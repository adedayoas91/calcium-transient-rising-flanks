"""Run empirical motoneuron null-control tables for saved A-D trace cases."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from calcium_transient_rising_flank import (
    CausalisedGC,
    build_representations,
    graph_summary,
    selected_frame_indices,
)
from calcium_transient_rising_flank.validation import (
    cross_recording_surrogate,
    cyclic_shift_surrogate,
    jitter_event_indices,
    permute_phase_event_indices,
    reverse_event_indices,
)

DEFAULT_DATA_DIR = Path("data/motoneurons")
DEFAULT_OUTPUT_DIR = Path("outputs/empirical_null_controls")

CASE_FILES = {
    "A": (
        "dff_dict_with_bad_neurons.pkl",
        "middle_dict.pkl",
        "bad neurons and artefacts retained",
    ),
    "B": (
        "dff_removed_dict.pkl",
        "middle_removed_dict.pkl",
        "bad neurons removed; artefacts retained",
    ),
    "C": (
        "dff_corrected_dict.pkl",
        "middle_removed_dict.pkl",
        "bad neurons removed; artefacts corrected",
    ),
    "D": (
        "dff_smoothed_dict.pkl",
        "middle_removed_dict.pkl",
        "bad neurons removed; artefacts corrected; smoothed",
    ),
}

DEFAULT_REPRESENTATIONS = ("full", "deconvolved", "rise", "fall", "fall_residual")
SELECTED_REPRESENTATIONS = {"rise", "fall", "fall_residual"}
METHOD_CHOICES = ("cgc", "fcgc", "cgc-star", "cgc*")
PARTIAL_ROWS_FILE = "null_control_rows.partial.csv"
PROGRESS_FILE = "progress.json"
EDGE_TESTING_FAMILY = (
    "eligible non-self ordered ROI pairs within each "
    "recording/case/method/representation"
)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def _recording_label(key: tuple[int, int]) -> str:
    fish, trial = key
    return f"F{fish}T{trial}"


def _parse_recordings(value: str | None) -> set[str] | None:
    if value is None or value.strip().lower() == "all":
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _parse_methods(value: str) -> tuple[str, ...]:
    methods = tuple(method.strip() for method in value.split(",") if method.strip())
    if not methods:
        raise ValueError("at least one method is required")
    invalid = sorted(set(methods) - set(METHOD_CHOICES))
    if invalid:
        raise ValueError(f"unsupported methods: {', '.join(invalid)}")
    return methods


def load_case_traces(
    data_dir: Path,
    *,
    cases: tuple[str, ...] = tuple(CASE_FILES),
    recordings: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Load declared motoneuron preprocessing cases from pickle dictionaries."""

    loaded: list[dict[str, Any]] = []
    for case_label in cases:
        trace_file, middle_file, description = CASE_FILES[case_label]
        traces_by_recording = _load_pickle(data_dir / trace_file)
        middle_by_recording = _load_pickle(data_dir / middle_file)
        for key, traces in sorted(traces_by_recording.items()):
            label = _recording_label(key)
            if recordings is not None and label not in recordings:
                continue
            if key not in middle_by_recording:
                continue
            fish, trial = key
            loaded.append(
                {
                    "case": case_label,
                    "description": description,
                    "recording": label,
                    "fish": fish,
                    "trial": trial,
                    "traces": np.asarray(traces, dtype=float),
                    "mid": int(middle_by_recording[key]),
                }
            )
    return loaded


def _next_donor(
    records: list[dict[str, Any]],
    index: int,
) -> np.ndarray | None:
    if len(records) < 2:
        return None
    current = records[index]
    for offset in range(1, len(records)):
        candidate = records[(index + offset) % len(records)]
        if candidate["case"] == current["case"] and candidate["fish"] != current["fish"]:
            return candidate["traces"]
    for offset in range(1, len(records)):
        candidate = records[(index + offset) % len(records)]
        if candidate["case"] == current["case"]:
            return candidate["traces"]
    return None


def _estimator_factory(
    args: argparse.Namespace,
    *,
    method: str,
) -> Callable[[int], CausalisedGC]:
    def factory(seed: int) -> CausalisedGC:
        return CausalisedGC(
            max_lag=args.max_lag,
            n_surrogates=args.n_estimator_surrogates,
            alpha=args.alpha,
            random_state=seed,
            fdr=not args.no_fdr,
            score_threshold=args.score_threshold,
            event_mode=args.event_mode,
            method=method,
        )

    return factory


def _fit_graph(
    estimator: CausalisedGC,
    traces: np.ndarray,
    representation_name: str,
    representation: np.ndarray,
    event_indices: tuple[np.ndarray, ...] | None,
):
    if representation_name in SELECTED_REPRESENTATIONS:
        return estimator.fit(traces, event_indices=event_indices)
    return estimator.fit(representation)


def _append_row(
    rows: list[dict[str, Any]],
    *,
    base: dict[str, Any],
    graph,
    mid: int,
    null_type: str,
    replicate: int,
) -> None:
    row = {
        **base,
        "null_type": null_type,
        "replicate": replicate,
        **graph_summary(graph.retained_scores, mid, binary=False),
    }
    rows.append(row)


def null_rows_for_recording(
    record: dict[str, Any],
    *,
    donor_traces: np.ndarray | None,
    estimator_factory: Callable[[int], CausalisedGC],
    method: str = "cgc",
    representations: tuple[str, ...] = DEFAULT_REPRESENTATIONS,
    n_null_replicates: int = 5,
    max_jitter: int = 1,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Run observed and null-control graph metrics for one recording/case."""

    if n_null_replicates < 0:
        raise ValueError("n_null_replicates cannot be negative")
    rng = np.random.default_rng(seed)
    traces = np.asarray(record["traces"], dtype=float)
    bundle = build_representations(traces)
    represented = bundle.as_dict()
    selected = {
        name: selected_frame_indices(represented[name])
        for name in SELECTED_REPRESENTATIONS
    }
    rows: list[dict[str, Any]] = []

    for representation_name in representations:
        values = represented[representation_name]
        event_indices = selected.get(representation_name)
        base = {
            "case": record["case"],
            "description": record["description"],
            "recording": record["recording"],
            "fish": record["fish"],
            "trial": record["trial"],
            "method": method,
            "representation": representation_name,
        }
        observed = _fit_graph(
            estimator_factory(seed),
            traces,
            representation_name,
            values,
            event_indices,
        )
        _append_row(
            rows,
            base=base,
            graph=observed,
            mid=record["mid"],
            null_type="observed",
            replicate=0,
        )

        reverse_indices = (
            None
            if event_indices is None
            else reverse_event_indices(event_indices, traces.shape[1])
        )
        reverse_graph = _fit_graph(
            estimator_factory(seed + 1),
            traces[:, ::-1],
            representation_name,
            values[:, ::-1],
            reverse_indices,
        )
        _append_row(
            rows,
            base=base,
            graph=reverse_graph,
            mid=record["mid"],
            null_type="reverse_time",
            replicate=0,
        )

        if donor_traces is not None:
            donor = cross_recording_surrogate(traces, donor_traces)
            donor_representation = build_representations(donor).as_dict()[
                representation_name
            ]
            donor_indices = (
                selected_frame_indices(donor_representation)
                if representation_name in SELECTED_REPRESENTATIONS
                else None
            )
            cross_graph = _fit_graph(
                estimator_factory(seed + 2),
                donor,
                representation_name,
                donor_representation,
                donor_indices,
            )
            _append_row(
                rows,
                base=base,
                graph=cross_graph,
                mid=record["mid"],
                null_type="cross_recording",
                replicate=0,
            )

        for replicate in range(1, n_null_replicates + 1):
            replicate_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            if representation_name in SELECTED_REPRESENTATIONS:
                shifted = cyclic_shift_surrogate(traces, random_state=replicate_seed)
                shifted_values = build_representations(shifted).as_dict()[
                    representation_name
                ]
                shifted_indices = selected_frame_indices(shifted_values)
            else:
                shifted = cyclic_shift_surrogate(values, random_state=replicate_seed)
                shifted_values = shifted
                shifted_indices = None
            cyclic_graph = _fit_graph(
                estimator_factory(replicate_seed),
                shifted,
                representation_name,
                shifted_values,
                shifted_indices,
            )
            _append_row(
                rows,
                base=base,
                graph=cyclic_graph,
                mid=record["mid"],
                null_type="cyclic_shift",
                replicate=replicate,
            )

            if event_indices is None:
                continue
            jittered = jitter_event_indices(
                event_indices,
                traces.shape[1],
                max_jitter=max_jitter,
                random_state=replicate_seed,
            )
            jitter_graph = estimator_factory(replicate_seed).fit(
                traces, event_indices=jittered
            )
            _append_row(
                rows,
                base=base,
                graph=jitter_graph,
                mid=record["mid"],
                null_type="event_jitter",
                replicate=replicate,
            )

            alternative_name = "fall" if representation_name == "rise" else "rise"
            permuted = permute_phase_event_indices(
                event_indices,
                selected[alternative_name],
                random_state=replicate_seed,
            )
            phase_graph = estimator_factory(replicate_seed).fit(
                traces, event_indices=permuted
            )
            _append_row(
                rows,
                base=base,
                graph=phase_graph,
                mid=record["mid"],
                null_type="phase_permutation",
                replicate=replicate,
            )
    return rows


def summarize_null_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["case"],
            row.get("method", "not_recorded"),
            row["representation"],
            row["null_type"],
        )
        grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        case, method, representation, null_type = key
        entry: dict[str, Any] = {
            "case": case,
            "method": method,
            "representation": representation,
            "null_type": null_type,
            "n": len(group),
        }
        for metric in ("w_ic", "w_rc", "edge_density", "retained_edges", "total_weight"):
            values = [row[metric] for row in group if row[metric] is not None]
            entry[f"{metric}_mean"] = float(np.mean(values)) if values else None
        summaries.append(entry)
    return summaries


def _finite_metric_values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        numeric = float(value)
        if np.isfinite(numeric):
            values.append(numeric)
    return values


def build_null_contrast_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare observed graph summaries against each empirical null family."""

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["case"],
            row.get("method", "not_recorded"),
            row["representation"],
            row["null_type"],
        )
        grouped.setdefault(key, []).append(row)

    metrics = ("w_ic", "w_rc", "edge_density", "retained_edges", "total_weight")
    contrasts: list[dict[str, Any]] = []
    observed_by_key = {
        (case, method, representation): group
        for (case, method, representation, null_type), group in grouped.items()
        if null_type == "observed"
    }
    for (case, method, representation, null_type), null_group in sorted(grouped.items()):
        if null_type == "observed":
            continue
        observed_group = observed_by_key.get((case, method, representation), [])
        entry: dict[str, Any] = {
            "case": case,
            "method": method,
            "representation": representation,
            "null_type": null_type,
            "n_observed": len(observed_group),
            "n_null": len(null_group),
        }
        for metric in metrics:
            observed_values = _finite_metric_values(observed_group, metric)
            null_values = _finite_metric_values(null_group, metric)
            observed_mean = (
                float(np.mean(observed_values)) if observed_values else None
            )
            null_mean = float(np.mean(null_values)) if null_values else None
            entry[f"observed_{metric}_mean"] = observed_mean
            entry[f"null_{metric}_mean"] = null_mean
            entry[f"observed_minus_null_{metric}_mean"] = (
                None
                if observed_mean is None or null_mean is None
                else observed_mean - null_mean
            )
            entry[f"p_null_ge_observed_{metric}_mean"] = (
                None
                if observed_mean is None or not null_values
                else (1 + sum(value >= observed_mean for value in null_values))
                / (len(null_values) + 1)
            )
        contrasts.append(entry)
    return contrasts


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _coerce_csv_value(value: str | None) -> Any:
    if value is None:
        return None
    if value == "":
        return None
    if value in {"True", "False"}:
        return value == "True"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return [
            {key: _coerce_csv_value(value) for key, value in row.items()}
            for row in csv.DictReader(file)
        ]


def _run_config(
    args: argparse.Namespace,
    *,
    cases: tuple[str, ...],
    representations: tuple[str, ...],
    methods: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "resume_schema_version": 2,
        "data_dir": str(args.data_dir),
        "cases": list(cases),
        "recordings": args.recordings,
        "representations": list(representations),
        "methods": list(methods),
        "n_null_replicates": args.n_null_replicates,
        "max_jitter": args.max_jitter,
        "max_lag": args.max_lag,
        "n_estimator_surrogates": args.n_estimator_surrogates,
        "alpha": args.alpha,
        "score_threshold": args.score_threshold,
        "event_mode": args.event_mode,
        "fdr": not args.no_fdr,
        "seed": args.seed,
        "edge_testing_family": EDGE_TESTING_FAMILY,
    }


def _unit_key(method: str, record: dict[str, Any]) -> str:
    return "|".join((method, str(record["case"]), str(record["recording"])))


def _expected_rows_for_recording(
    *,
    donor_available: bool,
    representations: tuple[str, ...],
    n_null_replicates: int,
) -> int:
    base_per_representation = 2 + int(donor_available)
    selected_count = sum(
        representation in SELECTED_REPRESENTATIONS
        for representation in representations
    )
    continuous_count = len(representations) - selected_count
    return (
        len(representations) * base_per_representation
        + n_null_replicates * (continuous_count + 3 * selected_count)
    )


def _recover_completed_units(
    rows: list[dict[str, Any]],
    *,
    records: list[dict[str, Any]],
    methods: tuple[str, ...],
    representations: tuple[str, ...],
    n_null_replicates: int,
) -> tuple[list[dict[str, Any]], set[str]]:
    expected_counts = {
        _unit_key(method, record): _expected_rows_for_recording(
            donor_available=_next_donor(records, index) is not None,
            representations=representations,
            n_null_replicates=n_null_replicates,
        )
        for method in methods
        for index, record in enumerate(records)
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        unit = "|".join(
            (str(row["method"]), str(row["case"]), str(row["recording"]))
        )
        grouped.setdefault(unit, []).append(row)
    unknown = set(grouped) - set(expected_counts)
    if unknown:
        raise SystemExit(
            "resume partial rows contain unexpected units: "
            + ", ".join(sorted(unknown))
        )
    completed: set[str] = set()
    for unit, unit_rows in grouped.items():
        row_keys = {
            (
                str(row["representation"]),
                str(row["null_type"]),
                int(row["replicate"]),
            )
            for row in unit_rows
        }
        expected_count = expected_counts[unit]
        if len(unit_rows) == expected_count and len(row_keys) == expected_count:
            completed.add(unit)
    recovered_rows = [
        row
        for row in rows
        if "|".join(
            (str(row["method"]), str(row["case"]), str(row["recording"]))
        )
        in completed
    ]
    return recovered_rows, completed


def _expected_top_level_fit_count(
    *,
    records: list[dict[str, Any]],
    methods: tuple[str, ...],
    representations: tuple[str, ...],
    n_null_replicates: int,
) -> int:
    selected_count = sum(
        representation in SELECTED_REPRESENTATIONS
        for representation in representations
    )
    continuous_count = len(representations) - selected_count
    total = 0
    for method in methods:
        for index, _record in enumerate(records):
            base_per_representation = 2 + int(_next_donor(records, index) is not None)
            total += len(representations) * base_per_representation
            total += n_null_replicates * (
                continuous_count + 3 * selected_count
            )
    return total


def _write_progress(
    output_dir: Path,
    *,
    config: dict[str, Any],
    completed_units: set[str],
    expected_units: int,
    expected_top_level_fits: int,
    status: str,
) -> None:
    payload = {
        "status": status,
        "config": config,
        "completed_units": sorted(completed_units),
        "completed_unit_count": len(completed_units),
        "expected_unit_count": expected_units,
        "expected_top_level_fit_count": expected_top_level_fits,
    }
    _atomic_write_text(
        output_dir / PROGRESS_FILE,
        json.dumps(payload, indent=2) + "\n",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cases", default="A,B,C,D")
    parser.add_argument("--recordings", default="all")
    parser.add_argument(
        "--representations",
        default=",".join(DEFAULT_REPRESENTATIONS),
    )
    parser.add_argument("--n-null-replicates", type=int, default=5)
    parser.add_argument("--max-jitter", type=int, default=1)
    parser.add_argument("--max-lag", type=int, default=1)
    parser.add_argument("--n-estimator-surrogates", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument(
        "--event-mode", choices=("compressed", "physical"), default="physical"
    )
    parser.add_argument(
        "--method",
        choices=METHOD_CHOICES,
        default="cgc",
        help="single estimator method; ignored when --methods is supplied",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="comma-separated estimator methods for combined output, e.g. cgc,cgc-star",
    )
    parser.add_argument("--no-fdr", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed method/case/recording units from this output directory",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = tuple(case.strip() for case in args.cases.split(",") if case.strip())
    invalid_cases = sorted(set(cases) - set(CASE_FILES))
    if invalid_cases:
        raise SystemExit(f"unsupported cases: {', '.join(invalid_cases)}")
    representations = tuple(
        item.strip() for item in args.representations.split(",") if item.strip()
    )
    invalid_representations = sorted(set(representations) - set(DEFAULT_REPRESENTATIONS))
    if invalid_representations:
        raise SystemExit(
            f"unsupported representations: {', '.join(invalid_representations)}"
        )

    records = load_case_traces(
        args.data_dir,
        cases=cases,
        recordings=_parse_recordings(args.recordings),
    )
    methods = _parse_methods(args.methods or args.method)
    config = _run_config(
        args,
        cases=cases,
        representations=representations,
        methods=methods,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / PROGRESS_FILE
    partial_rows_path = args.output_dir / PARTIAL_ROWS_FILE
    completed_units: set[str] = set()
    rows: list[dict[str, Any]] = []
    if args.resume and progress_path.exists():
        with progress_path.open() as file:
            progress = json.load(file)
        if progress.get("config") != config:
            raise SystemExit(
                "resume configuration does not match the saved empirical run"
            )
        rows = _read_csv(partial_rows_path)
        rows, completed_units = _recover_completed_units(
            rows,
            records=records,
            methods=methods,
            representations=representations,
            n_null_replicates=args.n_null_replicates,
        )

    expected_units = len(methods) * len(records)
    expected_top_level_fits = _expected_top_level_fit_count(
        records=records,
        methods=methods,
        representations=representations,
        n_null_replicates=args.n_null_replicates,
    )
    _write_progress(
        args.output_dir,
        config=config,
        completed_units=completed_units,
        expected_units=expected_units,
        expected_top_level_fits=expected_top_level_fits,
        status="running",
    )
    for method_index, method in enumerate(methods):
        factory = _estimator_factory(args, method=method)
        seed_offset = args.seed + method_index * 100_000
        for index, record in enumerate(records):
            unit = _unit_key(method, record)
            if unit in completed_units:
                continue
            rows.extend(
                null_rows_for_recording(
                    record,
                    donor_traces=_next_donor(records, index),
                    estimator_factory=factory,
                    method=method,
                    representations=representations,
                    n_null_replicates=args.n_null_replicates,
                    max_jitter=args.max_jitter,
                    seed=seed_offset + index,
                )
            )
            completed_units.add(unit)
            _write_csv(partial_rows_path, rows)
            _write_progress(
                args.output_dir,
                config=config,
                completed_units=completed_units,
                expected_units=expected_units,
                expected_top_level_fits=expected_top_level_fits,
                status="running",
            )

    if not rows:
        raise SystemExit("no null-control rows were generated")
    summaries = summarize_null_rows(rows)
    contrasts = build_null_contrast_rows(rows)
    _write_csv(args.output_dir / "null_control_rows.csv", rows)
    _write_csv(args.output_dir / "null_control_summary.csv", summaries)
    _write_csv(args.output_dir / "null_control_contrasts.csv", contrasts)
    _atomic_write_text(
        args.output_dir / "summary.json",
        json.dumps(
            {
                "status": "complete",
                "config": config,
                "data_dir": str(args.data_dir),
                "cases": cases,
                "recordings": args.recordings,
                "representations": representations,
                "methods": methods,
                "method": methods[0] if len(methods) == 1 else None,
                "n_rows": len(rows),
                "n_summary_rows": len(summaries),
                "n_contrast_rows": len(contrasts),
                "expected_top_level_fit_count": expected_top_level_fits,
                "edge_testing_family": EDGE_TESTING_FAMILY,
                "outputs": [
                    "null_control_rows.csv",
                    "null_control_summary.csv",
                    "null_control_contrasts.csv",
                ],
            },
            indent=2,
        )
        + "\n",
    )
    _write_progress(
        args.output_dir,
        config=config,
        completed_units=completed_units,
        expected_units=expected_units,
        expected_top_level_fits=expected_top_level_fits,
        status="complete",
    )
    print(f"wrote {len(rows)} null-control rows to {args.output_dir}")


if __name__ == "__main__":
    main()
