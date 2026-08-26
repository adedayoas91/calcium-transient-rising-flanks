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
from calcium_transient_rising_flank.checkpointing import format_progress
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
OBSERVED_GRAPH_ARTIFACTS_DIR = "observed_graph_artifacts"
OBSERVED_GRAPH_ARTIFACTS_MANIFEST_FILE = "observed_graph_artifacts_manifest.json"
OBSERVED_GRAPH_ARTIFACTS_SCHEMA_VERSION = 1
ARTIFACT_REPRESENTATIONS = frozenset({"rise", "fall"})
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


def _atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **arrays)
    temporary.replace(path)


def _artifact_slug(value: str) -> str:
    return "".join(
        character if character.isalnum() else "-"
        for character in str(value).strip().lower()
    ).strip("-")


def _should_stage_observed_graph_artifact(
    *,
    representation_name: str,
    null_type: str,
    replicate: int,
) -> bool:
    return (
        null_type == "observed"
        and replicate == 0
        and representation_name in ARTIFACT_REPRESENTATIONS
    )


def _stage_observed_graph_artifact(
    artifacts: list[dict[str, Any]] | None,
    *,
    base: dict[str, Any],
    graph,
    null_type: str,
    replicate: int,
    fdr: bool | None,
) -> None:
    if artifacts is None:
        return
    representation_name = str(base["representation"])
    if not _should_stage_observed_graph_artifact(
        representation_name=representation_name,
        null_type=null_type,
        replicate=replicate,
    ):
        return
    multiple_testing = None
    if fdr is not None:
        multiple_testing = "benjamini-hochberg" if fdr else "unadjusted"
    artifacts.append(
        {
            "case": str(base["case"]),
            "description": str(base["description"]),
            "recording": str(base["recording"]),
            "fish": int(base["fish"]),
            "trial": int(base["trial"]),
            "method": str(base["method"]),
            "representation": representation_name,
            "null_type": null_type,
            "replicate": replicate,
            "fdr": fdr,
            "multiple_testing": multiple_testing,
            "edge_testing_family": EDGE_TESTING_FAMILY,
            "estimator": graph.estimator,
            "n_rois": int(np.asarray(graph.adjacency).shape[0]),
            "adjacency": np.asarray(graph.adjacency, dtype=bool).copy(),
            "retained_scores": np.asarray(graph.retained_scores, dtype=float).copy(),
            "p_values": np.asarray(graph.p_values, dtype=float).copy(),
            "best_lags": np.asarray(graph.best_lags, dtype=int).copy(),
        }
    )


def _artifact_unit_key(entry: dict[str, Any]) -> str:
    return "|".join((str(entry["method"]), str(entry["case"]), str(entry["recording"])))


def _artifact_entry_key(entry: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(entry["case"]),
        str(entry["recording"]),
        str(entry["method"]),
        str(entry["representation"]),
    )


def _artifact_relative_path(entry: dict[str, Any]) -> str:
    filename = "__".join(
        (
            _artifact_slug(str(entry["case"])),
            _artifact_slug(str(entry["recording"])),
            _artifact_slug(str(entry["method"])),
            _artifact_slug(str(entry["representation"])),
        )
    )
    return f"{OBSERVED_GRAPH_ARTIFACTS_DIR}/{filename}.npz"


def _load_observed_graph_artifact_entries(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / OBSERVED_GRAPH_ARTIFACTS_MANIFEST_FILE
    if not path.exists():
        return []
    with path.open() as file:
        payload = json.load(file)
    if payload.get("schema_version") != OBSERVED_GRAPH_ARTIFACTS_SCHEMA_VERSION:
        raise SystemExit("observed graph artifact manifest schema mismatch")
    entries = payload.get("artifacts")
    if not isinstance(entries, list):
        raise SystemExit("observed graph artifact manifest is malformed")
    return [dict(entry) for entry in entries]


def _write_observed_graph_artifacts(
    output_dir: Path,
    *,
    staged_artifacts: list[dict[str, Any]],
    manifest_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not staged_artifacts and manifest_entries:
        payload = {
            "schema_version": OBSERVED_GRAPH_ARTIFACTS_SCHEMA_VERSION,
            "edge_testing_family": EDGE_TESTING_FAMILY,
            "artifacts": sorted(manifest_entries, key=_artifact_entry_key),
        }
        _atomic_write_text(
            output_dir / OBSERVED_GRAPH_ARTIFACTS_MANIFEST_FILE,
            json.dumps(payload, indent=2) + "\n",
        )
        return sorted(manifest_entries, key=_artifact_entry_key)

    merged = {_artifact_entry_key(entry): dict(entry) for entry in manifest_entries}
    for artifact in staged_artifacts:
        relative_path = _artifact_relative_path(artifact)
        path = output_dir / relative_path
        metadata = {
            "case": artifact["case"],
            "description": artifact["description"],
            "recording": artifact["recording"],
            "fish": artifact["fish"],
            "trial": artifact["trial"],
            "method": artifact["method"],
            "representation": artifact["representation"],
            "null_type": artifact["null_type"],
            "replicate": artifact["replicate"],
            "fdr": artifact["fdr"],
            "multiple_testing": artifact["multiple_testing"],
            "edge_testing_family": artifact["edge_testing_family"],
            "estimator": artifact["estimator"],
            "n_rois": artifact["n_rois"],
        }
        _atomic_write_npz(
            path,
            adjacency=np.asarray(artifact["adjacency"], dtype=bool),
            retained_scores=np.asarray(artifact["retained_scores"], dtype=float),
            p_values=np.asarray(artifact["p_values"], dtype=float),
            best_lags=np.asarray(artifact["best_lags"], dtype=int),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        manifest_entry = {
            **metadata,
            "path": relative_path,
        }
        merged[_artifact_entry_key(manifest_entry)] = manifest_entry
    updated_entries = sorted(merged.values(), key=_artifact_entry_key)
    payload = {
        "schema_version": OBSERVED_GRAPH_ARTIFACTS_SCHEMA_VERSION,
        "edge_testing_family": EDGE_TESTING_FAMILY,
        "artifacts": updated_entries,
    }
    _atomic_write_text(
        output_dir / OBSERVED_GRAPH_ARTIFACTS_MANIFEST_FILE,
        json.dumps(payload, indent=2) + "\n",
    )
    return updated_entries


def _expected_observed_graph_artifact_count(
    representations: tuple[str, ...],
) -> int:
    return sum(
        representation in ARTIFACT_REPRESENTATIONS
        for representation in representations
    )


def _artifact_entries_by_unit(
    entries: list[dict[str, Any]],
    *,
    output_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        path = output_dir / str(entry["path"])
        if not path.exists():
            continue
        grouped.setdefault(_artifact_unit_key(entry), []).append(entry)
    return grouped


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
    observed_artifacts: list[dict[str, Any]] | None = None,
    fdr: bool | None = None,
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
        _stage_observed_graph_artifact(
            observed_artifacts,
            base=base,
            graph=observed,
            null_type="observed",
            replicate=0,
            fdr=fdr,
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
        "resume_schema_version": 3,
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
    artifact_entries: list[dict[str, Any]] | None = None,
    output_dir: Path | None = None,
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
    expected_artifact_count = _expected_observed_graph_artifact_count(representations)
    enforce_artifacts = (
        expected_artifact_count > 0
        and artifact_entries is not None
        and output_dir is not None
    )
    artifacts_by_unit = (
        {}
        if not enforce_artifacts
        else _artifact_entries_by_unit(artifact_entries, output_dir=output_dir)
    )
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
        row_complete = len(unit_rows) == expected_count and len(row_keys) == expected_count
        artifact_complete = True
        if enforce_artifacts:
            unit_artifacts = artifacts_by_unit.get(unit, [])
            artifact_keys = {
                (
                    str(entry["representation"]),
                    str(entry["null_type"]),
                    int(entry["replicate"]),
                )
                for entry in unit_artifacts
            }
            artifact_complete = (
                len(unit_artifacts) == expected_artifact_count
                and len(artifact_keys) == expected_artifact_count
            )
        if row_complete and artifact_complete:
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
    active_unit: str | None = None,
) -> None:
    payload = {
        "status": status,
        "config": config,
        "completed_units": sorted(completed_units),
        "completed_unit_count": len(completed_units),
        "expected_unit_count": expected_units,
        "expected_top_level_fit_count": expected_top_level_fits,
        "active_unit": active_unit,
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
    manifest_entries: list[dict[str, Any]] = []
    previous_active_unit: str | None = None
    if args.resume and progress_path.exists():
        with progress_path.open() as file:
            progress = json.load(file)
        if progress.get("config") != config:
            raise SystemExit(
                "resume configuration does not match the saved empirical run"
            )
        previous_active_unit = progress.get("active_unit")
        rows = _read_csv(partial_rows_path)
        manifest_entries = _load_observed_graph_artifact_entries(args.output_dir)
        rows, completed_units = _recover_completed_units(
            rows,
            records=records,
            methods=methods,
            representations=representations,
            n_null_replicates=args.n_null_replicates,
            artifact_entries=manifest_entries,
            output_dir=args.output_dir,
        )
        manifest_entries = [
            entry
            for entry in manifest_entries
            if _artifact_unit_key(entry) in completed_units
            and (args.output_dir / str(entry["path"])).exists()
        ]
        manifest_entries = _write_observed_graph_artifacts(
            args.output_dir,
            staged_artifacts=[],
            manifest_entries=manifest_entries,
        )

    expected_units = len(methods) * len(records)
    expected_top_level_fits = _expected_top_level_fit_count(
        records=records,
        methods=methods,
        representations=representations,
        n_null_replicates=args.n_null_replicates,
    )
    print(
        "[plan] "
        f"{expected_units} checkpoint units; {expected_top_level_fits} top-level fits; "
        f"output={args.output_dir}",
        flush=True,
    )
    if args.resume and progress_path.exists():
        print(
            f"[resume] loaded and validated {len(completed_units)}/{expected_units} "
            "complete checkpoint units; completed units will be skipped",
            flush=True,
        )
        if previous_active_unit is not None and previous_active_unit not in completed_units:
            print(
                f"[resume] previous run stopped during {previous_active_unit}; "
                "that unit will be recomputed",
                flush=True,
            )
    elif args.resume:
        print(
            f"[resume] no saved progress found at {progress_path}; starting a new run",
            flush=True,
        )
    else:
        print("[resume] disabled; saved completed units will not be loaded", flush=True)

    print(
        format_progress(len(completed_units), expected_units, label="Overall units"),
        flush=True,
    )
    _write_progress(
        args.output_dir,
        config=config,
        completed_units=completed_units,
        expected_units=expected_units,
        expected_top_level_fits=expected_top_level_fits,
        status="running",
        active_unit=None,
    )
    for method_index, method in enumerate(methods):
        factory = _estimator_factory(args, method=method)
        seed_offset = args.seed + method_index * 100_000
        for index, record in enumerate(records):
            unit = _unit_key(method, record)
            if unit in completed_units:
                continue
            print(
                f"[unit] starting {len(completed_units) + 1}/{expected_units}: {unit}",
                flush=True,
            )
            _write_progress(
                args.output_dir,
                config=config,
                completed_units=completed_units,
                expected_units=expected_units,
                expected_top_level_fits=expected_top_level_fits,
                status="running",
                active_unit=unit,
            )
            staged_artifacts: list[dict[str, Any]] = []
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
                    observed_artifacts=staged_artifacts,
                    fdr=not args.no_fdr,
                )
            )
            manifest_entries = _write_observed_graph_artifacts(
                args.output_dir,
                staged_artifacts=staged_artifacts,
                manifest_entries=manifest_entries,
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
                active_unit=None,
            )
            print(
                format_progress(
                    len(completed_units),
                    expected_units,
                    label="Overall units",
                )
                + f" | completed {unit}",
                flush=True,
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
                    OBSERVED_GRAPH_ARTIFACTS_MANIFEST_FILE,
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
        active_unit=None,
    )
    print(
        format_progress(expected_units, expected_units, label="Overall units")
        + f" | complete; wrote {len(rows)} null-control rows to {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
