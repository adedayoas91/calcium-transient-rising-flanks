"""Run empirical motoneuron stability tables for saved A-D trace cases."""

from __future__ import annotations

import argparse
import csv
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
    StabilityResult,
    run_event_bootstrap_stability,
    run_leave_one_neuron_stability,
    run_leave_one_transient_stability,
    run_time_window_stability,
)
from calcium_transient_rising_flank.checkpointing import (
    JsonUnitCheckpointStore,
    atomic_write_json,
    format_progress,
)

DEFAULT_DATA_DIR = Path("data/motoneurons")
DEFAULT_OUTPUT_DIR = Path("outputs/empirical_stability")

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
METRICS = ("w_ic", "w_rc", "edge_density", "retained_edges", "total_weight")
METHOD_CHOICES = ("cgc", "fcgc", "cgc-star", "cgc*")


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


def _mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _graph_metric_means(result: StabilityResult, mid: int) -> dict[str, float | None]:
    summaries = [
        graph_summary(graph.retained_scores, mid, binary=False)
        for graph in result.graphs
    ]
    means: dict[str, float | None] = {}
    for metric in METRICS:
        values = [
            float(row[metric])
            for row in summaries
            if row.get(metric) is not None and np.isfinite(float(row[metric]))
        ]
        means[f"mean_{metric}"] = _mean(values)
    return means


def _ok_row(
    *,
    base: dict[str, Any],
    stability_type: str,
    result: StabilityResult,
    mid: int,
) -> dict[str, Any]:
    return {
        **base,
        "stability_type": stability_type,
        "status": "ok",
        "skip_reason": None,
        "n_graphs": len(result.graphs),
        "stability": result.stability,
        **_graph_metric_means(result, mid),
    }


def _skipped_row(
    *,
    base: dict[str, Any],
    stability_type: str,
    reason: str,
) -> dict[str, Any]:
    return {
        **base,
        "stability_type": stability_type,
        "status": "skipped",
        "skip_reason": reason,
        "n_graphs": 0,
        "stability": None,
        **{f"mean_{metric}": None for metric in METRICS},
    }


def _try_stability(
    rows: list[dict[str, Any]],
    *,
    base: dict[str, Any],
    stability_type: str,
    mid: int,
    fn,
) -> None:
    try:
        result = fn()
    except ValueError as exc:
        rows.append(
            _skipped_row(
                base=base,
                stability_type=stability_type,
                reason=str(exc),
            )
        )
        return
    rows.append(
        _ok_row(
            base=base,
            stability_type=stability_type,
            result=result,
            mid=mid,
        )
    )


def stability_rows_for_recording(
    record: dict[str, Any],
    *,
    estimator_factory: Callable[[int], CausalisedGC],
    method: str = "cgc",
    representations: tuple[str, ...] = DEFAULT_REPRESENTATIONS,
    n_event_bootstrap: int = 20,
    window_length: int = 120,
    window_step: int | None = None,
    include_leave_one_neuron: bool = True,
    include_leave_one_transient: bool = True,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Run graph-stability diagnostics for one recording/case."""

    traces = np.asarray(record["traces"], dtype=float)
    represented = build_representations(traces).as_dict()
    selected = {
        name: selected_frame_indices(represented[name])
        for name in SELECTED_REPRESENTATIONS
    }
    rows: list[dict[str, Any]] = []

    for index, representation_name in enumerate(representations):
        values = represented[representation_name]
        event_indices = selected.get(representation_name)
        fit_values = traces if event_indices is not None else values
        base = {
            "case": record["case"],
            "description": record["description"],
            "recording": record["recording"],
            "fish": record["fish"],
            "trial": record["trial"],
            "method": method,
            "representation": representation_name,
            "event_selected": event_indices is not None,
        }
        representation_seed = seed + index * 1000

        if event_indices is not None:
            if n_event_bootstrap >= 2:
                _try_stability(
                    rows,
                    base=base,
                    stability_type="event_bootstrap",
                    mid=record["mid"],
                    fn=lambda representation_seed=representation_seed: (
                        run_event_bootstrap_stability(
                            traces,
                            estimator_factory(representation_seed),
                            event_indices,
                            n_bootstrap=n_event_bootstrap,
                            random_state=representation_seed,
                        )
                    ),
                )
            else:
                rows.append(
                    _skipped_row(
                        base=base,
                        stability_type="event_bootstrap",
                        reason="n_event_bootstrap must be at least two",
                    )
                )

        _try_stability(
            rows,
            base=base,
            stability_type="time_window",
            mid=record["mid"],
            fn=lambda representation_seed=representation_seed: (
                run_time_window_stability(
                    fit_values,
                    estimator_factory(representation_seed + 1),
                    window_length=window_length,
                    step=window_step,
                    event_indices=event_indices,
                )
            ),
        )

        if include_leave_one_neuron:
            _try_stability(
                rows,
                base=base,
                stability_type="leave_one_neuron",
                mid=record["mid"],
                fn=lambda representation_seed=representation_seed: (
                    run_leave_one_neuron_stability(
                        fit_values,
                        estimator_factory(representation_seed + 2),
                        event_indices=event_indices,
                    )
                ),
            )

        if event_indices is not None and include_leave_one_transient:
            _try_stability(
                rows,
                base=base,
                stability_type="leave_one_transient",
                mid=record["mid"],
                fn=lambda representation_seed=representation_seed: (
                    run_leave_one_transient_stability(
                        traces,
                        estimator_factory(representation_seed + 3),
                        event_indices,
                    )
                ),
            )
    return rows


def summarize_stability_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate empirical stability rows by case, representation, and method."""

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["case"],
            row.get("method", "not_recorded"),
            row["representation"],
            row["stability_type"],
        )
        grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for (case, method, representation, stability_type), group in sorted(grouped.items()):
        ok = [row for row in group if row["status"] == "ok"]
        entry: dict[str, Any] = {
            "case": case,
            "method": method,
            "representation": representation,
            "stability_type": stability_type,
            "n_rows": len(group),
            "n_ok": len(ok),
            "n_skipped": len(group) - len(ok),
            "stability_mean": _mean(
                [
                    float(row["stability"])
                    for row in ok
                    if row["stability"] is not None
                ]
            ),
        }
        for metric in METRICS:
            key = f"mean_{metric}"
            values = [
                float(row[key])
                for row in ok
                if row.get(key) is not None and np.isfinite(float(row[key]))
            ]
            entry[f"{key}_mean"] = _mean(values)
        summaries.append(entry)
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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
    parser.add_argument("--n-event-bootstrap", type=int, default=20)
    parser.add_argument("--window-length", type=int, default=120)
    parser.add_argument("--window-step", type=int, default=None)
    parser.add_argument("--no-leave-one-neuron", action="store_true")
    parser.add_argument("--no-leave-one-transient", action="store_true")
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
        help="reuse complete method/case/recording stability checkpoint units",
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
    config = {
        "data_dir": str(args.data_dir),
        "cases": list(cases),
        "recordings": args.recordings,
        "representations": list(representations),
        "methods": list(methods),
        "n_event_bootstrap": args.n_event_bootstrap,
        "window_length": args.window_length,
        "window_step": args.window_step,
        "include_leave_one_neuron": not args.no_leave_one_neuron,
        "include_leave_one_transient": not args.no_leave_one_transient,
        "max_lag": args.max_lag,
        "n_estimator_surrogates": args.n_estimator_surrogates,
        "alpha": args.alpha,
        "score_threshold": args.score_threshold,
        "event_mode": args.event_mode,
        "fdr": not args.no_fdr,
        "seed": args.seed,
    }
    checkpoint_store = JsonUnitCheckpointStore(
        args.output_dir,
        "empirical_stability",
        config,
    )
    checkpoint_store.initialize(resume=args.resume)
    rows: list[dict[str, Any]] = []
    unit_ids = [
        f"{method}|{record['case']}|{record['recording']}"
        for method in methods
        for record in records
    ]
    total_units = len(unit_ids)
    resumable_units = sum(
        1
        for unit_id in unit_ids
        if args.resume and checkpoint_store.load_rows(unit_id) is not None
    )
    completed_units = resumable_units
    print(
        "[plan] "
        f"{total_units} checkpoint units; methods={','.join(methods)}; "
        f"output={args.output_dir}",
        flush=True,
    )
    if args.resume:
        print(
            f"[resume] loaded and validated {resumable_units}/{total_units} "
            "complete checkpoint units; incomplete units will be recomputed",
            flush=True,
        )
    else:
        print("[resume] disabled; saved completed units will not be loaded", flush=True)
    print(
        format_progress(resumable_units, total_units, label="Overall units"),
        flush=True,
    )
    for method_index, method in enumerate(methods):
        factory = _estimator_factory(args, method=method)
        seed_offset = args.seed + method_index * 100_000
        for index, record in enumerate(records):
            unit_id = f"{method}|{record['case']}|{record['recording']}"
            completed_rows = (
                checkpoint_store.load_rows(unit_id) if args.resume else None
            )
            if completed_rows is not None:
                rows.extend(completed_rows)
                print(
                    format_progress(
                        completed_units,
                        total_units,
                        label="Overall units",
                    )
                    + f" | loaded checkpoint {unit_id}",
                    flush=True,
                )
                continue
            print(
                f"[unit] starting {completed_units + 1}/{total_units}: {unit_id}",
                flush=True,
            )
            unit_rows = stability_rows_for_recording(
                record,
                estimator_factory=factory,
                method=method,
                representations=representations,
                n_event_bootstrap=args.n_event_bootstrap,
                window_length=args.window_length,
                window_step=args.window_step,
                include_leave_one_neuron=not args.no_leave_one_neuron,
                include_leave_one_transient=not args.no_leave_one_transient,
                seed=seed_offset + index,
            )
            checkpoint_store.save_rows(unit_id, unit_rows)
            rows.extend(unit_rows)
            completed_units += 1
            print(
                format_progress(
                    completed_units,
                    total_units,
                    label="Overall units",
                )
                + f" | completed {unit_id}",
                flush=True,
            )

    if not rows:
        raise SystemExit("no stability rows were generated")
    summaries = summarize_stability_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "stability_rows.csv", rows)
    _write_csv(args.output_dir / "stability_summary.csv", summaries)
    atomic_write_json(
        args.output_dir / "summary.json",
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
            "outputs": ["stability_rows.csv", "stability_summary.csv"],
        },
    )
    checkpoint_store.finish(
        completed_units=total_units,
        total_units=total_units,
    )
    print(
        format_progress(total_units, total_units, label="Overall units")
        + f" | complete; wrote {len(rows)} empirical stability rows to {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
