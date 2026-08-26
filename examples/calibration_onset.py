"""Run held-out threshold and adaptive-onset experiments.

The experiment is deliberately network-blind during calibration: thresholds
are estimated from a held-out prefix of each simulated trace and are never
selected by inspecting an inferred graph.  Seed-level checkpoints make the
grid safe to resume on a compute machine.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from calcium_transient_rising_flank import (
    CausalisedGC,
    build_representations,
    calibrate_roi_thresholds,
    calibration_signal,
    detect_bayesian_onsets,
    detect_change_point_onsets,
    edge_recovery,
    selected_frame_indices,
    w_ic,
    w_rc,
)
from calcium_transient_rising_flank.checkpointing import format_progress
from calcium_transient_rising_flank.validation import simulate_calcium_dataset


DEFAULT_OUTPUT_DIR = Path("outputs/calibration_onset")
CALIBRATION_METHODS = ("first_difference_mad", "ar_residual_mad")
FIXED_THRESHOLDS = (0.0, 0.02, 0.05)
MULTIPLIERS = (2.0, 3.0, 4.0)
GRAPH_SCORE_THRESHOLDS = (0.0, 0.05, 0.1, 0.2)
COMPONENTS = ("threshold", "onset")
PROGRESS_FILE = "progress.json"
THRESHOLD_PARTIAL_FILE = "threshold_graph_rows.partial.csv"
ONSET_PARTIAL_FILE = "onset_rows.partial.csv"
ONSET_DETECTORS = (
    "thresholded_increment",
    "change_point",
    "kinetics_template_posterior",
)


def _truth_graph() -> np.ndarray:
    return np.array(
        [
            [False, True, False, False],
            [False, False, False, False],
            [False, False, False, True],
            [False, False, False, False],
        ],
        dtype=bool,
    )


def _f1(precision: float, recall: float) -> float:
    denominator = precision + recall
    return 0.0 if denominator == 0.0 else 2.0 * precision * recall / denominator


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty row collection to {path}")
    fields = sorted({field for row in rows for field in row})
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _coerce(value: str | None) -> Any:
    if value in {None, ""}:
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
            {key: _coerce(value) for key, value in row.items()}
            for row in csv.DictReader(file)
        ]


def _parse_numbers(value: str, cast: type[int] | type[float]) -> tuple[Any, ...]:
    parsed = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("at least one value is required")
    return parsed


def _parse_components(value: str) -> tuple[str, ...]:
    components = tuple(item.strip() for item in value.split(",") if item.strip())
    if not components:
        raise argparse.ArgumentTypeError("at least one component is required")
    invalid = sorted(set(components) - set(COMPONENTS))
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unsupported components: {', '.join(invalid)}"
        )
    return components


def _threshold_row(
    *,
    seed: int,
    threshold_kind: str,
    threshold_method: str,
    threshold_value: float | None,
    threshold_multiplier: float | None,
    thresholds: np.ndarray,
    calibration_frames: int,
    evaluation_traces: np.ndarray,
    truth: np.ndarray,
    n_surrogates: int,
) -> list[dict[str, Any]]:
    rise = build_representations(evaluation_traces, tolerance=thresholds).rise
    selected = selected_frame_indices(rise)
    graph = CausalisedGC(
        max_lag=1,
        n_surrogates=n_surrogates,
        alpha=0.05,
        fdr=True,
        event_mode="physical",
        random_state=100_000 + seed,
    ).fit(evaluation_traces, event_indices=selected)
    off_diagonal = ~np.eye(truth.shape[0], dtype=bool)
    sides = np.array(["L", "L", "R", "R"])
    positions = np.array([0.0, 1.0, 0.0, 1.0])
    rows: list[dict[str, Any]] = []
    for graph_score_threshold in GRAPH_SCORE_THRESHOLDS:
        adjacency = graph.adjacency & (graph.scores >= graph_score_threshold)
        recovery = edge_recovery(truth, adjacency)
        ipsilateral = w_ic(adjacency, sides, binary=True)
        rostrocaudal = w_rc(adjacency, sides, positions, binary=True)
        rows.append(
            {
                "seed": seed,
                "threshold_kind": threshold_kind,
                "threshold_method": threshold_method,
                "threshold_value": threshold_value,
                "threshold_multiplier": threshold_multiplier,
                "graph_score_threshold": graph_score_threshold,
                "calibration_frames": calibration_frames,
                "evaluation_frames": evaluation_traces.shape[1],
                "threshold_min": float(np.min(thresholds)),
                "threshold_median": float(np.median(thresholds)),
                "threshold_max": float(np.max(thresholds)),
                "selected_frames_total": int(sum(frames.size for frames in selected)),
                "precision": recovery.precision,
                "recall": recovery.recall,
                "false_positive_rate": recovery.false_positive_rate,
                "f1": _f1(recovery.precision, recovery.recall),
                "orientation_accuracy": recovery.orientation_accuracy,
                "edge_density": float(np.mean(adjacency[off_diagonal])),
                "retained_edges": int(np.count_nonzero(adjacency)),
                "w_ic": ipsilateral.value,
                "w_rc": rostrocaudal.value,
                "n_estimator_surrogates": n_surrogates,
                "alpha": 0.05,
                "fdr": True,
                "event_mode": "physical",
                "calibration_uses_graph": False,
            }
        )
    return rows


def threshold_rows_for_seed(
    *, seed: int, n_steps: int, calibration_fraction: float, n_surrogates: int
) -> list[dict[str, Any]]:
    """Evaluate fixed and held-out calibrated thresholds on one shared trace."""

    truth = _truth_graph()
    dataset = simulate_calcium_dataset(
        truth,
        n_steps=n_steps,
        gamma=np.array([0.72, 0.80, 0.88, 0.92]),
        noise_std=0.035,
        shared_noise_std=0.02,
        random_state=seed,
    )
    split = int(round(n_steps * calibration_fraction))
    if split < 3 or n_steps - split < 3:
        raise ValueError("calibration and evaluation partitions need at least 3 frames")
    calibration_traces = dataset.fluorescence[:, :split]
    evaluation_traces = dataset.fluorescence[:, split:]
    rows: list[dict[str, Any]] = []
    for threshold in FIXED_THRESHOLDS:
        rows.extend(
            _threshold_row(
                seed=seed,
                threshold_kind="fixed_global",
                threshold_method="fixed",
                threshold_value=threshold,
                threshold_multiplier=None,
                thresholds=np.full(truth.shape[0], threshold),
                calibration_frames=split,
                evaluation_traces=evaluation_traces,
                truth=truth,
                n_surrogates=n_surrogates,
            )
        )
    for method in CALIBRATION_METHODS:
        for multiplier in MULTIPLIERS:
            calibrated = calibrate_roi_thresholds(
                calibration_traces,
                method=method,
                threshold_multiplier=multiplier,
            )
            rows.extend(
                _threshold_row(
                    seed=seed,
                    threshold_kind="held_out_per_roi",
                    threshold_method=method,
                    threshold_value=None,
                    threshold_multiplier=multiplier,
                    thresholds=calibrated.thresholds,
                    calibration_frames=split,
                    evaluation_traces=evaluation_traces,
                    truth=truth,
                    n_surrogates=n_surrogates,
                )
            )
    return rows


def simulate_onset_traces(
    *,
    seed: int,
    n_rois: int = 8,
    n_timepoints: int = 180,
    calibration_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate heterogeneous kinetics with signal and null ROIs."""

    if n_rois < 4 or n_timepoints < 80:
        raise ValueError("onset benchmark needs at least 4 ROIs and 80 frames")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must lie in (0, 1)")
    rng = np.random.default_rng(seed)
    calibration_frames = int(round(n_timepoints * calibration_fraction))
    if calibration_frames < 3 or n_timepoints - calibration_frames < 54:
        raise ValueError(
            "onset calibration needs at least 3 prefix frames and 54 evaluation frames"
        )
    true_onsets = np.full(n_rois, -1, dtype=int)
    signal_count = n_rois - 2
    true_onsets[:signal_count] = rng.integers(
        calibration_frames + 8,
        n_timepoints - 45,
        size=signal_count,
    )
    traces = rng.normal(scale=0.03, size=(n_rois, n_timepoints))
    shared = rng.normal(scale=0.015, size=n_timepoints)
    traces += shared[None, :]
    for roi, onset in enumerate(true_onsets[:signal_count]):
        rise = int(rng.integers(2, 7))
        decay = float(rng.uniform(8.0, 22.0))
        amplitude = float(rng.uniform(0.8, 1.4))
        for timepoint in range(int(onset), n_timepoints):
            delay = timepoint - int(onset)
            if delay < rise:
                response = amplitude * (delay + 1.0) / rise
            else:
                response = amplitude * np.exp(-(delay - rise + 1.0) / decay)
            traces[roi, timepoint] += response
    return traces, true_onsets, calibration_frames


def _first_crossings(signal: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    crossings = np.full(signal.shape[0], -1, dtype=int)
    for roi in range(signal.shape[0]):
        candidates = np.flatnonzero(signal[roi] > thresholds[roi])
        if candidates.size:
            crossings[roi] = int(candidates[0])
    return crossings


def _onset_metric_row(
    *, seed: int, calibration_method: str, detector: str,
    predicted: np.ndarray, truth: np.ndarray,
    n_steps: int | None = None,
    calibration_frames: int | None = None,
) -> dict[str, Any]:
    true_event = truth >= 0
    predicted_event = predicted >= 0
    true_positive = true_event & predicted_event
    errors = np.abs(predicted[true_positive] - truth[true_positive])
    tp = int(np.count_nonzero(true_positive))
    fp = int(np.count_nonzero(~true_event & predicted_event))
    fn = int(np.count_nonzero(true_event & ~predicted_event))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "seed": seed,
        "calibration_method": calibration_method,
        "detector": detector,
        "n_steps": n_steps,
        "calibration_frames": calibration_frames,
        "calibration_fraction_actual": None
        if n_steps is None or calibration_frames is None
        else calibration_frames / n_steps,
        "n_signal_rois": int(np.count_nonzero(true_event)),
        "n_null_rois": int(np.count_nonzero(~true_event)),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "within_2_frames": 0.0 if errors.size == 0 else float(np.mean(errors <= 2)),
        "within_5_frames": 0.0 if errors.size == 0 else float(np.mean(errors <= 5)),
        "median_absolute_error": None
        if errors.size == 0
        else float(np.median(errors)),
        "calibration_uses_graph": False,
    }


def onset_rows_for_seed(
    seed: int,
    *,
    n_steps: int = 180,
    calibration_fraction: float = 0.25,
) -> list[dict[str, Any]]:
    traces, truth, calibration_frames = simulate_onset_traces(
        seed=seed,
        n_timepoints=n_steps,
        calibration_fraction=calibration_fraction,
    )
    evaluation = traces[:, calibration_frames:]
    rows: list[dict[str, Any]] = []
    for method in CALIBRATION_METHODS:
        calibration = calibrate_roi_thresholds(
            traces[:, :calibration_frames],
            method=method,
            threshold_multiplier=3.0,
        )
        innovations = calibration_signal(evaluation, calibration, pad=True)
        thresholded = _first_crossings(innovations, calibration.thresholds)
        change_point = detect_change_point_onsets(
            evaluation,
            calibration=calibration,
        ).onset_indices.copy()
        bayesian = detect_bayesian_onsets(
            evaluation,
            calibration=calibration,
        ).onset_indices.copy()
        for predicted in (thresholded, change_point, bayesian):
            predicted[predicted >= 0] += calibration_frames
        rows.extend(
            (
                _onset_metric_row(
                    seed=seed,
                    calibration_method=method,
                    detector="thresholded_increment",
                    predicted=thresholded,
                    truth=truth,
                    n_steps=n_steps,
                    calibration_frames=calibration_frames,
                ),
                _onset_metric_row(
                    seed=seed,
                    calibration_method=method,
                    detector="change_point",
                    predicted=change_point,
                    truth=truth,
                    n_steps=n_steps,
                    calibration_frames=calibration_frames,
                ),
                _onset_metric_row(
                    seed=seed,
                    calibration_method=method,
                    detector="kinetics_template_posterior",
                    predicted=bayesian,
                    truth=truth,
                    n_steps=n_steps,
                    calibration_frames=calibration_frames,
                ),
            )
        )
    return rows


def _complete_threshold_seeds(rows: Sequence[dict[str, Any]]) -> set[int]:
    expected_count = (
        len(FIXED_THRESHOLDS) + len(CALIBRATION_METHODS) * len(MULTIPLIERS)
    ) * len(GRAPH_SCORE_THRESHOLDS)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["seed"]), []).append(row)
    complete: set[int] = set()
    for seed, seed_rows in grouped.items():
        keys = {
            (
                row["threshold_kind"],
                row["threshold_method"],
                row["threshold_value"],
                row["threshold_multiplier"],
                row["graph_score_threshold"],
            )
            for row in seed_rows
        }
        if len(seed_rows) == expected_count and len(keys) == expected_count:
            complete.add(seed)
    return complete


def _complete_onset_seeds(rows: Sequence[dict[str, Any]]) -> set[int]:
    expected_count = len(CALIBRATION_METHODS) * len(ONSET_DETECTORS)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["seed"]), []).append(row)
    complete: set[int] = set()
    for seed, seed_rows in grouped.items():
        keys = {
            (row["calibration_method"], row["detector"])
            for row in seed_rows
        }
        if len(seed_rows) == expected_count and len(keys) == expected_count:
            complete.add(seed)
    return complete


def _recover_completed_seeds(
    *,
    threshold_rows: Sequence[dict[str, Any]],
    onset_rows: Sequence[dict[str, Any]],
    components: Sequence[str],
    requested_seeds: Sequence[int],
) -> set[int]:
    completed = set(int(seed) for seed in requested_seeds)
    if "threshold" in components:
        completed &= _complete_threshold_seeds(threshold_rows)
    if "onset" in components:
        completed &= _complete_onset_seeds(onset_rows)
    return completed


def _config(args: argparse.Namespace, seeds: Sequence[int]) -> dict[str, Any]:
    return {
        "resume_schema_version": 2,
        "components": list(_parse_components(args.components)),
        "seeds": list(seeds),
        "n_steps": args.n_steps,
        "calibration_fraction": args.calibration_fraction,
        "n_estimator_surrogates": args.n_estimator_surrogates,
        "fixed_thresholds": list(FIXED_THRESHOLDS),
        "calibration_methods": list(CALIBRATION_METHODS),
        "threshold_multipliers": list(MULTIPLIERS),
        "graph_score_thresholds": list(GRAPH_SCORE_THRESHOLDS),
    }


def _write_progress(
    output_dir: Path,
    *,
    config: dict[str, Any],
    completed_seeds: set[int],
    status: str,
    active_seed: int | None = None,
) -> None:
    payload = {
        "status": status,
        "config": config,
        "completed_seeds": sorted(completed_seeds),
        "completed_seed_count": len(completed_seeds),
        "expected_seed_count": len(config["seeds"]),
        "active_seed": active_seed,
    }
    _atomic_write_text(
        output_dir / PROGRESS_FILE,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--n-steps", type=int, default=800)
    parser.add_argument("--calibration-fraction", type=float, default=0.4)
    parser.add_argument("--n-estimator-surrogates", type=int, default=1000)
    parser.add_argument("--components", default="threshold,onset")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _parse_numbers(args.seeds, int)
    components = _parse_components(args.components)
    if not 0.0 < args.calibration_fraction < 1.0:
        raise SystemExit("--calibration-fraction must lie in (0, 1)")
    if args.n_estimator_surrogates < 1:
        raise SystemExit("--n-estimator-surrogates must be positive")
    config = _config(args, seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / PROGRESS_FILE
    threshold_partial = args.output_dir / THRESHOLD_PARTIAL_FILE
    onset_partial = args.output_dir / ONSET_PARTIAL_FILE
    completed_seeds: set[int] = set()
    threshold_rows: list[dict[str, Any]] = []
    onset_rows: list[dict[str, Any]] = []
    total_seeds = len(seeds)
    previous_active_seed: int | None = None
    print(
        "[plan] "
        f"{total_seeds} checkpoint seeds; components={','.join(components)}; "
        f"output={args.output_dir}",
        flush=True,
    )
    if args.resume and progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get("config") != config:
            raise SystemExit("resume configuration does not match the saved run")
        previous_active_seed = progress.get("active_seed")
        if "threshold" in components:
            threshold_rows = _read_csv(threshold_partial)
        if "onset" in components:
            onset_rows = _read_csv(onset_partial)
        completed_seeds = _recover_completed_seeds(
            threshold_rows=threshold_rows,
            onset_rows=onset_rows,
            components=components,
            requested_seeds=seeds,
        )
        threshold_rows = [
            row for row in threshold_rows if int(row["seed"]) in completed_seeds
        ]
        onset_rows = [
            row for row in onset_rows if int(row["seed"]) in completed_seeds
        ]
        print(
            f"[resume] loaded and validated {len(completed_seeds)}/{total_seeds} "
            "complete checkpoint seeds; completed seeds will be skipped",
            flush=True,
        )
        if previous_active_seed is not None and previous_active_seed not in completed_seeds:
            print(
                f"[resume] previous run stopped during seed={previous_active_seed}; "
                "that seed will be recomputed",
                flush=True,
            )
    elif args.resume:
        print(
            f"[resume] no saved progress found at {progress_path}; starting a new run",
            flush=True,
        )
    else:
        print("[resume] disabled; saved completed seeds will not be loaded", flush=True)

    print(
        format_progress(len(completed_seeds), total_seeds, label="Overall seeds"),
        flush=True,
    )

    _write_progress(
        args.output_dir,
        config=config,
        completed_seeds=completed_seeds,
        status="running",
        active_seed=None,
    )
    for seed in seeds:
        if seed in completed_seeds:
            continue
        print(
            f"[seed] starting {len(completed_seeds) + 1}/{total_seeds}: seed={seed}",
            flush=True,
        )
        _write_progress(
            args.output_dir,
            config=config,
            completed_seeds=completed_seeds,
            status="running",
            active_seed=seed,
        )
        if "threshold" in components:
            threshold_rows.extend(
                threshold_rows_for_seed(
                    seed=seed,
                    n_steps=args.n_steps,
                    calibration_fraction=args.calibration_fraction,
                    n_surrogates=args.n_estimator_surrogates,
                )
            )
        if "onset" in components:
            onset_rows.extend(
                onset_rows_for_seed(
                    seed,
                    n_steps=args.n_steps,
                    calibration_fraction=args.calibration_fraction,
                )
            )
        completed_seeds.add(seed)
        if "threshold" in components:
            _write_csv(threshold_partial, threshold_rows)
        if "onset" in components:
            _write_csv(onset_partial, onset_rows)
        _write_progress(
            args.output_dir,
            config=config,
            completed_seeds=completed_seeds,
            status="running",
            active_seed=None,
        )
        print(
            format_progress(len(completed_seeds), total_seeds, label="Overall seeds")
            + f" | completed seed={seed}",
            flush=True,
        )

    outputs: list[str] = []
    if "threshold" in components:
        _write_csv(args.output_dir / "threshold_graph_rows.csv", threshold_rows)
        outputs.append("threshold_graph_rows.csv")
    if "onset" in components:
        _write_csv(args.output_dir / "onset_rows.csv", onset_rows)
        outputs.append("onset_rows.csv")
    summary = {
        "status": "complete",
        "config": config,
        "calibration_contract": {
            "uses_network_information": False,
            "uses_held_out_prefix": True,
            "selection_rule": "prespecified full grid; no preferred-graph tuning",
        },
        "n_threshold_rows": len(threshold_rows),
        "n_onset_rows": len(onset_rows),
        "outputs": outputs,
    }
    _atomic_write_text(
        args.output_dir / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    _write_progress(
        args.output_dir,
        config=config,
        completed_seeds=completed_seeds,
        status="complete",
        active_seed=None,
    )
    print(
        format_progress(total_seeds, total_seeds, label="Overall seeds")
        + f" | complete; summary={args.output_dir / 'summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
