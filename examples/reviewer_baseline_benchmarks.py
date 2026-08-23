"""Run matched OASIS and LPCMCI reviewer baselines on controlled simulations.

OASIS is evaluated first as an event estimator and then as preprocessing for
c-GC/LPCMCI.  LPCMCI raw PAG, p-value, and test-statistic tensors are retained;
the separate lagged binary projection is explicitly labeled as lossy.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from calcium_transient_rising_flank import (
    CausalisedGC,
    LPCMCIAdapter,
    OASISDeconvolver,
    build_representations,
    edge_recovery,
    selected_frame_indices,
)
from calcium_transient_rising_flank.representations import deconvolve_ar1
from calcium_transient_rising_flank.validation import simulate_calcium_dataset


DEFAULT_OUTPUT_DIR = Path("outputs/reviewer_baselines")
CONDITIONS = ("no_common_input", "latent_common_input")
REPRESENTATIONS = ("full", "ar1_innovation", "oasis", "rise")
PROGRESS_FILE = "progress.json"
EVENT_PARTIAL_FILE = "event_recovery_rows.partial.csv"
GRAPH_PARTIAL_FILE = "graph_recovery_rows.partial.csv"


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


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty rows to {path}")
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


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def add_latent_common_input(
    traces: np.ndarray,
    *,
    strength: float,
    random_state: int,
) -> np.ndarray:
    """Add an unobserved autocorrelated driver to two otherwise separate ROIs."""

    values = np.asarray(traces, dtype=float)
    if values.ndim != 2 or values.shape[0] != 4:
        raise ValueError("common-input benchmark expects four ROI traces")
    if strength < 0.0 or not np.isfinite(strength):
        raise ValueError("strength must be finite and non-negative")
    rng = np.random.default_rng(random_state)
    latent = np.zeros(values.shape[1], dtype=float)
    innovations = rng.normal(size=values.shape[1])
    for timepoint in range(1, latent.size):
        latent[timepoint] = 0.85 * latent[timepoint - 1] + innovations[timepoint]
    scale = float(np.std(latent))
    if scale > 0.0:
        latent /= scale
    loadings = np.array([1.0, 0.0, 0.8, 0.0])
    return values + strength * loadings[:, None] * latent[None, :]


def _event_counts(
    truth_indices: np.ndarray,
    predicted_indices: np.ndarray,
    *,
    tolerance: int,
) -> tuple[int, int, int, list[int]]:
    unmatched = list(int(value) for value in np.sort(predicted_indices))
    true_positives = 0
    errors: list[int] = []
    for truth in np.sort(truth_indices):
        candidates = [
            (abs(value - int(truth)), index, value)
            for index, value in enumerate(unmatched)
            if abs(value - int(truth)) <= tolerance
        ]
        if not candidates:
            continue
        error, index, value = min(candidates)
        true_positives += 1
        errors.append(int(value - int(truth)))
        unmatched.pop(index)
    false_positives = len(unmatched)
    false_negatives = int(truth_indices.size) - true_positives
    return true_positives, false_positives, false_negatives, errors


def event_recovery_metrics(
    truth: np.ndarray,
    scores: np.ndarray,
    *,
    thresholds: np.ndarray,
    tolerance: int,
) -> dict[str, float | int | None]:
    """Return one-to-one tolerance-window event recovery across all ROIs."""

    truth_values = np.asarray(truth) > 0
    score_values = np.asarray(scores, dtype=float)
    threshold_values = np.asarray(thresholds, dtype=float)
    if truth_values.shape != score_values.shape:
        raise ValueError("truth and scores must have matching ROI x time shape")
    if threshold_values.shape != (truth_values.shape[0],):
        raise ValueError("thresholds must provide one value per ROI")
    tp = fp = fn = 0
    errors: list[int] = []
    for roi in range(truth_values.shape[0]):
        counts = _event_counts(
            np.flatnonzero(truth_values[roi]),
            np.flatnonzero(score_values[roi] > threshold_values[roi]),
            tolerance=tolerance,
        )
        tp += counts[0]
        fp += counts[1]
        fn += counts[2]
        errors.extend(counts[3])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "median_signed_timing_error": None
        if not errors
        else float(np.median(errors)),
        "median_absolute_timing_error": None
        if not errors
        else float(np.median(np.abs(errors))),
    }


def _mad_thresholds(scores: np.ndarray, calibration_frames: int) -> np.ndarray:
    prefix = np.asarray(scores, dtype=float)[:, :calibration_frames]
    centers = np.median(prefix, axis=1)
    scales = 1.4826 * np.median(np.abs(prefix - centers[:, None]), axis=1)
    return np.maximum(centers + 3.0 * scales, 0.0)


def _event_rows(
    *,
    condition: str,
    seed: int,
    truth_events: np.ndarray,
    representations: dict[str, np.ndarray],
    calibration_frames: int,
    tolerance: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in ("ar1_innovation", "oasis"):
        scores = representations[method]
        for threshold_rule, thresholds in (
            ("positive", np.zeros(scores.shape[0], dtype=float)),
            ("held_out_mad3", _mad_thresholds(scores, calibration_frames)),
        ):
            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "event_method": method,
                    "threshold_rule": threshold_rule,
                    "calibration_frames": calibration_frames,
                    "tolerance_frames": tolerance,
                    "threshold_min": float(np.min(thresholds)),
                    "threshold_median": float(np.median(thresholds)),
                    "threshold_max": float(np.max(thresholds)),
                    **event_recovery_metrics(
                        truth_events,
                        scores,
                        thresholds=thresholds,
                        tolerance=tolerance,
                    ),
                }
            )
    return rows


def _skeleton_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    true_skeleton = np.asarray(truth, dtype=bool) | np.asarray(truth, dtype=bool).T
    predicted_skeleton = (
        np.asarray(predicted, dtype=bool) | np.asarray(predicted, dtype=bool).T
    )
    upper = np.triu(np.ones(true_skeleton.shape, dtype=bool), k=1)
    tp = int(np.count_nonzero(true_skeleton & predicted_skeleton & upper))
    fp = int(np.count_nonzero(~true_skeleton & predicted_skeleton & upper))
    fn = int(np.count_nonzero(true_skeleton & ~predicted_skeleton & upper))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "skeleton_true_positives": tp,
        "skeleton_false_positives": fp,
        "skeleton_false_negatives": fn,
        "skeleton_precision": precision,
        "skeleton_recall": recall,
    }


def _cgc_graph(
    traces: np.ndarray,
    *,
    representation_name: str,
    representation: np.ndarray,
    n_surrogates: int,
    seed: int,
) -> np.ndarray:
    estimator = CausalisedGC(
        max_lag=1,
        n_surrogates=n_surrogates,
        alpha=0.05,
        fdr=True,
        event_mode="physical",
        random_state=seed,
    )
    if representation_name == "rise":
        return estimator.fit(
            traces,
            event_indices=selected_frame_indices(representation),
        ).adjacency
    return estimator.fit(representation).adjacency


def _pag_mark_counts(graph: np.ndarray) -> str:
    counts = Counter(str(value) for value in np.asarray(graph).ravel() if str(value))
    return json.dumps(dict(sorted(counts.items())), sort_keys=True)


def _pag_p_descriptives(
    p_matrix: np.ndarray,
    truth: np.ndarray,
) -> dict[str, float | None]:
    lagged = np.asarray(p_matrix, dtype=float)[:, :, 1:]
    if lagged.size == 0:
        return {
            "true_edge_p_mean": None,
            "null_edge_p_mean": None,
            "null_edge_p_le_0_05_fraction": None,
        }
    off_diagonal = ~np.eye(truth.shape[0], dtype=bool)
    true_mask = np.broadcast_to(truth[:, :, None], lagged.shape)
    eligible = np.broadcast_to(off_diagonal[:, :, None], lagged.shape)
    true_values = lagged[eligible & true_mask]
    null_values = lagged[eligible & ~true_mask]
    return {
        "true_edge_p_mean": None
        if true_values.size == 0
        else float(np.mean(true_values)),
        "null_edge_p_mean": None
        if null_values.size == 0
        else float(np.mean(null_values)),
        "null_edge_p_le_0_05_fraction": None
        if null_values.size == 0
        else float(np.mean(null_values <= 0.05)),
    }


def rows_for_unit(
    *,
    condition: str,
    seed: int,
    n_steps: int,
    common_input_strength: float,
    n_cgc_surrogates: int,
    lpcmci_tau_max: int,
    lpcmci_pc_alpha: float,
    event_tolerance: int,
    pag_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    truth = _truth_graph()
    dataset = simulate_calcium_dataset(
        truth,
        n_steps=n_steps,
        gamma=np.array([0.72, 0.80, 0.88, 0.92]),
        noise_std=0.035,
        shared_noise_std=0.0,
        random_state=seed,
    )
    traces = dataset.fluorescence
    if condition == "latent_common_input":
        traces = add_latent_common_input(
            traces,
            strength=common_input_strength,
            random_state=900_000 + seed,
        )
    elif condition != "no_common_input":
        raise ValueError(f"unsupported condition: {condition}")

    ar1 = deconvolve_ar1(traces)
    oasis_result = OASISDeconvolver(
        backend_kwargs={"penalty": 1, "optimize_g": 0}
    ).fit(traces)
    rise = build_representations(traces).rise
    representations = {
        "full": traces,
        "ar1_innovation": ar1,
        "oasis": oasis_result.spikes,
        "rise": rise,
    }
    calibration_frames = max(20, n_steps // 4)
    event_rows = _event_rows(
        condition=condition,
        seed=seed,
        truth_events=dataset.events,
        representations=representations,
        calibration_frames=calibration_frames,
        tolerance=event_tolerance,
    )

    graph_rows: list[dict[str, Any]] = []
    for representation_index, representation_name in enumerate(REPRESENTATIONS):
        representation = representations[representation_name]
        cgc_adjacency = _cgc_graph(
            traces,
            representation_name=representation_name,
            representation=representation,
            n_surrogates=n_cgc_surrogates,
            seed=200_000 + seed * 10 + representation_index,
        )
        cgc_recovery = edge_recovery(truth, cgc_adjacency)
        graph_rows.append(
            {
                "condition": condition,
                "seed": seed,
                "method": "cgc",
                "representation": representation_name,
                "output_semantics": "directed_adjacency",
                "projection": "none",
                "comparison_metric": "directed_edge_recovery",
                "precision": cgc_recovery.precision,
                "recall": cgc_recovery.recall,
                "false_positive_rate": cgc_recovery.false_positive_rate,
                "orientation_accuracy": cgc_recovery.orientation_accuracy,
                "retained_edges": int(np.count_nonzero(cgc_adjacency)),
                "n_estimator_surrogates": n_cgc_surrogates,
                "alpha": 0.05,
                "fdr": True,
            }
        )

        lpcmci = LPCMCIAdapter(
            tau_max=lpcmci_tau_max,
            run_kwargs={"tau_min": 0, "pc_alpha": lpcmci_pc_alpha},
        ).fit(representation)
        pag_path = pag_dir / f"{condition}__seed-{seed}__{representation_name}.npz"
        pag_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            pag_path,
            graph=lpcmci.graph,
            p_matrix=lpcmci.p_matrix,
            val_matrix=lpcmci.val_matrix,
        )
        projected = lpcmci.lossy_lagged_skeleton()
        skeleton_metrics = _skeleton_metrics(truth, projected)
        graph_rows.append(
            {
                "condition": condition,
                "seed": seed,
                "method": "lpcmci",
                "representation": representation_name,
                "output_semantics": "raw_pag_with_separate_lossy_lagged_projection",
                "projection": "lossy_lagged_pag_skeleton",
                "comparison_metric": "undirected_skeleton_support_only",
                "retained_skeleton_edges": skeleton_metrics[
                    "skeleton_true_positives"
                ]
                + skeleton_metrics["skeleton_false_positives"],
                "pc_alpha": lpcmci_pc_alpha,
                "tau_max": lpcmci_tau_max,
                "conditional_independence_test": lpcmci.cond_ind_test,
                "raw_pag_path": str(pag_path),
                "pag_mark_counts": _pag_mark_counts(lpcmci.graph),
                **skeleton_metrics,
                **_pag_p_descriptives(lpcmci.p_matrix, truth),
            }
        )
    return event_rows, graph_rows


def _config(args: argparse.Namespace, seeds: Sequence[int]) -> dict[str, Any]:
    return {
        "resume_schema_version": 2,
        "conditions": list(CONDITIONS),
        "seeds": list(seeds),
        "n_steps": args.n_steps,
        "common_input_strength": args.common_input_strength,
        "n_cgc_surrogates": args.n_cgc_surrogates,
        "lpcmci_tau_max": args.lpcmci_tau_max,
        "lpcmci_pc_alpha": args.lpcmci_pc_alpha,
        "event_tolerance": args.event_tolerance,
        "representations": list(REPRESENTATIONS),
        "oasis": {"penalty": 1, "optimize_g": 0, "kinetics": "auto_ar1"},
        "pag_projection_policy": "raw PAG is primary; lagged skeleton is lossy",
    }


def _unit_key(condition: str, seed: int) -> str:
    return f"{condition}|{seed}"


def _recover_completed_units(
    *,
    event_rows: Sequence[dict[str, Any]],
    graph_rows: Sequence[dict[str, Any]],
    conditions: Sequence[str],
    seeds: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    expected_units = {
        _unit_key(condition, seed) for condition in conditions for seed in seeds
    }
    event_grouped: dict[str, list[dict[str, Any]]] = {}
    for row in event_rows:
        unit = _unit_key(str(row["condition"]), int(row["seed"]))
        event_grouped.setdefault(unit, []).append(row)
    graph_grouped: dict[str, list[dict[str, Any]]] = {}
    for row in graph_rows:
        unit = _unit_key(str(row["condition"]), int(row["seed"]))
        graph_grouped.setdefault(unit, []).append(row)
    unknown = (set(event_grouped) | set(graph_grouped)) - expected_units
    if unknown:
        raise SystemExit(
            "resume partial rows contain unexpected units: "
            + ", ".join(sorted(unknown))
        )

    completed: set[str] = set()
    for unit in expected_units:
        unit_event_rows = event_grouped.get(unit, [])
        event_keys = {
            (str(row["event_method"]), str(row["threshold_rule"]))
            for row in unit_event_rows
        }
        unit_graph_rows = graph_grouped.get(unit, [])
        graph_keys = {
            (str(row["method"]), str(row["representation"]))
            for row in unit_graph_rows
        }
        pag_files_present = all(
            Path(str(row["raw_pag_path"])).exists()
            for row in unit_graph_rows
            if row["method"] == "lpcmci"
        )
        if (
            len(unit_event_rows) == 4
            and len(event_keys) == 4
            and len(unit_graph_rows) == 2 * len(REPRESENTATIONS)
            and len(graph_keys) == 2 * len(REPRESENTATIONS)
            and pag_files_present
        ):
            completed.add(unit)

    recovered_event_rows = [
        row
        for row in event_rows
        if _unit_key(str(row["condition"]), int(row["seed"])) in completed
    ]
    recovered_graph_rows = [
        row
        for row in graph_rows
        if _unit_key(str(row["condition"]), int(row["seed"])) in completed
    ]
    return recovered_event_rows, recovered_graph_rows, completed


def _write_progress(
    output_dir: Path,
    *,
    config: dict[str, Any],
    completed_units: set[str],
    status: str,
) -> None:
    payload = {
        "status": status,
        "config": config,
        "completed_units": sorted(completed_units),
        "completed_unit_count": len(completed_units),
        "expected_unit_count": len(config["conditions"]) * len(config["seeds"]),
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
    parser.add_argument("--common-input-strength", type=float, default=0.3)
    parser.add_argument("--n-cgc-surrogates", type=int, default=1000)
    parser.add_argument("--lpcmci-tau-max", type=int, default=2)
    parser.add_argument("--lpcmci-pc-alpha", type=float, default=0.05)
    parser.add_argument("--event-tolerance", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _parse_seeds(args.seeds)
    if args.n_cgc_surrogates < 1:
        raise SystemExit("--n-cgc-surrogates must be positive")
    if args.lpcmci_tau_max < 1:
        raise SystemExit("--lpcmci-tau-max must be at least one")
    if not 0.0 < args.lpcmci_pc_alpha < 1.0:
        raise SystemExit("--lpcmci-pc-alpha must lie in (0, 1)")
    if args.event_tolerance < 0:
        raise SystemExit("--event-tolerance cannot be negative")
    config = _config(args, seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / PROGRESS_FILE
    event_partial = args.output_dir / EVENT_PARTIAL_FILE
    graph_partial = args.output_dir / GRAPH_PARTIAL_FILE
    completed_units: set[str] = set()
    event_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    if args.resume and progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get("config") != config:
            raise SystemExit("resume configuration does not match the saved run")
        event_rows = _read_csv(event_partial)
        graph_rows = _read_csv(graph_partial)
        event_rows, graph_rows, completed_units = _recover_completed_units(
            event_rows=event_rows,
            graph_rows=graph_rows,
            conditions=CONDITIONS,
            seeds=seeds,
        )

    _write_progress(
        args.output_dir,
        config=config,
        completed_units=completed_units,
        status="running",
    )
    for condition in CONDITIONS:
        for seed in seeds:
            unit = _unit_key(condition, seed)
            if unit in completed_units:
                continue
            new_event_rows, new_graph_rows = rows_for_unit(
                condition=condition,
                seed=seed,
                n_steps=args.n_steps,
                common_input_strength=args.common_input_strength,
                n_cgc_surrogates=args.n_cgc_surrogates,
                lpcmci_tau_max=args.lpcmci_tau_max,
                lpcmci_pc_alpha=args.lpcmci_pc_alpha,
                event_tolerance=args.event_tolerance,
                pag_dir=args.output_dir / "raw_pag",
            )
            event_rows.extend(new_event_rows)
            graph_rows.extend(new_graph_rows)
            completed_units.add(unit)
            _write_csv(event_partial, event_rows)
            _write_csv(graph_partial, graph_rows)
            _write_progress(
                args.output_dir,
                config=config,
                completed_units=completed_units,
                status="running",
            )

    _write_csv(args.output_dir / "event_recovery_rows.csv", event_rows)
    _write_csv(args.output_dir / "graph_recovery_rows.csv", graph_rows)
    summary = {
        "status": "complete",
        "config": config,
        "n_event_rows": len(event_rows),
        "n_graph_rows": len(graph_rows),
        "interpretation": {
            "oasis": "event estimator and preprocessing ablation, not a causal learner",
            "lpcmci": "raw PAG retained; binary lagged projection is explicitly lossy",
            "common_input": "controlled synthetic latent-driver stress test",
        },
        "outputs": [
            "event_recovery_rows.csv",
            "graph_recovery_rows.csv",
            "raw_pag/*.npz",
        ],
    }
    _atomic_write_text(
        args.output_dir / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    _write_progress(
        args.output_dir,
        config=config,
        completed_units=completed_units,
        status="complete",
    )


if __name__ == "__main__":
    main()
