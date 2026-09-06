"""Run OASIS and LPCMCI on the static c-GC/c-GC* simulation inputs.

Every run regenerates the exact network, fluorescence, latent events, seed
split, and representation inputs used by ``c-GC.ipynb`` and
``c-GC-star.ipynb``. Input digests are written with every row so outputs from
independently executed baseline notebooks can be checked for exact agreement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
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
    generate_static_validation_network,
    graph_summary,
    simulate_static_calcium_like,
    static_gamma_from_tau,
    static_input_digest,
)
from calcium_transient_rising_flank.checkpointing import format_progress


DEFAULT_OUTPUT_DIR = Path("outputs/simulation_baselines")
CONDITIONS = ("native", "noisy", "slow_decay", "low_framerate", "shared_input")
COMPONENTS = ("oasis", "lpcmci")
CGC_METHODS = ("cgc", "cgc-star")
REPRESENTATIONS = ("full", "deconvolved", "oasis", "rise", "fall")
REPRESENTATION_CHOICES = REPRESENTATIONS + ("fall_residual",)
PROGRESS_FILE = "progress.json"
EVENT_PARTIAL_FILE = "event_recovery_rows.partial.csv"
GRAPH_PARTIAL_FILE = "graph_recovery_rows.partial.csv"
BASE_SEED = 0
RISE_TAU = 5.0
DECAY_TAU = 14.0
NOISE_STD = 0.03
AMP_JITTER = 0.3
SMOOTH_WINDOW = 5
SPONTANEOUS_RATE = 0.012
TRANSMISSION_PROBABILITY = 0.85
REPRESENTATION_TOLERANCE = 0.05
IPSILATERAL_FRACTION = 0.65
PROPAGATION_DELAY = 1
CGC_ALPHA = 0.01
CGC_BETA = 0.001
CGC_N_PASTS = 2
CGC_N_LAGS = 1
CGC_FDR = False


def _condition_config(condition: str) -> dict[str, float | int]:
    configs: dict[str, dict[str, float | int]] = {
        "native": {
            "rise_tau": RISE_TAU,
            "decay_tau": DECAY_TAU,
            "noise_std": NOISE_STD,
            "shared_noise_std": 0.0,
            "downsample": 1,
        },
        "noisy": {
            "rise_tau": RISE_TAU,
            "decay_tau": DECAY_TAU,
            "noise_std": 0.08,
            "shared_noise_std": 0.0,
            "downsample": 1,
        },
        "slow_decay": {
            "rise_tau": RISE_TAU,
            "decay_tau": 2 * DECAY_TAU,
            "noise_std": NOISE_STD,
            "shared_noise_std": 0.0,
            "downsample": 1,
        },
        "low_framerate": {
            "rise_tau": RISE_TAU,
            "decay_tau": DECAY_TAU,
            "noise_std": NOISE_STD,
            "shared_noise_std": 0.0,
            "downsample": 2,
        },
        "shared_input": {
            "rise_tau": RISE_TAU,
            "decay_tau": DECAY_TAU,
            "noise_std": NOISE_STD,
            "shared_noise_std": 0.10,
            "downsample": 1,
        },
    }
    try:
        return configs[condition]
    except KeyError as exc:
        raise ValueError(f"unsupported condition: {condition}") from exc


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


def _atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **arrays)
    temporary.replace(path)


def _valid_pag_file(path: Path, *, input_digest: str) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            return {
                "graph",
                "p_matrix",
                "val_matrix",
                "input_digest",
            }.issubset(payload.files) and (
                str(np.asarray(payload["input_digest"]).item()) == input_digest
            )
    except (OSError, ValueError):
        return False


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


def _parse_choices(
    value: str,
    *,
    choices: Sequence[str],
    name: str,
) -> tuple[str, ...]:
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    if not selected:
        raise argparse.ArgumentTypeError(f"at least one {name} is required")
    invalid = sorted(set(selected) - set(choices))
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unsupported {name}: {', '.join(invalid)}"
        )
    if len(set(selected)) != len(selected):
        raise argparse.ArgumentTypeError(f"duplicate {name} values are not allowed")
    return selected


def _seeds_for_run(run_index: int, n_seeds: int) -> tuple[int, ...]:
    offset = run_index * (n_seeds + 100)
    return tuple(range(offset + 1, offset + n_seeds + 1))


def _unit_specs(
    n_runs_outer: int,
    n_seeds: int,
) -> tuple[tuple[int, str, int], ...]:
    return tuple(
        (run_index, condition, seed)
        for run_index in range(n_runs_outer)
        for condition in CONDITIONS
        for seed in _seeds_for_run(run_index, n_seeds)
    )


def _align_events_to_observation_grid(
    events: np.ndarray,
    downsample: int,
) -> np.ndarray:
    """Map latent events to the first retained frame at or after each event."""

    values = np.asarray(events, dtype=float)
    if downsample < 1:
        raise ValueError("downsample must be positive")
    if downsample == 1:
        return values.copy()
    aligned = np.zeros((values.shape[0], values[:, ::downsample].shape[1]))
    event_rois, event_times = np.nonzero(values)
    retained_times = (event_times + downsample - 1) // downsample
    within_grid = retained_times < aligned.shape[1]
    aligned[event_rois[within_grid], retained_times[within_grid]] = values[
        event_rois[within_grid], event_times[within_grid]
    ]
    return aligned


def matched_static_dataset(
    *,
    run_index: int,
    condition: str,
    seed: int,
    n_steps: int,
    n_seeds: int,
) -> dict[str, Any]:
    """Recreate one exact input unit from both static c-GC notebooks."""

    network_seed = BASE_SEED + run_index * 10
    n_rois = int(np.random.RandomState(network_seed).uniform(6, 13))
    truth, sides, positions, middle = generate_static_validation_network(
        n_rois,
        seed=network_seed,
        ipsilateral_fraction=IPSILATERAL_FRACTION,
    )
    config = _condition_config(condition)
    dataset = simulate_static_calcium_like(
        truth,
        n_steps,
        rise_tau=float(config["rise_tau"]),
        decay_tau=float(config["decay_tau"]),
        noise_std=float(config["noise_std"]),
        shared_noise_std=float(config["shared_noise_std"]),
        spontaneous_rate=SPONTANEOUS_RATE,
        transmission_probability=TRANSMISSION_PROBABILITY,
        amplitude_jitter=AMP_JITTER,
        smooth_window=SMOOTH_WINDOW,
        propagation_delay=PROPAGATION_DELAY,
        random_state=seed,
    )
    downsample = int(config["downsample"])
    traces = dataset.fluorescence[:, ::downsample]
    events = _align_events_to_observation_grid(dataset.events, downsample)
    seeds = _seeds_for_run(run_index, n_seeds)
    permutation = np.random.default_rng(BASE_SEED + run_index).permutation(seeds)
    calibration_count = max(1, len(seeds) // 2)
    calibration_seeds = {int(value) for value in permutation[:calibration_count]}
    return {
        "truth": truth,
        "traces": traces,
        "events": events,
        "sides": sides,
        "positions": positions,
        "middle": middle,
        "condition_config": config,
        "split": "calibration" if seed in calibration_seeds else "evaluation",
        "input_digest": static_input_digest(truth, traces),
    }


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
    unit_metadata: dict[str, Any],
    truth_events: np.ndarray,
    representations: dict[str, np.ndarray],
    calibration_frames: int,
    tolerance: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in ("deconvolved", "oasis"):
        scores = representations[method]
        for threshold_rule, thresholds in (
            ("positive", np.zeros(scores.shape[0], dtype=float)),
            ("held_out_mad3", _mad_thresholds(scores, calibration_frames)),
        ):
            rows.append(
                {
                    **unit_metadata,
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
    representation: np.ndarray,
    *,
    method: str,
    n_surrogates: int,
    seed: int,
) -> np.ndarray:
    estimator = CausalisedGC(
        max_lag=CGC_N_LAGS,
        n_surrogates=n_surrogates,
        alpha=CGC_ALPHA,
        beta=CGC_BETA,
        fdr=CGC_FDR,
        event_mode="compressed",
        method=method,
        simulation=True,
        n_pasts=CGC_N_PASTS,
        random_state=seed,
    )
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
    run_index: int,
    condition: str,
    seed: int,
    n_steps: int,
    n_seeds: int,
    n_cgc_surrogates: int,
    lpcmci_tau_max: int,
    lpcmci_pc_alpha: float,
    event_tolerance: int,
    pag_dir: Path,
    components: Sequence[str] = COMPONENTS,
    representation_names: Sequence[str] = REPRESENTATIONS,
    cgc_methods: Sequence[str] = CGC_METHODS,
    progress: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matched = matched_static_dataset(
        run_index=run_index,
        condition=condition,
        seed=seed,
        n_steps=n_steps,
        n_seeds=n_seeds,
    )
    truth = np.asarray(matched["truth"], dtype=bool)
    traces = np.asarray(matched["traces"], dtype=float)
    truth_events = np.asarray(matched["events"], dtype=float)
    condition_config = matched["condition_config"]
    downsample = int(condition_config["downsample"])
    bundle = build_representations(
        traces,
        tolerance=REPRESENTATION_TOLERANCE,
        gamma=static_gamma_from_tau(float(condition_config["decay_tau"]))
        ** downsample,
    )
    representations = {
        "full": bundle.full,
        "deconvolved": bundle.deconvolved,
        "rise": bundle.rise,
        "fall": bundle.fall,
        "fall_residual": bundle.fall_residual,
    }
    if "oasis" in components or "oasis" in representation_names:
        oasis_result = OASISDeconvolver(
            backend_kwargs={"penalty": 1, "optimize_g": 0}
        ).fit(traces)
        representations["oasis"] = oasis_result.spikes
    selected_representations = {
        name: representations[name] for name in representation_names
    }
    unit_metadata = {
        "run": run_index,
        "condition": condition,
        "seed": seed,
        "split": matched["split"],
        "input_digest": matched["input_digest"],
        "input_contract": "static-cgc-grid-v1",
        "n_rois": traces.shape[0],
        "n_timepoints": traces.shape[1],
        "true_w_ic": graph_summary(truth, int(matched["middle"]))["w_ic"],
    }
    calibration_frames = max(20, traces.shape[1] // 4)
    event_rows = (
        _event_rows(
            unit_metadata=unit_metadata,
            truth_events=truth_events,
            representations=representations,
            calibration_frames=calibration_frames,
            tolerance=event_tolerance,
        )
        if "oasis" in components
        else []
    )

    graph_rows: list[dict[str, Any]] = []
    for representation_index, representation_name in enumerate(representation_names):
        representation = selected_representations[representation_name]
        if "oasis" in components:
            for method_index, method in enumerate(cgc_methods):
                cgc_adjacency = _cgc_graph(
                    representation,
                    method=method,
                    n_surrogates=n_cgc_surrogates,
                    seed=(
                        200_000
                        + run_index * 10_000
                        + seed * 100
                        + representation_index * 10
                        + method_index
                    ),
                )
                cgc_recovery = edge_recovery(truth, cgc_adjacency)
                graph_rows.append(
                    {
                        **unit_metadata,
                        "method": method,
                        "representation": representation_name,
                        "output_semantics": "directed_adjacency",
                        "projection": "none",
                        "comparison_metric": "directed_edge_recovery",
                        "precision": cgc_recovery.precision,
                        "recall": cgc_recovery.recall,
                        "false_positive_rate": cgc_recovery.false_positive_rate,
                        "orientation_accuracy": cgc_recovery.orientation_accuracy,
                        "retained_edges": int(np.count_nonzero(cgc_adjacency)),
                        "recovered_w_ic": graph_summary(
                            cgc_adjacency,
                            int(matched["middle"]),
                        )["w_ic"],
                        "n_estimator_surrogates": n_cgc_surrogates,
                        "alpha": CGC_ALPHA,
                        "beta": CGC_BETA,
                        "n_pasts": CGC_N_PASTS,
                        "n_lags": CGC_N_LAGS,
                        "fdr": CGC_FDR,
                    }
                )

        if "lpcmci" in components:
            fit_started_at = time.perf_counter()
            if progress:
                print(
                    "[lpcmci] starting "
                    f"representation {representation_index + 1}/{len(representation_names)}: "
                    f"{representation_name} "
                    f"(rois={representation.shape[0]}, timepoints={representation.shape[1]})",
                    flush=True,
                )
            lpcmci = LPCMCIAdapter(
                tau_max=lpcmci_tau_max,
                run_kwargs={"tau_min": 0, "pc_alpha": lpcmci_pc_alpha},
                verbosity=1 if progress else 0,
            ).fit(representation)
            pag_path = (
                pag_dir
                / (
                    f"run-{run_index}__{condition}__seed-{seed}__"
                    f"{representation_name}.npz"
                )
            )
            _atomic_write_npz(
                pag_path,
                graph=lpcmci.graph,
                p_matrix=lpcmci.p_matrix,
                val_matrix=lpcmci.val_matrix,
                input_digest=np.asarray(matched["input_digest"]),
            )
            if progress:
                print(
                    f"[lpcmci] finished in {time.perf_counter() - fit_started_at:.1f}s; "
                    f"saved representation {representation_index + 1}/"
                    f"{len(representation_names)}: {pag_path}",
                    flush=True,
                )
            projected = lpcmci.lossy_lagged_skeleton()
            skeleton_metrics = _skeleton_metrics(truth, projected)
            graph_rows.append(
                {
                    **unit_metadata,
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


def _config(
    args: argparse.Namespace,
    *,
    components: Sequence[str],
    representations: Sequence[str],
    cgc_methods: Sequence[str],
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "resume_schema_version": 5 if "oasis" in components else 4,
        "input_contract": "static-cgc-grid-v1",
        "conditions": list(CONDITIONS),
        "base_seed": BASE_SEED,
        "n_runs_outer": args.n_runs_outer,
        "n_seeds_per_run": args.n_seeds,
        "n_steps": args.n_steps,
        "simulation": {
            "rise_tau": RISE_TAU,
            "decay_tau": DECAY_TAU,
            "noise_std": NOISE_STD,
            "amplitude_jitter": AMP_JITTER,
            "smooth_window": SMOOTH_WINDOW,
            "spontaneous_rate": SPONTANEOUS_RATE,
            "transmission_probability": TRANSMISSION_PROBABILITY,
            "representation_tolerance": REPRESENTATION_TOLERANCE,
            "ipsilateral_fraction": IPSILATERAL_FRACTION,
            "propagation_delay": PROPAGATION_DELAY,
        },
        "n_cgc_surrogates": args.n_cgc_surrogates,
        "cgc_methods": list(cgc_methods),
        "lpcmci_tau_max": args.lpcmci_tau_max,
        "lpcmci_pc_alpha": args.lpcmci_pc_alpha,
        "event_tolerance": args.event_tolerance,
        "components": list(components),
        "representations": list(representations),
        "oasis": {"penalty": 1, "optimize_g": 0, "kinetics": "auto_ar1"},
        "pag_projection_policy": "raw PAG is primary; lagged skeleton is lossy",
    }
    if "oasis" in components:
        config["cgc"] = {
            "alpha": CGC_ALPHA,
            "beta": CGC_BETA,
            "n_pasts": CGC_N_PASTS,
            "n_lags": CGC_N_LAGS,
            "fdr": CGC_FDR,
            "simulation": True,
        }
    return config


def _archive_incompatible_output(output_dir: Path, saved_config: Any) -> Path:
    """Preserve an incompatible run beside a newly initialized output directory."""

    digest = hashlib.sha256(
        json.dumps(saved_config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    candidate = output_dir.with_name(f"{output_dir.name}.incompatible-{digest}")
    suffix = 1
    while candidate.exists():
        candidate = output_dir.with_name(
            f"{output_dir.name}.incompatible-{digest}-{suffix}"
        )
        suffix += 1
    output_dir.replace(candidate)
    output_dir.mkdir(parents=True, exist_ok=False)
    return candidate


def _unit_key(run_index: int, condition: str, seed: int) -> str:
    return f"{run_index}|{condition}|{seed}"


def _recover_completed_units(
    *,
    event_rows: Sequence[dict[str, Any]],
    graph_rows: Sequence[dict[str, Any]],
    unit_specs: Sequence[tuple[int, str, int]],
    components: Sequence[str] = COMPONENTS,
    representations: Sequence[str] = REPRESENTATIONS,
    cgc_methods: Sequence[str] = CGC_METHODS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    expected_units = {
        _unit_key(run_index, condition, seed)
        for run_index, condition, seed in unit_specs
    }
    event_grouped: dict[str, list[dict[str, Any]]] = {}
    for row in event_rows:
        unit = _unit_key(
            int(row["run"]), str(row["condition"]), int(row["seed"])
        )
        event_grouped.setdefault(unit, []).append(row)
    graph_grouped: dict[str, list[dict[str, Any]]] = {}
    for row in graph_rows:
        unit = _unit_key(
            int(row["run"]), str(row["condition"]), int(row["seed"])
        )
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
            _valid_pag_file(
                Path(str(row["raw_pag_path"])),
                input_digest=str(row.get("input_digest")),
            )
            for row in unit_graph_rows
            if row["method"] == "lpcmci"
        )
        expected_event_keys = (
            {
                (method, rule)
                for method in ("deconvolved", "oasis")
                for rule in ("positive", "held_out_mad3")
            }
            if "oasis" in components
            else set()
        )
        expected_methods = (
            (tuple(cgc_methods) if "oasis" in components else ())
            + (("lpcmci",) if "lpcmci" in components else ())
        )
        expected_graph_keys = {
            (method, representation)
            for method in expected_methods
            for representation in representations
        }
        if (
            event_keys == expected_event_keys
            and graph_keys == expected_graph_keys
            and len(unit_event_rows) == len(expected_event_keys)
            and len(unit_graph_rows) == len(expected_graph_keys)
            and pag_files_present
            and len(
                {
                    str(row.get("input_digest"))
                    for row in (*unit_event_rows, *unit_graph_rows)
                }
            )
            == 1
        ):
            completed.add(unit)

    recovered_event_rows = [
        row
        for row in event_rows
        if _unit_key(
            int(row["run"]), str(row["condition"]), int(row["seed"])
        )
        in completed
    ]
    recovered_graph_rows = [
        row
        for row in graph_rows
        if _unit_key(
            int(row["run"]), str(row["condition"]), int(row["seed"])
        )
        in completed
    ]
    return recovered_event_rows, recovered_graph_rows, completed


def _write_progress(
    output_dir: Path,
    *,
    config: dict[str, Any],
    completed_units: set[str],
    status: str,
    active_unit: str | None = None,
) -> None:
    payload = {
        "status": status,
        "config": config,
        "completed_units": sorted(completed_units),
        "completed_unit_count": len(completed_units),
        "active_unit": active_unit,
        "expected_unit_count": (
            len(config["conditions"])
            * int(config["n_runs_outer"])
            * int(config["n_seeds_per_run"])
        ),
    }
    _atomic_write_text(
        output_dir / PROGRESS_FILE,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-runs-outer", type=int, default=10)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--n-steps", type=int, default=3000)
    parser.add_argument("--n-cgc-surrogates", type=int, default=1000)
    parser.add_argument("--lpcmci-tau-max", type=int, default=2)
    parser.add_argument("--lpcmci-pc-alpha", type=float, default=0.05)
    parser.add_argument("--event-tolerance", type=int, default=2)
    parser.add_argument(
        "--components",
        default=",".join(COMPONENTS),
        help="Comma-separated baseline components: oasis,lpcmci",
    )
    parser.add_argument(
        "--representations",
        default=",".join(REPRESENTATIONS),
        help="Comma-separated trace representations",
    )
    parser.add_argument(
        "--cgc-methods",
        default=",".join(CGC_METHODS),
        help="Downstream methods for OASIS preprocessing: cgc,cgc-star",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--restart-incompatible-resume",
        action="store_true",
        help=(
            "When --resume finds a different saved configuration, preserve the old "
            "output directory beside the new run and restart from zero"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    components = _parse_choices(
        args.components,
        choices=COMPONENTS,
        name="component",
    )
    representations = _parse_choices(
        args.representations,
        choices=REPRESENTATION_CHOICES,
        name="representation",
    )
    cgc_methods = _parse_choices(
        args.cgc_methods,
        choices=CGC_METHODS,
        name="c-GC method",
    )
    if "oasis" in components and "oasis" not in representations:
        raise SystemExit("the OASIS component requires the oasis representation")
    if args.n_runs_outer < 1 or args.n_seeds < 2 or args.n_steps < 3:
        raise SystemExit(
            "--n-runs-outer must be positive, --n-seeds at least two, and "
            "--n-steps at least three"
        )
    if args.n_cgc_surrogates < 1:
        raise SystemExit("--n-cgc-surrogates must be positive")
    if args.lpcmci_tau_max < 1:
        raise SystemExit("--lpcmci-tau-max must be at least one")
    if not 0.0 < args.lpcmci_pc_alpha < 1.0:
        raise SystemExit("--lpcmci-pc-alpha must lie in (0, 1)")
    if args.event_tolerance < 0:
        raise SystemExit("--event-tolerance cannot be negative")
    config = _config(
        args,
        components=components,
        representations=representations,
        cgc_methods=cgc_methods,
    )
    unit_specs = _unit_specs(args.n_runs_outer, args.n_seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / PROGRESS_FILE
    event_partial = args.output_dir / EVENT_PARTIAL_FILE
    graph_partial = args.output_dir / GRAPH_PARTIAL_FILE
    completed_units: set[str] = set()
    event_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    previous_active_unit: str | None = None
    total_units = len(unit_specs)
    total_lpcmci_fits = total_units * len(representations)
    print(
        "[plan] "
        f"{total_units} checkpoint units; "
        f"{total_lpcmci_fits if 'lpcmci' in components else 0} LPCMCI fits; "
        f"output={args.output_dir}",
        flush=True,
    )
    if args.resume and progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get("config") != config:
            if not args.restart_incompatible_resume:
                raise SystemExit("resume configuration does not match the saved run")
            archive_dir = _archive_incompatible_output(
                args.output_dir, progress.get("config")
            )
            print(
                "[resume] preserved incompatible output at "
                f"{archive_dir}; restarting with the current configuration",
                flush=True,
            )
            progress_path = args.output_dir / PROGRESS_FILE
            event_partial = args.output_dir / EVENT_PARTIAL_FILE
            graph_partial = args.output_dir / GRAPH_PARTIAL_FILE
            progress = {}
        else:
            previous_active_unit = progress.get("active_unit")
            event_rows = _read_csv(event_partial)
            graph_rows = _read_csv(graph_partial)
            event_rows, graph_rows, completed_units = _recover_completed_units(
                event_rows=event_rows,
                graph_rows=graph_rows,
                unit_specs=unit_specs,
                components=components,
                representations=representations,
                cgc_methods=cgc_methods,
            )
        print(
            f"[resume] loaded and validated {len(completed_units)}/{total_units} "
            "complete checkpoint units; completed units will be skipped",
            flush=True,
        )
        if previous_active_unit and previous_active_unit not in completed_units:
            print(
                f"[resume] previous run stopped during {previous_active_unit}; "
                "that whole outer-run/condition/seed unit will be recomputed",
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
        format_progress(len(completed_units), total_units, label="Overall units"),
        flush=True,
    )

    _write_progress(
        args.output_dir,
        config=config,
        completed_units=completed_units,
        status="running",
    )
    for run_index, condition, seed in unit_specs:
        unit = _unit_key(run_index, condition, seed)
        if unit in completed_units:
            continue
        print(
            f"[unit] starting {len(completed_units) + 1}/{total_units}: "
            f"run={run_index}, condition={condition}, seed={seed}",
            flush=True,
        )
        _write_progress(
            args.output_dir,
            config=config,
            completed_units=completed_units,
            status="running",
            active_unit=unit,
        )
        new_event_rows, new_graph_rows = rows_for_unit(
            run_index=run_index,
            condition=condition,
            seed=seed,
            n_steps=args.n_steps,
            n_seeds=args.n_seeds,
            n_cgc_surrogates=args.n_cgc_surrogates,
            lpcmci_tau_max=args.lpcmci_tau_max,
            lpcmci_pc_alpha=args.lpcmci_pc_alpha,
            event_tolerance=args.event_tolerance,
            pag_dir=args.output_dir / "raw_pag",
            components=components,
            representation_names=representations,
            cgc_methods=cgc_methods,
            progress=True,
        )
        event_rows.extend(new_event_rows)
        graph_rows.extend(new_graph_rows)
        completed_units.add(unit)
        if event_rows:
            _write_csv(event_partial, event_rows)
        _write_csv(graph_partial, graph_rows)
        _write_progress(
            args.output_dir,
            config=config,
            completed_units=completed_units,
            status="running",
        )
        print(
            format_progress(
                len(completed_units),
                total_units,
                label="Overall units",
            )
            + f" | completed {unit}",
            flush=True,
        )

    if event_rows:
        _write_csv(args.output_dir / "event_recovery_rows.csv", event_rows)
    _write_csv(args.output_dir / "graph_recovery_rows.csv", graph_rows)
    expected_methods = (
        (len(cgc_methods) if "oasis" in components else 0)
        + (1 if "lpcmci" in components else 0)
    )
    expected_graph_rows = total_units * len(representations) * expected_methods
    expected_event_rows = total_units * 4 if "oasis" in components else 0
    if len(graph_rows) != expected_graph_rows or len(event_rows) != expected_event_rows:
        raise RuntimeError(
            "Completed baseline row count does not match the declared grid: "
            f"graph {len(graph_rows)}/{expected_graph_rows}, "
            f"event {len(event_rows)}/{expected_event_rows}"
        )
    manifest_by_unit: dict[str, dict[str, Any]] = {}
    for row in graph_rows:
        unit = _unit_key(
            int(row["run"]), str(row["condition"]), int(row["seed"])
        )
        manifest_by_unit.setdefault(
            unit,
            {
                "run": row["run"],
                "condition": row["condition"],
                "seed": row["seed"],
                "split": row["split"],
                "n_rois": row["n_rois"],
                "n_timepoints": row["n_timepoints"],
                "input_digest": row["input_digest"],
                "input_contract": row["input_contract"],
            },
        )
    _write_csv(
        args.output_dir / "input_manifest.csv",
        list(manifest_by_unit.values()),
    )
    interpretation: dict[str, str] = {
        "matched_input": (
            "input digests identify the exact static c-GC/c-GC* grid units"
        ),
        "shared_input": (
            "the shared_input condition is the same common-noise condition used "
            "by both static c-GC notebooks"
        ),
    }
    if "oasis" in components:
        interpretation["oasis"] = (
            "event estimator and preprocessing ablation, not a causal learner"
        )
    if "lpcmci" in components:
        interpretation["lpcmci"] = (
            "raw PAG retained; binary lagged projection is explicitly lossy"
        )
    summary = {
        "status": "complete",
        "config": config,
        "expected_unit_count": total_units,
        "expected_event_rows": expected_event_rows,
        "expected_graph_rows": expected_graph_rows,
        "n_event_rows": len(event_rows),
        "n_graph_rows": len(graph_rows),
        "interpretation": interpretation,
        "outputs": (
            (["event_recovery_rows.csv"] if event_rows else [])
            + ["graph_recovery_rows.csv", "input_manifest.csv"]
            + (["raw_pag/*.npz"] if "lpcmci" in components else [])
        ),
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
    print(
        format_progress(total_units, total_units, label="Overall units")
        + f" | complete; summary={args.output_dir / 'summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
