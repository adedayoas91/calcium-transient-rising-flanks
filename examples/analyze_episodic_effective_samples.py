"""Quantify event-selection sample trade-offs in the episodic benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from calcium_transient_rising_flank import (
    DynamicSimulationConfig,
    build_representations,
    selected_frame_indices,
    simulate_calcium_dataset,
)


DEFAULT_GRID_PATH = Path(
    "outputs/validation_results/dynamic_episodic_locked/dynamic_grid_runs.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/method_validation/episodic_effective_samples"
)
REPRESENTATIONS = ("rise", "fall", "fall_residual")


def _truth_graphs() -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    first = np.array(
        [
            [False, True, False, False, False],
            [False, False, True, False, False],
            [False, False, False, True, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
        ]
    )
    second = np.array(
        [
            [False, False, False, False, True],
            [False, False, False, False, False],
            [False, True, False, False, False],
            [False, False, True, False, False],
            [False, False, False, False, False],
        ]
    )
    return first | second, (first, second)


def _active_node_sequence() -> tuple[np.ndarray, ...]:
    return (
        np.array([True, True, True, True, False]),
        np.array([True, True, True, True, True]),
    )


def _dynamic_config(sequence: tuple[np.ndarray, ...]) -> DynamicSimulationConfig:
    return DynamicSimulationConfig(
        n_episodes=3,
        min_rise_length=20,
        max_rise_length=28,
        rise_waveform_length=20,
        fall_to_rise_ratio_min=2.1,
        fall_to_rise_ratio_max=2.8,
        initial_activation_probability=0.7,
        fall_noise_rate=0.05,
        fall_noise_scale=0.02,
        fall_state_mode="stochastic_independent",
        fall_initial_scale=1.0,
        fall_initial_ceiling_fraction=1.0,
        edge_dropout_probability=0.0,
        adjacency_sequence=sequence,
        active_node_sequence=_active_node_sequence(),
    )


def _segment_ids(dataset: Any, phase: str) -> np.ndarray | None:
    if not dataset.episodes:
        return None
    values = np.full(dataset.fluorescence.shape[1], -1, dtype=int)
    for index, episode in enumerate(dataset.episodes, start=1):
        if phase == "rise":
            values[episode.rise_start : episode.rise_stop] = index
        else:
            values[episode.fall_start : episode.fall_stop] = index
    return values


def _ar1_effective_sample_proxy(x: np.ndarray, y: np.ndarray) -> float:
    """Return a bounded AR(1) approximation, not an exact independent-sample ESS."""

    count = int(min(x.size, y.size))
    if count < 3:
        return float(count)

    def lag1(values: np.ndarray) -> float:
        left = values[:-1] - np.mean(values[:-1])
        right = values[1:] - np.mean(values[1:])
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator <= np.finfo(float).eps:
            return 0.0
        return float(np.dot(left, right) / denominator)

    product = float(np.clip(lag1(x) * lag1(y), -0.999, 0.999))
    estimate = count * (1.0 - product) / (1.0 + product)
    return float(np.clip(estimate, 1.0, count))


def _compressed_samples(
    fluorescence: np.ndarray,
    selected: tuple[np.ndarray, ...],
    source: int,
    target: int,
    *,
    n_pasts: int,
    lag: int,
    segment_ids: np.ndarray | None,
) -> dict[str, Any]:
    common = np.intersect1d(selected[source], selected[target])
    if common.size <= n_pasts + 1:
        return {
            "common_selected_frames": int(common.size),
            "usable_samples": 0,
            "ar1_effective_sample_proxy": 0.0,
            "lag_pairs": 0,
            "one_frame_lag_pairs": 0,
            "same_segment_lag_pairs": 0,
            "estimable": False,
        }
    target_times = common[n_pasts:]
    source_start = n_pasts - lag
    source_stop = -lag if lag else None
    source_times = common[source_start:source_stop]
    if source_times.size != target_times.size:
        raise RuntimeError("compressed source and target samples are misaligned")
    one_frame = (target_times - source_times) == lag
    if segment_ids is None:
        same_segment = np.ones(target_times.size, dtype=bool)
    else:
        same_segment = (
            (segment_ids[target_times] >= 0)
            & (segment_ids[source_times] >= 0)
            & (segment_ids[target_times] == segment_ids[source_times])
        )
    x = fluorescence[source, source_times]
    y = fluorescence[target, target_times]
    return {
        "common_selected_frames": int(common.size),
        "usable_samples": int(target_times.size),
        "ar1_effective_sample_proxy": _ar1_effective_sample_proxy(x, y),
        "lag_pairs": int(target_times.size),
        "one_frame_lag_pairs": int(np.count_nonzero(one_frame)),
        "same_segment_lag_pairs": int(np.count_nonzero(same_segment)),
        "estimable": target_times.size >= 2,
    }


def _physical_samples(
    fluorescence: np.ndarray,
    selected: tuple[np.ndarray, ...],
    source: int,
    target: int,
    *,
    n_pasts: int,
    lag: int,
    segment_ids: np.ndarray | None,
) -> dict[str, Any]:
    n_steps = fluorescence.shape[1]
    target_mask = np.zeros(n_steps, dtype=bool)
    source_mask = np.zeros(n_steps, dtype=bool)
    target_mask[selected[target]] = True
    source_mask[selected[source]] = True
    times = np.arange(max(n_pasts, lag), n_steps, dtype=int)
    valid = target_mask[times] & source_mask[times - lag]
    if segment_ids is not None:
        valid &= segment_ids[times] >= 0
        valid &= segment_ids[times - lag] >= 0
        valid &= segment_ids[times] == segment_ids[times - lag]
    times = times[valid]
    x = fluorescence[source, times - lag]
    y = fluorescence[target, times]
    return {
        "usable_samples": int(times.size),
        "ar1_effective_sample_proxy": _ar1_effective_sample_proxy(x, y),
        "estimable": times.size >= 2,
    }


def pair_sample_row(
    fluorescence: np.ndarray,
    selected: tuple[np.ndarray, ...],
    source: int,
    target: int,
    *,
    truth: np.ndarray,
    n_pasts: int,
    lag: int,
    segment_ids: np.ndarray | None,
) -> dict[str, Any]:
    """Compare compressed and physical samples for one ordered ROI pair."""

    compressed = _compressed_samples(
        fluorescence,
        selected,
        source,
        target,
        n_pasts=n_pasts,
        lag=lag,
        segment_ids=segment_ids,
    )
    physical = _physical_samples(
        fluorescence,
        selected,
        source,
        target,
        n_pasts=n_pasts,
        lag=lag,
        segment_ids=segment_ids,
    )
    return {
        "source": source,
        "target": target,
        "true_edge": bool(truth[source, target]),
        "source_selected_frames": int(selected[source].size),
        "target_selected_frames": int(selected[target].size),
        "common_selected_frames": compressed["common_selected_frames"],
        "compressed_usable_samples": compressed["usable_samples"],
        "compressed_ar1_effective_sample_proxy": compressed[
            "ar1_effective_sample_proxy"
        ],
        "compressed_estimable": compressed["estimable"],
        "compressed_lag_pairs": compressed["lag_pairs"],
        "compressed_one_frame_lag_pairs": compressed["one_frame_lag_pairs"],
        "compressed_same_segment_lag_pairs": compressed[
            "same_segment_lag_pairs"
        ],
        "physical_usable_samples": physical["usable_samples"],
        "physical_ar1_effective_sample_proxy": physical[
            "ar1_effective_sample_proxy"
        ],
        "physical_estimable": physical["estimable"],
    }


def _phase_selected_fraction(
    selected: tuple[np.ndarray, ...], segment_ids: np.ndarray | None
) -> float:
    total = sum(frames.size for frames in selected)
    if total == 0:
        return 0.0
    if segment_ids is None:
        return 1.0
    retained = sum(np.count_nonzero(segment_ids[frames] >= 0) for frames in selected)
    return float(retained / total)


def build_pair_rows(
    *,
    n_seeds: int = 20,
    n_steps: int = 1500,
    n_pasts: int = 3,
    lag: int = 1,
) -> list[dict[str, Any]]:
    adjacency, sequence = _truth_graphs()
    conditions = (
        ("static_union", "static", None),
        ("dynamic_a_noncausal_fall", "episodic_dynamic", _dynamic_config(sequence)),
    )
    rows: list[dict[str, Any]] = []
    for condition, simulator_mode, dynamic_config in conditions:
        for seed in range(1, n_seeds + 1):
            dataset = simulate_calcium_dataset(
                adjacency,
                n_steps=n_steps,
                gamma=0.9,
                noise_std=0.05,
                shared_noise_std=0.0,
                spontaneous_rate=0.025,
                transmission_probability=0.8,
                random_state=seed,
                simulator_mode=simulator_mode,
                dynamic_config=dynamic_config,
            )
            representations = build_representations(
                dataset.fluorescence,
                tolerance=0.0,
                gamma=0.9,
            ).as_dict()
            for representation in REPRESENTATIONS:
                selected = selected_frame_indices(representations[representation])
                phase = "rise" if representation == "rise" else "fall"
                segments = _segment_ids(dataset, phase)
                truth = (
                    np.asarray(dataset.adjacency, dtype=bool)
                    if not dataset.episodes or representation == "rise"
                    else np.zeros_like(dataset.adjacency, dtype=bool)
                )
                phase_fraction = _phase_selected_fraction(selected, segments)
                for source in range(dataset.fluorescence.shape[0]):
                    for target in range(dataset.fluorescence.shape[0]):
                        if source == target:
                            continue
                        rows.append(
                            {
                                "condition": condition,
                                "simulator_mode": simulator_mode,
                                "seed": seed,
                                "representation": representation,
                                "n_rois": dataset.fluorescence.shape[0],
                                "n_steps": dataset.fluorescence.shape[1],
                                "n_pasts": n_pasts,
                                "tested_lag": lag,
                                "selected_roi_frames_total": int(
                                    sum(frames.size for frames in selected)
                                ),
                                "selected_roi_frame_fraction": float(
                                    sum(frames.size for frames in selected)
                                    / dataset.fluorescence.size
                                ),
                                "selected_frames_in_declared_phase_fraction": (
                                    phase_fraction
                                ),
                                **pair_sample_row(
                                    dataset.fluorescence,
                                    selected,
                                    source,
                                    target,
                                    truth=truth,
                                    n_pasts=n_pasts,
                                    lag=lag,
                                    segment_ids=segments,
                                ),
                            }
                        )
    return rows


def _safe_mean(values: Iterable[float]) -> float | None:
    array = np.asarray(tuple(values), dtype=float)
    array = array[np.isfinite(array)]
    return None if array.size == 0 else float(np.mean(array))


def _safe_median(values: Iterable[float]) -> float | None:
    array = np.asarray(tuple(values), dtype=float)
    array = array[np.isfinite(array)]
    return None if array.size == 0 else float(np.median(array))


def _conditioning_parameter_counts(
    n_rois: int,
    n_pasts: int,
    lag: int,
) -> tuple[int, int]:
    """Return regression parameter counts, including the intercept."""

    cgc_covariates = (
        (n_pasts - lag)
        + n_pasts
        + (n_rois - 2) * (n_pasts - lag + 1)
    )
    cgc_star_covariates = (n_pasts + 1) * n_rois - 2
    return cgc_covariates + 1, cgc_star_covariates + 1


def summarize_samples(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        grouped[(row["condition"], row["seed"], row["representation"])].append(row)
    summaries: list[dict[str, Any]] = []
    for (condition, seed, representation), group in sorted(grouped.items()):
        true_rows = [row for row in group if row["true_edge"]]
        nonedge_rows = [row for row in group if not row["true_edge"]]
        compressed_lag_pairs = sum(int(row["compressed_lag_pairs"]) for row in group)
        cgc_parameters, cgc_star_parameters = _conditioning_parameter_counts(
            int(group[0]["n_rois"]),
            int(group[0]["n_pasts"]),
            int(group[0]["tested_lag"]),
        )
        for mode in ("compressed", "physical"):
            usable_key = f"{mode}_usable_samples"
            ess_key = f"{mode}_ar1_effective_sample_proxy"
            usable = np.asarray([row[usable_key] for row in group], dtype=float)
            estimable = np.asarray([row[f"{mode}_estimable"] for row in group])
            true_usable = [row[usable_key] for row in true_rows]
            nonedge_usable = [row[usable_key] for row in nonedge_rows]
            true_median = _safe_median(true_usable)
            nonedge_median = _safe_median(nonedge_usable)
            summaries.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "representation": representation,
                    "event_mode": mode,
                    "n_pairs": len(group),
                    "selected_roi_frames_total": group[0][
                        "selected_roi_frames_total"
                    ],
                    "selected_roi_frame_fraction": group[0][
                        "selected_roi_frame_fraction"
                    ],
                    "selected_frames_in_declared_phase_fraction": group[0][
                        "selected_frames_in_declared_phase_fraction"
                    ],
                    "usable_samples_pair_mean": _safe_mean(usable),
                    "usable_samples_pair_median": _safe_median(usable),
                    "ar1_effective_sample_proxy_pair_median": _safe_median(
                        row[ess_key] for row in group
                    ),
                    "insufficient_pair_fraction": float(1.0 - np.mean(estimable)),
                    "true_edge_usable_samples_median": true_median,
                    "nonedge_usable_samples_median": nonedge_median,
                    "true_minus_nonedge_usable_samples": (
                        None
                        if true_median is None or nonedge_median is None
                        else true_median - nonedge_median
                    ),
                    "cgc_regression_parameter_count": cgc_parameters,
                    "cgc_star_regression_parameter_count": cgc_star_parameters,
                    "pairs_at_or_below_cgc_parameter_count_fraction": float(
                        np.mean(usable <= cgc_parameters)
                    ),
                    "pairs_at_or_below_cgc_star_parameter_count_fraction": float(
                        np.mean(usable <= cgc_star_parameters)
                    ),
                    "compressed_one_frame_lag_pair_fraction": (
                        None
                        if mode != "compressed" or compressed_lag_pairs == 0
                        else sum(
                            int(row["compressed_one_frame_lag_pairs"])
                            for row in group
                        )
                        / compressed_lag_pairs
                    ),
                    "compressed_same_segment_lag_pair_fraction": (
                        None
                        if mode != "compressed" or compressed_lag_pairs == 0
                        else sum(
                            int(row["compressed_same_segment_lag_pairs"])
                            for row in group
                        )
                        / compressed_lag_pairs
                    ),
                    "effective_sample_interpretation": (
                        "AR(1)-adjusted proxy, bounded by usable lag-pair count"
                    ),
                }
            )
    return summaries


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def join_recovery(
    sample_rows: list[dict[str, Any]], grid_path: Path
) -> list[dict[str, Any]]:
    lookup = {
        (
            str(row["condition"]),
            int(row["seed"]),
            str(row["representation"]),
            str(row["event_mode"]),
        ): row
        for row in sample_rows
    }
    joined: list[dict[str, Any]] = []
    for recovery in _read_csv(grid_path):
        representation = recovery["representation"]
        if representation not in REPRESENTATIONS:
            continue
        key = (
            recovery["condition"],
            int(recovery["seed"]),
            representation,
            recovery["event_mode"],
        )
        if key not in lookup:
            raise ValueError(f"no sample audit row for archived recovery key {key}")
        joined.append(
            {
                **lookup[key],
                "method": recovery["method"],
                "precision": float(recovery["precision"]),
                "recall": float(recovery["recall"]),
                "false_positive_rate": float(recovery["false_positive_rate"]),
                "f1": float(recovery["f1"]),
                "false_negatives": int(float(recovery["false_negatives"])),
            }
        )
    return joined


def summarize_recovery(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["condition"],
                row["method"],
                row["representation"],
                row["event_mode"],
            )
        ].append(row)
    summaries: list[dict[str, Any]] = []
    for (condition, method, representation, mode), group in sorted(grouped.items()):
        sample = np.asarray(
            [row["ar1_effective_sample_proxy_pair_median"] for row in group],
            dtype=float,
        )
        f1 = np.asarray([row["f1"] for row in group], dtype=float)
        correlation = None
        if sample.size >= 3 and np.std(sample) > 0 and np.std(f1) > 0:
            correlation = float(np.corrcoef(sample, f1)[0, 1])
        summaries.append(
            {
                "condition": condition,
                "method": method,
                "representation": representation,
                "event_mode": mode,
                "n_seeds": len(group),
                "usable_samples_pair_median_mean": _safe_mean(
                    row["usable_samples_pair_median"] for row in group
                ),
                "ar1_effective_sample_proxy_pair_median_mean": _safe_mean(sample),
                "insufficient_pair_fraction_mean": _safe_mean(
                    row["insufficient_pair_fraction"] for row in group
                ),
                "selected_roi_frame_fraction_mean": _safe_mean(
                    row["selected_roi_frame_fraction"] for row in group
                ),
                "selected_frames_in_declared_phase_fraction_mean": _safe_mean(
                    row["selected_frames_in_declared_phase_fraction"]
                    for row in group
                ),
                "true_minus_nonedge_usable_samples_mean": _safe_mean(
                    row["true_minus_nonedge_usable_samples"]
                    for row in group
                    if row["true_minus_nonedge_usable_samples"] is not None
                ),
                "compressed_one_frame_lag_pair_fraction_mean": _safe_mean(
                    row["compressed_one_frame_lag_pair_fraction"]
                    for row in group
                    if row["compressed_one_frame_lag_pair_fraction"] is not None
                ),
                "compressed_same_segment_lag_pair_fraction_mean": _safe_mean(
                    row["compressed_same_segment_lag_pair_fraction"]
                    for row in group
                    if row["compressed_same_segment_lag_pair_fraction"] is not None
                ),
                "pairs_at_or_below_cgc_parameter_count_fraction_mean": _safe_mean(
                    row["pairs_at_or_below_cgc_parameter_count_fraction"]
                    for row in group
                ),
                "pairs_at_or_below_cgc_star_parameter_count_fraction_mean": (
                    _safe_mean(
                        row["pairs_at_or_below_cgc_star_parameter_count_fraction"]
                        for row in group
                    )
                ),
                "f1_mean": _safe_mean(f1),
                "f1_std": float(np.std(f1, ddof=1)),
                "false_negative_mean": _safe_mean(
                    row["false_negatives"] for row in group
                ),
                "effective_sample_proxy_f1_correlation": correlation,
            }
        )
    return summaries


def _bootstrap_mean_ci(
    values: np.ndarray, rng: np.random.Generator, n_bootstrap: int
) -> tuple[float, float]:
    if values.size == 1:
        return float(values[0]), float(values[0])
    draws = rng.choice(values, size=(n_bootstrap, values.size), replace=True).mean(1)
    lower, upper = np.quantile(draws, (0.025, 0.975))
    return float(lower), float(upper)


def _sign_flip_pvalue(
    differences: np.ndarray, rng: np.random.Generator, n_permutations: int
) -> float:
    observed = abs(float(np.mean(differences)))
    signs = rng.choice((-1.0, 1.0), size=(n_permutations, differences.size))
    null = np.abs(np.mean(signs * differences, axis=1))
    return float((1 + np.count_nonzero(null >= observed)) / (n_permutations + 1))


def _holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted: np.ndarray = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def paired_mode_contrasts(
    rows: list[dict[str, Any]],
    *,
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Compare physical minus compressed time on identical simulation seeds."""

    metrics = (
        "f1",
        "usable_samples_pair_median",
        "ar1_effective_sample_proxy_pair_median",
        "insufficient_pair_fraction",
    )
    indexed: dict[
        tuple[str, str, str], dict[str, dict[int, dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        group = (
            str(row["condition"]),
            str(row["method"]),
            str(row["representation"]),
        )
        indexed[group][str(row["event_mode"])][int(row["seed"])] = row

    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    for (condition, method, representation), modes in sorted(indexed.items()):
        compressed_seeds = set(modes["compressed"])
        if compressed_seeds != set(modes["physical"]):
            raise ValueError(
                f"unpaired event modes for {condition}, {method}, {representation}"
            )
        for metric in metrics:
            differences = np.asarray(
                [
                    float(modes["physical"][unit_seed][metric])
                    - float(modes["compressed"][unit_seed][metric])
                    for unit_seed in sorted(compressed_seeds)
                ],
                dtype=float,
            )
            lower, upper = _bootstrap_mean_ci(differences, rng, n_bootstrap)
            mean = float(np.mean(differences))
            std = (
                float(np.std(differences, ddof=1))
                if differences.size > 1
                else 0.0
            )
            output.append(
                {
                    "condition": condition,
                    "method": method,
                    "representation": representation,
                    "contrast": "physical_minus_compressed",
                    "metric": metric,
                    "n_pairs": differences.size,
                    "mean_paired_difference": mean,
                    "std_paired_difference": std,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "standardized_mean_difference": (
                        mean / std if std > 0.0 else None
                    ),
                    "p_value": _sign_flip_pvalue(
                        differences, rng, n_permutations
                    ),
                }
            )
    adjusted = _holm_adjust([float(row["p_value"]) for row in output])
    for row, p_adjusted in zip(output, adjusted):
        row["p_holm"] = p_adjusted
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    sample_rows: list[dict[str, Any]],
    joined: list[dict[str, Any]],
    path: Path,
    *,
    n_bootstrap: int,
    seed: int,
) -> None:
    selected = [
        row
        for row in sample_rows
        if row["condition"] == "dynamic_a_noncausal_fall"
        and row["representation"] == "rise"
    ]
    recovery = [
        row
        for row in joined
        if row["condition"] == "dynamic_a_noncausal_fall"
        and row["representation"] == "rise"
    ]
    modes = ("compressed", "physical")
    colors = {"compressed": "#b45f06", "physical": "#2f6f8f"}
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    rng = np.random.default_rng(seed)
    sample_arrays = [
        np.asarray(
            [
                row["ar1_effective_sample_proxy_pair_median"]
                for row in selected
                if row["event_mode"] == mode
            ],
            dtype=float,
        )
        for mode in modes
    ]
    sample_means = np.asarray([np.mean(values) for values in sample_arrays])
    sample_intervals = [
        _bootstrap_mean_ci(values, rng, n_bootstrap) for values in sample_arrays
    ]
    axes[0].bar(
        modes,
        sample_means,
        yerr=np.asarray(
            [
                sample_means - np.asarray([item[0] for item in sample_intervals]),
                np.asarray([item[1] for item in sample_intervals]) - sample_means,
            ]
        ),
        capsize=3,
        color=[colors[mode] for mode in modes],
    )
    axes[0].set_ylabel("Median AR(1)-adjusted sample proxy")
    axes[0].set_title("Rise sample support")

    x: np.ndarray = np.arange(2, dtype=float)
    width = 0.34
    for index, mode in enumerate(modes):
        arrays = [
            np.asarray(
                [
                    row["f1"]
                    for row in recovery
                    if row["event_mode"] == mode and row["method"] == method
                ],
                dtype=float,
            )
            for method in ("cgc", "cgc-star")
        ]
        values = np.asarray([np.mean(array) for array in arrays])
        intervals = [
            _bootstrap_mean_ci(array, rng, n_bootstrap) for array in arrays
        ]
        axes[1].bar(
            x + (index - 0.5) * width,
            values,
            width=width,
            yerr=np.asarray(
                [
                    values - np.asarray([item[0] for item in intervals]),
                    np.asarray([item[1] for item in intervals]) - values,
                ]
            ),
            capsize=3,
            label=mode,
            color=colors[mode],
        )
    axes[1].set_xticks(x, ("c-GC", "c-GC*"))
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("F1")
    axes[1].set_title("Archived rise recovery")
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_tradeoff(rows: list[dict[str, Any]], path: Path) -> None:
    selected = [
        row
        for row in rows
        if row["condition"] == "dynamic_a_noncausal_fall"
        and row["representation"] == "rise"
    ]
    colors = {"compressed": "#E69F00", "physical": "#0072B2"}
    markers = {"cgc": "o", "cgc-star": "s"}
    figure, axis = plt.subplots(figsize=(5.6, 4.2))
    for method in ("cgc", "cgc-star"):
        for mode in ("compressed", "physical"):
            group = [
                row
                for row in selected
                if row["method"] == method and row["event_mode"] == mode
            ]
            axis.scatter(
                [row["ar1_effective_sample_proxy_pair_median"] for row in group],
                [row["f1"] for row in group],
                color=colors[mode],
                marker=markers[method],
                alpha=0.72,
                label=f"{method}, {mode}",
            )
    axis.set_xlabel("Median AR(1)-adjusted sample proxy")
    axis.set_ylabel("Union-truth F1")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(color="#dddddd", linewidth=0.7)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-path", type=Path, default=DEFAULT_GRID_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--n-steps", type=int, default=1500)
    parser.add_argument("--n-pasts", type=int, default=3)
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=97)
    args = parser.parse_args()
    if args.n_bootstrap < 100 or args.n_permutations < 99:
        raise SystemExit("use at least 100 bootstrap samples and 99 permutations")
    return args


def main() -> None:
    args = parse_args()
    pair_rows = build_pair_rows(
        n_seeds=args.n_seeds,
        n_steps=args.n_steps,
        n_pasts=args.n_pasts,
        lag=args.lag,
    )
    sample_rows = summarize_samples(pair_rows)
    joined = join_recovery(sample_rows, args.grid_path)
    recovery_summary = summarize_recovery(joined)
    contrasts = paired_mode_contrasts(
        joined,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "episodic_effective_sample_pair_rows.csv", pair_rows)
    _write_csv(args.output_dir / "episodic_effective_sample_rows.csv", sample_rows)
    _write_csv(args.output_dir / "episodic_effective_sample_recovery.csv", joined)
    _write_csv(args.output_dir / "episodic_effective_sample_summary.csv", recovery_summary)
    _write_csv(args.output_dir / "episodic_mode_contrasts.csv", contrasts)
    _plot(
        sample_rows,
        joined,
        args.output_dir / "episodic_effective_sample_summary",
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 1,
    )
    _plot_tradeoff(joined, args.output_dir / "episodic_support_recovery_tradeoff")
    dynamic_rise = [
        row
        for row in recovery_summary
        if row["condition"] == "dynamic_a_noncausal_fall"
        and row["representation"] == "rise"
        and row["method"] == "cgc"
    ]
    by_mode = {row["event_mode"]: row for row in dynamic_rise}
    compressed = by_mode["compressed"]
    physical = by_mode["physical"]
    summary = {
        "status": "complete",
        "n_pair_rows": len(pair_rows),
        "n_sample_rows": len(sample_rows),
        "n_recovery_rows": len(joined),
        "n_summary_rows": len(recovery_summary),
        "n_contrast_rows": len(contrasts),
        "episodic_rise_headline": {
            "compressed_usable_samples_pair_median_mean": compressed[
                "usable_samples_pair_median_mean"
            ],
            "physical_usable_samples_pair_median_mean": physical[
                "usable_samples_pair_median_mean"
            ],
            "physical_usable_sample_reduction_fraction": 1.0
            - physical["usable_samples_pair_median_mean"]
            / compressed["usable_samples_pair_median_mean"],
            "compressed_ar1_effective_sample_proxy_pair_median_mean": compressed[
                "ar1_effective_sample_proxy_pair_median_mean"
            ],
            "physical_ar1_effective_sample_proxy_pair_median_mean": physical[
                "ar1_effective_sample_proxy_pair_median_mean"
            ],
            "physical_effective_sample_proxy_reduction_fraction": 1.0
            - physical["ar1_effective_sample_proxy_pair_median_mean"]
            / compressed["ar1_effective_sample_proxy_pair_median_mean"],
            "compressed_one_frame_lag_pair_fraction": compressed[
                "compressed_one_frame_lag_pair_fraction_mean"
            ],
            "compressed_same_segment_lag_pair_fraction": compressed[
                "compressed_same_segment_lag_pair_fraction_mean"
            ],
            "selected_frames_in_declared_rise_phase_fraction": compressed[
                "selected_frames_in_declared_phase_fraction_mean"
            ],
            "true_minus_nonedge_usable_samples_compressed": compressed[
                "true_minus_nonedge_usable_samples_mean"
            ],
            "true_minus_nonedge_usable_samples_physical": physical[
                "true_minus_nonedge_usable_samples_mean"
            ],
        },
        "interpretation": (
            "Usable lag-pair counts are exact for the implemented sampling rules. "
            "The AR(1)-adjusted quantity is a descriptive effective-sample proxy, "
            "not an exact count of independent observations."
        ),
        "outputs": [
            "episodic_effective_sample_pair_rows.csv",
            "episodic_effective_sample_rows.csv",
            "episodic_effective_sample_recovery.csv",
            "episodic_effective_sample_summary.csv",
            "episodic_mode_contrasts.csv",
            "episodic_effective_sample_summary.png",
            "episodic_effective_sample_summary.pdf",
            "episodic_support_recovery_tradeoff.png",
            "episodic_support_recovery_tradeoff.pdf",
            "analysis-report.md",
            "stats-appendix.md",
            "figure-catalog.md",
        ],
    }
    report = """# Episodic effective-sample analysis

The analysis pairs compressed-event and physical-time fits by simulation seed,
method, condition, and representation. Exact usable lag-pair counts are reported
separately from the bounded AR(1)-adjusted effective-sample proxy. Recovery
changes are evaluated against the same archived truth for each paired seed.

Compression changes the time axis and therefore the causal estimand. The paired
contrasts quantify the resulting sensitivity and sample-support trade-off; they
do not establish that either time construction is universally preferable.
"""
    (args.output_dir / "analysis-report.md").write_text(report, encoding="utf-8")
    stats = f"""# Statistical appendix

- Unit of analysis: matched simulation seed (`n={args.n_seeds}`).
- Contrast: physical time minus compressed event time.
- Metrics: F1, exact usable lag-pair median, AR(1)-adjusted sample proxy, and
  insufficient-pair fraction.
- Uncertainty: percentile bootstrap 95% intervals over paired seeds.
- Inference: two-sided random sign-flip tests with Holm correction across the
  complete contrast table.
- Effect size: paired mean difference standardized by its sample standard deviation.
- Boundary: the AR(1) quantity is a descriptive proxy, not an independent-sample count.
"""
    (args.output_dir / "stats-appendix.md").write_text(stats, encoding="utf-8")
    catalog = """# Figure catalog

## `episodic_effective_sample_summary.pdf`

- Purpose: compare sample support and archived F1 between compressed and physical time.
- Error bars: seed-bootstrap 95% intervals.
- Interpretation: read with `episodic_mode_contrasts.csv`; compression alters the estimand.

## `episodic_support_recovery_tradeoff.pdf`

- Purpose: show seed-level association between effective-support proxy and F1.
- Points: one matched seed per method and time construction for dynamic-A rise inputs.
- Interpretation: association is descriptive and does not identify a mechanism.
"""
    (args.output_dir / "figure-catalog.md").write_text(catalog, encoding="utf-8")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
