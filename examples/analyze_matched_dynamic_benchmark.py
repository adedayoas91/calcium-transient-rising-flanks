"""Analyze the matched full-axis dynamic benchmark across four CSL algorithms.

The analysis is deliberately strict: every algorithm must use the same per-seed
input fingerprint, physical time axis, lag horizon, truth target, and evaluation
unit. c-GC and c-GC* use conditioning depth one in the primary comparison;
depths two and three are reported separately as a prespecified sensitivity.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


ALGORITHMS = ("cgc", "cgc-star", "pcmciplus", "var-granger")
ALGORITHM_LABELS = {
    "cgc": "c-GC",
    "cgc-star": "c-GC*",
    "pcmciplus": "PCMCI+",
    "var-granger": "VAR-Granger",
}
CONDITION_LABELS = {
    "dynamic_a_noncausal_fall": "Noncausal fall",
    "dynamic_b_hybrid_causal_fall_overlap": "Causal fall, overlap",
    "dynamic_c_hybrid_causal_fall_kinetic_misspecification": (
        "Causal fall, kinetic misspecification"
    ),
}
METRICS = (
    "f1",
    "precision",
    "recall",
    "false_positive_rate",
    "orientation_accuracy",
    "retained_edges",
)
KEY_FIELDS = ("condition", "seed", "representation", "truth_target")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing benchmark rows: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"benchmark file has no rows: {path}")
    for row in rows:
        row["seed"] = int(row["seed"])
        row["max_lag"] = int(row["max_lag"])
        row["n_timepoints"] = int(row["n_timepoints"])
        raw_depth = str(row.get("n_pasts", "")).strip()
        row["n_pasts"] = int(float(raw_depth)) if raw_depth else None
        for metric in METRICS:
            row[metric] = float(row[metric])
    return rows


def load_benchmark_rows(input_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for algorithm in ALGORITHMS:
        algorithm_rows = _read_rows(input_root / algorithm / "baseline_rows.csv")
        declared = {str(row["algorithm"]) for row in algorithm_rows}
        if declared != {algorithm}:
            raise ValueError(
                f"{algorithm} directory declares unexpected algorithms: {declared}"
            )
        rows.extend(algorithm_rows)
    return rows


def _key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in KEY_FIELDS)


def _primary_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["algorithm"] not in {"cgc", "cgc-star"} or row["n_pasts"] == 1
    ]


def validate_matched_design(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    dynamic = [row for row in rows if row.get("simulation_kind") == "dynamic-extension"]
    if len(dynamic) != len(rows):
        raise ValueError("all rows must come from the dynamic-extension simulation")
    if {row["max_lag"] for row in rows} != {1}:
        raise ValueError("the matched benchmark requires max_lag=1 for every algorithm")

    cgc_depths = {
        algorithm: sorted(
            {
                int(row["n_pasts"])
                for row in rows
                if row["algorithm"] == algorithm and row["n_pasts"] is not None
            }
        )
        for algorithm in ("cgc", "cgc-star")
    }
    for algorithm, depths in cgc_depths.items():
        if depths != [1, 2, 3]:
            raise ValueError(
                f"{algorithm} conditioning depths must be exactly [1, 2, 3], got {depths}"
            )

    by_algorithm: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {}
    for algorithm in ALGORITHMS:
        selected = [row for row in _primary_rows(rows) if row["algorithm"] == algorithm]
        index = {_key(row): row for row in selected}
        if len(index) != len(selected):
            raise ValueError(f"duplicate primary evaluation units for {algorithm}")
        by_algorithm[algorithm] = index

    reference_keys = set(by_algorithm[ALGORITHMS[0]])
    for algorithm, index in by_algorithm.items():
        if set(index) != reference_keys:
            missing = len(reference_keys - set(index))
            extra = len(set(index) - reference_keys)
            raise ValueError(
                f"unmatched evaluation units for {algorithm}: {missing} missing, {extra} extra"
            )
    for key in sorted(reference_keys):
        reference = by_algorithm[ALGORITHMS[0]][key]
        for algorithm in ALGORITHMS[1:]:
            observed = by_algorithm[algorithm][key]
            for field in ("input_digest", "n_timepoints", "truth_edges"):
                if str(observed[field]) != str(reference[field]):
                    raise ValueError(
                        f"matched-design failure for {key}: {field} differs between "
                        f"{ALGORITHMS[0]} and {algorithm}"
                    )

    depth_reference = {
        (row["algorithm"], _key(row)): (row["input_digest"], row["n_timepoints"])
        for row in rows
        if row["algorithm"] in {"cgc", "cgc-star"} and row["n_pasts"] == 1
    }
    for row in rows:
        if row["algorithm"] not in {"cgc", "cgc-star"}:
            continue
        expected = depth_reference[(row["algorithm"], _key(row))]
        if (row["input_digest"], row["n_timepoints"]) != expected:
            raise ValueError("conditioning-depth rows do not share identical inputs")

    return {
        "status": "matched",
        "n_primary_units_per_algorithm": len(reference_keys),
        "n_seeds": len({row["seed"] for row in rows}),
        "conditions": sorted({str(row["condition"]) for row in rows}),
        "representations": sorted({str(row["representation"]) for row in rows}),
        "truth_targets": sorted({str(row["truth_target"]) for row in rows}),
        "max_lag": 1,
        "cgc_depths": cgc_depths,
    }


def _bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    if data.size == 1:
        return float(data[0]), float(data[0])
    samples = rng.choice(data, size=(n_bootstrap, data.size), replace=True).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return float(lower), float(upper)


def summarize_rows(
    rows: Sequence[dict[str, Any]], *, n_bootstrap: int, seed: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        depth = row["n_pasts"] if row["n_pasts"] is not None else "native"
        group = (
            row["condition"],
            row["algorithm"],
            row["representation"],
            row["truth_target"],
            depth,
        )
        groups[group].append(row)

    rng = np.random.default_rng(seed)
    summaries: list[dict[str, Any]] = []
    for group, group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        condition, algorithm, representation, truth_target, depth = group
        for metric in METRICS:
            values = [float(row[metric]) for row in group_rows]
            lower, upper = _bootstrap_mean_ci(values, rng, n_bootstrap)
            summaries.append(
                {
                    "condition": condition,
                    "algorithm": algorithm,
                    "representation": representation,
                    "truth_target": truth_target,
                    "n_pasts": depth,
                    "metric": metric,
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
    return summaries


def _sign_flip_pvalue(
    differences: np.ndarray, rng: np.random.Generator, n_permutations: int
) -> float:
    observed = abs(float(np.mean(differences)))
    if observed == 0.0:
        return 1.0
    signs = rng.choice((-1.0, 1.0), size=(n_permutations, differences.size))
    permuted = np.abs(np.mean(signs * differences, axis=1))
    return float((1 + np.count_nonzero(permuted >= observed)) / (n_permutations + 1))


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted: np.ndarray = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def conditioning_depth_contrasts(
    rows: Sequence[dict[str, Any]],
    *,
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row["algorithm"] in {"cgc", "cgc-star"} and row["truth_target"] == "union"
    ]
    grouped: dict[tuple[Any, ...], dict[int, dict[int, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in selected:
        group = (row["condition"], row["algorithm"], row["representation"])
        grouped[group][int(row["n_pasts"])][int(row["seed"])] = row

    rng = np.random.default_rng(seed)
    contrasts: list[dict[str, Any]] = []
    for group, depth_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        condition, algorithm, representation = group
        reference_seeds = set(depth_rows[1])
        for depth in (2, 3):
            if set(depth_rows[depth]) != reference_seeds:
                raise ValueError(
                    f"unpaired depth comparison for {group}, depth {depth}"
                )
            for metric in ("f1", "false_positive_rate"):
                differences = np.asarray(
                    [
                        float(depth_rows[depth][unit_seed][metric])
                        - float(depth_rows[1][unit_seed][metric])
                        for unit_seed in sorted(reference_seeds)
                    ]
                )
                lower, upper = _bootstrap_mean_ci(differences, rng, n_bootstrap)
                contrasts.append(
                    {
                        "condition": condition,
                        "algorithm": algorithm,
                        "representation": representation,
                        "truth_target": "union",
                        "metric": metric,
                        "reference_n_pasts": 1,
                        "n_pasts": depth,
                        "n": differences.size,
                        "mean_paired_difference": float(np.mean(differences)),
                        "std_paired_difference": (
                            float(np.std(differences, ddof=1))
                            if differences.size > 1
                            else 0.0
                        ),
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "standardized_mean_difference": (
                            float(np.mean(differences) / np.std(differences, ddof=1))
                            if differences.size > 1
                            and np.std(differences, ddof=1) > 0.0
                            else None
                        ),
                        "p_value": _sign_flip_pvalue(differences, rng, n_permutations),
                    }
                )
    adjusted = _holm_adjust([float(row["p_value"]) for row in contrasts])
    for row, p_adjusted in zip(contrasts, adjusted):
        row["p_holm"] = p_adjusted
    return contrasts


def paired_method_contrasts(
    rows: Sequence[dict[str, Any]],
    *,
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Compare algorithms on identical seeds, inputs, representations, and truth."""

    selected = [
        row for row in _primary_rows(rows) if row["truth_target"] == "union"
    ]
    indexed: dict[
        tuple[str, str], dict[str, dict[int, dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(dict))
    for row in selected:
        group = (str(row["condition"]), str(row["representation"]))
        indexed[group][str(row["algorithm"])][int(row["seed"])] = row

    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    for (condition, representation), method_rows in sorted(indexed.items()):
        for left, right in combinations(ALGORITHMS, 2):
            left_seeds = set(method_rows[left])
            if left_seeds != set(method_rows[right]):
                raise ValueError(
                    f"unpaired method comparison for {condition}, {representation}, "
                    f"{left} versus {right}"
                )
            for metric in ("f1", "false_positive_rate"):
                differences = np.asarray(
                    [
                        float(method_rows[left][unit_seed][metric])
                        - float(method_rows[right][unit_seed][metric])
                        for unit_seed in sorted(left_seeds)
                    ],
                    dtype=float,
                )
                lower, upper = _bootstrap_mean_ci(
                    differences, rng, n_bootstrap
                )
                std = (
                    float(np.std(differences, ddof=1))
                    if differences.size > 1
                    else 0.0
                )
                mean = float(np.mean(differences))
                output.append(
                    {
                        "condition": condition,
                        "representation": representation,
                        "truth_target": "union",
                        "metric": metric,
                        "left_algorithm": left,
                        "right_algorithm": right,
                        "contrast": f"{left}_minus_{right}",
                        "n": differences.size,
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


def _ar1_effective_sample_proxy(x: np.ndarray, y: np.ndarray) -> float:
    count = min(x.size, y.size)
    if count < 3:
        return float(count)

    def lag_one(values: np.ndarray) -> float:
        left = values[:-1] - np.mean(values[:-1])
        right = values[1:] - np.mean(values[1:])
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator <= np.finfo(float).eps:
            return 0.0
        return float(np.dot(left, right) / denominator)

    product = float(np.clip(lag_one(x) * lag_one(y), -0.999, 0.999))
    estimate = count * (1.0 - product) / (1.0 + product)
    return float(np.clip(estimate, 1.0, count))


def effective_support_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Audit event sparsity and usable lag-1 support on each shared input."""

    references = {
        (row["condition"], row["seed"], row["representation"]): row
        for row in rows
        if row["algorithm"] == "cgc"
        and row["n_pasts"] == 1
        and row["truth_target"] == "union"
    }
    output: list[dict[str, Any]] = []
    for (condition, unit_seed, representation), row in sorted(references.items()):
        artifact = Path(str(row["artifact_path"]))
        if not artifact.is_file():
            raise FileNotFoundError(
                f"missing input artifact for support audit: {artifact}"
            )
        with np.load(artifact, allow_pickle=False) as payload:
            if "input_values" not in payload:
                raise ValueError(
                    f"artifact predates the effective-support schema: {artifact}"
                )
            values = np.asarray(payload["input_values"], dtype=float)
        active = np.abs(values) > np.finfo(float).eps
        pair_counts: list[int] = []
        effective_samples: list[float] = []
        for source in range(values.shape[0]):
            for target in range(values.shape[0]):
                if source == target:
                    continue
                valid = active[source, :-1] & active[target, 1:]
                pair_counts.append(int(np.count_nonzero(valid)))
                effective_samples.append(
                    _ar1_effective_sample_proxy(values[source, :-1], values[target, 1:])
                )
        output.append(
            {
                "condition": condition,
                "seed": unit_seed,
                "representation": representation,
                "n_rois": values.shape[0],
                "n_timepoints": values.shape[1],
                "lag_1_pairs_full_axis": values.shape[1] - 1,
                "global_active_frame_fraction": float(np.mean(np.any(active, axis=0))),
                "roi_active_frame_fraction_mean": float(np.mean(active)),
                "joint_active_lag_pairs_median": float(np.median(pair_counts)),
                "joint_active_lag_pair_fraction_median": float(
                    np.median(pair_counts) / max(1, values.shape[1] - 1)
                ),
                "ar1_effective_sample_proxy_pair_median": float(
                    np.median(effective_samples)
                ),
                "interpretation": (
                    "descriptive AR(1)-adjusted proxy and nonzero lag-pair support; "
                    "not an exact independent-sample count"
                ),
            }
        )
    return output


def selection_tradeoff_rows(
    rows: Sequence[dict[str, Any]], support_rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    support = {
        (row["condition"], row["seed"], row["representation"]): row
        for row in support_rows
    }
    primary = [row for row in _primary_rows(rows) if row["truth_target"] == "union"]
    indexed = {
        (row["algorithm"], row["condition"], row["seed"], row["representation"]): row
        for row in primary
    }
    output: list[dict[str, Any]] = []
    for row in primary:
        key = (row["condition"], row["seed"], row["representation"])
        reference_key = (row["algorithm"], row["condition"], row["seed"], "full")
        if reference_key not in indexed:
            raise ValueError("the selection audit requires the full representation")
        reference = indexed[reference_key]
        sample = support[key]
        output.append(
            {
                "condition": row["condition"],
                "seed": row["seed"],
                "algorithm": row["algorithm"],
                "representation": row["representation"],
                "truth_target": "union",
                "n_pasts": row["n_pasts"] if row["n_pasts"] is not None else "native",
                "f1": row["f1"],
                "f1_minus_full": float(row["f1"]) - float(reference["f1"]),
                "global_active_frame_fraction": sample["global_active_frame_fraction"],
                "joint_active_lag_pair_fraction_median": sample[
                    "joint_active_lag_pair_fraction_median"
                ],
                "ar1_effective_sample_proxy_pair_median": sample[
                    "ar1_effective_sample_proxy_pair_median"
                ],
            }
        )
    return output


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary_lookup(
    summaries: Sequence[dict[str, Any]],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {
        (
            row["condition"],
            row["algorithm"],
            row["representation"],
            row["truth_target"],
            str(row["n_pasts"]),
            row["metric"],
        ): row
        for row in summaries
    }


def plot_primary_comparison(
    summaries: Sequence[dict[str, Any]], output_dir: Path
) -> None:
    lookup = _summary_lookup(summaries)
    conditions = list(CONDITION_LABELS)
    representations = sorted(
        {
            str(row["representation"])
            for row in summaries
            if row["truth_target"] == "union" and row["metric"] == "f1"
        }
    )
    colors = dict(zip(ALGORITHMS, ("#0072B2", "#009E73", "#D55E00", "#CC79A7")))
    markers = dict(zip(ALGORITHMS, ("o", "s", "^", "D")))
    fig, axes = plt.subplots(1, len(conditions), figsize=(13.0, 4.2), sharey=True)
    offsets = np.linspace(-0.27, 0.27, len(ALGORITHMS))
    x = np.arange(len(representations))
    for axis, condition in zip(axes, conditions):
        for algorithm, offset in zip(ALGORITHMS, offsets):
            depth = "1" if algorithm in {"cgc", "cgc-star"} else "native"
            records = [
                lookup[(condition, algorithm, representation, "union", depth, "f1")]
                for representation in representations
            ]
            means = np.asarray([record["mean"] for record in records])
            lower = means - np.asarray([record["ci_lower"] for record in records])
            upper = np.asarray([record["ci_upper"] for record in records]) - means
            axis.errorbar(
                x + offset,
                means,
                yerr=np.vstack((lower, upper)),
                fmt=markers[algorithm],
                color=colors[algorithm],
                capsize=2.5,
                markersize=5,
                linewidth=1.2,
                label=ALGORITHM_LABELS[algorithm],
            )
        axis.set_title(CONDITION_LABELS[condition], fontsize=10)
        axis.set_xticks(x, [item.replace("_", "\n") for item in representations])
        axis.tick_params(axis="x", labelsize=8)
        axis.set_ylim(-0.02, 1.02)
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axes[0].set_ylabel("F1 against union truth")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Matched full-axis dynamic benchmark (physical lag 1)", y=1.02)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(
            output_dir / f"matched_dynamic_four_learner_f1.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_depth_sensitivity(
    summaries: Sequence[dict[str, Any]], output_dir: Path
) -> None:
    lookup = _summary_lookup(summaries)
    conditions = list(CONDITION_LABELS)
    methods = ("cgc", "cgc-star")
    representations = sorted(
        {
            str(row["representation"])
            for row in summaries
            if row["algorithm"] == "cgc"
            and row["truth_target"] == "union"
            and row["metric"] == "f1"
        }
    )
    palette = plt.get_cmap("tab10")
    fig, axes = plt.subplots(
        len(methods), len(conditions), figsize=(13.0, 7.0), sharex=True, sharey=True
    )
    for row_index, method in enumerate(methods):
        for column_index, condition in enumerate(conditions):
            axis = axes[row_index, column_index]
            for color_index, representation in enumerate(representations):
                records = [
                    lookup[
                        (
                            condition,
                            method,
                            representation,
                            "union",
                            str(depth),
                            "f1",
                        )
                    ]
                    for depth in (1, 2, 3)
                ]
                means = np.asarray([record["mean"] for record in records])
                lower = means - np.asarray([record["ci_lower"] for record in records])
                upper = np.asarray([record["ci_upper"] for record in records]) - means
                axis.errorbar(
                    (1, 2, 3),
                    means,
                    yerr=np.vstack((lower, upper)),
                    marker="o",
                    capsize=2,
                    linewidth=1.0,
                    color=palette(color_index),
                    label=representation.replace("_", " "),
                )
            if row_index == 0:
                axis.set_title(CONDITION_LABELS[condition], fontsize=10)
            if column_index == 0:
                axis.set_ylabel(f"{ALGORITHM_LABELS[method]} union-truth F1")
            if row_index == len(methods) - 1:
                axis.set_xlabel("Conditioning depth, n_pasts")
            axis.set_xticks((1, 2, 3))
            axis.set_ylim(-0.02, 1.02)
            axis.grid(color="#dddddd", linewidth=0.7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=min(5, len(labels)), frameon=False
    )
    fig.suptitle("c-GC/c-GC* conditioning-depth sensitivity at tested lag 1", y=1.02)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(
            output_dir / f"cgc_conditioning_depth_sensitivity.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_selection_tradeoff(
    tradeoffs: Sequence[dict[str, Any]], output_dir: Path
) -> None:
    algorithms = list(ALGORITHMS)
    representations = sorted({row["representation"] for row in tradeoffs})
    color_map = plt.get_cmap("tab10")
    colors = {
        representation: color_map(index)
        for index, representation in enumerate(representations)
    }
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2), sharex=True, sharey=True)
    for axis, algorithm in zip(axes.ravel(), algorithms):
        selected = [row for row in tradeoffs if row["algorithm"] == algorithm]
        for representation in representations:
            group = [row for row in selected if row["representation"] == representation]
            axis.scatter(
                [row["joint_active_lag_pair_fraction_median"] for row in group],
                [row["f1"] for row in group],
                s=22,
                alpha=0.65,
                color=colors[representation],
                label=representation.replace("_", " "),
            )
        axis.set_title(ALGORITHM_LABELS[algorithm])
        axis.grid(color="#dddddd", linewidth=0.7)
    for axis in axes[-1]:
        axis.set_xlabel("Median fraction of nonzero lag-1 pairs")
    for axis in axes[:, 0]:
        axis.set_ylabel("Union-truth F1")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=min(5, len(labels)), frameon=False
    )
    fig.suptitle(
        "Event sparsity and recovery on the retained physical time axis", y=1.02
    )
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(
            output_dir / f"selection_effective_support_tradeoff.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def _write_reports(
    output_dir: Path,
    validation: dict[str, Any],
    summaries: Sequence[dict[str, Any]],
    contrasts: Sequence[dict[str, Any]],
    method_contrasts: Sequence[dict[str, Any]],
    support_rows: Sequence[dict[str, Any]],
    tradeoffs: Sequence[dict[str, Any]],
) -> None:
    report = f"""# Matched dynamic benchmark

The design audit passed for {validation["n_seeds"]} paired stochastic seeds.
All four CSL algorithms received identical full-axis inputs, physical time indices,
truth targets, and a maximum tested lag of one. The primary comparison fixes
c-GC/c-GC* at `n_pasts=1`; `n_pasts=2` and `3` are sensitivity analyses and do
not change the tested lag-1 estimand.

The bootstrap intervals describe stochastic variation conditional on the fixed
simulated topologies. They do not quantify variation across network structures.
Conditioning-depth differences are paired by seed and Holm-adjusted across the
reported union-truth F1 and false-positive-rate contrasts. Depth should not be
selected from these evaluation results without a separate calibration split.

Cross-method differences are also paired by seed within the same condition,
representation, physical lag, and union-truth target. Their sign-flip p-values
are Holm-adjusted across the full reported cross-method contrast family.

The effective-support audit is shared by all four algorithms. It reports the
fraction of nonzero lag-1 pairs and a bounded AR(1) sample proxy for each input,
then pairs each representation's F1 with its full-trace F1. These are
descriptive selection-associated contrasts: changing representation also
changes signal content, so the analysis does not by itself identify a causal
effect of selection.
"""
    (output_dir / "analysis-report.md").write_text(report, encoding="utf-8")
    stats = f"""# Statistical appendix

- Unit of analysis: matched simulation seed (`n={validation['n_seeds']}`).
- Descriptive summaries: mean, sample standard deviation, and percentile
  bootstrap 95% interval.
- Cross-method and conditioning-depth contrasts: paired mean difference,
  paired-difference standard deviation, standardized paired mean difference,
  percentile bootstrap 95% interval, and two-sided random sign-flip test.
- Multiplicity: Holm correction across each complete reported contrast table.
- Scope: intervals condition on the fixed simulated topologies and do not
  represent variation across network structures.
- Selection trade-offs are descriptive because representation changes both
  sample support and signal content.
"""
    (output_dir / "stats-appendix.md").write_text(stats, encoding="utf-8")
    catalog = """# Figure catalog

## `matched_dynamic_four_learner_f1.pdf`

- Purpose: compare four CSL algorithms on identical full-axis inputs.
- Plotted quantity: union-truth F1; error bars are seed-bootstrap 95% intervals.
- Interpretation: use `method_contrasts.csv` for paired inference.

## `cgc_conditioning_depth_sensitivity.pdf`

- Purpose: assess c-GC/c-GC* sensitivity to `n_pasts=1,2,3` at fixed lag 1.
- Plotted quantity: union-truth F1; error bars are seed-bootstrap 95% intervals.
- Interpretation: depth is a sensitivity parameter, not a result-selected optimum.

## `selection_effective_support_tradeoff.pdf`

- Purpose: show how event selection changes lag-pair support alongside F1.
- Plotted quantities: nonzero lag-pair fraction and union-truth F1 by representation.
- Interpretation: association does not isolate a causal effect of selection.
"""
    (output_dir / "figure-catalog.md").write_text(catalog, encoding="utf-8")
    summary = {
        "status": "complete",
        "validation": validation,
        "n_summary_rows": len(summaries),
        "n_depth_contrasts": len(contrasts),
        "n_method_contrasts": len(method_contrasts),
        "outputs": [
            "metric_summary.csv",
            "conditioning_depth_contrasts.csv",
            "method_contrasts.csv",
            "effective_support_rows.csv",
            "selection_tradeoff_rows.csv",
            "matched_dynamic_four_learner_f1.pdf",
            "matched_dynamic_four_learner_f1.png",
            "cgc_conditioning_depth_sensitivity.pdf",
            "cgc_conditioning_depth_sensitivity.png",
            "selection_effective_support_tradeoff.pdf",
            "selection_effective_support_tradeoff.png",
            "analysis-report.md",
            "stats-appendix.md",
            "figure-catalog.md",
        ],
        "n_effective_support_rows": len(support_rows),
        "n_selection_tradeoff_rows": len(tradeoffs),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", type=Path, default=Path("outputs/matched_dynamic_benchmark")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/matched_dynamic_benchmark/analysis"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()
    if args.n_bootstrap < 100 or args.n_permutations < 99:
        raise SystemExit("use at least 100 bootstrap samples and 99 permutations")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_benchmark_rows(args.input_root)
    validation = validate_matched_design(rows)
    summaries = summarize_rows(rows, n_bootstrap=args.n_bootstrap, seed=args.seed)
    contrasts = conditioning_depth_contrasts(
        rows,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        seed=args.seed + 1,
    )
    method_contrasts = paired_method_contrasts(
        rows,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        seed=args.seed + 2,
    )
    support_rows = effective_support_rows(rows)
    tradeoffs = selection_tradeoff_rows(rows, support_rows)
    _write_csv(args.output_dir / "metric_summary.csv", summaries)
    _write_csv(args.output_dir / "conditioning_depth_contrasts.csv", contrasts)
    _write_csv(args.output_dir / "method_contrasts.csv", method_contrasts)
    _write_csv(args.output_dir / "effective_support_rows.csv", support_rows)
    _write_csv(args.output_dir / "selection_tradeoff_rows.csv", tradeoffs)
    plot_primary_comparison(summaries, args.output_dir)
    plot_depth_sensitivity(summaries, args.output_dir)
    plot_selection_tradeoff(tradeoffs, args.output_dir)
    _write_reports(
        args.output_dir,
        validation,
        summaries,
        contrasts,
        method_contrasts,
        support_rows,
        tradeoffs,
    )
    print(json.dumps({"status": "complete", "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
