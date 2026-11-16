"""Validate and summarize matched rectification and cross-representation audits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from calcium_transient_rising_flank.checkpointing import atomic_write_json  # noqa: E402


ALGORITHMS = ("cgc", "cgc-star", "pcmciplus", "var-granger")
LABELS = {
    "cgc": "c-GC",
    "cgc-star": "c-GC*",
    "pcmciplus": "PCMCI+",
    "var-granger": "VAR",
}
SIGNED_REPRESENTATIONS = (
    "full",
    "deconvolved_rectified",
    "rise",
    "fall",
    "signed_difference",
    "signed_innovation",
)
SIGNED_CONDITIONS = (
    "all_excitatory_control",
    "mixed_excitation_inhibition",
    "post_inhibitory_rebound",
)
CROSS_ROLES = (
    "rise_graph_context",
    "fall_residual_to_rise_falsification",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"required benchmark output is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _number(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in {"", "None", "nan", "NaN"}:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _mean_ci(
    values: Iterable[float],
    *,
    seed: int,
    n_bootstrap: int = 5000,
) -> tuple[float, float, float, float]:
    array = np.asarray(tuple(values), dtype=float)
    if array.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    if array.size == 1:
        return mean, std, mean, mean
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(n_bootstrap, array.size))
    bootstrap = np.mean(array[indices], axis=1)
    quantiles: np.ndarray = np.quantile(bootstrap, (0.025, 0.975))
    return mean, std, float(quantiles[0]), float(quantiles[1])


def _sign_flip_pvalue(values: Sequence[float], *, seed: int) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    observed = abs(float(np.mean(array)))
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(19999, array.size))
    null = np.abs(np.mean(signs * array, axis=1))
    return float((1 + np.count_nonzero(null >= observed)) / (null.size + 1))


def _holm(rows: list[dict[str, Any]], key: str = "p_value") -> None:
    eligible = [
        (index, float(row[key]))
        for index, row in enumerate(rows)
        if row.get(key) is not None and math.isfinite(float(row[key]))
    ]
    ordered = sorted(eligible, key=lambda item: item[1])
    running = 0.0
    adjusted: dict[int, float] = {}
    total = len(ordered)
    for rank, (index, p_value) in enumerate(ordered):
        running = max(running, (total - rank) * p_value)
        adjusted[index] = min(1.0, running)
    for index, row in enumerate(rows):
        row["holm_p_value"] = adjusted.get(index)


def _load_all(input_root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    signed: list[dict[str, str]] = []
    cross: list[dict[str, str]] = []
    for algorithm in ALGORITHMS:
        signed.extend(_read_csv(input_root / algorithm / "signed_information_rows.csv"))
        cross.extend(
            _read_csv(input_root / algorithm / "cross_representation_rows.csv")
        )
    return signed, cross


def _assert_artifacts(rows: Sequence[dict[str, str]]) -> None:
    for row in rows:
        artifact = Path(row.get("artifact_path", ""))
        if not artifact.is_file():
            raise FileNotFoundError(f"raw graph artifact is missing: {artifact}")


def _assert_permutation_resolution(
    rows: Sequence[dict[str, str]], *, allow_underresolved: bool = False
) -> None:
    for row in rows:
        if row.get("algorithm") not in {"cgc", "cgc-star"}:
            continue
        n_nodes = int(row["n_rois"])
        alpha = float(row["alpha"])
        n_surrogates = int(row["n_surrogates"])
        minimum = math.ceil(n_nodes * (n_nodes - 1) / alpha) - 1
        if n_surrogates < minimum and not allow_underresolved:
            raise ValueError(
                f"{row['algorithm']} uses {n_surrogates} surrogates for {n_nodes} "
                f"nodes; at least {minimum} are required for first-rank BH resolution"
            )


def _assert_unique(rows: Sequence[dict[str, str]], keys: Sequence[str]) -> None:
    observed: set[tuple[str, ...]] = set()
    for row in rows:
        key = tuple(row.get(name, "") for name in keys)
        if key in observed:
            raise ValueError(f"duplicate benchmark row for {dict(zip(keys, key))}")
        observed.add(key)


def _assert_labels(
    rows: Sequence[dict[str, str]], key: str, expected: Sequence[str]
) -> None:
    observed = {row.get(key, "") for row in rows}
    if observed != set(expected):
        raise ValueError(
            f"unexpected {key} labels: observed {sorted(observed)}, "
            f"expected {sorted(expected)}"
        )


def _assert_matched(
    rows: Sequence[dict[str, str]],
    *,
    unit_keys: Sequence[str],
    expected_algorithms: Sequence[str] = ALGORITHMS,
) -> None:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in unit_keys)].append(row)
    for key, group in grouped.items():
        algorithms = {row["algorithm"] for row in group}
        if algorithms != set(expected_algorithms):
            raise ValueError(
                f"unmatched method rows for {dict(zip(unit_keys, key))}: {sorted(algorithms)}"
            )
        fields = ["trace_digest", "n_timepoints", "n_rois", "max_lag"]
        fields.extend(
            field
            for field in ("representation_digest", "source_digest", "target_digest")
            if any(row.get(field, "") for row in group)
        )
        for field in fields:
            values = {row.get(field, "") for row in group}
            if len(values) != 1:
                raise ValueError(
                    f"unmatched {field} for {dict(zip(unit_keys, key))}: {sorted(values)}"
                )


def _simulation_validation(
    signed: Sequence[dict[str, str]],
    cross: Sequence[dict[str, str]],
    *,
    n_seeds: int,
    seed_start: int,
) -> None:
    expected_seed = {str(value) for value in range(seed_start, seed_start + n_seeds)}
    _assert_unique(signed, ("algorithm", "condition", "seed", "representation"))
    _assert_unique(cross, ("algorithm", "condition", "seed", "test_role"))
    _assert_labels(signed, "condition", SIGNED_CONDITIONS)
    _assert_labels(signed, "representation", SIGNED_REPRESENTATIONS)
    _assert_labels(cross, "condition", ("dynamic_a_noncausal_fall",))
    _assert_labels(cross, "test_role", CROSS_ROLES)
    for algorithm in ALGORITHMS:
        method_signed = [row for row in signed if row["algorithm"] == algorithm]
        expected = n_seeds * len(SIGNED_CONDITIONS) * len(SIGNED_REPRESENTATIONS)
        if len(method_signed) != expected:
            raise ValueError(
                f"{algorithm} has {len(method_signed)} signed rows, expected {expected}"
            )
        if {row["seed"] for row in method_signed} != expected_seed:
            raise ValueError(
                f"{algorithm} signed rows do not cover the requested seeds"
            )
        method_cross = [row for row in cross if row["algorithm"] == algorithm]
        if len(method_cross) != n_seeds * len(CROSS_ROLES):
            raise ValueError(f"{algorithm} cross rows are incomplete")
    _assert_matched(signed, unit_keys=("condition", "seed", "representation"))
    _assert_matched(cross, unit_keys=("condition", "seed", "test_role"))
    for row in signed:
        negative_fraction = float(row["negative_sample_fraction"])
        if (
            row["representation"] == "deconvolved_rectified"
            and negative_fraction != 0.0
        ):
            raise ValueError("rectified innovations contain negative samples")
        if row["representation"] == "signed_innovation" and negative_fraction <= 0.0:
            raise ValueError("signed innovations failed to retain negative samples")
    for row in cross:
        falsification = row["test_role"] == "fall_residual_to_rise_falsification"
        if falsification and row.get("cross_phase_truth") != "empty":
            raise ValueError("fall-residual-to-rise test lacks an explicit empty truth")
        if falsification and row.get("source_digest") == row.get("target_digest"):
            raise ValueError(
                "cross-representation source and target inputs are identical"
            )


def _motor_validation(
    signed: Sequence[dict[str, str]],
    cross: Sequence[dict[str, str]],
    *,
    cases: Sequence[str],
    recordings: Sequence[str],
) -> None:
    units = {(case, recording) for case in cases for recording in recordings}
    _assert_unique(signed, ("algorithm", "case", "recording", "representation"))
    _assert_unique(cross, ("algorithm", "case", "recording", "test_role"))
    _assert_labels(signed, "condition", ("observed",))
    _assert_labels(signed, "representation", SIGNED_REPRESENTATIONS)
    _assert_labels(cross, "condition", ("observed",))
    _assert_labels(cross, "test_role", CROSS_ROLES)
    for algorithm in ALGORITHMS:
        method_units = {
            (row["case"], row["recording"])
            for row in signed
            if row["algorithm"] == algorithm
        }
        if method_units != units:
            raise ValueError(
                f"{algorithm} motor signed rows do not cover requested units"
            )
        if len([row for row in signed if row["algorithm"] == algorithm]) != len(
            units
        ) * len(SIGNED_REPRESENTATIONS):
            raise ValueError(f"{algorithm} motor signed rows are incomplete")
        if len([row for row in cross if row["algorithm"] == algorithm]) != len(
            units
        ) * len(CROSS_ROLES):
            raise ValueError(f"{algorithm} motor cross rows are incomplete")
    _assert_matched(signed, unit_keys=("case", "recording", "representation"))
    _assert_matched(cross, unit_keys=("case", "recording", "test_role"))
    for row in cross:
        if (
            row["test_role"] == "fall_residual_to_rise_falsification"
            and row.get("source_digest") == row.get("target_digest")
        ):
            raise ValueError(
                "motor cross-representation source and target inputs are identical"
            )


def _summary_rows(
    rows: Sequence[dict[str, str]],
    *,
    group_keys: Sequence[str],
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in group_keys)].append(row)
    output: list[dict[str, Any]] = []
    for group_index, (key, group) in enumerate(sorted(grouped.items())):
        for metric in metrics:
            values = [
                value for row in group if (value := _number(row, metric)) is not None
            ]
            if not values:
                continue
            mean, std, lower, upper = _mean_ci(values, seed=11000 + group_index)
            output.append(
                {
                    **dict(zip(group_keys, key)),
                    "metric": metric,
                    "n": len(values),
                    "mean": mean,
                    "std": std,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                }
            )
    return output


def _paired_contrasts(
    rows: Sequence[dict[str, str]],
    *,
    unit_keys: Sequence[str],
    group_keys: Sequence[str],
    label_key: str,
    reference: str,
    comparator: str,
    metrics: Sequence[str],
    inferential: bool = True,
) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, ...], dict[str, dict[str, str]]] = defaultdict(dict)
    all_keys = tuple(group_keys) + tuple(unit_keys)
    for row in rows:
        indexed[tuple(row.get(key, "") for key in all_keys)][row[label_key]] = row
    deltas: dict[tuple[str, ...], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for key, pair in indexed.items():
        if reference not in pair or comparator not in pair:
            continue
        group = key[: len(group_keys)]
        for metric in metrics:
            left = _number(pair[comparator], metric)
            right = _number(pair[reference], metric)
            if left is not None and right is not None:
                deltas[group][metric].append(left - right)
    output: list[dict[str, Any]] = []
    for group_index, (group, metric_values) in enumerate(sorted(deltas.items())):
        for metric, values in sorted(metric_values.items()):
            mean, std, lower, upper = _mean_ci(values, seed=21000 + group_index)
            output.append(
                {
                    **dict(zip(group_keys, group)),
                    "metric": metric,
                    "contrast": f"{comparator}_minus_{reference}",
                    "n_pairs": len(values),
                    "mean_delta": mean,
                    "std_delta": std,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "standardized_mean_delta": (
                        mean / std if std > 0.0 else (0.0 if mean == 0.0 else None)
                    ),
                    "p_value": (
                        _sign_flip_pvalue(values, seed=31000 + group_index)
                        if inferential
                        else None
                    ),
                }
            )
    _holm(output)
    return output


def _plot_simulation(
    signed_summary: Sequence[dict[str, Any]],
    cross_summary: Sequence[dict[str, Any]],
    output_dir: Path,
) -> list[str]:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    condition_labels = {
        "all_excitatory_control": "excitatory",
        "mixed_excitation_inhibition": "mixed-sign",
        "post_inhibitory_rebound": "rebound",
    }
    fig, axes = plt.subplots(
        1, 3, figsize=(13, 4), sharey=True, constrained_layout=True
    )
    for axis, condition in zip(axes, SIGNED_CONDITIONS):
        x: np.ndarray = np.arange(len(ALGORITHMS), dtype=float)
        width = 0.36
        for offset, representation in (
            (-width / 2, "deconvolved_rectified"),
            (width / 2, "signed_innovation"),
        ):
            selected = {
                row["algorithm"]: row
                for row in signed_summary
                if row.get("condition") == condition
                and row.get("representation") == representation
                and row.get("metric") == "f1"
            }
            means = [float(selected[method]["mean"]) for method in ALGORITHMS]
            lower = [
                means[i] - float(selected[method]["ci95_lower"])
                for i, method in enumerate(ALGORITHMS)
            ]
            upper = [
                float(selected[method]["ci95_upper"]) - means[i]
                for i, method in enumerate(ALGORITHMS)
            ]
            axis.bar(
                x + offset,
                means,
                width,
                yerr=np.asarray([lower, upper]),
                capsize=3,
                label=(
                    "rectified AR(1)"
                    if "rectified" in representation
                    else "signed AR(1)"
                ),
            )
        axis.set_title(condition_labels[condition])
        axis.set_xticks(x, [LABELS[item] for item in ALGORITHMS], rotation=25)
        axis.set_ylim(0.0, 1.05)
        axis.set_ylabel("directed F1")
    axes[0].legend(fontsize=8)
    path = figures / "signed_vs_rectified_recovery.pdf"
    fig.savefig(path)
    plt.close(fig)
    paths.append(str(path))

    fig, axis = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    x = np.arange(len(ALGORITHMS), dtype=float)
    selected = {
        row["algorithm"]: row
        for row in cross_summary
        if row.get("test_role") == "fall_residual_to_rise_falsification"
        and row.get("metric") == "null_false_positive_rate"
    }
    means = [float(selected[method]["mean"]) for method in ALGORITHMS]
    lower = [
        means[i] - float(selected[method]["ci95_lower"])
        for i, method in enumerate(ALGORITHMS)
    ]
    upper = [
        float(selected[method]["ci95_upper"]) - means[i]
        for i, method in enumerate(ALGORITHMS)
    ]
    axis.bar(
        x,
        means,
        yerr=np.asarray([lower, upper]),
        capsize=3,
    )
    axis.set_xticks(x, [LABELS[item] for item in ALGORITHMS])
    axis.set_ylabel("false-positive rate under empty cross-phase truth")
    axis.set_title("Fall residual → rise falsification on dynamic-A")
    path = figures / "fall_residual_to_rise_falsification.pdf"
    fig.savefig(path)
    plt.close(fig)
    paths.append(str(path))
    return paths


def _plot_motor(
    signed_summary: Sequence[dict[str, Any]],
    cross_summary: Sequence[dict[str, Any]],
    output_dir: Path,
) -> list[str]:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    x: np.ndarray = np.arange(len(ALGORITHMS), dtype=float)
    width = 0.36
    for index, representation in enumerate(
        ("deconvolved_rectified", "signed_innovation")
    ):
        selected = {
            row["algorithm"]: row
            for row in signed_summary
            if row.get("representation") == representation
            and row.get("metric") == "edge_density"
        }
        means = [float(selected[method]["mean"]) for method in ALGORITHMS]
        axes[0].bar(
            x + (index - 0.5) * width,
            means,
            width,
            label=representation.replace("_", " "),
        )
    falsification = {
        row["algorithm"]: row
        for row in cross_summary
        if row.get("test_role") == "fall_residual_to_rise_falsification"
        and row.get("metric") == "edge_density"
    }
    axes[1].bar(x, [float(falsification[method]["mean"]) for method in ALGORITHMS])
    for axis in axes:
        axis.set_xticks(x, [LABELS[item] for item in ALGORITHMS], rotation=25)
        axis.set_ylabel("edge density")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Signed-input sensitivity")
    axes[1].set_title("Fall residual → rise sensitivity")
    path = figures / "motor_representation_sensitivity.pdf"
    fig.savefig(path)
    plt.close(fig)
    return [str(path)]


def _write_reports(
    output_dir: Path,
    *,
    dataset: str,
    figures: Sequence[str],
    signed_count: int,
    cross_count: int,
    signed_contrasts: Sequence[dict[str, Any]],
) -> None:
    simulation = dataset == "simulations"
    report = [
        "# Representation-bias analysis",
        "",
        f"Dataset scope: `{dataset}`.",
        f"Validated {signed_count} signed-information rows and {cross_count} cross-representation rows.",
        "",
        "The primary rectification contrast is signed AR(1) innovation minus its paired nonnegative projection.",
        "The cross-representation falsification tests lag-one source fall residuals against target rises on the complete physical timeseries.",
        "Rise-graph recovery is reported only as contextual calibration; it is not contrasted with the cross-representation fit because the two fits have different estimands and conditioning systems.",
        "",
        "## Interpretation boundary",
        "",
    ]
    if simulation:
        report.extend(
            [
                "Simulation recovery is evaluated against declared excitatory, inhibitory, and biphasic rebound edge masks.",
                "Lag-specific recall measures support plus selected-lag localization; they do not establish formal edge-sign recovery.",
                "Dynamic-A declares an empty fall-to-rise cross-phase truth, so retained cross-block edges are false positives for that falsification estimand.",
                "Method comparisons use matched inputs and the same lag-one cross-block estimand, while preserving each algorithm's native conditioning system.",
            ]
        )
    else:
        report.extend(
            [
                "Motor-neuron analyses are descriptive: graph changes cannot be interpreted as inhibitory, rebound, or cross-phase accuracy without ground truth.",
            ]
        )
    report.extend(["", "## Paired contrast highlights", ""])
    highlight_metrics = {
        "f1",
        "inhibitory_lag1_recall",
        "rebound_lag2_recall",
        "edge_density",
    }
    for row in signed_contrasts:
        if row.get("metric") not in highlight_metrics:
            continue
        interval = f"[{float(row['ci95_lower']):.3f}, {float(row['ci95_upper']):.3f}]"
        method = LABELS.get(str(row.get("algorithm")), str(row.get("algorithm")))
        report.append(
            f"- {method}, {row.get('condition')}, {row.get('metric')}: "
            f"mean paired delta {float(row['mean_delta']):.3f}, 95% CI {interval}."
        )
    (output_dir / "analysis-report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    stats = ["# Statistical appendix", ""]
    if simulation:
        stats.extend(
            [
                "Intervals are percentile bootstrap 95% confidence intervals over matched simulation seeds.",
                "Paired contrasts use two-sided random sign-flip tests and Holm correction over the reported contrast family.",
            ]
        )
    else:
        stats.extend(
            [
                "Intervals summarize matched case-recording units descriptively.",
                "No inferential p-values are assigned because preprocessing cases and repeated recordings are not treated as independent biological replicates.",
            ]
        )
    stats.append(
        "Effect sizes are paired standardized mean differences when the paired-difference standard deviation is nonzero."
    )
    (output_dir / "stats-appendix.md").write_text(
        "\n".join(stats) + "\n", encoding="utf-8"
    )
    catalog = ["# Figure catalog", ""]
    for path in figures:
        name = Path(path).name
        if name == "motor_representation_sensitivity.pdf":
            purpose = (
                "show the matched signed-versus-rectified sensitivity and the separate "
                "fall-residual-to-rise result"
            )
            interpretation = (
                "treat both motor-neuron panels descriptively and do not contrast the "
                "cross-representation fit with the contextual rise graph"
            )
        elif "fall_residual" in name:
            purpose = (
                "quantify the standalone fall-residual-to-rise falsification result; "
                "the contextual rise graph is intentionally not used as a comparator"
            )
            interpretation = (
                "interpret the simulation panel against its declared empty cross-phase "
                "truth and the motor-neuron panel descriptively"
            )
        else:
            purpose = "quantify the matched signed-versus-rectified sensitivity analysis"
            interpretation = "use the paired contrast table for inference"
        catalog.extend(
            [
                f"## {name}",
                "",
                f"Purpose: {purpose}.",
                "Error bars: bootstrap 95% confidence intervals where shown.",
                f"Interpretation: {interpretation}.",
                "",
            ]
        )
    (output_dir / "figure-catalog.md").write_text("\n".join(catalog), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("simulations", "motorneurons"), required=True
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--cases", default="C,D")
    parser.add_argument("--recordings", default="F3T1,F3T2,F5T2")
    parser.add_argument(
        "--allow-underresolved-surrogates",
        action="store_true",
        help="Development-smoke override; never use for publication analysis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    signed, cross = _load_all(args.input_root)
    _assert_artifacts((*signed, *cross))
    _assert_permutation_resolution(
        (*signed, *cross),
        allow_underresolved=args.allow_underresolved_surrogates,
    )
    unit_keys: tuple[str, ...]
    signed_group_keys: tuple[str, ...]
    if args.dataset == "simulations":
        _simulation_validation(
            signed,
            cross,
            n_seeds=args.n_seeds,
            seed_start=args.seed_start,
        )
        unit_keys = ("seed",)
        signed_group_keys = ("algorithm", "condition")
    else:
        cases = tuple(
            item.strip().upper() for item in args.cases.split(",") if item.strip()
        )
        recordings = tuple(
            item.strip() for item in args.recordings.split(",") if item.strip()
        )
        _motor_validation(signed, cross, cases=cases, recordings=recordings)
        unit_keys = ("case", "recording")
        signed_group_keys = ("algorithm", "condition")

    signed_metrics = (
        "edge_density" if args.dataset == "motorneurons" else "f1",
        "retained_edges",
        "negative_sample_fraction",
        "precision",
        "recall",
        "false_positive_rate",
        "excitatory_recall",
        "inhibitory_recall",
        "inhibitory_lag1_recall",
        "rebound_recall",
        "rebound_lag2_recall",
        "signed_vs_rectified_jaccard",
    )
    cross_metrics = (
        "edge_density",
        "retained_edges",
        "precision",
        "recall",
        "f1",
        "null_false_positive_rate",
    )
    signed_summary = _summary_rows(
        signed,
        group_keys=(*signed_group_keys, "representation"),
        metrics=signed_metrics,
    )
    cross_summary = _summary_rows(
        cross,
        group_keys=("algorithm", "condition", "test_role"),
        metrics=cross_metrics,
    )
    signed_contrasts = _paired_contrasts(
        signed,
        unit_keys=unit_keys,
        group_keys=signed_group_keys,
        label_key="representation",
        reference="deconvolved_rectified",
        comparator="signed_innovation",
        metrics=signed_metrics,
        inferential=args.dataset == "simulations",
    )
    _write_csv(args.output_dir / "signed_information_summary.csv", signed_summary)
    _write_csv(args.output_dir / "signed_rectification_contrasts.csv", signed_contrasts)
    _write_csv(args.output_dir / "cross_representation_summary.csv", cross_summary)
    figures = (
        _plot_simulation(signed_summary, cross_summary, args.output_dir)
        if args.dataset == "simulations"
        else _plot_motor(signed_summary, cross_summary, args.output_dir)
    )
    _write_reports(
        args.output_dir,
        dataset=args.dataset,
        figures=figures,
        signed_count=len(signed),
        cross_count=len(cross),
        signed_contrasts=signed_contrasts,
    )
    summary = {
        "status": "complete",
        "dataset": args.dataset,
        "input_root": str(args.input_root),
        "n_signed_rows": len(signed),
        "n_cross_rows": len(cross),
        "n_signed_contrasts": len(signed_contrasts),
        "figures": figures,
        "primary_rectification_estimand": "signed_innovation minus deconvolved_rectified",
        "underresolved_surrogate_override": args.allow_underresolved_surrogates,
        "primary_cross_estimand": (
            "fall_residual(t-1) to rise(t) under empty dynamic-A cross-phase truth"
            if args.dataset == "simulations"
            else "descriptive fall_residual(t-1) to rise(t) sensitivity without graph truth"
        ),
        "rise_graph_role": (
            "contextual calibration only; not directly contrasted with the cross-representation estimand"
        ),
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
