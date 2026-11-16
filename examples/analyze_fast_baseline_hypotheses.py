"""Run H1--H4 diagnostics for PCMCI+ or conditional VAR-Granger."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from calcium_transient_rising_flank import (
    characterize_transients,
    edge_recovery,
    simulate_calcium_dataset,
)
from calcium_transient_rising_flank.checkpointing import atomic_write_json
from calcium_transient_rising_flank.validation import cyclic_shift_surrogate

EXAMPLES_ROOT = Path(__file__).resolve().parent
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from run_fast_causal_baselines import _fit, representation_map  # noqa: E402


METHODS = ("pcmciplus", "var-granger")
LABELS = {"pcmciplus": "PCMCI+", "var-granger": "VAR-Granger"}
NOISE_LEVELS = (0.02, 0.05, 0.10, 0.20)
DOWNSAMPLE_FACTORS = (1, 2, 3)


def chain_truth() -> np.ndarray:
    truth: np.ndarray = np.zeros((5, 5), dtype=bool)
    truth[0, 1] = True
    truth[1, 2] = True
    truth[2, 3] = True
    truth[3, 4] = True
    return truth


def chain_dataset(
    *, seed: int, n_steps: int, noise_std: float
) -> tuple[np.ndarray, np.ndarray]:
    truth = chain_truth()
    dataset = simulate_calcium_dataset(
        truth,
        n_steps=n_steps,
        gamma=0.88,
        noise_std=noise_std,
        shared_noise_std=0.0,
        spontaneous_rate=0.02,
        transmission_probability=0.85,
        random_state=seed,
    )
    return truth, dataset.fluorescence


def _recovery_row(
    truth: np.ndarray,
    adjacency: np.ndarray,
    **metadata: Any,
) -> dict[str, Any]:
    recovery = edge_recovery(truth, adjacency)
    denominator = recovery.precision + recovery.recall
    return {
        **metadata,
        "precision": recovery.precision,
        "recall": recovery.recall,
        "false_positive_rate": recovery.false_positive_rate,
        "orientation_accuracy": recovery.orientation_accuracy,
        "f1": (
            2.0 * recovery.precision * recovery.recall / denominator
            if denominator
            else 0.0
        ),
        "retained_edges": int(np.count_nonzero(adjacency)),
    }


def _fit_adjacency(method: str, values: np.ndarray) -> np.ndarray:
    adjacency, _, _ = _fit(method, values, max_lag=1, alpha=0.05)
    return adjacency


def h1_rows(*, n_steps: int) -> list[dict[str, Any]]:
    _, traces = chain_dataset(seed=101, n_steps=n_steps, noise_std=0.05)
    transient = characterize_transients(traces, tolerance=0.0)
    return [
        {
            "roi": roi,
            "gamma_hat": transient.gamma[roi],
            "event_density": transient.event_density[roi],
            "rise_count": transient.rise_count[roi],
            "median_rise_duration": transient.median_rise_duration[roi],
            "signal_to_noise": transient.signal_to_noise[roi],
        }
        for roi in range(traces.shape[0])
    ]


def h2_rows(baseline_dir: Path, method: str) -> list[dict[str, Any]]:
    path = baseline_dir / "baseline_rows.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    rows = [
        row
        for row in source
        if row["algorithm"] == method and row.get("truth_target") == "union"
    ]
    if not rows:
        raise ValueError(f"no union-truth {method} rows in {path}")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["representation"])].append(row)
    rng = np.random.default_rng(53)
    output: list[dict[str, Any]] = []
    for (condition, representation), group in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "method": method,
            "condition": condition,
            "representation": representation,
            "n_seeds": len(group),
        }
        for metric in (
            "precision",
            "recall",
            "false_positive_rate",
            "orientation_accuracy",
            "f1",
        ):
            values = np.asarray([float(row[metric]) for row in group], dtype=float)
            lower, upper = _bootstrap_mean_ci(values, rng, 2000)
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            )
            summary[f"{metric}_ci_lower"] = lower
            summary[f"{metric}_ci_upper"] = upper
        output.append(summary)
    return output


def h3_rows(method: str, *, n_steps: int, n_null: int) -> list[dict[str, Any]]:
    truth, traces = chain_dataset(seed=103, n_steps=n_steps, noise_std=0.05)
    represented = representation_map(traces)
    observed = represented["rise"]
    inputs: list[tuple[str, np.ndarray]] = [("observed", observed)]
    inputs.extend(
        (
            f"cyclic_shift_{index}",
            cyclic_shift_surrogate(
                observed, random_state=2000 + index, minimum_shift=3
            ),
        )
        for index in range(1, n_null + 1)
    )
    inputs.extend(
        (
            ("reverse_time", observed[:, ::-1]),
            ("fall", represented["fall"]),
        )
    )
    return [
        _recovery_row(
            truth,
            _fit_adjacency(method, values),
            method=method,
            representation="rise" if comparator != "fall" else "fall",
            comparator=comparator,
        )
        for comparator, values in inputs
    ]


def h4_rows(
    method: str,
    *,
    n_steps: int,
    n_sweep_seeds: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in range(1, n_sweep_seeds + 1):
        for noise_std in NOISE_LEVELS:
            truth, traces = chain_dataset(
                seed=3000 + seed, n_steps=n_steps, noise_std=noise_std
            )
            values = representation_map(traces)["rise"]
            rows.append(
                _recovery_row(
                    truth,
                    _fit_adjacency(method, values),
                    method=method,
                    sweep="noise",
                    seed=seed,
                    parameter=noise_std,
                    representation="rise",
                )
            )
        truth, traces = chain_dataset(seed=4000 + seed, n_steps=n_steps, noise_std=0.05)
        for factor in DOWNSAMPLE_FACTORS:
            values = representation_map(traces[:, ::factor])["rise"]
            rows.append(
                _recovery_row(
                    truth,
                    _fit_adjacency(method, values),
                    method=method,
                    sweep="frame_rate",
                    seed=seed,
                    parameter=factor,
                    representation="rise",
                )
            )
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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


def _holm_adjust(rows: list[dict[str, Any]]) -> None:
    order = sorted(range(len(rows)), key=lambda index: float(rows[index]["p_value"]))
    running = 0.0
    total = len(rows)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * float(rows[index]["p_value"]))
        rows[index]["p_holm"] = min(1.0, running)


def h1_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for metric in (
        "gamma_hat",
        "event_density",
        "rise_count",
        "median_rise_duration",
        "signal_to_noise",
    ):
        values = np.asarray([float(row[metric]) for row in rows], dtype=float)
        output.append(
            {
                "metric": metric,
                "n_rois": values.size,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
        )
    return output


def h2_paired_contrasts(
    baseline_dir: Path,
    method: str,
    *,
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
) -> list[dict[str, Any]]:
    with (baseline_dir / "baseline_rows.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        source = list(csv.DictReader(handle))
    selected = [
        row
        for row in source
        if row["algorithm"] == method and row.get("truth_target") == "union"
    ]
    indexed = {
        (row["condition"], row["representation"], int(row["seed"])): row
        for row in selected
    }
    conditions = sorted({str(row["condition"]) for row in selected})
    representations = sorted(
        {str(row["representation"]) for row in selected} - {"full"}
    )
    seeds = sorted({int(row["seed"]) for row in selected})
    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    for condition in conditions:
        for representation in representations:
            for metric in ("f1", "false_positive_rate"):
                differences = np.asarray(
                    [
                        float(indexed[(condition, representation, unit_seed)][metric])
                        - float(indexed[(condition, "full", unit_seed)][metric])
                        for unit_seed in seeds
                    ],
                    dtype=float,
                )
                lower, upper = _bootstrap_mean_ci(
                    differences, rng, n_bootstrap
                )
                mean = float(np.mean(differences))
                std = (
                    float(np.std(differences, ddof=1))
                    if differences.size > 1
                    else 0.0
                )
                output.append(
                    {
                        "method": method,
                        "condition": condition,
                        "representation": representation,
                        "reference": "full",
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
    _holm_adjust(output)
    return output


def h3_null_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    observed = next(row for row in rows if row["comparator"] == "observed")
    nulls = [row for row in rows if str(row["comparator"]).startswith("cyclic_shift_")]
    output: list[dict[str, Any]] = []
    for metric in ("f1", "retained_edges"):
        null_values = np.asarray([float(row[metric]) for row in nulls], dtype=float)
        observed_value = float(observed[metric])
        std = float(np.std(null_values, ddof=1)) if null_values.size > 1 else 0.0
        output.append(
            {
                "metric": metric,
                "observed": observed_value,
                "n_cyclic_nulls": null_values.size,
                "null_mean": float(np.mean(null_values)),
                "null_std": std,
                "standardized_observed_minus_null": (
                    (observed_value - float(np.mean(null_values))) / std
                    if std > 0.0
                    else None
                ),
                "p_value": float(
                    (1 + np.count_nonzero(null_values >= observed_value))
                    / (null_values.size + 1)
                ),
            }
        )
    _holm_adjust(output)
    return output


def h4_summary(
    rows: Sequence[dict[str, Any]], *, n_bootstrap: int, seed: int
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    for sweep in ("noise", "frame_rate"):
        selected = [row for row in rows if row["sweep"] == sweep]
        for parameter in sorted({float(row["parameter"]) for row in selected}):
            group = [row for row in selected if float(row["parameter"]) == parameter]
            for metric in ("precision", "recall", "false_positive_rate", "f1"):
                values = np.asarray([float(row[metric]) for row in group], dtype=float)
                lower, upper = _bootstrap_mean_ci(values, rng, n_bootstrap)
                output.append(
                    {
                        "sweep": sweep,
                        "parameter": parameter,
                        "metric": metric,
                        "n_seeds": values.size,
                        "mean": float(np.mean(values)),
                        "std": (
                            float(np.std(values, ddof=1))
                            if values.size > 1
                            else 0.0
                        ),
                        "ci_lower": lower,
                        "ci_upper": upper,
                    }
                )
    return output


def h4_extreme_contrasts(
    rows: Sequence[dict[str, Any]],
    *,
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    for sweep in ("noise", "frame_rate"):
        selected = [row for row in rows if row["sweep"] == sweep]
        parameters = sorted({float(row["parameter"]) for row in selected})
        low, high = parameters[0], parameters[-1]
        indexed = {
            (int(row["seed"]), float(row["parameter"])): row for row in selected
        }
        seeds = sorted({int(row["seed"]) for row in selected})
        for metric in ("precision", "recall", "false_positive_rate", "f1"):
            differences = np.asarray(
                [
                    float(indexed[(unit_seed, high)][metric])
                    - float(indexed[(unit_seed, low)][metric])
                    for unit_seed in seeds
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
                    "sweep": sweep,
                    "low_parameter": low,
                    "high_parameter": high,
                    "contrast": "high_minus_low",
                    "metric": metric,
                    "n_pairs": differences.size,
                    "mean_paired_difference": mean,
                    "std_paired_difference": std,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "standardized_mean_difference": mean / std if std > 0.0 else None,
                    "p_value": _sign_flip_pvalue(
                        differences, rng, n_permutations
                    ),
                }
            )
    _holm_adjust(output)
    return output


def _plot_h1(rows: Sequence[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.3))
    roi = [row["roi"] for row in rows]
    axes[0].bar(roi, [row["event_density"] for row in rows], color="#0072B2")
    axes[0].set_title("H1: event density")
    axes[0].set_xlabel("ROI")
    axes[1].bar(roi, [row["median_rise_duration"] for row in rows], color="#D55E00")
    axes[1].set_title("H1: median rise duration")
    axes[1].set_xlabel("ROI")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_h2(rows: Sequence[dict[str, Any]], path: Path) -> None:
    conditions = sorted({row["condition"] for row in rows})
    representations = sorted({row["representation"] for row in rows})
    fig, axes = plt.subplots(1, len(conditions), figsize=(12.0, 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, condition in zip(axes, conditions):
        group = {
            row["representation"]: row for row in rows if row["condition"] == condition
        }
        means = np.asarray([group[item]["f1_mean"] for item in representations])
        lower = means - np.asarray(
            [group[item]["f1_ci_lower"] for item in representations]
        )
        upper = np.asarray(
            [group[item]["f1_ci_upper"] for item in representations]
        ) - means
        axis.bar(
            np.arange(len(representations)),
            means,
            yerr=np.vstack((lower, upper)),
            capsize=3,
            color="#0072B2",
        )
        axis.set_title(condition.replace("_", " "))
        axis.set_xticks(
            np.arange(len(representations)),
            [item.replace("_", "\n") for item in representations],
            fontsize=8,
        )
        axis.set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Mean union-truth F1")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_h3(rows: Sequence[dict[str, Any]], path: Path) -> None:
    indexed = {str(row["comparator"]): row for row in rows}
    nulls = [
        row for row in rows if str(row["comparator"]).startswith("cyclic_shift_")
    ]
    controls = (
        ("observed", "Observed rise", "#D55E00", "-"),
        ("reverse_time", "Reverse time", "#009E73", "--"),
        ("fall", "Fall", "#CC79A7", ":"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for axis, metric, label in zip(
        axes,
        ("f1", "retained_edges"),
        ("Union-truth F1", "Retained directed edges"),
    ):
        values = np.asarray([float(row[metric]) for row in nulls], dtype=float)
        bins = min(15, max(5, int(np.sqrt(values.size))))
        axis.hist(
            values,
            bins=bins,
            color="#8BB8D8",
            edgecolor="white",
            alpha=0.85,
            label="Cyclic-shift null",
        )
        for comparator, control_label, color, linestyle in controls:
            axis.axvline(
                float(indexed[comparator][metric]),
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                label=control_label,
            )
        axis.set_xlabel(label)
        axis.set_ylabel("Null count")
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("H3: observed recovery against the cyclic-shift null", y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_h4(rows: Sequence[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    rng = np.random.default_rng(61)
    for axis, sweep in zip(axes, ("noise", "frame_rate")):
        selected = [row for row in rows if row["sweep"] == sweep]
        parameters = sorted({row["parameter"] for row in selected})
        for metric, marker in (("precision", "o"), ("recall", "s")):
            arrays = [
                np.asarray(
                    [row[metric] for row in selected if row["parameter"] == value],
                    dtype=float,
                )
                for value in parameters
            ]
            means = np.asarray([np.mean(values) for values in arrays])
            intervals = [
                _bootstrap_mean_ci(values, rng, 2000) for values in arrays
            ]
            axis.errorbar(
                parameters,
                means,
                yerr=np.asarray(
                    [
                        means - np.asarray([item[0] for item in intervals]),
                        np.asarray([item[1] for item in intervals]) - means,
                    ]
                ),
                marker=marker,
                capsize=3,
                label=metric,
            )
        axis.set_title(f"H4: {sweep.replace('_', ' ')}")
        axis.set_xlabel("noise SD" if sweep == "noise" else "downsample factor")
        axis.grid(color="#dddddd")
    axes[0].set_ylabel("Recovery metric")
    axes[0].set_ylim(0.0, 1.0)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-steps", type=int, default=1500)
    parser.add_argument("--n-null", type=int, default=99)
    parser.add_argument("--n-sweep-seeds", type=int, default=20)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    if (
        args.n_steps < 100
        or args.n_null < 1
        or args.n_sweep_seeds < 1
        or args.n_bootstrap < 100
        or args.n_permutations < 99
    ):
        raise SystemExit("invalid H1--H4 diagnostic counts")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_h1 = h1_rows(n_steps=args.n_steps)
    rows_h2 = h2_rows(args.baseline_dir, args.method)
    rows_h3 = h3_rows(args.method, n_steps=args.n_steps, n_null=args.n_null)
    rows_h4 = h4_rows(
        args.method,
        n_steps=args.n_steps,
        n_sweep_seeds=args.n_sweep_seeds,
    )
    h1_descriptive = h1_summary(rows_h1)
    h2_contrasts = h2_paired_contrasts(
        args.baseline_dir,
        args.method,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
    h3_summary = h3_null_summary(rows_h3)
    h4_descriptive = h4_summary(
        rows_h4, n_bootstrap=args.n_bootstrap, seed=args.seed + 1
    )
    h4_contrasts = h4_extreme_contrasts(
        rows_h4,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        seed=args.seed + 2,
    )
    outputs = {
        "h1_transient_characterization.csv": rows_h1,
        "h1_characterization_summary.csv": h1_descriptive,
        "h2_representation_recovery.csv": rows_h2,
        "h2_paired_representation_contrasts.csv": h2_contrasts,
        "h3_null_comparators.csv": rows_h3,
        "h3_empirical_null_summary.csv": h3_summary,
        "h4_robustness.csv": rows_h4,
        "h4_robustness_summary.csv": h4_descriptive,
        "h4_extreme_contrasts.csv": h4_contrasts,
    }
    for filename, rows in outputs.items():
        _write_csv(args.output_dir / filename, rows)
    _plot_h1(rows_h1, args.output_dir / "h1_transient.png")
    _plot_h2(rows_h2, args.output_dir / "h2_representations.png")
    _plot_h3(rows_h3, args.output_dir / "h3_nulls.png")
    _plot_h4(rows_h4, args.output_dir / "h4_sweeps.png")
    report = f"""# {LABELS[args.method]} H1--H4 analysis

H1 characterizes the simulated calcium traces and is descriptive at the ROI
level. H2 compares representation-dependent recovery across matched stochastic
seeds, with paired contrasts against the full trace. H3 evaluates the observed
rise result against an empirical cyclic-shift null distribution; reverse-time
and fall inputs remain contextual falsification checks. H4 measures robustness
across noise and frame-rate sweeps using matched seeds.

The four hypotheses answer different questions and are not pooled into a single
method score. All recovery statements refer to the declared synthetic graph.
"""
    (args.output_dir / "analysis-report.md").write_text(report, encoding="utf-8")
    stats = f"""# Statistical appendix

- H1 unit: ROI (`n={len(rows_h1)}`); descriptive statistics only.
- H2 unit: matched simulation seed. Representation-minus-full contrasts report
  paired mean difference, sample standard deviation, standardized difference,
  percentile bootstrap 95% interval, sign-flip p-value, and Holm correction.
- H3 null: `{args.n_null}` cyclic shifts. Empirical upper-tail p-values use the
  finite-null correction `(1 + exceedances)/(1 + n_null)` and Holm correction.
- H4 unit: matched simulation seed (`n={args.n_sweep_seeds}`). Extreme-setting
  contrasts use the same paired bootstrap, effect-size, sign-flip, and Holm procedure.
- Reverse-time and fall checks are contextual and are not treated as independent null draws.
"""
    (args.output_dir / "stats-appendix.md").write_text(stats, encoding="utf-8")
    catalog = """# Figure catalog

## `h1_transient.pdf`
- Purpose: characterize event density and rise duration by ROI.
- Points/bars: individual ROI summaries; no inferential error bars are implied.

## `h2_representations.pdf`
- Purpose: compare union-truth F1 across representations and dynamic conditions.
- Error bars: seed-bootstrap 95% intervals.
- Interpretation: use the paired representation-contrast table for inference.

## `h3_nulls.pdf`
- Purpose: show the empirical cyclic-shift distributions and mark the observed,
  reverse-time, and fall controls.
- Interpretation: only cyclic shifts form the empirical null distribution; the
  vertical control lines are not pooled as null draws.

## `h4_sweeps.pdf`
- Purpose: assess recovery sensitivity to observation noise and downsampling.
- Error bars: seed-bootstrap 95% intervals.
- Interpretation: use the extreme-setting contrast table for paired inference.
"""
    (args.output_dir / "figure-catalog.md").write_text(catalog, encoding="utf-8")
    summary = {
        "status": "complete",
        "method": args.method,
        "method_label": LABELS[args.method],
        "hypotheses": {
            "H1": "kinetic asymmetry and transient characterization",
            "H2": "representation-dependent directed recovery",
            "H3": "cyclic-shift, reverse-time, and fall comparators",
            "H4": "noise and frame-rate robustness",
        },
        "config": {
            "n_steps": args.n_steps,
            "n_null": args.n_null,
            "n_sweep_seeds": args.n_sweep_seeds,
            "n_bootstrap": args.n_bootstrap,
            "n_permutations": args.n_permutations,
            "max_lag": 1,
        },
        "outputs": [
            *outputs,
            "h1_transient.png",
            "h2_representations.png",
            "h3_nulls.png",
            "h4_sweeps.png",
            "h1_transient.pdf",
            "h2_representations.pdf",
            "h3_nulls.pdf",
            "h4_sweeps.pdf",
            "analysis-report.md",
            "stats-appendix.md",
            "figure-catalog.md",
        ],
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
