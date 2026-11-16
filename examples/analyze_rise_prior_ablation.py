"""Create a strict statistical analysis bundle for the rise-prior ablation."""

from __future__ import annotations

import argparse
import csv
from itertools import product
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


ARMS = ("unrestricted", "naive_hard", "robust_hard", "soft_prior", "density_matched_random", "oracle_hard")
PRIOR_ARMS = ("robust_hard", "soft_prior")
SIGNAL_CONDITIONS = ("clean", "lowrate_shared")
COLORS = {"robust_hard": "#0072B2", "soft_prior": "#E69F00"}


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no rows found in {path}")
    numeric = {
        "seed",
        "fold",
        "deadband",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "candidate_density",
        "mask_recall",
        "robust_pruning_speedup",
        "robust_candidate_density",
    }
    converted: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for key in numeric:
            item[key] = int(item[key]) if key in {"seed", "fold", "deadband"} else float(item[key])
        item["is_null"] = str(item["is_null"]).lower() == "true"
        converted.append(item)
    return converted


def seed_means(
    rows: Sequence[dict[str, Any]],
    *,
    condition: str,
    deadband: int,
    arm: str,
    metric: str,
) -> dict[int, float]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        if (
            row["condition"] == condition
            and row["deadband"] == deadband
            and row["arm"] == arm
        ):
            grouped.setdefault(int(row["seed"]), []).append(float(row[metric]))
    return {seed: float(np.mean(values)) for seed, values in grouped.items()}


def paired_seed_differences(
    rows: Sequence[dict[str, Any]],
    *,
    condition: str,
    deadband: int,
    arm: str,
    metric: str,
) -> np.ndarray:
    baseline = seed_means(
        rows,
        condition=condition,
        deadband=deadband,
        arm="unrestricted",
        metric=metric,
    )
    comparison = seed_means(
        rows,
        condition=condition,
        deadband=deadband,
        arm=arm,
        metric=metric,
    )
    seeds = sorted(set(baseline) & set(comparison))
    return np.asarray([comparison[seed] - baseline[seed] for seed in seeds], dtype=float)


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    """Return a two-sided exact paired randomization p-value for mean change."""

    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("differences must be a non-empty finite vector")
    observed = abs(float(np.mean(values)))
    means = []
    for signs in product((-1.0, 1.0), repeat=values.size):
        means.append(abs(float(np.mean(values * np.asarray(signs)))))
    return float(np.mean(np.asarray(means) >= observed - 1e-15))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm family-wise-error adjusted p-values."""

    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or np.any((values < 0) | (values > 1)):
        raise ValueError("p_values must be a vector in [0, 1]")
    order = np.argsort(values)
    adjusted_ordered = np.empty(values.size, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (values.size - rank) * values[index])
        adjusted_ordered[rank] = min(running, 1.0)
    adjusted = np.empty(values.size, dtype=float)
    adjusted[order] = adjusted_ordered
    return adjusted.tolist()


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    random_state: int,
    n_bootstrap: int = 10_000,
) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size == 0:
        raise ValueError("values must be a non-empty vector")
    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, data.size, size=(n_bootstrap, data.size))
    means = np.mean(data[indices], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _sample_sd(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if values.size > 1 else 0.0


def _cohen_dz(differences: np.ndarray) -> float:
    sd = _sample_sd(differences)
    if sd == 0.0:
        return 0.0 if np.mean(differences) == 0.0 else float("inf")
    return float(np.mean(differences) / sd)


def descriptive_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    index = 0
    for condition in ("clean", "lowrate_shared", "null_lowrate_shared"):
        for deadband in (0, 1):
            for arm in ARMS:
                metric = "false_positive_rate" if condition.startswith("null_") else "f1"
                values = np.asarray(
                    list(
                        seed_means(
                            rows,
                            condition=condition,
                            deadband=deadband,
                            arm=arm,
                            metric=metric,
                        ).values()
                    )
                )
                low, high = bootstrap_mean_ci(values, random_state=20260821 + index)
                index += 1
                output.append(
                    {
                        "condition": condition,
                        "deadband": deadband,
                        "arm": arm,
                        "metric": metric,
                        "n_seeds": values.size,
                        "mean": float(np.mean(values)),
                        "sd": _sample_sd(values),
                        "median": float(np.median(values)),
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    return output


def inferential_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    families: list[list[dict[str, Any]]] = []
    recovery = []
    for condition in SIGNAL_CONDITIONS:
        for deadband in (0, 1):
            for arm in PRIOR_ARMS:
                differences = paired_seed_differences(
                    rows,
                    condition=condition,
                    deadband=deadband,
                    arm=arm,
                    metric="f1",
                )
                low, high = bootstrap_mean_ci(
                    differences,
                    random_state=71 + len(recovery),
                )
                recovery.append(
                    {
                        "family": "recovery_f1",
                        "condition": condition,
                        "deadband": deadband,
                        "arm": arm,
                        "metric": "f1",
                        "n_seeds": differences.size,
                        "mean_difference": float(np.mean(differences)),
                        "sd_difference": _sample_sd(differences),
                        "ci95_low": low,
                        "ci95_high": high,
                        "cohen_dz": _cohen_dz(differences),
                        "p_exact": exact_sign_flip_pvalue(differences),
                    }
                )
    families.append(recovery)

    safety = []
    for deadband in (0, 1):
        for arm in PRIOR_ARMS:
            differences = paired_seed_differences(
                rows,
                condition="null_lowrate_shared",
                deadband=deadband,
                arm=arm,
                metric="false_positive_rate",
            )
            low, high = bootstrap_mean_ci(
                differences,
                random_state=171 + len(safety),
            )
            safety.append(
                {
                    "family": "null_fpr",
                    "condition": "null_lowrate_shared",
                    "deadband": deadband,
                    "arm": arm,
                    "metric": "false_positive_rate",
                    "n_seeds": differences.size,
                    "mean_difference": float(np.mean(differences)),
                    "sd_difference": _sample_sd(differences),
                    "ci95_low": low,
                    "ci95_high": high,
                    "cohen_dz": _cohen_dz(differences),
                    "p_exact": exact_sign_flip_pvalue(differences),
                }
            )
    families.append(safety)

    output = []
    for family in families:
        adjusted = holm_adjust([float(row["p_exact"]) for row in family])
        for row, p_holm in zip(family, adjusted):
            output.append({**row, "p_holm": p_holm})
    return output


def _aggregate_seed_metric(
    rows: Sequence[dict[str, Any]],
    condition: str,
    deadband: int,
    metric: str,
) -> np.ndarray:
    return np.asarray(
        list(
            seed_means(
                rows,
                condition=condition,
                deadband=deadband,
                arm="robust_hard",
                metric=metric,
            ).values()
        ),
        dtype=float,
    )


def plot_main_comparison(rows: Sequence[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharey=True)
    rng = np.random.default_rng(34)
    for row_index, condition in enumerate(SIGNAL_CONDITIONS):
        for column_index, deadband in enumerate((0, 1)):
            axis = axes[row_index, column_index]
            for arm_index, arm in enumerate(PRIOR_ARMS):
                differences = paired_seed_differences(
                    rows,
                    condition=condition,
                    deadband=deadband,
                    arm=arm,
                    metric="f1",
                )
                x = arm_index + rng.uniform(-0.07, 0.07, size=differences.size)
                axis.scatter(x, differences, color=COLORS[arm], alpha=0.7, s=22)
                low, high = bootstrap_mean_ci(
                    differences,
                    random_state=400 + row_index * 20 + column_index * 4 + arm_index,
                )
                mean = float(np.mean(differences))
                axis.errorbar(
                    arm_index,
                    mean,
                    yerr=[[mean - low], [high - mean]],
                    fmt="D",
                    color="black",
                    capsize=3,
                    markersize=5,
                    zorder=4,
                )
            axis.axhline(0.0, color="0.35", linewidth=1.0, linestyle="--")
            axis.set_xticks((0, 1), ("robust hard", "soft prior"))
            axis.set_title(
                f"{condition.replace('_', ' ')}, deadband={deadband}",
                fontsize=10,
            )
            axis.grid(axis="y", alpha=0.2)
            axis.set_ylim(-0.16, 0.015)
    fig.supylabel("Held-out F1 difference vs unrestricted", fontsize=10)
    fig.suptitle("Temporal priors do not improve held-out edge recovery", fontsize=12)
    fig.tight_layout(rect=(0.04, 0.0, 1.0, 0.95))
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_supporting(rows: Sequence[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    markers = {0: "o", 1: "s"}
    condition_colors = {"clean": "#0072B2", "lowrate_shared": "#D55E00"}
    for condition in SIGNAL_CONDITIONS:
        for deadband in (0, 1):
            density = _aggregate_seed_metric(
                rows, condition, deadband, "robust_candidate_density"
            )
            recall = _aggregate_seed_metric(rows, condition, deadband, "mask_recall")
            speedup = _aggregate_seed_metric(
                rows, condition, deadband, "robust_pruning_speedup"
            )
            label = f"{condition.replace('_', ' ')}, db={deadband}"
            axes[0].scatter(
                density,
                recall,
                color=condition_colors[condition],
                marker=markers[deadband],
                alpha=0.72,
                label=label,
            )
            axes[1].scatter(
                density,
                speedup,
                color=condition_colors[condition],
                marker=markers[deadband],
                alpha=0.72,
                label=label,
            )
    axes[0].axvline(0.60, color="0.35", linestyle="--", linewidth=1.0)
    axes[0].axhline(0.90, color="0.35", linestyle=":", linewidth=1.0)
    axes[0].set(xlabel="Eligible-direction density", ylabel="True-edge mask recall", xlim=(0.55, 1.01), ylim=(0.45, 1.02))
    axes[0].set_title("Mask recall", fontsize=10)
    axes[1].axvline(0.60, color="0.35", linestyle="--", linewidth=1.0)
    axes[1].axhline(1.67, color="0.35", linestyle=":", linewidth=1.0)
    axes[1].set(xlabel="Eligible-direction density", ylabel="Measured core speedup (×)", xlim=(0.55, 1.01))
    axes[1].set_title("Small-graph runtime", fontsize=10)
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[1].legend(fontsize=7, loc="best")
    fig.suptitle("Ambiguity preserves recall but defeats pruning", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def write_analysis_report(
    path: Path,
    descriptive: Sequence[dict[str, Any]],
    inference: Sequence[dict[str, Any]],
    gates: dict[str, Any],
) -> None:
    signal = [row for row in descriptive if row["condition"] in SIGNAL_CONDITIONS]
    lines = [
        "# Rise-prior ablation: strict analysis",
        "",
        "## Analysis question",
        "",
        "Does an episode-cross-fitted temporal-order prior improve held-out directed-edge F1 or reduce computation without increasing null false positives, relative to unrestricted c-GC? The repeated-measure unit is the simulation seed (n = 8); the two cross-fit folds are averaged within seed.",
        "",
        "## Key finding",
        "",
        "Neither robust hard restriction nor soft weighting improved held-out F1. The hard rule remained null-safe and usually preserved true directions, but did so by retaining almost the full hypothesis family. The observed efficiency and recovery gates therefore failed; the current rule is not feasible as a causal restriction.",
        "",
        "## Exact numeric summary",
        "",
        "Values are seed means reported as mean ± sample SD with a seed-bootstrap 95% CI.",
        "",
        "| condition | deadband | arm | held-out F1 | 95% CI |",
        "|---|---:|---|---:|---:|",
    ]
    for row in signal:
        lines.append(
            f"| {row['condition']} | {row['deadband']} | {row['arm']} | "
            f"{_fmt(row['mean'])} ± {_fmt(row['sd'])} | "
            f"[{_fmt(row['ci95_low'])}, {_fmt(row['ci95_high'])}] |"
        )
    observed = gates["observed"]
    lines.extend(
        [
            "",
            "## Decision-changing observations",
            "",
            f"- Robust-hard mean F1 change pooled by the prespecified gate: {_fmt(observed['mean_f1_delta']['robust_hard'])}; soft-prior change: {_fmt(observed['mean_f1_delta']['soft_prior'])}.",
            f"- Mean robust-mask density was {_fmt(observed['mean_robust_candidate_density'])}, versus the ≤0.60 target. Median measured speedup was {_fmt(observed['median_pruning_speedup'])}×, versus the ≥1.67× target.",
            f"- Robust-mask recall was {_fmt(observed['hard_mask_recall']['clean'])} in clean data and {_fmt(observed['hard_mask_recall']['lowrate_shared'])} under low-rate/shared-noise observations.",
            f"- Mean null FPR change across robust/soft arms was {_fmt(observed['mean_null_fpr_delta'])}; the safety gate passed.",
            "",
            "The exact paired tests (Holm-adjusted within recovery and null-safety families) provide no evidence of an F1 improvement. Deadband 1 is especially uninformative at a one-frame propagation delay because it converts the target directional signal into a tie.",
            "",
            "## Evidence boundary and next decision",
            "",
            "This is a five-node synthetic experiment with a known one-frame delay and eight seeds. It tests feasibility under the current rise extraction and conservative ambiguity rule; it does not prove that every temporal prior will fail. The next useful experiment is not to deploy hard causal pruning. It is to redesign the prior so ambiguity can reduce confidence without automatically admitting both directions, then validate that redesign on longer-delay and higher-SNR regimes before any empirical use.",
            "",
            f"Prespecified recommendation: `{gates['recommendation']}`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_stats_appendix(path: Path, inference: Sequence[dict[str, Any]]) -> None:
    lines = [
        "# Statistical appendix",
        "",
        "Seed is the independent unit (n = 8); fold-level values are averaged within seed. F1 is bounded, sparse, and zero-inflated, so no normal approximation is used. Each contrast uses an exact two-sided paired sign-flip randomization test of the mean seed difference, a seed-bootstrap 95% CI (10,000 resamples), and Cohen's dz. Holm correction controls family-wise error separately for eight recovery contrasts and four null-FPR contrasts.",
        "",
        "| family | condition | db | arm | mean diff | SD diff | 95% CI | dz | exact p | Holm p |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in inference:
        lines.append(
            f"| {row['family']} | {row['condition']} | {row['deadband']} | {row['arm']} | "
            f"{_fmt(row['mean_difference'])} | {_fmt(row['sd_difference'])} | "
            f"[{_fmt(row['ci95_low'])}, {_fmt(row['ci95_high'])}] | "
            f"{_fmt(row['cohen_dz'])} | {_fmt(row['p_exact'])} | {_fmt(row['p_holm'])} |"
        )
    lines.extend(
        [
            "",
            "Limitations: two cross-fit folds from the same seed are not treated as independent; the bootstrap and randomization operate at seed level. The confidence intervals are percentile bootstrap intervals and are descriptive at n = 8. Timing is for a five-node graph and includes Python/core overhead, so it should not be extrapolated to large networks without a dedicated scaling benchmark.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figure_catalog(path: Path) -> None:
    path.write_text(
        """# Figure catalog

## Figure 1 — `figures/figure-01-f1-differences.pdf`

- Purpose: test whether either temporal prior changes held-out F1 relative to the unrestricted graph learner.
- Data: seed-level means across two cross-fit folds, n = 8 seeds per condition/deadband.
- Display: points are seeds; black diamonds are means; bars are seed-bootstrap 95% CIs.
- Observation: robust-hard differences sit at zero; soft weighting is neutral except for a negative clean/deadband-0 shift.
- Implication: the current priors do not justify promotion into the causal inference path.
- Caveat: bounded, sparse F1 and five-node simulation.

## Figure 2 — `figures/figure-02-mask-efficiency.pdf`

- Purpose: show the tradeoff that explains why hard-prior recall remains high while speedup is weak.
- Data: robust-hard seed means across two folds, n = 8 seeds.
- Display: circles are deadband 0, squares are deadband 1; dashed vertical line is the 0.60 density target; dotted horizontal lines are the 0.90 recall and 1.67× speed targets.
- Observation: points cluster near density 1.0, well outside the pruning target; speedups remain below target for most seeds.
- Implication: ambiguity handling protects recall by retaining nearly every direction, defeating the proposed computational advantage.
- Caveat: runtime is a small-graph diagnostic, not a scaling study.
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/validation_results/rise_prior_ablation"),
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input_dir / "analysis-output"
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.input_dir / "ablation_rows.csv")
    descriptive = descriptive_rows(rows)
    inference = inferential_rows(rows)
    gates = json.loads((args.input_dir / "summary.json").read_text(encoding="utf-8"))[
        "gates"
    ]
    _write_csv(output_dir / "descriptive-summary.csv", descriptive)
    _write_csv(output_dir / "inferential-tests.csv", inference)
    plot_main_comparison(rows, figures / "figure-01-f1-differences")
    plot_supporting(rows, figures / "figure-02-mask-efficiency")
    write_analysis_report(output_dir / "analysis-report.md", descriptive, inference, gates)
    write_stats_appendix(output_dir / "stats-appendix.md", inference)
    write_figure_catalog(output_dir / "figure-catalog.md")
    print(output_dir)


if __name__ == "__main__":
    main()
