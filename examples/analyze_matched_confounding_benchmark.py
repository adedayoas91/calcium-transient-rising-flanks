"""Analyze matched shared-noise and hidden-common-driver sensitivity runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


ALGORITHMS = ("cgc", "cgc-star", "pcmciplus", "var-granger", "lpcmci")
LABELS = {
    "cgc": "c-GC",
    "cgc-star": "c-GC*",
    "pcmciplus": "PCMCI+",
    "var-granger": "VAR-Granger",
    "lpcmci": "LPCMCI",
}
CONDITIONS = (
    "native",
    "shared_observation_noise",
    "latent_common_driver",
)
CONDITION_LABELS = {
    "native": "Native",
    "shared_observation_noise": "Shared observation noise",
    "latent_common_driver": "Hidden common driver",
}
KEY_FIELDS = ("condition", "seed", "representation", "truth_target")
SKELETON_METRICS = (
    "skeleton_f1",
    "skeleton_precision",
    "skeleton_recall",
    "skeleton_false_positive_rate",
)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing confounding benchmark rows: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty confounding benchmark file: {path}")
    for row in rows:
        row["seed"] = int(row["seed"])
        row["max_lag"] = int(row["max_lag"])
        row["n_timepoints"] = int(row["n_timepoints"])
        raw_depth = str(row.get("n_pasts", "")).strip()
        row["n_pasts"] = int(float(raw_depth)) if raw_depth else None
        for metric in SKELETON_METRICS:
            row[metric] = float(row[metric])
    return rows


def load_rows(input_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for algorithm in ALGORITHMS:
        algorithm_rows = _read_rows(input_root / algorithm / "baseline_rows.csv")
        if {row["algorithm"] for row in algorithm_rows} != {algorithm}:
            raise ValueError(f"unexpected algorithm declaration for {algorithm}")
        rows.extend(algorithm_rows)
    return rows


def _key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in KEY_FIELDS)


def _primary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["algorithm"] not in {"cgc", "cgc-star"} or row["n_pasts"] == 1
    ]


def validate_design(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if {row["simulation_kind"] for row in rows} != {"confounding"}:
        raise ValueError("all rows must use the confounding simulation kind")
    if {row["condition"] for row in rows} != set(CONDITIONS):
        raise ValueError(
            "native, shared-noise, and latent-driver conditions are required"
        )
    if {row["max_lag"] for row in rows} != {1}:
        raise ValueError("all algorithms must use a one-frame lag horizon")
    for algorithm in ("cgc", "cgc-star"):
        depths = sorted(
            {int(row["n_pasts"]) for row in rows if row["algorithm"] == algorithm}
        )
        if depths != [1, 2, 3]:
            raise ValueError(f"{algorithm} depths must be [1, 2, 3], got {depths}")

    indexes: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {}
    for algorithm in ALGORITHMS:
        selected = [row for row in _primary(rows) if row["algorithm"] == algorithm]
        index = {_key(row): row for row in selected}
        if len(index) != len(selected):
            raise ValueError(f"duplicate primary rows for {algorithm}")
        indexes[algorithm] = index
    reference_keys = set(indexes[ALGORITHMS[0]])
    for algorithm, index in indexes.items():
        if set(index) != reference_keys:
            raise ValueError(f"unmatched evaluation units for {algorithm}")
    for key in sorted(reference_keys):
        reference = indexes[ALGORITHMS[0]][key]
        for algorithm in ALGORITHMS[1:]:
            observed = indexes[algorithm][key]
            for field in ("input_digest", "n_timepoints", "truth_edges"):
                if str(reference[field]) != str(observed[field]):
                    raise ValueError(
                        f"matched-design failure for {key}: {field} differs for {algorithm}"
                    )
    lpcmci_rows = [row for row in rows if row["algorithm"] == "lpcmci"]
    if {row.get("projection") for row in lpcmci_rows} != {"lossy_lagged_pag_skeleton"}:
        raise ValueError(
            "LPCMCI must preserve its PAG and declare the lossy projection"
        )
    return {
        "status": "matched",
        "n_seeds": len({row["seed"] for row in rows}),
        "conditions": list(CONDITIONS),
        "representations": sorted({row["representation"] for row in rows}),
        "max_lag": 1,
        "common_estimand": "lagged skeleton",
        "lpcmci_projection": "lossy_lagged_pag_skeleton",
    }


def _bootstrap_ci(
    values: np.ndarray, rng: np.random.Generator, n_bootstrap: int
) -> tuple[float, float]:
    if values.size == 1:
        return float(values[0]), float(values[0])
    samples = rng.choice(values, size=(n_bootstrap, values.size), replace=True).mean(1)
    lower, upper = np.quantile(samples, (0.025, 0.975))
    return float(lower), float(upper)


def summarize(
    rows: Sequence[dict[str, Any]], *, n_bootstrap: int, seed: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        depth = row["n_pasts"] if row["n_pasts"] is not None else "native"
        grouped[
            (
                row["condition"],
                row["algorithm"],
                row["representation"],
                depth,
            )
        ].append(row)
    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    for group, group_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        condition, algorithm, representation, depth = group
        for metric in SKELETON_METRICS:
            values = np.asarray([row[metric] for row in group_rows], dtype=float)
            lower, upper = _bootstrap_ci(values, rng, n_bootstrap)
            output.append(
                {
                    "condition": condition,
                    "algorithm": algorithm,
                    "representation": representation,
                    "n_pasts": depth,
                    "metric": metric,
                    "n": values.size,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
    return output


def _sign_flip_pvalue(
    differences: np.ndarray, rng: np.random.Generator, n_permutations: int
) -> float:
    observed = abs(float(np.mean(differences)))
    signs = rng.choice((-1.0, 1.0), size=(n_permutations, differences.size))
    null = np.abs(np.mean(signs * differences, axis=1))
    return float((1 + np.count_nonzero(null >= observed)) / (n_permutations + 1))


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted: np.ndarray = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * float(p_values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def paired_degradation(
    rows: Sequence[dict[str, Any]],
    *,
    n_bootstrap: int = 2000,
    n_permutations: int = 9999,
    seed: int = 84,
) -> list[dict[str, Any]]:
    primary = _primary(rows)
    indexed = {
        (row["algorithm"], row["representation"], row["condition"], row["seed"]): row
        for row in primary
    }
    output: list[dict[str, Any]] = []
    algorithms = sorted({row["algorithm"] for row in primary})
    representations = sorted({row["representation"] for row in primary})
    seeds = sorted({row["seed"] for row in primary})
    rng = np.random.default_rng(seed)
    for algorithm in algorithms:
        for representation in representations:
            for condition in CONDITIONS[1:]:
                for metric in ("skeleton_f1", "skeleton_false_positive_rate"):
                    differences = np.asarray(
                        [
                            indexed[(algorithm, representation, condition, seed)][
                                metric
                            ]
                            - indexed[(algorithm, representation, "native", seed)][
                                metric
                            ]
                            for seed in seeds
                        ],
                        dtype=float,
                    )
                    lower, upper = _bootstrap_ci(differences, rng, n_bootstrap)
                    mean = float(np.mean(differences))
                    std = (
                        float(np.std(differences, ddof=1))
                        if differences.size > 1
                        else 0.0
                    )
                    output.append(
                        {
                            "algorithm": algorithm,
                            "representation": representation,
                            "condition": condition,
                            "reference": "native",
                            "metric": metric,
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


def confounder_edge_rows(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Inspect the co-driven target pair and all false skeleton adjacencies."""

    primary = _primary(rows)
    target_specs = {
        str(row.get("latent_driver_targets", "")).strip()
        for row in primary
        if str(row.get("latent_driver_targets", "")).strip()
        not in {"", "None", "nan"}
    }
    if len(target_specs) != 1:
        raise ValueError(
            "the hidden-driver benchmark must declare one target-pair specification"
        )
    target_pair = tuple(int(item) for item in target_specs.pop().split(","))
    if len(target_pair) != 2 or target_pair[0] == target_pair[1]:
        raise ValueError("latent_driver_targets must declare two distinct observed nodes")

    output: list[dict[str, Any]] = []
    for row in primary:
        artifact = Path(str(row["artifact_path"]))
        if not artifact.is_file():
            raise FileNotFoundError(f"missing confounding artifact: {artifact}")
        with np.load(artifact, allow_pickle=False) as payload:
            adjacency = np.asarray(payload["adjacency"], dtype=bool)
            truth = np.asarray(payload["truth"], dtype=bool)
        skeleton = adjacency | adjacency.T
        truth_skeleton = truth | truth.T
        false_skeleton = skeleton & ~truth_skeleton
        upper_false = np.triu(false_skeleton, k=1)
        source, target = target_pair
        pair_selected = bool(skeleton[source, target])
        pair_is_false = not bool(truth_skeleton[source, target])
        if not pair_is_false:
            raise ValueError("the declared co-driven target pair must be absent from truth")
        false_count = int(np.count_nonzero(upper_false))
        output.append(
            {
                "condition": row["condition"],
                "seed": row["seed"],
                "algorithm": row["algorithm"],
                "representation": row["representation"],
                "n_pasts": (
                    row["n_pasts"] if row["n_pasts"] is not None else "native"
                ),
                "target_pair": f"{source}-{target}",
                "target_pair_is_false_in_observed_truth": pair_is_false,
                "target_pair_selected": pair_selected,
                "false_skeleton_edge_count": false_count,
                "other_false_skeleton_edge_count": false_count - int(pair_selected),
                "artifact_path": str(artifact),
            }
        )
    return output


def summarize_confounder_edges(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["condition"]),
                str(row["algorithm"]),
                str(row["representation"]),
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    for (condition, algorithm, representation), group in sorted(grouped.items()):
        selected = np.asarray(
            [float(bool(row["target_pair_selected"])) for row in group], dtype=float
        )
        false_counts = np.asarray(
            [float(row["false_skeleton_edge_count"]) for row in group], dtype=float
        )
        other_counts = np.asarray(
            [float(row["other_false_skeleton_edge_count"]) for row in group],
            dtype=float,
        )
        output.append(
            {
                "condition": condition,
                "algorithm": algorithm,
                "representation": representation,
                "target_pair": group[0]["target_pair"],
                "n_seeds": len(group),
                "target_pair_selection_rate": float(np.mean(selected)),
                "false_skeleton_edge_count_mean": float(np.mean(false_counts)),
                "false_skeleton_edge_count_std": (
                    float(np.std(false_counts, ddof=1)) if len(group) > 1 else 0.0
                ),
                "other_false_skeleton_edge_count_mean": float(np.mean(other_counts)),
            }
        )
    return output


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summaries: Sequence[dict[str, Any]], output_dir: Path) -> None:
    lookup = {
        (
            row["condition"],
            row["algorithm"],
            row["representation"],
            str(row["n_pasts"]),
            row["metric"],
        ): row
        for row in summaries
    }
    representations = sorted({row["representation"] for row in summaries})
    x = np.arange(len(ALGORITHMS))
    offsets = np.linspace(-0.24, 0.24, len(CONDITIONS))
    colors = ("#0072B2", "#D55E00", "#009E73")
    fig, axes = plt.subplots(1, len(representations), figsize=(12.5, 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, representation in zip(axes, representations):
        for condition, offset, color in zip(CONDITIONS, offsets, colors):
            records = []
            for algorithm in ALGORITHMS:
                depth = "1" if algorithm in {"cgc", "cgc-star"} else "native"
                records.append(
                    lookup[
                        (
                            condition,
                            algorithm,
                            representation,
                            depth,
                            "skeleton_f1",
                        )
                    ]
                )
            means = np.asarray([row["mean"] for row in records])
            lower = means - np.asarray([row["ci_lower"] for row in records])
            upper = np.asarray([row["ci_upper"] for row in records]) - means
            axis.errorbar(
                x + offset,
                means,
                yerr=np.vstack((lower, upper)),
                marker="o",
                linestyle="none",
                capsize=2.5,
                color=color,
                label=CONDITION_LABELS[condition],
            )
        axis.set_title(representation.replace("_", " ").title())
        axis.set_xticks(x, [LABELS[item] for item in ALGORITHMS], rotation=25)
        axis.set_ylim(-0.02, 1.02)
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axes[0].set_ylabel("Lagged-skeleton F1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Shared-noise and hidden-common-driver sensitivity", y=1.04)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(
            output_dir / f"matched_confounding_skeleton_f1.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_degradation(
    degradation: Sequence[dict[str, Any]], output_dir: Path
) -> None:
    representations = sorted({str(row["representation"]) for row in degradation})
    fig, axes = plt.subplots(
        1, len(representations), figsize=(6.2 * len(representations), 4.0), sharey=True
    )
    axes = np.atleast_1d(axes)
    x: np.ndarray = np.arange(len(ALGORITHMS), dtype=float)
    width = 0.34
    colors = ("#E69F00", "#0072B2")
    for axis, representation in zip(axes, representations):
        for index, condition in enumerate(CONDITIONS[1:]):
            selected = {
                str(row["algorithm"]): row
                for row in degradation
                if row["representation"] == representation
                and row["condition"] == condition
                and row["metric"] == "skeleton_f1"
            }
            means = np.asarray(
                [selected[algorithm]["mean_paired_difference"] for algorithm in ALGORITHMS]
            )
            lower = means - np.asarray(
                [selected[algorithm]["ci_lower"] for algorithm in ALGORITHMS]
            )
            upper = np.asarray(
                [selected[algorithm]["ci_upper"] for algorithm in ALGORITHMS]
            ) - means
            axis.bar(
                x + (index - 0.5) * width,
                means,
                width,
                yerr=np.vstack((lower, upper)),
                capsize=2.5,
                color=colors[index],
                label=CONDITION_LABELS[condition],
            )
        axis.axhline(0.0, color="#444444", linewidth=0.8)
        axis.set_xticks(x, [LABELS[item] for item in ALGORITHMS], rotation=25)
        axis.set_title(representation.replace("_", " ").title())
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axes[0].set_ylabel("Paired skeleton-F1 change from native")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(
            output_dir / f"paired_confounding_degradation.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_confounder_edge_mechanism(
    summaries: Sequence[dict[str, Any]], output_dir: Path
) -> None:
    representations = sorted({str(row["representation"]) for row in summaries})
    fig, axes = plt.subplots(
        1, len(representations), figsize=(6.2 * len(representations), 4.0), sharey=True
    )
    axes = np.atleast_1d(axes)
    x: np.ndarray = np.arange(len(ALGORITHMS), dtype=float)
    width = 0.24
    colors = ("#999999", "#E69F00", "#0072B2")
    for axis, representation in zip(axes, representations):
        for index, condition in enumerate(CONDITIONS):
            selected = {
                str(row["algorithm"]): row
                for row in summaries
                if row["representation"] == representation
                and row["condition"] == condition
            }
            axis.bar(
                x + (index - 1) * width,
                [selected[algorithm]["target_pair_selection_rate"] for algorithm in ALGORITHMS],
                width,
                color=colors[index],
                label=CONDITION_LABELS[condition],
            )
        axis.set_xticks(x, [LABELS[item] for item in ALGORITHMS], rotation=25)
        axis.set_title(representation.replace("_", " ").title())
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axes[0].set_ylabel("Co-driven target-pair selection rate")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(
            output_dir / f"confounder_target_pair_selection.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", type=Path, default=Path("outputs/matched_confounding_benchmark")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/matched_confounding_benchmark/analysis"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=83)
    args = parser.parse_args()
    if args.n_bootstrap < 100 or args.n_permutations < 99:
        raise SystemExit("use at least 100 bootstrap samples and 99 permutations")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.input_root)
    validation = validate_design(rows)
    summaries = summarize(rows, n_bootstrap=args.n_bootstrap, seed=args.seed)
    degradation = paired_degradation(
        rows,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        seed=args.seed + 1,
    )
    edge_rows = confounder_edge_rows(rows)
    edge_summaries = summarize_confounder_edges(edge_rows)
    _write_csv(args.output_dir / "skeleton_metric_summary.csv", summaries)
    _write_csv(args.output_dir / "paired_native_degradation.csv", degradation)
    _write_csv(args.output_dir / "confounder_edge_mechanism_rows.csv", edge_rows)
    _write_csv(
        args.output_dir / "confounder_edge_mechanism_summary.csv", edge_summaries
    )
    plot_summary(summaries, args.output_dir)
    plot_degradation(degradation, args.output_dir)
    plot_confounder_edge_mechanism(edge_summaries, args.output_dir)
    report = """# Confounding sensitivity benchmark

All algorithms received the same observed matrices within each condition and seed.
The hidden-driver condition simulates an unobserved sixth node with lagged edges
to observed nodes 1 and 3; those two edges are excluded from the observed truth.
Shared observation noise and an explicit hidden common cause are reported
separately.

The common comparison is lagged-skeleton recovery. LPCMCI's raw PAG is retained
in every artifact, and its skeleton is an explicitly lossy projection. Therefore
this analysis does not rank LPCMCI's directed orientations against the directed
outputs of c-GC/c-GC*, PCMCI+, or VAR-Granger. It tests robustness of adjacencies
to confounding mechanisms under one fixed topology.

Native-to-perturbed skeleton-F1 and false-positive-rate changes are paired by
seed. Their percentile bootstrap intervals, standardized paired differences,
and random sign-flip p-values are reported with Holm correction across the
complete degradation family.

The artifact-level mechanism audit additionally tracks the adjacency between
the two observed nodes that share the hidden parent. That pair is absent from
the observed truth, so its selection rate is a targeted false-adjacency check.
Native and shared-noise runs provide controls for the same pair. Total and
other false-skeleton counts are retained so the targeted check is not mistaken
for a complete account of confounding errors.
"""
    (args.output_dir / "analysis-report.md").write_text(report, encoding="utf-8")
    stats = f"""# Statistical appendix

- Unit of analysis: matched simulation seed (`n={validation['n_seeds']}`).
- Descriptive summaries: mean, sample standard deviation, and percentile
  bootstrap 95% interval.
- Robustness contrasts: perturbed-condition minus native-condition skeleton F1
  and skeleton false-positive rate on the same seed and representation.
- Inference: two-sided random sign-flip test with Holm correction over all
  reported degradation contrasts.
- Effect size: paired mean difference standardized by its sample standard deviation.
- Boundary: the fixed topology does not establish robustness across network structures.
- Mechanism audit: the co-driven target-pair selection rate and unordered false
  skeleton-edge counts are descriptive seed-level diagnostics.
"""
    (args.output_dir / "stats-appendix.md").write_text(stats, encoding="utf-8")
    catalog = """# Figure catalog

## `matched_confounding_skeleton_f1.pdf`

- Purpose: compare lagged-skeleton recovery under native, shared-noise, and
  hidden-driver conditions.
- Error bars: seed-bootstrap 95% intervals.
- Interpretation: LPCMCI is shown only through its declared lossy PAG-skeleton projection.

## `paired_confounding_degradation.pdf`

- Purpose: isolate within-seed F1 change from the native condition.
- Error bars: bootstrap 95% intervals of paired differences.
- Interpretation: negative values indicate degraded skeleton recovery; use the
  corrected contrast table for inferential support.

## `confounder_target_pair_selection.pdf`

- Purpose: test whether the two observed nodes sharing the latent parent are
  preferentially connected by a spurious skeleton adjacency.
- Controls: the same absent edge under native and shared-noise conditions.
- Interpretation: selection rate is descriptive; the raw table also reports all
  other false skeleton adjacencies.
"""
    (args.output_dir / "figure-catalog.md").write_text(catalog, encoding="utf-8")
    summary = {
        "status": "complete",
        "validation": validation,
        "n_summary_rows": len(summaries),
        "n_degradation_rows": len(degradation),
        "n_confounder_edge_rows": len(edge_rows),
        "outputs": [
            "skeleton_metric_summary.csv",
            "paired_native_degradation.csv",
            "confounder_edge_mechanism_rows.csv",
            "confounder_edge_mechanism_summary.csv",
            "matched_confounding_skeleton_f1.pdf",
            "matched_confounding_skeleton_f1.png",
            "paired_confounding_degradation.pdf",
            "paired_confounding_degradation.png",
            "confounder_target_pair_selection.pdf",
            "confounder_target_pair_selection.png",
            "analysis-report.md",
            "stats-appendix.md",
            "figure-catalog.md",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
