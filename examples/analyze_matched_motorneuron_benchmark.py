"""Analyze the matched full-axis four-algorithm motoneuron comparison."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


ALGORITHMS = ("cgc", "cgc-star", "pcmciplus", "var-granger")
LABELS = {
    "cgc": "c-GC",
    "cgc-star": "c-GC*",
    "pcmciplus": "PCMCI+",
    "var-granger": "VAR-Granger",
}
METRICS = ("edge_density", "retained_edges", "w_ic", "w_rc")
KEY_FIELDS = ("fluo_type", "recording", "representation")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing benchmark rows: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"benchmark file has no rows: {path}")
    for row in rows:
        row["max_lag"] = int(row["max_lag"])
        row["n_timepoints"] = int(row["n_timepoints"])
        raw_depth = str(row.get("n_pasts", "")).strip()
        row["n_pasts"] = int(float(raw_depth)) if raw_depth else None
        for metric in METRICS:
            raw = str(row.get(metric, "")).strip()
            row[metric] = float(raw) if raw not in {"", "None", "nan"} else np.nan
    return rows


def load_benchmark_rows(input_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for algorithm in ALGORITHMS:
        algorithm_rows = _read_rows(input_root / algorithm / "baseline_rows.csv")
        if {row["algorithm"] for row in algorithm_rows} != {algorithm}:
            raise ValueError(f"unexpected algorithm declaration under {algorithm}")
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
    if {row.get("dataset") for row in rows} != {"motorneurons"}:
        raise ValueError("all rows must come from the motoneuron dataset")
    if {row["max_lag"] for row in rows} != {1}:
        raise ValueError("the matched comparison requires max_lag=1")
    depths = {
        algorithm: sorted(
            {
                int(row["n_pasts"])
                for row in rows
                if row["algorithm"] == algorithm and row["n_pasts"] is not None
            }
        )
        for algorithm in ("cgc", "cgc-star")
    }
    for algorithm, values in depths.items():
        if values != [1, 2, 3]:
            raise ValueError(f"{algorithm} depths must be [1, 2, 3], got {values}")

    indexes: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {}
    for algorithm in ALGORITHMS:
        selected = [row for row in _primary_rows(rows) if row["algorithm"] == algorithm]
        index = {_key(row): row for row in selected}
        if len(index) != len(selected):
            raise ValueError(f"duplicate primary units for {algorithm}")
        indexes[algorithm] = index
    expected = set(indexes[ALGORITHMS[0]])
    for algorithm, index in indexes.items():
        if set(index) != expected:
            raise ValueError(f"unmatched motoneuron units for {algorithm}")
    for key in sorted(expected):
        reference = indexes[ALGORITHMS[0]][key]
        for algorithm in ALGORITHMS[1:]:
            observed = indexes[algorithm][key]
            for field in ("input_digest", "n_timepoints", "n_rois"):
                if str(observed[field]) != str(reference[field]):
                    raise ValueError(
                        f"matched-design failure for {key}: {field} differs for {algorithm}"
                    )

    for algorithm in ("cgc", "cgc-star"):
        depth_reference: dict[tuple[Any, ...], tuple[Any, Any]] = {
            _key(row): (row["input_digest"], row["n_timepoints"])
            for row in rows
            if row["algorithm"] == algorithm and row["n_pasts"] == 1
        }
        for row in rows:
            if (
                row["algorithm"] == algorithm
                and (row["input_digest"], row["n_timepoints"])
                != depth_reference[_key(row)]
            ):
                raise ValueError("conditioning-depth inputs are not identical")
    return {
        "status": "matched",
        "n_primary_units_per_algorithm": len(expected),
        "cases": sorted({row["fluo_type"] for row in rows}),
        "recordings": sorted({row["recording"] for row in rows}),
        "representations": sorted({row["representation"] for row in rows}),
        "max_lag": 1,
        "cgc_depths": depths,
    }


def _bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan, np.nan
    if finite.size == 1:
        return float(finite[0]), float(finite[0])
    means = rng.choice(finite, size=(n_bootstrap, finite.size), replace=True).mean(
        axis=1
    )
    lower, upper = np.quantile(means, (0.025, 0.975))
    return float(lower), float(upper)


def summarize_rows(
    rows: Sequence[dict[str, Any]], *, n_bootstrap: int, seed: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        depth = row["n_pasts"] if row["n_pasts"] is not None else "native"
        grouped[
            (
                row["fluo_type"],
                row["algorithm"],
                row["representation"],
                depth,
            )
        ].append(row)
    rng = np.random.default_rng(seed)
    summaries: list[dict[str, Any]] = []
    for group, group_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        case, algorithm, representation, depth = group
        for metric in METRICS:
            values = np.asarray([row[metric] for row in group_rows], dtype=float)
            finite = values[np.isfinite(values)]
            lower, upper = _bootstrap_ci(finite, rng, n_bootstrap)
            summaries.append(
                {
                    "case": case,
                    "algorithm": algorithm,
                    "representation": representation,
                    "n_pasts": depth,
                    "metric": metric,
                    "n": int(finite.size),
                    "mean": float(np.mean(finite)) if finite.size else np.nan,
                    "std": (float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0),
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
    return summaries


def descriptive_paired_contrasts(
    rows: Sequence[dict[str, Any]], *, n_bootstrap: int, seed: int
) -> list[dict[str, Any]]:
    """Summarize paired sensitivities without assigning inferential p-values."""

    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    primary = list(_primary_rows(rows))
    representation_index = {
        (
            str(row["fluo_type"]),
            str(row["algorithm"]),
            str(row["recording"]),
            str(row["representation"]),
        ): row
        for row in primary
    }
    cases = sorted({str(row["fluo_type"]) for row in primary})
    recordings = sorted({str(row["recording"]) for row in primary})
    for case in cases:
        for algorithm in ALGORITHMS:
            for metric in METRICS:
                pairs = []
                for recording in recordings:
                    rise = float(
                        representation_index[(case, algorithm, recording, "rise")][
                            metric
                        ]
                    )
                    fall = float(
                        representation_index[(case, algorithm, recording, "fall")][
                            metric
                        ]
                    )
                    if np.isfinite(rise) and np.isfinite(fall):
                        pairs.append(fall - rise)
                values = np.asarray(pairs, dtype=float)
                if values.size == 0:
                    continue
                lower, upper = _bootstrap_ci(values, rng, n_bootstrap)
                output.append(
                    {
                        "contrast_family": "representation",
                        "case": case,
                        "algorithm": algorithm,
                        "representation": "fall_minus_rise",
                        "metric": metric,
                        "n_pairs": values.size,
                        "mean_paired_difference": float(np.mean(values)),
                        "std_paired_difference": (
                            float(np.std(values, ddof=1))
                            if values.size > 1
                            else 0.0
                        ),
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "p_value": None,
                    }
                )

    for case in cases:
        for algorithm in ("cgc", "cgc-star"):
            for representation in ("rise", "fall"):
                for depth in (2, 3):
                    depth_differences: list[float] = []
                    for recording in recordings:
                        selected = {
                            int(row["n_pasts"]): row
                            for row in rows
                            if row["fluo_type"] == case
                            and row["algorithm"] == algorithm
                            and row["recording"] == recording
                            and row["representation"] == representation
                        }
                        depth_differences.append(
                            float(selected[depth]["edge_density"])
                            - float(selected[1]["edge_density"])
                        )
                    array = np.asarray(depth_differences, dtype=float)
                    lower, upper = _bootstrap_ci(array, rng, n_bootstrap)
                    output.append(
                        {
                            "contrast_family": "conditioning_depth",
                            "case": case,
                            "algorithm": algorithm,
                            "representation": representation,
                            "metric": "edge_density",
                            "reference_n_pasts": 1,
                            "n_pasts": depth,
                            "n_pairs": array.size,
                            "mean_paired_difference": float(np.mean(array)),
                            "std_paired_difference": (
                                float(np.std(array, ddof=1))
                                if array.size > 1
                                else 0.0
                            ),
                            "ci_lower": lower,
                            "ci_upper": upper,
                            "p_value": None,
                        }
                    )
    return output


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _lookup(
    summaries: Sequence[dict[str, Any]],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {
        (
            row["case"],
            row["algorithm"],
            row["representation"],
            str(row["n_pasts"]),
            row["metric"],
        ): row
        for row in summaries
    }


def plot_primary_density(summaries: Sequence[dict[str, Any]], output_dir: Path) -> None:
    lookup = _lookup(summaries)
    cases = sorted({row["case"] for row in summaries})
    representations = sorted({row["representation"] for row in summaries})
    colors = dict(zip(cases, ("#0072B2", "#D55E00", "#009E73")))
    fig, axes = plt.subplots(1, len(representations), figsize=(9.5, 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(ALGORITHMS))
    offsets = np.linspace(-0.12, 0.12, len(cases))
    for axis, representation in zip(axes, representations):
        for case, offset in zip(cases, offsets):
            records = []
            for algorithm in ALGORITHMS:
                depth = "1" if algorithm in {"cgc", "cgc-star"} else "native"
                records.append(
                    lookup[(case, algorithm, representation, depth, "edge_density")]
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
                capsize=3,
                color=colors[case],
                label=f"Case {case}",
            )
        axis.set_title(representation.capitalize())
        axis.set_xticks(x, [LABELS[item] for item in ALGORITHMS], rotation=20)
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axes[0].set_ylabel("Directed edge density")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(cases), frameon=False)
    fig.suptitle("Matched full-axis motoneuron comparison (physical lag 1)", y=1.03)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(
            output_dir / f"matched_motorneuron_graph_density.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_depth_sensitivity(
    summaries: Sequence[dict[str, Any]], output_dir: Path
) -> None:
    lookup = _lookup(summaries)
    cases = sorted({row["case"] for row in summaries})
    representations = sorted({row["representation"] for row in summaries})
    colors = dict(zip(cases, ("#0072B2", "#D55E00", "#009E73")))
    fig, axes = plt.subplots(2, len(representations), figsize=(9.5, 7.0), sharex=True)
    axes = np.asarray(axes).reshape(2, len(representations))
    for row_index, algorithm in enumerate(("cgc", "cgc-star")):
        for column_index, representation in enumerate(representations):
            axis = axes[row_index, column_index]
            for case in cases:
                records = [
                    lookup[
                        (
                            case,
                            algorithm,
                            representation,
                            str(depth),
                            "edge_density",
                        )
                    ]
                    for depth in (1, 2, 3)
                ]
                means = np.asarray([row["mean"] for row in records])
                lower = means - np.asarray([row["ci_lower"] for row in records])
                upper = np.asarray([row["ci_upper"] for row in records]) - means
                axis.errorbar(
                    (1, 2, 3),
                    means,
                    yerr=np.vstack((lower, upper)),
                    marker="o",
                    capsize=2,
                    color=colors[case],
                    label=f"Case {case}",
                )
            if row_index == 0:
                axis.set_title(representation.capitalize())
            if column_index == 0:
                axis.set_ylabel(f"{LABELS[algorithm]} edge density")
            if row_index == 1:
                axis.set_xlabel("Conditioning depth, n_pasts")
            axis.set_xticks((1, 2, 3))
            axis.grid(color="#dddddd", linewidth=0.7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(cases), frameon=False)
    fig.suptitle("Motoneuron c-GC/c-GC* conditioning-depth sensitivity", y=1.03)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(
            output_dir / f"motorneuron_cgc_conditioning_depth_sensitivity.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", type=Path, default=Path("outputs/matched_motorneuron_benchmark")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/matched_motorneuron_benchmark/analysis"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=79)
    args = parser.parse_args()
    if args.n_bootstrap < 100:
        raise SystemExit("use at least 100 bootstrap samples")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_benchmark_rows(args.input_root)
    validation = validate_matched_design(rows)
    summaries = summarize_rows(rows, n_bootstrap=args.n_bootstrap, seed=args.seed)
    contrasts = descriptive_paired_contrasts(
        rows, n_bootstrap=args.n_bootstrap, seed=args.seed + 1
    )
    _write_csv(args.output_dir / "graph_summary.csv", summaries)
    _write_csv(args.output_dir / "descriptive_paired_contrasts.csv", contrasts)
    plot_primary_density(summaries, args.output_dir)
    plot_depth_sensitivity(summaries, args.output_dir)
    report = """# Matched motoneuron benchmark

The input audit passed: all four CSL algorithms received identical full-axis matrices,
recordings, physical lag horizon, and representation definitions. c-GC/c-GC*
use `n_pasts=1` in the primary plot; depths 2 and 3 are sensitivity settings.

No directed edge truth is available for these recordings. Edge density, W_IC,
and W_RC therefore describe method sensitivity and graph structure, not causal
recovery or comparative accuracy. Bootstrap intervals span only the three
recordings within each preprocessing case and should be interpreted cautiously.
Paired representation and conditioning-depth differences are supplied as
descriptive sensitivity estimates only. No p-values are assigned because the
preprocessing cases and repeated recordings are not independent biological
replicates.
"""
    (args.output_dir / "analysis-report.md").write_text(report, encoding="utf-8")
    stats = f"""# Statistical appendix

- Units summarized: {validation['recordings']} within preprocessing cases
  {validation['cases']}.
- Descriptive outputs: mean, sample standard deviation, and recording-bootstrap
  95% interval.
- Paired contrasts: fall minus rise within a recording, and c-GC/c-GC* depth 2
  or 3 minus depth 1 on the same recording.
- Inferential p-values: not reported, because these units are not treated as
  independent biological replicates.
- Accuracy boundary: no directed graph truth is available.
"""
    (args.output_dir / "stats-appendix.md").write_text(stats, encoding="utf-8")
    catalog = """# Figure catalog

## `matched_motorneuron_graph_density.pdf`

- Purpose: compare four CSL algorithms on matched rise and fall matrices.
- Plotted quantity: directed edge density; error bars are descriptive
  recording-bootstrap 95% intervals.
- Interpretation: differences indicate algorithm sensitivity, not recovery accuracy.

## `motorneuron_cgc_conditioning_depth_sensitivity.pdf`

- Purpose: assess c-GC/c-GC* graph-density sensitivity to `n_pasts=1,2,3`.
- Error bars: descriptive recording-bootstrap 95% intervals.
- Interpretation: use the paired table to quantify changes; no depth is selected
  from these recordings.
"""
    (args.output_dir / "figure-catalog.md").write_text(catalog, encoding="utf-8")
    summary = {
        "status": "complete",
        "validation": validation,
        "n_summary_rows": len(summaries),
        "n_descriptive_contrasts": len(contrasts),
        "outputs": [
            "graph_summary.csv",
            "descriptive_paired_contrasts.csv",
            "matched_motorneuron_graph_density.pdf",
            "matched_motorneuron_graph_density.png",
            "motorneuron_cgc_conditioning_depth_sensitivity.pdf",
            "motorneuron_cgc_conditioning_depth_sensitivity.png",
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
