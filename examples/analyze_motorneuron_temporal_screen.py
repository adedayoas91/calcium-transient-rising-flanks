"""Build a strict descriptive analysis bundle for the motoneuron timing screen."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


RUNS = (
    ("window_120", "Fixed windows (120 frames)"),
    ("window_240", "Fixed windows (240 frames)"),
    ("window_480", "Fixed windows (480 frames)"),
    ("population_bouts", "Population bouts"),
)
PRIMARY_RECORDINGS = frozenset({"F3T1", "F3T2", "F5T2"})
MAXIMUM_DENSITY = 0.60
MINIMUM_REPLICATION = 0.80
COLORS = {
    "window_120": "#0072B2",
    "window_240": "#009E73",
    "population_bouts": "#CC79A7",
    "not_null_separated": "#B8B8B8",
    "pass": "#E69F00",
}


def _optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def read_summary(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    integer_fields = {
        "n_rois",
        "n_frames",
        "inferred_episode_count",
        "max_onset_lag_frames",
        "deadband_frames",
        "n_folds",
    }
    float_fields = {
        "directional_pair_fraction_mean",
        "ambiguous_pair_fraction_mean",
        "unmatched_pair_fraction_mean",
        "candidate_density_mean",
        "heldout_directional_replication_mean",
        "null_directional_pair_fraction_p95_mean",
        "directional_fraction_above_null_p95_mean",
    }
    for row in raw:
        item: dict[str, Any] = dict(row)
        for field in integer_fields:
            item[field] = int(item[field])
        for field in float_fields:
            item[field] = _optional_float(item[field])
        item["passes_empirical_diagnostics"] = (
            item["passes_empirical_diagnostics"].lower() == "true"
        )
        rows.append(item)
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_bundle(root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    rows_by_run: dict[str, list[dict[str, Any]]] = {}
    metadata_by_run: dict[str, dict[str, Any]] = {}
    expected_recordings: set[str] | None = None
    for run, _ in RUNS:
        run_dir = root / run
        metadata = _load_json(run_dir / "summary.json")
        metadata_by_run[run] = metadata
        summary_path = run_dir / "screen_summary.csv"
        rows = read_summary(summary_path) if summary_path.exists() else []
        rows_by_run[run] = rows
        if len(rows) != int(metadata["summary_cell_count"]):
            raise ValueError(f"{run}: CSV/JSON summary-cell count mismatch")
        if len(rows) != len(metadata["summary"]):
            raise ValueError(f"{run}: CSV/embedded-summary count mismatch")
        if sum(row["passes_empirical_diagnostics"] for row in rows) != int(
            metadata["diagnostic_pass_count"]
        ):
            raise ValueError(f"{run}: diagnostic-pass count mismatch")
        for row in rows:
            fractions = [
                row["directional_pair_fraction_mean"],
                row["ambiguous_pair_fraction_mean"],
                row["unmatched_pair_fraction_mean"],
            ]
            if any(value is None for value in fractions) or not np.isclose(
                sum(float(value) for value in fractions), 1.0, atol=1e-10
            ):
                raise ValueError(f"{run}: invalid three-state fractions")
            density_expected = (
                float(row["ambiguous_pair_fraction_mean"])
                + 0.5 * float(row["directional_pair_fraction_mean"])
            )
            if not np.isclose(
                float(row["candidate_density_mean"]), density_expected, atol=1e-10
            ):
                raise ValueError(f"{run}: candidate-density identity failed")
        recordings = {str(row["recording"]) for row in rows}
        if recordings:
            expected_recordings = expected_recordings or recordings
            if recordings != expected_recordings:
                raise ValueError(f"{run}: recording set differs from other valid runs")
    if rows_by_run["window_480"]:
        raise ValueError("480-frame sensitivity was expected to be non-estimable")
    if int(metadata_by_run["window_480"]["recording_count"]) != 9:
        raise ValueError("480-frame run did not attempt all nine recordings")
    if expected_recordings is None or len(expected_recordings) != 9:
        raise ValueError("expected nine valid case-D recordings")
    return rows_by_run, metadata_by_run


def gate_flags(row: dict[str, Any]) -> tuple[bool, bool, bool, bool]:
    density = float(row["candidate_density_mean"]) <= MAXIMUM_DENSITY
    replication_value = row["heldout_directional_replication_mean"]
    replication = (
        replication_value is not None
        and float(replication_value) >= MINIMUM_REPLICATION
    )
    null_separation = float(row["directional_fraction_above_null_p95_mean"]) > 0.0
    combined = density and replication and null_separation
    if combined != bool(row["passes_empirical_diagnostics"]):
        raise ValueError("recomputed diagnostic gate disagrees with saved output")
    return density, replication, null_separation, combined


def gate_summary_rows(
    rows_by_run: dict[str, list[dict[str, Any]]],
    metadata_by_run: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    labels = dict(RUNS)
    for run, _ in RUNS:
        rows = rows_by_run[run]
        flags = [gate_flags(row) for row in rows]
        output.append(
            {
                "run": run,
                "label": labels[run],
                "recordings_attempted": int(metadata_by_run[run]["recording_count"]),
                "valid_cells": len(rows),
                "density_gate_count": sum(flag[0] for flag in flags),
                "replication_gate_count": sum(flag[1] for flag in flags),
                "null_separation_count": sum(flag[2] for flag in flags),
                "all_gates_count": sum(flag[3] for flag in flags),
                "estimability": "estimable" if rows else "not_estimable",
            }
        )
    return output


def passing_rows(rows_by_run: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for run, label in RUNS:
        for row in rows_by_run[run]:
            if not row["passes_empirical_diagnostics"]:
                continue
            output.append(
                {
                    "run": run,
                    "label": label,
                    "recording": row["recording"],
                    "in_primary_manuscript_subset": row["recording"] in PRIMARY_RECORDINGS,
                    "inferred_episode_count": row["inferred_episode_count"],
                    "max_onset_lag_frames": row["max_onset_lag_frames"],
                    "deadband_frames": row["deadband_frames"],
                    "directional_pair_fraction_mean": row[
                        "directional_pair_fraction_mean"
                    ],
                    "ambiguous_pair_fraction_mean": row[
                        "ambiguous_pair_fraction_mean"
                    ],
                    "unmatched_pair_fraction_mean": row[
                        "unmatched_pair_fraction_mean"
                    ],
                    "candidate_density_mean": row["candidate_density_mean"],
                    "heldout_directional_replication_mean": row[
                        "heldout_directional_replication_mean"
                    ],
                    "null_directional_pair_fraction_p95_mean": row[
                        "null_directional_pair_fraction_p95_mean"
                    ],
                    "directional_fraction_above_null_p95_mean": row[
                        "directional_fraction_above_null_p95_mean"
                    ],
                }
            )
    return output


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_gate_space(
    rows_by_run: dict[str, list[dict[str, Any]]], output: Path
) -> None:
    runs = ("window_120", "window_240", "population_bouts")
    labels = dict(RUNS)
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.1), sharex=True, sharey=True)
    for axis, run in zip(axes, runs):
        rows = rows_by_run[run]
        for row in rows:
            replication = row["heldout_directional_replication_mean"]
            if replication is None:
                continue
            null_separated = float(row["directional_fraction_above_null_p95_mean"]) > 0
            passed = bool(row["passes_empirical_diagnostics"])
            axis.scatter(
                float(row["candidate_density_mean"]),
                float(replication),
                s=72 if passed else 34,
                marker="*" if passed else "o",
                color=COLORS["pass"] if passed else (
                    COLORS[run] if null_separated else COLORS["not_null_separated"]
                ),
                edgecolor="black" if passed else "none",
                linewidth=0.6,
                alpha=0.9,
                zorder=3,
            )
        missing = sum(
            row["heldout_directional_replication_mean"] is None for row in rows
        )
        axis.axvline(MAXIMUM_DENSITY, color="black", linestyle="--", linewidth=0.9)
        axis.axhline(MINIMUM_REPLICATION, color="black", linestyle="--", linewidth=0.9)
        axis.set_title(labels[run], fontsize=10)
        axis.text(
            0.02,
            0.03,
            f"{missing} cells without a replication estimate",
            transform=axis.transAxes,
            fontsize=7.5,
            color="#555555",
        )
        axis.grid(color="#E5E5E5", linewidth=0.6, zorder=0)
    axes[0].set_ylabel("Held-out directional replication")
    for axis in axes:
        axis.set_xlabel("Candidate density")
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gate_counts(gate_rows: Sequence[dict[str, Any]], output: Path) -> None:
    estimable = [row for row in gate_rows if row["estimability"] == "estimable"]
    labels = [
        "Density",
        "Replication",
        "Positive null excess",
        "All gates",
    ]
    fields = [
        "density_gate_count",
        "replication_gate_count",
        "null_separation_count",
        "all_gates_count",
    ]
    x = np.arange(len(labels))
    width = 0.24
    fig, axis = plt.subplots(figsize=(8.6, 4.5))
    offsets = np.linspace(-width, width, len(estimable))
    for offset, row in zip(offsets, estimable):
        counts = [int(row[field]) for field in fields]
        bars = axis.bar(
            x + offset,
            counts,
            width=width,
            label=str(row["label"]),
            color=COLORS[str(row["run"])],
        )
        axis.bar_label(bars, padding=2, fontsize=8)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Passing cells (of 54)")
    axis.set_ylim(0, 34)
    axis.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    axis.legend(frameon=False, fontsize=8)
    axis.text(
        0.5,
        -0.22,
        "The 480-frame run was not estimable: each recording yielded fewer than four windows.",
        transform=axis.transAxes,
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _format(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_reports(
    output_dir: Path,
    gate_rows: Sequence[dict[str, Any]],
    passes: Sequence[dict[str, Any]],
    rows_by_run: dict[str, list[dict[str, Any]]],
) -> None:
    gate_lines = [
        "| Analysis | Valid cells | Density | Replication | Positive null excess | All gates |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in gate_rows:
        valid = str(row["valid_cells"]) if row["estimability"] == "estimable" else "0 (not estimable)"
        gate_lines.append(
            f"| {row['label']} | {valid} | {row['density_gate_count']} | "
            f"{row['replication_gate_count']} | {row['null_separation_count']} | "
            f"{row['all_gates_count']} |"
        )
    pass_lines = [
        "| Recording | Segmentation | Episodes | Lag | Deadband | Density | Replication | Null-p95 excess |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in passes:
        pass_lines.append(
            f"| {row['recording']} | {row['label']} | {row['inferred_episode_count']} | "
            f"{row['max_onset_lag_frames']} | {row['deadband_frames']} | "
            f"{_format(row['candidate_density_mean'])} | "
            f"{_format(row['heldout_directional_replication_mean'])} | "
            f"{_format(row['directional_fraction_above_null_p95_mean'])} |"
        )
    report = f"""# Motoneuron temporal-screen analysis

## Question

Can rise-onset order in case-D motoneuron fluorescence reduce the candidate direction family while reproducing across held-out temporal segments and exceeding a within-segment circular-shift null?

## Decision

**Do not apply a temporal hard mask or soft prior to the primary empirical c-GC analysis.** The prespecified fixed-window screens yielded 0/54 diagnostic passes at both 120 and 240 frames. The 480-frame setting was not estimable because the recordings yielded only three non-overlapping windows, fewer than the four required for two-way episode cross-fitting.

Population-bout segmentation yielded 2/54 passes, for F1T2 and F6T2 at lag 1 and deadband 0. Neither recording belongs to the retained manuscript subset (F3T1, F3T2, F5T2), and neither pass reproduced under either fixed-window analysis. Across those three retained recordings, 0/18 cells passed in each estimable analysis.

## Exact gate counts

{chr(10).join(gate_lines)}

The gates were candidate density <= 0.60, held-out directional replication >= 0.80, and positive mean excess of observed directional fraction over fold-specific 95th-percentile thresholds from 20 independent within-episode, per-ROI circular-shift nulls.

## Sensitivity-only passing cells

{chr(10).join(pass_lines)}

The two passes are segmentation-sensitive empirical diagnostics, not evidence of causal connectivity. They authorize, at most, an explicitly sensitivity-labelled learner comparison in those recordings; they do not support pruning the manuscript's primary empirical graphs.

## Interpretation

The fixed-window result is a no-go for the current temporal-prior rule on these recordings. At 120 and 240 frames, direction calls either failed to reproduce across folds or did not exceed the timing-shift null. Population bouts improved apparent replication and null separation, but the only joint passes were absent from the fixed-window screens and from the retained primary subset. The result therefore does not justify a c-GC/c-GC* learner run with temporal pruning.

## Evidence boundary

- This analysis is truth-free and ran no c-GC or c-GC*. It cannot estimate true-edge coverage, direction accuracy, or graph-recovery F1.
- Population bouts are inferred analysis strata rather than experimentally annotated trials.
- A displayed cell is one recording x lag x deadband configuration; cells from the same recording are repeated summaries, not independent biological replicates.
- Stored outputs retain fold-averaged null means and 95th percentiles, not all null draws. No p-values, confidence intervals, or standardized effect sizes are defensible from this bundle.
- The 480-frame result is non-estimability, not evidence that the gates failed.

## Reproducibility

Entry point: `examples/motorneuron_temporal_screen.py`. Data: case D (`dff_smoothed_dict.pkl` with `middle_removed_dict.pkl`) for all nine available recordings. Common settings were lags 1/2/3, deadbands 0/1, tolerance 0, minimum rise run 2, 20 nulls, and random seed 20260821. See `experiment-log.md` in the parent directory and the four run-level `summary.json` files for exact configurations.
"""
    (output_dir / "analysis-report.md").write_text(report, encoding="utf-8")

    stats = f"""# Statistical appendix

## Estimands and units

The screen estimates three descriptive quantities for each recording x maximum-lag x deadband cell: admitted candidate density, held-out directional replication, and directional-fraction excess over a within-episode timing-shift null p95. There are 54 cells per estimable segmentation analysis (9 recordings x 3 lags x 2 deadbands).

The biological recording is the highest independent observational unit available. Lag and deadband cells within a recording reuse the same fluorescence data. The two cross-fit folds are paired validation directions, not independent replicates.

## Decision rules

A cell passes only when density <= {MAXIMUM_DENSITY:.2f}, replication >= {MINIMUM_REPLICATION:.2f}, and the mean excess of observed directional fraction over the fold-specific null p95 thresholds is strictly positive. These are prespecified diagnostic gates; counts are exact descriptions of the stored outputs.

## Inferential limits

Each fold used 20 circular-shift null replicates, but the current artifacts retain only fold-level null means and p95 values. The individual null draws are unavailable. With five fish represented by nine recordings, repeated tuning cells, and inferred rather than annotated bouts, a conventional cell-level hypothesis test would overstate precision. Consequently this analysis reports no p-values, multiplicity-adjusted tests, bootstrap intervals, or standardized effect sizes.

## Data-integrity checks

- CSV row counts agree with the embedded JSON summaries and saved diagnostic-pass counts.
- Directional, ambiguous, and unmatched fractions sum to one for every valid cell.
- Candidate density equals ambiguous fraction plus half the directional fraction for every valid cell.
- Fixed 120, fixed 240, and population-bout analyses each contain all nine case-D recordings and 54 cells.
- Fixed 480 attempted all nine recordings but produced no valid cells because each recording had fewer than four non-overlapping windows.

## Exact gate summary

{chr(10).join(gate_lines)}
"""
    (output_dir / "stats-appendix.md").write_text(stats, encoding="utf-8")

    catalog = """# Figure catalog

## Figure 1: Empirical temporal-screen gate space

- **Why it exists:** shows whether individual recording/configuration cells jointly approach the two numerical gates and distinguishes cells that also exceed the timing-shift null.
- **Source:** `window_120/screen_summary.csv`, `window_240/screen_summary.csv`, and `population_bouts/screen_summary.csv`.
- **What to notice:** fixed-window cells produce no joint passes; the only two stars occur in the population-bout sensitivity. Cells lacking an eligible held-out replication denominator are omitted from the scatter and counted within each panel.
- **Decision implication:** no temporal prior should be applied to the primary empirical graph analysis.
- **Caveat:** points are repeated configurations within recordings, not independent biological replicates.

## Figure 2: Exact diagnostic-gate counts by segmentation

- **Why it exists:** separates the reason a joint pass failed into density, replication, and null-separation components.
- **Source:** the same three estimable summary tables; counts are exact and have no sampling-error bars.
- **What to notice:** neither fixed-window analysis has any null-separated cell, while the population-bout sensitivity has 29 density, 13 replication, 8 null-separation, and 2 joint passes among 54 cells.
- **Decision implication:** the apparent feasibility depends on inferred bout segmentation and does not reproduce under prespecified fixed windows.
- **Caveat:** the 480-frame configuration is annotated as not estimable rather than counted as 54 failures.
"""
    (output_dir / "figure-catalog.md").write_text(catalog, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/motorneurons/temporal_screen_manual"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.input_dir
    output_dir = args.output_dir or root / "analysis-output"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows_by_run, metadata_by_run = validate_bundle(root)
    gate_rows = gate_summary_rows(rows_by_run, metadata_by_run)
    passes = passing_rows(rows_by_run)
    if len(passes) != 2:
        raise ValueError(f"expected two sensitivity-only passes, found {len(passes)}")
    if any(row["in_primary_manuscript_subset"] for row in passes):
        raise ValueError("unexpected pass in the primary manuscript subset")

    write_csv(output_dir / "gate-summary.csv", gate_rows)
    write_csv(output_dir / "passing-cells.csv", passes)
    plot_gate_space(rows_by_run, figures_dir / "figure-01-gate-space")
    plot_gate_counts(gate_rows, figures_dir / "figure-02-gate-counts")
    write_reports(output_dir, gate_rows, passes, rows_by_run)
    print(f"Wrote strict analysis bundle to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
