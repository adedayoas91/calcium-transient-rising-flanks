"""Assemble manuscript-facing evidence tables from saved outputs.

This script is deliberately a packaging step. It reads CSV/JSON artifacts that
already exist under ``outputs/`` and writes interpretation tables, a figure
manifest, and a concise Markdown evidence report. It does not run simulations
or re-estimate empirical graphs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_OUTPUT_ROOT = Path("outputs")
DEFAULT_OUTPUT_DIR = Path("outputs/manuscript_evidence")
DEFAULT_REPRESENTATIVE_FIGURES = ("Rise3.png", "Fall3.png")

CHEN_SUMMARY = Path("chen_comparison/chen_comparison_summary.csv")
SYNTHETIC_TRADEOFFS = Path("validation_tradeoffs/rise_tradeoff_contrasts.csv")
DYNAMIC_A_SUMMARY = Path("validation_results/dynamic_episodic_locked/summary.json")
DYNAMIC_A_REPRESENTATION_SUMMARY = Path(
    "validation_results/dynamic_episodic_locked/representation_summary.csv"
)
DYNAMIC_A_CONTRASTS = Path(
    "validation_results/dynamic_episodic_locked/rise_fall_contrasts.csv"
)
EMPIRICAL_TESTS = Path("empirical_stats/paired_signflip_tests.csv")
GRAPH_STABILITY = Path("graph_stability/graph_stability_summary.csv")
EMPIRICAL_NULL_CONTRASTS = Path("empirical_null_controls/null_control_contrasts.csv")
EMPIRICAL_STABILITY_SUMMARY = Path("empirical_stability/stability_summary.csv")
READINESS_ROWS = Path("result_readiness/result_readiness_rows.csv")

PRIMARY_METRICS = (
    "w_ic",
    "w_rc",
    "edge_density",
    "retained_edges",
    "total_weight",
)

EMPIRICAL_NULL_FIELDS = (
    "case",
    "method",
    "representation",
    "null_type",
    "metric",
    "n_observed",
    "n_null",
    "observed_mean",
    "null_mean",
    "observed_minus_null_mean",
    "p_null_ge_observed_mean",
    "evidence_strength",
    "manuscript_use",
)

EMPIRICAL_STABILITY_FIELDS = (
    "case",
    "method",
    "representation",
    "stability_type",
    "n_rows",
    "n_ok",
    "n_skipped",
    "stability_mean",
    "mean_w_ic_mean",
    "mean_w_rc_mean",
    "mean_edge_density_mean",
    "mean_retained_edges_mean",
    "mean_total_weight_mean",
    "evidence_strength",
    "manuscript_use",
)

DYNAMIC_A_FIELDS = (
    "method",
    "event_mode",
    "condition",
    "comparator",
    "n",
    "delta_precision",
    "delta_recall",
    "delta_false_positive_rate",
    "delta_f1",
    "delta_orientation_accuracy",
    "delta_edge_density",
    "comparator_truth_edges_mean",
    "comparator_fall_propagated_total_mean",
    "evidence_strength",
    "manuscript_use",
)

FINAL_FIGURE_PLAN_FIELDS = (
    "figure_id",
    "manuscript_role",
    "source_drafts",
    "status",
    "next_step",
)

FALLBACK_PUBLICATION_GATES = (
    {
        "category": "dynamic_a_locked",
        "artifact": "locked dynamic-A summary",
        "relative_path": "validation_results/dynamic_episodic_locked/summary.json",
        "required_for": "publication-grade causal-rise versus noncausal-fall claim",
        "user_run_required": "True",
        "status": "missing",
    },
    {
        "category": "dynamic_a_locked",
        "artifact": "locked dynamic-A representation summary",
        "relative_path": "validation_results/dynamic_episodic_locked/representation_summary.csv",
        "required_for": "publication-grade dynamic-A representation comparison",
        "user_run_required": "True",
        "status": "missing",
    },
    {
        "category": "dynamic_a_locked",
        "artifact": "locked dynamic-A rise/fall contrasts",
        "relative_path": "validation_results/dynamic_episodic_locked/rise_fall_contrasts.csv",
        "required_for": "publication-grade dynamic-A rise-minus-comparator claim",
        "user_run_required": "True",
        "status": "missing",
    },
    {
        "category": "empirical_null_controls",
        "artifact": "empirical null-control contrasts",
        "relative_path": "empirical_null_controls/null_control_contrasts.csv",
        "required_for": "empirical observed-versus-null claim",
        "user_run_required": "True",
        "status": "missing",
    },
    {
        "category": "empirical_stability",
        "artifact": "empirical re-estimation stability summary",
        "relative_path": "empirical_stability/stability_summary.csv",
        "required_for": "empirical graph stability claim",
        "user_run_required": "True",
        "status": "missing",
    },
)


def _empty_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return None if stripped == "" else stripped


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = _empty_to_none(value)
        if value is None:
            return None
    return float(value)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: tuple[str, ...] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or tuple(sorted({field for row in rows for field in row}))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def _tex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _tex_path(value: Any) -> str:
    return rf"\texttt{{{_tex_escape(value)}}}"


def build_chen_interpretation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select manuscript-facing Chen comparison rows from the aggregate table."""

    selected: list[dict[str, Any]] = []
    for row in rows:
        representation = row.get("representation")
        method = row.get("method")
        if representation not in {
            "published",
            "published_direct",
            "full_trace",
            "deconvolved",
            "rise",
            "fall",
            "fall_residual",
            "oasis_spikes",
            "oasis_denoised",
        }:
            continue
        if method not in {
            "chen_improved_gc",
            "chen_bvgc",
            "chen_mvgc",
            "cgc",
            "cgc-star",
            "lpcmci",
            "oasis+cgc",
            "oasis+cgc-star",
        }:
            continue
        if row.get("dataset") != "motoneurons":
            continue
        role = "published comparator"
        if representation == "published_direct":
            role = "direct Chen BVGC/MVGC matrix reproduction"
        elif method == "lpcmci":
            role = "lossy undirected lagged PAG-skeleton baseline"
        elif str(method).startswith("oasis+"):
            role = "OASIS preprocessing with named downstream graph estimator"
        elif representation == "full_trace":
            role = "full-trace method baseline"
        elif representation == "deconvolved":
            role = "deconvolved-trace comparator"
        elif representation == "rise":
            role = "primary rising-flank candidate"
        elif representation == "fall":
            role = "falling-flank negative comparator"
        elif representation == "fall_residual":
            role = "falling-residual negative comparator"
        selected.append(
            {
                "method": method,
                "case": row.get("case") or row.get("fluo_type") or "published",
                "representation": representation,
                "recordings": row.get("recordings"),
                "w_ic_mean": row.get("w_ic_mean"),
                "w_rc_mean": row.get("w_rc_mean"),
                "edge_density_mean": row.get("edge_density_mean"),
                "retained_edges_mean": row.get("retained_edges_mean"),
                "total_weight_mean": row.get("total_weight_mean"),
                "delta_w_ic_rise_minus_fall_mean": row.get(
                    "delta_w_ic_rise_minus_fall_mean"
                ),
                "delta_w_rc_rise_minus_fall_mean": row.get(
                    "delta_w_rc_rise_minus_fall_mean"
                ),
                "manuscript_role": role,
            }
        )
    return selected


def _delta_direction(value: float | None) -> str:
    if value is None:
        return "not_available"
    if value > 0.0:
        return "rise_higher"
    if value < 0.0:
        return "fall_higher"
    return "tie"


def build_empirical_pairing_interpretation_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Turn sign-flip rows into manuscript interpretation rows."""

    selected: list[dict[str, Any]] = []
    for row in rows:
        metric = row.get("metric")
        if metric not in PRIMARY_METRICS:
            continue
        mean_delta = _float_or_none(row.get("mean_delta"))
        p_value = _float_or_none(row.get("p_two_sided_signflip"))
        n_pairs = int(float(row.get("n_pairs") or 0))
        if p_value is None:
            strength = "not_tested"
        elif p_value < 0.05:
            strength = "nominal"
        else:
            strength = "descriptive"
        selected.append(
            {
                "case": row.get("case") or row.get("fluo_type"),
                "method": row.get("method"),
                "metric": metric,
                "n_pairs": n_pairs,
                "mean_delta_rise_minus_fall": mean_delta,
                "median_delta_rise_minus_fall": _float_or_none(
                    row.get("median_delta")
                ),
                "n_positive": int(float(row.get("n_positive") or 0)),
                "n_negative": int(float(row.get("n_negative") or 0)),
                "p_two_sided_signflip": p_value,
                "direction": _delta_direction(mean_delta),
                "evidence_strength": strength,
                "manuscript_use": (
                    "descriptive until empirical null and stability gates are complete"
                ),
            }
        )
    return selected


def _classify_tradeoff(delta_recall: float | None, delta_fpr: float | None) -> str:
    if delta_recall is None:
        return "insufficient"
    if delta_recall <= 0.0:
        return "no_rise_recall_gain"
    if delta_fpr is None:
        return "rise_recall_gain_fpr_unknown"
    if delta_fpr <= 0.0:
        return "rise_recall_gain_without_fpr_cost"
    return "rise_recall_gain_with_fpr_cost"


def build_synthetic_tradeoff_interpretation_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Interpret rise-minus-comparator synthetic recovery tradeoffs."""

    selected: list[dict[str, Any]] = []
    for row in rows:
        split = row.get("split") or "all"
        if split not in {"evaluation", "all"}:
            continue
        delta_recall = _float_or_none(row.get("delta_recall"))
        delta_fpr = _float_or_none(row.get("delta_false_positive_rate"))
        selected.append(
            {
                "method": row.get("method"),
                "event_mode": row.get("event_mode"),
                "condition": row.get("condition"),
                "simulator_mode": row.get("simulator_mode"),
                "split": split,
                "comparator": row.get("comparator"),
                "delta_recall": delta_recall,
                "delta_false_positive_rate": delta_fpr,
                "delta_precision": _float_or_none(row.get("delta_precision")),
                "delta_f1": _float_or_none(row.get("delta_f1")),
                "delta_orientation_accuracy": _float_or_none(
                    row.get("delta_orientation_accuracy")
                ),
                "tradeoff_class": _classify_tradeoff(delta_recall, delta_fpr),
                "manuscript_use": (
                    "method-level synthetic evidence, not empirical biological proof"
                ),
            }
        )
    return selected


def _summary_lookup(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("method") or "not_recorded"),
            str(row.get("event_mode") or ""),
            str(row.get("condition") or ""),
            str(row.get("representation") or ""),
        )
        lookup[key] = row
    return lookup


def _dynamic_delta(row: dict[str, Any], comparator: str, metric: str) -> float | None:
    return _float_or_none(row.get(f"rise_minus_{comparator}_{metric}"))


def _dynamic_evidence_strength(
    *,
    comparator: str,
    delta_recall: float | None,
    delta_fpr: float | None,
    comparator_truth_edges: float | None,
    comparator_fall_propagated: float | None,
) -> str:
    if delta_recall is None:
        return "not_tested"
    zero_truth_control = (
        comparator in {"fall", "fall_residual"}
        and comparator_truth_edges == 0.0
        and comparator_fall_propagated == 0.0
    )
    if delta_recall <= 0.0:
        return "no_rise_recovery_gain"
    if zero_truth_control and (delta_fpr is None or delta_fpr <= 0.0):
        return "rise_gain_zero_truth_fall_without_fpr_cost"
    if zero_truth_control:
        return "rise_gain_zero_truth_fall_with_fpr_cost"
    if delta_fpr is not None and delta_fpr > 0.0:
        return "rise_gain_with_fpr_cost"
    return "rise_gain_without_fpr_cost"


def build_dynamic_a_interpretation_rows(
    summary_rows: list[dict[str, Any]],
    contrast_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize locked dynamic-A contrasts into manuscript-facing rows."""

    summaries = _summary_lookup(summary_rows)
    selected: list[dict[str, Any]] = []
    for row in contrast_rows:
        method = str(row.get("method") or "not_recorded")
        event_mode = str(row.get("event_mode") or "")
        condition = str(row.get("condition") or "")
        for comparator in ("fall", "fall_residual", "full", "deconvolved"):
            delta_recall = _dynamic_delta(row, comparator, "recall")
            delta_fpr = _dynamic_delta(row, comparator, "false_positive_rate")
            if delta_recall is None and delta_fpr is None:
                continue
            comparator_summary = summaries.get(
                (method, event_mode, condition, comparator), {}
            )
            comparator_truth_edges = _float_or_none(
                comparator_summary.get("truth_edges_mean")
            )
            comparator_fall_propagated = _float_or_none(
                comparator_summary.get("fall_propagated_total_mean")
            )
            selected.append(
                {
                    "method": method,
                    "event_mode": event_mode,
                    "condition": condition,
                    "comparator": comparator,
                    "n": int(float(comparator_summary.get("n") or 0)),
                    "delta_precision": _dynamic_delta(row, comparator, "precision"),
                    "delta_recall": delta_recall,
                    "delta_false_positive_rate": delta_fpr,
                    "delta_f1": _dynamic_delta(row, comparator, "f1"),
                    "delta_orientation_accuracy": _dynamic_delta(
                        row, comparator, "orientation_accuracy"
                    ),
                    "delta_edge_density": _dynamic_delta(
                        row, comparator, "edge_density"
                    ),
                    "comparator_truth_edges_mean": comparator_truth_edges,
                    "comparator_fall_propagated_total_mean": (
                        comparator_fall_propagated
                    ),
                    "evidence_strength": _dynamic_evidence_strength(
                        comparator=comparator,
                        delta_recall=delta_recall,
                        delta_fpr=delta_fpr,
                        comparator_truth_edges=comparator_truth_edges,
                        comparator_fall_propagated=comparator_fall_propagated,
                    ),
                    "manuscript_use": (
                        "locked dynamic-A synthetic evidence after user-run "
                        "artifact generation"
                    ),
                }
            )
    return selected


def build_graph_support_interpretation_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Prepare graph-support rows for descriptive manuscript reporting."""

    selected: list[dict[str, Any]] = []
    for row in rows:
        dataset = row.get("dataset")
        representation = row.get("representation")
        if representation not in {
            "full_trace",
            "deconvolved",
            "rise",
            "fall",
            "fall_residual",
            "oasis_spikes",
            "oasis_denoised",
        }:
            continue
        selected.append(
            {
                "dataset": dataset,
                "case": row.get("case") or row.get("fluo_type") or row.get("subset"),
                "method": row.get("method"),
                "representation": representation,
                "n_graphs": row.get("n_graphs"),
                "edge_density_mean": row.get("edge_density_mean"),
                "edge_density_std": row.get("edge_density_std"),
                "retained_edges_mean": row.get("retained_edges_mean"),
                "retained_edges_cv": row.get("retained_edges_cv"),
                "total_weight_mean": row.get("total_weight_mean"),
                "total_weight_cv": row.get("total_weight_cv"),
                "manuscript_use": (
                    "descriptive support; re-estimation stability still needed "
                    "where readiness gates are missing"
                ),
            }
        )
    return selected


def _null_evidence_strength(delta: float | None, p_value: float | None) -> str:
    if delta is None:
        return "not_tested"
    if delta <= 0.0:
        return "observed_not_above_null"
    if p_value is not None and p_value < 0.05:
        return "nominal_observed_above_null"
    return "descriptive_observed_above_null"


def build_empirical_null_interpretation_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Turn empirical null-control contrasts into metric-level claim rows."""

    selected: list[dict[str, Any]] = []
    for row in rows:
        for metric in PRIMARY_METRICS:
            observed_mean = _float_or_none(row.get(f"observed_{metric}_mean"))
            null_mean = _float_or_none(row.get(f"null_{metric}_mean"))
            delta = _float_or_none(row.get(f"observed_minus_null_{metric}_mean"))
            p_value = _float_or_none(row.get(f"p_null_ge_observed_{metric}_mean"))
            selected.append(
                {
                    "case": row.get("case"),
                    "method": row.get("method") or "not_recorded",
                    "representation": row.get("representation"),
                    "null_type": row.get("null_type"),
                    "metric": metric,
                    "n_observed": int(float(row.get("n_observed") or 0)),
                    "n_null": int(float(row.get("n_null") or 0)),
                    "observed_mean": observed_mean,
                    "null_mean": null_mean,
                    "observed_minus_null_mean": delta,
                    "p_null_ge_observed_mean": p_value,
                    "evidence_strength": _null_evidence_strength(delta, p_value),
                    "manuscript_use": (
                        "empirical observed-versus-null evidence after user-run "
                        "artifact generation"
                    ),
                }
            )
    return selected


def _stability_evidence_strength(
    n_ok: int,
    stability_mean: float | None,
) -> str:
    if n_ok <= 0:
        return "not_available"
    if stability_mean is None:
        return "no_stability_value"
    if stability_mean >= 0.75:
        return "high_stability"
    if stability_mean >= 0.5:
        return "moderate_stability"
    return "low_stability"


def build_empirical_stability_interpretation_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize empirical re-estimation stability summaries for the manuscript."""

    selected: list[dict[str, Any]] = []
    for row in rows:
        n_ok = int(float(row.get("n_ok") or 0))
        stability_mean = _float_or_none(row.get("stability_mean"))
        selected.append(
            {
                "case": row.get("case"),
                "method": row.get("method") or "not_recorded",
                "representation": row.get("representation"),
                "stability_type": row.get("stability_type"),
                "n_rows": int(float(row.get("n_rows") or 0)),
                "n_ok": n_ok,
                "n_skipped": int(float(row.get("n_skipped") or 0)),
                "stability_mean": stability_mean,
                "mean_w_ic_mean": _float_or_none(row.get("mean_w_ic_mean")),
                "mean_w_rc_mean": _float_or_none(row.get("mean_w_rc_mean")),
                "mean_edge_density_mean": _float_or_none(
                    row.get("mean_edge_density_mean")
                ),
                "mean_retained_edges_mean": _float_or_none(
                    row.get("mean_retained_edges_mean")
                ),
                "mean_total_weight_mean": _float_or_none(
                    row.get("mean_total_weight_mean")
                ),
                "evidence_strength": _stability_evidence_strength(
                    n_ok, stability_mean
                ),
                "manuscript_use": (
                    "empirical re-estimation stability evidence after user-run "
                    "artifact generation"
                ),
            }
        )
    return selected


def build_evidence_gate_rows(readiness_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add manuscript-facing impact labels to readiness rows."""

    gates: list[dict[str, Any]] = []
    source_rows = readiness_rows if readiness_rows else list(FALLBACK_PUBLICATION_GATES)
    for row in source_rows:
        missing = row.get("status") == "missing"
        user_run = str(row.get("user_run_required")) == "True"
        if missing and user_run:
            impact = "blocks final publication claim"
        elif missing:
            impact = "needs regeneration or inspection"
        else:
            impact = "available for current evidence package"
        gates.append(
            {
                "category": row.get("category"),
                "artifact": row.get("artifact"),
                "relative_path": row.get("relative_path"),
                "status": row.get("status"),
                "required_for": row.get("required_for"),
                "claim_impact": impact,
            }
        )
    return gates


def build_figure_manifest(
    gates: list[dict[str, Any]],
    *,
    has_dynamic_rows: bool = False,
    has_empirical_null_rows: bool = False,
    has_empirical_stability_rows: bool = False,
    figures_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """List planned figures and whether saved evidence is sufficient today."""

    status_by_path = {row["relative_path"]: row["status"] for row in gates}
    representative_ready = (
        figures_dir is not None
        and all((figures_dir / name).is_file() for name in DEFAULT_REPRESENTATIVE_FIGURES)
    )
    null_ready = (
        status_by_path.get("empirical_null_controls/null_control_contrasts.csv")
        == "available"
    )
    stability_ready = (
        status_by_path.get("empirical_stability/stability_summary.csv") == "available"
    )
    dynamic_ready = (
        status_by_path.get("validation_results/dynamic_episodic_locked/summary.json")
        == "available"
        and has_dynamic_rows
    )
    null_ready = null_ready and has_empirical_null_rows
    stability_ready = stability_ready and has_empirical_stability_rows
    return [
        {
            "figure": "metric_summary_rise_vs_fall",
            "status": "available_from_saved_tables",
            "source_artifacts": "chen_comparison/chen_comparison_summary.csv; empirical_stats/paired_signflip_tests.csv",
            "draft_output": "empirical_metric_summary.png",
            "next_step": "assemble final manuscript panel styling",
        },
        {
            "figure": "chen_comparison_panel_or_table",
            "status": "available_from_saved_tables",
            "source_artifacts": "chen_comparison/chen_comparison_summary.csv",
            "draft_output": "chen_comparison_summary.png",
            "next_step": "decide primary rows versus supplement",
        },
        {
            "figure": "synthetic_recall_fpr_tradeoff",
            "status": "available_from_saved_tables",
            "source_artifacts": "validation_tradeoffs/rise_tradeoff_contrasts.csv",
            "draft_output": "synthetic_tradeoff_summary.png",
            "next_step": "keep claim method-level unless locked dynamic-A is added",
        },
        {
            "figure": "representative_network_case_c",
            "status": (
                "available_from_saved_images"
                if representative_ready
                else "missing_saved_images"
            ),
            "source_artifacts": "figures/Rise3.png; figures/Fall3.png",
            "draft_output": (
                "representative_network_case_c.png"
                if representative_ready
                else ""
            ),
            "next_step": (
                "use as representative Case C rise/fall network panel"
                if representative_ready
                else "provide saved Rise3/Fall3 network images"
            ),
        },
        {
            "figure": "dynamic_a_locked_validation",
            "status": "available" if dynamic_ready else "missing_user_run",
            "source_artifacts": "validation_results/dynamic_episodic_locked/summary.json; validation_results/dynamic_episodic_locked/representation_summary.csv; validation_results/dynamic_episodic_locked/rise_fall_contrasts.csv",
            "draft_output": (
                "dynamic_a_locked_summary.png"
                if dynamic_ready
                else "evidence_gate_status.png"
            ),
            "next_step": (
                "inspect locked dynamic-A interpretation"
                if dynamic_ready
                else "run locked dynamic-A script before publication claim"
            ),
        },
        {
            "figure": "empirical_null_and_stability_panel",
            "status": "available" if null_ready and stability_ready else "missing_user_run",
            "source_artifacts": "empirical_null_controls/null_control_contrasts.csv; empirical_stability/stability_summary.csv",
            "draft_output": (
                "empirical_null_summary.png; empirical_stability_summary.png"
                if null_ready and stability_ready
                else "evidence_gate_status.png"
            ),
            "next_step": (
                "inspect null-control and stability interpretations"
                if null_ready and stability_ready
                else "run empirical null controls and re-estimation stability"
            ),
        },
        {
            "figure": "motoneuron_method_graph_support",
            "status": "available_from_saved_tables",
            "source_artifacts": "graph_stability/graph_stability_summary.csv",
            "draft_output": "graph_support_summary.png",
            "next_step": "compare graph support across motorneuron method outputs",
        },
    ]


def _metric_deltas(
    rows: list[dict[str, Any]], metric: str
) -> tuple[int, float | None, float | None]:
    values = [
        float(row["mean_delta_rise_minus_fall"])
        for row in rows
        if row["metric"] == metric and row["mean_delta_rise_minus_fall"] is not None
    ]
    return len(values), _mean(values), max(values) if values else None


def _tradeoff_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row["tradeoff_class"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_markdown_report(
    chen_rows: list[dict[str, Any]],
    empirical_rows: list[dict[str, Any]],
    synthetic_rows: list[dict[str, Any]],
    dynamic_rows: list[dict[str, Any]],
    graph_rows: list[dict[str, Any]],
    empirical_null_rows: list[dict[str, Any]],
    empirical_stability_rows: list[dict[str, Any]],
    gates: list[dict[str, Any]],
    figures: list[dict[str, Any]],
) -> str:
    missing_gates = [
        row for row in gates if row["claim_impact"] == "blocks final publication claim"
    ]
    wic_n, wic_mean, wic_max = _metric_deltas(empirical_rows, "w_ic")
    wrc_n, wrc_mean, wrc_max = _metric_deltas(empirical_rows, "w_rc")
    tradeoff_counts = _tradeoff_counts(synthetic_rows)
    full_trace_rows = [
        row for row in chen_rows if row["representation"] == "full_trace"
    ]
    direct_chen_rows = [
        row for row in chen_rows if row["representation"] == "published_direct"
    ]
    rise_rows = [row for row in chen_rows if row["representation"] == "rise"]
    fall_rows = [row for row in chen_rows if row["representation"] == "fall"]
    motoneuron_graph_rows = [
        row for row in graph_rows if row["dataset"] == "motoneurons"
    ]
    null_nominal = [
        row
        for row in empirical_null_rows
        if row["evidence_strength"] == "nominal_observed_above_null"
    ]
    null_descriptive = [
        row
        for row in empirical_null_rows
        if row["evidence_strength"] == "descriptive_observed_above_null"
    ]
    dynamic_zero_truth_support = [
        row
        for row in dynamic_rows
        if row["evidence_strength"]
        in {
            "rise_gain_zero_truth_fall_without_fpr_cost",
            "rise_gain_zero_truth_fall_with_fpr_cost",
        }
    ]
    stable_rows = [
        row
        for row in empirical_stability_rows
        if row["evidence_strength"] in {"high_stability", "moderate_stability"}
    ]

    lines = [
        "# Manuscript Evidence Package",
        "",
        "## Evidence Boundary",
        "",
    ]
    if missing_gates:
        lines.append(
            "Do not promote the empirical superiority claim yet; required "
            "user-run publication gates are still missing."
        )
        for gate in missing_gates:
            lines.append(f"- `{gate['relative_path']}`: {gate['required_for']}.")
    else:
        lines.append("All tracked publication gates are available.")
    lines.extend(
        [
            "",
            "## Chen-Style Comparison",
            "",
            f"- Full-trace baseline rows available: {len(full_trace_rows)}.",
            f"- Direct Chen BVGC/MVGC matrix rows available: {len(direct_chen_rows)}.",
            f"- Rising-flank rows available: {len(rise_rows)}.",
            f"- Falling-flank rows available: {len(fall_rows)}.",
            (
                "- Published Chen comparator is represented only for the reported "
                "ipsilateral-consistency value; missing fields must not be inferred."
            ),
            "",
            "## Empirical Paired Tests",
            "",
            (
                f"- W_IC rise-minus-fall rows: {wic_n}; mean delta "
                f"{_format_float(wic_mean)}; max delta {_format_float(wic_max)}."
            ),
            (
                f"- W_RC rise-minus-fall rows: {wrc_n}; mean delta "
                f"{_format_float(wrc_mean)}; max delta {_format_float(wrc_max)}."
            ),
            (
                "- These paired tests are descriptive until empirical null-control "
                "and stability outputs exist."
            ),
            "",
            "## Synthetic Tradeoffs",
            "",
        ]
    )
    for key in sorted(tradeoff_counts):
        lines.append(f"- {key}: {tradeoff_counts[key]} rows.")
    lines.extend(["", "## Locked Dynamic-A Validation", ""])
    if dynamic_rows:
        lines.append(f"- Locked dynamic-A interpretation rows available: {len(dynamic_rows)}.")
        lines.append(
            f"- Rise-gain rows against zero-truth fall controls: "
            f"{len(dynamic_zero_truth_support)}."
        )
    else:
        lines.append(
            "- No locked dynamic-A interpretation rows are available in the "
            "evidence package yet."
        )
    lines.extend(["", "## Empirical Null Controls", ""])
    if empirical_null_rows:
        lines.append(f"- Null-control interpretation rows available: {len(empirical_null_rows)}.")
        lines.append(
            f"- Nominal observed-above-null rows: {len(null_nominal)}; "
            f"descriptive observed-above-null rows: {len(null_descriptive)}."
        )
    else:
        lines.append(
            "- No empirical null-control contrast rows are available in the "
            "evidence package yet."
        )
    lines.extend(["", "## Empirical Re-Estimation Stability", ""])
    if empirical_stability_rows:
        lines.append(
            f"- Stability interpretation rows available: "
            f"{len(empirical_stability_rows)}."
        )
        lines.append(
            f"- Rows classified as moderate or high stability: {len(stable_rows)}."
        )
    else:
        lines.append(
            "- No empirical re-estimation stability rows are available in the "
            "evidence package yet."
        )
    lines.extend(["", "## Motorneuron Method Graph Support", ""])
    lines.append(
        f"- Method-specific graph-support rows available: "
        f"{len(motoneuron_graph_rows)}."
    )
    lines.append(
        "- LPCMCI values summarize a lossy undirected lagged PAG skeleton; "
        "they are not directed-weight equivalents of c-GC/c-GC*."
    )
    lines.extend(["", "## Figure Manifest", ""])
    for figure in figures:
        lines.append(
            f"- `{figure['figure']}`: {figure['status']}; "
            f"{figure['next_step']}."
        )
    lines.append("")
    return "\n".join(lines)


def build_latex_snippet(
    dynamic_rows: list[dict[str, Any]],
    empirical_null_rows: list[dict[str, Any]],
    empirical_stability_rows: list[dict[str, Any]],
    gates: list[dict[str, Any]],
    figures: list[dict[str, Any]],
) -> str:
    """Write paste-ready results prose driven only by saved evidence artifacts."""

    missing_gates = [
        row for row in gates if row["claim_impact"] == "blocks final publication claim"
    ]
    dynamic_zero_truth_support = [
        row
        for row in dynamic_rows
        if row["evidence_strength"]
        in {
            "rise_gain_zero_truth_fall_without_fpr_cost",
            "rise_gain_zero_truth_fall_with_fpr_cost",
        }
    ]
    null_nominal = [
        row
        for row in empirical_null_rows
        if row["evidence_strength"] == "nominal_observed_above_null"
    ]
    null_descriptive = [
        row
        for row in empirical_null_rows
        if row["evidence_strength"] == "descriptive_observed_above_null"
    ]
    stable_rows = [
        row
        for row in empirical_stability_rows
        if row["evidence_strength"] in {"high_stability", "moderate_stability"}
    ]
    figure_outputs = [
        row.get("draft_output")
        for row in figures
        if row.get("draft_output") and row.get("status") != "missing_user_run"
    ]

    lines = [
        "% Generated by examples/build_manuscript_evidence_package.py.",
        "% Paste into the Results section only after checking the saved CSVs.",
        r"\paragraph{Publication-gate status.}",
    ]
    if missing_gates:
        lines.append(
            "The current evidence package still lacks "
            f"{len(missing_gates)} user-run publication-gate artifacts: "
            + ", ".join(_tex_path(row["relative_path"]) for row in missing_gates)
            + ". Final empirical superiority claims therefore remain gated."
        )
    else:
        lines.append(
            "All tracked user-run publication-gate artifacts are available in "
            "the saved evidence package."
        )

    lines.extend(["", r"\paragraph{Locked dynamic-A validation.}"])
    if dynamic_rows:
        lines.append(
            f"The locked dynamic-A package produced {len(dynamic_rows)} interpreted "
            f"rise-minus-comparator contrasts, including "
            f"{len(dynamic_zero_truth_support)} contrasts against zero-truth "
            "fall or fall-residual controls."
        )
    else:
        lines.append(
            "Locked dynamic-A interpretation is not yet available because the "
            "publication-grade user-run dynamic-A artifacts have not been saved."
        )

    lines.extend(["", r"\paragraph{Empirical null controls.}"])
    if empirical_null_rows:
        lines.append(
            f"The empirical null-control package produced "
            f"{len(empirical_null_rows)} metric-level interpretation rows. "
            f"{len(null_nominal)} rows are nominal observed-above-null results, "
            f"and {len(null_descriptive)} rows are descriptive observed-above-null "
            "results."
        )
    else:
        lines.append(
            "Empirical null-control interpretation is not yet available because "
            "the user-run null-control contrasts have not been saved."
        )

    lines.extend(["", r"\paragraph{Empirical re-estimation stability.}"])
    if empirical_stability_rows:
        lines.append(
            f"The empirical stability package produced "
            f"{len(empirical_stability_rows)} interpretation rows, with "
            f"{len(stable_rows)} rows classified as moderate or high stability."
        )
    else:
        lines.append(
            "Empirical re-estimation stability interpretation is not yet "
            "available because the user-run stability summary has not been saved."
        )

    lines.extend(["", r"\paragraph{Draft figure artifacts.}"])
    if figure_outputs:
        lines.append(
            "The current figure drafts are "
            + ", ".join(_tex_path(output) for output in figure_outputs)
            + "."
        )
    else:
        lines.append("No draft figure artifacts are currently available.")
    lines.append("")
    return "\n".join(lines)


def _split_draft_outputs(value: Any) -> list[str]:
    if not value:
        return []
    return [
        item.strip()
        for item in str(value).split(";")
        if item.strip()
    ]


def _figure_by_name(figures: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next((row for row in figures if row.get("figure") == name), {})


def _plan_row(
    *,
    figure_id: str,
    manuscript_role: str,
    manifest_row: dict[str, Any],
    available_outputs: set[str],
    fallback_next_step: str,
) -> dict[str, Any]:
    source_drafts = _split_draft_outputs(manifest_row.get("draft_output"))
    all_available = bool(source_drafts) and all(
        draft in available_outputs for draft in source_drafts
    )
    manifest_status = str(manifest_row.get("status") or "missing")
    if manifest_status == "missing_user_run":
        status = "missing_user_run"
    elif all_available:
        status = "ready"
    elif manifest_status.startswith("available"):
        status = "needs_regeneration"
    else:
        status = manifest_status
    return {
        "figure_id": figure_id,
        "manuscript_role": manuscript_role,
        "source_drafts": "; ".join(source_drafts),
        "status": status,
        "next_step": manifest_row.get("next_step") or fallback_next_step,
    }


def build_final_figure_plan(
    figures: list[dict[str, Any]],
    figure_outputs: list[str],
) -> list[dict[str, Any]]:
    """Summarize final manuscript figure assembly state from generated drafts."""

    available_outputs = set(figure_outputs)
    rows = [
        _plan_row(
            figure_id="fig_primary_metric_summary",
            manuscript_role="main rise/fall metric panel",
            manifest_row=_figure_by_name(figures, "metric_summary_rise_vs_fall"),
            available_outputs=available_outputs,
            fallback_next_step="regenerate empirical metric summary",
        ),
        _plan_row(
            figure_id="fig_representative_network_case_c",
            manuscript_role="representative rise/fall network panel",
            manifest_row=_figure_by_name(figures, "representative_network_case_c"),
            available_outputs=available_outputs,
            fallback_next_step="provide saved Case C rise/fall network images",
        ),
        _plan_row(
            figure_id="fig_chen_comparison",
            manuscript_role="published Chen versus local c-GC/c-GC* comparison",
            manifest_row=_figure_by_name(figures, "chen_comparison_panel_or_table"),
            available_outputs=available_outputs,
            fallback_next_step="regenerate Chen comparison summary",
        ),
        _plan_row(
            figure_id="fig_synthetic_tradeoff",
            manuscript_role="synthetic recall versus false-positive tradeoff",
            manifest_row=_figure_by_name(figures, "synthetic_recall_fpr_tradeoff"),
            available_outputs=available_outputs,
            fallback_next_step="regenerate synthetic tradeoff summary",
        ),
        _plan_row(
            figure_id="fig_dynamic_a_locked",
            manuscript_role="locked dynamic-A causal-rise/noncausal-fall panel",
            manifest_row=_figure_by_name(figures, "dynamic_a_locked_validation"),
            available_outputs=available_outputs,
            fallback_next_step="run locked dynamic-A gate and repackage evidence",
        ),
        _plan_row(
            figure_id="fig_empirical_null_robustness",
            manuscript_role="empirical null-control and stability panel",
            manifest_row=_figure_by_name(figures, "empirical_null_and_stability_panel"),
            available_outputs=available_outputs,
            fallback_next_step=(
                "run empirical null controls and re-estimation stability, "
                "then repackage evidence"
            ),
        ),
        _plan_row(
            figure_id="fig_motoneuron_method_graph_support",
            manuscript_role="motorneuron method graph-support panel",
            manifest_row=_figure_by_name(figures, "motoneuron_method_graph_support"),
            available_outputs=available_outputs,
            fallback_next_step="regenerate graph support summary",
        ),
    ]
    return rows


def build_figure_layout_snippet(final_plan: list[dict[str, Any]]) -> str:
    """Write a LaTeX-ready figure assembly status snippet."""

    ready = [row for row in final_plan if row["status"] == "ready"]
    gated = [row for row in final_plan if row["status"] != "ready"]
    lines = [
        "% Generated by examples/build_manuscript_evidence_package.py.",
        r"\paragraph{Final figure assembly.}",
        (
            f"{len(ready)} planned manuscript figure panels are ready from saved "
            f"drafts; {len(gated)} panels still require regeneration, user-run "
            "artifacts, or final styling."
        ),
    ]
    for row in final_plan:
        lines.append(
            rf"\noindent {_tex_escape(row['figure_id'])}: "
            rf"{_tex_escape(row['status'])}; "
            rf"{_tex_escape(row['next_step'])}.\\"
        )
    lines.append("")
    return "\n".join(lines)


def _plot_empirical_metric_summary(
    rows: list[dict[str, Any]], output_path: Path
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    paired = [
        row
        for row in rows
        if row["method"] in {"cgc", "cgc-star", "lpcmci"}
        and row["representation"] in {"rise", "fall"}
    ]
    if not paired:
        return False
    groups = sorted({(str(row["method"]), str(row["case"])) for row in paired})
    group_labels = [f"{method}\n{case}" for method, case in groups]
    metrics = ("w_ic_mean", "w_rc_mean", "edge_density_mean")
    labels = ("W_IC", "W_RC", "Edge density")
    fig, axes = plt.subplots(1, len(metrics), figsize=(10, 3.2), constrained_layout=True)
    for axis, metric, label in zip(axes, metrics, labels, strict=True):
        for representation, marker in (("rise", "o"), ("fall", "s")):
            values = []
            for method, case in groups:
                match = next(
                    (
                        row
                        for row in paired
                        if str(row["method"]) == method
                        and str(row["case"]) == case
                        and row["representation"] == representation
                    ),
                    None,
                )
                values.append(_float_or_none(None if match is None else match[metric]))
            axis.plot(group_labels, values, marker=marker, label=representation)
        axis.set_title(label)
        axis.set_xlabel("Method / fluorescence")
        axis.tick_params(axis="x", labelrotation=35, labelsize=7)
    axes[0].legend(frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def _plot_synthetic_tradeoffs(
    rows: list[dict[str, Any]], output_path: Path
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    plotted = [
        row
        for row in rows
        if row["delta_recall"] is not None
        and row["delta_false_positive_rate"] is not None
    ]
    if not plotted:
        return False
    fig, axis = plt.subplots(figsize=(5, 4), constrained_layout=True)
    for row in plotted:
        axis.scatter(row["delta_false_positive_rate"], row["delta_recall"], s=28)
    axis.axhline(0.0, color="0.4", linewidth=0.8)
    axis.axvline(0.0, color="0.4", linewidth=0.8)
    axis.set_xlabel("Rise minus comparator FPR")
    axis.set_ylabel("Rise minus comparator recall")
    axis.set_title("Synthetic rise tradeoff summary")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def _plot_dynamic_a(
    rows: list[dict[str, Any]], output_path: Path
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    plotted = [
        row
        for row in rows
        if row["delta_recall"] is not None
        and row["delta_false_positive_rate"] is not None
    ]
    if not plotted:
        return False
    labels = [
        f"{row['method']} {row['event_mode']}\n{row['condition']} vs {row['comparator']}"
        for row in plotted
    ]
    recall = [float(row["delta_recall"]) for row in plotted]
    fpr = [float(row["delta_false_positive_rate"]) for row in plotted]
    x = list(range(len(plotted)))
    fig, axis = plt.subplots(
        figsize=(max(6.0, 0.65 * len(plotted)), 3.6), constrained_layout=True
    )
    axis.bar([value - 0.18 for value in x], recall, width=0.36, label="Delta recall")
    axis.bar([value + 0.18 for value in x], fpr, width=0.36, label="Delta FPR")
    axis.axhline(0.0, color="0.3", linewidth=0.8)
    axis.set_xticks(x, labels, rotation=45, ha="right", fontsize=7)
    axis.set_title("Locked dynamic-A rise-minus-comparator contrasts")
    axis.legend(frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def _plot_representative_networks(figures_dir: Path, output_path: Path) -> bool:
    try:
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
    except Exception:
        return False

    paths = [figures_dir / name for name in DEFAULT_REPRESENTATIVE_FIGURES]
    if not all(path.is_file() for path in paths):
        return False

    titles = ("Case C rise", "Case C fall")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.0), constrained_layout=True)
    for axis, path, title in zip(axes, paths, titles, strict=True):
        axis.imshow(mpimg.imread(path))
        axis.set_title(title)
        axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def _bar_labels(rows: list[dict[str, Any]]) -> list[str]:
    labels = []
    for row in rows:
        method = str(row.get("method") or "")
        case = str(row.get("case") or "")
        representation = str(row.get("representation") or "")
        if method == "chen_improved_gc":
            labels.append("Chen\npublished")
        elif representation == "published_direct":
            labels.append(f"{method}\n{case}")
        elif representation == "full_trace":
            labels.append(f"{method}\n{case}")
        else:
            labels.append(f"{method}\n{case} {representation}")
    return labels


def _plot_chen_comparison(rows: list[dict[str, Any]], output_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    plotted = [
        row
        for row in rows
        if row["representation"]
        in {
            "published",
            "published_direct",
            "full_trace",
            "deconvolved",
            "rise",
            "fall",
            "fall_residual",
            "oasis_spikes",
            "oasis_denoised",
        }
    ]
    if not plotted:
        return False
    labels = _bar_labels(plotted)
    x = range(len(plotted))
    metrics = (("w_ic_mean", "W_IC"), ("w_rc_mean", "W_RC"))
    colors = {
        "published": "#555555",
        "published_direct": "#777777",
        "full_trace": "#4c78a8",
        "deconvolved": "#72b7b2",
        "rise": "#2f855a",
        "fall": "#c05621",
        "fall_residual": "#f2cf5b",
        "oasis_spikes": "#b279a2",
        "oasis_denoised": "#ff9da6",
    }
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6), constrained_layout=True)
    for axis, (metric, label) in zip(axes, metrics, strict=True):
        values = [_float_or_none(row.get(metric)) for row in plotted]
        heights = [0.0 if value is None else value for value in values]
        axis.bar(
            list(x),
            heights,
            color=[colors.get(str(row["representation"]), "#666666") for row in plotted],
        )
        for index, value in enumerate(values):
            if value is None:
                axis.text(index, 0.03, "NA", ha="center", va="bottom", fontsize=7)
        axis.set_ylabel(label)
        axis.set_ylim(0.0, 1.05)
    axes[-1].set_xticks(list(x), labels, rotation=45, ha="right", fontsize=7)
    axes[0].set_title("Chen-style motoneuron graph summaries")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def _plot_graph_support(rows: list[dict[str, Any]], output_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    motoneuron = [row for row in rows if row["dataset"] == "motoneurons"]
    if not motoneuron:
        return False

    labels = [
        f"{row['method']}\n{row['case']} {row['representation']}"
        for row in motoneuron
    ]
    values = [
        _float_or_none(row.get("edge_density_mean")) or 0.0 for row in motoneuron
    ]
    fig, axis = plt.subplots(
        figsize=(max(7.0, 0.48 * len(motoneuron)), 3.8),
        constrained_layout=True,
    )
    axis.bar(range(len(motoneuron)), values, color="#4c78a8")
    axis.set_xticks(
        range(len(motoneuron)), labels, rotation=45, ha="right", fontsize=7
    )
    axis.set_title("Motorneuron graph support by method and representation")
    axis.set_ylabel("Edge density")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def _plot_evidence_gates(rows: list[dict[str, Any]], output_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    if not rows:
        return False
    labels = [str(row["category"]) for row in rows]
    values = [1 for _row in rows]
    colors = [
        "#2f855a" if row["status"] == "available" else "#b83232" for row in rows
    ]
    fig, axis = plt.subplots(figsize=(8, max(3.0, 0.32 * len(rows))), constrained_layout=True)
    axis.barh(range(len(rows)), values, color=colors)
    axis.set_yticks(range(len(rows)), labels, fontsize=7)
    axis.set_xticks([])
    axis.set_xlim(0, 1)
    axis.set_title("Publication evidence gates")
    for index, row in enumerate(rows):
        if row["claim_impact"] == "blocks final publication claim":
            text = "missing user-run"
        elif row["status"] == "missing":
            text = "missing"
        else:
            text = "available"
        axis.text(0.03, index, text, va="center", fontsize=7, color="white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def _plot_empirical_nulls(
    rows: list[dict[str, Any]], output_path: Path
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    plotted = [
        row
        for row in rows
        if row["metric"] in {"w_ic", "w_rc"}
        and row["observed_minus_null_mean"] is not None
    ]
    if not plotted:
        return False
    labels = [
        f"{row['case']} {row['representation']}\n{row['null_type']} {row['metric']}"
        for row in plotted
    ]
    values = [float(row["observed_minus_null_mean"]) for row in plotted]
    colors = [
        "#2f855a" if value > 0.0 else "#b83232"
        for value in values
    ]
    height = max(3.0, 0.28 * len(plotted))
    fig, axis = plt.subplots(figsize=(8, height), constrained_layout=True)
    axis.barh(range(len(plotted)), values, color=colors)
    axis.axvline(0.0, color="0.3", linewidth=0.8)
    axis.set_yticks(range(len(plotted)), labels, fontsize=6)
    axis.set_xlabel("Observed minus null mean")
    axis.set_title("Empirical null-control contrasts")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def _plot_empirical_stability(
    rows: list[dict[str, Any]], output_path: Path
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    plotted = [row for row in rows if row["stability_mean"] is not None]
    if not plotted:
        return False
    labels = [
        f"{row['case']} {row['representation']}\n{row['stability_type']}"
        for row in plotted
    ]
    values = [float(row["stability_mean"]) for row in plotted]
    fig, axis = plt.subplots(
        figsize=(8, max(3.0, 0.28 * len(plotted))), constrained_layout=True
    )
    axis.barh(range(len(plotted)), values, color="#4c78a8")
    axis.set_yticks(range(len(plotted)), labels, fontsize=6)
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Mean graph Jaccard stability")
    axis.set_title("Empirical re-estimation stability")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def _plot_main_results_overview(
    output_dir: Path,
    figure_outputs: list[str],
    output_path: Path,
) -> bool:
    try:
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
    except Exception:
        return False

    panel_names = [
        "empirical_metric_summary.png",
        "representative_network_case_c.png",
        "chen_comparison_summary.png",
        "synthetic_tradeoff_summary.png",
        "dynamic_a_locked_summary.png",
        "empirical_null_summary.png",
        "empirical_stability_summary.png",
    ]
    available = [
        name
        for name in panel_names
        if name in figure_outputs and (output_dir / name).is_file()
    ]
    if len(available) < 2:
        return False
    n_cols = 2
    n_rows = (len(available) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(8.5, 3.2 * n_rows),
        constrained_layout=True,
    )
    flat_axes = np.asarray(axes).reshape(-1)
    for axis, name in zip(flat_axes, available, strict=False):
        axis.imshow(mpimg.imread(output_dir / name))
        axis.set_title(Path(name).stem.replace("_", " "))
        axis.axis("off")
    for axis in flat_axes[len(available):]:
        axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def build_package(
    output_root: Path,
    output_dir: Path,
    *,
    make_figures: bool = True,
    figures_dir: Path | None = None,
    dynamic_output_dir: Path | None = None,
    null_output_dir: Path | None = None,
    stability_output_dir: Path | None = None,
    readiness_output_dir: Path | None = None,
) -> dict[str, Any]:
    resolved_figures_dir = resolve_figures_dir(figures_dir)
    resolved_dynamic_output_dir = dynamic_output_dir or (
        output_root / "validation_results/dynamic_episodic_locked"
    )
    resolved_null_output_dir = null_output_dir or (
        output_root / "empirical_null_controls"
    )
    resolved_stability_output_dir = stability_output_dir or (
        output_root / "empirical_stability"
    )
    resolved_readiness_output_dir = readiness_output_dir or (
        output_root / "result_readiness"
    )
    chen_rows = build_chen_interpretation_rows(_read_csv(output_root / CHEN_SUMMARY))
    empirical_rows = build_empirical_pairing_interpretation_rows(
        _read_csv(output_root / EMPIRICAL_TESTS)
    )
    synthetic_rows = build_synthetic_tradeoff_interpretation_rows(
        _read_csv(output_root / SYNTHETIC_TRADEOFFS)
    )
    dynamic_rows = build_dynamic_a_interpretation_rows(
        _read_csv(resolved_dynamic_output_dir / DYNAMIC_A_REPRESENTATION_SUMMARY.name),
        _read_csv(resolved_dynamic_output_dir / DYNAMIC_A_CONTRASTS.name),
    )
    graph_rows = build_graph_support_interpretation_rows(
        _read_csv(output_root / GRAPH_STABILITY)
    )
    empirical_null_rows = build_empirical_null_interpretation_rows(
        _read_csv(resolved_null_output_dir / EMPIRICAL_NULL_CONTRASTS.name)
    )
    empirical_stability_rows = build_empirical_stability_interpretation_rows(
        _read_csv(resolved_stability_output_dir / EMPIRICAL_STABILITY_SUMMARY.name)
    )
    gates = build_evidence_gate_rows(
        _read_csv(resolved_readiness_output_dir / READINESS_ROWS.name)
    )
    figures = build_figure_manifest(
        gates,
        has_dynamic_rows=bool(dynamic_rows),
        has_empirical_null_rows=bool(empirical_null_rows),
        has_empirical_stability_rows=bool(empirical_stability_rows),
        figures_dir=resolved_figures_dir,
    )
    markdown = build_markdown_report(
        chen_rows,
        empirical_rows,
        synthetic_rows,
        dynamic_rows,
        graph_rows,
        empirical_null_rows,
        empirical_stability_rows,
        gates,
        figures,
    )
    latex_snippet = build_latex_snippet(
        dynamic_rows,
        empirical_null_rows,
        empirical_stability_rows,
        gates,
        figures,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "chen_interpretation_table.csv", chen_rows)
    _write_csv(output_dir / "empirical_pairing_interpretation.csv", empirical_rows)
    _write_csv(output_dir / "synthetic_tradeoff_interpretation.csv", synthetic_rows)
    _write_csv(
        output_dir / "dynamic_a_interpretation.csv",
        dynamic_rows,
        DYNAMIC_A_FIELDS,
    )
    _write_csv(output_dir / "graph_support_interpretation.csv", graph_rows)
    _write_csv(
        output_dir / "empirical_null_interpretation.csv",
        empirical_null_rows,
        EMPIRICAL_NULL_FIELDS,
    )
    _write_csv(
        output_dir / "empirical_stability_interpretation.csv",
        empirical_stability_rows,
        EMPIRICAL_STABILITY_FIELDS,
    )
    _write_csv(output_dir / "evidence_gates.csv", gates)
    _write_csv(output_dir / "figure_manifest.csv", figures)
    (output_dir / "manuscript_evidence_report.md").write_text(markdown)
    (output_dir / "manuscript_results_snippets.tex").write_text(latex_snippet)

    figure_outputs: list[str] = []
    if make_figures:
        if _plot_empirical_metric_summary(
            chen_rows, output_dir / "empirical_metric_summary.png"
        ):
            figure_outputs.append("empirical_metric_summary.png")
        if _plot_synthetic_tradeoffs(
            synthetic_rows, output_dir / "synthetic_tradeoff_summary.png"
        ):
            figure_outputs.append("synthetic_tradeoff_summary.png")
        if _plot_dynamic_a(
            dynamic_rows, output_dir / "dynamic_a_locked_summary.png"
        ):
            figure_outputs.append("dynamic_a_locked_summary.png")
        if _plot_chen_comparison(
            chen_rows, output_dir / "chen_comparison_summary.png"
        ):
            figure_outputs.append("chen_comparison_summary.png")
        if _plot_representative_networks(
            resolved_figures_dir,
            output_dir / "representative_network_case_c.png",
        ):
            figure_outputs.append("representative_network_case_c.png")
        if _plot_graph_support(
            graph_rows, output_dir / "graph_support_summary.png"
        ):
            figure_outputs.append("graph_support_summary.png")
        if _plot_empirical_nulls(
            empirical_null_rows, output_dir / "empirical_null_summary.png"
        ):
            figure_outputs.append("empirical_null_summary.png")
        if _plot_empirical_stability(
            empirical_stability_rows,
            output_dir / "empirical_stability_summary.png",
        ):
            figure_outputs.append("empirical_stability_summary.png")
        if _plot_evidence_gates(gates, output_dir / "evidence_gate_status.png"):
            figure_outputs.append("evidence_gate_status.png")
        if _plot_main_results_overview(
            output_dir,
            figure_outputs,
            output_dir / "main_results_overview.png",
        ):
            figure_outputs.append("main_results_overview.png")

    final_figure_plan = build_final_figure_plan(figures, figure_outputs)
    figure_layout = build_figure_layout_snippet(final_figure_plan)
    _write_csv(
        output_dir / "final_figure_plan.csv",
        final_figure_plan,
        FINAL_FIGURE_PLAN_FIELDS,
    )
    (output_dir / "manuscript_figure_layout.tex").write_text(figure_layout)

    summary = {
        "status": "complete",
        "output_root": str(output_root),
        "dynamic_output_dir": str(resolved_dynamic_output_dir),
        "null_output_dir": str(resolved_null_output_dir),
        "stability_output_dir": str(resolved_stability_output_dir),
        "readiness_output_dir": str(resolved_readiness_output_dir),
        "n_chen_rows": len(chen_rows),
        "n_empirical_pairing_rows": len(empirical_rows),
        "n_synthetic_tradeoff_rows": len(synthetic_rows),
        "n_dynamic_a_interpretation_rows": len(dynamic_rows),
        "n_graph_support_rows": len(graph_rows),
        "n_empirical_null_interpretation_rows": len(empirical_null_rows),
        "n_empirical_stability_interpretation_rows": len(empirical_stability_rows),
        "n_ready_final_figure_panels": sum(
            row["status"] == "ready" for row in final_figure_plan
        ),
        "n_gated_final_figure_panels": sum(
            row["status"] != "ready" for row in final_figure_plan
        ),
        "n_missing_publication_gates": sum(
            row["claim_impact"] == "blocks final publication claim" for row in gates
        ),
        "figure_outputs": figure_outputs,
        "outputs": [
            "chen_interpretation_table.csv",
            "empirical_pairing_interpretation.csv",
            "synthetic_tradeoff_interpretation.csv",
            "dynamic_a_interpretation.csv",
            "graph_support_interpretation.csv",
            "empirical_null_interpretation.csv",
            "empirical_stability_interpretation.csv",
            "evidence_gates.csv",
            "figure_manifest.csv",
            "final_figure_plan.csv",
            "manuscript_evidence_report.md",
            "manuscript_results_snippets.tex",
            "manuscript_figure_layout.tex",
            *figure_outputs,
        ],
    }
    with (output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")
    return summary


def resolve_figures_dir(figures_dir: Path | None) -> Path:
    """Resolve saved manuscript figure inputs across repo/package cwd layouts."""

    if figures_dir is not None:
        return figures_dir
    candidates = (Path("figures"), Path("../figures"))
    for candidate in candidates:
        if all((candidate / name).is_file() for name in DEFAULT_REPRESENTATIVE_FIGURES):
            return candidate
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dynamic-output-dir", type=Path, default=None)
    parser.add_argument("--null-output-dir", type=Path, default=None)
    parser.add_argument("--stability-output-dir", type=Path, default=None)
    parser.add_argument("--readiness-output-dir", type=Path, default=None)
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="directory containing saved Rise3.png/Fall3.png representative graphs",
    )
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_package(
        args.output_root,
        args.output_dir,
        make_figures=not args.no_figures,
        figures_dir=args.figures_dir,
        dynamic_output_dir=args.dynamic_output_dir,
        null_output_dir=args.null_output_dir,
        stability_output_dir=args.stability_output_dir,
        readiness_output_dir=args.readiness_output_dir,
    )
    print(
        "wrote manuscript evidence package with "
        f"{summary['n_missing_publication_gates']} missing gates to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
