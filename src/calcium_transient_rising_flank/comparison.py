"""Comparison summaries for Chen-style motoneuron benchmarks."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import BilateralMetric, w_ic, w_rc


def sides_from_mid(total: int, mid: int) -> np.ndarray:
    """Return left/right labels for the manuscript's bilateral node ordering."""

    if not 0 < mid < total:
        raise ValueError("mid must divide the node count into two nonempty sides")
    return np.array(["L"] * mid + ["R"] * (total - mid))


def positions_from_mid(total: int, mid: int) -> np.ndarray:
    """Return within-side rostrocaudal indices for left/right ordered matrices."""

    if not 0 < mid < total:
        raise ValueError("mid must divide the node count into two nonempty sides")
    return np.concatenate([np.arange(mid), np.arange(total - mid)]).astype(float)


def _weighted_matrix(payload_or_matrix: Any) -> np.ndarray:
    if isinstance(payload_or_matrix, dict):
        matrix = payload_or_matrix.get(
            "weighted_adjacency", payload_or_matrix["adjacency"]
        )
    else:
        matrix = payload_or_matrix
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("adjacency matrix must be square")
    values = values.copy()
    np.fill_diagonal(values, 0.0)
    if np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("adjacency weights must be finite and nonnegative")
    return values


def _safe_value(metric: BilateralMetric | None) -> float | None:
    return None if metric is None else metric.value


def _safe_metric(
    fn, matrix: np.ndarray, *args, binary: bool
) -> BilateralMetric | None:
    try:
        return fn(matrix, *args, binary=binary)
    except ValueError:
        return None


def graph_summary(
    payload_or_matrix: Any,
    mid: int,
    *,
    binary: bool = False,
    positions: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    """Summarize one weighted directed graph with Chen-style quantities."""

    matrix = _weighted_matrix(payload_or_matrix)
    total = matrix.shape[0]
    sides = sides_from_mid(total, mid)
    node_positions = positions_from_mid(total, mid) if positions is None else positions
    if np.asarray(node_positions).shape != (total,):
        raise ValueError("positions must contain one value per node")

    values = matrix.astype(bool).astype(float) if binary else matrix
    off_diag = ~np.eye(total, dtype=bool)
    same_side = (sides[:, None] == sides[None, :]) & off_diag
    cross_side = (sides[:, None] != sides[None, :]) & off_diag
    forward = same_side & (node_positions[:, None] < node_positions[None, :])
    reverse = same_side & (node_positions[:, None] > node_positions[None, :])
    ic = _safe_metric(w_ic, matrix, sides, binary=binary)
    rc = _safe_metric(w_rc, matrix, sides, node_positions, binary=binary)
    retained_edges = int(np.count_nonzero(values[off_diag]))
    opportunities = int(np.count_nonzero(off_diag))
    return {
        "n_nodes": total,
        "mid": mid,
        "w_ic": _safe_value(ic),
        "w_rc": _safe_value(rc),
        "edge_density": retained_edges / opportunities if opportunities else None,
        "retained_edges": retained_edges,
        "edge_opportunities": opportunities,
        "total_weight": float(values[off_diag].sum()),
        "ipsilateral_weight": float(values[same_side].sum()),
        "contralateral_weight": float(values[cross_side].sum()),
        "rostrocaudal_weight": float(values[forward].sum()),
        "caudorostral_weight": float(values[reverse].sum()),
        "ipsilateral_edges": int(np.count_nonzero(values[same_side])),
        "contralateral_edges": int(np.count_nonzero(values[cross_side])),
        "rostrocaudal_edges": int(np.count_nonzero(values[forward])),
        "caudorostral_edges": int(np.count_nonzero(values[reverse])),
    }


def _iter_cache_cases(
    adjacency_cache: dict[str, Any],
) -> tuple[tuple[str, dict[str, Any]], ...]:
    if "cases" in adjacency_cache:
        return tuple(adjacency_cache["cases"].items())
    if "case" in adjacency_cache:
        case = adjacency_cache["case"]
        label = str(case.get("label", "case"))
        return ((label, case),)
    raise ValueError("cache must contain either 'cases' or 'case'")


def summarize_adjacency_cache(
    adjacency_cache: dict[str, Any],
    *,
    binary: bool = False,
) -> list[dict[str, Any]]:
    """Return one summary row per case, recording, and representation/phase."""

    rows: list[dict[str, Any]] = []
    for case_label, case in _iter_cache_cases(adjacency_cache):
        middle = case["middle"]
        for key, graph_payloads in case["graphs"].items():
            fish, trace = key
            for graph_label, payload in graph_payloads.items():
                graph_name = str(graph_label)
                if "__" in graph_name:
                    method, representation = graph_name.split("__", 1)
                else:
                    method, representation = None, graph_name
                row: dict[str, Any] = {
                    "case": case_label,
                    "description": case.get("description"),
                    "fish": fish,
                    "trace": trace,
                    "recording": f"F{fish}T{trace}",
                    "graph_label": graph_name,
                    "method": method,
                    "representation": representation,
                    "binary": binary,
                }
                row.update(graph_summary(payload, int(middle[key]), binary=binary))
                rows.append(row)
    return rows


def add_paired_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add rise-minus-fall deltas within each case and recording."""

    indexed = {
        (
            row["case"],
            row["fish"],
            row["trace"],
            row.get("method"),
            row["representation"],
        ): row
        for row in rows
    }
    enriched = [dict(row) for row in rows]
    for row in enriched:
        key = (row["case"], row["fish"], row["trace"], row.get("method"))
        rise = indexed.get((*key, "rise"))
        fall = indexed.get((*key, "fall"))
        if rise is None or fall is None:
            row["delta_w_ic_rise_minus_fall"] = None
            row["delta_w_rc_rise_minus_fall"] = None
            continue
        row["delta_w_ic_rise_minus_fall"] = (
            None
            if rise["w_ic"] is None or fall["w_ic"] is None
            else rise["w_ic"] - fall["w_ic"]
        )
        row["delta_w_rc_rise_minus_fall"] = (
            None
            if rise["w_rc"] is None or fall["w_rc"] is None
            else rise["w_rc"] - fall["w_rc"]
        )
    return enriched


def write_summary_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write summary rows to CSV for manuscript tables or notebook display."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output
