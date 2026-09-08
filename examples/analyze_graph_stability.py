"""Summarize graph support, overlap, and stability from saved adjacency artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np

DEFAULT_INPUT_DIR = Path("outputs/motorneurons")
DEFAULT_OUTPUT_DIR = Path("outputs/graph_stability")
DEFAULT_LPCMCI_INPUT_DIR = Path("outputs/revision_campaign/motorneurons_lpcmci")
DEFAULT_OASIS_INPUT_DIR = Path("outputs/revision_campaign/motorneurons_oasis")

METHOD_CACHE_FILES = (
    ("cgc", "cgc_motoneurons_weighted_adjacency_matrices.pkl"),
    (
        "cgc-star",
        "cgc_star_motoneurons_weighted_adjacency_matrices.pkl",
    ),
)

PAIR_GROUP_FIELDS = (
    "dataset",
    "method",
    "case",
    "fluo_type",
    "subset",
    "recording",
    "fish",
    "trial",
    "trace",
    "binary",
)

STABILITY_GROUP_FIELDS = (
    "dataset",
    "method",
    "case",
    "fluo_type",
    "subset",
    "representation",
    "binary",
)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def _matrix(payload_or_matrix: Any) -> np.ndarray:
    if isinstance(payload_or_matrix, dict):
        values = payload_or_matrix.get(
            "weighted_adjacency", payload_or_matrix.get("adjacency")
        )
    else:
        values = payload_or_matrix
    if values is None:
        raise ValueError("graph payload must contain an adjacency matrix")
    matrix = np.asarray(values, dtype=float).copy()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency matrix must be square")
    if np.any(matrix < 0.0) or not np.isfinite(matrix).all():
        raise ValueError("adjacency weights must be finite and nonnegative")
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _generic_summary(matrix: np.ndarray) -> dict[str, float | int]:
    values = _matrix(matrix)
    off_diag: np.ndarray = ~np.eye(values.shape[0], dtype=bool)
    retained = values > 0.0
    out_strength = values.sum(axis=1)
    in_strength = values.sum(axis=0)
    opportunities = int(np.count_nonzero(off_diag))
    retained_edges = int(np.count_nonzero(retained & off_diag))
    return {
        "n_nodes": values.shape[0],
        "retained_edges": retained_edges,
        "edge_opportunities": opportunities,
        "edge_density": retained_edges / opportunities if opportunities else 0.0,
        "total_weight": float(values[off_diag].sum()),
        "mean_out_strength": float(np.mean(out_strength)),
        "mean_in_strength": float(np.mean(in_strength)),
        "max_out_strength": float(np.max(out_strength)),
        "max_in_strength": float(np.max(in_strength)),
    }


def _record(
    *,
    matrix: np.ndarray,
    dataset: str,
    method: str,
    representation: str,
    source_file: str,
    **metadata: Any,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "method": method,
        "representation": representation,
        "source_file": source_file,
        **metadata,
        **_generic_summary(matrix),
        "matrix": matrix,
    }


def _load_method_records(path: Path, method: str) -> list[dict[str, Any]]:
    """Load every representation from a method-specific c-GC cache."""

    if not path.exists():
        return []
    payload = _load_pickle(path)
    raw_records = payload.get("records")
    if raw_records is None and "record" in payload:
        raw_records = {"record": payload["record"]}
    if raw_records is None:
        return []

    records: list[dict[str, Any]] = []
    for raw in raw_records.values():
        representation = str(raw.get("representation") or "full")
        if representation == "full":
            representation = "full_trace"
        fluo_type = raw.get("fluo_type")
        records.append(
            _record(
                matrix=_matrix(raw),
                dataset="motoneurons",
                method=method,
                representation=representation,
                source_file=path.name,
                case=raw.get("case") or fluo_type,
                description=raw.get("description"),
                fluo_type=fluo_type,
                subset=raw.get("subset"),
                recording=raw.get("recording"),
                fish=raw.get("fish"),
                trial=raw.get("trial"),
                trace=raw.get("trace") or raw.get("trial"),
                binary=False,
                method_internal=raw.get("method_internal"),
                metric_semantics="directed_weighted_adjacency",
            )
        )
    return records


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _int_or_none(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(float(value))


def _resolve_artifact_path(input_dir: Path, value: Any) -> Path:
    """Resolve saved artifact paths after moving outputs between computers."""

    raw = Path(str(value))
    if raw.is_file():
        return raw
    direct = input_dir / raw
    if direct.is_file():
        return direct
    matches = list(input_dir.rglob(raw.name))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"saved graph artifact does not exist: {raw}")
    raise ValueError(f"ambiguous saved graph artifact name under {input_dir}: {raw.name}")


def _load_npz_records(
    input_dir: Path | None,
    *,
    matrix_key: str,
    method_from_row: Any,
    binary: bool,
    metric_semantics: str,
) -> list[dict[str, Any]]:
    if input_dir is None:
        return []
    records: list[dict[str, Any]] = []
    for raw in _read_csv(input_dir / "graph_summary_rows.csv"):
        artifact = _resolve_artifact_path(input_dir, raw.get("artifact_path"))
        with np.load(artifact, allow_pickle=False) as payload:
            matrix = np.asarray(payload[matrix_key], dtype=float)
        fluo_type = raw.get("fluo_type")
        records.append(
            _record(
                matrix=matrix,
                dataset="motoneurons",
                method=method_from_row(raw),
                representation=str(raw.get("representation") or "unknown"),
                source_file=str(artifact),
                case=raw.get("case") or fluo_type,
                description=raw.get("output_semantics"),
                fluo_type=fluo_type,
                subset=raw.get("subset"),
                recording=raw.get("recording"),
                fish=_int_or_none(raw.get("fish")),
                trial=_int_or_none(raw.get("trial")),
                trace=_int_or_none(raw.get("trial")),
                binary=binary,
                method_internal=raw.get("method"),
                metric_semantics=metric_semantics,
            )
        )
    return records


def load_graph_records(
    input_dir: Path,
    *,
    lpcmci_input_dir: Path | None = None,
    oasis_input_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Load all supported saved weighted-adjacency artifacts."""

    records: list[dict[str, Any]] = []
    for method, filename in METHOD_CACHE_FILES:
        records.extend(_load_method_records(input_dir / filename, method))
    records.extend(
        _load_npz_records(
            lpcmci_input_dir,
            matrix_key="lossy_lagged_skeleton",
            method_from_row=lambda row: "lpcmci",
            binary=True,
            metric_semantics="lossy_undirected_lagged_pag_skeleton",
        )
    )
    records.extend(
        _load_npz_records(
            oasis_input_dir,
            matrix_key="retained_scores",
            method_from_row=lambda row: f"oasis+{row.get('method') or 'unknown'}",
            binary=False,
            metric_semantics="directed_weighted_adjacency_after_oasis_preprocessing",
        )
    )
    return records


def _jaccard(first: np.ndarray, second: np.ndarray) -> float | None:
    if first.shape != second.shape:
        return None
    a = _matrix(first) > 0.0
    b = _matrix(second) > 0.0
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 1.0
    return int(np.count_nonzero(a & b)) / union


def _weighted_jaccard(first: np.ndarray, second: np.ndarray) -> float | None:
    if first.shape != second.shape:
        return None
    a = _matrix(first)
    b = _matrix(second)
    denominator = float(np.maximum(a, b).sum())
    if denominator == 0.0:
        return 1.0
    return float(np.minimum(a, b).sum() / denominator)


def build_pairwise_overlaps(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare graph overlap among representations within each recording."""

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[tuple(row.get(field) for field in PAIR_GROUP_FIELDS)].append(row)

    overlaps: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        if len(group) < 2:
            continue
        base = dict(zip(PAIR_GROUP_FIELDS, key, strict=True))
        ordered = sorted(group, key=lambda row: row["representation"])
        for first, second in combinations(ordered, 2):
            first_edges = int(first["retained_edges"])
            second_edges = int(second["retained_edges"])
            first_matrix = first["matrix"]
            second_matrix = second["matrix"]
            if first_matrix.shape != second_matrix.shape:
                continue
            first_binary = first_matrix > 0.0
            second_binary = second_matrix > 0.0
            intersection = int(np.count_nonzero(first_binary & second_binary))
            union = int(np.count_nonzero(first_binary | second_binary))
            overlaps.append(
                {
                    **base,
                    "source_representation": first["representation"],
                    "target_representation": second["representation"],
                    "source_edges": first_edges,
                    "target_edges": second_edges,
                    "edge_intersection": intersection,
                    "edge_union": union,
                    "jaccard": 1.0 if union == 0 else intersection / union,
                    "weighted_jaccard": _weighted_jaccard(first_matrix, second_matrix),
                    "overlap_fraction_source": (
                        None if first_edges == 0 else intersection / first_edges
                    ),
                    "overlap_fraction_target": (
                        None if second_edges == 0 else intersection / second_edges
                    ),
                }
            )
    return overlaps


def _mean(values: Iterable[float]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return None if not numeric else float(np.mean(numeric))


def _std(values: Iterable[float]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if len(numeric) < 2:
        return None
    return float(np.std(numeric, ddof=1))


def _cv(values: Iterable[float]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if len(numeric) < 2:
        return None
    mean = float(np.mean(numeric))
    if mean == 0.0:
        return None
    return float(np.std(numeric, ddof=1) / abs(mean))


def build_stability_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize support variability and shape-compatible graph stability."""

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[tuple(row.get(field) for field in STABILITY_GROUP_FIELDS)].append(row)

    summaries: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        base = dict(zip(STABILITY_GROUP_FIELDS, key, strict=True))
        pairwise_jaccards: list[float] = []
        pairwise_weighted_jaccards: list[float] = []
        by_shape: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
        for row in group:
            by_shape[row["matrix"].shape].append(row["matrix"])
        for matrices in by_shape.values():
            for first, second in combinations(matrices, 2):
                jaccard = _jaccard(first, second)
                weighted = _weighted_jaccard(first, second)
                if jaccard is not None:
                    pairwise_jaccards.append(jaccard)
                if weighted is not None:
                    pairwise_weighted_jaccards.append(weighted)

        summaries.append(
            {
                **base,
                "n_graphs": len(group),
                "n_shape_groups": len(by_shape),
                "n_shape_compatible_pairs": len(pairwise_jaccards),
                "mean_pairwise_jaccard": _mean(pairwise_jaccards),
                "mean_pairwise_weighted_jaccard": _mean(pairwise_weighted_jaccards),
                "edge_density_mean": _mean(row["edge_density"] for row in group),
                "edge_density_std": _std(row["edge_density"] for row in group),
                "retained_edges_mean": _mean(row["retained_edges"] for row in group),
                "retained_edges_cv": _cv(row["retained_edges"] for row in group),
                "total_weight_mean": _mean(row["total_weight"] for row in group),
                "total_weight_cv": _cv(row["total_weight"] for row in group),
            }
        )
    return summaries


def _csv_ready(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key != "matrix"}
        for row in rows
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--lpcmci-input-dir", type=Path, default=DEFAULT_LPCMCI_INPUT_DIR
    )
    parser.add_argument("--oasis-input-dir", type=Path, default=DEFAULT_OASIS_INPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_graph_records(
        args.input_dir,
        lpcmci_input_dir=args.lpcmci_input_dir,
        oasis_input_dir=args.oasis_input_dir,
    )
    if not records:
        raise SystemExit(f"no graph artifacts found under {args.input_dir}")
    overlaps = build_pairwise_overlaps(records)
    stability = build_stability_summaries(records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "graph_support_rows.csv", _csv_ready(records))
    _write_csv(args.output_dir / "representation_overlaps.csv", overlaps)
    _write_csv(args.output_dir / "graph_stability_summary.csv", stability)
    with (args.output_dir / "summary.json").open("w") as file:
        json.dump(
            {
                "input_dir": str(args.input_dir),
                "lpcmci_input_dir": str(args.lpcmci_input_dir),
                "oasis_input_dir": str(args.oasis_input_dir),
                "n_graph_rows": len(records),
                "n_overlap_rows": len(overlaps),
                "n_stability_rows": len(stability),
                "outputs": [
                    "graph_support_rows.csv",
                    "representation_overlaps.csv",
                    "graph_stability_summary.csv",
                ],
            },
            file,
            indent=2,
        )
        file.write("\n")
    print(f"wrote {len(stability)} graph-stability rows to {args.output_dir}")


if __name__ == "__main__":
    main()
