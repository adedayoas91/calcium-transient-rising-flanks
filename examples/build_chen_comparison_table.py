"""Build Chen-style comparison tables from saved weighted adjacency artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from calcium_transient_rising_flank import (
    add_paired_deltas,
    graph_summary,
    summarize_adjacency_cache,
)

DEFAULT_INPUT_DIR = Path("outputs/motorneurons")
DEFAULT_OUTPUT_DIR = Path("outputs/chen_comparison")
DEFAULT_CHEN_MATRIX_MANIFEST = Path("outputs/chen_direct_matrices/manifest.csv")
CHEN_W_IC = 1.0

MOTONEURON_SUMMARY_FILES = (
    ("cgc", "cgc_motoneurons_summary_rows.pkl"),
    ("cgc-star", "cgc_star_motoneurons_summary_rows.pkl"),
)

NUMERIC_SUMMARY_FIELDS = (
    "w_ic",
    "w_rc",
    "edge_density",
    "retained_edges",
    "edge_opportunities",
    "total_weight",
    "ipsilateral_weight",
    "contralateral_weight",
    "rostrocaudal_weight",
    "caudorostral_weight",
    "ipsilateral_edges",
    "contralateral_edges",
    "rostrocaudal_edges",
    "caudorostral_edges",
    "delta_w_ic_rise_minus_fall",
    "delta_w_rc_rise_minus_fall",
)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def _empty_to_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def _int_or_none(value: Any) -> int | None:
    value = _empty_to_none(value)
    return None if value is None else int(float(value))


def _bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _method_from_manifest(row: dict[str, Any], path: Path) -> str:
    method = str(row.get("method") or "").strip().lower()
    if method in {"bvgc", "chen_bvgc", "chen-bvgc"}:
        return "chen_bvgc"
    if method in {"mvgc", "chen_mvgc", "chen-mvgc"}:
        return "chen_mvgc"
    name = path.stem.lower()
    if "bvgc" in name:
        return "chen_bvgc"
    if "mvgc" in name:
        return "chen_mvgc"
    raise ValueError(
        "Chen direct matrix manifest rows must declare method BVGC or MVGC"
    )


def _load_matrix(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path), dtype=float)
    if suffix == ".npz":
        payload = np.load(path)
        key = "matrix" if "matrix" in payload.files else payload.files[0]
        return np.asarray(payload[key], dtype=float)
    if suffix in {".csv", ".txt"}:
        delimiter = "," if suffix == ".csv" else None
        return np.loadtxt(path, delimiter=delimiter, dtype=float)
    if suffix == ".json":
        with path.open() as file:
            return np.asarray(json.load(file), dtype=float)
    raise ValueError(
        f"unsupported matrix format for {path}; use .csv, .txt, .json, .npy, or .npz"
    )


def _published_chen_row() -> dict[str, Any]:
    return {
        "dataset": "motoneurons",
        "recording": "published aggregate",
        "fish": None,
        "trial": None,
        "case": "published",
        "description": "Chen et al. improved GC published ipsilateral result",
        "fluo_type": None,
        "method": "chen_improved_gc",
        "method_internal": "published",
        "representation": "published",
        "graph_label": "published",
        "n_nodes": None,
        "mid": None,
        "w_ic": CHEN_W_IC,
        "w_rc": None,
        "edge_density": None,
        "retained_edges": None,
        "edge_opportunities": None,
        "total_weight": None,
        "ipsilateral_weight": None,
        "contralateral_weight": None,
        "rostrocaudal_weight": None,
        "caudorostral_weight": None,
        "ipsilateral_edges": None,
        "contralateral_edges": None,
        "rostrocaudal_edges": None,
        "caudorostral_edges": None,
        "delta_w_ic_rise_minus_fall": None,
        "delta_w_rc_rise_minus_fall": None,
        "source_file": "literature",
        "source_note": "Only W_IC=1.00 is encoded here; W_RC was not inferred.",
    }


def _rising_flank_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    cache = _load_pickle(path)
    rows = add_paired_deltas(summarize_adjacency_cache(cache))
    normalized = []
    for row in rows:
        values = dict(row)
        values["dataset"] = "motoneurons"
        values["method"] = values.get("method") or "rising_flank_cgc"
        values["method_internal"] = values["method"]
        values["fluo_type"] = values["case"]
        values["source_file"] = path.name
        values["source_note"] = "case-specific rise/fall c-GC weighted cache"
        normalized.append(values)
    return normalized


def _full_trace_rows(path: Path, method: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = _load_pickle(path)
    normalized = []
    for row in rows:
        values = dict(row)
        values["case"] = values.get("case") or values.get("fluo_type")
        values["description"] = values.get("description") or (
            "full-trace baseline from saved c-GC weighted cache"
        )
        values["method"] = values.get("method") or method
        values["representation"] = values.get("representation") or "full_trace"
        values["graph_label"] = values.get("graph_label") or values["representation"]
        values["delta_w_ic_rise_minus_fall"] = None
        values["delta_w_rc_rise_minus_fall"] = None
        values["source_file"] = path.name
        values["source_note"] = "saved full-trace motoneuron summary row"
        normalized.append(values)
    return normalized


def _chen_direct_matrix_rows(manifest_path: Path | None) -> list[dict[str, Any]]:
    if manifest_path is None or not manifest_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with manifest_path.open(newline="") as file:
        for manifest_row in csv.DictReader(file):
            matrix_path = Path(str(manifest_row.get("path") or ""))
            if not matrix_path.is_absolute():
                matrix_path = manifest_path.parent / matrix_path
            matrix = _load_matrix(matrix_path)
            mid = _int_or_none(manifest_row.get("mid"))
            if mid is None:
                raise ValueError("Chen direct matrix manifest rows must declare mid")
            method = _method_from_manifest(manifest_row, matrix_path)
            values: dict[str, Any] = {
                "dataset": manifest_row.get("dataset") or "motoneurons",
                "recording": manifest_row.get("recording")
                or matrix_path.stem,
                "fish": _int_or_none(manifest_row.get("fish")),
                "trial": _int_or_none(
                    manifest_row.get("trial") or manifest_row.get("trace")
                ),
                "case": manifest_row.get("case") or "published_direct",
                "description": manifest_row.get("description")
                or "direct Chen-style BVGC/MVGC matrix summary",
                "fluo_type": _empty_to_none(manifest_row.get("fluo_type")),
                "method": method,
                "method_internal": manifest_row.get("method") or method,
                "representation": manifest_row.get("representation")
                or "published_direct",
                "graph_label": manifest_row.get("graph_label")
                or f"{method}__published_direct",
                "binary": _bool_value(manifest_row.get("binary")),
                "source_file": str(matrix_path),
                "source_note": manifest_row.get("source_note")
                or "direct supplied Chen-style BVGC/MVGC matrix",
                "delta_w_ic_rise_minus_fall": None,
                "delta_w_rc_rise_minus_fall": None,
            }
            values.update(
                graph_summary(
                    matrix,
                    mid,
                    binary=bool(values["binary"]),
                )
            )
            rows.append(values)
    return rows


def build_comparison_rows(
    input_dir: Path,
    *,
    include_published_chen: bool = True,
    chen_matrix_manifest: Path | None = DEFAULT_CHEN_MATRIX_MANIFEST,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if include_published_chen:
        rows.append(_published_chen_row())
    rows.extend(_chen_direct_matrix_rows(chen_matrix_manifest))
    rows.extend(
        _rising_flank_rows(input_dir / "rising_flanks_weighted_adjacency_matrices.pkl")
    )
    for method, filename in MOTONEURON_SUMMARY_FILES:
        rows.extend(_full_trace_rows(input_dir / filename, method))
    return rows


def _mean(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return None if not numeric else float(np.mean(numeric))


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("dataset"),
            row.get("method"),
            row.get("case"),
            row.get("fluo_type"),
            row.get("representation"),
        )
        grouped[key].append(row)

    aggregates = []
    def sort_key(item: tuple[tuple[Any, ...], list[dict[str, Any]]]) -> tuple[str, ...]:
        return tuple("" if value is None else str(value) for value in item[0])

    for key, group in sorted(grouped.items(), key=sort_key):
        dataset, method, case, fluo_type, representation = key
        aggregate: dict[str, Any] = {
            "dataset": dataset,
            "method": method,
            "case": case,
            "fluo_type": fluo_type,
            "representation": representation,
            "n_rows": len(group),
            "recordings": ";".join(
                sorted(str(row.get("recording")) for row in group if row.get("recording"))
            ),
        }
        for field in NUMERIC_SUMMARY_FIELDS:
            aggregate[f"{field}_mean"] = _mean([row.get(field) for row in group])
        aggregates.append(aggregate)
    return aggregates


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
        "--omit-published-chen",
        action="store_true",
        help="omit the literature W_IC=1.00 reference row",
    )
    parser.add_argument(
        "--chen-matrix-manifest",
        type=Path,
        default=DEFAULT_CHEN_MATRIX_MANIFEST,
        help=(
            "optional CSV manifest of supplied Chen-style BVGC/MVGC matrices "
            "to summarize with the same graph metrics"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_comparison_rows(
        args.input_dir,
        include_published_chen=not args.omit_published_chen,
        chen_matrix_manifest=args.chen_matrix_manifest,
    )
    if not rows:
        raise SystemExit(f"no comparison rows found under {args.input_dir}")
    aggregates = aggregate_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "chen_comparison_rows.csv", rows)
    _write_csv(args.output_dir / "chen_comparison_summary.csv", aggregates)
    with (args.output_dir / "summary.json").open("w") as file:
        json.dump(
            {
                "input_dir": str(args.input_dir),
                "n_rows": len(rows),
                "n_summary_rows": len(aggregates),
                "includes_published_chen": not args.omit_published_chen,
                "chen_matrix_manifest": str(args.chen_matrix_manifest),
                "n_direct_chen_matrix_rows": sum(
                    row.get("method") in {"chen_bvgc", "chen_mvgc"}
                    for row in rows
                ),
                "outputs": [
                    "chen_comparison_rows.csv",
                    "chen_comparison_summary.csv",
                ],
            },
            file,
            indent=2,
        )
        file.write("\n")
    print(f"wrote {len(rows)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
