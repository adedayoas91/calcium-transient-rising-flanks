"""Summarize recall-vs-false-positive tradeoffs from validation CSV outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_INPUT_DIR = Path("outputs/validation_results")
DEFAULT_OUTPUT_DIR = Path("outputs/validation_tradeoffs")
GRID_FILENAMES = {"grid_runs.csv", "dynamic_grid_runs.csv"}

NUMERIC_FIELDS = (
    "precision",
    "recall",
    "false_positive_rate",
    "f1",
    "orientation_accuracy",
    "w_ic",
    "edge_density",
)

GROUP_FIELDS = (
    "method",
    "event_mode",
    "condition",
    "simulator_mode",
    "split",
)

COMPARATOR_ORDER = ("fall", "full", "deconvolved", "fall_residual")


def _empty_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return None if stripped == "" else stripped


def _float_or_none(value: str | None) -> float | None:
    text = _empty_to_none(value)
    if text is None:
        return None
    return float(text)


def _method_from_path(path: Path, input_dir: Path) -> str:
    try:
        relative = path.relative_to(input_dir)
    except ValueError:
        return path.parent.name
    parts = relative.parts
    return parts[0] if len(parts) > 1 else path.parent.name


def _normalized_row(row: dict[str, str], path: Path, input_dir: Path) -> dict[str, Any]:
    method = row.get("method") or _method_from_path(path, input_dir)
    normalized = {
        "source_file": str(path),
        "method": method,
        "event_mode": _empty_to_none(row.get("event_mode")) or "compressed",
        "condition": _empty_to_none(row.get("condition")) or "unspecified",
        "simulator_mode": _empty_to_none(row.get("simulator_mode")) or "static",
        "split": _empty_to_none(row.get("split")) or "all",
        "seed": _empty_to_none(row.get("seed")),
        "representation": _empty_to_none(row.get("representation")),
        "precision": _float_or_none(row.get("precision")),
        "recall": _float_or_none(row.get("recall")),
        "false_positive_rate": _float_or_none(
            row.get("false_positive_rate", row.get("fpr"))
        ),
        "f1": _float_or_none(row.get("f1")),
        "orientation_accuracy": _float_or_none(
            row.get("orientation_accuracy", row.get("orientation"))
        ),
        "w_ic": _float_or_none(row.get("W_IC", row.get("w_ic"))),
        "edge_density": _float_or_none(row.get("edge_density")),
    }
    if normalized["representation"] is None:
        raise ValueError(f"missing representation column in {path}")
    return normalized


def load_tradeoff_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(input_dir.rglob("*.csv")):
        if path.name not in GRID_FILENAMES:
            continue
        with path.open(newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                rows.append(_normalized_row(row, path, input_dir))
    return rows


def _mean(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return None if not numeric else float(np.mean(numeric))


def summarize_by_representation(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in (*GROUP_FIELDS, "representation"))
        grouped[key].append(row)

    summaries = []
    for key, group in sorted(grouped.items()):
        values = dict(zip((*GROUP_FIELDS, "representation"), key, strict=True))
        values["n"] = len(group)
        for field in NUMERIC_FIELDS:
            values[f"{field}_mean"] = _mean([row.get(field) for row in group])
        summaries.append(values)
    return summaries


def _indexed_summaries(
    summaries: list[dict[str, Any]],
) -> dict[tuple[Any, ...], dict[str, dict[str, Any]]]:
    indexed: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summaries:
        key = tuple(row[field] for field in GROUP_FIELDS)
        indexed[key][row["representation"]] = row
    return indexed


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def build_tradeoff_contrasts(
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    contrasts: list[dict[str, Any]] = []
    for key, reps in sorted(_indexed_summaries(summaries).items()):
        rise = reps.get("rise")
        if rise is None:
            continue
        base = dict(zip(GROUP_FIELDS, key, strict=True))
        for comparator in COMPARATOR_ORDER:
            other = reps.get(comparator)
            if other is None:
                continue
            delta_recall = None
            delta_fpr = None
            if rise["recall_mean"] is not None and other["recall_mean"] is not None:
                delta_recall = rise["recall_mean"] - other["recall_mean"]
            if (
                rise["false_positive_rate_mean"] is not None
                and other["false_positive_rate_mean"] is not None
            ):
                delta_fpr = (
                    rise["false_positive_rate_mean"]
                    - other["false_positive_rate_mean"]
                )
            contrast = {
                **base,
                "comparator": comparator,
                "rise_recall_mean": rise["recall_mean"],
                "comparator_recall_mean": other["recall_mean"],
                "delta_recall": delta_recall,
                "rise_false_positive_rate_mean": rise["false_positive_rate_mean"],
                "comparator_false_positive_rate_mean": other[
                    "false_positive_rate_mean"
                ],
                "delta_false_positive_rate": delta_fpr,
                "recall_gain_per_fpr_increase": _ratio(delta_recall, delta_fpr),
                "rise_precision_mean": rise["precision_mean"],
                "comparator_precision_mean": other["precision_mean"],
                "delta_precision": None
                if rise["precision_mean"] is None or other["precision_mean"] is None
                else rise["precision_mean"] - other["precision_mean"],
                "rise_f1_mean": rise["f1_mean"],
                "comparator_f1_mean": other["f1_mean"],
                "delta_f1": None
                if rise["f1_mean"] is None or other["f1_mean"] is None
                else rise["f1_mean"] - other["f1_mean"],
                "rise_orientation_accuracy_mean": rise["orientation_accuracy_mean"],
                "comparator_orientation_accuracy_mean": other[
                    "orientation_accuracy_mean"
                ],
                "delta_orientation_accuracy": None
                if (
                    rise["orientation_accuracy_mean"] is None
                    or other["orientation_accuracy_mean"] is None
                )
                else (
                    rise["orientation_accuracy_mean"]
                    - other["orientation_accuracy_mean"]
                ),
            }
            contrasts.append(contrast)
    return contrasts


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_tradeoff_rows(args.input_dir)
    if not rows:
        raise SystemExit(f"no grid CSV files found under {args.input_dir}")
    summaries = summarize_by_representation(rows)
    contrasts = build_tradeoff_contrasts(summaries)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "representation_tradeoff_summary.csv", summaries)
    _write_csv(args.output_dir / "rise_tradeoff_contrasts.csv", contrasts)
    with (args.output_dir / "summary.json").open("w") as file:
        json.dump(
            {
                "input_dir": str(args.input_dir),
                "n_rows": len(rows),
                "n_summary_rows": len(summaries),
                "n_contrast_rows": len(contrasts),
                "grid_filenames": sorted(GRID_FILENAMES),
                "outputs": [
                    "representation_tradeoff_summary.csv",
                    "rise_tradeoff_contrasts.csv",
                ],
            },
            file,
            indent=2,
        )
        file.write("\n")
    print(f"wrote {len(contrasts)} contrast rows to {args.output_dir}")


if __name__ == "__main__":
    main()
