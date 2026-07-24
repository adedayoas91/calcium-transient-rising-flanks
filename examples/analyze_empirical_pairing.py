"""Compute paired rise-minus-fall empirical tests from Chen-comparison rows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from itertools import product
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np

DEFAULT_INPUT = Path("outputs/chen_comparison/chen_comparison_rows.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/empirical_stats")

METRIC_FIELDS = (
    "w_ic",
    "w_rc",
    "edge_density",
    "retained_edges",
    "total_weight",
    "ipsilateral_weight",
    "contralateral_weight",
    "rostrocaudal_weight",
    "caudorostral_weight",
)

PAIR_ID_FIELDS = (
    "dataset",
    "method",
    "method_internal",
    "case",
    "fluo_type",
    "binary",
    "recording",
    "fish",
    "trial",
    "trace",
)

TEST_GROUP_FIELDS = (
    "dataset",
    "method",
    "method_internal",
    "case",
    "fluo_type",
    "binary",
    "metric",
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


def load_comparison_rows(path: Path) -> list[dict[str, Any]]:
    """Read Chen-comparison rows and normalize configured numeric metrics."""

    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        rows: list[dict[str, Any]] = []
        for row in reader:
            normalized: dict[str, Any] = dict(row)
            for field in METRIC_FIELDS:
                normalized[field] = _float_or_none(row.get(field))
            rows.append(normalized)
    return rows


def _pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in PAIR_ID_FIELDS)


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.mean(array))


def build_paired_deltas(
    rows: list[dict[str, Any]],
    *,
    metric_fields: tuple[str, ...] = METRIC_FIELDS,
) -> list[dict[str, Any]]:
    """Compute rise-minus-fall deltas for every paired recording and metric."""

    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        representation = row.get("representation")
        if representation not in {"rise", "fall"}:
            continue
        grouped[_pair_key(row)][representation] = row

    deltas: list[dict[str, Any]] = []
    for key, pair in sorted(grouped.items()):
        rise = pair.get("rise")
        fall = pair.get("fall")
        if rise is None or fall is None:
            continue
        pair_values = dict(zip(PAIR_ID_FIELDS, key, strict=True))
        for metric in metric_fields:
            rise_value = rise.get(metric)
            fall_value = fall.get(metric)
            if rise_value is None or fall_value is None:
                continue
            deltas.append(
                {
                    **pair_values,
                    "description": rise.get("description") or fall.get("description"),
                    "metric": metric,
                    "rise_value": rise_value,
                    "fall_value": fall_value,
                    "delta_rise_minus_fall": float(rise_value) - float(fall_value),
                }
            )
    return deltas


def _exact_signflip_p_value(deltas: list[float], *, tolerance: float = 1e-12) -> float:
    observed = abs(_mean(deltas))
    total = 0
    at_least_observed = 0
    for signs in product((-1.0, 1.0), repeat=len(deltas)):
        total += 1
        statistic = abs(_mean(sign * delta for sign, delta in zip(signs, deltas)))
        if statistic + tolerance >= observed:
            at_least_observed += 1
    return at_least_observed / total


def _monte_carlo_signflip_p_value(
    deltas: list[float],
    *,
    n_resamples: int,
    seed: int,
    tolerance: float = 1e-12,
) -> float:
    observed = abs(_mean(deltas))
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n_resamples, len(deltas)))
    statistics = np.abs(np.mean(signs * np.asarray(deltas, dtype=float), axis=1))
    count = np.count_nonzero(statistics + tolerance >= observed)
    return float((count + 1) / (n_resamples + 1))


def summarize_paired_tests(
    deltas: list[dict[str, Any]],
    *,
    max_exact_pairs: int = 20,
    n_resamples: int = 10000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Aggregate paired deltas and run a two-sided sign-flip test per group."""

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in deltas:
        grouped[tuple(row.get(field) for field in TEST_GROUP_FIELDS)].append(row)

    summaries: list[dict[str, Any]] = []
    for group_index, (key, group) in enumerate(sorted(grouped.items())):
        values = [float(row["delta_rise_minus_fall"]) for row in group]
        n_pairs = len(values)
        if n_pairs == 0:
            continue
        if n_pairs <= max_exact_pairs:
            p_value = _exact_signflip_p_value(values)
            test_mode = "exact"
        else:
            p_value = _monte_carlo_signflip_p_value(
                values,
                n_resamples=n_resamples,
                seed=seed + group_index,
            )
            test_mode = "monte_carlo"
        summaries.append(
            {
                **dict(zip(TEST_GROUP_FIELDS, key, strict=True)),
                "n_pairs": n_pairs,
                "mean_delta": float(np.mean(values)),
                "median_delta": float(median(values)),
                "min_delta": float(np.min(values)),
                "max_delta": float(np.max(values)),
                "n_positive": sum(value > 0.0 for value in values),
                "n_negative": sum(value < 0.0 for value in values),
                "n_zero": sum(value == 0.0 for value in values),
                "p_two_sided_signflip": p_value,
                "test_mode": test_mode,
            }
        )
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-exact-pairs", type=int, default=20)
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_comparison_rows(args.input)
    deltas = build_paired_deltas(rows)
    tests = summarize_paired_tests(
        deltas,
        max_exact_pairs=args.max_exact_pairs,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "rise_fall_recording_deltas.csv", deltas)
    _write_csv(args.output_dir / "paired_signflip_tests.csv", tests)
    with (args.output_dir / "summary.json").open("w") as file:
        json.dump(
            {
                "input": str(args.input),
                "n_input_rows": len(rows),
                "n_delta_rows": len(deltas),
                "n_test_rows": len(tests),
                "metric_fields": list(METRIC_FIELDS),
                "outputs": [
                    "rise_fall_recording_deltas.csv",
                    "paired_signflip_tests.csv",
                ],
            },
            file,
            indent=2,
        )
        file.write("\n")
    print(f"wrote {len(tests)} paired test rows to {args.output_dir}")


if __name__ == "__main__":
    main()
