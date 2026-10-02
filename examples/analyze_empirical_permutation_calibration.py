"""Audit finite-surrogate calibration in saved empirical BH graph artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from calcium_transient_rising_flank.estimators import (
    benjamini_hochberg,
    finite_sample_permutation_p_values,
)


DEFAULT_INPUT_DIR = Path(
    "outputs/validation_campaign/empirical_fdr/bh/observed_graph_artifacts"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/method_validation/permutation_calibration"
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero(left | right)
    if union == 0:
        return 1.0
    return float(np.count_nonzero(left & right) / union)


def audit_artifact(
    path: Path,
    *,
    n_surrogates: int,
    alpha: float,
) -> dict[str, Any]:
    """Return pre/post-correction BH diagnostics for one saved graph."""

    with np.load(path, allow_pickle=False) as archive:
        required = {"adjacency", "p_values", "metadata_json"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"{path} is missing: {', '.join(missing)}")
        saved = np.asarray(archive["adjacency"], dtype=bool)
        p_values = np.asarray(archive["p_values"], dtype=float)
        metadata = json.loads(str(archive["metadata_json"]))

    if p_values.ndim != 2 or p_values.shape[0] != p_values.shape[1]:
        raise ValueError(f"{path} has a non-square p-value matrix")
    if saved.shape != p_values.shape:
        raise ValueError(f"{path} adjacency and p-values have different shapes")
    if np.any(~np.isfinite(p_values)) or np.any((p_values < 0) | (p_values > 1)):
        raise ValueError(f"{path} contains invalid p-values")

    eligible: np.ndarray = ~np.eye(p_values.shape[0], dtype=bool)
    raw_bh = benjamini_hochberg(p_values, alpha)
    corrected = finite_sample_permutation_p_values(p_values, n_surrogates)
    corrected_bh = benjamini_hochberg(corrected, alpha)
    saved_edges = int(np.count_nonzero(saved))
    corrected_edges = int(np.count_nonzero(corrected_bh))
    removed = saved & ~corrected_bh
    added = corrected_bh & ~saved

    return {
        "artifact": str(path),
        "case": metadata.get("case"),
        "recording": metadata.get("recording"),
        "fish": metadata.get("fish"),
        "trial": metadata.get("trial"),
        "method": metadata.get("method"),
        "representation": metadata.get("representation"),
        "n_rois": int(p_values.shape[0]),
        "n_tests": int(np.count_nonzero(eligible)),
        "n_surrogates": n_surrogates,
        "alpha": alpha,
        "zero_p_values": int(np.count_nonzero((p_values == 0.0) & eligible)),
        "minimum_raw_p": float(np.min(p_values[eligible])),
        "minimum_corrected_p": float(np.min(corrected[eligible])),
        "saved_edges": saved_edges,
        "raw_bh_edges": int(np.count_nonzero(raw_bh)),
        "corrected_edges": corrected_edges,
        "removed_edges": int(np.count_nonzero(removed)),
        "added_edges": int(np.count_nonzero(added)),
        "saved_edge_density": float(saved_edges / np.count_nonzero(eligible)),
        "corrected_edge_density": float(
            corrected_edges / np.count_nonzero(eligible)
        ),
        "saved_matches_recomputed_raw_bh": bool(np.array_equal(saved, raw_bh)),
        "saved_corrected_jaccard": _jaccard(saved, corrected_bh),
    }


def resolution_rows(
    n_rois_values: Iterable[int],
    *,
    alpha: float,
    n_surrogates: int,
) -> list[dict[str, Any]]:
    """Return first-rank BH resolution bounds for the requested graph sizes."""

    rows: list[dict[str, Any]] = []
    for n_rois in sorted(set(int(value) for value in n_rois_values)):
        n_tests = n_rois * (n_rois - 1)
        first_rank_cutoff = alpha / n_tests
        minimum_surrogates = max(1, math.ceil(n_tests / alpha) - 1)
        rows.append(
            {
                "n_rois": n_rois,
                "n_tests": n_tests,
                "alpha": alpha,
                "first_rank_bh_cutoff": first_rank_cutoff,
                "current_n_surrogates": n_surrogates,
                "current_minimum_add_one_p": 1.0 / (n_surrogates + 1.0),
                "minimum_surrogates_for_first_rank_resolution": minimum_surrogates,
                "current_count_resolves_first_rank": bool(
                    n_surrogates >= minimum_surrogates
                ),
                "interpretation": (
                    "analytical probability-resolution bound; not an empirical "
                    "stabilization estimate"
                ),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one artifact is required")
    return {
        "n_graphs": len(rows),
        "all_saved_match_recomputed_raw_bh": all(
            bool(row["saved_matches_recomputed_raw_bh"]) for row in rows
        ),
        "saved_edges_total": int(sum(int(row["saved_edges"]) for row in rows)),
        "corrected_edges_total": int(
            sum(int(row["corrected_edges"]) for row in rows)
        ),
        "removed_edges_total": int(sum(int(row["removed_edges"]) for row in rows)),
        "added_edges_total": int(sum(int(row["added_edges"]) for row in rows)),
        "saved_edges_mean": float(np.mean([row["saved_edges"] for row in rows])),
        "corrected_edges_mean": float(
            np.mean([row["corrected_edges"] for row in rows])
        ),
        "saved_corrected_jaccard_mean": float(
            np.mean([row["saved_corrected_jaccard"] for row in rows])
        ),
        "graphs_with_changed_edges": int(
            sum(int(row["removed_edges"]) + int(row["added_edges"]) > 0 for row in rows)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-surrogates", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_surrogates < 1:
        raise SystemExit("--n-surrogates must be positive")
    if not 0.0 < args.alpha < 1.0:
        raise SystemExit("--alpha must lie in (0, 1)")
    paths = sorted(args.input_dir.glob("*.npz"))
    if not paths:
        raise SystemExit(f"no NPZ artifacts found in {args.input_dir}")

    rows = [
        audit_artifact(
            path,
            n_surrogates=args.n_surrogates,
            alpha=args.alpha,
        )
        for path in paths
    ]
    bounds = resolution_rows(
        [int(row["n_rois"]) for row in rows],
        alpha=args.alpha,
        n_surrogates=args.n_surrogates,
    )
    summary = {
        "status": "complete",
        "input_dir": str(args.input_dir),
        "n_surrogates": args.n_surrogates,
        "alpha": args.alpha,
        "interpretation": (
            "The add-one audit is a deterministic reanalysis of saved p-value "
            "matrices. Resolution bounds do not establish graph stabilization."
        ),
        **summarize(rows),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "empirical_add_one_sensitivity.csv", rows)
    _write_csv(args.output_dir / "permutation_resolution.csv", bounds)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
