"""Run one deterministic rising-flank analysis with known directed truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from calcium_transient_rising_flank import (
    CausalisedGC,
    build_representations,
    edge_recovery,
    selected_frame_indices,
)
from calcium_transient_rising_flank.preprocessing import smooth_traces
from calcium_transient_rising_flank.validation import simulate_calcium_dataset


def run_analysis(
    *,
    seed: int = 21,
    n_steps: int = 600,
    smoothing_window: int = 3,
    tolerance: float = 0.02,
    n_surrogates: int = 199,
) -> dict[str, Any]:
    """Return configuration and recovery metrics for a small locked example."""

    truth = np.array(
        [
            [False, True, False, False],
            [False, False, False, False],
            [False, False, False, True],
            [False, False, False, False],
        ]
    )
    dataset = simulate_calcium_dataset(
        truth,
        n_steps=n_steps,
        gamma=0.8,
        noise_std=0.03,
        shared_noise_std=0.02,
        random_state=seed,
    )
    traces = smooth_traces(dataset.fluorescence, smoothing_window)
    rise = build_representations(traces, tolerance=tolerance, gamma=0.8).rise
    selected = selected_frame_indices(rise)
    graph = CausalisedGC(
        max_lag=1,
        n_surrogates=n_surrogates,
        alpha=0.05,
        random_state=seed,
        fdr=True,
        event_mode="physical",
    ).fit(traces, event_indices=selected)
    recovery = edge_recovery(truth, graph.adjacency)
    off_diagonal = ~np.eye(truth.shape[0], dtype=bool)

    return {
        "config": {
            "seed": seed,
            "n_steps": n_steps,
            "gamma": 0.8,
            "noise_std": 0.03,
            "shared_noise_std": 0.02,
            "smoothing_window": smoothing_window,
            "tolerance": tolerance,
            "max_lag": 1,
            "event_mode": "physical",
            "n_surrogates": n_surrogates,
            "alpha": 0.05,
            "fdr": True,
        },
        "selected_frames_per_roi": [int(values.size) for values in selected],
        "truth_edges": int(np.count_nonzero(truth)),
        "retained_edges": int(np.count_nonzero(graph.adjacency)),
        "edge_density": float(np.mean(graph.adjacency[off_diagonal])),
        "recovery": {
            "precision": recovery.precision,
            "recall": recovery.recall,
            "false_positive_rate": recovery.false_positive_rate,
            "orientation_accuracy": recovery.orientation_accuracy,
            "true_positives": recovery.true_positives,
            "false_positives": recovery.false_positives,
            "false_negatives": recovery.false_negatives,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--n-steps", type=int, default=600)
    parser.add_argument("--smoothing-window", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--n-surrogates", type=int, default=199)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_analysis(
        seed=args.seed,
        n_steps=args.n_steps,
        smoothing_window=args.smoothing_window,
        tolerance=args.tolerance,
        n_surrogates=args.n_surrogates,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
