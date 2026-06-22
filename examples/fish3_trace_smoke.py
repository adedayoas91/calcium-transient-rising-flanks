"""Verify that the supplied Fish 3 trace passes through the implemented core."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from calcium_transient_rising_flank import (
    CausalGranger,
    build_representations,
    characterize_transients,
    selected_frame_indices,
)


def main() -> None:
    trace_path = Path(__file__).parents[1] / "data" / "fish3_trace2_dff.npy"
    traces = np.load(trace_path, allow_pickle=False)
    summary = characterize_transients(traces)
    representations = build_representations(traces, gamma=summary.gamma)
    estimator = CausalGranger(max_lag=1, n_surrogates=0)
    print(f"loaded {trace_path.name}: shape={traces.shape}")
    print(f"median estimated gamma={np.median(summary.gamma):.4f}")
    for label in ("rise", "fall", "fall_residual"):
        graph = estimator.fit(
            traces,
            event_indices=selected_frame_indices(representations.as_dict()[label]),
        )
        print(f"{label}: score matrix shape={graph.scores.shape}")
    print("No biological interpretation is made without declared analysis settings.")


if __name__ == "__main__":
    main()
