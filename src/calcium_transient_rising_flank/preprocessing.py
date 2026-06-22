"""Declared preprocessing operations for calcium-fluorescence traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ScenarioData:
    """A trace matrix and retained-neuron mapping for one analysis scenario."""

    name: str
    description: str
    traces: np.ndarray
    node_indices: np.ndarray


def validate_traces(traces: np.ndarray) -> np.ndarray:
    """Return a finite two-dimensional float trace matrix.

    Rows are ROIs and columns are time samples.
    """

    values = np.asarray(traces, dtype=float)
    if values.ndim != 2:
        raise ValueError("traces must have shape (n_rois, n_timepoints)")
    if values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("traces must contain at least one ROI and two samples")
    if not np.isfinite(values).all():
        raise ValueError("traces must contain only finite values")
    return values.copy()


def normalize_dff(
    fluorescence: np.ndarray,
    baseline: np.ndarray | None = None,
    percentile: float = 10.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute baseline-normalized fluorescence `(F - F0) / F0`.

    When no baseline is supplied, each ROI uses its declared percentile
    baseline. Baseline selection should be reported with empirical results.
    """

    values = validate_traces(fluorescence)
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must lie in [0, 100]")
    if baseline is None:
        base = np.percentile(values, percentile, axis=1)
    else:
        base = np.asarray(baseline, dtype=float)
        if base.shape != (values.shape[0],):
            raise ValueError("baseline must have one value per ROI")
    denom = np.where(np.abs(base) < eps, eps, np.abs(base))
    return (values - base[:, None]) / denom[:, None]


def smooth_traces(traces: np.ndarray, window: int) -> np.ndarray:
    """Smooth each ROI with an edge-padded centered moving average."""

    values = validate_traces(traces)
    if window < 1:
        raise ValueError("window must be positive")
    if window == 1:
        return values
    if window > values.shape[1]:
        raise ValueError("window cannot exceed the recording length")
    left = window // 2
    right = window - left - 1
    padded = np.pad(values, ((0, 0), (left, right)), mode="edge")
    kernel = np.full(window, 1.0 / window)
    return np.stack([np.convolve(row, kernel, mode="valid") for row in padded])


def remove_bad_neurons(
    traces: np.ndarray, bad_neurons: Iterable[int] = ()
) -> tuple[np.ndarray, np.ndarray]:
    """Remove externally labelled poor-quality ROIs.

    This function intentionally does not infer exclusions from a preferred
    network result.
    """

    values = validate_traces(traces)
    excluded = np.asarray(sorted(set(int(i) for i in bad_neurons)), dtype=int)
    if excluded.size and (
        np.any(excluded < 0) or np.any(excluded >= values.shape[0])
    ):
        raise ValueError("bad_neurons includes an out-of-range ROI index")
    keep = np.ones(values.shape[0], dtype=bool)
    keep[excluded] = False
    if not keep.any():
        raise ValueError("bad_neurons cannot exclude every ROI")
    indices = np.flatnonzero(keep)
    return values[keep], indices


def correct_artifact_frames(
    traces: np.ndarray, artifact_frames: Iterable[int] = ()
) -> np.ndarray:
    """Interpolate declared artifact frames without selecting on graph output."""

    values = validate_traces(traces)
    artifacts = np.asarray(sorted(set(int(t) for t in artifact_frames)), dtype=int)
    if artifacts.size == 0:
        return values
    if np.any(artifacts < 0) or np.any(artifacts >= values.shape[1]):
        raise ValueError("artifact_frames includes an out-of-range sample")
    valid = np.ones(values.shape[1], dtype=bool)
    valid[artifacts] = False
    if valid.sum() < 2:
        raise ValueError("at least two non-artifact samples are required")
    time = np.arange(values.shape[1])
    result = values.copy()
    for roi, row in enumerate(values):
        result[roi, artifacts] = np.interp(artifacts, time[valid], row[valid])
    return result


def build_scenarios(
    traces: np.ndarray,
    bad_neurons: Iterable[int] = (),
    artifact_frames: Iterable[int] = (),
    smoothing_window: int = 3,
) -> dict[str, ScenarioData]:
    """Construct the four preprocessing scenarios described in the manuscript."""

    original = validate_traces(traces)
    retained, node_indices = remove_bad_neurons(original, bad_neurons)
    corrected = correct_artifact_frames(retained, artifact_frames)
    smoothed = smooth_traces(corrected, smoothing_window)
    return {
        "A": ScenarioData(
            "A",
            "Includes labelled bad neurons and retained artefacts.",
            original,
            np.arange(original.shape[0]),
        ),
        "B": ScenarioData(
            "B",
            "Excludes labelled bad neurons but retains artefacts.",
            retained,
            node_indices,
        ),
        "C": ScenarioData(
            "C",
            "Excludes labelled bad neurons and removes artefacts.",
            corrected,
            node_indices,
        ),
        "D": ScenarioData(
            "D",
            "Smoothed data after labelled-neuron exclusion and artefact removal.",
            smoothed,
            node_indices,
        ),
    }
