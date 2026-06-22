"""Fixed-length calcium-transient representations and decay controls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocessing import validate_traces


@dataclass(frozen=True)
class RepresentationBundle:
    """Representations defined on the same original time axis."""

    full: np.ndarray
    deconvolved: np.ndarray
    rise: np.ndarray
    fall: np.ndarray
    fall_residual: np.ndarray
    gamma: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "full": self.full,
            "deconvolved": self.deconvolved,
            "rise": self.rise,
            "fall": self.fall,
            "fall_residual": self.fall_residual,
        }


def _per_roi_parameter(
    value: float | np.ndarray, n_rois: int, name: str
) -> np.ndarray:
    parameter = np.asarray(value, dtype=float)
    if parameter.ndim == 0:
        parameter = np.full(n_rois, float(parameter))
    if parameter.shape != (n_rois,):
        raise ValueError(f"{name} must be a scalar or one value per ROI")
    return parameter


def rising_flank(
    traces: np.ndarray, tolerance: float | np.ndarray = 0.0
) -> np.ndarray:
    """Return thresholded positive increments while retaining all frames."""

    values = validate_traces(traces)
    delta = _per_roi_parameter(tolerance, values.shape[0], "tolerance")
    result = np.zeros_like(values)
    result[:, 1:] = np.maximum(np.diff(values, axis=1) - delta[:, None], 0.0)
    return result


def falling_flank(
    traces: np.ndarray, tolerance: float | np.ndarray = 0.0
) -> np.ndarray:
    """Return thresholded negative increments while retaining all frames."""

    values = validate_traces(traces)
    delta = _per_roi_parameter(tolerance, values.shape[0], "tolerance")
    result = np.zeros_like(values)
    result[:, 1:] = np.maximum(-np.diff(values, axis=1) - delta[:, None], 0.0)
    return result


def estimate_decay(traces: np.ndarray, default: float = 0.95) -> np.ndarray:
    """Estimate an AR(1) persistence parameter from declining samples.

    This lightweight estimator is intended for prespecified analysis and
    synthetic validation; empirical use should report sensitivity to the
    selected kinetic model.
    """

    values = validate_traces(traces)
    gamma = np.full(values.shape[0], default, dtype=float)
    for roi, row in enumerate(values):
        previous = row[:-1]
        current = row[1:]
        mask = (previous > 0) & (current >= 0) & (current <= previous)
        if mask.any() and np.dot(previous[mask], previous[mask]) > 0:
            estimate = np.dot(previous[mask], current[mask]) / np.dot(
                previous[mask], previous[mask]
            )
            gamma[roi] = np.clip(estimate, 0.0, 0.999)
    return gamma


def deconvolve_ar1(
    traces: np.ndarray, gamma: float | np.ndarray | None = None
) -> np.ndarray:
    """Estimate nonnegative innovations under `C_t = gamma C_(t-1) + S_t`."""

    values = validate_traces(traces)
    decay = estimate_decay(values) if gamma is None else _per_roi_parameter(
        gamma, values.shape[0], "gamma"
    )
    if np.any((decay < 0) | (decay >= 1)):
        raise ValueError("gamma values must lie in [0, 1)")
    events = np.zeros_like(values)
    events[:, 0] = np.maximum(values[:, 0], 0.0)
    events[:, 1:] = np.maximum(
        values[:, 1:] - decay[:, None] * values[:, :-1], 0.0
    )
    return events


def decay_null_residual(
    traces: np.ndarray,
    gamma: float | np.ndarray | None = None,
    tolerance: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Compute falling increments exceeding the expected AR(1) decline."""

    values = validate_traces(traces)
    decay = estimate_decay(values) if gamma is None else _per_roi_parameter(
        gamma, values.shape[0], "gamma"
    )
    fall = falling_flank(values, tolerance)
    expected = np.zeros_like(values)
    expected[:, 1:] = np.maximum(
        (1.0 - decay[:, None]) * values[:, :-1], 0.0
    )
    return np.maximum(fall - expected, 0.0)


def build_representations(
    traces: np.ndarray,
    tolerance: float | np.ndarray = 0.0,
    gamma: float | np.ndarray | None = None,
) -> RepresentationBundle:
    """Generate all primary and comparator representations in the manuscript."""

    full = validate_traces(traces)
    decay = estimate_decay(full) if gamma is None else _per_roi_parameter(
        gamma, full.shape[0], "gamma"
    )
    return RepresentationBundle(
        full=full,
        deconvolved=deconvolve_ar1(full, decay),
        rise=rising_flank(full, tolerance),
        fall=falling_flank(full, tolerance),
        fall_residual=decay_null_residual(full, decay, tolerance),
        gamma=decay,
    )
