"""Pre-network transient and residual diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocessing import validate_traces
from .representations import build_representations


@dataclass(frozen=True)
class TransientSummary:
    """Per-ROI summaries computed before graph inspection."""

    gamma: np.ndarray
    event_density: np.ndarray
    rise_count: np.ndarray
    median_rise_duration: np.ndarray
    signal_to_noise: np.ndarray


@dataclass(frozen=True)
class ResidualDiagnostics:
    """Per-ROI descriptive checks for a fitted-model residual series."""

    mean: np.ndarray
    standard_deviation: np.ndarray
    lag1_autocorrelation: np.ndarray
    skewness: np.ndarray


def _positive_run_durations(active: np.ndarray) -> list[int]:
    padded = np.pad(active.astype(int), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return (stops - starts).tolist()


def characterize_transients(
    traces: np.ndarray,
    tolerance: float | np.ndarray = 0.0,
    gamma: float | np.ndarray | None = None,
) -> TransientSummary:
    """Summarize rise occurrence, duration, decay, and robust SNR per ROI.

    These quantities should be computed before viewing directed graphs when
    they are used to choose analysis settings.
    """

    values = validate_traces(traces)
    representations = build_representations(values, tolerance=tolerance, gamma=gamma)
    rise = representations.rise
    active = rise > 0
    event_density = np.mean(active, axis=1)
    rise_count = np.zeros(values.shape[0], dtype=int)
    median_duration = np.zeros(values.shape[0], dtype=float)
    signal_to_noise = np.zeros(values.shape[0], dtype=float)
    increments = np.diff(values, axis=1)
    eps = np.finfo(float).eps
    for roi in range(values.shape[0]):
        durations = _positive_run_durations(active[roi])
        rise_count[roi] = len(durations)
        median_duration[roi] = 0.0 if not durations else float(np.median(durations))
        noise_samples = increments[roi][~active[roi, 1:]]
        if noise_samples.size == 0:
            noise_samples = increments[roi]
        center = float(np.median(noise_samples))
        scale = 1.4826 * float(np.median(np.abs(noise_samples - center)))
        if scale <= eps:
            scale = max(float(np.std(noise_samples)), eps)
        amplitude = float(np.mean(rise[roi][active[roi]])) if active[roi].any() else 0.0
        signal_to_noise[roi] = amplitude / scale
    return TransientSummary(
        gamma=representations.gamma,
        event_density=event_density,
        rise_count=rise_count,
        median_rise_duration=median_duration,
        signal_to_noise=signal_to_noise,
    )


def residual_diagnostics(residuals: np.ndarray) -> ResidualDiagnostics:
    """Compute descriptive residual checks without parametric normality claims."""

    values = validate_traces(residuals)
    mean = np.mean(values, axis=1)
    standard_deviation = np.std(values, axis=1)
    lag1 = np.zeros(values.shape[0], dtype=float)
    skewness = np.zeros(values.shape[0], dtype=float)
    eps = np.finfo(float).eps
    for roi, row in enumerate(values):
        left = row[:-1] - np.mean(row[:-1])
        right = row[1:] - np.mean(row[1:])
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        lag1[roi] = 0.0 if denominator <= eps else float(np.dot(left, right) / denominator)
        if standard_deviation[roi] > eps:
            standardized = (row - mean[roi]) / standard_deviation[roi]
            skewness[roi] = float(np.mean(standardized**3))
    return ResidualDiagnostics(
        mean=mean,
        standard_deviation=standard_deviation,
        lag1_autocorrelation=lag1,
        skewness=skewness,
    )
