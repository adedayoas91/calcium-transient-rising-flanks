"""Deterministic onset detectors for calcium-transient traces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import (
    ThresholdCalibrationResult,
    calibrate_roi_thresholds,
    calibration_signal,
)
from .preprocessing import validate_traces

_CHANGE_POINT_ASSUMPTIONS = (
    "Onsets appear as sustained positive innovations relative to a robust noise threshold.",
    "The earliest strong score within the dominant local maximum is reported as the onset.",
)
_BAYESIAN_ASSUMPTIONS = (
    "Each onset is approximated by a non-negative rise-then-decay template.",
    "Posterior mass is computed from deterministic Gaussian-style residual scores and a threshold-informed prior bonus.",
)


@dataclass(frozen=True)
class ChangePointOnsetResult:
    """Per-ROI onset calls from thresholded change-point scoring."""

    onset_indices: np.ndarray
    onset_scores: np.ndarray
    confidences: np.ndarray
    uncertainty_widths: np.ndarray
    candidate_mask: np.ndarray
    calibration: ThresholdCalibrationResult
    assumptions: tuple[str, ...] = _CHANGE_POINT_ASSUMPTIONS

    def __post_init__(self) -> None:
        onset_indices = np.asarray(self.onset_indices, dtype=int)
        scores = np.asarray(self.onset_scores, dtype=float)
        confidences = np.asarray(self.confidences, dtype=float)
        widths = np.asarray(self.uncertainty_widths, dtype=int)
        mask = np.asarray(self.candidate_mask, dtype=bool)
        if scores.ndim != 2:
            raise ValueError("onset_scores must have shape (n_rois, n_timepoints)")
        if onset_indices.shape != (scores.shape[0],):
            raise ValueError("onset_indices must provide one value per ROI")
        if confidences.shape != onset_indices.shape or widths.shape != onset_indices.shape:
            raise ValueError("confidences and uncertainty_widths must match onset_indices")
        if mask.shape != scores.shape:
            raise ValueError("candidate_mask must match onset_scores")
        if np.any(onset_indices < -1) or np.any(onset_indices >= scores.shape[1]):
            raise ValueError("onset_indices must lie in [-1, n_timepoints)")
        if not np.isfinite(scores).all() or not np.isfinite(confidences).all():
            raise ValueError("change-point outputs must be finite")
        if np.any((confidences < 0.0) | (confidences > 1.0)):
            raise ValueError("confidences must lie in [0, 1]")
        if np.any(widths < 0):
            raise ValueError("uncertainty_widths cannot be negative")
        object.__setattr__(self, "onset_indices", onset_indices)
        object.__setattr__(self, "onset_scores", scores)
        object.__setattr__(self, "confidences", confidences)
        object.__setattr__(self, "uncertainty_widths", widths)
        object.__setattr__(self, "candidate_mask", mask)
        object.__setattr__(self, "assumptions", tuple(self.assumptions))

    @property
    def detected(self) -> np.ndarray:
        return self.onset_indices >= 0


@dataclass(frozen=True)
class BayesianOnsetResult:
    """Per-ROI posterior onset summaries from a kinetics-aware template model."""

    onset_indices: np.ndarray
    posterior_probabilities: np.ndarray
    null_probabilities: np.ndarray
    map_probabilities: np.ndarray
    credible_intervals: np.ndarray
    posterior_entropies: np.ndarray
    calibration: ThresholdCalibrationResult
    assumptions: tuple[str, ...] = _BAYESIAN_ASSUMPTIONS

    def __post_init__(self) -> None:
        onset_indices = np.asarray(self.onset_indices, dtype=int)
        posterior = np.asarray(self.posterior_probabilities, dtype=float)
        null_probabilities = np.asarray(self.null_probabilities, dtype=float)
        map_probabilities = np.asarray(self.map_probabilities, dtype=float)
        credible_intervals = np.asarray(self.credible_intervals, dtype=int)
        entropies = np.asarray(self.posterior_entropies, dtype=float)
        if posterior.ndim != 2:
            raise ValueError(
                "posterior_probabilities must have shape (n_rois, n_timepoints)"
            )
        if onset_indices.shape != (posterior.shape[0],):
            raise ValueError("onset_indices must provide one value per ROI")
        expected = onset_indices.shape
        if (
            null_probabilities.shape != expected
            or map_probabilities.shape != expected
            or entropies.shape != expected
        ):
            raise ValueError("probability summaries must provide one value per ROI")
        if credible_intervals.shape != (posterior.shape[0], 2):
            raise ValueError("credible_intervals must have shape (n_rois, 2)")
        if np.any(onset_indices < -1) or np.any(onset_indices >= posterior.shape[1]):
            raise ValueError("onset_indices must lie in [-1, n_timepoints)")
        if not (
            np.isfinite(posterior).all()
            and np.isfinite(null_probabilities).all()
            and np.isfinite(map_probabilities).all()
            and np.isfinite(entropies).all()
        ):
            raise ValueError("Bayesian onset outputs must be finite")
        if np.any(posterior < 0.0) or np.any(null_probabilities < 0.0):
            raise ValueError("posterior probabilities cannot be negative")
        object.__setattr__(self, "onset_indices", onset_indices)
        object.__setattr__(self, "posterior_probabilities", posterior)
        object.__setattr__(self, "null_probabilities", null_probabilities)
        object.__setattr__(self, "map_probabilities", map_probabilities)
        object.__setattr__(self, "credible_intervals", credible_intervals)
        object.__setattr__(self, "posterior_entropies", entropies)
        object.__setattr__(self, "assumptions", tuple(self.assumptions))

    @property
    def event_probabilities(self) -> np.ndarray:
        return 1.0 - self.null_probabilities


def detect_change_point_onsets(
    traces: np.ndarray,
    *,
    calibration: ThresholdCalibrationResult | None = None,
    calibration_method: str = "first_difference_mad",
    threshold_multiplier: float = 3.0,
    min_pre_samples: int = 2,
    min_post_samples: int = 3,
    min_persistence: int = 2,
    uncertainty_fraction: float = 0.75,
) -> ChangePointOnsetResult:
    """Detect onsets from sustained thresholded innovation changes."""

    values = validate_traces(traces)
    min_pre_samples = _validate_positive_integer("min_pre_samples", min_pre_samples)
    min_post_samples = _validate_positive_integer("min_post_samples", min_post_samples)
    min_persistence = _validate_positive_integer("min_persistence", min_persistence)
    fraction = float(uncertainty_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("uncertainty_fraction must lie in (0, 1]")
    calibration = _resolve_calibration(
        values,
        calibration=calibration,
        calibration_method=calibration_method,
        threshold_multiplier=threshold_multiplier,
    )
    innovations = calibration_signal(values, calibration, pad=True)
    excess = np.maximum(innovations - calibration.thresholds[:, None], 0.0)
    n_rois, n_timepoints = values.shape

    scores = np.zeros_like(values)
    candidate_mask = np.zeros_like(values, dtype=bool)
    onsets = np.full(n_rois, -1, dtype=int)
    confidences = np.zeros(n_rois, dtype=float)
    widths = np.zeros(n_rois, dtype=int)

    for roi in range(n_rois):
        for timepoint in range(1, n_timepoints):
            post_stop = min(n_timepoints, timepoint + min_post_samples)
            post = excess[roi, timepoint:post_stop]
            if post.size < min_post_samples:
                continue
            pre_start = max(1, timepoint - min_pre_samples)
            pre = excess[roi, pre_start:timepoint]
            if pre.size < min_pre_samples:
                continue
            persistence = np.count_nonzero(post > 0.0)
            if persistence < min_persistence:
                continue
            candidate_mask[roi, timepoint] = True
            scores[roi, timepoint] = float(post.mean() - pre.mean())

        peak_score = float(scores[roi].max())
        scale = max(float(calibration.scales[roi]), calibration.min_scale)
        if peak_score <= 0.0 or (peak_score / scale) <= 1.0:
            continue
        support = np.flatnonzero(scores[roi] >= peak_score * fraction)
        chosen = int(support[0])
        onsets[roi] = chosen
        widths[roi] = int(support[-1] - support[0] + 1)
        confidences[roi] = float(1.0 - np.exp(-peak_score / scale))

    return ChangePointOnsetResult(
        onset_indices=onsets,
        onset_scores=scores,
        confidences=confidences,
        uncertainty_widths=widths,
        candidate_mask=candidate_mask,
        calibration=calibration,
    )


def detect_bayesian_onsets(
    traces: np.ndarray,
    *,
    calibration: ThresholdCalibrationResult | None = None,
    calibration_method: str = "ar_residual_mad",
    threshold_multiplier: float = 3.0,
    rise_samples: int = 4,
    decay_samples: float = 12.0,
    prior_strength: float = 1.0,
    credibility: float = 0.8,
) -> BayesianOnsetResult:
    """Detect onsets via a deterministic kinetics-aware posterior approximation."""

    values = validate_traces(traces)
    rise_samples = _validate_positive_integer("rise_samples", rise_samples)
    decay = float(decay_samples)
    prior_strength = float(prior_strength)
    credibility = float(credibility)
    if not np.isfinite(decay) or decay <= 0.0:
        raise ValueError("decay_samples must be positive and finite")
    if not np.isfinite(prior_strength) or prior_strength < 0.0:
        raise ValueError("prior_strength must be finite and non-negative")
    if not 0.0 < credibility < 1.0:
        raise ValueError("credibility must lie in (0, 1)")

    calibration = _resolve_calibration(
        values,
        calibration=calibration,
        calibration_method=calibration_method,
        threshold_multiplier=threshold_multiplier,
    )
    innovations = calibration_signal(values, calibration, pad=True)
    n_rois, n_timepoints = values.shape
    posterior = np.zeros_like(values)
    null_probabilities = np.zeros(n_rois, dtype=float)
    map_probabilities = np.zeros(n_rois, dtype=float)
    onsets = np.full(n_rois, -1, dtype=int)
    credible_intervals = np.full((n_rois, 2), -1, dtype=int)
    entropies = np.zeros(n_rois, dtype=float)

    for roi in range(n_rois):
        sigma = max(float(calibration.scales[roi]), calibration.min_scale)
        trace = values[roi]
        event_penalty = np.log(float(n_timepoints))
        null_baseline = float(np.median(trace))
        null_residual = trace - null_baseline
        null_log_weight = -0.5 * float(np.dot(null_residual, null_residual)) / (
            sigma * sigma
        )
        log_weights = np.full(n_timepoints, -np.inf, dtype=float)

        for onset in range(1, n_timepoints):
            baseline = float(np.median(trace[:onset]))
            template = _kinetics_template(
                n_timepoints,
                onset=onset,
                rise_samples=rise_samples,
                decay_samples=decay,
            )
            response = trace - baseline
            denominator = float(np.dot(template, template))
            amplitude = 0.0
            if denominator > 0.0:
                amplitude = max(0.0, float(np.dot(response, template)) / denominator)
            residual = trace - (baseline + amplitude * template)
            log_likelihood = -0.5 * float(np.dot(residual, residual)) / (sigma * sigma)
            prior_bonus = prior_strength * max(
                0.0,
                float(innovations[roi, onset] - calibration.thresholds[roi]) / sigma,
            )
            log_weights[onset] = log_likelihood + prior_bonus - event_penalty

        stacked = np.concatenate(([null_log_weight], log_weights[1:]))
        posterior_with_null = _softmax(stacked)
        null_probabilities[roi] = float(posterior_with_null[0])
        posterior[roi, 1:] = posterior_with_null[1:]
        map_time = int(np.argmax(posterior[roi]))
        map_probabilities[roi] = float(posterior[roi, map_time])
        if null_probabilities[roi] >= map_probabilities[roi] or (
            1.0 - null_probabilities[roi]
        ) <= 0.5:
            entropies[roi] = _entropy(posterior_with_null)
            continue
        onsets[roi] = map_time
        credible_intervals[roi] = _shortest_credible_interval(
            posterior[roi],
            credibility=credibility,
        )
        entropies[roi] = _entropy(posterior_with_null)

    return BayesianOnsetResult(
        onset_indices=onsets,
        posterior_probabilities=posterior,
        null_probabilities=null_probabilities,
        map_probabilities=map_probabilities,
        credible_intervals=credible_intervals,
        posterior_entropies=entropies,
        calibration=calibration,
    )


def _resolve_calibration(
    traces: np.ndarray,
    *,
    calibration: ThresholdCalibrationResult | None,
    calibration_method: str,
    threshold_multiplier: float,
) -> ThresholdCalibrationResult:
    if calibration is not None:
        if calibration.n_rois != traces.shape[0]:
            raise ValueError(
                "calibration and traces must contain the same number of ROIs"
            )
        return calibration
    return calibrate_roi_thresholds(
        traces,
        method=calibration_method,
        threshold_multiplier=threshold_multiplier,
    )


def _validate_positive_integer(name: str, value: int) -> int:
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    integer = int(value)
    if integer < 1:
        raise ValueError(f"{name} must be at least 1")
    return integer


def _kinetics_template(
    n_timepoints: int,
    *,
    onset: int,
    rise_samples: int,
    decay_samples: float,
) -> np.ndarray:
    template = np.zeros(n_timepoints, dtype=float)
    for timepoint in range(onset, n_timepoints):
        delay = timepoint - onset
        if delay < rise_samples:
            template[timepoint] = (delay + 1.0) / rise_samples
        else:
            template[timepoint] = np.exp(-(delay - rise_samples + 1.0) / decay_samples)
    return template


def _softmax(log_weights: np.ndarray) -> np.ndarray:
    maximum = float(np.max(log_weights))
    shifted = np.exp(log_weights - maximum)
    total = float(shifted.sum())
    if total == 0.0:
        return np.full(log_weights.shape, 1.0 / log_weights.size)
    return shifted / total


def _shortest_credible_interval(
    probabilities: np.ndarray,
    *,
    credibility: float,
) -> np.ndarray:
    support = np.flatnonzero(probabilities > 0.0)
    if support.size == 0:
        return np.array([-1, -1], dtype=int)
    cumulative = np.concatenate(([0.0], np.cumsum(probabilities)))
    best: tuple[int, int] | None = None
    for start in range(1, probabilities.size):
        if probabilities[start] == 0.0:
            continue
        target = cumulative[start] + credibility
        stop = int(np.searchsorted(cumulative, target, side="left"))
        stop = min(stop, probabilities.size)
        mass = cumulative[stop] - cumulative[start]
        if mass < credibility:
            continue
        candidate = (start, stop - 1)
        if best is None or (candidate[1] - candidate[0]) < (best[1] - best[0]):
            best = candidate
    if best is None:
        best = (int(support[0]), int(support[-1]))
    return np.asarray(best, dtype=int)


def _entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0.0]
    if positive.size == 0:
        return 0.0
    return float(-np.sum(positive * np.log(positive)))
