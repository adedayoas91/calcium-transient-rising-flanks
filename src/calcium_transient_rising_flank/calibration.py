"""Network-blind per-ROI threshold calibration for onset detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .preprocessing import validate_traces

CalibrationMethod = Literal["first_difference_mad", "ar_residual_mad"]

_METHODS = {"first_difference_mad", "ar_residual_mad"}
_DEFAULT_ASSUMPTIONS = (
    "Each ROI is calibrated independently without network information.",
    "Noise scale is estimated robustly with a median-absolute-deviation rule.",
    "Positive innovations above the calibrated threshold indicate candidate onsets.",
)


@dataclass(frozen=True)
class ThresholdCalibrationResult:
    """Validated per-ROI thresholds derived from robust innovation scales."""

    method: CalibrationMethod
    centers: np.ndarray
    scales: np.ndarray
    thresholds: np.ndarray
    threshold_multiplier: float
    min_scale: float
    ar_coefficients: np.ndarray | None = None
    ar_intercepts: np.ndarray | None = None
    assumptions: tuple[str, ...] = _DEFAULT_ASSUMPTIONS

    def __post_init__(self) -> None:
        _validate_method(self.method)
        centers = np.asarray(self.centers, dtype=float)
        scales = np.asarray(self.scales, dtype=float)
        thresholds = np.asarray(self.thresholds, dtype=float)
        if centers.ndim != 1 or scales.ndim != 1 or thresholds.ndim != 1:
            raise ValueError("centers, scales, and thresholds must be one-dimensional")
        if not (centers.shape == scales.shape == thresholds.shape):
            raise ValueError("centers, scales, and thresholds must have the same shape")
        if centers.size < 1:
            raise ValueError("at least one ROI must be calibrated")
        if not (
            np.isfinite(centers).all()
            and np.isfinite(scales).all()
            and np.isfinite(thresholds).all()
        ):
            raise ValueError("calibration outputs must be finite")
        if np.any(scales <= 0.0):
            raise ValueError("calibration scales must be strictly positive")
        if np.any(thresholds < 0.0):
            raise ValueError("calibration thresholds must be non-negative")
        multiplier = float(self.threshold_multiplier)
        min_scale = float(self.min_scale)
        if not np.isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError("threshold_multiplier must be positive and finite")
        if not np.isfinite(min_scale) or min_scale <= 0.0:
            raise ValueError("min_scale must be positive and finite")
        if self.method == "ar_residual_mad":
            if self.ar_coefficients is None or self.ar_intercepts is None:
                raise ValueError(
                    "ar_residual_mad calibration requires AR coefficients and intercepts"
                )
            coefficients = np.asarray(self.ar_coefficients, dtype=float)
            intercepts = np.asarray(self.ar_intercepts, dtype=float)
            if coefficients.shape != centers.shape or intercepts.shape != centers.shape:
                raise ValueError(
                    "AR coefficients and intercepts must match the ROI dimension"
                )
            if not (np.isfinite(coefficients).all() and np.isfinite(intercepts).all()):
                raise ValueError("AR parameters must be finite")
            object.__setattr__(self, "ar_coefficients", coefficients)
            object.__setattr__(self, "ar_intercepts", intercepts)
        else:
            if self.ar_coefficients is not None or self.ar_intercepts is not None:
                raise ValueError(
                    "first_difference_mad calibration cannot include AR parameters"
                )
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(self, "threshold_multiplier", multiplier)
        object.__setattr__(self, "min_scale", min_scale)
        object.__setattr__(self, "assumptions", tuple(self.assumptions))

    @property
    def n_rois(self) -> int:
        return int(self.thresholds.size)


def calibrate_roi_thresholds(
    traces: np.ndarray,
    *,
    method: CalibrationMethod = "first_difference_mad",
    threshold_multiplier: float = 3.0,
    min_scale: float = 1e-6,
    ar_clip: float = 0.995,
) -> ThresholdCalibrationResult:
    """Calibrate deterministic per-ROI thresholds from robust innovation scales."""

    values = validate_traces(traces)
    _validate_method(method)
    multiplier = float(threshold_multiplier)
    min_scale = float(min_scale)
    clip = float(ar_clip)
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("threshold_multiplier must be positive and finite")
    if not np.isfinite(min_scale) or min_scale <= 0.0:
        raise ValueError("min_scale must be positive and finite")
    if not np.isfinite(clip) or clip <= 0.0:
        raise ValueError("ar_clip must be positive and finite")

    if method == "first_difference_mad":
        innovations = _first_difference_signal(values)
        centers, scales = _robust_location_scale(innovations, min_scale=min_scale)
        thresholds = np.maximum(centers + multiplier * scales, 0.0)
        return ThresholdCalibrationResult(
            method=method,
            centers=centers,
            scales=scales,
            thresholds=thresholds,
            threshold_multiplier=multiplier,
            min_scale=min_scale,
        )

    if values.shape[1] < 3:
        raise ValueError("ar_residual_mad calibration requires at least three samples")
    ar_coefficients, ar_intercepts = _fit_ar1_parameters(values, ar_clip=clip)
    residuals = _ar_residual_signal(
        values,
        ar_coefficients=ar_coefficients,
        ar_intercepts=ar_intercepts,
    )
    centers, scales = _robust_location_scale(residuals, min_scale=min_scale)
    thresholds = np.maximum(centers + multiplier * scales, 0.0)
    return ThresholdCalibrationResult(
        method=method,
        centers=centers,
        scales=scales,
        thresholds=thresholds,
        threshold_multiplier=multiplier,
        min_scale=min_scale,
        ar_coefficients=ar_coefficients,
        ar_intercepts=ar_intercepts,
    )


def calibration_signal(
    traces: np.ndarray,
    calibration: ThresholdCalibrationResult,
    *,
    pad: bool = True,
) -> np.ndarray:
    """Return the innovation/residual signal aligned to the trace time axis."""

    values = validate_traces(traces)
    if values.shape[0] != calibration.n_rois:
        raise ValueError("traces and calibration must contain the same number of ROIs")
    if calibration.method == "first_difference_mad":
        signal = _first_difference_signal(values)
    else:
        signal = _ar_residual_signal(
            values,
            ar_coefficients=np.asarray(calibration.ar_coefficients, dtype=float),
            ar_intercepts=np.asarray(calibration.ar_intercepts, dtype=float),
        )
    if not pad:
        return signal
    padded = np.zeros_like(values)
    padded[:, 1:] = signal
    return padded


def _validate_method(method: str) -> None:
    if method not in _METHODS:
        choices = ", ".join(sorted(_METHODS))
        raise ValueError(f"method must be one of {{{choices}}}")


def _robust_location_scale(
    values: np.ndarray,
    *,
    min_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    centers = np.median(values, axis=1)
    deviations = np.abs(values - centers[:, None])
    scales = 1.4826 * np.median(deviations, axis=1)
    return centers, np.maximum(scales, min_scale)


def _first_difference_signal(traces: np.ndarray) -> np.ndarray:
    return np.diff(traces, axis=1)


def _fit_ar1_parameters(
    traces: np.ndarray,
    *,
    ar_clip: float,
) -> tuple[np.ndarray, np.ndarray]:
    previous = traces[:, :-1]
    current = traces[:, 1:]
    previous_mean = previous.mean(axis=1)
    current_mean = current.mean(axis=1)
    previous_centered = previous - previous_mean[:, None]
    current_centered = current - current_mean[:, None]
    denominator = np.sum(previous_centered * previous_centered, axis=1)
    numerator = np.sum(previous_centered * current_centered, axis=1)
    coefficients = np.divide(
        numerator,
        denominator,
        out=np.zeros(traces.shape[0], dtype=float),
        where=denominator > 0.0,
    )
    coefficients = np.clip(coefficients, -ar_clip, ar_clip)
    intercepts = current_mean - coefficients * previous_mean
    return coefficients, intercepts


def _ar_residual_signal(
    traces: np.ndarray,
    *,
    ar_coefficients: np.ndarray,
    ar_intercepts: np.ndarray,
) -> np.ndarray:
    previous = traces[:, :-1]
    current = traces[:, 1:]
    return current - (
        ar_coefficients[:, None] * previous + ar_intercepts[:, None]
    )
