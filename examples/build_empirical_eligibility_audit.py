"""Build a descriptive eligibility ledger for archived motoneuron recordings."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from calcium_transient_rising_flank.diagnostics import characterize_transients


DEFAULT_DATA_DIR = Path("data/motoneurons")
DEFAULT_OUTPUT_DIR = Path(
    "outputs/method_validation/empirical_eligibility"
)
MATCHED_RECORDINGS = {"F3T1", "F3T2", "F5T2"}


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _recording_label(key: tuple[int, int]) -> str:
    return f"F{key[0]}T{key[1]}"


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _median(values: np.ndarray) -> float | None:
    selected = _finite(np.asarray(values, dtype=float))
    return None if selected.size == 0 else float(np.median(selected))


def _maximum(values: np.ndarray) -> float | None:
    selected = _finite(np.asarray(values, dtype=float))
    return None if selected.size == 0 else float(np.max(selected))


def _per_roi_drift(values: np.ndarray) -> np.ndarray:
    n_frames = values.shape[1]
    time = np.arange(n_frames, dtype=float)
    centered_time = time - np.mean(time)
    denominator = float(np.dot(centered_time, centered_time))
    result = np.full(values.shape[0], np.nan)
    for roi, row in enumerate(values):
        if not np.isfinite(row).all() or denominator == 0.0:
            continue
        centered = row - np.mean(row)
        slope = float(np.dot(centered_time, centered) / denominator)
        scale = float(np.subtract(*np.percentile(row, [75.0, 25.0])))
        if scale <= np.finfo(float).eps:
            scale = float(np.std(row))
        if scale > np.finfo(float).eps:
            result[roi] = slope * max(n_frames - 1, 1) / scale
    return result


def _split_half_diagnostics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    midpoint = values.shape[1] // 2
    mean_shift = np.full(values.shape[0], np.nan)
    log2_variance_ratio = np.full(values.shape[0], np.nan)
    eps = np.finfo(float).eps
    for roi, row in enumerate(values):
        if midpoint == 0 or not np.isfinite(row).all():
            continue
        left, right = row[:midpoint], row[midpoint:]
        scale = float(np.subtract(*np.percentile(row, [75.0, 25.0])))
        if scale <= eps:
            scale = float(np.std(row))
        if scale > eps:
            mean_shift[roi] = (float(np.mean(right)) - float(np.mean(left))) / scale
        left_var = float(np.var(left))
        right_var = float(np.var(right))
        log2_variance_ratio[roi] = float(
            np.log2((right_var + eps) / (left_var + eps))
        )
    return mean_shift, log2_variance_ratio


def _lag1_autocorrelation(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape[0], np.nan)
    eps = np.finfo(float).eps
    for roi, row in enumerate(values):
        if row.size < 2 or not np.isfinite(row).all():
            continue
        left = row[:-1] - np.mean(row[:-1])
        right = row[1:] - np.mean(row[1:])
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator > eps:
            result[roi] = float(np.dot(left, right) / denominator)
    return result


def _extreme_repeat_fraction(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape[0], np.nan)
    for roi, row in enumerate(values):
        if row.size == 0 or not np.isfinite(row).all():
            continue
        extremes = (row == np.min(row)) | (row == np.max(row))
        repeated = np.zeros_like(extremes)
        if row.size > 1:
            repeated[1:] |= extremes[1:] & extremes[:-1] & (row[1:] == row[:-1])
            repeated[:-1] |= extremes[:-1] & extremes[1:] & (row[:-1] == row[1:])
        result[roi] = float(np.mean(repeated))
    return result


def audit_recording(
    key: tuple[int, int],
    traces: np.ndarray,
    *,
    n_rois_before_exclusion: int,
    artifact_frame: int | None,
) -> dict[str, Any]:
    """Return processed-trace diagnostics without an eligibility decision."""

    values = np.asarray(traces, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("traces must have shape (n_rois, n_frames)")
    label = _recording_label(key)
    finite = np.isfinite(values)
    standard_deviation = np.nanstd(values, axis=1)
    drift = _per_roi_drift(values)
    half_mean, half_variance = _split_half_diagnostics(values)
    lag1 = _lag1_autocorrelation(values)
    extreme_repeat = _extreme_repeat_fraction(values)

    transient_status = "ok"
    transient_values: dict[str, float | None]
    if finite.all():
        transient = characterize_transients(values)
        transient_values = {
            "gamma_median": _median(transient.gamma),
            "event_density_median": _median(transient.event_density),
            "rise_count_median": _median(transient.rise_count),
            "median_rise_duration_median": _median(
                transient.median_rise_duration
            ),
            "signal_to_noise_median": _median(transient.signal_to_noise),
        }
    else:
        transient_status = "not_computed_nonfinite_values"
        transient_values = {
            "gamma_median": None,
            "event_density_median": None,
            "rise_count_median": None,
            "median_rise_duration_median": None,
            "signal_to_noise_median": None,
        }

    n_rois = int(values.shape[0])
    return {
        "recording": label,
        "fish": int(key[0]),
        "trial": int(key[1]),
        "matched_legacy_subset": label in MATCHED_RECORDINGS,
        "selection_provenance": (
            "legacy combined archive; original selection rationale unavailable"
            if label in MATCHED_RECORDINGS
            else "available recording outside the legacy paired subset"
        ),
        "n_frames": int(values.shape[1]),
        "n_rois_before_exclusion": int(n_rois_before_exclusion),
        "n_rois_after_exclusion": n_rois,
        "n_rois_excluded": int(n_rois_before_exclusion - n_rois),
        "nonfinite_value_count": int(values.size - np.count_nonzero(finite)),
        "nonfinite_value_fraction": float(1.0 - np.mean(finite)),
        "constant_roi_count": int(
            np.count_nonzero(standard_deviation <= np.finfo(float).eps)
        ),
        "absolute_normalized_drift_median": _median(np.abs(drift)),
        "absolute_normalized_drift_max": _maximum(np.abs(drift)),
        "absolute_split_half_mean_shift_median": _median(np.abs(half_mean)),
        "absolute_split_half_mean_shift_max": _maximum(np.abs(half_mean)),
        "absolute_log2_split_half_variance_ratio_median": _median(
            np.abs(half_variance)
        ),
        "lag1_autocorrelation_median": _median(lag1),
        "lag1_autocorrelation_max": _maximum(lag1),
        "repeated_extreme_fraction_median": _median(extreme_repeat),
        "repeated_extreme_fraction_max": _maximum(extreme_repeat),
        "artifact_annotation_available": artifact_frame is not None,
        "archived_artifact_frame": artifact_frame,
        "transient_summary_status": transient_status,
        **transient_values,
        "stationarity_interpretation": (
            "descriptive drift and split-half proxies; no binary stationarity gate"
        ),
        "saturation_interpretation": (
            "repeated processed-trace extrema only; true sensor saturation unavailable"
        ),
        "motion_audit": "unavailable_without raw movie or motion traces",
        "neuropil_audit": "unavailable_without raw movie or extraction metadata",
        "raw_movie_available": False,
        "eligibility_decision": "not assigned retrospectively",
    }


def build_rows(data_dir: Path) -> list[dict[str, Any]]:
    before: Mapping[tuple[int, int], Any] = _load_pickle(
        data_dir / "dff_dict_with_bad_neurons.pkl"
    )
    processed: Mapping[tuple[int, int], Any] = _load_pickle(
        data_dir / "dff_smoothed_dict.pkl"
    )
    artifacts: Mapping[tuple[int, int], Any] = _load_pickle(
        data_dir / "artifact_dict.pkl"
    )
    missing = sorted(set(processed) - set(before))
    if missing:
        raise ValueError(f"recordings missing from pre-exclusion data: {missing}")
    return [
        audit_recording(
            key,
            np.asarray(processed[key], dtype=float),
            n_rois_before_exclusion=int(np.asarray(before[key]).shape[0]),
            artifact_frame=(None if key not in artifacts else int(artifacts[key])),
        )
        for key in sorted(processed)
    ]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one recording is required")
    return {
        "status": "complete",
        "n_recordings": len(rows),
        "n_fish": len({int(row["fish"]) for row in rows}),
        "n_legacy_subset_recordings": sum(
            bool(row["matched_legacy_subset"]) for row in rows
        ),
        "recordings_with_artifact_annotation": sum(
            bool(row["artifact_annotation_available"]) for row in rows
        ),
        "frames_min": min(int(row["n_frames"]) for row in rows),
        "frames_max": max(int(row["n_frames"]) for row in rows),
        "rois_after_exclusion_min": min(
            int(row["n_rois_after_exclusion"]) for row in rows
        ),
        "rois_after_exclusion_max": max(
            int(row["n_rois_after_exclusion"]) for row in rows
        ),
        "total_excluded_rois": sum(int(row["n_rois_excluded"]) for row in rows),
        "total_nonfinite_values": sum(
            int(row["nonfinite_value_count"]) for row in rows
        ),
        "total_constant_rois": sum(int(row["constant_roi_count"]) for row in rows),
        "limitations": [
            "The audit uses processed ROI traces, not raw movies.",
            "Motion, neuropil contamination, and true sensor saturation are unavailable.",
            "The original rationale for the three-recording legacy subset is unavailable.",
            "No retrospective eligibility decision was assigned.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(args.data_dir)
    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "empirical_eligibility_audit.csv", rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
