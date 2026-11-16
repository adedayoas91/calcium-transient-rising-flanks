import unittest

import numpy as np

from calcium_transient_rising_flank.calibration import calibrate_roi_thresholds
from calcium_transient_rising_flank.onset_detection import (
    ChangePointOnsetResult,
    detect_bayesian_onsets,
    detect_change_point_onsets,
)


def _synthetic_trace(
    *,
    n_timepoints: int,
    onset: int | None,
    rise_samples: int = 4,
    decay_samples: float = 12.0,
    amplitude: float = 3.0,
    noise_scale: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    trace = rng.normal(scale=noise_scale, size=n_timepoints)
    if onset is None:
        return trace
    for timepoint in range(onset, n_timepoints):
        delay = timepoint - onset
        if delay < rise_samples:
            trace[timepoint] += amplitude * (delay + 1.0) / rise_samples
        else:
            trace[timepoint] += amplitude * np.exp(
                -(delay - rise_samples + 1.0) / decay_samples
            )
    return trace


class OnsetDetectionTests(unittest.TestCase):
    def test_change_point_detector_finds_known_onset_and_rejects_flat_trace(self) -> None:
        traces = np.vstack(
            [
                _synthetic_trace(n_timepoints=60, onset=20, amplitude=3.5, seed=2),
                _synthetic_trace(n_timepoints=60, onset=None, noise_scale=0.02, seed=3),
            ]
        )

        result = detect_change_point_onsets(
            traces,
            threshold_multiplier=2.0,
            min_pre_samples=2,
            min_post_samples=2,
            min_persistence=1,
        )

        self.assertLessEqual(abs(int(result.onset_indices[0]) - 20), 1)
        self.assertEqual(int(result.onset_indices[1]), -1)
        self.assertGreater(float(result.confidences[0]), 0.5)
        self.assertLess(float(result.confidences[1]), 0.1)
        self.assertTrue(bool(result.candidate_mask[0, result.onset_indices[0]]))

    def test_bayesian_detector_returns_posterior_and_credible_interval(self) -> None:
        traces = np.vstack(
            [
                _synthetic_trace(n_timepoints=70, onset=24, amplitude=4.0, seed=5),
                _synthetic_trace(n_timepoints=70, onset=None, noise_scale=0.02, seed=6),
            ]
        )

        result = detect_bayesian_onsets(
            traces,
            threshold_multiplier=2.5,
            rise_samples=4,
            decay_samples=10.0,
            prior_strength=1.0,
            credibility=0.8,
        )

        self.assertLessEqual(abs(int(result.onset_indices[0]) - 24), 1)
        self.assertGreater(float(result.map_probabilities[0]), 0.2)
        self.assertGreaterEqual(int(result.credible_intervals[0, 0]), 1)
        self.assertLessEqual(int(result.credible_intervals[0, 0]), 24)
        self.assertGreaterEqual(int(result.credible_intervals[0, 1]), 24)
        self.assertEqual(int(result.onset_indices[1]), -1)
        self.assertGreater(float(result.null_probabilities[1]), 0.5)
        np.testing.assert_allclose(
            result.posterior_probabilities[0].sum() + result.null_probabilities[0],
            1.0,
            atol=1e-8,
        )

    def test_detectors_are_reproducible_with_shared_calibration(self) -> None:
        traces = np.vstack(
            [
                _synthetic_trace(n_timepoints=55, onset=18, amplitude=3.2, seed=11),
                _synthetic_trace(n_timepoints=55, onset=30, amplitude=2.8, seed=12),
            ]
        )
        calibration = calibrate_roi_thresholds(
            traces,
            method="ar_residual_mad",
            threshold_multiplier=2.5,
        )

        first_cp = detect_change_point_onsets(
            traces,
            calibration=calibration,
            min_pre_samples=2,
            min_post_samples=2,
            min_persistence=1,
        )
        second_cp = detect_change_point_onsets(
            traces,
            calibration=calibration,
            min_pre_samples=2,
            min_post_samples=2,
            min_persistence=1,
        )
        np.testing.assert_array_equal(first_cp.onset_indices, second_cp.onset_indices)
        np.testing.assert_allclose(first_cp.onset_scores, second_cp.onset_scores)

        first_bayes = detect_bayesian_onsets(
            traces,
            calibration=calibration,
            rise_samples=4,
            decay_samples=8.0,
        )
        second_bayes = detect_bayesian_onsets(
            traces,
            calibration=calibration,
            rise_samples=4,
            decay_samples=8.0,
        )
        np.testing.assert_array_equal(first_bayes.onset_indices, second_bayes.onset_indices)
        np.testing.assert_allclose(
            first_bayes.posterior_probabilities,
            second_bayes.posterior_probabilities,
        )

    def test_invalid_detector_parameters_are_rejected(self) -> None:
        traces = np.vstack(
            [
                _synthetic_trace(n_timepoints=20, onset=8, seed=21),
                _synthetic_trace(n_timepoints=20, onset=None, seed=22),
            ]
        )

        with self.assertRaisesRegex(ValueError, "uncertainty_fraction"):
            detect_change_point_onsets(traces, uncertainty_fraction=0.0)
        with self.assertRaisesRegex(ValueError, "calibration and traces"):
            detect_change_point_onsets(
                traces,
                calibration=calibrate_roi_thresholds(traces[:1]),
            )
        with self.assertRaisesRegex(ValueError, "credibility"):
            detect_bayesian_onsets(traces, credibility=1.0)
        with self.assertRaisesRegex(ValueError, "prior_strength"):
            detect_bayesian_onsets(traces, prior_strength=-1.0)

    def test_result_normalizes_array_like_inputs(self) -> None:
        calibration = calibrate_roi_thresholds(np.array([[0.0, 0.1, 0.0]]))
        result = ChangePointOnsetResult(
            onset_indices=[-1],
            onset_scores=[[0.0, 0.0, 0.0]],
            confidences=[0.0],
            uncertainty_widths=[0],
            candidate_mask=[[False, False, False]],
            calibration=calibration,
        )

        self.assertIsInstance(result.onset_indices, np.ndarray)
        np.testing.assert_array_equal(result.detected, np.array([False]))


if __name__ == "__main__":
    unittest.main()
