import unittest

import numpy as np

from calcium_transient_rising_flank.calibration import (
    ThresholdCalibrationResult,
    calibrate_roi_thresholds,
    calibration_signal,
)


class ThresholdCalibrationTests(unittest.TestCase):
    def test_first_difference_mad_uses_min_scale_for_constant_trace(self) -> None:
        traces = np.array(
            [
                [0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )

        result = calibrate_roi_thresholds(
            traces,
            method="first_difference_mad",
            threshold_multiplier=2.0,
            min_scale=0.1,
        )

        self.assertEqual(result.method, "first_difference_mad")
        np.testing.assert_allclose(result.scales, np.array([0.1, 0.1]))
        np.testing.assert_allclose(result.thresholds, result.centers + 2.0 * result.scales)
        signal = calibration_signal(traces, result)
        self.assertEqual(signal.shape, traces.shape)
        np.testing.assert_allclose(signal[:, 0], 0.0)

    def test_ar_residual_mad_recovers_ar1_structure(self) -> None:
        rng = np.random.default_rng(9)
        innovations = rng.normal(scale=0.08, size=240)
        trace = np.zeros_like(innovations)
        for index in range(1, innovations.size):
            trace[index] = 0.7 * trace[index - 1] + innovations[index]
        traces = trace[None, 40:]

        result = calibrate_roi_thresholds(
            traces,
            method="ar_residual_mad",
            threshold_multiplier=3.0,
            min_scale=1e-4,
        )

        self.assertEqual(result.method, "ar_residual_mad")
        self.assertIsNotNone(result.ar_coefficients)
        self.assertIsNotNone(result.ar_intercepts)
        self.assertAlmostEqual(float(result.ar_coefficients[0]), 0.7, delta=0.15)
        self.assertGreater(float(result.scales[0]), 0.0)
        signal = calibration_signal(traces, result, pad=False)
        self.assertEqual(signal.shape, (1, traces.shape[1] - 1))

    def test_calibration_is_deterministic(self) -> None:
        rng = np.random.default_rng(4)
        traces = rng.normal(size=(3, 20))

        first = calibrate_roi_thresholds(traces, method="ar_residual_mad")
        second = calibrate_roi_thresholds(traces, method="ar_residual_mad")

        np.testing.assert_allclose(first.scales, second.scales)
        np.testing.assert_allclose(first.thresholds, second.thresholds)
        np.testing.assert_allclose(first.ar_coefficients, second.ar_coefficients)

    def test_thresholds_are_clamped_to_representation_domain(self) -> None:
        traces = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
        result = calibrate_roi_thresholds(
            traces,
            method="first_difference_mad",
            threshold_multiplier=2.0,
            min_scale=0.1,
        )
        np.testing.assert_allclose(result.thresholds, 0.0)

    def test_result_normalizes_array_like_inputs(self) -> None:
        result = ThresholdCalibrationResult(
            method="first_difference_mad",
            centers=[0.0],
            scales=[0.1],
            thresholds=[0.3],
            threshold_multiplier=3,
            min_scale=0.1,
        )

        self.assertIsInstance(result.thresholds, np.ndarray)
        self.assertEqual(result.n_rois, 1)

    def test_invalid_configuration_is_rejected(self) -> None:
        traces = np.array([[0.0, 1.0], [1.0, 2.0]])

        with self.assertRaisesRegex(ValueError, "method"):
            calibrate_roi_thresholds(traces, method="unknown")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "threshold_multiplier"):
            calibrate_roi_thresholds(traces, threshold_multiplier=0.0)
        with self.assertRaisesRegex(ValueError, "at least three samples"):
            calibrate_roi_thresholds(traces, method="ar_residual_mad")


if __name__ == "__main__":
    unittest.main()
