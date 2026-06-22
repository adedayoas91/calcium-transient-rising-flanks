import unittest

import numpy as np

from calcium_transient_rising_flank.diagnostics import (
    characterize_transients,
    residual_diagnostics,
)


class DiagnosticTests(unittest.TestCase):
    def test_transient_characterization_reports_prespecified_roi_summaries(self) -> None:
        traces = np.array(
            [
                [0.0, 1.0, 0.8, 0.6, 0.0, 0.0, 1.0, 0.7],
                [0.0, 0.0, 0.9, 0.7, 0.5, 0.0, 0.0, 0.0],
            ]
        )

        summary = characterize_transients(traces, tolerance=0.05, gamma=0.8)

        np.testing.assert_allclose(summary.gamma, [0.8, 0.8])
        np.testing.assert_array_equal(summary.rise_count, [2, 1])
        self.assertTrue(np.all(summary.event_density > 0))
        self.assertEqual(summary.median_rise_duration.shape, (2,))
        self.assertTrue(np.isfinite(summary.signal_to_noise).all())

    def test_residual_diagnostics_include_lag_one_autocorrelation_and_skewness(self) -> None:
        residuals = np.array(
            [[-1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, -1.0]]
        )

        diagnostics = residual_diagnostics(residuals)

        self.assertEqual(diagnostics.lag1_autocorrelation.shape, (2,))
        self.assertEqual(diagnostics.skewness.shape, (2,))
        self.assertTrue(np.isfinite(diagnostics.standard_deviation).all())


if __name__ == "__main__":
    unittest.main()
