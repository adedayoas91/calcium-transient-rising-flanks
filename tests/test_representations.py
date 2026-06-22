import unittest

import numpy as np

from calcium_transient_rising_flank.representations import (
    build_representations,
    decay_null_residual,
    deconvolve_ar1,
    falling_flank,
    rising_flank,
)


class RepresentationTests(unittest.TestCase):
    def test_rising_and_falling_flanks_retain_common_time_axis(self) -> None:
        traces = np.array([[0.0, 0.10, 0.40, 0.30, 0.10]])

        rise = rising_flank(traces, tolerance=0.05)
        fall = falling_flank(traces, tolerance=0.05)

        np.testing.assert_allclose(rise, [[0.0, 0.05, 0.25, 0.0, 0.0]])
        np.testing.assert_allclose(fall, [[0.0, 0.0, 0.0, 0.05, 0.15]])

    def test_deconvolution_and_decay_null_residual_use_declared_gamma(self) -> None:
        traces = np.array([[0.0, 1.0, 0.8, 0.64, 0.30]])

        events = deconvolve_ar1(traces, gamma=np.array([0.8]))
        residual = decay_null_residual(traces, gamma=np.array([0.8]))

        np.testing.assert_allclose(events, [[0.0, 1.0, 0.0, 0.0, 0.0]])
        self.assertAlmostEqual(residual[0, 2], 0.0)
        self.assertGreater(residual[0, 4], 0.0)

    def test_representation_bundle_contains_all_manuscript_comparators(self) -> None:
        traces = np.array([[0.0, 1.0, 0.8], [0.0, 0.2, 0.1]])

        bundle = build_representations(traces, tolerance=0.0, gamma=0.8)

        self.assertEqual(
            set(bundle.as_dict()),
            {"full", "deconvolved", "rise", "fall", "fall_residual"},
        )
        self.assertEqual(bundle.rise.shape, traces.shape)


if __name__ == "__main__":
    unittest.main()
