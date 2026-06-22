import unittest

import numpy as np

from calcium_transient_rising_flank.metrics import (
    delta_w_ic,
    edge_recovery,
    graph_stability,
    w_ic,
    w_rc,
)


class MetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sides = np.array(["L", "L", "R", "R"])
        self.position = np.array([0, 1, 0, 1])

    def test_weighted_w_ic_normalizes_available_pair_opportunities(self) -> None:
        scores = np.zeros((4, 4))
        scores[0, 1] = scores[1, 0] = 1.0
        scores[2, 3] = scores[3, 2] = 1.0
        scores[0, 2] = scores[1, 3] = scores[2, 0] = scores[3, 1] = 1.0

        metric = w_ic(scores, self.sides)

        self.assertAlmostEqual(metric.value, 2 / 3)
        self.assertEqual(metric.ipsilateral_edges, 4)
        self.assertEqual(metric.cross_side_edges, 4)
        self.assertEqual(metric.preferred_density, 1.0)
        self.assertEqual(metric.alternative_density, 0.5)

    def test_binary_w_ic_and_paired_delta(self) -> None:
        rise = np.zeros((4, 4), dtype=bool)
        rise[0, 1] = rise[2, 3] = True
        fall = rise.copy()
        fall[0, 2] = fall[2, 0] = True

        self.assertEqual(w_ic(rise, self.sides, binary=True).value, 1.0)
        self.assertGreater(delta_w_ic(rise, fall, self.sides, binary=True), 0.0)

    def test_w_rc_detects_rostral_to_caudal_orientation(self) -> None:
        scores = np.zeros((4, 4))
        scores[0, 1] = scores[2, 3] = 1.0

        metric = w_rc(scores, self.sides, self.position)

        self.assertEqual(metric.value, 1.0)

    def test_recovery_and_stability(self) -> None:
        truth = np.array([[False, True], [False, False]])
        recovered = truth.copy()

        summary = edge_recovery(truth, recovered)

        self.assertEqual(summary.precision, 1.0)
        self.assertEqual(summary.recall, 1.0)
        self.assertEqual(graph_stability([truth, recovered]), 1.0)

    def test_orientation_accuracy_excludes_unrelated_false_positive_pairs(self) -> None:
        truth = np.zeros((3, 3), dtype=bool)
        truth[0, 1] = True
        recovered = truth.copy()
        recovered[0, 2] = True

        summary = edge_recovery(truth, recovered)

        self.assertEqual(summary.precision, 0.5)
        self.assertEqual(summary.orientation_accuracy, 1.0)

    def test_legacy_ratio_helper_maps_to_w_ic(self) -> None:
        from others_funcs import get_ratio_from_GC

        scores = np.zeros((4, 4))
        scores[0, 1] = scores[2, 3] = 1.0

        self.assertEqual(get_ratio_from_GC(scores, mid=2, ratio_type="ipsi"), 1.0)


if __name__ == "__main__":
    unittest.main()
