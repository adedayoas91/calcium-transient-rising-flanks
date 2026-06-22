import unittest

import numpy as np

from calcium_transient_rising_flank.estimators import (
    CausalGranger,
    _load_gcstar_class,
    selected_frame_indices,
)
from core.rising_flanks import RisingFlanks


class EstimatorTests(unittest.TestCase):
    def test_supplied_rising_flanks_executes_on_selected_frames(self) -> None:
        rng = np.random.default_rng(3)
        driver = rng.normal(size=500)
        target = np.zeros(500)
        target[1:] = 0.95 * driver[:-1] + rng.normal(scale=0.05, size=499)
        data = np.vstack([driver, target])
        indices = (np.arange(data.shape[1]), np.arange(data.shape[1]))

        fitted = RisingFlanks(
            n_perm=0, n_pasts=1, n_lags=1, f_s=1.0, seg_len=0
        ).fit_rising(data, indices, verbose=0)

        self.assertEqual(fitted.inv_corr_.shape, (4, 2))
        self.assertGreater(fitted.inv_corr_[2, 1], fitted.inv_corr_[3, 0])

    def test_adapter_uses_selected_frame_contract_and_shapes_result(self) -> None:
        rng = np.random.default_rng(12)
        data = rng.normal(size=(3, 120))
        selected = selected_frame_indices(np.maximum(data, 0.0))

        result = CausalGranger(
            max_lag=1, n_surrogates=9, alpha=0.2, random_state=2
        ).fit(data, event_indices=selected)

        self.assertEqual(result.scores.shape, (3, 3))
        self.assertEqual(result.p_values.shape, (3, 3))
        self.assertEqual(result.adjacency.shape, (3, 3))
        self.assertEqual(result.estimator, "rising_flanks_cgc")
        self.assertFalse(np.any(np.diag(result.adjacency)))

    def test_adapter_rejects_unimplemented_segment_extension(self) -> None:
        with self.assertRaises(NotImplementedError):
            CausalGranger(max_lag=1).fit(
                np.ones((2, 20)), segment_ids=np.zeros(20, dtype=int)
            )

    def test_physical_event_mode_preserves_frame_lag_interpretation(self) -> None:
        rng = np.random.default_rng(21)
        driver = rng.normal(size=180)
        target = np.zeros(180)
        target[1:] = driver[:-1] + rng.normal(scale=0.01, size=179)
        data = np.vstack([driver, target])

        result = CausalGranger(max_lag=1, event_mode="physical").fit(data)

        self.assertEqual(result.estimator, "physical_time_cgc")
        self.assertTrue(result.adjacency[0, 1])
        self.assertGreater(result.scores[0, 1], result.scores[1, 0])

    def test_physical_event_mode_does_not_compress_discontinuous_events(self) -> None:
        rng = np.random.default_rng(22)
        data = rng.normal(size=(2, 100))
        selected = (np.array([10, 30, 50, 70]), np.array([10, 30, 50, 70]))

        result = CausalGranger(max_lag=1, event_mode="physical").fit(
            data, event_indices=selected
        )

        self.assertEqual(result.scores[0, 1], 0.0)
        self.assertFalse(result.adjacency[0, 1])

    def test_invalid_event_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CausalGranger(max_lag=1, event_mode="unknown")

    def test_gcstar_engine_returns_graph_result_for_cgc_and_cgc_star(self) -> None:
        rng = np.random.default_rng(23)
        data = rng.normal(size=(3, 90))

        cgc = CausalGranger(
            max_lag=1,
            n_surrogates=0,
            engine="gcstar",
            gcstar_method="cgc",
        ).fit(data)
        cgc_star = CausalGranger(
            max_lag=1,
            n_surrogates=0,
            engine="gcstar",
            gcstar_method="cgc-star",
        ).fit(data)

        self.assertEqual(cgc.estimator, "cgc")
        self.assertEqual(cgc_star.estimator, "cgc_star")
        self.assertEqual(cgc.scores.shape, (3, 3))
        self.assertFalse(np.any(np.diag(cgc_star.adjacency)))

    def test_gcstar_accepts_cgc_star_alias(self) -> None:
        GcStar = _load_gcstar_class()

        estimator = GcStar(n_perm=0, n_pasts=1, n_lags=1, method="cgc-star")

        self.assertEqual(estimator.method, "fcgc")


if __name__ == "__main__":
    unittest.main()
