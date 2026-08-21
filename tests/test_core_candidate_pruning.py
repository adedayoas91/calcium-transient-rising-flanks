import sys
import unittest

import numpy as np

from calcium_transient_rising_flank.estimators import _load_gcstar_class


class CoreCandidatePruningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.GcStar = _load_gcstar_class()
        self.module = sys.modules[self.GcStar.__module__]

    def _driver_target_data(self) -> np.ndarray:
        rng = np.random.default_rng(2026)
        driver = rng.normal(size=96)
        target = np.zeros(96, dtype=float)
        target[1:] = 0.95 * driver[:-1] + rng.normal(scale=0.01, size=95)
        return np.vstack([driver, target])

    @staticmethod
    def _eligible_pairs() -> np.ndarray:
        return np.array([[False, True], [False, False]], dtype=bool)

    def _run_with_counted_work(self, fit_call):
        counts = {"perm": 0, "regression": 0}
        original_perm = self.module._perm_test_numba
        original_regression = self.module.regression_residual

        def counting_perm(x, y, n_perm):
            counts["perm"] += 1
            return original_perm(x, y, n_perm)

        def counting_regression(x, z):
            counts["regression"] += 1
            return original_regression(x, z)

        self.module._perm_test_numba = counting_perm
        self.module.regression_residual = counting_regression
        try:
            estimator = fit_call()
        finally:
            self.module._perm_test_numba = original_perm
            self.module.regression_residual = original_regression
        return estimator, counts

    def _assert_mask_pruning_outcome(self, estimator) -> None:
        self.assertEqual(estimator.corr_.shape, (4, 2))
        self.assertEqual(estimator.inv_corr_.shape, (4, 2))
        self.assertGreater(estimator.inv_corr_[2, 1], 0.9)
        self.assertEqual(estimator.corr_[3, 0], 0.0)
        self.assertEqual(estimator.pVal_corr_[3, 0], 1.0)
        self.assertEqual(estimator.inv_corr_[3, 0], 0.0)
        self.assertEqual(estimator.pVal_inv_corr_[3, 0], 1.0)

        diagnostics = estimator.pair_diagnostics_
        self.assertEqual(diagnostics["attempted"], 8)
        self.assertEqual(diagnostics["completed"], 2)
        self.assertEqual(diagnostics["mask_skipped"], 6)
        self.assertEqual(diagnostics["insufficient_sample"], 0)
        self.assertEqual(
            diagnostics["per_lag"],
            [
                {
                    "lag": 0,
                    "attempted": 4,
                    "completed": 1,
                    "mask_skipped": 3,
                    "insufficient_sample": 0,
                },
                {
                    "lag": 1,
                    "attempted": 4,
                    "completed": 1,
                    "mask_skipped": 3,
                    "insufficient_sample": 0,
                },
            ],
        )

    def test_full_fit_skips_masked_pairs_before_core_work(self) -> None:
        data = self._driver_target_data()
        eligible_pairs = self._eligible_pairs()

        estimator, counts = self._run_with_counted_work(
            lambda: self.GcStar(n_perm=3, n_pasts=1, n_lags=1, method="cgc").fit(
                data,
                eligible_pairs=eligible_pairs,
                verbose=0,
            )
        )

        self.assertEqual(counts, {"perm": 4, "regression": 4})
        self._assert_mask_pruning_outcome(estimator)

    def test_compressed_event_fit_skips_masked_pairs_before_core_work(self) -> None:
        data = self._driver_target_data()
        event_indices = tuple(np.arange(data.shape[1], dtype=int) for _ in range(2))
        eligible_pairs = self._eligible_pairs()

        estimator, counts = self._run_with_counted_work(
            lambda: self.GcStar(n_perm=3, n_pasts=1, n_lags=1, method="cgc").fit_event_compressed(
                data,
                event_indices,
                eligible_pairs=eligible_pairs,
                verbose=0,
            )
        )

        self.assertEqual(counts, {"perm": 4, "regression": 4})
        self._assert_mask_pruning_outcome(estimator)

    def test_physical_event_fit_skips_masked_pairs_before_core_work(self) -> None:
        data = self._driver_target_data()
        event_indices = tuple(np.arange(data.shape[1], dtype=int) for _ in range(2))
        eligible_pairs = self._eligible_pairs()

        estimator, counts = self._run_with_counted_work(
            lambda: self.GcStar(n_perm=3, n_pasts=1, n_lags=1, method="cgc").fit_event_physical(
                data,
                event_indices=event_indices,
                eligible_pairs=eligible_pairs,
                verbose=0,
            )
        )

        self.assertEqual(counts, {"perm": 4, "regression": 4})
        self._assert_mask_pruning_outcome(estimator)


if __name__ == "__main__":
    unittest.main()
