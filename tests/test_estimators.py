import unittest

import numpy as np

from calcium_transient_rising_flank.estimators import (
    CausalisedGC,
    _load_gcstar_class,
    finite_sample_permutation_p_values,
    extract_rise_flank_runs,
    rise_flank_candidate_pairs,
    selected_frame_indices,
    weighted_benjamini_hochberg,
)
from core.rising_flanks import RisingFlanks


class EstimatorTests(unittest.TestCase):
    def test_rise_flank_runs_drop_short_segments_and_match_shifted_candidates(
        self,
    ) -> None:
        representation = np.zeros((2, 150), dtype=float)
        neuron_1_runs = (
            np.arange(2, 13),
            np.arange(23, 26),
            np.arange(57, 69),
            np.arange(89, 93),
            np.arange(120, 135),
        )
        neuron_2_runs = (
            np.arange(4, 15),
            np.arange(22, 26),
            np.arange(59, 72),
            np.arange(88, 92),
            np.arange(123, 137),
        )
        for run in neuron_1_runs:
            representation[0, run] = 1.0
        for run in neuron_2_runs:
            representation[1, run] = 1.0

        runs = extract_rise_flank_runs(representation, min_run_samples=7)
        result = rise_flank_candidate_pairs(
            representation,
            min_run_samples=7,
            max_lag=4,
            min_overlap_samples=7,
            min_overlap_fraction=0.6,
        )

        self.assertEqual(
            [run.tolist() for run in runs.dropped_runs[0]],
            [[23, 24, 25], [89, 90, 91, 92]],
        )
        self.assertEqual(
            [run.tolist() for run in runs.dropped_runs[1]],
            [[22, 23, 24, 25], [88, 89, 90, 91]],
        )
        self.assertTrue(result.candidate_adjacency[0, 1])
        self.assertFalse(result.candidate_adjacency[1, 0])
        self.assertEqual(result.best_lags[0, 1], 2)
        expanded = runs.expanded_event_indices(150, context_samples=5)
        np.testing.assert_array_equal(
            expanded[0][-25:],
            np.arange(115, 140),
        )
        first_match = next(
            match
            for match in result.matches
            if match.source == 0 and match.target == 1 and match.source_run == 0
        )
        self.assertEqual(first_match.lag, 2)
        self.assertEqual(first_match.overlap_count, 11)

    def test_compressed_event_mode_matches_old_rising_flank_logic(self) -> None:
        rng = np.random.default_rng(3)
        driver = rng.normal(size=160)
        target = np.zeros(160)
        target[1:] = 0.95 * driver[:-1] + rng.normal(scale=0.05, size=159)
        data = np.vstack([driver, target])
        indices = (np.arange(data.shape[1]), np.arange(data.shape[1]))

        old = RisingFlanks(n_perm=0, n_pasts=1, n_lags=1, f_s=1.0, seg_len=0)
        old.fit_rising(data, indices, verbose=0)
        new = _load_gcstar_class()(n_perm=0, n_pasts=1, n_lags=1, method="cgc")
        new.fit_event_compressed(data, indices, verbose=0)

        np.testing.assert_allclose(new.inv_corr_, old.inv_corr_)
        np.testing.assert_allclose(new.pVal_inv_corr_, old.pVal_inv_corr_)

    def test_compressed_event_mode_matches_old_logic_with_multilag_events(
        self,
    ) -> None:
        rng = np.random.default_rng(4)
        data = rng.normal(size=(3, 180))
        data[1, 1:] += 0.7 * data[0, :-1]
        data[2, 2:] += 0.4 * data[1, :-2]
        indices = (
            np.arange(0, data.shape[1], 2),
            np.arange(0, data.shape[1], 3),
            np.arange(0, data.shape[1], 5),
        )

        old = RisingFlanks(n_perm=0, n_pasts=2, n_lags=2, f_s=1.0, seg_len=0)
        old.fit_rising(data, indices, verbose=0)
        new = _load_gcstar_class()(n_perm=0, n_pasts=2, n_lags=2, method="cgc")
        new.fit_event_compressed(data, indices, verbose=0)
        off_diagonal = np.tile(~np.eye(data.shape[0], dtype=bool), (3, 1))

        np.testing.assert_allclose(
            new.inv_corr_[off_diagonal], old.inv_corr_[off_diagonal]
        )
        np.testing.assert_allclose(
            new.pVal_inv_corr_[off_diagonal], old.pVal_inv_corr_[off_diagonal]
        )

    def test_adapter_uses_modified_cgc_on_selected_frames(self) -> None:
        rng = np.random.default_rng(12)
        data = rng.normal(size=(3, 120))
        selected = selected_frame_indices(np.maximum(data, 0.0))

        result = CausalisedGC(
            max_lag=1,
            n_surrogates=0,
            event_mode="compressed",
            method="cgc",
        ).fit(data, event_indices=selected)

        self.assertEqual(result.scores.shape, (3, 3))
        self.assertEqual(result.p_values.shape, (3, 3))
        self.assertEqual(result.adjacency.shape, (3, 3))
        self.assertEqual(result.estimator, "cgc")
        self.assertFalse(np.any(np.diag(result.adjacency)))

    def test_candidate_adjacency_masks_graph_output(self) -> None:
        rng = np.random.default_rng(13)
        driver = rng.normal(size=120)
        target = np.zeros(120)
        target[1:] = driver[:-1] + rng.normal(scale=0.01, size=119)
        data = np.vstack([driver, target])
        candidate_adjacency = np.array(
            [[False, True], [False, False]],
            dtype=bool,
        )

        result = CausalisedGC(
            max_lag=1,
            n_surrogates=0,
            score_threshold=0.0,
        ).fit(data, candidate_adjacency=candidate_adjacency)

        self.assertTrue(result.adjacency[0, 1])
        self.assertFalse(result.adjacency[1, 0])
        self.assertEqual(result.scores[1, 0], 0.0)
        self.assertEqual(result.p_values[1, 0], 1.0)
        np.testing.assert_array_equal(result.candidate_adjacency, candidate_adjacency)

    def test_weighted_bh_prioritizes_without_removing_hypotheses(self) -> None:
        p_values = np.array(
            [
                [1.0, 0.03, 1.0],
                [0.03, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        weights = np.ones_like(p_values)
        weights[0, 1] = 9.0
        weights[1, 0] = 1.0

        discoveries = weighted_benjamini_hochberg(
            p_values,
            weights,
            alpha=0.05,
        )

        self.assertTrue(discoveries[0, 1])
        self.assertFalse(discoveries[1, 0])
        self.assertFalse(np.any(np.diag(discoveries)))

    def test_weighted_bh_rejects_nonpositive_weights(self) -> None:
        p_values = np.full((2, 2), 0.5)
        weights = np.ones((2, 2))
        weights[0, 1] = 0.0

        with self.assertRaisesRegex(ValueError, "positive"):
            weighted_benjamini_hochberg(p_values, weights, alpha=0.05)

    def test_finite_sample_permutation_correction_prevents_zero_p_values(
        self,
    ) -> None:
        corrected = finite_sample_permutation_p_values(
            np.array([[0.0, 1.0 / 19.0], [10.0 / 19.0, 1.0]]),
            n_surrogates=19,
        )

        self.assertEqual(corrected[0, 0], 0.05)
        self.assertEqual(corrected[0, 1], 0.1)
        self.assertEqual(corrected[1, 1], 1.0)

    def test_fixed_tau_is_separate_from_history_depth(self) -> None:
        rng = np.random.default_rng(14)
        data = rng.normal(size=(3, 120))

        result = CausalisedGC(
            max_lag=1,
            tau=2,
            n_pasts=5,
            n_surrogates=0,
        ).fit(data)

        off_diagonal = ~np.eye(data.shape[0], dtype=bool)
        self.assertEqual(result.best_lags[off_diagonal].tolist(), [2] * 6)

    def test_physical_event_mode_preserves_frame_lag_interpretation(self) -> None:
        rng = np.random.default_rng(21)
        driver = rng.normal(size=180)
        target = np.zeros(180)
        target[1:] = driver[:-1] + rng.normal(scale=0.01, size=179)
        data = np.vstack([driver, target])

        result = CausalisedGC(max_lag=1, event_mode="physical").fit(data)

        self.assertEqual(result.estimator, "cgc")
        self.assertTrue(result.adjacency[0, 1])
        self.assertGreater(result.scores[0, 1], result.scores[1, 0])

    def test_physical_event_mode_does_not_compress_discontinuous_events(self) -> None:
        rng = np.random.default_rng(22)
        data = rng.normal(size=(2, 100))
        selected = (np.array([10, 30, 50, 70]), np.array([10, 30, 50, 70]))

        result = CausalisedGC(max_lag=1, event_mode="physical").fit(
            data, event_indices=selected
        )

        self.assertEqual(result.estimator, "segment_aware_cgc")
        self.assertEqual(result.scores[0, 1], 0.0)
        self.assertFalse(result.adjacency[0, 1])

    def test_physical_event_mode_rejects_cross_segment_pairs(self) -> None:
        data = np.zeros((2, 50), dtype=float)
        data[0, [10, 30]] = [1.0, 2.0]
        data[1, [11, 31]] = [1.0, 2.0]
        selected = (np.array([10, 30]), np.array([11, 31]))
        estimator = CausalisedGC(
            max_lag=1, event_mode="physical", score_threshold=0.5
        )

        ungated = estimator.fit(data, event_indices=selected)
        segment_ids = np.full(data.shape[1], -1, dtype=int)
        segment_ids[[10, 30]] = [1, 3]
        segment_ids[[11, 31]] = [2, 4]
        gated = estimator.fit(data, segment_ids=segment_ids, event_indices=selected)

        self.assertTrue(ungated.adjacency[0, 1])
        self.assertEqual(gated.scores[0, 1], 0.0)
        self.assertFalse(gated.adjacency[0, 1])

    def test_invalid_event_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CausalisedGC(max_lag=1, event_mode="unknown")

    def test_adapter_returns_cgc_and_cgc_star_graph_results(self) -> None:
        rng = np.random.default_rng(23)
        data = rng.normal(size=(3, 90))

        cgc = CausalisedGC(max_lag=1, n_surrogates=0, method="cgc").fit(data)
        cgc_star = CausalisedGC(
            max_lag=1, n_surrogates=0, method="cgc-star"
        ).fit(data)

        self.assertEqual(cgc.estimator, "cgc")
        self.assertEqual(cgc_star.estimator, "cgc_star")
        self.assertEqual(cgc.scores.shape, (3, 3))
        self.assertFalse(np.any(np.diag(cgc_star.adjacency)))

    def test_cgc_star_alias_is_supported_by_supplied_core(self) -> None:
        GcStar = _load_gcstar_class()

        estimator = GcStar(n_perm=0, n_pasts=1, n_lags=1, method="cgc-star")

        self.assertEqual(estimator.method, "fcgc")


if __name__ == "__main__":
    unittest.main()
