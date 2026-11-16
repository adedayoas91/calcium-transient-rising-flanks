import unittest

import numpy as np

from calcium_transient_rising_flank.temporal_priors import build_temporal_prior


def _set_run(
    representation: np.ndarray,
    roi: int,
    start: int,
    stop: int,
) -> None:
    representation[roi, start:stop] = 1.0


class TemporalPriorTests(unittest.TestCase):
    def test_consistent_forward_support_yields_one_sided_hard_mask(self) -> None:
        rise = np.zeros((2, 30), dtype=float)
        segments = np.repeat(np.arange(3), 10)
        _set_run(rise, 0, 1, 3)
        _set_run(rise, 1, 3, 5)
        _set_run(rise, 0, 11, 13)
        _set_run(rise, 1, 14, 16)
        _set_run(rise, 0, 21, 23)
        _set_run(rise, 1, 24, 26)

        result = build_temporal_prior(
            rise,
            segments,
            min_run_samples=2,
            max_onset_lag=4,
            timing_deadband=0,
            minimum_decisive_support=2,
            consistency_threshold=0.75,
            beta_prior_concentration=1.0,
        )

        self.assertTrue(result.robust_hard_mask[0, 1])
        self.assertFalse(result.robust_hard_mask[1, 0])
        self.assertEqual(result.decisive_episode_counts[0, 1], 3)
        self.assertEqual(result.decisive_episode_counts[1, 0], 0)
        self.assertEqual(result.matched_episode_counts[0, 1], 3)
        self.assertEqual(result.tie_episode_counts[0, 1], 0)
        self.assertAlmostEqual(result.direction_consistency[0, 1], 1.0)
        self.assertAlmostEqual(result.mean_decisive_lag[0, 1], 8.0 / 3.0)
        self.assertAlmostEqual(result.median_decisive_lag[0, 1], 3.0)
        self.assertEqual(result.screen_episode_ids[0][1], (0, 1, 2))
        self.assertAlmostEqual(result.hypothesis_weights[0, 1], 1.6)
        self.assertAlmostEqual(result.hypothesis_weights[1, 0], 0.4)

    def test_reversing_episodes_fall_back_to_bidirectional_hard_mask(self) -> None:
        rise = np.zeros((2, 20), dtype=float)
        segments = np.repeat(np.arange(2), 10)
        _set_run(rise, 0, 1, 3)
        _set_run(rise, 1, 3, 5)
        _set_run(rise, 0, 15, 17)
        _set_run(rise, 1, 12, 14)

        result = build_temporal_prior(
            rise,
            segments,
            min_run_samples=2,
            max_onset_lag=4,
            timing_deadband=0,
            minimum_decisive_support=1,
            consistency_threshold=0.75,
            beta_prior_concentration=1.0,
        )

        self.assertTrue(result.robust_hard_mask[0, 1])
        self.assertTrue(result.robust_hard_mask[1, 0])
        self.assertEqual(result.decisive_episode_counts[0, 1], 1)
        self.assertEqual(result.decisive_episode_counts[1, 0], 1)
        self.assertEqual(result.matched_episode_counts[0, 1], 2)
        self.assertEqual(result.tie_episode_counts[0, 1], 0)
        self.assertAlmostEqual(result.direction_consistency[0, 1], 0.5)
        self.assertEqual(result.screen_episode_ids[0][1], (0, 1))
        self.assertAlmostEqual(result.hypothesis_weights[0, 1], 1.0)
        self.assertAlmostEqual(result.hypothesis_weights[1, 0], 1.0)

    def test_deadband_ties_count_as_supported_but_nondecisive(self) -> None:
        rise = np.zeros((2, 10), dtype=float)
        segments = np.zeros(10, dtype=int)
        _set_run(rise, 0, 1, 3)
        _set_run(rise, 1, 2, 4)

        result = build_temporal_prior(
            rise,
            segments,
            min_run_samples=2,
            max_onset_lag=3,
            timing_deadband=1,
            minimum_decisive_support=1,
            consistency_threshold=0.75,
            beta_prior_concentration=1.0,
        )

        self.assertTrue(result.robust_hard_mask[0, 1])
        self.assertTrue(result.robust_hard_mask[1, 0])
        self.assertEqual(result.matched_episode_counts[0, 1], 1)
        self.assertEqual(result.tie_episode_counts[0, 1], 1)
        self.assertEqual(result.decisive_episode_counts[0, 1], 0)
        self.assertEqual(result.decisive_episode_counts[1, 0], 0)
        self.assertAlmostEqual(result.hypothesis_weights[0, 1], 1.0)
        self.assertAlmostEqual(result.hypothesis_weights[1, 0], 1.0)

    def test_unmatched_pairs_keep_neutral_weights_and_empty_screen_ids(self) -> None:
        rise = np.zeros((2, 12), dtype=float)
        segments = np.zeros(12, dtype=int)
        _set_run(rise, 0, 1, 3)
        _set_run(rise, 1, 8, 10)

        result = build_temporal_prior(
            rise,
            segments,
            min_run_samples=2,
            max_onset_lag=2,
            timing_deadband=0,
            minimum_decisive_support=1,
            consistency_threshold=0.75,
            beta_prior_concentration=1.0,
        )

        self.assertFalse(result.robust_hard_mask[0, 1])
        self.assertFalse(result.robust_hard_mask[1, 0])
        self.assertEqual(result.matched_episode_counts[0, 1], 0)
        self.assertEqual(result.screen_episode_ids[0][1], ())
        self.assertAlmostEqual(result.hypothesis_weights[0, 1], 1.0)
        self.assertAlmostEqual(result.hypothesis_weights[1, 0], 1.0)

    def test_weights_are_positive_normalized_and_monotone_in_directional_support(
        self,
    ) -> None:
        one_episode = np.zeros((2, 10), dtype=float)
        one_segments = np.zeros(10, dtype=int)
        _set_run(one_episode, 0, 1, 3)
        _set_run(one_episode, 1, 3, 5)
        weak = build_temporal_prior(
            one_episode,
            one_segments,
            min_run_samples=2,
            max_onset_lag=4,
            timing_deadband=0,
            minimum_decisive_support=1,
            consistency_threshold=0.75,
            beta_prior_concentration=1.0,
        )

        two_episode = np.zeros((2, 20), dtype=float)
        two_segments = np.repeat(np.arange(2), 10)
        _set_run(two_episode, 0, 1, 3)
        _set_run(two_episode, 1, 3, 5)
        _set_run(two_episode, 0, 11, 13)
        _set_run(two_episode, 1, 14, 16)
        strong = build_temporal_prior(
            two_episode,
            two_segments,
            min_run_samples=2,
            max_onset_lag=4,
            timing_deadband=0,
            minimum_decisive_support=1,
            consistency_threshold=0.75,
            beta_prior_concentration=1.0,
        )

        self.assertGreater(weak.hypothesis_weights[0, 1], 1.0)
        self.assertGreater(strong.hypothesis_weights[0, 1], weak.hypothesis_weights[0, 1])
        self.assertLess(strong.hypothesis_weights[1, 0], weak.hypothesis_weights[1, 0])
        self.assertGreater(np.min(weak.hypothesis_weights[~np.eye(2, dtype=bool)]), 0.0)
        self.assertGreater(
            np.min(strong.hypothesis_weights[~np.eye(2, dtype=bool)]), 0.0
        )
        self.assertAlmostEqual(
            weak.hypothesis_weights[0, 1] + weak.hypothesis_weights[1, 0],
            2.0,
        )
        self.assertAlmostEqual(
            strong.hypothesis_weights[0, 1] + strong.hypothesis_weights[1, 0],
            2.0,
        )

    def test_negative_one_segments_are_excluded_from_episode_screen(self) -> None:
        rise = np.zeros((2, 12), dtype=float)
        segments = np.array([0, 0, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1], dtype=int)
        _set_run(rise, 0, 0, 2)
        _set_run(rise, 1, 2, 4)
        _set_run(rise, 0, 6, 8)
        _set_run(rise, 1, 8, 10)

        result = build_temporal_prior(
            rise,
            segments,
            min_run_samples=2,
            max_onset_lag=3,
            timing_deadband=0,
            minimum_decisive_support=1,
            consistency_threshold=0.75,
            beta_prior_concentration=1.0,
        )

        self.assertEqual(result.matched_episode_counts[0, 1], 1)
        self.assertEqual(result.screen_episode_ids[0][1], (1,))
        self.assertEqual(result.decisive_episode_counts[0, 1], 1)
        self.assertFalse(0 in result.screen_episode_ids[0][1])

    def test_invalid_inputs_are_rejected(self) -> None:
        rise = np.ones((2, 6), dtype=float)
        segments = np.zeros(6, dtype=int)

        with self.assertRaises(ValueError):
            build_temporal_prior(
                -rise,
                segments,
                min_run_samples=2,
                max_onset_lag=2,
                timing_deadband=0,
                minimum_decisive_support=1,
                consistency_threshold=0.75,
                beta_prior_concentration=1.0,
            )
        with self.assertRaises(ValueError):
            build_temporal_prior(
                rise,
                np.array([0, 0, 0, 0, 0.5, 0]),
                min_run_samples=2,
                max_onset_lag=2,
                timing_deadband=0,
                minimum_decisive_support=1,
                consistency_threshold=0.75,
                beta_prior_concentration=1.0,
            )
        with self.assertRaises(ValueError):
            build_temporal_prior(
                rise,
                segments,
                min_run_samples=0,
                max_onset_lag=2,
                timing_deadband=0,
                minimum_decisive_support=1,
                consistency_threshold=0.75,
                beta_prior_concentration=1.0,
            )
        with self.assertRaises(ValueError):
            build_temporal_prior(
                rise,
                segments,
                min_run_samples=2,
                max_onset_lag=2,
                timing_deadband=0,
                minimum_decisive_support=1,
                consistency_threshold=1.5,
                beta_prior_concentration=1.0,
            )
        with self.assertRaises(ValueError):
            build_temporal_prior(
                rise,
                segments,
                min_run_samples=2,
                max_onset_lag=2,
                timing_deadband=0,
                minimum_decisive_support=1,
                consistency_threshold=0.75,
                beta_prior_concentration=0.0,
            )


if __name__ == "__main__":
    unittest.main()
