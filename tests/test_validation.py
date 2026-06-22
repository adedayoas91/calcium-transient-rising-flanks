import unittest

import numpy as np

from calcium_transient_rising_flank.estimators import CausalGranger
from calcium_transient_rising_flank.validation import (
    cross_recording_surrogate,
    jitter_event_indices,
    reverse_event_indices,
    run_event_bootstrap_stability,
    run_event_null_controls,
    run_null_controls,
    simulate_calcium_dataset,
    validate_representations,
)


class ValidationTests(unittest.TestCase):
    def test_null_controls_include_shifted_and_reverse_time_graphs(self) -> None:
        rng = np.random.default_rng(7)
        traces = rng.normal(size=(3, 80))
        estimator = CausalGranger(max_lag=1)

        controls = run_null_controls(
            traces, estimator=estimator, n_surrogates=3, random_state=2
        )

        self.assertEqual(len(controls.shifted), 3)
        self.assertEqual(controls.observed.scores.shape, (3, 3))
        self.assertEqual(controls.reverse_time.scores.shape, (3, 3))

    def test_reverse_time_control_reverses_declared_segments(self) -> None:
        class RecordingEstimator:
            max_lag = 1

            def fit(self, traces: np.ndarray, segment_ids: np.ndarray | None = None):
                return None if segment_ids is None else segment_ids.copy()

        segments = np.array([0, 0, 1, 1, 2, 2])
        controls = run_null_controls(
            np.ones((2, len(segments))),
            estimator=RecordingEstimator(),
            n_surrogates=1,
            segment_ids=segments,
        )

        np.testing.assert_array_equal(controls.reverse_time, segments[::-1])

    def test_event_null_controls_include_jitter_phase_and_reverse_time(self) -> None:
        rng = np.random.default_rng(10)
        traces = rng.normal(size=(3, 90))
        rise = tuple(np.arange(10, 80, 5) for _ in range(3))
        fall = tuple(np.arange(12, 82, 5) for _ in range(3))

        controls = run_event_null_controls(
            traces,
            estimator=CausalGranger(max_lag=1, event_mode="physical"),
            event_indices=rise,
            alternative_event_indices=fall,
            n_jittered=2,
            n_phase_permutations=2,
            random_state=4,
        )

        self.assertEqual(len(controls.jittered), 2)
        self.assertEqual(len(controls.phase_permuted), 2)
        self.assertEqual(controls.observed.scores.shape, (3, 3))
        self.assertEqual(controls.reverse_time.scores.shape, (3, 3))

    def test_event_index_transforms_stay_on_axis(self) -> None:
        indices = (np.array([0, 5, 9]), np.array([2, 8]))

        jittered = jitter_event_indices(indices, n_steps=10, max_jitter=2, random_state=1)
        reversed_indices = reverse_event_indices(indices, n_steps=10)

        self.assertTrue(all(np.all(values >= 0) for values in jittered))
        self.assertTrue(all(np.all(values < 10) for values in jittered))
        np.testing.assert_array_equal(reversed_indices[0], np.array([0, 4, 9]))

    def test_cross_recording_surrogate_matches_reference_shape(self) -> None:
        reference = np.ones((4, 20))
        donor = np.arange(30, dtype=float).reshape(3, 10)

        surrogate = cross_recording_surrogate(reference, donor)

        self.assertEqual(surrogate.shape, reference.shape)

    def test_event_bootstrap_stability_returns_graphs_and_score(self) -> None:
        rng = np.random.default_rng(11)
        traces = rng.normal(size=(3, 90))
        indices = tuple(np.arange(10, 80, 4) for _ in range(3))

        result = run_event_bootstrap_stability(
            traces,
            estimator=CausalGranger(max_lag=1, event_mode="physical"),
            event_indices=indices,
            n_bootstrap=3,
            random_state=5,
        )

        self.assertEqual(len(result.graphs), 3)
        self.assertGreaterEqual(result.stability, 0.0)

    def test_synthetic_validation_scores_all_comparator_representations(self) -> None:
        truth = np.array(
            [[False, True, False], [False, False, True], [False, False, False]]
        )
        dataset = simulate_calcium_dataset(
            truth, n_steps=250, gamma=0.8, noise_std=0.02, random_state=4
        )

        validation = validate_representations(
            dataset,
            estimator=CausalGranger(max_lag=1),
            gamma=0.8,
        )

        self.assertEqual(
            set(validation.graphs),
            {"full", "deconvolved", "rise", "fall", "fall_residual"},
        )
        self.assertEqual(set(validation.recovery), set(validation.graphs))


if __name__ == "__main__":
    unittest.main()
