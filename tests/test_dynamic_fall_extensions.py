import unittest

import numpy as np

from calcium_transient_rising_flank.estimators import GraphResult
from calcium_transient_rising_flank.validation import (
    DynamicSimulationConfig,
    simulate_calcium_dataset,
    validate_representations,
)


class RecordingEstimator:
    def fit(self, traces: np.ndarray, **_: object) -> GraphResult:
        n_nodes = traces.shape[0]
        adjacency = np.zeros((n_nodes, n_nodes), dtype=bool)
        if n_nodes > 1:
            adjacency[0, 1] = True
        return GraphResult(
            scores=adjacency.astype(float),
            p_values=np.ones((n_nodes, n_nodes), dtype=float),
            adjacency=adjacency,
            best_lags=np.zeros((n_nodes, n_nodes), dtype=int),
            estimator="recording",
        )


class DynamicFallExtensionTests(unittest.TestCase):
    def test_default_dynamic_regime_keeps_noncausal_fall_truth(self) -> None:
        truth = np.array([[False, True], [False, False]])
        config = DynamicSimulationConfig(
            n_episodes=1,
            min_rise_length=20,
            max_rise_length=20,
            fall_to_rise_ratio_min=2.1,
            fall_to_rise_ratio_max=2.1,
            initial_activation_probability=1.0,
            fall_noise_rate=0.0,
            fall_initial_scale=0.0,
            adjacency_sequence=(truth,),
        )

        dataset = simulate_calcium_dataset(
            truth,
            n_steps=64,
            gamma=0.9,
            noise_std=0.0,
            shared_noise_std=0.0,
            spontaneous_rate=0.0,
            transmission_probability=1.0,
            random_state=5,
            simulator_mode="episodic_dynamic",
            dynamic_config=config,
        )
        validation = validate_representations(dataset, RecordingEstimator(), gamma=0.9)

        episode = dataset.episodes[0]
        fall_slice = slice(episode.fall_start, episode.fall_stop)
        np.testing.assert_array_equal(
            dataset.fall_union_adjacency,
            np.zeros_like(truth, dtype=bool),
        )
        np.testing.assert_array_equal(
            episode.fall_adjacency,
            np.zeros_like(truth, dtype=bool),
        )
        self.assertTrue(np.all(dataset.propagated_events[:, fall_slice] == 0.0))
        np.testing.assert_array_equal(
            validation.truth["fall"],
            np.zeros_like(truth, dtype=bool),
        )
        np.testing.assert_array_equal(
            validation.truth["fall_residual"],
            np.zeros_like(truth, dtype=bool),
        )

    def test_causal_fall_truth_is_stored_and_scored(self) -> None:
        truth = np.array([[False, True], [False, False]])
        config = DynamicSimulationConfig(
            n_episodes=1,
            min_rise_length=20,
            max_rise_length=20,
            rise_waveform_length=4,
            fall_to_rise_ratio_min=2.1,
            fall_to_rise_ratio_max=2.1,
            propagation_delay=1,
            initial_activation_probability=1.0,
            initial_activation_mode="source_nodes",
            fall_state_mode="passive_decay",
            fall_noise_rate=0.0,
            fall_initial_scale=0.0,
            adjacency_sequence=(truth,),
            fall_adjacency_sequence=(truth,),
            fall_initial_activation_probability=1.0,
            fall_transmission_probability=1.0,
        )

        dataset = simulate_calcium_dataset(
            truth,
            n_steps=64,
            gamma=0.9,
            noise_std=0.0,
            shared_noise_std=0.0,
            spontaneous_rate=0.0,
            transmission_probability=1.0,
            random_state=7,
            simulator_mode="episodic_dynamic",
            dynamic_config=config,
        )
        validation = validate_representations(dataset, RecordingEstimator(), gamma=0.9)

        episode = dataset.episodes[0]
        fall_slice = slice(episode.fall_start, episode.fall_stop)
        np.testing.assert_array_equal(dataset.fall_union_adjacency, truth)
        np.testing.assert_array_equal(episode.fall_adjacency, truth)
        self.assertGreater(dataset.propagated_events[:, fall_slice].sum(), 0.0)
        propagated_target_frames = np.flatnonzero(
            dataset.propagated_events[1, fall_slice] > 0.0
        )
        self.assertGreater(propagated_target_frames.size, 0)
        first_target = episode.fall_start + int(propagated_target_frames[0])
        self.assertLess(
            dataset.calcium[1, first_target],
            0.9 * dataset.calcium[1, first_target - 1],
        )
        np.testing.assert_array_equal(validation.truth["fall"], truth)
        np.testing.assert_array_equal(validation.truth["fall_residual"], truth)
        self.assertEqual(validation.recovery["fall"].true_positives, 1)
        self.assertEqual(validation.recovery["fall_residual"].true_positives, 1)

    def test_fall_overlap_exclusion_spaces_repeated_truth_events(self) -> None:
        truth = np.array([[False, True], [False, False]])
        config = DynamicSimulationConfig(
            n_episodes=1,
            min_rise_length=20,
            max_rise_length=20,
            fall_to_rise_ratio_min=2.1,
            fall_to_rise_ratio_max=2.1,
            propagation_delay=1,
            initial_activation_probability=1.0,
            initial_activation_mode="source_nodes",
            fall_state_mode="passive_decay",
            fall_noise_rate=0.0,
            fall_initial_scale=0.0,
            adjacency_sequence=(truth,),
            fall_adjacency_sequence=(truth,),
            fall_initial_activation_probability=1.0,
            fall_spontaneous_rate=1.0,
            fall_transmission_probability=1.0,
            fall_overlap_exclusion_samples=2,
        )

        dataset = simulate_calcium_dataset(
            truth,
            n_steps=64,
            gamma=0.9,
            noise_std=0.0,
            shared_noise_std=0.0,
            spontaneous_rate=0.0,
            transmission_probability=1.0,
            random_state=11,
            simulator_mode="episodic_dynamic",
            dynamic_config=config,
        )

        episode = dataset.episodes[0]
        fall_events = dataset.events[:, episode.fall_start : episode.fall_stop]
        for roi_events in fall_events:
            frames = np.flatnonzero(roi_events > 0.0)
            if frames.size > 1:
                self.assertTrue(np.all(np.diff(frames) > 2))


if __name__ == "__main__":
    unittest.main()
