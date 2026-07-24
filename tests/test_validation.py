import unittest

import numpy as np

from calcium_transient_rising_flank.estimators import CausalisedGC
from calcium_transient_rising_flank.estimators import GraphResult
from calcium_transient_rising_flank.validation import (
    DynamicSimulationConfig,
    SyntheticDataset,
    cross_recording_surrogate,
    jitter_event_indices,
    reverse_event_indices,
    run_event_bootstrap_stability,
    run_event_null_controls,
    run_leave_one_neuron_stability,
    run_leave_one_transient_stability,
    run_null_controls,
    run_time_window_stability,
    simulate_calcium_dataset,
    validate_representations,
)


class RecordingEstimator:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, int], tuple[tuple[int, ...], ...] | None]] = []

    def fit(
        self,
        traces: np.ndarray,
        event_indices: tuple[np.ndarray, ...] | list[np.ndarray] | None = None,
        **_: object,
    ) -> GraphResult:
        selected = None
        if event_indices is not None:
            selected = tuple(
                tuple(np.asarray(frames, dtype=int)) for frames in event_indices
            )
        self.calls.append((traces.shape, selected))
        n_nodes = traces.shape[0]
        adjacency = np.zeros((n_nodes, n_nodes), dtype=bool)
        if n_nodes > 1:
            adjacency[0, 1] = True
        return GraphResult(
            scores=adjacency.astype(float),
            p_values=np.ones((n_nodes, n_nodes)),
            adjacency=adjacency,
            best_lags=np.zeros((n_nodes, n_nodes), dtype=int),
            estimator="recording",
        )


class ValidationTests(unittest.TestCase):
    def test_dynamic_simulator_default_fall_state_is_stochastic_independent(
        self,
    ) -> None:
        self.assertEqual(
            DynamicSimulationConfig().fall_state_mode,
            "stochastic_independent",
        )

    def test_null_controls_include_shifted_and_reverse_time_graphs(self) -> None:
        rng = np.random.default_rng(7)
        traces = rng.normal(size=(3, 80))
        estimator = CausalisedGC(max_lag=1)

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
            estimator=CausalisedGC(max_lag=1, event_mode="physical"),
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
            estimator=CausalisedGC(max_lag=1, event_mode="physical"),
            event_indices=indices,
            n_bootstrap=3,
            random_state=5,
        )

        self.assertEqual(len(result.graphs), 3)
        self.assertGreaterEqual(result.stability, 0.0)

    def test_time_window_stability_uses_declared_windows(self) -> None:
        traces = np.ones((3, 12))
        estimator = RecordingEstimator()

        result = run_time_window_stability(
            traces,
            estimator=estimator,
            window_length=5,
            step=3,
        )

        self.assertEqual(len(result.graphs), 3)
        self.assertEqual([shape for shape, _ in estimator.calls], [(3, 5)] * 3)
        self.assertGreaterEqual(result.stability, 0.0)

    def test_time_window_stability_rebases_event_indices(self) -> None:
        traces = np.ones((2, 10))
        estimator = RecordingEstimator()
        event_indices = (np.array([1, 2, 6, 7]), np.array([2, 3, 7, 8]))

        run_time_window_stability(
            traces,
            estimator=estimator,
            window_length=5,
            step=5,
            event_indices=event_indices,
        )

        self.assertEqual(
            estimator.calls[0][1],
            ((1, 2), (2, 3)),
        )
        self.assertEqual(
            estimator.calls[1][1],
            ((1, 2), (2, 3)),
        )

    def test_leave_one_neuron_stability_embeds_subset_graphs(self) -> None:
        traces = np.ones((4, 20))

        result = run_leave_one_neuron_stability(
            traces,
            estimator=RecordingEstimator(),
        )

        self.assertEqual(len(result.graphs), 4)
        self.assertTrue(all(graph.adjacency.shape == (4, 4) for graph in result.graphs))
        for omitted, graph in enumerate(result.graphs):
            self.assertFalse(graph.adjacency[omitted].any())
            self.assertFalse(graph.adjacency[:, omitted].any())

    def test_leave_one_transient_stability_drops_one_contiguous_segment(
        self,
    ) -> None:
        traces = np.ones((2, 12))
        event_indices = (np.array([1, 2, 3, 8, 9]), np.array([4, 5, 10, 11]))
        estimator = RecordingEstimator()

        result = run_leave_one_transient_stability(
            traces,
            estimator=estimator,
            event_indices=event_indices,
        )

        self.assertEqual(len(result.graphs), 4)
        self.assertEqual(estimator.calls[0][1], ((8, 9), (4, 5, 10, 11)))
        self.assertEqual(estimator.calls[1][1], ((1, 2, 3), (4, 5, 10, 11)))
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
            estimator=CausalisedGC(max_lag=1),
            gamma=0.8,
        )

        self.assertEqual(
            set(validation.graphs),
            {"full", "deconvolved", "rise", "fall", "fall_residual"},
        )
        self.assertEqual(set(validation.recovery), set(validation.graphs))

    def test_synthetic_validation_uses_rise_candidate_prefilter(self) -> None:
        traces = np.zeros((3, 40), dtype=float)
        traces[0, 2:12] = np.linspace(0.1, 1.0, 10)
        traces[1, 4:14] = np.linspace(0.1, 1.0, 10)
        traces[2, 20:30] = np.linspace(0.1, 1.0, 10)
        dataset = SyntheticDataset(
            adjacency=np.zeros((3, 3), dtype=bool),
            events=traces,
            calcium=traces,
            fluorescence=traces,
        )

        validation = validate_representations(
            dataset,
            estimator=CausalisedGC(
                max_lag=1,
                tau=2,
                n_pasts=5,
                min_rise_run_samples=7,
                rise_candidate_filter=True,
                rise_match_max_lag=3,
                rise_match_min_overlap_samples=6,
                rise_match_min_overlap_fraction=0.6,
            ),
            gamma=0.0,
        )

        self.assertIsNotNone(validation.rise_flank_candidates)
        candidates = validation.rise_flank_candidates.candidate_adjacency
        self.assertTrue(candidates[0, 1])
        self.assertFalse(candidates[1, 0])
        np.testing.assert_array_equal(
            validation.rise_flank_candidates.event_indices[0],
            np.arange(0, 17),
        )
        self.assertEqual(validation.graphs["rise"].best_lags[0, 1], 2)
        np.testing.assert_array_equal(
            validation.graphs["rise"].candidate_adjacency,
            candidates,
        )

    def test_dynamic_simulator_records_causal_rises_and_noncausal_falls(
        self,
    ) -> None:
        a1 = np.array(
            [
                [False, True, False, False],
                [False, False, True, False],
                [False, False, False, False],
                [False, False, False, False],
            ]
        )
        a2 = np.array(
            [
                [False, False, False, True],
                [False, False, False, False],
                [False, True, False, False],
                [False, False, False, False],
            ]
        )
        config = DynamicSimulationConfig(
            n_episodes=3,
            min_rise_length=20,
            max_rise_length=22,
            fall_to_rise_ratio_min=2.1,
            fall_to_rise_ratio_max=2.2,
            initial_activation_probability=1.0,
            fall_noise_rate=0.5,
            fall_noise_scale=0.01,
            adjacency_sequence=(a1, a2),
        )

        dataset = simulate_calcium_dataset(
            a1,
            n_steps=210,
            gamma=0.92,
            noise_std=0.0,
            shared_noise_std=0.0,
            spontaneous_rate=0.03,
            transmission_probability=1.0,
            random_state=8,
            simulator_mode="episodic_dynamic",
            dynamic_config=config,
        )

        self.assertEqual(len(dataset.episodes), 3)
        self.assertIsNotNone(dataset.phase_labels)
        self.assertIsNotNone(dataset.propagated_events)
        self.assertTrue({1, 2}.issubset(set(dataset.phase_labels)))
        np.testing.assert_array_equal(dataset.episodes[0].adjacency, a1)
        np.testing.assert_array_equal(dataset.episodes[1].adjacency, a2)
        np.testing.assert_array_equal(dataset.episodes[2].adjacency, a1)
        np.testing.assert_array_equal(dataset.union_adjacency, a1 | a2)
        self.assertEqual(len(dataset.rise_adjacencies), 3)
        np.testing.assert_array_equal(dataset.rise_adjacencies[0], a1)
        np.testing.assert_array_equal(dataset.rise_adjacencies[1], a2)
        np.testing.assert_array_equal(
            dataset.edge_presence_counts,
            a1.astype(int) * 2 + a2.astype(int),
        )
        np.testing.assert_array_equal(
            dataset.edge_counts,
            a1.astype(int) * 2 + a2.astype(int),
        )
        np.testing.assert_array_equal(
            dataset.edge_prevalence,
            dataset.edge_presence_counts / len(dataset.episodes),
        )
        np.testing.assert_array_equal(
            dataset.node_presence_counts,
            np.full(a1.shape[0], len(dataset.episodes), dtype=int),
        )
        np.testing.assert_array_equal(
            dataset.node_prevalence,
            np.ones(a1.shape[0], dtype=float),
        )

        for episode in dataset.episodes:
            self.assertGreaterEqual(episode.rise_length, 20)
            self.assertGreater(episode.fall_length, 2 * episode.rise_length)
            self.assertIsNotNone(episode.active_nodes)
            self.assertTrue(np.all(episode.active_nodes))
            self.assertEqual(episode.node_count, a1.shape[0])
            self.assertEqual(episode.edge_count, int(np.count_nonzero(episode.adjacency)))
            fall_slice = slice(episode.fall_start, episode.fall_stop)
            np.testing.assert_array_equal(
                dataset.phase_labels[episode.rise_start : episode.rise_stop],
                np.ones(episode.rise_length, dtype=int),
            )
            np.testing.assert_array_equal(
                dataset.phase_labels[fall_slice],
                np.full(episode.fall_length, 2, dtype=int),
            )
            self.assertTrue(np.all(dataset.propagated_events[:, fall_slice] == 0.0))
            self.assertTrue(
                np.all(np.diff(dataset.calcium[:, fall_slice], axis=1) <= 0.0)
            )

    def test_dynamic_simulator_tracks_active_nodes_per_rise(self) -> None:
        a1 = np.array(
            [
                [False, True, False],
                [False, False, False],
                [False, False, False],
            ]
        )
        a2 = np.array(
            [
                [False, False, False],
                [False, False, True],
                [False, False, False],
            ]
        )
        node_sequence = (
            np.array([True, True, False]),
            np.array([False, True, True]),
        )
        config = DynamicSimulationConfig(
            n_episodes=2,
            min_rise_length=20,
            max_rise_length=20,
            fall_to_rise_ratio_min=2.1,
            fall_to_rise_ratio_max=2.1,
            initial_activation_probability=1.0,
            fall_noise_rate=0.0,
            fall_initial_scale=0.0,
            adjacency_sequence=(a1, a2),
            active_node_sequence=node_sequence,
        )

        dataset = simulate_calcium_dataset(
            a1,
            n_steps=126,
            gamma=0.9,
            noise_std=0.0,
            shared_noise_std=0.0,
            spontaneous_rate=0.0,
            transmission_probability=1.0,
            random_state=9,
            simulator_mode="episodic_dynamic",
            dynamic_config=config,
        )

        np.testing.assert_array_equal(dataset.episodes[0].active_nodes, node_sequence[0])
        np.testing.assert_array_equal(dataset.episodes[1].active_nodes, node_sequence[1])
        np.testing.assert_array_equal(
            dataset.node_presence_counts,
            np.array([1, 2, 1]),
        )
        np.testing.assert_array_equal(
            dataset.node_prevalence,
            np.array([0.5, 1.0, 0.5]),
        )

        for episode in dataset.episodes:
            inactive = ~episode.active_nodes
            rise_slice = slice(episode.rise_start, episode.rise_stop)
            fall_slice = slice(episode.fall_start, episode.fall_stop)
            self.assertTrue(np.all(dataset.events[inactive, rise_slice] == 0.0))
            self.assertTrue(np.all(dataset.events[inactive, fall_slice] == 0.0))
            self.assertTrue(np.all(dataset.calcium[inactive, rise_slice] == 0.0))
            self.assertTrue(np.all(dataset.calcium[inactive, fall_slice] == 0.0))

    def test_dynamic_simulator_generates_source_dropout_and_recruitment(
        self,
    ) -> None:
        base = np.array(
            [
                [False, True, False],
                [False, False, True],
                [False, False, False],
            ]
        )
        config = DynamicSimulationConfig(
            n_episodes=2,
            min_rise_length=20,
            max_rise_length=20,
            fall_to_rise_ratio_min=2.1,
            fall_to_rise_ratio_max=2.1,
            initial_activation_probability=1.0,
            source_dropout_probability=1.0,
            source_recruitment_probability=1.0,
            source_recruitment_edge_probability=1.0,
            fall_noise_rate=0.0,
            fall_initial_scale=0.0,
        )

        dataset = simulate_calcium_dataset(
            base,
            n_steps=126,
            gamma=0.9,
            noise_std=0.0,
            shared_noise_std=0.0,
            spontaneous_rate=0.0,
            transmission_probability=1.0,
            random_state=10,
            simulator_mode="episodic_dynamic",
            dynamic_config=config,
        )

        expected = np.array(
            [
                [False, False, False],
                [False, False, False],
                [True, True, False],
            ]
        )
        self.assertEqual(len(dataset.episode_adjacencies), 2)
        for graph in dataset.episode_adjacencies:
            np.testing.assert_array_equal(graph, expected)
        np.testing.assert_array_equal(dataset.union_adjacency, expected)
        np.testing.assert_array_equal(
            dataset.edge_presence_counts,
            expected.astype(int) * len(dataset.episode_adjacencies),
        )
        np.testing.assert_array_equal(dataset.edge_prevalence, expected.astype(float))

    def test_dynamic_simulator_can_reset_falls_to_independent_stochastic_state(
        self,
    ) -> None:
        truth = np.array([[False, True], [False, False]])
        config = DynamicSimulationConfig(
            n_episodes=1,
            min_rise_length=20,
            max_rise_length=20,
            fall_to_rise_ratio_min=2.1,
            fall_to_rise_ratio_max=2.1,
            initial_activation_probability=1.0,
            fall_noise_rate=0.0,
            fall_state_mode="stochastic_independent",
            fall_initial_scale=0.0,
            adjacency_sequence=(truth,),
        )

        dataset = simulate_calcium_dataset(
            truth,
            n_steps=64,
            gamma=0.9,
            noise_std=0.0,
            shared_noise_std=0.0,
            simulator_mode="episodic_dynamic",
            dynamic_config=config,
        )

        episode = dataset.episodes[0]
        self.assertTrue(np.any(dataset.calcium[:, episode.rise_stop - 1] > 0.0))
        np.testing.assert_array_equal(
            dataset.calcium[:, episode.fall_start : episode.fall_stop],
            np.zeros((truth.shape[0], episode.fall_length)),
        )

    def test_dynamic_simulator_bounded_fall_reset_does_not_jump_upward(self) -> None:
        truth = np.array([[False, True], [False, False]])
        config = DynamicSimulationConfig(
            n_episodes=1,
            min_rise_length=20,
            max_rise_length=20,
            rise_waveform_length=20,
            fall_to_rise_ratio_min=2.1,
            fall_to_rise_ratio_max=2.1,
            initial_activation_probability=1.0,
            fall_noise_rate=1.0,
            fall_noise_scale=0.01,
            fall_state_mode="stochastic_independent",
            fall_initial_scale=100.0,
            fall_initial_ceiling_fraction=1.0,
            adjacency_sequence=(truth,),
        )

        dataset = simulate_calcium_dataset(
            truth,
            n_steps=64,
            gamma=0.9,
            noise_std=0.0,
            shared_noise_std=0.0,
            simulator_mode="episodic_dynamic",
            dynamic_config=config,
        )

        episode = dataset.episodes[0]
        rise_endpoint = dataset.calcium[:, episode.rise_stop - 1]
        fall_start = dataset.calcium[:, episode.fall_start]
        self.assertTrue(np.all(fall_start <= rise_endpoint))
        self.assertTrue(
            np.all(
                np.diff(
                    dataset.calcium[:, episode.fall_start : episode.fall_stop],
                    axis=1,
                )
                <= 0.0
            )
        )

    def test_dynamic_validation_scores_falls_against_zero_truth(self) -> None:
        truth = np.array([[False, True], [False, False]])
        config = DynamicSimulationConfig(
            n_episodes=1,
            min_rise_length=20,
            max_rise_length=20,
            fall_to_rise_ratio_min=2.1,
            fall_to_rise_ratio_max=2.1,
            initial_activation_probability=1.0,
            adjacency_sequence=(truth,),
        )
        dataset = simulate_calcium_dataset(
            truth,
            n_steps=64,
            gamma=0.9,
            noise_std=0.0,
            shared_noise_std=0.0,
            simulator_mode="episodic_dynamic",
            dynamic_config=config,
        )

        validation = validate_representations(dataset, RecordingEstimator(), gamma=0.9)

        np.testing.assert_array_equal(validation.truth["rise"], truth)
        np.testing.assert_array_equal(
            validation.truth["fall"], np.zeros_like(truth, dtype=bool)
        )
        self.assertEqual(validation.recovery["rise"].true_positives, 1)
        self.assertEqual(validation.recovery["fall"].true_positives, 0)
        self.assertEqual(validation.recovery["fall"].false_positives, 1)

    def test_dynamic_simulator_requires_at_least_twenty_rise_samples(self) -> None:
        truth = np.array([[False, True], [False, False]])

        with self.assertRaisesRegex(ValueError, "at least 20 samples"):
            simulate_calcium_dataset(
                truth,
                n_steps=100,
                simulator_mode="episodic_dynamic",
                dynamic_config=DynamicSimulationConfig(min_rise_length=19),
            )

    def test_dynamic_simulator_spreads_rise_onsets_over_configured_samples(
        self,
    ) -> None:
        truth = np.zeros((1, 1), dtype=bool)
        config = DynamicSimulationConfig(
            n_episodes=1,
            min_rise_length=20,
            max_rise_length=20,
            rise_waveform_length=5,
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
            transmission_probability=0.0,
            random_state=3,
            simulator_mode="episodic_dynamic",
            dynamic_config=config,
        )

        episode = dataset.episodes[0]
        rise = dataset.calcium[0, episode.rise_start : episode.rise_stop]
        np.testing.assert_allclose(
            rise[:6],
            np.array([0.2, 0.38, 0.542, 0.6878, 0.81902, 0.737118]),
        )
        self.assertGreater(rise[4], rise[0])
        self.assertLess(rise[5], rise[4])
        self.assertEqual(
            int(
                np.count_nonzero(
                    dataset.events[:, episode.rise_start : episode.rise_stop]
                )
            ),
            1,
        )

    def test_dynamic_simulator_rejects_invalid_rise_waveform_length(self) -> None:
        truth = np.array([[False, True], [False, False]])

        with self.assertRaisesRegex(ValueError, "rise_waveform_length"):
            simulate_calcium_dataset(
                truth,
                n_steps=100,
                simulator_mode="episodic_dynamic",
                dynamic_config=DynamicSimulationConfig(rise_waveform_length=0),
            )

    def test_dynamic_simulator_rejects_invalid_fall_initial_ceiling(self) -> None:
        truth = np.array([[False, True], [False, False]])

        with self.assertRaisesRegex(ValueError, "fall_initial_ceiling_fraction"):
            simulate_calcium_dataset(
                truth,
                n_steps=100,
                simulator_mode="episodic_dynamic",
                dynamic_config=DynamicSimulationConfig(
                    fall_initial_ceiling_fraction=-0.1
                ),
            )

    def test_dynamic_simulator_rejects_unknown_fall_state_mode(self) -> None:
        truth = np.array([[False, True], [False, False]])

        with self.assertRaisesRegex(ValueError, "fall_state_mode"):
            simulate_calcium_dataset(
                truth,
                n_steps=100,
                simulator_mode="episodic_dynamic",
                dynamic_config=DynamicSimulationConfig(
                    min_rise_length=20,
                    fall_state_mode="coupled",
                ),
            )


if __name__ == "__main__":
    unittest.main()
