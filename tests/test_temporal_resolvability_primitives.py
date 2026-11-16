import unittest

import numpy as np

from calcium_transient_rising_flank import (
    DynamicSimulationConfig,
    build_temporal_prior,
    simulate_calcium_dataset,
    temporal_prior_state_masks,
)


class TemporalResolvabilityPrimitiveTests(unittest.TestCase):
    def test_state_masks_partition_every_unordered_pair(self) -> None:
        rise = np.zeros((3, 30), dtype=float)
        segments = np.repeat(np.arange(3), 10)
        for offset in (0, 10, 20):
            rise[0, offset + 1 : offset + 3] = 1.0
            rise[1, offset + 3 : offset + 5] = 1.0
        rise[2, 8:10] = 1.0

        prior = build_temporal_prior(
            rise,
            segments,
            min_run_samples=2,
            max_onset_lag=3,
            timing_deadband=0,
            minimum_decisive_support=2,
            consistency_threshold=0.75,
            beta_prior_concentration=1.0,
        )
        states = temporal_prior_state_masks(prior)

        self.assertTrue(states.directional[0, 1])
        self.assertFalse(states.directional[1, 0])
        self.assertTrue(states.unmatched[0, 2])
        self.assertTrue(states.unmatched[2, 0])
        self.assertFalse(np.any(np.diag(states.directional)))
        partition = (
            states.directional
            | states.directional.T
            | states.ambiguous
            | states.unmatched
        )
        np.testing.assert_array_equal(
            partition,
            ~np.eye(3, dtype=bool),
        )

    def test_source_node_activation_mode_excludes_downstream_initial_events(
        self,
    ) -> None:
        truth = np.array(
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
            rise_waveform_length=4,
            fall_to_rise_ratio_min=2.1,
            fall_to_rise_ratio_max=2.1,
            propagation_delay=2,
            initial_activation_probability=1.0,
            initial_activation_mode="source_nodes",
            edge_dropout_probability=0.0,
            fall_noise_rate=0.0,
            adjacency_sequence=(truth,),
        )

        dataset = simulate_calcium_dataset(
            truth,
            n_steps=130,
            gamma=0.9,
            noise_std=0.0,
            spontaneous_rate=0.0,
            transmission_probability=1.0,
            random_state=3,
            simulator_mode="episodic_dynamic",
            dynamic_config=config,
        )

        for episode in dataset.episodes:
            np.testing.assert_array_equal(
                dataset.events[:, episode.rise_start],
                np.array([1.0, 0.0, 0.0]),
            )

    def test_invalid_initial_activation_mode_is_rejected(self) -> None:
        truth = np.array([[False, True], [False, False]])
        with self.assertRaisesRegex(ValueError, "initial_activation_mode"):
            simulate_calcium_dataset(
                truth,
                n_steps=70,
                simulator_mode="episodic_dynamic",
                dynamic_config=DynamicSimulationConfig(
                    n_episodes=1,
                    min_rise_length=20,
                    max_rise_length=20,
                    fall_to_rise_ratio_min=2.1,
                    fall_to_rise_ratio_max=2.1,
                    initial_activation_mode="invalid",
                ),
            )


if __name__ == "__main__":
    unittest.main()
