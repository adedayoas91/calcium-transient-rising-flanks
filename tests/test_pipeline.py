import unittest

import numpy as np

from calcium_transient_rising_flank.pipeline import AnalysisConfig, run_pipeline
from calcium_transient_rising_flank.preprocessing import build_scenarios
from calcium_transient_rising_flank.validation import simulate_calcium_dataset


class PipelineTests(unittest.TestCase):
    def test_preprocessing_scenarios_apply_declared_operations(self) -> None:
        traces = np.array(
            [[0.0, 1.0, 20.0, 3.0], [0.0, 1.0, 20.0, 3.0], [1.0, 1.0, 1.0, 1.0]]
        )

        scenarios = build_scenarios(
            traces, bad_neurons=[2], artifact_frames=[2], smoothing_window=3
        )

        self.assertEqual(set(scenarios), {"A", "B", "C", "D"})
        self.assertEqual(scenarios["A"].traces.shape[0], 3)
        self.assertEqual(scenarios["B"].traces.shape[0], 2)
        self.assertLess(scenarios["C"].traces[0, 2], scenarios["B"].traces[0, 2])

    def test_synthetic_dataset_and_pipeline_execute_end_to_end(self) -> None:
        truth = np.array([[False, True, False], [False, False, True], [False, False, False]])
        dataset = simulate_calcium_dataset(
            truth, n_steps=300, gamma=0.75, noise_std=0.02, random_state=5
        )
        config = AnalysisConfig(
            max_lag=1,
            n_surrogates=0,
            tolerance=0.0,
            gamma=0.75,
            smoothing_window=3,
        )

        result = run_pipeline(
            dataset.fluorescence,
            sides=["L", "L", "R"],
            positions=[0, 1, 0],
            config=config,
        )

        self.assertEqual(set(result.scenarios), {"A", "B", "C", "D"})
        self.assertIn("rise", result.scenarios["A"].cgc)
        self.assertIsNotNone(result.scenarios["A"].w_ic["rise"].value)

    def test_pipeline_can_use_physical_time_event_mode(self) -> None:
        traces = np.vstack(
            [
                np.linspace(0, 1, 80),
                np.roll(np.linspace(0, 1, 80), 1),
                np.linspace(1, 0, 80),
            ]
        )

        result = run_pipeline(
            traces,
            sides=["L", "L", "R"],
            positions=[0, 1, 0],
            config=AnalysisConfig(max_lag=1, event_mode="physical"),
        )

        self.assertEqual(result.config.event_mode, "physical")
        self.assertIn("full", result.scenarios["A"].cgc)
        self.assertEqual(result.scenarios["A"].cgc["rise"].estimator, "physical_time_cgc")

    def test_roi_specific_parameters_follow_declared_neuron_exclusion(self) -> None:
        traces = np.array(
            [
                [0.0, 1.0, 0.8, 0.64, 0.4, 0.3],
                [0.0, 0.8, 0.6, 0.4, 0.3, 0.2],
                [0.0, 1.0, 0.9, 0.8, 0.7, 0.6],
                [0.0, 0.4, 0.3, 0.2, 0.1, 0.0],
            ]
        )
        config = AnalysisConfig(
            bad_neurons=(3,),
            gamma=np.array([0.8, 0.7, 0.9, 0.6]),
            tolerance=np.array([0.0, 0.0, 0.0, 0.0]),
            smoothing_window=1,
        )

        result = run_pipeline(traces, sides=["L", "L", "R", "R"], config=config)

        np.testing.assert_allclose(
            result.scenarios["B"].representations.gamma, [0.8, 0.7, 0.9]
        )

    def test_declared_segment_ids_are_rejected_until_supported_by_core(self) -> None:
        traces = np.array(
            [
                [0.0, 1.0, 0.8, 0.0, 1.0, 0.8, 0.0, 1.0, 0.8],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 0.5, 0.2, 0.0, 0.4, 0.2, 0.0, 0.5, 0.2],
            ]
        )
        segments = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        config = AnalysisConfig(
            max_lag=1,
            smoothing_window=1,
            segment_ids_by_representation={"rise": segments, "fall": segments},
        )

        with self.assertRaises(NotImplementedError):
            run_pipeline(traces, sides=["L", "L", "R"], config=config)


if __name__ == "__main__":
    unittest.main()
