import unittest

import numpy as np

from calcium_transient_rising_flank.estimators import CausalGranger
from calcium_transient_rising_flank.pipeline import AnalysisConfig
from calcium_transient_rising_flank.robustness import (
    SyntheticCondition,
    downsample_dataset,
    run_analysis_sensitivity,
    run_synthetic_grid,
    split_calibration_evaluation,
)
from calcium_transient_rising_flank.sensitivity import (
    PartialAncestralGraph,
    run_latent_confounding_sensitivity,
)
from calcium_transient_rising_flank.validation import simulate_calcium_dataset


class RobustnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.truth = np.array(
            [[False, True, False], [False, False, True], [False, False, False]]
        )

    def test_downsampling_reduces_observation_rate_without_changing_truth(self) -> None:
        dataset = simulate_calcium_dataset(self.truth, n_steps=40, random_state=1)

        sampled = downsample_dataset(dataset, factor=2)

        np.testing.assert_array_equal(sampled.adjacency, dataset.adjacency)
        self.assertEqual(sampled.fluorescence.shape[1], 20)

    def test_synthetic_grid_can_be_split_before_locked_evaluation(self) -> None:
        runs = run_synthetic_grid(
            self.truth,
            conditions=[
                SyntheticCondition(name="baseline", gamma=0.8),
                SyntheticCondition(name="sampled", gamma=0.8, downsample=2),
            ],
            seeds=[1, 2, 3, 4],
            estimator_factory=lambda: CausalGranger(max_lag=1),
            n_steps=80,
        )

        calibration, evaluation = split_calibration_evaluation(
            runs, calibration_fraction=0.5, random_state=3
        )

        self.assertEqual(len(runs), 8)
        self.assertEqual(len(calibration), 4)
        self.assertEqual(len(evaluation), 4)
        self.assertEqual(
            {run.seed for run in calibration}.intersection(
                run.seed for run in evaluation
            ),
            set(),
        )

    def test_analysis_sensitivity_runs_declared_parameter_settings(self) -> None:
        dataset = simulate_calcium_dataset(self.truth, n_steps=100, random_state=2)
        runs = run_analysis_sensitivity(
            dataset.fluorescence,
            sides=["L", "L", "R"],
            positions=[0, 1, 0],
            configs=[
                AnalysisConfig(max_lag=1, gamma=0.8, smoothing_window=1),
                AnalysisConfig(max_lag=2, gamma=0.8, smoothing_window=3),
            ],
        )

        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[1].config.max_lag, 2)

    def test_latent_confounding_adapter_preserves_external_pag_output(self) -> None:
        class DeclaredPAGModel:
            def fit(self, traces: np.ndarray) -> PartialAncestralGraph:
                return PartialAncestralGraph(
                    endpoint_marks=np.zeros((traces.shape[0], traces.shape[0])),
                    method="declared-test-model",
                    assumptions=("configured externally",),
                )

        pag = run_latent_confounding_sensitivity(
            np.ones((3, 10)), estimator=DeclaredPAGModel()
        )

        self.assertEqual(pag.method, "declared-test-model")


if __name__ == "__main__":
    unittest.main()
