"""Tests for the calibration/onset runner."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "examples" / "calibration_onset.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("calibration_onset", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load calibration/onset script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CalibrationOnsetScriptTests(unittest.TestCase):
    def test_first_crossings_returns_minus_one_for_null_rows(self) -> None:
        script = _load_script_module()
        signal = np.array([[0.0, 0.1, 0.4], [0.0, 0.1, 0.2]])
        result = script._first_crossings(signal, np.array([0.3, 0.5]))
        np.testing.assert_array_equal(result, np.array([2, -1]))

    def test_onset_metrics_distinguish_missed_and_null_calls(self) -> None:
        script = _load_script_module()
        row = script._onset_metric_row(
            seed=1,
            calibration_method="first_difference_mad",
            detector="example",
            predicted=np.array([10, -1, 5, 8]),
            truth=np.array([11, 12, -1, -1]),
        )
        self.assertEqual(row["true_positives"], 1)
        self.assertEqual(row["false_positives"], 2)
        self.assertEqual(row["false_negatives"], 1)
        self.assertEqual(row["within_2_frames"], 1.0)

    def test_synthetic_onset_truth_reserves_two_null_rois(self) -> None:
        script = _load_script_module()
        traces, onsets, calibration_frames = script.simulate_onset_traces(seed=2)
        self.assertEqual(traces.shape, (8, 180))
        self.assertEqual(np.count_nonzero(onsets < 0), 2)
        self.assertTrue(np.all(onsets[onsets >= 0] > calibration_frames))

    def test_onset_configuration_controls_length_and_split(self) -> None:
        script = _load_script_module()
        traces, _, calibration_frames = script.simulate_onset_traces(
            seed=2,
            n_timepoints=240,
            calibration_fraction=0.4,
        )
        self.assertEqual(traces.shape, (8, 240))
        self.assertEqual(calibration_frames, 96)

    def test_resume_completion_is_derived_from_full_partial_rows(self) -> None:
        script = _load_script_module()
        complete = [
            {
                "seed": 1,
                "calibration_method": method,
                "detector": detector,
            }
            for method in script.CALIBRATION_METHODS
            for detector in script.ONSET_DETECTORS
        ]
        incomplete = complete[:-1]

        self.assertEqual(script._complete_onset_seeds(complete), {1})
        self.assertEqual(script._complete_onset_seeds(incomplete), set())


if __name__ == "__main__":
    unittest.main()
