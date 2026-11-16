import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "examples" / (
        "motorneuron_temporal_screen.py"
    )
    spec = importlib.util.spec_from_file_location(
        "motorneuron_temporal_screen",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MotorneuronTemporalScreenScriptTests(unittest.TestCase):
    def test_resume_reuses_completed_recording_unit_without_rebuilding_prior(self) -> None:
        script = _load_script_module()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_store = script.JsonUnitCheckpointStore(
                Path(directory),
                "motorneuron_temporal_screen",
                {"cases": ["D"], "recordings": "F1T1"},
            )
            checkpoint_store.initialize(resume=False)
            expected_rows = [{"recording": "F1T1", "status": "ok"}]
            checkpoint_store.save_rows("D|F1T1", expected_rows)

            with patch.object(
                script,
                "build_representations",
                side_effect=AssertionError("resume should skip recomputation"),
            ):
                rows = script.run_screen(
                    [
                        {
                            "case": "D",
                            "description": "test",
                            "recording": "F1T1",
                            "fish": 1,
                            "trial": 1,
                            "mid": 1,
                            "traces": np.zeros((3, 96), dtype=float),
                        }
                    ],
                    max_onset_lags=(3,),
                    deadbands=(0,),
                    tolerance=0.0,
                    min_run_samples=2,
                    segment_mode="fixed_windows",
                    window_frames=12,
                    merge_gap_frames=3,
                    minimum_participating_rois=2,
                    n_nulls=2,
                    random_state=10,
                    checkpoint_store=checkpoint_store,
                    resume=True,
                )

        self.assertEqual(rows, expected_rows)

    def test_fixed_windows_and_crossfit_have_disjoint_episode_halves(self) -> None:
        script = _load_script_module()

        segments = script.fixed_window_segments(48, window_frames=12)
        folds = script.crossfit_episode_folds(segments)

        self.assertEqual(tuple(np.unique(segments)), (0, 1, 2, 3))
        self.assertEqual(folds[0], ((0, 2), (1, 3)))
        self.assertEqual(folds[1], ((1, 3), (0, 2)))

    def test_population_bouts_merge_short_gaps_and_require_participants(self) -> None:
        script = _load_script_module()
        rise = np.zeros((3, 20), dtype=float)
        rise[0, 2:4] = 1.0
        rise[1, 5:7] = 1.0
        rise[2, 15:17] = 1.0

        segments = script.population_episode_segments(
            rise,
            merge_gap_frames=1,
            minimum_participating_rois=2,
        )

        np.testing.assert_array_equal(np.unique(segments), [-1, 0])
        np.testing.assert_array_equal(np.flatnonzero(segments == 0), np.arange(2, 7))

    def test_episode_shift_null_is_deterministic_and_preserves_episode_values(self) -> None:
        script = _load_script_module()
        rise = np.arange(24, dtype=float).reshape(2, 12)
        segments = script.fixed_window_segments(12, window_frames=4)

        first = script.episode_shift_null(rise, segments, random_state=7)
        second = script.episode_shift_null(rise, segments, random_state=7)

        np.testing.assert_array_equal(first, second)
        for episode_id in range(3):
            frames = np.flatnonzero(segments == episode_id)
            for roi in range(2):
                np.testing.assert_array_equal(
                    np.sort(first[roi, frames]),
                    np.sort(rise[roi, frames]),
                )

    def test_small_truth_free_screen_writes_auditable_artifacts(self) -> None:
        script = _load_script_module()
        traces = np.zeros((3, 96), dtype=float)
        for start in range(0, 96, 12):
            traces[0, start + 1 : start + 4] = (1.0, 2.0, 3.0)
            traces[1, start + 3 : start + 6] = (1.0, 2.0, 3.0)
            traces[2, start + 5 : start + 8] = (1.0, 2.0, 3.0)
        record = {
            "case": "D",
            "description": "test",
            "recording": "F1T1",
            "fish": 1,
            "trial": 1,
            "mid": 1,
            "traces": traces,
        }

        rows = script.run_screen(
            [record],
            max_onset_lags=(3,),
            deadbands=(0,),
            tolerance=0.0,
            min_run_samples=2,
            segment_mode="fixed_windows",
            window_frames=12,
            merge_gap_frames=3,
            minimum_participating_rois=2,
            n_nulls=2,
            random_state=10,
        )
        summary = script.summarize_rows(rows)

        self.assertEqual(len(rows), 2)
        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(
            rows[0]["directional_pair_fraction"]
            + rows[0]["ambiguous_pair_fraction"]
            + rows[0]["unmatched_pair_fraction"],
            1.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            script.write_outputs(output, rows, summary, {"smoke": True})
            payload = json.loads((output / "summary.json").read_text())

            self.assertTrue((output / "screen_rows.csv").exists())
            self.assertTrue((output / "screen_summary.csv").exists())
            self.assertTrue((output / "report.md").exists())
            self.assertEqual(payload["summary_cell_count"], 1)
            self.assertIn("does not estimate true-edge coverage", (output / "report.md").read_text())

    def test_null_gate_can_be_disabled_without_affecting_other_gates(self) -> None:
        script = _load_script_module()
        row = {
            "case": "D",
            "description": "test",
            "recording": "F1T1",
            "status": "ok",
            "n_rois": 3,
            "n_frames": 96,
            "inferred_episode_count": 8,
            "max_onset_lag_frames": 3,
            "deadband_frames": 0,
            "min_run_samples": 2,
            "segment_mode": "fixed_windows",
            "window_frames": 12,
            "merge_gap_frames": 3,
            "minimum_participating_rois": 2,
            "tolerance": 0.0,
            "n_nulls": 2,
            "directional_pair_fraction": 0.4,
            "ambiguous_pair_fraction": 0.3,
            "unmatched_pair_fraction": 0.3,
            "candidate_density": 0.4,
            "heldout_directional_replication": 0.9,
            "null_directional_pair_fraction_mean": 0.5,
            "null_directional_pair_fraction_p95": 0.6,
            "null_candidate_density_mean": 0.5,
            "null_heldout_directional_replication_mean": 0.5,
            "directional_fraction_above_null_p95": -0.2,
        }

        self.assertFalse(script.summarize_rows([row])[0]["passes_empirical_diagnostics"])
        with patch.dict(
            script.DIAGNOSTIC_GATES,
            {"require_directional_fraction_above_null_p95": False},
        ):
            self.assertTrue(
                script.summarize_rows([row])[0]["passes_empirical_diagnostics"]
            )


if __name__ == "__main__":
    unittest.main()
