import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from calcium_transient_rising_flank import build_temporal_prior


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "examples" / (
        "temporal_resolvability_map.py"
    )
    spec = importlib.util.spec_from_file_location("temporal_resolvability_map", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TemporalResolvabilityScriptTests(unittest.TestCase):
    def test_state_metrics_keep_ambiguity_and_unmatched_separate(self) -> None:
        script = _load_script_module()
        rise = np.zeros((3, 24), dtype=float)
        segments = np.repeat(np.arange(2), 12)
        for offset in (0, 12):
            rise[0, offset + 1 : offset + 3] = 1.0
            rise[1, offset + 3 : offset + 5] = 1.0
        truth = np.array(
            [
                [False, True, False],
                [False, False, False],
                [False, False, False],
            ]
        )
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

        metrics = script.temporal_state_metrics(prior, truth)

        self.assertEqual(metrics["directional_pair_count"], 1)
        self.assertEqual(metrics["ambiguous_pair_count"], 0)
        self.assertEqual(metrics["unmatched_pair_count"], 2)
        self.assertAlmostEqual(
            metrics["directional_pair_fraction"]
            + metrics["ambiguous_pair_fraction"]
            + metrics["unmatched_pair_fraction"],
            1.0,
        )
        self.assertEqual(metrics["conditional_direction_accuracy"], 1.0)
        self.assertEqual(metrics["unconditional_direction_accuracy"], 1.0)
        self.assertEqual(metrics["true_edge_coverage"], 1.0)

    def test_onset_lag_rule_reports_native_to_observed_mapping(self) -> None:
        script = _load_script_module()

        self.assertEqual(script.onset_lag_limit(8, 1), 8)
        self.assertEqual(script.onset_lag_limit(8, 2), 4)
        self.assertEqual(script.onset_lag_limit(8, 4), 2)

    def test_phase_downsampling_preserves_half_open_episode_intervals(self) -> None:
        script = _load_script_module()
        values = np.arange(10, dtype=float)[None, :]
        episode = script.SyntheticEpisode(
            rise_start=1,
            rise_stop=5,
            fall_start=5,
            fall_stop=9,
            adjacency=np.zeros((1, 1), dtype=bool),
        )
        dataset = script.SyntheticDataset(
            adjacency=np.zeros((1, 1), dtype=bool),
            events=values,
            calcium=values,
            fluorescence=values,
            episodes=(episode,),
        )

        phase_zero = script.downsample_with_phase(dataset, 2, 0)
        phase_one = script.downsample_with_phase(dataset, 2, 1)

        np.testing.assert_array_equal(phase_zero.events, values[:, ::2])
        np.testing.assert_array_equal(phase_one.events, values[:, 1::2])
        self.assertEqual(
            (phase_zero.episodes[0].rise_start, phase_zero.episodes[0].rise_stop),
            (1, 3),
        )
        self.assertEqual(
            (phase_one.episodes[0].rise_start, phase_one.episodes[0].rise_stop),
            (0, 2),
        )

    def test_expected_lag_uses_realized_acquisition_phase(self) -> None:
        script = _load_script_module()
        truth = np.array([[False, True], [False, False]])
        events = np.zeros((2, 6), dtype=float)
        events[0, 1] = 1.0
        events[1, 2] = 1.0
        episode = script.SyntheticEpisode(
            rise_start=0,
            rise_stop=4,
            fall_start=4,
            fall_stop=6,
            adjacency=truth,
        )
        dataset = script.SyntheticDataset(
            adjacency=truth,
            events=events,
            calcium=events,
            fluorescence=events,
            episodes=(episode,),
        )

        phase_zero = script.expected_edge_lag_matrix(
            dataset, truth, (0,), factor=2, offset=0
        )
        phase_one = script.expected_edge_lag_matrix(
            dataset, truth, (0,), factor=2, offset=1
        )

        self.assertEqual(phase_zero[0, 1], 0.0)
        self.assertEqual(phase_one[0, 1], 1.0)

    def test_gate_requires_coverage_accuracy_density_and_replication(self) -> None:
        script = _load_script_module()
        base = {
            "regime": "clean_homogeneous",
            "signal_layer": script.PRIMARY_SIGNAL_LAYER,
            "native_delay_frames": 2,
            "downsample": 1,
            "observed_delay_frames": 2.0,
            "deadband_frames": 0,
            "seed": 1,
            "directional_pair_fraction": 0.4,
            "ambiguous_pair_fraction": 0.6,
            "unmatched_pair_fraction": 0.0,
            "directional_candidate_density": 0.2,
            "robust_candidate_density": 0.8,
            "candidate_density": 0.2,
            "true_edge_coverage": 1.0,
            "unconditional_direction_accuracy": 1.0,
            "true_edge_decisive_fraction": 1.0,
            "nonedge_directional_fraction": 0.0,
            "nonedge_exclusion": 1.0,
            "heldout_directional_replication": 1.0,
            "heldout_true_direction_replication": 1.0,
            "soft_true_direction_advantage": 1.0,
            "median_observed_lag_absolute_error": 0.0,
            "estimated_gamma_mean": 0.9,
            "estimated_gamma_sd": 0.0,
            "median_screen_rise_run_samples": 3.0,
            "correct_direction_count": 8,
            "wrong_direction_count": 0,
            "screen_directional_count": 8,
            "replicated_directional_count": 8,
        }

        neighbor = {
            **base,
            "native_delay_frames": 4,
            "observed_delay_frames": 4.0,
        }
        passing = script.apply_adjacency_gate(
            script.summarize_cells(
                [
                    base,
                    neighbor,
                    {**base, "signal_layer": "native_latent_events"},
                ]
            )
        )
        failing = script.apply_adjacency_gate(
            script.summarize_cells(
                [
                    {**base, "true_edge_coverage": 0.5},
                    {**neighbor, "true_edge_coverage": 0.5},
                ]
            )
        )

        primary = [
            cell
            for cell in passing
            if cell["signal_layer"] == script.PRIMARY_SIGNAL_LAYER
        ]
        diagnostic = [
            cell
            for cell in passing
            if cell["signal_layer"] != script.PRIMARY_SIGNAL_LAYER
        ]
        self.assertTrue(all(cell["viable"] for cell in primary))
        self.assertFalse(any(cell["viable"] for cell in diagnostic))
        self.assertFalse(any(cell["viable"] for cell in failing))

    def test_small_grid_is_deterministic_and_writes_auditable_artifacts(self) -> None:
        script = _load_script_module()
        kwargs = {
            "regimes": (script.REGIMES[0],),
            "native_delays": (2,),
            "downsample_factors": (1,),
            "deadbands": (0,),
            "seeds": (1,),
        }

        first = script.run_resolvability_grid(**kwargs)
        second = script.run_resolvability_grid(**kwargs)
        cells = script.apply_adjacency_gate(script.summarize_cells(first))
        evaluation = script.evaluate_map(cells)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertEqual(len(cells), 4)
        required = {
            "regime",
            "seed",
            "native_delay_frames",
            "downsample",
            "deadband_frames",
            "signal_layer",
            "directional_pair_fraction",
            "unconditional_direction_accuracy",
            "true_edge_coverage",
            "candidate_density",
        }
        self.assertTrue(required.issubset(first[0]))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            script.write_outputs(
                output,
                first,
                cells,
                evaluation,
                {"smoke": True},
            )
            self.assertTrue((output / "resolvability_rows.csv").exists())
            self.assertTrue((output / "resolvability_cells.csv").exists())
            self.assertTrue((output / "report.md").exists())
            payload = json.loads((output / "summary.json").read_text())
            self.assertEqual(set(payload), {"config", "gates", "summary"})
            self.assertIn(
                "does not establish causal identification",
                (output / "report.md").read_text(),
            )


if __name__ == "__main__":
    unittest.main()
