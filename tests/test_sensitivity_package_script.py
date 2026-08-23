"""Tests for sensitivity aggregation and interpretation."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


def _load_script_module(filename: str):
    path = Path(__file__).resolve().parents[1] / "examples" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SensitivityPackageTests(unittest.TestCase):
    def test_bootstrap_interval_is_deterministic_and_contains_mean(self) -> None:
        script = _load_script_module("build_sensitivity_package.py")

        first = script.bootstrap_mean_ci(
            [0.0, 1.0, 2.0, 3.0], random_state=7, n_bootstrap=500
        )
        second = script.bootstrap_mean_ci(
            [0.0, 1.0, 2.0, 3.0], random_state=7, n_bootstrap=500
        )

        self.assertEqual(first, second)
        self.assertLessEqual(first[1], first[0])
        self.assertGreaterEqual(first[2], first[0])

    def test_dynamic_summaries_pair_rows_by_seed_and_deduplicate_generator(self) -> None:
        script = _load_script_module("build_sensitivity_package.py")
        rows = []
        for seed, rise_precision, full_precision in ((1, 0.8, 0.6), (2, 1.0, 0.5)):
            for representation, precision in (
                ("rise", rise_precision),
                ("full", full_precision),
            ):
                rows.append(
                    {
                        "method": "cgc",
                        "event_mode": "physical",
                        "condition": "dynamic_a_noncausal_fall",
                        "simulator_mode": "episodic_dynamic",
                        "seed": str(seed),
                        "representation": representation,
                        "precision": str(precision),
                        "recall": "0.5",
                        "false_positive_rate": "0.1",
                        "f1": "0.6",
                        "orientation_accuracy": "0.75",
                        "edge_density": "0.2",
                        "n_episodes": "3",
                        "min_rise_length": "20",
                        "rise_waveform_length": "20",
                        "min_fall_to_rise": "2.1",
                        "fall_propagated_total": "0",
                        "fall_initial_scale": "1",
                        "fall_initial_ceiling_fraction": "1",
                        "edge_presence_total": "9",
                        "edge_prevalence_mean": "0.5",
                        "edge_prevalence_max": "0.67",
                        "node_presence_total": "13",
                        "node_prevalence_mean": "0.87",
                        "active_union_nodes": "5",
                        "fall_state_mode": "stochastic_independent",
                    }
                )

        summaries = script.summarize_dynamic_rows(
            rows, random_state=1, n_bootstrap=100
        )
        contrasts = script.build_dynamic_paired_contrasts(
            rows, random_state=1, n_bootstrap=100
        )
        generator_rows, generator_summary = script.build_generator_summary(
            rows,
            {"n_steps": 1500, "dynamic_episodes": 3},
            random_state=1,
            n_bootstrap=100,
        )

        self.assertEqual(len(summaries), 2)
        self.assertEqual(len(contrasts), 1)
        self.assertEqual(contrasts[0]["contrast"], "rise_minus_full")
        self.assertAlmostEqual(contrasts[0]["precision_mean"], 0.35)
        self.assertEqual(len(generator_rows), 2)
        self.assertEqual(generator_summary[0]["n_seeds"], 2)
        self.assertEqual(generator_summary[0]["config_n_steps"], 1500)

    def test_tiny_threshold_grid_and_minimal_example_are_runnable(self) -> None:
        sensitivity = _load_script_module("build_sensitivity_package.py")
        minimal = _load_script_module("minimal_reproducible_analysis.py")

        rows = sensitivity.run_threshold_smoothing_grid(
            seeds=(3,),
            thresholds=(0.01,),
            smoothing_windows=(1,),
            n_steps=160,
            n_surrogates=0,
        )
        summary = sensitivity.summarize_threshold_grid(
            rows, random_state=1, n_bootstrap=10
        )
        result = minimal.run_analysis(seed=3, n_steps=160, n_surrogates=0)
        with tempfile.TemporaryDirectory() as tmp:
            plot_path = Path(tmp) / "sensitivity.png"
            sensitivity.plot_threshold_sensitivity(summary, plot_path)
            self.assertGreater(plot_path.stat().st_size, 0)

        self.assertEqual(len(rows), 1)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["n_seeds"], 1)
        self.assertEqual(result["config"]["event_mode"], "physical")
        self.assertEqual(result["truth_edges"], 2)
        self.assertEqual(len(result["selected_frames_per_roi"]), 4)


if __name__ == "__main__":
    unittest.main()
