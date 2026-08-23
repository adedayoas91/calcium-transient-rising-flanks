import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "examples" / (
        "reviewer_dynamic_extensions.py"
    )
    spec = importlib.util.spec_from_file_location("reviewer_dynamic_extensions", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReviewerDynamicExtensionsScriptTests(unittest.TestCase):
    def test_parse_grid_exposes_multi_lag_and_bout_bounded_specs(self) -> None:
        script = _load_script_module()

        specs = script._parse_grid(
            "lag1_context1,lag2_context2,bout_bounded_lag3_context4"
        )

        self.assertEqual([spec.name for spec in specs], list(script.GRID_SPECS))
        self.assertEqual(specs[0].max_lag, 1)
        self.assertEqual(specs[1].rise_run_context_samples, 2)
        self.assertTrue(specs[2].rise_candidate_filter)
        self.assertEqual(specs[2].min_rise_run_samples, 3)

    def test_dynamic_conditions_include_explicit_causal_fall_truth(self) -> None:
        script = _load_script_module()
        _, rise_sequence, fall_sequence = script._truth_graphs()

        conditions = script._dynamic_conditions(rise_sequence, fall_sequence)
        by_name = {condition.name: condition for condition in conditions}

        causal = by_name["dynamic_b_hybrid_causal_fall_overlap"].dynamic_config
        self.assertIsNotNone(causal.fall_adjacency_sequence)
        self.assertEqual(len(causal.fall_adjacency_sequence), len(fall_sequence))
        for observed, expected in zip(causal.fall_adjacency_sequence, fall_sequence):
            self.assertTrue((observed == expected).all())
        self.assertEqual(causal.fall_state_mode, "passive_decay")
        self.assertEqual(causal.fall_transmission_probability, 1.0)
        self.assertEqual(causal.fall_overlap_exclusion_samples, 0)
        misspecified = by_name[
            "dynamic_c_hybrid_causal_fall_kinetic_misspecification"
        ].dynamic_config
        self.assertEqual(misspecified.fall_overlap_exclusion_samples, 2)
        self.assertIsNotNone(misspecified.gamma_fall)

    def test_summary_and_contrasts_group_rows_by_grid(self) -> None:
        script = _load_script_module()
        base = {
            "method": "cgc",
            "event_mode": "physical",
            "grid": "lag2_context2",
            "condition": "dynamic_b_hybrid_causal_fall",
            "simulator_mode": "episodic_dynamic",
            "seed": 3,
            "orientation_accuracy": 0.8,
            "edge_density": 0.3,
            "truth_edges": 2,
            "fall_truth_edges": 2,
            "fall_propagated_total": 4.0,
        }
        rows = [
            {
                **base,
                "representation": "rise",
                "precision": 0.7,
                "recall": 0.8,
                "false_positive_rate": 0.1,
                "f1": 0.7466666667,
            },
            {
                **base,
                "representation": "fall",
                "precision": 0.6,
                "recall": 0.7,
                "false_positive_rate": 0.2,
                "f1": 0.6461538462,
            },
            {
                **base,
                "representation": "fall_residual",
                "precision": 0.65,
                "recall": 0.75,
                "false_positive_rate": 0.18,
                "f1": 0.6964285714,
            },
            {
                **base,
                "representation": "full",
                "precision": 0.5,
                "recall": 0.6,
                "false_positive_rate": 0.3,
                "f1": 0.5454545455,
            },
            {
                **base,
                "representation": "deconvolved",
                "precision": 0.55,
                "recall": 0.65,
                "false_positive_rate": 0.25,
                "f1": 0.5958333333,
            },
        ]

        summaries = script._summary_rows(rows)
        contrasts = script._contrast_rows(summaries)

        self.assertEqual(len(summaries), 5)
        self.assertEqual(len(contrasts), 1)
        self.assertEqual(contrasts[0]["grid"], "lag2_context2")
        self.assertAlmostEqual(contrasts[0]["rise_minus_fall_precision"], 0.1)
        self.assertAlmostEqual(
            contrasts[0]["rise_minus_fall_residual_recall"],
            0.05,
        )

    def test_resume_completion_keys_include_grid_dimension(self) -> None:
        script = _load_script_module()

        def row(grid: str, seed: int, representation: str):
            return {
                "method": "cgc",
                "grid": grid,
                "condition": "dynamic_b_hybrid_causal_fall",
                "seed": seed,
                "representation": representation,
                "precision": 0.5,
            }

        rows = [
            row("lag1_context1", 1, representation)
            for representation in script.REPRESENTATION_ORDER
        ]
        rows.extend([row("lag2_context2", 1, "full"), row("lag2_context2", 1, "rise")])

        complete_rows = script._complete_grid_rows(rows)
        completed_keys = script._completed_run_keys(rows)

        self.assertEqual(len(complete_rows), len(script.REPRESENTATION_ORDER))
        self.assertEqual(
            completed_keys,
            {("cgc", "lag1_context1", "dynamic_b_hybrid_causal_fall", 1)},
        )

    def test_resume_signature_rejects_changed_script_settings(self) -> None:
        script = _load_script_module()

        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            script._write_resume_state(
                output_dir,
                config_signature={"n_steps": 360, "grid": ["lag1_context1"]},
                total_units=2,
                completed_units=1,
                status="running",
            )

            script._validate_resume_signature(
                output_dir,
                {"n_steps": 360, "grid": ["lag1_context1"]},
            )
            with self.assertRaises(ValueError):
                script._validate_resume_signature(
                    output_dir,
                    {"n_steps": 480, "grid": ["lag1_context1"]},
                )


if __name__ == "__main__":
    unittest.main()
