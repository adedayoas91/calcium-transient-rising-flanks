import unittest
from pathlib import Path

import pandas as pd

from examples import simulation_baseline_diagnostics as script


def _config(*, oasis: bool, n_runs: int = 10, n_seeds: int = 20) -> dict:
    representations = ["full", "deconvolved", "rise", "fall", "fall_residual"]
    if oasis:
        representations.insert(2, "oasis")
    return {
        "conditions": ["native"],
        "representations": representations,
        "cgc_methods": ["cgc", "cgc-star"],
        "n_runs_outer": n_runs,
        "n_seeds_per_run": n_seeds,
        "n_steps": 80,
        "n_cgc_surrogates": 5,
        "lpcmci_tau_max": 1,
        "lpcmci_pc_alpha": 0.05,
    }


class SimulationBaselineDiagnosticTests(unittest.TestCase):
    def test_full_analysis_unit_counts_match_notebook_contract(self) -> None:
        lpcmci = script.diagnostic_units("lpcmci", _config(oasis=False), n_null=6)
        oasis = script.diagnostic_units("oasis", _config(oasis=True), n_null=6)

        self.assertEqual(len(lpcmci), 224)
        self.assertEqual(len(oasis), 225)
        self.assertEqual(len(oasis) * 2, 450)
        self.assertEqual(
            sum(unit.phase == "null" for unit in lpcmci),
            9,
        )
        self.assertEqual(
            sum(unit.phase in {"noise", "frame_rate"} for unit in lpcmci),
            210,
        )

    def test_oasis_is_primary_for_its_null_and_robustness_analyses(self) -> None:
        units = script.diagnostic_units(
            "oasis", _config(oasis=True, n_runs=1, n_seeds=3), n_null=2
        )
        diagnostic = [unit for unit in units if unit.phase != "representative"]

        self.assertEqual({unit.representation for unit in diagnostic}, {"oasis"})

    def test_diagnostic_plan_requires_all_comparator_representations(self) -> None:
        config = _config(oasis=False)
        config["representations"].remove("fall_residual")

        with self.assertRaisesRegex(ValueError, "fall_residual"):
            script.diagnostic_units("lpcmci", config, n_null=6)

    def test_signflip_test_is_exact_for_small_samples(self) -> None:
        observed, p_value = script.paired_signflip_pvalue([1.0, 1.0, 1.0])

        self.assertEqual(observed, 1.0)
        self.assertEqual(p_value, 0.25)

    def test_oasis_grid_validation_requires_every_unique_fit(self) -> None:
        config = _config(oasis=True, n_runs=1, n_seeds=2)
        rows = []
        for seed in (1, 2):
            for method in config["cgc_methods"]:
                for representation in config["representations"]:
                    rows.append(
                        {
                            "run": 0,
                            "condition": "native",
                            "seed": seed,
                            "split": "evaluation",
                            "method": method,
                            "representation": representation,
                            "precision": 0.5,
                            "recall": 0.4,
                            "false_positive_rate": 0.1,
                            "orientation_accuracy": 0.5,
                            "recovered_w_ic": 0.2,
                        }
                    )
        frame = pd.DataFrame(rows)

        normalized = script.prepare_grid_analysis(
            frame,
            baseline="oasis",
            baseline_dir=Path("unused"),
            config=config,
        )
        with self.assertRaisesRegex(ValueError, "expected 24 unique rows"):
            script.prepare_grid_analysis(
                frame.iloc[:-1],
                baseline="oasis",
                baseline_dir=Path("unused"),
                config=config,
            )

        self.assertEqual(len(normalized), 24)
        self.assertEqual(set(normalized["wic_semantics"]), {"directed_adjacency"})

    def test_grid_summaries_use_locked_rows_and_preserve_metric_semantics(self) -> None:
        rows = []
        for run, seed in ((0, 1), (1, 2)):
            for representation, recall in (("rise", 0.8), ("fall", 0.3)):
                rows.append(
                    {
                        "method": "cgc",
                        "condition": "native",
                        "run": run,
                        "seed": seed,
                        "split": "evaluation",
                        "representation": representation,
                        "precision": 0.5,
                        "recall": recall,
                        "false_positive_rate": 0.1,
                        "comparison_w_ic": recall,
                        "true_w_ic": 1.0,
                        "wic_semantics": "directed_adjacency",
                        "metric_semantics": "directed_edge_recovery",
                    }
                )
        locked, _, paired, wic = script.build_grid_summaries(pd.DataFrame(rows))

        self.assertEqual(len(locked), 4)
        self.assertEqual(paired.loc[0, "mean_rise_minus_fall"], 0.5)
        self.assertEqual(paired.loc[0, "n_runs"], 2)
        self.assertEqual(paired.loc[0, "metric_semantics"], "directed_edge_recovery")
        self.assertIn("delta_w_ic_rise_minus_fall", wic)

    def test_phase_summaries_keep_all_runs(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "phase": "noise",
                    "method": "lpcmci",
                    "representation": "rise",
                    "run": run,
                    "parameter": 0.03,
                    "precision": 0.5,
                    "recall": 0.4 + run / 10,
                    "false_positive_rate": 0.1,
                }
                for run in range(3)
            ]
        )

        summary = script._robustness_summary(rows, "noise")

        self.assertEqual(len(summary), 3)
        self.assertEqual(summary["run"].tolist(), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
