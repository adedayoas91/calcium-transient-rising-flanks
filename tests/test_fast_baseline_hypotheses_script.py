"""Tests for PCMCI+/VAR H1--H4 diagnostics."""

import csv
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "analyze_fast_baseline_hypotheses.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_fast_baseline_hypotheses", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(script)


class FastBaselineHypothesisTests(unittest.TestCase):
    def test_chain_truth_is_directed_and_acyclic(self) -> None:
        truth = script.chain_truth()

        self.assertEqual(truth.shape, (5, 5))
        self.assertEqual(np.count_nonzero(truth), 4)
        self.assertFalse(np.any(truth & truth.T))

    def test_h2_summary_separates_conditions_and_representations(self) -> None:
        fields = [
            "algorithm",
            "truth_target",
            "condition",
            "representation",
            "precision",
            "recall",
            "false_positive_rate",
            "orientation_accuracy",
            "f1",
        ]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "baseline_rows.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for condition in ("a", "b"):
                    for representation in ("full", "rise"):
                        for _ in range(2):
                            writer.writerow(
                                {
                                    "algorithm": "pcmciplus",
                                    "truth_target": "union",
                                    "condition": condition,
                                    "representation": representation,
                                    "precision": 0.5,
                                    "recall": 0.5,
                                    "false_positive_rate": 0.1,
                                    "orientation_accuracy": 0.5,
                                    "f1": 0.5,
                                }
                            )

            rows = script.h2_rows(root, "pcmciplus")

        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["n_seeds"] == 2 for row in rows))
        self.assertEqual({row["f1_mean"] for row in rows}, {0.5})
        self.assertTrue(all("f1_ci_lower" in row for row in rows))
        self.assertTrue(all("f1_std" in row for row in rows))

    def test_h3_summary_uses_only_cyclic_shifts_as_nulls(self) -> None:
        rows = [
            {"comparator": "observed", "f1": 0.8, "retained_edges": 4},
            {"comparator": "cyclic_shift_1", "f1": 0.1, "retained_edges": 1},
            {"comparator": "cyclic_shift_2", "f1": 0.2, "retained_edges": 2},
            {"comparator": "reverse_time", "f1": 0.9, "retained_edges": 5},
            {"comparator": "fall", "f1": 0.7, "retained_edges": 4},
        ]

        summary = script.h3_null_summary(rows)

        self.assertEqual(len(summary), 2)
        self.assertTrue(all(row["n_cyclic_nulls"] == 2 for row in summary))
        self.assertTrue(all("p_holm" in row for row in summary))

        with TemporaryDirectory() as directory:
            output = Path(directory) / "h3_nulls.png"
            script._plot_h3(rows, output)
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".pdf").is_file())

    def test_h4_extreme_contrasts_are_seed_paired(self) -> None:
        rows = []
        for sweep in ("noise", "frame_rate"):
            for seed in (1, 2):
                for parameter, value in ((1.0, 0.8), (3.0, 0.5)):
                    rows.append(
                        {
                            "sweep": sweep,
                            "seed": seed,
                            "parameter": parameter,
                            "precision": value,
                            "recall": value,
                            "false_positive_rate": 1.0 - value,
                            "f1": value,
                        }
                    )

        contrasts = script.h4_extreme_contrasts(
            rows,
            n_bootstrap=100,
            n_permutations=99,
            seed=4,
        )

        self.assertEqual(len(contrasts), 8)
        self.assertTrue(all(row["n_pairs"] == 2 for row in contrasts))
        self.assertTrue(all("p_holm" in row for row in contrasts))


if __name__ == "__main__":
    unittest.main()
