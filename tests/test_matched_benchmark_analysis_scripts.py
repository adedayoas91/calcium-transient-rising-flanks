"""Tests for strict matched-benchmark validation and summaries."""

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load(name: str):
    path = EXAMPLES / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dynamic = _load("analyze_matched_dynamic_benchmark")
motor = _load("analyze_matched_motorneuron_benchmark")
confounding = _load("analyze_matched_confounding_benchmark")


class MatchedDynamicAnalysisTests(unittest.TestCase):
    def _rows(self):
        rows = []
        for algorithm in dynamic.ALGORITHMS:
            depths = (1, 2, 3) if algorithm in {"cgc", "cgc-star"} else (None,)
            for seed in (1, 2):
                for depth in depths:
                    rows.append(
                        {
                            "dataset": "simulations",
                            "simulation_kind": "dynamic-extension",
                            "condition": "dynamic_a_noncausal_fall",
                            "seed": seed,
                            "algorithm": algorithm,
                            "representation": "rise",
                            "truth_target": "union",
                            "truth_edges": 4,
                            "n_timepoints": 1500,
                            "max_lag": 1,
                            "n_pasts": depth,
                            "input_digest": f"input-{seed}",
                            "f1": 0.5 + 0.01 * (depth or 1),
                            "precision": 0.5,
                            "recall": 0.5,
                            "false_positive_rate": 0.1,
                            "orientation_accuracy": 0.5,
                            "retained_edges": 4.0,
                        }
                    )
        return rows

    def test_validation_and_depth_contrasts_use_paired_units(self) -> None:
        rows = self._rows()
        validation = dynamic.validate_matched_design(rows)
        contrasts = dynamic.conditioning_depth_contrasts(
            rows,
            n_bootstrap=100,
            n_permutations=99,
            seed=5,
        )
        method_contrasts = dynamic.paired_method_contrasts(
            rows,
            n_bootstrap=100,
            n_permutations=99,
            seed=6,
        )

        self.assertEqual(validation["status"], "matched")
        self.assertEqual(validation["max_lag"], 1)
        self.assertEqual(len(contrasts), 8)
        self.assertTrue(all(row["n"] == 2 for row in contrasts))
        self.assertTrue(all("p_holm" in row for row in contrasts))
        self.assertEqual(len(method_contrasts), 12)
        self.assertTrue(all(row["n"] == 2 for row in method_contrasts))
        self.assertTrue(all("p_holm" in row for row in method_contrasts))

    def test_validation_rejects_a_changed_input_digest(self) -> None:
        rows = self._rows()
        next(
            row for row in rows if row["algorithm"] == "pcmciplus" and row["seed"] == 1
        )["input_digest"] = "different"

        with self.assertRaisesRegex(ValueError, "input_digest"):
            dynamic.validate_matched_design(rows)


class MatchedMotorneuronAnalysisTests(unittest.TestCase):
    def _rows(self):
        rows = []
        for algorithm in motor.ALGORITHMS:
            depths = (1, 2, 3) if algorithm in {"cgc", "cgc-star"} else (None,)
            for recording in ("F3T1", "F3T2"):
                for representation in ("rise", "fall"):
                    for depth in depths:
                        rows.append(
                            {
                                "dataset": "motorneurons",
                                "fluo_type": "C",
                                "recording": recording,
                                "representation": representation,
                                "algorithm": algorithm,
                                "max_lag": 1,
                                "n_pasts": depth,
                                "n_timepoints": 800,
                                "n_rois": 6,
                                "input_digest": f"{recording}-{representation}",
                                "edge_density": 0.2,
                                "retained_edges": 6.0,
                                "w_ic": 0.8,
                                "w_rc": 0.7,
                            }
                        )
        return rows

    def test_validation_and_summary_keep_depth_separate(self) -> None:
        rows = self._rows()
        validation = motor.validate_matched_design(rows)
        summaries = motor.summarize_rows(rows, n_bootstrap=100, seed=7)
        contrasts = motor.descriptive_paired_contrasts(
            rows, n_bootstrap=100, seed=8
        )

        self.assertEqual(validation["status"], "matched")
        self.assertEqual(validation["cgc_depths"]["cgc"], [1, 2, 3])
        cgc_density = [
            row
            for row in summaries
            if row["algorithm"] == "cgc" and row["metric"] == "edge_density"
        ]
        self.assertEqual({row["n_pasts"] for row in cgc_density}, {1, 2, 3})
        self.assertTrue(contrasts)
        self.assertTrue(all(row["p_value"] is None for row in contrasts))

        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "contrasts.csv"
            motor._write_csv(output_path, contrasts)
            header = output_path.read_text(encoding="utf-8").splitlines()[0]

        self.assertIn("reference_n_pasts", header)
        self.assertIn("n_pasts", header)


class MatchedConfoundingAnalysisTests(unittest.TestCase):
    def test_validation_uses_a_common_skeleton_estimand(self) -> None:
        rows = []
        for algorithm in confounding.ALGORITHMS:
            depths = (1, 2, 3) if algorithm in {"cgc", "cgc-star"} else (None,)
            for condition in confounding.CONDITIONS:
                for seed in (1, 2):
                    for depth in depths:
                        rows.append(
                            {
                                "simulation_kind": "confounding",
                                "condition": condition,
                                "seed": seed,
                                "algorithm": algorithm,
                                "representation": "rise",
                                "truth_target": "union",
                                "truth_edges": 4,
                                "n_timepoints": 1500,
                                "max_lag": 1,
                                "n_pasts": depth,
                                "input_digest": f"{condition}-{seed}",
                                "projection": (
                                    "lossy_lagged_pag_skeleton"
                                    if algorithm == "lpcmci"
                                    else "directed"
                                ),
                                "skeleton_f1": 0.5,
                                "skeleton_precision": 0.5,
                                "skeleton_recall": 0.5,
                                "skeleton_false_positive_rate": 0.1,
                            }
                        )

        validation = confounding.validate_design(rows)
        degradation = confounding.paired_degradation(rows)

        self.assertEqual(validation["status"], "matched")
        self.assertEqual(validation["common_estimand"], "lagged skeleton")
        self.assertEqual(len(degradation), 20)
        self.assertEqual(
            {row["metric"] for row in degradation},
            {"skeleton_f1", "skeleton_false_positive_rate"},
        )
        self.assertTrue(all("ci_lower" in row for row in degradation))
        self.assertTrue(all("p_holm" in row for row in degradation))

    def test_artifact_audit_tracks_the_latent_target_pair(self) -> None:
        truth = np.zeros((5, 5), dtype=bool)
        truth[0, 1] = True
        adjacency = truth.copy()
        adjacency[1, 3] = True
        with TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact.npz"
            np.savez_compressed(artifact, adjacency=adjacency, truth=truth)
            rows = [
                {
                    "condition": condition,
                    "seed": 1,
                    "algorithm": algorithm,
                    "representation": "rise",
                    "n_pasts": 1 if algorithm in {"cgc", "cgc-star"} else None,
                    "latent_driver_targets": (
                        "1,3" if condition == "latent_common_driver" else ""
                    ),
                    "artifact_path": str(artifact),
                }
                for algorithm in confounding.ALGORITHMS
                for condition in confounding.CONDITIONS
            ]

            edge_rows = confounding.confounder_edge_rows(rows)
            summaries = confounding.summarize_confounder_edges(edge_rows)
            confounding.plot_confounder_edge_mechanism(summaries, Path(directory))

            self.assertTrue(
                (Path(directory) / "confounder_target_pair_selection.pdf").is_file()
            )

        self.assertEqual(len(edge_rows), 15)
        self.assertTrue(all(row["target_pair_selected"] for row in edge_rows))
        self.assertTrue(
            all(row["false_skeleton_edge_count"] == 1 for row in edge_rows)
        )
        self.assertEqual(len(summaries), 15)


if __name__ == "__main__":
    unittest.main()
