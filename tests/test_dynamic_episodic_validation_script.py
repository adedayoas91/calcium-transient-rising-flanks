import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "examples" / (
        "dynamic_episodic_validation.py"
    )
    spec = importlib.util.spec_from_file_location("dynamic_episodic_validation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DynamicEpisodicValidationScriptTests(unittest.TestCase):
    def test_parse_methods_accepts_combined_cgc_variants(self) -> None:
        script = _load_script_module()

        self.assertEqual(script._parse_methods("cgc,cgc-star"), ("cgc", "cgc-star"))

        with self.assertRaises(ValueError):
            script._parse_methods("cgc,unsupported")

    def test_summary_and_contrast_rows_capture_representation_tradeoffs(self) -> None:
        script = _load_script_module()
        base = {
            "method": "cgc-star",
            "event_mode": "compressed",
            "condition": "dynamic_a_noncausal_fall",
            "simulator_mode": "episodic_dynamic",
            "seed": 1,
            "false_positive_rate": 0.2,
            "orientation_accuracy": 0.8,
            "edge_density": 0.3,
            "truth_edges": 4,
            "fall_propagated_total": 0.0,
            "edge_presence_total": 5,
            "edge_prevalence_mean": 0.75,
            "edge_prevalence_max": 1.0,
            "node_presence_total": 10,
            "node_prevalence_mean": 1.0,
            "active_union_nodes": 5,
        }
        rows = [
            {
                **base,
                "representation": "rise",
                "precision": 0.6,
                "recall": 0.8,
                "f1": 0.6857142857,
            },
            {
                **base,
                "representation": "fall",
                "precision": 0.2,
                "recall": 0.3,
                "f1": 0.24,
            },
            {
                **base,
                "representation": "full",
                "precision": 0.4,
                "recall": 0.5,
                "f1": 0.4444444444,
            },
            {
                **base,
                "representation": "fall_residual",
                "precision": 0.3,
                "recall": 0.4,
                "f1": 0.3428571429,
            },
            {
                **base,
                "representation": "deconvolved",
                "precision": 0.7,
                "recall": 0.9,
                "f1": 0.7875,
            },
        ]

        summaries = script._summary_rows(rows)
        contrasts = script._contrast_rows(summaries)

        self.assertEqual(len(summaries), 5)
        self.assertEqual(len(contrasts), 1)
        self.assertTrue(all(row["method"] == "cgc-star" for row in summaries))
        self.assertEqual(contrasts[0]["method"], "cgc-star")
        self.assertAlmostEqual(contrasts[0]["rise_minus_fall_precision"], 0.4)
        self.assertAlmostEqual(
            contrasts[0]["rise_minus_fall_residual_recall"], 0.4
        )
        self.assertAlmostEqual(contrasts[0]["rise_minus_full_recall"], 0.3)
        self.assertAlmostEqual(
            contrasts[0]["rise_minus_deconvolved_f1"],
            0.6857142857 - 0.7875,
        )
        self.assertEqual(contrasts[0]["rise_minus_fall_fall_propagated_total"], 0.0)

    def test_dynamic_metadata_reports_topology_counts_and_prevalence(self) -> None:
        script = _load_script_module()
        episode = SimpleNamespace(rise_length=20, fall_length=42, fall_start=20, fall_stop=62)
        run = SimpleNamespace(
            dataset=SimpleNamespace(
                episodes=(episode, episode),
                propagated_events=np.zeros((3, 80)),
                edge_presence_counts=np.array(
                    [
                        [0, 2, 0],
                        [0, 0, 1],
                        [0, 0, 0],
                    ]
                ),
                edge_prevalence=np.array(
                    [
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.5],
                        [0.0, 0.0, 0.0],
                    ]
                ),
                node_presence_counts=np.array([2, 2, 1]),
                node_prevalence=np.array([1.0, 1.0, 0.5]),
            ),
            condition=SimpleNamespace(
                dynamic_config=SimpleNamespace(
                    rise_waveform_length=20,
                    fall_state_mode="stochastic_independent",
                    fall_initial_scale=1.0,
                    fall_initial_ceiling_fraction=1.0,
                )
            ),
        )

        metadata = script._dynamic_metadata(run)

        self.assertEqual(metadata["n_episodes"], 2)
        self.assertEqual(metadata["edge_presence_total"], 3)
        self.assertAlmostEqual(metadata["edge_prevalence_mean"], 0.75)
        self.assertAlmostEqual(metadata["edge_prevalence_max"], 1.0)
        self.assertEqual(metadata["node_presence_total"], 5)
        self.assertAlmostEqual(metadata["node_prevalence_mean"], 5 / 6)
        self.assertEqual(metadata["active_union_nodes"], 3)

    def test_rise_candidate_metadata_reports_retained_and_dropped_runs(self) -> None:
        script = _load_script_module()
        candidate = SimpleNamespace(
            candidate_adjacency=np.array(
                [
                    [False, True, False],
                    [False, False, False],
                    [False, True, False],
                ]
            ),
            matches=(object(), object(), object()),
            support_counts=np.array(
                [
                    [0, 2, 0],
                    [0, 0, 0],
                    [0, 1, 0],
                ]
            ),
            mean_overlap_fraction=np.array(
                [
                    [0.0, 0.75, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.5, 0.0],
                ]
            ),
            event_indices=(np.arange(6), np.arange(5), np.arange(4)),
            runs=SimpleNamespace(
                kept_counts=np.array([2, 1, 1]),
                dropped_counts=np.array([1, 0, 2]),
                event_indices=(np.arange(4), np.arange(3), np.arange(2)),
            ),
        )
        run = SimpleNamespace(
            validation=SimpleNamespace(rise_flank_candidates=candidate)
        )

        metadata = script._rise_candidate_metadata(run, "rise")
        fall_metadata = script._rise_candidate_metadata(run, "fall")

        self.assertEqual(metadata["rise_candidate_edges"], 2)
        self.assertEqual(metadata["rise_candidate_matches"], 3)
        self.assertEqual(metadata["rise_candidate_retained_runs"], 4)
        self.assertEqual(metadata["rise_candidate_dropped_runs"], 3)
        self.assertEqual(metadata["rise_candidate_retained_frames"], 9)
        self.assertEqual(metadata["rise_candidate_context_frames"], 15)
        self.assertAlmostEqual(metadata["rise_candidate_mean_overlap"], 0.625)
        self.assertIsNone(fall_metadata["rise_candidate_edges"])


if __name__ == "__main__":
    unittest.main()
