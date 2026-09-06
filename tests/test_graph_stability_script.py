import importlib.util
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "examples" / (
        "analyze_graph_stability.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_graph_stability", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GraphStabilityScriptTests(unittest.TestCase):
    def test_full_trace_loader_ignores_other_matched_representations(self) -> None:
        script = _load_script_module()
        matrix = np.eye(3)
        cache = {
            "records": {
                (1, 1, "dff", "full"): {
                    "weighted_adjacency": matrix,
                    "mid": 1,
                    "representation": "full",
                    "recording": "F1T1",
                    "fluo_type": "dff",
                },
                (1, 1, "dff", "deconvolved"): {
                    "weighted_adjacency": matrix,
                    "mid": 1,
                    "representation": "deconvolved",
                    "recording": "F1T1",
                    "fluo_type": "dff",
                },
            }
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "full.pkl"
            with path.open("wb") as file:
                pickle.dump(cache, file)
            records = script._load_full_trace_records(path, "motoneurons", "cgc")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["representation"], "full_trace")

    def test_summarizes_rise_fall_overlap_and_pairwise_stability(self) -> None:
        script = _load_script_module()
        rise = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        )
        fall = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        cache = {
            "cases": {
                "A": {
                    "description": "test case",
                    "middle": {(1, 1): 1, (1, 2): 1},
                    "graphs": {
                        (1, 1): {"rise": rise, "fall": fall},
                        (1, 2): {"rise": rise, "fall": fall},
                    },
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (root / script.RISING_MOTONEURON_FILE).open("wb") as file:
                pickle.dump(cache, file)

            records = script.load_graph_records(root)
            overlaps = script.build_pairwise_overlaps(records)
            stability = script.build_stability_summaries(records)

        self.assertEqual(len(records), 4)
        self.assertEqual(len(overlaps), 2)
        self.assertTrue(
            all(row["source_representation"] == "fall" for row in overlaps)
        )
        self.assertTrue(
            all(row["target_representation"] == "rise" for row in overlaps)
        )
        self.assertAlmostEqual(overlaps[0]["jaccard"], 1 / 3)
        rise_stability = next(row for row in stability if row["representation"] == "rise")
        self.assertEqual(rise_stability["n_graphs"], 2)
        self.assertEqual(rise_stability["n_shape_compatible_pairs"], 1)
        self.assertAlmostEqual(rise_stability["mean_pairwise_jaccard"], 1.0)


if __name__ == "__main__":
    unittest.main()
