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
    def test_method_loader_keeps_all_matched_representations(self) -> None:
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
            records = script._load_method_records(path, "cgc")

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["representation"], "full_trace")
        self.assertEqual(records[1]["representation"], "deconvolved")

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
            "records": {
                (1, 1, "dff", "rise"): {
                    "weighted_adjacency": rise,
                    "representation": "rise",
                    "recording": "F1T1",
                    "fish": 1,
                    "trial": 1,
                    "fluo_type": "dff",
                },
                (1, 1, "dff", "fall"): {
                    "weighted_adjacency": fall,
                    "representation": "fall",
                    "recording": "F1T1",
                    "fish": 1,
                    "trial": 1,
                    "fluo_type": "dff",
                },
                (1, 2, "dff", "rise"): {
                    "weighted_adjacency": rise,
                    "representation": "rise",
                    "recording": "F1T2",
                    "fish": 1,
                    "trial": 2,
                    "fluo_type": "dff",
                },
                (1, 2, "dff", "fall"): {
                    "weighted_adjacency": fall,
                    "representation": "fall",
                    "recording": "F1T2",
                    "fish": 1,
                    "trial": 2,
                    "fluo_type": "dff",
                },
            }
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (root / "cgc_motoneurons_weighted_adjacency_matrices.pkl").open(
                "wb"
            ) as file:
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

    def test_loads_lpcmci_and_oasis_artifacts_without_hindbrain_inputs(self) -> None:
        script = _load_script_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            methods = root / "methods"
            lpcmci = root / "lpcmci"
            oasis = root / "oasis"
            methods.mkdir()
            (lpcmci / "raw_pag" / "dff" / "F1T1").mkdir(parents=True)
            (oasis / "oasis_graphs" / "dff" / "F1T1").mkdir(parents=True)

            skeleton_path = lpcmci / "raw_pag" / "dff" / "F1T1" / "rise.npz"
            np.savez_compressed(
                skeleton_path,
                lossy_lagged_skeleton=np.array(
                    [[False, True], [True, False]], dtype=bool
                ),
            )
            (lpcmci / "graph_summary_rows.csv").write_text(
                "fluo_type,recording,fish,trial,method,representation,artifact_path\n"
                "dff,F1T1,1,1,lpcmci,rise,/foreign/computer/rise.npz\n"
            )

            oasis_path = (
                oasis / "oasis_graphs" / "dff" / "F1T1" / "cgc__spikes.npz"
            )
            np.savez_compressed(
                oasis_path,
                retained_scores=np.array([[0.0, 0.5], [0.0, 0.0]]),
            )
            (oasis / "graph_summary_rows.csv").write_text(
                "fluo_type,recording,fish,trial,method,representation,artifact_path\n"
                "dff,F1T1,1,1,cgc,oasis_spikes,"
                "/foreign/computer/cgc__spikes.npz\n"
            )

            records = script.load_graph_records(
                methods,
                lpcmci_input_dir=lpcmci,
                oasis_input_dir=oasis,
            )

        self.assertEqual({row["method"] for row in records}, {"lpcmci", "oasis+cgc"})
        self.assertTrue(all(row["dataset"] == "motoneurons" for row in records))
        self.assertFalse(hasattr(script, "RISING_HINDBRAIN_FILE"))


if __name__ == "__main__":
    unittest.main()
