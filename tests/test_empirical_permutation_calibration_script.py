import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "analyze_empirical_permutation_calibration.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_empirical_permutation_calibration", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


class EmpiricalPermutationCalibrationScriptTests(unittest.TestCase):
    def test_audit_applies_add_one_before_bh(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.npz"
            p_values = np.array(
                [
                    [1.0, 0.0, 0.2],
                    [0.0, 1.0, 0.4],
                    [0.2, 0.4, 1.0],
                ]
            )
            adjacency = script.benjamini_hochberg(p_values, 0.05)
            np.savez_compressed(
                path,
                adjacency=adjacency,
                p_values=p_values,
                metadata_json=json.dumps(
                    {
                        "case": "D",
                        "recording": "F1T1",
                        "method": "cgc",
                        "representation": "rise",
                    }
                ),
            )

            row = script.audit_artifact(path, n_surrogates=19, alpha=0.05)

            self.assertTrue(row["saved_matches_recomputed_raw_bh"])
            self.assertEqual(row["saved_edges"], 2)
            self.assertEqual(row["corrected_edges"], 0)
            self.assertEqual(row["removed_edges"], 2)
            self.assertEqual(row["minimum_corrected_p"], 0.05)

    def test_resolution_bound_is_labeled_and_conservative(self) -> None:
        row = script.resolution_rows([17], alpha=0.05, n_surrogates=1000)[0]

        self.assertEqual(row["n_tests"], 272)
        self.assertEqual(
            row["minimum_surrogates_for_first_rank_resolution"],
            5439,
        )
        self.assertFalse(row["current_count_resolves_first_rank"])
        self.assertIn("not an empirical", row["interpretation"])


if __name__ == "__main__":
    unittest.main()
