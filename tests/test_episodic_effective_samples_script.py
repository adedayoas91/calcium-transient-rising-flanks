import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "analyze_episodic_effective_samples.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_episodic_effective_samples", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


class EpisodicEffectiveSamplesScriptTests(unittest.TestCase):
    def test_conditioning_parameter_counts_include_intercept(self) -> None:
        self.assertEqual(script._conditioning_parameter_counts(5, 3, 1), (15, 19))

    def test_pair_counts_distinguish_pseudotime_from_physical_time(self) -> None:
        fluorescence = np.vstack(
            [np.arange(10, dtype=float), np.arange(10, dtype=float) ** 2]
        )
        selected = (
            np.array([1, 2, 3, 4, 8, 9]),
            np.array([1, 2, 3, 4, 5, 9]),
        )
        segments = np.full(10, -1, dtype=int)
        segments[1:6] = 1
        segments[8:10] = 2
        truth = np.array([[False, True], [False, False]])

        row = script.pair_sample_row(
            fluorescence,
            selected,
            0,
            1,
            truth=truth,
            n_pasts=1,
            lag=1,
            segment_ids=segments,
        )

        self.assertTrue(row["true_edge"])
        self.assertEqual(row["common_selected_frames"], 5)
        self.assertEqual(row["compressed_usable_samples"], 4)
        self.assertEqual(row["compressed_one_frame_lag_pairs"], 3)
        self.assertEqual(row["compressed_same_segment_lag_pairs"], 3)
        self.assertEqual(row["physical_usable_samples"], 5)

    def test_short_intersection_is_recorded_as_insufficient(self) -> None:
        fluorescence = np.ones((2, 8), dtype=float)
        selected = (np.array([1, 4]), np.array([1, 4]))
        truth = np.zeros((2, 2), dtype=bool)

        row = script.pair_sample_row(
            fluorescence,
            selected,
            0,
            1,
            truth=truth,
            n_pasts=3,
            lag=1,
            segment_ids=None,
        )

        self.assertFalse(row["compressed_estimable"])
        self.assertEqual(row["compressed_usable_samples"], 0)

    def test_mode_contrasts_pair_identical_seeds(self) -> None:
        rows = []
        for seed in (1, 2):
            for mode, offset in (("compressed", 0.0), ("physical", -0.1)):
                rows.append(
                    {
                        "condition": "dynamic_a_noncausal_fall",
                        "method": "cgc",
                        "representation": "rise",
                        "event_mode": mode,
                        "seed": seed,
                        "f1": 0.7 + offset,
                        "usable_samples_pair_median": 20.0 + 10.0 * offset,
                        "ar1_effective_sample_proxy_pair_median": (
                            15.0 + 10.0 * offset
                        ),
                        "insufficient_pair_fraction": 0.1 - offset,
                    }
                )

        contrasts = script.paired_mode_contrasts(
            rows,
            n_bootstrap=100,
            n_permutations=99,
            seed=5,
        )

        self.assertEqual(len(contrasts), 4)
        self.assertTrue(all(row["n_pairs"] == 2 for row in contrasts))
        self.assertTrue(all("p_holm" in row for row in contrasts))


if __name__ == "__main__":
    unittest.main()
