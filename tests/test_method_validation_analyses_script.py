import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "run_method_validation_analyses.py"
)
SPEC = importlib.util.spec_from_file_location("run_method_validation_analyses", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(script)


class MethodValidationAnalysisTests(unittest.TestCase):
    def test_jaccard_handles_empty_and_nonempty_graphs(self) -> None:
        empty = np.zeros((2, 2), dtype=bool)
        first = np.array([[False, True], [False, False]])
        second = np.array([[False, True], [True, False]])

        self.assertEqual(script._jaccard(empty, empty), 1.0)
        self.assertEqual(script._jaccard(first, second), 0.5)

    def test_cross_representation_smoke_reports_both_sources(self) -> None:
        args = SimpleNamespace(
            seed_start=1,
            simulation_seeds=1,
            simulation_steps=240,
            simulation_surrogates=0,
            default_depth=2,
        )

        rows = script.cross_representation_rows(args)

        self.assertEqual(len(rows), 8)
        self.assertEqual(
            {row["source_representation"] for row in rows},
            {"rise", "fall_residual"},
        )
        self.assertEqual(
            {row["simulation_kind"] for row in rows}, {"static", "episodic"}
        )
        self.assertTrue(all(row["target_representation"] == "rise" for row in rows))
        self.assertTrue(
            all(
                row["cross_representation"]
                == (row["source_representation"] == "fall_residual")
                for row in rows
            )
        )

    def test_mad_scale_is_per_roi_and_nonnegative(self) -> None:
        traces = np.array(
            [
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.1, 0.2, 0.3, 0.4],
            ]
        )

        scale = script._mad_scale(traces)

        self.assertEqual(scale.shape, (2,))
        self.assertTrue(np.all(scale >= 0.0))
        self.assertNotEqual(scale[0], scale[1])

    def test_bh_adjustment_excludes_diagonal(self) -> None:
        p_values = np.array([[0.0, 0.01], [0.04, 0.0]])

        adjusted = script._bh_adjusted_p_values(p_values)

        np.testing.assert_allclose(adjusted, [[1.0, 0.02], [0.04, 1.0]])

    def test_signed_event_calcium_has_both_edge_signs_and_is_deterministic(
        self,
    ) -> None:
        first = script.signed_event_calcium_dataset(seed=3, n_steps=200)
        second = script.signed_event_calcium_dataset(seed=3, n_steps=200)

        self.assertGreater(np.count_nonzero(first["excitatory_mask"]), 0)
        self.assertGreater(np.count_nonzero(first["inhibitory_mask"]), 0)
        np.testing.assert_allclose(first["traces"], second["traces"])
        self.assertEqual(first["input_digest"], second["input_digest"])

    def test_first_rank_bh_resolution_accounts_for_add_one_floor(self) -> None:
        resolution = script._first_rank_bh_resolution(14, alpha=0.05)

        self.assertEqual(resolution["n_tests"], 182)
        self.assertEqual(
            resolution["minimum_surrogates_for_first_rank_bh"], 3639
        )


if __name__ == "__main__":
    unittest.main()
