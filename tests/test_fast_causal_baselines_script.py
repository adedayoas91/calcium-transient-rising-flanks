import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "examples" / "run_fast_causal_baselines.py"
SPEC = importlib.util.spec_from_file_location("run_fast_causal_baselines", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(script)


class FastCausalBaselineScriptTests(unittest.TestCase):
    def test_representation_map_retains_signed_samples(self) -> None:
        traces = np.array([[0.0, 1.0, 0.7, 0.1], [0.0, 0.2, 0.5, 0.3]])

        represented = script.representation_map(traces)

        self.assertEqual(set(represented), set(script.REPRESENTATIONS))
        self.assertLess(float(np.min(represented["signed_difference"])), 0.0)
        self.assertLess(float(np.min(represented["signed_innovation"])), 0.0)
        self.assertTrue(np.all(represented["rise"] >= 0.0))

    def test_episodic_dataset_is_deterministic_and_has_truth(self) -> None:
        first = script.episodic_dataset(seed=4, n_steps=400)
        second = script.episodic_dataset(seed=4, n_steps=400)

        np.testing.assert_array_equal(first["truth"], second["truth"])
        np.testing.assert_allclose(first["traces"], second["traces"])
        self.assertGreater(np.count_nonzero(first["truth"]), 0)
        self.assertEqual(first["input_digest"], second["input_digest"])

    def test_var_granger_fit_returns_auditable_arrays(self) -> None:
        rng = np.random.default_rng(5)
        values = rng.normal(size=(3, 300))
        values[1, 1:] += values[0, :-1]

        adjacency, metadata, arrays = script._fit(
            "var-granger", values, max_lag=1, alpha=0.05
        )

        self.assertEqual(adjacency.shape, (3, 3))
        self.assertIn("p_values", arrays)
        self.assertIn("coefficients", arrays)
        self.assertIn("companion_spectral_radius", arrays)
        self.assertIn("var_stable", metadata)
        self.assertEqual(metadata["projection"], "none")

    def test_pcmciplus_fit_uses_requested_alpha_and_bh(self) -> None:
        graph = np.full((2, 2, 2), "", dtype=object)
        result = type(
            "Result",
            (),
            {
                "graph": graph,
                "p_matrix": np.ones((2, 2, 2)),
                "val_matrix": np.zeros((2, 2, 2)),
                "cond_ind_test": "ParCorr",
                "assumptions": (),
                "lagged_adjacency": lambda self: np.zeros((2, 2), dtype=bool),
            },
        )()
        with patch.object(script, "PCMCIPlusAdapter") as adapter:
            adapter.return_value.fit.return_value = result
            script._fit("pcmciplus", np.ones((2, 20)), max_lag=2, alpha=0.01)

        self.assertEqual(
            adapter.call_args.kwargs["run_kwargs"],
            {
                "tau_min": 0,
                "pc_alpha": 0.01,
                "fdr_method": "fdr_bh",
            },
        )


if __name__ == "__main__":
    unittest.main()
