import importlib.util
import unittest
from pathlib import Path

import numpy as np

from calcium_transient_rising_flank import GraphResult


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "examples" / (
        "run_empirical_stability.py"
    )
    spec = importlib.util.spec_from_file_location("run_empirical_stability", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StableEstimator:
    def fit(
        self,
        traces: np.ndarray,
        *,
        event_indices: tuple[np.ndarray, ...] | None = None,
    ) -> GraphResult:
        n_nodes = traces.shape[0]
        adjacency = np.zeros((n_nodes, n_nodes), dtype=bool)
        scores = np.zeros((n_nodes, n_nodes), dtype=float)
        if n_nodes > 1:
            adjacency[0, 1] = True
            scores[0, 1] = 1.0
        return GraphResult(
            scores=scores,
            p_values=np.ones((n_nodes, n_nodes), dtype=float),
            adjacency=adjacency,
            best_lags=np.zeros((n_nodes, n_nodes), dtype=int),
            estimator="stable",
        )


class EmpiricalStabilityScriptTests(unittest.TestCase):
    def test_parse_methods_accepts_combined_cgc_variants(self) -> None:
        script = _load_script_module()

        self.assertEqual(script._parse_methods("cgc,cgc-star"), ("cgc", "cgc-star"))

        with self.assertRaises(ValueError):
            script._parse_methods("cgc,unsupported")

    def test_builds_empirical_stability_rows_and_summaries(self) -> None:
        script = _load_script_module()
        traces = np.array(
            [
                [0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            ]
        )
        record = {
            "case": "A",
            "description": "test",
            "recording": "F1T1",
            "fish": 1,
            "trial": 1,
            "traces": traces,
            "mid": 1,
        }

        rows = script.stability_rows_for_recording(
            record,
            estimator_factory=lambda _seed: StableEstimator(),
            method="cgc-star",
            representations=("rise", "full"),
            n_event_bootstrap=2,
            window_length=5,
            window_step=5,
            seed=4,
        )
        summaries = script.summarize_stability_rows(rows)
        row_keys = {
            (row["representation"], row["stability_type"], row["status"])
            for row in rows
        }

        self.assertIn(("rise", "event_bootstrap", "ok"), row_keys)
        self.assertIn(("rise", "leave_one_transient", "ok"), row_keys)
        self.assertIn(("rise", "time_window", "ok"), row_keys)
        self.assertIn(("full", "time_window", "ok"), row_keys)
        self.assertNotIn(("full", "event_bootstrap", "ok"), row_keys)
        self.assertTrue(all(row["method"] == "cgc-star" for row in rows))
        rise_bootstrap = next(
            row
            for row in summaries
            if row["representation"] == "rise"
            and row["stability_type"] == "event_bootstrap"
        )
        self.assertEqual(rise_bootstrap["method"], "cgc-star")
        self.assertEqual(rise_bootstrap["n_ok"], 1)
        self.assertAlmostEqual(rise_bootstrap["stability_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
