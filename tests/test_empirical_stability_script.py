import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
    def test_main_resume_reuses_completed_checkpoint_unit(self) -> None:
        script = _load_script_module()
        record = {
            "case": "D",
            "description": "test",
            "recording": "F1T1",
            "fish": 1,
            "trial": 1,
            "mid": 1,
            "traces": np.zeros((2, 8), dtype=float),
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "outputs"
            data_dir = root / "data"
            config = {
                "data_dir": str(data_dir),
                "cases": ["D"],
                "recordings": "F1T1",
                "representations": list(script.DEFAULT_REPRESENTATIONS),
                "methods": ["cgc"],
                "n_event_bootstrap": 20,
                "window_length": 120,
                "window_step": None,
                "include_leave_one_neuron": True,
                "include_leave_one_transient": True,
                "max_lag": 1,
                "n_estimator_surrogates": 0,
                "alpha": 0.05,
                "score_threshold": 0.0,
                "event_mode": "physical",
                "fdr": True,
                "seed": 0,
            }
            checkpoint_store = script.JsonUnitCheckpointStore(
                output_dir,
                "empirical_stability",
                config,
            )
            checkpoint_store.initialize(resume=False)
            checkpoint_store.save_rows(
                "cgc|D|F1T1",
                [
                    {
                        "case": "D",
                        "description": "test",
                        "recording": "F1T1",
                        "fish": 1,
                        "trial": 1,
                        "method": "cgc",
                        "representation": "rise",
                        "event_selected": True,
                        "stability_type": "event_bootstrap",
                        "status": "ok",
                        "skip_reason": None,
                        "n_graphs": 1,
                        "stability": 1.0,
                        "mean_w_ic": 1.0,
                        "mean_w_rc": None,
                        "mean_edge_density": 0.5,
                        "mean_retained_edges": 1.0,
                        "mean_total_weight": 1.0,
                    }
                ],
            )
            checkpoint_store.finish(completed_units=1, total_units=1)

            argv = [
                "run_empirical_stability.py",
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(output_dir),
                "--cases",
                "D",
                "--recordings",
                "F1T1",
                "--resume",
            ]
            with patch.object(sys, "argv", argv):
                with patch.object(script, "load_case_traces", return_value=[record]):
                    with patch.object(
                        script,
                        "stability_rows_for_recording",
                        side_effect=AssertionError("resume should skip recomputation"),
                    ):
                        script.main()

            payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(payload["n_rows"], 1)

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
