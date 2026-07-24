import importlib.util
import unittest
from pathlib import Path

import numpy as np

from calcium_transient_rising_flank import (
    GraphResult,
    build_representations,
    selected_frame_indices,
)


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "examples" / (
        "run_empirical_null_controls.py"
    )
    spec = importlib.util.spec_from_file_location("run_empirical_null_controls", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecordingEstimator:
    def __init__(self, calls: list[dict[str, object]]) -> None:
        self.calls = calls

    def fit(
        self,
        traces: np.ndarray,
        *,
        event_indices: tuple[np.ndarray, ...] | None = None,
    ) -> GraphResult:
        selected = None
        if event_indices is not None:
            selected = tuple(tuple(np.asarray(values, dtype=int)) for values in event_indices)
        self.calls.append({"shape": traces.shape, "event_indices": selected})
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
            estimator="recording",
        )


class EmpiricalNullControlScriptTests(unittest.TestCase):
    def test_parse_methods_accepts_combined_cgc_variants(self) -> None:
        script = _load_script_module()

        self.assertEqual(script._parse_methods("cgc,cgc-star"), ("cgc", "cgc-star"))

        with self.assertRaises(ValueError):
            script._parse_methods("cgc,unsupported")

    def test_cross_recording_selected_null_uses_donor_event_indices(self) -> None:
        script = _load_script_module()
        traces = np.array(
            [
                [0.0, 1.0, 0.0, 2.0, 1.0, 3.0],
                [0.0, 0.0, 1.0, 0.0, 2.0, 1.0],
            ]
        )
        donor = np.array(
            [
                [0.0, 0.0, 0.0, 5.0, 0.0, 0.0],
                [0.0, 2.0, 1.0, 1.0, 1.0, 3.0],
            ]
        )
        calls: list[dict[str, object]] = []
        record = {
            "case": "A",
            "description": "test",
            "recording": "F1T1",
            "fish": 1,
            "trial": 1,
            "traces": traces,
            "mid": 1,
        }

        rows = script.null_rows_for_recording(
            record,
            donor_traces=donor,
            estimator_factory=lambda _seed: RecordingEstimator(calls),
            method="cgc-star",
            representations=("rise",),
            n_null_replicates=0,
        )

        donor_rise = build_representations(donor).as_dict()["rise"]
        expected = tuple(
            tuple(values) for values in selected_frame_indices(donor_rise)
        )
        self.assertEqual([row["null_type"] for row in rows], [
            "observed",
            "reverse_time",
            "cross_recording",
        ])
        self.assertTrue(all(row["method"] == "cgc-star" for row in rows))
        self.assertEqual(calls[2]["event_indices"], expected)

    def test_null_contrasts_compare_observed_to_each_null_family(self) -> None:
        script = _load_script_module()
        base = {
            "case": "A",
            "method": "cgc",
            "representation": "rise",
            "w_rc": None,
            "edge_density": 0.5,
            "retained_edges": 1,
            "total_weight": 1.0,
        }
        rows = [
            {**base, "null_type": "observed", "w_ic": 0.8},
            {**base, "null_type": "cyclic_shift", "w_ic": 0.2},
            {**base, "null_type": "cyclic_shift", "w_ic": 0.9},
        ]

        contrasts = script.build_null_contrast_rows(rows)

        self.assertEqual(len(contrasts), 1)
        self.assertEqual(contrasts[0]["method"], "cgc")
        self.assertEqual(contrasts[0]["null_type"], "cyclic_shift")
        self.assertAlmostEqual(contrasts[0]["observed_w_ic_mean"], 0.8)
        self.assertAlmostEqual(contrasts[0]["null_w_ic_mean"], 0.55)
        self.assertAlmostEqual(
            contrasts[0]["observed_minus_null_w_ic_mean"], 0.25
        )
        self.assertAlmostEqual(
            contrasts[0]["p_null_ge_observed_w_ic_mean"], 2 / 3
        )


if __name__ == "__main__":
    unittest.main()
