import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "examples" / "reviewer_baseline_benchmarks.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("reviewer_baseline_benchmarks", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load reviewer baseline benchmark script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReviewerBaselineBenchmarkScriptTests(unittest.TestCase):
    def test_event_matching_is_one_to_one_with_tolerance(self) -> None:
        script = _load_script_module()
        tp, fp, fn, errors = script._event_counts(
            np.array([10, 20]),
            np.array([9, 11, 30]),
            tolerance=2,
        )
        self.assertEqual((tp, fp, fn), (1, 2, 1))
        self.assertEqual(errors, [-1])

    def test_event_recovery_aggregates_rois(self) -> None:
        script = _load_script_module()
        truth = np.zeros((2, 8))
        truth[0, 3] = 1.0
        truth[1, 6] = 1.0
        scores = np.zeros_like(truth)
        scores[0, 4] = 1.0
        scores[1, 1] = 1.0
        metrics = script.event_recovery_metrics(
            truth,
            scores,
            thresholds=np.zeros(2),
            tolerance=1,
        )
        self.assertEqual(metrics["true_positives"], 1)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 1)

    def test_common_input_is_deterministic_and_affects_declared_rois(self) -> None:
        script = _load_script_module()
        traces = np.zeros((4, 40))
        first = script.add_latent_common_input(
            traces, strength=0.3, random_state=4
        )
        second = script.add_latent_common_input(
            traces, strength=0.3, random_state=4
        )
        np.testing.assert_allclose(first, second)
        self.assertGreater(np.std(first[0]), 0.0)
        self.assertGreater(np.std(first[2]), 0.0)
        np.testing.assert_allclose(first[1], 0.0)
        np.testing.assert_allclose(first[3], 0.0)

    def test_resume_completion_requires_full_unique_artifacts(self) -> None:
        script = _load_script_module()
        with tempfile.TemporaryDirectory() as temporary:
            event_rows = [
                {
                    "condition": "no_common_input",
                    "seed": 1,
                    "event_method": method,
                    "threshold_rule": rule,
                }
                for method in ("ar1_innovation", "oasis")
                for rule in ("positive", "held_out_mad3")
            ]
            graph_rows = []
            for method in ("cgc", "lpcmci"):
                for representation in script.REPRESENTATIONS:
                    row = {
                        "condition": "no_common_input",
                        "seed": 1,
                        "method": method,
                        "representation": representation,
                    }
                    if method == "lpcmci":
                        pag_path = Path(temporary) / f"{representation}.npz"
                        pag_path.touch()
                        row["raw_pag_path"] = str(pag_path)
                    graph_rows.append(row)

            _, _, complete = script._recover_completed_units(
                event_rows=event_rows,
                graph_rows=graph_rows,
                conditions=("no_common_input",),
                seeds=(1,),
            )
            _, _, incomplete = script._recover_completed_units(
                event_rows=event_rows,
                graph_rows=graph_rows[:-1],
                conditions=("no_common_input",),
                seeds=(1,),
            )

        self.assertEqual(complete, {"no_common_input|1"})
        self.assertEqual(incomplete, set())


if __name__ == "__main__":
    unittest.main()
