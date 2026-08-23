import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "examples" / "simulation_baselines.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("simulation_baselines", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load simulation baseline script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SimulationBaselineScriptTests(unittest.TestCase):
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

    def test_downsampled_events_align_to_next_observed_frame(self) -> None:
        script = _load_script_module()
        events = np.zeros((1, 6))
        events[0, [0, 1, 4, 5]] = 1.0
        aligned = script._align_events_to_observation_grid(events, 2)
        np.testing.assert_array_equal(aligned, np.array([[1.0, 1.0, 1.0]]))

    def test_matched_dataset_is_deterministic_and_uses_static_grid(self) -> None:
        script = _load_script_module()
        first = script.matched_static_dataset(
            run_index=0,
            condition="native",
            seed=1,
            n_steps=80,
            n_seeds=4,
        )
        second = script.matched_static_dataset(
            run_index=0,
            condition="native",
            seed=1,
            n_steps=80,
            n_seeds=4,
        )
        np.testing.assert_array_equal(first["truth"], second["truth"])
        np.testing.assert_array_equal(first["traces"], second["traces"])
        self.assertEqual(first["input_digest"], second["input_digest"])
        self.assertEqual(first["traces"].shape[0], 9)
        self.assertIn(first["split"], {"calibration", "evaluation"})

    def test_resume_completion_requires_full_unique_artifacts(self) -> None:
        script = _load_script_module()
        with tempfile.TemporaryDirectory() as temporary:
            event_rows = [
                {
                    "run": 0,
                    "condition": "native",
                    "seed": 1,
                    "input_digest": "abc",
                    "event_method": method,
                    "threshold_rule": rule,
                }
                for method in ("deconvolved", "oasis")
                for rule in ("positive", "held_out_mad3")
            ]
            graph_rows = []
            for method in ("cgc", "cgc-star", "lpcmci"):
                for representation in script.REPRESENTATIONS:
                    row = {
                        "run": 0,
                        "condition": "native",
                        "seed": 1,
                        "input_digest": "abc",
                        "method": method,
                        "representation": representation,
                    }
                    if method == "lpcmci":
                        pag_path = Path(temporary) / f"{representation}.npz"
                        np.savez_compressed(
                            pag_path,
                            graph=np.empty((1, 1, 1), dtype="<U1"),
                            p_matrix=np.ones((1, 1, 1)),
                            val_matrix=np.zeros((1, 1, 1)),
                            input_digest=np.asarray("abc"),
                        )
                        row["raw_pag_path"] = str(pag_path)
                    graph_rows.append(row)

            _, _, complete = script._recover_completed_units(
                event_rows=event_rows,
                graph_rows=graph_rows,
                unit_specs=((0, "native", 1),),
            )
            _, _, incomplete = script._recover_completed_units(
                event_rows=event_rows,
                graph_rows=graph_rows[:-1],
                unit_specs=((0, "native", 1),),
            )

        self.assertEqual(complete, {"0|native|1"})
        self.assertEqual(incomplete, set())

    def test_lpcmci_only_resume_does_not_require_oasis_rows(self) -> None:
        script = _load_script_module()
        with tempfile.TemporaryDirectory() as temporary:
            graph_rows = []
            for representation in ("full", "rise"):
                pag_path = Path(temporary) / f"{representation}.npz"
                np.savez_compressed(
                    pag_path,
                    graph=np.empty((1, 1, 1), dtype="<U1"),
                    p_matrix=np.ones((1, 1, 1)),
                    val_matrix=np.zeros((1, 1, 1)),
                    input_digest=np.asarray("abc"),
                )
                graph_rows.append(
                    {
                        "condition": "native",
                        "run": 0,
                        "seed": 1,
                        "input_digest": "abc",
                        "method": "lpcmci",
                        "representation": representation,
                        "raw_pag_path": str(pag_path),
                    }
                )
            recovered_events, recovered_graphs, completed = (
                script._recover_completed_units(
                    event_rows=[],
                    graph_rows=graph_rows,
                    unit_specs=((0, "native", 1),),
                    components=("lpcmci",),
                    representations=("full", "rise"),
                )
            )
            Path(graph_rows[0]["raw_pag_path"]).write_bytes(b"partial")
            _, _, corrupt = script._recover_completed_units(
                event_rows=[],
                graph_rows=graph_rows,
                unit_specs=((0, "native", 1),),
                components=("lpcmci",),
                representations=("full", "rise"),
            )

        self.assertEqual(recovered_events, [])
        self.assertEqual(len(recovered_graphs), 2)
        self.assertEqual(completed, {"0|native|1"})
        self.assertEqual(corrupt, set())


if __name__ == "__main__":
    unittest.main()
