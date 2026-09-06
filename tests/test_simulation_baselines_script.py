import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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
    def test_oasis_cgc_configuration_matches_the_reference_notebooks(self) -> None:
        script = _load_script_module()
        args = SimpleNamespace(
            n_runs_outer=10,
            n_seeds=20,
            n_steps=3000,
            n_cgc_surrogates=1000,
            lpcmci_tau_max=2,
            lpcmci_pc_alpha=0.05,
            event_tolerance=2,
        )

        oasis = script._config(
            args,
            components=("oasis",),
            representations=("full", "deconvolved", "oasis", "rise", "fall"),
            cgc_methods=("cgc", "cgc-star"),
        )
        lpcmci = script._config(
            args,
            components=("lpcmci",),
            representations=("full", "deconvolved", "rise", "fall"),
            cgc_methods=("cgc", "cgc-star"),
        )

        self.assertEqual(
            oasis["cgc"],
            {
                "alpha": 0.01,
                "beta": 0.001,
                "n_pasts": 2,
                "n_lags": 1,
                "fdr": False,
                "simulation": True,
            },
        )
        self.assertEqual(oasis["resume_schema_version"], 5)
        self.assertEqual(lpcmci["resume_schema_version"], 4)
        self.assertNotIn("cgc", lpcmci)

    def test_incompatible_output_is_preserved_before_restart(self) -> None:
        script = _load_script_module()
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "oasis"
            output_dir.mkdir()
            (output_dir / "progress.json").write_text("old")

            archive = script._archive_incompatible_output(
                output_dir, {"resume_schema_version": 4}
            )

            self.assertTrue((archive / "progress.json").is_file())
            self.assertTrue(output_dir.is_dir())
            self.assertEqual(list(output_dir.iterdir()), [])

    def test_progress_state_records_the_in_flight_resume_unit(self) -> None:
        script = _load_script_module()
        config = {
            "conditions": ["native"],
            "n_runs_outer": 1,
            "n_seeds_per_run": 2,
        }
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            script._write_progress(
                output_dir,
                config=config,
                completed_units={"0|native|1"},
                status="running",
                active_unit="0|native|2",
            )
            payload = json.loads((output_dir / script.PROGRESS_FILE).read_text())

        self.assertEqual(payload["active_unit"], "0|native|2")
        self.assertEqual(payload["completed_unit_count"], 1)
        self.assertEqual(payload["expected_unit_count"], 2)

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
