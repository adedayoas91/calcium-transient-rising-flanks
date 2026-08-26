import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).parents[1] / "examples" / "empirical_baselines.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("empirical_baselines", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load empirical baselines script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EmpiricalBaselinesScriptTests(unittest.TestCase):
    def test_progress_state_records_the_in_flight_fit(self) -> None:
        script = _load_script_module()
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            script._write_progress(
                output_dir,
                config={"components": ["lpcmci"]},
                completed={"lpcmci|full|dff|F3T1"},
                expected_count=2,
                status="running",
                active_unit="lpcmci|rise|dff|F3T1",
            )
            payload = json.loads((output_dir / script.PROGRESS_FILE).read_text())

        self.assertEqual(payload["active_unit"], "lpcmci|rise|dff|F3T1")
        self.assertEqual(payload["completed_unit_count"], 1)
        self.assertEqual(payload["expected_unit_count"], 2)

    def test_load_records_matches_cgc_dataframe_and_requires_every_unit(self) -> None:
        script = _load_script_module()
        with tempfile.TemporaryDirectory() as temporary:
            data_file = Path(temporary) / "motoneurons.pkl"
            traces = np.zeros((4, 20))
            pd.DataFrame(
                [
                    {
                        "Fish": 3,
                        "Trace": 1,
                        "fluo_type": "dff",
                        "fluo": traces,
                        "mid": 2,
                        "n_cells": 4,
                    },
                    {
                        "Fish": 3,
                        "Trace": 1,
                        "fluo_type": "dff",
                        "fluo": traces.copy(),
                        "mid": 2,
                        "n_cells": 4,
                    },
                ]
            ).to_pickle(data_file)

            records = script.load_motoneuron_records(
                data_file,
                fluo_types=("dff",),
                recordings=("F3T1",),
            )
            with self.assertRaises(FileNotFoundError):
                script.load_motoneuron_records(
                    data_file,
                    fluo_types=("dff",),
                    recordings=("F3T1", "F5T2"),
                )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["traces"].shape, (4, 20))
        self.assertEqual(records[0]["mid"], 2)
        self.assertEqual(records[0]["fluo_type"], "dff")
        self.assertEqual(len(records[0]["input_digest"]), 64)

    def test_resume_requires_one_row_and_artifact_per_expected_unit(self) -> None:
        script = _load_script_module()
        record = {
            "fluo_type": "dff",
            "recording": "F3T1",
            "fish": 3,
            "trial": 1,
        }
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            expected = script._expected_units(
                (record,),
                components=("oasis", "lpcmci"),
                representations=("full", "rise"),
                oasis_outputs=("spikes",),
                cgc_methods=("cgc",),
            )
            for unit, relative in expected.items():
                path = output_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if unit.startswith("oasis-transform|"):
                    arrays = {
                        "denoised": np.zeros((1, 2)),
                        "spikes": np.zeros((1, 2)),
                        "baseline": np.zeros(1),
                        "penalty_lambda": np.zeros(1),
                        "ar_params_json": np.asarray("{}"),
                    }
                elif unit.startswith("oasis-graph|"):
                    arrays = {
                        "adjacency": np.zeros((1, 1), dtype=bool),
                        "retained_scores": np.zeros((1, 1)),
                        "p_values": np.ones((1, 1)),
                        "best_lags": np.zeros((1, 1), dtype=int),
                    }
                else:
                    arrays = {
                        "graph": np.empty((1, 1, 1), dtype="<U1"),
                        "p_matrix": np.ones((1, 1, 1)),
                        "val_matrix": np.zeros((1, 1, 1)),
                        "lossy_lagged_skeleton": np.zeros((1, 1), dtype=bool),
                    }
                np.savez_compressed(path, **arrays, input_digest=np.asarray("abc"))
            oasis_unit = script._oasis_transform_unit(record)
            oasis_rows = [{"unit": oasis_unit, "input_digest": "abc"}]
            graph_rows = [
                {"unit": unit, "input_digest": "abc"}
                for unit in expected
                if unit != oasis_unit
            ]
            recovered_oasis, recovered_graphs, completed = (
                script._recover_completed_units(
                    output_dir=output_dir,
                    expected=expected,
                    oasis_rows=oasis_rows,
                    graph_rows=graph_rows,
                )
            )
            missing_unit = next(unit for unit in expected if unit.startswith("lpcmci"))
            (output_dir / expected[missing_unit]).unlink()
            _, _, incomplete = script._recover_completed_units(
                output_dir=output_dir,
                expected=expected,
                oasis_rows=oasis_rows,
                graph_rows=graph_rows,
            )
            transform_path = output_dir / expected[oasis_unit]
            transform_path.write_bytes(b"partial")
            _, _, corrupt_parent = script._recover_completed_units(
                output_dir=output_dir,
                expected=expected,
                oasis_rows=oasis_rows,
                graph_rows=graph_rows,
            )
            oasis_graph_unit = next(
                unit for unit in expected if unit.startswith("oasis-graph|")
            )

        self.assertEqual(recovered_oasis, oasis_rows)
        self.assertEqual(recovered_graphs, graph_rows)
        self.assertEqual(completed, set(expected))
        self.assertNotIn(missing_unit, incomplete)
        self.assertNotIn(oasis_unit, corrupt_parent)
        self.assertNotIn(oasis_graph_unit, corrupt_parent)

    def test_stable_seed_is_process_independent(self) -> None:
        script = _load_script_module()
        self.assertEqual(script._stable_seed(10, "unit"), script._stable_seed(10, "unit"))
        self.assertNotEqual(
            script._stable_seed(10, "unit"),
            script._stable_seed(10, "different-unit"),
        )

    def test_method_units_write_reloadable_artifacts_without_optional_packages(self) -> None:
        script = _load_script_module()
        record = {
            "fluo_type": "dff",
            "recording": "F3T1",
            "fish": 3,
            "trial": 1,
            "traces": np.arange(80, dtype=float).reshape(4, 20),
            "mid": 2,
            "input_digest": "abc",
        }

        class FakeOasis:
            def __init__(self, **_kwargs):
                pass

            def fit(self, traces):
                values = np.asarray(traces, dtype=float)
                return SimpleNamespace(
                    denoised=values / 2,
                    spikes=(values > 10).astype(float),
                    baseline=np.zeros(values.shape[0]),
                    penalty_lambda=np.ones(values.shape[0]),
                    ar_params=((0.9,),) * values.shape[0],
                )

        class FakeCausalisedGC:
            def __init__(self, **_kwargs):
                pass

            def fit(self, traces):
                size = np.asarray(traces).shape[0]
                adjacency = np.zeros((size, size), dtype=bool)
                adjacency[0, 1] = True
                return SimpleNamespace(
                    adjacency=adjacency,
                    retained_scores=adjacency.astype(float),
                    p_values=np.full((size, size), 0.5),
                    best_lags=np.zeros((size, size), dtype=int),
                )

        class FakeLPCMCIResult:
            cond_ind_test = "FakeParCorr"

            def __init__(self):
                self.graph = np.full((4, 4, 2), "", dtype="<U3")
                self.graph[0, 1, 1] = "-->"
                self.p_matrix = np.full((4, 4, 2), 0.5)
                self.val_matrix = np.zeros((4, 4, 2))

            def lossy_lagged_skeleton(self):
                skeleton = np.zeros((4, 4), dtype=bool)
                skeleton[0, 1] = True
                return skeleton

        class FakeLPCMCI:
            def __init__(self, **_kwargs):
                pass

            def fit(self, _traces):
                return FakeLPCMCIResult()

        script.OASISDeconvolver = FakeOasis
        script.CausalisedGC = FakeCausalisedGC
        script.LPCMCIAdapter = FakeLPCMCI
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            oasis_row = script.run_oasis_transform(record, output_dir=output_dir)
            graph_row = script.run_oasis_graph(
                record,
                output_dir=output_dir,
                method="cgc",
                oasis_output="spikes",
                max_lag=1,
                n_surrogates=10,
                alpha=0.05,
                seed=10,
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                pag_row = script.run_lpcmci(
                    record,
                    output_dir=output_dir,
                    representation="full",
                    tau_max=1,
                    pc_alpha=0.05,
                    progress=True,
                )
            for row in (oasis_row, graph_row, pag_row):
                artifact = Path(row["artifact_path"])
                self.assertTrue(artifact.is_file())
                with np.load(artifact, allow_pickle=False):
                    pass

        self.assertEqual(graph_row["retained_edges"], 1)
        self.assertEqual(pag_row["skeleton_retained_edges"], 1)
        self.assertIn("[lpcmci] entering Tigramite", output.getvalue())
        self.assertIn("[lpcmci] fit finished", output.getvalue())


if __name__ == "__main__":
    unittest.main()
