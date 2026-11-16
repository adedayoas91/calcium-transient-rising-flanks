import importlib.util
import json
import tempfile
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


class ArtifactEstimator:
    def fit(
        self,
        traces: np.ndarray,
        *,
        event_indices: tuple[np.ndarray, ...] | None = None,
    ) -> GraphResult:
        del event_indices
        n_nodes = traces.shape[0]
        scores = np.zeros((n_nodes, n_nodes), dtype=float)
        p_values = np.ones((n_nodes, n_nodes), dtype=float)
        adjacency = np.zeros((n_nodes, n_nodes), dtype=bool)
        best_lags = np.zeros((n_nodes, n_nodes), dtype=int)
        if n_nodes > 1:
            scores[0, 1] = 2.5
            scores[1, 0] = 1.25
            p_values[0, 1] = 0.01
            p_values[1, 0] = 0.2
            adjacency[0, 1] = True
            best_lags[0, 1] = 2
        return GraphResult(
            scores=scores,
            p_values=p_values,
            adjacency=adjacency,
            best_lags=best_lags,
            estimator="artifact",
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

    def test_expected_fit_count_matches_matched_fdr_campaign(self) -> None:
        script = _load_script_module()
        records = [
            {
                "case": case,
                "recording": recording,
                "fish": fish,
                "traces": np.zeros((2, 8)),
            }
            for case in ("C", "D")
            for recording, fish in (("F3T1", 3), ("F3T2", 3), ("F5T2", 5))
        ]

        count = script._expected_top_level_fit_count(
            records=records,
            methods=("cgc", "cgc-star"),
            representations=("rise", "fall"),
            n_null_replicates=0,
        )

        self.assertEqual(count, 72)

    def test_partial_csv_round_trip_preserves_numeric_values(self) -> None:
        script = _load_script_module()
        rows = [
            {
                "case": "C",
                "method": "cgc",
                "recording": "F3T1",
                "replicate": 0,
                "w_ic": 0.75,
                "w_rc": None,
                "binary": False,
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.csv"
            script._write_csv(path, rows)
            restored = script._read_csv(path)

        self.assertEqual(restored[0]["replicate"], 0)
        self.assertAlmostEqual(restored[0]["w_ic"], 0.75)
        self.assertIsNone(restored[0]["w_rc"])
        self.assertFalse(restored[0]["binary"])

    def test_resume_recovers_complete_rows_without_progress_update(self) -> None:
        script = _load_script_module()
        record = {
            "case": "C",
            "recording": "F3T1",
            "fish": 3,
            "traces": np.zeros((2, 8)),
        }
        rows = [
            {
                "method": "cgc",
                "case": "C",
                "recording": "F3T1",
                "representation": "rise",
                "null_type": null_type,
                "replicate": 0,
            }
            for null_type in ("observed", "reverse_time")
        ]

        recovered, completed = script._recover_completed_units(
            rows,
            records=[record],
            methods=("cgc",),
            representations=("rise",),
            n_null_replicates=0,
        )

        self.assertEqual(recovered, rows)
        self.assertEqual(completed, {"cgc|C|F3T1"})

    def test_observed_graph_artifacts_capture_edge_level_matrices_and_metadata(self) -> None:
        script = _load_script_module()
        traces = np.array(
            [
                [0.0, 1.0, 0.0, 2.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, 2.0, 0.0],
            ]
        )
        record = {
            "case": "D",
            "description": "artifact test",
            "recording": "F5T2",
            "fish": 5,
            "trial": 2,
            "traces": traces,
            "mid": 1,
        }
        staged: list[dict[str, object]] = []

        rows = script.null_rows_for_recording(
            record,
            donor_traces=None,
            estimator_factory=lambda _seed: ArtifactEstimator(),
            method="cgc-star",
            representations=("rise", "full", "fall"),
            n_null_replicates=0,
            observed_artifacts=staged,
            fdr=True,
        )

        self.assertEqual(len(rows), 6)
        self.assertEqual(
            [(entry["method"], entry["representation"]) for entry in staged],
            [("cgc-star", "rise"), ("cgc-star", "fall")],
        )

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            manifest_entries = script._write_observed_graph_artifacts(
                output_dir,
                staged_artifacts=staged,
                manifest_entries=[],
            )
            manifest_path = output_dir / script.OBSERVED_GRAPH_ARTIFACTS_MANIFEST_FILE
            payload = json.loads(manifest_path.read_text())

            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(
                [entry["path"] for entry in payload["artifacts"]],
                [
                    "observed_graph_artifacts/d__f5t2__cgc-star__fall.npz",
                    "observed_graph_artifacts/d__f5t2__cgc-star__rise.npz",
                ],
            )
            self.assertEqual(len(manifest_entries), 2)

            rise_path = output_dir / manifest_entries[1]["path"]
            with np.load(rise_path) as saved:
                np.testing.assert_array_equal(
                    saved["adjacency"],
                    np.array([[False, True], [False, False]], dtype=bool),
                )
                np.testing.assert_array_equal(
                    saved["retained_scores"],
                    np.array([[0.0, 2.5], [0.0, 0.0]], dtype=float),
                )
                np.testing.assert_array_equal(
                    saved["p_values"],
                    np.array([[1.0, 0.01], [0.2, 1.0]], dtype=float),
                )
                np.testing.assert_array_equal(
                    saved["best_lags"],
                    np.array([[0, 2], [0, 0]], dtype=int),
                )
                metadata = json.loads(saved["metadata_json"].item())
            self.assertTrue(metadata["fdr"])
            self.assertEqual(metadata["multiple_testing"], "benjamini-hochberg")
            self.assertEqual(metadata["edge_testing_family"], script.EDGE_TESTING_FAMILY)

    def test_observed_graph_artifact_manifest_marks_unadjusted_runs(self) -> None:
        script = _load_script_module()
        record = {
            "case": "A",
            "description": "artifact test",
            "recording": "F1T1",
            "fish": 1,
            "trial": 1,
            "traces": np.zeros((2, 6)),
            "mid": 1,
        }
        staged: list[dict[str, object]] = []

        script.null_rows_for_recording(
            record,
            donor_traces=None,
            estimator_factory=lambda _seed: ArtifactEstimator(),
            method="cgc",
            representations=("rise",),
            n_null_replicates=0,
            observed_artifacts=staged,
            fdr=False,
        )

        with tempfile.TemporaryDirectory() as temporary:
            manifest_entries = script._write_observed_graph_artifacts(
                Path(temporary),
                staged_artifacts=staged,
                manifest_entries=[],
            )

        self.assertFalse(manifest_entries[0]["fdr"])
        self.assertEqual(manifest_entries[0]["multiple_testing"], "unadjusted")

    def test_resume_requires_observed_graph_artifacts_for_complete_units(self) -> None:
        script = _load_script_module()
        record = {
            "case": "C",
            "recording": "F3T1",
            "description": "resume",
            "fish": 3,
            "trial": 1,
            "traces": np.zeros((2, 8)),
            "mid": 1,
        }
        rows = [
            {
                "method": "cgc",
                "case": "C",
                "recording": "F3T1",
                "representation": representation,
                "null_type": null_type,
                "replicate": 0,
            }
            for representation, null_type in (
                ("rise", "observed"),
                ("rise", "reverse_time"),
                ("fall", "observed"),
                ("fall", "reverse_time"),
            )
        ]
        manifest_entries = [
            {
                "case": "C",
                "description": "resume",
                "recording": "F3T1",
                "fish": 3,
                "trial": 1,
                "method": "cgc",
                "representation": "rise",
                "null_type": "observed",
                "replicate": 0,
                "fdr": True,
                "multiple_testing": "benjamini-hochberg",
                "edge_testing_family": script.EDGE_TESTING_FAMILY,
                "estimator": "artifact",
                "n_rois": 2,
                "path": "observed_graph_artifacts/c__f3t1__cgc__rise.npz",
            }
        ]

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            recovered, completed = script._recover_completed_units(
                rows,
                records=[record],
                methods=("cgc",),
                representations=("rise", "fall"),
                n_null_replicates=0,
                artifact_entries=manifest_entries,
                output_dir=output_dir,
            )
            self.assertEqual(recovered, [])
            self.assertEqual(completed, set())

            script._write_observed_graph_artifacts(
                output_dir,
                staged_artifacts=[
                    {
                        **manifest_entries[0],
                        "adjacency": np.array([[False, True], [False, False]], dtype=bool),
                        "retained_scores": np.array([[0.0, 1.0], [0.0, 0.0]], dtype=float),
                        "p_values": np.array([[1.0, 0.02], [0.5, 1.0]], dtype=float),
                        "best_lags": np.array([[0, 1], [0, 0]], dtype=int),
                    },
                    {
                        **manifest_entries[0],
                        "representation": "fall",
                        "path": "observed_graph_artifacts/c__f3t1__cgc__fall.npz",
                        "adjacency": np.array([[False, True], [False, False]], dtype=bool),
                        "retained_scores": np.array([[0.0, 1.0], [0.0, 0.0]], dtype=float),
                        "p_values": np.array([[1.0, 0.02], [0.5, 1.0]], dtype=float),
                        "best_lags": np.array([[0, 1], [0, 0]], dtype=int),
                    },
                ],
                manifest_entries=[],
            )
            persisted_entries = script._load_observed_graph_artifact_entries(output_dir)
            recovered, completed = script._recover_completed_units(
                rows,
                records=[record],
                methods=("cgc",),
                representations=("rise", "fall"),
                n_null_replicates=0,
                artifact_entries=persisted_entries,
                output_dir=output_dir,
            )

        self.assertEqual(recovered, rows)
        self.assertEqual(completed, {"cgc|C|F3T1"})


if __name__ == "__main__":
    unittest.main()
