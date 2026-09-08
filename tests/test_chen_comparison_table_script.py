import importlib.util
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "examples" / (
        "build_chen_comparison_table.py"
    )
    spec = importlib.util.spec_from_file_location("build_chen_comparison_table", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ChenComparisonTableScriptTests(unittest.TestCase):
    def test_builds_rows_from_all_method_specific_motorneuron_outputs(self) -> None:
        script = _load_script_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lpcmci = root / "lpcmci"
            oasis = root / "oasis"
            lpcmci.mkdir()
            oasis.mkdir()
            cgc_rows = [
                {
                    "dataset": "motoneurons",
                    "recording": "F1T1",
                    "fish": 1,
                    "trial": 1,
                    "fluo_type": "dff",
                    "method": "cgc",
                    "n_nodes": 4,
                    "mid": 2,
                    "w_ic": 1.0,
                    "w_rc": None,
                    "edge_density": 0.5,
                    "retained_edges": 1,
                    "edge_opportunities": 2,
                    "total_weight": 1.0,
                    "representation": "full",
                },
                {
                    "dataset": "motoneurons",
                    "recording": "F1T1",
                    "fish": 1,
                    "trial": 1,
                    "fluo_type": "dff",
                    "method": "cgc",
                    "representation": "deconvolved",
                    "n_nodes": 4,
                    "mid": 2,
                    "w_ic": 0.5,
                    "w_rc": None,
                    "edge_density": 0.25,
                    "retained_edges": 1,
                    "edge_opportunities": 2,
                    "total_weight": 0.5,
                },
                {
                    "dataset": "motoneurons",
                    "recording": "F1T1",
                    "fish": 1,
                    "trial": 1,
                    "fluo_type": "dff",
                    "method": "cgc",
                    "representation": "rise",
                    "w_ic": 0.9,
                    "w_rc": 0.7,
                    "edge_density": 0.4,
                    "retained_edges": 4,
                    "total_weight": 3.0,
                },
                {
                    "dataset": "motoneurons",
                    "recording": "F1T1",
                    "fish": 1,
                    "trial": 1,
                    "fluo_type": "dff",
                    "method": "cgc",
                    "representation": "fall",
                    "w_ic": 0.4,
                    "w_rc": 0.2,
                    "edge_density": 0.2,
                    "retained_edges": 2,
                    "total_weight": 1.0,
                },
            ]
            with (root / "cgc_motoneurons_summary_rows.pkl").open("wb") as file:
                pickle.dump(cgc_rows, file)
            with (root / "cgc_star_motoneurons_summary_rows.pkl").open("wb") as file:
                pickle.dump([{**cgc_rows[0], "method": "cgc-star"}], file)
            (lpcmci / "graph_summary_rows.csv").write_text(
                "fluo_type,recording,fish,trial,method,representation,"
                "skeleton_w_ic,skeleton_edge_density,skeleton_retained_edges\n"
                "dff,F1T1,1,1,lpcmci,rise,0.8,0.3,3\n"
                "dff,F1T1,1,1,lpcmci,fall,0.3,0.2,2\n"
            )
            (oasis / "graph_summary_rows.csv").write_text(
                "fluo_type,recording,fish,trial,method,representation,w_ic,w_rc,"
                "edge_density,retained_edges,total_weight\n"
                "dff,F1T1,1,1,cgc,oasis_spikes,0.7,0.6,0.2,2,1.5\n"
                "dff,F1T1,1,1,cgc-star,oasis_spikes,0.6,0.5,0.2,2,1.4\n"
            )

            rows = script.build_comparison_rows(
                root,
                lpcmci_input_dir=lpcmci,
                oasis_input_dir=oasis,
            )
            aggregates = script.aggregate_rows(rows)

        methods = {row["method"] for row in rows}
        self.assertIn("chen_improved_gc", methods)
        self.assertIn("cgc", methods)
        self.assertIn("cgc-star", methods)
        self.assertIn("lpcmci", methods)
        self.assertIn("oasis+cgc", methods)
        self.assertIn("oasis+cgc-star", methods)
        self.assertNotIn("rising_flank_cgc", methods)
        cgc_rows = [row for row in rows if row["method"] == "cgc"]
        self.assertEqual(len(cgc_rows), 4)
        self.assertEqual(cgc_rows[0]["representation"], "full_trace")
        rise = next(
            row
            for row in rows
            if row["method"] == "cgc" and row.get("representation") == "rise"
        )
        fall = next(
            row
            for row in rows
            if row["method"] == "cgc" and row.get("representation") == "fall"
        )
        self.assertGreater(rise["delta_w_ic_rise_minus_fall"], 0.0)
        self.assertGreater(fall["delta_w_ic_rise_minus_fall"], 0.0)
        self.assertTrue(any(row["method"] == "cgc" for row in aggregates))

    def test_optionally_ingests_direct_chen_bvgc_and_mvgc_matrices(self) -> None:
        script = _load_script_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix_dir = root / "chen_direct"
            matrix_dir.mkdir()
            bvgc = np.array(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.8],
                    [0.0, 0.0, 0.2, 0.0],
                ]
            )
            mvgc = np.array(
                [
                    [0.0, 0.0, 0.3, 0.0],
                    [0.0, 0.0, 0.0, 0.4],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )
            np.savetxt(matrix_dir / "caseA_bvgc.csv", bvgc, delimiter=",")
            np.save(matrix_dir / "caseA_mvgc.npy", mvgc)
            manifest = matrix_dir / "manifest.csv"
            manifest.write_text(
                "\n".join(
                    [
                        "path,method,case,recording,fish,trial,fluo_type,"
                        "representation,mid,binary,source_note",
                        "caseA_bvgc.csv,BVGC,A,F1T1,1,1,dff,published_direct,"
                        "2,false,supplied direct BVGC",
                        "caseA_mvgc.npy,MVGC,A,F1T1,1,1,dff,published_direct,"
                        "2,true,supplied direct MVGC",
                    ]
                )
                + "\n"
            )

            rows = script.build_comparison_rows(
                root,
                include_published_chen=False,
                chen_matrix_manifest=manifest,
            )
            aggregates = script.aggregate_rows(rows)

        methods = {row["method"] for row in rows}
        self.assertEqual(methods, {"chen_bvgc", "chen_mvgc"})
        self.assertTrue(all(row["representation"] == "published_direct" for row in rows))
        self.assertTrue(all(row["w_ic"] is not None for row in rows))
        self.assertTrue(any(row["method"] == "chen_bvgc" for row in aggregates))
        self.assertTrue(any(row["method"] == "chen_mvgc" for row in aggregates))


if __name__ == "__main__":
    unittest.main()
