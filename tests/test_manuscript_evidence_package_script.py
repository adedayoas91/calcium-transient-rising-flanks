import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "examples" / (
        "build_manuscript_evidence_package.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_manuscript_evidence_package", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ManuscriptEvidencePackageScriptTests(unittest.TestCase):
    def test_missing_readiness_rows_fall_back_to_missing_publication_gates(
        self,
    ) -> None:
        script = _load_script_module()

        gates = script.build_evidence_gate_rows([])

        self.assertEqual(len(gates), 5)
        self.assertTrue(
            all(gate["claim_impact"] == "blocks final publication claim" for gate in gates)
        )

    def test_builds_evidence_package_from_saved_artifacts(self) -> None:
        script = _load_script_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "outputs"
            out = Path(temp_dir) / "package"
            figures = Path(temp_dir) / "figures"
            figures.mkdir()
            script_figures = np.ones((3, 3, 3), dtype=float)
            try:
                import matplotlib.pyplot as plt

                plt.imsave(figures / "Rise3.png", script_figures)
                plt.imsave(figures / "Fall3.png", 1.0 - script_figures)
            except Exception:
                pass
            (root / "chen_comparison").mkdir(parents=True)
            (root / "chen_comparison" / "chen_comparison_summary.csv").write_text(
                "\n".join(
                    [
                        "dataset,method,case,fluo_type,representation,recordings,"
                        "w_ic_mean,w_rc_mean,edge_density_mean,retained_edges_mean,"
                        "total_weight_mean,delta_w_ic_rise_minus_fall_mean,"
                        "delta_w_rc_rise_minus_fall_mean",
                        "motoneurons,chen_improved_gc,published,,published,"
                        "published aggregate,1.0,,,,,,",
                        "motoneurons,rising_flank_cgc,A,A,rise,F1;F2,"
                        "0.9,0.6,0.2,10,4.0,0.4,0.3",
                        "motoneurons,rising_flank_cgc,A,A,fall,F1;F2,"
                        "0.5,0.3,0.1,5,2.0,0.4,0.3",
                        "motoneurons,cgc,dff,dff,full_trace,F1;F2,"
                        "0.8,0.5,0.3,12,5.0,,",
                    ]
                )
                + "\n"
            )
            (root / "validation_tradeoffs").mkdir()
            (root / "validation_tradeoffs" / "rise_tradeoff_contrasts.csv").write_text(
                "\n".join(
                    [
                        "method,event_mode,condition,simulator_mode,split,comparator,"
                        "delta_recall,delta_false_positive_rate,delta_precision,"
                        "delta_f1,delta_orientation_accuracy",
                        "cgc,physical,native,static,evaluation,fall,"
                        "0.3,0.1,-0.2,0.15,0.2",
                    ]
                )
                + "\n"
            )
            (root / "empirical_stats").mkdir()
            (root / "empirical_stats" / "paired_signflip_tests.csv").write_text(
                "\n".join(
                    [
                        "case,method,metric,n_pairs,mean_delta,median_delta,"
                        "n_positive,n_negative,p_two_sided_signflip",
                        "A,rising_flank_cgc,w_ic,2,0.4,0.4,2,0,0.5",
                    ]
                )
                + "\n"
            )
            (root / "graph_stability").mkdir()
            (root / "graph_stability" / "graph_stability_summary.csv").write_text(
                "\n".join(
                    [
                        "dataset,case,method,representation,n_graphs,"
                        "edge_density_mean,retained_edges_mean,total_weight_mean",
                        "motoneurons,A,rising_flank_cgc,rise,2,0.2,10,4.0",
                        "hindbrain,medial,rising_flank_cgc,rise,1,0.16,63,9.8",
                    ]
                )
                + "\n"
            )
            (root / "result_readiness").mkdir()
            (root / "result_readiness" / "result_readiness_rows.csv").write_text(
                "\n".join(
                    [
                        "category,artifact,relative_path,required_for,"
                        "user_run_required,evidence_role,path,status,size_bytes",
                        "dynamic_a_locked,locked dynamic-A summary,"
                        "validation_results/dynamic_episodic_locked/summary.json,"
                        "publication claim,True,user-run,,missing,",
                    ]
                )
                + "\n"
            )

            summary = script.build_package(
                root,
                out,
                make_figures=True,
                figures_dir=figures,
            )
            report = (out / "manuscript_evidence_report.md").read_text()
            snippet = (out / "manuscript_results_snippets.tex").read_text()
            figure_layout = (out / "manuscript_figure_layout.tex").read_text()
            final_plan = (out / "final_figure_plan.csv").read_text()
            figure_manifest_exists = (out / "figure_manifest.csv").is_file()
            figure_outputs = set(summary["figure_outputs"])
            expected_figures_exist = all(
                (out / name).is_file()
                for name in (
                    "empirical_metric_summary.png",
                    "synthetic_tradeoff_summary.png",
                    "chen_comparison_summary.png",
                    "representative_network_case_c.png",
                    "graph_support_summary.png",
                    "evidence_gate_status.png",
                    "main_results_overview.png",
                )
            )

        self.assertEqual(summary["n_chen_rows"], 4)
        self.assertEqual(summary["n_empirical_pairing_rows"], 1)
        self.assertEqual(summary["n_synthetic_tradeoff_rows"], 1)
        self.assertEqual(summary["n_dynamic_a_interpretation_rows"], 0)
        self.assertEqual(summary["n_hindbrain_graph_support_rows"], 1)
        self.assertEqual(summary["n_empirical_null_interpretation_rows"], 0)
        self.assertEqual(summary["n_empirical_stability_interpretation_rows"], 0)
        self.assertEqual(summary["n_missing_publication_gates"], 1)
        self.assertIn("Do not promote the empirical superiority claim yet", report)
        self.assertIn("Hindbrain Descriptive Extension", report)
        self.assertIn("do not report W_IC or W_RC", report)
        self.assertIn("Final empirical superiority claims therefore remain gated", snippet)
        self.assertIn("representative\\_network\\_case\\_c.png", snippet)
        self.assertIn("Final figure assembly", figure_layout)
        self.assertIn("fig_representative_network_case_c", final_plan)
        self.assertIn("fig_dynamic_a_locked", final_plan)
        self.assertIn("missing_user_run", final_plan)
        self.assertTrue(figure_manifest_exists)
        self.assertIn("chen_comparison_summary.png", figure_outputs)
        self.assertIn("representative_network_case_c.png", figure_outputs)
        self.assertIn("main_results_overview.png", figure_outputs)
        self.assertTrue(expected_figures_exist)

    def test_merges_user_run_locked_dynamic_a_artifacts(self) -> None:
        script = _load_script_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "outputs"
            out = Path(temp_dir) / "package"
            locked = root / "validation_results" / "dynamic_episodic_locked"
            locked.mkdir(parents=True)
            (locked / "summary.json").write_text("{}\n")
            (locked / "representation_summary.csv").write_text(
                "\n".join(
                    [
                        "method,event_mode,condition,representation,n,"
                        "precision_mean,recall_mean,false_positive_rate_mean,"
                        "f1_mean,orientation_accuracy_mean,edge_density_mean,"
                        "truth_edges_mean,fall_propagated_total_mean",
                        "cgc,physical,dynamic_a_noncausal_fall,rise,5,"
                        "0.8,0.9,0.1,0.85,0.7,0.2,4,0",
                        "cgc,physical,dynamic_a_noncausal_fall,fall,5,"
                        "0.1,0.2,0.05,0.13,0.3,0.1,0,0",
                        "cgc,physical,dynamic_a_noncausal_fall,fall_residual,5,"
                        "0.1,0.25,0.04,0.14,0.3,0.1,0,0",
                        "cgc,physical,dynamic_a_noncausal_fall,full,5,"
                        "0.5,0.6,0.08,0.55,0.5,0.15,4,0",
                    ]
                )
                + "\n"
            )
            (locked / "rise_fall_contrasts.csv").write_text(
                "\n".join(
                    [
                        "method,event_mode,condition,"
                        "rise_minus_fall_precision,rise_minus_fall_recall,"
                        "rise_minus_fall_false_positive_rate,"
                        "rise_minus_fall_f1,"
                        "rise_minus_fall_orientation_accuracy,"
                        "rise_minus_fall_edge_density,"
                        "rise_minus_fall_residual_precision,"
                        "rise_minus_fall_residual_recall,"
                        "rise_minus_fall_residual_false_positive_rate,"
                        "rise_minus_fall_residual_f1,"
                        "rise_minus_fall_residual_orientation_accuracy,"
                        "rise_minus_fall_residual_edge_density,"
                        "rise_minus_full_precision,rise_minus_full_recall,"
                        "rise_minus_full_false_positive_rate,rise_minus_full_f1,"
                        "rise_minus_full_orientation_accuracy,"
                        "rise_minus_full_edge_density",
                        "cgc,physical,dynamic_a_noncausal_fall,"
                        "0.7,0.7,0.05,0.72,0.4,0.1,"
                        "0.7,0.65,0.06,0.71,0.4,0.1,"
                        "0.3,0.3,0.02,0.3,0.2,0.05",
                    ]
                )
                + "\n"
            )
            (root / "result_readiness").mkdir()
            (root / "result_readiness" / "result_readiness_rows.csv").write_text(
                "\n".join(
                    [
                        "category,artifact,relative_path,required_for,"
                        "user_run_required,evidence_role,path,status,size_bytes",
                        "dynamic_a_locked,locked dynamic-A summary,"
                        "validation_results/dynamic_episodic_locked/summary.json,"
                        "publication claim,True,user-run,"
                        "validation_results/dynamic_episodic_locked/summary.json,"
                        "available,2",
                    ]
                )
                + "\n"
            )

            summary = script.build_package(root, out, make_figures=False)
            dynamic_report = (out / "dynamic_a_interpretation.csv").read_text()
            markdown = (out / "manuscript_evidence_report.md").read_text()
            snippet = (out / "manuscript_results_snippets.tex").read_text()
            manifest = (out / "figure_manifest.csv").read_text()

        self.assertEqual(summary["n_dynamic_a_interpretation_rows"], 3)
        self.assertIn("rise_gain_zero_truth_fall_with_fpr_cost", dynamic_report)
        self.assertIn("Locked Dynamic-A Validation", markdown)
        self.assertIn("Rise-gain rows against zero-truth fall controls: 2", markdown)
        self.assertIn("locked dynamic-A package produced 3 interpreted", snippet)
        self.assertIn("dynamic_a_locked_summary.png", manifest)

    def test_merges_user_run_null_and_stability_artifacts(self) -> None:
        script = _load_script_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "outputs"
            out = Path(temp_dir) / "package"
            (root / "empirical_null_controls").mkdir(parents=True)
            (
                root / "empirical_null_controls" / "null_control_contrasts.csv"
            ).write_text(
                "\n".join(
                    [
                        "case,method,representation,null_type,n_observed,n_null,"
                        "observed_w_ic_mean,null_w_ic_mean,"
                        "observed_minus_null_w_ic_mean,"
                        "p_null_ge_observed_w_ic_mean",
                        "A,cgc,rise,cyclic_shift,3,30,0.9,0.4,0.5,0.02",
                    ]
                )
                + "\n"
            )
            (root / "empirical_stability").mkdir()
            (root / "empirical_stability" / "stability_summary.csv").write_text(
                "\n".join(
                    [
                        "case,method,representation,stability_type,n_rows,n_ok,"
                        "n_skipped,stability_mean,mean_w_ic_mean,"
                        "mean_w_rc_mean,mean_edge_density_mean,"
                        "mean_retained_edges_mean,mean_total_weight_mean",
                        "A,cgc,rise,event_bootstrap,3,3,0,0.8,0.9,0.6,0.2,10,4.0",
                    ]
                )
                + "\n"
            )

            summary = script.build_package(root, out, make_figures=False)
            null_report = (out / "empirical_null_interpretation.csv").read_text()
            stability_report = (
                out / "empirical_stability_interpretation.csv"
            ).read_text()
            markdown = (out / "manuscript_evidence_report.md").read_text()
            snippet = (out / "manuscript_results_snippets.tex").read_text()

        self.assertEqual(summary["n_empirical_null_interpretation_rows"], 5)
        self.assertEqual(summary["n_empirical_stability_interpretation_rows"], 1)
        self.assertIn("nominal_observed_above_null", null_report)
        self.assertIn("high_stability", stability_report)
        self.assertIn("Empirical Null Controls", markdown)
        self.assertIn("Empirical Re-Estimation Stability", markdown)
        self.assertIn("empirical null-control package produced 5", snippet)
        self.assertIn("empirical stability package produced 1", snippet)


if __name__ == "__main__":
    unittest.main()
