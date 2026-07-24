import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "examples" / (
        "build_todo_completion_audit.py"
    )
    spec = importlib.util.spec_from_file_location("build_todo_completion_audit", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TodoCompletionAuditScriptTests(unittest.TestCase):
    def test_marks_missing_user_run_gates_as_pending(self) -> None:
        script = _load_script_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            rows = script.build_audit_rows(Path(temp_dir))
            summary = script.summarize_audit(rows)
            report = script.build_markdown_report(rows, summary)

        self.assertEqual(summary["complete_gates"], 0)
        self.assertGreater(summary["pending_gates"], 0)
        self.assertFalse(summary["ready_to_close_user_run_todos"])
        self.assertIn("locked_dynamic_a_outputs", report)
        self.assertIn("pending", report)

    def test_marks_complete_when_expected_artifacts_exist(self) -> None:
        script = _load_script_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            locked = root / "validation_results" / "dynamic_episodic_locked"
            locked.mkdir(parents=True)
            (locked / "summary.json").write_text("{}\n")
            (locked / "representation_summary.csv").write_text("method\ncgc\n")
            (locked / "rise_fall_contrasts.csv").write_text("method\ncgc\n")
            null_dir = root / "empirical_null_controls"
            null_dir.mkdir()
            (null_dir / "null_control_contrasts.csv").write_text("case\nA\n")
            stability_dir = root / "empirical_stability"
            stability_dir.mkdir()
            (stability_dir / "stability_summary.csv").write_text("case\nA\n")
            evidence = root / "manuscript_evidence"
            evidence.mkdir()
            (evidence / "dynamic_a_interpretation.csv").write_text("method\ncgc\n")
            (evidence / "empirical_null_interpretation.csv").write_text("case\nA\n")
            (evidence / "empirical_stability_interpretation.csv").write_text(
                "case\nA\n"
            )
            (evidence / "summary.json").write_text(
                '{"n_missing_publication_gates": 0}\n'
            )
            (evidence / "final_figure_plan.csv").write_text(
                "figure_id,status\nfig_a,ready\nfig_b,ready\n"
            )

            rows = script.build_audit_rows(root)
            summary = script.summarize_audit(rows)

        self.assertEqual(summary["pending_gates"], 0)
        self.assertTrue(summary["ready_to_close_user_run_todos"])


if __name__ == "__main__":
    unittest.main()
