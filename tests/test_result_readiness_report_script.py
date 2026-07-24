import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "examples" / (
        "build_result_readiness_report.py"
    )
    spec = importlib.util.spec_from_file_location("build_result_readiness_report", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ResultReadinessReportScriptTests(unittest.TestCase):
    def test_report_marks_available_and_missing_user_run_artifacts(self) -> None:
        script = _load_script_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            available = output_root / "validation_results" / "cgc" / "summary.json"
            available.parent.mkdir(parents=True)
            available.write_text("{}\n")

            rows = script.build_readiness_rows(output_root)
            summary = script.summarize_readiness(rows)
            report = script.build_markdown_report(rows, summary)

        by_path = {row["relative_path"]: row for row in rows}
        self.assertEqual(
            by_path["validation_results/cgc/summary.json"]["status"], "available"
        )
        self.assertEqual(
            by_path[
                "validation_results/dynamic_episodic_locked/summary.json"
            ]["status"],
            "missing",
        )
        self.assertEqual(
            by_path[
                "validation_results/dynamic_episodic_locked/representation_summary.csv"
            ]["status"],
            "missing",
        )
        self.assertEqual(
            by_path[
                "validation_results/dynamic_episodic_locked/rise_fall_contrasts.csv"
            ]["status"],
            "missing",
        )
        self.assertGreater(summary["missing_user_run_artifacts"], 0)
        self.assertFalse(summary["ready_for_publication_claims"])
        self.assertIn("validation_results/dynamic_episodic_locked/summary.json", report)
        self.assertIn("empirical_null_controls/null_control_contrasts.csv", report)


if __name__ == "__main__":
    unittest.main()
