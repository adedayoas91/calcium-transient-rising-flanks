import importlib.util
import tempfile
import unittest
from pathlib import Path


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "examples" / (
        "analyze_false_positive_tradeoffs.py"
    )
    spec = importlib.util.spec_from_file_location(
        "analyze_false_positive_tradeoffs", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FalsePositiveTradeoffScriptTests(unittest.TestCase):
    def test_tradeoff_summary_normalizes_legacy_and_dynamic_columns(self) -> None:
        script = _load_script_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "cgc"
            legacy.mkdir()
            (legacy / "grid_runs.csv").write_text(
                "\n".join(
                    [
                        "condition,seed,representation,precision,recall,fpr,orientation,f1,split",
                        "native,1,rise,0.8,1.0,0.2,0.9,0.88,evaluation",
                        "native,1,fall,0.6,0.4,0.1,0.7,0.48,evaluation",
                    ]
                )
                + "\n"
            )
            dynamic = root / "dynamic"
            dynamic.mkdir()
            (dynamic / "dynamic_grid_runs.csv").write_text(
                "\n".join(
                    [
                        "event_mode,condition,simulator_mode,seed,representation,precision,recall,false_positive_rate,f1,orientation_accuracy,edge_density",
                        "physical,dynamic_a,episodic_dynamic,1,rise,0.7,0.8,0.3,0.74,0.8,0.5",
                        "physical,dynamic_a,episodic_dynamic,1,full,0.5,0.6,0.2,0.54,0.6,0.4",
                    ]
                )
                + "\n"
            )

            rows = script.load_tradeoff_rows(root)
            summaries = script.summarize_by_representation(rows)
            contrasts = script.build_tradeoff_contrasts(summaries)

        self.assertEqual(len(rows), 4)
        self.assertEqual(len(summaries), 4)
        legacy_contrast = next(row for row in contrasts if row["method"] == "cgc")
        self.assertEqual(legacy_contrast["comparator"], "fall")
        self.assertAlmostEqual(legacy_contrast["delta_recall"], 0.6)
        self.assertAlmostEqual(legacy_contrast["delta_false_positive_rate"], 0.1)
        dynamic_contrast = next(row for row in contrasts if row["method"] == "dynamic")
        self.assertEqual(dynamic_contrast["event_mode"], "physical")
        self.assertEqual(dynamic_contrast["simulator_mode"], "episodic_dynamic")
        self.assertEqual(dynamic_contrast["comparator"], "full")


if __name__ == "__main__":
    unittest.main()
