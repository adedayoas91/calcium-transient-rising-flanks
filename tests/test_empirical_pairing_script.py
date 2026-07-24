import importlib.util
import tempfile
import unittest
from pathlib import Path


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "examples" / (
        "analyze_empirical_pairing.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_empirical_pairing", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EmpiricalPairingScriptTests(unittest.TestCase):
    def test_builds_paired_deltas_and_exact_signflip_tests(self) -> None:
        script = _load_script_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chen_comparison_rows.csv"
            path.write_text(
                "\n".join(
                    [
                        "dataset,method,method_internal,case,fluo_type,binary,"
                        "recording,fish,trial,trace,description,representation,"
                        "w_ic,w_rc,edge_density,retained_edges,total_weight",
                        "motoneurons,rising_flank_cgc,rising_flank_cgc,A,A,"
                        "False,F1T1,1,1,,raw,rise,0.9,0.8,0.3,6,2.0",
                        "motoneurons,rising_flank_cgc,rising_flank_cgc,A,A,"
                        "False,F1T1,1,1,,raw,fall,0.4,0.2,0.2,4,1.0",
                        "motoneurons,rising_flank_cgc,rising_flank_cgc,A,A,"
                        "False,F1T2,1,2,,raw,rise,0.8,,0.4,8,3.0",
                        "motoneurons,rising_flank_cgc,rising_flank_cgc,A,A,"
                        "False,F1T2,1,2,,raw,fall,0.6,,0.1,2,1.5",
                        "motoneurons,cgc,cgc,A,A,,F1T1,1,1,,raw,full_trace,"
                        "0.7,0.6,0.3,5,2.0",
                    ]
                )
                + "\n"
            )

            rows = script.load_comparison_rows(path)
            deltas = script.build_paired_deltas(rows)
            tests = script.summarize_paired_tests(deltas)

        w_ic_deltas = [
            row
            for row in deltas
            if row["metric"] == "w_ic" and row["case"] == "A"
        ]
        w_rc_deltas = [
            row
            for row in deltas
            if row["metric"] == "w_rc" and row["case"] == "A"
        ]
        self.assertEqual(len(w_ic_deltas), 2)
        self.assertEqual(len(w_rc_deltas), 1)
        self.assertAlmostEqual(w_ic_deltas[0]["delta_rise_minus_fall"], 0.5)

        w_ic_test = next(row for row in tests if row["metric"] == "w_ic")
        self.assertEqual(w_ic_test["n_pairs"], 2)
        self.assertEqual(w_ic_test["n_positive"], 2)
        self.assertEqual(w_ic_test["test_mode"], "exact")
        self.assertAlmostEqual(w_ic_test["p_two_sided_signflip"], 0.5)


if __name__ == "__main__":
    unittest.main()
