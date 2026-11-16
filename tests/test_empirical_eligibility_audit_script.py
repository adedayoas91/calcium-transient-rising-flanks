import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "build_empirical_eligibility_audit.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_empirical_eligibility_audit", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


class EmpiricalEligibilityAuditScriptTests(unittest.TestCase):
    def test_audit_is_descriptive_and_records_unavailable_fields(self) -> None:
        time = np.linspace(0.0, 1.0, 40)
        traces = np.vstack([time, np.sin(2.0 * np.pi * time)])

        row = script.audit_recording(
            (3, 1),
            traces,
            n_rois_before_exclusion=3,
            artifact_frame=20,
        )

        self.assertEqual(row["recording"], "F3T1")
        self.assertEqual(row["n_rois_excluded"], 1)
        self.assertTrue(row["matched_legacy_subset"])
        self.assertTrue(row["artifact_annotation_available"])
        self.assertEqual(row["eligibility_decision"], "not assigned retrospectively")
        self.assertIn("unavailable", row["motion_audit"])
        self.assertGreater(row["absolute_normalized_drift_max"], 0.0)

    def test_nonfinite_values_are_counted_without_imputation(self) -> None:
        traces = np.array([[0.0, 1.0, np.nan], [0.0, 0.5, 1.0]])

        row = script.audit_recording(
            (1, 2),
            traces,
            n_rois_before_exclusion=2,
            artifact_frame=None,
        )

        self.assertEqual(row["nonfinite_value_count"], 1)
        self.assertEqual(
            row["transient_summary_status"],
            "not_computed_nonfinite_values",
        )
        self.assertIsNone(row["event_density_median"])


if __name__ == "__main__":
    unittest.main()
