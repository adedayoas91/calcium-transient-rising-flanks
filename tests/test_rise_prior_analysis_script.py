import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "analyze_rise_prior_ablation.py"
    spec = importlib.util.spec_from_file_location("analyze_rise_prior_ablation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RisePriorAnalysisScriptTests(unittest.TestCase):
    def test_exact_sign_flip_detects_zero_and_consistent_differences(self) -> None:
        script = _load_script_module()

        self.assertEqual(script.exact_sign_flip_pvalue(np.zeros(4)), 1.0)
        self.assertEqual(script.exact_sign_flip_pvalue(np.ones(4)), 0.125)

    def test_holm_adjustment_is_monotone_in_sorted_p_values(self) -> None:
        script = _load_script_module()

        adjusted = script.holm_adjust([0.01, 0.04, 0.03])

        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])

    def test_seed_means_average_folds_before_comparison(self) -> None:
        script = _load_script_module()
        rows = []
        for seed in (1, 2):
            for fold in (0, 1):
                rows.append(
                    {
                        "condition": "clean",
                        "deadband": 0,
                        "arm": "unrestricted",
                        "seed": seed,
                        "fold": fold,
                        "f1": 0.1 * seed,
                    }
                )
                rows.append(
                    {
                        "condition": "clean",
                        "deadband": 0,
                        "arm": "robust_hard",
                        "seed": seed,
                        "fold": fold,
                        "f1": 0.1 * seed + 0.05,
                    }
                )

        differences = script.paired_seed_differences(
            rows,
            condition="clean",
            deadband=0,
            arm="robust_hard",
            metric="f1",
        )

        np.testing.assert_allclose(differences, [0.05, 0.05])


if __name__ == "__main__":
    unittest.main()
