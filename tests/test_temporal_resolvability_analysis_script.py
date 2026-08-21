import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "examples" / (
        "analyze_temporal_resolvability_map.py"
    )
    spec = importlib.util.spec_from_file_location(
        "analyze_temporal_resolvability_map",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TemporalResolvabilityAnalysisScriptTests(unittest.TestCase):
    def test_exact_sign_flip_handles_zero_and_consistent_differences(self) -> None:
        script = _load_script_module()

        self.assertEqual(script.exact_sign_flip_pvalue(np.zeros(4)), 1.0)
        self.assertEqual(script.exact_sign_flip_pvalue(np.ones(4)), 0.125)

    def test_holm_adjustment_is_monotone_in_sorted_p_values(self) -> None:
        script = _load_script_module()

        adjusted = script.holm_adjust([0.01, 0.04, 0.03])

        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])

    def test_seed_cell_rows_average_folds_and_acquisition_phases(self) -> None:
        script = _load_script_module()
        rows = []
        for fold in (0, 1):
            for phase in (0, 1):
                row = {
                    "regime": "clean_homogeneous",
                    "signal_layer": script.PRIMARY_LAYER,
                    "native_delay_frames": 2,
                    "downsample": 2,
                    "observed_delay_frames": 1.0,
                    "deadband_frames": 0,
                    "seed": 1,
                }
                for metric in script.SEED_METRICS:
                    row[metric] = float(fold + phase)
                rows.append(row)

        aggregated = script.seed_cell_rows(rows)

        self.assertEqual(len(aggregated), 1)
        self.assertEqual(aggregated[0]["unconditional_direction_accuracy"], 1.0)

    def test_writers_derive_seed_and_gate_counts_from_artifacts(self) -> None:
        script = _load_script_module()
        cell = {
            "signal_layer": script.PRIMARY_LAYER,
            "deadband_frames": 0,
            "n_seeds": 2,
            "viable": False,
            "regime": "clean_homogeneous",
        }
        seed_rows = [
            {
                "seed": seed,
                "signal_layer": script.PRIMARY_LAYER,
                "deadband_frames": 0,
                "native_delay_frames": 1,
                "downsample": 1,
            }
            for seed in (3, 4)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script.write_figure_catalog(root / "figures.md", [cell])
            script.write_stats_appendix(root / "stats.md", [], seed_rows)

            figure_text = (root / "figures.md").read_text()
            stats_text = (root / "stats.md").read_text()

        self.assertIn("n = 2", figure_text)
        self.assertIn("No primary cell passes", figure_text)
        self.assertIn("seeds 3–4", stats_text)
        self.assertIn("2^2 assignments", stats_text)


if __name__ == "__main__":
    unittest.main()
