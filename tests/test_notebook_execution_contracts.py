import json
import re
import unittest
from pathlib import Path


NOTEBOOK_ROOT = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOK_DIRS = (
    NOTEBOOK_ROOT / "simulations",
    NOTEBOOK_ROOT / "motorneurons",
)
RUNNER_NOTEBOOKS = {
    "01_dynamic_episodic_validation_run.ipynb": {
        "examples/dynamic_episodic_validation.py",
    },
    "03_publication_gate_pipeline_run.ipynb": {
        "examples/run_publication_gate_pipeline.py",
    },
    "04_temporal_resolvability_map.ipynb": {
        "examples/temporal_resolvability_map.py",
        "examples/analyze_temporal_resolvability_map.py",
    },
    "05_reviewer_calibration_onset_run.ipynb": {
        "examples/reviewer_calibration_onset.py",
    },
    "06_reviewer_dynamic_extensions_run.ipynb": {
        "examples/reviewer_dynamic_extensions.py",
    },
    "07_reviewer_baseline_benchmarks_run.ipynb": {
        "examples/reviewer_baseline_benchmarks.py",
    },
    "08_reviewer_revision_campaign_run.ipynb": {
        "examples/run_reviewer_revision_campaign.py",
    },
    "Reviewer_FDR_reestimation.ipynb": {
        "examples/run_empirical_null_controls.py",
    },
    "Temporal_resolvability_screen.ipynb": {
        "examples/motorneuron_temporal_screen.py",
    },
}
DIRECT_HEAVY_NOTEBOOKS = {
    "00_hyperparameter_timeseries_explorer.ipynb",
    "Rising_flanks_Hindbrain.ipynb",
    "Rising_flanks_WithSections.ipynb",
    "c-GC-star_Hindbrain.ipynb",
    "c-GC-star_Motoneurons.ipynb",
    "c-GC_Hindbrain.ipynb",
    "c-GC_Motoneurons.ipynb",
    "c-GC-star.ipynb",
    "c-GC.ipynb",
}
SAFE_RUN_TOGGLES = {
    "01_dynamic_episodic_validation_run.ipynb": ("RUN_VALIDATION",),
    "02_saved_artifact_analysis_run.ipynb": ("RUN_ANALYSIS",),
    "03_publication_gate_pipeline_run.ipynb": ("RUN_PIPELINE",),
    "04_temporal_resolvability_map.ipynb": (
        "RUN_SMOKE",
        "RUN_LOCKED_GRID",
        "RUN_STRICT_ANALYSIS",
    ),
    "05_reviewer_calibration_onset_run.ipynb": ("RUN_THRESHOLD", "RUN_ONSET"),
    "06_reviewer_dynamic_extensions_run.ipynb": (
        "RUN_MIXED_FALL",
        "RUN_HYBRID",
    ),
    "07_reviewer_baseline_benchmarks_run.ipynb": ("RUN_BASELINES",),
    "08_reviewer_revision_campaign_run.ipynb": ("RUN_CAMPAIGN",),
    "Reviewer_FDR_reestimation.ipynb": ("RUN_BH", "RUN_UNADJUSTED"),
    "Temporal_resolvability_screen.ipynb": ("RUN_SCREEN",),
}


def _all_notebooks() -> list[Path]:
    notebooks: list[Path] = []
    for directory in NOTEBOOK_DIRS:
        notebooks.extend(sorted(directory.glob("*.ipynb")))
    return notebooks


def _code_cells(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in payload.get("cells", [])
        if cell.get("cell_type") == "code"
    ]


class NotebookExecutionContractTests(unittest.TestCase):
    def test_notebooks_have_valid_json_and_compilable_code_cells(self) -> None:
        for path in _all_notebooks():
            with self.subTest(notebook=path.name):
                code_cells = _code_cells(path)
                self.assertTrue(code_cells, msg=f"{path.name} has no code cells")
                for index, source in enumerate(code_cells):
                    compile(source, f"{path.name}#cell{index}", "exec")

    def test_notebooks_do_not_disable_fdr_inline(self) -> None:
        for path in _all_notebooks():
            source = "\n".join(_code_cells(path))
            with self.subTest(notebook=path.name):
                self.assertNotIn("fdr=False", source)

    def test_runner_notebooks_forward_resume_to_long_running_scripts(self) -> None:
        for path in _all_notebooks():
            expected_scripts = RUNNER_NOTEBOOKS.get(path.name)
            if expected_scripts is None:
                continue
            source = "\n".join(_code_cells(path))
            with self.subTest(notebook=path.name):
                self.assertTrue(
                    any(script in source for script in expected_scripts),
                    msg=f"{path.name} does not call the expected runner script",
                )
                self.assertIn("--resume", source)

    def test_direct_heavy_notebooks_declare_resume_and_checkpoint_primitives(self) -> None:
        for path in _all_notebooks():
            if path.name not in DIRECT_HEAVY_NOTEBOOKS:
                continue
            source = "\n".join(_code_cells(path))
            with self.subTest(notebook=path.name):
                self.assertRegex(source, r"\bRESUME\s*=\s*True\b")
                self.assertIn("IMPLEMENTATION_REVISION", source)
                self.assertTrue(
                    any(
                        marker in source
                        for marker in (
                            "atomic_write_json",
                            "atomic_write_pickle",
                            "JsonUnitCheckpointStore",
                        )
                    ),
                    msg=f"{path.name} is missing an explicit checkpoint primitive",
                )

    def test_safe_run_toggles_remain_disabled_by_default(self) -> None:
        for path in _all_notebooks():
            toggles = SAFE_RUN_TOGGLES.get(path.name, ())
            if not toggles:
                continue
            source = "\n".join(_code_cells(path))
            for toggle in toggles:
                with self.subTest(notebook=path.name, toggle=toggle):
                    self.assertRegex(source, rf"\b{re.escape(toggle)}\s*=\s*False\b")


if __name__ == "__main__":
    unittest.main()
