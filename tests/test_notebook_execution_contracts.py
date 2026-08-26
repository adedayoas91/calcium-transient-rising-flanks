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
    "05_calibration_onset_run.ipynb": {
        "examples/calibration_onset.py",
    },
    "06_dynamic_extensions_run.ipynb": {
        "examples/dynamic_extensions.py",
    },
    "simulations/lpcmci.ipynb": {
        "examples/simulation_baselines.py",
    },
    "simulations/oasis.ipynb": {
        "examples/simulation_baselines.py",
    },
    "motorneurons/lpcmci.ipynb": {
        "examples/empirical_baselines.py",
    },
    "motorneurons/oasis.ipynb": {
        "examples/empirical_baselines.py",
    },
    "08_revision_campaign_run.ipynb": {
        "examples/run_revision_campaign.py",
    },
    "fdr_reestimation.ipynb": {
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
    "05_calibration_onset_run.ipynb": ("RUN_THRESHOLD", "RUN_ONSET"),
    "06_dynamic_extensions_run.ipynb": (
        "RUN_MIXED_FALL",
        "RUN_HYBRID",
    ),
    "08_revision_campaign_run.ipynb": ("RUN_CAMPAIGN",),
    "fdr_reestimation.ipynb": ("RUN_BH", "RUN_UNADJUSTED"),
    "Temporal_resolvability_screen.ipynb": ("RUN_SCREEN",),
}
READY_TO_RUN_TOGGLES = {
    "simulations/lpcmci.ipynb": "RUN_LPCMCI",
    "simulations/oasis.ipynb": "RUN_OASIS",
    "motorneurons/lpcmci.ipynb": "RUN_LPCMCI",
    "motorneurons/oasis.ipynb": "RUN_OASIS",
}
PORTABLE_RUNNER_NOTEBOOKS = (
    "simulations/05_calibration_onset_run.ipynb",
    "simulations/06_dynamic_extensions_run.ipynb",
    "simulations/08_revision_campaign_run.ipynb",
    "simulations/lpcmci.ipynb",
    "simulations/oasis.ipynb",
    "motorneurons/fdr_reestimation.ipynb",
    "motorneurons/lpcmci.ipynb",
    "motorneurons/oasis.ipynb",
)


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


def _notebook_key(path: Path) -> str:
    return path.relative_to(NOTEBOOK_ROOT).as_posix()


class NotebookExecutionContractTests(unittest.TestCase):
    def test_notebooks_have_valid_json_and_compilable_code_cells(self) -> None:
        for path in _all_notebooks():
            with self.subTest(notebook=path.name):
                code_cells = _code_cells(path)
                self.assertTrue(code_cells, msg=f"{path.name} has no code cells")
                for index, source in enumerate(code_cells):
                    compile(source, f"{path.name}#cell{index}", "exec")

    def test_long_running_notebooks_expose_live_progress(self) -> None:
        for path in _all_notebooks():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not payload.get("metadata", {}).get("rising_flanks", {}).get(
                "long_running"
            ):
                continue
            source = "\n".join(_code_cells(path))
            with self.subTest(notebook=_notebook_key(path)):
                self.assertTrue(
                    "format_progress" in source or "progress_path" in source,
                    msg=f"{path.name} has no visible progress instrumentation",
                )
                if "subprocess.run" in source:
                    self.assertIn("PYTHONUNBUFFERED", source)

    def test_notebooks_do_not_disable_fdr_inline(self) -> None:
        for path in _all_notebooks():
            source = "\n".join(_code_cells(path))
            with self.subTest(notebook=path.name):
                self.assertNotIn("fdr=False", source)

    def test_runner_notebooks_forward_resume_to_long_running_scripts(self) -> None:
        for path in _all_notebooks():
            expected_scripts = RUNNER_NOTEBOOKS.get(
                _notebook_key(path), RUNNER_NOTEBOOKS.get(path.name)
            )
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
            toggles = SAFE_RUN_TOGGLES.get(
                _notebook_key(path), SAFE_RUN_TOGGLES.get(path.name, ())
            )
            if not toggles:
                continue
            source = "\n".join(_code_cells(path))
            for toggle in toggles:
                with self.subTest(notebook=path.name, toggle=toggle):
                    self.assertRegex(source, rf"\b{re.escape(toggle)}\s*=\s*False\b")

    def test_baseline_notebooks_are_ready_to_run_by_default(self) -> None:
        for relative_path, toggle in READY_TO_RUN_TOGGLES.items():
            path = NOTEBOOK_ROOT / relative_path
            source = "\n".join(_code_cells(path))
            with self.subTest(notebook=relative_path):
                self.assertRegex(source, rf"\b{re.escape(toggle)}\s*=\s*True\b")
                self.assertNotIn("--dry-run", source)

    def test_baseline_environment_setup_reproduces_shared_environment(self) -> None:
        setup_command = "uv sync --frozen --all-extras"
        documentation = (NOTEBOOK_ROOT / "exec_order.md").read_text()
        self.assertIn(setup_command, documentation)
        self.assertIn("uv run --no-sync", documentation)
        for relative_path in READY_TO_RUN_TOGGLES:
            notebook_text = (NOTEBOOK_ROOT / relative_path).read_text()
            with self.subTest(notebook=relative_path):
                self.assertIn(setup_command, notebook_text)
                self.assertNotIn("uv sync --extra pag", notebook_text)
                self.assertNotIn("uv sync --extra deconvolution", notebook_text)

    def test_runner_notebooks_find_root_from_common_jupyter_directories(self) -> None:
        package_root = NOTEBOOK_ROOT.parent.resolve()
        outer_root = package_root.parent
        for relative_path in PORTABLE_RUNNER_NOTEBOOKS:
            path = NOTEBOOK_ROOT / relative_path
            source = "\n".join(_code_cells(path))
            function_start = source.index("def find_package_root")
            assignment_start = source.index(
                "PACKAGE_ROOT = find_package_root()", function_start
            )
            namespace = {"Path": Path}
            exec(source[function_start:assignment_start], namespace)
            find_package_root = namespace["find_package_root"]
            with self.subTest(notebook=relative_path):
                self.assertNotIn("PACKAGE_ROOT = Path.cwd()", source)
                self.assertNotIn("Start this notebook from", source)
                self.assertRegex(
                    source,
                    r"OUTPUT_(?:DIR|ROOT)\s*=\s*PACKAGE_ROOT\s*/",
                )
                self.assertIn("RUNNER_ENV['PYTHONPATH']", source)
                self.assertIn("RUNNER_ENV['MPLBACKEND'] = 'Agg'", source)
                self.assertIn("env=RUNNER_ENV", source)
                self.assertIn("RUNNER_PYTHON =", source)
                self.assertNotIn("sys.executable, 'examples/", source)
                self.assertEqual(find_package_root(path.parent), package_root)
                self.assertEqual(find_package_root(package_root), package_root)
                self.assertEqual(find_package_root(outer_root), package_root)

    def test_baseline_notebooks_use_the_cgc_input_contracts(self) -> None:
        for name in ("lpcmci.ipynb", "oasis.ipynb"):
            simulation_source = "\n".join(
                _code_cells(NOTEBOOK_ROOT / "simulations" / name)
            )
            motorneuron_source = "\n".join(
                _code_cells(NOTEBOOK_ROOT / "motorneurons" / name)
            )
            with self.subTest(notebook=f"simulations/{name}"):
                self.assertIn("N_RUNS_OUTER = 10", simulation_source)
                self.assertIn("N_SEEDS = 20", simulation_source)
                self.assertIn("N_STEPS = 3000", simulation_source)
            with self.subTest(notebook=f"motorneurons/{name}"):
                self.assertIn("FLUO_TYPES = 'dff,f_smooth'", motorneuron_source)
                self.assertNotIn("--cases", motorneuron_source)

        for name in ("c-GC.ipynb", "c-GC-star.ipynb"):
            source = "\n".join(_code_cells(NOTEBOOK_ROOT / "simulations" / name))
            with self.subTest(notebook=f"simulations/{name}"):
                self.assertIn("static_input_digest(truth, fluo)", source)
                self.assertIn('RESULTS_DIR / "input_manifest.csv"', source)

        for name in ("c-GC_Motoneurons.ipynb", "c-GC-star_Motoneurons.ipynb"):
            source = "\n".join(_code_cells(NOTEBOOK_ROOT / "motorneurons" / name))
            with self.subTest(notebook=f"motorneurons/{name}"):
                self.assertIn("array_input_digest(traces)", source)
                self.assertIn('"input_digest": record["input_digest"]', source)


if __name__ == "__main__":
    unittest.main()
