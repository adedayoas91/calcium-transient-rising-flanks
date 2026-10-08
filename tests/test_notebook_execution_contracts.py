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
        "examples/run_fast_causal_baselines.py",
        "examples/analyze_matched_dynamic_benchmark.py",
        "examples/analyze_matched_confounding_benchmark.py",
        "examples/analyze_episodic_effective_samples.py",
    },
    "simulations/lpcmci.ipynb": {
        "examples/simulation_baselines.py",
        "examples/simulation_baseline_diagnostics.py",
        "examples/run_fast_causal_baselines.py",
        "examples/analyze_matched_confounding_benchmark.py",
    },
    "simulations/oasis.ipynb": {
        "examples/simulation_baselines.py",
        "examples/simulation_baseline_diagnostics.py",
    },
    "motorneurons/lpcmci.ipynb": {
        "examples/empirical_baselines.py",
    },
    "motorneurons/oasis.ipynb": {
        "examples/empirical_baselines.py",
    },
    "simulations/pcmciplus.ipynb": {
        "examples/run_fast_causal_baselines.py",
        "examples/analyze_matched_dynamic_benchmark.py",
        "examples/analyze_matched_confounding_benchmark.py",
        "examples/analyze_fast_baseline_hypotheses.py",
        "examples/run_representation_bias_benchmarks.py",
        "examples/analyze_representation_bias_benchmarks.py",
    },
    "simulations/c-GC.ipynb": {
        "examples/dynamic_episodic_validation.py",
        "examples/run_fast_causal_baselines.py",
        "examples/analyze_matched_dynamic_benchmark.py",
        "examples/analyze_matched_confounding_benchmark.py",
        "examples/analyze_episodic_effective_samples.py",
    },
    "simulations/c-GC-star.ipynb": {
        "examples/dynamic_episodic_validation.py",
        "examples/run_fast_causal_baselines.py",
        "examples/analyze_matched_dynamic_benchmark.py",
        "examples/analyze_matched_confounding_benchmark.py",
        "examples/analyze_episodic_effective_samples.py",
    },
    "simulations/var_granger.ipynb": {
        "examples/run_fast_causal_baselines.py",
        "examples/analyze_matched_dynamic_benchmark.py",
        "examples/analyze_matched_confounding_benchmark.py",
        "examples/analyze_fast_baseline_hypotheses.py",
        "examples/run_representation_bias_benchmarks.py",
        "examples/analyze_representation_bias_benchmarks.py",
    },
    "motorneurons/pcmciplus.ipynb": {
        "examples/run_fast_causal_baselines.py",
        "examples/analyze_matched_motorneuron_benchmark.py",
        "examples/run_representation_bias_benchmarks.py",
        "examples/analyze_representation_bias_benchmarks.py",
    },
    "motorneurons/var_granger.ipynb": {
        "examples/run_fast_causal_baselines.py",
        "examples/analyze_matched_motorneuron_benchmark.py",
        "examples/run_representation_bias_benchmarks.py",
        "examples/analyze_representation_bias_benchmarks.py",
    },
    "motorneurons/c-GC_Motoneurons.ipynb": {
        "examples/run_fast_causal_baselines.py",
        "examples/analyze_matched_motorneuron_benchmark.py",
    },
    "motorneurons/c-GC-star_Motoneurons.ipynb": {
        "examples/run_fast_causal_baselines.py",
        "examples/analyze_matched_motorneuron_benchmark.py",
    },
    "08_validation_campaign_run.ipynb": {
        "examples/run_validation_campaign.py",
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
    "c-GC-star_Motoneurons.ipynb",
    "c-GC_Motoneurons.ipynb",
    "c-GC-star.ipynb",
    "c-GC.ipynb",
}
SAFE_RUN_TOGGLES = {
    "01_dynamic_episodic_validation_run.ipynb": ("RUN_VALIDATION",),
    "04_temporal_resolvability_map.ipynb": ("RUN_SMOKE",),
    "06_dynamic_extensions_run.ipynb": (
        "RUN_MATCHED_FULL_AXIS",
        "RUN_CONFOUNDING_SENSITIVITY",
        "RUN_SELECTION_AUDIT",
    ),
    "08_validation_campaign_run.ipynb": ("RUN_CAMPAIGN",),
}
FINAL_RUN_TOGGLES = {
    "simulations/02_saved_artifact_analysis_run.ipynb": ("RUN_ANALYSIS",),
    "simulations/03_publication_gate_pipeline_run.ipynb": ("RUN_PIPELINE",),
    "simulations/04_temporal_resolvability_map.ipynb": (
        "RUN_LOCKED_GRID",
        "RUN_STRICT_ANALYSIS",
    ),
    "simulations/05_calibration_onset_run.ipynb": (
        "RUN_THRESHOLD",
        "RUN_ONSET",
    ),
    "simulations/06_dynamic_extensions_run.ipynb": (
        "RUN_MIXED_FALL",
        "RUN_HYBRID",
    ),
    "motorneurons/fdr_reestimation.ipynb": ("RUN_BH", "RUN_UNADJUSTED"),
    "motorneurons/Temporal_resolvability_screen.ipynb": ("RUN_SCREEN",),
}
CANONICAL_NOTEBOOK_ORDER = (
    "simulations/04_temporal_resolvability_map.ipynb",
    "simulations/05_calibration_onset_run.ipynb",
    "simulations/06_dynamic_extensions_run.ipynb",
    "simulations/c-GC.ipynb",
    "simulations/c-GC-star.ipynb",
    "simulations/lpcmci.ipynb",
    "simulations/oasis.ipynb",
    "simulations/pcmciplus.ipynb",
    "simulations/var_granger.ipynb",
    "motorneurons/fdr_reestimation.ipynb",
    "motorneurons/Temporal_resolvability_screen.ipynb",
    "motorneurons/c-GC_Motoneurons.ipynb",
    "motorneurons/c-GC-star_Motoneurons.ipynb",
    "motorneurons/lpcmci.ipynb",
    "motorneurons/oasis.ipynb",
    "motorneurons/pcmciplus.ipynb",
    "motorneurons/var_granger.ipynb",
    "simulations/03_publication_gate_pipeline_run.ipynb",
    "simulations/02_saved_artifact_analysis_run.ipynb",
)
READY_TO_RUN_TOGGLES = {
    "simulations/lpcmci.ipynb": "RUN_LPCMCI",
    "simulations/oasis.ipynb": "RUN_OASIS",
    "motorneurons/lpcmci.ipynb": "RUN_LPCMCI",
    "motorneurons/oasis.ipynb": "RUN_OASIS",
    "simulations/pcmciplus.ipynb": "RUN_PCMCIPLUS",
    "simulations/var_granger.ipynb": "RUN_VAR_GRANGER",
    "motorneurons/pcmciplus.ipynb": "RUN_PCMCIPLUS",
    "motorneurons/var_granger.ipynb": "RUN_VAR_GRANGER",
    "simulations/c-GC.ipynb": "RUN_ADDITIONAL_BIAS_AUDITS",
    "simulations/c-GC-star.ipynb": "RUN_ADDITIONAL_BIAS_AUDITS",
    "motorneurons/c-GC_Motoneurons.ipynb": "RUN_MATCHED_FULL_AXIS",
    "motorneurons/c-GC-star_Motoneurons.ipynb": "RUN_MATCHED_FULL_AXIS",
}
PORTABLE_RUNNER_NOTEBOOKS = (
    "simulations/05_calibration_onset_run.ipynb",
    "simulations/06_dynamic_extensions_run.ipynb",
    "simulations/08_validation_campaign_run.ipynb",
    "simulations/lpcmci.ipynb",
    "simulations/oasis.ipynb",
    "motorneurons/fdr_reestimation.ipynb",
    "motorneurons/lpcmci.ipynb",
    "motorneurons/oasis.ipynb",
    "simulations/pcmciplus.ipynb",
    "simulations/var_granger.ipynb",
    "motorneurons/pcmciplus.ipynb",
    "motorneurons/var_granger.ipynb",
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
    def test_execution_guide_lists_every_notebook_in_canonical_order(self) -> None:
        guide = (NOTEBOOK_ROOT / "exec_order.md").read_text(encoding="utf-8")
        positions = [guide.index(path) for path in CANONICAL_NOTEBOOK_ORDER]
        self.assertEqual(positions, sorted(positions))

        documented = set(CANONICAL_NOTEBOOK_ORDER)
        documented.update(
            {
                "simulations/00_hyperparameter_timeseries_explorer.ipynb",
                "simulations/01_dynamic_episodic_validation_run.ipynb",
                "simulations/08_validation_campaign_run.ipynb",
            }
        )
        discovered = {
            path.relative_to(NOTEBOOK_ROOT).as_posix() for path in _all_notebooks()
        }
        self.assertEqual(documented, discovered)

    def test_retired_legacy_and_hindbrain_notebooks_are_absent(self) -> None:
        retired = (
            "Rising_flanks_WithSections.ipynb",
            "Rising_flanks_Hindbrain.ipynb",
            "c-GC_Hindbrain.ipynb",
            "c-GC-star_Hindbrain.ipynb",
        )
        for name in retired:
            self.assertFalse((NOTEBOOK_ROOT / "motorneurons" / name).exists())

    def test_saved_artifact_analysis_reads_method_specific_baselines(self) -> None:
        source = "\n".join(
            _code_cells(
                NOTEBOOK_ROOT / "simulations" / "02_saved_artifact_analysis_run.ipynb"
            )
        )
        self.assertIn("--lpcmci-input-dir", source)
        self.assertIn("--oasis-input-dir", source)
        self.assertNotIn("rising_flanks_weighted_adjacency_matrices.pkl", source)

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
            if (
                not payload.get("metadata", {})
                .get("rising_flanks", {})
                .get("long_running")
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
                    all(script in source for script in expected_scripts),
                    msg=f"{path.name} does not call every expected runner script",
                )
                self.assertIn("--resume", source)

    def test_dynamic_notebook_preserves_incompatible_resume_outputs(self) -> None:
        source = "\n".join(
            _code_cells(
                NOTEBOOK_ROOT
                / "simulations"
                / "01_dynamic_episodic_validation_run.ipynb"
            )
        )
        self.assertIn("--restart-incompatible-resume", source)

    def test_direct_heavy_notebooks_declare_resume_and_checkpoint_primitives(
        self,
    ) -> None:
        for path in _all_notebooks():
            if path.name not in DIRECT_HEAVY_NOTEBOOKS:
                continue
            source = "\n".join(_code_cells(path))
            with self.subTest(notebook=path.name):
                self.assertRegex(source, r"\bRESUME\s*=\s*True\b")
                self.assertIn("VALIDATION_VERSION", source)
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

    def test_supporting_notebooks_are_ready_for_the_final_run(self) -> None:
        for relative_path, toggles in FINAL_RUN_TOGGLES.items():
            source = "\n".join(_code_cells(NOTEBOOK_ROOT / relative_path))
            for toggle in toggles:
                with self.subTest(notebook=relative_path, toggle=toggle):
                    self.assertRegex(
                        source, rf"\b{re.escape(toggle)}\s*=\s*True\b"
                    )

        publication_source = "\n".join(
            _code_cells(
                NOTEBOOK_ROOT / "simulations/03_publication_gate_pipeline_run.ipynb"
            )
        )
        self.assertRegex(publication_source, r"\bSKIP_DYNAMIC\s*=\s*True\b")

    def test_matched_benchmark_notebooks_lock_common_estimands(self) -> None:
        simulation_paths = (
            "simulations/06_dynamic_extensions_run.ipynb",
            "simulations/pcmciplus.ipynb",
            "simulations/var_granger.ipynb",
        )
        for relative_path in simulation_paths:
            source = "\n".join(_code_cells(NOTEBOOK_ROOT / relative_path))
            with self.subTest(notebook=relative_path):
                self.assertIn("outputs/matched_dynamic_benchmark", source)
                self.assertIn("--max-lag', '1'", source)

        lpcmci_source = "\n".join(
            _code_cells(NOTEBOOK_ROOT / "simulations" / "lpcmci.ipynb")
        )
        self.assertIn("RUN_CONFOUNDING_SENSITIVITY = True", lpcmci_source)
        self.assertIn("'baseline_rows.csv', 'summary.json'", lpcmci_source)

    def test_four_method_notebooks_execute_reviewer_bias_audits(self) -> None:
        simulation_paths = (
            "simulations/c-GC.ipynb",
            "simulations/c-GC-star.ipynb",
            "simulations/pcmciplus.ipynb",
            "simulations/var_granger.ipynb",
        )
        motor_paths = (
            "motorneurons/c-GC_Motoneurons.ipynb",
            "motorneurons/c-GC-star_Motoneurons.ipynb",
            "motorneurons/pcmciplus.ipynb",
            "motorneurons/var_granger.ipynb",
        )
        expected_algorithms = ("cgc", "cgc-star", "pcmciplus", "var-granger")
        for relative_path, algorithm in zip(simulation_paths, expected_algorithms):
            source = "\n".join(_code_cells(NOTEBOOK_ROOT / relative_path))
            with self.subTest(notebook=relative_path):
                self.assertIn("RUN_REVIEWER_AUDITS = True", source)
                self.assertIn("examples/run_representation_bias_benchmarks.py", source)
                self.assertIn(
                    "examples/analyze_representation_bias_benchmarks.py", source
                )
                self.assertIn(
                    "outputs/representation_bias_benchmark/simulations", source
                )
                self.assertIn(f"'--algorithm', '{algorithm}'", source)
                self.assertIn("'--n-seeds', '20'", source)
                self.assertIn("'--n-steps', '1500'", source)
                self.assertIn("'--n-surrogates', '1000'", source)
                self.assertIn("--resume", source)
                self.assertIn("'cross_representation_rows.csv'", source)
                self.assertIn("'summary.json'", source)
        for relative_path, algorithm in zip(motor_paths, expected_algorithms):
            source = "\n".join(_code_cells(NOTEBOOK_ROOT / relative_path))
            with self.subTest(notebook=relative_path):
                self.assertIn("RUN_REVIEWER_AUDITS = True", source)
                self.assertIn("examples/run_representation_bias_benchmarks.py", source)
                self.assertIn(
                    "examples/analyze_representation_bias_benchmarks.py", source
                )
                self.assertIn(
                    "outputs/representation_bias_benchmark/motorneurons", source
                )
                self.assertIn(f"'--algorithm', '{algorithm}'", source)
                self.assertIn("'--cases', 'C,D'", source)
                self.assertIn("'--recordings', 'F3T1,F3T2,F5T2'", source)
                self.assertIn("'--n-surrogates', '6000'", source)
                self.assertIn("--resume", source)
                self.assertIn("'cross_representation_rows.csv'", source)
                self.assertIn("'summary.json'", source)

    def test_fast_baseline_notebooks_expose_all_hypotheses(self) -> None:
        for relative_path in (
            "simulations/pcmciplus.ipynb",
            "simulations/var_granger.ipynb",
        ):
            source = "\n".join(_code_cells(NOTEBOOK_ROOT / relative_path))
            notebook_text = (NOTEBOOK_ROOT / relative_path).read_text()
            with self.subTest(notebook=relative_path):
                self.assertIn("examples/analyze_fast_baseline_hypotheses.py", source)
                self.assertIn("RUN_HYPOTHESIS_ANALYSIS = True", source)
                for hypothesis in ("H1", "H2", "H3", "H4"):
                    self.assertIn(f"## {hypothesis}", notebook_text)
                for output in (
                    "h1_transient_characterization.csv",
                    "h1_characterization_summary.csv",
                    "h2_representation_recovery.csv",
                    "h2_paired_representation_contrasts.csv",
                    "h3_null_comparators.csv",
                    "h3_empirical_null_summary.csv",
                    "h4_robustness.csv",
                    "h4_robustness_summary.csv",
                    "h4_extreme_contrasts.csv",
                    "analysis-report.md",
                    "stats-appendix.md",
                ):
                    self.assertIn(output, source)
                self.assertIn("dynamic-extension", source)
                self.assertIn("1500", source)
                self.assertIn("'--n-null', '99'", source)
                self.assertIn("'--n-sweep-seeds', '20'", source)
        dynamic_source = "\n".join(
            _code_cells(
                NOTEBOOK_ROOT / "simulations" / "06_dynamic_extensions_run.ipynb"
            )
        )
        self.assertIn("--cgc-depths', '1,2,3'", dynamic_source)
        self.assertIn("--cgc-tau', '1'", dynamic_source)
        self.assertIn("examples/analyze_episodic_effective_samples.py", dynamic_source)

        for relative_path in (
            "simulations/c-GC.ipynb",
            "simulations/c-GC-star.ipynb",
        ):
            source = "\n".join(_code_cells(NOTEBOOK_ROOT / relative_path))
            with self.subTest(notebook=relative_path, audit="episodic-grid"):
                self.assertIn("examples/dynamic_episodic_validation.py", source)
                self.assertIn("'--methods', 'cgc,cgc-star'", source)
                self.assertIn("'--event-modes', 'compressed,physical'", source)
                self.assertIn("'--n-pasts', '1'", source)
                self.assertIn("--restart-incompatible-resume", source)

        confounding_paths = (
            "simulations/06_dynamic_extensions_run.ipynb",
            "simulations/c-GC.ipynb",
            "simulations/c-GC-star.ipynb",
            "simulations/pcmciplus.ipynb",
            "simulations/var_granger.ipynb",
            "simulations/lpcmci.ipynb",
        )
        for relative_path in confounding_paths:
            source = "\n".join(_code_cells(NOTEBOOK_ROOT / relative_path))
            with self.subTest(notebook=relative_path):
                self.assertIn("outputs/matched_confounding_benchmark", source)
                self.assertIn("latent_common_driver", source)
                self.assertIn("shared_observation_noise", source)
                self.assertIn("--max-lag', '1'", source)

        motor_paths = (
            "motorneurons/c-GC_Motoneurons.ipynb",
            "motorneurons/c-GC-star_Motoneurons.ipynb",
            "motorneurons/pcmciplus.ipynb",
            "motorneurons/var_granger.ipynb",
        )
        for relative_path in motor_paths:
            source = "\n".join(_code_cells(NOTEBOOK_ROOT / relative_path))
            with self.subTest(notebook=relative_path):
                self.assertIn("outputs/matched_motorneuron_benchmark", source)
                self.assertIn("--cases', 'C,D'", source)
                self.assertIn("--recordings', 'F3T1,F3T2,F5T2'", source)
                self.assertIn("--representations', 'rise,fall'", source)
                self.assertIn("--max-lag', '1'", source)
                self.assertIn("examples/analyze_matched_motorneuron_benchmark.py", source)
                self.assertIn("'baseline_rows.csv', 'summary.json'", source)

    def test_matched_analyzers_emit_strict_analysis_bundles(self) -> None:
        for filename in (
            "analyze_matched_dynamic_benchmark.py",
            "analyze_matched_confounding_benchmark.py",
            "analyze_matched_motorneuron_benchmark.py",
            "analyze_episodic_effective_samples.py",
            "analyze_fast_baseline_hypotheses.py",
        ):
            source = (NOTEBOOK_ROOT.parent / "examples" / filename).read_text()
            with self.subTest(analyzer=filename):
                self.assertIn("analysis-report.md", source)
                self.assertIn("stats-appendix.md", source)
                self.assertIn("figure-catalog.md", source)

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

    def test_oasis_notebooks_preflight_the_selected_python_environment(self) -> None:
        for relative_path in (
            "simulations/oasis.ipynb",
            "motorneurons/oasis.ipynb",
        ):
            source = "\n".join(_code_cells(NOTEBOOK_ROOT / relative_path))
            with self.subTest(notebook=relative_path):
                self.assertIn("def has_oasis", source)
                self.assertIn("import oasis", source)
                self.assertIn("uv sync --frozen --all-extras", source)

    def test_matched_notebooks_explain_primary_fit_count(self) -> None:
        for relative_path in (
            "simulations/c-GC.ipynb",
            "simulations/c-GC-star.ipynb",
            "simulations/pcmciplus.ipynb",
            "simulations/var_granger.ipynb",
        ):
            notebook_text = (NOTEBOOK_ROOT / relative_path).read_text()
            with self.subTest(notebook=relative_path):
                self.assertIn("300 primary fits per algorithm", notebook_text)

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
                self.assertIn("N_NULL = 6", simulation_source)
                self.assertIn(
                    "examples/simulation_baseline_diagnostics.py",
                    simulation_source,
                )
            with self.subTest(notebook=f"motorneurons/{name}"):
                self.assertIn("FLUO_TYPES = 'dff,f_smooth'", motorneuron_source)
                self.assertNotIn("--cases", motorneuron_source)

        oasis_source = "\n".join(
            _code_cells(NOTEBOOK_ROOT / "simulations" / "oasis.ipynb")
        )
        self.assertIn(
            "REPRESENTATIONS = 'full,deconvolved,oasis,rise,fall,fall_residual'",
            oasis_source,
        )
        self.assertIn("--restart-incompatible-resume", oasis_source)

    def test_baseline_notebooks_expose_each_reference_analysis_section(self) -> None:
        required_sections = (
            "H1 — kinetic asymmetry and transient characterization",
            "H2 — representation recovery and locked condition analysis",
            "Null and negative-comparator checks",
            "Ipsilateral consistency",
            "H4 — robustness to noise and frame-rate reduction",
        )
        for name in ("lpcmci.ipynb", "oasis.ipynb"):
            text = (NOTEBOOK_ROOT / "simulations" / name).read_text()
            with self.subTest(notebook=name):
                for section in required_sections:
                    self.assertIn(section, text)
                self.assertIn("null_comparator_rows.csv", text)
                self.assertIn("wic_delta_by_condition.csv", text)
                self.assertIn("robustness_framerate.csv", text)

        oasis_text = (NOTEBOOK_ROOT / "simulations" / "oasis.ipynb").read_text()
        self.assertIn("kept separate for downstream c-GC and c-GC*", oasis_text)

        for name in ("c-GC.ipynb", "c-GC-star.ipynb"):
            source = "\n".join(_code_cells(NOTEBOOK_ROOT / "simulations" / name))
            with self.subTest(notebook=f"simulations/{name}"):
                self.assertIn("static_input_digest(truth, fluo)", source)
                self.assertIn('RESULTS_DIR / "input_manifest.csv"', source)
                self.assertIn('os.environ.get("RF_N_STEPS", "3000")', source)
                self.assertIn('os.environ.get("RF_N_SEEDS", "20")', source)
                self.assertIn(
                    'GRID_REPS = ["full", "deconvolved", "rise", "fall", "fall_residual"]',
                    source,
                )
                self.assertIn("all_grid_records.extend", source)
                self.assertIn("grid_df = pd.DataFrame(all_grid_records)", source)
                self.assertIn('"expected_grid_rows": int(TOTAL_GRID_FITS)', source)

        for name in ("c-GC_Motoneurons.ipynb", "c-GC-star_Motoneurons.ipynb"):
            source = "\n".join(_code_cells(NOTEBOOK_ROOT / "motorneurons" / name))
            with self.subTest(notebook=f"motorneurons/{name}"):
                self.assertIn("array_input_digest(traces)", source)
                self.assertIn('"input_digest": record["input_digest"]', source)
                self.assertIn(
                    'REPRESENTATIONS = ("full", "deconvolved", "rise", "fall", "fall_residual")',
                    source,
                )
                self.assertIn('os.environ.get("RF_FLUO_TYPES", "dff,f_smooth")', source)
                self.assertIn(
                    'os.environ.get("RF_RECORDINGS", "F3T1,F3T2,F5T2")', source
                )
                self.assertIn("expected - actual", source)
                self.assertIn(
                    "total_fits = len(records) * len(REPRESENTATIONS)", source
                )
                self.assertIn("representation_inputs[representation]", source)
                self.assertIn('"expected_fit_count": total_fits', source)


if __name__ == "__main__":
    unittest.main()
