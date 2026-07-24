import importlib.util
import sys
import unittest
from pathlib import Path


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "examples" / (
        "run_publication_gate_pipeline.py"
    )
    spec = importlib.util.spec_from_file_location("run_publication_gate_pipeline", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PublicationGatePipelineScriptTests(unittest.TestCase):
    def test_builds_ordered_publication_gate_commands(self) -> None:
        script = _load_script_module()

        config = script.PipelineConfig(python="python", methods="cgc,cgc-star")
        steps = script.build_steps(config)
        names = [step.name for step in steps]

        self.assertEqual(
            names,
            [
                "locked dynamic-A validation",
                "empirical null controls",
                "empirical re-estimation stability",
                "result readiness report",
                "manuscript evidence package",
                "todo completion audit",
            ],
        )
        dynamic = steps[0].command
        self.assertIn("--methods", dynamic)
        self.assertIn("cgc,cgc-star", dynamic)
        self.assertIn("outputs/validation_results/dynamic_episodic_locked", dynamic)
        self.assertIn("--fall-state-mode", dynamic)
        self.assertIn("stochastic_independent", dynamic)
        self.assertIn("--topology-mode", dynamic)
        self.assertIn("sequence", dynamic)
        self.assertIn("--source-recruitment-probability", dynamic)
        self.assertIn("--max-lag", dynamic)
        self.assertIn("--min-rise-run-samples", dynamic)
        self.assertIn("--rise-match-min-overlap-fraction", dynamic)

    def test_publication_gate_forwards_optional_rise_candidate_switches(self) -> None:
        script = _load_script_module()

        config = script.PipelineConfig(
            python="python",
            skip_null=True,
            skip_stability=True,
            min_rise_run_samples=7,
            rise_candidate_filter=True,
            dynamic_tau=2,
            dynamic_n_pasts=5,
            rise_match_max_lag=4,
            rise_match_min_overlap_samples=6,
            rise_run_context_samples=5,
        )
        steps = script.build_steps(config)
        dynamic = steps[0].command

        self.assertIn("--rise-candidate-filter", dynamic)
        self.assertIn("--tau", dynamic)
        self.assertIn("--n-pasts", dynamic)
        self.assertIn("2", dynamic)
        self.assertIn("--rise-match-max-lag", dynamic)
        self.assertIn("4", dynamic)
        self.assertIn("--rise-match-min-overlap-samples", dynamic)
        self.assertIn("6", dynamic)
        self.assertIn("--rise-run-context-samples", dynamic)
        self.assertIn("5", dynamic)

    def test_skip_options_remove_only_heavy_gate_steps(self) -> None:
        script = _load_script_module()

        config = script.PipelineConfig(
            python="python",
            skip_dynamic=True,
            skip_null=True,
            skip_stability=True,
        )
        steps = script.build_steps(config)

        self.assertEqual(
            [step.name for step in steps],
            [
                "result readiness report",
                "manuscript evidence package",
                "todo completion audit",
            ],
        )

    def test_formats_commands_without_shell_execution(self) -> None:
        script = _load_script_module()

        formatted = script.format_command(
            ("python", "script.py", "--methods", "cgc,cgc-star")
        )

        self.assertEqual(formatted, "python script.py --methods cgc,cgc-star")


if __name__ == "__main__":
    unittest.main()
