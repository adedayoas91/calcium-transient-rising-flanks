"""Run or print the remaining publication-gate artifact pipeline.

This script orchestrates the user-run heavy gates needed before the manuscript
can promote the rise-primary empirical claim. By default it performs a dry run
and prints commands only. Pass ``--execute`` to actually run the commands.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DYNAMIC_OUTPUT_DIR = Path("outputs/validation_results/dynamic_episodic_locked")
DEFAULT_METHODS = "cgc,cgc-star"
DEFAULT_EVENT_MODES = "compressed,physical"


@dataclass(frozen=True)
class PipelineStep:
    name: str
    command: tuple[str, ...]


@dataclass(frozen=True)
class PipelineConfig:
    python: str
    methods: str = DEFAULT_METHODS
    event_modes: str = DEFAULT_EVENT_MODES
    dynamic_output_dir: Path = DEFAULT_DYNAMIC_OUTPUT_DIR
    dynamic_n_seeds: int = 20
    dynamic_n_steps: int = 1500
    dynamic_n_surrogates: int = 1000
    dynamic_tau: int | None = None
    dynamic_n_pasts: int | None = None
    dynamic_max_lag: int = 1
    dynamic_rise_waveform_length: int = 20
    dynamic_topology_mode: str = "sequence"
    fall_state_mode: str = "stochastic_independent"
    fall_initial_ceiling_fraction: float = 1.0
    edge_dropout_probability: float = 0.2
    edge_addition_probability: float = 0.02
    source_dropout_probability: float = 0.15
    source_recruitment_probability: float = 0.25
    source_recruitment_edge_probability: float = 0.25
    min_rise_run_samples: int = 1
    rise_candidate_filter: bool = False
    rise_match_min_lag: int = 1
    rise_match_max_lag: int | None = None
    rise_match_min_overlap_samples: int | None = None
    rise_match_min_overlap_fraction: float = 0.5
    rise_run_context_samples: int | None = None
    resume_dynamic: bool = False
    null_replicates: int = 20
    stability_bootstrap: int = 20
    skip_dynamic: bool = False
    skip_null: bool = False
    skip_stability: bool = False


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "src")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    env.setdefault("XDG_CACHE_HOME", "/tmp/font-cache")
    return env


def build_steps(config: PipelineConfig) -> list[PipelineStep]:
    """Build the ordered user-run publication-gate commands."""

    steps: list[PipelineStep] = []
    if not config.skip_dynamic:
        dynamic_command = [
            config.python,
            "examples/dynamic_episodic_validation.py",
            "--output-dir",
            str(config.dynamic_output_dir),
            "--n-seeds",
            str(config.dynamic_n_seeds),
            "--n-steps",
            str(config.dynamic_n_steps),
            "--n-surrogates",
            str(config.dynamic_n_surrogates),
            "--max-lag",
            str(config.dynamic_max_lag),
            "--rise-waveform-length",
            str(config.dynamic_rise_waveform_length),
            "--topology-mode",
            config.dynamic_topology_mode,
            "--fall-state-mode",
            config.fall_state_mode,
            "--fall-initial-ceiling-fraction",
            str(config.fall_initial_ceiling_fraction),
            "--edge-dropout-probability",
            str(config.edge_dropout_probability),
            "--edge-addition-probability",
            str(config.edge_addition_probability),
            "--source-dropout-probability",
            str(config.source_dropout_probability),
            "--source-recruitment-probability",
            str(config.source_recruitment_probability),
            "--source-recruitment-edge-probability",
            str(config.source_recruitment_edge_probability),
            "--min-rise-run-samples",
            str(config.min_rise_run_samples),
            "--rise-match-min-lag",
            str(config.rise_match_min_lag),
            "--rise-match-min-overlap-fraction",
            str(config.rise_match_min_overlap_fraction),
            "--event-modes",
            config.event_modes,
            "--methods",
            config.methods,
        ]
        if config.resume_dynamic:
            dynamic_command.append("--resume")
        if config.rise_candidate_filter:
            dynamic_command.append("--rise-candidate-filter")
        if config.dynamic_tau is not None:
            dynamic_command.extend(["--tau", str(config.dynamic_tau)])
        if config.dynamic_n_pasts is not None:
            dynamic_command.extend(["--n-pasts", str(config.dynamic_n_pasts)])
        if config.rise_match_max_lag is not None:
            dynamic_command.extend(
                ["--rise-match-max-lag", str(config.rise_match_max_lag)]
            )
        if config.rise_match_min_overlap_samples is not None:
            dynamic_command.extend(
                [
                    "--rise-match-min-overlap-samples",
                    str(config.rise_match_min_overlap_samples),
                ]
            )
        if config.rise_run_context_samples is not None:
            dynamic_command.extend(
                ["--rise-run-context-samples", str(config.rise_run_context_samples)]
            )
        steps.append(
            PipelineStep(
                "locked dynamic-A validation",
                tuple(dynamic_command),
            )
        )
    if not config.skip_null:
        steps.append(
            PipelineStep(
                "empirical null controls",
                (
                    config.python,
                    "examples/run_empirical_null_controls.py",
                    "--methods",
                    config.methods,
                    "--n-null-replicates",
                    str(config.null_replicates),
                ),
            )
        )
    if not config.skip_stability:
        steps.append(
            PipelineStep(
                "empirical re-estimation stability",
                (
                    config.python,
                    "examples/run_empirical_stability.py",
                    "--methods",
                    config.methods,
                    "--n-event-bootstrap",
                    str(config.stability_bootstrap),
                ),
            )
        )
    steps.extend(
        [
            PipelineStep(
                "result readiness report",
                (config.python, "examples/build_result_readiness_report.py"),
            ),
            PipelineStep(
                "manuscript evidence package",
                (config.python, "examples/build_manuscript_evidence_package.py"),
            ),
            PipelineStep(
                "todo completion audit",
                (config.python, "examples/build_todo_completion_audit.py"),
            ),
        ]
    )
    return steps


def format_command(command: tuple[str, ...]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_steps(
    steps: list[PipelineStep],
    *,
    execute: bool = False,
    cwd: Path | None = None,
) -> None:
    env = _base_env()
    for index, step in enumerate(steps, start=1):
        print(f"{index}. {step.name}")
        print(f"   {format_command(step.command)}")
        if execute:
            subprocess.run(step.command, cwd=cwd, env=env, check=True)
    if not execute:
        print("\nDry run only. Re-run with --execute to launch these commands.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--event-modes", default=DEFAULT_EVENT_MODES)
    parser.add_argument(
        "--dynamic-output-dir",
        type=Path,
        default=DEFAULT_DYNAMIC_OUTPUT_DIR,
    )
    parser.add_argument("--dynamic-n-seeds", type=int, default=20)
    parser.add_argument("--dynamic-n-steps", type=int, default=1500)
    parser.add_argument("--dynamic-n-surrogates", type=int, default=1000)
    parser.add_argument("--dynamic-max-lag", type=int, default=1)
    parser.add_argument("--dynamic-tau", type=int, default=None)
    parser.add_argument("--dynamic-n-pasts", type=int, default=None)
    parser.add_argument("--dynamic-rise-waveform-length", type=int, default=20)
    parser.add_argument(
        "--dynamic-topology-mode",
        choices=("sequence", "generated"),
        default="sequence",
    )
    parser.add_argument(
        "--fall-state-mode",
        choices=("passive_decay", "stochastic_independent"),
        default="stochastic_independent",
    )
    parser.add_argument("--fall-initial-ceiling-fraction", type=float, default=1.0)
    parser.add_argument("--edge-dropout-probability", type=float, default=0.2)
    parser.add_argument("--edge-addition-probability", type=float, default=0.02)
    parser.add_argument("--source-dropout-probability", type=float, default=0.15)
    parser.add_argument("--source-recruitment-probability", type=float, default=0.25)
    parser.add_argument(
        "--source-recruitment-edge-probability",
        type=float,
        default=0.25,
    )
    parser.add_argument("--min-rise-run-samples", type=int, default=1)
    parser.add_argument("--rise-candidate-filter", action="store_true")
    parser.add_argument("--rise-match-min-lag", type=int, default=1)
    parser.add_argument("--rise-match-max-lag", type=int, default=None)
    parser.add_argument("--rise-match-min-overlap-samples", type=int, default=None)
    parser.add_argument("--rise-match-min-overlap-fraction", type=float, default=0.5)
    parser.add_argument("--rise-run-context-samples", type=int, default=None)
    parser.add_argument(
        "--resume-dynamic",
        action="store_true",
        help="pass --resume to the locked dynamic-A validation step",
    )
    parser.add_argument("--null-replicates", type=int, default=20)
    parser.add_argument("--stability-bootstrap", type=int, default=20)
    parser.add_argument("--skip-dynamic", action="store_true")
    parser.add_argument("--skip-null", action="store_true")
    parser.add_argument("--skip-stability", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        python=args.python,
        methods=args.methods,
        event_modes=args.event_modes,
        dynamic_output_dir=args.dynamic_output_dir,
        dynamic_n_seeds=args.dynamic_n_seeds,
        dynamic_n_steps=args.dynamic_n_steps,
        dynamic_n_surrogates=args.dynamic_n_surrogates,
        dynamic_tau=args.dynamic_tau,
        dynamic_n_pasts=args.dynamic_n_pasts,
        dynamic_max_lag=args.dynamic_max_lag,
        dynamic_rise_waveform_length=args.dynamic_rise_waveform_length,
        dynamic_topology_mode=args.dynamic_topology_mode,
        fall_state_mode=args.fall_state_mode,
        fall_initial_ceiling_fraction=args.fall_initial_ceiling_fraction,
        edge_dropout_probability=args.edge_dropout_probability,
        edge_addition_probability=args.edge_addition_probability,
        source_dropout_probability=args.source_dropout_probability,
        source_recruitment_probability=args.source_recruitment_probability,
        source_recruitment_edge_probability=args.source_recruitment_edge_probability,
        min_rise_run_samples=args.min_rise_run_samples,
        rise_candidate_filter=args.rise_candidate_filter,
        rise_match_min_lag=args.rise_match_min_lag,
        rise_match_max_lag=args.rise_match_max_lag,
        rise_match_min_overlap_samples=args.rise_match_min_overlap_samples,
        rise_match_min_overlap_fraction=args.rise_match_min_overlap_fraction,
        rise_run_context_samples=args.rise_run_context_samples,
        resume_dynamic=args.resume_dynamic,
        null_replicates=args.null_replicates,
        stability_bootstrap=args.stability_bootstrap,
        skip_dynamic=args.skip_dynamic,
        skip_null=args.skip_null,
        skip_stability=args.skip_stability,
    )
    run_steps(build_steps(config), execute=args.execute, cwd=Path.cwd())


if __name__ == "__main__":
    main()
