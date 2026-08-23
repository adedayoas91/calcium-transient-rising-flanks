"""Run the complete reviewer-revision experiment campaign in dependency order.

Every scientific stage is invoked with its own ``--resume`` flag.  This driver
also skips only outputs whose ``summary.json`` explicitly reports
``status=complete``; a stale or partial directory is resumed, not trusted.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path("outputs/reviewer_revision_campaign")
CAMPAIGN_STAGES = (
    "empirical_fdr_bh",
    "empirical_fdr_unadjusted",
    "threshold_calibration",
    "mixed_fall",
    "lpcmci_oasis",
    "hybrid_event",
    "adaptive_onset",
)


@dataclass(frozen=True)
class Stage:
    name: str
    command: tuple[str, ...]
    summary_path: Path


def _parse_stage_names(value: str | None) -> set[str]:
    if value is None:
        return set()
    names = {item.strip() for item in value.split(",") if item.strip()}
    invalid = sorted(names - set(CAMPAIGN_STAGES))
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unsupported campaign stages: {', '.join(invalid)}"
        )
    return names


def _summary_is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "complete"


def build_stages(
    *,
    python: str,
    output_root: Path,
    seeds: str,
    dynamic_n_seeds: int,
    n_surrogates: int,
) -> tuple[Stage, ...]:
    """Return immutable commands; long-running scripts always receive resume."""

    empirical_common = (
        python,
        "examples/run_empirical_null_controls.py",
        "--cases",
        "C,D",
        "--recordings",
        "F3T1,F3T2,F5T2",
        "--representations",
        "rise,fall",
        "--methods",
        "cgc,cgc-star",
        "--n-null-replicates",
        "0",
        "--n-estimator-surrogates",
        str(n_surrogates),
        "--alpha",
        "0.05",
        "--event-mode",
        "physical",
        "--seed",
        "10",
        "--resume",
    )
    bh_dir = output_root / "empirical_fdr" / "bh"
    unadjusted_dir = output_root / "empirical_fdr" / "unadjusted"
    threshold_dir = output_root / "threshold_calibration"
    mixed_fall_dir = output_root / "mixed_fall"
    baselines_dir = output_root / "lpcmci_oasis"
    hybrid_dir = output_root / "hybrid_event"
    onset_dir = output_root / "adaptive_onset"
    return (
        Stage(
            "empirical_fdr_bh",
            (*empirical_common, "--output-dir", str(bh_dir)),
            bh_dir / "summary.json",
        ),
        Stage(
            "empirical_fdr_unadjusted",
            (
                *empirical_common,
                "--no-fdr",
                "--output-dir",
                str(unadjusted_dir),
            ),
            unadjusted_dir / "summary.json",
        ),
        Stage(
            "threshold_calibration",
            (
                python,
                "examples/reviewer_calibration_onset.py",
                "--components",
                "threshold",
                "--seeds",
                seeds,
                "--n-estimator-surrogates",
                str(n_surrogates),
                "--output-dir",
                str(threshold_dir),
                "--resume",
            ),
            threshold_dir / "summary.json",
        ),
        Stage(
            "mixed_fall",
            (
                python,
                "examples/reviewer_dynamic_extensions.py",
                "--methods",
                "cgc,cgc-star",
                "--grid",
                "lag1_context1",
                "--n-seeds",
                str(dynamic_n_seeds),
                "--n-surrogates",
                str(n_surrogates),
                "--output-dir",
                str(mixed_fall_dir),
                "--resume",
            ),
            mixed_fall_dir / "summary.json",
        ),
        Stage(
            "lpcmci_oasis",
            (
                python,
                "examples/reviewer_baseline_benchmarks.py",
                "--seeds",
                seeds,
                "--n-cgc-surrogates",
                str(n_surrogates),
                "--output-dir",
                str(baselines_dir),
                "--resume",
            ),
            baselines_dir / "summary.json",
        ),
        Stage(
            "hybrid_event",
            (
                python,
                "examples/reviewer_dynamic_extensions.py",
                "--methods",
                "cgc,cgc-star",
                "--grid",
                "lag2_context2,bout_bounded_lag3_context4",
                "--n-seeds",
                str(dynamic_n_seeds),
                "--n-surrogates",
                str(n_surrogates),
                "--output-dir",
                str(hybrid_dir),
                "--resume",
            ),
            hybrid_dir / "summary.json",
        ),
        Stage(
            "adaptive_onset",
            (
                python,
                "examples/reviewer_calibration_onset.py",
                "--components",
                "onset",
                "--seeds",
                seeds,
                "--output-dir",
                str(onset_dir),
                "--resume",
            ),
            onset_dir / "summary.json",
        ),
    )


def _write_campaign_state(
    path: Path,
    *,
    status: str,
    selected_stages: Sequence[str],
    completed_stages: Sequence[str],
    stage_records: Sequence[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "selected_stages": list(selected_stages),
        "completed_stages": list(completed_stages),
        "stage_records": list(stage_records),
    }
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seeds", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--dynamic-n-seeds", type=int, default=8)
    parser.add_argument("--n-surrogates", type=int, default=1000)
    parser.add_argument("--only", default=None)
    parser.add_argument("--skip", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="explicit provenance flag; child-stage resume is always enabled",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dynamic_n_seeds < 1:
        raise SystemExit("--dynamic-n-seeds must be positive")
    if args.n_surrogates < 1:
        raise SystemExit("--n-surrogates must be positive")
    only = _parse_stage_names(args.only)
    skip = _parse_stage_names(args.skip)
    if only & skip:
        raise SystemExit("the same stage cannot appear in both --only and --skip")
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else PACKAGE_ROOT / args.output_root
    )
    stages = build_stages(
        python=args.python,
        output_root=output_root,
        seeds=args.seeds,
        dynamic_n_seeds=args.dynamic_n_seeds,
        n_surrogates=args.n_surrogates,
    )
    selected = [
        stage for stage in stages if (not only or stage.name in only) and stage.name not in skip
    ]
    if not selected:
        raise SystemExit("no campaign stages were selected")

    state_path = output_root / "campaign_state.json"
    completed: list[str] = []
    records: list[dict[str, object]] = []
    _write_campaign_state(
        state_path,
        status="preview" if args.dry_run else "running",
        selected_stages=[stage.name for stage in selected],
        completed_stages=completed,
        stage_records=records,
    )
    for stage in selected:
        command_text = " ".join(stage.command)
        if _summary_is_complete(stage.summary_path):
            completed.append(stage.name)
            records.append(
                {
                    "stage": stage.name,
                    "status": "skipped_complete",
                    "command": list(stage.command),
                    "summary": str(stage.summary_path),
                }
            )
            continue
        if args.dry_run:
            print(f"[{stage.name}] {command_text}")
            records.append(
                {
                    "stage": stage.name,
                    "status": "preview",
                    "command": list(stage.command),
                    "summary": str(stage.summary_path),
                }
            )
            continue
        print(f"[{stage.name}] {command_text}", flush=True)
        completed_process = subprocess.run(
            stage.command,
            cwd=PACKAGE_ROOT,
            check=False,
        )
        if completed_process.returncode != 0:
            records.append(
                {
                    "stage": stage.name,
                    "status": "failed",
                    "returncode": completed_process.returncode,
                    "command": list(stage.command),
                    "summary": str(stage.summary_path),
                }
            )
            _write_campaign_state(
                state_path,
                status="failed",
                selected_stages=[item.name for item in selected],
                completed_stages=completed,
                stage_records=records,
            )
            raise SystemExit(
                f"stage {stage.name} failed with code {completed_process.returncode}; "
                "rerun the same campaign command to resume"
            )
        if not _summary_is_complete(stage.summary_path):
            records.append(
                {
                    "stage": stage.name,
                    "status": "failed_missing_complete_summary",
                    "returncode": 0,
                    "command": list(stage.command),
                    "summary": str(stage.summary_path),
                }
            )
            _write_campaign_state(
                state_path,
                status="failed",
                selected_stages=[item.name for item in selected],
                completed_stages=completed,
                stage_records=records,
            )
            raise SystemExit(
                f"stage {stage.name} exited successfully without a complete summary"
            )
        completed.append(stage.name)
        records.append(
            {
                "stage": stage.name,
                "status": "complete",
                "returncode": 0,
                "command": list(stage.command),
                "summary": str(stage.summary_path),
            }
        )
        _write_campaign_state(
            state_path,
            status="running",
            selected_stages=[item.name for item in selected],
            completed_stages=completed,
            stage_records=records,
        )

    _write_campaign_state(
        state_path,
        status="preview" if args.dry_run else "complete",
        selected_stages=[stage.name for stage in selected],
        completed_stages=completed,
        stage_records=records,
    )


if __name__ == "__main__":
    main()
