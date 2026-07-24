"""Audit remaining todo.md gates against saved result artifacts.

This script is deliberately file-based. It does not run simulations,
re-estimate empirical graphs, or regenerate manuscript artifacts. It checks the
saved outputs that prove the remaining user-run TODO gates are complete.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path("outputs")
DEFAULT_OUTPUT_DIR = Path("outputs/todo_completion")
DEFAULT_TODO_PATH = Path("../todo.md")


@dataclass(frozen=True)
class TodoGate:
    item_id: str
    todo_text: str
    required_evidence: str
    checker: Callable[[Path], tuple[bool, str]]


def _csv_data_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    return len(rows)


def _all_files_exist(output_root: Path, relative_paths: tuple[str, ...]) -> tuple[bool, str]:
    missing = [path for path in relative_paths if not (output_root / path).is_file()]
    if missing:
        return False, "missing " + "; ".join(missing)
    return True, "available " + "; ".join(relative_paths)


def _csv_has_rows(output_root: Path, relative_path: str) -> tuple[bool, str]:
    path = output_root / relative_path
    rows = _csv_data_rows(path)
    if rows <= 0:
        return False, f"{relative_path} has no data rows"
    return True, f"{relative_path} has {rows} data rows"


def _summary_has_no_missing_gates(output_root: Path) -> tuple[bool, str]:
    path = output_root / "manuscript_evidence/summary.json"
    if not path.is_file():
        return False, "manuscript_evidence/summary.json is missing"
    with path.open() as file:
        summary = json.load(file)
    missing = int(summary.get("n_missing_publication_gates") or 0)
    if missing:
        return False, f"manuscript evidence still reports {missing} missing gates"
    return True, "manuscript evidence reports zero missing gates"


def _final_figures_ready(output_root: Path) -> tuple[bool, str]:
    path = output_root / "manuscript_evidence/final_figure_plan.csv"
    if not path.is_file():
        return False, "manuscript_evidence/final_figure_plan.csv is missing"
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return False, "final_figure_plan.csv has no rows"
    not_ready = [
        f"{row.get('figure_id')}={row.get('status')}"
        for row in rows
        if row.get("status") != "ready"
    ]
    if not_ready:
        return False, "not ready: " + "; ".join(not_ready)
    return True, f"all {len(rows)} planned figure panels are ready"


def todo_gates() -> tuple[TodoGate, ...]:
    return (
        TodoGate(
            item_id="locked_dynamic_a_outputs",
            todo_text=(
                "User-run publication-grade locked dynamic-A simulation result set "
                "that directly tests causal rises versus noncausal stochastic falls."
            ),
            required_evidence=(
                "validation_results/dynamic_episodic_locked/summary.json; "
                "validation_results/dynamic_episodic_locked/representation_summary.csv; "
                "validation_results/dynamic_episodic_locked/rise_fall_contrasts.csv"
            ),
            checker=lambda root: _all_files_exist(
                root,
                (
                    "validation_results/dynamic_episodic_locked/summary.json",
                    "validation_results/dynamic_episodic_locked/representation_summary.csv",
                    "validation_results/dynamic_episodic_locked/rise_fall_contrasts.csv",
                ),
            ),
        ),
        TodoGate(
            item_id="dynamic_a_interpretation",
            todo_text=(
                "User-run analysis compares recovery across full, deconvolved, "
                "rise, fall, and fall-residual dynamic-A representations."
            ),
            required_evidence="manuscript_evidence/dynamic_a_interpretation.csv",
            checker=lambda root: _csv_has_rows(
                root, "manuscript_evidence/dynamic_a_interpretation.csv"
            ),
        ),
        TodoGate(
            item_id="empirical_null_interpretation",
            todo_text=(
                "Manuscript-ready interpretation of empirical null-control "
                "contrast tables across preprocessing cases A-D and representations."
            ),
            required_evidence="manuscript_evidence/empirical_null_interpretation.csv",
            checker=lambda root: _csv_has_rows(
                root, "manuscript_evidence/empirical_null_interpretation.csv"
            ),
        ),
        TodoGate(
            item_id="empirical_stability_interpretation",
            todo_text=(
                "Empirical re-estimation stability interpretation is available "
                "for final robustness claims."
            ),
            required_evidence=(
                "empirical_stability/stability_summary.csv; "
                "manuscript_evidence/empirical_stability_interpretation.csv"
            ),
            checker=lambda root: (
                _all_files_exist(root, ("empirical_stability/stability_summary.csv",))
                if not (root / "empirical_stability/stability_summary.csv").is_file()
                else _csv_has_rows(
                    root, "manuscript_evidence/empirical_stability_interpretation.csv"
                )
            ),
        ),
        TodoGate(
            item_id="post_run_manuscript_update_ready",
            todo_text=(
                "Manuscript can be updated after locked dynamic-A, null-control, "
                "and stability gates are saved and inspected."
            ),
            required_evidence="manuscript_evidence/summary.json reports zero missing gates",
            checker=_summary_has_no_missing_gates,
        ),
        TodoGate(
            item_id="final_figures_ready",
            todo_text=(
                "Final main figure panels are ready after dynamic-A and "
                "null/robustness user-run gates are resolved."
            ),
            required_evidence="manuscript_evidence/final_figure_plan.csv all statuses ready",
            checker=_final_figures_ready,
        ),
    )


def build_audit_rows(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gate in todo_gates():
        complete, evidence_status = gate.checker(output_root)
        rows.append(
            {
                "item_id": gate.item_id,
                "todo_text": gate.todo_text,
                "required_evidence": gate.required_evidence,
                "status": "complete" if complete else "pending",
                "evidence_status": evidence_status,
            }
        )
    return rows


def summarize_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pending = [row for row in rows if row["status"] != "complete"]
    return {
        "total_gates": len(rows),
        "complete_gates": len(rows) - len(pending),
        "pending_gates": len(pending),
        "ready_to_close_user_run_todos": len(pending) == 0,
    }


def build_markdown_report(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# TODO Completion Audit",
        "",
        (
            f"Complete gates: {summary['complete_gates']} / "
            f"{summary['total_gates']}."
        ),
        (
            "Ready to close user-run TODOs: "
            f"{summary['ready_to_close_user_run_todos']}."
        ),
        "",
        "## Gate Status",
        "",
    ]
    for row in rows:
        marker = "complete" if row["status"] == "complete" else "pending"
        lines.append(f"- {marker} `{row['item_id']}`: {row['evidence_status']}.")
    lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--todo-path",
        type=Path,
        default=DEFAULT_TODO_PATH,
        help="kept for command provenance; the audit maps current known TODO gates",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_audit_rows(args.output_root)
    summary = summarize_audit(rows)
    report = build_markdown_report(rows, summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "todo_completion_rows.csv", rows)
    (args.output_dir / "todo_completion_audit.md").write_text(report)
    with (args.output_dir / "summary.json").open("w") as file:
        json.dump(
            {
                **summary,
                "output_root": str(args.output_root),
                "todo_path": str(args.todo_path),
            },
            file,
            indent=2,
        )
        file.write("\n")
    print(
        "wrote TODO completion audit with "
        f"{summary['pending_gates']} pending gates to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
