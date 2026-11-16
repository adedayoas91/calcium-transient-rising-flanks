"""Build a readiness report for saved result artifacts.

The report is intentionally file-based. It does not run simulations or
re-estimate empirical graphs; it only records which publication-gating outputs
are already present and which still need user-run generation.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RequiredArtifact:
    category: str
    artifact: str
    relative_path: str
    required_for: str
    user_run_required: bool
    evidence_role: str


REQUIRED_ARTIFACTS = (
    RequiredArtifact(
        category="static_validation",
        artifact="c-GC summary",
        relative_path="validation_results/cgc/summary.json",
        required_for="saved static synthetic c-GC interpretation",
        user_run_required=False,
        evidence_role="core synthetic validation",
    ),
    RequiredArtifact(
        category="static_validation",
        artifact="c-GC* summary",
        relative_path="validation_results/cgc_star/summary.json",
        required_for="saved static synthetic c-GC* interpretation",
        user_run_required=False,
        evidence_role="core synthetic validation",
    ),
    RequiredArtifact(
        category="dynamic_a_initial",
        artifact="initial dynamic-A summary",
        relative_path="validation_results/dynamic_episodic/summary.json",
        required_for="smoke evidence for dynamic-A script",
        user_run_required=False,
        evidence_role="implementation smoke output",
    ),
    RequiredArtifact(
        category="dynamic_a_locked",
        artifact="locked dynamic-A summary",
        relative_path="validation_results/dynamic_episodic_locked/summary.json",
        required_for="publication-grade causal-rise versus noncausal-fall claim",
        user_run_required=True,
        evidence_role="user-run locked synthetic validation",
    ),
    RequiredArtifact(
        category="dynamic_a_locked",
        artifact="locked dynamic-A representation summary",
        relative_path=(
            "validation_results/dynamic_episodic_locked/representation_summary.csv"
        ),
        required_for="publication-grade dynamic-A representation comparison",
        user_run_required=True,
        evidence_role="user-run locked synthetic validation",
    ),
    RequiredArtifact(
        category="dynamic_a_locked",
        artifact="locked dynamic-A rise/fall contrasts",
        relative_path="validation_results/dynamic_episodic_locked/rise_fall_contrasts.csv",
        required_for="publication-grade dynamic-A rise-minus-comparator claim",
        user_run_required=True,
        evidence_role="user-run locked synthetic validation",
    ),
    RequiredArtifact(
        category="chen_comparison",
        artifact="Chen comparison rows",
        relative_path="chen_comparison/chen_comparison_rows.csv",
        required_for="row-level empirical metric comparison",
        user_run_required=False,
        evidence_role="saved empirical artifact summary",
    ),
    RequiredArtifact(
        category="chen_comparison",
        artifact="Chen comparison summary",
        relative_path="chen_comparison/chen_comparison_summary.csv",
        required_for="aggregate empirical metric comparison",
        user_run_required=False,
        evidence_role="saved empirical artifact summary",
    ),
    RequiredArtifact(
        category="false_positive_tradeoff",
        artifact="representation tradeoff summary",
        relative_path="validation_tradeoffs/representation_tradeoff_summary.csv",
        required_for="recall versus false-positive interpretation",
        user_run_required=False,
        evidence_role="saved synthetic tradeoff summary",
    ),
    RequiredArtifact(
        category="false_positive_tradeoff",
        artifact="rise tradeoff contrasts",
        relative_path="validation_tradeoffs/rise_tradeoff_contrasts.csv",
        required_for="rise-minus-comparator synthetic interpretation",
        user_run_required=False,
        evidence_role="saved synthetic tradeoff summary",
    ),
    RequiredArtifact(
        category="empirical_pairing",
        artifact="paired sign-flip tests",
        relative_path="empirical_stats/paired_signflip_tests.csv",
        required_for="paired rise-minus-fall empirical statistics",
        user_run_required=False,
        evidence_role="saved empirical paired-statistic summary",
    ),
    RequiredArtifact(
        category="graph_stability",
        artifact="saved-artifact graph stability summary",
        relative_path="graph_stability/graph_stability_summary.csv",
        required_for="support and overlap interpretation from saved graphs",
        user_run_required=False,
        evidence_role="saved graph-support summary",
    ),
    RequiredArtifact(
        category="empirical_null_controls",
        artifact="empirical null-control contrasts",
        relative_path="empirical_null_controls/null_control_contrasts.csv",
        required_for="empirical observed-versus-null claim",
        user_run_required=True,
        evidence_role="user-run empirical null controls",
    ),
    RequiredArtifact(
        category="empirical_stability",
        artifact="empirical re-estimation stability summary",
        relative_path="empirical_stability/stability_summary.csv",
        required_for="empirical graph stability claim",
        user_run_required=True,
        evidence_role="user-run empirical stability",
    ),
)


def _artifact_path(
    artifact: RequiredArtifact,
    *,
    output_root: Path,
    dynamic_output_dir: Path | None,
    null_output_dir: Path | None,
    stability_output_dir: Path | None,
) -> Path:
    if artifact.category == "dynamic_a_locked" and dynamic_output_dir is not None:
        return dynamic_output_dir / Path(artifact.relative_path).name
    if artifact.category == "empirical_null_controls" and null_output_dir is not None:
        return null_output_dir / Path(artifact.relative_path).name
    if artifact.category == "empirical_stability" and stability_output_dir is not None:
        return stability_output_dir / Path(artifact.relative_path).name
    return output_root / artifact.relative_path


def _artifact_available(artifact: RequiredArtifact, path: Path) -> bool:
    if not path.is_file():
        return False
    if artifact.category != "dynamic_a_locked" or path.name != "summary.json":
        return True
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "complete"


def build_readiness_rows(
    output_root: Path,
    *,
    dynamic_output_dir: Path | None = None,
    null_output_dir: Path | None = None,
    stability_output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact in REQUIRED_ARTIFACTS:
        path = _artifact_path(
            artifact,
            output_root=output_root,
            dynamic_output_dir=dynamic_output_dir,
            null_output_dir=null_output_dir,
            stability_output_dir=stability_output_dir,
        )
        available = _artifact_available(artifact, path)
        row = asdict(artifact)
        row.update(
            {
                "path": str(path),
                "status": "available" if available else "missing",
                "size_bytes": path.stat().st_size if available else None,
            }
        )
        rows.append(row)
    return rows


def summarize_readiness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    missing = [row for row in rows if row["status"] == "missing"]
    missing_user_run = [row for row in missing if row["user_run_required"]]
    available = [row for row in rows if row["status"] == "available"]
    return {
        "total_artifacts": len(rows),
        "available_artifacts": len(available),
        "missing_artifacts": len(missing),
        "missing_user_run_artifacts": len(missing_user_run),
        "ready_for_publication_claims": len(missing_user_run) == 0
        and len(missing) == 0,
    }


def build_markdown_report(
    rows: list[dict[str, Any]], summary: dict[str, Any]
) -> str:
    lines = [
        "# Result Readiness Report",
        "",
        (
            f"Available artifacts: {summary['available_artifacts']} / "
            f"{summary['total_artifacts']}."
        ),
        f"Missing user-run publication gates: {summary['missing_user_run_artifacts']}.",
        "",
        "## Missing User-Run Gates",
        "",
    ]
    missing_user_run = [
        row
        for row in rows
        if row["status"] == "missing" and row["user_run_required"]
    ]
    if missing_user_run:
        for row in missing_user_run:
            lines.append(
                f"- `{row['relative_path']}`: {row['required_for']}."
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## All Artifacts", ""])
    for row in rows:
        marker = "ok" if row["status"] == "available" else "missing"
        lines.append(
            f"- {marker} `{row['relative_path']}` "
            f"({row['category']}): {row['evidence_role']}."
        )
    lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/result_readiness")
    )
    parser.add_argument("--dynamic-output-dir", type=Path, default=None)
    parser.add_argument("--null-output-dir", type=Path, default=None)
    parser.add_argument("--stability-output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_readiness_rows(
        args.output_root,
        dynamic_output_dir=args.dynamic_output_dir,
        null_output_dir=args.null_output_dir,
        stability_output_dir=args.stability_output_dir,
    )
    summary = {
        "status": "complete",
        "output_root": str(args.output_root),
        "dynamic_output_dir": str(args.dynamic_output_dir)
        if args.dynamic_output_dir is not None
        else None,
        "null_output_dir": str(args.null_output_dir)
        if args.null_output_dir is not None
        else None,
        "stability_output_dir": str(args.stability_output_dir)
        if args.stability_output_dir is not None
        else None,
        **summarize_readiness(rows),
    }
    report = build_markdown_report(rows, summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "result_readiness_rows.csv", rows)
    (args.output_dir / "result_readiness_report.md").write_text(report)
    with (args.output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    print(
        "wrote readiness report with "
        f"{summary['missing_artifacts']} missing artifacts to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
