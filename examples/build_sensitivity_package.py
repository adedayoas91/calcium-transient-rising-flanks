"""Build sensitivity tables from locked synthetic evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from calcium_transient_rising_flank import (
    CausalisedGC,
    build_representations,
    edge_recovery,
    selected_frame_indices,
)
from calcium_transient_rising_flank.preprocessing import smooth_traces
from calcium_transient_rising_flank.validation import simulate_calcium_dataset


DEFAULT_DYNAMIC_DIR = Path("outputs/validation_results/dynamic_episodic_locked")
DEFAULT_OUTPUT_DIR = Path("outputs/sensitivity_package")
RECOVERY_METRICS = (
    "precision",
    "recall",
    "false_positive_rate",
    "f1",
    "orientation_accuracy",
    "edge_density",
)
GENERATOR_METRICS = (
    "n_episodes",
    "min_rise_length",
    "rise_waveform_length",
    "min_fall_to_rise",
    "fall_propagated_total",
    "fall_initial_scale",
    "fall_initial_ceiling_fraction",
    "edge_presence_total",
    "edge_prevalence_mean",
    "edge_prevalence_max",
    "node_presence_total",
    "node_prevalence_mean",
    "active_union_nodes",
)
GENERATOR_CONFIG_KEYS = (
    "n_steps",
    "dynamic_episodes",
    "topology_mode",
    "min_rise_length",
    "max_rise_length",
    "rise_waveform_length",
    "fall_state_mode",
    "fall_initial_scale",
    "fall_initial_ceiling_fraction",
    "edge_dropout_probability",
    "edge_addition_probability",
    "source_dropout_probability",
    "source_recruitment_probability",
    "source_recruitment_edge_probability",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows were generated for {path.name}")
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ValueError(f"inconsistent fields for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _float(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    if value in {"", None}:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    random_state: int,
    n_bootstrap: int,
) -> tuple[float, float, float]:
    """Return a deterministic percentile-bootstrap mean and 95% interval."""

    sample = np.asarray(list(values), dtype=float)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        raise ValueError("at least one finite value is required")
    mean = float(np.mean(sample))
    if sample.size == 1:
        return mean, mean, mean
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, sample.size, size=(n_bootstrap, sample.size))
    means = np.mean(sample[indices], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return mean, float(low), float(high)


def _add_ci_fields(
    output: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    metrics: Sequence[str],
    *,
    random_state: int,
    n_bootstrap: int,
) -> None:
    for metric_index, metric in enumerate(metrics):
        values = [
            float(row[metric])
            for row in rows
            if row.get(metric) not in {"", None}
        ]
        if not values:
            output[f"{metric}_mean"] = None
            output[f"{metric}_ci95_low"] = None
            output[f"{metric}_ci95_high"] = None
            continue
        mean, low, high = bootstrap_mean_ci(
            values,
            random_state=random_state + metric_index,
            n_bootstrap=n_bootstrap,
        )
        output[f"{metric}_mean"] = mean
        output[f"{metric}_ci95_low"] = low
        output[f"{metric}_ci95_high"] = high


def summarize_dynamic_rows(
    raw_rows: Sequence[dict[str, str]],
    *,
    random_state: int,
    n_bootstrap: int,
) -> list[dict[str, Any]]:
    """Summarize locked dynamic validation metrics across seed-level rows."""

    keys = ("method", "event_mode", "condition", "simulator_mode", "representation")
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        normalized: dict[str, Any] = dict(row)
        for metric in RECOVERY_METRICS:
            normalized[metric] = _float(row, metric)
        grouped[tuple(row.get(key, "") for key in keys)].append(normalized)

    output_rows: list[dict[str, Any]] = []
    for group_index, (key, rows) in enumerate(sorted(grouped.items())):
        output: dict[str, Any] = dict(zip(keys, key, strict=True))
        output["n_seeds"] = len({int(row["seed"]) for row in rows})
        _add_ci_fields(
            output,
            rows,
            RECOVERY_METRICS,
            random_state=random_state + group_index * 100,
            n_bootstrap=n_bootstrap,
        )
        output_rows.append(output)
    return output_rows


def build_dynamic_paired_contrasts(
    raw_rows: Sequence[dict[str, str]],
    *,
    random_state: int,
    n_bootstrap: int,
) -> list[dict[str, Any]]:
    """Compute seed-paired rise-minus-comparator intervals."""

    index = {
        (
            row.get("method", ""),
            row.get("event_mode", ""),
            row.get("condition", ""),
            row.get("simulator_mode", ""),
            int(row["seed"]),
            row["representation"],
        ): row
        for row in raw_rows
    }
    group_keys = sorted({key[:4] for key in index})
    comparators = ("full", "deconvolved", "fall", "fall_residual")
    outputs: list[dict[str, Any]] = []
    for group_index, group_key in enumerate(group_keys):
        seeds = sorted({key[4] for key in index if key[:4] == group_key})
        for comparator_index, comparator in enumerate(comparators):
            paired: list[dict[str, Any]] = []
            for seed in seeds:
                rise = index.get((*group_key, seed, "rise"))
                other = index.get((*group_key, seed, comparator))
                if rise is None or other is None:
                    continue
                pair: dict[str, Any] = {"seed": seed}
                for metric in RECOVERY_METRICS:
                    rise_value = _float(rise, metric)
                    other_value = _float(other, metric)
                    pair[metric] = (
                        None
                        if rise_value is None or other_value is None
                        else rise_value - other_value
                    )
                paired.append(pair)
            if not paired:
                continue
            output: dict[str, Any] = {
                "method": group_key[0],
                "event_mode": group_key[1],
                "condition": group_key[2],
                "simulator_mode": group_key[3],
                "contrast": f"rise_minus_{comparator}",
                "n_paired_seeds": len(paired),
            }
            _add_ci_fields(
                output,
                paired,
                RECOVERY_METRICS,
                random_state=(
                    random_state + group_index * 1000 + comparator_index * 100
                ),
                n_bootstrap=n_bootstrap,
            )
            outputs.append(output)
    return outputs


def build_generator_summary(
    raw_rows: Sequence[dict[str, str]],
    config: dict[str, Any],
    *,
    random_state: int,
    n_bootstrap: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicate and summarize generator metadata already saved per seed."""

    by_seed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in raw_rows:
        if row.get("simulator_mode") != "episodic_dynamic":
            continue
        key = (row["condition"], int(row["seed"]))
        candidate: dict[str, Any] = {
            "condition": key[0],
            "seed": key[1],
            "fall_state_mode": row.get("fall_state_mode", ""),
        }
        for metric in GENERATOR_METRICS:
            candidate[metric] = _float(row, metric)
        if key in by_seed and by_seed[key] != candidate:
            raise ValueError(f"inconsistent generator metadata for {key}")
        by_seed[key] = candidate

    seed_rows = [by_seed[key] for key in sorted(by_seed)]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[str(row["condition"])].append(row)
    summary_rows: list[dict[str, Any]] = []
    for group_index, (condition, rows) in enumerate(sorted(grouped.items())):
        output: dict[str, Any] = {
            "condition": condition,
            "n_seeds": len(rows),
            "fall_state_mode": ";".join(
                sorted({str(row["fall_state_mode"]) for row in rows})
            ),
        }
        for key in GENERATOR_CONFIG_KEYS:
            output[f"config_{key}"] = config.get(key)
        _add_ci_fields(
            output,
            rows,
            GENERATOR_METRICS,
            random_state=random_state + group_index * 100,
            n_bootstrap=n_bootstrap,
        )
        summary_rows.append(output)
    return seed_rows, summary_rows


def _f1(precision: float, recall: float) -> float:
    return (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )


def run_threshold_smoothing_grid(
    *,
    seeds: Sequence[int],
    thresholds: Sequence[float],
    smoothing_windows: Sequence[int],
    n_steps: int,
    n_surrogates: int,
) -> list[dict[str, Any]]:
    """Run a locked grid where each seed's trace is reused for every setting."""

    truth = np.array(
        [
            [False, True, False, False],
            [False, False, False, False],
            [False, False, False, True],
            [False, False, False, False],
        ]
    )
    off_diagonal = ~np.eye(truth.shape[0], dtype=bool)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        dataset = simulate_calcium_dataset(
            truth,
            n_steps=n_steps,
            gamma=0.8,
            noise_std=0.03,
            shared_noise_std=0.02,
            random_state=seed,
        )
        for window in smoothing_windows:
            traces = smooth_traces(dataset.fluorescence, window)
            for threshold in thresholds:
                rise = build_representations(
                    traces,
                    tolerance=threshold,
                    gamma=0.8,
                ).rise
                selected = selected_frame_indices(rise)
                graph = CausalisedGC(
                    max_lag=1,
                    n_surrogates=n_surrogates,
                    alpha=0.05,
                    random_state=100_000 + seed,
                    fdr=True,
                    event_mode="physical",
                ).fit(traces, event_indices=selected)
                recovery = edge_recovery(truth, graph.adjacency)
                rows.append(
                    {
                        "seed": seed,
                        "smoothing_window": window,
                        "tolerance": threshold,
                        "n_steps": n_steps,
                        "n_surrogates": n_surrogates,
                        "alpha": 0.05,
                        "fdr": True,
                        "event_mode": "physical",
                        "selected_frames_total": int(
                            sum(values.size for values in selected)
                        ),
                        "selected_frame_fraction": float(
                            sum(values.size for values in selected)
                            / (truth.shape[0] * n_steps)
                        ),
                        "retained_edges": int(np.count_nonzero(graph.adjacency)),
                        "precision": recovery.precision,
                        "recall": recovery.recall,
                        "false_positive_rate": recovery.false_positive_rate,
                        "f1": _f1(recovery.precision, recovery.recall),
                        "orientation_accuracy": recovery.orientation_accuracy,
                        "edge_density": float(np.mean(graph.adjacency[off_diagonal])),
                    }
                )
    return rows


def summarize_threshold_grid(
    rows: Sequence[dict[str, Any]],
    *,
    random_state: int,
    n_bootstrap: int,
) -> list[dict[str, Any]]:
    metrics = (
        "selected_frame_fraction",
        "retained_edges",
        *RECOVERY_METRICS,
    )
    grouped: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["smoothing_window"]), float(row["tolerance"]))].append(row)
    outputs: list[dict[str, Any]] = []
    for group_index, (key, group) in enumerate(sorted(grouped.items())):
        output: dict[str, Any] = {
            "smoothing_window": key[0],
            "tolerance": key[1],
            "n_seeds": len({int(row["seed"]) for row in group}),
        }
        _add_ci_fields(
            output,
            group,
            metrics,
            random_state=random_state + group_index * 100,
            n_bootstrap=n_bootstrap,
        )
        outputs.append(output)
    return outputs


def plot_threshold_sensitivity(
    summary_rows: Sequence[dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot sensitivity curves with seed-bootstrap intervals."""

    import matplotlib.pyplot as plt

    panels = (
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("false_positive_rate", "False-positive rate"),
        ("edge_density", "Edge density"),
    )
    windows = sorted({int(row["smoothing_window"]) for row in summary_rows})
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.5), sharex=True)
    for axis, (metric, label) in zip(axes.flat, panels, strict=True):
        for window in windows:
            rows = sorted(
                (
                    row
                    for row in summary_rows
                    if int(row["smoothing_window"]) == window
                ),
                key=lambda row: float(row["tolerance"]),
            )
            tolerance = np.asarray([float(row["tolerance"]) for row in rows])
            mean = np.asarray([float(row[f"{metric}_mean"]) for row in rows])
            low = np.asarray([float(row[f"{metric}_ci95_low"]) for row in rows])
            high = np.asarray([float(row[f"{metric}_ci95_high"]) for row in rows])
            (line,) = axis.plot(
                tolerance,
                mean,
                marker="o",
                label=f"window={window}",
            )
            axis.fill_between(tolerance, low, high, color=line.get_color(), alpha=0.16)
        axis.set_title(label)
        axis.set_ylabel("Mean across seeds")
        axis.grid(alpha=0.25)
        axis.set_ylim(-0.04, 1.04)
    for axis in axes[-1]:
        axis.set_xlabel("Rise tolerance (fluorescence increment)")
    axes[0, 0].legend(frameon=False, title="Smoothing")
    figure.suptitle("Rising-flank proposal sensitivity diagnostic")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _parse_values(value: str, cast: type[int] | type[float]) -> tuple[Any, ...]:
    values = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic-dir", type=Path, default=DEFAULT_DYNAMIC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--thresholds", default="0.0,0.02,0.05")
    parser.add_argument("--smoothing-windows", default="1,3,5")
    parser.add_argument("--n-steps", type=int, default=600)
    parser.add_argument("--n-surrogates", type=int, default=199)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260808)
    parser.add_argument("--skip-threshold-grid", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dynamic_csv = args.dynamic_dir / "dynamic_grid_runs.csv"
    dynamic_json = args.dynamic_dir / "summary.json"
    if not dynamic_csv.exists() or not dynamic_json.exists():
        raise SystemExit(
            "locked dynamic artifacts are missing; expected "
            f"{dynamic_csv} and {dynamic_json}"
        )
    dynamic_rows = _read_csv(dynamic_csv)
    dynamic_payload = json.loads(dynamic_json.read_text())
    dynamic_config = dynamic_payload.get("config", {})

    dynamic_summary = summarize_dynamic_rows(
        dynamic_rows,
        random_state=args.bootstrap_seed,
        n_bootstrap=args.n_bootstrap,
    )
    dynamic_contrasts = build_dynamic_paired_contrasts(
        dynamic_rows,
        random_state=args.bootstrap_seed + 100_000,
        n_bootstrap=args.n_bootstrap,
    )
    generator_seeds, generator_summary = build_generator_summary(
        dynamic_rows,
        dynamic_config,
        random_state=args.bootstrap_seed + 200_000,
        n_bootstrap=args.n_bootstrap,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "dynamic_seed_summary.csv", dynamic_summary)
    _write_csv(
        args.output_dir / "dynamic_paired_contrasts.csv",
        dynamic_contrasts,
    )
    _write_csv(
        args.output_dir / "dynamic_generator_seed_rows.csv",
        generator_seeds,
    )
    _write_csv(
        args.output_dir / "dynamic_generator_summary.csv",
        generator_summary,
    )

    threshold_rows: list[dict[str, Any]] = []
    threshold_summary: list[dict[str, Any]] = []
    if not args.skip_threshold_grid:
        seeds = _parse_values(args.seeds, int)
        thresholds = _parse_values(args.thresholds, float)
        smoothing_windows = _parse_values(args.smoothing_windows, int)
        threshold_rows = run_threshold_smoothing_grid(
            seeds=seeds,
            thresholds=thresholds,
            smoothing_windows=smoothing_windows,
            n_steps=args.n_steps,
            n_surrogates=args.n_surrogates,
        )
        threshold_summary = summarize_threshold_grid(
            threshold_rows,
            random_state=args.bootstrap_seed + 300_000,
            n_bootstrap=args.n_bootstrap,
        )
        _write_csv(
            args.output_dir / "threshold_smoothing_seed_rows.csv",
            threshold_rows,
        )
        _write_csv(
            args.output_dir / "threshold_smoothing_summary.csv",
            threshold_summary,
        )
        plot_threshold_sensitivity(
            threshold_summary,
            args.output_dir / "threshold_smoothing_sensitivity.png",
        )

    empirical_base = (
        "PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-cache "
        "XDG_CACHE_HOME=/tmp/font-cache .venv/bin/python "
        "examples/run_empirical_null_controls.py --cases C,D "
        "--recordings F3T1,F3T2,F5T2 --representations rise,fall "
        "--methods cgc,cgc-star --n-null-replicates 0 "
        "--n-estimator-surrogates 1000 --alpha 0.05 "
        "--event-mode physical --seed 10"
    )
    summary = {
        "analysis_scope": (
            "Proposal stress tests and sensitivity diagnostics; not a "
            "comprehensive benchmark or validation guarantee."
        ),
        "ci_method": {
            "name": "seed-level percentile bootstrap of the mean",
            "confidence_level": 0.95,
            "n_bootstrap": args.n_bootstrap,
            "random_state": args.bootstrap_seed,
        },
        "dynamic_source": {
            "rows": str(dynamic_csv),
            "config": str(dynamic_json),
            "source_row_count": len(dynamic_rows),
            "source_config": dynamic_config,
        },
        "threshold_smoothing": {
            "status": "skipped" if args.skip_threshold_grid else "completed",
            "design": {
                "same_simulated_trace_reused_across_settings_within_seed": True,
                "truth_edges": [[0, 1], [2, 3]],
                "gamma": 0.8,
                "noise_std": 0.03,
                "shared_noise_std": 0.02,
                "n_steps": args.n_steps,
                "seeds": []
                if args.skip_threshold_grid
                else list(_parse_values(args.seeds, int)),
                "thresholds": []
                if args.skip_threshold_grid
                else list(_parse_values(args.thresholds, float)),
                "smoothing_windows": []
                if args.skip_threshold_grid
                else list(_parse_values(args.smoothing_windows, int)),
                "estimator": "cgc",
                "event_mode": "physical",
                "n_surrogates": args.n_surrogates,
                "alpha": 0.05,
                "fdr": True,
            },
            "seed_rows": len(threshold_rows),
            "summary_rows": len(threshold_summary),
            "plot": None
            if args.skip_threshold_grid
            else str(args.output_dir / "threshold_smoothing_sensitivity.png"),
        },
        "empirical_fdr_comparison": {
            "status": "not_run",
            "cached_trace_inputs_available": Path("data/motoneurons").exists(),
            "blocker": (
                "No per-edge empirical p-value matrices are cached. The current "
                "runner re-estimates observed, reverse-time, and cross-recording "
                "graphs; the requested two-arm design requires 144 graph fits "
                "with 1000 estimator surrogates (72 per arm), outside this "
                "bounded analysis run."
            ),
            "bh_command": (
                empirical_base
                + " --output-dir outputs/empirical_fdr_comparison/bh"
            ),
            "unadjusted_command": (
                empirical_base
                + " --no-fdr --output-dir "
                "outputs/empirical_fdr_comparison/unadjusted"
            ),
        },
        "limitations": [
            "Threshold/smoothing results use the static four-node simulator and "
            "do not substitute for empirical sensitivity.",
            "The bounded threshold grid uses 199 surrogates; the locked dynamic "
            "validation used 1000 surrogates.",
            "Bootstrap intervals quantify between-seed Monte Carlo variation, "
            "not uncertainty over animals or recordings.",
            "Fall representations in dynamic rise-minus-fall contrasts have zero "
            "truth by construction; those contrasts are descriptive controls.",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
