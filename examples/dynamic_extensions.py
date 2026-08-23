"""Run dynamic fall extensions on physical event frames."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from calcium_transient_rising_flank import CausalisedGC, DynamicSimulationConfig
from calcium_transient_rising_flank.robustness import (
    SyntheticCondition,
    run_synthetic_grid,
)

SUMMARY_METRICS = (
    "precision",
    "recall",
    "false_positive_rate",
    "f1",
    "orientation_accuracy",
    "edge_density",
    "truth_edges",
    "fall_truth_edges",
    "fall_propagated_total",
)
METHOD_CHOICES = ("cgc", "fcgc", "cgc-star", "cgc*")
GRID_ROWS_CSV = "dynamic_extensions_rows.csv"
REPRESENTATION_SUMMARY_CSV = "dynamic_extensions_summary.csv"
RISE_FALL_CONTRASTS_CSV = "dynamic_extensions_contrasts.csv"
SUMMARY_JSON = "summary.json"
RESUME_STATE_JSON = "resume_state.json"
REPRESENTATION_ORDER = ("full", "deconvolved", "rise", "fall", "fall_residual")
EXPECTED_REPRESENTATIONS = frozenset(REPRESENTATION_ORDER)
RESUME_AXIS_CONFIG_KEYS = {"methods", "n_seeds"}
INTEGER_ROW_COLUMNS = {
    "seed",
    "max_lag",
    "tau",
    "n_pasts",
    "min_rise_run_samples",
    "rise_match_max_lag",
    "rise_match_min_overlap_samples",
    "rise_run_context_samples",
    "truth_edges",
    "fall_truth_edges",
    "n_episodes",
    "fall_overlap_exclusion_samples",
}
FLOAT_ROW_COLUMNS = set(SUMMARY_METRICS) | {
    "rise_match_min_overlap_fraction",
    "fall_propagation_drop_fraction",
}


@dataclass(frozen=True)
class EstimatorGridSpec:
    name: str
    max_lag: int
    tau: int | None = None
    n_pasts: int | None = None
    min_rise_run_samples: int = 1
    rise_candidate_filter: bool = False
    rise_match_min_lag: int = 1
    rise_match_max_lag: int | None = None
    rise_match_min_overlap_samples: int | None = None
    rise_match_min_overlap_fraction: float = 0.5
    rise_run_context_samples: int | None = None


GRID_SPECS = {
    "lag1_context1": EstimatorGridSpec(
        name="lag1_context1",
        max_lag=1,
        tau=1,
        n_pasts=1,
        rise_match_max_lag=1,
        rise_run_context_samples=1,
    ),
    "lag2_context2": EstimatorGridSpec(
        name="lag2_context2",
        max_lag=2,
        tau=2,
        n_pasts=2,
        rise_match_max_lag=2,
        rise_run_context_samples=2,
    ),
    "bout_bounded_lag3_context4": EstimatorGridSpec(
        name="bout_bounded_lag3_context4",
        max_lag=3,
        tau=3,
        n_pasts=4,
        min_rise_run_samples=3,
        rise_candidate_filter=True,
        rise_match_max_lag=3,
        rise_match_min_overlap_samples=2,
        rise_match_min_overlap_fraction=0.5,
        rise_run_context_samples=4,
    ),
}


def _parse_methods(value: str) -> tuple[str, ...]:
    methods = tuple(method.strip() for method in value.split(",") if method.strip())
    if not methods:
        raise ValueError("at least one method is required")
    invalid = sorted(set(methods) - set(METHOD_CHOICES))
    if invalid:
        raise ValueError(f"unsupported methods: {', '.join(invalid)}")
    return methods


def _parse_grid(value: str) -> tuple[EstimatorGridSpec, ...]:
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    if not names:
        raise ValueError("at least one estimator grid name is required")
    invalid = sorted(set(names) - set(GRID_SPECS))
    if invalid:
        raise ValueError(f"unsupported grid names: {', '.join(invalid)}")
    return tuple(GRID_SPECS[name] for name in names)


def _truth_graphs() -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    rise_a = np.array(
        [
            [False, True, False, False, False],
            [False, False, True, False, False],
            [False, False, False, True, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
        ]
    )
    rise_b = np.array(
        [
            [False, False, False, False, True],
            [False, False, False, False, False],
            [False, True, False, False, False],
            [False, False, True, False, False],
            [False, False, False, False, False],
        ]
    )
    fall_a = np.array(
        [
            [False, True, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, True],
            [False, False, False, False, False],
            [False, False, False, False, False],
        ]
    )
    fall_b = np.array(
        [
            [False, False, True, False, False],
            [False, False, False, False, False],
            [False, False, False, True, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
        ]
    )
    return rise_a | rise_b, (rise_a, rise_b), (fall_a, fall_b)


def _active_node_sequence() -> tuple[np.ndarray, ...]:
    return (
        np.array([True, True, True, True, False]),
        np.array([True, True, True, True, True]),
    )


def _dynamic_conditions(
    rise_sequence: tuple[np.ndarray, ...],
    fall_sequence: tuple[np.ndarray, ...],
) -> tuple[SyntheticCondition, ...]:
    common = {
        "n_episodes": 4,
        "min_rise_length": 24,
        "max_rise_length": 24,
        "rise_waveform_length": 8,
        "fall_to_rise_ratio_min": 2.1,
        "fall_to_rise_ratio_max": 2.2,
        "propagation_delay": 1,
        "initial_activation_probability": 1.0,
        "initial_activation_mode": "source_nodes",
        "fall_noise_rate": 0.0,
        "fall_noise_scale": 0.0,
        "adjacency_sequence": rise_sequence,
        "active_node_sequence": _active_node_sequence(),
    }
    return (
        SyntheticCondition(
            name="dynamic_a_noncausal_fall",
            gamma=0.9,
            simulator_mode="episodic_dynamic",
            dynamic_config=DynamicSimulationConfig(
                **common,
                fall_state_mode="stochastic_independent",
                fall_initial_scale=0.0,
                fall_initial_ceiling_fraction=1.0,
            ),
        ),
        SyntheticCondition(
            name="dynamic_b_hybrid_causal_fall_overlap",
            gamma=0.9,
            simulator_mode="episodic_dynamic",
            dynamic_config=DynamicSimulationConfig(
                **common,
                fall_state_mode="passive_decay",
                fall_initial_scale=0.0,
                fall_initial_ceiling_fraction=1.0,
                fall_adjacency_sequence=fall_sequence,
                fall_initial_activation_probability=1.0,
                fall_transmission_probability=1.0,
                fall_spontaneous_rate=0.0,
                fall_overlap_exclusion_samples=0,
                fall_propagation_drop_fraction=0.35,
            ),
        ),
        SyntheticCondition(
            name="dynamic_c_hybrid_causal_fall_kinetic_misspecification",
            gamma=0.9,
            simulator_mode="episodic_dynamic",
            dynamic_config=DynamicSimulationConfig(
                **common,
                fall_state_mode="passive_decay",
                fall_initial_scale=0.0,
                fall_initial_ceiling_fraction=1.0,
                gamma_fall=np.array([0.72, 0.80, 0.86, 0.92, 0.95]),
                fall_adjacency_sequence=fall_sequence,
                fall_initial_activation_probability=1.0,
                fall_transmission_probability=1.0,
                fall_spontaneous_rate=0.0,
                fall_overlap_exclusion_samples=2,
                fall_propagation_drop_fraction=0.35,
            ),
        ),
    )


def _edge_density(adjacency: np.ndarray) -> float:
    values = np.asarray(adjacency, dtype=bool)
    off_diag = ~np.eye(values.shape[0], dtype=bool)
    return float(np.count_nonzero(values[off_diag]) / np.count_nonzero(off_diag))


def _f1(precision: float, recall: float) -> float:
    denominator = precision + recall
    return 0.0 if denominator == 0.0 else 2.0 * precision * recall / denominator


def _dynamic_metadata(run) -> dict[str, float | int | bool | str | None]:
    episodes = run.dataset.episodes
    dynamic_config = getattr(run.condition, "dynamic_config", None)
    fall_truth = run.dataset.fall_union_adjacency
    fall_propagated = 0.0
    if run.dataset.propagated_events is not None:
        for episode in episodes:
            fall_propagated += float(
                run.dataset.propagated_events[:, episode.fall_start : episode.fall_stop].sum()
            )
    return {
        "n_episodes": len(episodes),
        "fall_truth_edges": 0 if fall_truth is None else int(np.count_nonzero(fall_truth)),
        "fall_propagated_total": fall_propagated,
        "fall_overlap_exclusion_samples": (
            None
            if dynamic_config is None
            else dynamic_config.fall_overlap_exclusion_samples
        ),
        "fall_propagation_drop_fraction": (
            None
            if dynamic_config is None
            else dynamic_config.fall_propagation_drop_fraction
        ),
        "gamma_fall": (
            None
            if dynamic_config is None or dynamic_config.gamma_fall is None
            else json.dumps(np.asarray(dynamic_config.gamma_fall).tolist())
        ),
        "segment_policy": "episode_bounded" if episodes else "none",
        "cross_boundary_pairs_allowed": False if episodes else None,
    }


def _rows_for_spec(
    *,
    adjacency: np.ndarray,
    condition: SyntheticCondition,
    seed: int,
    method: str,
    spec: EstimatorGridSpec,
    n_steps: int,
    n_surrogates: int,
    score_threshold: float,
    no_fdr: bool,
) -> list[dict[str, float | int | bool | str | None]]:
    runs = run_synthetic_grid(
        adjacency,
        conditions=(condition,),
        seeds=(seed,),
        estimator_factory=lambda: CausalisedGC(
            max_lag=spec.max_lag,
            tau=spec.tau,
            n_pasts=spec.n_pasts,
            n_surrogates=n_surrogates,
            score_threshold=score_threshold,
            event_mode="physical",
            method=method,
            fdr=not no_fdr,
            min_rise_run_samples=spec.min_rise_run_samples,
            rise_candidate_filter=spec.rise_candidate_filter,
            rise_match_min_lag=spec.rise_match_min_lag,
            rise_match_max_lag=spec.rise_match_max_lag,
            rise_match_min_overlap_samples=spec.rise_match_min_overlap_samples,
            rise_match_min_overlap_fraction=spec.rise_match_min_overlap_fraction,
            rise_run_context_samples=spec.rise_run_context_samples,
        ),
        n_steps=n_steps,
    )
    rows: list[dict[str, float | int | bool | str | None]] = []
    for run in runs:
        metadata = _dynamic_metadata(run)
        for label, graph in run.validation.graphs.items():
            recovery = run.validation.recovery[label]
            truth = run.validation.truth[label]
            rows.append(
                {
                    "method": method,
                    "event_mode": "physical",
                    "grid": spec.name,
                    "condition": run.condition.name,
                    "simulator_mode": run.condition.simulator_mode,
                    "seed": seed,
                    "representation": label,
                    "max_lag": spec.max_lag,
                    "tau": spec.tau,
                    "n_pasts": spec.n_pasts,
                    "min_rise_run_samples": spec.min_rise_run_samples,
                    "rise_candidate_filter": spec.rise_candidate_filter,
                    "rise_match_max_lag": spec.rise_match_max_lag,
                    "rise_match_min_overlap_samples": spec.rise_match_min_overlap_samples,
                    "rise_match_min_overlap_fraction": spec.rise_match_min_overlap_fraction,
                    "rise_run_context_samples": spec.rise_run_context_samples,
                    "precision": recovery.precision,
                    "recall": recovery.recall,
                    "false_positive_rate": recovery.false_positive_rate,
                    "f1": _f1(recovery.precision, recovery.recall),
                    "orientation_accuracy": recovery.orientation_accuracy,
                    "edge_density": _edge_density(graph.adjacency),
                    "truth_edges": int(np.count_nonzero(truth)),
                    **metadata,
                }
            )
    return rows


def _summary_rows(
    rows: list[dict[str, float | int | bool | str | None]],
) -> list[dict[str, float | int | bool | str | None]]:
    grouped: dict[
        tuple[str, str, str, str], list[dict[str, float | int | bool | str | None]]
    ] = defaultdict(list)
    for row in rows:
        key = (
            str(row["method"]),
            str(row["grid"]),
            str(row["condition"]),
            str(row["representation"]),
        )
        grouped[key].append(row)

    summaries: list[dict[str, float | int | bool | str | None]] = []
    for key, values in sorted(grouped.items()):
        method, grid, condition, representation = key
        entry: dict[str, float | int | bool | str | None] = {
            "method": method,
            "grid": grid,
            "condition": condition,
            "representation": representation,
            "event_mode": "physical",
            "n": len(values),
        }
        for metric in SUMMARY_METRICS:
            samples = [
                float(value)
                for row in values
                if (value := row.get(metric)) is not None
            ]
            entry[f"{metric}_mean"] = float(np.mean(samples)) if samples else None
        summaries.append(entry)
    return summaries


def _contrast_rows(
    summaries: list[dict[str, float | int | bool | str | None]],
) -> list[dict[str, float | int | bool | str | None]]:
    by_group: dict[
        tuple[str, str, str], dict[str, dict[str, float | int | bool | str | None]]
    ] = defaultdict(dict)
    for row in summaries:
        by_group[
            (str(row["method"]), str(row["grid"]), str(row["condition"]))
        ][str(row["representation"])] = row

    contrasts: list[dict[str, float | int | bool | str | None]] = []
    for (method, grid, condition), representation_rows in sorted(by_group.items()):
        contrast: dict[str, float | int | bool | str | None] = {
            "method": method,
            "grid": grid,
            "condition": condition,
            "event_mode": "physical",
        }
        for metric in SUMMARY_METRICS:
            column = f"{metric}_mean"
            rise = representation_rows.get("rise", {}).get(column)
            fall = representation_rows.get("fall", {}).get(column)
            residual = representation_rows.get("fall_residual", {}).get(column)
            contrast[f"rise_minus_fall_{metric}"] = (
                None if rise is None or fall is None else float(rise) - float(fall)
            )
            contrast[f"rise_minus_fall_residual_{metric}"] = (
                None
                if rise is None or residual is None
                else float(rise) - float(residual)
            )
        contrasts.append(contrast)
    return contrasts


def _write_csv(path: Path, rows: list[dict[str, float | int | bool | str | None]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if not rows:
        temporary.write_text("", encoding="utf-8")
        temporary.replace(path)
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _coerce_grid_row(
    row: dict[str, str | None],
) -> dict[str, float | int | bool | str | None]:
    coerced: dict[str, float | int | bool | str | None] = {}
    for key, value in row.items():
        if value in {"", None}:
            coerced[key] = None
        elif key in INTEGER_ROW_COLUMNS:
            coerced[key] = int(value)
        elif key in FLOAT_ROW_COLUMNS:
            coerced[key] = float(value)
        elif value in {"True", "False"}:
            coerced[key] = value == "True"
        else:
            coerced[key] = value
    return coerced


def _read_grid_rows(path: Path) -> list[dict[str, float | int | bool | str | None]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return [_coerce_grid_row(row) for row in csv.DictReader(file)]


def _row_run_key(row: dict[str, float | int | bool | str | None]) -> tuple[str, str, str, int]:
    method = row.get("method")
    grid = row.get("grid")
    condition = row.get("condition")
    seed = row.get("seed")
    if method is None or grid is None or condition is None or seed is None:
        raise ValueError("resume row is missing method, grid, condition, or seed")
    return (str(method), str(grid), str(condition), int(seed))


def _complete_grid_rows(
    rows: list[dict[str, float | int | bool | str | None]],
) -> list[dict[str, float | int | bool | str | None]]:
    ordered_keys: list[tuple[str, str, str, int]] = []
    grouped: dict[
        tuple[str, str, str, int], dict[str, dict[str, float | int | bool | str | None]]
    ] = {}
    for row in rows:
        key = _row_run_key(row)
        representation = row.get("representation")
        if representation is None:
            raise ValueError("resume row is missing representation")
        if key not in grouped:
            grouped[key] = {}
            ordered_keys.append(key)
        grouped[key][str(representation)] = row

    complete: list[dict[str, float | int | bool | str | None]] = []
    for key in ordered_keys:
        representation_rows = grouped[key]
        if not EXPECTED_REPRESENTATIONS.issubset(representation_rows):
            continue
        complete.extend(
            representation_rows[representation]
            for representation in REPRESENTATION_ORDER
        )
    return complete


def _completed_run_keys(
    rows: list[dict[str, float | int | bool | str | None]],
) -> set[tuple[str, str, str, int]]:
    return {_row_run_key(row) for row in _complete_grid_rows(rows)}


def _jsonable(value):
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(_jsonable(payload), file, indent=2)
        file.write("\n")
    temporary.replace(path)


def _resume_signature(config: dict) -> dict:
    return {
        key: _jsonable(value)
        for key, value in config.items()
        if key not in RESUME_AXIS_CONFIG_KEYS
    }


def _load_resume_signature(output_dir: Path) -> dict | None:
    state_path = output_dir / RESUME_STATE_JSON
    if state_path.exists():
        try:
            with state_path.open(encoding="utf-8") as file:
                state = json.load(file)
        except json.JSONDecodeError:
            state = {}
        signature = state.get("config_signature")
        if isinstance(signature, dict):
            return signature
    summary_path = output_dir / SUMMARY_JSON
    if not summary_path.exists():
        return None
    try:
        with summary_path.open(encoding="utf-8") as file:
            summary = json.load(file)
    except json.JSONDecodeError:
        return None
    config = summary.get("config")
    if not isinstance(config, dict):
        return None
    return _resume_signature(config)


def _validate_resume_signature(output_dir: Path, current_signature: dict) -> None:
    existing_signature = _load_resume_signature(output_dir)
    if existing_signature is None:
        return
    shared_keys = set(existing_signature).intersection(current_signature)
    mismatches = [
        key
        for key in sorted(shared_keys)
        if existing_signature[key] != current_signature[key]
    ]
    if mismatches:
        details = "; ".join(
            (
                f"{key}: existing={existing_signature[key]!r}, "
                f"requested={current_signature[key]!r}"
            )
            for key in mismatches
        )
        raise ValueError(
            "cannot resume dynamic extensions with different settings: " + details
        )


def _write_resume_state(
    output_dir: Path,
    *,
    config_signature: dict,
    total_units: int,
    completed_units: int,
    status: str,
) -> None:
    _write_json(
        output_dir / RESUME_STATE_JSON,
        {
            "status": status,
            "completed_units": completed_units,
            "total_units": total_units,
            "config_signature": config_signature,
        },
    )


def _summary_mapping(
    rows: list[dict[str, float | int | bool | str | None]],
) -> dict[str, dict[str, float | int | bool | str | None]]:
    return {
        f"{row['method']}|{row['grid']}|{row['condition']}|{row['representation']}": {
            key: value
            for key, value in row.items()
            if key not in {"method", "grid", "condition", "representation"}
        }
        for row in _summary_rows(rows)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/validation_results/dynamic_extensions"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n-seeds", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=360)
    parser.add_argument("--n-surrogates", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument("--no-fdr", action="store_true")
    parser.add_argument("--methods", default="cgc,cgc-star")
    parser.add_argument(
        "--grid",
        default="lag1_context1,lag2_context2,bout_bounded_lag3_context4",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = _parse_methods(args.methods)
    specs = _parse_grid(args.grid)
    adjacency, rise_sequence, fall_sequence = _truth_graphs()
    conditions = _dynamic_conditions(rise_sequence, fall_sequence)
    seeds = tuple(range(args.seed_start, args.seed_start + args.n_seeds))
    if not seeds:
        raise ValueError("n_seeds must be positive")

    config = {
        "resume_schema_version": 2,
        "methods": methods,
        "grid": tuple(spec.name for spec in specs),
        "n_seeds": args.n_seeds,
        "seed_start": args.seed_start,
        "n_steps": args.n_steps,
        "n_surrogates": args.n_surrogates,
        "score_threshold": args.score_threshold,
        "no_fdr": args.no_fdr,
    }
    run_specs = [
        (method, spec, condition, seed)
        for method in methods
        for spec in specs
        for condition in conditions
        for seed in seeds
    ]
    target_keys = {
        (method, spec.name, condition.name, seed)
        for method, spec, condition, seed in run_specs
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = args.output_dir / GRID_ROWS_CSV
    rows: list[dict[str, float | int | bool | str | None]] = []
    completed_keys: set[tuple[str, str, str, int]] = set()
    skipped_units = 0
    if args.resume:
        config_signature = _resume_signature(config)
        _validate_resume_signature(args.output_dir, config_signature)
        existing_rows = _complete_grid_rows(_read_grid_rows(grid_path))
        rows = [row for row in existing_rows if _row_run_key(row) in target_keys]
        completed_keys = _completed_run_keys(rows)
        skipped_units = len(completed_keys)
        _write_resume_state(
            args.output_dir,
            config_signature=config_signature,
            total_units=len(run_specs),
            completed_units=len(completed_keys),
            status="running",
        )

    for method, spec, condition, seed in run_specs:
        key = (method, spec.name, condition.name, seed)
        if args.resume and key in completed_keys:
            continue
        rows.extend(
            _rows_for_spec(
                adjacency=adjacency,
                condition=condition,
                seed=seed,
                method=method,
                spec=spec,
                n_steps=args.n_steps,
                n_surrogates=args.n_surrogates,
                score_threshold=args.score_threshold,
                no_fdr=args.no_fdr,
            )
        )
        if args.resume:
            completed_keys.add(key)
            _write_csv(grid_path, rows)
            _write_resume_state(
                args.output_dir,
                config_signature=config_signature,
                total_units=len(run_specs),
                completed_units=len(completed_keys),
                status="running",
            )

    summary_rows = _summary_rows(rows)
    contrast_rows = _contrast_rows(summary_rows)
    _write_csv(grid_path, rows)
    _write_csv(args.output_dir / REPRESENTATION_SUMMARY_CSV, summary_rows)
    _write_csv(args.output_dir / RISE_FALL_CONTRASTS_CSV, contrast_rows)
    _write_json(
        args.output_dir / SUMMARY_JSON,
        {
            "status": "complete",
            "config": config,
            "rows": len(rows),
            "means": _summary_mapping(rows),
            "contrasts": contrast_rows,
        },
    )

    if args.resume:
        _write_resume_state(
            args.output_dir,
            config_signature=config_signature,
            total_units=len(run_specs),
            completed_units=len(completed_keys),
            status="complete",
        )
        print(
            f"wrote {len(rows)} rows to {args.output_dir} "
            f"(resumed {skipped_units} completed units; "
            f"ran {len(run_specs) - skipped_units} units)"
        )
        return

    print(f"wrote {len(rows)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
