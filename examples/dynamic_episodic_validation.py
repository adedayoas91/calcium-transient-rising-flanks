"""Generate validation tables for dynamic-A episodic synthetic simulations."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from calcium_transient_rising_flank import (
    CausalisedGC,
    DynamicSimulationConfig,
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
    "fall_propagated_total",
    "edge_presence_total",
    "edge_prevalence_mean",
    "edge_prevalence_max",
    "node_presence_total",
    "node_prevalence_mean",
    "active_union_nodes",
    "rise_candidate_edges",
    "rise_candidate_matches",
    "rise_candidate_retained_runs",
    "rise_candidate_dropped_runs",
    "rise_candidate_retained_frames",
    "rise_candidate_context_frames",
    "rise_candidate_mean_overlap",
)
METHOD_CHOICES = ("cgc", "fcgc", "cgc-star", "cgc*")
TOPOLOGY_CHOICES = ("sequence", "generated")
GRID_RUNS_CSV = "dynamic_grid_runs.csv"
REPRESENTATION_SUMMARY_CSV = "representation_summary.csv"
RISE_FALL_CONTRASTS_CSV = "rise_fall_contrasts.csv"
SUMMARY_JSON = "summary.json"
RESUME_STATE_JSON = "resume_state.json"
REPRESENTATION_ORDER = ("full", "deconvolved", "rise", "fall", "fall_residual")
EXPECTED_REPRESENTATIONS = frozenset(REPRESENTATION_ORDER)
RESUME_KEY_COLUMNS = ("method", "event_mode", "condition", "seed")
RESUME_AXIS_CONFIG_KEYS = {"event_modes", "methods", "method", "n_seeds"}
INTEGER_ROW_COLUMNS = {
    "seed",
    "true_positives",
    "false_positives",
    "false_negatives",
    "truth_edges",
    "n_episodes",
    "min_rise_length",
    "rise_waveform_length",
    "edge_presence_total",
    "node_presence_total",
    "active_union_nodes",
    "rise_candidate_edges",
    "rise_candidate_matches",
    "rise_candidate_retained_runs",
    "rise_candidate_dropped_runs",
    "rise_candidate_retained_frames",
    "rise_candidate_context_frames",
}
FLOAT_ROW_COLUMNS = set(SUMMARY_METRICS) | {
    "precision",
    "recall",
    "false_positive_rate",
    "f1",
    "orientation_accuracy",
    "edge_density",
    "min_fall_to_rise",
    "fall_initial_scale",
    "fall_initial_ceiling_fraction",
}


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return int(value)


def _parse_event_modes(value: str) -> tuple[str, ...]:
    modes = tuple(mode.strip() for mode in value.split(",") if mode.strip())
    if not modes:
        raise ValueError("at least one event mode is required")
    invalid = sorted(set(modes) - {"compressed", "physical"})
    if invalid:
        raise ValueError(f"unsupported event modes: {', '.join(invalid)}")
    return modes


def _parse_methods(value: str) -> tuple[str, ...]:
    methods = tuple(method.strip() for method in value.split(",") if method.strip())
    if not methods:
        raise ValueError("at least one method is required")
    invalid = sorted(set(methods) - set(METHOD_CHOICES))
    if invalid:
        raise ValueError(f"unsupported methods: {', '.join(invalid)}")
    return methods


def _truth_graphs() -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    a1 = np.array(
        [
            [False, True, False, False, False],
            [False, False, True, False, False],
            [False, False, False, True, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
        ]
    )
    a2 = np.array(
        [
            [False, False, False, False, True],
            [False, False, False, False, False],
            [False, True, False, False, False],
            [False, False, True, False, False],
            [False, False, False, False, False],
        ]
    )
    return a1 | a2, (a1, a2)


def _active_node_sequence() -> tuple[np.ndarray, ...]:
    return (
        np.array([True, True, True, True, False]),
        np.array([True, True, True, True, True]),
    )


def _edge_density(adjacency: np.ndarray) -> float:
    values = np.asarray(adjacency, dtype=bool)
    off_diag: np.ndarray = ~np.eye(values.shape[0], dtype=bool)
    return float(np.count_nonzero(values[off_diag]) / np.count_nonzero(off_diag))


def _f1(precision: float, recall: float) -> float:
    denominator = precision + recall
    return 0.0 if denominator == 0.0 else 2.0 * precision * recall / denominator


def _dynamic_metadata(run) -> dict[str, float | int | str | None]:
    episodes = run.dataset.episodes
    dynamic_config = getattr(run.condition, "dynamic_config", None)
    edge_presence_counts = getattr(run.dataset, "edge_presence_counts", None)
    edge_prevalence = getattr(run.dataset, "edge_prevalence", None)
    node_presence_counts = getattr(run.dataset, "node_presence_counts", None)
    node_prevalence = getattr(run.dataset, "node_prevalence", None)
    if not episodes:
        return {
            "n_episodes": 0,
            "min_rise_length": None,
            "rise_waveform_length": None,
            "min_fall_to_rise": None,
            "fall_propagated_total": None,
            "fall_state_mode": None,
            "fall_initial_scale": None,
            "fall_initial_ceiling_fraction": None,
            "edge_presence_total": None,
            "edge_prevalence_mean": None,
            "edge_prevalence_max": None,
            "node_presence_total": None,
            "node_prevalence_mean": None,
            "active_union_nodes": None,
        }
    fall_propagated = 0.0
    if run.dataset.propagated_events is not None:
        for episode in episodes:
            fall_propagated += float(
                run.dataset.propagated_events[
                    :, episode.fall_start : episode.fall_stop
                ].sum()
            )
    edge_mask = None
    if edge_presence_counts is not None:
        edge_mask = np.asarray(edge_presence_counts) > 0
    edge_prevalence_values = (
        None
        if edge_prevalence is None or edge_mask is None or not np.any(edge_mask)
        else np.asarray(edge_prevalence, dtype=float)[edge_mask]
    )
    return {
        "n_episodes": len(episodes),
        "min_rise_length": min(episode.rise_length for episode in episodes),
        "rise_waveform_length": None
        if dynamic_config is None
        else dynamic_config.rise_waveform_length,
        "min_fall_to_rise": min(
            episode.fall_length / episode.rise_length for episode in episodes
        ),
        "fall_propagated_total": fall_propagated,
        "fall_state_mode": None
        if dynamic_config is None
        else dynamic_config.fall_state_mode,
        "fall_initial_scale": None
        if dynamic_config is None
        else dynamic_config.fall_initial_scale,
        "fall_initial_ceiling_fraction": None
        if dynamic_config is None
        else dynamic_config.fall_initial_ceiling_fraction,
        "edge_presence_total": None
        if edge_presence_counts is None
        else int(np.asarray(edge_presence_counts, dtype=int).sum()),
        "edge_prevalence_mean": None
        if edge_prevalence_values is None
        else float(np.mean(edge_prevalence_values)),
        "edge_prevalence_max": None
        if edge_prevalence_values is None
        else float(np.max(edge_prevalence_values)),
        "node_presence_total": None
        if node_presence_counts is None
        else int(np.asarray(node_presence_counts, dtype=int).sum()),
        "node_prevalence_mean": None
        if node_prevalence is None
        else float(np.mean(np.asarray(node_prevalence, dtype=float))),
        "active_union_nodes": None
        if node_presence_counts is None
        else int(np.count_nonzero(np.asarray(node_presence_counts, dtype=int))),
    }


def _rise_candidate_metadata(run, representation: str) -> dict[str, float | int | None]:
    candidates = getattr(run.validation, "rise_flank_candidates", None)
    if representation != "rise" or candidates is None:
        return {
            "rise_candidate_edges": None,
            "rise_candidate_matches": None,
            "rise_candidate_retained_runs": None,
            "rise_candidate_dropped_runs": None,
            "rise_candidate_retained_frames": None,
            "rise_candidate_context_frames": None,
            "rise_candidate_mean_overlap": None,
        }
    supported = candidates.support_counts > 0
    context_indices = candidates.event_indices or candidates.runs.event_indices
    return {
        "rise_candidate_edges": int(np.count_nonzero(candidates.candidate_adjacency)),
        "rise_candidate_matches": len(candidates.matches),
        "rise_candidate_retained_runs": int(candidates.runs.kept_counts.sum()),
        "rise_candidate_dropped_runs": int(candidates.runs.dropped_counts.sum()),
        "rise_candidate_retained_frames": int(
            sum(frames.size for frames in candidates.runs.event_indices)
        ),
        "rise_candidate_context_frames": int(
            sum(frames.size for frames in context_indices)
        ),
        "rise_candidate_mean_overlap": None
        if not np.any(supported)
        else float(np.mean(candidates.mean_overlap_fraction[supported])),
    }


def _rows_for_mode(
    *,
    adjacency: np.ndarray,
    conditions: tuple[SyntheticCondition, ...],
    seeds: tuple[int, ...],
    method: str,
    event_mode: str,
    args: argparse.Namespace,
) -> list[dict[str, float | int | str | None]]:
    runs = run_synthetic_grid(
        adjacency,
        conditions=conditions,
        seeds=seeds,
        estimator_factory=lambda: CausalisedGC(
            max_lag=args.max_lag,
            tau=args.tau,
            n_pasts=args.n_pasts,
            n_surrogates=args.n_surrogates,
            score_threshold=args.score_threshold,
            event_mode=event_mode,
            method=method,
            fdr=not args.no_fdr,
            min_rise_run_samples=args.min_rise_run_samples,
            rise_candidate_filter=args.rise_candidate_filter,
            rise_match_min_lag=args.rise_match_min_lag,
            rise_match_max_lag=args.rise_match_max_lag,
            rise_match_min_overlap_samples=args.rise_match_min_overlap_samples,
            rise_match_min_overlap_fraction=args.rise_match_min_overlap_fraction,
            rise_run_context_samples=args.rise_run_context_samples,
        ),
        n_steps=args.n_steps,
    )
    rows: list[dict[str, float | int | str | None]] = []
    for run in runs:
        metadata = _dynamic_metadata(run)
        for label, graph in run.validation.graphs.items():
            recovery = run.validation.recovery[label]
            truth = run.validation.truth[label]
            rows.append(
                {
                    "event_mode": event_mode,
                    "method": method,
                    "condition": run.condition.name,
                    "simulator_mode": run.condition.simulator_mode,
                    "seed": run.seed,
                    "representation": label,
                    "precision": recovery.precision,
                    "recall": recovery.recall,
                    "false_positive_rate": recovery.false_positive_rate,
                    "f1": _f1(recovery.precision, recovery.recall),
                    "orientation_accuracy": recovery.orientation_accuracy,
                    "true_positives": recovery.true_positives,
                    "false_positives": recovery.false_positives,
                    "false_negatives": recovery.false_negatives,
                    "edge_density": _edge_density(graph.adjacency),
                    "truth_edges": int(np.count_nonzero(truth)),
                    **metadata,
                    **_rise_candidate_metadata(run, label),
                }
            )
    return rows


def _summary_rows(
    rows: list[dict[str, float | int | str | None]],
) -> list[dict[str, float | int | str | None]]:
    grouped: dict[
        tuple[str, str, str, str], list[dict[str, float | int | str | None]]
    ] = defaultdict(list)
    for row in rows:
        key = (
            str(row["method"]),
            str(row["event_mode"]),
            str(row["condition"]),
            str(row["representation"]),
        )
        grouped[key].append(row)

    summaries: list[dict[str, float | int | str | None]] = []
    for key, values in sorted(grouped.items()):
        method, event_mode, condition, representation = key
        entry: dict[str, float | int | str | None] = {
            "method": method,
            "event_mode": event_mode,
            "condition": condition,
            "representation": representation,
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


def _summary_mapping(
    rows: list[dict[str, float | int | str | None]],
) -> dict[str, dict[str, float | int | str | None]]:
    return {
        (
            f"{row['method']}|{row['event_mode']}|"
            f"{row['condition']}|{row['representation']}"
        ): {
            key: value
            for key, value in row.items()
            if key not in {"method", "event_mode", "condition", "representation"}
        }
        for row in _summary_rows(rows)
    }


def _contrast_rows(
    summaries: list[dict[str, float | int | str | None]],
) -> list[dict[str, float | int | str | None]]:
    by_condition: dict[
        tuple[str, str, str], dict[str, dict[str, float | int | str | None]]
    ]
    by_condition = defaultdict(dict)
    for row in summaries:
        by_condition[
            (str(row["method"]), str(row["event_mode"]), str(row["condition"]))
        ][str(row["representation"])] = row

    contrasts: list[dict[str, float | int | str | None]] = []
    for (method, event_mode, condition), representation_rows in sorted(
        by_condition.items()
    ):
        contrast: dict[str, float | int | str | None] = {
            "method": method,
            "event_mode": event_mode,
            "condition": condition,
        }
        for metric in SUMMARY_METRICS:
            column = f"{metric}_mean"
            rise = representation_rows.get("rise", {}).get(column)
            fall = representation_rows.get("fall", {}).get(column)
            full = representation_rows.get("full", {}).get(column)
            deconvolved = representation_rows.get("deconvolved", {}).get(column)
            fall_residual = representation_rows.get("fall_residual", {}).get(column)
            contrast[f"rise_minus_fall_{metric}"] = (
                None if rise is None or fall is None else float(rise) - float(fall)
            )
            contrast[f"rise_minus_fall_residual_{metric}"] = (
                None
                if rise is None or fall_residual is None
                else float(rise) - float(fall_residual)
            )
            contrast[f"rise_minus_full_{metric}"] = (
                None if rise is None or full is None else float(rise) - float(full)
            )
            contrast[f"rise_minus_deconvolved_{metric}"] = (
                None
                if rise is None or deconvolved is None
                else float(rise) - float(deconvolved)
            )
            contrast[f"full_minus_fall_{metric}"] = (
                None if full is None or fall is None else float(full) - float(fall)
            )
        contrasts.append(contrast)
    return contrasts


def _write_csv(path: Path, rows: list[dict[str, float | int | str | None]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _coerce_grid_row(
    row: dict[str, str | None],
) -> dict[str, float | int | str | None]:
    coerced: dict[str, float | int | str | None] = {}
    for key, value in row.items():
        if value is None or value == "":
            coerced[key] = None
        elif key in INTEGER_ROW_COLUMNS:
            coerced[key] = int(value)
        elif key in FLOAT_ROW_COLUMNS:
            coerced[key] = float(value)
        else:
            coerced[key] = value
    return coerced


def _read_grid_rows(path: Path) -> list[dict[str, float | int | str | None]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as file:
        return [_coerce_grid_row(row) for row in csv.DictReader(file)]


def _row_run_key(row: dict[str, float | int | str | None]) -> tuple[str, str, str, int]:
    method = row.get("method")
    event_mode = row.get("event_mode")
    condition = row.get("condition")
    seed = row.get("seed")
    if method is None or event_mode is None or condition is None or seed is None:
        values = {
            "method": method,
            "event_mode": event_mode,
            "condition": condition,
            "seed": seed,
        }
        missing = [column for column, value in values.items() if value is None]
        raise ValueError(
            "resume grid row is missing required columns: " + ", ".join(missing)
        )
    return (
        str(method),
        str(event_mode),
        str(condition),
        int(seed),
    )


def _run_key(
    method: str,
    event_mode: str,
    condition: SyntheticCondition,
    seed: int,
) -> tuple[str, str, str, int]:
    return (method, event_mode, condition.name, seed)


def _complete_grid_rows(
    rows: list[dict[str, float | int | str | None]],
) -> list[dict[str, float | int | str | None]]:
    ordered_keys: list[tuple[str, str, str, int]] = []
    grouped: dict[
        tuple[str, str, str, int], dict[str, dict[str, float | int | str | None]]
    ] = {}
    for row in rows:
        key = _row_run_key(row)
        representation = row.get("representation")
        if representation is None:
            raise ValueError("resume grid row is missing required column: representation")
        if key not in grouped:
            grouped[key] = {}
            ordered_keys.append(key)
        grouped[key][str(representation)] = row

    complete: list[dict[str, float | int | str | None]] = []
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
    rows: list[dict[str, float | int | str | None]],
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
    with temporary.open("w") as file:
        json.dump(_jsonable(payload), file, indent=2)
        file.write("\n")
    temporary.replace(path)


def _resume_signature(
    config: dict[str, float | int | str | tuple[str, ...] | None | bool],
) -> dict:
    return {
        key: _jsonable(value)
        for key, value in config.items()
        if key not in RESUME_AXIS_CONFIG_KEYS
    }


def _load_resume_signature(output_dir: Path) -> dict | None:
    state_path = output_dir / RESUME_STATE_JSON
    if state_path.exists():
        try:
            with state_path.open() as file:
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
        with summary_path.open() as file:
            summary = json.load(file)
    except json.JSONDecodeError:
        return None
    config = summary.get("config")
    if not isinstance(config, dict):
        return None
    return _resume_signature(config)


def _validate_resume_signature(
    output_dir: Path,
    current_signature: dict,
) -> None:
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
            "cannot resume dynamic validation with different run settings: " + details
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/validation_results/dynamic_episodic"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "reuse complete method/event/condition/seed rows from dynamic_grid_runs.csv "
            "and run only missing or incomplete simulation units"
        ),
    )
    parser.add_argument("--n-seeds", type=int, default=_env_int("RF_DYNAMIC_N_SEEDS", 5))
    parser.add_argument(
        "--n-steps", type=int, default=_env_int("RF_DYNAMIC_N_STEPS", 240)
    )
    parser.add_argument(
        "--event-modes",
        default=os.environ.get("RF_DYNAMIC_EVENT_MODES", "compressed,physical"),
        help="comma-separated subset of: compressed,physical",
    )
    parser.add_argument("--max-lag", type=int, default=_env_int("RF_DYNAMIC_MAX_LAG", 1))
    parser.add_argument(
        "--tau",
        type=int,
        default=_env_optional_int("RF_DYNAMIC_TAU"),
        help=(
            "fixed c-GC/c-GC* tested lag; omit to choose the strongest lag "
            "from 1..--max-lag"
        ),
    )
    parser.add_argument(
        "--n-pasts",
        type=int,
        default=_env_optional_int("RF_DYNAMIC_N_PASTS"),
        help="c-GC/c-GC* conditioning/history depth; defaults to max(--max-lag, tau)",
    )
    parser.add_argument(
        "--n-surrogates", type=int, default=_env_int("RF_DYNAMIC_N_SURROGATES", 0)
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=_env_float("RF_DYNAMIC_SCORE_THRESHOLD", 0.1),
    )
    parser.add_argument("--no-fdr", action="store_true")
    parser.add_argument(
        "--method",
        choices=METHOD_CHOICES,
        default="cgc",
        help="single estimator method; ignored when --methods is supplied",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="comma-separated estimator methods for combined output, e.g. cgc,cgc-star",
    )
    parser.add_argument(
        "--dynamic-episodes", type=int, default=_env_int("RF_DYNAMIC_EPISODES", 3)
    )
    parser.add_argument(
        "--topology-mode",
        choices=TOPOLOGY_CHOICES,
        default=os.environ.get("RF_DYNAMIC_TOPOLOGY_MODE", "sequence"),
        help=(
            "'sequence' cycles through locked A_k matrices; 'generated' samples "
            "edge/source additions and deletions around the base union graph"
        ),
    )
    parser.add_argument(
        "--min-rise-length", type=int, default=_env_int("RF_DYNAMIC_MIN_RISE", 20)
    )
    parser.add_argument(
        "--max-rise-length", type=int, default=_env_int("RF_DYNAMIC_MAX_RISE", 28)
    )
    parser.add_argument(
        "--rise-waveform-length",
        type=int,
        default=_env_int("RF_DYNAMIC_RISE_WAVEFORM_LENGTH", 20),
        help="number of samples over which each rise onset is spread",
    )
    parser.add_argument(
        "--fall-state-mode",
        choices=("passive_decay", "stochastic_independent"),
        default=os.environ.get("RF_DYNAMIC_FALL_STATE_MODE", "stochastic_independent"),
    )
    parser.add_argument(
        "--fall-initial-scale",
        type=float,
        default=_env_float("RF_DYNAMIC_FALL_INITIAL_SCALE", 1.0),
    )
    parser.add_argument(
        "--fall-initial-ceiling-fraction",
        type=float,
        default=_env_float("RF_DYNAMIC_FALL_INITIAL_CEILING_FRACTION", 1.0),
        help="maximum fall reset amplitude as a fraction of the rise endpoint",
    )
    parser.add_argument(
        "--edge-dropout-probability",
        type=float,
        default=_env_float("RF_DYNAMIC_EDGE_DROPOUT_PROBABILITY", 0.2),
        help="generated-topology probability that a base edge is absent in one rise",
    )
    parser.add_argument(
        "--edge-addition-probability",
        type=float,
        default=_env_float("RF_DYNAMIC_EDGE_ADDITION_PROBABILITY", 0.02),
        help="generated-topology probability that a non-base edge appears in one rise",
    )
    parser.add_argument(
        "--source-dropout-probability",
        type=float,
        default=_env_float("RF_DYNAMIC_SOURCE_DROPOUT_PROBABILITY", 0.15),
        help="generated-topology probability that a base source stops sending",
    )
    parser.add_argument(
        "--source-recruitment-probability",
        type=float,
        default=_env_float("RF_DYNAMIC_SOURCE_RECRUITMENT_PROBABILITY", 0.25),
        help="generated-topology probability that a non-source becomes causal",
    )
    parser.add_argument(
        "--source-recruitment-edge-probability",
        type=float,
        default=_env_float("RF_DYNAMIC_SOURCE_RECRUITMENT_EDGE_PROBABILITY", 0.25),
        help="target probability for edges from a recruited source neuron",
    )
    parser.add_argument(
        "--min-rise-run-samples",
        type=int,
        default=_env_int("RF_DYNAMIC_MIN_RISE_RUN_SAMPLES", 1),
        help="drop extracted rising-flank runs shorter than this sample count",
    )
    parser.add_argument(
        "--rise-candidate-filter",
        action="store_true",
        default=_env_bool("RF_DYNAMIC_RISE_CANDIDATE_FILTER", False),
        help="mask rise c-GC/c-GC* outputs to shifted-flank candidate pairs",
    )
    parser.add_argument(
        "--rise-match-min-lag",
        type=int,
        default=_env_int("RF_DYNAMIC_RISE_MATCH_MIN_LAG", 1),
        help="minimum source-before-target onset lag for rise candidate matching",
    )
    parser.add_argument(
        "--rise-match-max-lag",
        type=int,
        default=_env_optional_int("RF_DYNAMIC_RISE_MATCH_MAX_LAG"),
        help="maximum onset/overlap lag for rise candidates; defaults to --max-lag",
    )
    parser.add_argument(
        "--rise-match-min-overlap-samples",
        type=int,
        default=_env_optional_int("RF_DYNAMIC_RISE_MATCH_MIN_OVERLAP_SAMPLES"),
        help="minimum shifted-overlap sample count for a candidate match",
    )
    parser.add_argument(
        "--rise-match-min-overlap-fraction",
        type=float,
        default=_env_float("RF_DYNAMIC_RISE_MATCH_MIN_OVERLAP_FRACTION", 0.5),
        help="minimum shifted-overlap fraction for a candidate match",
    )
    parser.add_argument(
        "--rise-run-context-samples",
        type=int,
        default=_env_optional_int("RF_DYNAMIC_RISE_RUN_CONTEXT_SAMPLES"),
        help=(
            "consecutive samples added before/after each retained rise run for "
            "c-GC/c-GC* lag construction; defaults to --max-lag"
        ),
    )
    return parser.parse_args()


def _dynamic_config_from_args(
    args: argparse.Namespace,
    sequence: tuple[np.ndarray, ...],
) -> DynamicSimulationConfig:
    common = {
        "n_episodes": args.dynamic_episodes,
        "min_rise_length": args.min_rise_length,
        "max_rise_length": args.max_rise_length,
        "rise_waveform_length": args.rise_waveform_length,
        "fall_to_rise_ratio_min": 2.1,
        "fall_to_rise_ratio_max": 2.8,
        "initial_activation_probability": 0.7,
        "fall_noise_rate": 0.05,
        "fall_noise_scale": 0.02,
        "fall_state_mode": args.fall_state_mode,
        "fall_initial_scale": args.fall_initial_scale,
        "fall_initial_ceiling_fraction": args.fall_initial_ceiling_fraction,
    }
    if args.topology_mode == "sequence":
        return DynamicSimulationConfig(
            **common,
            edge_dropout_probability=0.0,
            adjacency_sequence=sequence,
            active_node_sequence=_active_node_sequence(),
        )
    return DynamicSimulationConfig(
        **common,
        edge_dropout_probability=args.edge_dropout_probability,
        edge_addition_probability=args.edge_addition_probability,
        source_dropout_probability=args.source_dropout_probability,
        source_recruitment_probability=args.source_recruitment_probability,
        source_recruitment_edge_probability=args.source_recruitment_edge_probability,
    )


def _summary_config(
    args: argparse.Namespace,
    dynamic_config: DynamicSimulationConfig,
    event_modes: tuple[str, ...],
    methods: tuple[str, ...],
) -> dict[str, float | int | str | tuple[str, ...] | None | bool]:
    return {
        "n_seeds": args.n_seeds,
        "n_steps": args.n_steps,
        "event_modes": event_modes,
        "max_lag": args.max_lag,
        "tau": args.tau,
        "n_pasts": args.n_pasts,
        "n_surrogates": args.n_surrogates,
        "score_threshold": args.score_threshold,
        "no_fdr": args.no_fdr,
        "methods": methods,
        "method": methods[0] if len(methods) == 1 else None,
        "dynamic_episodes": args.dynamic_episodes,
        "topology_mode": args.topology_mode,
        "min_rise_length": args.min_rise_length,
        "max_rise_length": args.max_rise_length,
        "rise_waveform_length": args.rise_waveform_length,
        "fall_state_mode": args.fall_state_mode,
        "fall_initial_scale": args.fall_initial_scale,
        "fall_initial_ceiling_fraction": args.fall_initial_ceiling_fraction,
        "edge_dropout_probability": dynamic_config.edge_dropout_probability,
        "edge_addition_probability": dynamic_config.edge_addition_probability,
        "source_dropout_probability": dynamic_config.source_dropout_probability,
        "source_recruitment_probability": dynamic_config.source_recruitment_probability,
        "source_recruitment_edge_probability": (
            dynamic_config.source_recruitment_edge_probability
        ),
        "min_rise_run_samples": args.min_rise_run_samples,
        "rise_candidate_filter": args.rise_candidate_filter,
        "rise_match_min_lag": args.rise_match_min_lag,
        "rise_match_max_lag": args.rise_match_max_lag,
        "rise_match_min_overlap_samples": args.rise_match_min_overlap_samples,
        "rise_match_min_overlap_fraction": args.rise_match_min_overlap_fraction,
        "rise_run_context_samples": args.rise_run_context_samples,
    }


def main() -> None:
    args = parse_args()
    event_modes = _parse_event_modes(args.event_modes)
    methods = _parse_methods(args.methods or args.method)
    adjacency, sequence = _truth_graphs()
    dynamic_config = _dynamic_config_from_args(args, sequence)
    conditions = (
        SyntheticCondition(name="static_union", gamma=0.9),
        SyntheticCondition(
            name="dynamic_a_noncausal_fall",
            gamma=0.9,
            simulator_mode="episodic_dynamic",
            dynamic_config=dynamic_config,
        ),
    )
    seeds = tuple(range(1, args.n_seeds + 1))
    if not seeds:
        raise ValueError("n_seeds must be positive")
    config = _summary_config(args, dynamic_config, event_modes, methods)
    run_specs: list[tuple[str, str, SyntheticCondition, int]] = [
        (method, mode, condition, seed)
        for method in methods
        for mode in event_modes
        for condition in conditions
        for seed in seeds
    ]
    target_keys = {
        _run_key(method, mode, condition, seed)
        for method, mode, condition, seed in run_specs
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = args.output_dir / GRID_RUNS_CSV
    rows: list[dict[str, float | int | str | None]] = []
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

    if args.resume:
        for method, mode, condition, seed in run_specs:
            key = _run_key(method, mode, condition, seed)
            if key in completed_keys:
                continue
            rows.extend(
                _rows_for_mode(
                    adjacency=adjacency,
                    conditions=(condition,),
                    seeds=(seed,),
                    method=method,
                    event_mode=mode,
                    args=args,
                )
            )
            completed_keys.add(key)
            _write_csv(grid_path, rows)
            _write_resume_state(
                args.output_dir,
                config_signature=config_signature,
                total_units=len(run_specs),
                completed_units=len(completed_keys),
                status="running",
            )
    else:
        for method in methods:
            for mode in event_modes:
                rows.extend(
                    _rows_for_mode(
                        adjacency=adjacency,
                        conditions=conditions,
                        seeds=seeds,
                        method=method,
                        event_mode=mode,
                        args=args,
                    )
                )

    summary_rows = _summary_rows(rows)
    contrast_rows = _contrast_rows(summary_rows)
    _write_csv(grid_path, rows)
    _write_csv(args.output_dir / REPRESENTATION_SUMMARY_CSV, summary_rows)
    _write_csv(args.output_dir / RISE_FALL_CONTRASTS_CSV, contrast_rows)

    summary = {
        "config": config,
        "rows": len(rows),
        "means": _summary_mapping(rows),
        "contrasts": contrast_rows,
    }
    _write_json(args.output_dir / SUMMARY_JSON, summary)

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
