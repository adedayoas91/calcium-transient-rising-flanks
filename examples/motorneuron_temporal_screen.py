"""Run a truth-free temporal-resolvability screen on motoneuron recordings.

This empirical screen does not estimate causal validity because directed ground
truth is unavailable. It reports three-state candidate density, cross-fitted
direction replication, and episode-wise timing-shift null diagnostics before
any c-GC fit.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from calcium_transient_rising_flank import (
    TemporalPriorResult,
    build_representations,
    build_temporal_prior,
    temporal_prior_state_masks,
)
from calcium_transient_rising_flank.checkpointing import (
    JsonUnitCheckpointStore,
    atomic_write_json,
)


DEFAULT_DATA_DIR = Path("data/motoneurons")
DEFAULT_OUTPUT_DIR = Path("outputs/motorneurons/temporal_screen")
CASE_FILES = {
    "A": (
        "dff_dict_with_bad_neurons.pkl",
        "middle_dict.pkl",
        "bad neurons and artefacts retained",
    ),
    "B": (
        "dff_removed_dict.pkl",
        "middle_removed_dict.pkl",
        "bad neurons removed; artefacts retained",
    ),
    "C": (
        "dff_corrected_dict.pkl",
        "middle_removed_dict.pkl",
        "bad neurons removed; artefacts corrected",
    ),
    "D": (
        "dff_smoothed_dict.pkl",
        "middle_removed_dict.pkl",
        "bad neurons removed; artefacts corrected; smoothed",
    ),
}
DIAGNOSTIC_GATES = {
    "maximum_candidate_density": 0.60,
    "minimum_directional_replication": 0.80,
    "require_directional_fraction_above_null_p95": True,
}


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _recording_label(key: tuple[int, int]) -> str:
    return f"F{key[0]}T{key[1]}"


def load_case_traces(
    data_dir: Path,
    *,
    cases: Sequence[str],
    recordings: set[str] | None,
) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    for case_label in cases:
        trace_file, middle_file, description = CASE_FILES[case_label]
        trace_store = _load_pickle(data_dir / trace_file)
        middle_store = _load_pickle(data_dir / middle_file)
        for key, values in sorted(trace_store.items()):
            recording = _recording_label(key)
            if recordings is not None and recording not in recordings:
                continue
            if key not in middle_store:
                continue
            loaded.append(
                {
                    "case": case_label,
                    "description": description,
                    "recording": recording,
                    "fish": int(key[0]),
                    "trial": int(key[1]),
                    "mid": int(middle_store[key]),
                    "traces": np.asarray(values, dtype=float),
                }
            )
    return loaded


def population_episode_segments(
    rise: np.ndarray,
    *,
    merge_gap_frames: int,
    minimum_participating_rois: int,
) -> np.ndarray:
    """Infer population activity bouts used only as screening strata."""

    values = np.asarray(rise, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("rise must have shape (n_rois, n_frames)")
    if merge_gap_frames < 0:
        raise ValueError("merge_gap_frames cannot be negative")
    if minimum_participating_rois < 1:
        raise ValueError("minimum_participating_rois must be positive")
    active_frames = np.flatnonzero(np.any(values > 0.0, axis=0))
    segments = np.full(values.shape[1], -1, dtype=int)
    if active_frames.size == 0:
        return segments
    breaks = np.flatnonzero(
        np.diff(active_frames) > merge_gap_frames + 1
    ) + 1
    episode_id = 0
    for group in np.split(active_frames, breaks):
        start = int(group[0])
        stop = int(group[-1]) + 1
        participants = int(np.count_nonzero(np.any(values[:, start:stop] > 0.0, axis=1)))
        if participants < minimum_participating_rois:
            continue
        segments[start:stop] = episode_id
        episode_id += 1
    return segments


def fixed_window_segments(n_frames: int, *, window_frames: int) -> np.ndarray:
    """Partition a recording into non-overlapping temporal validation blocks."""

    if n_frames < 1 or window_frames < 2:
        raise ValueError("n_frames must be positive and window_frames at least two")
    return np.arange(n_frames, dtype=int) // int(window_frames)


def crossfit_episode_folds(
    segment_ids: np.ndarray,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    episodes = tuple(int(value) for value in np.unique(segment_ids) if value >= 0)
    if len(episodes) < 4:
        raise ValueError("at least four inferred episodes are required")
    first = episodes[::2]
    second = episodes[1::2]
    if len(first) < 2 or len(second) < 2:
        raise ValueError("each cross-fit half must contain at least two episodes")
    return ((first, second), (second, first))


def subset_segments(segment_ids: np.ndarray, episode_ids: Iterable[int]) -> np.ndarray:
    selected = np.asarray(segment_ids, dtype=int).copy()
    keep = np.isin(selected, np.asarray(tuple(episode_ids), dtype=int))
    selected[~keep] = -1
    return selected


def episode_shift_null(
    rise: np.ndarray,
    segment_ids: np.ndarray,
    *,
    random_state: int,
) -> np.ndarray:
    """Shift every ROI independently inside each inferred episode."""

    values = np.asarray(rise, dtype=float)
    segments = np.asarray(segment_ids, dtype=int)
    if values.ndim != 2 or segments.shape != (values.shape[1],):
        raise ValueError("rise and segment_ids must share the time axis")
    rng = np.random.default_rng(random_state)
    shifted = np.zeros_like(values)
    for episode_id in np.unique(segments):
        if episode_id < 0:
            continue
        frames = np.flatnonzero(segments == episode_id)
        if frames.size == 0:
            continue
        episode_values = values[:, frames]
        for roi in range(values.shape[0]):
            offset = int(rng.integers(0, frames.size))
            shifted[roi, frames] = np.roll(episode_values[roi], offset)
    return shifted


def build_prior(
    rise: np.ndarray,
    segment_ids: np.ndarray,
    *,
    min_run_samples: int,
    max_onset_lag: int,
    deadband: int,
) -> TemporalPriorResult:
    return build_temporal_prior(
        rise,
        segment_ids,
        min_run_samples=min_run_samples,
        max_onset_lag=max_onset_lag,
        timing_deadband=deadband,
        minimum_decisive_support=2,
        consistency_threshold=0.75,
        beta_prior_concentration=1.0,
    )


def empirical_prior_metrics(
    prior: TemporalPriorResult,
    *,
    heldout: TemporalPriorResult | None = None,
) -> dict[str, float | int | None]:
    states = temporal_prior_state_masks(prior)
    n_nodes = states.directional.shape[0]
    upper = np.triu(np.ones((n_nodes, n_nodes), dtype=bool), 1)
    off_diagonal = ~np.eye(n_nodes, dtype=bool)
    directional_pairs = (states.directional | states.directional.T) & upper
    ambiguous_pairs = states.ambiguous & upper
    unmatched_pairs = states.unmatched & upper
    pair_count = int(np.count_nonzero(upper))
    direction_count = int(np.count_nonzero(off_diagonal))
    directional_count = int(np.count_nonzero(states.directional))
    replicated = 0
    if heldout is not None:
        heldout_states = temporal_prior_state_masks(heldout)
        if heldout_states.directional.shape != states.directional.shape:
            raise ValueError("screen and heldout priors must have matching shapes")
        replicated = int(
            np.count_nonzero(states.directional & heldout_states.directional)
        )
    return {
        "pair_count": pair_count,
        "direction_count": direction_count,
        "directional_pair_fraction": (
            0.0 if pair_count == 0 else np.count_nonzero(directional_pairs) / pair_count
        ),
        "ambiguous_pair_fraction": (
            0.0 if pair_count == 0 else np.count_nonzero(ambiguous_pairs) / pair_count
        ),
        "unmatched_pair_fraction": (
            0.0 if pair_count == 0 else np.count_nonzero(unmatched_pairs) / pair_count
        ),
        "candidate_density": (
            0.0
            if direction_count == 0
            else np.count_nonzero(prior.robust_hard_mask & off_diagonal)
            / direction_count
        ),
        "screen_directional_count": directional_count,
        "replicated_directional_count": replicated,
        "heldout_directional_replication": (
            None if directional_count == 0 else replicated / directional_count
        ),
    }


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=float)))


def _quantile(values: Sequence[float], probability: float) -> float | None:
    return (
        None
        if not values
        else float(np.quantile(np.asarray(values, dtype=float), probability))
    )


def run_screen(
    records: Sequence[dict[str, Any]],
    *,
    max_onset_lags: Sequence[int],
    deadbands: Sequence[int],
    tolerance: float,
    min_run_samples: int,
    merge_gap_frames: int,
    minimum_participating_rois: int,
    segment_mode: str,
    window_frames: int,
    n_nulls: int,
    random_state: int,
    progress: bool = False,
    checkpoint_store: JsonUnitCheckpointStore | None = None,
    resume: bool = False,
) -> list[dict[str, Any]]:
    if n_nulls < 1:
        raise ValueError("n_nulls must be positive")
    rows: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        unit_id = f"{record['case']}|{record['recording']}"
        if resume and checkpoint_store is not None:
            completed_rows = checkpoint_store.load_rows(unit_id)
            if completed_rows is not None:
                rows.extend(completed_rows)
                if progress:
                    print(f"resumed {unit_id}", flush=True)
                continue
        unit_start = len(rows)
        traces = np.asarray(record["traces"], dtype=float)
        rise = build_representations(traces, tolerance=tolerance).rise
        if segment_mode == "fixed_windows":
            segment_ids = fixed_window_segments(
                traces.shape[1],
                window_frames=window_frames,
            )
        elif segment_mode == "population_bouts":
            segment_ids = population_episode_segments(
                rise,
                merge_gap_frames=merge_gap_frames,
                minimum_participating_rois=minimum_participating_rois,
            )
        else:
            raise ValueError(
                "segment_mode must be 'fixed_windows' or 'population_bouts'"
            )
        episode_count = int(np.count_nonzero(np.unique(segment_ids) >= 0))
        try:
            folds = crossfit_episode_folds(segment_ids)
        except ValueError as error:
            rows.append(
                {
                    "case": record["case"],
                    "recording": record["recording"],
                    "status": "skipped",
                    "skip_reason": str(error),
                    "n_rois": traces.shape[0],
                    "n_frames": traces.shape[1],
                    "inferred_episode_count": episode_count,
                }
            )
            if checkpoint_store is not None:
                checkpoint_store.save_rows(unit_id, rows[unit_start:])
            continue
        for lag_index, max_onset_lag in enumerate(max_onset_lags):
            for deadband_index, deadband in enumerate(deadbands):
                for fold_index, (screen_ids, test_ids) in enumerate(folds):
                    screen_segments = subset_segments(segment_ids, screen_ids)
                    test_segments = subset_segments(segment_ids, test_ids)
                    screen_prior = build_prior(
                        rise,
                        screen_segments,
                        min_run_samples=min_run_samples,
                        max_onset_lag=int(max_onset_lag),
                        deadband=int(deadband),
                    )
                    test_prior = build_prior(
                        rise,
                        test_segments,
                        min_run_samples=min_run_samples,
                        max_onset_lag=int(max_onset_lag),
                        deadband=int(deadband),
                    )
                    observed = empirical_prior_metrics(
                        screen_prior,
                        heldout=test_prior,
                    )
                    null_directional: list[float] = []
                    null_density: list[float] = []
                    null_replication: list[float] = []
                    seed_base = (
                        random_state
                        + record_index * 100_000
                        + lag_index * 10_000
                        + deadband_index * 1_000
                        + fold_index * 100
                    )
                    for null_index in range(n_nulls):
                        shifted = episode_shift_null(
                            rise,
                            segment_ids,
                            random_state=seed_base + null_index,
                        )
                        null_screen = build_prior(
                            shifted,
                            screen_segments,
                            min_run_samples=min_run_samples,
                            max_onset_lag=int(max_onset_lag),
                            deadband=int(deadband),
                        )
                        null_test = build_prior(
                            shifted,
                            test_segments,
                            min_run_samples=min_run_samples,
                            max_onset_lag=int(max_onset_lag),
                            deadband=int(deadband),
                        )
                        null_metrics = empirical_prior_metrics(
                            null_screen,
                            heldout=null_test,
                        )
                        null_directional.append(
                            float(null_metrics["directional_pair_fraction"])
                        )
                        null_density.append(float(null_metrics["candidate_density"]))
                        if null_metrics["heldout_directional_replication"] is not None:
                            null_replication.append(
                                float(null_metrics["heldout_directional_replication"])
                            )
                    null_directional_p95 = _quantile(null_directional, 0.95)
                    rows.append(
                        {
                            "case": record["case"],
                            "description": record["description"],
                            "recording": record["recording"],
                            "fish": record["fish"],
                            "trial": record["trial"],
                            "status": "ok",
                            "skip_reason": None,
                            "n_rois": traces.shape[0],
                            "n_frames": traces.shape[1],
                            "inferred_episode_count": episode_count,
                            "fold": fold_index,
                            "screen_episode_count": len(screen_ids),
                            "heldout_episode_count": len(test_ids),
                            "max_onset_lag_frames": int(max_onset_lag),
                            "deadband_frames": int(deadband),
                            "min_run_samples": min_run_samples,
                            "segment_mode": segment_mode,
                            "window_frames": window_frames,
                            "merge_gap_frames": merge_gap_frames,
                            "minimum_participating_rois": minimum_participating_rois,
                            "tolerance": tolerance,
                            "n_nulls": n_nulls,
                            **observed,
                            "null_directional_pair_fraction_mean": _mean(
                                null_directional
                            ),
                            "null_directional_pair_fraction_p95": null_directional_p95,
                            "null_candidate_density_mean": _mean(null_density),
                            "null_heldout_directional_replication_mean": _mean(
                                null_replication
                            ),
                            "directional_fraction_above_null_p95": (
                                None
                                if null_directional_p95 is None
                                else float(observed["directional_pair_fraction"])
                                - null_directional_p95
                            ),
                        }
                    )
        if checkpoint_store is not None:
            checkpoint_store.save_rows(unit_id, rows[unit_start:])
        if progress:
            print(
                f"completed case={record['case']} recording={record['recording']}",
                flush=True,
            )
    return rows


def summarize_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [row for row in rows if row["status"] == "ok"]
    keys = (
        "case",
        "description",
        "recording",
        "n_rois",
        "n_frames",
        "inferred_episode_count",
        "max_onset_lag_frames",
        "deadband_frames",
        "min_run_samples",
        "segment_mode",
        "window_frames",
        "merge_gap_frames",
        "minimum_participating_rois",
        "tolerance",
        "n_nulls",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in valid:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    metrics = (
        "directional_pair_fraction",
        "ambiguous_pair_fraction",
        "unmatched_pair_fraction",
        "candidate_density",
        "heldout_directional_replication",
        "null_directional_pair_fraction_mean",
        "null_directional_pair_fraction_p95",
        "null_candidate_density_mean",
        "null_heldout_directional_replication_mean",
        "directional_fraction_above_null_p95",
    )
    output: list[dict[str, Any]] = []
    for key, members in sorted(groups.items()):
        item: dict[str, Any] = dict(zip(keys, key))
        item["n_folds"] = len(members)
        for metric in metrics:
            values = [
                float(member[metric])
                for member in members
                if member[metric] is not None
            ]
            item[f"{metric}_mean"] = _mean(values)
        replication = item["heldout_directional_replication_mean"]
        null_excess = item["directional_fraction_above_null_p95_mean"]
        null_gate_passes = (
            not DIAGNOSTIC_GATES[
                "require_directional_fraction_above_null_p95"
            ]
            or (null_excess is not None and null_excess > 0.0)
        )
        item["passes_empirical_diagnostics"] = bool(
            item["candidate_density_mean"]
            <= DIAGNOSTIC_GATES["maximum_candidate_density"]
            and replication is not None
            and replication
            >= DIAGNOSTIC_GATES["minimum_directional_replication"]
            and null_gate_passes
        )
        output.append(item)
    return output


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    summary: Sequence[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "screen_rows.csv", rows)
    if summary:
        _write_csv(output_dir / "screen_summary.csv", summary)
    payload = {
        "config": config,
        "diagnostic_gates": dict(DIAGNOSTIC_GATES),
        "recording_count": len({(row.get("case"), row.get("recording")) for row in rows}),
        "summary_cell_count": len(summary),
        "diagnostic_pass_count": sum(
            bool(row["passes_empirical_diagnostics"]) for row in summary
        ),
        "summary": list(summary),
    }
    payload["status"] = "complete"
    atomic_write_json(output_dir / "summary.json", payload)
    (output_dir / "report.md").write_text(
        "\n".join(
            [
                "# Motoneuron temporal screen",
                "",
                "This truth-free empirical screen reports candidate density, split-episode direction replication, and episode-wise timing-shift null diagnostics. It does not estimate true-edge coverage or direction accuracy because the recordings have no directed ground truth.",
                "",
                f"Diagnostic cells passing all empirical checks: {payload['diagnostic_pass_count']} of {payload['summary_cell_count']}.",
                "",
                "A diagnostic pass is permission to run a separately controlled unrestricted-versus-prior c-GC comparison, not evidence of causal connectivity. Inferred population activity bouts are analysis strata rather than experimentally annotated trials, so segmentation sensitivity must be reported.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _parse_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("provide at least one integer")
    return parsed


def _parse_cases(value: str) -> tuple[str, ...]:
    cases = tuple(item.strip().upper() for item in value.split(",") if item.strip())
    invalid = sorted(set(cases) - set(CASE_FILES))
    if not cases or invalid:
        raise argparse.ArgumentTypeError(
            "cases must be selected from A,B,C,D"
        )
    return cases


def _parse_recordings(value: str) -> set[str] | None:
    if value.strip().lower() == "all":
        return None
    parsed = {item.strip().upper() for item in value.split(",") if item.strip()}
    if not parsed:
        raise argparse.ArgumentTypeError("provide recording labels or 'all'")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cases", type=_parse_cases, default=("D",))
    parser.add_argument("--recordings", type=_parse_recordings, default=None)
    parser.add_argument("--max-onset-lags", type=_parse_ints, default=(1, 2, 3))
    parser.add_argument("--deadbands", type=_parse_ints, default=(0, 1))
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--min-run-samples", type=int, default=2)
    parser.add_argument(
        "--segment-mode",
        choices=("fixed_windows", "population_bouts"),
        default="fixed_windows",
    )
    parser.add_argument("--window-frames", type=int, default=240)
    parser.add_argument("--merge-gap-frames", type=int, default=3)
    parser.add_argument("--minimum-participating-rois", type=int, default=2)
    parser.add_argument("--n-nulls", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=20260821)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse complete case/recording checkpoint units",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_case_traces(
        args.data_dir,
        cases=args.cases,
        recordings=args.recordings,
    )
    if not records:
        raise ValueError("no recordings matched the declared selection")
    config = {
        "data_dir": str(args.data_dir),
        "cases": list(args.cases),
        "recordings": "all" if args.recordings is None else sorted(args.recordings),
        "max_onset_lags": list(args.max_onset_lags),
        "deadbands": list(args.deadbands),
        "tolerance": args.tolerance,
        "min_run_samples": args.min_run_samples,
        "segment_mode": args.segment_mode,
        "window_frames": args.window_frames,
        "merge_gap_frames": args.merge_gap_frames,
        "minimum_participating_rois": args.minimum_participating_rois,
        "n_nulls": args.n_nulls,
        "random_state": args.random_state,
        "causal_identification": False,
    }
    checkpoint_store = JsonUnitCheckpointStore(
        args.output_dir,
        "motorneuron_temporal_screen",
        config,
    )
    checkpoint_store.initialize(resume=args.resume)
    rows = run_screen(
        records,
        max_onset_lags=args.max_onset_lags,
        deadbands=args.deadbands,
        tolerance=args.tolerance,
        min_run_samples=args.min_run_samples,
        segment_mode=args.segment_mode,
        window_frames=args.window_frames,
        merge_gap_frames=args.merge_gap_frames,
        minimum_participating_rois=args.minimum_participating_rois,
        n_nulls=args.n_nulls,
        random_state=args.random_state,
        progress=True,
        checkpoint_store=checkpoint_store,
        resume=args.resume,
    )
    summary = summarize_rows(rows)
    write_outputs(args.output_dir, rows, summary, config)
    checkpoint_store.finish(
        completed_units=len(records),
        total_units=len(records),
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
