"""Map when rising-flank timing can resolve directed propagation order.

This screening-only experiment isolates measurement resolution from c-GC
power. A stable source-triggered graph is observed across a locked grid of
native delays, downsampling factors, noise, and calcium-kinetic heterogeneity.
Temporal states are learned on four episodes and checked on four disjoint
episodes. No causal learner is run in this experiment.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from math import ceil
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from calcium_transient_rising_flank import (
    DynamicSimulationConfig,
    SyntheticDataset,
    SyntheticEpisode,
    TemporalPriorResult,
    build_representations,
    build_temporal_prior,
    rising_flank,
    simulate_calcium_dataset,
    temporal_prior_state_masks,
)
from calcium_transient_rising_flank.estimators import extract_rise_flank_runs
from calcium_transient_rising_flank.representations import estimate_decay


DEFAULT_OUTPUT = Path("outputs/validation_results/temporal_resolvability_map")
NATIVE_DELAYS = (1, 2, 4, 8)
DOWNSAMPLE_FACTORS = (1, 2, 4)
DEADBANDS = (0, 1)
HETEROGENEOUS_GAMMA = (0.84, 0.87, 0.90, 0.93, 0.96)
HOMOGENEOUS_GAMMA = (0.90,) * 5
PRIMARY_DEADBAND = 0
PRIMARY_SIGNAL_LAYER = "sampled_noisy_fluorescence"
SIGNAL_LAYERS = (
    "native_latent_events",
    "sampled_latent_events",
    "sampled_noiseless_calcium",
    PRIMARY_SIGNAL_LAYER,
)
GATES = {
    "minimum_true_edge_coverage": 0.90,
    "minimum_unconditional_direction_accuracy": 0.90,
    "maximum_candidate_density": 0.60,
    "minimum_seed_pass_fraction": 0.80,
}


@dataclass(frozen=True)
class ObservationRegime:
    name: str
    noise_std: float
    shared_noise_std: float
    heterogeneous_gamma: bool


REGIMES = (
    ObservationRegime("clean_homogeneous", 0.01, 0.0, False),
    ObservationRegime("independent_stress", 0.04, 0.0, False),
    ObservationRegime("shared_noise_stress", 0.04, 0.03, False),
    ObservationRegime("shared_heterogeneous", 0.04, 0.03, True),
)


def stable_truth_graph() -> np.ndarray:
    """Return a stable acyclic graph with one source and four true edges."""

    return np.array(
        [
            [False, True, False, False, False],
            [False, False, True, False, False],
            [False, False, False, True, False],
            [False, False, False, False, True],
            [False, False, False, False, False],
        ]
    )


def crossfit_episode_folds() -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    first = (0, 1, 4, 5)
    second = (2, 3, 6, 7)
    return ((first, second), (second, first))


def episode_segments(dataset: Any, episode_ids: Iterable[int]) -> np.ndarray:
    segments = np.full(dataset.fluorescence.shape[1], -1, dtype=int)
    for episode_id in episode_ids:
        episode = dataset.episodes[int(episode_id)]
        segments[episode.rise_start : episode.rise_stop] = int(episode_id)
    return segments


def onset_lag_limit(maximum_native_delay: int, downsample: int) -> int:
    """Convert a physical matching window to sampled-frame coordinates."""

    if maximum_native_delay < 1 or downsample < 1:
        raise ValueError("maximum_native_delay and downsample must be positive")
    return max(1, int(ceil(maximum_native_delay / downsample)))


def gamma_values(regime: ObservationRegime, seed: int) -> np.ndarray:
    """Return deterministic per-ROI kinetics without confounding node identity."""

    base = (
        np.asarray(HETEROGENEOUS_GAMMA, dtype=float)
        if regime.heterogeneous_gamma
        else np.asarray(HOMOGENEOUS_GAMMA, dtype=float)
    )
    return np.roll(base, seed % base.size)


def dynamic_config(truth: np.ndarray, native_delay: int) -> DynamicSimulationConfig:
    return DynamicSimulationConfig(
        n_episodes=8,
        min_rise_length=48,
        max_rise_length=48,
        rise_waveform_length=12,
        fall_to_rise_ratio_min=2.1,
        fall_to_rise_ratio_max=2.2,
        propagation_delay=native_delay,
        initial_activation_probability=1.0,
        initial_activation_mode="source_nodes",
        edge_dropout_probability=0.0,
        edge_addition_probability=0.0,
        source_dropout_probability=0.0,
        source_recruitment_probability=0.0,
        fall_noise_rate=0.0,
        fall_noise_scale=0.0,
        adjacency_sequence=(truth,),
    )


def _sampled_boundary(frame: int, factor: int, offset: int) -> int:
    """Map an exclusive native-frame boundary onto a phase-shifted sample axis."""

    return max(0, int(ceil((frame - offset) / factor)))


def downsample_with_phase(
    dataset: SyntheticDataset,
    factor: int,
    offset: int,
) -> SyntheticDataset:
    """Downsample at one acquisition phase while preserving episode intervals."""

    if not isinstance(factor, int) or factor < 1:
        raise ValueError("factor must be a positive integer")
    if not isinstance(offset, int) or not 0 <= offset < factor:
        raise ValueError("offset must lie in [0, factor)")
    sample = slice(offset, None, factor)
    episodes = tuple(
        SyntheticEpisode(
            rise_start=_sampled_boundary(episode.rise_start, factor, offset),
            rise_stop=_sampled_boundary(episode.rise_stop, factor, offset),
            fall_start=_sampled_boundary(episode.fall_start, factor, offset),
            fall_stop=_sampled_boundary(episode.fall_stop, factor, offset),
            adjacency=episode.adjacency.copy(),
            active_nodes=(
                None
                if episode.active_nodes is None
                else episode.active_nodes.copy()
            ),
        )
        for episode in dataset.episodes
    )
    return SyntheticDataset(
        adjacency=dataset.adjacency.copy(),
        events=dataset.events[:, sample],
        calcium=dataset.calcium[:, sample],
        fluorescence=dataset.fluorescence[:, sample],
        phase_labels=(
            None if dataset.phase_labels is None else dataset.phase_labels[sample]
        ),
        episodes=episodes,
        propagated_events=(
            None
            if dataset.propagated_events is None
            else dataset.propagated_events[:, sample]
        ),
        rise_adjacencies=tuple(graph.copy() for graph in dataset.rise_adjacencies),
        edge_presence_counts=(
            None
            if dataset.edge_presence_counts is None
            else dataset.edge_presence_counts.copy()
        ),
        edge_prevalence=(
            None
            if dataset.edge_prevalence is None
            else dataset.edge_prevalence.copy()
        ),
        node_presence_counts=(
            None
            if dataset.node_presence_counts is None
            else dataset.node_presence_counts.copy()
        ),
        node_prevalence=(
            None
            if dataset.node_prevalence is None
            else dataset.node_prevalence.copy()
        ),
    )


def expected_edge_lag_matrix(
    dataset: SyntheticDataset,
    truth: np.ndarray,
    episode_ids: Iterable[int],
    *,
    factor: int,
    offset: int,
) -> np.ndarray:
    """Return phase-realized sampled lags for each true direct edge."""

    directed_truth = np.asarray(truth, dtype=bool)
    if directed_truth.shape != dataset.adjacency.shape:
        raise ValueError("truth and dataset adjacency must have matching shapes")
    if factor < 1 or not 0 <= offset < factor:
        raise ValueError("factor and offset must define a valid sampling phase")
    expected = np.full(directed_truth.shape, np.nan, dtype=float)
    for source, target in np.argwhere(directed_truth):
        lags: list[int] = []
        for episode_id in episode_ids:
            episode = dataset.episodes[int(episode_id)]
            source_events = np.flatnonzero(
                dataset.events[source, episode.rise_start : episode.rise_stop] > 0
            )
            target_events = np.flatnonzero(
                dataset.events[target, episode.rise_start : episode.rise_stop] > 0
            )
            if source_events.size == 0 or target_events.size == 0:
                continue
            source_frame = episode.rise_start + int(source_events[0])
            target_frame = episode.rise_start + int(target_events[0])
            sampled_source = _sampled_boundary(source_frame, factor, offset)
            sampled_target = _sampled_boundary(target_frame, factor, offset)
            lags.append(sampled_target - sampled_source)
        if lags:
            expected[source, target] = float(np.median(lags))
    return expected


def build_prior(
    rise: np.ndarray,
    segments: np.ndarray,
    *,
    deadband: int,
    max_onset_lag: int,
    min_run_samples: int,
) -> TemporalPriorResult:
    return build_temporal_prior(
        rise,
        segments,
        min_run_samples=min_run_samples,
        max_onset_lag=max_onset_lag,
        timing_deadband=deadband,
        minimum_decisive_support=2,
        consistency_threshold=0.75,
        beta_prior_concentration=1.0,
    )


def temporal_state_metrics(
    prior: TemporalPriorResult,
    truth: np.ndarray,
    *,
    heldout_prior: TemporalPriorResult | None = None,
    expected_edge_lag: float | np.ndarray | None = None,
) -> dict[str, float | int | None]:
    """Summarize pair states without hiding abstention inside accuracy."""

    directed_truth = np.asarray(truth, dtype=bool).copy()
    if directed_truth.ndim != 2 or directed_truth.shape[0] != directed_truth.shape[1]:
        raise ValueError("truth must be square")
    np.fill_diagonal(directed_truth, False)
    if np.any(directed_truth & directed_truth.T):
        raise ValueError("resolvability truth cannot contain reciprocal pairs")

    states = temporal_prior_state_masks(prior)
    n_nodes = directed_truth.shape[0]
    if states.directional.shape != directed_truth.shape:
        raise ValueError("prior and truth must have matching shapes")
    upper = np.triu(np.ones_like(directed_truth, dtype=bool), 1)
    off_diagonal = ~np.eye(n_nodes, dtype=bool)
    truth_skeleton = directed_truth | directed_truth.T
    directional_pairs = (states.directional | states.directional.T) & upper
    ambiguous_pairs = states.ambiguous & upper
    unmatched_pairs = states.unmatched & upper
    pair_count = int(np.count_nonzero(upper))
    direction_count = int(np.count_nonzero(off_diagonal))

    truth_edges = int(np.count_nonzero(directed_truth))
    correct = int(np.count_nonzero(states.directional & directed_truth))
    wrong = int(np.count_nonzero(states.directional & directed_truth.T))
    decisive_true = correct + wrong
    conditional_accuracy = None if decisive_true == 0 else correct / decisive_true
    true_admitted = int(
        np.count_nonzero(prior.robust_hard_mask & directed_truth)
    )

    nonedge_pairs = upper & ~truth_skeleton
    nonedge_count = int(np.count_nonzero(nonedge_pairs))
    directional_nonedge = int(np.count_nonzero(directional_pairs & nonedge_pairs))
    nonedge_directions = np.broadcast_to(
        nonedge_pairs | nonedge_pairs.T,
        directed_truth.shape,
    )
    admitted_nonedge_directions = int(
        np.count_nonzero(prior.robust_hard_mask & nonedge_directions)
    )

    screen_directional = int(np.count_nonzero(states.directional))
    replicated = 0
    true_replicated = 0
    correct_screen = int(np.count_nonzero(states.directional & directed_truth))
    if heldout_prior is not None:
        heldout = temporal_prior_state_masks(heldout_prior)
        if heldout.directional.shape != states.directional.shape:
            raise ValueError("heldout prior must match the screen prior")
        replicated = int(
            np.count_nonzero(states.directional & heldout.directional)
        )
        true_replicated = int(
            np.count_nonzero(
                states.directional & heldout.directional & directed_truth
            )
        )

    true_weight = prior.hypothesis_weights[directed_truth]
    reverse_weight = prior.hypothesis_weights.T[directed_truth]
    soft_advantage = (
        None
        if truth_edges == 0
        else float(np.mean(true_weight - reverse_weight))
    )
    lag_errors: list[float] = []
    if expected_edge_lag is not None:
        expected = np.asarray(expected_edge_lag, dtype=float)
        if expected.ndim == 0:
            if float(expected) < 0:
                raise ValueError("expected_edge_lag cannot be negative")
            expected = np.full(directed_truth.shape, float(expected))
        if expected.shape != directed_truth.shape:
            raise ValueError("expected_edge_lag must be scalar or match truth")
        if np.any(expected[np.isfinite(expected)] < 0):
            raise ValueError("expected_edge_lag cannot be negative")
        correct_edges = states.directional & directed_truth
        observed = prior.median_decisive_lag[correct_edges]
        expected_values = expected[correct_edges]
        lag_errors = [
            abs(float(value) - float(target))
            for value, target in zip(observed, expected_values)
            if np.isfinite(target)
        ]
    return {
        "pair_count": pair_count,
        "direction_count": direction_count,
        "directional_pair_count": int(np.count_nonzero(directional_pairs)),
        "ambiguous_pair_count": int(np.count_nonzero(ambiguous_pairs)),
        "unmatched_pair_count": int(np.count_nonzero(unmatched_pairs)),
        "directional_pair_fraction": float(
            np.count_nonzero(directional_pairs) / pair_count
        ),
        "ambiguous_pair_fraction": float(
            np.count_nonzero(ambiguous_pairs) / pair_count
        ),
        "unmatched_pair_fraction": float(
            np.count_nonzero(unmatched_pairs) / pair_count
        ),
        "directional_candidate_density": float(
            np.count_nonzero(states.directional) / direction_count
        ),
        "robust_candidate_density": float(
            np.count_nonzero(prior.robust_hard_mask & off_diagonal) / direction_count
        ),
        "truth_edge_count": truth_edges,
        "true_edge_admitted_count": true_admitted,
        "correct_direction_count": correct,
        "wrong_direction_count": wrong,
        "decisive_true_edge_count": decisive_true,
        "true_edge_coverage": (
            0.0 if truth_edges == 0 else true_admitted / truth_edges
        ),
        "unconditional_direction_accuracy": (
            0.0 if truth_edges == 0 else correct / truth_edges
        ),
        "true_edge_decisive_fraction": (
            0.0 if truth_edges == 0 else decisive_true / truth_edges
        ),
        "conditional_direction_accuracy": conditional_accuracy,
        "candidate_density": float(
            np.count_nonzero(prior.robust_hard_mask & off_diagonal)
            / direction_count
        ),
        "nonedge_directional_fraction": (
            None if nonedge_count == 0 else directional_nonedge / nonedge_count
        ),
        "nonedge_exclusion": (
            None
            if nonedge_count == 0
            else 1.0
            - admitted_nonedge_directions / (2.0 * nonedge_count)
        ),
        "screen_directional_count": screen_directional,
        "replicated_directional_count": replicated,
        "heldout_directional_replication": (
            None if screen_directional == 0 else replicated / screen_directional
        ),
        "correct_screen_direction_count": correct_screen,
        "true_replicated_direction_count": true_replicated,
        "heldout_true_direction_replication": (
            None if correct_screen == 0 else true_replicated / correct_screen
        ),
        "soft_true_direction_advantage": soft_advantage,
        "median_observed_lag_absolute_error": (
            None if not lag_errors else float(np.median(lag_errors))
        ),
    }


def _median_screen_run_length(
    rise: np.ndarray,
    segments: np.ndarray,
    min_run_samples: int,
) -> float | None:
    masked = np.where(segments[np.newaxis, :] >= 0, rise, 0.0)
    runs = extract_rise_flank_runs(masked, min_run_samples).kept_runs
    lengths = [run.size for roi_runs in runs for run in roi_runs]
    return None if not lengths else float(np.median(lengths))


def run_resolvability_grid(
    *,
    regimes: Sequence[ObservationRegime],
    native_delays: Sequence[int],
    downsample_factors: Sequence[int],
    deadbands: Sequence[int],
    seeds: Sequence[int],
    n_steps: int = 1280,
    tolerance: float = 0.02,
    min_run_samples: int = 2,
    matching_window_native_frames: int = max(NATIVE_DELAYS),
    progress: bool = False,
) -> list[dict[str, Any]]:
    """Run the locked screening grid and return fold-level rows."""

    if not regimes or not native_delays or not downsample_factors or not deadbands or not seeds:
        raise ValueError("all grid dimensions must be non-empty")
    truth = stable_truth_graph()
    rows: list[dict[str, Any]] = []
    for regime in regimes:
        for native_delay in native_delays:
            if native_delay >= 48:
                raise ValueError("native delays must be shorter than the rise phase")
            for seed in seeds:
                gamma = gamma_values(regime, int(seed))
                dataset = simulate_calcium_dataset(
                    truth,
                    n_steps=n_steps,
                    gamma=gamma,
                    noise_std=regime.noise_std,
                    shared_noise_std=regime.shared_noise_std,
                    spontaneous_rate=0.0,
                    transmission_probability=1.0,
                    random_state=int(seed),
                    simulator_mode="episodic_dynamic",
                    dynamic_config=dynamic_config(truth, int(native_delay)),
                )
                for downsample in downsample_factors:
                    sampled_gamma = gamma ** int(downsample)
                    sampled_lag_limit = onset_lag_limit(
                        matching_window_native_frames,
                        int(downsample),
                    )
                    for acquisition_phase in range(int(downsample)):
                        sampled = downsample_with_phase(
                            dataset,
                            int(downsample),
                            acquisition_phase,
                        )
                        bundle = build_representations(
                            sampled.fluorescence,
                            tolerance=tolerance,
                            gamma=sampled_gamma,
                        )
                        estimated_gamma = estimate_decay(sampled.fluorescence)
                        layers = {
                            "native_latent_events": (
                                dataset.events,
                                dataset,
                                1,
                                matching_window_native_frames,
                                1,
                                0,
                            ),
                            "sampled_latent_events": (
                                sampled.events,
                                sampled,
                                1,
                                sampled_lag_limit,
                                int(downsample),
                                acquisition_phase,
                            ),
                            "sampled_noiseless_calcium": (
                                rising_flank(sampled.calcium, tolerance=tolerance),
                                sampled,
                                min_run_samples,
                                sampled_lag_limit,
                                int(downsample),
                                acquisition_phase,
                            ),
                            PRIMARY_SIGNAL_LAYER: (
                                bundle.rise,
                                sampled,
                                min_run_samples,
                                sampled_lag_limit,
                                int(downsample),
                                acquisition_phase,
                            ),
                        }
                        for fold, (screen_ids, test_ids) in enumerate(
                            crossfit_episode_folds()
                        ):
                            for signal_layer, layer_values in layers.items():
                                (
                                    rise,
                                    layer_dataset,
                                    layer_min_run,
                                    max_onset_lag,
                                    expected_lag_factor,
                                    expected_lag_offset,
                                ) = layer_values
                                screen_segments = episode_segments(
                                    layer_dataset, screen_ids
                                )
                                test_segments = episode_segments(
                                    layer_dataset, test_ids
                                )
                                expected_edge_lag = expected_edge_lag_matrix(
                                    dataset,
                                    truth,
                                    screen_ids,
                                    factor=expected_lag_factor,
                                    offset=expected_lag_offset,
                                )
                                layer_deadbands = (
                                    deadbands
                                    if signal_layer == PRIMARY_SIGNAL_LAYER
                                    else (PRIMARY_DEADBAND,)
                                )
                                for deadband in layer_deadbands:
                                    screen_prior = build_prior(
                                        rise,
                                        screen_segments,
                                        deadband=int(deadband),
                                        max_onset_lag=max_onset_lag,
                                        min_run_samples=layer_min_run,
                                    )
                                    test_prior = build_prior(
                                        rise,
                                        test_segments,
                                        deadband=int(deadband),
                                        max_onset_lag=max_onset_lag,
                                        min_run_samples=layer_min_run,
                                    )
                                    metrics = temporal_state_metrics(
                                        screen_prior,
                                        truth,
                                        heldout_prior=test_prior,
                                        expected_edge_lag=expected_edge_lag,
                                    )
                                    rows.append(
                                        {
                                            "regime": regime.name,
                                            "signal_layer": signal_layer,
                                            "seed": int(seed),
                                            "fold": fold,
                                            "acquisition_phase": acquisition_phase,
                                            "acquisition_phase_count": int(downsample),
                                            "screen_episode_ids": ";".join(
                                                map(str, screen_ids)
                                            ),
                                            "test_episode_ids": ";".join(
                                                map(str, test_ids)
                                            ),
                                            "native_delay_frames": int(native_delay),
                                            "downsample": int(downsample),
                                            "observed_delay_frames": float(
                                                native_delay / downsample
                                            ),
                                            "deadband_frames": int(deadband),
                                            "max_onset_lag_frames": max_onset_lag,
                                            "matching_window_native_frames": (
                                                matching_window_native_frames
                                            ),
                                            "noise_std": regime.noise_std,
                                            "shared_noise_std": (
                                                regime.shared_noise_std
                                            ),
                                            "heterogeneous_gamma": (
                                                regime.heterogeneous_gamma
                                            ),
                                            "native_gamma_min": float(np.min(gamma)),
                                            "native_gamma_max": float(np.max(gamma)),
                                            "native_gamma_mean": float(np.mean(gamma)),
                                            "native_gamma_sd": float(np.std(gamma)),
                                            "sampled_gamma_min": float(
                                                np.min(sampled_gamma)
                                            ),
                                            "sampled_gamma_max": float(
                                                np.max(sampled_gamma)
                                            ),
                                            "estimated_gamma_mean": float(
                                                np.mean(estimated_gamma)
                                            ),
                                            "estimated_gamma_sd": float(
                                                np.std(estimated_gamma)
                                            ),
                                            "median_screen_rise_run_samples": (
                                                _median_screen_run_length(
                                                    rise,
                                                    screen_segments,
                                                    layer_min_run,
                                                )
                                            ),
                                            **metrics,
                                        }
                                    )
            if progress:
                print(
                    f"completed regime={regime.name} delay={native_delay}",
                    flush=True,
                )
    return rows


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def _sd(values: Sequence[float]) -> float:
    data = np.asarray(values, dtype=float)
    return float(np.std(data, ddof=1)) if data.size > 1 else 0.0


def _seed_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "regime",
        "signal_layer",
        "native_delay_frames",
        "downsample",
        "observed_delay_frames",
        "deadband_frames",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[name] for name in keys) + (row["seed"],)
        groups.setdefault(key, []).append(row)
    metrics = (
        "directional_pair_fraction",
        "ambiguous_pair_fraction",
        "unmatched_pair_fraction",
        "directional_candidate_density",
        "robust_candidate_density",
        "candidate_density",
        "true_edge_coverage",
        "unconditional_direction_accuracy",
        "true_edge_decisive_fraction",
        "nonedge_directional_fraction",
        "nonedge_exclusion",
        "heldout_directional_replication",
        "heldout_true_direction_replication",
        "soft_true_direction_advantage",
        "median_observed_lag_absolute_error",
        "estimated_gamma_mean",
        "estimated_gamma_sd",
        "median_screen_rise_run_samples",
    )
    output = []
    for key, members in sorted(groups.items()):
        item = dict(zip(keys + ("seed",), key))
        for metric in metrics:
            values = [
                float(row[metric])
                for row in members
                if row[metric] is not None
            ]
            item[metric] = None if not values else _mean(values)
        item["correct_direction_count"] = int(
            sum(int(row["correct_direction_count"]) for row in members)
        )
        item["wrong_direction_count"] = int(
            sum(int(row["wrong_direction_count"]) for row in members)
        )
        item["screen_directional_count"] = int(
            sum(int(row["screen_directional_count"]) for row in members)
        )
        item["replicated_directional_count"] = int(
            sum(int(row["replicated_directional_count"]) for row in members)
        )
        item["passes_seed_gates"] = bool(
            item["true_edge_coverage"] >= GATES["minimum_true_edge_coverage"]
            and item["unconditional_direction_accuracy"]
            >= GATES["minimum_unconditional_direction_accuracy"]
            and item["candidate_density"] <= GATES["maximum_candidate_density"]
        )
        output.append(item)
    return output


def summarize_cells(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate folds within seed, then seeds within each locked grid cell."""

    seed_rows = _seed_rows(rows)
    keys = (
        "regime",
        "signal_layer",
        "native_delay_frames",
        "downsample",
        "observed_delay_frames",
        "deadband_frames",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in seed_rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    metrics = (
        "directional_pair_fraction",
        "ambiguous_pair_fraction",
        "unmatched_pair_fraction",
        "directional_candidate_density",
        "robust_candidate_density",
        "candidate_density",
        "true_edge_coverage",
        "unconditional_direction_accuracy",
        "true_edge_decisive_fraction",
        "nonedge_directional_fraction",
        "nonedge_exclusion",
        "heldout_directional_replication",
        "heldout_true_direction_replication",
        "soft_true_direction_advantage",
        "median_observed_lag_absolute_error",
        "estimated_gamma_mean",
        "estimated_gamma_sd",
        "median_screen_rise_run_samples",
    )
    output = []
    for key, members in sorted(groups.items()):
        item: dict[str, Any] = dict(zip(keys, key))
        item["n_seeds"] = len(members)
        for metric in metrics:
            values = [
                float(member[metric])
                for member in members
                if member[metric] is not None
            ]
            item[f"{metric}_mean"] = None if not values else _mean(values)
            item[f"{metric}_sd"] = None if not values else _sd(values)
        correct = sum(int(member["correct_direction_count"]) for member in members)
        wrong = sum(int(member["wrong_direction_count"]) for member in members)
        screen = sum(int(member["screen_directional_count"]) for member in members)
        replicated = sum(
            int(member["replicated_directional_count"]) for member in members
        )
        item["conditional_direction_accuracy_pooled"] = (
            None if correct + wrong == 0 else correct / (correct + wrong)
        )
        item["heldout_directional_replication_pooled"] = (
            None if screen == 0 else replicated / screen
        )
        seed_passes = sum(bool(member["passes_seed_gates"]) for member in members)
        item["seed_pass_count"] = seed_passes
        item["seed_pass_fraction"] = seed_passes / len(members)
        item["passes_coverage"] = (
            item["true_edge_coverage_mean"]
            >= GATES["minimum_true_edge_coverage"]
        )
        item["passes_accuracy"] = (
            item["unconditional_direction_accuracy_mean"]
            >= GATES["minimum_unconditional_direction_accuracy"]
        )
        item["passes_density"] = (
            item["candidate_density_mean"] <= GATES["maximum_candidate_density"]
        )
        item["passes_seed_replication"] = (
            item["seed_pass_fraction"] >= GATES["minimum_seed_pass_fraction"]
        )
        item["passes_mean_gates"] = bool(
            item["passes_coverage"]
            and item["passes_accuracy"]
            and item["passes_density"]
        )
        item["passes_adjacency"] = False
        item["viable"] = False
        output.append(item)
    return output


def apply_adjacency_gate(
    cells: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Require a passing cell to have a passing grid neighbor."""

    finalized = [dict(cell) for cell in cells]
    native_values = sorted({int(cell["native_delay_frames"]) for cell in cells})
    sample_values = sorted({int(cell["downsample"]) for cell in cells})
    native_index = {value: index for index, value in enumerate(native_values)}
    sample_index = {value: index for index, value in enumerate(sample_values)}
    eligible = {
        (
            str(cell["regime"]),
            int(cell["native_delay_frames"]),
            int(cell["downsample"]),
        )
        for cell in finalized
        if cell["signal_layer"] == PRIMARY_SIGNAL_LAYER
        and int(cell["deadband_frames"]) == PRIMARY_DEADBAND
        and cell["passes_mean_gates"]
        and cell["passes_seed_replication"]
    }
    for cell in finalized:
        if (
            cell["signal_layer"] != PRIMARY_SIGNAL_LAYER
            or int(cell["deadband_frames"]) != PRIMARY_DEADBAND
        ):
            continue
        key = (
            str(cell["regime"]),
            int(cell["native_delay_frames"]),
            int(cell["downsample"]),
        )
        if key not in eligible:
            continue
        _, native_delay, downsample = key
        has_neighbor = any(
            other_regime == key[0]
            and (
                (
                    abs(native_index[other_delay] - native_index[native_delay]) == 1
                    and other_sample == downsample
                )
                or (
                    other_delay == native_delay
                    and abs(sample_index[other_sample] - sample_index[downsample]) == 1
                )
            )
            for other_regime, other_delay, other_sample in eligible
            if (other_regime, other_delay, other_sample) != key
        )
        cell["passes_adjacency"] = has_neighbor
        cell["viable"] = has_neighbor
    return finalized


def evaluate_map(cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    viable = [cell for cell in cells if cell["viable"]]
    primary = [
        cell
        for cell in cells
        if cell["signal_layer"] == PRIMARY_SIGNAL_LAYER
        and int(cell["deadband_frames"]) == PRIMARY_DEADBAND
    ]
    viable_regimes = sorted({str(cell["regime"]) for cell in viable})
    return {
        "thresholds": dict(GATES),
        "cell_count": len(cells),
        "primary_cell_count": len(primary),
        "viable_cell_count": len(viable),
        "viable_regimes": viable_regimes,
        "recommendation": (
            "proceed_to_cgc_in_viable_regimes"
            if viable
            else "no_temporally_resolvable_regime_found"
        ),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _report(cells: Sequence[dict[str, Any]], evaluation: dict[str, Any]) -> str:
    viable = [cell for cell in cells if cell["viable"]]
    lines = [
        "# Temporal resolvability map",
        "",
        "This screening-only experiment asks when rising-flank onset order can resolve directed propagation. It does not run c-GC and does not establish causal identification.",
        "",
        f"Recommendation: `{evaluation['recommendation']}`.",
        "",
        "Viable primary fluorescence cells: "
        f"{evaluation['viable_cell_count']} of {evaluation['primary_cell_count']}.",
        "",
        "## Viable regimes",
        "",
        "| regime | native delay | downsample | observed delay | coverage | direction accuracy | candidate density | passing seeds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in viable:
        lines.append(
            "| {regime} | {native_delay_frames} | {downsample} | "
            "{observed_delay_frames:.2f} | "
            "{true_edge_coverage_mean:.3f} | "
            "{unconditional_direction_accuracy_mean:.3f} | "
            "{candidate_density_mean:.3f} | "
            "{seed_pass_count}/{n_seeds} |".format(**cell)
        )
    if not viable:
        lines.append("| _none_ | | | | | | | |")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The source-triggered stable graph is an upper-bound measurement test. Passing cells identify conditions in which temporal ordering is sufficiently resolved to justify a later graph-learning ablation. A cell passes only when mean coverage and unconditional direction accuracy are at least 0.90, candidate density is at most 0.60, at least 80% of seeds pass those three criteria, and a neighboring grid cell also passes. Failure here argues against direction pruning at the corresponding acquisition resolution.",
            "",
            "Deadband 0 is primary; deadband 1 is descriptive sensitivity. Native events, sampled events, and sampled noiseless calcium are diagnostic layers used to localize information loss, not additional decision opportunities.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    cells: Sequence[dict[str, Any]],
    evaluation: dict[str, Any],
    config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "resolvability_rows.csv", rows)
    _write_csv(output_dir / "resolvability_cells.csv", cells)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": config,
                "gates": evaluation,
                "summary": list(cells),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        _report(cells, evaluation), encoding="utf-8"
    )


def _parse_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("provide at least one integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=_parse_ints, default=tuple(range(1, 21)))
    parser.add_argument("--native-delays", type=_parse_ints, default=NATIVE_DELAYS)
    parser.add_argument(
        "--downsample-factors",
        type=_parse_ints,
        default=DOWNSAMPLE_FACTORS,
    )
    parser.add_argument("--deadbands", type=_parse_ints, default=DEADBANDS)
    parser.add_argument("--n-steps", type=int, default=1280)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--min-run-samples", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_resolvability_grid(
        regimes=REGIMES,
        native_delays=args.native_delays,
        downsample_factors=args.downsample_factors,
        deadbands=args.deadbands,
        seeds=args.seeds,
        n_steps=args.n_steps,
        tolerance=args.tolerance,
        min_run_samples=args.min_run_samples,
        progress=True,
    )
    cells = apply_adjacency_gate(summarize_cells(rows))
    evaluation = evaluate_map(cells)
    config = {
        "seeds": list(args.seeds),
        "native_delays": list(args.native_delays),
        "downsample_factors": list(args.downsample_factors),
        "deadbands": list(args.deadbands),
        "n_steps": args.n_steps,
        "tolerance": args.tolerance,
        "min_run_samples": args.min_run_samples,
        "n_episodes": 8,
        "screen_episodes_per_fold": 4,
        "heldout_episodes_per_fold": 4,
        "rise_length_native_frames": 48,
        "rise_waveform_length_native_frames": 12,
        "initial_activation_mode": "source_nodes",
        "transmission_probability": 1.0,
        "spontaneous_rate": 0.0,
        "regimes": [asdict(regime) for regime in REGIMES],
        "gamma_homogeneous": list(HOMOGENEOUS_GAMMA),
        "gamma_heterogeneous": list(HETEROGENEOUS_GAMMA),
        "matching_window_native_frames": max(NATIVE_DELAYS),
        "onset_lag_rule": "ceil(maximum_native_matching_window/downsample)",
        "primary_signal_layer": PRIMARY_SIGNAL_LAYER,
        "diagnostic_signal_layers": list(SIGNAL_LAYERS[:-1]),
        "primary_deadband_frames": PRIMARY_DEADBAND,
        "acquisition_phase_rule": "average_all_offsets_0_to_downsample_minus_1",
        "decision_rule": "mean gates + >=80% seed passes + adjacent passing cell",
    }
    write_outputs(args.output_dir, rows, cells, evaluation, config)
    print(json.dumps(evaluation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
