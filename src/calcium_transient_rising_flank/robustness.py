"""Prespecified sensitivity and locked synthetic-evaluation utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .estimators import CausalisedGC
from .pipeline import AnalysisConfig, PipelineResult, run_pipeline
from .validation import (
    DynamicSimulationConfig,
    RepresentationValidation,
    SyntheticDataset,
    SyntheticEpisode,
    simulate_calcium_dataset,
    validate_representations,
)


@dataclass(frozen=True)
class SyntheticCondition:
    """One declared observation regime in a synthetic validation grid."""

    name: str
    gamma: float | np.ndarray = 0.8
    noise_std: float = 0.05
    shared_noise_std: float = 0.0
    spontaneous_rate: float = 0.025
    transmission_probability: float = 0.8
    downsample: int = 1
    simulator_mode: str = "static"
    dynamic_config: DynamicSimulationConfig | None = None


@dataclass(frozen=True)
class SyntheticGridRun:
    """Output for one condition/seed pair before calibration or evaluation."""

    condition: SyntheticCondition
    seed: int
    dataset: SyntheticDataset
    validation: RepresentationValidation


@dataclass(frozen=True)
class SensitivityRun:
    """Pipeline output associated with one declared empirical setting."""

    config: AnalysisConfig
    result: PipelineResult


def downsample_dataset(dataset: SyntheticDataset, factor: int) -> SyntheticDataset:
    """Reduce the observation rate while retaining directed-event truth."""

    if not isinstance(factor, int) or factor < 1:
        raise ValueError("factor must be a positive integer")
    if factor == 1:
        return dataset
    episodes = tuple(
        SyntheticEpisode(
            rise_start=episode.rise_start // factor,
            rise_stop=(episode.rise_stop + factor - 1) // factor,
            fall_start=episode.fall_start // factor,
            fall_stop=(episode.fall_stop + factor - 1) // factor,
            adjacency=episode.adjacency.copy(),
            active_nodes=None
            if episode.active_nodes is None
            else episode.active_nodes.copy(),
            fall_adjacency=None
            if episode.fall_adjacency is None
            else episode.fall_adjacency.copy(),
        )
        for episode in dataset.episodes
    )
    return SyntheticDataset(
        adjacency=dataset.adjacency.copy(),
        events=dataset.events[:, ::factor],
        calcium=dataset.calcium[:, ::factor],
        fluorescence=dataset.fluorescence[:, ::factor],
        phase_labels=None
        if dataset.phase_labels is None
        else dataset.phase_labels[::factor],
        episodes=episodes,
        propagated_events=None
        if dataset.propagated_events is None
        else dataset.propagated_events[:, ::factor],
        rise_adjacencies=tuple(graph.copy() for graph in dataset.rise_adjacencies),
        edge_presence_counts=None
        if dataset.edge_presence_counts is None
        else dataset.edge_presence_counts.copy(),
        edge_prevalence=None
        if dataset.edge_prevalence is None
        else dataset.edge_prevalence.copy(),
        node_presence_counts=None
        if dataset.node_presence_counts is None
        else dataset.node_presence_counts.copy(),
        node_prevalence=None
        if dataset.node_prevalence is None
        else dataset.node_prevalence.copy(),
        fall_truth_adjacency=None
        if dataset.fall_truth_adjacency is None
        else dataset.fall_truth_adjacency.copy(),
        fall_truth_adjacencies=tuple(
            graph.copy() for graph in dataset.fall_truth_adjacencies
        ),
    )


def _sampled_gamma(gamma: float | np.ndarray, factor: int) -> float | np.ndarray:
    sampled = np.asarray(gamma, dtype=float) ** factor
    return float(sampled) if sampled.ndim == 0 else sampled


def run_synthetic_grid(
    adjacency: np.ndarray,
    conditions: Iterable[SyntheticCondition],
    seeds: Iterable[int],
    estimator_factory: Callable[[], CausalisedGC],
    n_steps: int = 1000,
    tolerance: float | np.ndarray = 0.0,
) -> tuple[SyntheticGridRun, ...]:
    """Evaluate representations across declared observation regimes and seeds."""

    declared_conditions = tuple(conditions)
    declared_seeds = tuple(seeds)
    if not declared_conditions or not declared_seeds:
        raise ValueError("conditions and seeds must be non-empty")
    runs: list[SyntheticGridRun] = []
    for condition in declared_conditions:
        if condition.downsample < 1:
            raise ValueError("downsample must be a positive integer")
        for seed in declared_seeds:
            dataset = simulate_calcium_dataset(
                adjacency,
                n_steps=n_steps,
                gamma=condition.gamma,
                noise_std=condition.noise_std,
                shared_noise_std=condition.shared_noise_std,
                spontaneous_rate=condition.spontaneous_rate,
                transmission_probability=condition.transmission_probability,
                random_state=seed,
                simulator_mode=condition.simulator_mode,
                dynamic_config=condition.dynamic_config,
            )
            sampled = downsample_dataset(dataset, condition.downsample)
            validation = validate_representations(
                sampled,
                estimator=estimator_factory(),
                tolerance=tolerance,
                gamma=_sampled_gamma(condition.gamma, condition.downsample),
            )
            runs.append(SyntheticGridRun(condition, seed, sampled, validation))
    return tuple(runs)


def split_calibration_evaluation(
    runs: Sequence[SyntheticGridRun],
    calibration_fraction: float = 0.5,
    random_state: int | None = None,
) -> tuple[tuple[SyntheticGridRun, ...], tuple[SyntheticGridRun, ...]]:
    """Freeze disjoint calibration and evaluation collections grouped by seed."""

    seeds = tuple(dict.fromkeys(run.seed for run in runs))
    if len(seeds) < 2:
        raise ValueError("at least two distinct seeds are required for a held-out split")
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must lie strictly between zero and one")
    order = np.random.default_rng(random_state).permutation(len(seeds))
    count = max(1, min(len(seeds) - 1, int(round(len(seeds) * calibration_fraction))))
    calibration_seeds = {seeds[index] for index in order[:count]}
    calibration = tuple(run for run in runs if run.seed in calibration_seeds)
    evaluation = tuple(run for run in runs if run.seed not in calibration_seeds)
    return calibration, evaluation


def run_analysis_sensitivity(
    traces: np.ndarray,
    sides: Iterable[str],
    configs: Iterable[AnalysisConfig],
    positions: Iterable[float] | None = None,
) -> tuple[SensitivityRun, ...]:
    """Run a prespecified empirical parameter grid without selecting results."""

    declared_configs = tuple(configs)
    if not declared_configs:
        raise ValueError("configs must be non-empty")
    side_labels = tuple(sides)
    node_positions = None if positions is None else tuple(positions)
    return tuple(
        SensitivityRun(
            config,
            run_pipeline(traces, side_labels, node_positions, config),
        )
        for config in declared_configs
    )
