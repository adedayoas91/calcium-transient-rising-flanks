"""Synthetic and null-control utilities for pipeline validation."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np

from .estimators import (
    CausalisedGC,
    GraphResult,
    RiseFlankCandidateResult,
    selected_frame_indices,
)
from .metrics import RecoverySummary, edge_recovery, graph_stability
from .preprocessing import validate_traces
from .representations import RepresentationBundle
from .representations import build_representations


@dataclass(frozen=True)
class SyntheticEpisode:
    """One rise/fall episode in an episodic dynamic simulation."""

    rise_start: int
    rise_stop: int
    fall_start: int
    fall_stop: int
    adjacency: np.ndarray
    active_nodes: np.ndarray | None = None
    fall_adjacency: np.ndarray | None = None

    @property
    def rise_length(self) -> int:
        return self.rise_stop - self.rise_start

    @property
    def fall_length(self) -> int:
        return self.fall_stop - self.fall_start

    @property
    def node_count(self) -> int:
        if self.active_nodes is None:
            return int(self.adjacency.shape[0])
        return int(np.count_nonzero(self.active_nodes))

    @property
    def edge_count(self) -> int:
        return int(np.count_nonzero(self.adjacency))


@dataclass(frozen=True)
class DynamicSimulationConfig:
    """Controls for episodic simulations with optional rise and fall propagation.

    ``rise_waveform_length`` spreads each rise-phase onset over a finite number
    of samples in the calcium drive. A value of 1 preserves impulse-like
    innovations; larger values create slower, inspectable rising flanks.

    ``fall_state_mode="stochastic_independent"`` resets each fall to an
    ROI-local stochastic state before passive decay, making the fall segment a
    stricter graph-independent negative control. The reset is capped by
    ``fall_initial_ceiling_fraction`` of the preceding rise endpoint to avoid
    non-calcium-like upward jumps at the rise/fall boundary.

    Dynamic rise graphs are event specific. When ``adjacency_sequence`` is
    supplied, episode ``k`` uses the corresponding user-declared matrix
    ``A_k``. Otherwise the base adjacency is treated as a stable backbone and
    each episode samples edge dropout/addition and optional source-neuron
    de-recruitment/recruitment around that backbone. The dataset-level
    adjacency is the logical OR over the episode matrices.
    """

    n_episodes: int = 5
    min_rise_length: int = 20
    max_rise_length: int = 35
    rise_waveform_length: int = 1
    fall_to_rise_ratio_min: float = 2.1
    fall_to_rise_ratio_max: float = 3.5
    propagation_delay: int = 1
    initial_activation_probability: float = 0.35
    initial_activation_mode: str = "all_active"
    edge_dropout_probability: float = 0.2
    edge_addition_probability: float = 0.0
    source_dropout_probability: float = 0.0
    source_recruitment_probability: float = 0.0
    source_recruitment_edge_probability: float = 0.25
    fall_noise_rate: float = 0.03
    fall_noise_scale: float = 0.02
    fall_state_mode: str = "stochastic_independent"
    fall_initial_scale: float = 1.0
    fall_initial_ceiling_fraction: float = 1.0
    rise_gain: float = 1.0
    gamma_fall: float | np.ndarray | None = None
    adjacency_sequence: tuple[np.ndarray, ...] | None = None
    active_node_sequence: tuple[np.ndarray, ...] | None = None
    fall_adjacency_sequence: tuple[np.ndarray, ...] | None = None
    fall_initial_activation_probability: float = 0.0
    fall_spontaneous_rate: float = 0.0
    fall_transmission_probability: float = 0.0
    fall_overlap_exclusion_samples: int = 0
    fall_propagation_drop_fraction: float = 0.25


@dataclass(frozen=True)
class SyntheticDataset:
    adjacency: np.ndarray
    events: np.ndarray
    calcium: np.ndarray
    fluorescence: np.ndarray
    phase_labels: np.ndarray | None = None
    episodes: tuple[SyntheticEpisode, ...] = ()
    propagated_events: np.ndarray | None = None
    rise_adjacencies: tuple[np.ndarray, ...] = ()
    edge_presence_counts: np.ndarray | None = None
    edge_prevalence: np.ndarray | None = None
    node_presence_counts: np.ndarray | None = None
    node_prevalence: np.ndarray | None = None
    fall_truth_adjacency: np.ndarray | None = None
    fall_truth_adjacencies: tuple[np.ndarray, ...] = ()

    @property
    def union_adjacency(self) -> np.ndarray:
        return np.asarray(self.adjacency, dtype=bool).copy()

    @property
    def episode_adjacencies(self) -> tuple[np.ndarray, ...]:
        """Return the event-specific adjacency matrix used for each rise."""

        if self.rise_adjacencies:
            return tuple(graph.copy() for graph in self.rise_adjacencies)
        return tuple(episode.adjacency.copy() for episode in self.episodes)

    @property
    def edge_counts(self) -> np.ndarray | None:
        """Alias for edge presence counts across dynamic rise graphs."""

        if self.edge_presence_counts is None:
            return None
        return self.edge_presence_counts.copy()

    @property
    def fall_union_adjacency(self) -> np.ndarray | None:
        """Return the explicit fall-phase truth when the simulator provides it."""

        if self.fall_truth_adjacency is None:
            return None
        return np.asarray(self.fall_truth_adjacency, dtype=bool).copy()

    @property
    def episode_fall_adjacencies(self) -> tuple[np.ndarray, ...]:
        """Return the explicit event-specific fall truth graphs."""

        if self.fall_truth_adjacencies:
            return tuple(graph.copy() for graph in self.fall_truth_adjacencies)
        return tuple(
            np.zeros_like(episode.adjacency, dtype=bool)
            if episode.fall_adjacency is None
            else episode.fall_adjacency.copy()
            for episode in self.episodes
        )


@dataclass(frozen=True)
class NullControlResult:
    """Observed, cyclic-shift, and reverse-time graph estimates."""

    observed: GraphResult
    shifted: tuple[GraphResult, ...]
    reverse_time: GraphResult


@dataclass(frozen=True)
class EventNullControlResult:
    """Event-aware nulls for rise/fall selected-frame analyses."""

    observed: GraphResult
    jittered: tuple[GraphResult, ...]
    reverse_time: GraphResult
    phase_permuted: tuple[GraphResult, ...] = ()
    cross_recording: GraphResult | None = None


@dataclass(frozen=True)
class StabilityResult:
    """Bootstrap graph estimates and their mean pairwise edge stability."""

    graphs: tuple[GraphResult, ...]
    stability: float


@dataclass(frozen=True)
class RepresentationValidation:
    """Graphs and directed-edge recovery for each signal representation."""

    graphs: dict[str, GraphResult]
    recovery: dict[str, RecoverySummary]
    truth: dict[str, np.ndarray]
    rise_flank_candidates: RiseFlankCandidateResult | None = None


def simulate_events(
    adjacency: np.ndarray,
    n_steps: int,
    spontaneous_rate: float = 0.025,
    transmission_probability: float = 0.8,
    random_state: int | None = None,
    propagation_delay: int = 1,
) -> np.ndarray:
    """Generate binary events on a directed network with known structure.

    ``propagation_delay`` is the number of frames between a source event and the
    event it may trigger downstream (the latent causal lag). The default of 1
    reproduces the original behaviour exactly, including the random-draw order.
    Exposing it lets validation sweep the rise-duration-to-delay ratio, which is
    what governs whether time-lagged Granger causality can resolve edge direction.
    """

    network = np.asarray(adjacency, dtype=bool).copy()
    if network.ndim != 2 or network.shape[0] != network.shape[1]:
        raise ValueError("adjacency must be square")
    if n_steps < 3:
        raise ValueError("n_steps must be at least three")
    if not 0 <= spontaneous_rate <= 1 or not 0 <= transmission_probability <= 1:
        raise ValueError("event probabilities must lie in [0, 1]")
    if not isinstance(propagation_delay, (int, np.integer)) or propagation_delay < 1:
        raise ValueError("propagation_delay must be a positive integer (frames)")
    if propagation_delay >= n_steps:
        raise ValueError("propagation_delay must be smaller than n_steps")
    np.fill_diagonal(network, False)
    rng = np.random.default_rng(random_state)
    events = np.zeros((network.shape[0], n_steps), dtype=float)
    events[:, 0] = rng.random(network.shape[0]) < spontaneous_rate
    for time in range(1, n_steps):
        spontaneous = rng.random(network.shape[0]) < spontaneous_rate
        if time >= propagation_delay:
            incoming = network.T @ events[:, time - propagation_delay] > 0
        else:
            incoming = np.zeros(network.shape[0], dtype=bool)
        propagated = incoming & (
            rng.random(network.shape[0]) < transmission_probability
        )
        events[:, time] = spontaneous | propagated
    return events


def calcium_observation_model(
    events: np.ndarray,
    gamma: float | np.ndarray = 0.8,
    noise_std: float = 0.05,
    shared_noise_std: float = 0.0,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map latent events to persistent calcium and noisy fluorescence traces."""

    spikes = validate_traces(events)
    decay = np.asarray(gamma, dtype=float)
    if decay.ndim == 0:
        decay = np.full(spikes.shape[0], float(decay))
    if decay.shape != (spikes.shape[0],) or np.any((decay < 0) | (decay >= 1)):
        raise ValueError("gamma must provide values in [0, 1) for each ROI")
    if noise_std < 0 or shared_noise_std < 0:
        raise ValueError("noise levels cannot be negative")
    calcium = np.zeros_like(spikes)
    calcium[:, 0] = spikes[:, 0]
    for time in range(1, spikes.shape[1]):
        calcium[:, time] = decay * calcium[:, time - 1] + spikes[:, time]
    rng = np.random.default_rng(random_state)
    shared = rng.normal(scale=shared_noise_std, size=(1, spikes.shape[1]))
    noise = rng.normal(scale=noise_std, size=spikes.shape)
    return calcium, calcium + shared + noise


def _decay_values(gamma: float | np.ndarray, n_rois: int, name: str) -> np.ndarray:
    decay = np.asarray(gamma, dtype=float)
    if decay.ndim == 0:
        decay = np.full(n_rois, float(decay))
    if decay.shape != (n_rois,) or np.any((decay < 0) | (decay >= 1)):
        raise ValueError(f"{name} must provide values in [0, 1) for each ROI")
    return decay


def _probability(value: float, name: str) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must lie in [0, 1]")


def _minimum_fall_length(config: DynamicSimulationConfig, rise_length: int) -> int:
    return max(
        int(2 * rise_length) + 1,
        int(ceil(config.fall_to_rise_ratio_min * rise_length)),
    )


def _minimum_rise_length(config: DynamicSimulationConfig) -> int:
    return max(config.min_rise_length, int(config.rise_waveform_length))


def _rise_waveform(config: DynamicSimulationConfig) -> np.ndarray:
    length = int(config.rise_waveform_length)
    return np.full(length, 1.0 / length, dtype=float)


def _validate_dynamic_config(
    config: DynamicSimulationConfig,
    adjacency: np.ndarray,
    n_steps: int,
) -> None:
    if config.n_episodes < 1:
        raise ValueError("n_episodes must be positive")
    if config.min_rise_length < 20:
        raise ValueError("dynamic rises must contain at least 20 samples")
    if (
        not isinstance(config.rise_waveform_length, (int, np.integer))
        or config.rise_waveform_length < 1
    ):
        raise ValueError("rise_waveform_length must be a positive integer")
    min_rise = _minimum_rise_length(config)
    if config.max_rise_length < min_rise:
        raise ValueError(
            "max_rise_length must be at least the effective minimum rise length"
        )
    if config.fall_to_rise_ratio_min <= 2.0:
        raise ValueError("fall_to_rise_ratio_min must be greater than 2")
    if config.fall_to_rise_ratio_max < config.fall_to_rise_ratio_min:
        raise ValueError("fall_to_rise_ratio_max must be at least the minimum ratio")
    if config.propagation_delay < 1:
        raise ValueError("propagation_delay must be positive")
    if config.propagation_delay >= min_rise:
        raise ValueError("propagation_delay must be shorter than the rise phase")
    if config.initial_activation_mode not in {"all_active", "source_nodes"}:
        raise ValueError(
            "initial_activation_mode must be 'all_active' or 'source_nodes'"
        )
    if config.rise_gain <= 0:
        raise ValueError("rise_gain must be positive")
    if config.fall_noise_scale < 0:
        raise ValueError("fall_noise_scale cannot be negative")
    if config.fall_initial_scale < 0:
        raise ValueError("fall_initial_scale cannot be negative")
    if config.fall_initial_ceiling_fraction < 0:
        raise ValueError("fall_initial_ceiling_fraction cannot be negative")
    if (
        not isinstance(config.fall_overlap_exclusion_samples, (int, np.integer))
        or config.fall_overlap_exclusion_samples < 0
    ):
        raise ValueError("fall_overlap_exclusion_samples must be a nonnegative integer")
    if config.fall_state_mode not in {"passive_decay", "stochastic_independent"}:
        raise ValueError(
            "fall_state_mode must be 'passive_decay' or 'stochastic_independent'"
        )
    _probability(config.initial_activation_probability, "initial_activation_probability")
    _probability(
        config.fall_initial_activation_probability,
        "fall_initial_activation_probability",
    )
    _probability(config.fall_spontaneous_rate, "fall_spontaneous_rate")
    _probability(config.fall_transmission_probability, "fall_transmission_probability")
    _probability(
        config.fall_propagation_drop_fraction,
        "fall_propagation_drop_fraction",
    )
    _probability(config.edge_dropout_probability, "edge_dropout_probability")
    _probability(config.edge_addition_probability, "edge_addition_probability")
    _probability(config.source_dropout_probability, "source_dropout_probability")
    _probability(config.source_recruitment_probability, "source_recruitment_probability")
    _probability(
        config.source_recruitment_edge_probability,
        "source_recruitment_edge_probability",
    )
    _probability(config.fall_noise_rate, "fall_noise_rate")
    min_fall = _minimum_fall_length(config, min_rise)
    min_total = config.n_episodes * (min_rise + min_fall)
    if n_steps < min_total:
        raise ValueError("n_steps is too short for the requested dynamic episodes")
    if config.adjacency_sequence is None:
        adjacency_sequence = None
    else:
        adjacency_sequence = config.adjacency_sequence
        if not adjacency_sequence:
            raise ValueError("adjacency_sequence cannot be empty")
        for graph in adjacency_sequence:
            values = np.asarray(graph)
            if values.shape != adjacency.shape:
                raise ValueError(
                    "each dynamic adjacency must match the base adjacency shape"
                )
    if config.active_node_sequence is None:
        pass
    else:
        if not config.active_node_sequence:
            raise ValueError("active_node_sequence cannot be empty")
        for active_nodes in config.active_node_sequence:
            values = np.asarray(active_nodes, dtype=bool)
            if values.shape != (adjacency.shape[0],):
                raise ValueError(
                    "each active-node mask must provide one value per ROI"
                )
            if not values.any():
                raise ValueError("each active-node mask must retain at least one ROI")
    if config.fall_adjacency_sequence is None:
        return
    if not config.fall_adjacency_sequence:
        raise ValueError("fall_adjacency_sequence cannot be empty")
    for graph in config.fall_adjacency_sequence:
        values = np.asarray(graph)
        if values.shape != adjacency.shape:
            raise ValueError(
                "each fall-truth adjacency must match the base adjacency shape"
            )


def _episode_active_nodes(
    adjacency: np.ndarray,
    config: DynamicSimulationConfig,
    episode_index: int,
) -> np.ndarray:
    if config.active_node_sequence is None:
        return np.ones(adjacency.shape[0], dtype=bool)
    return np.asarray(
        config.active_node_sequence[
            episode_index % len(config.active_node_sequence)
        ],
        dtype=bool,
    ).copy()


def _episode_adjacency(
    base_adjacency: np.ndarray,
    config: DynamicSimulationConfig,
    episode_index: int,
    rng: np.random.Generator,
    active_nodes: np.ndarray,
) -> np.ndarray:
    if config.adjacency_sequence is not None:
        graph = np.asarray(
            config.adjacency_sequence[episode_index % len(config.adjacency_sequence)],
            dtype=bool,
        ).copy()
    else:
        graph = base_adjacency.copy()
        if config.edge_dropout_probability > 0:
            graph &= rng.random(graph.shape) >= config.edge_dropout_probability
            if base_adjacency.any() and not graph.any():
                edges = np.argwhere(base_adjacency)
                source, target = edges[int(rng.integers(0, len(edges)))]
                graph[source, target] = True
        if config.edge_addition_probability > 0:
            candidates = ~base_adjacency & ~np.eye(base_adjacency.shape[0], dtype=bool)
            graph |= candidates & (
                rng.random(base_adjacency.shape) < config.edge_addition_probability
            )
        if config.source_dropout_probability > 0:
            base_sources = np.flatnonzero(base_adjacency.any(axis=1) & active_nodes)
            dropped = (
                rng.random(base_sources.size) < config.source_dropout_probability
            )
            if dropped.any():
                graph[base_sources[dropped], :] = False
        if config.source_recruitment_probability > 0:
            base_sources = base_adjacency.any(axis=1)
            recruitable = np.flatnonzero((~base_sources) & active_nodes)
            recruited = (
                rng.random(recruitable.size) < config.source_recruitment_probability
            )
            for source in recruitable[recruited]:
                candidates = active_nodes.copy()
                candidates[source] = False
                targets = (
                    rng.random(base_adjacency.shape[0])
                    < config.source_recruitment_edge_probability
                ) & candidates
                if (
                    config.source_recruitment_edge_probability > 0
                    and not targets.any()
                ):
                    eligible = np.flatnonzero(candidates)
                    if eligible.size:
                        targets[
                            int(eligible[int(rng.integers(0, len(eligible)))])
                        ] = True
                graph[source, targets] = True
    graph[~active_nodes, :] = False
    graph[:, ~active_nodes] = False
    np.fill_diagonal(graph, False)
    return graph


def _episode_fall_adjacency(
    base_adjacency: np.ndarray,
    rise_graph: np.ndarray,
    config: DynamicSimulationConfig,
    episode_index: int,
    active_nodes: np.ndarray,
) -> np.ndarray:
    if config.fall_adjacency_sequence is None:
        graph = np.zeros_like(base_adjacency, dtype=bool)
    else:
        graph = np.asarray(
            config.fall_adjacency_sequence[
                episode_index % len(config.fall_adjacency_sequence)
            ],
            dtype=bool,
        ).copy()
    if graph.shape != rise_graph.shape:
        raise ValueError("fall-truth adjacency must match the rise adjacency shape")
    graph[~active_nodes, :] = False
    graph[:, ~active_nodes] = False
    np.fill_diagonal(graph, False)
    return graph


def _simulate_dynamic_calcium_dataset(
    adjacency: np.ndarray,
    n_steps: int,
    gamma: float | np.ndarray,
    noise_std: float,
    shared_noise_std: float,
    spontaneous_rate: float,
    transmission_probability: float,
    random_state: int | None,
    config: DynamicSimulationConfig,
) -> SyntheticDataset:
    """Produce an episodic trace with separately declared rise and fall truth."""

    base = np.asarray(adjacency, dtype=bool).copy()
    if base.ndim != 2 or base.shape[0] != base.shape[1]:
        raise ValueError("adjacency must be square")
    np.fill_diagonal(base, False)
    if not 0 <= spontaneous_rate <= 1 or not 0 <= transmission_probability <= 1:
        raise ValueError("event probabilities must lie in [0, 1]")
    if noise_std < 0 or shared_noise_std < 0:
        raise ValueError("noise levels cannot be negative")
    _validate_dynamic_config(config, base, n_steps)

    n_rois = base.shape[0]
    gamma_rise = _decay_values(gamma, n_rois, "gamma")
    gamma_fall = gamma_rise if config.gamma_fall is None else _decay_values(
        config.gamma_fall, n_rois, "gamma_fall"
    )
    rng = np.random.default_rng(random_state)
    events = np.zeros((n_rois, n_steps), dtype=float)
    propagated_events = np.zeros_like(events)
    rise_drive = np.zeros_like(events)
    calcium = np.zeros_like(events)
    phase_labels = np.zeros(n_steps, dtype=int)
    episodes: list[SyntheticEpisode] = []
    rise_adjacencies: list[np.ndarray] = []
    fall_truth_adjacencies: list[np.ndarray] = []
    union_adjacency = np.zeros_like(base, dtype=bool)
    fall_truth_union = np.zeros_like(base, dtype=bool)
    edge_presence_counts = np.zeros(base.shape, dtype=int)
    node_presence_counts = np.zeros(n_rois, dtype=int)
    state = np.zeros(n_rois, dtype=float)
    fall_truth_events = np.zeros_like(events)
    cursor = 0
    min_rise_length = _minimum_rise_length(config)
    rise_waveform = _rise_waveform(config)
    min_episode_total = min_rise_length + _minimum_fall_length(
        config, min_rise_length
    )

    for episode_index in range(config.n_episodes):
        remaining_episodes = config.n_episodes - episode_index - 1
        available = n_steps - cursor - remaining_episodes * min_episode_total
        max_rise = min(config.max_rise_length, available)
        while max_rise >= min_rise_length:
            if max_rise + _minimum_fall_length(config, max_rise) <= available:
                break
            max_rise -= 1
        if max_rise < min_rise_length:
            raise ValueError("n_steps cannot fit the requested dynamic episode schedule")
        rise_length = int(
            rng.integers(min_rise_length, max_rise + 1)
        )
        min_fall = _minimum_fall_length(config, rise_length)
        max_fall = min(
            int(ceil(config.fall_to_rise_ratio_max * rise_length)),
            available - rise_length,
        )
        fall_length = int(rng.integers(min_fall, max_fall + 1))
        rise_start = cursor
        rise_stop = rise_start + rise_length
        fall_start = rise_stop
        fall_stop = fall_start + fall_length
        active_nodes = _episode_active_nodes(base, config, episode_index)
        graph = _episode_adjacency(base, config, episode_index, rng, active_nodes)
        fall_graph = _episode_fall_adjacency(
            base,
            graph,
            config,
            episode_index,
            active_nodes,
        )
        rise_adjacencies.append(graph.copy())
        fall_truth_adjacencies.append(fall_graph.copy())
        union_adjacency |= graph
        fall_truth_union |= fall_graph
        edge_presence_counts += graph.astype(int)
        node_presence_counts += active_nodes.astype(int)
        episodes.append(
            SyntheticEpisode(
                rise_start,
                rise_stop,
                fall_start,
                fall_stop,
                graph,
                active_nodes=active_nodes.copy(),
                fall_adjacency=fall_graph.copy(),
            )
        )
        state = state.copy()
        state[~active_nodes] = 0.0

        for time in range(rise_start, rise_stop):
            phase_labels[time] = 1
            if time == rise_start:
                initial_candidates = active_nodes
                if config.initial_activation_mode == "source_nodes":
                    source_nodes = active_nodes & ~np.any(graph, axis=0)
                    if source_nodes.any():
                        initial_candidates = source_nodes
                spontaneous = (
                    rng.random(n_rois) < config.initial_activation_probability
                ) & initial_candidates
                if initial_candidates.any() and not spontaneous.any():
                    available_nodes = np.flatnonzero(initial_candidates)
                    spontaneous[
                        available_nodes[
                            int(rng.integers(0, len(available_nodes)))
                        ]
                    ] = True
            else:
                spontaneous = (rng.random(n_rois) < spontaneous_rate) & active_nodes
            if time - config.propagation_delay >= rise_start:
                incoming = (
                    graph.T @ (events[:, time - config.propagation_delay] > 0) > 0
                )
            else:
                incoming = np.zeros(n_rois, dtype=bool)
            propagated = (
                incoming
                & (rng.random(n_rois) < transmission_probability)
                & active_nodes
            )
            onset = (spontaneous | propagated).astype(float) * config.rise_gain
            events[:, time] = onset
            propagated_events[:, time] = propagated.astype(float) * config.rise_gain
            drive_stop = min(rise_stop, time + rise_waveform.size)
            drive_length = drive_stop - time
            if drive_length > 0:
                rise_drive[:, time:drive_stop] += (
                    onset[:, None] * rise_waveform[:drive_length][None, :]
                )
            state = gamma_rise * state + rise_drive[:, time]
            state[~active_nodes] = 0.0
            calcium[:, time] = state

        if config.fall_state_mode == "stochastic_independent":
            reset_state = np.zeros(n_rois, dtype=float)
            ceiling = np.maximum(config.fall_initial_ceiling_fraction * state, 0.0)
            if active_nodes.any():
                reset_state[active_nodes] = np.minimum(
                    rng.exponential(
                        config.fall_initial_scale, size=int(np.count_nonzero(active_nodes))
                    ),
                    ceiling[active_nodes],
                )
            state = reset_state
        else:
            state = state.copy()
            state[~active_nodes] = 0.0

        for time in range(fall_start, fall_stop):
            phase_labels[time] = 2
            truth_events = np.zeros(n_rois, dtype=bool)
            propagated = np.zeros(n_rois, dtype=bool)
            if fall_graph.any():
                eligible = np.ones(n_rois, dtype=bool)
                if config.fall_overlap_exclusion_samples > 0:
                    recent = np.flatnonzero(
                        np.any(
                            fall_truth_events[
                                :,
                                max(fall_start, time - config.fall_overlap_exclusion_samples) : time,
                            ]
                            > 0,
                            axis=1,
                        )
                    )
                    eligible[recent] = False
                if time == fall_start:
                    sources = active_nodes & np.any(fall_graph, axis=1)
                    initial_candidates = sources if sources.any() else active_nodes
                    spontaneous = (
                        rng.random(n_rois) < config.fall_initial_activation_probability
                    ) & initial_candidates
                    if (
                        config.fall_initial_activation_probability > 0
                        and initial_candidates.any()
                        and not spontaneous.any()
                    ):
                        candidates = np.flatnonzero(initial_candidates)
                        spontaneous[
                            candidates[int(rng.integers(0, len(candidates)))]
                        ] = True
                else:
                    spontaneous = (
                        rng.random(n_rois) < config.fall_spontaneous_rate
                    ) & active_nodes
                spontaneous &= eligible
                if time - config.propagation_delay >= fall_start:
                    incoming = (
                        fall_graph.T
                        @ (fall_truth_events[:, time - config.propagation_delay] > 0)
                        > 0
                    )
                    propagated = (
                        incoming
                        & (rng.random(n_rois) < config.fall_transmission_probability)
                        & active_nodes
                        & eligible
                    )
                truth_events = spontaneous | propagated
                fall_truth_events[:, time] = truth_events.astype(float)
            local_noise = np.zeros(n_rois, dtype=float)
            noisy_nodes = (rng.random(n_rois) < config.fall_noise_rate) & active_nodes
            if noisy_nodes.any():
                local_noise[noisy_nodes] = rng.exponential(
                    config.fall_noise_scale,
                    size=int(np.count_nonzero(noisy_nodes)),
                )
            max_noise = np.maximum((1.0 - gamma_fall) * state * 0.8, 0.0)
            noise_innovation = np.minimum(local_noise, max_noise)
            fall_drop = (
                truth_events.astype(float)
                * config.fall_propagation_drop_fraction
                * np.maximum(state, 0.0)
            )
            events[:, time] = truth_events.astype(float) + noise_innovation
            propagated_events[:, time] = propagated.astype(float)
            state = np.maximum(
                gamma_fall * state - fall_drop + noise_innovation,
                0.0,
            )
            state[~active_nodes] = 0.0
            calcium[:, time] = state

        cursor = fall_stop

    for time in range(cursor, n_steps):
        state = gamma_fall * state
        calcium[:, time] = state

    shared = rng.normal(scale=shared_noise_std, size=(1, n_steps))
    noise = rng.normal(scale=noise_std, size=calcium.shape)
    edge_prevalence = edge_presence_counts / len(episodes)
    node_prevalence = node_presence_counts / len(episodes)
    return SyntheticDataset(
        adjacency=union_adjacency,
        events=events,
        calcium=calcium,
        fluorescence=calcium + shared + noise,
        phase_labels=phase_labels,
        episodes=tuple(episodes),
        propagated_events=propagated_events,
        rise_adjacencies=tuple(rise_adjacencies),
        edge_presence_counts=edge_presence_counts,
        edge_prevalence=edge_prevalence,
        node_presence_counts=node_presence_counts,
        node_prevalence=node_prevalence,
        fall_truth_adjacency=fall_truth_union,
        fall_truth_adjacencies=tuple(fall_truth_adjacencies),
    )


def simulate_calcium_dataset(
    adjacency: np.ndarray,
    n_steps: int = 1000,
    gamma: float | np.ndarray = 0.8,
    noise_std: float = 0.05,
    shared_noise_std: float = 0.0,
    spontaneous_rate: float = 0.025,
    transmission_probability: float = 0.8,
    random_state: int | None = None,
    propagation_delay: int = 1,
    simulator_mode: str = "static",
    dynamic_config: DynamicSimulationConfig | None = None,
) -> SyntheticDataset:
    """Produce synthetic calcium observations with directed-event truth."""

    if simulator_mode not in {"static", "episodic_dynamic"}:
        raise ValueError("simulator_mode must be 'static' or 'episodic_dynamic'")
    if simulator_mode == "episodic_dynamic":
        return _simulate_dynamic_calcium_dataset(
            adjacency,
            n_steps,
            gamma,
            noise_std,
            shared_noise_std,
            spontaneous_rate,
            transmission_probability,
            random_state,
            DynamicSimulationConfig() if dynamic_config is None else dynamic_config,
        )

    events = simulate_events(
        adjacency,
        n_steps,
        spontaneous_rate,
        transmission_probability,
        random_state,
        propagation_delay,
    )
    calcium, fluorescence = calcium_observation_model(
        events, gamma, noise_std, shared_noise_std, random_state
    )
    return SyntheticDataset(
        adjacency=np.asarray(adjacency, dtype=bool),
        events=events,
        calcium=calcium,
        fluorescence=fluorescence,
    )


def cyclic_shift_surrogate(
    traces: np.ndarray,
    random_state: int | None = None,
    minimum_shift: int = 1,
) -> np.ndarray:
    """Independently cyclic-shift ROIs to disrupt between-ROI temporal order."""

    values = validate_traces(traces)
    if minimum_shift < 1 or minimum_shift >= values.shape[1]:
        raise ValueError("minimum_shift must allow at least one nonzero shift")
    rng = np.random.default_rng(random_state)
    shifted = values.copy()
    for roi in range(values.shape[0]):
        shift = int(rng.integers(minimum_shift, values.shape[1]))
        shifted[roi] = np.roll(values[roi], shift)
    return shifted


def jitter_event_indices(
    event_indices: tuple[np.ndarray, ...] | list[np.ndarray],
    n_steps: int,
    max_jitter: int = 1,
    random_state: int | None = None,
) -> tuple[np.ndarray, ...]:
    """Randomly perturb selected event frames while staying on the time axis."""

    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if max_jitter < 0:
        raise ValueError("max_jitter cannot be negative")
    rng = np.random.default_rng(random_state)
    jittered = []
    for values in event_indices:
        frames = np.asarray(values, dtype=int)
        if frames.size == 0 or max_jitter == 0:
            jittered.append(frames.copy())
            continue
        offsets = rng.integers(-max_jitter, max_jitter + 1, size=frames.size)
        jittered.append(np.unique(np.clip(frames + offsets, 0, n_steps - 1)))
    return tuple(jittered)


def reverse_event_indices(
    event_indices: tuple[np.ndarray, ...] | list[np.ndarray],
    n_steps: int,
) -> tuple[np.ndarray, ...]:
    """Map selected frames onto the reversed time axis."""

    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    return tuple(
        np.sort(n_steps - 1 - np.asarray(values, dtype=int))
        for values in event_indices
    )


def permute_phase_event_indices(
    primary: tuple[np.ndarray, ...] | list[np.ndarray],
    alternative: tuple[np.ndarray, ...] | list[np.ndarray],
    random_state: int | None = None,
) -> tuple[np.ndarray, ...]:
    """Swap primary and alternative event sets independently by ROI."""

    if len(primary) != len(alternative):
        raise ValueError("primary and alternative must have one entry per ROI")
    rng = np.random.default_rng(random_state)
    permuted = []
    for first, second in zip(primary, alternative):
        selected = second if rng.random() < 0.5 else first
        permuted.append(np.asarray(selected, dtype=int).copy())
    return tuple(permuted)


def bootstrap_event_indices(
    event_indices: tuple[np.ndarray, ...] | list[np.ndarray],
    random_state: int | None = None,
) -> tuple[np.ndarray, ...]:
    """Sample selected frames with replacement within each ROI."""

    rng = np.random.default_rng(random_state)
    sampled = []
    for values in event_indices:
        frames = np.asarray(values, dtype=int)
        if frames.size == 0:
            sampled.append(frames.copy())
        else:
            sampled.append(np.sort(rng.choice(frames, size=frames.size, replace=True)))
    return tuple(sampled)


def cross_recording_surrogate(reference: np.ndarray, donor: np.ndarray) -> np.ndarray:
    """Build a shape-matched surrogate from another recording's ROI traces."""

    values = validate_traces(reference)
    donor_values = validate_traces(donor)
    if donor_values.shape[1] < values.shape[1]:
        repeats = int(np.ceil(values.shape[1] / donor_values.shape[1]))
        donor_values = np.tile(donor_values, (1, repeats))
    surrogate = np.zeros_like(values)
    for roi in range(values.shape[0]):
        surrogate[roi] = donor_values[roi % donor_values.shape[0], : values.shape[1]]
    return surrogate


def run_null_controls(
    traces: np.ndarray,
    estimator: CausalisedGC,
    n_surrogates: int = 20,
    random_state: int | None = None,
    segment_ids: np.ndarray | None = None,
) -> NullControlResult:
    """Run temporal nulls that preserve within-ROI marginal trace structure."""

    if n_surrogates < 1:
        raise ValueError("n_surrogates must be positive")
    values = validate_traces(traces)
    rng = np.random.default_rng(random_state)
    shifted = []
    for _ in range(n_surrogates):
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        surrogate = cyclic_shift_surrogate(
            values,
            random_state=seed,
            minimum_shift=getattr(estimator, "n_pasts", estimator.max_lag) + 1,
        )
        shifted.append(estimator.fit(surrogate, segment_ids=segment_ids))
    return NullControlResult(
        observed=estimator.fit(values, segment_ids=segment_ids),
        shifted=tuple(shifted),
        reverse_time=estimator.fit(
            values[:, ::-1],
            segment_ids=None if segment_ids is None else segment_ids[::-1],
        ),
    )


def run_event_null_controls(
    traces: np.ndarray,
    estimator: CausalisedGC,
    event_indices: tuple[np.ndarray, ...] | list[np.ndarray],
    *,
    alternative_event_indices: tuple[np.ndarray, ...] | list[np.ndarray] | None = None,
    donor_traces: np.ndarray | None = None,
    n_jittered: int = 20,
    n_phase_permutations: int = 20,
    max_jitter: int = 1,
    random_state: int | None = None,
) -> EventNullControlResult:
    """Run event-aware nulls for selected-frame c-GC analyses."""

    if n_jittered < 0 or n_phase_permutations < 0:
        raise ValueError("null replicate counts cannot be negative")
    values = validate_traces(traces)
    selected = tuple(np.asarray(values_, dtype=int) for values_ in event_indices)
    rng = np.random.default_rng(random_state)
    jittered = []
    for _ in range(n_jittered):
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        jittered.append(
            estimator.fit(
                values,
                event_indices=jitter_event_indices(
                    selected, values.shape[1], max_jitter, seed
                ),
            )
        )
    phase_permuted = []
    if alternative_event_indices is not None:
        alternative = tuple(
            np.asarray(values_, dtype=int) for values_ in alternative_event_indices
        )
        for _ in range(n_phase_permutations):
            seed = int(rng.integers(0, np.iinfo(np.int32).max))
            phase_permuted.append(
                estimator.fit(
                    values,
                    event_indices=permute_phase_event_indices(
                        selected, alternative, seed
                    ),
                )
            )
    cross_recording = None
    if donor_traces is not None:
        cross_recording = estimator.fit(
            cross_recording_surrogate(values, donor_traces),
            event_indices=selected,
        )
    return EventNullControlResult(
        observed=estimator.fit(values, event_indices=selected),
        jittered=tuple(jittered),
        reverse_time=estimator.fit(
            values[:, ::-1],
            event_indices=reverse_event_indices(selected, values.shape[1]),
        ),
        phase_permuted=tuple(phase_permuted),
        cross_recording=cross_recording,
    )


def run_event_bootstrap_stability(
    traces: np.ndarray,
    estimator: CausalisedGC,
    event_indices: tuple[np.ndarray, ...] | list[np.ndarray],
    n_bootstrap: int = 20,
    random_state: int | None = None,
) -> StabilityResult:
    """Estimate selected-frame graph stability by event bootstrap."""

    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least two")
    values = validate_traces(traces)
    selected = tuple(np.asarray(values_, dtype=int) for values_ in event_indices)
    rng = np.random.default_rng(random_state)
    graphs = []
    for _ in range(n_bootstrap):
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        graphs.append(
            estimator.fit(
                values,
                event_indices=bootstrap_event_indices(selected, seed),
            )
        )
    return StabilityResult(
        graphs=tuple(graphs),
        stability=graph_stability([graph.adjacency for graph in graphs]),
    )


def _windowed_event_indices(
    event_indices: tuple[np.ndarray, ...],
    start: int,
    stop: int,
) -> tuple[np.ndarray, ...]:
    selected = []
    for frames in event_indices:
        values = frames[(frames >= start) & (frames < stop)] - start
        selected.append(values.astype(int, copy=False))
    return tuple(selected)


def run_time_window_stability(
    traces: np.ndarray,
    estimator: CausalisedGC,
    window_length: int,
    step: int | None = None,
    *,
    event_indices: tuple[np.ndarray, ...] | list[np.ndarray] | None = None,
) -> StabilityResult:
    """Estimate graph stability across contiguous time windows."""

    values = validate_traces(traces)
    if not isinstance(window_length, int) or window_length < 3:
        raise ValueError("window_length must be an integer of at least three")
    if window_length > values.shape[1]:
        raise ValueError("window_length cannot exceed the trace length")
    stride = window_length if step is None else step
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("step must be a positive integer")
    selected = (
        None
        if event_indices is None
        else tuple(np.asarray(frames, dtype=int) for frames in event_indices)
    )
    if selected is not None and len(selected) != values.shape[0]:
        raise ValueError("event_indices must contain one entry per ROI")

    graphs = []
    for start in range(0, values.shape[1] - window_length + 1, stride):
        stop = start + window_length
        window_events = (
            None if selected is None else _windowed_event_indices(selected, start, stop)
        )
        graphs.append(
            estimator.fit(values[:, start:stop], event_indices=window_events)
        )
    if len(graphs) < 2:
        raise ValueError("at least two windows are required for stability")
    return StabilityResult(
        graphs=tuple(graphs),
        stability=graph_stability([graph.adjacency for graph in graphs]),
    )


def run_leave_one_neuron_stability(
    traces: np.ndarray,
    estimator: CausalisedGC,
    *,
    event_indices: tuple[np.ndarray, ...] | list[np.ndarray] | None = None,
) -> StabilityResult:
    """Estimate graph stability after removing each ROI in turn."""

    values = validate_traces(traces)
    if values.shape[0] < 3:
        raise ValueError("at least three ROIs are required for leave-one-neuron stability")
    selected = (
        None
        if event_indices is None
        else tuple(np.asarray(frames, dtype=int) for frames in event_indices)
    )
    if selected is not None and len(selected) != values.shape[0]:
        raise ValueError("event_indices must contain one entry per ROI")

    graphs = []
    for omitted in range(values.shape[0]):
        keep = np.array([index for index in range(values.shape[0]) if index != omitted])
        subset_events = (
            None if selected is None else tuple(selected[index] for index in keep)
        )
        graph = estimator.fit(values[keep], event_indices=subset_events)
        scores = np.zeros((values.shape[0], values.shape[0]), dtype=float)
        p_values = np.ones_like(scores)
        adjacency = np.zeros((values.shape[0], values.shape[0]), dtype=bool)
        best_lags = np.zeros_like(adjacency, dtype=int)
        scores[np.ix_(keep, keep)] = graph.scores
        p_values[np.ix_(keep, keep)] = graph.p_values
        adjacency[np.ix_(keep, keep)] = graph.adjacency
        best_lags[np.ix_(keep, keep)] = graph.best_lags
        graphs.append(
            GraphResult(
                scores=scores,
                p_values=p_values,
                adjacency=adjacency,
                best_lags=best_lags,
                estimator=graph.estimator,
            )
        )
    return StabilityResult(
        graphs=tuple(graphs),
        stability=graph_stability([graph.adjacency for graph in graphs]),
    )


def _contiguous_segments(frames: np.ndarray) -> tuple[np.ndarray, ...]:
    if frames.size == 0:
        return ()
    values = np.unique(np.asarray(frames, dtype=int))
    breaks = np.flatnonzero(np.diff(values) > 1) + 1
    return tuple(np.asarray(segment, dtype=int) for segment in np.split(values, breaks))


def _transient_segments(
    event_indices: tuple[np.ndarray, ...],
) -> tuple[tuple[int, np.ndarray], ...]:
    segments = []
    for roi, frames in enumerate(event_indices):
        for segment in _contiguous_segments(frames):
            segments.append((roi, segment))
    return tuple(segments)


def run_leave_one_transient_stability(
    traces: np.ndarray,
    estimator: CausalisedGC,
    event_indices: tuple[np.ndarray, ...] | list[np.ndarray],
) -> StabilityResult:
    """Estimate selected-frame graph stability after dropping one transient."""

    values = validate_traces(traces)
    selected = tuple(np.asarray(frames, dtype=int) for frames in event_indices)
    if len(selected) != values.shape[0]:
        raise ValueError("event_indices must contain one entry per ROI")
    segments = _transient_segments(selected)
    if len(segments) < 2:
        raise ValueError("at least two transient segments are required")

    graphs = []
    for roi, segment in segments:
        reduced = [frames.copy() for frames in selected]
        reduced[roi] = np.setdiff1d(reduced[roi], segment, assume_unique=True)
        graphs.append(estimator.fit(values, event_indices=tuple(reduced)))
    return StabilityResult(
        graphs=tuple(graphs),
        stability=graph_stability([graph.adjacency for graph in graphs]),
    )


def _representation_truths(dataset: SyntheticDataset) -> dict[str, np.ndarray]:
    union = np.asarray(dataset.adjacency, dtype=bool).copy()
    fall_truth = dataset.fall_union_adjacency
    truth = {
        "full": union.copy(),
        "deconvolved": union.copy(),
        "rise": union.copy(),
        "fall": union.copy(),
        "fall_residual": union.copy(),
    }
    if dataset.episodes:
        if fall_truth is None:
            fall_truth = np.zeros_like(union, dtype=bool)
        truth["fall"] = fall_truth.copy()
        truth["fall_residual"] = fall_truth.copy()
    return truth


def _episode_segment_ids(dataset: SyntheticDataset, phase: str) -> np.ndarray | None:
    if not dataset.episodes:
        return None
    if phase not in {"rise", "fall"}:
        raise ValueError("phase must be 'rise' or 'fall'")
    n_steps = dataset.fluorescence.shape[1]
    segment_ids = np.full(n_steps, -1, dtype=int)
    for index, episode in enumerate(dataset.episodes, start=1):
        if phase == "rise":
            segment_ids[episode.rise_start : episode.rise_stop] = index
        else:
            segment_ids[episode.fall_start : episode.fall_stop] = index
    return segment_ids


def _fit_selected_representation(
    dataset: SyntheticDataset,
    estimator: CausalisedGC,
    representation: np.ndarray,
    phase: str,
) -> GraphResult:
    event_indices = selected_frame_indices(representation)
    segment_ids = None
    if getattr(estimator, "event_mode", None) == "physical":
        segment_ids = _episode_segment_ids(dataset, phase)
    return estimator.fit(
        dataset.fluorescence,
        segment_ids=segment_ids,
        event_indices=event_indices,
    )


def _fit_rise_representation(
    dataset: SyntheticDataset,
    estimator: CausalisedGC,
    representation: np.ndarray,
) -> tuple[GraphResult, RiseFlankCandidateResult | None]:
    candidates = None
    candidate_adjacency = None
    if (
        getattr(estimator, "min_rise_run_samples", 1) > 1
        or getattr(estimator, "rise_candidate_filter", False)
    ):
        candidates = estimator.rise_flank_candidates(representation)
        event_indices = candidates.event_indices
        if estimator.rise_candidate_filter:
            candidate_adjacency = candidates.candidate_adjacency
    else:
        event_indices = selected_frame_indices(representation)

    segment_ids = None
    if getattr(estimator, "event_mode", None) == "physical":
        segment_ids = _episode_segment_ids(dataset, "rise")
    graph = estimator.fit(
        dataset.fluorescence,
        segment_ids=segment_ids,
        event_indices=event_indices,
        candidate_adjacency=candidate_adjacency,
    )
    return graph, candidates


def validate_representations(
    dataset: SyntheticDataset,
    estimator: CausalisedGC,
    tolerance: float | np.ndarray = 0.0,
    gamma: float | np.ndarray | None = None,
) -> RepresentationValidation:
    """Estimate each representation and score recovery against known truth."""

    representations = build_representations(
        dataset.fluorescence, tolerance=tolerance, gamma=gamma
    )
    rise_graph, rise_flank_candidates = _fit_rise_representation(
        dataset, estimator, representations.rise
    )
    graphs = {
        "full": estimator.fit(representations.full),
        "deconvolved": estimator.fit(representations.deconvolved),
        "rise": rise_graph,
        "fall": _fit_selected_representation(
            dataset, estimator, representations.fall, "fall"
        ),
        "fall_residual": _fit_selected_representation(
            dataset, estimator, representations.fall_residual, "fall"
        ),
    }
    truth = _representation_truths(dataset)
    recovery = {
        label: evaluate_against_truth(truth[label], graph)
        for label, graph in graphs.items()
    }
    return RepresentationValidation(
        graphs=graphs,
        recovery=recovery,
        truth=truth,
        rise_flank_candidates=rise_flank_candidates,
    )


def evaluate_against_truth(
    truth: np.ndarray, graph: GraphResult
) -> RecoverySummary:
    """Score a discovered graph against a synthetic directed network."""

    return edge_recovery(truth, graph.adjacency)
