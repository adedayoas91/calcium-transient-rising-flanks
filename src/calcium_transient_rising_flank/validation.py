"""Synthetic and null-control utilities for pipeline validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .estimators import CausalGranger, GraphResult, selected_frame_indices
from .metrics import RecoverySummary, edge_recovery, graph_stability
from .preprocessing import validate_traces
from .representations import build_representations


@dataclass(frozen=True)
class SyntheticDataset:
    adjacency: np.ndarray
    events: np.ndarray
    calcium: np.ndarray
    fluorescence: np.ndarray


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


def simulate_events(
    adjacency: np.ndarray,
    n_steps: int,
    spontaneous_rate: float = 0.025,
    transmission_probability: float = 0.8,
    random_state: int | None = None,
) -> np.ndarray:
    """Generate binary events on a directed network with known structure."""

    network = np.asarray(adjacency, dtype=bool).copy()
    if network.ndim != 2 or network.shape[0] != network.shape[1]:
        raise ValueError("adjacency must be square")
    if n_steps < 3:
        raise ValueError("n_steps must be at least three")
    if not 0 <= spontaneous_rate <= 1 or not 0 <= transmission_probability <= 1:
        raise ValueError("event probabilities must lie in [0, 1]")
    np.fill_diagonal(network, False)
    rng = np.random.default_rng(random_state)
    events = np.zeros((network.shape[0], n_steps), dtype=float)
    events[:, 0] = rng.random(network.shape[0]) < spontaneous_rate
    for time in range(1, n_steps):
        spontaneous = rng.random(network.shape[0]) < spontaneous_rate
        incoming = network.T @ events[:, time - 1] > 0
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


def simulate_calcium_dataset(
    adjacency: np.ndarray,
    n_steps: int = 1000,
    gamma: float | np.ndarray = 0.8,
    noise_std: float = 0.05,
    shared_noise_std: float = 0.0,
    spontaneous_rate: float = 0.025,
    transmission_probability: float = 0.8,
    random_state: int | None = None,
) -> SyntheticDataset:
    """Produce synthetic calcium observations with directed-event truth."""

    events = simulate_events(
        adjacency,
        n_steps,
        spontaneous_rate,
        transmission_probability,
        random_state,
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
    estimator: CausalGranger,
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
            values, random_state=seed, minimum_shift=estimator.max_lag + 1
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
    estimator: CausalGranger,
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
    estimator: CausalGranger,
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


def validate_representations(
    dataset: SyntheticDataset,
    estimator: CausalGranger,
    tolerance: float | np.ndarray = 0.0,
    gamma: float | np.ndarray | None = None,
) -> RepresentationValidation:
    """Estimate each representation and score recovery against known truth."""

    representations = build_representations(
        dataset.fluorescence, tolerance=tolerance, gamma=gamma
    )
    graphs = {
        "full": estimator.fit(representations.full),
        "deconvolved": estimator.fit(representations.deconvolved),
        "rise": estimator.fit(
            dataset.fluorescence,
            event_indices=selected_frame_indices(representations.rise),
        ),
        "fall": estimator.fit(
            dataset.fluorescence,
            event_indices=selected_frame_indices(representations.fall),
        ),
        "fall_residual": estimator.fit(
            dataset.fluorescence,
            event_indices=selected_frame_indices(representations.fall_residual),
        ),
    }
    recovery = {
        label: evaluate_against_truth(dataset.adjacency, graph)
        for label, graph in graphs.items()
    }
    return RepresentationValidation(graphs=graphs, recovery=recovery)


def evaluate_against_truth(
    truth: np.ndarray, graph: GraphResult
) -> RecoverySummary:
    """Score a discovered graph against a synthetic directed network."""

    return edge_recovery(truth, graph.adjacency)
