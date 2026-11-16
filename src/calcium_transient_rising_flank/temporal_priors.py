"""Episode-aware temporal priors for rising-flank onset ordering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocessing import validate_traces


@dataclass(frozen=True)
class TemporalPriorResult:
    """Pairwise hard/soft temporal priors derived from episode onsets."""

    robust_hard_mask: np.ndarray
    hypothesis_weights: np.ndarray
    decisive_episode_counts: np.ndarray
    matched_episode_counts: np.ndarray
    tie_episode_counts: np.ndarray
    direction_consistency: np.ndarray
    mean_decisive_lag: np.ndarray
    median_decisive_lag: np.ndarray
    screen_episode_ids: tuple[tuple[tuple[int, ...], ...], ...]


@dataclass(frozen=True)
class TemporalPriorStateMasks:
    """Explicit directional, ambiguous, and unmatched pair states.

    ``directional`` is oriented. ``ambiguous`` and ``unmatched`` are symmetric
    masks because those states belong to an unordered ROI pair.
    """

    directional: np.ndarray
    ambiguous: np.ndarray
    unmatched: np.ndarray


def temporal_prior_state_masks(
    prior: TemporalPriorResult,
) -> TemporalPriorStateMasks:
    """Resolve the robust hard mask into mutually exclusive pair states."""

    hard = np.asarray(prior.robust_hard_mask, dtype=bool)
    if hard.ndim != 2 or hard.shape[0] != hard.shape[1]:
        raise ValueError("robust_hard_mask must be square")
    off_diagonal = ~np.eye(hard.shape[0], dtype=bool)
    directional = hard & ~hard.T & off_diagonal
    ambiguous = hard & hard.T & off_diagonal
    unmatched = ~hard & ~hard.T & off_diagonal
    return TemporalPriorStateMasks(
        directional=directional,
        ambiguous=ambiguous,
        unmatched=unmatched,
    )


@dataclass(frozen=True)
class _EpisodeVote:
    label: str
    lag_summary: float | None


def _validate_nonnegative_integer(name: str, value: int, *, minimum: int = 0) -> int:
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    integer = int(value)
    if integer < minimum:
        if minimum == 0:
            raise ValueError(f"{name} cannot be negative")
        raise ValueError(f"{name} must be at least {minimum}")
    return integer


def _validate_segment_ids(segment_ids: np.ndarray, n_steps: int) -> np.ndarray:
    values = np.asarray(segment_ids)
    if values.shape != (n_steps,):
        raise ValueError("segment_ids must contain one value per timepoint")
    if not np.isfinite(values).all():
        raise ValueError("segment_ids must contain only finite values")
    rounded = np.rint(values)
    if not np.array_equal(values, rounded):
        raise ValueError("segment_ids must contain only integer-valued labels")
    return rounded.astype(int, copy=False)


def _extract_episode_onsets(
    representation: np.ndarray,
    segment_ids: np.ndarray,
    min_run_samples: int,
) -> tuple[dict[int, tuple[int, ...]], ...]:
    positive = representation > 0.0
    onsets_by_roi: list[dict[int, list[int]]] = []
    for row in positive:
        valid_frames = np.flatnonzero(row & (segment_ids >= 0))
        onsets: dict[int, list[int]] = {}
        if valid_frames.size:
            frame_steps = np.diff(valid_frames)
            segment_steps = np.diff(segment_ids[valid_frames])
            breaks = np.flatnonzero((frame_steps > 1) | (segment_steps != 0)) + 1
            for run in np.split(valid_frames, breaks):
                if run.size < min_run_samples:
                    continue
                episode_id = int(segment_ids[run[0]])
                onsets.setdefault(episode_id, []).append(int(run[0]))
        onsets_by_roi.append({key: tuple(values) for key, values in onsets.items()})
    return tuple(onsets_by_roi)


def _match_onsets(
    left_onsets: tuple[int, ...],
    right_onsets: tuple[int, ...],
    max_onset_lag: int,
) -> list[int]:
    candidates: list[tuple[int, int, int]] = []
    for left_index, left in enumerate(left_onsets):
        for right_index, right in enumerate(right_onsets):
            lag = int(right - left)
            if abs(lag) <= max_onset_lag:
                candidates.append((abs(lag), left_index, right_index))
    candidates.sort()
    used_left: set[int] = set()
    used_right: set[int] = set()
    matched_lags: list[int] = []
    for _, left_index, right_index in candidates:
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        matched_lags.append(int(right_onsets[right_index] - left_onsets[left_index]))
    return matched_lags


def _classify_episode(
    left_onsets: tuple[int, ...],
    right_onsets: tuple[int, ...],
    max_onset_lag: int,
    timing_deadband: int,
) -> _EpisodeVote | None:
    lags = _match_onsets(left_onsets, right_onsets, max_onset_lag)
    if not lags:
        return None
    decisive_forward = [lag for lag in lags if lag > timing_deadband]
    decisive_reverse = [abs(lag) for lag in lags if lag < -timing_deadband]
    tie_count = sum(abs(lag) <= timing_deadband for lag in lags)
    if tie_count or (decisive_forward and decisive_reverse):
        return _EpisodeVote(label="tie", lag_summary=None)
    if decisive_forward:
        return _EpisodeVote(
            label="forward",
            lag_summary=float(np.median(np.asarray(decisive_forward, dtype=float))),
        )
    if decisive_reverse:
        return _EpisodeVote(
            label="reverse",
            lag_summary=float(np.median(np.asarray(decisive_reverse, dtype=float))),
        )
    return _EpisodeVote(label="tie", lag_summary=None)


def build_temporal_prior(
    rise_representation: np.ndarray,
    segment_ids: np.ndarray,
    *,
    min_run_samples: int,
    max_onset_lag: int,
    timing_deadband: int,
    minimum_decisive_support: int,
    consistency_threshold: float,
    beta_prior_concentration: float,
) -> TemporalPriorResult:
    """Build an episode-aware temporal prior from segmented rising flanks."""

    rise = validate_traces(rise_representation)
    if np.any(rise < 0.0):
        raise ValueError("rise_representation cannot contain negative values")
    min_run_samples = _validate_nonnegative_integer(
        "min_run_samples", min_run_samples, minimum=1
    )
    max_onset_lag = _validate_nonnegative_integer("max_onset_lag", max_onset_lag)
    timing_deadband = _validate_nonnegative_integer("timing_deadband", timing_deadband)
    minimum_decisive_support = _validate_nonnegative_integer(
        "minimum_decisive_support",
        minimum_decisive_support,
        minimum=1,
    )
    threshold = float(consistency_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("consistency_threshold must lie in [0, 1]")
    beta = float(beta_prior_concentration)
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta_prior_concentration must be positive and finite")

    segments = _validate_segment_ids(segment_ids, rise.shape[1])
    onsets_by_roi = _extract_episode_onsets(rise, segments, min_run_samples)
    n_rois = rise.shape[0]

    hard_mask = np.zeros((n_rois, n_rois), dtype=bool)
    weights = np.ones((n_rois, n_rois), dtype=float)
    decisive = np.zeros((n_rois, n_rois), dtype=int)
    matched = np.zeros((n_rois, n_rois), dtype=int)
    ties = np.zeros((n_rois, n_rois), dtype=int)
    consistency = np.zeros((n_rois, n_rois), dtype=float)
    mean_lag = np.zeros((n_rois, n_rois), dtype=float)
    median_lag = np.zeros((n_rois, n_rois), dtype=float)
    screen_episode_ids = [
        [tuple() for _ in range(n_rois)]
        for _ in range(n_rois)
    ]
    np.fill_diagonal(consistency, 1.0)

    for left in range(n_rois):
        for right in range(left + 1, n_rois):
            left_episodes = onsets_by_roi[left]
            right_episodes = onsets_by_roi[right]
            shared_episode_ids = sorted(set(left_episodes) & set(right_episodes))
            matched_ids: list[int] = []
            forward_lags: list[float] = []
            reverse_lags: list[float] = []
            tie_count = 0

            for episode_id in shared_episode_ids:
                vote = _classify_episode(
                    left_episodes[episode_id],
                    right_episodes[episode_id],
                    max_onset_lag,
                    timing_deadband,
                )
                if vote is None:
                    continue
                matched_ids.append(int(episode_id))
                if vote.label == "forward":
                    decisive[left, right] += 1
                    if vote.lag_summary is not None:
                        forward_lags.append(vote.lag_summary)
                elif vote.label == "reverse":
                    decisive[right, left] += 1
                    if vote.lag_summary is not None:
                        reverse_lags.append(vote.lag_summary)
                else:
                    tie_count += 1

            matched_count = len(matched_ids)
            matched[left, right] = matched[right, left] = matched_count
            ties[left, right] = ties[right, left] = tie_count
            screen_episode_ids[left][right] = tuple(matched_ids)
            screen_episode_ids[right][left] = tuple(matched_ids)

            forward = int(decisive[left, right])
            reverse = int(decisive[right, left])
            total_decisive = forward + reverse
            majority = max(forward, reverse)
            pair_consistency = (
                0.0 if total_decisive == 0 else float(majority / total_decisive)
            )
            consistency[left, right] = consistency[right, left] = pair_consistency

            if forward_lags:
                mean_lag[left, right] = float(np.mean(forward_lags))
                median_lag[left, right] = float(np.median(forward_lags))
            if reverse_lags:
                mean_lag[right, left] = float(np.mean(reverse_lags))
                median_lag[right, left] = float(np.median(reverse_lags))

            posterior_forward = (beta + forward) / (2.0 * beta + total_decisive)
            posterior_reverse = (beta + reverse) / (2.0 * beta + total_decisive)
            weights[left, right] = 2.0 * posterior_forward
            weights[right, left] = 2.0 * posterior_reverse

            if matched_count == 0:
                continue
            if (
                majority < minimum_decisive_support
                or forward == reverse
                or pair_consistency < threshold
            ):
                hard_mask[left, right] = True
                hard_mask[right, left] = True
                continue
            if forward > reverse:
                hard_mask[left, right] = True
            else:
                hard_mask[right, left] = True

    return TemporalPriorResult(
        robust_hard_mask=hard_mask,
        hypothesis_weights=weights,
        decisive_episode_counts=decisive,
        matched_episode_counts=matched,
        tie_episode_counts=ties,
        direction_consistency=consistency,
        mean_decisive_lag=mean_lag,
        median_decisive_lag=median_lag,
        screen_episode_ids=tuple(
            tuple(tuple(episode_ids) for episode_ids in row)
            for row in screen_episode_ids
        ),
    )
