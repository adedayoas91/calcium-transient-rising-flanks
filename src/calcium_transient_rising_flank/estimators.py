"""Typed adapters for the supplied c-GC/c-GC* implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .preprocessing import validate_traces


def _load_gcstar_class():
    module_name = "calcium_transient_rising_flank_core_causalised_gc"
    if module_name in sys.modules:
        return sys.modules[module_name].GcStar
    path = Path(__file__).resolve().parents[1] / "core" / "causalised-GC.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load GcStar from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.GcStar


@dataclass(frozen=True)
class GraphResult:
    """Directed estimator output with ``[source, target]`` orientation."""

    scores: np.ndarray
    p_values: np.ndarray
    adjacency: np.ndarray
    best_lags: np.ndarray
    estimator: str
    candidate_adjacency: np.ndarray | None = None
    hypothesis_weights: np.ndarray | None = None
    pair_diagnostics: dict[str, Any] | None = None

    @property
    def retained_scores(self) -> np.ndarray:
        return np.where(self.adjacency, self.scores, 0.0)


@dataclass(frozen=True)
class RiseFlankRunSummary:
    """Contiguous rising-flank runs retained after duration filtering."""

    kept_runs: tuple[tuple[np.ndarray, ...], ...]
    dropped_runs: tuple[tuple[np.ndarray, ...], ...]
    event_indices: tuple[np.ndarray, ...]
    min_run_samples: int

    @property
    def kept_counts(self) -> np.ndarray:
        return np.asarray([len(runs) for runs in self.kept_runs], dtype=int)

    @property
    def dropped_counts(self) -> np.ndarray:
        return np.asarray([len(runs) for runs in self.dropped_runs], dtype=int)

    def expanded_event_indices(
        self,
        n_steps: int,
        context_samples: int = 0,
    ) -> tuple[np.ndarray, ...]:
        """Return retained rise runs plus consecutive context frames."""

        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        if context_samples < 0:
            raise ValueError("context_samples cannot be negative")
        expanded: list[np.ndarray] = []
        for roi_runs in self.kept_runs:
            if not roi_runs:
                expanded.append(np.array([], dtype=int))
                continue
            frames = [
                np.arange(
                    max(0, int(run[0]) - context_samples),
                    min(n_steps, int(run[-1]) + context_samples + 1),
                    dtype=int,
                )
                for run in roi_runs
            ]
            expanded.append(np.unique(np.concatenate(frames)))
        return tuple(expanded)


@dataclass(frozen=True)
class RiseFlankCandidateMatch:
    """One shifted-overlap match between a retained source and target rise."""

    source: int
    target: int
    source_run: int
    target_run: int
    lag: int
    overlap_count: int
    overlap_fraction: float
    source_start: int
    target_start: int
    start_lag: int


@dataclass(frozen=True)
class RiseFlankCandidateResult:
    """Candidate directed pairs proposed by shifted rising-flank alignment."""

    runs: RiseFlankRunSummary
    matches: tuple[RiseFlankCandidateMatch, ...]
    candidate_adjacency: np.ndarray
    support_counts: np.ndarray
    best_lags: np.ndarray
    mean_overlap_fraction: np.ndarray
    event_indices: tuple[np.ndarray, ...] | None = None


def selected_frame_indices(representation: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return nonzero frame indices for each ROI-specific flank signal."""

    values = validate_traces(representation)
    return tuple(np.flatnonzero(row > 0.0) for row in values)


def _contiguous_runs(frames: np.ndarray) -> tuple[np.ndarray, ...]:
    if frames.size == 0:
        return ()
    breaks = np.flatnonzero(np.diff(frames) > 1) + 1
    return tuple(run.astype(int, copy=True) for run in np.split(frames, breaks))


def extract_rise_flank_runs(
    representation: np.ndarray,
    min_run_samples: int = 1,
) -> RiseFlankRunSummary:
    """Return list-of-lists style contiguous rising-flank runs per ROI.

    Runs shorter than ``min_run_samples`` are omitted from ``event_indices`` and
    preserved in ``dropped_runs`` for reporting.
    """

    if min_run_samples < 1:
        raise ValueError("min_run_samples must be at least 1")
    values = validate_traces(representation)
    kept_by_roi: list[tuple[np.ndarray, ...]] = []
    dropped_by_roi: list[tuple[np.ndarray, ...]] = []
    event_indices: list[np.ndarray] = []
    for row in values:
        runs = _contiguous_runs(np.flatnonzero(row > 0.0))
        kept = tuple(run for run in runs if run.size >= min_run_samples)
        dropped = tuple(run for run in runs if run.size < min_run_samples)
        kept_by_roi.append(kept)
        dropped_by_roi.append(dropped)
        if kept:
            event_indices.append(np.concatenate(kept).astype(int, copy=False))
        else:
            event_indices.append(np.array([], dtype=int))
    return RiseFlankRunSummary(
        kept_runs=tuple(kept_by_roi),
        dropped_runs=tuple(dropped_by_roi),
        event_indices=tuple(event_indices),
        min_run_samples=min_run_samples,
    )


def _normalise_rise_runs(
    runs_by_roi: Sequence[Sequence[np.ndarray]],
) -> tuple[tuple[np.ndarray, ...], ...]:
    normalised: list[tuple[np.ndarray, ...]] = []
    for roi_runs in runs_by_roi:
        converted = []
        for run in roi_runs:
            values = np.asarray(run, dtype=int)
            if values.ndim != 1:
                raise ValueError("each rise run must be one-dimensional")
            if np.any(values < 0):
                raise ValueError("rise runs cannot contain negative frame indices")
            if values.size and not np.array_equal(values, np.unique(values)):
                raise ValueError("rise runs must be sorted and unique")
            converted.append(values)
        normalised.append(tuple(converted))
    return tuple(normalised)


def match_shifted_rise_flank_runs(
    runs_by_roi: Sequence[Sequence[np.ndarray]],
    *,
    max_lag: int,
    min_lag: int = 1,
    min_overlap_samples: int = 1,
    min_overlap_fraction: float = 0.5,
) -> RiseFlankCandidateResult:
    """Match retained rise runs by shifted overlap and summarize candidates.

    A source-to-target match at lag ``l`` means the target run starts after the
    source run within the allowed lag window and ``source_run + l`` overlaps the
    target run strongly enough. This is only a candidate screen; causal testing
    is still performed by c-GC/c-GC*.
    """

    if min_lag < 0:
        raise ValueError("min_lag cannot be negative")
    if max_lag < min_lag:
        raise ValueError("max_lag must be greater than or equal to min_lag")
    if min_overlap_samples < 1:
        raise ValueError("min_overlap_samples must be at least 1")
    if not 0 < min_overlap_fraction <= 1:
        raise ValueError("min_overlap_fraction must lie in (0, 1]")

    runs = _normalise_rise_runs(runs_by_roi)
    n_nodes = len(runs)
    candidate = np.zeros((n_nodes, n_nodes), dtype=bool)
    support = np.zeros((n_nodes, n_nodes), dtype=int)
    best_lags = np.zeros((n_nodes, n_nodes), dtype=int)
    overlap_sums = np.zeros((n_nodes, n_nodes), dtype=float)
    lag_support: dict[tuple[int, int], dict[int, int]] = {}
    matches: list[RiseFlankCandidateMatch] = []

    for source, source_runs in enumerate(runs):
        for target, target_runs in enumerate(runs):
            if source == target:
                continue
            for source_index, source_run in enumerate(source_runs):
                for target_index, target_run in enumerate(target_runs):
                    if source_run.size == 0 or target_run.size == 0:
                        continue
                    best_match: RiseFlankCandidateMatch | None = None
                    best_key: tuple[float, int, int] | None = None
                    denominator = min(source_run.size, target_run.size)
                    start_lag = int(target_run[0] - source_run[0])
                    if start_lag < min_lag or start_lag > max_lag:
                        continue
                    for lag in range(min_lag, max_lag + 1):
                        shifted_source = source_run + lag
                        overlap = np.intersect1d(
                            shifted_source,
                            target_run,
                            assume_unique=True,
                        ).size
                        overlap_fraction = overlap / denominator
                        if (
                            overlap < min_overlap_samples
                            or overlap_fraction < min_overlap_fraction
                        ):
                            continue
                        key = (
                            float(overlap_fraction),
                            int(overlap),
                            -abs(lag - start_lag),
                        )
                        if best_key is None or key > best_key:
                            best_key = key
                            best_match = RiseFlankCandidateMatch(
                                source=source,
                                target=target,
                                source_run=source_index,
                                target_run=target_index,
                                lag=lag,
                                overlap_count=int(overlap),
                                overlap_fraction=float(overlap_fraction),
                                source_start=int(source_run[0]),
                                target_start=int(target_run[0]),
                                start_lag=start_lag,
                            )
                    if best_match is None:
                        continue
                    matches.append(best_match)
                    candidate[source, target] = True
                    support[source, target] += 1
                    overlap_sums[source, target] += best_match.overlap_fraction
                    pair_key = (source, target)
                    lag_counts = lag_support.setdefault(pair_key, {})
                    lag_counts[best_match.lag] = (
                        lag_counts.get(best_match.lag, 0) + 1
                    )

    for (source, target), counts in lag_support.items():
        best_lags[source, target] = max(counts, key=lambda lag: (counts[lag], -lag))
    mean_overlap = np.divide(
        overlap_sums,
        support,
        out=np.zeros_like(overlap_sums),
        where=support > 0,
    )
    empty_summary = RiseFlankRunSummary(
        kept_runs=runs,
        dropped_runs=tuple(() for _ in runs),
        event_indices=tuple(
            np.concatenate(roi_runs) if roi_runs else np.array([], dtype=int)
            for roi_runs in runs
        ),
        min_run_samples=1,
    )
    return RiseFlankCandidateResult(
        runs=empty_summary,
        matches=tuple(matches),
        candidate_adjacency=candidate,
        support_counts=support,
        best_lags=best_lags,
        mean_overlap_fraction=mean_overlap,
        event_indices=empty_summary.event_indices,
    )


def rise_flank_candidate_pairs(
    representation: np.ndarray,
    *,
    min_run_samples: int = 1,
    max_lag: int,
    min_lag: int = 1,
    min_overlap_samples: int | None = None,
    min_overlap_fraction: float = 0.5,
    context_samples: int = 0,
) -> RiseFlankCandidateResult:
    """Extract retained rise runs and propose shifted-overlap candidate pairs."""

    values = validate_traces(representation)
    if context_samples < 0:
        raise ValueError("context_samples cannot be negative")
    runs = extract_rise_flank_runs(representation, min_run_samples)
    overlap_samples = (
        max(1, int(np.ceil(min_run_samples * min_overlap_fraction)))
        if min_overlap_samples is None
        else min_overlap_samples
    )
    candidates = match_shifted_rise_flank_runs(
        runs.kept_runs,
        max_lag=max_lag,
        min_lag=min_lag,
        min_overlap_samples=overlap_samples,
        min_overlap_fraction=min_overlap_fraction,
    )
    return RiseFlankCandidateResult(
        runs=runs,
        matches=candidates.matches,
        candidate_adjacency=candidates.candidate_adjacency,
        support_counts=candidates.support_counts,
        best_lags=candidates.best_lags,
        mean_overlap_fraction=candidates.mean_overlap_fraction,
        event_indices=runs.expanded_event_indices(values.shape[1], context_samples),
    )


def benjamini_hochberg(
    p_values: np.ndarray,
    alpha: float,
    eligible_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return discoveries after Benjamini-Hochberg FDR control."""

    values = np.asarray(p_values, dtype=float)
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    eligible = np.isfinite(values)
    if values.ndim == 2 and values.shape[0] == values.shape[1]:
        eligible &= ~np.eye(values.shape[0], dtype=bool)
    if eligible_mask is not None:
        mask = np.asarray(eligible_mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("eligible_mask must match p_values")
        eligible &= mask
    flattened = values[eligible]
    discoveries = np.zeros(values.shape, dtype=bool)
    if flattened.size == 0:
        return discoveries
    ordered = np.sort(flattened)
    thresholds = alpha * (np.arange(1, ordered.size + 1) / ordered.size)
    accepted = np.flatnonzero(ordered <= thresholds)
    if accepted.size:
        discoveries[eligible] = flattened <= ordered[accepted[-1]]
    return discoveries


def weighted_benjamini_hochberg(
    p_values: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    eligible_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return weighted-BH discoveries using fixed, positive prior weights.

    Weights are normalized to mean one over the eligible hypothesis family.
    They must be learned independently of the p-values under test for the usual
    weighted-BH error-control interpretation to apply.
    """

    values = np.asarray(p_values, dtype=float)
    prior = np.asarray(weights, dtype=float)
    if values.shape != prior.shape:
        raise ValueError("weights must match p_values")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if np.any(~np.isfinite(prior)) or np.any(prior <= 0):
        raise ValueError("weights must be finite and positive")
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p_values must be finite and lie in [0, 1]")

    eligible = np.ones(values.shape, dtype=bool)
    if values.ndim == 2 and values.shape[0] == values.shape[1]:
        eligible &= ~np.eye(values.shape[0], dtype=bool)
    if eligible_mask is not None:
        mask = np.asarray(eligible_mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("eligible_mask must match p_values")
        eligible &= mask

    discoveries = np.zeros(values.shape, dtype=bool)
    hypothesis_count = int(np.count_nonzero(eligible))
    if hypothesis_count == 0:
        return discoveries
    normalized = prior[eligible] / float(np.mean(prior[eligible]))
    weighted_p = values[eligible] / normalized
    order = np.argsort(weighted_p)
    ordered = weighted_p[order]
    thresholds = alpha * (
        np.arange(1, hypothesis_count + 1, dtype=float) / hypothesis_count
    )
    accepted = np.flatnonzero(ordered <= thresholds)
    if accepted.size:
        cutoff = ordered[accepted[-1]]
        discoveries[eligible] = weighted_p <= cutoff
    return discoveries


def finite_sample_permutation_p_values(
    p_values: np.ndarray,
    n_surrogates: int,
) -> np.ndarray:
    """Apply the plus-one correction to Monte Carlo permutation p-values."""

    if n_surrogates < 1:
        raise ValueError("n_surrogates must be positive")
    values = np.asarray(p_values, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p_values must be finite and lie in [0, 1]")
    return np.minimum(
        (values * n_surrogates + 1.0) / (n_surrogates + 1.0),
        1.0,
    )


class CausalisedGC:
    """Run supplied c-GC/c-GC* and return a typed graph result.

    ``event_mode="physical"`` invokes the modified c-GC/c-GC* path in
    ``src/core/causalised-GC.py`` so selected samples keep their original frame
    lag and optional segment IDs prevent cross-transient discontinuities.
    """

    def __init__(
        self,
        max_lag: int = 1,
        n_surrogates: int = 0,
        alpha: float = 0.05,
        random_state: int | None = None,
        fdr: bool = True,
        score_threshold: float = 0.0,
        event_mode: str = "compressed",
        method: str = "cgc",
        beta: float | None = None,
        simulation: bool = True,
        tau: int | None = None,
        n_pasts: int | None = None,
        min_rise_run_samples: int = 1,
        rise_candidate_filter: bool = False,
        rise_match_min_lag: int = 1,
        rise_match_max_lag: int | None = None,
        rise_match_min_overlap_samples: int | None = None,
        rise_match_min_overlap_fraction: float = 0.5,
        rise_run_context_samples: int | None = None,
    ) -> None:
        if max_lag < 1:
            raise ValueError("max_lag must be positive")
        if tau is not None and tau < 1:
            raise ValueError("tau must be positive when provided")
        if n_surrogates < 0:
            raise ValueError("n_surrogates cannot be negative")
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        if event_mode not in {"compressed", "physical"}:
            raise ValueError("event_mode must be 'compressed' or 'physical'")
        self.tau = tau
        self.max_lag = max(max_lag, tau) if tau is not None else max_lag
        self.n_pasts = self.max_lag if n_pasts is None else n_pasts
        if self.n_pasts < self.max_lag:
            raise ValueError("n_pasts must be at least max_lag or tau")
        self.n_surrogates = n_surrogates
        self.alpha = alpha
        self.random_state = random_state
        self.fdr = fdr
        self.score_threshold = score_threshold
        self.event_mode = event_mode
        self.method = method
        self.beta = alpha if beta is None else beta
        if not 0 < self.beta < 1:
            raise ValueError("beta must lie in (0, 1)")
        self.simulation = simulation
        if min_rise_run_samples < 1:
            raise ValueError("min_rise_run_samples must be at least 1")
        if rise_match_min_lag < 0:
            raise ValueError("rise_match_min_lag cannot be negative")
        default_match_max_lag = self.tau if self.tau is not None else self.max_lag
        self.rise_match_max_lag = (
            default_match_max_lag
            if rise_match_max_lag is None
            else rise_match_max_lag
        )
        if self.rise_match_max_lag < rise_match_min_lag:
            raise ValueError(
                "rise_match_max_lag must be greater than or equal to "
                "rise_match_min_lag"
            )
        if (
            rise_match_min_overlap_samples is not None
            and rise_match_min_overlap_samples < 1
        ):
            raise ValueError("rise_match_min_overlap_samples must be at least 1")
        if not 0 < rise_match_min_overlap_fraction <= 1:
            raise ValueError("rise_match_min_overlap_fraction must lie in (0, 1]")
        self.min_rise_run_samples = min_rise_run_samples
        self.rise_candidate_filter = rise_candidate_filter
        self.rise_match_min_lag = rise_match_min_lag
        self.rise_match_min_overlap_samples = rise_match_min_overlap_samples
        self.rise_match_min_overlap_fraction = rise_match_min_overlap_fraction
        self.rise_run_context_samples = (
            self.n_pasts
            if rise_run_context_samples is None
            else rise_run_context_samples
        )
        if self.rise_run_context_samples < 0:
            raise ValueError("rise_run_context_samples cannot be negative")

    @staticmethod
    def _indices(
        data: np.ndarray,
        event_indices: Sequence[np.ndarray] | None,
    ) -> tuple[np.ndarray, ...] | None:
        if event_indices is None:
            return None
        if len(event_indices) != data.shape[0]:
            raise ValueError("event_indices must contain one entry per ROI")
        return tuple(np.asarray(values, dtype=int) for values in event_indices)

    @staticmethod
    def _mask_to_event_indices(
        data: np.ndarray,
        event_indices: tuple[np.ndarray, ...],
    ) -> np.ndarray:
        masked = np.zeros_like(data)
        for roi, frames in enumerate(event_indices):
            masked[roi, frames] = data[roi, frames]
        return masked

    @staticmethod
    def _validate_segment_ids(segment_ids: np.ndarray, n_steps: int) -> np.ndarray:
        values = np.asarray(segment_ids, dtype=int)
        if values.shape != (n_steps,):
            raise ValueError("segment_ids must contain one value per timepoint")
        return values

    @staticmethod
    def _at_best_lag(values: np.ndarray, best: np.ndarray) -> np.ndarray:
        return np.take_along_axis(values, best[np.newaxis, :, :], axis=0)[0]

    @staticmethod
    def _candidate_mask(
        data: np.ndarray,
        candidate_adjacency: np.ndarray | None,
    ) -> np.ndarray | None:
        if candidate_adjacency is None:
            return None
        mask = np.asarray(candidate_adjacency, dtype=bool)
        if mask.shape != (data.shape[0], data.shape[0]):
            raise ValueError("candidate_adjacency must be square with one row per ROI")
        mask = mask.copy()
        np.fill_diagonal(mask, False)
        return mask

    @staticmethod
    def _hypothesis_weights(
        data: np.ndarray,
        hypothesis_weights: np.ndarray | None,
    ) -> np.ndarray | None:
        if hypothesis_weights is None:
            return None
        weights = np.asarray(hypothesis_weights, dtype=float)
        if weights.shape != (data.shape[0], data.shape[0]):
            raise ValueError("hypothesis_weights must be square with one row per ROI")
        if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("hypothesis_weights must be finite and positive")
        return weights.copy()

    def rise_flank_candidates(
        self,
        representation: np.ndarray,
    ) -> RiseFlankCandidateResult:
        """Extract run-filtered rises and shifted-overlap candidate pairs."""

        return rise_flank_candidate_pairs(
            representation,
            min_run_samples=self.min_rise_run_samples,
            max_lag=self.rise_match_max_lag,
            min_lag=self.rise_match_min_lag,
            min_overlap_samples=self.rise_match_min_overlap_samples,
            min_overlap_fraction=self.rise_match_min_overlap_fraction,
            context_samples=self.rise_run_context_samples,
        )

    def _method_label(self, physical: bool) -> str:
        GcStar = _load_gcstar_class()
        core = GcStar(n_perm=0, n_pasts=1, n_lags=1, method=self.method)
        base = "cgc_star" if core.method == "fcgc" else "cgc"
        if physical:
            return f"segment_aware_{base}"
        return base

    def _core(self):
        GcStar = _load_gcstar_class()
        return GcStar(
            n_perm=self.n_surrogates,
            n_pasts=self.n_pasts,
            n_lags=self.max_lag,
            temporal=True,
            method=self.method,
        )

    def _tested_lags(self) -> np.ndarray:
        if self.tau is not None:
            return np.array([self.tau], dtype=int)
        return np.arange(1, self.max_lag + 1, dtype=int)

    def _fit_core(
        self,
        data: np.ndarray,
        event_indices: tuple[np.ndarray, ...] | None,
        segment_ids: np.ndarray | None,
        candidate_mask: np.ndarray | None,
    ):
        core = self._core()
        if self.event_mode == "physical" and (
            event_indices is not None or segment_ids is not None
        ):
            return core.fit_event_physical(
                data,
                event_indices=event_indices,
                segment_ids=segment_ids,
                verbose=0,
                eligible_pairs=candidate_mask,
            )
        if event_indices is not None:
            return core.fit_event_compressed(
                data,
                event_indices,
                verbose=0,
                eligible_pairs=candidate_mask,
            )
        values = (
            data
            if event_indices is None
            else self._mask_to_event_indices(data, event_indices)
        )
        return core.fit(values, verbose=0, eligible_pairs=candidate_mask)

    def _graph_from_core(
        self,
        core,
        *,
        physical: bool,
        candidate_adjacency: np.ndarray | None = None,
        hypothesis_weights: np.ndarray | None = None,
    ) -> GraphResult:
        n_nodes = core.data.shape[0]
        candidate_mask = self._candidate_mask(core.data, candidate_adjacency)
        weights = self._hypothesis_weights(core.data, hypothesis_weights)
        tested_lags = self._tested_lags()
        lag_rows = np.concatenate(
            [
                np.arange(lag * n_nodes, (lag + 1) * n_nodes, dtype=int)
                for lag in tested_lags
            ]
        )
        scores_by_lag = np.nan_to_num(
            core.inv_corr_[lag_rows].reshape(tested_lags.size, n_nodes, n_nodes),
            nan=0.0,
        )
        p_values_by_lag = np.nan_to_num(
            np.maximum(
                core.pVal_corr_[lag_rows],
                core.pVal_inv_corr_[lag_rows],
            ).reshape(tested_lags.size, n_nodes, n_nodes),
            nan=1.0,
        )
        best = np.argmax(scores_by_lag, axis=0)
        scores = self._at_best_lag(scores_by_lag, best)
        p_values = self._at_best_lag(p_values_by_lag, best)
        best_lags = np.take_along_axis(
            np.broadcast_to(tested_lags[:, None, None], scores_by_lag.shape),
            best[np.newaxis, :, :],
            axis=0,
        )[0]
        if self.n_surrogates:
            if self.fdr and weights is not None:
                adjacency = weighted_benjamini_hochberg(
                    p_values,
                    weights,
                    self.alpha,
                    eligible_mask=candidate_mask,
                )
            elif self.fdr:
                adjacency = benjamini_hochberg(
                    p_values,
                    self.alpha,
                    eligible_mask=candidate_mask,
                )
            else:
                adjacency = p_values <= self.alpha
        else:
            adjacency = scores > self.score_threshold
        adjacency = np.asarray(adjacency, dtype=bool)
        if candidate_mask is not None:
            adjacency &= candidate_mask
            scores = np.where(candidate_mask, scores, 0.0)
            p_values = np.where(candidate_mask, p_values, 1.0)
            best_lags = np.where(candidate_mask, best_lags, 0)
        np.fill_diagonal(adjacency, False)
        np.fill_diagonal(best_lags, 0)
        return GraphResult(
            scores=scores,
            p_values=p_values,
            adjacency=adjacency,
            best_lags=best_lags,
            estimator=self._method_label(physical),
            candidate_adjacency=candidate_mask,
            hypothesis_weights=weights,
            pair_diagnostics=(
                None
                if getattr(core, "pair_diagnostics_", None) is None
                else dict(core.pair_diagnostics_)
            ),
        )

    def fit(
        self,
        traces: np.ndarray,
        segment_ids: np.ndarray | None = None,
        outcomes: np.ndarray | None = None,
        *,
        event_indices: Sequence[np.ndarray] | None = None,
        candidate_adjacency: np.ndarray | None = None,
        hypothesis_weights: np.ndarray | None = None,
    ) -> GraphResult:
        """Fit c-GC/c-GC* on full traces or selected event frames."""

        if outcomes is not None:
            raise NotImplementedError("cross-representation GC is not implemented")
        if segment_ids is not None and self.event_mode != "physical":
            raise NotImplementedError(
                "segment-aware c-GC/c-GC* requires event_mode='physical'"
            )

        data = validate_traces(traces)
        indices = self._indices(data, event_indices)
        candidate_mask = self._candidate_mask(data, candidate_adjacency)
        core_eligible_pairs = None if candidate_adjacency is None else candidate_mask
        weights = self._hypothesis_weights(data, hypothesis_weights)
        if weights is not None and (self.n_surrogates == 0 or not self.fdr):
            raise ValueError(
                "hypothesis_weights require positive n_surrogates and fdr=True"
            )
        segments = (
            None
            if segment_ids is None
            else self._validate_segment_ids(segment_ids, data.shape[1])
        )
        physical = self.event_mode == "physical" and (
            indices is not None or segments is not None
        )
        if self.random_state is None:
            core = self._fit_core(data, indices, segments, core_eligible_pairs)
        else:
            random_state = np.random.get_state()
            try:
                np.random.seed(self.random_state)
                core = self._fit_core(data, indices, segments, core_eligible_pairs)
            finally:
                np.random.set_state(random_state)
        return self._graph_from_core(
            core,
            physical=physical,
            candidate_adjacency=candidate_mask,
            hypothesis_weights=weights,
        )
