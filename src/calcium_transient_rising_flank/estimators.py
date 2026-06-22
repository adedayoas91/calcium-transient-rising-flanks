"""Typed result adapter for the supplied rising-flank c-GC implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys

import numpy as np

from core.rising_flanks import RisingFlanks

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

    @property
    def retained_scores(self) -> np.ndarray:
        return np.where(self.adjacency, self.scores, 0.0)


def selected_frame_indices(representation: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return nonzero frame indices for each ROI-specific flank signal."""

    values = validate_traces(representation)
    return tuple(np.flatnonzero(row > 0.0) for row in values)


def benjamini_hochberg(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """Return discoveries after Benjamini-Hochberg FDR control."""

    values = np.asarray(p_values, dtype=float)
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    eligible = np.isfinite(values)
    if values.ndim == 2 and values.shape[0] == values.shape[1]:
        eligible &= ~np.eye(values.shape[0], dtype=bool)
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


class CausalGranger:
    """Adapt ``core.rising_flanks.RisingFlanks`` to typed graph outputs.

    This class does not implement Granger causality. It passes raw traces and
    optionally selected rising/falling frame indices to the supplied core
    method, then reduces its lag blocks to the public ``GraphResult`` shape.
    """

    def __init__(
        self,
        max_lag: int = 1,
        n_surrogates: int = 0,
        alpha: float = 0.05,
        random_state: int | None = None,
        fdr: bool = True,
        score_threshold: float = 0.0,
        sampling_frequency: float = 1.0,
        segment_length: int = 0,
        event_mode: str = "compressed",
        engine: str = "rising_flanks",
        gcstar_method: str = "cgc",
        beta: float | None = None,
        gcstar_simulation: bool = True,
    ) -> None:
        if max_lag < 1:
            raise ValueError("max_lag must be positive")
        if n_surrogates < 0:
            raise ValueError("n_surrogates cannot be negative")
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        self.max_lag = max_lag
        self.n_surrogates = n_surrogates
        self.alpha = alpha
        self.random_state = random_state
        self.fdr = fdr
        self.score_threshold = score_threshold
        self.sampling_frequency = sampling_frequency
        self.segment_length = segment_length
        if event_mode not in {"compressed", "physical"}:
            raise ValueError("event_mode must be 'compressed' or 'physical'")
        self.event_mode = event_mode
        if engine not in {"rising_flanks", "gcstar"}:
            raise ValueError("engine must be 'rising_flanks' or 'gcstar'")
        self.engine = engine
        self.gcstar_method = gcstar_method
        self.beta = alpha if beta is None else beta
        if not 0 < self.beta < 1:
            raise ValueError("beta must lie in (0, 1)")
        self.gcstar_simulation = gcstar_simulation

    @staticmethod
    def _indices(
        data: np.ndarray,
        event_indices: Sequence[np.ndarray] | None,
    ) -> tuple[np.ndarray, ...]:
        if event_indices is None:
            all_frames = np.arange(data.shape[1], dtype=int)
            return tuple(all_frames.copy() for _ in range(data.shape[0]))
        return tuple(np.asarray(values, dtype=int) for values in event_indices)

    def _fit_core(
        self,
        data: np.ndarray,
        event_indices: tuple[np.ndarray, ...],
    ) -> RisingFlanks:
        core = RisingFlanks(
            n_perm=self.n_surrogates,
            n_pasts=self.max_lag,
            n_lags=self.max_lag,
            f_s=self.sampling_frequency,
            seg_len=self.segment_length,
        )
        if self.random_state is None:
            return core.fit_rising(data, event_indices, verbose=0)

        random_state = np.random.get_state()
        try:
            np.random.seed(self.random_state)
            return core.fit_rising(data, event_indices, verbose=0)
        finally:
            np.random.set_state(random_state)

    @staticmethod
    def _at_best_lag(values: np.ndarray, best: np.ndarray) -> np.ndarray:
        return np.take_along_axis(values, best[np.newaxis, :, :], axis=0)[0]

    @staticmethod
    def _absolute_corr(x: np.ndarray, y: np.ndarray) -> float:
        if x.size < 2 or y.size < 2 or np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        return float(np.abs(np.corrcoef(x, y)[1, 0]))

    def _perm_test(self, x: np.ndarray, y: np.ndarray) -> float:
        if self.n_surrogates <= 0 or x.size < 2 or y.size < 2:
            return 1.0
        observed = self._absolute_corr(x, y)
        if observed == 0.0:
            return 1.0
        count = 0
        for shift in np.random.randint(1, len(x), self.n_surrogates):
            if self._absolute_corr(np.roll(x, shift), y) >= observed:
                count += 1
        return count / self.n_surrogates

    @staticmethod
    def _residual(values: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        if covariates.size == 0:
            return values - np.mean(values)
        design = np.column_stack([np.ones(values.size), covariates.T])
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        return values - design @ coefficients

    def _physical_sample_times(
        self,
        event_indices: tuple[np.ndarray, ...],
        source: int,
        target: int,
        lag: int,
        n_steps: int,
    ) -> np.ndarray:
        selected_target = event_indices[target]
        selected_source = event_indices[source]
        if selected_target.size == n_steps and selected_source.size == n_steps:
            return np.arange(self.max_lag, n_steps, dtype=int)
        target_mask = np.zeros(n_steps, dtype=bool)
        source_mask = np.zeros(n_steps, dtype=bool)
        target_mask[selected_target] = True
        source_mask[selected_source] = True
        times = np.arange(self.max_lag, n_steps, dtype=int)
        return times[target_mask[times] & source_mask[times - lag]]

    def _physical_conditioning_set(
        self,
        data: np.ndarray,
        times: np.ndarray,
        source: int,
        target: int,
        lag: int,
    ) -> np.ndarray:
        rows = []
        for past_lag in range(1, self.max_lag + 1):
            rows.append(data[target, times - past_lag])
        for past_lag in range(lag + 1, self.max_lag + 1):
            rows.append(data[source, times - past_lag])
        for other in range(data.shape[0]):
            if other in {source, target}:
                continue
            for past_lag in range(lag, self.max_lag + 1):
                rows.append(data[other, times - past_lag])
        return np.vstack(rows) if rows else np.empty((0, times.size))

    def _fit_physical(
        self,
        data: np.ndarray,
        event_indices: tuple[np.ndarray, ...],
    ) -> GraphResult:
        n_nodes, n_steps = data.shape
        scores_by_lag = np.zeros((self.max_lag, n_nodes, n_nodes), dtype=float)
        p_values_by_lag = np.ones_like(scores_by_lag)
        for lag in range(1, self.max_lag + 1):
            lag_index = lag - 1
            for source in range(n_nodes):
                for target in range(n_nodes):
                    if source == target:
                        continue
                    times = self._physical_sample_times(
                        event_indices, source, target, lag, n_steps
                    )
                    if times.size < 2:
                        continue
                    x = data[source, times - lag]
                    y = data[target, times]
                    z = self._physical_conditioning_set(
                        data, times, source, target, lag
                    )
                    x_res = self._residual(x, z)
                    y_res = self._residual(y, z)
                    scores_by_lag[lag_index, source, target] = self._absolute_corr(
                        x_res, y_res
                    )
                    p_values_by_lag[lag_index, source, target] = self._perm_test(
                        x_res, y_res
                    )
        best = np.argmax(scores_by_lag, axis=0)
        scores = self._at_best_lag(scores_by_lag, best)
        p_values = self._at_best_lag(p_values_by_lag, best)
        best_lags = best + 1
        return self._graph_from_scores(scores, p_values, best_lags, "physical_time_cgc")

    @staticmethod
    def _mask_to_event_indices(
        data: np.ndarray,
        event_indices: tuple[np.ndarray, ...],
    ) -> np.ndarray:
        masked = np.zeros_like(data)
        for roi, frames in enumerate(event_indices):
            masked[roi, frames] = data[roi, frames]
        return masked

    def _fit_gcstar(
        self,
        data: np.ndarray,
        event_indices: tuple[np.ndarray, ...] | None,
    ) -> GraphResult:
        values = data if event_indices is None else self._mask_to_event_indices(
            data, event_indices
        )
        GcStar = _load_gcstar_class()
        core = GcStar(
            n_perm=self.n_surrogates,
            n_pasts=self.max_lag,
            n_lags=self.max_lag,
            temporal=True,
            method=self.gcstar_method,
        )
        core.fit(values, verbose=0)
        retained = core.get_connectivity_matrix(
            simulation=self.gcstar_simulation,
            alpha=self.alpha,
            beta=self.beta,
        )
        retained = np.nan_to_num(retained, nan=0.0)
        n_nodes = values.shape[0]
        lag_slice = slice(n_nodes, (self.max_lag + 1) * n_nodes)
        scores_by_lag = np.nan_to_num(
            core.inv_corr_[lag_slice].reshape(self.max_lag, n_nodes, n_nodes),
            nan=0.0,
        )
        p_values_by_lag = np.nan_to_num(
            np.maximum(
                core.pVal_corr_[lag_slice],
                core.pVal_inv_corr_[lag_slice],
            ).reshape(self.max_lag, n_nodes, n_nodes),
            nan=1.0,
        )
        best = np.argmax(scores_by_lag, axis=0)
        scores = self._at_best_lag(scores_by_lag, best)
        p_values = self._at_best_lag(p_values_by_lag, best)
        best_lags = best + 1
        adjacency = np.asarray(retained > 0.0, dtype=bool)
        np.fill_diagonal(adjacency, False)
        np.fill_diagonal(best_lags, 0)
        method_label = "cgc_star" if core.method == "fcgc" else "cgc"
        return GraphResult(scores, p_values, adjacency, best_lags, method_label)

    def _graph_from_scores(
        self,
        scores: np.ndarray,
        p_values: np.ndarray,
        best_lags: np.ndarray,
        estimator: str,
    ) -> GraphResult:
        if self.n_surrogates:
            adjacency = (
                benjamini_hochberg(p_values, self.alpha)
                if self.fdr
                else p_values <= self.alpha
            )
        else:
            adjacency = scores > self.score_threshold
        adjacency = np.asarray(adjacency, dtype=bool)
        np.fill_diagonal(adjacency, False)
        np.fill_diagonal(best_lags, 0)
        return GraphResult(scores, p_values, adjacency, best_lags, estimator)

    def fit(
        self,
        traces: np.ndarray,
        segment_ids: np.ndarray | None = None,
        outcomes: np.ndarray | None = None,
        *,
        event_indices: Sequence[np.ndarray] | None = None,
    ) -> GraphResult:
        """Fit by invoking the supplied core implementation.

        ``segment_ids`` and ``outcomes`` are retained only to provide explicit
        errors for formerly advertised extension paths that the supplied
        implementation does not expose.
        """

        if segment_ids is not None:
            raise NotImplementedError(
                "segment-aware GC is not exposed by core.rising_flanks.RisingFlanks"
            )
        if outcomes is not None:
            raise NotImplementedError(
                "cross-representation GC is not exposed by core.rising_flanks.RisingFlanks"
            )

        data = validate_traces(traces)
        indices = self._indices(data, event_indices)
        if self.engine == "gcstar":
            if self.random_state is None:
                return self._fit_gcstar(data, None if event_indices is None else indices)
            random_state = np.random.get_state()
            try:
                np.random.seed(self.random_state)
                return self._fit_gcstar(data, None if event_indices is None else indices)
            finally:
                np.random.set_state(random_state)

        if self.event_mode == "physical":
            if self.random_state is None:
                return self._fit_physical(data, indices)
            random_state = np.random.get_state()
            try:
                np.random.seed(self.random_state)
                return self._fit_physical(data, indices)
            finally:
                np.random.set_state(random_state)

        core = self._fit_core(data, indices)
        n_nodes = data.shape[0]
        lag_slice = slice(n_nodes, (self.max_lag + 1) * n_nodes)
        scores_by_lag = core.inv_corr_[lag_slice].reshape(
            self.max_lag, n_nodes, n_nodes
        )
        p_values_by_lag = np.maximum(
            core.pVal_corr_[lag_slice],
            core.pVal_inv_corr_[lag_slice],
        ).reshape(self.max_lag, n_nodes, n_nodes)
        best = np.argmax(scores_by_lag, axis=0)
        scores = self._at_best_lag(scores_by_lag, best)
        p_values = self._at_best_lag(p_values_by_lag, best)
        best_lags = best + 1

        return self._graph_from_scores(scores, p_values, best_lags, "rising_flanks_cgc")
