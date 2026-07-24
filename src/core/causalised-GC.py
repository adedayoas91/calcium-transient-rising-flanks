"""Core c-GC/fcGC estimator implementation.

This module contains a cleaned-up version of the original ``GcStar``
implementation. It is intended to be imported both from the experiment harness
and from interactive notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional

import numpy as np

try:
    from numba import jit
except Exception:  # pragma: no cover - numba is optional at runtime
    def jit(*_args, **_kwargs):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator


SHD = None
SID = None


@jit(nopython=True)
def _perm_test_numba(x: np.ndarray, y: np.ndarray, n_perm: int) -> float:
    """Compute a circular-shift permutation p-value for correlation.

    The implementation mirrors the original estimator logic but guards against
    degenerate short series by using the largest valid shift range available.
    """

    if x.size <= 1 or y.size <= 1 or n_perm <= 0:
        return 1.0

    count = 0
    corr_obs = np.corrcoef(x, y)[1, 0]
    x_copy = x.copy()
    low = 1
    high = x.size

    for _ in range(n_perm):
        shift = np.random.randint(low, high)
        rolled = np.hstack((x_copy[shift:], x_copy[:shift]))
        corr_perm = np.corrcoef(rolled, y)[1, 0]
        if np.abs(corr_perm) >= np.abs(corr_obs):
            count += 1

    return count / n_perm


def regression_residual(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Return residuals after regressing ``z`` out of ``x``.

    Parameters
    ----------
    x:
        One-dimensional target series of shape ``(T,)``.
    z:
        Conditioning set shaped ``(n_covariates, T)``.
    """

    if z.size == 0:
        return x - np.mean(x)

    if z.ndim == 1:
        z = z[np.newaxis, :]

    design = np.column_stack([np.ones(x.size), z.T])
    coef, *_ = np.linalg.lstsq(design, x, rcond=None)
    fitted = design @ coef
    return x - fitted


@dataclass
class GcStar:
    """Granger-causality estimator from a causal-Bayesian-network viewpoint.

    Parameters
    ----------
    n_perm:
        Number of circular-shift permutations used for p-value estimation.
    n_pasts:
        Conditioning depth used to construct the shifted state.
    n_lags:
        Number of lag blocks merged into the final connectivity matrix.
    temporal:
        Reserved compatibility flag retained from the original implementation.
    method:
        Either ``"cgc"`` or the full-conditioned extension.  The extension can
        be requested as ``"fcgc"``, ``"cgc-star"``, or ``"cgc*"``.
    parallel:
        If ``True``, run unconditional and conditional dependence passes in
        separate worker threads.
    """

    n_perm: int = 1000
    n_pasts: int = 1
    n_lags: int = 1
    temporal: bool = True
    method: str = "cgc"
    parallel: bool = False
    corr_: Optional[np.ndarray] = None
    pVal_corr_: Optional[np.ndarray] = None
    inv_corr_: Optional[np.ndarray] = None
    pVal_inv_corr_: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Validate basic configuration."""

        aliases = {"cgc-star": "fcgc", "cgc*": "fcgc", "gc-star": "fcgc"}
        self.method = aliases.get(self.method, self.method)
        if self.method not in {"cgc", "fcgc"}:
            raise ValueError("method must be 'cgc', 'fcgc', 'cgc-star', or 'cgc*'.")
        if self.n_pasts < 0:
            raise ValueError("n_pasts must be non-negative.")
        if self.n_lags < 1:
            raise ValueError("n_lags must be at least 1.")
        self.logger = logging.getLogger(__name__)

    def _configure_logging(self, verbose: int) -> None:
        """Configure module logging for interactive runs."""

        level = logging.WARNING
        if verbose == 1:
            level = logging.INFO
        elif verbose >= 2:
            level = logging.DEBUG
        logging.basicConfig(level=level, force=True)

    def shift_data(self, arr: np.ndarray) -> np.ndarray:
        """Create stacked lagged views of the data.

        Parameters
        ----------
        arr:
            Array shaped ``(n_variables, T)``.
        """

        self.n_neur = arr.shape[0]
        if self.n_pasts == 0:
            return arr.copy()

        trimmed = arr[:, self.n_pasts :]
        for index in range(self.n_pasts):
            start = self.n_pasts - 1 - index
            stop = -index - 1
            trimmed = np.r_[trimmed, arr[:, start:stop]]
        return trimmed

    def get_conditioning_set(self, data: np.ndarray, i: int, j: int) -> np.ndarray:
        """Build the c-GC conditioning set for a single directed pair."""

        self.shifted_data = self.shift_data(data.copy())
        source_index = i % self.n_neur
        excluded_history = [
            source_index + lag * self.n_neur for lag in range(i // self.n_neur)
        ]
        excluded = np.r_[np.array(excluded_history, dtype=int), [i, j]]
        return np.delete(self.shifted_data, excluded, axis=0)

    def correlation_func(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute unconditional dependence and permutation p-values."""

        self.n_neur = data.shape[0]
        shifted = self.shift_data(data.copy())
        n_rows = shifted.shape[0]
        corr = np.abs(np.corrcoef(shifted))
        pvals = np.zeros((n_rows, self.n_neur))

        for i in range(n_rows):
            for j in range(self.n_neur):
                pvals[i, j] = _perm_test_numba(shifted[i, :], shifted[j, :], self.n_perm)

        return corr[:, : self.n_neur], pvals

    def inv_correlation_func(
        self,
        data: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute conditional dependence using residual correlations."""

        self.n_neur = data.shape[0]
        shifted = self.shift_data(data.copy())
        n_rows = shifted.shape[0]
        inv_corr = np.zeros((n_rows, self.n_neur))
        pvals = np.zeros((n_rows, self.n_neur))

        for i in range(n_rows):
            for j in range(self.n_neur):
                x = shifted[i]
                y = shifted[j]
                if self.method == "fcgc":
                    z = np.delete(shifted.copy(), [i, j], axis=0)
                else:
                    z = self.get_conditioning_set(data, i, j)

                x_res = regression_residual(x, z)
                y_res = regression_residual(y, z)
                inv_corr[i, j] = np.abs(np.corrcoef(x_res, y_res)[1, 0])
                pvals[i, j] = _perm_test_numba(x_res, y_res, self.n_perm)

        return inv_corr, pvals

    @staticmethod
    def _safe_abs_corr(x: np.ndarray, y: np.ndarray) -> float:
        if x.size < 2 or y.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
            return 0.0
        value = np.corrcoef(x, y)[1, 0]
        return 0.0 if not np.isfinite(value) else float(abs(value))

    def _safe_perm_test(self, x: np.ndarray, y: np.ndarray) -> float:
        if (
            self.n_perm <= 0
            or x.size < 2
            or y.size < 2
            or np.std(x) == 0.0
            or np.std(y) == 0.0
        ):
            return 1.0
        return float(_perm_test_numba(x, y, self.n_perm))

    @staticmethod
    def _normalise_event_indices(
        event_indices: tuple[np.ndarray, ...] | list[np.ndarray] | None,
        n_nodes: int,
        n_steps: int,
    ) -> tuple[np.ndarray, ...]:
        if event_indices is None:
            all_frames = np.arange(n_steps, dtype=int)
            return tuple(all_frames.copy() for _ in range(n_nodes))
        if len(event_indices) != n_nodes:
            raise ValueError("event_indices must contain one frame list per ROI.")
        normalised = []
        for frames in event_indices:
            values = np.asarray(frames, dtype=int)
            if np.any(values < 0) or np.any(values >= n_steps):
                raise ValueError("event_indices include an out-of-range frame.")
            normalised.append(np.unique(values))
        return tuple(normalised)

    @staticmethod
    def _normalise_segment_ids(
        segment_ids: np.ndarray | None,
        n_steps: int,
    ) -> np.ndarray | None:
        if segment_ids is None:
            return None
        values = np.asarray(segment_ids, dtype=int)
        if values.shape != (n_steps,):
            raise ValueError("segment_ids must contain one value per timepoint.")
        return values

    def _physical_sample_times(
        self,
        event_indices: tuple[np.ndarray, ...],
        source: int,
        target: int,
        lag: int,
        n_steps: int,
        segment_ids: np.ndarray | None,
    ) -> np.ndarray:
        target_mask = np.zeros(n_steps, dtype=bool)
        source_mask = np.zeros(n_steps, dtype=bool)
        target_mask[event_indices[target]] = True
        source_mask[event_indices[source]] = True
        times = np.arange(max(self.n_pasts, lag), n_steps, dtype=int)
        valid = target_mask[times] & source_mask[times - lag]
        if segment_ids is not None:
            valid &= segment_ids[times] >= 0
            valid &= segment_ids[times - lag] >= 0
            valid &= segment_ids[times] == segment_ids[times - lag]
        return times[valid]

    def _shifted_rows_at_times(self, data: np.ndarray, times: np.ndarray) -> np.ndarray:
        rows = [data[:, times]]
        for lag in range(1, self.n_pasts + 1):
            rows.append(data[:, times - lag])
        return np.vstack(rows)

    def _physical_conditioning_set(
        self,
        shifted: np.ndarray,
        row_index: int,
        target: int,
    ) -> np.ndarray:
        if self.method == "fcgc":
            excluded = np.array([row_index, target], dtype=int)
            return np.delete(shifted, np.unique(excluded), axis=0)

        return self._rising_flank_conditioning_set(shifted, row_index, target)

    def _rising_flank_conditioning_set(
        self,
        shifted: np.ndarray,
        row_index: int,
        target: int,
    ) -> np.ndarray:
        """Condition using the original rising-flank c-GC variable logic."""

        source = row_index % self.n_neur
        source_lag = row_index // self.n_neur
        rows: list[int] = []
        for lag in range(source_lag + 1, self.n_pasts + 1):
            rows.append(lag * self.n_neur + source)
        for lag in range(1, self.n_pasts + 1):
            rows.append(lag * self.n_neur + target)
        for other in range(self.n_neur):
            if other in {source, target}:
                continue
            for lag in range(source_lag, self.n_pasts + 1):
                rows.append(lag * self.n_neur + other)
        if not rows:
            return np.empty((0, shifted.shape[1]))
        return shifted[np.asarray(rows, dtype=int)]

    def fit_event_compressed(
        self,
        data: np.ndarray,
        event_indices: tuple[np.ndarray, ...] | list[np.ndarray],
        verbose: int = 0,
    ) -> "GcStar":
        """Fit modified c-GC/c-GC* using old selected-frame compression logic."""

        self.data = data.copy()
        self.n_neur = data.shape[0]
        self._configure_logging(verbose)
        selected = self._normalise_event_indices(
            event_indices, self.n_neur, data.shape[1]
        )
        n_rows = (self.n_pasts + 1) * self.n_neur
        corr = np.zeros((n_rows, self.n_neur), dtype=float)
        p_corr = np.ones((n_rows, self.n_neur), dtype=float)
        inv_corr = np.zeros((n_rows, self.n_neur), dtype=float)
        p_inv = np.ones((n_rows, self.n_neur), dtype=float)

        for row_index in range(n_rows):
            source = row_index % self.n_neur
            for target in range(self.n_neur):
                common = np.intersect1d(selected[source], selected[target])
                if common.size <= self.n_pasts + 1:
                    continue
                shifted = self.shift_data(data[:, common])
                if shifted.shape[1] < 2:
                    continue
                x = shifted[row_index]
                y = shifted[target]
                corr[row_index, target] = self._safe_abs_corr(x, y)
                p_corr[row_index, target] = self._safe_perm_test(x, y)
                z = self._physical_conditioning_set(shifted, row_index, target)
                x_res = regression_residual(x, z)
                y_res = regression_residual(y, z)
                inv_corr[row_index, target] = self._safe_abs_corr(x_res, y_res)
                p_inv[row_index, target] = self._safe_perm_test(x_res, y_res)

        self.corr_, self.pVal_corr_ = corr, p_corr
        self.inv_corr_, self.pVal_inv_corr_ = inv_corr, p_inv
        return self

    def physical_event_correlation_func(
        self,
        data: np.ndarray,
        event_indices: tuple[np.ndarray, ...] | list[np.ndarray] | None = None,
        segment_ids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute modified c-GC/c-GC* on valid physical event-lag samples.

        Unlike event-compressed analyses, this only evaluates source-target
        samples where target time ``t`` and source time ``t-lag`` are selected
        on the original time axis. When ``segment_ids`` is provided, both times
        must also belong to the same non-negative segment, preventing lag pairs
        from crossing transient or episode boundaries.
        """

        self.n_neur = data.shape[0]
        selected = self._normalise_event_indices(
            event_indices, self.n_neur, data.shape[1]
        )
        segments = self._normalise_segment_ids(segment_ids, data.shape[1])
        n_rows = (self.n_pasts + 1) * self.n_neur
        corr = np.zeros((n_rows, self.n_neur), dtype=float)
        p_corr = np.ones((n_rows, self.n_neur), dtype=float)
        inv_corr = np.zeros((n_rows, self.n_neur), dtype=float)
        p_inv = np.ones((n_rows, self.n_neur), dtype=float)

        for lag in range(self.n_pasts + 1):
            for source in range(self.n_neur):
                row_index = lag * self.n_neur + source
                for target in range(self.n_neur):
                    times = self._physical_sample_times(
                        selected, source, target, lag, data.shape[1], segments
                    )
                    if times.size < 2:
                        continue
                    shifted = self._shifted_rows_at_times(data, times)
                    x = shifted[row_index]
                    y = shifted[target]
                    corr[row_index, target] = self._safe_abs_corr(x, y)
                    p_corr[row_index, target] = self._safe_perm_test(x, y)
                    z = self._physical_conditioning_set(shifted, row_index, target)
                    x_res = regression_residual(x, z)
                    y_res = regression_residual(y, z)
                    inv_corr[row_index, target] = self._safe_abs_corr(x_res, y_res)
                    p_inv[row_index, target] = self._safe_perm_test(x_res, y_res)
        return corr, p_corr, inv_corr, p_inv

    def fit_event_physical(
        self,
        data: np.ndarray,
        event_indices: tuple[np.ndarray, ...] | list[np.ndarray] | None = None,
        segment_ids: np.ndarray | None = None,
        verbose: int = 0,
    ) -> "GcStar":
        """Fit modified c-GC/c-GC* while preserving physical event lags."""

        self.data = data.copy()
        self.shifted_data = self.shift_data(self.data)
        self._configure_logging(verbose)
        (
            self.corr_,
            self.pVal_corr_,
            self.inv_corr_,
            self.pVal_inv_corr_,
        ) = self.physical_event_correlation_func(
            self.data,
            event_indices=event_indices,
            segment_ids=segment_ids,
        )
        return self

    def fit(self, data: np.ndarray, verbose: int = 0) -> "GcStar":
        """Fit the estimator on an array of shape ``(n_variables, T)``."""

        self.data = data.copy()
        self.shifted_data = self.shift_data(self.data)
        self._configure_logging(verbose)

        if self.parallel:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                corr_future = executor.submit(self.correlation_func, data)
                inv_future = executor.submit(self.inv_correlation_func, data)
                self.corr_, self.pVal_corr_ = corr_future.result()
                self.inv_corr_, self.pVal_inv_corr_ = inv_future.result()
        else:
            self.corr_, self.pVal_corr_ = self.correlation_func(data)
            self.inv_corr_, self.pVal_inv_corr_ = self.inv_correlation_func(data)

        return self

    def get_connectivity_matrix(
        self,
        *,
        simulation: bool = True,
        alpha: float = 0.01,
        beta: float = 0.001,
    ) -> np.ndarray:
        """Construct a weighted connectivity matrix from significance masks."""

        if self.corr_ is None or self.inv_corr_ is None:
            raise RuntimeError("fit must be called before get_connectivity_matrix.")

        sig_corr = np.multiply(self.corr_, self.pVal_corr_ <= alpha)
        sig_inv = np.multiply(self.inv_corr_, self.pVal_inv_corr_ <= beta)
        inferred = np.logical_and(sig_corr, sig_inv)

        all_: list[np.ndarray] = []
        n_neur = inferred.shape[1]
        for lag in range(self.n_pasts + 1):
            start = lag * n_neur
            stop = (lag + 1) * n_neur
            all_.append(inferred[start:stop, 0:n_neur])

        if simulation:
            self.conn_mat = all_[1] if len(all_) > 1 else all_[0]
            if self.n_lags > 1:
                max_lag = min(self.n_lags, len(all_) - 1)
                for i in range(2, max_lag + 1):
                    self.conn_mat = np.logical_or(self.conn_mat, all_[i])
        elif self.n_lags == 1:
            self.conn_mat = np.logical_or(all_[0], all_[1]) if len(all_) > 1 else all_[0]
        elif self.n_lags > 1:
            self.conn_mat = all_[0]
            max_lag = min(self.n_lags, len(all_) - 1)
            for i in range(1, max_lag + 1):
                self.conn_mat = np.logical_or(self.conn_mat, all_[i])

        self.conn_mat = np.multiply(self.corr_[:n_neur, :], self.conn_mat)
        return self.conn_mat

    def compute_confusion_matrix(
        self,
        truth: np.ndarray,
        *,
        simulation: bool = True,
    ) -> np.ndarray:
        """Compute the confusion matrix against a ground-truth adjacency."""

        if not hasattr(self, "conn_mat"):
            raise RuntimeError("get_connectivity_matrix must be called first.")

        target = truth.T if simulation else truth
        tp = np.sum(np.logical_and(target != 0, self.conn_mat != 0))
        fn = np.sum(np.logical_and(target != 0, self.conn_mat == 0))
        fp = np.sum(np.logical_and(target == 0, self.conn_mat != 0))
        tn = np.sum(np.logical_and(target == 0, self.conn_mat == 0))
        self.confusion_matrix = np.array([[tp, fp], [fn, tn]])
        return self.confusion_matrix

    def compute_metrics(self) -> np.ndarray:
        """Return accuracy, precision, recall, FPR, BA, and F1."""

        if not hasattr(self, "confusion_matrix"):
            raise RuntimeError("compute_confusion_matrix must be called first.")

        tp, fp, fn, tn = self.confusion_matrix.flatten()
        total = tp + fp + fn + tn
        accuracy = (tp + tn) / total if total else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        balanced_accuracy = (specificity + recall) / 2
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0
        return np.array([accuracy, precision, recall, fpr, balanced_accuracy, f1])

    def compute_shd_sid(
        self,
        truth: np.ndarray,
        inferred: np.ndarray,
        *,
        simulation: bool = True,
    ) -> np.ndarray:
        """Compute SHD, importing ``cdt`` only when needed."""

        global SHD, SID
        if SHD is None:
            try:
                from cdt.metrics import SHD as _SHD, SID as _SID
            except Exception as exc:  # pragma: no cover - optional dependency
                raise ImportError("cdt.metrics is required for SHD/SID computations.") from exc
            SHD, SID = _SHD, _SID

        target = truth.T if simulation else truth
        self.shd_ = SHD(target=target, pred=inferred, double_for_anticausal=False)
        # self.sid_ = SID(target=target, pred=inferred)
        return self.shd_ #, self.sid_
