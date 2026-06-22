"""Compatibility utilities retained from the starter notebook export."""

from __future__ import annotations

import numpy as np

from calcium_transient_rising_flank.plotting import (
    bilateral_coordinates as get_coords,
    bilateral_labels as get_label_and_color_lists,
    plot_directed_graph,
    plot_matrix as plot_gc_matrix,
    plot_topographic_graph,
    plot_topographic_pair,
)
from calcium_transient_rising_flank.metrics import w_ic, w_rc


def cross_corr(x: np.ndarray, y: np.ndarray, n_lags: int) -> np.ndarray:
    """Return absolute correlations over symmetric integer lags."""

    first = np.asarray(x, dtype=float)
    second = np.asarray(y, dtype=float)
    if first.ndim != 1 or first.shape != second.shape:
        raise ValueError("x and y must be one-dimensional arrays of equal length")
    if n_lags < 1:
        raise ValueError("n_lags must be positive")
    lags = np.arange(-n_lags + 1, n_lags)
    output = np.zeros(lags.size)
    for index, lag in enumerate(lags):
        if lag < 0:
            left, right = first[:lag], second[-lag:]
        elif lag > 0:
            left, right = first[lag:], second[:-lag]
        else:
            left, right = first, second
        output[index] = 0.0 if np.std(left) == 0 or np.std(right) == 0 else abs(
            np.corrcoef(left, right)[0, 1]
        )
    return output


def perm_test_shift(
    x: np.ndarray,
    y: np.ndarray,
    shuffle: int,
    random_state: int | None = None,
) -> float:
    """Cyclic-shift permutation p-value for absolute correlation."""

    first = np.asarray(x, dtype=float)
    second = np.asarray(y, dtype=float)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("x and y must be equally sized vectors")
    if shuffle < 1:
        raise ValueError("shuffle must be positive")
    if first.size < 5:
        raise ValueError("vectors are too short for cyclic-shift testing")
    observed = abs(np.corrcoef(first, second)[0, 1])
    rng = np.random.default_rng(random_state)
    null = [
        abs(np.corrcoef(np.roll(first, int(rng.integers(1, first.size))), second)[0, 1])
        for _ in range(shuffle)
    ]
    return float((1 + np.count_nonzero(np.asarray(null) >= observed)) / (shuffle + 1))


def cross_correlation(
    data: np.ndarray,
    n_perm: int,
    n_lags: int,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute zero-lag and maximum-lag correlations with shift p-values."""

    values = np.asarray(data, dtype=float)
    if values.ndim != 2:
        raise ValueError("data must have shape (n_variables, n_timepoints)")
    zero = np.abs(np.corrcoef(values))
    p_zero = np.ones_like(zero)
    maximum = np.zeros_like(zero)
    best_lag = np.zeros_like(zero, dtype=int)
    p_maximum = np.ones_like(zero)
    lags = np.arange(-n_lags + 1, n_lags)
    rng = np.random.default_rng(random_state)
    for first in range(values.shape[0]):
        for second in range(first, values.shape[0]):
            seed_a = int(rng.integers(0, np.iinfo(np.int32).max))
            p_zero[first, second] = p_zero[second, first] = perm_test_shift(
                values[first], values[second], n_perm, seed_a
            )
            correlations = cross_corr(values[first], values[second], n_lags)
            lag = int(lags[np.argmax(correlations)])
            maximum[first, second] = maximum[second, first] = float(
                correlations.max()
            )
            best_lag[first, second] = best_lag[second, first] = lag
            if lag < 0:
                left, right = values[first, :lag], values[second, -lag:]
            elif lag > 0:
                left, right = values[first, lag:], values[second, :-lag]
            else:
                left, right = values[first], values[second]
            seed_b = int(rng.integers(0, np.iinfo(np.int32).max))
            p_maximum[first, second] = p_maximum[second, first] = perm_test_shift(
                left, right, n_perm, seed_b
            )
    return zero, p_zero, maximum, best_lag, p_maximum


def pval_to_star(pvalue: float) -> str:
    if pvalue <= 0.0001:
        return "****"
    if pvalue <= 0.001:
        return "***"
    if pvalue <= 0.01:
        return "**"
    if pvalue <= 0.05:
        return "*"
    return "ns"


def get_ratio_from_GC(
    gc: np.ndarray,
    mid: int,
    ratio_type: str = "ipsi",
    binary: bool = False,
) -> float | None:
    """Compatibility metric for notebook calls on bilateral motor-neuron graphs.

    `ratio_type="ipsi"` computes `W_IC`. `ratio_type="rostrocaudal"` computes
    `W_RC` under the notebook labeling convention: nodes on each side are
    ordered from rostral to caudal.
    """

    matrix = np.asarray(gc)
    sides = np.array(["L"] * mid + ["R"] * (matrix.shape[0] - mid))
    if ratio_type in {"ipsi", "W_IC", "wic"}:
        return w_ic(matrix, sides, binary=binary).value
    if ratio_type in {"rostrocaudal", "W_RC", "wrc"}:
        positions = np.concatenate([np.arange(mid), np.arange(matrix.shape[0] - mid)])
        return w_rc(matrix, sides, positions, binary=binary).value
    raise ValueError("ratio_type must be 'ipsi'/'W_IC' or 'rostrocaudal'/'W_RC'")


__all__ = [
    "cross_corr",
    "cross_correlation",
    "get_coords",
    "get_label_and_color_lists",
    "get_ratio_from_GC",
    "perm_test_shift",
    "plot_directed_graph",
    "plot_gc_matrix",
    "plot_topographic_graph",
    "plot_topographic_pair",
    "pval_to_star",
]
