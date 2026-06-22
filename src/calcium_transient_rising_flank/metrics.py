"""Directed-structure metrics for bilateral spinal recordings."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class BilateralMetric:
    """A lateral or directional score with support information."""

    value: float | None
    preferred_mean: float
    alternative_mean: float
    preferred_edges: int
    alternative_edges: int
    preferred_opportunities: int
    alternative_opportunities: int

    @property
    def ipsilateral_edges(self) -> int:
        return self.preferred_edges

    @property
    def cross_side_edges(self) -> int:
        return self.alternative_edges

    @property
    def preferred_density(self) -> float:
        return self.preferred_edges / self.preferred_opportunities

    @property
    def alternative_density(self) -> float:
        return self.alternative_edges / self.alternative_opportunities


@dataclass(frozen=True)
class RecoverySummary:
    precision: float
    recall: float
    false_positive_rate: float
    orientation_accuracy: float
    true_positives: int
    false_positives: int
    false_negatives: int


def _matrix_and_sides(
    matrix: np.ndarray, sides: np.ndarray | list[str], binary: bool
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(matrix)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("matrix must be square with orientation [source, target]")
    side = np.asarray(sides)
    if side.shape != (scores.shape[0],):
        raise ValueError("sides must contain one label per node")
    if not np.isin(side, ["L", "R"]).all():
        raise ValueError("sides must contain only 'L' and 'R'")
    values = scores.astype(bool).astype(float) if binary else scores.astype(float)
    if np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("link scores must be finite and nonnegative")
    values = values.copy()
    np.fill_diagonal(values, 0.0)
    return values, side


def _ratio_metric(
    values: np.ndarray, preferred: np.ndarray, alternative: np.ndarray
) -> BilateralMetric:
    p_count = int(preferred.sum())
    a_count = int(alternative.sum())
    if p_count == 0 or a_count == 0:
        raise ValueError("both comparison pair sets must contain opportunities")
    p_mean = float(values[preferred].sum() / p_count)
    a_mean = float(values[alternative].sum() / a_count)
    denominator = p_mean + a_mean
    value = None if denominator == 0 else p_mean / denominator
    return BilateralMetric(
        value=value,
        preferred_mean=p_mean,
        alternative_mean=a_mean,
        preferred_edges=int(np.count_nonzero(values[preferred])),
        alternative_edges=int(np.count_nonzero(values[alternative])),
        preferred_opportunities=p_count,
        alternative_opportunities=a_count,
    )


def w_ic(
    matrix: np.ndarray, sides: np.ndarray | list[str], binary: bool = False
) -> BilateralMetric:
    """Compute opportunity-normalized ipsilateral consistency.

    Matrix orientation is `[source, target]`. Use `binary=True` when only
    retained oriented relations are available.
    """

    values, side = _matrix_and_sides(matrix, sides, binary)
    off_diag = ~np.eye(values.shape[0], dtype=bool)
    same_side = side[:, None] == side[None, :]
    return _ratio_metric(values, off_diag & same_side, off_diag & ~same_side)


def delta_w_ic(
    rise: np.ndarray,
    fall: np.ndarray,
    sides: np.ndarray | list[str],
    binary: bool = False,
) -> float:
    """Compute the paired rise-minus-fall ipsilateral-consistency contrast."""

    rise_value = w_ic(rise, sides, binary).value
    fall_value = w_ic(fall, sides, binary).value
    if rise_value is None or fall_value is None:
        raise ValueError("delta W_IC is undefined when either graph has no score")
    return rise_value - fall_value


def w_rc(
    matrix: np.ndarray,
    sides: np.ndarray | list[str],
    positions: np.ndarray | list[float],
    binary: bool = False,
) -> BilateralMetric:
    """Compute rostrocaudal concentration among ipsilateral relations.

    Smaller `positions` are treated as more rostral; a source-to-target edge
    with `positions[source] < positions[target]` is rostrocaudal.
    """

    values, side = _matrix_and_sides(matrix, sides, binary)
    position = np.asarray(positions, dtype=float)
    if position.shape != (values.shape[0],):
        raise ValueError("positions must contain one value per node")
    same_side = side[:, None] == side[None, :]
    forward = same_side & (position[:, None] < position[None, :])
    reverse = same_side & (position[:, None] > position[None, :])
    return _ratio_metric(values, forward, reverse)


def edge_recovery(truth: np.ndarray, recovered: np.ndarray) -> RecoverySummary:
    """Compute directed-edge recovery statistics against synthetic truth."""

    true_edges = np.asarray(truth, dtype=bool).copy()
    pred_edges = np.asarray(recovered, dtype=bool).copy()
    if true_edges.shape != pred_edges.shape or true_edges.ndim != 2:
        raise ValueError("truth and recovered must be matching square matrices")
    if true_edges.shape[0] != true_edges.shape[1]:
        raise ValueError("truth and recovered must be square")
    np.fill_diagonal(true_edges, False)
    np.fill_diagonal(pred_edges, False)
    tp = int(np.count_nonzero(true_edges & pred_edges))
    fp = int(np.count_nonzero(~true_edges & pred_edges))
    fn = int(np.count_nonzero(true_edges & ~pred_edges))
    negatives = int(np.count_nonzero(~true_edges)) - true_edges.shape[0]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / negatives if negatives else 0.0
    true_skeleton = true_edges | true_edges.T
    oriented_on_true_pairs = int(np.count_nonzero(pred_edges & true_skeleton))
    orientation = tp / oriented_on_true_pairs if oriented_on_true_pairs else 0.0
    return RecoverySummary(precision, recall, fpr, orientation, tp, fp, fn)


def graph_stability(graphs: list[np.ndarray]) -> float:
    """Mean pairwise Jaccard similarity of retained directed edges."""

    if len(graphs) < 2:
        raise ValueError("at least two graphs are required")
    adjacency = [np.asarray(graph, dtype=bool).copy() for graph in graphs]
    for graph in adjacency:
        if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
            raise ValueError("each graph must be square")
        np.fill_diagonal(graph, False)
    similarities = []
    for first, second in combinations(adjacency, 2):
        union = np.count_nonzero(first | second)
        intersection = np.count_nonzero(first & second)
        similarities.append(1.0 if union == 0 else intersection / union)
    return float(np.mean(similarities))
