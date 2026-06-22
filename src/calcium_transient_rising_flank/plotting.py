"""Non-mutating visualizations for bilateral directed structures."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


LEFT_COLOR = (57 / 255, 87 / 255, 225 / 255)
RIGHT_COLOR = (255 / 255, 138 / 255, 0.0)
DRIVER_COLOR = "firebrick"
RECEIVER_COLOR = "navy"
BALANCED_COLOR = "#6a3d9a"
TOPOGRAPHIC_EDGE_COLOR = "crimson"


def bilateral_labels(mid: int, total: int) -> tuple[list[int], list[tuple[float, ...]]]:
    if mid < 0 or mid > total:
        raise ValueError("mid must divide the node count into left and right sides")
    labels = [2 * index + 1 for index in range(mid)]
    labels.extend(2 * index + 2 for index in range(total - mid))
    colors = [LEFT_COLOR] * mid + [RIGHT_COLOR] * (total - mid)
    return labels, colors


def bilateral_coordinates(mid: int, total: int) -> np.ndarray:
    if mid < 1 or total - mid < 1:
        raise ValueError("at least one node is required on each side")
    return np.vstack(
        [
            _semicircle_coordinates(mid, side=-1),
            _semicircle_coordinates(total - mid, side=1),
        ]
    )


def _semicircle_coordinates(n_points: int, side: int) -> np.ndarray:
    """Return the manuscript's left/right abstract motoneuron layout."""

    if n_points == 1:
        return np.array([[1.1 * side, 0.0]])
    half = n_points // 2
    inner = side / max(half, 1)
    outer = 1.0 * side
    x_values = np.linspace(inner, outer, half)
    if n_points % 2 == 1:
        x_values = np.concatenate((x_values, [1.1 * side]))
    x_values = np.concatenate((x_values, np.linspace(outer, inner, half)))
    y_values = np.linspace(1.0, -1.0, n_points)
    return np.column_stack((x_values, y_values))


def plot_matrix(
    scores: np.ndarray, mid: int, ax: plt.Axes | None = None
) -> plt.Axes:
    """Plot a directed score matrix without mutating caller data."""

    matrix = np.asarray(scores, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("scores must be square")
    displayed = matrix.copy()
    displayed[displayed == 0] = np.nan
    axis = plt.gca() if ax is None else ax
    axis.imshow(displayed, cmap="YlOrRd")
    labels, colors = bilateral_labels(mid, matrix.shape[0])
    axis.set_xticks(np.arange(matrix.shape[0]), labels=labels)
    axis.set_yticks(np.arange(matrix.shape[0]), labels=labels)
    for tick, color in zip(axis.get_xticklabels(), colors):
        tick.set_color(color)
    for tick, color in zip(axis.get_yticklabels(), colors):
        tick.set_color(color)
    axis.set_xlabel("to neuron")
    axis.set_ylabel("from neuron")
    return axis


def plot_directed_graph(
    scores: np.ndarray,
    mid: int,
    ax: plt.Axes | None = None,
    hide_digits: bool = False,
) -> plt.Axes:
    """Plot bilateral directed edges using `[source, target]` scores."""

    textsize = 22
    matrix = np.asarray(scores, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("scores must be square")
    axis = plt.gca() if ax is None else ax
    coords = bilateral_coordinates(mid, matrix.shape[0])
    ipsilateral = matrix.copy()
    ipsilateral[:mid, mid:] = 0.0
    ipsilateral[mid:, :mid] = 0.0
    drive = ipsilateral.sum(axis=1) - ipsilateral.sum(axis=0)
    maximum = np.max(np.abs(drive))
    scaled = drive if maximum == 0 else drive / maximum
    labels, colors = bilateral_labels(mid, matrix.shape[0])
    for node, center in enumerate(coords):
        color = (
            DRIVER_COLOR if scaled[node] > 0
            else RECEIVER_COLOR if scaled[node] < 0
            else BALANCED_COLOR
        )
        axis.scatter(*center, s=100 + 400 * abs(scaled[node]), c=color, zorder=2)
        if not hide_digits:
            axis.text(
                center[0] + np.sign(center[0]) * 0.4,
                center[1],
                str(labels[node]),
                color=colors[node],
                ha="center",
                va="center",
                size=textsize,
                fontweight="bold",
            )
    positive = matrix[matrix > 0]
    scale = float(positive.max()) if positive.size else 1.0
    for source, target in zip(*np.nonzero(matrix > 0)):
        if source == target:
            continue
        color = "black" if (source < mid) == (target < mid) else "gray"
        axis.annotate(
            "",
            xy=coords[target],
            xytext=coords[source],
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "lw": 0.8 + 2.2 * matrix[source, target] / scale,
                "alpha": 0.8,
            },
        )
    axis.set_aspect("equal")
    axis.axis("off")
    return axis


def _score_matrix(scores: np.ndarray) -> np.ndarray:
    matrix = np.asarray(scores, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("scores must be square")
    return np.nan_to_num(matrix, nan=0.0)


def _topographic_inputs(
    scores: np.ndarray, cell_centers: np.ndarray, background: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = _score_matrix(scores)
    centers = np.asarray(cell_centers, dtype=float)
    image = np.asarray(background)
    if centers.shape != (matrix.shape[0], 2):
        raise ValueError("cell_centers must have shape (n_nodes, 2)")
    if image.ndim not in {2, 3}:
        raise ValueError("background must be a grayscale or RGB image")
    return matrix, centers, image


def _display_centers(
    cell_centers: np.ndarray, image: np.ndarray, invert: bool
) -> np.ndarray:
    centers = cell_centers.copy()
    if invert:
        centers[:, 0] = image.shape[1] - 1 - centers[:, 0]
    return centers


def _node_drive(matrix: np.ndarray, mid: int | None) -> np.ndarray:
    displayed = matrix.copy()
    if mid is not None:
        if not 0 < mid < displayed.shape[0]:
            raise ValueError("mid must divide the node count into two nonempty sides")
        displayed[:mid, mid:] = 0.0
        displayed[mid:, :mid] = 0.0
    return displayed.sum(axis=1) - displayed.sum(axis=0)


def _show_topographic_background(axis: plt.Axes, image: np.ndarray) -> None:
    kwargs: dict[str, object] = {"cmap": plt.cm.gist_yarg, "aspect": "equal"}
    if image.ndim == 2 and image.size:
        maximum = float(np.nanmax(image))
        if maximum > 0.0:
            kwargs["vmax"] = maximum / 2.0
    axis.imshow(image, **kwargs)


def _add_topographic_legend(
    axis: plt.Axes, mid: int | None, edge_color: str | None
) -> None:
    edge_handles = (
        [
            Line2D([], [], color="black", lw=1.5, label="ipsilateral edge"),
            Line2D([], [], color="gray", lw=1.5, ls="--", label="contralateral edge"),
        ]
        if mid is not None and edge_color is None
        else [
            Line2D(
                [], [], color=edge_color or TOPOGRAPHIC_EDGE_COLOR, lw=1.5,
                label="directed edge",
            )
        ]
    )
    handles = [
        Line2D(
            [], [], marker="o", linestyle="none", color=DRIVER_COLOR,
            label="net ipsilateral source",
        ),
        Line2D(
            [], [], marker="o", linestyle="none", color=RECEIVER_COLOR,
            label="net ipsilateral receiver",
        ),
        Line2D(
            [], [], marker="o", linestyle="none", color=BALANCED_COLOR,
            label="balanced",
        ),
        *edge_handles,
    ]
    axis.figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=len(handles),
        fontsize=8,
        frameon=False,
        borderpad=0.4,
        handlelength=1.6,
    )


def plot_topographic_graph(
    scores: np.ndarray,
    cell_centers: np.ndarray,
    background: np.ndarray,
    *,
    mid: int | None = None,
    ax: plt.Axes | None = None,
    invert: bool = False,
    show_labels: bool = True,
    show_legend: bool = False,
    edge_color: str | None = None,
    edge_vmax: float | None = None,
    crop: tuple[float, float, float, float] | None = None,
) -> plt.Axes:
    """Overlay a directed c-GC score matrix on anatomical centroid positions.

    Parameters
    ----------
    scores:
        Square directed-score matrix in ``[source, target]`` orientation.
    cell_centers:
        Array of image coordinates with columns ``(x, y)`` in node order.
    background:
        Anatomical background image corresponding to ``cell_centers``.
    mid:
        Optional left/right split. When supplied, node drive and the black
        within-side/gray cross-side edge convention match the manuscript's
        bilateral graph panels. Without it, every edge is drawn in crimson as
        in the topographic information-flow routine being adapted.
    invert:
        Mirror x-coordinates for recordings whose orientation is reversed.
        Mirroring uses the supplied image width rather than a fixed canvas size.
    crop:
        Optional ``(xmin, xmax, ymin, ymax)`` displayed region.
    show_legend:
        Show a compact legend explaining signed ipsilateral node drive and
        directed edge colors.
    """

    matrix, centers, image = _topographic_inputs(scores, cell_centers, background)
    axis = plt.gca() if ax is None else ax
    positioned = _display_centers(centers, image, invert)
    drive = _node_drive(matrix, mid)
    maximum_drive = float(np.max(np.abs(drive))) if drive.size else 0.0
    scaled_drive = drive if maximum_drive == 0.0 else drive / maximum_drive

    _show_topographic_background(axis, image)
    labels, colors = (
        bilateral_labels(mid, matrix.shape[0])
        if mid is not None
        else (list(range(1, matrix.shape[0] + 1)), ["black"] * matrix.shape[0])
    )
    for node, center in enumerate(positioned):
        color = (
            DRIVER_COLOR if scaled_drive[node] > 0
            else RECEIVER_COLOR if scaled_drive[node] < 0
            else BALANCED_COLOR
        )
        axis.scatter(
            *center,
            s=85.0 + 270.0 * abs(scaled_drive[node]),
            c=color,
            edgecolors="black",
            linewidths=0.6,
            alpha=0.9,
            zorder=3,
        )
        if show_labels:
            label = axis.annotate(
                str(labels[node]),
                center,
                xytext=(8, 0),
                textcoords="offset points",
                color=colors[node],
                fontsize=9,
                fontweight="bold",
                ha="left",
                va="center",
                zorder=4,
            )
            label.set_path_effects(
                [path_effects.Stroke(linewidth=2.4, foreground="white"), path_effects.Normal()]
            )

    positive = matrix[matrix > 0.0]
    scale = (
        float(edge_vmax)
        if edge_vmax is not None and edge_vmax > 0.0
        else float(positive.max()) if positive.size else 1.0
    )
    for source, target in zip(*np.nonzero(matrix > 0.0)):
        if source == target:
            continue
        color = edge_color
        line_style = "-"
        if color is None:
            color = (
                "black" if mid is not None and (source < mid) == (target < mid)
                else "gray" if mid is not None
                else TOPOGRAPHIC_EDGE_COLOR
            )
            if mid is not None and (source < mid) != (target < mid):
                line_style = "--"
        line_width = 0.9 + 2.6 * matrix[source, target] / scale
        annotation = axis.annotate(
            "",
            xy=positioned[target],
            xytext=positioned[source],
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "lw": line_width,
                "linestyle": line_style,
                "alpha": 0.85,
                "mutation_scale": 12,
                "shrinkA": 8,
                "shrinkB": 8,
            },
            zorder=2,
        )
        annotation.arrow_patch.set_path_effects(
            [
                path_effects.Stroke(linewidth=line_width + 1.4, foreground="white"),
                path_effects.Normal(),
            ]
        )

    if crop is not None:
        xmin, xmax, ymin, ymax = crop
        axis.set_xlim(xmin, xmax)
        axis.set_ylim(ymax, ymin)
    if show_legend:
        _add_topographic_legend(axis, mid, edge_color)
    axis.axis("off")
    return axis


def plot_topographic_pair(
    rise_scores: np.ndarray,
    fall_scores: np.ndarray,
    cell_centers: np.ndarray,
    background: np.ndarray,
    *,
    mid: int | None = None,
    axes: Sequence[plt.Axes] | None = None,
    invert: bool = False,
    show_labels: bool = True,
    show_legend: bool = False,
    edge_color: str | None = None,
    crop: tuple[float, float, float, float] | None = None,
    titles: tuple[str, str] = ("Rising transients", "Falling transients"),
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot paired rising/falling c-GC outputs on the same topography."""

    rise = _score_matrix(rise_scores)
    fall = _score_matrix(fall_scores)
    if rise.shape != fall.shape:
        raise ValueError("rise_scores and fall_scores must have identical shapes")
    if axes is None:
        fig, created = plt.subplots(
            1, 2, figsize=(10, 4.8),
            gridspec_kw={"wspace": 0.02},
        )
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.14, top=0.96)
        pair = (created[0], created[1])
    else:
        if len(axes) != 2:
            raise ValueError("axes must contain exactly two matplotlib axes")
        pair = (axes[0], axes[1])
        fig = pair[0].figure
    maximum = float(np.max([rise.max(initial=0.0), fall.max(initial=0.0)]))
    for axis, matrix, title in zip(pair, (rise, fall), titles):
        plot_topographic_graph(
            matrix,
            cell_centers,
            background,
            mid=mid,
            ax=axis,
            invert=invert,
            show_labels=show_labels,
            show_legend=show_legend and axis is pair[0],
            edge_color=edge_color,
            edge_vmax=maximum if maximum > 0.0 else None,
            crop=crop,
        )
        axis.set_title(title)
    return fig, pair
