"""Render paired rising/falling c-GC matrices on anatomical topography.

Example:

    PYTHONPATH=src python examples/topographic_pair.py \
        --rise results/rise.npy --fall results/fall.npy \
        --centroids data/fish3_trace2_centroids.npy \
        --background data/fish3_trace2_background.npy \
        --mid 6 --output figures/rise_fall_topography.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from calcium_transient_rising_flank import plot_topographic_pair


def _parse_crop(text: str) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in text.split(","))
    if len(values) != 4:
        raise argparse.ArgumentTypeError("crop must be xmin,xmax,ymin,ymax")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay paired c-GC score matrices on cell topography."
    )
    parser.add_argument("--rise", required=True, type=Path, help="Rise matrix .npy file.")
    parser.add_argument("--fall", required=True, type=Path, help="Fall matrix .npy file.")
    parser.add_argument(
        "--centroids", required=True, type=Path, help="Centroid array .npy file."
    )
    parser.add_argument(
        "--background", required=True, type=Path, help="Background image array .npy file."
    )
    parser.add_argument("--mid", type=int, default=None, help="Left/right node split.")
    parser.add_argument("--invert", action="store_true", help="Mirror topography in x.")
    parser.add_argument(
        "--crop",
        type=_parse_crop,
        default=None,
        help="Displayed extent as xmin,xmax,ymin,ymax.",
    )
    parser.add_argument(
        "--crimson-edges",
        action="store_true",
        help="Use the reference topographic crimson edge overlay instead of bilateral edge colors.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output figure file.")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    rise = np.load(args.rise)
    fall = np.load(args.fall)
    centers = np.load(args.centroids)
    background = np.load(args.background)
    fig, _ = plot_topographic_pair(
        rise,
        fall,
        centers,
        background,
        mid=args.mid,
        invert=args.invert,
        edge_color="crimson" if args.crimson_edges else None,
        crop=args.crop,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
