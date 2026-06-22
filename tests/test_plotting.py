import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np

from calcium_transient_rising_flank.plotting import (
    bilateral_coordinates,
    plot_directed_graph,
    plot_topographic_graph,
    plot_topographic_pair,
)


class TopographicPlottingTests(unittest.TestCase):
    DATA_DIR = Path(__file__).resolve().parents[1] / "data"

    def setUp(self) -> None:
        self.background = np.arange(80 * 100, dtype=float).reshape(80, 100)
        self.centers = np.array(
            [[15.0, 20.0], [15.0, 55.0], [75.0, 20.0], [75.0, 55.0]]
        )
        self.rise = np.array(
            [
                [0.0, 0.7, 0.2, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.6],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        self.fall = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.4, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0, 0.0],
            ]
        )

    def tearDown(self) -> None:
        plt.close("all")

    def test_topographic_graph_renders_without_mutating_scores(self) -> None:
        original = self.rise.copy()
        fig, axis = plt.subplots()

        result = plot_topographic_graph(
            self.rise, self.centers, self.background, mid=2, ax=axis
        )

        self.assertIs(result, axis)
        np.testing.assert_array_equal(self.rise, original)
        self.assertEqual(len(axis.images), 1)
        self.assertEqual(len(axis.collections), 4)
        self.assertGreaterEqual(len(axis.texts), 4)
        arrow_colors = [patch.arrow_patch.get_edgecolor()[:3] for patch in axis.texts[4:]]
        self.assertIn(to_rgba("black")[:3], arrow_colors)
        self.assertIn(to_rgba("gray")[:3], arrow_colors)
        dashed_edges = [
            patch.arrow_patch
            for patch in axis.texts[4:]
            if patch.arrow_patch.get_edgecolor()[:3] == to_rgba("gray")[:3]
        ]
        self.assertTrue(all(edge.get_linestyle() == "--" for edge in dashed_edges))

    def test_topographic_graph_mirrors_coordinates_using_image_width(self) -> None:
        fig, axis = plt.subplots()

        plot_topographic_graph(
            self.rise,
            self.centers,
            self.background,
            mid=2,
            ax=axis,
            invert=True,
            show_labels=False,
        )

        displayed = axis.collections[0].get_offsets()[0]
        self.assertAlmostEqual(displayed[0], 99.0 - self.centers[0, 0])

    def test_pair_shares_inputs_and_accepts_reference_edge_color(self) -> None:
        fig, axes = plot_topographic_pair(
            self.rise,
            self.fall,
            self.centers,
            self.background,
            mid=2,
            show_legend=True,
            edge_color="crimson",
        )

        self.assertIs(axes[0].figure, fig)
        self.assertEqual(
            [axis.get_title() for axis in axes],
            ["Rising transients", "Falling transients"],
        )
        self.assertEqual([len(axis.images) for axis in axes], [1, 1])
        self.assertEqual(len(fig.legends), 1)
        self.assertIsNone(axes[0].get_legend())
        self.assertIsNone(axes[1].get_legend())
        self.assertIn(
            "net ipsilateral source",
            [text.get_text() for text in fig.legends[0].get_texts()],
        )

    def test_supplied_topography_arrays_are_compatible(self) -> None:
        background = np.load(self.DATA_DIR / "fish3_trace2_background.npy")
        centers = np.load(self.DATA_DIR / "fish3_trace2_centroids.npy")
        matrix = np.zeros((centers.shape[0], centers.shape[0]))

        fig, axes = plot_topographic_pair(
            matrix, matrix, centers, background, mid=6, show_labels=False
        )

        self.assertEqual(background.shape, (512, 512))
        self.assertEqual(centers.shape, (11, 2))
        self.assertEqual(len(axes), 2)

    def test_directed_graph_uses_abstract_bilateral_layout(self) -> None:
        matrix = np.zeros((5, 5))
        matrix[0, 1] = 0.8
        matrix[1, 3] = 0.5
        fig, axis = plt.subplots()

        plot_directed_graph(matrix, mid=3, ax=axis)
        coords = bilateral_coordinates(mid=3, total=5)

        self.assertEqual(len(axis.images), 0)
        self.assertAlmostEqual(coords[0, 0], -1.0)
        self.assertAlmostEqual(coords[1, 0], -1.1)
        self.assertAlmostEqual(coords[3, 0], 1.0)
        self.assertEqual(len(axis.collections), 5)


if __name__ == "__main__":
    unittest.main()
