import tempfile
import unittest
from pathlib import Path

import numpy as np

from calcium_transient_rising_flank.comparison import (
    add_paired_deltas,
    graph_summary,
    summarize_adjacency_cache,
    write_summary_csv,
)


class ComparisonTests(unittest.TestCase):
    def test_graph_summary_reports_weighted_support(self) -> None:
        matrix = np.zeros((4, 4))
        matrix[0, 1] = 2.0
        matrix[2, 3] = 1.0
        matrix[0, 2] = 0.5

        summary = graph_summary(matrix, mid=2)

        self.assertAlmostEqual(summary["ipsilateral_weight"], 3.0)
        self.assertAlmostEqual(summary["contralateral_weight"], 0.5)
        self.assertEqual(summary["retained_edges"], 3)
        self.assertGreater(summary["w_ic"], 0.5)

    def test_cache_summary_adds_rise_minus_fall_deltas(self) -> None:
        rise = np.zeros((4, 4))
        rise[0, 1] = rise[2, 3] = 1.0
        fall = rise.copy()
        fall[0, 2] = 1.0
        cache = {
            "cases": {
                "A": {
                    "description": "test",
                    "middle": {(3, 1): 2},
                    "graphs": {
                        (3, 1): {
                            "cgc__rise": {"adjacency": rise},
                            "cgc__fall": {"adjacency": fall},
                        }
                    },
                }
            }
        }

        rows = add_paired_deltas(summarize_adjacency_cache(cache))
        rise_row = next(row for row in rows if row["representation"] == "rise")

        self.assertEqual(len(rows), 2)
        self.assertEqual(rise_row["method"], "cgc")
        self.assertGreater(rise_row["delta_w_ic_rise_minus_fall"], 0.0)

    def test_summary_rows_can_be_written_to_csv(self) -> None:
        rows = [{"case": "A", "w_ic": 1.0}, {"case": "B", "w_rc": 0.5}]

        with tempfile.TemporaryDirectory() as tmp:
            path = write_summary_csv(rows, Path(tmp) / "summary.csv")
            content = path.read_text()

        self.assertIn("case", content)
        self.assertIn("w_ic", content)
        self.assertIn("w_rc", content)


if __name__ == "__main__":
    unittest.main()
