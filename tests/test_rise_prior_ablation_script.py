import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "rise_prior_ablation.py"
    spec = importlib.util.spec_from_file_location("rise_prior_ablation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RisePriorAblationScriptTests(unittest.TestCase):
    def test_episode_segments_only_label_selected_rise_windows(self) -> None:
        script = _load_script_module()
        dataset = SimpleNamespace(
            fluorescence=np.zeros((2, 12)),
            episodes=(
                SimpleNamespace(rise_start=1, rise_stop=3),
                SimpleNamespace(rise_start=7, rise_stop=10),
            ),
        )

        segments = script.episode_segments(dataset, (1,))

        np.testing.assert_array_equal(
            segments,
            np.array([-1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1]),
        )

    def test_density_matched_random_mask_is_exact_and_off_diagonal(self) -> None:
        script = _load_script_module()
        mask = np.array(
            [[False, True, False], [False, False, True], [True, False, False]]
        )

        sampled = script.density_matched_random_mask(mask, random_state=4)

        self.assertEqual(np.count_nonzero(sampled), 3)
        self.assertFalse(np.any(np.diag(sampled)))

    def test_all_arms_share_p_values_and_change_only_hypothesis_policy(self) -> None:
        script = _load_script_module()
        p_values = np.array(
            [
                [1.0, 0.001, 0.04],
                [0.04, 1.0, 0.001],
                [0.04, 0.04, 1.0],
            ]
        )
        robust = np.array(
            [[False, True, False], [False, False, True], [False, False, False]]
        )
        weights = np.ones((3, 3))
        weights[0, 1] = 1.8
        weights[1, 0] = 0.2

        arms = script.arm_adjacencies(
            p_values,
            alpha=0.05,
            naive_mask=robust,
            robust_mask=robust,
            hypothesis_weights=weights,
            random_mask=robust,
            oracle_mask=robust,
        )

        self.assertEqual(set(arms), set(script.ARMS))
        for arm in ("naive_hard", "robust_hard", "density_matched_random", "oracle_hard"):
            self.assertFalse(np.any(arms[arm] & ~robust))
        self.assertTrue(arms["unrestricted"][0, 1])
        self.assertTrue(arms["unrestricted"][1, 2])

    def test_crossfit_folds_are_disjoint_and_cover_all_episodes(self) -> None:
        script = _load_script_module()

        folds = script.crossfit_episode_folds(8)

        self.assertEqual(len(folds), 2)
        for screen, test in folds:
            self.assertFalse(set(screen) & set(test))
            self.assertEqual(set(screen) | set(test), set(range(8)))
            self.assertEqual({episode % 2 for episode in screen}, {0, 1})
            self.assertEqual({episode % 2 for episode in test}, {0, 1})


if __name__ == "__main__":
    unittest.main()
