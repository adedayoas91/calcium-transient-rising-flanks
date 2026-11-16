import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


CORE_PATH = Path(__file__).resolve().parents[1] / "src" / "core" / "causalised-GC.py"
SPEC = importlib.util.spec_from_file_location("test_core_causalised_gc", CORE_PATH)
assert SPEC is not None and SPEC.loader is not None
core = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = core
SPEC.loader.exec_module(core)


def direct_permutation_p_value(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int,
) -> float:
    observed = np.corrcoef(x, y)[1, 0]
    count = 0
    for _ in range(n_perm):
        shift = np.random.randint(1, x.size)
        permuted = np.corrcoef(np.roll(x, -shift), y)[1, 0]
        count += int(abs(permuted) >= abs(observed))
    return count / n_perm


class CircularPermutationTests(unittest.TestCase):
    def test_fft_implementation_matches_direct_circular_shifts(self) -> None:
        rng = np.random.default_rng(11)
        x = rng.normal(size=97)
        y = 0.4 * np.roll(x, 2) + rng.normal(size=97)

        np.random.seed(23)
        expected = direct_permutation_p_value(x, y, 500)
        np.random.seed(23)
        observed = core._perm_test_numba(x, y, 500)

        self.assertEqual(observed, expected)

    def test_constant_input_preserves_zero_exceedance_behavior(self) -> None:
        self.assertEqual(
            core._perm_test_numba(np.ones(20), np.arange(20.0), 50),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
