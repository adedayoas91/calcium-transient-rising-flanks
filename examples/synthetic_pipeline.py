"""Run an unthresholded synthetic smoke analysis of rise versus fall scores."""

from __future__ import annotations

import numpy as np

from calcium_transient_rising_flank import AnalysisConfig, run_pipeline
from calcium_transient_rising_flank.validation import simulate_calcium_dataset


def main() -> None:
    truth = np.array(
        [
            [False, True, False, False],
            [False, False, False, False],
            [False, False, False, True],
            [False, False, False, False],
        ]
    )
    data = simulate_calcium_dataset(
        truth, n_steps=600, gamma=0.8, noise_std=0.03, random_state=21
    )
    result = run_pipeline(
        data.fluorescence,
        sides=["L", "L", "R", "R"],
        positions=[0, 1, 0, 1],
        config=AnalysisConfig(
            max_lag=1,
            n_surrogates=0,
            gamma=0.8,
            smoothing_window=3,
            random_state=21,
        ),
    )
    for label, scenario in result.scenarios.items():
        rise = scenario.w_ic["rise"]
        fall = scenario.w_ic["fall"]
        print(
            f"{label} (unthresholded smoke): rise W_IC={rise.value if rise else None}, "
            f"fall W_IC={fall.value if fall else None}, "
            f"delta={scenario.delta_w_ic}"
        )


if __name__ == "__main__":
    main()
