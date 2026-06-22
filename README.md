# Calcium Transient Rising-Flank Pipeline

Python implementation of the event-aware analysis pipeline specified in the
companion manuscript. The package compares fixed-length rising and falling
calcium-transient representations using directed structure learners and
bilateral spinal-circuit summary metrics.

## Status

This repository now provides a runnable, tested reference implementation for:

- preprocessing scenarios A-D: retained outliers/artefacts, bad-ROI removal,
  artefact correction, and smoothing;
- full, AR(1)-deconvolved, rising-flank, falling-flank, and decay-null
  residual representations;
- a typed adapter over the supplied `src/core/rising_flanks.py` c-GC
  implementation;
- `W_IC`, binary `W_IC`, paired `Delta W_IC`, and `W_RC`;
- cyclic-shift and reverse-time null controls;
- synthetic calcium-observation validation and directed-edge recovery metrics;
- prespecified transient and residual diagnostics;
- synthetic observation/sampling grids, locked calibration/evaluation splitting,
  and empirical parameter sensitivity execution;
- a latent-confounding sensitivity adapter boundary for future LPCMCI or
  SVAR-FCI integrations; and
- bilateral structure plots, anatomical topographic overlays for notebook
  results, and compatibility wrappers for the starter module names.

The supplied notebook suite is not currently reproducible as-is: although
`data/` contains the supplied Fish 3 trace, centroid, and background arrays,
the notebooks import external modules that are not in this repository.
`src/core/rising_flanks.py` is nevertheless present and is treated as the
canonical estimator source. `CausalGranger` is an adapter that passes raw
traces plus selected-frame indices to `RisingFlanks.fit_rising` and shapes its
output for the metric pipeline; it does not implement a second GC algorithm.
`src/core/causalised-GC.py` is retained as supplied supporting source and is
not needed by the current adapter path.

## Package Layout

| Module | Responsibility |
| --- | --- |
| `preprocessing.py` | Trace validation, normalization, declared exclusions, artefact interpolation, smoothing, A-D scenarios |
| `representations.py` | Full/deconvolved/rise/fall representations and decay-null residual |
| `diagnostics.py` | Pre-network decay, rise-density/duration, robust SNR, and residual checks |
| `estimators.py` | `CausalGranger` adapter over supplied `RisingFlanks`, graph-result shaping, FDR control |
| `metrics.py` | `W_IC`, `W_RC`, `Delta W_IC`, recovery and stability summaries |
| `validation.py` | Synthetic data, cyclic-shift/reverse-time controls, representation validation |
| `robustness.py` | Sampling/observation grids, locked splits, and empirical parameter sensitivity |
| `pipeline.py` | Full paired analysis across preprocessing scenarios |
| `sensitivity.py` | Adapter protocol for PAG-producing latent-confounding analyses |
| `plotting.py` | Non-mutating bilateral and anatomical-topographic plots |

## Scientific Conventions

- Trace arrays have shape `(n_rois, n_timepoints)`.
- Directed matrices are oriented `[source, target]`.
- Rising and falling representations retain the original time axis and define
  selected-frame indices passed with the raw traces to the supplied core
  method.
- This package does not implement bivariate or multivariate Granger
  causality. It also does not add segment-aware or cross-representation GC
  behavior that is absent from the supplied core API; attempts to configure
  those extensions fail explicitly.
- `W_IC` is an ipsilateral-consistency statistic, not a synaptic-recovery
  metric. If only retained arrows are meaningful, set `metric_binary=True` and
  report binary `W_IC` rather than treating significance as an edge strength.
- No claim of improvement over published GC is justified without quantitative
  rise/fall summaries, null controls, and an accompanying `W_RC` analysis.

## Quick Start

Run the unit tests without installing another test runner:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-cache XDG_CACHE_HOME=/tmp/font-cache \
  .venv/bin/python -m unittest discover -s tests -v
```

Run a synthetic end-to-end analysis:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-cache XDG_CACHE_HOME=/tmp/font-cache \
  .venv/bin/python examples/synthetic_pipeline.py
```

Run a settings-free compatibility check on the supplied Fish 3 trace:

```bash
PYTHONPATH=src .venv/bin/python examples/fish3_trace_smoke.py
```

Core usage:

```python
import numpy as np

from calcium_transient_rising_flank import AnalysisConfig, run_pipeline

config = AnalysisConfig(
    max_lag=3,
    n_surrogates=199,
    gamma=0.8,
    tolerance=0.0,
    bad_neurons=(12,),
    artifact_frames=(139,),
    smoothing_window=3,
)

result = run_pipeline(
    fluorescence,
    sides=["L", "L", "R", "R"],
    positions=[0, 1, 0, 1],
    config=config,
)

rise_wic = result.scenarios["D"].w_ic["rise"]
fall_wic = result.scenarios["D"].w_ic["fall"]
paired_delta = result.scenarios["D"].delta_w_ic
```

`n_surrogates=0` is useful for smoke execution only: it treats every positive
score above `score_threshold` as retained. Use declared surrogate/FDR settings
for empirical reporting.

Prespecified diagnostics and synthetic robustness checks are available without
selecting parameters from observed network outputs:

```python
from calcium_transient_rising_flank import (
    CausalGranger,
    SyntheticCondition,
    characterize_transients,
    run_synthetic_grid,
    split_calibration_evaluation,
)

summary = characterize_transients(fluorescence, tolerance=0.01)
runs = run_synthetic_grid(
    truth,
    conditions=[
        SyntheticCondition(name="native", gamma=0.8),
        SyntheticCondition(name="low_rate", gamma=0.8, downsample=2),
    ],
    seeds=range(20),
    estimator_factory=lambda: CausalGranger(max_lag=1),
)
calibration, locked_evaluation = split_calibration_evaluation(runs)
```

Plot a paired result directly on a supplied anatomical background:

```python
from calcium_transient_rising_flank import plot_topographic_pair

fig, axes = plot_topographic_pair(
    scenario.cgc["rise"].retained_scores,
    scenario.cgc["fall"].retained_scores,
    cell_centers,
    background,
    mid=mid,
)
```

`cell_centers` must be an `(n_rois, 2)` array in image `(x, y)` coordinates,
and `background` may be any grayscale or RGB NumPy image, including a later
hindbrain background. With `mid` set, the plot keeps the manuscript bilateral
color convention: node fill reports signed within-side drive and edges are
black within side or gray across sides. Set `edge_color="crimson"` to use the
reference anatomical information-flow overlay style. For saved arrays, run:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-cache XDG_CACHE_HOME=/tmp/font-cache \
  .venv/bin/python examples/topographic_pair.py \
  --rise results/rise.npy --fall results/fall.npy \
  --centroids data/fish3_trace2_centroids.npy \
  --background data/fish3_trace2_background.npy \
  --mid 6 --output figures/rise_fall_topography.png
```

## Starter Compatibility

The original notebook-style imports remain available as forwards to the
retained core files when `src` is on `PYTHONPATH`:

```python
from rising_flanks import RisingFlanks
from others_funcs import get_ratio_from_GC, plot_directed_graph
```

`get_ratio_from_GC(matrix, mid, ratio_type="ipsi")` maps to `W_IC`; use
`ratio_type="rostrocaudal"` for `W_RC`.
