# Calcium Transient Rising-Flank Pipeline

## Scientific Context: Calcium Imaging and Rising Flanks

This work is solely for calcium imaging data analysis.

Calcium flows into the soma of neurons, activating green fluorescence protein (GFP), which then fluoresces. A key characteristic of GFP is its **very fast rise but slow fall** in fluorescence. It is essential to understand that the pumping out of calcium from the soma does not have any contribution to the causal relation between any two neurons.

![Calcium inflow visualization showing GFP response with fast rise and slow fall characteristics](https://user-images.githubusercontent.com/47278559/209997772-998b1c87-c6ff-463b-9162-d960484d3e82.png)

This project explores only the activations of neurons and finds meaningful structures in the data by analyzing the rising flanks of calcium transients—the rapid activation phase where causal relationships are most informative.

## Overview

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
- a typed adapter over the supplied `src/core/causalised-GC.py` c-GC/c-GC*
  implementation, including compressed and physical event modes;
- `W_IC`, binary `W_IC`, paired `Delta W_IC`, and `W_RC`;
- cyclic-shift, reverse-time, event-jitter, phase-permutation, and
  cross-recording null controls;
- synthetic calcium-observation validation and directed-edge recovery metrics;
- dynamic-A episodic simulations where rise phases use episode-specific
  adjacency matrices and fall phases default to stochastic graph-independent
  noncausal controls, with passive decay available as an ablation;
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
`src/core/causalised-GC.py` is the active estimator source for c-GC and c-GC*.
`CausalisedGC` shapes that supplied implementation for the metric pipeline and
adds modified event-aware fitting paths: compressed mode follows the old
rising-flank selected-frame logic, and physical mode preserves original-frame
lags while blocking cross-segment discontinuities.

## Package Layout

| Module | Responsibility |
| --- | --- |
| `preprocessing.py` | Trace validation, normalization, declared exclusions, artefact interpolation, smoothing, A-D scenarios |
| `representations.py` | Full/deconvolved/rise/fall representations and decay-null residual |
| `diagnostics.py` | Pre-network decay, rise-density/duration, robust SNR, and residual checks |
| `estimators.py` | `CausalisedGC` adapter over supplied c-GC/c-GC*, graph-result shaping, selected-frame and segment-aware fitting |
| `metrics.py` | `W_IC`, `W_RC`, `Delta W_IC`, recovery and stability summaries |
| `validation.py` | Synthetic data, null controls, representation validation, and stability resampling |
| `robustness.py` | Sampling/observation grids, locked splits, and empirical parameter sensitivity |
| `pipeline.py` | Full paired analysis across preprocessing scenarios |
| `sensitivity.py` | Adapter protocol for PAG-producing latent-confounding analyses |
| `plotting.py` | Non-mutating bilateral and anatomical-topographic plots |

## Scientific Conventions

- Trace arrays have shape `(n_rois, n_timepoints)`.
- Directed matrices are oriented `[source, target]`.
- Rising and falling representations retain the original time axis and define
  selected-frame indices passed with the raw traces to the supplied c-GC/c-GC*
  core.
- `event_mode="compressed"` uses the legacy event-selected pseudo-time
  sequence. `event_mode="physical"` preserves physical-frame lags and accepts
  segment IDs so selected samples at `t` and `t-lag` must come from the same
  contiguous episode segment.
- This package does not implement bivariate or multivariate Granger
  causality. It also does not add cross-representation GC behavior that is
  absent from the supplied core API; attempts to configure that extension fail
  explicitly.
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

Generate dynamic-A episodic validation tables where rise phases are driven by
episode-specific adjacency matrices and fall phases have no cross-ROI
propagation. By default the script uses stochastic graph-independent fall
initialization; pass `--fall-state-mode passive_decay` to keep falls as passive
carryover from the preceding rise:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-cache XDG_CACHE_HOME=/tmp/font-cache \
  .venv/bin/python examples/dynamic_episodic_validation.py
```

For a locked run, increase the seed count, trace length, and surrogate count:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-cache XDG_CACHE_HOME=/tmp/font-cache \
  .venv/bin/python examples/dynamic_episodic_validation.py \
  --output-dir outputs/validation_results/dynamic_episodic_locked \
  --n-seeds 20 --n-steps 1500 --n-surrogates 1000 \
  --fall-state-mode stochastic_independent \
  --methods cgc,cgc-star
```

The script writes `dynamic_grid_runs.csv`, `representation_summary.csv`,
`rise_fall_contrasts.csv`, and `summary.json`. For dynamic-A rows, rise/full
representations are scored against the union of active episode adjacencies,
while fall and fall-residual representations are scored against zero truth.
The raw rows, summaries, and contrasts include the selected method label so
c-GC and c-GC* locked runs can be kept separate. Use `--method cgc` for a
single-method run or `--methods cgc,cgc-star` for one combined output set.

Run all remaining publication-gate jobs and then refresh the readiness,
manuscript evidence, and TODO completion packages:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-cache XDG_CACHE_HOME=/tmp/font-cache \
  .venv/bin/python examples/run_publication_gate_pipeline.py
```

This command is a dry run by default and prints the exact heavy commands. Add
`--execute` to run them. Use `--skip-dynamic`, `--skip-null`, or
`--skip-stability` to rerun only the remaining gates plus the final reports.

Summarize recall-versus-false-positive tradeoffs from saved validation CSVs:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-cache XDG_CACHE_HOME=/tmp/font-cache \
  .venv/bin/python examples/analyze_false_positive_tradeoffs.py
```

This writes `outputs/validation_tradeoffs/representation_tradeoff_summary.csv`,
`outputs/validation_tradeoffs/rise_tradeoff_contrasts.csv`, and
`outputs/validation_tradeoffs/summary.json`.

Build Chen-style comparison tables from saved weighted adjacency artifacts:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-cache XDG_CACHE_HOME=/tmp/font-cache \
  .venv/bin/python examples/build_chen_comparison_table.py
```

This writes `outputs/chen_comparison/chen_comparison_rows.csv`,
`outputs/chen_comparison/chen_comparison_summary.csv`, and
`outputs/chen_comparison/summary.json`.

If direct Chen-style BVGC/MVGC weighted matrices are supplied, list them in
`outputs/chen_direct_matrices/manifest.csv` and rerun the same command. The
manifest columns are:

```text
path,method,case,recording,fish,trial,fluo_type,representation,mid,binary,source_note
```

`path` is resolved relative to the manifest, `method` must identify BVGC or
MVGC, `mid` gives the left/right split used for `W_IC` and `W_RC`, and
`binary` controls whether nonzero entries are summarized as binary retained
edges. These optional rows are labeled as direct Chen BVGC/MVGC matrix
reproduction in downstream evidence outputs.

Compute paired rise-minus-fall empirical tests from the Chen-style table:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-cache XDG_CACHE_HOME=/tmp/font-cache \
  .venv/bin/python examples/analyze_empirical_pairing.py
```

This writes `outputs/empirical_stats/rise_fall_recording_deltas.csv`,
`outputs/empirical_stats/paired_signflip_tests.csv`, and
`outputs/empirical_stats/summary.json`.

Generate empirical null-control tables for the saved motoneuron A-D cases:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-cache XDG_CACHE_HOME=/tmp/font-cache \
  .venv/bin/python examples/run_empirical_null_controls.py \
  --methods cgc,cgc-star
```

This writes `outputs/empirical_null_controls/null_control_rows.csv`,
`outputs/empirical_null_controls/null_control_summary.csv`,
`outputs/empirical_null_controls/null_control_contrasts.csv`, and
`outputs/empirical_null_controls/summary.json`. The contrast table reports
observed-minus-null means and empirical upper-tail null p-values for
`W_IC`, `W_RC`, edge density, retained edges, and total retained weight.
Use `--method cgc` for a single-method run.

Summarize graph support, rise/fall overlap, and saved-artifact stability:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-cache XDG_CACHE_HOME=/tmp/font-cache \
  .venv/bin/python examples/analyze_graph_stability.py
```

This writes `outputs/graph_stability/graph_support_rows.csv`,
`outputs/graph_stability/representation_overlaps.csv`,
`outputs/graph_stability/graph_stability_summary.csv`, and
`outputs/graph_stability/summary.json`.

Generate empirical stability tables by re-estimating graphs under event
bootstrap, time-window, leave-one-neuron, and leave-one-transient resampling:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-cache XDG_CACHE_HOME=/tmp/font-cache \
  .venv/bin/python examples/run_empirical_stability.py \
  --methods cgc,cgc-star
```

This writes `outputs/empirical_stability/stability_rows.csv`,
`outputs/empirical_stability/stability_summary.csv`, and
`outputs/empirical_stability/summary.json`. Rows that cannot be estimated
because a representation has too few windows, neurons, or transient segments
are retained with `status="skipped"` and a skip reason. Use `--method cgc`
for a single-method run.

Build a file-based readiness report without running simulations:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-cache XDG_CACHE_HOME=/tmp/font-cache \
  .venv/bin/python examples/build_result_readiness_report.py
```

This writes `outputs/result_readiness/result_readiness_rows.csv`,
`outputs/result_readiness/result_readiness_report.md`, and
`outputs/result_readiness/summary.json`, flagging missing user-run publication
gates separately from already saved support artifacts. Locked dynamic-A
readiness requires `summary.json`, `representation_summary.csv`, and
`rise_fall_contrasts.csv` from the locked output directory.

Assemble manuscript-facing evidence tables and current figure drafts from saved
outputs:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-cache XDG_CACHE_HOME=/tmp/font-cache \
  .venv/bin/python examples/build_manuscript_evidence_package.py
```

This writes interpretation CSVs, `manuscript_evidence_report.md`, a figure
manifest, and current saved-table figures under
`outputs/manuscript_evidence/`, including empirical metric, Chen-comparison,
synthetic-tradeoff, representative Case C rise/fall network, graph-support,
and evidence-gate draft panels. When user-run locked dynamic-A, empirical
null-control, or empirical stability artifacts exist, the same command also
writes dynamic-A, null-control, and stability interpretation tables and draft
panels. The package also writes `manuscript_results_snippets.tex`, a
saved-evidence-only LaTeX snippet that reports missing gates now and switches
to result-summary prose after the user-run artifacts are present. Final figure
assembly support is written to `final_figure_plan.csv` and
`manuscript_figure_layout.tex`; when enough draft PNGs exist, the packager also
builds `main_results_overview.png` as a composite starting point for final
styling. It is a packaging step only; it does not run simulations or
re-estimate empirical graphs.

Audit the remaining `todo.md` user-run gates against saved outputs:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-cache XDG_CACHE_HOME=/tmp/font-cache \
  .venv/bin/python examples/build_todo_completion_audit.py
```

This writes `outputs/todo_completion/todo_completion_rows.csv`,
`outputs/todo_completion/todo_completion_audit.md`, and
`outputs/todo_completion/summary.json`. It is also file-based only: it checks
whether the locked dynamic-A, null-control, stability, manuscript-evidence,
and final-figure plan artifacts exist and contain data, but it does not
generate those artifacts.

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
    CausalisedGC,
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
    estimator_factory=lambda: CausalisedGC(max_lag=1),
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
