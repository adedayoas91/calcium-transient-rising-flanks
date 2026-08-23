# Notebook Execution Order

Run notebooks from the `calcium-transient-rising-flank` package root so relative paths resolve cleanly. The newer wrapper notebooks default to dry-run mode; set their `RUN_*` toggles to `True` when you want them to launch the underlying scripts.

## Reviewer-Revision Campaign (Recommended Next Run)

The reviewer-revision implementation is complete, but its scientific outputs
have intentionally not been generated on this machine. Run the campaign on the
compute machine before updating any result claim. IAAFT and bout-preserving
surrogates are excluded from this campaign by author decision.

### 1. Prepare the optional baseline environment

The core package remains lightweight. LPCMCI and OASIS are isolated in optional
extras because they are external GPL-3.0 packages and LPCMCI is an experimental
Tigramite method.

```bash
uv sync --extra pag --extra deconvolution
```

The committed `uv.lock` contains the resolved baseline dependencies. Use the
same environment for every stage so OASIS and LPCMCI versions remain matched.

### 2. Preferred one-command execution

Preview the exact commands first:

```bash
uv run --extra pag --extra deconvolution python \
  examples/run_reviewer_revision_campaign.py --resume --dry-run
```

Then launch the full resumable campaign:

```bash
uv run --extra pag --extra deconvolution python \
  examples/run_reviewer_revision_campaign.py --resume
```

The corresponding preview-first notebook is:

1. `notebooks/simulations/08_reviewer_revision_campaign_run.ipynb`

Set `RUN_CAMPAIGN = True` only after inspecting the preview. The notebook passes
`--resume`, and the campaign passes `--resume` to every potentially long child
runner. Re-executing the cell skips a stage only when its `summary.json`
contains `"status": "complete"`; partial stages continue from their checkpoints.

The campaign writes its top-level state to
`outputs/reviewer_revision_campaign/campaign_state.json` and executes these
stages in the reviewer-recommended dependency order:

| Stage | Implementation | Primary completion marker |
|---|---|---|
| 1 | Matched BH empirical re-estimation | `outputs/reviewer_revision_campaign/empirical_fdr/bh/summary.json` |
| 2 | Matched unadjusted empirical re-estimation | `outputs/reviewer_revision_campaign/empirical_fdr/unadjusted/summary.json` |
| 3 | Held-out MAD/AR-residual per-ROI thresholds plus score sweeps for recovery, `W_IC`, and `W_RC` | `outputs/reviewer_revision_campaign/threshold_calibration/summary.json` |
| 4 | Mixed noncausal/causal-fall benchmark, overlap-allowed condition, and kinetic-misspecification condition | `outputs/reviewer_revision_campaign/mixed_fall/summary.json` |
| 5 | OASIS event/preprocessing baseline and raw LPCMCI PAG baseline under controlled common input | `outputs/reviewer_revision_campaign/lpcmci_oasis/summary.json` |
| 6 | Multi-lag, run-context, and bout-bounded physical-event hybrid grids | `outputs/reviewer_revision_campaign/hybrid_event/summary.json` |
| 7 | Thresholded-increment, change-point, and kinetics-template onset comparison | `outputs/reviewer_revision_campaign/adaptive_onset/summary.json` |

The two empirical arms use the same cases, recordings, representations,
estimators, seeds, alpha level, and 1,000 estimator surrogates. The only arm
difference is BH correction versus `--no-fdr`. Each arm has 72 top-level graph
fits for cases C/D, recordings F3T1/F3T2/F5T2, rise/fall representations, and
c-GC/c-GC*. The declared edge-testing family is the eligible non-self ordered
ROI pairs within each recording/case/method/representation.

### 3. Individual resumable notebooks

Use these when a scheduler or failure requires running one family at a time:

1. `notebooks/motorneurons/Reviewer_FDR_reestimation.ipynb`
2. `notebooks/simulations/05_reviewer_calibration_onset_run.ipynb`
3. `notebooks/simulations/06_reviewer_dynamic_extensions_run.ipynb`
4. `notebooks/simulations/07_reviewer_baseline_benchmarks_run.ipynb`

Every heavy launch command in these notebooks contains `--resume`; there is no
notebook toggle that disables checkpoint reuse. The mixed-fall and hybrid grids
write to separate directories, as do threshold calibration and adaptive onset,
so later stages cannot overwrite earlier evidence.

Direct equivalents are:

```bash
# Matched BH arm
uv run python examples/run_empirical_null_controls.py \
  --cases C,D --recordings F3T1,F3T2,F5T2 \
  --representations rise,fall --methods cgc,cgc-star \
  --n-null-replicates 0 --n-estimator-surrogates 1000 \
  --alpha 0.05 --event-mode physical --seed 10 --resume \
  --output-dir outputs/reviewer_revision_campaign/empirical_fdr/bh

# Matched unadjusted arm
uv run python examples/run_empirical_null_controls.py \
  --cases C,D --recordings F3T1,F3T2,F5T2 \
  --representations rise,fall --methods cgc,cgc-star \
  --n-null-replicates 0 --n-estimator-surrogates 1000 \
  --alpha 0.05 --event-mode physical --seed 10 --no-fdr --resume \
  --output-dir outputs/reviewer_revision_campaign/empirical_fdr/unadjusted

# Held-out per-ROI threshold and W_IC/W_RC calibration
uv run python examples/reviewer_calibration_onset.py \
  --components threshold --n-estimator-surrogates 1000 --resume \
  --output-dir outputs/reviewer_revision_campaign/threshold_calibration

# Mixed fall benchmark
uv run python examples/reviewer_dynamic_extensions.py \
  --methods cgc,cgc-star --grid lag1_context1 --n-seeds 8 \
  --n-surrogates 1000 --resume \
  --output-dir outputs/reviewer_revision_campaign/mixed_fall

# OASIS and LPCMCI baselines
uv run --extra pag --extra deconvolution python \
  examples/reviewer_baseline_benchmarks.py --n-cgc-surrogates 1000 --resume \
  --output-dir outputs/reviewer_revision_campaign/lpcmci_oasis

# Hybrid physical-event grids
uv run python examples/reviewer_dynamic_extensions.py \
  --methods cgc,cgc-star \
  --grid lag2_context2,bout_bounded_lag3_context4 --n-seeds 8 \
  --n-surrogates 1000 --resume \
  --output-dir outputs/reviewer_revision_campaign/hybrid_event

# Adaptive-onset comparison
uv run python examples/reviewer_calibration_onset.py \
  --components onset --resume \
  --output-dir outputs/reviewer_revision_campaign/adaptive_onset
```

### 4. Resume and failure rules

- Rerun the identical command after an interruption. Each runner validates its
  saved configuration before accepting partial rows.
- Do not change seeds, methods, grids, thresholds, or output directories while
  resuming. A configuration mismatch exits instead of mixing incompatible rows.
- Do not delete partial CSV or progress JSON files independently. A progress
  counter is advisory: the runners reconstruct completion from the full
  expected row keys. Incomplete or duplicated units are discarded and rerun;
  fully written units are recovered even if interruption occurred before the
  progress JSON update.
- Raw LPCMCI `graph`, `p_matrix`, and `val_matrix` tensors are retained under
  `outputs/reviewer_revision_campaign/lpcmci_oasis/raw_pag/`. The binary lagged
  projection is lossy, is scored only as an undirected skeleton, and must not
  replace the PAG in reporting.
- OASIS rows are event-recovery and preprocessing-ablation evidence; OASIS is
  not reported as a causal discovery method.

### 5. Post-run verification on the compute machine

After all stage markers report complete, run the authored regression suite and
inspect the campaign state before revising the manuscript:

```bash
uv run --extra pag --extra deconvolution python -m unittest discover -s tests -v
```

Then confirm that `campaign_state.json` and every stage `summary.json` reports
`"status": "complete"`, inspect row counts against the saved configurations,
and review raw PAG mark distributions before interpreting the lossy projection.
No manuscript result should be changed from proposed/awaiting execution to
completed solely because the code or notebook exists.

## 0. Explore The Generator

1. `notebooks/simulations/00_hyperparameter_timeseries_explorer.ipynb`

This is exploratory only. Use it to tune dynamic-A simulation hyperparameters and visually inspect rise/fall episode schedules, stochastic-independent falls, event rasters, fluorescence traces, and representation transforms.

Use `RISE_WAVEFORM_LENGTH` to control how many samples each rise-phase onset is spread across. The default is 20 in the exploration and dynamic validation notebooks. Use `FALL_INITIAL_CEILING_FRACTION` to cap the stochastic fall reset against the preceding rise endpoint; keep it at `1.0` or lower to avoid upward fall-boundary spikes.

Use `TOPOLOGY_MODE = "sequence"` for the locked hand-declared `A_1, A_2, ...` rise graphs, including the active-node sequence that makes a neuron appear in later rises. Use `TOPOLOGY_MODE = "generated"` to sample per-rise edge dropout/addition and causal-source dropout/recruitment around the base union graph.

Use `TAU`, `N_PASTS`, `MIN_RISE_RUN_SAMPLES`, and the `RISE_MATCH_*` settings in this notebook to preview which sustained rising flanks enter the candidate screen and how much lag-history context is retained.

## 1. Static Synthetic Baselines

2. `notebooks/simulations/c-GC.ipynb`
3. `notebooks/simulations/c-GC-star.ipynb`

Run these when you need to regenerate the static synthetic validation outputs under `outputs/validation_results/`. They are existing notebooks and are separate from the dynamic-A scenario.

Set `TAU = None` to preserve the existing merged-`N_LAGS` behavior, or set `TAU` to a positive integer to keep only that causal lag after fitting. `N_PASTS` controls the conditioning-history depth and must be at least `TAU`.

## 2. Dynamic-A Synthetic Validation

4. `notebooks/simulations/01_dynamic_episodic_validation_run.ipynb`

This wraps `examples/dynamic_episodic_validation.py` and writes the locked dynamic-A outputs:

- `outputs/validation_results/dynamic_episodic_locked/dynamic_grid_runs.csv`
- `outputs/validation_results/dynamic_episodic_locked/representation_summary.csv`
- `outputs/validation_results/dynamic_episodic_locked/rise_fall_contrasts.csv`
- `outputs/validation_results/dynamic_episodic_locked/summary.json`

The notebook passes `--rise-waveform-length 20`, `--topology-mode sequence`, and `--fall-initial-ceiling-fraction 1.0` by default. Increase or decrease `RISE_WAVEFORM_LENGTH` before running if you want a different rise sampling density. Lower `FALL_INITIAL_CEILING_FRACTION` if you want falls to begin below the rise endpoint. Switch to `TOPOLOGY_MODE = "generated"` when you want stochastic addition/deletion of causal sources and edges instead of the locked hand-designed sequence.

Set `TAU` and `N_PASTS` in the parameter cell before running. When `TAU` is set, the notebook also defaults `RISE_MATCH_MIN_LAG` and `RISE_MATCH_MAX_LAG` to that same lag so the candidate screen and c-GC/c-GC* test the same offset.

## 3. Empirical Graph Artifacts

5. `notebooks/motorneurons/Rising_flanks_WithSections.ipynb`
6. `notebooks/motorneurons/Rising_flanks_Hindbrain.ipynb`
7. `notebooks/motorneurons/c-GC_Motoneurons.ipynb`
8. `notebooks/motorneurons/c-GC-star_Motoneurons.ipynb`
9. `notebooks/motorneurons/c-GC_Hindbrain.ipynb`
10. `notebooks/motorneurons/c-GC-star_Hindbrain.ipynb`

Run these only when the saved graph pickle artifacts under `outputs/motorneurons/` need to be rebuilt.

## 4. Saved-Artifact Summaries

11. `notebooks/simulations/02_saved_artifact_analysis_run.ipynb`

This wraps the scripts that summarize already-generated artifacts:

- `examples/build_chen_comparison_table.py`
- `examples/analyze_empirical_pairing.py`
- `examples/analyze_graph_stability.py`
- `examples/analyze_false_positive_tradeoffs.py`
- `examples/build_result_readiness_report.py`
- `examples/build_manuscript_evidence_package.py`
- `examples/build_todo_completion_audit.py`

## 5. Publication-Gate Runs

12. `notebooks/simulations/03_publication_gate_pipeline_run.ipynb`

This wraps `examples/run_publication_gate_pipeline.py` for the heavier user-run gates:

- locked dynamic-A validation
- empirical null controls
- empirical re-estimation stability
- result readiness report
- manuscript evidence package
- todo completion audit

The notebook now keeps global stage numbers in the runner output and leaves
completed heavy gates visible as skipped. By default, `SKIP_COMPLETED = True`
skips dynamic, null-control, or stability gates whose `summary.json` marker
already exists. To force a full locked rerun, set `SKIP_COMPLETED = False` and
leave the explicit skip flags as `False`.

For a manual resume after dynamic validation and empirical null controls, set
`SKIP_DYNAMIC = True`, `SKIP_NULL = True`, and `SKIP_STABILITY = False`.

Set `DYNAMIC_TAU`, `DYNAMIC_N_PASTS`, `MIN_RISE_RUN_SAMPLES`, and the `RISE_MATCH_*` settings in the parameter cell to forward those hyperparameters into the dynamic-A validation stage.

## 6. Three-State Temporal-Prior Follow-Up

13. `notebooks/simulations/04_temporal_resolvability_map.ipynb`
14. `notebooks/motorneurons/Temporal_resolvability_screen.ipynb`

Run the synthetic notebook first. It is the user-run entry point for the locked
three-state temporal-resolvability experiment and keeps manual results separate
from the existing reference run. Enable its toggles in order:

1. `RUN_SMOKE = True`
2. `RUN_LOCKED_GRID = True`
3. `RUN_STRICT_ANALYSIS = True`

The smoke run writes to
`outputs/validation_results/temporal_resolvability_map_manual_smoke/`; the
locked grid and strict analysis write to
`outputs/validation_results/temporal_resolvability_map_manual/`. The analysis
toggle requires the locked grid's `resolvability_rows.csv` and does not fall
back to results produced by another run.

The motoneuron screen was completed on 2026-08-22 for case D and all nine
recordings. The estimable fixed-window analyses produced no joint diagnostic
passes: 0/54 cells at 120 frames and 0/54 at the primary 240-frame setting.
The 480-frame setting was not estimable because each recording provided only
three non-overlapping windows, fewer than the four required for two-way
cross-fitting. The population-bout sensitivity produced 2/54 passes (F1T2 and
F6T2 at lag 1, deadband 0), but neither pass reproduced under fixed windows and
neither recording is in the retained primary subset F3T1/F3T2/F5T2.

Run-level artifacts are under
`outputs/motorneurons/temporal_screen_manual/{window_120,window_240,window_480,population_bouts}/`.
The strict descriptive analysis, exact tables, and figures are under
`outputs/motorneurons/temporal_screen_manual/analysis-output/`, with execution
metadata in `outputs/motorneurons/temporal_screen_manual/experiment-log.md`.

The empirical screen is truth-free and ran no c-GC or c-GC*. Its result is a
no-go for applying a temporal hard mask or soft prior to the primary empirical
graphs. The two population-bout passes may support only an explicitly
exploratory, segmentation-sensitive learner comparison; they are not evidence
of causal connectivity.

## Optional Smoke Checks

The scripts below are quick development checks and are not required for the full simulation execution order:

- `examples/synthetic_pipeline.py`
- `examples/fish3_trace_smoke.py`
- `examples/topographic_pair.py`
