# Notebook Execution Order

Run notebooks from the `calcium-transient-rising-flank` package root so relative paths resolve cleanly. The newer wrapper notebooks default to dry-run mode; set their `RUN_*` toggles to `True` when you want them to launch the underlying scripts.

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

If you already ran step 2 and do not want to repeat dynamic validation, set `SKIP_DYNAMIC = True` in this notebook. Leave all skip flags as `False` for a single locked publication-gate rerun.

Set `DYNAMIC_TAU`, `DYNAMIC_N_PASTS`, `MIN_RISE_RUN_SAMPLES`, and the `RISE_MATCH_*` settings in the parameter cell to forward those hyperparameters into the dynamic-A validation stage.

## Optional Smoke Checks

The scripts below are quick development checks and are not required for the full simulation execution order:

- `examples/synthetic_pipeline.py`
- `examples/fish3_trace_smoke.py`
- `examples/topographic_pair.py`
