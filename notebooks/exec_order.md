# Notebook execution order

From the package root, prepare the locked environment once:

```bash
uv sync --frozen --all-extras
```

Use `uv run --no-sync` for any accompanying terminal commands.

Run the notebooks below from top to bottom. Required launch toggles are already enabled.

## Simulation generation

1. `notebooks/simulations/04_temporal_resolvability_map.ipynb` — generates and analyzes the locked temporal resolvability grid.
2. `notebooks/simulations/05_calibration_onset_run.ipynb` — runs threshold calibration and adaptive onset sensitivity.
3. `notebooks/simulations/06_dynamic_extensions_run.ipynb` — runs the unique mixed-fall and hybrid/context simulation arms.
4. `notebooks/simulations/c-GC.ipynb` — runs the c-GC simulation analyses and reviewer audits.
5. `notebooks/simulations/c-GC-star.ipynb` — runs the c-GC* simulation analyses and reviewer audits.
6. `notebooks/simulations/oasis.ipynb` — runs the OASIS event/preprocessing comparison.
7. `notebooks/simulations/pcmciplus.ipynb` — runs the matched PCMCI+ analyses, hypotheses, and reviewer audits.
8. `notebooks/simulations/var_granger.ipynb` — runs the matched VAR-Granger analyses, hypotheses, and reviewer audits.
9. `notebooks/simulations/lpcmci.ipynb` — runs only the bounded LPCMCI latent-confounding sensitivity and the five-method aggregate.

## Motorneuron analysis

10. `notebooks/motorneurons/fdr_reestimation.ipynb` — runs matched BH-adjusted and unadjusted empirical re-estimation.
11. `notebooks/motorneurons/Temporal_resolvability_screen.ipynb` — runs the descriptive temporal resolvability screen.
12. `notebooks/motorneurons/c-GC_Motoneurons.ipynb` — runs the c-GC motorneuron analyses and reviewer audits.
13. `notebooks/motorneurons/c-GC-star_Motoneurons.ipynb` — runs the c-GC* motorneuron analyses and reviewer audits.
14. `notebooks/motorneurons/oasis.ipynb` — runs OASIS preprocessing followed by c-GC, c-GC*, PCMCI+, and VAR.
15. `notebooks/motorneurons/pcmciplus.ipynb` — runs the matched PCMCI+ motorneuron analyses and reviewer audits.
16. `notebooks/motorneurons/var_granger.ipynb` — runs the matched VAR-Granger motorneuron analyses and reviewer audits.

## Final aggregation

17. `notebooks/simulations/03_publication_gate_pipeline_run.ipynb` — runs empirical null controls, stability checks, and publication-readiness packaging while reusing the method results.
18. `notebooks/simulations/02_saved_artifact_analysis_run.ipynb` — builds the final aggregate tables, statistical summaries, evidence package, and completion audit.

## Not part of the canonical run

- `notebooks/simulations/00_hyperparameter_timeseries_explorer.ipynb` — exploratory simulator inspection only.
- `notebooks/simulations/01_dynamic_episodic_validation_run.ipynb` — standalone duplicate of the episodic grid run by the c-GC and c-GC* notebooks.
- `notebooks/simulations/08_validation_campaign_run.ipynb` — alternative orchestration wrapper that overlaps the individual notebooks above.
- `notebooks/motorneurons/lpcmci.ipynb` — documents the intentional exclusion of the unbounded motorneuron LPCMCI run and exits without fitting.
