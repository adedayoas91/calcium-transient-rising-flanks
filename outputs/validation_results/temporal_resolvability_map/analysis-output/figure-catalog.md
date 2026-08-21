# Figure catalog

## Figure 1 — `figures/figure-01-primary-resolution-map.pdf`

- Purpose: identify where the primary noisy-fluorescence screen satisfies the temporal-direction requirements before c-GC.
- Data: seed means after averaging two cross-fit folds and all acquisition phases, n = 20 seeds per cell; deadband 0.
- Display: color is unconditional direction accuracy; each cell prints accuracy (A) and total three-state candidate density (D); orange outlines mark cells that also pass coverage, seed replication, and adjacency gates.
- Observation: 7 cells pass in clean homogeneous; all other cells fail.
- Interpretation: useful pruning requires both sufficiently stable onset order and low enough ambiguity, not merely true-edge admission.
- Implication: any c-GC follow-up should be limited to the outlined viable cells.
- Caveat: the source-triggered chain is an upper-bound measurability test, not causal identification.

## Figure 2 — `figures/figure-02-information-loss.pdf`

- Purpose: localize whether failures arise in latent timing, sampling, calcium convolution, or observation noise.
- Data: clean homogeneous regime, deadband 0, n = 20 seed means per cell.
- Display: color and annotations show mean unconditional direction accuracy; annotations also show sample SD across seeds.
- Observation: native latent order is perfectly measurable; point sampling removes impulses, noiseless calcium integration restores much of the order, and fluorescence noise creates additional losses.
- Interpretation: the observed boundary is a measurement/extraction boundary rather than failure of the imposed latent propagation order.
- Implication: better event extraction may expand the feasible region, but the current rule should not be generalized beyond observed passing cells.
- Caveat: sampled impulse events are a diagnostic layer and do not mimic the temporal integration of a physical camera exposure.
