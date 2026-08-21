# Temporal-resolvability map: strict analysis

## Analysis question

Under what acquisition and observation conditions can the three-state rising-flank screen retain true directions, orient them correctly, and reduce the candidate family before any c-GC fit? The independent unit is the simulation seed (n = 20). Cross-fit folds and all acquisition-phase offsets are averaged within seed.

## Key finding

7 of 48 primary fluorescence cells passed the locked conjunctive gate. Passing cells occurred in: clean homogeneous. No cell with stronger independent noise, shared noise, or heterogeneous decay passed. The three-state screen is therefore conditionally feasible as a candidate-pruning pre-screen only within the passing synthetic upper-bound conditions.

## Passing region

| regime | native delay | downsample | observed delay | coverage | unconditional accuracy | density | passing seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean_homogeneous | 1 | 1 | 1.000 | 1.000 | 0.994 | 0.503 | 19/20 |
| clean_homogeneous | 2 | 1 | 2.000 | 1.000 | 1.000 | 0.546 | 19/20 |
| clean_homogeneous | 2 | 2 | 1.000 | 1.000 | 0.994 | 0.501 | 20/20 |
| clean_homogeneous | 4 | 2 | 2.000 | 1.000 | 1.000 | 0.351 | 20/20 |
| clean_homogeneous | 4 | 4 | 1.000 | 1.000 | 0.995 | 0.432 | 20/20 |
| clean_homogeneous | 8 | 2 | 4.000 | 1.000 | 0.994 | 0.205 | 20/20 |
| clean_homogeneous | 8 | 4 | 2.000 | 1.000 | 0.997 | 0.210 | 20/20 |

The passing cells are not described by a single delay-to-frame ratio. For example, native delay 4 at downsampling 1 passes the mean thresholds but only 6/20 seeds pass jointly, so it fails the seed-replication gate. This is why the map and conjunctive seed criterion are more informative than a global one-frame cutoff.

## Where information is lost

Clean-regime values below average the complete 4 × 3 delay/downsampling grid; uncertainty is across 20 seed-level grid averages.

| signal layer | accuracy mean ± SD | 95% CI | candidate density |
|---|---:|---:|---:|
| native_latent_events | 1.000 ± 0.000 | [1.000, 1.000] | 0.388 |
| sampled_latent_events | 0.551 ± 0.040 | [0.533, 0.568] | 0.332 |
| sampled_noiseless_calcium | 0.876 ± 0.014 | [0.870, 0.882] | 0.422 |
| sampled_noisy_fluorescence | 0.835 ± 0.022 | [0.825, 0.843] | 0.473 |

Native latent events preserve the imposed order. Point sampling loses many impulse events, while noiseless calcium integration recovers much of the ordering. Observation noise then reduces accuracy and increases ambiguity. These layer comparisons localize measurability loss; they are not additional hypothesis tests or causal evidence.

## Sequential stress contrasts

Contrasts average the 12 primary grid cells within seed and report right-minus-left changes. Exact paired sign-flip p-values are Holm-adjusted within each three-contrast metric family.

| metric | added factor | mean change | 95% CI | dz | Holm p |
|---|---|---:|---:|---:|---:|
| true_edge_coverage | independent_noise | -0.018 | [-0.023, -0.014] | -1.687 | <0.001 |
| true_edge_coverage | shared_noise | 0.010 | [0.004, 0.015] | 0.766 | 0.006 |
| true_edge_coverage | decay_heterogeneity | -0.006 | [-0.011, -0.003] | -0.680 | 0.006 |
| unconditional_direction_accuracy | independent_noise | -0.332 | [-0.352, -0.314] | -7.238 | <0.001 |
| unconditional_direction_accuracy | shared_noise | -0.108 | [-0.126, -0.090] | -2.588 | <0.001 |
| unconditional_direction_accuracy | decay_heterogeneity | -0.005 | [-0.016, 0.007] | -0.172 | 0.459 |
| candidate_density | independent_noise | 0.243 | [0.235, 0.250] | 13.902 | <0.001 |
| candidate_density | shared_noise | 0.055 | [0.048, 0.062] | 3.501 | <0.001 |
| candidate_density | decay_heterogeneity | 0.000 | [-0.005, 0.004] | 0.023 | 0.933 |

## Decision and evidence boundary

Proceed only to a small c-GC follow-up restricted to the 7 viable coordinates (clean homogeneous), with the three-state mask treated as a screening prior rather than ground-truth causality. Do not use the rule as a universal hard direction constraint, and do not promote it to noisy empirical recordings on the strength of this experiment.

The simulator is a deterministic source-triggered five-node chain with no spontaneous events or transmission failures. It measures temporal observability under known propagation; it does not establish causal identification, handle reciprocal edges, model ROI-specific rise kernels, or demonstrate graph-recovery improvement. Deadband 1 remains descriptive sensitivity only.

Prespecified recommendation: `proceed_to_cgc_in_viable_regimes`.
