# Temporal resolvability map

This screening-only experiment asks when rising-flank onset order can resolve directed propagation. It does not run c-GC and does not establish causal identification.

Recommendation: `proceed_to_cgc_in_viable_regimes`.

Viable primary fluorescence cells: 7 of 48.

## Viable regimes

| regime | native delay | downsample | observed delay | coverage | direction accuracy | candidate density | passing seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean_homogeneous | 1 | 1 | 1.00 | 1.000 | 0.994 | 0.503 | 19/20 |
| clean_homogeneous | 2 | 1 | 2.00 | 1.000 | 1.000 | 0.546 | 19/20 |
| clean_homogeneous | 2 | 2 | 1.00 | 1.000 | 0.994 | 0.501 | 20/20 |
| clean_homogeneous | 4 | 2 | 2.00 | 1.000 | 1.000 | 0.351 | 20/20 |
| clean_homogeneous | 4 | 4 | 1.00 | 1.000 | 0.995 | 0.432 | 20/20 |
| clean_homogeneous | 8 | 2 | 4.00 | 1.000 | 0.994 | 0.205 | 20/20 |
| clean_homogeneous | 8 | 4 | 2.00 | 1.000 | 0.997 | 0.210 | 20/20 |

## Interpretation boundary

The source-triggered stable graph is an upper-bound measurement test. Passing cells identify conditions in which temporal ordering is sufficiently resolved to justify a later graph-learning ablation. A cell passes only when mean coverage and unconditional direction accuracy are at least 0.90, candidate density is at most 0.60, at least 80% of seeds pass those three criteria, and a neighboring grid cell also passes. Failure here argues against direction pruning at the corresponding acquisition resolution.

Deadband 0 is primary; deadband 1 is descriptive sensitivity. Native events, sampled events, and sampled noiseless calcium are diagnostic layers used to localize information loss, not additional decision opportunities.
