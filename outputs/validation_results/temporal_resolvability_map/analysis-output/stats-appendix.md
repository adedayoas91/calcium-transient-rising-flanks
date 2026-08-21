# Statistical appendix

The independent repeated-measure unit is the random seed (n = 20, seeds 1–20). Two cross-fit episode folds and all q acquisition offsets for downsampling factor q are averaged within seed. Sequential regime contrasts then average the 12 locked delay/downsampling cells within each seed, preserving common-random-number pairing.

Metrics are bounded and often concentrated at thresholds, so inference does not rely on normality. Each right-minus-left regime contrast uses a two-sided exact paired sign-flip randomization test over all 2^20 assignments, a percentile seed-bootstrap 95% confidence interval with 10,000 resamples, and Cohen's dz. Holm correction controls family-wise error separately for the three contrasts of each metric. Descriptive cell intervals use the same 10,000-resample seed bootstrap.

| family | contrast | n | mean diff | SD diff | 95% CI | dz | exact p | Holm p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| true_edge_coverage | independent_noise | 20 | -0.018 | 0.011 | [-0.023, -0.014] | -1.687 | <0.001 | <0.001 |
| true_edge_coverage | shared_noise | 20 | 0.010 | 0.013 | [0.004, 0.015] | 0.766 | 0.003 | 0.006 |
| true_edge_coverage | decay_heterogeneity | 20 | -0.006 | 0.009 | [-0.011, -0.003] | -0.680 | 0.003 | 0.006 |
| unconditional_direction_accuracy | independent_noise | 20 | -0.332 | 0.046 | [-0.352, -0.314] | -7.238 | <0.001 | <0.001 |
| unconditional_direction_accuracy | shared_noise | 20 | -0.108 | 0.042 | [-0.126, -0.090] | -2.588 | <0.001 | <0.001 |
| unconditional_direction_accuracy | decay_heterogeneity | 20 | -0.005 | 0.027 | [-0.016, 0.007] | -0.172 | 0.459 | 0.459 |
| candidate_density | independent_noise | 20 | 0.243 | 0.017 | [0.235, 0.250] | 13.902 | <0.001 | <0.001 |
| candidate_density | shared_noise | 20 | 0.055 | 0.016 | [0.048, 0.062] | 3.501 | <0.001 | <0.001 |
| candidate_density | decay_heterogeneity | 20 | 0.000 | 0.010 | [-0.005, 0.004] | 0.023 | 0.933 | 0.933 |

The inferential contrasts estimate sequential additions in the locked nested regimes; they are not factorial main effects or interactions. Grid cells are deliberately not treated as independent replicates. The pass/fail map is governed by prespecified effect thresholds and seed replication, not p-values.
