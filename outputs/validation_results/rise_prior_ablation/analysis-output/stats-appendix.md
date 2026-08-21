# Statistical appendix

Seed is the independent unit (n = 8); fold-level values are averaged within seed. F1 is bounded, sparse, and zero-inflated, so no normal approximation is used. Each contrast uses an exact two-sided paired sign-flip randomization test of the mean seed difference, a seed-bootstrap 95% CI (10,000 resamples), and Cohen's dz. Holm correction controls family-wise error separately for eight recovery contrasts and four null-FPR contrasts.

| family | condition | db | arm | mean diff | SD diff | 95% CI | dz | exact p | Holm p |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| recovery_f1 | clean | 0 | robust_hard | 0.000 | 0.000 | [0.000, 0.000] | 0.000 | 1.000 | 1.000 |
| recovery_f1 | clean | 0 | soft_prior | -0.036 | 0.066 | [-0.071, 0.000] | -0.540 | 0.500 | 1.000 |
| recovery_f1 | clean | 1 | robust_hard | 0.000 | 0.000 | [0.000, 0.000] | 0.000 | 1.000 | 1.000 |
| recovery_f1 | clean | 1 | soft_prior | 0.000 | 0.000 | [0.000, 0.000] | 0.000 | 1.000 | 1.000 |
| recovery_f1 | lowrate_shared | 0 | robust_hard | 0.000 | 0.000 | [0.000, 0.000] | 0.000 | 1.000 | 1.000 |
| recovery_f1 | lowrate_shared | 0 | soft_prior | 0.000 | 0.000 | [0.000, 0.000] | 0.000 | 1.000 | 1.000 |
| recovery_f1 | lowrate_shared | 1 | robust_hard | 0.000 | 0.000 | [0.000, 0.000] | 0.000 | 1.000 | 1.000 |
| recovery_f1 | lowrate_shared | 1 | soft_prior | 0.000 | 0.000 | [0.000, 0.000] | 0.000 | 1.000 | 1.000 |
| null_fpr | null_lowrate_shared | 0 | robust_hard | -0.003 | 0.009 | [-0.009, 0.000] | -0.354 | 1.000 | 1.000 |
| null_fpr | null_lowrate_shared | 0 | soft_prior | 0.000 | 0.000 | [0.000, 0.000] | 0.000 | 1.000 | 1.000 |
| null_fpr | null_lowrate_shared | 1 | robust_hard | 0.000 | 0.000 | [0.000, 0.000] | 0.000 | 1.000 | 1.000 |
| null_fpr | null_lowrate_shared | 1 | soft_prior | 0.000 | 0.000 | [0.000, 0.000] | 0.000 | 1.000 | 1.000 |

Limitations: two cross-fit folds from the same seed are not treated as independent; the bootstrap and randomization operate at seed level. The confidence intervals are percentile bootstrap intervals and are descriptive at n = 8. Timing is for a five-node graph and includes Python/core overhead, so it should not be extrapolated to large networks without a dedicated scaling benchmark.
