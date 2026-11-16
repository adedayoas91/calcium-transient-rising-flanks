# Robust/soft temporal-prior ablation

Temporal priors were learned from four rise episodes and evaluated on four disjoint episodes, with the folds swapped. All scientific arms use the same held-out c-GC p-value matrix; the mask-aware refit is timing-only.

Recommendation: `no_go_for_causal_restriction`.

## Prespecified gate observations

```json
{
  "clean_hard_recall_loss": -0.0,
  "hard_mask_recall": {
    "clean": 0.9739583333333334,
    "lowrate_shared": 0.9635416666666667
  },
  "mean_f1_delta": {
    "robust_hard": 0.0,
    "soft_prior": -0.008928571428571428
  },
  "mean_null_fpr_delta": -0.00078125,
  "mean_robust_candidate_density": 0.96640625,
  "median_pruning_speedup": 1.278828610462038
}
```

## Mean recovery by condition, deadband, and arm

| condition | deadband | arm | F1 | recall | FPR | mask recall | candidate density |
|---|---:|---|---:|---:|---:|---:|---:|
| clean | 0 | density_matched_random | 0.181 | 0.115 | 0.027 | 0.958 | 0.944 |
| clean | 0 | naive_hard | 0.049 | 0.031 | 0.018 | 0.625 | 0.438 |
| clean | 0 | oracle_hard | 0.198 | 0.125 | 0.000 | 1.000 | 0.300 |
| clean | 0 | robust_hard | 0.180 | 0.115 | 0.031 | 0.948 | 0.944 |
| clean | 0 | soft_prior | 0.144 | 0.094 | 0.022 | 1.000 | 1.000 |
| clean | 0 | unrestricted | 0.180 | 0.115 | 0.031 | 1.000 | 1.000 |
| clean | 1 | density_matched_random | 0.180 | 0.115 | 0.027 | 0.990 | 0.991 |
| clean | 1 | naive_hard | 0.049 | 0.031 | 0.018 | 0.625 | 0.438 |
| clean | 1 | oracle_hard | 0.198 | 0.125 | 0.000 | 1.000 | 0.300 |
| clean | 1 | robust_hard | 0.180 | 0.115 | 0.031 | 1.000 | 0.991 |
| clean | 1 | soft_prior | 0.180 | 0.115 | 0.031 | 1.000 | 1.000 |
| clean | 1 | unrestricted | 0.180 | 0.115 | 0.031 | 1.000 | 1.000 |
| lowrate_shared | 0 | density_matched_random | 0.033 | 0.021 | 0.031 | 0.896 | 0.931 |
| lowrate_shared | 0 | naive_hard | 0.018 | 0.010 | 0.004 | 0.542 | 0.478 |
| lowrate_shared | 0 | oracle_hard | 0.036 | 0.021 | 0.000 | 1.000 | 0.300 |
| lowrate_shared | 0 | robust_hard | 0.033 | 0.021 | 0.031 | 0.927 | 0.931 |
| lowrate_shared | 0 | soft_prior | 0.033 | 0.021 | 0.031 | 1.000 | 1.000 |
| lowrate_shared | 0 | unrestricted | 0.033 | 0.021 | 0.031 | 1.000 | 1.000 |
| lowrate_shared | 1 | density_matched_random | 0.033 | 0.021 | 0.031 | 1.000 | 1.000 |
| lowrate_shared | 1 | naive_hard | 0.018 | 0.010 | 0.004 | 0.542 | 0.478 |
| lowrate_shared | 1 | oracle_hard | 0.036 | 0.021 | 0.000 | 1.000 | 0.300 |
| lowrate_shared | 1 | robust_hard | 0.033 | 0.021 | 0.031 | 1.000 | 1.000 |
| lowrate_shared | 1 | soft_prior | 0.033 | 0.021 | 0.031 | 1.000 | 1.000 |
| lowrate_shared | 1 | unrestricted | 0.033 | 0.021 | 0.031 | 1.000 | 1.000 |
| null_lowrate_shared | 0 | density_matched_random | 0.000 | 0.000 | 0.038 | 1.000 | 0.959 |
| null_lowrate_shared | 0 | naive_hard | 0.000 | 0.000 | 0.009 | 1.000 | 0.403 |
| null_lowrate_shared | 0 | oracle_hard | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| null_lowrate_shared | 0 | robust_hard | 0.000 | 0.000 | 0.034 | 1.000 | 0.959 |
| null_lowrate_shared | 0 | soft_prior | 0.000 | 0.000 | 0.038 | 1.000 | 1.000 |
| null_lowrate_shared | 0 | unrestricted | 0.000 | 0.000 | 0.038 | 1.000 | 1.000 |
| null_lowrate_shared | 1 | density_matched_random | 0.000 | 0.000 | 0.038 | 1.000 | 1.000 |
| null_lowrate_shared | 1 | naive_hard | 0.000 | 0.000 | 0.009 | 1.000 | 0.403 |
| null_lowrate_shared | 1 | oracle_hard | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| null_lowrate_shared | 1 | robust_hard | 0.000 | 0.000 | 0.038 | 1.000 | 1.000 |
| null_lowrate_shared | 1 | soft_prior | 0.000 | 0.000 | 0.038 | 1.000 | 1.000 |
| null_lowrate_shared | 1 | unrestricted | 0.000 | 0.000 | 0.038 | 1.000 | 1.000 |

## Interpretation boundary

Lead/lag order is evaluated here as an external temporal prior, not as proof of causation. Masked-out directions are untested hypotheses, not established absences. Deadband 0 retains one-frame ordering; deadband 1 treats one-frame differences as ambiguous and is expected to be conservative at the simulated one-frame propagation delay.
