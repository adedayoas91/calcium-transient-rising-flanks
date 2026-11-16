# Robust/soft temporal-prior ablation

Temporal priors were learned from four rise episodes and evaluated on four disjoint episodes, with the folds swapped. All scientific arms use the same held-out c-GC p-value matrix; the mask-aware refit is timing-only.

Recommendation: `no_go_for_causal_restriction`.

## Prespecified gate observations

```json
{
  "clean_hard_recall_loss": -0.0,
  "hard_mask_recall": {
    "clean": 0.9583333333333334,
    "lowrate_shared": 0.9791666666666667
  },
  "mean_f1_delta": {
    "robust_hard": 0.0,
    "soft_prior": 0.0
  },
  "mean_null_fpr_delta": 0.0,
  "mean_robust_candidate_density": 0.95625,
  "median_pruning_speedup": 1.3093882023271748
}
```

## Mean recovery by condition, deadband, and arm

| condition | deadband | arm | F1 | recall | FPR | mask recall | candidate density |
|---|---:|---|---:|---:|---:|---:|---:|
| clean | 0 | density_matched_random | 0.000 | 0.000 | 0.000 | 0.875 | 0.887 |
| clean | 0 | naive_hard | 0.000 | 0.000 | 0.054 | 0.708 | 0.512 |
| clean | 0 | oracle_hard | 0.000 | 0.000 | 0.000 | 1.000 | 0.300 |
| clean | 0 | robust_hard | 0.050 | 0.042 | 0.054 | 0.917 | 0.887 |
| clean | 0 | soft_prior | 0.050 | 0.042 | 0.054 | 1.000 | 1.000 |
| clean | 0 | unrestricted | 0.050 | 0.042 | 0.054 | 1.000 | 1.000 |
| clean | 1 | density_matched_random | 0.050 | 0.042 | 0.054 | 1.000 | 0.975 |
| clean | 1 | naive_hard | 0.000 | 0.000 | 0.054 | 0.708 | 0.512 |
| clean | 1 | oracle_hard | 0.000 | 0.000 | 0.000 | 1.000 | 0.300 |
| clean | 1 | robust_hard | 0.050 | 0.042 | 0.054 | 1.000 | 0.975 |
| clean | 1 | soft_prior | 0.050 | 0.042 | 0.054 | 1.000 | 1.000 |
| clean | 1 | unrestricted | 0.050 | 0.042 | 0.054 | 1.000 | 1.000 |
| lowrate_shared | 0 | density_matched_random | 0.000 | 0.000 | 0.000 | 0.917 | 0.962 |
| lowrate_shared | 0 | naive_hard | 0.071 | 0.042 | 0.000 | 0.500 | 0.425 |
| lowrate_shared | 0 | oracle_hard | 0.000 | 0.000 | 0.000 | 1.000 | 0.300 |
| lowrate_shared | 0 | robust_hard | 0.000 | 0.000 | 0.000 | 0.958 | 0.962 |
| lowrate_shared | 0 | soft_prior | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| lowrate_shared | 0 | unrestricted | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| lowrate_shared | 1 | density_matched_random | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| lowrate_shared | 1 | naive_hard | 0.071 | 0.042 | 0.000 | 0.500 | 0.425 |
| lowrate_shared | 1 | oracle_hard | 0.000 | 0.000 | 0.000 | 1.000 | 0.300 |
| lowrate_shared | 1 | robust_hard | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| lowrate_shared | 1 | soft_prior | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| lowrate_shared | 1 | unrestricted | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| null_lowrate_shared | 0 | density_matched_random | 0.000 | 0.000 | 0.000 | 1.000 | 0.988 |
| null_lowrate_shared | 0 | naive_hard | 0.000 | 0.000 | 0.000 | 1.000 | 0.325 |
| null_lowrate_shared | 0 | oracle_hard | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| null_lowrate_shared | 0 | robust_hard | 0.000 | 0.000 | 0.000 | 1.000 | 0.988 |
| null_lowrate_shared | 0 | soft_prior | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| null_lowrate_shared | 0 | unrestricted | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| null_lowrate_shared | 1 | density_matched_random | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| null_lowrate_shared | 1 | naive_hard | 0.000 | 0.000 | 0.000 | 1.000 | 0.325 |
| null_lowrate_shared | 1 | oracle_hard | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| null_lowrate_shared | 1 | robust_hard | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| null_lowrate_shared | 1 | soft_prior | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| null_lowrate_shared | 1 | unrestricted | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |

## Interpretation boundary

Lead/lag order is evaluated here as an external temporal prior, not as proof of causation. Masked-out directions are untested hypotheses, not established absences. Deadband 0 retains one-frame ordering; deadband 1 treats one-frame differences as ambiguous and is expected to be conservative at the simulated one-frame propagation delay.
