# Rise-prior ablation: strict analysis

## Analysis question

Does an episode-cross-fitted temporal-order prior improve held-out directed-edge F1 or reduce computation without increasing null false positives, relative to unrestricted c-GC? The repeated-measure unit is the simulation seed (n = 8); the two cross-fit folds are averaged within seed.

## Key finding

Neither robust hard restriction nor soft weighting improved held-out F1. The hard rule remained null-safe and usually preserved true directions, but did so by retaining almost the full hypothesis family. The observed efficiency and recovery gates therefore failed; the current rule is not feasible as a causal restriction.

## Exact numeric summary

Values are seed means reported as mean ± sample SD with a seed-bootstrap 95% CI.

| condition | deadband | arm | held-out F1 | 95% CI |
|---|---:|---|---:|---:|
| clean | 0 | unrestricted | 0.180 ± 0.118 | [0.103, 0.259] |
| clean | 0 | naive_hard | 0.049 ± 0.095 | [0.000, 0.116] |
| clean | 0 | robust_hard | 0.180 ± 0.118 | [0.104, 0.257] |
| clean | 0 | soft_prior | 0.144 ± 0.124 | [0.067, 0.228] |
| clean | 0 | density_matched_random | 0.181 ± 0.117 | [0.103, 0.259] |
| clean | 0 | oracle_hard | 0.198 ± 0.141 | [0.116, 0.295] |
| clean | 1 | unrestricted | 0.180 ± 0.118 | [0.104, 0.257] |
| clean | 1 | naive_hard | 0.049 ± 0.095 | [0.000, 0.116] |
| clean | 1 | robust_hard | 0.180 ± 0.118 | [0.107, 0.256] |
| clean | 1 | soft_prior | 0.180 ± 0.118 | [0.104, 0.259] |
| clean | 1 | density_matched_random | 0.180 ± 0.118 | [0.103, 0.257] |
| clean | 1 | oracle_hard | 0.198 ± 0.141 | [0.113, 0.295] |
| lowrate_shared | 0 | unrestricted | 0.033 ± 0.062 | [0.000, 0.080] |
| lowrate_shared | 0 | naive_hard | 0.018 ± 0.051 | [0.000, 0.054] |
| lowrate_shared | 0 | robust_hard | 0.033 ± 0.062 | [0.000, 0.080] |
| lowrate_shared | 0 | soft_prior | 0.033 ± 0.062 | [0.000, 0.083] |
| lowrate_shared | 0 | density_matched_random | 0.033 ± 0.062 | [0.000, 0.083] |
| lowrate_shared | 0 | oracle_hard | 0.036 ± 0.066 | [0.000, 0.089] |
| lowrate_shared | 1 | unrestricted | 0.033 ± 0.062 | [0.000, 0.080] |
| lowrate_shared | 1 | naive_hard | 0.018 ± 0.051 | [0.000, 0.054] |
| lowrate_shared | 1 | robust_hard | 0.033 ± 0.062 | [0.000, 0.078] |
| lowrate_shared | 1 | soft_prior | 0.033 ± 0.062 | [0.000, 0.071] |
| lowrate_shared | 1 | density_matched_random | 0.033 ± 0.062 | [0.000, 0.080] |
| lowrate_shared | 1 | oracle_hard | 0.036 ± 0.066 | [0.000, 0.089] |

## Decision-changing observations

- Robust-hard mean F1 change pooled by the prespecified gate: 0.000; soft-prior change: -0.009.
- Mean robust-mask density was 0.966, versus the ≤0.60 target. Median measured speedup was 1.259×, versus the ≥1.67× target.
- Robust-mask recall was 0.974 in clean data and 0.964 under low-rate/shared-noise observations.
- Mean null FPR change across robust/soft arms was -0.001; the safety gate passed.

The exact paired tests (Holm-adjusted within recovery and null-safety families) provide no evidence of an F1 improvement. Deadband 1 is especially uninformative at a one-frame propagation delay because it converts the target directional signal into a tie.

## Evidence boundary and next decision

This is a five-node synthetic experiment with a known one-frame delay and eight seeds. It tests feasibility under the current rise extraction and conservative ambiguity rule; it does not prove that every temporal prior will fail. The next useful experiment is not to deploy hard causal pruning. It is to redesign the prior so ambiguity can reduce confidence without automatically admitting both directions, then validate that redesign on longer-delay and higher-SNR regimes before any empirical use.

Prespecified recommendation: `no_go_for_causal_restriction`.
