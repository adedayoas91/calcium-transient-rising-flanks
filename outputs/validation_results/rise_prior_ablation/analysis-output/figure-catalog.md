# Figure catalog

## Figure 1 — `figures/figure-01-f1-differences.pdf`

- Purpose: test whether either temporal prior changes held-out F1 relative to the unrestricted graph learner.
- Data: seed-level means across two cross-fit folds, n = 8 seeds per condition/deadband.
- Display: points are seeds; black diamonds are means; bars are seed-bootstrap 95% CIs.
- Observation: robust-hard differences sit at zero; soft weighting is neutral except for a negative clean/deadband-0 shift.
- Implication: the current priors do not justify promotion into the causal inference path.
- Caveat: bounded, sparse F1 and five-node simulation.

## Figure 2 — `figures/figure-02-mask-efficiency.pdf`

- Purpose: show the tradeoff that explains why hard-prior recall remains high while speedup is weak.
- Data: robust-hard seed means across two folds, n = 8 seeds.
- Display: circles are deadband 0, squares are deadband 1; dashed vertical line is the 0.60 density target; dotted horizontal lines are the 0.90 recall and 1.67× speed targets.
- Observation: points cluster near density 1.0, well outside the pruning target; speedups remain below target for most seeds.
- Implication: ambiguity handling protects recall by retaining nearly every direction, defeating the proposed computational advantage.
- Caveat: runtime is a small-graph diagnostic, not a scaling study.
