# Result Readiness Report

Available artifacts: 9 / 14.
Missing user-run publication gates: 5.

## Missing User-Run Gates

- `validation_results/dynamic_episodic_locked/summary.json`: publication-grade causal-rise versus noncausal-fall claim.
- `validation_results/dynamic_episodic_locked/representation_summary.csv`: publication-grade dynamic-A representation comparison.
- `validation_results/dynamic_episodic_locked/rise_fall_contrasts.csv`: publication-grade dynamic-A rise-minus-comparator claim.
- `empirical_null_controls/null_control_contrasts.csv`: empirical observed-versus-null claim.
- `empirical_stability/stability_summary.csv`: empirical graph stability claim.

## All Artifacts

- ok `validation_results/cgc/summary.json` (static_validation): core synthetic validation.
- ok `validation_results/cgc_star/summary.json` (static_validation): core synthetic validation.
- ok `validation_results/dynamic_episodic/summary.json` (dynamic_a_initial): implementation smoke output.
- missing `validation_results/dynamic_episodic_locked/summary.json` (dynamic_a_locked): user-run locked synthetic validation.
- missing `validation_results/dynamic_episodic_locked/representation_summary.csv` (dynamic_a_locked): user-run locked synthetic validation.
- missing `validation_results/dynamic_episodic_locked/rise_fall_contrasts.csv` (dynamic_a_locked): user-run locked synthetic validation.
- ok `chen_comparison/chen_comparison_rows.csv` (chen_comparison): saved empirical artifact summary.
- ok `chen_comparison/chen_comparison_summary.csv` (chen_comparison): saved empirical artifact summary.
- ok `validation_tradeoffs/representation_tradeoff_summary.csv` (false_positive_tradeoff): saved synthetic tradeoff summary.
- ok `validation_tradeoffs/rise_tradeoff_contrasts.csv` (false_positive_tradeoff): saved synthetic tradeoff summary.
- ok `empirical_stats/paired_signflip_tests.csv` (empirical_pairing): saved empirical paired-statistic summary.
- ok `graph_stability/graph_stability_summary.csv` (graph_stability): saved graph-support summary.
- missing `empirical_null_controls/null_control_contrasts.csv` (empirical_null_controls): user-run empirical null controls.
- missing `empirical_stability/stability_summary.csv` (empirical_stability): user-run empirical stability.
