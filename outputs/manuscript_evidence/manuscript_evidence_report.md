# Manuscript Evidence Package

## Evidence Boundary

Do not promote the empirical superiority claim yet; required user-run publication gates are still missing.
- `validation_results/dynamic_episodic_locked/summary.json`: publication-grade causal-rise versus noncausal-fall claim.
- `validation_results/dynamic_episodic_locked/representation_summary.csv`: publication-grade dynamic-A representation comparison.
- `validation_results/dynamic_episodic_locked/rise_fall_contrasts.csv`: publication-grade dynamic-A rise-minus-comparator claim.
- `empirical_null_controls/null_control_contrasts.csv`: empirical observed-versus-null claim.
- `empirical_stability/stability_summary.csv`: empirical graph stability claim.

## Chen-Style Comparison

- Full-trace baseline rows available: 4.
- Direct Chen BVGC/MVGC matrix rows available: 0.
- Rising-flank rows available: 4.
- Falling-flank rows available: 4.
- Published Chen comparator is represented only for the reported ipsilateral-consistency value; missing fields must not be inferred.

## Empirical Paired Tests

- W_IC rise-minus-fall rows: 4; mean delta 0.387; max delta 0.726.
- W_RC rise-minus-fall rows: 4; mean delta 0.547; max delta 0.690.
- These paired tests are descriptive until empirical null-control and stability outputs exist.

## Synthetic Tradeoffs

- no_rise_recall_gain: 7 rows.
- rise_recall_gain_with_fpr_cost: 15 rows.
- rise_recall_gain_without_fpr_cost: 4 rows.

## Locked Dynamic-A Validation

- No locked dynamic-A interpretation rows are available in the evidence package yet.

## Empirical Null Controls

- No empirical null-control contrast rows are available in the evidence package yet.

## Empirical Re-Estimation Stability

- No empirical re-estimation stability rows are available in the evidence package yet.

## Hindbrain Descriptive Extension

- Hindbrain graph-support rows available: 4.
- Report edge density, retained edges, and total retained weight only; do not report W_IC or W_RC without a declared anatomical partition.
- medial cgc full_trace: edge density 0.092, retained edges 35.0, total weight 23.528.
- medial cgc-star full_trace: edge density 0.050, retained edges 19.0, total weight 13.030.
- medial rising_flank_cgc fall: edge density 0.061, retained edges 23.0, total weight 3.374.
- medial rising_flank_cgc rise: edge density 0.166, retained edges 63.0, total weight 9.709.

## Figure Manifest

- `metric_summary_rise_vs_fall`: available_from_saved_tables; assemble final manuscript panel styling.
- `chen_comparison_panel_or_table`: available_from_saved_tables; decide primary rows versus supplement.
- `synthetic_recall_fpr_tradeoff`: available_from_saved_tables; keep claim method-level unless locked dynamic-A is added.
- `representative_network_case_c`: available_from_saved_images; use as representative Case C rise/fall network panel.
- `dynamic_a_locked_validation`: missing_user_run; run locked dynamic-A script before publication claim.
- `empirical_null_and_stability_panel`: missing_user_run; run empirical null controls and re-estimation stability.
- `hindbrain_descriptive_extension`: available_from_saved_tables; use as descriptive supplement unless anatomical partitions are defined.
