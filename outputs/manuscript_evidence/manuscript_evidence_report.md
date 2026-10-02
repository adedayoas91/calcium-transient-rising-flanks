# Manuscript Evidence Package

## Evidence Boundary

All tracked publication gates are available.

## Chen-Style Comparison

- Full-trace baseline rows available: 4.
- Direct Chen BVGC/MVGC matrix rows available: 0.
- Rising-flank rows available: 4.
- Falling-flank rows available: 4.
- Published Chen comparator is represented only for the reported ipsilateral-consistency value; missing fields must not be inferred.

## Empirical Paired Tests

- W_IC rise-minus-fall rows: 4; mean delta 0.050; max delta 0.100.
- W_RC rise-minus-fall rows: 4; mean delta -0.003; max delta 0.011.
- These paired tests are descriptive until empirical null-control and stability outputs exist.

## Synthetic Tradeoffs

- no_rise_recall_gain: 34 rows.
- rise_recall_gain_with_fpr_cost: 32 rows.
- rise_recall_gain_without_fpr_cost: 22 rows.

## Locked Dynamic-A Validation

- Locked dynamic-A interpretation rows available: 32.
- Rise-gain rows against zero-truth fall controls: 8.

## Empirical Null Controls

- Null-control interpretation rows available: 840.
- Nominal observed-above-null rows: 34; descriptive observed-above-null rows: 231.

## Empirical Re-Estimation Stability

- Stability interpretation rows available: 128.
- Rows classified as moderate or high stability: 128.

## Motorneuron Method Graph Support

- Method-specific graph-support rows available: 28.
- LPCMCI values summarize a lossy undirected lagged PAG skeleton; they are not directed-weight equivalents of c-GC/c-GC*.

## Figure Manifest

- `metric_summary_rise_vs_fall`: available_from_saved_tables; assemble final manuscript panel styling.
- `chen_comparison_panel_or_table`: available_from_saved_tables; decide primary rows versus supplement.
- `synthetic_recall_fpr_tradeoff`: available_from_saved_tables; keep claim method-level unless locked dynamic-A is added.
- `representative_network_case_c`: available_from_saved_images; use as representative Case C rise/fall network panel.
- `dynamic_a_locked_validation`: available; inspect locked dynamic-A interpretation.
- `empirical_null_and_stability_panel`: available; inspect null-control and stability interpretations.
- `motoneuron_method_graph_support`: available_from_saved_tables; compare graph support across motorneuron method outputs.
