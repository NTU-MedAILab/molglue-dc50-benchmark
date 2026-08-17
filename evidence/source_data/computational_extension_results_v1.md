# Computational extension results v1

Analysis identity: `post_hoc_extension_float32_v4`  
Source-data generation ID: `2dfa3b400e36d870b4b2f7b6860c6ffd38e07ff16aaf756aab930b5ca0844f6c`

Evidence identity: **post-hoc**. Positive paired contrasts favour the first named model; no multiplicity-adjusted significance claim is made.

## Primary robustness effects

| Analysis | Condition | Metric | Estimate [95% interval] | Interval relation to zero |
|---|---|---|---:|---|
| held_axis_portable_context | source_ood:full_vs_chemistry | delta_domain_macro_spearman | -0.086 [-0.154, -0.011] | unfavourable_interval_below_zero |
| held_axis_portable_context | source_ood:full_vs_chemistry | delta_domain_macro_rmse | -0.002 [-0.034, +0.025] | interval_includes_zero |
| held_axis_portable_context | source_ood:portable_vs_chemistry | delta_domain_macro_spearman | -0.108 [-0.176, -0.025] | unfavourable_interval_below_zero |
| held_axis_portable_context | source_ood:portable_vs_chemistry | delta_domain_macro_rmse | -0.010 [-0.046, +0.019] | interval_includes_zero |
| held_axis_portable_context | source_ood:portable_vs_full | delta_domain_macro_spearman | -0.022 [-0.038, -0.001] | unfavourable_interval_below_zero |
| held_axis_portable_context | source_ood:portable_vs_full | delta_domain_macro_rmse | -0.008 [-0.014, -0.003] | unfavourable_interval_below_zero |
| held_axis_portable_context | target_ood:full_vs_chemistry | delta_domain_macro_spearman | -0.066 [-0.131, -0.007] | unfavourable_interval_below_zero |
| held_axis_portable_context | target_ood:full_vs_chemistry | delta_domain_macro_rmse | -0.008 [-0.018, +0.003] | interval_includes_zero |
| held_axis_portable_context | target_ood:portable_vs_chemistry | delta_domain_macro_spearman | -0.051 [-0.093, -0.009] | unfavourable_interval_below_zero |
| held_axis_portable_context | target_ood:portable_vs_chemistry | delta_domain_macro_rmse | -0.012 [-0.018, -0.005] | unfavourable_interval_below_zero |
| held_axis_portable_context | target_ood:portable_vs_full | delta_domain_macro_spearman | +0.015 [-0.037, +0.072] | interval_includes_zero |
| held_axis_portable_context | target_ood:portable_vs_full | delta_domain_macro_rmse | -0.004 [-0.014, +0.005] | interval_includes_zero |
| internal_fit_weight | uniform_full_vs_chemistry | delta_spearman | +0.055 [+0.023, +0.094] | favourable_interval_above_zero |
| internal_fit_weight | uniform_full_vs_chemistry | delta_rmse | +0.040 [+0.025, +0.057] | favourable_interval_above_zero |
| internal_fit_weight | compound_equal_full_vs_chemistry | delta_spearman | +0.051 [+0.022, +0.087] | favourable_interval_above_zero |
| internal_fit_weight | compound_equal_full_vs_chemistry | delta_rmse | +0.038 [+0.023, +0.054] | favourable_interval_above_zero |
| internal_fit_weight | domain_balanced_full_vs_chemistry | delta_spearman | +0.050 [+0.023, +0.082] | favourable_interval_above_zero |
| internal_fit_weight | domain_balanced_full_vs_chemistry | delta_rmse | +0.036 [+0.021, +0.052] | favourable_interval_above_zero |
| generic_murcko_scaffold | full_vs_chemistry | delta_spearman | +0.037 [+0.010, +0.066] | favourable_interval_above_zero |
| generic_murcko_scaffold | full_vs_chemistry | delta_rmse | +0.027 [+0.014, +0.041] | favourable_interval_above_zero |
| target_ood_training_source_deletion | none | delta_spearman | -0.066 [-0.131, -0.006] | unfavourable_interval_below_zero |
| target_ood_training_source_deletion | none | delta_rmse | -0.008 [-0.017, +0.003] | interval_includes_zero |
| target_ood_training_source_deletion | MGTbind | delta_spearman | -0.033 [-0.088, +0.027] | interval_includes_zero |
| target_ood_training_source_deletion | MGTbind | delta_rmse | +0.013 [+0.001, +0.025] | favourable_interval_above_zero |
| target_ood_training_source_deletion | MGDB | delta_spearman | -0.032 [-0.069, +0.014] | interval_includes_zero |
| target_ood_training_source_deletion | MGDB | delta_rmse | -0.002 [-0.008, +0.004] | interval_includes_zero |
| target_ood_training_source_deletion | MolGlueDB | delta_spearman | -0.089 [-0.139, -0.033] | unfavourable_interval_below_zero |
| target_ood_training_source_deletion | MolGlueDB | delta_rmse | +0.003 [-0.012, +0.018] | interval_includes_zero |
| target_ood_training_source_deletion | TPDdb | delta_spearman | -0.052 [-0.095, -0.001] | unfavourable_interval_below_zero |
| target_ood_training_source_deletion | TPDdb | delta_rmse | +0.013 [+0.002, +0.024] | favourable_interval_above_zero |
| target_ood_deletion_vs_within_script_baseline | MGTbind:chemistry_extra_trees | delta_spearman | -0.066 [-0.123, -0.017] | unfavourable_interval_below_zero |
| target_ood_deletion_vs_within_script_baseline | MGTbind:chemistry_extra_trees | delta_rmse | -0.022 [-0.036, -0.008] | unfavourable_interval_below_zero |
| target_ood_deletion_vs_within_script_baseline | MGTbind:full_context_extra_trees | delta_spearman | -0.034 [-0.086, +0.022] | interval_includes_zero |
| target_ood_deletion_vs_within_script_baseline | MGTbind:full_context_extra_trees | delta_rmse | -0.001 [-0.015, +0.012] | interval_includes_zero |
| target_ood_deletion_vs_within_script_baseline | MGDB:chemistry_extra_trees | delta_spearman | -0.074 [-0.156, +0.002] | interval_includes_zero |
| target_ood_deletion_vs_within_script_baseline | MGDB:chemistry_extra_trees | delta_rmse | -0.021 [-0.040, -0.000] | unfavourable_interval_below_zero |
| target_ood_deletion_vs_within_script_baseline | MGDB:full_context_extra_trees | delta_spearman | -0.040 [-0.096, +0.023] | interval_includes_zero |
| target_ood_deletion_vs_within_script_baseline | MGDB:full_context_extra_trees | delta_rmse | -0.015 [-0.027, -0.001] | unfavourable_interval_below_zero |
| target_ood_deletion_vs_within_script_baseline | MolGlueDB:chemistry_extra_trees | delta_spearman | +0.068 [-0.045, +0.141] | interval_includes_zero |
| target_ood_deletion_vs_within_script_baseline | MolGlueDB:chemistry_extra_trees | delta_rmse | +0.009 [-0.017, +0.026] | interval_includes_zero |
| target_ood_deletion_vs_within_script_baseline | MolGlueDB:full_context_extra_trees | delta_spearman | +0.045 [-0.049, +0.109] | interval_includes_zero |
| target_ood_deletion_vs_within_script_baseline | MolGlueDB:full_context_extra_trees | delta_rmse | +0.019 [+0.002, +0.029] | favourable_interval_above_zero |
| target_ood_deletion_vs_within_script_baseline | TPDdb:chemistry_extra_trees | delta_spearman | +0.002 [-0.057, +0.056] | interval_includes_zero |
| target_ood_deletion_vs_within_script_baseline | TPDdb:chemistry_extra_trees | delta_rmse | -0.011 [-0.023, +0.001] | interval_includes_zero |
| target_ood_deletion_vs_within_script_baseline | TPDdb:full_context_extra_trees | delta_spearman | +0.016 [-0.028, +0.066] | interval_includes_zero |
| target_ood_deletion_vs_within_script_baseline | TPDdb:full_context_extra_trees | delta_rmse | +0.009 [+0.001, +0.017] | favourable_interval_above_zero |

Generic Murcko grouping reduced 667 Bemis–Murcko groups to 376 generic groups; 183 were singletons and the largest contained 76 rows.

## Fit-weight diagnostics

| Weight mode | Fit records | Clipped fraction, mean–max | Effective-sample fraction, min–median–max |
|---|---:|---:|---:|
| compound_equal | 125 | 0.000–0.000 | 0.818–0.838–0.868 |
| domain_balanced_clipped | 125 | 0.031–0.044 | 0.164–0.181–0.208 |

## Chemical-novelty applicability contrasts

| Regime, protocol and Tanimoto bin | Metric | Estimate [95% interval] | Interval relation to zero | Rows | Scaffolds |
|---|---|---:|---|---:|---:|
| internal_scaffold_disjoint:scaffold:0.0_to_lt_0.4 | delta_spearman | +0.061 [-0.069, +0.240] | interval_includes_zero | 39 | 22 |
| internal_scaffold_disjoint:scaffold:0.0_to_lt_0.4 | delta_rmse | +0.011 [-0.005, +0.031] | interval_includes_zero | 39 | 22 |
| internal_scaffold_disjoint:scaffold:0.4_to_lt_0.6 | delta_spearman | +0.121 [+0.024, +0.209] | favourable_interval_above_zero | 168 | 55 |
| internal_scaffold_disjoint:scaffold:0.4_to_lt_0.6 | delta_rmse | +0.056 [+0.027, +0.087] | favourable_interval_above_zero | 168 | 55 |
| internal_scaffold_disjoint:scaffold:0.6_to_lt_0.8 | delta_spearman | +0.044 [+0.007, +0.093] | favourable_interval_above_zero | 874 | 371 |
| internal_scaffold_disjoint:scaffold:0.6_to_lt_0.8 | delta_rmse | +0.042 [+0.021, +0.064] | favourable_interval_above_zero | 874 | 371 |
| internal_scaffold_disjoint:scaffold:0.8_to_1.000001 | delta_spearman | +0.058 [+0.011, +0.111] | favourable_interval_above_zero | 479 | 282 |
| internal_scaffold_disjoint:scaffold:0.8_to_1.000001 | delta_rmse | +0.034 [+0.013, +0.056] | favourable_interval_above_zero | 479 | 282 |
| frozen_ood:source_ood:0.0_to_lt_0.4 | delta_domain_macro_spearman | +0.062 [interval not estimable] | not_estimable | 798 | 429 |
| frozen_ood:source_ood:0.0_to_lt_0.4 | delta_domain_macro_rmse | -0.017 [-0.036, +0.026] | interval_includes_zero | 798 | 429 |
| frozen_ood:source_ood:0.4_to_lt_0.6 | delta_domain_macro_spearman | -0.118 [-0.230, +0.035] | interval_includes_zero | 360 | 170 |
| frozen_ood:source_ood:0.4_to_lt_0.6 | delta_domain_macro_rmse | -0.033 [-0.064, -0.001] | unfavourable_interval_below_zero | 360 | 170 |
| frozen_ood:source_ood:0.6_to_lt_0.8 | delta_domain_macro_spearman | -0.053 [-0.251, +0.030] | interval_includes_zero | 239 | 95 |
| frozen_ood:source_ood:0.6_to_lt_0.8 | delta_domain_macro_rmse | -0.022 [-0.076, +0.035] | interval_includes_zero | 239 | 95 |
| frozen_ood:source_ood:0.8_to_1.000001 | delta_domain_macro_spearman | +0.029 [-0.066, +0.148] | interval_includes_zero | 123 | 62 |
| frozen_ood:source_ood:0.8_to_1.000001 | delta_domain_macro_rmse | +0.013 [-0.066, +0.075] | interval_includes_zero | 123 | 62 |
| frozen_ood:target_ood:0.0_to_lt_0.4 | delta_domain_macro_spearman | -0.037 [interval not estimable] | not_estimable | 740 | 380 |
| frozen_ood:target_ood:0.0_to_lt_0.4 | delta_domain_macro_rmse | -0.015 [interval not estimable] | not_estimable | 740 | 380 |
| frozen_ood:target_ood:0.4_to_lt_0.6 | delta_domain_macro_spearman | -0.087 [-0.203, -0.001] | unfavourable_interval_below_zero | 391 | 207 |
| frozen_ood:target_ood:0.4_to_lt_0.6 | delta_domain_macro_rmse | +0.012 [-0.003, +0.027] | interval_includes_zero | 391 | 207 |
| frozen_ood:target_ood:0.6_to_lt_0.8 | delta_domain_macro_spearman | -0.047 [-0.117, +0.076] | interval_includes_zero | 118 | 38 |
| frozen_ood:target_ood:0.6_to_lt_0.8 | delta_domain_macro_rmse | -0.040 [-0.063, +0.009] | interval_includes_zero | 118 | 38 |
| frozen_ood:target_ood:0.8_to_1.000001 | delta_domain_macro_spearman | +0.001 [-0.136, +0.092] | interval_includes_zero | 41 | 18 |
| frozen_ood:target_ood:0.8_to_1.000001 | delta_domain_macro_rmse | +0.002 [-0.020, +0.060] | interval_includes_zero | 41 | 18 |
| strict_domain_scaffold_ood:source_ood:0.0_to_lt_0.4 | delta_domain_macro_spearman | +0.015 [interval not estimable] | not_estimable | 814 | 439 |
| strict_domain_scaffold_ood:source_ood:0.0_to_lt_0.4 | delta_domain_macro_rmse | -0.010 [-0.025, +0.022] | interval_includes_zero | 814 | 439 |
| strict_domain_scaffold_ood:source_ood:0.4_to_lt_0.6 | delta_domain_macro_spearman | -0.223 [-0.318, -0.098] | unfavourable_interval_below_zero | 432 | 182 |
| strict_domain_scaffold_ood:source_ood:0.4_to_lt_0.6 | delta_domain_macro_rmse | -0.046 [-0.065, -0.028] | unfavourable_interval_below_zero | 432 | 182 |
| strict_domain_scaffold_ood:source_ood:0.6_to_lt_0.8 | delta_domain_macro_spearman | -0.091 [-0.249, -0.002] | unfavourable_interval_below_zero | 207 | 86 |
| strict_domain_scaffold_ood:source_ood:0.6_to_lt_0.8 | delta_domain_macro_rmse | -0.021 [-0.047, +0.013] | interval_includes_zero | 207 | 86 |
| strict_domain_scaffold_ood:source_ood:0.8_to_1.000001 | delta_domain_macro_spearman | +0.127 [+0.023, +0.363] | favourable_interval_above_zero | 67 | 42 |
| strict_domain_scaffold_ood:source_ood:0.8_to_1.000001 | delta_domain_macro_rmse | +0.076 [+0.027, +0.115] | favourable_interval_above_zero | 67 | 42 |
| strict_domain_scaffold_ood:target_ood:0.0_to_lt_0.4 | delta_domain_macro_spearman | +0.013 [interval not estimable] | not_estimable | 744 | 384 |
| strict_domain_scaffold_ood:target_ood:0.0_to_lt_0.4 | delta_domain_macro_rmse | -0.015 [interval not estimable] | not_estimable | 744 | 384 |
| strict_domain_scaffold_ood:target_ood:0.4_to_lt_0.6 | delta_domain_macro_spearman | -0.101 [-0.215, -0.010] | unfavourable_interval_below_zero | 454 | 212 |
| strict_domain_scaffold_ood:target_ood:0.4_to_lt_0.6 | delta_domain_macro_rmse | +0.000 [-0.014, +0.017] | interval_includes_zero | 454 | 212 |
| strict_domain_scaffold_ood:target_ood:0.6_to_lt_0.8 | delta_domain_macro_spearman | -0.062 [-0.173, +0.071] | interval_includes_zero | 77 | 31 |
| strict_domain_scaffold_ood:target_ood:0.6_to_lt_0.8 | delta_domain_macro_rmse | -0.046 [-0.076, +0.004] | interval_includes_zero | 77 | 31 |
| strict_domain_scaffold_ood:target_ood:0.8_to_1.000001 | delta_domain_macro_spearman | -0.001 [-0.182, +0.111] | interval_includes_zero | 15 | 9 |
| strict_domain_scaffold_ood:target_ood:0.8_to_1.000001 | delta_domain_macro_rmse | +0.015 [-0.018, +0.078] | interval_includes_zero | 15 | 9 |

## Conditional label-randomization controls

| Estimand | Metric | Tail | Observed | Null mean | Null 95% range | Empirical p |
|---|---|---|---:|---:|---:|---:|
| chemistry_extra_trees | spearman | greater | 0.566 | 0.384 | [0.361, 0.412] | 0.010 |
| chemistry_extra_trees | rmse | smaller | 0.832 | 0.908 | [0.899, 0.916] | 0.010 |
| chemistry_extra_trees | mae | smaller | 0.612 | 0.695 | [0.688, 0.703] | 0.010 |
| full_context_extra_trees | spearman | greater | 0.621 | 0.426 | [0.403, 0.452] | 0.010 |
| full_context_extra_trees | rmse | smaller | 0.792 | 0.892 | [0.881, 0.901] | 0.010 |
| full_context_extra_trees | mae | smaller | 0.580 | 0.676 | [0.669, 0.684] | 0.010 |
| full_minus_chemistry | delta_spearman | greater | 0.055 | 0.042 | [0.027, 0.057] | 0.059 |
| full_minus_chemistry | delta_rmse | greater | 0.040 | 0.016 | [0.011, 0.021] | 0.010 |
| full_minus_chemistry | delta_mae | greater | 0.031 | 0.019 | [0.014, 0.024] | 0.010 |

## Endpoint-selection boundary

- Parsed records: 3,117.
- Relation counts: equality=1,825, range=535, one_sided=731, missing_or_unparsed=26.
- Mean singleton-row fraction in conditional permutation fit folds: 2.663%.
- Mean changed-label fraction in conditional permutation fit folds: 91.199%.

## Claim boundary

- All analyses are post-hoc sensitivities, diagnostics or repeated negative controls.
- Intervals are conditional on the frozen observations and fixed OOD domains.
- No result is prospective validation, calibrated absolute prediction or mechanistic evidence.
- No publication-eligible independent prospective dataset was available.
