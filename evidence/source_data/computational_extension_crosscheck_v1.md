# Computational extension independent cross-check v1

Overall status: **PASS**

Checks: 54/54 passed.

| Check | Status | Detail |
|---|---:|---|
| `core_sha256` | PASS | observed=e9494246bcbd198bc09d5b0d09721910c33f70f46ae859b76377a13775e3e4b2 |
| `parsed_sha256` | PASS | observed=ae1a16613b41ce84f92f539dbd31f16ada272ac5e401250acc7e97fd5e8bd970 |
| `master_protocol_sha256` | PASS | observed=7dee28f45e3ddf8d6299622a8e494c50492b3b2614f39ce8b2fdd2a1be754ff8 |
| `float32_correction_sha256` | PASS | observed=75a8b0e4f894cead9d72039eab9e6162a89d2a20c79afceda8a2aae1ffd78f3f |
| `context_v2_protocol_sha256` | PASS | observed=07ab2cc9a85f9fffd55974fe0b0b7a040dbb3dbc8b5cc99ac409a504faf712ad |
| `scaffold_v3_protocol_sha256` | PASS | observed=5e67fe833b3b5d3f87d38ac3b28b6b7ac2600aebb2756676e031aed7bd2aab83 |
| `null_v3_protocol_sha256` | PASS | observed=84178cde0edfc0ff31be3d7e12135351f95a2578b9ac69b61d9a5d517caa0dc5 |
| `publication_lineage_frozen` | PASS | analysis_identity=post_hoc_extension_float32_v4 |
| `context_weight_exact_directory_and_supersession_guard` | PASS | directory=./reports/post_hoc_context_weight_sensitivity_v2; expected=post_hoc_context_weight_sensitivity_v2 |
| `context_weight_artifact_inventory_rowwise` | PASS | rows=26; sha256=6554790882c515632480dfbc6a560ff3d93235111da1990f2a847b4e05ca42d8 |
| `context_weight_formal_provenance_files` | PASS | all present |
| `context_weight_formal_manifest_identity` | PASS | protocol=post_hoc_context_weight_sensitivity_v2.0; qa=PASS |
| `scaffold_source_exact_directory_and_supersession_guard` | PASS | directory=./reports/post_hoc_scaffold_source_sensitivity_v3; expected=post_hoc_scaffold_source_sensitivity_v3 |
| `scaffold_source_artifact_inventory_rowwise` | PASS | rows=29; sha256=55f38a3ba956f2a7979e1bc42ad7149896d3ac1b6b83218192f89cda24fb9cc7 |
| `scaffold_source_formal_provenance_files` | PASS | all present |
| `scaffold_source_formal_manifest_identity` | PASS | protocol=post_hoc_scaffold_source_sensitivity_v3.0; qa=PASS |
| `null_applicability_censoring_exact_directory_and_supersession_guard` | PASS | directory=./reports/post_hoc_null_applicability_censoring_v3; expected=post_hoc_null_applicability_censoring_v3 |
| `null_applicability_censoring_artifact_inventory_rowwise` | PASS | rows=28; sha256=0f51b938d00730eb2b6f3596f180b63a5ab89c2c0f0d216ea584503ec1768e2d |
| `null_applicability_censoring_formal_provenance_files` | PASS | all present |
| `null_applicability_censoring_formal_manifest_identity` | PASS | protocol=post_hoc_null_applicability_censoring_v3.0; qa=PASS |
| `formal_source_tables_exact_schema_dtype_and_counts` | PASS | all 20 exact |
| `context_weight_upstream_qa` | PASS | status=PASS |
| `scaffold_source_upstream_qa` | PASS | status=PASS |
| `null_applicability_censoring_upstream_qa` | PASS | status=PASS |
| `context_required_artifacts` | PASS | all present |
| `context_formal_configuration` | PASS | formal_post_hoc_sensitivity |
| `context_internal_model_set` | PASS | models=['chemistry_extra_trees__compound_equal', 'chemistry_extra_trees__domain_balanced_clipped', 'chemistry_extra_trees__uniform', 'full_context_extra_trees__compound_equal', 'full_context_extra_trees__domain_balanced_clipped', 'full_context_extra_trees__uniform'] |
| `context_internal_prediction_shape` | PASS | rows=9360; models=6 |
| `context_internal_metrics_independent_recompute` | PASS | max_abs_difference=1.11e-16 |
| `context_25_scaffold_disjoint_splits` | PASS | splits=25; overlap_columns=['n_scaffold_overlap'] |
| `context_bootstrap_counts` | PASS | internal=[np.int64(10000)]; ood=[np.int64(10000)] |
| `portable_ood_macro_independent_recompute` | PASS | max_abs_difference=1.11e-16 |
| `scaffold_source_required_artifacts` | PASS | all present |
| `scaffold_source_formal_configuration` | PASS | args={'mode': 'both', 'data_file': './data/processed/all_molglue_dc50_qc_train_test_standardized_context.csv', 'output_dir': 'reports/post_hoc_scaffold_source_sensitivity_v3', 'identity_archive_dir': './reports/confirmatory_ood_cpu_v1', 'outer_folds': 5, 'outer_repeats': 5, 'inner_folds': 4, 'n_estimators': 600, 'n_jobs': 4, 'bootstrap_replicates': 10000, 'seed': 260531, 'max_outer_splits': None, 'max_targets': None, 'max_deletions': None, 'only_target': None, 'baseline_only': False, 'overwrite': True, 'resume': False} |
| `scaffold_source_v3_native_thread_and_identity_gate` | PASS | thread=PASS; identity=PASS |
| `generic_scaffold_metrics_independent_recompute` | PASS | max_abs_difference=1.11e-16 |
| `generic_25_group_disjoint_splits` | PASS | splits=25; overlap_columns=['n_group_overlap'] |
| `source_deletion_macro_independent_recompute` | PASS | max_abs_difference=1.11e-16 |
| `source_deletion_condition_completeness` | PASS | metric_rows=80; split_rows=40; targets=8 |
| `source_deletion_all_8_confirmatory_prediction_identity` | PASS | all 8 targets, both models, exact row/parameter/y_true/y_pred identity |
| `scaffold_source_bootstrap_counts` | PASS | generic=[np.int64(10000)]; source=[np.int64(10000)] |
| `null_applicability_required_artifacts` | PASS | all present |
| `null_formal_configuration` | PASS | config={'n_permutations': 100, 'n_estimators': 600, 'n_jobs': 4, 'bootstrap_replicates': 10000, 'max_repeats': 5, 'outer_folds_per_repeat': 5, 'seed': 260531, 'similarity_edges': [0.0, 0.4, 0.6, 0.8, 1.000001], 'models': ['chemistry_extra_trees', 'full_context_extra_trees'], 'training_response_dtype': 'float32', 'applicability_domain_macro_bootstrap': 'global scaffold clusters; metric-specific frozen eligible domain set; incomplete fixed-domain replicates are NA', 'applicability_minimum_valid_bootstrap': 'max(100, ceil(0.50 * requested_replicates))', 'applicability_sparse_domain_rule': 'domain-macro CI is non-estimable when any fixed eligible domain has fewer than two unique global scaffolds', 'label_permutation_strata': 'source_database × target_protein within each outer fit fold', 'hyperparameters': 'reuse corresponding original-data outer-fold selections; no retuning'} |
| `null_locked_environment_identity` | PASS | python=3.10.19 \| packaged by conda-forge \| (main, Oct 22 2025, 22:29:10) [GCC 14.3.0]; scikit_learn=1.7.2; rdkit=2025.09.4 |
| `permutation_id_completeness` | PASS | models=100; contrasts=100 |
| `permutation_table_shape_and_uniqueness` | PASS | model_rows=200; contrast_rows=100; inference_rows=9 |
| `permutation_fold_diagnostic_completeness` | PASS | ids=100; rows=2500 |
| `permutation_shard_merge_provenance` | PASS | status=PASS; scientific_change=False; shards=3; snapshot=reports/post_hoc_null_applicability_censoring_v3/premerge_primary_snapshot_v3 |
| `permutation_empirical_p_independent_recompute` | PASS | max_abs_difference=1.73e-18 |
| `applicability_domain_macro_independent_recompute` | PASS | max_abs_difference=2.22e-16 |
| `applicability_strata_and_bootstrap_completeness` | PASS | models=40; contrasts=20 |
| `censoring_total_and_classes` | PASS | total=3117; classes=['equality', 'missing_or_unparsed', 'one_sided', 'range'] |
| `aggregate_source_exact_hash_and_lineage_binding` | PASS | analysis=post_hoc_extension_float32_v4; index=8cd3acaf120fe09bc1b92b894d6e8b365f3af550b1f165ccad31b459324bc3f4; checksums=103276a4b71ab80d6db41447faf955a749b2a8f69f0099b55de5328ecd5e71b4; provenance=b67e1f3581d536bca09650a3afeafa09a6dfba2c2f969809041a539528039bc5 |
| `computational_extension_results_independent_recompute` | PASS | sha256=188f23ace005d3121ee2f8f6adc7ff964fa16c2d360b67ca3f79b2ad07c4b302 |

This verifier independently recomputes central metrics from stored predictions or domain-level records. It does not re-estimate bootstrap intervals.
