#!/usr/bin/env python3
"""Build publication-safe aggregate source data for the post-hoc extensions.

The builder deliberately excludes row-level predictions, molecular
structures, identifiers and split assignments. It reads only completed,
QA-passing formal extension outputs and emits aggregate tables suitable for
figure generation and manuscript evidence checks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from computational_extension_lineage_v1 import (
    ANALYSIS_IDENTITY,
    FLOAT32_CORRECTION_SHA256,
    MASTER_PROTOCOL_SHA256,
    OUTPUT_NAMES,
    PROVENANCE_CONTRACT_SHA256,
    PROVENANCE_FILENAME,
    PROVENANCE_VERSION,
    SOURCE_FILENAMES,
    TABLE_WORKFLOW,
    WORKFLOW_IDENTITIES as CENTRAL_WORKFLOW_IDENTITIES,
    downstream_software_bindings,
    require_exact_columns,
    require_exact_integer_column,
    require_finite_numeric,
    require_publication_lineage_frozen,
    require_strict_boolean_column,
    validate_artifact_inventory,
    validate_aggregate_value_binding,
    validate_scaffold_v3_execution,
    workflow_directory,
    workflow_protocol_path,
    workflow_runner_path,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ROUTE_DIR = SCRIPT_DIR.parent
REPORT_DIR = ROUTE_DIR / "reports"
CONTEXT_DIR = workflow_directory("context_weight")
SCAFFOLD_DIR = workflow_directory("scaffold_source")
NULL_DIR = workflow_directory("null_applicability_censoring")
OUTPUT_DIR = REPORT_DIR / "publication_validation_v1"

MASTER_PROTOCOL = (
    ROUTE_DIR
    / "docs"
    / "post_hoc_computational_extension_master_protocol_v1.md"
)
EXPECTED_MASTER_SHA256 = MASTER_PROTOCOL_SHA256
CORRECTION_DOCUMENT = (
    ROUTE_DIR / "docs" / "post_hoc_float32_response_correction_v2.md"
)
EXPECTED_CORRECTION_SHA256 = FLOAT32_CORRECTION_SHA256
WORKFLOW_IDENTITIES = {
    workflow: {
        **identity,
        "directory": workflow_directory(workflow),
        "protocol_path": workflow_protocol_path(workflow),
        "runner_path": workflow_runner_path(workflow),
    }
    for workflow, identity in CENTRAL_WORKFLOW_IDENTITIES.items()
}

INPUTS: dict[str, tuple[Path, set[str]]] = {
    "portable_context": (
        CONTEXT_DIR / "ood_paired_scaffold_bootstrap.csv",
        {
            "record_type",
            "protocol",
            "model_id",
            "contrast_id",
            "domain_macro_spearman_observed",
            "domain_macro_spearman_ci_low",
            "domain_macro_spearman_ci_high",
            "delta_domain_macro_spearman_observed",
            "delta_domain_macro_spearman_ci_low",
            "delta_domain_macro_spearman_ci_high",
            "n_bootstrap",
        },
    ),
    "weighting_bootstrap": (
        CONTEXT_DIR / "internal_paired_scaffold_bootstrap.csv",
        {
            "record_type",
            "model_id",
            "contrast_id",
            "spearman_observed",
            "spearman_ci_low",
            "spearman_ci_high",
            "delta_spearman_observed",
            "delta_spearman_ci_low",
            "delta_spearman_ci_high",
            "n_bootstrap",
        },
    ),
    "weighting_diagnostics": (
        CONTEXT_DIR / "internal_weight_diagnostics.csv",
        {
            "weight_mode",
            "fit_stage",
            "n_fit_rows",
            "clipping_fraction",
            "effective_sample_size",
            "effective_sample_fraction",
        },
    ),
    "generic_scaffold": (
        SCAFFOLD_DIR / "generic_scaffold_paired_cluster_bootstrap.csv",
        {
            "record_type",
            "model_id",
            "contrast_id",
            "metric",
            "observed",
            "ci_low",
            "ci_high",
            "n_bootstrap",
            "n_clusters",
        },
    ),
    "generic_scaffold_groups": (
        SCAFFOLD_DIR / "generic_scaffold_group_summary.csv",
        {
            "n_rows",
            "n_unique_smiles",
            "n_bemis_murcko_groups",
            "n_generic_murcko_groups",
            "n_singleton_generic_groups",
            "largest_generic_group_rows",
        },
    ),
    "source_deletion": (
        SCAFFOLD_DIR / "source_deletion_paired_cluster_bootstrap.csv",
        {
            "record_type",
            "contrast_type",
            "deletion_source",
            "model_id",
            "heldout_target",
            "metric",
            "observed",
            "ci_low",
            "ci_high",
            "n_bootstrap",
            "n_global_scaffold_clusters",
        },
    ),
    "applicability_models": (
        NULL_DIR / "applicability_model_metrics.csv",
        {
            "regime",
            "protocol",
            "similarity_bin",
            "model_id",
            "n_rows",
            "n_unique_smiles",
            "n_scaffolds",
            "n_heldout_groups",
            "n_domains_expected",
            "n_domains_total",
            "spearman",
            "rmse",
            "mae",
            "spearman_non_estimable_reason",
            "rmse_non_estimable_reason",
            "mae_non_estimable_reason",
            "domain_macro_spearman",
            "domain_macro_spearman_finite_domains",
            "domain_macro_rmse",
            "domain_macro_rmse_finite_domains",
            "domain_macro_mae",
            "domain_macro_mae_finite_domains",
            "domain_macro_non_estimable_reason",
        },
    ),
    "applicability_domains": (
        NULL_DIR / "applicability_domain_metrics.csv",
        {
            "regime",
            "protocol",
            "similarity_bin",
            "heldout_group",
            "model_id",
            "n_rows",
            "n_unique_smiles",
            "n_scaffolds",
            "spearman",
            "rmse",
            "mae",
            "spearman_non_estimable_reason",
            "rmse_non_estimable_reason",
            "mae_non_estimable_reason",
        },
    ),
    "applicability_contrasts": (
        NULL_DIR / "applicability_paired_deltas.csv",
        {
            "regime",
            "protocol",
            "similarity_bin",
            "contrast_id",
            "n_rows",
            "n_unique_smiles",
            "n_scaffolds",
            "n_heldout_groups",
            "n_domains_expected",
            "n_domains_total",
            "delta_spearman",
            "delta_spearman_ci_low",
            "delta_spearman_ci_high",
            "delta_rmse",
            "delta_rmse_ci_low",
            "delta_rmse_ci_high",
            "delta_mae",
            "delta_mae_ci_low",
            "delta_mae_ci_high",
            "delta_domain_macro_spearman",
            "delta_domain_macro_spearman_finite_domains",
            "delta_domain_macro_spearman_n_bootstrap_finite",
            "delta_domain_macro_spearman_ci_low",
            "delta_domain_macro_spearman_ci_high",
            "delta_domain_macro_rmse",
            "delta_domain_macro_rmse_finite_domains",
            "delta_domain_macro_rmse_n_bootstrap_finite",
            "delta_domain_macro_rmse_ci_low",
            "delta_domain_macro_rmse_ci_high",
            "delta_domain_macro_mae",
            "delta_domain_macro_mae_finite_domains",
            "delta_domain_macro_mae_n_bootstrap_finite",
            "delta_domain_macro_mae_ci_low",
            "delta_domain_macro_mae_ci_high",
            "n_bootstrap_requested",
            "bootstrap_non_estimable_reason",
        },
    ),
    "permutation_inference": (
        NULL_DIR / "permutation_null_inference.csv",
        {
            "record_type",
            "estimand_id",
            "metric",
            "observed",
            "null_mean",
            "null_percentile_2p5",
            "null_percentile_97p5",
            "n_permutations_finite",
            "empirical_tail",
            "empirical_p",
        },
    ),
    "permutation_models": (
        NULL_DIR / "permutation_model_metrics.csv",
        {
            "permutation_id",
            "model_id",
            "spearman",
            "rmse",
            "mae",
        },
    ),
    "permutation_contrasts": (
        NULL_DIR / "permutation_contrast_metrics.csv",
        {
            "permutation_id",
            "contrast_id",
            "delta_spearman",
            "delta_rmse",
            "delta_mae",
        },
    ),
    "permutation_diagnostics": (
        NULL_DIR / "permutation_fold_diagnostics.csv",
        {
            "permutation_id",
            "repeat",
            "outer_fold",
            "permutation_seed",
            "n_train",
            "n_test",
            "permutation_n_strata",
            "permutation_n_singleton_strata",
            "permutation_singleton_row_fraction",
            "permutation_changed_label_fraction",
            "permutation_label_pearson",
            "within_stratum_multiset_preserved",
        },
    ),
    "censoring_overall": (
        NULL_DIR / "censoring_relation_overall.csv",
        {"relation_class", "n_records", "fraction_within_group"},
    ),
    "censoring_by_source": (
        NULL_DIR / "censoring_relation_by_source.csv",
        {
            "source_database",
            "relation_class",
            "n_records",
            "fraction_within_group",
        },
    ),
    "censoring_by_target": (
        NULL_DIR / "censoring_relation_by_target.csv",
        {
            "target_protein",
            "relation_class",
            "n_records",
            "fraction_within_group",
        },
    ),
    "censoring_overlap": (
        NULL_DIR / "censoring_overlap_with_core.csv",
        {
            "relation_class",
            "n_records",
            "n_unique_compounds",
            "n_unique_scaffolds",
            "fraction_unique_compounds_in_core",
            "fraction_unique_scaffolds_in_core",
        },
    ),
    "censoring_operator_counts": (
        NULL_DIR / "censoring_relation_operator_counts.csv",
        {
            "relation_operator",
            "relation_class",
            "n_records",
            "fraction_all_records",
        },
    ),
    "censoring_selection_flow": (
        NULL_DIR / "selection_flow_overall.csv",
        {"stage_order", "stage", "n", "unit", "note"},
    ),
    "censoring_source_selection": (
        NULL_DIR / "selection_representation_by_source.csv",
        {
            "source_database",
            "n_parsed_records",
            "n_equality_records",
            "n_range_records",
            "n_one_sided_records",
            "n_missing_or_unparsed_records",
            "n_parsed_unique_record_ids",
            "n_core_unique_component_record_ids",
            "n_core_aggregated_rows_with_source_token",
            "note",
        },
    ),
}


def _columns(value: str) -> tuple[str, ...]:
    return tuple(value.split(","))


# Exact ordered source schemas.  These are intentionally broader than the
# publication projection for weighting diagnostics, whose split/fold and
# fit-index fields are verified upstream but never copied into source data.
SOURCE_COLUMN_ALLOWLISTS = {
    "portable_context": _columns(
        "record_type,protocol,model_id,cluster_unit,n_clusters,n_domains,"
        "n_bootstrap,domain_macro_spearman_observed,"
        "domain_macro_spearman_bootstrap_mean,domain_macro_spearman_ci_low,"
        "domain_macro_spearman_ci_high,domain_macro_spearman_n_valid,"
        "domain_macro_rmse_observed,domain_macro_rmse_bootstrap_mean,"
        "domain_macro_rmse_ci_low,domain_macro_rmse_ci_high,"
        "domain_macro_rmse_n_valid,pooled_spearman_observed,"
        "pooled_spearman_bootstrap_mean,pooled_spearman_ci_low,"
        "pooled_spearman_ci_high,pooled_spearman_n_valid,"
        "pooled_rmse_observed,pooled_rmse_bootstrap_mean,pooled_rmse_ci_low,"
        "pooled_rmse_ci_high,pooled_rmse_n_valid,contrast_id,first_model,"
        "comparator_model,delta_direction,delta_domain_macro_spearman_observed,"
        "delta_domain_macro_spearman_bootstrap_mean,"
        "delta_domain_macro_spearman_ci_low,"
        "delta_domain_macro_spearman_ci_high,"
        "delta_domain_macro_spearman_n_valid,"
        "delta_domain_macro_rmse_observed,"
        "delta_domain_macro_rmse_bootstrap_mean,"
        "delta_domain_macro_rmse_ci_low,delta_domain_macro_rmse_ci_high,"
        "delta_domain_macro_rmse_n_valid,delta_pooled_spearman_observed,"
        "delta_pooled_spearman_bootstrap_mean,delta_pooled_spearman_ci_low,"
        "delta_pooled_spearman_ci_high,delta_pooled_spearman_n_valid,"
        "delta_pooled_rmse_observed,delta_pooled_rmse_bootstrap_mean,"
        "delta_pooled_rmse_ci_low,delta_pooled_rmse_ci_high,"
        "delta_pooled_rmse_n_valid"
    ),
    "weighting_bootstrap": _columns(
        "record_type,protocol,model_id,cluster_unit,n_clusters,n_bootstrap,"
        "spearman_observed,spearman_bootstrap_mean,spearman_ci_low,"
        "spearman_ci_high,rmse_observed,rmse_bootstrap_mean,rmse_ci_low,"
        "rmse_ci_high,reference_model,delta_direction,"
        "delta_spearman_observed,delta_spearman_mean,delta_spearman_ci_low,"
        "delta_spearman_ci_high,delta_rmse_observed,delta_rmse_mean,"
        "delta_rmse_ci_low,delta_rmse_ci_high,contrast_id,first_model,"
        "comparator_model"
    ),
    "weighting_diagnostics": _columns(
        "protocol,repeat,outer_fold,fit_stage,inner_fold,fit_indices_sha256,"
        "applies_to_models,weight_mode,n_fit_rows,n_weight_groups,weight_min,"
        "weight_max,weight_mean,weight_sd,pre_clip_weight_max,clip_threshold,"
        "post_clip_prenorm_mean,post_clip_prenorm_max,clipping_fraction,"
        "effective_sample_size,effective_sample_fraction,"
        "compound_total_relative_range"
    ),
    "generic_scaffold": _columns(
        "record_type,model_id,contrast_id,metric,observed,bootstrap_mean,"
        "ci_low,ci_high,n_valid,n_bootstrap,cluster_unit,n_clusters,"
        "positive_direction"
    ),
    "generic_scaffold_groups": _columns(
        "n_rows,n_unique_smiles,n_bemis_murcko_groups,n_generic_murcko_groups,"
        "n_singleton_generic_groups,largest_generic_group_rows,"
        "median_generic_group_rows,generic_definition"
    ),
    "source_deletion": _columns(
        "record_type,contrast_type,deletion_source,model_id,heldout_target,"
        "metric,observed,bootstrap_mean,ci_low,ci_high,n_valid,n_bootstrap,"
        "cluster_unit,n_global_scaffold_clusters,positive_direction"
    ),
    "applicability_models": _columns(
        "regime,protocol,similarity_bin,similarity_bin_left,"
        "similarity_bin_right_exclusive,model_id,n_rows,n_unique_smiles,"
        "n_scaffolds,n_heldout_groups,n_domains_expected,n_domains_total,"
        "spearman,rmse,mae,spearman_non_estimable_reason,"
        "rmse_non_estimable_reason,mae_non_estimable_reason,"
        "domain_macro_spearman,domain_macro_spearman_finite_domains,"
        "domain_macro_rmse,domain_macro_rmse_finite_domains,domain_macro_mae,"
        "domain_macro_mae_finite_domains,domain_macro_non_estimable_reason"
    ),
    "applicability_domains": _columns(
        "regime,protocol,similarity_bin,heldout_group,model_id,n_rows,"
        "n_unique_smiles,n_scaffolds,spearman,rmse,mae,"
        "spearman_non_estimable_reason,rmse_non_estimable_reason,"
        "mae_non_estimable_reason"
    ),
    "applicability_contrasts": _columns(
        "regime,protocol,similarity_bin,contrast_id,n_rows,n_unique_smiles,"
        "n_scaffolds,n_heldout_groups,n_domains_expected,"
        "n_bootstrap_requested,delta_definition,direction,delta_spearman,"
        "delta_rmse,delta_mae,n_domains_total,represented_domains_json,"
        "represented_domains_sha256,delta_domain_macro_spearman,"
        "delta_domain_macro_spearman_finite_domains,"
        "delta_domain_macro_spearman_n_domains_eligible,"
        "delta_domain_macro_spearman_eligible_domains_json,"
        "delta_domain_macro_spearman_eligible_domains_sha256,"
        "delta_domain_macro_spearman_n_sparse_eligible_domains,"
        "delta_domain_macro_spearman_sparse_eligible_domains_json,"
        "delta_domain_macro_spearman_sparse_eligible_domains_sha256,"
        "delta_domain_macro_rmse,delta_domain_macro_rmse_finite_domains,"
        "delta_domain_macro_rmse_n_domains_eligible,"
        "delta_domain_macro_rmse_eligible_domains_json,"
        "delta_domain_macro_rmse_eligible_domains_sha256,"
        "delta_domain_macro_rmse_n_sparse_eligible_domains,"
        "delta_domain_macro_rmse_sparse_eligible_domains_json,"
        "delta_domain_macro_rmse_sparse_eligible_domains_sha256,"
        "delta_domain_macro_mae,delta_domain_macro_mae_finite_domains,"
        "delta_domain_macro_mae_n_domains_eligible,"
        "delta_domain_macro_mae_eligible_domains_json,"
        "delta_domain_macro_mae_eligible_domains_sha256,"
        "delta_domain_macro_mae_n_sparse_eligible_domains,"
        "delta_domain_macro_mae_sparse_eligible_domains_json,"
        "delta_domain_macro_mae_sparse_eligible_domains_sha256,"
        "delta_spearman_n_bootstrap_finite,delta_spearman_ci_low,"
        "delta_spearman_ci_high,delta_spearman_non_estimable_reason,"
        "delta_rmse_n_bootstrap_finite,delta_rmse_ci_low,delta_rmse_ci_high,"
        "delta_rmse_non_estimable_reason,delta_mae_n_bootstrap_finite,"
        "delta_mae_ci_low,delta_mae_ci_high,"
        "delta_mae_non_estimable_reason,"
        "delta_domain_macro_spearman_n_bootstrap_requested,"
        "delta_domain_macro_spearman_n_bootstrap_finite,"
        "delta_domain_macro_spearman_n_bootstrap_valid,"
        "delta_domain_macro_spearman_n_bootstrap_invalid_missing_domain,"
        "delta_domain_macro_spearman_n_bootstrap_invalid_metric_nonestimable,"
        "delta_domain_macro_spearman_n_bootstrap_not_attempted_structural,"
        "delta_domain_macro_spearman_minimum_valid_required,"
        "delta_domain_macro_spearman_ci_low,"
        "delta_domain_macro_spearman_ci_high,"
        "delta_domain_macro_spearman_ci_status,"
        "delta_domain_macro_spearman_non_estimable_reason,"
        "delta_domain_macro_rmse_n_bootstrap_requested,"
        "delta_domain_macro_rmse_n_bootstrap_finite,"
        "delta_domain_macro_rmse_n_bootstrap_valid,"
        "delta_domain_macro_rmse_n_bootstrap_invalid_missing_domain,"
        "delta_domain_macro_rmse_n_bootstrap_invalid_metric_nonestimable,"
        "delta_domain_macro_rmse_n_bootstrap_not_attempted_structural,"
        "delta_domain_macro_rmse_minimum_valid_required,"
        "delta_domain_macro_rmse_ci_low,delta_domain_macro_rmse_ci_high,"
        "delta_domain_macro_rmse_ci_status,"
        "delta_domain_macro_rmse_non_estimable_reason,"
        "delta_domain_macro_mae_n_bootstrap_requested,"
        "delta_domain_macro_mae_n_bootstrap_finite,"
        "delta_domain_macro_mae_n_bootstrap_valid,"
        "delta_domain_macro_mae_n_bootstrap_invalid_missing_domain,"
        "delta_domain_macro_mae_n_bootstrap_invalid_metric_nonestimable,"
        "delta_domain_macro_mae_n_bootstrap_not_attempted_structural,"
        "delta_domain_macro_mae_minimum_valid_required,"
        "delta_domain_macro_mae_ci_low,delta_domain_macro_mae_ci_high,"
        "delta_domain_macro_mae_ci_status,"
        "delta_domain_macro_mae_non_estimable_reason,"
        "bootstrap_non_estimable_reason"
    ),
    "permutation_inference": _columns(
        "record_type,estimand_id,metric,observed,null_mean,null_sd,"
        "null_percentile_2p5,null_percentile_97p5,n_permutations_finite,"
        "empirical_tail,empirical_p"
    ),
    "permutation_models": _columns(
        "permutation_id,model_id,n_rows,n_repeats,spearman,rmse,mae"
    ),
    "permutation_contrasts": _columns(
        "permutation_id,contrast_id,n_rows,n_repeats,delta_spearman,"
        "delta_rmse,delta_mae"
    ),
    "permutation_diagnostics": _columns(
        "permutation_id,repeat,outer_fold,permutation_seed,n_train,n_test,"
        "permutation_n_strata,permutation_n_singleton_strata,"
        "permutation_singleton_row_fraction,"
        "permutation_changed_label_fraction,permutation_label_pearson,"
        "within_stratum_multiset_preserved"
    ),
    "censoring_overall": _columns(
        "relation_class,n_records,group_total_records,fraction_within_group"
    ),
    "censoring_by_source": _columns(
        "source_database,relation_class,n_records,group_total_records,"
        "fraction_within_group"
    ),
    "censoring_by_target": _columns(
        "target_protein,relation_class,n_records,group_total_records,"
        "fraction_within_group"
    ),
    "censoring_overlap": _columns(
        "relation_class,n_records,n_structure_derivable_records,"
        "n_missing_or_invalid_structure_records,n_unique_compounds,"
        "n_unique_scaffolds,n_records_with_compound_in_core,"
        "fraction_records_with_compound_in_core,n_unique_compounds_in_core,"
        "fraction_unique_compounds_in_core,n_records_with_scaffold_in_core,"
        "fraction_records_with_scaffold_in_core,n_unique_scaffolds_in_core,"
        "fraction_unique_scaffolds_in_core"
    ),
    "censoring_operator_counts": _columns(
        "relation_operator,relation_class,n_records,fraction_all_records"
    ),
    "censoring_selection_flow": _columns(
        "stage_order,stage,n,unit,note"
    ),
    "censoring_source_selection": _columns(
        "source_database,n_parsed_records,n_equality_records,n_range_records,"
        "n_one_sided_records,n_missing_or_unparsed_records,"
        "n_parsed_unique_record_ids,n_core_unique_component_record_ids,"
        "n_core_aggregated_rows_with_source_token,note"
    ),
}

PUBLISH_COLUMN_ALLOWLISTS = dict(SOURCE_COLUMN_ALLOWLISTS)
PUBLISH_COLUMN_ALLOWLISTS["weighting_diagnostics"] = (
    "weight_mode",
    "fit_stage",
    "n_fit_rows",
    "clipping_fraction",
    "effective_sample_size",
    "effective_sample_fraction",
)

EXPECTED_ROW_COUNTS = {
    "portable_context": 12,
    "weighting_bootstrap": 13,
    "weighting_diagnostics": 250,
    "generic_scaffold": 6,
    "generic_scaffold_groups": 1,
    "source_deletion": 254,
    "applicability_models": 40,
    "applicability_domains": 162,
    "applicability_contrasts": 20,
    "permutation_inference": 9,
    "permutation_models": 200,
    "permutation_contrasts": 100,
    "permutation_diagnostics": 2_500,
    "censoring_overall": 4,
    "censoring_by_source": 16,
    "censoring_by_target": 296,
    "censoring_overlap": 5,
    "censoring_operator_counts": 7,
    "censoring_selection_flow": 7,
    "censoring_source_selection": 4,
}

CATEGORICAL_COLUMNS = {
    "portable_context": {
        "record_type", "protocol", "model_id", "cluster_unit", "contrast_id",
        "first_model", "comparator_model", "delta_direction",
    },
    "weighting_bootstrap": {
        "record_type", "protocol", "model_id", "cluster_unit",
        "reference_model", "delta_direction", "contrast_id", "first_model",
        "comparator_model",
    },
    "weighting_diagnostics": {
        "protocol", "fit_stage", "fit_indices_sha256", "applies_to_models",
        "weight_mode",
    },
    "generic_scaffold": {
        "record_type", "model_id", "contrast_id", "metric", "cluster_unit",
        "positive_direction",
    },
    "generic_scaffold_groups": {"generic_definition"},
    "source_deletion": {
        "record_type", "contrast_type", "deletion_source", "model_id",
        "heldout_target", "metric", "cluster_unit", "positive_direction",
    },
    "applicability_models": {
        "regime", "protocol", "similarity_bin", "model_id",
        "spearman_non_estimable_reason", "rmse_non_estimable_reason",
        "mae_non_estimable_reason", "domain_macro_non_estimable_reason",
    },
    "applicability_domains": {
        "regime", "protocol", "similarity_bin", "heldout_group", "model_id",
        "spearman_non_estimable_reason", "rmse_non_estimable_reason",
        "mae_non_estimable_reason",
    },
    "applicability_contrasts": {
        "regime", "protocol", "similarity_bin", "contrast_id",
        "delta_definition", "direction", "represented_domains_json",
        "represented_domains_sha256",
        "delta_domain_macro_spearman_eligible_domains_json",
        "delta_domain_macro_spearman_eligible_domains_sha256",
        "delta_domain_macro_spearman_sparse_eligible_domains_json",
        "delta_domain_macro_spearman_sparse_eligible_domains_sha256",
        "delta_domain_macro_rmse_eligible_domains_json",
        "delta_domain_macro_rmse_eligible_domains_sha256",
        "delta_domain_macro_rmse_sparse_eligible_domains_json",
        "delta_domain_macro_rmse_sparse_eligible_domains_sha256",
        "delta_domain_macro_mae_eligible_domains_json",
        "delta_domain_macro_mae_eligible_domains_sha256",
        "delta_domain_macro_mae_sparse_eligible_domains_json",
        "delta_domain_macro_mae_sparse_eligible_domains_sha256",
        "delta_spearman_non_estimable_reason",
        "delta_rmse_non_estimable_reason",
        "delta_mae_non_estimable_reason",
        "delta_domain_macro_spearman_ci_status",
        "delta_domain_macro_spearman_non_estimable_reason",
        "delta_domain_macro_rmse_ci_status",
        "delta_domain_macro_rmse_non_estimable_reason",
        "delta_domain_macro_mae_ci_status",
        "delta_domain_macro_mae_non_estimable_reason",
        "bootstrap_non_estimable_reason",
    },
    "permutation_inference": {
        "record_type", "estimand_id", "metric", "empirical_tail",
    },
    "permutation_models": {"model_id"},
    "permutation_contrasts": {"contrast_id"},
    "permutation_diagnostics": set(),
    "censoring_overall": {"relation_class"},
    "censoring_by_source": {"source_database", "relation_class"},
    "censoring_by_target": {"target_protein", "relation_class"},
    "censoring_overlap": {"relation_class"},
    "censoring_operator_counts": {"relation_operator", "relation_class"},
    "censoring_selection_flow": {"stage", "unit", "note"},
    "censoring_source_selection": {"source_database", "note"},
}

NULLABLE_NUMERIC_COLUMNS = {
    "portable_context": set(SOURCE_COLUMN_ALLOWLISTS["portable_context"])
    - CATEGORICAL_COLUMNS["portable_context"]
    - {"n_clusters", "n_domains", "n_bootstrap"},
    "weighting_bootstrap": set(
        SOURCE_COLUMN_ALLOWLISTS["weighting_bootstrap"]
    )
    - CATEGORICAL_COLUMNS["weighting_bootstrap"]
    - {"n_clusters", "n_bootstrap"},
    "weighting_diagnostics": {
        "inner_fold", "clip_threshold", "compound_total_relative_range"
    },
    "applicability_models": {
        "domain_macro_spearman", "domain_macro_rmse", "domain_macro_mae"
    },
    "applicability_domains": {"spearman"},
    "applicability_contrasts": {
        "delta_domain_macro_spearman", "delta_domain_macro_rmse",
        "delta_domain_macro_mae", "delta_domain_macro_spearman_ci_low",
        "delta_domain_macro_spearman_ci_high",
        "delta_domain_macro_rmse_ci_low", "delta_domain_macro_rmse_ci_high",
        "delta_domain_macro_mae_ci_low", "delta_domain_macro_mae_ci_high",
    },
}

FORBIDDEN_COLUMNS = {
    "canonical_smiles",
    "smiles",
    "qc_id",
    "record_id",
    "row_index",
    "source_id",
    "y_true",
    "y_pred",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def require_value(
    observed: object,
    expected: object,
    *,
    label: str,
) -> None:
    if observed != expected:
        raise RuntimeError(
            f"{label} mismatch: observed={observed!r}, expected={expected!r}"
        )


def strict_integer(value: object, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be an integer, not bool")
    numeric = float(value)
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{label} must be a finite integer")
    return int(numeric)


def require_completed_qa(directory: Path) -> dict[str, Any]:
    qa_path = directory / "qa_summary.json"
    qa = read_json(qa_path)
    status = str(
        qa.get("overall_status", qa.get("status", ""))
    ).strip().upper()
    if status != "PASS":
        raise RuntimeError(f"Formal QA is not PASS: {qa_path} ({status})")
    return qa


def require_passed_check(
    qa: dict[str, Any],
    *check_ids: str,
) -> None:
    checks = qa.get("checks", [])
    by_id = {
        str(item.get("check_id", item.get("check", ""))): str(
            item.get("status", "")
        ).upper()
        for item in checks
        if isinstance(item, dict)
    }
    missing_or_failed = [
        check_id for check_id in check_ids if by_id.get(check_id) != "PASS"
    ]
    if missing_or_failed:
        raise RuntimeError(
            "Required upstream QA checks are absent or failed: "
            + ", ".join(missing_or_failed)
        )


def require_exact_workflow_directory(
    directory: Path,
    expected_name: str,
) -> None:
    if directory.name != expected_name:
        raise RuntimeError(
            f"Cross-version input rejected: {directory}; "
            f"expected directory basename {expected_name!r}"
        )
    if directory.is_symlink():
        raise RuntimeError(f"Symlinked workflow directory is not permitted: {directory}")
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    marker = directory / "SUPERSEDED_DO_NOT_USE.md"
    if marker.exists():
        raise RuntimeError(f"Superseded workflow input rejected: {marker}")


def require_formal_args(
    args: dict[str, Any],
    expected: dict[str, object],
    *,
    workflow: str,
) -> None:
    for key, value in expected.items():
        require_value(
            args.get(key),
            value,
            label=f"{workflow} formal argument {key}",
        )


def validate_context_workflow(
    directory: Path,
    identity: dict[str, object],
    qa: dict[str, Any],
) -> dict[str, Any]:
    config = read_json(directory / "configuration.json")
    manifest = read_json(directory / "run_manifest.json")
    protocol_version = str(identity["protocol_version"])
    protocol_path = Path(identity["protocol_path"])
    runner_path = Path(identity["runner_path"])
    require_value(
        config.get("protocol_version"),
        protocol_version,
        label="context protocol version",
    )
    require_value(
        manifest.get("protocol_version"),
        protocol_version,
        label="context manifest protocol version",
    )
    require_value(
        config.get("analysis_mode"),
        "formal_post_hoc_sensitivity",
        label="context analysis mode",
    )
    require_value(
        manifest.get("status"), "complete", label="context manifest status"
    )
    require_value(
        str(manifest.get("qa_status", "")).upper(),
        "PASS",
        label="context manifest QA status",
    )
    require_value(
        config.get("training_response_dtype"),
        "float32",
        label="context training-response dtype",
    )
    require_value(
        manifest.get("training_response_dtype"),
        "float32",
        label="context manifest training-response dtype",
    )
    require_value(
        config.get("correction_document_sha256"),
        EXPECTED_CORRECTION_SHA256,
        label="context correction identity",
    )
    correction = manifest.get("correction_document", {})
    require_value(
        correction.get("sha256") if isinstance(correction, dict) else None,
        EXPECTED_CORRECTION_SHA256,
        label="context manifest correction identity",
    )
    require_value(
        config.get("child_protocol_sha256"),
        sha256_file(protocol_path),
        label="context child-protocol identity",
    )
    require_value(
        config.get("script_sha256"),
        sha256_file(runner_path),
        label="context runner identity",
    )
    require_formal_args(
        config,
        {
            "outer_folds": 5,
            "outer_repeats": 5,
            "inner_folds": 4,
            "n_estimators": 600,
            "bootstrap_replicates": 10_000,
            "seed": 260531,
            "max_internal_splits": None,
            "max_ood_groups": None,
        },
        workflow="context",
    )
    require_passed_check(
        qa,
        "float32_correction_document_sha256",
        "training_response_dtype",
        "publication_input_allowlist",
    )
    return {
        "manifest": manifest,
        "configuration": config,
        "protocol_version": protocol_version,
    }


def validate_scaffold_workflow(
    directory: Path,
    identity: dict[str, object],
    qa: dict[str, Any],
) -> dict[str, Any]:
    execution = validate_scaffold_v3_execution(directory)
    config = execution["configuration"]
    manifest = execution["manifest"]
    protocol_version = str(identity["protocol_version"])
    protocol_path = Path(identity["protocol_path"])
    runner_path = Path(identity["runner_path"])
    for label, payload in (
        ("configuration", config),
        ("manifest", manifest),
    ):
        require_value(
            payload.get("protocol_version"),
            protocol_version,
            label=f"scaffold/source {label} protocol version",
        )
        require_value(
            payload.get("response_dtype"),
            "float32",
            label=f"scaffold/source {label} response dtype",
        )
        require_value(
            payload.get("float32_correction_document_sha256"),
            EXPECTED_CORRECTION_SHA256,
            label=f"scaffold/source {label} correction identity",
        )
        require_value(
            payload.get("formal_settings_match"),
            True,
            label=f"scaffold/source {label} formal-settings flag",
        )
    require_value(
        manifest.get("status"),
        "complete",
        label="scaffold/source manifest status",
    )
    require_value(
        str(manifest.get("qa_status", "")).upper(),
        "PASS",
        label="scaffold/source manifest QA status",
    )
    require_value(
        config.get("protocol_sha256"),
        sha256_file(protocol_path),
        label="scaffold/source child-protocol identity",
    )
    require_value(
        config.get("this_script_sha256"),
        sha256_file(runner_path),
        label="scaffold/source runner identity",
    )
    args = config.get("args", {})
    if not isinstance(args, dict):
        raise RuntimeError("scaffold/source configuration args must be an object")
    require_formal_args(
        args,
        {
            "mode": "both",
            "outer_folds": 5,
            "outer_repeats": 5,
            "inner_folds": 4,
            "n_estimators": 600,
            "n_jobs": 4,
            "bootstrap_replicates": 10_000,
            "seed": 260531,
            "max_outer_splits": None,
            "max_targets": None,
            "max_deletions": None,
            "only_target": None,
            "baseline_only": False,
        },
        workflow="scaffold/source",
    )
    require_value(
        config.get("identity_settings_match"),
        True,
        label="scaffold/source configuration identity-settings flag",
    )
    require_value(
        manifest.get("identity_settings_match"),
        True,
        label="scaffold/source manifest identity-settings flag",
    )
    require_value(
        manifest.get("native_thread_status"),
        "PASS",
        label="scaffold/source manifest native-thread status",
    )
    require_value(
        manifest.get("independent_identity_status"),
        "PASS",
        label="scaffold/source manifest independent-identity status",
    )
    require_value(
        qa.get("formal_settings_match"),
        True,
        label="scaffold/source QA formal-settings flag",
    )
    require_value(
        qa.get("protocol_version"),
        protocol_version,
        label="scaffold/source QA protocol version",
    )
    require_passed_check(
        qa,
        "training_response_dtype_float32",
        "native_threadpool_contract",
        "native_threadpool_stage_completeness",
        "all_target_baseline_prediction_identity",
        "independent_all_target_identity_qa",
    )
    return {
        "manifest": manifest,
        "configuration": config,
        "protocol_version": protocol_version,
    }


def validate_null_workflow(
    directory: Path,
    identity: dict[str, object],
    qa: dict[str, Any],
) -> dict[str, Any]:
    manifest = read_json(directory / "run_manifest.json")
    merge = read_json(directory / "permutation_shard_merge_manifest.json")
    protocol_version = str(identity["protocol_version"])
    protocol_path = Path(identity["protocol_path"])
    runner_path = Path(identity["runner_path"])
    wrapper_path = SCRIPT_DIR / str(identity["wrapper_filename"])
    merge_helper_path = SCRIPT_DIR / str(identity["merge_filename"])
    require_value(
        manifest.get("protocol_version"),
        protocol_version,
        label="null/applicability manifest protocol version",
    )
    require_value(
        manifest.get("local_protocol_sha256"),
        sha256_file(protocol_path),
        label="null/applicability child-protocol identity",
    )
    require_value(
        manifest.get("float32_correction_sha256"),
        EXPECTED_CORRECTION_SHA256,
        label="null/applicability correction identity",
    )
    require_value(
        manifest.get("applicability_correction_sha256"),
        identity["fixed_domain_correction_sha256"],
        label="null/applicability fixed-domain correction identity",
    )
    require_value(
        str(manifest.get("qa_status", "")).upper(),
        "PASS",
        label="null/applicability manifest QA status",
    )
    config = manifest.get("config", {})
    if not isinstance(config, dict):
        raise RuntimeError("null/applicability manifest config must be an object")
    require_formal_args(
        config,
        {
            "n_permutations": 100,
            "n_estimators": 600,
            "bootstrap_replicates": 10_000,
            "max_repeats": 5,
            "outer_folds_per_repeat": 5,
            "seed": 260531,
            "training_response_dtype": "float32",
        },
        workflow="null/applicability",
    )
    require_value(
        qa.get("protocol_version"),
        protocol_version,
        label="null/applicability QA protocol version",
    )
    require_value(
        qa.get("run_kind"),
        "formal",
        label="null/applicability QA run kind",
    )
    for key, expected in (
        ("status", "PASS"),
        ("protocol_version", protocol_version),
        ("training_response_dtype", "float32"),
        ("float32_correction_sha256", EXPECTED_CORRECTION_SHA256),
        (
            "applicability_correction_sha256",
            identity["fixed_domain_correction_sha256"],
        ),
        ("master_protocol_sha256", EXPECTED_MASTER_SHA256),
        ("local_protocol_sha256", sha256_file(protocol_path)),
        ("scientific_change", False),
        ("n_permutations", 100),
    ):
        require_value(
            merge.get(key),
            expected,
            label=f"permutation merge {key}",
        )
    require_value(
        merge.get("scientific_implementation_sha256"),
        sha256_file(runner_path),
        label="permutation merge parent-runner identity",
    )
    require_value(
        merge.get("execution_wrapper_sha256"),
        sha256_file(wrapper_path),
        label="permutation merge wrapper identity",
    )
    require_value(
        sha256_file(merge_helper_path),
        identity["merge_sha256"],
        label="permutation merge helper identity",
    )
    snapshot_relative = str(merge.get("recoverable_snapshot", ""))
    snapshot_candidate = Path(snapshot_relative)
    snapshot = ROUTE_DIR / snapshot_candidate
    if (
        snapshot_candidate.is_absolute()
        or ".." in snapshot_candidate.parts
        or snapshot.name != identity["snapshot_name"]
        or snapshot.parent.resolve() != directory.resolve()
        or not snapshot.is_dir()
        or snapshot.is_symlink()
    ):
        raise RuntimeError(
            f"Invalid or missing recoverable v3 permutation snapshot: {snapshot}"
        )
    primary_hashes = merge.get("primary_premerge_sha256")
    if not isinstance(primary_hashes, dict) or not primary_hashes:
        raise RuntimeError("Permutation merge lacks primary snapshot hashes")
    snapshot_files = {
        path.name
        for path in snapshot.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if snapshot_files != set(primary_hashes):
        raise RuntimeError("Permutation primary snapshot exact file set mismatch")
    for name, expected_hash in primary_hashes.items():
        snapshot_file = snapshot / str(name)
        if (
            snapshot_file.is_symlink()
            or sha256_file(snapshot_file) != str(expected_hash)
        ):
            raise RuntimeError(
                f"Permutation primary snapshot hash mismatch: {name}"
            )
    shards = merge.get("shards", [])
    if not isinstance(shards, list) or len(shards) != 3:
        raise RuntimeError("Permutation merge must bind exactly three v3 shards")
    shard_relatives: set[str] = set()
    for shard in shards:
        if not isinstance(shard, dict):
            raise RuntimeError("Permutation merge shard record must be an object")
        shard_relative = str(shard.get("path", ""))
        shard_candidate = Path(shard_relative)
        shard_path = ROUTE_DIR / shard_candidate
        if (
            shard_candidate.is_absolute()
            or ".." in shard_candidate.parts
            or shard_relative in shard_relatives
            or not shard_path.name.endswith(str(identity["shard_suffix"]))
            or not shard_path.is_dir()
            or shard_path.is_symlink()
            or (shard_path / "SUPERSEDED_DO_NOT_USE.md").exists()
            or (shard_path / "shard_manifest.json").is_symlink()
            or (shard_path / "shard_qa.json").is_symlink()
            or sha256_file(shard_path / "shard_manifest.json")
            != str(shard.get("manifest_sha256", ""))
            or sha256_file(shard_path / "shard_qa.json")
            != str(shard.get("qa_sha256", ""))
        ):
            raise RuntimeError(f"Invalid permutation v3 shard provenance: {shard_path}")
        shard_relatives.add(shard_relative)
    input_hashes = pd.read_csv(directory / "input_sha256.csv")
    require_exact_columns(
        input_hashes,
        ("relative_path", "sha256", "size_bytes"),
        label="null/applicability input inventory",
    )
    expected_input_paths = {
        "data/processed/all_molglue_dc50_qc_train_test_standardized_context.csv",
        "data/processed/all_molglue_dc50_parsed_records.csv",
        "reports/confirmatory_cpu_v1/scaffold_cross_fitted_predictions.csv",
        "reports/confirmatory_cpu_v1/scaffold_selected_hyperparameters.csv",
        "reports/confirmatory_cpu_v1/applicability_by_row.csv",
        "reports/confirmatory_ood_cpu_v1/ood_predictions.csv",
        (
            "reports/post_hoc_strict_domain_scaffold_ood_cpu_v1/"
            "strict_ood_predictions.csv"
        ),
        "docs/post_hoc_computational_extension_master_protocol_v1.md",
        "docs/post_hoc_null_applicability_censoring_protocol_v3.md",
        "docs/post_hoc_float32_response_correction_v2.md",
        "docs/post_hoc_applicability_fixed_domain_correction_v3.md",
        "scripts/run_post_hoc_null_applicability_censoring_v3.py",
        "scripts/run_confirmatory_cpu_v1.py",
    }
    observed_input_paths = input_hashes["relative_path"].astype(str).tolist()
    if (
        set(observed_input_paths) != expected_input_paths
        or len(observed_input_paths) != len(set(observed_input_paths))
    ):
        raise RuntimeError(
            "Null/applicability input inventory exact path set mismatch"
        )
    for _, row in input_hashes.iterrows():
        relative = str(row["relative_path"])
        candidate = Path(relative)
        path = ROUTE_DIR / candidate
        if (
            candidate.is_absolute()
            or ".." in candidate.parts
            or candidate.as_posix() != relative
            or not path.is_file()
            or path.is_symlink()
            or strict_integer(
                row["size_bytes"],
                label=f"null input {relative} size",
            )
            != path.stat().st_size
            or str(row["sha256"]) != sha256_file(path)
        ):
            raise RuntimeError(
                f"Null/applicability input inventory mismatch: {relative}"
            )
    expected_runner_relative = str(runner_path.relative_to(ROUTE_DIR))
    runner_rows = input_hashes[
        input_hashes["relative_path"].astype(str) == expected_runner_relative
    ]
    if len(runner_rows) != 1:
        raise RuntimeError(
            "Null/applicability input identity lacks exactly one v3 parent runner"
        )
    require_value(
        str(runner_rows.iloc[0]["sha256"]),
        sha256_file(runner_path),
        label="null/applicability input runner identity",
    )
    merge_tables = {
        "permutation_model_metrics.csv",
        "permutation_contrast_metrics.csv",
        "permutation_fold_diagnostics.csv",
    }
    merged_hashes = merge.get("merged_sha256")
    if (
        not isinstance(merged_hashes, dict)
        or set(merged_hashes) != merge_tables
        or any(
            sha256_file(directory / name) != str(expected_hash)
            for name, expected_hash in merged_hashes.items()
        )
    ):
        raise RuntimeError("Permutation merged-table SHA-256 binding mismatch")
    if (
        set(primary_hashes) != merge_tables
        or merge.get("feature_manifest_sha256")
        != sha256_file(directory / "feature_manifest.json")
        or set(merge.get("overlapping_ids_exactly_matched", {}))
        != merge_tables
    ):
        raise RuntimeError("Permutation merge manifest table set mismatch")

    expected_ranges = {(26, 50), (51, 75), (76, 100)}
    observed_ranges: set[tuple[int, int]] = set()
    for shard in shards:
        shard_path = ROUTE_DIR / str(shard["path"])
        shard_manifest = read_json(shard_path / "shard_manifest.json")
        shard_qa = read_json(shard_path / "shard_qa.json")
        shard_config = shard_manifest.get("config", {})
        if not isinstance(shard_config, dict):
            raise RuntimeError("Permutation shard config must be an object")
        shard_range = (
            int(shard_config.get("start_id", -1)),
            int(shard_config.get("end_id", -1)),
        )
        observed_ranges.add(shard_range)
        expected_shard_files = {
            "artifact_sha256.csv",
            "feature_manifest.json",
            "permutation_contrast_metrics.csv",
            "permutation_fold_diagnostics.csv",
            "permutation_model_metrics.csv",
            "shard_manifest.json",
            "shard_qa.json",
        }
        observed_shard_files = {
            path.name
            for path in shard_path.iterdir()
            if path.is_file() and not path.is_symlink()
        }
        if observed_shard_files != expected_shard_files:
            raise RuntimeError(
                f"Permutation shard exact artefact set mismatch: {shard_path}"
            )
        shard_inventory = pd.read_csv(shard_path / "artifact_sha256.csv")
        require_exact_columns(
            shard_inventory,
            ("relative_path", "sha256", "size_bytes"),
            label=f"{shard_path.name} artifact inventory",
        )
        shard_inventory_names = shard_inventory[
            "relative_path"
        ].astype(str).tolist()
        if (
            set(shard_inventory_names)
            != expected_shard_files - {"artifact_sha256.csv"}
            or len(shard_inventory_names)
            != len(set(shard_inventory_names))
        ):
            raise RuntimeError(
                f"Permutation shard inventory row set mismatch: {shard_path}"
            )
        for _, inventory_row in shard_inventory.iterrows():
            name = str(inventory_row["relative_path"])
            candidate = Path(name)
            artifact = shard_path / candidate
            if (
                candidate.name != name
                or not artifact.is_file()
                or artifact.is_symlink()
                or strict_integer(
                    inventory_row["size_bytes"],
                    label=f"{shard_path.name}/{name} size",
                )
                != artifact.stat().st_size
                or str(inventory_row["sha256"]) != sha256_file(artifact)
            ):
                raise RuntimeError(
                    f"Permutation shard inventory mismatch: "
                    f"{shard_path.name}/{name}"
                )
        if (
            shard_range
            != (int(shard["start_id"]), int(shard["end_id"]))
            or shard_manifest.get("protocol_version") != protocol_version
            or shard_manifest.get("scientific_implementation_sha256")
            != identity["runner_sha256"]
            or shard_manifest.get("float32_correction_sha256")
            != EXPECTED_CORRECTION_SHA256
            or shard_manifest.get("applicability_correction_sha256")
            != identity["fixed_domain_correction_sha256"]
            or shard_manifest.get("scientific_change") is not False
            or shard_qa.get("overall_status") != "PASS"
            or int(shard_qa.get("start_id", -1)) != shard_range[0]
            or int(shard_qa.get("end_id", -1)) != shard_range[1]
            or shard_qa.get("all_metrics_finite") is not True
            or shard_qa.get("all_multisets_preserved") is not True
            or str(shard["feature_manifest_sha256"])
            != sha256_file(shard_path / "feature_manifest.json")
            or str(shard["feature_manifest_sha256"])
            != merge["feature_manifest_sha256"]
        ):
            raise RuntimeError(
                f"Permutation shard scientific identity mismatch: {shard_path}"
            )
    if observed_ranges != expected_ranges:
        raise RuntimeError(
            f"Permutation shard range set mismatch: {sorted(observed_ranges)}"
        )
    return {
        "manifest": manifest,
        "merge_manifest": merge,
        "protocol_version": protocol_version,
    }


def validate_formal_workflows() -> dict[str, dict[str, Any]]:
    require_publication_lineage_frozen()
    if sha256_file(CORRECTION_DOCUMENT) != EXPECTED_CORRECTION_SHA256:
        raise RuntimeError("Float32 correction document hash mismatch")
    validated: dict[str, dict[str, Any]] = {}
    validators = {
        "context_weight": validate_context_workflow,
        "scaffold_source": validate_scaffold_workflow,
        "null_applicability_censoring": validate_null_workflow,
    }
    for workflow, identity in WORKFLOW_IDENTITIES.items():
        directory = Path(identity["directory"])
        require_value(
            sha256_file(Path(identity["protocol_path"])),
            identity["protocol_sha256"],
            label=f"{workflow} frozen protocol SHA-256",
        )
        require_exact_workflow_directory(
            directory,
            str(identity["directory_name"]),
        )
        qa = require_completed_qa(directory)
        validated[workflow] = validators[workflow](directory, identity, qa)
        validated[workflow]["qa"] = qa
        validated[workflow]["artifact_inventory"] = (
            validate_artifact_inventory(directory, workflow=workflow)
        )
    return validated


def _has_text(values: pd.Series) -> pd.Series:
    return values.notna() & values.astype(str).str.strip().ne("")


def _stable_domain_sequence(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}: domain sequence must be non-empty JSON text")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label}: invalid domain-sequence JSON") from error
    if (
        not isinstance(decoded, list)
        or any(not isinstance(item, str) or not item for item in decoded)
        or decoded != sorted(set(decoded))
    ):
        raise ValueError(f"{label}: domains must be unique sorted strings")
    canonical = json.dumps(decoded, ensure_ascii=False, separators=(",", ":"))
    if canonical != value:
        raise ValueError(f"{label}: domain JSON is not canonical")
    return tuple(decoded)


def _require_domain_identity(
    row: pd.Series,
    *,
    json_column: str,
    hash_column: str,
    count_column: str,
    label: str,
    allow_empty_hash: bool = False,
) -> tuple[str, ...]:
    domains = _stable_domain_sequence(row[json_column], label=label)
    observed_hash = (
        ""
        if pd.isna(row[hash_column])
        else str(row[hash_column]).strip()
    )
    expected_hash = hashlib.sha256(
        str(row[json_column]).encode("utf-8")
    ).hexdigest()
    if observed_hash != expected_hash and not (
        allow_empty_hash and not domains and not observed_hash
    ):
        raise ValueError(f"{label}: domain-sequence SHA-256 mismatch")
    if int(row[count_column]) != len(domains):
        raise ValueError(f"{label}: domain-sequence count mismatch")
    return domains


def _require_ci(
    frame: pd.DataFrame,
    observed: str,
    low: str,
    high: str,
    *,
    label: str,
) -> None:
    selected = frame[[observed, low, high]].apply(
        pd.to_numeric, errors="raise"
    )
    complete = selected.notna().all(axis=1)
    partial = selected.notna().any(axis=1) & ~complete
    if partial.any():
        raise ValueError(f"{label}: partially missing confidence interval")
    values = selected.loc[complete]
    if (
        not np.isfinite(values.to_numpy(dtype=float)).all()
        or not (values[low] <= values[observed]).all()
        or not (values[observed] <= values[high]).all()
    ):
        raise ValueError(f"{label}: CI does not contain its observed estimate")


def _validate_non_estimable_metric(
    frame: pd.DataFrame,
    metric: str,
    reason: str,
    *,
    label: str,
) -> None:
    values = pd.to_numeric(frame[metric], errors="raise")
    missing = values.isna()
    reasons = _has_text(frame[reason])
    if not (missing == reasons).all():
        raise ValueError(
            f"{label}.{metric}: missing values must have exactly one explicit reason"
        )


def validate_source_table(
    key: str,
    path: Path,
    frame: pd.DataFrame,
) -> None:
    if len(frame) != EXPECTED_ROW_COUNTS[key]:
        raise ValueError(
            f"{path.name}: expected {EXPECTED_ROW_COUNTS[key]} rows, "
            f"observed {len(frame)}"
        )
    boolean_columns = (
        {"within_stratum_multiset_preserved"}
        if key == "permutation_diagnostics"
        else set()
    )
    numeric_columns = (
        set(SOURCE_COLUMN_ALLOWLISTS[key])
        - CATEGORICAL_COLUMNS[key]
        - boolean_columns
    )
    require_finite_numeric(
        frame,
        sorted(numeric_columns),
        label=path.name,
        allow_missing=NULLABLE_NUMERIC_COLUMNS.get(key, set()),
    )

    integer_candidates = {
        column
        for column in numeric_columns
        if (
            column.startswith("n_")
            or column.endswith("_n_valid")
            or column.endswith("_finite_domains")
            or column.endswith("_n_bootstrap_finite")
            or column
            in {
                "repeat",
                "outer_fold",
                "inner_fold",
                "permutation_id",
                "permutation_seed",
                "stage_order",
            }
        )
    }
    for column in sorted(integer_candidates):
        values = pd.to_numeric(frame[column], errors="raise")
        finite = values.dropna().to_numpy(dtype=float)
        if (
            (finite < 0).any()
            or not np.equal(finite, np.floor(finite)).all()
        ):
            raise ValueError(f"{path.name}.{column} is not a non-negative integer")

    if "n_bootstrap" in frame:
        require_exact_integer_column(
            frame, "n_bootstrap", 10_000, label=path.name
        )
    if "n_bootstrap_requested" in frame:
        require_exact_integer_column(
            frame, "n_bootstrap_requested", 10_000, label=path.name
        )
    if key == "permutation_inference":
        require_exact_integer_column(
            frame, "n_permutations_finite", 100, label=path.name
        )
    if key == "permutation_diagnostics":
        require_strict_boolean_column(
            path,
            frame,
            "within_stratum_multiset_preserved",
            require_all_true=True,
        )

    if key == "portable_context":
        for record_type, prefixes in {
            "model": (
                "domain_macro_spearman",
                "domain_macro_rmse",
                "pooled_spearman",
                "pooled_rmse",
            ),
            "contrast": (
                "delta_domain_macro_spearman",
                "delta_domain_macro_rmse",
                "delta_pooled_spearman",
                "delta_pooled_rmse",
            ),
        }.items():
            subset = frame[frame["record_type"] == record_type]
            if len(subset) != 6:
                raise ValueError(f"{path.name}: incomplete {record_type} rows")
            for prefix in prefixes:
                required_columns = [
                    f"{prefix}_observed",
                    f"{prefix}_bootstrap_mean",
                    f"{prefix}_ci_low",
                    f"{prefix}_ci_high",
                    f"{prefix}_n_valid",
                ]
                if subset[required_columns].isna().any().any():
                    raise ValueError(
                        f"{path.name}: NaN in required {record_type}/{prefix}"
                    )
                _require_ci(
                    subset,
                    f"{prefix}_observed",
                    f"{prefix}_ci_low",
                    f"{prefix}_ci_high",
                    label=f"{path.name}/{record_type}/{prefix}",
                )
    elif key == "weighting_bootstrap":
        model = frame[frame["record_type"] == "model"]
        contrast = frame[frame["record_type"] == "contrast"]
        if len(model) != 6 or len(contrast) != 7:
            raise ValueError(f"{path.name}: incomplete model/contrast rows")
        for prefix in ("spearman", "rmse"):
            model_columns = [
                f"{prefix}_observed",
                f"{prefix}_bootstrap_mean",
                f"{prefix}_ci_low",
                f"{prefix}_ci_high",
            ]
            contrast_columns = [
                f"delta_{prefix}_observed",
                f"delta_{prefix}_mean",
                f"delta_{prefix}_ci_low",
                f"delta_{prefix}_ci_high",
            ]
            if model[model_columns].isna().any().any():
                raise ValueError(f"{path.name}: NaN in model {prefix}")
            if contrast[contrast_columns].isna().any().any():
                raise ValueError(f"{path.name}: NaN in contrast {prefix}")
            _require_ci(
                model,
                f"{prefix}_observed",
                f"{prefix}_ci_low",
                f"{prefix}_ci_high",
                label=f"{path.name}/model/{prefix}",
            )
            _require_ci(
                contrast,
                f"delta_{prefix}_observed",
                f"delta_{prefix}_ci_low",
                f"delta_{prefix}_ci_high",
                label=f"{path.name}/contrast/{prefix}",
            )
    elif key in {"generic_scaffold", "source_deletion"}:
        _require_ci(
            frame,
            "observed",
            "ci_low",
            "ci_high",
            label=path.name,
        )
    elif key == "applicability_domains":
        _validate_non_estimable_metric(
            frame,
            "spearman",
            "spearman_non_estimable_reason",
            label=path.name,
        )
    elif key == "applicability_models":
        for metric in ("spearman", "rmse", "mae"):
            domain_metric = f"domain_macro_{metric}"
            count = pd.to_numeric(
                frame[f"{domain_metric}_finite_domains"], errors="raise"
            )
            values = pd.to_numeric(frame[domain_metric], errors="raise")
            reason = _has_text(frame["domain_macro_non_estimable_reason"])
            if not (
                ((count == 0) & values.isna() & reason)
                | ((count > 0) & values.notna() & ~reason)
            ).all():
                raise ValueError(
                    f"{path.name}.{domain_metric}: invalid non-estimable pattern"
                )
    elif key == "applicability_contrasts":
        similarity_bins = {
            "0.0_to_lt_0.4",
            "0.4_to_lt_0.6",
            "0.6_to_lt_0.8",
            "0.8_to_1.000001",
        }
        combinations = {
            ("internal_scaffold_disjoint", "scaffold"),
            ("frozen_ood", "source_ood"),
            ("frozen_ood", "target_ood"),
            ("strict_domain_scaffold_ood", "source_ood"),
            ("strict_domain_scaffold_ood", "target_ood"),
        }
        keys = frame[["regime", "protocol", "similarity_bin"]].astype(str)
        observed_keys = {
            tuple(row)
            for row in keys.itertuples(index=False, name=None)
        }
        expected_keys = {
            (regime, protocol, similarity_bin)
            for regime, protocol in combinations
            for similarity_bin in similarity_bins
        }
        if (
            len(observed_keys) != len(frame)
            or observed_keys != expected_keys
            or not frame["contrast_id"].astype(str).eq(
                "full_minus_chemistry"
            ).all()
        ):
            raise ValueError(
                f"{path.name}: applicability contrast strata are incomplete, "
                "duplicated, or non-canonical"
            )
        for metric in ("spearman", "rmse", "mae"):
            require_exact_integer_column(
                frame,
                f"delta_{metric}_n_bootstrap_finite",
                10_000,
                label=path.name,
            )
            if _has_text(
                frame[f"delta_{metric}_non_estimable_reason"]
            ).any():
                raise ValueError(
                    f"{path.name}: pooled delta_{metric} unexpectedly "
                    "non-estimable"
                )
            _require_ci(
                frame,
                f"delta_{metric}",
                f"delta_{metric}_ci_low",
                f"delta_{metric}_ci_high",
                label=f"{path.name}/pooled/{metric}",
            )
            prefix = f"delta_domain_macro_{metric}"
            eligible = pd.to_numeric(
                frame[f"{prefix}_n_domains_eligible"], errors="raise"
            )
            observed = pd.to_numeric(frame[prefix], errors="raise")
            requested = pd.to_numeric(
                frame[f"{prefix}_n_bootstrap_requested"], errors="raise"
            )
            finite = pd.to_numeric(
                frame[f"{prefix}_n_bootstrap_finite"], errors="raise"
            )
            valid = pd.to_numeric(
                frame[f"{prefix}_n_bootstrap_valid"], errors="raise"
            )
            minimum = pd.to_numeric(
                frame[f"{prefix}_minimum_valid_required"], errors="raise"
            )
            invalid_missing = pd.to_numeric(
                frame[
                    f"{prefix}_n_bootstrap_invalid_missing_domain"
                ],
                errors="raise",
            )
            invalid_metric = pd.to_numeric(
                frame[
                    f"{prefix}_n_bootstrap_invalid_metric_nonestimable"
                ],
                errors="raise",
            )
            structural = pd.to_numeric(
                frame[
                    f"{prefix}_n_bootstrap_not_attempted_structural"
                ],
                errors="raise",
            )
            triple = frame[
                [prefix, f"{prefix}_ci_low", f"{prefix}_ci_high"]
            ].apply(pd.to_numeric, errors="raise")
            reason = _has_text(frame[f"{prefix}_non_estimable_reason"])
            status = frame[f"{prefix}_ci_status"].astype(str)
            estimated = status.eq("ESTIMATED")
            protocol = frame["protocol"].astype(str)
            internal = protocol.eq("scaffold")
            ood = ~internal
            sparse = pd.to_numeric(
                frame[f"{prefix}_n_sparse_eligible_domains"],
                errors="raise",
            )
            global_scaffolds = pd.to_numeric(
                frame["n_scaffolds"], errors="raise"
            )
            expected_status = pd.Series(
                np.where(
                    internal,
                    "NOT_APPLICABLE_INTERNAL_POOLED",
                    np.where(
                        (eligible == 0)
                        | (sparse > 0)
                        | (global_scaffolds < 2),
                        "NON_ESTIMABLE_STRUCTURAL",
                        np.where(
                            valid >= minimum,
                            "ESTIMATED",
                            "NON_ESTIMABLE_TOO_FEW_VALID",
                        ),
                    ),
                ),
                index=frame.index,
            )
            if not status.eq(expected_status).all():
                raise ValueError(
                    f"{path.name}.{prefix}: CI status violates the frozen rule"
                )
            reason_text = frame[
                f"{prefix}_non_estimable_reason"
            ].fillna("").astype(str)
            expected_reason = pd.Series(
                np.where(
                    status.eq("ESTIMATED"),
                    "",
                    np.where(
                        status.eq("NON_ESTIMABLE_STRUCTURAL"),
                        np.where(
                            eligible.eq(0),
                            "no_observed_eligible_domains",
                            np.where(
                                sparse.gt(0),
                                "fixed_eligible_domain_fewer_than_2_scaffold_clusters",
                                "fewer_than_2_global_scaffold_clusters",
                            ),
                        ),
                        np.where(
                            status.eq("NON_ESTIMABLE_TOO_FEW_VALID"),
                            "too_few_complete_fixed_domain_bootstrap_replicates",
                            "not_applicable_internal_pooled",
                        ),
                    ),
                ),
                index=frame.index,
            )
            if not reason_text.eq(expected_reason).all():
                raise ValueError(
                    f"{path.name}.{prefix}: non-estimable reason mismatch"
                )
            ci_missing = triple[
                [f"{prefix}_ci_low", f"{prefix}_ci_high"]
            ].isna().all(axis=1)
            if not (
                (
                    estimated
                    & (eligible > 0)
                    & (valid >= minimum)
                    & triple.notna().all(axis=1)
                    & ~reason
                )
                | (
                    ~estimated
                    & ci_missing
                    & reason
                )
            ).all():
                raise ValueError(
                    f"{path.name}.{prefix}: invalid CI status/completeness"
                )
            if not (
                (internal & eligible.eq(0) & observed.isna())
                | (ood & eligible.gt(0) & observed.notna())
            ).all():
                raise ValueError(
                    f"{path.name}.{prefix}: observed macro/eligible domains "
                    "are inconsistent"
                )
            accounting = valid + invalid_missing + invalid_metric + structural
            if not (
                requested.eq(10_000).all()
                and minimum.eq(5_000).all()
                and finite.eq(valid).all()
                and pd.to_numeric(
                    frame[f"{prefix}_finite_domains"], errors="raise"
                ).eq(eligible).all()
                and accounting.loc[ood].eq(requested.loc[ood]).all()
                and accounting.loc[internal].eq(0).all()
                and structural.loc[
                    status.eq("NON_ESTIMABLE_STRUCTURAL")
                ].eq(requested.loc[
                    status.eq("NON_ESTIMABLE_STRUCTURAL")
                ]).all()
                and structural.loc[
                    ~status.eq("NON_ESTIMABLE_STRUCTURAL")
                ].eq(0).all()
            ):
                raise ValueError(
                    f"{path.name}.{prefix}: bootstrap accounting mismatch"
                )
            _require_ci(
                frame.loc[estimated],
                prefix,
                f"{prefix}_ci_low",
                f"{prefix}_ci_high",
                label=f"{path.name}/domain-macro/{metric}",
            )
        for row_index, row in frame.iterrows():
            represented = _require_domain_identity(
                row,
                json_column="represented_domains_json",
                hash_column="represented_domains_sha256",
                count_column="n_domains_total",
                label=f"{path.name}[{row_index}]/represented",
            )
            expected_domains = (
                1
                if str(row["protocol"]) == "scaffold"
                else (4 if str(row["protocol"]) == "source_ood" else 8)
            )
            if (
                int(row["n_domains_expected"]) != expected_domains
                or int(row["n_heldout_groups"]) != len(represented)
                or not (1 <= len(represented) <= expected_domains)
            ):
                raise ValueError(
                    f"{path.name}[{row_index}]: represented-domain count "
                    "violates the protocol"
                )
            for metric in ("spearman", "rmse", "mae"):
                prefix = f"delta_domain_macro_{metric}"
                eligible = _require_domain_identity(
                    row,
                    json_column=f"{prefix}_eligible_domains_json",
                    hash_column=f"{prefix}_eligible_domains_sha256",
                    count_column=f"{prefix}_n_domains_eligible",
                    label=f"{path.name}[{row_index}]/{metric}/eligible",
                    allow_empty_hash=str(row["protocol"]) == "scaffold",
                )
                sparse = _require_domain_identity(
                    row,
                    json_column=f"{prefix}_sparse_eligible_domains_json",
                    hash_column=f"{prefix}_sparse_eligible_domains_sha256",
                    count_column=f"{prefix}_n_sparse_eligible_domains",
                    label=f"{path.name}[{row_index}]/{metric}/sparse",
                    allow_empty_hash=str(row["protocol"]) == "scaffold",
                )
                if not set(eligible).issubset(represented) or not set(
                    sparse
                ).issubset(eligible):
                    raise ValueError(
                        f"{path.name}[{row_index}]/{metric}: eligible-domain "
                        "sets are inconsistent"
                    )
            expected_non_estimable = sorted(
                f"delta_domain_macro_{metric}:"
                f"{row[f'delta_domain_macro_{metric}_non_estimable_reason']}"
                for metric in ("spearman", "rmse", "mae")
                if str(
                    row[f"delta_domain_macro_{metric}_ci_status"]
                )
                not in {"ESTIMATED", "NOT_APPLICABLE_INTERNAL_POOLED"}
            )
            observed_non_estimable = (
                ""
                if pd.isna(row["bootstrap_non_estimable_reason"])
                else str(row["bootstrap_non_estimable_reason"])
            )
            if observed_non_estimable != "|".join(expected_non_estimable):
                raise ValueError(
                    f"{path.name}[{row_index}]: aggregate bootstrap "
                    "non-estimable reason mismatch"
                )
    elif key == "permutation_inference":
        if not (
            frame["null_percentile_2p5"]
            <= frame["null_percentile_97p5"]
        ).all():
            raise ValueError(f"{path.name}: reversed null percentile interval")


def read_checked(
    path: Path,
    required: Iterable[str],
    *,
    key: str,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    require_exact_columns(
        frame, SOURCE_COLUMN_ALLOWLISTS[key], label=path.name
    )
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(
            f"{path.name} is missing required publication columns: {missing}"
        )
    leaked = sorted(FORBIDDEN_COLUMNS.intersection(frame.columns))
    if leaked:
        raise ValueError(
            f"{path.name} is not aggregate-safe; forbidden columns: {leaked}"
        )
    validate_source_table(key, path, frame)
    return frame.loc[:, PUBLISH_COLUMN_ALLOWLISTS[key]].copy()


def validate_source_deletion_baseline_identity(directory: Path) -> None:
    source_path = directory / "source_deletion_predictions.csv"
    confirmatory_path = (
        REPORT_DIR / "confirmatory_ood_cpu_v1" / "ood_predictions.csv"
    )
    if (
        not source_path.is_file()
        or source_path.is_symlink()
        or not confirmatory_path.is_file()
        or confirmatory_path.is_symlink()
    ):
        raise FileNotFoundError(
            "source-deletion or confirmatory prediction identity input missing"
        )
    source = pd.read_csv(source_path)
    confirmatory = pd.read_csv(confirmatory_path)
    source = source[source["deletion_source"] == "none"].copy()
    confirmatory = confirmatory[
        (confirmatory["protocol"] == "target_ood")
        & confirmatory["model_id"].isin(
            {"chemistry_extra_trees", "full_context_extra_trees"}
        )
    ].rename(columns={"heldout_group": "heldout_target"})
    expected_targets = {
        "VAV1",
        "CSNK1A1",
        "GSPT1",
        "WIZ",
        "CDK2",
        "CCNK+CDK12",
        "IKZF2",
        "IKZF1",
    }
    if (
        set(source["heldout_target"]) != expected_targets
        or set(confirmatory["heldout_target"]) != expected_targets
    ):
        raise RuntimeError("source-deletion identity does not cover all 8 targets")
    columns = [
        "heldout_target",
        "row_index",
        "qc_id",
        "model_id",
        "param_id",
        "y_true",
        "y_pred",
        "canonical_smiles",
        "source_database",
        "target_protein",
    ]
    sort_columns = ["heldout_target", "row_index", "model_id", "qc_id"]
    observed = source[columns].sort_values(
        sort_columns, kind="mergesort"
    ).reset_index(drop=True)
    expected = confirmatory[columns].sort_values(
        sort_columns, kind="mergesort"
    ).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            observed,
            expected,
            check_exact=True,
            check_dtype=True,
            check_like=False,
        )
    except AssertionError as error:
        raise RuntimeError(
            "All 8 no-deletion target predictions must exactly equal the "
            "confirmatory target-OOD predictions"
        ) from error

    domain = pd.read_csv(directory / "source_deletion_domain_metrics.csv")
    baseline = domain[domain["deletion_source"] == "none"]
    for model_id, group in baseline.groupby("model_id", sort=False):
        spearman = pd.to_numeric(group["spearman"], errors="raise").to_numpy(
            dtype=float
        )
        if (
            len(group) != 8
            or set(group["heldout_target"]) != expected_targets
            or not np.isfinite(spearman).all()
        ):
            raise RuntimeError(
                "Fixed 8-domain macro is invalid because a no-deletion target "
                f"Spearman is missing/non-finite for {model_id}"
            )


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    if sha256_file(MASTER_PROTOCOL) != EXPECTED_MASTER_SHA256:
        raise RuntimeError("Master extension protocol hash mismatch")
    validated = validate_formal_workflows()
    validate_source_deletion_baseline_identity(SCAFFOLD_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, object]] = []
    emitted: list[Path] = []
    loaded: dict[str, pd.DataFrame] = {}
    for key, (input_path, required) in INPUTS.items():
        loaded[key] = read_checked(input_path, required, key=key)

    if (
        loaded["permutation_models"]["permutation_id"].nunique() != 100
        or loaded["permutation_contrasts"]["permutation_id"].nunique() != 100
    ):
        raise RuntimeError("Permutation source data do not contain 100 runs")
    diagnostics = loaded["permutation_diagnostics"]
    if (
        diagnostics["permutation_id"].nunique() != 100
        or len(diagnostics) != 100 * 5 * 5
        or not diagnostics["within_stratum_multiset_preserved"].all()
    ):
        raise RuntimeError("Permutation fold diagnostics are incomplete")
    if not loaded["permutation_inference"][
        "n_permutations_finite"
    ].eq(100).all():
        raise RuntimeError("Permutation inference is incomplete")
    selection = loaded["censoring_selection_flow"].set_index("stage")
    if (
        int(selection.loc["all_parsed_records", "n"]) != 3_117
        or int(selection.loc["final_aggregated_modeling_rows", "n"]) != 1_560
    ):
        raise RuntimeError("Censoring selection flow does not match frozen totals")

    for key, (input_path, _) in INPUTS.items():
        frame = loaded[key].copy()
        output_path = OUTPUT_DIR / OUTPUT_NAMES[key]
        frame.insert(0, "analysis_identity", ANALYSIS_IDENTITY)
        atomic_csv(frame, output_path)
        validate_aggregate_value_binding(
            input_path,
            output_path,
            PUBLISH_COLUMN_ALLOWLISTS[key],
            table_id=key,
        )
        emitted.append(output_path)
        workflow = TABLE_WORKFLOW[key]
        workflow_identity = WORKFLOW_IDENTITIES[workflow]
        inventory = validated[workflow]["artifact_inventory"]
        index_rows.append(
            {
                "table_id": key,
                "output_file": output_path.name,
                "source_file": str(input_path.relative_to(ROUTE_DIR)),
                "n_rows": int(len(frame)),
                "n_columns": int(frame.shape[1]),
                "source_size_bytes": int(input_path.stat().st_size),
                "source_sha256": sha256_file(input_path),
                "source_inventory_sha256": inventory["inventory_sha256"],
                "evidence_identity": ANALYSIS_IDENTITY,
                "protocol_version": workflow_identity["protocol_version"],
                "response_dtype": "float32",
                "float32_correction_sha256": EXPECTED_CORRECTION_SHA256,
                "fixed_domain_correction_sha256": (
                    workflow_identity["fixed_domain_correction_sha256"]
                    if workflow == "null_applicability_censoring"
                    else "not_applicable"
                ),
                "contains_row_level_records": False,
                "contains_molecular_structures": False,
            }
        )

    index = pd.DataFrame(index_rows)
    index_path = OUTPUT_DIR / "source_data_extension_index.csv"
    atomic_csv(index, index_path)
    emitted.append(index_path)

    workflow_provenance: dict[str, dict[str, object]] = {}
    for workflow in validated:
        identity = WORKFLOW_IDENTITIES[workflow]
        directory = Path(identity["directory"])
        entry: dict[str, object] = {
            "source_directory": str(directory.relative_to(ROUTE_DIR)),
            "protocol_version": identity["protocol_version"],
            "response_dtype": "float32",
            "protocol_sha256": sha256_file(Path(identity["protocol_path"])),
            "runner_sha256": sha256_file(Path(identity["runner_path"])),
            "artifact_inventory_sha256": validated[workflow][
                "artifact_inventory"
            ]["inventory_sha256"],
            "artifact_inventory_verified_rows": validated[workflow][
                "artifact_inventory"
            ]["verified_row_count"],
            "artifact_inventory_self_reference_exclusions": validated[
                workflow
            ]["artifact_inventory"]["self_reference_exclusions"],
            "qa_summary_sha256": sha256_file(directory / "qa_summary.json"),
            "run_manifest_sha256": sha256_file(directory / "run_manifest.json"),
            "superseded_marker_present": False,
        }
        if workflow == "context_weight":
            entry["configuration_sha256"] = sha256_file(
                directory / "configuration.json"
            )
        elif workflow == "scaffold_source":
            entry["scientific_configuration_sha256"] = sha256_file(
                directory / "scientific_configuration.json"
            )
            entry["native_threadpool_audit_sha256"] = sha256_file(
                directory / str(identity["thread_audit_filename"])
            )
            entry["all_target_exact_identity_audit_sha256"] = sha256_file(
                directory / str(identity["identity_audit_filename"])
            )
            entry[
                "all_target_exact_identity_audit_markdown_sha256"
            ] = sha256_file(
                directory / str(identity["identity_audit_markdown_filename"])
            )
            entry["identity_qa_script_sha256"] = identity["qa_sha256"]
            entry["execution_correction_sha256"] = identity[
                "execution_correction_sha256"
            ]
            entry["mathematical_runner_sha256"] = identity[
                "math_runner_sha256"
            ]
        else:
            entry["permutation_merge_manifest_sha256"] = sha256_file(
                directory / "permutation_shard_merge_manifest.json"
            )
            entry["permutation_wrapper_sha256"] = sha256_file(
                SCRIPT_DIR / str(identity["wrapper_filename"])
            )
            entry["permutation_merge_helper_sha256"] = sha256_file(
                SCRIPT_DIR / str(identity["merge_filename"])
            )
            entry["fixed_domain_correction_sha256"] = identity[
                "fixed_domain_correction_sha256"
            ]
        workflow_provenance[workflow] = entry
    software_bindings = downstream_software_bindings()
    provenance = {
        "provenance_version": PROVENANCE_VERSION,
        "analysis_identity": ANALYSIS_IDENTITY,
        "provenance_contract_sha256": PROVENANCE_CONTRACT_SHA256,
        "master_protocol_sha256": EXPECTED_MASTER_SHA256,
        "float32_correction_sha256": EXPECTED_CORRECTION_SHA256,
        "response_dtype": "float32",
        "software_bindings": software_bindings,
        "workflows": workflow_provenance,
        "publication_controls": {
            "superseded_workflow_results_included": False,
            "cross_version_workflow_mixing_permitted": False,
            "row_level_predictions_in_aggregate_source_data": False,
            (
                "publication_eligible_independent_prospective_"
                "dataset_available"
            ): False,
        },
    }
    provenance_path = OUTPUT_DIR / PROVENANCE_FILENAME
    atomic_json(provenance, provenance_path)
    emitted.append(provenance_path)

    checksums = pd.DataFrame(
        [
            {
                "relative_path": path.name,
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            for path in emitted
        ]
    ).sort_values("relative_path", kind="mergesort")
    checksum_path = OUTPUT_DIR / "source_data_extension_sha256.csv"
    atomic_csv(checksums, checksum_path)

    print(
        "COMPUTATIONAL_EXTENSION_SOURCE_DATA: PASS "
        f"({len(index)} tables; {int(index['n_rows'].sum())} aggregate rows)"
    )


if __name__ == "__main__":
    main()
