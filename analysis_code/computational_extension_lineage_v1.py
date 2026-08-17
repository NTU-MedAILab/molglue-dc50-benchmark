#!/usr/bin/env python3
"""Single source of truth for the formal computational-extension lineage.

Downstream publication builders must call
:func:`require_publication_lineage_frozen` before emitting or synchronising
any artefact.  The gate accepts only context v2, scaffold/source v3 and
null/applicability/censoring v3.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype


SCRIPT_DIR = Path(__file__).resolve().parent
ROUTE_DIR = SCRIPT_DIR.parent
REPORT_ROOT = ROUTE_DIR / "reports"

MASTER_PROTOCOL_FILENAME = "post_hoc_computational_extension_master_protocol_v1.md"
MASTER_PROTOCOL_SHA256 = (
    "7dee28f45e3ddf8d6299622a8e494c50492b3b2614f39ce8b2fdd2a1be754ff8"
)
FLOAT32_CORRECTION_FILENAME = "post_hoc_float32_response_correction_v2.md"
FLOAT32_CORRECTION_SHA256 = (
    "75a8b0e4f894cead9d72039eab9e6162a89d2a20c79afceda8a2aae1ffd78f3f"
)

# Final aggregate/release identity after the native-thread scaffold/source
# correction.  It must never certify scaffold/source v1/v2 or null v1/v2
# results.
ANALYSIS_IDENTITY = "post_hoc_extension_float32_v4"
RELEASE_ANALYSIS_LABEL = "post_hoc_computational_extension_float32_v4"
PROVENANCE_VERSION = "computational_extension_float32_v4.0"
PROVENANCE_FILENAME = "source_data_extension_provenance_v4.json"
PROVENANCE_CONTRACT_FILENAME = (
    "computational_extension_float32_v4_provenance_contract.md"
)
PROVENANCE_CONTRACT_SHA256 = (
    "e45fe57721f9cfd1c8d3773356d71db738c924c7dd2331b0bb7ee3c94086e815"
)
FIGURE_DIRECTORY_NAME = "publication_figures_v4"
FIGURE_PROVENANCE_FILENAME = "computational_extension_figure_provenance.json"
FIGURE_QA_FILENAME = "computational_extension_figure_qa_summary.json"
FIGURE_CONTRACT_FILENAME = "computational_extension_figure_contract_v1.md"
FIGURE_CONTRACT_SHA256 = (
    "98140192a53c0fcad1a83cb736fbe8c9b22cad01fb7597223304dd2a78e0d748"
)
FIGURE_QA_SCRIPT_FILENAME = "qa_computational_extension_figures_v1.py"
FIGURE_QA_SCRIPT_SHA256 = (
    "eff8189674424e00aa3e9532ca4f082b09c8724851eac03b74458622400e955c"
)

PUBLIC_PROSPECTIVE_DATASET_STATEMENT = (
    "No publication-eligible independent prospective dataset was available."
)

DOWNSTREAM_SCRIPT_FILENAMES = {
    "source_data_builder_sha256": (
        "build_computational_extension_source_data_v1.py"
    ),
    "results_summarizer_sha256": (
        "summarize_computational_extension_results_v1.py"
    ),
    "independent_crosscheck_sha256": (
        "qa_computational_extension_crosscheck_v1.py"
    ),
}

# Exact order emitted by qa_computational_extension_crosscheck_v1.py.  Binding
# the complete set prevents a stale, shortened or selectively omitted audit
# from being accepted merely because its top-level status says PASS.
CROSSCHECK_EXPECTED_CHECK_IDS = (
    "core_sha256",
    "parsed_sha256",
    "master_protocol_sha256",
    "float32_correction_sha256",
    "context_v2_protocol_sha256",
    "scaffold_v3_protocol_sha256",
    "null_v3_protocol_sha256",
    "publication_lineage_frozen",
    "context_weight_exact_directory_and_supersession_guard",
    "context_weight_artifact_inventory_rowwise",
    "context_weight_formal_provenance_files",
    "context_weight_formal_manifest_identity",
    "scaffold_source_exact_directory_and_supersession_guard",
    "scaffold_source_artifact_inventory_rowwise",
    "scaffold_source_formal_provenance_files",
    "scaffold_source_formal_manifest_identity",
    "null_applicability_censoring_exact_directory_and_supersession_guard",
    "null_applicability_censoring_artifact_inventory_rowwise",
    "null_applicability_censoring_formal_provenance_files",
    "null_applicability_censoring_formal_manifest_identity",
    "formal_source_tables_exact_schema_dtype_and_counts",
    "context_weight_upstream_qa",
    "scaffold_source_upstream_qa",
    "null_applicability_censoring_upstream_qa",
    "context_required_artifacts",
    "context_formal_configuration",
    "context_internal_model_set",
    "context_internal_prediction_shape",
    "context_internal_metrics_independent_recompute",
    "context_25_scaffold_disjoint_splits",
    "context_bootstrap_counts",
    "portable_ood_macro_independent_recompute",
    "scaffold_source_required_artifacts",
    "scaffold_source_formal_configuration",
    "scaffold_source_v3_native_thread_and_identity_gate",
    "generic_scaffold_metrics_independent_recompute",
    "generic_25_group_disjoint_splits",
    "source_deletion_macro_independent_recompute",
    "source_deletion_condition_completeness",
    "source_deletion_all_8_confirmatory_prediction_identity",
    "scaffold_source_bootstrap_counts",
    "null_applicability_required_artifacts",
    "null_formal_configuration",
    "null_locked_environment_identity",
    "permutation_id_completeness",
    "permutation_table_shape_and_uniqueness",
    "permutation_fold_diagnostic_completeness",
    "permutation_shard_merge_provenance",
    "permutation_empirical_p_independent_recompute",
    "applicability_domain_macro_independent_recompute",
    "applicability_strata_and_bootstrap_completeness",
    "censoring_total_and_classes",
    "aggregate_source_exact_hash_and_lineage_binding",
    "computational_extension_results_independent_recompute",
)


CONTEXT_ARTIFACTS = (
    "artifact_sha256.csv",
    "configuration.json",
    "context_sanitization_audit.csv",
    "internal_canonical_compound_metrics.csv",
    "internal_domain_metrics.csv",
    "internal_paired_scaffold_bootstrap.csv",
    "internal_predictions.csv",
    "internal_repeat_averaged_predictions.csv",
    "internal_repeat_metrics.csv",
    "internal_selected_hyperparameters.csv",
    "internal_split_audit.csv",
    "internal_summary_metrics.csv",
    "internal_tuning_metrics.csv",
    "internal_weight_diagnostics.csv",
    "ood_aggregate_metrics.csv",
    "ood_domain_metrics.csv",
    "ood_domain_scaffold_bootstrap.csv",
    "ood_paired_scaffold_bootstrap.csv",
    "ood_portable_feature_audit.csv",
    "ood_portable_inner_split_audit.csv",
    "ood_portable_selected_hyperparameters.csv",
    "ood_portable_tuning_metrics.csv",
    "ood_predictions.csv",
    "qa_summary.json",
    "qa_summary.md",
    "results_brief_zh.md",
    "run_manifest.json",
)

SCAFFOLD_ARTIFACTS = (
    "all_target_exact_identity_audit.json",
    "all_target_exact_identity_audit.md",
    "analysis_index.csv",
    "artifact_sha256.csv",
    "context_sanitization_audit.csv",
    "generic_scaffold_group_summary.csv",
    "generic_scaffold_inner_split_audit.csv",
    "generic_scaffold_inner_tuning_metrics.csv",
    "generic_scaffold_metrics_by_repeat.csv",
    "generic_scaffold_outer_fold_metrics.csv",
    "generic_scaffold_outer_split_audit.csv",
    "generic_scaffold_paired_cluster_bootstrap.csv",
    "generic_scaffold_predictions.csv",
    "generic_scaffold_repeat_averaged_metrics.csv",
    "generic_scaffold_repeat_averaged_predictions.csv",
    "generic_scaffold_selected_hyperparameters.csv",
    "native_threadpool_audit.json",
    "qa_summary.json",
    "qa_summary.md",
    "results_brief_zh.md",
    "run_manifest.json",
    "scientific_configuration.json",
    "source_deletion_domain_metrics.csv",
    "source_deletion_domain_paired_deltas.csv",
    "source_deletion_equal_domain_macro.csv",
    "source_deletion_inner_split_audit.csv",
    "source_deletion_inner_tuning_metrics.csv",
    "source_deletion_paired_cluster_bootstrap.csv",
    "source_deletion_predictions.csv",
    "source_deletion_selected_hyperparameters.csv",
    "source_deletion_split_audit.csv",
)

SCAFFOLD_SOURCE_DOMAIN_METRICS_COLUMNS = (
    "deletion_source",
    "heldout_target",
    "model_id",
    "param_id",
    "n_train_rows",
    "n_train_unique_smiles",
    "n_train_scaffolds",
    "n_test_rows",
    "n_test_unique_smiles",
    "n_test_scaffolds",
    "spearman",
    "pearson",
    "rmse",
    "mae",
    "r2",
    "calibration_intercept",
    "calibration_slope",
    "rdkit_kept",
    "rdkit_total",
    "rdkit_train_missing",
    "rdkit_eval_missing",
    "context_dim",
    "chemistry_dim",
    "full_dim",
)

# The anticipated v3 output names are listed so the inventory gate is exact.
# If the v3 implementation legitimately changes this set, update it together
# with the five pending v3 hashes below in this file only.
NULL_V3_ARTIFACTS = (
    "applicability_macro_bootstrap_audit.csv",
    "applicability_macro_domain_eligibility.csv",
    "applicability_domain_metrics.csv",
    "applicability_model_metrics.csv",
    "applicability_paired_deltas.csv",
    "applicability_paired_source_data.csv",
    "artifact_sha256.csv",
    "censoring_overlap_with_core.csv",
    "censoring_relation_by_source.csv",
    "censoring_relation_by_target.csv",
    "censoring_relation_operator_counts.csv",
    "censoring_relation_overall.csv",
    "context_sanitization_audit.csv",
    "core_modeling_distribution_by_target.csv",
    "feature_manifest.json",
    "input_sha256.csv",
    "permutation_contrast_metrics.csv",
    "permutation_fold_diagnostics.csv",
    "permutation_model_metrics.csv",
    "permutation_null_inference.csv",
    "permutation_observed_contrast_metrics.csv",
    "permutation_observed_model_metrics.csv",
    "permutation_shard_merge_manifest.json",
    "qa_summary.json",
    "qa_summary.md",
    "results_brief_zh.md",
    "run_manifest.json",
    "selection_flow_overall.csv",
    "selection_representation_by_source.csv",
)


# PENDING-V3 FREEZE BLOCK.  This is the sole update point after the null v3
# protocol, runner, wrapper and merge helper are final and their formal output
# inventory has completed.
NULL_V3_PUBLICATION_READY = True
NULL_V3_PROTOCOL_SHA256: str | None = (
    "84178cde0edfc0ff31be3d7e12135351f95a2578b9ac69b61d9a5d517caa0dc5"
)
NULL_V3_RUNNER_SHA256: str | None = (
    "41b99a803b0e090d1bd55950983441eee0ce8edc8b099f86f56387e2f29f3650"
)
NULL_V3_WRAPPER_SHA256: str | None = (
    "5aa68bfdc69083b5f30fa1eaedb938f2b3af4ceb86d774c294d672373319637e"
)
NULL_V3_MERGE_SHA256: str | None = (
    "95a5dbc4927e53f0a22eba824076026b371261e4f9a8fdfc4afb577f1ff9be43"
)
NULL_V3_FIXED_DOMAIN_CORRECTION_SHA256: str | None = (
    "9c2d8f90fa546480a37b6e7546cbd9e9a740916325e5dec973f9fa7f8ef899ce"
)


WORKFLOW_IDENTITIES: dict[str, dict[str, Any]] = {
    "context_weight": {
        "directory_name": "post_hoc_context_weight_sensitivity_v2",
        "protocol_version": "post_hoc_context_weight_sensitivity_v2.0",
        "protocol_filename": "post_hoc_context_weight_sensitivity_protocol_v2.md",
        "protocol_sha256": (
            "07ab2cc9a85f9fffd55974fe0b0b7a040dbb3dbc8b5cc99ac409a504faf712ad"
        ),
        "runner_filename": "run_post_hoc_context_weight_sensitivity_v2.py",
        "runner_sha256": (
            "a007246e01cbfefd68286a506c9031a0d33c643a3f80164b963c99798d86e701"
        ),
        "artifact_files": CONTEXT_ARTIFACTS,
        "inventory_kind": "context",
        "inventory_exclusions": ("artifact_sha256.csv",),
        "allowed_directories": (),
    },
    "scaffold_source": {
        "directory_name": "post_hoc_scaffold_source_sensitivity_v3",
        "protocol_version": "post_hoc_scaffold_source_sensitivity_v3.0",
        "protocol_filename": "post_hoc_scaffold_source_sensitivity_protocol_v3.md",
        "protocol_sha256": (
            "5e67fe833b3b5d3f87d38ac3b28b6b7ac2600aebb2756676e031aed7bd2aab83"
        ),
        "runner_filename": "run_post_hoc_scaffold_source_sensitivity_v3.py",
        "runner_sha256": (
            "6000af04d0abcc5e3f97adf55c8be8876f9abe1bc0cfc8363d173f30aa979f0a"
        ),
        "qa_filename": "qa_post_hoc_scaffold_source_v3_all_target_identity.py",
        "qa_sha256": (
            "3636760961c858716d0eb087a4be5d8491058da45a61bc1a9821f73fb262b298"
        ),
        "execution_correction_filename": (
            "post_hoc_scaffold_source_thread_execution_correction_v3.md"
        ),
        "execution_correction_sha256": (
            "733b56540fb3917c6c2e3189b7c4d8c4bc33118a5f1fd2db3f132e46e3210456"
        ),
        "math_runner_filename": "run_post_hoc_scaffold_source_sensitivity_v2.py",
        "math_runner_sha256": (
            "449d8821ff31e9a5392a227a86ce78fc09c1660f293449a1fbc850a3ee8ffb34"
        ),
        "thread_audit_filename": "native_threadpool_audit.json",
        "identity_audit_filename": "all_target_exact_identity_audit.json",
        "identity_audit_markdown_filename": (
            "all_target_exact_identity_audit.md"
        ),
        "artifact_files": SCAFFOLD_ARTIFACTS,
        "inventory_kind": "scaffold",
        "inventory_exclusions": ("artifact_sha256.csv", "run_manifest.json"),
        "allowed_directories": (),
    },
    "null_applicability_censoring": {
        "directory_name": "post_hoc_null_applicability_censoring_v3",
        "protocol_version": "post_hoc_null_applicability_censoring_v3.0",
        "protocol_filename": "post_hoc_null_applicability_censoring_protocol_v3.md",
        "protocol_sha256": NULL_V3_PROTOCOL_SHA256,
        "runner_filename": "run_post_hoc_null_applicability_censoring_v3.py",
        "runner_sha256": NULL_V3_RUNNER_SHA256,
        "wrapper_filename": "run_post_hoc_permutation_shard_v3.py",
        "wrapper_sha256": NULL_V3_WRAPPER_SHA256,
        "merge_filename": "merge_post_hoc_permutation_shards_v3.py",
        "merge_sha256": NULL_V3_MERGE_SHA256,
        "qa_filename": "qa_post_hoc_applicability_fixed_domain_v3.py",
        "qa_sha256": (
            "7e51d44a9f0b352475b564047ad99c14e8c81583b4f0ed963abbbde6ab839e83"
        ),
        "fixed_domain_correction_filename": (
            "post_hoc_applicability_fixed_domain_correction_v3.md"
        ),
        "fixed_domain_correction_sha256": NULL_V3_FIXED_DOMAIN_CORRECTION_SHA256,
        "snapshot_name": "premerge_primary_snapshot_v3",
        "shard_suffix": "_v3",
        "artifact_files": NULL_V3_ARTIFACTS,
        "inventory_kind": "null",
        "inventory_exclusions": ("artifact_sha256.csv",),
        "allowed_directories": ("premerge_primary_snapshot_v3",),
    },
}


TABLE_WORKFLOW = {
    "portable_context": "context_weight",
    "weighting_bootstrap": "context_weight",
    "weighting_diagnostics": "context_weight",
    "generic_scaffold": "scaffold_source",
    "generic_scaffold_groups": "scaffold_source",
    "source_deletion": "scaffold_source",
    "applicability_models": "null_applicability_censoring",
    "applicability_domains": "null_applicability_censoring",
    "applicability_contrasts": "null_applicability_censoring",
    "permutation_inference": "null_applicability_censoring",
    "permutation_models": "null_applicability_censoring",
    "permutation_contrasts": "null_applicability_censoring",
    "permutation_diagnostics": "null_applicability_censoring",
    "censoring_overall": "null_applicability_censoring",
    "censoring_by_source": "null_applicability_censoring",
    "censoring_by_target": "null_applicability_censoring",
    "censoring_overlap": "null_applicability_censoring",
    "censoring_operator_counts": "null_applicability_censoring",
    "censoring_selection_flow": "null_applicability_censoring",
    "censoring_source_selection": "null_applicability_censoring",
}

SOURCE_FILENAMES = {
    "portable_context": "ood_paired_scaffold_bootstrap.csv",
    "weighting_bootstrap": "internal_paired_scaffold_bootstrap.csv",
    "weighting_diagnostics": "internal_weight_diagnostics.csv",
    "generic_scaffold": "generic_scaffold_paired_cluster_bootstrap.csv",
    "generic_scaffold_groups": "generic_scaffold_group_summary.csv",
    "source_deletion": "source_deletion_paired_cluster_bootstrap.csv",
    "applicability_models": "applicability_model_metrics.csv",
    "applicability_domains": "applicability_domain_metrics.csv",
    "applicability_contrasts": "applicability_paired_deltas.csv",
    "permutation_inference": "permutation_null_inference.csv",
    "permutation_models": "permutation_model_metrics.csv",
    "permutation_contrasts": "permutation_contrast_metrics.csv",
    "permutation_diagnostics": "permutation_fold_diagnostics.csv",
    "censoring_overall": "censoring_relation_overall.csv",
    "censoring_by_source": "censoring_relation_by_source.csv",
    "censoring_by_target": "censoring_relation_by_target.csv",
    "censoring_overlap": "censoring_overlap_with_core.csv",
    "censoring_operator_counts": "censoring_relation_operator_counts.csv",
    "censoring_selection_flow": "selection_flow_overall.csv",
    "censoring_source_selection": "selection_representation_by_source.csv",
}

OUTPUT_NAMES = {
    key: f"source_data_extension_{key}.csv" for key in TABLE_WORKFLOW
}
EXPECTED_TABLE_IDS = frozenset(TABLE_WORKFLOW)

INDEX_COLUMNS = (
    "table_id",
    "output_file",
    "source_file",
    "n_rows",
    "n_columns",
    "source_size_bytes",
    "source_sha256",
    "source_inventory_sha256",
    "evidence_identity",
    "protocol_version",
    "response_dtype",
    "float32_correction_sha256",
    "fixed_domain_correction_sha256",
    "contains_row_level_records",
    "contains_molecular_structures",
)

FORBIDDEN_AGGREGATE_COLUMNS = frozenset(
    {
        "canonical_smiles",
        "smiles",
        "qc_id",
        "record_id",
        "row_index",
        "source_id",
        "y_true",
        "y_pred",
        "split",
        "split_assignment",
        "fit_indices_sha256",
        "train_indices_sha256",
        "test_indices_sha256",
    }
)

FIGURE_STEMS = (
    "fig1_validation_gradient",
    "fig2_internal_evidence",
    "fig3_domain_transfer",
    "fig4_boundary_and_calibration",
    "fig5_robustness_map",
    "supp_fig_s1_applicability_profile",
    "supp_fig_s2_permutation_controls",
    "supp_fig_s3_endpoint_selection_boundary",
)
FIGURE_FORMATS = (".svg", ".pdf", ".tiff", ".png")

_FIGURE_QA_EXPECTED_TEXT = {
    "fig5_robustness_map": (
        "Held-axis context portability",
        "Scaffold-definition sensitivity",
        "Training-source deletion",
        "95% CI",
        "10,000 resamples",
    ),
    "supp_fig_s1_applicability_profile": (
        "Maximum train-set Tanimoto bin",
        "RMSE",
        "resamples",
        "No lines connect bins",
        "CI NE",
    ),
    "supp_fig_s2_permutation_controls": (
        "Permutation count",
        "one-sided",
        "All 100",
        "within-source",
    ),
    "supp_fig_s3_endpoint_selection_boundary": (
        "parsed-record rows",
        "not unique record IDs",
        "not fitted as point labels",
        "target categories are shown",
    ),
}


def _figure_qa_required_check_ids() -> tuple[str, ...]:
    check_ids: list[str] = []
    for stem in FIGURE_STEMS:
        check_ids.extend(
            f"{stem}:exists:{suffix}" for suffix in FIGURE_FORMATS
        )
        check_ids.extend(
            f"{stem}:size:{suffix}" for suffix in FIGURE_FORMATS
        )
        check_ids.extend(
            (
                f"{stem}:svg-width",
                f"{stem}:svg-height",
                f"{stem}:svg-editable-text",
            )
        )
        check_ids.extend(
            f"{stem}:text:{phrase}"
            for phrase in _FIGURE_QA_EXPECTED_TEXT.get(stem, ())
        )
        if stem in _FIGURE_QA_EXPECTED_TEXT:
            check_ids.append(f"{stem}:panel-labels")
        check_ids.extend(
            (
                f"{stem}:pdf-width",
                f"{stem}:pdf-height",
                f"{stem}:pdf-fonts",
                f"{stem}:.tiff:pixels",
                f"{stem}:.tiff:dpi",
                f"{stem}:tiff-rgb",
                f"{stem}:tiff-lzw",
                f"{stem}:.png:pixels",
                f"{stem}:.png:dpi",
            )
        )
    check_ids.extend(
        (
            "formal:figure-contract-sha256",
            "formal:figure-qa-script-sha256",
            "formal:source-provenance",
            "existing-figures:complete",
            "manifest:exists",
            "provenance-record:exists",
            "manifest:exact-figure-set",
            "manifest:unique-sha256",
        )
    )
    check_ids.extend(
        f"manifest:{stem}{suffix}"
        for stem in FIGURE_STEMS
        for suffix in FIGURE_FORMATS
    )
    check_ids.extend(
        (
            "provenance-record:identity",
            "formal:required-check-id-set",
        )
    )
    return tuple(sorted(check_ids))


FIGURE_QA_REQUIRED_CHECK_IDS = _figure_qa_required_check_ids()
FIGURE_QA_REQUIRED_CHECK_COUNT = 224
FIGURE_QA_REQUIRED_CHECK_IDS_SHA256 = (
    "f7c6f1901f9d04ebe9970be7fdb823b59150f95347ab52f799feb54bb891835e"
)
_figure_qa_check_ids_json = json.dumps(
    list(FIGURE_QA_REQUIRED_CHECK_IDS),
    ensure_ascii=True,
    separators=(",", ":"),
)
if (
    len(FIGURE_QA_REQUIRED_CHECK_IDS) != FIGURE_QA_REQUIRED_CHECK_COUNT
    or len(set(FIGURE_QA_REQUIRED_CHECK_IDS))
    != FIGURE_QA_REQUIRED_CHECK_COUNT
    or hashlib.sha256(
        _figure_qa_check_ids_json.encode("utf-8")
    ).hexdigest()
    != FIGURE_QA_REQUIRED_CHECK_IDS_SHA256
):
    raise RuntimeError("Frozen formal figure-QA check-ID contract is inconsistent")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def figure_generation_id(
    *,
    source_data_generation_id: str,
    figure_bundle_manifest_sha256: str,
    plot_script_sha256: str,
) -> str:
    """Bind a figure bundle to source, code, QA and its frozen contract."""

    return hashlib.sha256(
        "|".join(
            (
                ANALYSIS_IDENTITY,
                source_data_generation_id,
                figure_bundle_manifest_sha256,
                plot_script_sha256,
                FIGURE_QA_SCRIPT_SHA256,
                FIGURE_CONTRACT_SHA256,
                FIGURE_QA_REQUIRED_CHECK_IDS_SHA256,
                str(FIGURE_QA_REQUIRED_CHECK_COUNT),
            )
        ).encode("utf-8")
    ).hexdigest()


def require_figure_publication_contract_frozen() -> None:
    """Require the immutable figure contract, QA implementation and ID set."""

    frozen_files = (
        (
            "figure contract",
            ROUTE_DIR / "docs" / FIGURE_CONTRACT_FILENAME,
            FIGURE_CONTRACT_SHA256,
        ),
        (
            "figure QA script",
            SCRIPT_DIR / FIGURE_QA_SCRIPT_FILENAME,
            FIGURE_QA_SCRIPT_SHA256,
        ),
    )
    for label, path, expected in frozen_files:
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != expected
        ):
            raise RuntimeError(
                f"{label} is missing, symlinked or hash-mismatched"
            )


def downstream_software_bindings() -> dict[str, Any]:
    """Return the exact downstream code and cross-check contract identity."""

    bindings: dict[str, Any] = {
        key: sha256_file(SCRIPT_DIR / filename)
        for key, filename in DOWNSTREAM_SCRIPT_FILENAMES.items()
    }
    check_ids_json = json.dumps(
        list(CROSSCHECK_EXPECTED_CHECK_IDS),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    bindings.update(
        {
            "expected_crosscheck_check_ids": list(
                CROSSCHECK_EXPECTED_CHECK_IDS
            ),
            "expected_crosscheck_check_count": len(
                CROSSCHECK_EXPECTED_CHECK_IDS
            ),
            "expected_crosscheck_check_ids_sha256": hashlib.sha256(
                check_ids_json.encode("utf-8")
            ).hexdigest(),
        }
    )
    return bindings


def results_generation_id(
    source_hashes: Mapping[str, Any],
    software_bindings: Mapping[str, Any] | None = None,
) -> str:
    """Create the deterministic aggregate-to-summary generation identity."""

    software = (
        downstream_software_bindings()
        if software_bindings is None
        else software_bindings
    )
    return hashlib.sha256(
        "|".join(
            (
                str(source_hashes["source_data_generation_id"]),
                str(software["source_data_builder_sha256"]),
                str(software["results_summarizer_sha256"]),
                str(software["independent_crosscheck_sha256"]),
                str(software["expected_crosscheck_check_ids_sha256"]),
                str(software["expected_crosscheck_check_count"]),
            )
        ).encode("utf-8")
    ).hexdigest()


def crosscheck_generation_id(
    source_hashes: Mapping[str, Any],
    *,
    results_sha256: str,
    software_bindings: Mapping[str, Any] | None = None,
) -> str:
    """Create the deterministic identity of the independent audit record."""

    software = (
        downstream_software_bindings()
        if software_bindings is None
        else software_bindings
    )
    return hashlib.sha256(
        "|".join(
            (
                str(source_hashes["source_data_generation_id"]),
                str(results_sha256),
                str(software["independent_crosscheck_sha256"]),
                str(software["expected_crosscheck_check_ids_sha256"]),
                str(software["expected_crosscheck_check_count"]),
            )
        ).encode("utf-8")
    ).hexdigest()


def workflow_directory(workflow: str) -> Path:
    return REPORT_ROOT / str(WORKFLOW_IDENTITIES[workflow]["directory_name"])


def workflow_protocol_path(workflow: str) -> Path:
    return ROUTE_DIR / "docs" / str(
        WORKFLOW_IDENTITIES[workflow]["protocol_filename"]
    )


def workflow_runner_path(workflow: str) -> Path:
    return SCRIPT_DIR / str(WORKFLOW_IDENTITIES[workflow]["runner_filename"])


def expected_workflow_sources() -> dict[str, str]:
    return {
        workflow: f"reports/{identity['directory_name']}"
        for workflow, identity in WORKFLOW_IDENTITIES.items()
    }


def require_publication_lineage_frozen() -> None:
    """Verify every immutable code/protocol identity in the final lineage."""

    pending = []
    null_identity = WORKFLOW_IDENTITIES["null_applicability_censoring"]
    if not NULL_V3_PUBLICATION_READY:
        pending.append("NULL_V3_PUBLICATION_READY")
    for key in (
        "protocol_sha256",
        "runner_sha256",
        "wrapper_sha256",
        "merge_sha256",
        "fixed_domain_correction_sha256",
    ):
        value = null_identity.get(key)
        if not isinstance(value, str) or len(value) != 64:
            pending.append(f"null_applicability_censoring.{key}")
    if pending:
        raise RuntimeError(
            "Publication lineage is intentionally blocked pending the frozen "
            "null/applicability v3 workflow: " + ", ".join(pending)
        )

    master = ROUTE_DIR / "docs" / MASTER_PROTOCOL_FILENAME
    correction = ROUTE_DIR / "docs" / FLOAT32_CORRECTION_FILENAME
    frozen_files = (
        ("master protocol", master, MASTER_PROTOCOL_SHA256),
        ("float32 correction", correction, FLOAT32_CORRECTION_SHA256),
        (
            "mixed-lineage provenance contract",
            ROUTE_DIR / "docs" / PROVENANCE_CONTRACT_FILENAME,
            PROVENANCE_CONTRACT_SHA256,
        ),
    )
    for label, path, expected in frozen_files:
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != expected
        ):
            raise RuntimeError(f"{label} is missing, symlinked or hash-mismatched")
    for workflow, identity in WORKFLOW_IDENTITIES.items():
        for label, path, expected in (
            (
                "protocol",
                workflow_protocol_path(workflow),
                identity["protocol_sha256"],
            ),
            (
                "runner",
                workflow_runner_path(workflow),
                identity["runner_sha256"],
            ),
        ):
            if (
                not path.is_file()
                or path.is_symlink()
                or sha256_file(path) != expected
            ):
                raise RuntimeError(
                    f"{workflow} {label} is missing, symlinked or hash-mismatched"
                )
        if workflow == "null_applicability_censoring":
            fixed_correction = ROUTE_DIR / "docs" / str(
                identity["fixed_domain_correction_filename"]
            )
            if (
                not fixed_correction.is_file()
                or fixed_correction.is_symlink()
                or sha256_file(fixed_correction)
                != identity["fixed_domain_correction_sha256"]
            ):
                raise RuntimeError(
                    f"{workflow} fixed-domain correction is missing, "
                    "symlinked or hash-mismatched"
                )
            for label, filename_key, hash_key in (
                ("wrapper", "wrapper_filename", "wrapper_sha256"),
                ("merge helper", "merge_filename", "merge_sha256"),
                ("independent QA", "qa_filename", "qa_sha256"),
            ):
                path = SCRIPT_DIR / str(identity[filename_key])
                if (
                    not path.is_file()
                    or path.is_symlink()
                    or sha256_file(path) != identity[hash_key]
                ):
                    raise RuntimeError(
                        f"{workflow} {label} is missing, symlinked or hash-mismatched"
                    )
        elif workflow == "scaffold_source":
            execution_correction = ROUTE_DIR / "docs" / str(
                identity["execution_correction_filename"]
            )
            scaffold_dependencies = (
                (
                    "native-thread correction",
                    execution_correction,
                    identity["execution_correction_sha256"],
                ),
                (
                    "independent all-target identity QA",
                    SCRIPT_DIR / str(identity["qa_filename"]),
                    identity["qa_sha256"],
                ),
                (
                    "imported mathematical runner",
                    SCRIPT_DIR / str(identity["math_runner_filename"]),
                    identity["math_runner_sha256"],
                ),
            )
            for label, path, expected in scaffold_dependencies:
                if (
                    not path.is_file()
                    or path.is_symlink()
                    or sha256_file(path) != expected
                ):
                    raise RuntimeError(
                        f"{workflow} {label} is missing, symlinked or "
                        "hash-mismatched"
                    )


def _safe_basename(value: object) -> str:
    text = str(value)
    candidate = Path(text)
    if (
        not text
        or candidate.name != text
        or candidate.is_absolute()
        or "/" in text
        or "\\" in text
        or text in {".", ".."}
        or ".." in candidate.parts
    ):
        raise ValueError(f"unsafe inventory path: {text!r}")
    return text


def _strict_int(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, not bool")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{label} is not a finite integer: {value!r}")
    return int(numeric)


def validate_artifact_inventory(
    directory: Path,
    *,
    workflow: str,
    require_exact_directory_files: bool = True,
) -> dict[str, Any]:
    """Verify an upstream ``artifact_sha256.csv`` row by row.

    The inventory itself is the sole self-reference exception for context and
    null.  The scaffold runner also explicitly excludes its final
    ``run_manifest.json``; that file is still required in the exact directory
    set and its hash is bound separately by aggregate provenance.
    """

    identity = WORKFLOW_IDENTITIES[workflow]
    expected_files = set(map(str, identity["artifact_files"]))
    exclusions = set(map(str, identity["inventory_exclusions"]))
    inventory_path = directory / "artifact_sha256.csv"
    if (
        not directory.is_dir()
        or directory.is_symlink()
        or not inventory_path.is_file()
        or inventory_path.is_symlink()
    ):
        raise ValueError(f"invalid workflow directory or inventory: {directory}")

    entries = list(directory.iterdir())
    observed_files = {
        path.name
        for path in entries
        if path.is_file() and not path.is_symlink()
    }
    observed_directories = {
        path.name
        for path in entries
        if path.is_dir() and not path.is_symlink()
    }
    expected_directories = set(map(str, identity["allowed_directories"]))
    symlinked_files = [path.name for path in entries if path.is_symlink()]
    unsafe_entries = [
        path.name
        for path in entries
        if not path.is_symlink() and not path.is_file() and not path.is_dir()
    ]
    if symlinked_files:
        raise ValueError(
            f"{workflow} contains symlinked artefacts: {sorted(symlinked_files)}"
        )
    if unsafe_entries:
        raise ValueError(
            f"{workflow} contains unsupported filesystem entries: "
            f"{sorted(unsafe_entries)}"
        )
    if require_exact_directory_files and observed_files != expected_files:
        raise ValueError(
            f"{workflow} exact artefact set mismatch: "
            f"missing={sorted(expected_files - observed_files)}, "
            f"extra={sorted(observed_files - expected_files)}"
        )
    if require_exact_directory_files and observed_directories != expected_directories:
        raise ValueError(
            f"{workflow} exact auxiliary-directory set mismatch: "
            f"missing={sorted(expected_directories - observed_directories)}, "
            f"extra={sorted(observed_directories - expected_directories)}"
        )

    kind = str(identity["inventory_kind"])
    if kind == "context":
        expected_columns = (
            "relative_path",
            "sha256",
            "size_bytes",
            "kind",
            "n_rows",
            "n_columns",
            "verification",
        )
        path_column, size_column = "relative_path", "size_bytes"
    elif kind == "scaffold":
        expected_columns = ("file", "bytes", "sha256")
        path_column, size_column = "file", "bytes"
    elif kind == "null":
        expected_columns = ("relative_path", "sha256", "size_bytes")
        path_column, size_column = "relative_path", "size_bytes"
    else:
        raise ValueError(f"unknown inventory kind: {kind}")

    with inventory_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected_columns:
            raise ValueError(
                f"{workflow} inventory schema mismatch: {reader.fieldnames}"
            )
        rows = list(reader)
    row_names = [_safe_basename(row[path_column]) for row in rows]
    expected_rows = expected_files - exclusions
    if (
        len(row_names) != len(set(row_names))
        or set(row_names) != expected_rows
        or "artifact_sha256.csv" in row_names
    ):
        raise ValueError(
            f"{workflow} inventory exact row set mismatch: "
            f"missing={sorted(expected_rows - set(row_names))}, "
            f"extra={sorted(set(row_names) - expected_rows)}"
        )

    for row, name in zip(rows, row_names, strict=True):
        path = directory / name
        if (
            not path.is_file()
            or path.is_symlink()
            or path.resolve().parent != directory.resolve()
        ):
            raise ValueError(f"{workflow} inventory path is unsafe: {name}")
        observed_size = _strict_int(
            row[size_column], label=f"{workflow}/{name} size"
        )
        if observed_size != path.stat().st_size:
            raise ValueError(f"{workflow}/{name} size mismatch")
        observed_hash = str(row["sha256"])
        if len(observed_hash) != 64 or sha256_file(path) != observed_hash:
            raise ValueError(f"{workflow}/{name} SHA-256 mismatch")
        if kind == "context":
            if row["verification"] != "PASS":
                raise ValueError(
                    f"{workflow}/{name} inventory verification is not PASS"
                )
            expected_kind = path.suffix.lower().lstrip(".") or "file"
            if row["kind"] != expected_kind:
                raise ValueError(f"{workflow}/{name} kind mismatch")
            if path.suffix.lower() == ".csv":
                frame = pd.read_csv(path)
                if _strict_int(
                    row["n_rows"], label=f"{workflow}/{name} n_rows"
                ) != len(frame):
                    raise ValueError(f"{workflow}/{name} row-count mismatch")
                if _strict_int(
                    row["n_columns"], label=f"{workflow}/{name} n_columns"
                ) != len(frame.columns):
                    raise ValueError(f"{workflow}/{name} column-count mismatch")
            elif (
                str(row["n_rows"]).strip()
                or str(row["n_columns"]).strip()
            ):
                raise ValueError(
                    f"{workflow}/{name} non-CSV row/column fields must be blank"
                )

    return {
        "workflow": workflow,
        "inventory_sha256": sha256_file(inventory_path),
        "inventory_size_bytes": inventory_path.stat().st_size,
        "verified_row_count": len(rows),
        "expected_file_count": len(expected_files),
        "self_reference_exclusions": sorted(exclusions),
    }


def require_exact_columns(
    frame: pd.DataFrame,
    expected: Iterable[str],
    *,
    label: str,
) -> None:
    expected_tuple = tuple(expected)
    observed_tuple = tuple(map(str, frame.columns))
    if observed_tuple != expected_tuple:
        raise ValueError(
            f"{label} exact column allowlist mismatch: "
            f"expected={expected_tuple}, observed={observed_tuple}"
        )


def require_finite_numeric(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    label: str,
    allow_missing: Iterable[str] = (),
) -> None:
    allowed = set(allow_missing)
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="raise")
        values = numeric.to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError(f"{label}.{column} contains infinity")
        if column not in allowed and np.isnan(values).any():
            raise ValueError(f"{label}.{column} contains NaN")


def require_exact_integer_column(
    frame: pd.DataFrame,
    column: str,
    expected: int,
    *,
    label: str,
) -> None:
    values = pd.to_numeric(frame[column], errors="raise")
    array = values.to_numpy(dtype=float)
    if (
        len(array) == 0
        or not np.isfinite(array).all()
        or not np.equal(array, np.floor(array)).all()
        or not np.equal(array, expected).all()
        or len(np.unique(array)) != 1
    ):
        raise ValueError(
            f"{label}.{column} must be a complete, uniform integer "
            f"column equal to {expected}"
        )


def require_strict_boolean_column(
    path: Path,
    frame: pd.DataFrame,
    column: str,
    *,
    require_all_true: bool = False,
) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        raw = [row.get(column, "") for row in reader]
    if not raw or any(value not in {"True", "False"} for value in raw):
        raise ValueError(
            f"{path.name}.{column} must use canonical True/False tokens"
        )
    if not is_bool_dtype(frame[column].dtype):
        raise ValueError(f"{path.name}.{column} did not parse as bool")
    if require_all_true and not bool(frame[column].all()):
        raise ValueError(f"{path.name}.{column} contains False")


def validate_scaffold_v3_clean_run_args(
    configuration: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Require a fresh, unresumed, full formal scaffold/source execution."""

    workflow_identity = (
        WORKFLOW_IDENTITIES["scaffold_source"]
        if identity is None
        else identity
    )
    args = configuration.get("args")
    manifest_args = manifest.get("args")
    expected_args = {
        "mode": "both",
        "outer_folds": 5,
        "outer_repeats": 5,
        "inner_folds": 4,
        "n_estimators": 600,
        "n_jobs": 4,
        "bootstrap_replicates": 10_000,
        "seed": 260_531,
        "max_outer_splits": None,
        "max_targets": None,
        "max_deletions": None,
        "only_target": None,
        "baseline_only": False,
        "overwrite": True,
        "resume": False,
    }
    expected_arg_keys = set(expected_args) | {
        "data_file",
        "output_dir",
        "identity_archive_dir",
    }
    expected_data_file = (
        ROUTE_DIR
        / "data"
        / "processed"
        / "all_molglue_dc50_qc_train_test_standardized_context.csv"
    ).resolve()
    expected_archive = (REPORT_ROOT / "confirmatory_ood_cpu_v1").resolve()
    if not isinstance(args, dict) or not isinstance(manifest_args, dict):
        raise ValueError("scaffold/source v3 args must be JSON objects")
    output_candidate = Path(str(args.get("output_dir", "")))
    if (
        manifest_args != args
        or set(args) != expected_arg_keys
        or any(args.get(key) != value for key, value in expected_args.items())
        or Path(str(args.get("data_file", ""))).resolve()
        != expected_data_file
        or Path(str(args.get("identity_archive_dir", ""))).resolve()
        != expected_archive
        or output_candidate.is_absolute()
        or output_candidate.as_posix()
        != f"reports/{workflow_identity['directory_name']}"
    ):
        raise ValueError(
            "scaffold/source v3 publication run must be a clean, unresumed "
            "full formal execution"
        )
    return args


def validate_scaffold_v3_execution(directory: Path) -> dict[str, Any]:
    """Validate the formal native-thread and all-target identity evidence."""

    identity = WORKFLOW_IDENTITIES["scaffold_source"]

    def read_object(filename: str) -> dict[str, Any]:
        path = directory / filename
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or symlinked scaffold/source v3 file: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"scaffold/source v3 JSON is not an object: {path}")
        return payload

    configuration = read_object("scientific_configuration.json")
    manifest = read_object("run_manifest.json")
    qa = read_object("qa_summary.json")
    thread_audit = read_object(str(identity["thread_audit_filename"]))
    exact_identity = read_object(str(identity["identity_audit_filename"]))
    protocol_version = str(identity["protocol_version"])

    for label, payload in (
        ("configuration", configuration),
        ("manifest", manifest),
    ):
        if (
            payload.get("protocol_version") != protocol_version
            or payload.get("response_dtype") != "float32"
            or payload.get("data_sha256")
            != "e9494246bcbd198bc09d5b0d09721910c33f70f46ae859b76377a13775e3e4b2"
            or payload.get("master_protocol_sha256") != MASTER_PROTOCOL_SHA256
            or payload.get("float32_correction_document_sha256")
            != FLOAT32_CORRECTION_SHA256
            or payload.get("protocol_sha256") != identity["protocol_sha256"]
            or payload.get("thread_correction_document_sha256")
            != identity["execution_correction_sha256"]
            or payload.get("this_script_sha256") != identity["runner_sha256"]
            or payload.get("identity_qa_script_sha256")
            != identity["qa_sha256"]
        ):
            raise ValueError(f"scaffold/source v3 {label} identity mismatch")
        implementation = payload.get("mathematical_implementation")
        if (
            not isinstance(implementation, dict)
            or implementation.get("module")
            != identity["math_runner_filename"]
            or implementation.get("required_sha256")
            != identity["math_runner_sha256"]
            or implementation.get("observed_sha256")
            != identity["math_runner_sha256"]
        ):
            raise ValueError(
                f"scaffold/source v3 {label} mathematical runner mismatch"
            )

    validate_scaffold_v3_clean_run_args(
        configuration,
        manifest,
        identity=identity,
    )
    if (
        configuration.get("formal_settings_match") is not True
        or configuration.get("identity_settings_match") is not True
        or manifest.get("formal_settings_match") is not True
        or manifest.get("identity_settings_match") is not True
        or manifest.get("status") != "complete"
        or str(manifest.get("qa_status", "")).upper() != "PASS"
        or manifest.get("native_thread_status") != "PASS"
        or manifest.get("independent_identity_status") != "PASS"
    ):
        raise ValueError("scaffold/source v3 formal manifest is incomplete")

    checks = qa.get("checks")
    check_status = {
        str(item.get("check_id")): str(item.get("status", "")).upper()
        for item in checks
        if isinstance(item, dict)
    } if isinstance(checks, list) else {}
    required_checks = {
        "training_response_dtype_float32",
        "native_threadpool_contract",
        "native_threadpool_stage_completeness",
        "all_target_baseline_prediction_identity",
        "independent_all_target_identity_qa",
    }
    if (
        qa.get("protocol_version") != protocol_version
        or qa.get("status") != "PASS"
        or qa.get("formal_settings_match") is not True
        or qa.get("identity_settings_match") is not True
        or _strict_int(qa.get("n_fail"), label="scaffold/source QA n_fail") != 0
        or not required_checks.issubset(check_status)
        or any(check_status[name] != "PASS" for name in required_checks)
    ):
        raise ValueError("scaffold/source v3 runner QA is incomplete")

    required_stages = {
        "startup",
        "before_generic",
        "after_generic",
        "before_source",
        "after_source",
    }
    snapshots = thread_audit.get("snapshots")
    required_environment = {
        "OMP_NUM_THREADS": "24",
        "OPENBLAS_NUM_THREADS": "24",
    }
    if (
        thread_audit.get("protocol_version") != protocol_version
        or thread_audit.get("contract_version")
        != "scaffold_source_native_threads_v3.0"
        or thread_audit.get("status") != "PASS"
        or thread_audit.get("required_environment") != required_environment
        or thread_audit.get("required_native_pool_threads") != 24
        or not isinstance(snapshots, list)
        or len(snapshots) != len(required_stages)
        or {str(item.get("stage")) for item in snapshots if isinstance(item, dict)}
        != required_stages
    ):
        raise ValueError("scaffold/source v3 native-thread audit is incomplete")
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            raise ValueError("scaffold/source v3 thread snapshot is not an object")
        pools = snapshot.get("threadpools")
        relevant = [
            pool
            for pool in pools
            if isinstance(pool, dict)
            and str(pool.get("user_api", "")).lower() in {"openmp", "blas"}
        ] if isinstance(pools, list) else []
        if (
            snapshot.get("status") != "PASS"
            or snapshot.get("required_environment") != required_environment
            or snapshot.get("observed_environment", {}).get("OMP_NUM_THREADS")
            != "24"
            or snapshot.get("observed_environment", {}).get(
                "OPENBLAS_NUM_THREADS"
            )
            != "24"
            or _strict_int(
                snapshot.get("cpu_affinity_count"),
                label="scaffold/source CPU affinity",
            )
            < 24
            or snapshot.get("issues") != []
            or {str(pool.get("user_api", "")).lower() for pool in relevant}
            != {"openmp", "blas"}
            or any(
                _strict_int(
                    pool.get("num_threads"),
                    label="scaffold/source native pool threads",
                )
                != 24
                for pool in relevant
            )
        ):
            raise ValueError("scaffold/source v3 native-thread snapshot failed")

    model_differences = exact_identity.get("model_prediction_differences")
    if (
        exact_identity.get("status") != "PASS"
        or exact_identity.get("execution_profile") != "formal_complete_package"
        or exact_identity.get("fatal_error") is not None
        or _strict_int(
            exact_identity.get("n_fail"),
            label="scaffold/source identity n_fail",
        )
        != 0
        or _strict_int(
            exact_identity.get("n_pass"),
            label="scaffold/source identity n_pass",
        )
        != _strict_int(
            exact_identity.get("n_checks"),
            label="scaffold/source identity n_checks",
        )
        or not isinstance(model_differences, dict)
        or len(model_differences) != 16
        or any(
            not isinstance(record, dict)
            or record.get("array_equal") is not True
            or float(record.get("maximum_absolute_difference", math.nan)) != 0.0
            for record in model_differences.values()
        )
    ):
        raise ValueError(
            "scaffold/source v3 independent all-target identity audit failed"
        )
    identities = exact_identity.get("identities")
    if (
        not isinstance(identities, dict)
        or identities.get("audit_script_sha256") != identity["qa_sha256"]
        or identities.get("v3_runner_sha256") != identity["runner_sha256"]
        or identities.get("v3_protocol_sha256") != identity["protocol_sha256"]
        or identities.get("v3_native_threadpool_audit_sha256")
        != sha256_file(directory / str(identity["thread_audit_filename"]))
    ):
        raise ValueError("scaffold/source v3 identity-audit bindings mismatch")

    domain_path = directory / "source_deletion_domain_metrics.csv"
    domain = pd.read_csv(domain_path)
    require_exact_columns(
        domain,
        SCAFFOLD_SOURCE_DOMAIN_METRICS_COLUMNS,
        label=domain_path.name,
    )
    if (
        len(domain) != 80
        or domain[
            [
                "n_train_rows",
                "n_train_unique_smiles",
                "n_train_scaffolds",
                "n_test_rows",
                "n_test_unique_smiles",
                "n_test_scaffolds",
            ]
        ]
        .isna()
        .any()
        .any()
        or domain.groupby(
            ["heldout_target", "deletion_source", "model_id"],
            dropna=False,
        ).size().ne(1).any()
    ):
        raise ValueError(
            "scaffold/source v3 enriched domain metrics are incomplete"
        )

    return {
        "manifest": manifest,
        "configuration": configuration,
        "qa": qa,
        "native_threadpool_audit": thread_audit,
        "all_target_exact_identity_audit": exact_identity,
    }


def _publication_table_columns() -> Mapping[str, Iterable[str]]:
    """Load the builder's publication projection without duplicating schemas.

    The import is deliberately lazy: the builder imports this lineage module,
    but it never calls the aggregate validator while its own module is still
    initialising.
    """

    try:
        from build_computational_extension_source_data_v1 import (
            PUBLISH_COLUMN_ALLOWLISTS,
        )
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "Unable to load the frozen aggregate schema allowlists"
        ) from error
    if set(PUBLISH_COLUMN_ALLOWLISTS) != EXPECTED_TABLE_IDS:
        raise RuntimeError("Aggregate schema allowlist table set is incomplete")
    return PUBLISH_COLUMN_ALLOWLISTS


def validate_aggregate_value_binding(
    source_path: Path,
    aggregate_path: Path,
    publication_columns: Iterable[str],
    *,
    table_id: str,
) -> None:
    """Require exact cell-for-cell equality to the selected upstream columns."""

    if (
        not source_path.is_file()
        or source_path.is_symlink()
        or not aggregate_path.is_file()
        or aggregate_path.is_symlink()
    ):
        raise ValueError(f"{table_id}: source or aggregate is missing/symlinked")
    columns = tuple(publication_columns)
    source = pd.read_csv(source_path)
    aggregate = pd.read_csv(aggregate_path)
    require_exact_columns(
        aggregate,
        ("analysis_identity", *columns),
        label=aggregate_path.name,
    )
    # The aggregate is a CSV projection of an already parsed upstream CSV.
    # Normalise the expected projection through the identical pandas
    # writer/parser round trip before requiring bit-exact equality.  This
    # removes harmless last-bit changes introduced by CSV decimal
    # serialisation while preserving an exact post-serialisation tamper gate.
    expected_buffer = io.StringIO()
    source.loc[:, columns].reset_index(drop=True).to_csv(
        expected_buffer,
        index=False,
    )
    expected_buffer.seek(0)
    expected = pd.read_csv(expected_buffer)
    observed = aggregate.drop(columns=["analysis_identity"]).reset_index(
        drop=True
    )
    try:
        pd.testing.assert_frame_equal(
            observed,
            expected,
            check_exact=True,
            check_dtype=True,
            check_like=False,
            check_names=True,
        )
    except AssertionError as error:
        raise ValueError(
            f"{table_id}: aggregate values differ from selected upstream "
            "columns"
        ) from error


def aggregate_bundle_hashes(source_dir: Path) -> dict[str, str]:
    """Validate and bind the current aggregate index/checksum/provenance trio."""

    require_publication_lineage_frozen()
    validate_scaffold_v3_execution(workflow_directory("scaffold_source"))
    inventory_audits = {
        workflow: validate_artifact_inventory(
            workflow_directory(workflow),
            workflow=workflow,
        )
        for workflow in WORKFLOW_IDENTITIES
    }
    publication_columns = _publication_table_columns()
    index_path = source_dir / "source_data_extension_index.csv"
    checksum_path = source_dir / "source_data_extension_sha256.csv"
    provenance_path = source_dir / PROVENANCE_FILENAME
    for path in (index_path, checksum_path, provenance_path):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or symlinked aggregate identity file: {path}")

    index = pd.read_csv(index_path)
    require_exact_columns(index, INDEX_COLUMNS, label="extension index")
    require_strict_boolean_column(
        index_path,
        index,
        "contains_row_level_records",
    )
    require_strict_boolean_column(
        index_path,
        index,
        "contains_molecular_structures",
    )
    if (
        len(index) != len(EXPECTED_TABLE_IDS)
        or set(index["table_id"].astype(str)) != EXPECTED_TABLE_IDS
        or index["table_id"].duplicated().any()
    ):
        raise ValueError("extension index table identities are incomplete")
    for _, row in index.iterrows():
        table_id = str(row["table_id"])
        workflow = TABLE_WORKFLOW[table_id]
        identity = WORKFLOW_IDENTITIES[workflow]
        expected_output = OUTPUT_NAMES[table_id]
        expected_source = (
            Path("reports")
            / str(identity["directory_name"])
            / SOURCE_FILENAMES[table_id]
        ).as_posix()
        if (
            str(row["output_file"]) != expected_output
            or str(row["source_file"]) != expected_source
            or str(row["evidence_identity"]) != ANALYSIS_IDENTITY
            or str(row["protocol_version"]) != identity["protocol_version"]
            or str(row["response_dtype"]) != "float32"
            or str(row["float32_correction_sha256"])
            != FLOAT32_CORRECTION_SHA256
            or str(row["fixed_domain_correction_sha256"])
            != (
                str(identity["fixed_domain_correction_sha256"])
                if workflow == "null_applicability_censoring"
                else "not_applicable"
            )
            or row["contains_row_level_records"] not in (False, np.bool_(False))
            or row["contains_molecular_structures"] not in (
                False,
                np.bool_(False),
            )
        ):
            raise ValueError(f"invalid extension index lineage row: {table_id}")
        output_path = source_dir / expected_output
        source_path = ROUTE_DIR / expected_source
        inventory = inventory_audits[workflow]
        if (
            not output_path.is_file()
            or output_path.is_symlink()
            or not source_path.is_file()
            or source_path.is_symlink()
            or sha256_file(source_path) != str(row["source_sha256"])
            or source_path.stat().st_size
            != _strict_int(
                row["source_size_bytes"], label=f"{table_id} source size"
            )
            or sha256_file(
                source_path.parent / "artifact_sha256.csv"
            )
            != str(row["source_inventory_sha256"])
            or str(row["source_inventory_sha256"])
            != inventory["inventory_sha256"]
        ):
            raise ValueError(f"extension index source binding mismatch: {table_id}")
        aggregate = pd.read_csv(output_path)
        require_exact_columns(
            aggregate,
            ("analysis_identity", *publication_columns[table_id]),
            label=expected_output,
        )
        if (
            len(aggregate)
            != _strict_int(row["n_rows"], label=f"{table_id} n_rows")
            or len(aggregate.columns)
            != _strict_int(row["n_columns"], label=f"{table_id} n_columns")
            or "analysis_identity" not in aggregate.columns
            or aggregate.empty
            or not aggregate["analysis_identity"]
            .astype(str)
            .eq(ANALYSIS_IDENTITY)
            .all()
            or set(aggregate.columns) & FORBIDDEN_AGGREGATE_COLUMNS
        ):
            raise ValueError(f"invalid aggregate table: {expected_output}")
        validate_aggregate_value_binding(
            source_path,
            output_path,
            publication_columns[table_id],
            table_id=table_id,
        )

    checksum = pd.read_csv(checksum_path)
    require_exact_columns(
        checksum,
        ("relative_path", "size_bytes", "sha256"),
        label="extension checksum inventory",
    )
    expected_checksum_rows = set(index["output_file"].astype(str)) | {
        "source_data_extension_index.csv",
        PROVENANCE_FILENAME,
    }
    observed_checksum_rows = list(checksum["relative_path"].astype(str))
    if (
        set(observed_checksum_rows) != expected_checksum_rows
        or len(observed_checksum_rows) != len(set(observed_checksum_rows))
        or observed_checksum_rows != sorted(expected_checksum_rows)
    ):
        raise ValueError("extension checksum inventory exact set mismatch")
    for _, row in checksum.iterrows():
        relative = _safe_basename(row["relative_path"])
        path = source_dir / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != str(row["sha256"])
            or path.stat().st_size
            != _strict_int(row["size_bytes"], label=f"{relative} size")
        ):
            raise ValueError(f"extension checksum mismatch: {relative}")

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(provenance, dict):
        raise ValueError("aggregate provenance must be a JSON object")
    expected_provenance_keys = {
        "provenance_version",
        "analysis_identity",
        "provenance_contract_sha256",
        "master_protocol_sha256",
        "float32_correction_sha256",
        "response_dtype",
        "software_bindings",
        "workflows",
        "publication_controls",
    }
    workflows = provenance.get("workflows")
    controls = provenance.get("publication_controls")
    software_bindings = downstream_software_bindings()
    if (
        set(provenance) != expected_provenance_keys
        or provenance.get("provenance_version") != PROVENANCE_VERSION
        or provenance.get("analysis_identity") != ANALYSIS_IDENTITY
        or provenance.get("provenance_contract_sha256")
        != PROVENANCE_CONTRACT_SHA256
        or provenance.get("master_protocol_sha256") != MASTER_PROTOCOL_SHA256
        or provenance.get("response_dtype") != "float32"
        or provenance.get("float32_correction_sha256")
        != FLOAT32_CORRECTION_SHA256
        or provenance.get("software_bindings") != software_bindings
        or not isinstance(workflows, dict)
        or set(workflows) != set(WORKFLOW_IDENTITIES)
        or not isinstance(controls, dict)
        or set(controls)
        != {
            "superseded_workflow_results_included",
            "cross_version_workflow_mixing_permitted",
            "row_level_predictions_in_aggregate_source_data",
            (
                "publication_eligible_independent_prospective_"
                "dataset_available"
            ),
        }
        or controls.get("superseded_workflow_results_included") is not False
        or controls.get(
            "publication_eligible_independent_prospective_dataset_available"
        )
        is not False
        or controls.get("cross_version_workflow_mixing_permitted") is not False
        or controls.get("row_level_predictions_in_aggregate_source_data")
        is not False
    ):
        raise ValueError("aggregate provenance identity mismatch")
    for workflow, identity in WORKFLOW_IDENTITIES.items():
        entry = workflows.get(workflow)
        directory = workflow_directory(workflow)
        inventory = inventory_audits[workflow]
        expected_entry_keys = {
            "source_directory",
            "protocol_version",
            "response_dtype",
            "protocol_sha256",
            "runner_sha256",
            "artifact_inventory_sha256",
            "artifact_inventory_verified_rows",
            "artifact_inventory_self_reference_exclusions",
            "qa_summary_sha256",
            "run_manifest_sha256",
            "superseded_marker_present",
        }
        workflow_specific: dict[str, object]
        if workflow == "context_weight":
            workflow_specific = {
                "configuration_sha256": sha256_file(
                    directory / "configuration.json"
                ),
            }
        elif workflow == "scaffold_source":
            workflow_specific = {
                "scientific_configuration_sha256": sha256_file(
                    directory / "scientific_configuration.json"
                ),
                "native_threadpool_audit_sha256": sha256_file(
                    directory / str(identity["thread_audit_filename"])
                ),
                "all_target_exact_identity_audit_sha256": sha256_file(
                    directory / str(identity["identity_audit_filename"])
                ),
                "all_target_exact_identity_audit_markdown_sha256": sha256_file(
                    directory
                    / str(identity["identity_audit_markdown_filename"])
                ),
                "identity_qa_script_sha256": identity["qa_sha256"],
                "execution_correction_sha256": identity[
                    "execution_correction_sha256"
                ],
                "mathematical_runner_sha256": identity["math_runner_sha256"],
            }
        else:
            workflow_specific = {
                "permutation_merge_manifest_sha256": sha256_file(
                    directory / "permutation_shard_merge_manifest.json"
                ),
                "permutation_wrapper_sha256": sha256_file(
                    SCRIPT_DIR / str(identity["wrapper_filename"])
                ),
                "permutation_merge_helper_sha256": sha256_file(
                    SCRIPT_DIR / str(identity["merge_filename"])
                ),
                "fixed_domain_correction_sha256": identity[
                    "fixed_domain_correction_sha256"
                ],
            }
        expected_entry_keys.update(workflow_specific)
        if (
            not isinstance(entry, dict)
            or set(entry) != expected_entry_keys
            or entry.get("source_directory")
            != f"reports/{identity['directory_name']}"
            or entry.get("protocol_version") != identity["protocol_version"]
            or entry.get("response_dtype") != "float32"
            or entry.get("protocol_sha256") != identity["protocol_sha256"]
            or entry.get("runner_sha256") != identity["runner_sha256"]
            or entry.get("artifact_inventory_sha256")
            != inventory["inventory_sha256"]
            or entry.get("artifact_inventory_verified_rows")
            != inventory["verified_row_count"]
            or entry.get("artifact_inventory_self_reference_exclusions")
            != inventory["self_reference_exclusions"]
            or entry.get("qa_summary_sha256")
            != sha256_file(directory / "qa_summary.json")
            or entry.get("run_manifest_sha256")
            != sha256_file(directory / "run_manifest.json")
            or any(
                entry.get(key) != expected
                for key, expected in workflow_specific.items()
            )
            or entry.get("superseded_marker_present") is not False
            or (directory / "SUPERSEDED_DO_NOT_USE.md").exists()
        ):
            raise ValueError(f"aggregate workflow provenance mismatch: {workflow}")

    bound = {
        "analysis_identity": ANALYSIS_IDENTITY,
        "source_data_extension_index_sha256": sha256_file(index_path),
        "source_data_extension_checksums_sha256": sha256_file(checksum_path),
        "source_data_extension_provenance_sha256": sha256_file(provenance_path),
        "source_data_builder_sha256": software_bindings[
            "source_data_builder_sha256"
        ],
        "results_summarizer_sha256": software_bindings[
            "results_summarizer_sha256"
        ],
        "independent_crosscheck_sha256": software_bindings[
            "independent_crosscheck_sha256"
        ],
        "expected_crosscheck_check_ids_sha256": software_bindings[
            "expected_crosscheck_check_ids_sha256"
        ],
    }
    bound["source_data_generation_id"] = hashlib.sha256(
        "|".join(
            (
                bound["analysis_identity"],
                bound["source_data_extension_index_sha256"],
                bound["source_data_extension_checksums_sha256"],
                bound["source_data_extension_provenance_sha256"],
                bound["source_data_builder_sha256"],
                bound["results_summarizer_sha256"],
                bound["independent_crosscheck_sha256"],
                bound["expected_crosscheck_check_ids_sha256"],
                str(software_bindings["expected_crosscheck_check_count"]),
            )
        ).encode("utf-8")
    ).hexdigest()
    return bound


def _results_frame(source_dir: Path, table_id: str) -> pd.DataFrame:
    path = source_dir / OUTPUT_NAMES[table_id]
    frame = pd.read_csv(path)
    if (
        "analysis_identity" not in frame
        or frame.empty
        or not frame["analysis_identity"]
        .astype(str)
        .eq(ANALYSIS_IDENTITY)
        .all()
    ):
        raise ValueError(f"{table_id}: invalid summary input identity")
    return frame


def _results_one(frame: pd.DataFrame, *, label: str) -> pd.Series:
    if len(frame) != 1:
        raise ValueError(f"{label}: expected one row, observed {len(frame)}")
    return frame.iloc[0]


def _results_scalar(row: pd.Series, column: str) -> float:
    value = float(row[column])
    if not math.isfinite(value):
        raise ValueError(f"{column}: selected summary value is non-finite")
    return value


def _results_interval_direction(low: float, high: float) -> str:
    if low > 0:
        return "favourable_interval_above_zero"
    if high < 0:
        return "unfavourable_interval_below_zero"
    return "interval_includes_zero"


def _results_effect(
    analysis: str,
    condition: str,
    metric: str,
    estimate: float,
    low: float,
    high: float,
    direction: str,
    n_resamples: int,
) -> dict[str, Any]:
    return {
        "analysis": analysis,
        "condition": condition,
        "metric": metric,
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "interval_direction": _results_interval_direction(low, high),
        "positive_direction": direction,
        "n_resamples": n_resamples,
        "evidence_identity": "post_hoc_sensitivity",
    }


def recompute_results_components(source_dir: Path) -> dict[str, Any]:
    """Independently rebuild every numerical JSON result from aggregates."""

    effects: list[dict[str, Any]] = []
    portable = _results_frame(source_dir, "portable_context")
    for protocol in ("source_ood", "target_ood"):
        for contrast in (
            "full_vs_chemistry",
            "portable_vs_chemistry",
            "portable_vs_full",
        ):
            row = _results_one(
                portable[
                    portable["record_type"].eq("contrast")
                    & portable["protocol"].eq(protocol)
                    & portable["contrast_id"].eq(contrast)
                ],
                label=f"{protocol}/{contrast}",
            )
            for metric in ("spearman", "rmse"):
                prefix = f"delta_domain_macro_{metric}"
                effects.append(
                    _results_effect(
                        "held_axis_portable_context",
                        f"{protocol}:{contrast}",
                        prefix,
                        _results_scalar(row, f"{prefix}_observed"),
                        _results_scalar(row, f"{prefix}_ci_low"),
                        _results_scalar(row, f"{prefix}_ci_high"),
                        "positive_favours_first_named_model",
                        int(row["n_bootstrap"]),
                    )
                )

    weights = _results_frame(source_dir, "weighting_bootstrap")
    for contrast in (
        "uniform_full_vs_chemistry",
        "compound_equal_full_vs_chemistry",
        "domain_balanced_full_vs_chemistry",
    ):
        row = _results_one(
            weights[
                weights["record_type"].eq("contrast")
                & weights["contrast_id"].eq(contrast)
            ],
            label=contrast,
        )
        for metric in ("spearman", "rmse"):
            prefix = f"delta_{metric}"
            effects.append(
                _results_effect(
                    "internal_fit_weight",
                    contrast,
                    prefix,
                    _results_scalar(row, f"{prefix}_observed"),
                    _results_scalar(row, f"{prefix}_ci_low"),
                    _results_scalar(row, f"{prefix}_ci_high"),
                    "positive_favours_full_context",
                    int(row["n_bootstrap"]),
                )
            )

    generic = _results_frame(source_dir, "generic_scaffold")
    for metric in ("delta_spearman", "delta_rmse"):
        row = _results_one(
            generic[
                generic["record_type"].eq("contrast")
                & generic["contrast_id"].eq("full_vs_chemistry")
                & generic["metric"].eq(metric)
            ],
            label=f"generic/{metric}",
        )
        effects.append(
            _results_effect(
                "generic_murcko_scaffold",
                "full_vs_chemistry",
                metric,
                _results_scalar(row, "observed"),
                _results_scalar(row, "ci_low"),
                _results_scalar(row, "ci_high"),
                "positive_favours_full_context",
                int(row["n_bootstrap"]),
            )
        )

    source_deletion = _results_frame(source_dir, "source_deletion")
    deletions = ("none", "MGTbind", "MGDB", "MolGlueDB", "TPDdb")
    for deletion in deletions:
        for metric in ("delta_spearman", "delta_rmse"):
            row = _results_one(
                source_deletion[
                    source_deletion["record_type"].eq("full_vs_chemistry")
                    & source_deletion["contrast_type"].eq(
                        "full_vs_chemistry"
                    )
                    & source_deletion["deletion_source"].eq(deletion)
                    & source_deletion["heldout_target"].isna()
                    & source_deletion["metric"].eq(metric)
                ],
                label=f"source-deletion/{deletion}/{metric}",
            )
            effects.append(
                _results_effect(
                    "target_ood_training_source_deletion",
                    deletion,
                    metric,
                    _results_scalar(row, "observed"),
                    _results_scalar(row, "ci_low"),
                    _results_scalar(row, "ci_high"),
                    "positive_favours_full_context",
                    int(row["n_bootstrap"]),
                )
            )
    for deletion in deletions[1:]:
        for model_id in (
            "chemistry_extra_trees",
            "full_context_extra_trees",
        ):
            for metric in ("delta_spearman", "delta_rmse"):
                row = _results_one(
                    source_deletion[
                        source_deletion["record_type"].eq(
                            "deletion_vs_baseline"
                        )
                        & source_deletion["contrast_type"].eq(
                            "deletion_vs_baseline"
                        )
                        & source_deletion["deletion_source"].eq(deletion)
                        & source_deletion["model_id"].eq(model_id)
                        & source_deletion["heldout_target"].isna()
                        & source_deletion["metric"].eq(metric)
                    ],
                    label=f"deletion-baseline/{deletion}/{model_id}/{metric}",
                )
                effects.append(
                    _results_effect(
                        "target_ood_deletion_vs_within_script_baseline",
                        f"{deletion}:{model_id}",
                        metric,
                        _results_scalar(row, "observed"),
                        _results_scalar(row, "ci_low"),
                        _results_scalar(row, "ci_high"),
                        "positive_favours_deletion_condition",
                        int(row["n_bootstrap"]),
                    )
                )

    applicability = _results_frame(source_dir, "applicability_contrasts")
    for _, row in applicability.iterrows():
        internal = (
            row["regime"] == "internal_scaffold_disjoint"
            and row["protocol"] == "scaffold"
        )
        for metric in ("spearman", "rmse"):
            prefix = (
                f"delta_{metric}"
                if internal
                else f"delta_domain_macro_{metric}"
            )
            estimate = pd.to_numeric(
                pd.Series([row.get(prefix)]), errors="coerce"
            ).iloc[0]
            low = pd.to_numeric(
                pd.Series([row.get(f"{prefix}_ci_low")]), errors="coerce"
            ).iloc[0]
            high = pd.to_numeric(
                pd.Series([row.get(f"{prefix}_ci_high")]), errors="coerce"
            ).iloc[0]
            effects.append(
                {
                    "analysis": "chemical_novelty_applicability",
                    "condition": (
                        f"{row['regime']}:{row['protocol']}:"
                        f"{row['similarity_bin']}"
                    ),
                    "metric": prefix,
                    "estimate": None if pd.isna(estimate) else float(estimate),
                    "ci_low": None if pd.isna(low) else float(low),
                    "ci_high": None if pd.isna(high) else float(high),
                    "interval_direction": (
                        "not_estimable"
                        if pd.isna(low) or pd.isna(high)
                        else _results_interval_direction(
                            float(low), float(high)
                        )
                    ),
                    "positive_direction": "positive_favours_full_context",
                    "n_resamples": int(row["n_bootstrap_requested"]),
                    "n_rows": int(row["n_rows"]),
                    "n_scaffolds": int(row["n_scaffolds"]),
                    "n_domains_expected": int(row["n_domains_expected"]),
                    "evidence_identity": "post_hoc_diagnostic",
                }
            )

    weight_diagnostics: list[dict[str, Any]] = []
    diagnostic_frame = _results_frame(source_dir, "weighting_diagnostics")
    for weight_mode, group in diagnostic_frame.groupby(
        "weight_mode", sort=False
    ):
        clipping = pd.to_numeric(
            group["clipping_fraction"], errors="raise"
        )
        effective = pd.to_numeric(
            group["effective_sample_fraction"], errors="raise"
        )
        weight_diagnostics.append(
            {
                "weight_mode": str(weight_mode),
                "n_fit_records": int(len(group)),
                "fit_stage_counts": {
                    str(key): int(value)
                    for key, value in group["fit_stage"]
                    .value_counts()
                    .items()
                },
                "n_fit_rows_min": int(group["n_fit_rows"].min()),
                "n_fit_rows_max": int(group["n_fit_rows"].max()),
                "clipping_fraction_mean": float(clipping.mean()),
                "clipping_fraction_max": float(clipping.max()),
                "effective_sample_fraction_min": float(effective.min()),
                "effective_sample_fraction_median": float(effective.median()),
                "effective_sample_fraction_max": float(effective.max()),
            }
        )

    group_row = _results_one(
        _results_frame(source_dir, "generic_scaffold_groups"),
        label="generic scaffold groups",
    )
    generic_groups = {
        key: int(group_row[key])
        for key in (
            "n_rows",
            "n_unique_smiles",
            "n_bemis_murcko_groups",
            "n_generic_murcko_groups",
            "n_singleton_generic_groups",
            "largest_generic_group_rows",
        )
    }

    permutation_records: list[dict[str, Any]] = []
    inference = _results_frame(source_dir, "permutation_inference")
    for _, row in inference.iterrows():
        permutation_records.append(
            {
                "analysis": "conditional_label_randomization",
                "estimand_id": row["estimand_id"],
                "metric": row["metric"],
                "observed": _results_scalar(row, "observed"),
                "null_mean": _results_scalar(row, "null_mean"),
                "null_percentile_2p5": _results_scalar(
                    row, "null_percentile_2p5"
                ),
                "null_percentile_97p5": _results_scalar(
                    row, "null_percentile_97p5"
                ),
                "empirical_tail": row["empirical_tail"],
                "empirical_p": _results_scalar(row, "empirical_p"),
                "n_permutations": int(row["n_permutations_finite"]),
                "evidence_identity": (
                    "post_hoc_repeated_negative_control"
                ),
            }
        )

    overall = _results_frame(source_dir, "censoring_overall")
    permutation_diagnostics = _results_frame(
        source_dir, "permutation_diagnostics"
    )
    selection = _results_frame(source_dir, "censoring_selection_flow").drop(
        columns=["analysis_identity"]
    )
    censoring = {
        "parsed_total": int(overall["n_records"].sum()),
        "relation_counts": {
            str(row["relation_class"]): int(row["n_records"])
            for _, row in overall.iterrows()
        },
        "relation_fractions": {
            str(row["relation_class"]): float(row["fraction_within_group"])
            for _, row in overall.iterrows()
        },
        "permutation_singleton_row_fraction_mean": float(
            permutation_diagnostics[
                "permutation_singleton_row_fraction"
            ].mean()
        ),
        "permutation_changed_label_fraction_mean": float(
            permutation_diagnostics[
                "permutation_changed_label_fraction"
            ].mean()
        ),
        "selection_flow": selection.to_dict(orient="records"),
        "evidence_identity": "post_hoc_descriptive_audit",
    }
    return {
        "effect_records": effects,
        "weight_diagnostics": weight_diagnostics,
        "generic_scaffold_group_summary": generic_groups,
        "permutation_records": permutation_records,
        "censoring_and_randomization_diagnostics": censoring,
        "claim_boundary": [
            (
                "All analyses are post-hoc sensitivities, diagnostics or "
                "repeated negative controls."
            ),
            (
                "Intervals are conditional on the frozen observations and "
                "fixed OOD domains."
            ),
            (
                "No result is prospective validation, calibrated absolute "
                "prediction or mechanistic evidence."
            ),
            PUBLIC_PROSPECTIVE_DATASET_STATEMENT,
        ],
    }


def validate_results_payload_from_aggregates(
    payload: Mapping[str, Any],
    source_dir: Path,
    source_hashes: Mapping[str, str] | None = None,
) -> None:
    """Reject any numerical brief that is not exactly aggregate-derived."""

    hashes = (
        aggregate_bundle_hashes(source_dir)
        if source_hashes is None
        else dict(source_hashes)
    )
    validate_bound_payload(payload, hashes, label="extension results")
    software = downstream_software_bindings()
    expected_keys = {
        "analysis_identity",
        "source_data_provenance",
        "software_bindings",
        "results_generation_id",
        "effect_records",
        "weight_diagnostics",
        "generic_scaffold_group_summary",
        "permutation_records",
        "censoring_and_randomization_diagnostics",
        "claim_boundary",
    }
    if (
        set(payload) != expected_keys
        or payload.get("software_bindings") != software
        or payload.get("results_generation_id")
        != results_generation_id(hashes, software)
    ):
        raise ValueError("extension results software/generation binding mismatch")
    expected_components = recompute_results_components(source_dir)
    for key, expected in expected_components.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"extension results are not exact aggregate recomputation: {key}"
            )


def validate_crosscheck_payload(
    payload: Mapping[str, Any],
    expected_hashes: Mapping[str, str],
    *,
    results_path: Path,
) -> None:
    """Require the complete independently recomputed audit and code identity."""

    software = downstream_software_bindings()
    if not results_path.is_file() or results_path.is_symlink():
        raise ValueError("crosscheck-bound results JSON is missing or symlinked")
    results_sha256 = sha256_file(results_path)
    checks = payload.get("checks")
    provenance = payload.get("provenance")
    if not isinstance(checks, list) or not isinstance(provenance, Mapping):
        raise ValueError("crosscheck checks/provenance are malformed")
    observed_ids = [
        str(item.get("check_id", ""))
        for item in checks
        if isinstance(item, Mapping)
    ]
    expected_ids = list(CROSSCHECK_EXPECTED_CHECK_IDS)
    expected_root_keys = {
        "status",
        "n_checks",
        "n_pass",
        "n_fail",
        "analysis_identity",
        "audit_identity",
        "expected_check_ids",
        "expected_check_count",
        "provenance",
        "checks",
    }
    if (
        set(payload) != expected_root_keys
        or payload.get("status") != "PASS"
        or payload.get("analysis_identity") != ANALYSIS_IDENTITY
        or payload.get("audit_identity")
        != "independent_computational_extension_crosscheck_v1"
        or payload.get("expected_check_ids") != expected_ids
        or payload.get("expected_check_count") != len(expected_ids)
        or observed_ids != expected_ids
        or len(observed_ids) != len(set(observed_ids))
        or any(
            not isinstance(item, Mapping)
            or set(item) != {"check_id", "status", "detail"}
            or item.get("status") != "PASS"
            for item in checks
        )
        or payload.get("n_checks") != len(expected_ids)
        or payload.get("n_pass") != len(expected_ids)
        or payload.get("n_fail") != 0
    ):
        raise ValueError("crosscheck exact check-ID/status contract mismatch")
    expected_provenance_keys = set(expected_hashes) | {
        "software_bindings",
        "computational_extension_results_sha256",
        "crosscheck_generation_id",
        "final_analysis_identity",
        "response_dtype",
        "float32_correction_sha256",
        "workflow_directories",
        "superseded_workflow_results_used",
    }
    if (
        set(provenance) != expected_provenance_keys
        or provenance.get("final_analysis_identity") != ANALYSIS_IDENTITY
        or provenance.get("response_dtype") != "float32"
        or provenance.get("float32_correction_sha256")
        != FLOAT32_CORRECTION_SHA256
        or provenance.get("workflow_directories")
        != expected_workflow_sources()
        or provenance.get("superseded_workflow_results_used") is not False
    ):
        raise ValueError("crosscheck exact final-lineage provenance mismatch")
    for key, expected in expected_hashes.items():
        if provenance.get(key) != expected:
            raise ValueError(f"crosscheck stale aggregate binding: {key}")
    if (
        provenance.get("software_bindings") != software
        or provenance.get("computational_extension_results_sha256")
        != results_sha256
        or provenance.get("crosscheck_generation_id")
        != crosscheck_generation_id(
            expected_hashes,
            results_sha256=results_sha256,
            software_bindings=software,
        )
    ):
        raise ValueError("crosscheck software/results generation mismatch")


def validate_bound_payload(
    payload: Mapping[str, Any],
    expected_hashes: Mapping[str, str],
    *,
    label: str,
) -> None:
    if payload.get("analysis_identity") != ANALYSIS_IDENTITY:
        raise ValueError(f"{label} analysis identity mismatch")
    provenance = payload.get("source_data_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{label} lacks source-data provenance")
    if set(provenance) != set(expected_hashes):
        raise ValueError(f"{label} source-data provenance key set mismatch")
    for key, expected in expected_hashes.items():
        if provenance.get(key) != expected:
            raise ValueError(f"{label} stale or mismatched binding: {key}")
