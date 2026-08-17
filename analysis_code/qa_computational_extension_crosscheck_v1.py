#!/usr/bin/env python3
"""Independent cross-check of the post-hoc computational extension.

This verifier does not reuse the analysis scripts' metric helpers.  It
recomputes central point estimates from stored predictions or domain tables,
checks the permutation tail formula, confirms formal replicate counts, audits
aggregate source-data safety, and writes a concise PASS/FAIL report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from build_computational_extension_source_data_v1 import (
    INPUTS,
    SOURCE_COLUMN_ALLOWLISTS,
    validate_source_deletion_baseline_identity,
    validate_source_table,
)
from computational_extension_lineage_v1 import (
    ANALYSIS_IDENTITY,
    CROSSCHECK_EXPECTED_CHECK_IDS,
    FLOAT32_CORRECTION_FILENAME,
    FLOAT32_CORRECTION_SHA256,
    MASTER_PROTOCOL_FILENAME,
    MASTER_PROTOCOL_SHA256,
    PROVENANCE_FILENAME,
    WORKFLOW_IDENTITIES as CENTRAL_WORKFLOW_IDENTITIES,
    aggregate_bundle_hashes,
    crosscheck_generation_id,
    downstream_software_bindings,
    expected_workflow_sources,
    require_exact_columns,
    require_publication_lineage_frozen,
    validate_artifact_inventory,
    validate_results_payload_from_aggregates,
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
SOURCE_DIR = REPORT_DIR / "publication_validation_v1"
DEFAULT_JSON = SOURCE_DIR / "computational_extension_crosscheck_v1.json"
DEFAULT_MD = SOURCE_DIR / "computational_extension_crosscheck_v1.md"
DEFAULT_RESULTS_JSON = SOURCE_DIR / "computational_extension_results_v1.json"

CORE_FILE = (
    ROUTE_DIR
    / "data"
    / "processed"
    / "all_molglue_dc50_qc_train_test_standardized_context.csv"
)
PARSED_FILE = (
    ROUTE_DIR / "data" / "processed" / "all_molglue_dc50_parsed_records.csv"
)
MASTER_PROTOCOL = ROUTE_DIR / "docs" / MASTER_PROTOCOL_FILENAME
CORRECTION_DOCUMENT = ROUTE_DIR / "docs" / FLOAT32_CORRECTION_FILENAME
CONTEXT_PROTOCOL = workflow_protocol_path("context_weight")
SCAFFOLD_PROTOCOL = workflow_protocol_path("scaffold_source")
NULL_PROTOCOL = workflow_protocol_path("null_applicability_censoring")
EXPECTED_HASHES = {
    "core": "e9494246bcbd198bc09d5b0d09721910c33f70f46ae859b76377a13775e3e4b2",
    "parsed": "ae1a16613b41ce84f92f539dbd31f16ada272ac5e401250acc7e97fd5e8bd970",
    "master": MASTER_PROTOCOL_SHA256,
    "correction": FLOAT32_CORRECTION_SHA256,
    "context_protocol": CENTRAL_WORKFLOW_IDENTITIES["context_weight"][
        "protocol_sha256"
    ],
    "scaffold_protocol": CENTRAL_WORKFLOW_IDENTITIES["scaffold_source"][
        "protocol_sha256"
    ],
    "null_protocol": CENTRAL_WORKFLOW_IDENTITIES[
        "null_applicability_censoring"
    ]["protocol_sha256"],
}
NULL_IDENTITY = CENTRAL_WORKFLOW_IDENTITIES["null_applicability_censoring"]
EXPECTED_NULL_PARENT_SHA256 = NULL_IDENTITY["runner_sha256"]
EXPECTED_PERMUTATION_WRAPPER_SHA256 = NULL_IDENTITY["wrapper_sha256"]
EXPECTED_PERMUTATION_MERGE_SHA256 = NULL_IDENTITY["merge_sha256"]
WORKFLOW_IDENTITIES = {
    workflow: {
        **identity,
        "protocol_path": workflow_protocol_path(workflow),
        "runner_path": workflow_runner_path(workflow),
    }
    for workflow, identity in CENTRAL_WORKFLOW_IDENTITIES.items()
}
MODELS = ["chemistry_extra_trees", "full_context_extra_trees"]
WEIGHT_SUFFIXES = ["uniform", "compound_equal", "domain_balanced_clipped"]
FORBIDDEN_AGGREGATE_COLUMNS = {
    "canonical_smiles",
    "smiles",
    "qc_id",
    "record_id",
    "row_index",
    "source_id",
    "y_true",
    "y_pred",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-dir", type=Path, default=CONTEXT_DIR)
    parser.add_argument("--scaffold-dir", type=Path, default=SCAFFOLD_DIR)
    parser.add_argument("--null-dir", type=Path, default=NULL_DIR)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    parser.add_argument(
        "--results-json",
        type=Path,
        default=DEFAULT_RESULTS_JSON,
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def route_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROUTE_DIR.resolve()))
    except ValueError as error:
        raise ValueError(
            f"Publication cross-check path must be inside the route: {path.name}"
        ) from error


def portable_detail(value: object) -> str:
    """Remove host-specific route prefixes from public audit details."""

    return str(value).replace(str(ROUTE_DIR.resolve()), ".")


def status_from_qa(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload.get("status", payload.get("overall_status", ""))).upper()


def close(a: object, b: object, tolerance: float = 1e-10) -> bool:
    try:
        left = float(a)
        right = float(b)
    except (TypeError, ValueError):
        return False
    if math.isnan(left) and math.isnan(right):
        return True
    return bool(np.isfinite(left) and np.isfinite(right) and abs(left - right) <= tolerance)


def independent_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y_true = pd.to_numeric(frame["y_true"], errors="coerce").to_numpy(float)
    y_pred = pd.to_numeric(frame["y_pred"], errors="coerce").to_numpy(float)
    if y_true.size == 0 or not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("Non-finite or empty prediction stratum")
    correlation = float(spearmanr(y_true, y_pred).statistic)
    residual = y_true - y_pred
    return {
        "spearman": correlation,
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
    }


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(self, check_id: str, passed: bool, detail: str) -> None:
        self.checks.append(
            {
                "check_id": check_id,
                "status": "PASS" if passed else "FAIL",
                "detail": portable_detail(detail),
            }
        )

    def require_files(self, check_id: str, paths: Iterable[Path]) -> None:
        paths = list(paths)
        missing = [str(path) for path in paths if not path.is_file()]
        self.add(
            check_id,
            not missing,
            "all present" if not missing else "missing=" + "; ".join(missing),
        )

    @property
    def passed(self) -> bool:
        return all(item["status"] == "PASS" for item in self.checks)


def audit_input_identity(audit: Audit) -> None:
    for check_id, path, expected in (
        ("core_sha256", CORE_FILE, EXPECTED_HASHES["core"]),
        ("parsed_sha256", PARSED_FILE, EXPECTED_HASHES["parsed"]),
        ("master_protocol_sha256", MASTER_PROTOCOL, EXPECTED_HASHES["master"]),
        (
            "float32_correction_sha256",
            CORRECTION_DOCUMENT,
            EXPECTED_HASHES["correction"],
        ),
        (
            "context_v2_protocol_sha256",
            CONTEXT_PROTOCOL,
            EXPECTED_HASHES["context_protocol"],
        ),
        (
            "scaffold_v3_protocol_sha256",
            SCAFFOLD_PROTOCOL,
            EXPECTED_HASHES["scaffold_protocol"],
        ),
        (
            "null_v3_protocol_sha256",
            NULL_PROTOCOL,
            EXPECTED_HASHES["null_protocol"],
        ),
    ):
        if not path.is_file():
            audit.add(check_id, False, f"missing={path}")
            continue
        observed = sha256_file(path)
        audit.add(check_id, observed == expected, f"observed={observed}")


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def qa_check_passed(qa: dict[str, Any], check_id: str) -> bool:
    for check in qa.get("checks", []):
        if not isinstance(check, dict):
            continue
        observed_id = str(check.get("check_id", check.get("check", "")))
        if observed_id == check_id:
            return str(check.get("status", "")).upper() == "PASS"
    return False


def audit_formal_workflow_provenance(
    audit: Audit,
    context_dir: Path,
    scaffold_dir: Path,
    null_dir: Path,
) -> None:
    try:
        require_publication_lineage_frozen()
    except (OSError, RuntimeError, ValueError) as error:
        audit.add("publication_lineage_frozen", False, str(error))
        return
    audit.add(
        "publication_lineage_frozen",
        True,
        f"analysis_identity={ANALYSIS_IDENTITY}",
    )

    directories = {
        "context_weight": context_dir,
        "scaffold_source": scaffold_dir,
        "null_applicability_censoring": null_dir,
    }
    for workflow, directory in directories.items():
        identity = WORKFLOW_IDENTITIES[workflow]
        directory_ok = (
            directory.name == identity["directory_name"]
            and directory.is_dir()
            and not directory.is_symlink()
            and not (directory / "SUPERSEDED_DO_NOT_USE.md").exists()
        )
        audit.add(
            f"{workflow}_exact_directory_and_supersession_guard",
            directory_ok,
            f"directory={directory}; expected={identity['directory_name']}",
        )
        if not directory_ok:
            continue
        try:
            inventory = validate_artifact_inventory(
                directory, workflow=workflow
            )
            audit.add(
                f"{workflow}_artifact_inventory_rowwise",
                True,
                (
                    f"rows={inventory['verified_row_count']}; "
                    f"sha256={inventory['inventory_sha256']}"
                ),
            )
        except (OSError, ValueError, TypeError) as error:
            audit.add(
                f"{workflow}_artifact_inventory_rowwise",
                False,
                str(error),
            )

        required = [
            directory / "run_manifest.json",
            directory / "qa_summary.json",
        ]
        audit.require_files(f"{workflow}_formal_provenance_files", required)
        if not all(path.is_file() for path in required):
            continue
        try:
            manifest = read_json_object(required[0])
            qa = read_json_object(required[1])
            status = str(
                qa.get("overall_status", qa.get("status", ""))
            ).upper()
            common_ok = (
                manifest.get("protocol_version")
                == identity["protocol_version"]
                and status == "PASS"
                and str(manifest.get("qa_status", "PASS")).upper() == "PASS"
            )
            if workflow == "context_weight":
                common_ok &= (
                    manifest.get("training_response_dtype") == "float32"
                    and isinstance(manifest.get("correction_document"), dict)
                    and manifest["correction_document"].get("sha256")
                    == EXPECTED_HASHES["correction"]
                )
            elif workflow == "scaffold_source":
                execution = validate_scaffold_v3_execution(directory)
                common_ok &= (
                    manifest.get("response_dtype") == "float32"
                    and manifest.get(
                        "float32_correction_document_sha256"
                    )
                    == EXPECTED_HASHES["correction"]
                    and manifest.get("formal_settings_match") is True
                    and manifest.get("identity_settings_match") is True
                    and manifest.get("status") == "complete"
                    and manifest.get("native_thread_status") == "PASS"
                    and manifest.get("independent_identity_status") == "PASS"
                    and execution["native_threadpool_audit"].get("status")
                    == "PASS"
                    and execution["all_target_exact_identity_audit"].get(
                        "status"
                    )
                    == "PASS"
                )
            else:
                merge = read_json_object(
                    directory / "permutation_shard_merge_manifest.json"
                )
                common_ok &= (
                    manifest.get("float32_correction_sha256")
                    == EXPECTED_HASHES["correction"]
                    and manifest.get("applicability_correction_sha256")
                    == identity["fixed_domain_correction_sha256"]
                    and merge.get("status") == "PASS"
                    and merge.get("protocol_version")
                    == identity["protocol_version"]
                    and merge.get("scientific_implementation_sha256")
                    == identity["runner_sha256"]
                    and merge.get("execution_wrapper_sha256")
                    == identity["wrapper_sha256"]
                    and merge.get("applicability_correction_sha256")
                    == identity["fixed_domain_correction_sha256"]
                    and sha256_file(
                        SCRIPT_DIR / str(identity["merge_filename"])
                    )
                    == identity["merge_sha256"]
                    and merge.get("scientific_change") is False
                )
            audit.add(
                f"{workflow}_formal_manifest_identity",
                bool(common_ok),
                (
                    f"protocol={manifest.get('protocol_version')}; "
                    f"qa={status}"
                ),
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            audit.add(
                f"{workflow}_formal_manifest_identity",
                False,
                str(error),
            )

    exact_schema_ok = True
    schema_errors: list[str] = []
    for key, (default_path, _) in INPUTS.items():
        workflow = {
            "context_weight": context_dir,
            "scaffold_source": scaffold_dir,
            "null_applicability_censoring": null_dir,
        }[
            next(
                workflow
                for workflow, identity in WORKFLOW_IDENTITIES.items()
                if default_path.parent.name == identity["directory_name"]
            )
        ]
        path = workflow / default_path.name
        try:
            frame = pd.read_csv(path)
            require_exact_columns(
                frame, SOURCE_COLUMN_ALLOWLISTS[key], label=path.name
            )
            validate_source_table(key, path, frame)
        except (OSError, ValueError, TypeError, KeyError) as error:
            exact_schema_ok = False
            schema_errors.append(f"{key}: {error}")
    audit.add(
        "formal_source_tables_exact_schema_dtype_and_counts",
        exact_schema_ok,
        "all 20 exact" if exact_schema_ok else "; ".join(schema_errors[:8]),
    )


def audit_upstream_qa(
    audit: Audit, context_dir: Path, scaffold_dir: Path, null_dir: Path
) -> None:
    for label, directory in (
        ("context_weight", context_dir),
        ("scaffold_source", scaffold_dir),
        ("null_applicability_censoring", null_dir),
    ):
        path = directory / "qa_summary.json"
        if not path.is_file():
            audit.add(f"{label}_upstream_qa", False, f"missing={path}")
            continue
        status = status_from_qa(path)
        audit.add(f"{label}_upstream_qa", status == "PASS", f"status={status}")


def audit_context(audit: Audit, directory: Path) -> None:
    required = [
        directory / "configuration.json",
        directory / "internal_repeat_averaged_predictions.csv",
        directory / "internal_summary_metrics.csv",
        directory / "internal_paired_scaffold_bootstrap.csv",
        directory / "internal_split_audit.csv",
        directory / "ood_domain_metrics.csv",
        directory / "ood_aggregate_metrics.csv",
        directory / "ood_paired_scaffold_bootstrap.csv",
    ]
    audit.require_files("context_required_artifacts", required)
    if not all(path.is_file() for path in required):
        return
    config = json.loads(required[0].read_text(encoding="utf-8"))
    formal = (
        config.get("analysis_mode") == "formal_post_hoc_sensitivity"
        and int(config.get("outer_folds", 0)) == 5
        and int(config.get("outer_repeats", 0)) == 5
        and int(config.get("n_estimators", 0)) == 600
        and int(config.get("bootstrap_replicates", 0)) == 10_000
    )
    audit.add("context_formal_configuration", formal, str(config.get("analysis_mode")))

    predictions = pd.read_csv(required[1])
    summary = pd.read_csv(required[2])
    model_ids = {
        f"{model}__{suffix}" for model in MODELS for suffix in WEIGHT_SUFFIXES
    }
    audit.add(
        "context_internal_model_set",
        set(predictions["model_id"]) == model_ids,
        f"models={sorted(predictions['model_id'].unique())}",
    )
    prediction_shape_ok = (
        len(predictions) == 1_560 * len(model_ids)
        and predictions.groupby("model_id").size().eq(1_560).all()
        and not predictions.duplicated(["model_id", "row_index"]).any()
    )
    audit.add(
        "context_internal_prediction_shape",
        prediction_shape_ok,
        f"rows={len(predictions)}; models={len(model_ids)}",
    )
    metric_ok = True
    max_difference = 0.0
    for model_id, group in predictions.groupby("model_id", sort=False):
        observed = independent_metrics(group)
        stored_rows = summary[summary["model_id"] == model_id]
        if len(stored_rows) != 1:
            metric_ok = False
            continue
        stored = stored_rows.iloc[0]
        for metric in ("spearman", "rmse", "mae"):
            difference = abs(observed[metric] - float(stored[metric]))
            max_difference = max(max_difference, difference)
            metric_ok &= close(observed[metric], stored[metric])
    audit.add(
        "context_internal_metrics_independent_recompute",
        metric_ok,
        f"max_abs_difference={max_difference:.3g}",
    )

    splits = pd.read_csv(required[4])
    overlap_columns = [column for column in splits if "overlap" in column.lower()]
    overlap_zero = True
    for column in overlap_columns:
        values = pd.to_numeric(splits[column], errors="coerce").dropna()
        overlap_zero &= bool(values.eq(0).all())
    audit.add(
        "context_25_scaffold_disjoint_splits",
        len(splits) == 25 and bool(overlap_columns) and overlap_zero,
        f"splits={len(splits)}; overlap_columns={overlap_columns}",
    )

    internal_boot = pd.read_csv(required[3])
    internal_counts = pd.to_numeric(
        internal_boot["n_bootstrap"], errors="coerce"
    )
    ood_boot = pd.read_csv(required[7])
    ood_counts = pd.to_numeric(ood_boot["n_bootstrap"], errors="coerce")
    audit.add(
        "context_bootstrap_counts",
        not internal_counts.empty
        and internal_counts.notna().all()
        and internal_counts.eq(10_000).all()
        and not ood_counts.empty
        and ood_counts.notna().all()
        and ood_counts.eq(10_000).all(),
        f"internal={sorted(internal_counts.unique())}; ood={sorted(ood_counts.unique())}",
    )

    domain = pd.read_csv(required[5])
    aggregate = pd.read_csv(required[6])
    macro_ok = True
    max_macro_difference = 0.0
    for (protocol, model_id), group in domain.groupby(
        ["protocol", "model_id"], sort=False
    ):
        stored_rows = aggregate[
            (aggregate["protocol"] == protocol)
            & (aggregate["model_id"] == model_id)
        ]
        if len(stored_rows) != 1:
            macro_ok = False
            continue
        stored = stored_rows.iloc[0]
        for metric in ("spearman", "rmse", "mae"):
            observed = pd.to_numeric(group[metric], errors="coerce").dropna().mean()
            expected = stored[f"domain_macro_{metric}"]
            difference = abs(float(observed) - float(expected))
            max_macro_difference = max(max_macro_difference, difference)
            macro_ok &= close(observed, expected)
    audit.add(
        "portable_ood_macro_independent_recompute",
        macro_ok,
        f"max_abs_difference={max_macro_difference:.3g}",
    )


def audit_scaffold_source(audit: Audit, directory: Path) -> None:
    required = [
        directory / "scientific_configuration.json",
        directory / "generic_scaffold_repeat_averaged_predictions.csv",
        directory / "generic_scaffold_repeat_averaged_metrics.csv",
        directory / "generic_scaffold_outer_split_audit.csv",
        directory / "generic_scaffold_paired_cluster_bootstrap.csv",
        directory / "source_deletion_domain_metrics.csv",
        directory / "source_deletion_equal_domain_macro.csv",
        directory / "source_deletion_paired_cluster_bootstrap.csv",
        directory / "source_deletion_split_audit.csv",
        directory / "source_deletion_predictions.csv",
        directory / "native_threadpool_audit.json",
        directory / "all_target_exact_identity_audit.json",
        directory / "all_target_exact_identity_audit.md",
    ]
    audit.require_files("scaffold_source_required_artifacts", required)
    if not all(path.is_file() for path in required):
        return
    config = json.loads(required[0].read_text(encoding="utf-8"))
    args = config.get("args", {})
    formal = (
        bool(config.get("formal_settings_match"))
        and bool(config.get("identity_settings_match"))
        and int(args.get("outer_folds", 0)) == 5
        and int(args.get("outer_repeats", 0)) == 5
        and int(args.get("n_estimators", 0)) == 600
        and int(args.get("n_jobs", 0)) == 4
        and int(args.get("bootstrap_replicates", 0)) == 10_000
    )
    audit.add("scaffold_source_formal_configuration", formal, f"args={args}")
    try:
        execution = validate_scaffold_v3_execution(directory)
        audit.add(
            "scaffold_source_v3_native_thread_and_identity_gate",
            True,
            (
                f"thread={execution['native_threadpool_audit']['status']}; "
                f"identity={execution['all_target_exact_identity_audit']['status']}"
            ),
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        audit.add(
            "scaffold_source_v3_native_thread_and_identity_gate",
            False,
            str(error),
        )

    predictions = pd.read_csv(required[1])
    summary = pd.read_csv(required[2])
    metric_ok = True
    max_difference = 0.0
    for model_id, group in predictions.groupby("model_id", sort=False):
        observed = independent_metrics(group)
        stored_rows = summary[summary["model_id"] == model_id]
        if len(stored_rows) != 1:
            metric_ok = False
            continue
        stored = stored_rows.iloc[0]
        for metric in ("spearman", "rmse", "mae"):
            difference = abs(observed[metric] - float(stored[metric]))
            max_difference = max(max_difference, difference)
            metric_ok &= close(observed[metric], stored[metric])
    audit.add(
        "generic_scaffold_metrics_independent_recompute",
        metric_ok,
        f"max_abs_difference={max_difference:.3g}",
    )

    generic_splits = pd.read_csv(required[3])
    overlap_columns = [
        column for column in generic_splits if "overlap" in column.lower()
    ]
    overlap_zero = bool(overlap_columns)
    for column in overlap_columns:
        values = pd.to_numeric(generic_splits[column], errors="coerce").dropna()
        overlap_zero &= bool(values.eq(0).all())
    audit.add(
        "generic_25_group_disjoint_splits",
        len(generic_splits) == 25 and overlap_zero,
        f"splits={len(generic_splits)}; overlap_columns={overlap_columns}",
    )

    domain = pd.read_csv(required[5])
    macro = pd.read_csv(required[6])
    macro_ok = True
    max_difference = 0.0
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
    for (deletion, model_id), group in domain.groupby(
        ["deletion_source", "model_id"], sort=False
    ):
        stored_rows = macro[
            (macro["record_type"] == "model")
            & (macro["deletion_source"] == deletion)
            & (macro["model_id"] == model_id)
        ]
        if len(stored_rows) != 1:
            macro_ok = False
            continue
        stored = stored_rows.iloc[0]
        for metric in ("spearman", "rmse", "mae"):
            values = pd.to_numeric(
                group[metric], errors="coerce"
            ).to_numpy(dtype=float)
            if (
                len(values) != 8
                or set(group["heldout_target"]) != expected_targets
                or not np.isfinite(values).all()
            ):
                macro_ok = False
                continue
            observed = float(np.mean(values))
            expected = stored[f"domain_macro_{metric}"]
            difference = abs(observed - float(expected))
            max_difference = max(max_difference, difference)
            macro_ok &= close(observed, expected)
    audit.add(
        "source_deletion_macro_independent_recompute",
        macro_ok,
        f"max_abs_difference={max_difference:.3g}",
    )
    split_audit = pd.read_csv(required[8])
    deletion_completeness = (
        set(domain["deletion_source"]) == {"none", "MGTbind", "MGDB", "MolGlueDB", "TPDdb"}
        and domain["heldout_target"].nunique() == 8
        and set(domain["model_id"]) == set(MODELS)
        and len(domain) == 80
        and domain.groupby(
            ["heldout_target", "deletion_source"]
        ).size().eq(2).all()
        and len(split_audit) == 40
        and split_audit.groupby(
            ["heldout_target", "deletion_source"]
        ).size().eq(1).all()
        and split_audit["status"].astype(str).str.lower().eq("complete").all()
    )
    audit.add(
        "source_deletion_condition_completeness",
        deletion_completeness,
        (
            f"metric_rows={len(domain)}; split_rows={len(split_audit)}; "
            f"targets={domain['heldout_target'].nunique()}"
        ),
    )
    try:
        validate_source_deletion_baseline_identity(directory)
        audit.add(
            "source_deletion_all_8_confirmatory_prediction_identity",
            True,
            "all 8 targets, both models, exact row/parameter/y_true/y_pred identity",
        )
    except (OSError, ValueError, RuntimeError, AssertionError) as error:
        audit.add(
            "source_deletion_all_8_confirmatory_prediction_identity",
            False,
            str(error),
        )

    generic_boot = pd.read_csv(required[4])
    source_boot = pd.read_csv(required[7])
    counts_a = pd.to_numeric(generic_boot["n_bootstrap"], errors="coerce")
    counts_b = pd.to_numeric(source_boot["n_bootstrap"], errors="coerce")
    audit.add(
        "scaffold_source_bootstrap_counts",
        not counts_a.empty
        and counts_a.notna().all()
        and counts_a.eq(10_000).all()
        and not counts_b.empty
        and counts_b.notna().all()
        and counts_b.eq(10_000).all(),
        f"generic={sorted(counts_a.unique())}; source={sorted(counts_b.unique())}",
    )


def audit_null_applicability_censoring(audit: Audit, directory: Path) -> None:
    required = [
        directory / "run_manifest.json",
        directory / "permutation_model_metrics.csv",
        directory / "permutation_contrast_metrics.csv",
        directory / "permutation_null_inference.csv",
        directory / "applicability_domain_metrics.csv",
        directory / "applicability_model_metrics.csv",
        directory / "applicability_paired_deltas.csv",
        directory / "censoring_relation_overall.csv",
    ]
    audit.require_files("null_applicability_required_artifacts", required)
    if not all(path.is_file() for path in required):
        return
    manifest = json.loads(required[0].read_text(encoding="utf-8"))
    config = manifest.get("config", {})
    formal = (
        int(config.get("n_permutations", 0)) == 100
        and int(config.get("n_estimators", 0)) == 600
        and int(config.get("bootstrap_replicates", 0)) == 10_000
        and int(config.get("max_repeats", 0)) == 5
    )
    audit.add("null_formal_configuration", formal, f"config={config}")
    software = manifest.get("software", {})
    locked_environment = (
        str(software.get("python", "")).startswith("3.10.19")
        and str(software.get("scikit_learn", "")) == "1.7.2"
        and str(software.get("rdkit", "")) == "2025.09.4"
    )
    audit.add(
        "null_locked_environment_identity",
        locked_environment,
        (
            f"python={software.get('python')}; "
            f"scikit_learn={software.get('scikit_learn')}; "
            f"rdkit={software.get('rdkit')}"
        ),
    )

    models = pd.read_csv(required[1])
    contrasts = pd.read_csv(required[2])
    inference = pd.read_csv(required[3])
    ids_models = set(models["permutation_id"].astype(int))
    ids_contrasts = set(contrasts["permutation_id"].astype(int))
    expected_ids = set(range(1, 101))
    audit.add(
        "permutation_id_completeness",
        ids_models == ids_contrasts == expected_ids,
        f"models={len(ids_models)}; contrasts={len(ids_contrasts)}",
    )
    table_shape_ok = (
        len(models) == 200
        and len(contrasts) == 100
        and models.groupby("permutation_id").size().eq(2).all()
        and contrasts.groupby("permutation_id").size().eq(1).all()
        and not models.duplicated(["permutation_id", "model_id"]).any()
        and not contrasts.duplicated(
            ["permutation_id", "contrast_id"]
        ).any()
        and len(inference) == 9
        and not inference.duplicated(
            ["record_type", "estimand_id", "metric"]
        ).any()
    )
    audit.add(
        "permutation_table_shape_and_uniqueness",
        table_shape_ok,
        (
            f"model_rows={len(models)}; contrast_rows={len(contrasts)}; "
            f"inference_rows={len(inference)}"
        ),
    )
    diagnostics_path = directory / "permutation_fold_diagnostics.csv"
    if diagnostics_path.is_file():
        diagnostics = pd.read_csv(diagnostics_path)
        diagnostic_ids = set(diagnostics["permutation_id"].astype(int))
        diagnostics_ok = (
            diagnostic_ids == expected_ids
            and len(diagnostics) == 2_500
            and diagnostics.groupby("permutation_id").size().eq(25).all()
            and diagnostics[
                "within_stratum_multiset_preserved"
            ].dtype == bool
            and diagnostics["within_stratum_multiset_preserved"].all()
        )
        audit.add(
            "permutation_fold_diagnostic_completeness",
            diagnostics_ok,
            f"ids={len(diagnostic_ids)}; rows={len(diagnostics)}",
        )
    else:
        audit.add(
            "permutation_fold_diagnostic_completeness",
            False,
            f"missing={diagnostics_path}",
        )

    merge_path = directory / "permutation_shard_merge_manifest.json"
    if merge_path.is_file():
        merge = json.loads(merge_path.read_text(encoding="utf-8"))
        snapshot_relative = str(merge.get("recoverable_snapshot", ""))
        snapshot_candidate = Path(snapshot_relative)
        snapshot = ROUTE_DIR / snapshot_candidate
        merge_ok = (
            not snapshot_candidate.is_absolute()
            and ".." not in snapshot_candidate.parts
            and
            merge.get("status") == "PASS"
            and merge.get("scientific_change") is False
            and int(merge.get("n_permutations", 0)) == 100
            and merge.get("protocol_version")
            == NULL_IDENTITY["protocol_version"]
            and merge.get("training_response_dtype") == "float32"
            and merge.get("float32_correction_sha256")
            == EXPECTED_HASHES["correction"]
            and merge.get("applicability_correction_sha256")
            == NULL_IDENTITY["fixed_domain_correction_sha256"]
            and merge.get("local_protocol_sha256")
            == EXPECTED_HASHES["null_protocol"]
            and merge.get("scientific_implementation_sha256")
            == EXPECTED_NULL_PARENT_SHA256
            and merge.get("execution_wrapper_sha256")
            == EXPECTED_PERMUTATION_WRAPPER_SHA256
            and sha256_file(
                SCRIPT_DIR / str(NULL_IDENTITY["merge_filename"])
            )
            == EXPECTED_PERMUTATION_MERGE_SHA256
            and snapshot.name == NULL_IDENTITY["snapshot_name"]
            and snapshot.parent.resolve() == directory.resolve()
            and snapshot.is_dir()
            and not snapshot.is_symlink()
        )
        audit.add(
            "permutation_shard_merge_provenance",
            merge_ok,
            (
                f"status={merge.get('status')}; "
                f"scientific_change={merge.get('scientific_change')}; "
                f"shards={len(merge.get('shards', []))}; "
                f"snapshot={snapshot_relative}"
            ),
        )
    else:
        audit.add(
            "permutation_shard_merge_provenance",
            False,
            f"missing={merge_path}",
        )
    p_ok = True
    max_difference = 0.0
    for _, row in inference.iterrows():
        if row["record_type"] == "model":
            subset = models[models["model_id"] == row["estimand_id"]]
        else:
            subset = contrasts[contrasts["contrast_id"] == row["estimand_id"]]
        values = pd.to_numeric(subset[row["metric"]], errors="coerce").dropna()
        observed = float(row["observed"])
        tail = str(row["empirical_tail"])
        if tail == "greater":
            extreme = int((values >= observed).sum())
        elif tail == "smaller":
            extreme = int((values <= observed).sum())
        else:
            p_ok = False
            continue
        recomputed = (1 + extreme) / (len(values) + 1)
        difference = abs(recomputed - float(row["empirical_p"]))
        max_difference = max(max_difference, difference)
        p_ok &= len(values) == 100 and close(recomputed, row["empirical_p"])
    audit.add(
        "permutation_empirical_p_independent_recompute",
        p_ok,
        f"max_abs_difference={max_difference:.3g}",
    )

    domains = pd.read_csv(required[4])
    aggregate = pd.read_csv(required[5])
    macro_ok = True
    max_difference = 0.0
    ood_domains = domains[domains["heldout_group"] != "pooled"]
    for keys, group in ood_domains.groupby(
        ["regime", "protocol", "similarity_bin", "model_id"], sort=False
    ):
        regime, protocol, similarity_bin, model_id = keys
        stored_rows = aggregate[
            (aggregate["regime"] == regime)
            & (aggregate["protocol"] == protocol)
            & (aggregate["similarity_bin"] == similarity_bin)
            & (aggregate["model_id"] == model_id)
        ]
        if len(stored_rows) != 1:
            macro_ok = False
            continue
        stored = stored_rows.iloc[0]
        for metric in ("spearman", "rmse", "mae"):
            observed = pd.to_numeric(group[metric], errors="coerce").dropna().mean()
            expected = stored[f"domain_macro_{metric}"]
            if pd.isna(observed) and pd.isna(expected):
                continue
            difference = abs(float(observed) - float(expected))
            max_difference = max(max_difference, difference)
            macro_ok &= close(observed, expected)
    audit.add(
        "applicability_domain_macro_independent_recompute",
        macro_ok,
        f"max_abs_difference={max_difference:.3g}",
    )

    paired = pd.read_csv(required[6])
    audit.add(
        "applicability_strata_and_bootstrap_completeness",
        len(aggregate) == 40
        and len(paired) == 20
        and pd.to_numeric(
            paired["n_bootstrap_requested"], errors="coerce"
        ).eq(10_000).all(),
        f"models={len(aggregate)}; contrasts={len(paired)}",
    )

    censor = pd.read_csv(required[7])
    audit.add(
        "censoring_total_and_classes",
        int(censor["n_records"].sum()) == 3_117
        and set(censor["relation_class"])
        == {"equality", "range", "one_sided", "missing_or_unparsed"},
        f"total={int(censor['n_records'].sum())}; classes={sorted(censor['relation_class'])}",
    )


def audit_bound_aggregate_source_data(
    audit: Audit,
    source_dir: Path,
) -> dict[str, str]:
    """Fail-closed aggregate audit that also returns immutable input hashes."""

    try:
        hashes = aggregate_bundle_hashes(source_dir)
    except (OSError, ValueError, TypeError, KeyError) as error:
        audit.add(
            "aggregate_source_exact_hash_and_lineage_binding",
            False,
            str(error),
        )
        return {}
    audit.add(
        "aggregate_source_exact_hash_and_lineage_binding",
        True,
        (
            f"analysis={hashes['analysis_identity']}; "
            f"index={hashes['source_data_extension_index_sha256']}; "
            f"checksums={hashes['source_data_extension_checksums_sha256']}; "
            f"provenance={hashes['source_data_extension_provenance_sha256']}"
        ),
    )
    return hashes


def audit_results_payload(
    audit: Audit,
    *,
    source_dir: Path,
    source_hashes: dict[str, str],
    results_path: Path,
) -> str:
    """Independently rebuild every reported result and bind its JSON bytes."""

    try:
        if not results_path.is_file() or results_path.is_symlink():
            raise ValueError("results JSON is missing or symlinked")
        payload = read_json_object(results_path)
        if not source_hashes:
            raise ValueError("aggregate source-data identity is unavailable")
        validate_results_payload_from_aggregates(
            payload,
            source_dir,
            source_hashes,
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        audit.add(
            "computational_extension_results_independent_recompute",
            False,
            str(error),
        )
        return ""
    digest = sha256_file(results_path)
    audit.add(
        "computational_extension_results_independent_recompute",
        True,
        f"sha256={digest}",
    )
    return digest


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    audit = Audit()
    audit_input_identity(audit)
    audit_formal_workflow_provenance(
        audit,
        args.context_dir,
        args.scaffold_dir,
        args.null_dir,
    )
    audit_upstream_qa(
        audit, args.context_dir, args.scaffold_dir, args.null_dir
    )
    audit_context(audit, args.context_dir)
    audit_scaffold_source(audit, args.scaffold_dir)
    audit_null_applicability_censoring(audit, args.null_dir)
    aggregate_hashes = audit_bound_aggregate_source_data(
        audit, args.source_dir
    )
    results_sha256 = audit_results_payload(
        audit,
        source_dir=args.source_dir,
        source_hashes=aggregate_hashes,
        results_path=args.results_json,
    )
    expected_check_ids = list(CROSSCHECK_EXPECTED_CHECK_IDS)
    observed_check_ids = [item["check_id"] for item in audit.checks]
    check_contract_complete = observed_check_ids == expected_check_ids
    n_pass = sum(item["status"] == "PASS" for item in audit.checks)
    software_bindings = downstream_software_bindings()
    overall_pass = audit.passed and check_contract_complete
    payload = {
        "status": "PASS" if overall_pass else "FAIL",
        "n_checks": len(audit.checks),
        "n_pass": n_pass,
        "n_fail": len(audit.checks) - n_pass,
        "analysis_identity": ANALYSIS_IDENTITY,
        "audit_identity": "independent_computational_extension_crosscheck_v1",
        "expected_check_ids": expected_check_ids,
        "expected_check_count": len(expected_check_ids),
        "provenance": {
            **aggregate_hashes,
            "software_bindings": software_bindings,
            "computational_extension_results_sha256": results_sha256,
            "crosscheck_generation_id": (
                crosscheck_generation_id(
                    aggregate_hashes,
                    results_sha256=results_sha256,
                    software_bindings=software_bindings,
                )
                if aggregate_hashes and results_sha256
                else ""
            ),
            "final_analysis_identity": ANALYSIS_IDENTITY,
            "response_dtype": "float32",
            "float32_correction_sha256": EXPECTED_HASHES["correction"],
            "workflow_directories": {
                "context_weight": route_relative(args.context_dir),
                "scaffold_source": route_relative(
                    args.scaffold_dir
                ),
                "null_applicability_censoring": route_relative(
                    args.null_dir
                ),
            },
            "superseded_workflow_results_used": False,
        },
        "checks": audit.checks,
    }
    atomic_text(
        args.output_json,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )
    lines = [
        "# Computational extension independent cross-check v1",
        "",
        f"Overall status: **{payload['status']}**",
        "",
        f"Checks: {n_pass}/{len(audit.checks)} passed.",
        "",
        "| Check | Status | Detail |",
        "|---|---:|---|",
    ]
    for item in audit.checks:
        detail = str(item["detail"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| `{item['check_id']}` | {item['status']} | {detail} |"
        )
    lines.extend(
        [
            "",
            "This verifier independently recomputes central metrics from stored "
            "predictions or domain-level records. It does not re-estimate "
            "bootstrap intervals.",
            "",
        ]
    )
    atomic_text(args.output_md, "\n".join(lines))
    print(
        f"COMPUTATIONAL_EXTENSION_CROSSCHECK: {payload['status']} "
        f"({n_pass}/{len(audit.checks)})"
    )
    if not overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
