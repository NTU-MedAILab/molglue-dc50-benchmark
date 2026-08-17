#!/usr/bin/env python
"""Exact all-target identity gate for scaffold/source sensitivity v3.

The gate compares the no-source-deletion rows from the v3 post-hoc package
with the frozen confirmatory target-OOD artifacts.  It deliberately checks
the full eight-target baseline rather than a single diagnostic target.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_confirmatory_cpu_v1 as core  # noqa: E402
import run_confirmatory_ood_cpu_v1 as ood  # noqa: E402


DEFAULT_CURRENT = (
    PROJECT_DIR / "reports" / "post_hoc_scaffold_source_sensitivity_v3"
)
DEFAULT_ARCHIVE = PROJECT_DIR / "reports" / "confirmatory_ood_cpu_v1"
OUTPUT_JSON = "all_target_exact_identity_audit.json"
OUTPUT_MARKDOWN = "all_target_exact_identity_audit.md"
EXPECTED_PROTOCOL_VERSION = "post_hoc_scaffold_source_sensitivity_v3.0"
EXPECTED_MODELS = (
    "chemistry_extra_trees",
    "full_context_extra_trees",
)
EXPECTED_TARGETS = tuple(ood.TARGET_OOD_GROUPS)
EXPECTED_TARGET_COUNT = 8
EXPECTED_BASELINE_PREDICTION_ROWS = 2_580
EXPECTED_SELECTED_ROWS = 16
EXPECTED_TUNING_ROWS = 96
EXPECTED_OUTER_SPLIT_ROWS = 8
EXPECTED_INNER_SPLIT_ROWS = 32
EXPECTED_OMP_THREADS = 24
EXPECTED_OPENBLAS_THREADS = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-dir", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(values: Iterable[Any]) -> str:
    payload = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(type(value).__name__)


def nested_key_values(value: Any, requested_key: str) -> list[Any]:
    """Collect values for an exact key from an arbitrarily nested JSON value."""
    found: list[Any] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == requested_key:
                found.append(child)
            found.extend(nested_key_values(child, requested_key))
    elif isinstance(value, list):
        for child in value:
            found.extend(nested_key_values(child, requested_key))
    return found


def threadpool_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if (
            "internal_api" in value
            and "num_threads" in value
            and isinstance(value.get("internal_api"), str)
        ):
            records.append(value)
        for child in value.values():
            records.extend(threadpool_records(child))
    elif isinstance(value, list):
        for child in value:
            records.extend(threadpool_records(child))
    return records


def equal_values(left: pd.Series, right: pd.Series) -> tuple[bool, int, float | None]:
    """Return exact equality, mismatch count, and numeric maximum difference."""
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        left_values = left.to_numpy()
        right_values = right.to_numpy()
        exact_mask = (left_values == right_values) | (
            pd.isna(left_values) & pd.isna(right_values)
        )
        mismatch = int(np.sum(~exact_mask))
        maximum: float | None = None
        if len(left_values):
            left_float = left.to_numpy(dtype=np.float64)
            right_float = right.to_numpy(dtype=np.float64)
            finite_pair = np.isfinite(left_float) & np.isfinite(right_float)
            if finite_pair.any():
                maximum = float(
                    np.max(np.abs(left_float[finite_pair] - right_float[finite_pair]))
                )
        return bool(mismatch == 0), mismatch, maximum
    left_values = left.fillna("<NA>").astype(str).to_numpy()
    right_values = right.fillna("<NA>").astype(str).to_numpy()
    mismatch = int(np.sum(left_values != right_values))
    return bool(mismatch == 0), mismatch, None


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(
        self,
        check_id: str,
        passed: bool,
        observed: Any,
        expected: Any,
    ) -> None:
        self.checks.append(
            {
                "check_id": check_id,
                "status": "PASS" if bool(passed) else "FAIL",
                "observed": observed,
                "expected": expected,
            }
        )

    def required_columns(
        self,
        check_prefix: str,
        frame: pd.DataFrame,
        columns: Iterable[str],
    ) -> bool:
        required = list(columns)
        missing = sorted(set(required) - set(frame.columns))
        self.check(
            f"{check_prefix}_required_columns",
            not missing,
            {"missing": missing, "available": sorted(frame.columns)},
            {"missing": []},
        )
        return not missing

    def compare_columns(
        self,
        check_prefix: str,
        current: pd.DataFrame,
        archive: pd.DataFrame,
        mappings: Iterable[tuple[str, str]],
    ) -> None:
        for current_column, archive_column in mappings:
            exact, mismatch, maximum = equal_values(
                current[current_column],
                archive[archive_column],
            )
            observed: dict[str, Any] = {"n_mismatch": mismatch}
            if maximum is not None:
                observed["maximum_absolute_difference"] = maximum
            self.check(
                f"{check_prefix}_{current_column}",
                exact,
                observed,
                {"n_mismatch": 0, "maximum_absolute_difference": 0.0}
                if maximum is not None
                else {"n_mismatch": 0},
            )


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def load_v3_module() -> Any:
    return importlib.import_module("run_post_hoc_scaffold_source_sensitivity_v3")


def audit_protocol_and_threads(
    audit: Audit,
    current_dir: Path,
    archive_dir: Path,
    v3: Any,
) -> dict[str, Any]:
    configuration = read_json(current_dir / "scientific_configuration.json")
    manifest = read_json(current_dir / "run_manifest.json")
    runner_qa = read_json(current_dir / "qa_summary.json")
    archive_manifest = read_json(archive_dir / "ood_run_manifest.json")
    archive_qa_text = (archive_dir / "qa_summary.md").read_text(
        encoding="utf-8"
    )
    threadpool_audit = read_json(current_dir / "native_threadpool_audit.json")

    audit.check(
        "target_contract_has_exactly_eight_targets",
        len(EXPECTED_TARGETS) == EXPECTED_TARGET_COUNT
        and len(set(EXPECTED_TARGETS)) == EXPECTED_TARGET_COUNT,
        list(EXPECTED_TARGETS),
        f"{EXPECTED_TARGET_COUNT} unique frozen target-OOD groups",
    )
    audit.check(
        "v3_module_protocol_identity",
        getattr(v3, "PROTOCOL_VERSION", None) == EXPECTED_PROTOCOL_VERSION,
        getattr(v3, "PROTOCOL_VERSION", None),
        EXPECTED_PROTOCOL_VERSION,
    )
    for label, payload in (
        ("configuration", configuration),
        ("manifest", manifest),
    ):
        audit.check(
            f"{label}_protocol_identity",
            payload.get("protocol_version") == EXPECTED_PROTOCOL_VERSION,
            payload.get("protocol_version"),
            EXPECTED_PROTOCOL_VERSION,
        )
        audit.check(
            f"{label}_response_dtype_float32",
            payload.get("response_dtype") == "float32",
            payload.get("response_dtype"),
            "float32",
        )
        audit.check(
            f"{label}_frozen_data_identity",
            payload.get("data_sha256") == core.FROZEN_DATA_SHA256,
            payload.get("data_sha256"),
            core.FROZEN_DATA_SHA256,
        )
    audit.check(
        "v3_runner_qa_pass",
        runner_qa.get("status") == "PASS",
        runner_qa.get("status"),
        "PASS",
    )
    audit.check(
        "archive_protocol_identity",
        archive_manifest.get("protocol_version") == "confirmatory_ood_cpu_v1.0",
        archive_manifest.get("protocol_version"),
        "confirmatory_ood_cpu_v1.0",
    )
    audit.check(
        "archive_status_complete",
        archive_manifest.get("status") == "complete",
        archive_manifest.get("status"),
        "complete",
    )
    audit.check(
        "archive_qa_pass",
        "Overall result: **PASS**" in archive_qa_text,
        "PASS marker present"
        if "Overall result: **PASS**" in archive_qa_text
        else "PASS marker absent",
        "PASS marker present",
    )

    run_args = configuration.get("args", {})
    shared_expected_args = {
        "outer_folds": 5,
        "outer_repeats": 5,
        "inner_folds": 4,
        "n_estimators": 600,
        "n_jobs": 4,
        "seed": 260_531,
        "max_outer_splits": None,
        "max_targets": None,
        "max_deletions": None,
        "only_target": None,
    }
    formal_profile = {
        **shared_expected_args,
        "mode": "both",
        "bootstrap_replicates": 10_000,
        "baseline_only": False,
    }
    preflight_profile = {
        **shared_expected_args,
        "mode": "source-deletion",
        "bootstrap_replicates": 0,
        "baseline_only": True,
    }

    def profile_matches(profile: dict[str, Any]) -> bool:
        return all(run_args.get(key) == expected for key, expected in profile.items())

    if profile_matches(formal_profile):
        execution_profile = "formal_complete_package"
    elif profile_matches(preflight_profile):
        execution_profile = "all_target_baseline_identity_preflight"
    else:
        execution_profile = "ineligible"
    audit.check(
        "eligible_execution_profile",
        execution_profile != "ineligible",
        {
            "profile": execution_profile,
            "args": {
                key: run_args.get(key)
                for key in sorted(set(formal_profile) | set(preflight_profile))
            },
        },
        {
            "eligible_profiles": [
                "formal_complete_package",
                "all_target_baseline_identity_preflight",
            ]
        },
    )

    configured_models = tuple(configuration.get("models", []))
    configured_targets = tuple(configuration.get("target_ood_groups", []))
    audit.check(
        "configuration_model_identity",
        configured_models == EXPECTED_MODELS,
        list(configured_models),
        list(EXPECTED_MODELS),
    )
    audit.check(
        "configuration_target_identity",
        configured_targets == EXPECTED_TARGETS,
        list(configured_targets),
        list(EXPECTED_TARGETS),
    )

    protocol_document = Path(v3.PROTOCOL_DOCUMENT)
    runner_path = Path(v3.__file__).resolve()
    audit.check(
        "protocol_document_exists",
        protocol_document.is_file(),
        str(protocol_document),
        "existing file",
    )
    if protocol_document.is_file():
        actual_protocol_sha = sha256_file(protocol_document)
        audit.check(
            "protocol_document_sha256",
            configuration.get("protocol_sha256") == actual_protocol_sha,
            configuration.get("protocol_sha256"),
            actual_protocol_sha,
        )
    actual_runner_sha = sha256_file(runner_path)
    audit.check(
        "runner_sha256",
        configuration.get("this_script_sha256") == actual_runner_sha,
        configuration.get("this_script_sha256"),
        actual_runner_sha,
    )
    expected_archive_hashes = getattr(v3, "EXPECTED_CONFIRMATORY_HASHES", {})
    audit.check(
        "frozen_archive_hash_contract_present",
        isinstance(expected_archive_hashes, dict)
        and set(expected_archive_hashes)
        == {
            "ood_predictions.csv",
            "ood_domain_metrics.csv",
            "ood_selected_hyperparameters.csv",
            "ood_inner_tuning_metrics.csv",
            "ood_split_audit.csv",
            "ood_inner_split_audit.csv",
        },
        sorted(expected_archive_hashes)
        if isinstance(expected_archive_hashes, dict)
        else type(expected_archive_hashes).__name__,
        "the six frozen confirmatory identity artifacts",
    )
    if isinstance(expected_archive_hashes, dict):
        for filename, expected_sha in sorted(expected_archive_hashes.items()):
            path = archive_dir / filename
            observed_sha = sha256_file(path) if path.is_file() else None
            audit.check(
                f"frozen_archive_sha256::{filename}",
                observed_sha == expected_sha,
                observed_sha,
                expected_sha,
            )

    contract = configuration.get("native_thread_contract")
    audit.check(
        "native_thread_contract_present",
        isinstance(contract, dict),
        type(contract).__name__,
        "dict",
    )
    if not isinstance(contract, dict):
        contract = {}
    audit.check(
        "manifest_native_thread_contract_exact",
        manifest.get("native_thread_contract") == contract,
        manifest.get("native_thread_contract"),
        contract,
    )
    for env_name, expected in (
        ("OMP_NUM_THREADS", EXPECTED_OMP_THREADS),
        ("OPENBLAS_NUM_THREADS", EXPECTED_OPENBLAS_THREADS),
    ):
        values = nested_key_values(contract, env_name)
        normalized = [str(value) for value in values]
        audit.check(
            f"native_thread_contract_{env_name}",
            bool(values) and all(value == str(expected) for value in normalized),
            normalized,
            [str(expected)],
        )

    records = threadpool_records(threadpool_audit)
    snapshots = threadpool_audit.get("snapshots", [])
    if not isinstance(snapshots, list):
        snapshots = []
    required_stages = {"startup", "before_source", "after_source"}
    if execution_profile == "formal_complete_package":
        required_stages.update({"before_generic", "after_generic"})
    observed_stages = {
        str(snapshot.get("stage"))
        for snapshot in snapshots
        if isinstance(snapshot, dict)
    }
    audit.check(
        "native_threadpool_audit_status",
        threadpool_audit.get("status") == "PASS",
        threadpool_audit.get("status"),
        "PASS",
    )
    audit.check(
        "native_threadpool_required_stage_completeness",
        required_stages.issubset(observed_stages),
        sorted(observed_stages),
        sorted(required_stages),
    )
    affinity_counts = [
        snapshot.get("cpu_affinity_count")
        for snapshot in snapshots
        if isinstance(snapshot, dict)
    ]
    audit.check(
        "native_threadpool_cpu_affinity",
        bool(affinity_counts)
        and all(
            isinstance(value, (int, np.integer)) and int(value) >= 24
            for value in affinity_counts
        ),
        affinity_counts,
        "one value per snapshot, all >=24",
    )
    relevant_pool_records = [
        record
        for record in records
        if str(record.get("user_api", "")).lower() in {"openmp", "blas"}
    ]
    audit.check(
        "all_active_openmp_and_blas_pools_locked",
        bool(relevant_pool_records)
        and all(
            isinstance(record.get("num_threads"), (int, np.integer))
            and int(record["num_threads"]) == EXPECTED_OMP_THREADS
            for record in relevant_pool_records
        ),
        [
            {
                "user_api": record.get("user_api"),
                "internal_api": record.get("internal_api"),
                "num_threads": record.get("num_threads"),
            }
            for record in relevant_pool_records
        ],
        "one or more OpenMP/BLAS pool records, all num_threads=24",
    )
    openmp_records = [
        record
        for record in records
        if str(record.get("internal_api", "")).lower() == "openmp"
    ]
    openblas_records = [
        record
        for record in records
        if str(record.get("internal_api", "")).lower() == "openblas"
    ]
    for api_name, records_for_api, expected in (
        ("openmp", openmp_records, EXPECTED_OMP_THREADS),
        ("openblas", openblas_records, EXPECTED_OPENBLAS_THREADS),
    ):
        observed = [record.get("num_threads") for record in records_for_api]
        audit.check(
            f"native_threadpool_runtime_{api_name}",
            bool(observed)
            and all(
                isinstance(value, (int, np.integer)) and int(value) == expected
                for value in observed
            ),
            observed,
            f"one or more snapshots, all num_threads={expected}",
        )

    audit.check(
        "native_threadpool_audit_protocol_identity",
        threadpool_audit.get("protocol_version") == EXPECTED_PROTOCOL_VERSION,
        threadpool_audit.get("protocol_version"),
        EXPECTED_PROTOCOL_VERSION,
    )

    return {
        "configuration": configuration,
        "manifest": manifest,
        "threadpool_audit": threadpool_audit,
        "execution_profile": execution_profile,
    }


def filter_and_sort_artifacts(
    audit: Audit,
    current_dir: Path,
    archive_dir: Path,
    v3: Any,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    source_files = v3.SOURCE_FILES
    current_predictions_all = pd.read_csv(
        current_dir / source_files["predictions"]
    )
    archive_predictions_all = pd.read_csv(archive_dir / "ood_predictions.csv")
    current_predictions_baseline = current_predictions_all[
        current_predictions_all["deletion_source"] == v3.BASELINE_CONDITION
    ]
    audit.check(
        "prediction_baseline_target_set_exact",
        set(current_predictions_baseline["heldout_target"].astype(str))
        == set(EXPECTED_TARGETS),
        sorted(set(current_predictions_baseline["heldout_target"].astype(str))),
        sorted(EXPECTED_TARGETS),
    )
    audit.check(
        "prediction_baseline_model_set_exact",
        set(current_predictions_baseline["model_id"].astype(str))
        == set(EXPECTED_MODELS),
        sorted(set(current_predictions_baseline["model_id"].astype(str))),
        sorted(EXPECTED_MODELS),
    )
    current_predictions = (
        current_predictions_baseline[
            current_predictions_baseline["model_id"].isin(EXPECTED_MODELS)
            & current_predictions_baseline["heldout_target"].isin(
                EXPECTED_TARGETS
            )
        ]
        .sort_values(["heldout_target", "model_id", "row_index"])
        .reset_index(drop=True)
    )
    archive_predictions = (
        archive_predictions_all[
            (archive_predictions_all["protocol"] == "target_ood")
            & archive_predictions_all["model_id"].isin(EXPECTED_MODELS)
            & archive_predictions_all["heldout_group"].isin(EXPECTED_TARGETS)
        ]
        .sort_values(["heldout_group", "model_id", "row_index"])
        .reset_index(drop=True)
    )

    current_selected_all = pd.read_csv(current_dir / source_files["selected"])
    archive_selected_all = pd.read_csv(
        archive_dir / "ood_selected_hyperparameters.csv"
    )
    current_selected = (
        current_selected_all[
            (current_selected_all["deletion_source"] == v3.BASELINE_CONDITION)
            & current_selected_all["model_id"].isin(EXPECTED_MODELS)
            & current_selected_all["heldout_target"].isin(EXPECTED_TARGETS)
        ]
        .sort_values(["heldout_target", "model_id"])
        .reset_index(drop=True)
    )
    archive_selected = (
        archive_selected_all[
            (archive_selected_all["protocol"] == "target_ood")
            & archive_selected_all["model_id"].isin(EXPECTED_MODELS)
            & archive_selected_all["heldout_group"].isin(EXPECTED_TARGETS)
        ]
        .sort_values(["heldout_group", "model_id"])
        .reset_index(drop=True)
    )

    current_tuning_all = pd.read_csv(current_dir / source_files["tuning"])
    archive_tuning_all = pd.read_csv(
        archive_dir / "ood_inner_tuning_metrics.csv"
    )
    current_tuning = (
        current_tuning_all[
            (current_tuning_all["deletion_source"] == v3.BASELINE_CONDITION)
            & current_tuning_all["model_id"].isin(EXPECTED_MODELS)
            & current_tuning_all["heldout_target"].isin(EXPECTED_TARGETS)
        ]
        .sort_values(["heldout_target", "model_id", "param_id"])
        .reset_index(drop=True)
    )
    archive_tuning = (
        archive_tuning_all[
            (archive_tuning_all["protocol"] == "target_ood")
            & archive_tuning_all["model_id"].isin(EXPECTED_MODELS)
            & archive_tuning_all["heldout_group"].isin(EXPECTED_TARGETS)
        ]
        .sort_values(["heldout_group", "model_id", "param_id"])
        .reset_index(drop=True)
    )

    current_domain_all = pd.read_csv(
        current_dir / source_files["domain_metrics"]
    )
    archive_domain_all = pd.read_csv(archive_dir / "ood_domain_metrics.csv")
    current_domain = (
        current_domain_all[
            (current_domain_all["deletion_source"] == v3.BASELINE_CONDITION)
            & current_domain_all["model_id"].isin(EXPECTED_MODELS)
            & current_domain_all["heldout_target"].isin(EXPECTED_TARGETS)
        ]
        .sort_values(["heldout_target", "model_id"])
        .reset_index(drop=True)
    )
    archive_domain = (
        archive_domain_all[
            (archive_domain_all["protocol"] == "target_ood")
            & archive_domain_all["model_id"].isin(EXPECTED_MODELS)
            & archive_domain_all["heldout_group"].isin(EXPECTED_TARGETS)
        ]
        .sort_values(["heldout_group", "model_id"])
        .reset_index(drop=True)
    )

    current_outer_all = pd.read_csv(
        current_dir / source_files["split_audit"]
    )
    archive_outer_all = pd.read_csv(archive_dir / "ood_split_audit.csv")
    current_outer = (
        current_outer_all[
            (current_outer_all["deletion_source"] == v3.BASELINE_CONDITION)
            & current_outer_all["heldout_target"].isin(EXPECTED_TARGETS)
        ]
        .sort_values("heldout_target")
        .reset_index(drop=True)
    )
    archive_outer = (
        archive_outer_all[
            (archive_outer_all["protocol"] == "target_ood")
            & archive_outer_all["heldout_group"].isin(EXPECTED_TARGETS)
        ]
        .sort_values("heldout_group")
        .reset_index(drop=True)
    )

    current_inner_all = pd.read_csv(
        current_dir / source_files["inner_audit"]
    )
    archive_inner_all = pd.read_csv(
        archive_dir / "ood_inner_split_audit.csv"
    )
    current_inner = (
        current_inner_all[
            (current_inner_all["deletion_source"] == v3.BASELINE_CONDITION)
            & current_inner_all["heldout_target"].isin(EXPECTED_TARGETS)
        ]
        .sort_values(["heldout_target", "inner_fold"])
        .reset_index(drop=True)
    )
    archive_inner = (
        archive_inner_all[
            (archive_inner_all["protocol"] == "target_ood")
            & archive_inner_all["heldout_group"].isin(EXPECTED_TARGETS)
        ]
        .sort_values(["heldout_group", "inner_fold"])
        .reset_index(drop=True)
    )

    pairs = {
        "predictions": (current_predictions, archive_predictions),
        "selected": (current_selected, archive_selected),
        "tuning": (current_tuning, archive_tuning),
        "domain": (current_domain, archive_domain),
        "outer": (current_outer, archive_outer),
        "inner": (current_inner, archive_inner),
    }
    expected_rows = {
        "predictions": EXPECTED_BASELINE_PREDICTION_ROWS,
        "selected": EXPECTED_SELECTED_ROWS,
        "tuning": EXPECTED_TUNING_ROWS,
        "domain": EXPECTED_SELECTED_ROWS,
        "outer": EXPECTED_OUTER_SPLIT_ROWS,
        "inner": EXPECTED_INNER_SPLIT_ROWS,
    }
    for name, (current, archive) in pairs.items():
        expected = expected_rows[name]
        audit.check(
            f"{name}_row_count",
            len(current) == len(archive) == expected,
            {"current": len(current), "archive": len(archive)},
            {"current": expected, "archive": expected},
        )
    return pairs


def audit_prediction_selected_tuning_and_features(
    audit: Audit,
    pairs: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> dict[str, dict[str, Any]]:
    current_predictions, archive_predictions = pairs["predictions"]
    prediction_mappings = (
        ("heldout_target", "heldout_group"),
        ("row_index", "row_index"),
        ("qc_id", "qc_id"),
        ("model_id", "model_id"),
        ("param_id", "param_id"),
        ("y_true", "y_true"),
        ("y_pred", "y_pred"),
        ("canonical_smiles", "canonical_smiles"),
        ("bemis_murcko_scaffold_id", "scaffold_id"),
        ("source_database", "source_database"),
        ("target_protein", "target_protein"),
    )
    if audit.required_columns(
        "predictions_current",
        current_predictions,
        [item[0] for item in prediction_mappings],
    ) and audit.required_columns(
        "predictions_archive",
        archive_predictions,
        [item[1] for item in prediction_mappings],
    ):
        audit.compare_columns(
            "prediction_column_exact",
            current_predictions,
            archive_predictions,
            prediction_mappings,
        )

    model_differences: dict[str, dict[str, Any]] = {}
    for target in EXPECTED_TARGETS:
        for model_id in EXPECTED_MODELS:
            key = f"{target}::{model_id}"
            current_values = current_predictions[
                (current_predictions["heldout_target"] == target)
                & (current_predictions["model_id"] == model_id)
            ]["y_pred"].to_numpy(dtype=np.float64)
            archive_values = archive_predictions[
                (archive_predictions["heldout_group"] == target)
                & (archive_predictions["model_id"] == model_id)
            ]["y_pred"].to_numpy(dtype=np.float64)
            same_length = len(current_values) == len(archive_values)
            exact = bool(
                same_length
                and np.array_equal(current_values, archive_values, equal_nan=True)
            )
            absolute = (
                np.abs(current_values - archive_values)
                if same_length
                else np.asarray([], dtype=np.float64)
            )
            maximum = float(np.max(absolute)) if len(absolute) else None
            mean = float(np.mean(absolute)) if len(absolute) else None
            model_differences[key] = {
                "n_current": int(len(current_values)),
                "n_archive": int(len(archive_values)),
                "array_equal": exact,
                "mean_absolute_difference": mean,
                "maximum_absolute_difference": maximum,
            }
            audit.check(
                f"prediction_vector_exact::{key}",
                exact,
                {
                    "n_current": len(current_values),
                    "n_archive": len(archive_values),
                    "maximum_absolute_difference": maximum,
                },
                {
                    "equal_lengths": True,
                    "maximum_absolute_difference": 0.0,
                },
            )

    current_selected, archive_selected = pairs["selected"]
    selected_mappings = (
        ("heldout_target", "heldout_group"),
        ("model_id", "model_id"),
        ("param_id", "param_id"),
        ("min_samples_leaf", "min_samples_leaf"),
        ("max_features", "max_features"),
    )
    if audit.required_columns(
        "selected_current",
        current_selected,
        [item[0] for item in selected_mappings],
    ) and audit.required_columns(
        "selected_archive",
        archive_selected,
        [item[1] for item in selected_mappings],
    ):
        audit.compare_columns(
            "selected_column_exact",
            current_selected,
            archive_selected,
            selected_mappings,
        )

    current_tuning, archive_tuning = pairs["tuning"]
    tuning_mappings = (
        ("heldout_target", "heldout_group"),
        ("model_id", "model_id"),
        ("param_id", "param_id"),
        ("inner_n", "inner_n"),
        ("spearman", "spearman"),
        ("pearson", "pearson"),
        ("rmse", "rmse"),
        ("mae", "mae"),
        ("r2", "r2"),
        ("calibration_intercept", "calibration_intercept"),
        ("calibration_slope", "calibration_slope"),
    )
    if audit.required_columns(
        "tuning_current",
        current_tuning,
        [item[0] for item in tuning_mappings],
    ) and audit.required_columns(
        "tuning_archive",
        archive_tuning,
        [item[1] for item in tuning_mappings],
    ):
        audit.compare_columns(
            "tuning_column_exact",
            current_tuning,
            archive_tuning,
            tuning_mappings,
        )

    current_domain, archive_domain = pairs["domain"]
    domain_mappings = (
        ("heldout_target", "heldout_group"),
        ("model_id", "model_id"),
        ("param_id", "param_id"),
        ("n_train_rows", "n_train_rows"),
        ("n_train_unique_smiles", "n_train_unique_smiles"),
        ("n_train_scaffolds", "n_train_scaffolds"),
        ("n_test_rows", "n_test_rows"),
        ("n_test_unique_smiles", "n_test_unique_smiles"),
        ("n_test_scaffolds", "n_test_scaffolds"),
        ("spearman", "spearman"),
        ("pearson", "pearson"),
        ("rmse", "rmse"),
        ("mae", "mae"),
        ("r2", "r2"),
        ("calibration_intercept", "calibration_intercept"),
        ("calibration_slope", "calibration_slope"),
        ("rdkit_kept", "rdkit_kept"),
        ("rdkit_total", "rdkit_total"),
        ("rdkit_train_missing", "rdkit_train_missing"),
        ("rdkit_eval_missing", "rdkit_eval_missing"),
        ("context_dim", "context_dim"),
        ("chemistry_dim", "chemistry_dim"),
        ("full_dim", "full_dim"),
    )
    if audit.required_columns(
        "domain_current",
        current_domain,
        [item[0] for item in domain_mappings],
    ) and audit.required_columns(
        "domain_archive",
        archive_domain,
        [item[1] for item in domain_mappings],
    ):
        audit.compare_columns(
            "domain_and_feature_column_exact",
            current_domain,
            archive_domain,
            domain_mappings,
        )
    return model_differences


def audit_reconstructed_splits(
    audit: Audit,
    pairs: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    v3: Any,
) -> None:
    current_outer, archive_outer = pairs["outer"]
    current_inner, archive_inner = pairs["inner"]
    rows, _ = core.load_rows(v3.DATA_FILE)
    scaffold_ids = core.build_scaffold_ids(rows["canonical_smiles"])
    splits, _ = ood.build_ood_splits(
        rows,
        scaffold_ids,
        ["target_ood"],
        core.DEFAULT_SEED,
        None,
    )
    target_splits = {
        split.heldout_group: split
        for split in splits
        if split.heldout_group in EXPECTED_TARGETS
    }
    audit.check(
        "reconstructed_outer_target_set",
        set(target_splits) == set(EXPECTED_TARGETS),
        sorted(target_splits),
        sorted(EXPECTED_TARGETS),
    )

    for target in EXPECTED_TARGETS:
        if target not in target_splits:
            audit.check(
                f"outer_split_reconstruction::{target}",
                False,
                "missing reconstructed split",
                "one reconstructed target-OOD split",
            )
            continue
        split = target_splits[target]
        current_rows = current_outer[current_outer["heldout_target"] == target]
        archive_rows = archive_outer[archive_outer["heldout_group"] == target]
        if len(current_rows) != 1 or len(archive_rows) != 1:
            audit.check(
                f"outer_split_unique::{target}",
                False,
                {"current": len(current_rows), "archive": len(archive_rows)},
                {"current": 1, "archive": 1},
            )
            continue
        current_row = current_rows.iloc[0]
        archive_row = archive_rows.iloc[0]
        expected_train_hash = sequence_sha256(split.train_idx)
        expected_test_hash = sequence_sha256(split.test_idx)
        outer_checks = {
            "split_seed": (
                int(current_row["split_seed"])
                == int(archive_row["split_seed"])
                == int(split.split_seed)
            ),
            "train_n": (
                int(current_row["n_train_rows"])
                == int(archive_row["n_train_rows"])
                == len(split.train_idx)
            ),
            "test_n": (
                int(current_row["n_test_rows"])
                == int(archive_row["n_test_rows"])
                == len(split.test_idx)
            ),
            "train_sequence": (
                current_row["train_row_indices_sha256"] == expected_train_hash
            ),
            "test_sequence": (
                current_row["test_row_indices_sha256"] == expected_test_hash
            ),
        }
        for name, passed in outer_checks.items():
            audit.check(
                f"outer_split_{name}::{target}",
                passed,
                {
                    "current_seed": current_row.get("split_seed"),
                    "archive_seed": archive_row.get("split_seed"),
                    "reconstructed_seed": split.split_seed,
                    "current_train_hash": current_row.get(
                        "train_row_indices_sha256"
                    ),
                    "reconstructed_train_hash": expected_train_hash,
                    "current_test_hash": current_row.get(
                        "test_row_indices_sha256"
                    ),
                    "reconstructed_test_hash": expected_test_hash,
                },
                "current, archive, and reconstructed values exactly agree",
            )

        for model_id in EXPECTED_MODELS:
            prediction_rows = pairs["predictions"][0][
                (pairs["predictions"][0]["heldout_target"] == target)
                & (pairs["predictions"][0]["model_id"] == model_id)
            ]
            observed_indices = prediction_rows["row_index"].to_numpy(
                dtype=np.int64
            )
            audit.check(
                f"prediction_test_order::{target}::{model_id}",
                bool(np.array_equal(observed_indices, split.test_idx)),
                sequence_sha256(observed_indices),
                expected_test_hash,
            )

        reconstructed_inner = core.balanced_group_splits(
            split.train_idx,
            scaffold_ids,
            4,
            split.split_seed + 101,
        )
        target_current_inner = (
            current_inner[current_inner["heldout_target"] == target]
            .sort_values("inner_fold")
            .reset_index(drop=True)
        )
        target_archive_inner = (
            archive_inner[archive_inner["heldout_group"] == target]
            .sort_values("inner_fold")
            .reset_index(drop=True)
        )
        audit.check(
            f"inner_split_count::{target}",
            len(target_current_inner)
            == len(target_archive_inner)
            == len(reconstructed_inner)
            == 4,
            {
                "current": len(target_current_inner),
                "archive": len(target_archive_inner),
                "reconstructed": len(reconstructed_inner),
            },
            {"current": 4, "archive": 4, "reconstructed": 4},
        )
        if not (
            len(target_current_inner)
            == len(target_archive_inner)
            == len(reconstructed_inner)
            == 4
        ):
            continue
        for position, (fit_idx, eval_idx) in enumerate(reconstructed_inner):
            current_row = target_current_inner.iloc[position]
            archive_row = target_archive_inner.iloc[position]
            fit_scaffolds = sorted(set(scaffold_ids[fit_idx].astype(str)))
            eval_scaffolds = sorted(set(scaffold_ids[eval_idx].astype(str)))
            expected = {
                "inner_seed": split.split_seed + 101,
                "current_fit_rows": sequence_sha256(fit_idx),
                "current_eval_rows": sequence_sha256(eval_idx),
                "current_fit_groups": sequence_sha256(fit_scaffolds),
                "current_eval_groups": sequence_sha256(eval_scaffolds),
                "archive_fit_rows": ood.sequence_sha256(
                    int(value) for value in fit_idx
                ),
                "archive_eval_rows": ood.sequence_sha256(
                    int(value) for value in eval_idx
                ),
                "archive_fit_groups": ood.sequence_sha256(fit_scaffolds),
                "archive_eval_groups": ood.sequence_sha256(eval_scaffolds),
            }
            observed = {
                "current_inner_seed": int(current_row["inner_seed"]),
                "archive_inner_seed": int(archive_row["inner_seed"]),
                "current_fit_rows": current_row["fit_row_indices_sha256"],
                "archive_fit_rows": archive_row["fit_row_indices_sha256"],
                "current_eval_rows": current_row["eval_row_indices_sha256"],
                "archive_eval_rows": archive_row["eval_row_indices_sha256"],
                "current_fit_groups": current_row["fit_group_ids_sha256"],
                "archive_fit_groups": archive_row["fit_scaffold_ids_sha256"],
                "current_eval_groups": current_row["eval_group_ids_sha256"],
                "archive_eval_groups": archive_row["eval_scaffold_ids_sha256"],
            }
            passed = (
                observed["current_inner_seed"]
                == observed["archive_inner_seed"]
                == expected["inner_seed"]
                and observed["current_fit_rows"]
                == expected["current_fit_rows"]
                and observed["archive_fit_rows"]
                == expected["archive_fit_rows"]
                and observed["current_eval_rows"]
                == expected["current_eval_rows"]
                and observed["archive_eval_rows"]
                == expected["archive_eval_rows"]
                and observed["current_fit_groups"]
                == expected["current_fit_groups"]
                and observed["archive_fit_groups"]
                == expected["archive_fit_groups"]
                and observed["current_eval_groups"]
                == expected["current_eval_groups"]
                and observed["archive_eval_groups"]
                == expected["archive_eval_groups"]
                and int(current_row["n_fit_rows"])
                == int(archive_row["n_fit_rows"])
                == len(fit_idx)
                and int(current_row["n_eval_rows"])
                == int(archive_row["n_eval_rows"])
                == len(eval_idx)
            )
            audit.check(
                f"inner_split_sequence::{target}::fold_{position + 1}",
                passed,
                observed,
                expected,
            )


def write_outputs(
    current_dir: Path,
    audit: Audit,
    model_differences: dict[str, dict[str, Any]],
    identities: dict[str, Any],
    execution_profile: str,
    fatal_error: str | None,
) -> dict[str, Any]:
    passed = sum(item["status"] == "PASS" for item in audit.checks)
    payload = {
        "status": (
            "PASS"
            if fatal_error is None and passed == len(audit.checks)
            else "FAIL"
        ),
        "audit_scope": (
            "All eight frozen target-OOD groups, no source deletion, "
            "two matched ExtraTrees models"
        ),
        "execution_profile": execution_profile,
        "n_checks": len(audit.checks),
        "n_pass": passed,
        "n_fail": len(audit.checks) - passed,
        "fatal_error": fatal_error,
        "model_prediction_differences": model_differences,
        "identities": identities,
        "checks": audit.checks,
    }
    current_dir.mkdir(parents=True, exist_ok=True)
    output_json = current_dir / OUTPUT_JSON
    output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_ready)
        + "\n",
        encoding="utf-8",
    )
    markdown = [
        "# Scaffold/source v3 all-target exact-identity audit",
        "",
        f"- Status: **{payload['status']}**",
        f"- Checks: {passed}/{len(audit.checks)} PASS",
        f"- Eligible execution profile: `{execution_profile}`",
        (
            "- Scope: all eight frozen target-OOD groups, no source deletion, "
            "both matched ExtraTrees models."
        ),
        (
            "- Identity standard: exact prediction values and keys, responses, "
            "selected hyperparameters, inner-tuning fields, domain metrics, "
            "feature dimensions, and reconstructed outer/inner split sequences."
        ),
        (
            "- Execution provenance: v3 protocol/configuration plus the locked "
            "OpenMP/OpenBLAS runtime snapshots."
        ),
        "",
    ]
    if fatal_error is not None:
        markdown.extend(["## Fatal error", "", f"`{fatal_error}`", ""])
    markdown.extend(
        [
            "## Prediction identity by target and model",
            "",
            "| Target/model | Rows (v3/archive) | Mean absolute difference | Maximum absolute difference | Exact |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key, values in model_differences.items():
        mean = values["mean_absolute_difference"]
        maximum = values["maximum_absolute_difference"]
        mean_text = "NA" if mean is None else f"{mean:.17g}"
        maximum_text = "NA" if maximum is None else f"{maximum:.17g}"
        markdown.append(
            f"| {key} | {values['n_current']}/{values['n_archive']} | "
            f"{mean_text} | {maximum_text} | {values['array_equal']} |"
        )
    failed_checks = [
        item["check_id"]
        for item in audit.checks
        if item["status"] != "PASS"
    ]
    markdown.extend(
        [
            "",
            "## Failed checks",
            "",
            (
                "None."
                if not failed_checks
                else "\n".join(f"- `{check_id}`" for check_id in failed_checks)
            ),
            "",
        ]
    )
    (current_dir / OUTPUT_MARKDOWN).write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    args = parse_args()
    current_dir = args.current_dir.resolve()
    archive_dir = args.archive_dir.resolve()
    audit = Audit()
    model_differences: dict[str, dict[str, Any]] = {}
    identities: dict[str, Any] = {
        "audit_script_sha256": sha256_file(Path(__file__).resolve()),
        "current_dir": str(current_dir),
        "archive_dir": str(archive_dir),
    }
    execution_profile = "unavailable"
    fatal_error: str | None = None
    try:
        v3 = load_v3_module()
        identities["v3_runner_sha256"] = sha256_file(Path(v3.__file__).resolve())
        provenance = audit_protocol_and_threads(
            audit,
            current_dir,
            archive_dir,
            v3,
        )
        execution_profile = provenance["execution_profile"]
        pairs = filter_and_sort_artifacts(
            audit,
            current_dir,
            archive_dir,
            v3,
        )
        model_differences = audit_prediction_selected_tuning_and_features(
            audit,
            pairs,
        )
        audit_reconstructed_splits(audit, pairs, v3)
        identities.update(
            {
                "v3_protocol_sha256": sha256_file(Path(v3.PROTOCOL_DOCUMENT)),
                "v3_predictions_sha256": sha256_file(
                    current_dir / v3.SOURCE_FILES["predictions"]
                ),
                "archive_predictions_sha256": sha256_file(
                    archive_dir / "ood_predictions.csv"
                ),
                "v3_native_threadpool_audit_sha256": sha256_file(
                    current_dir / "native_threadpool_audit.json"
                ),
                "configuration_protocol_version": provenance[
                    "configuration"
                ].get("protocol_version"),
                "manifest_protocol_version": provenance["manifest"].get(
                    "protocol_version"
                ),
            }
        )
    except Exception as exc:  # Fail closed while still writing audit artifacts.
        fatal_error = f"{type(exc).__name__}: {exc}"
        audit.check(
            "audit_completed_without_exception",
            False,
            fatal_error,
            "no exception",
        )

    payload = write_outputs(
        current_dir,
        audit,
        model_differences,
        identities,
        execution_profile,
        fatal_error,
    )
    print(
        f"[ALL-TARGET IDENTITY] {payload['status']} "
        f"{payload['n_pass']}/{payload['n_checks']} "
        f"output={current_dir / OUTPUT_JSON}"
    )
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
