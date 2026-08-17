#!/usr/bin/env python
"""Run the thread-locked v3 scaffold/source sensitivity package.

The mathematical estimands and fitting implementation are imported from the
frozen v2 runner after its exact SHA-256 is verified.  No v2 result artifact is
read or resumed.  Version 3 adds a fail-closed native-thread contract and an
all-target exact confirmatory-baseline identity gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


# These variables must already be present before NumPy/scikit-learn are
# imported.  A wrong or implicit value is not publication-auditable.
REQUIRED_NATIVE_THREAD_ENV = {
    "OMP_NUM_THREADS": "24",
    "OPENBLAS_NUM_THREADS": "24",
}
_native_env_mismatch = {
    key: os.environ.get(key)
    for key, expected in REQUIRED_NATIVE_THREAD_ENV.items()
    if os.environ.get(key) != expected
}
if _native_env_mismatch:
    raise RuntimeError(
        "The v3 process must be launched with native thread variables set "
        "before Python imports third-party libraries: "
        "OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24. "
        f"Observed mismatches: {_native_env_mismatch}"
    )

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
from rdkit import rdBase  # noqa: E402
from sklearn import __version__ as sklearn_version  # noqa: E402
from threadpoolctl import threadpool_info  # noqa: E402

import run_post_hoc_scaffold_source_sensitivity_v2 as math_v2  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_FILE = math_v2.DATA_FILE
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "reports" / "post_hoc_scaffold_source_sensitivity_v3"
)
DEFAULT_IDENTITY_ARCHIVE_DIR = (
    PROJECT_DIR / "reports" / "confirmatory_ood_cpu_v1"
)
PROTOCOL_DOCUMENT = (
    PROJECT_DIR / "docs" / "post_hoc_scaffold_source_sensitivity_protocol_v3.md"
)
THREAD_CORRECTION_DOCUMENT = (
    PROJECT_DIR
    / "docs"
    / "post_hoc_scaffold_source_thread_execution_correction_v3.md"
)
FLOAT32_CORRECTION_DOCUMENT = math_v2.CORRECTION_DOCUMENT
MASTER_PROTOCOL_DOCUMENT = math_v2.MASTER_PROTOCOL_DOCUMENT
IDENTITY_QA_SCRIPT = (
    SCRIPT_DIR / "qa_post_hoc_scaffold_source_v3_all_target_identity.py"
)

PROTOCOL_VERSION = "post_hoc_scaffold_source_sensitivity_v3.0"
RESPONSE_DTYPE = np.dtype(np.float32)
EXPECTED_DATA_SHA256 = math_v2.EXPECTED_DATA_SHA256
EXPECTED_N_ROWS = math_v2.EXPECTED_N_ROWS
DEFAULT_SEED = math_v2.DEFAULT_SEED
MODELS = math_v2.MODELS
TARGETS = math_v2.TARGETS
SOURCE_DELETIONS = math_v2.SOURCE_DELETIONS
BASELINE_CONDITION = math_v2.BASELINE_CONDITION
GENERIC_FILES = math_v2.GENERIC_FILES
SOURCE_FILES = math_v2.SOURCE_FILES

EXPECTED_MASTER_PROTOCOL_SHA256 = (
    "7dee28f45e3ddf8d6299622a8e494c50492b3b2614f39ce8b2fdd2a1be754ff8"
)
EXPECTED_FLOAT32_CORRECTION_SHA256 = (
    "75a8b0e4f894cead9d72039eab9e6162a89d2a20c79afceda8a2aae1ffd78f3f"
)
EXPECTED_PROTOCOL_SHA256 = (
    "5e67fe833b3b5d3f87d38ac3b28b6b7ac2600aebb2756676e031aed7bd2aab83"
)
EXPECTED_THREAD_CORRECTION_SHA256 = (
    "733b56540fb3917c6c2e3189b7c4d8c4bc33118a5f1fd2db3f132e46e3210456"
)
EXPECTED_V2_MATH_RUNNER_SHA256 = (
    "449d8821ff31e9a5392a227a86ce78fc09c1660f293449a1fbc850a3ee8ffb34"
)
EXPECTED_IDENTITY_QA_SHA256 = (
    "3636760961c858716d0eb087a4be5d8491058da45a61bc1a9821f73fb262b298"
)

# Frozen confirmatory files used by the all-target implementation identity
# gate.  They are not new observations or a new estimand.
EXPECTED_CONFIRMATORY_HASHES = {
    "ood_predictions.csv": (
        "0af39d248adff5cd17fc6d21cce1bfbd0c38051a740d14019d2e80b49c0cef5a"
    ),
    "ood_domain_metrics.csv": (
        "0409b4072403847f0316652ad2c16f713c07a60abab0c837b03612c7a5beff77"
    ),
    "ood_selected_hyperparameters.csv": (
        "f3c031f2b7a73a09edbbd0d5ba20828d2a396d619e50742e2c7f51aa81e3a36f"
    ),
    "ood_inner_tuning_metrics.csv": (
        "83e23ac28c7ff6e9d96efc03b0146e0c3376eba684747a7e9bf782c4cb02c5de"
    ),
    "ood_split_audit.csv": (
        "18c281f1ef66797288b741bd9522f91097182ee4a4366f9d60315c747238fd27"
    ),
    "ood_inner_split_audit.csv": (
        "e70cfed6ae8bfe1550495d07478bad04a0ec7cef2333cbec8493234a4735be83"
    ),
}

THREAD_AUDIT_FILENAME = "native_threadpool_audit.json"
IDENTITY_AUDIT_JSON = "all_target_exact_identity_audit.json"
IDENTITY_AUDIT_MD = "all_target_exact_identity_audit.md"

FINAL_FILES = (
    *math_v2.FINAL_FILES,
    THREAD_AUDIT_FILENAME,
    IDENTITY_AUDIT_JSON,
    IDENTITY_AUDIT_MD,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("both", "generic-scaffold", "source-deletion"),
        default="both",
    )
    parser.add_argument("--data-file", type=Path, default=DATA_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--identity-archive-dir",
        type=Path,
        default=DEFAULT_IDENTITY_ARCHIVE_DIR,
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--outer-repeats", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-outer-splits", type=int, default=None)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--max-deletions", type=int, default=None)
    parser.add_argument("--only-target", choices=TARGETS, default=None)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.outer_folds < 2 or args.inner_folds < 2:
        parser.error("outer-folds and inner-folds must be >= 2")
    if args.outer_repeats < 1:
        parser.error("outer-repeats must be >= 1")
    if args.n_estimators < 10:
        parser.error("n-estimators must be >= 10")
    if args.n_jobs != 4:
        parser.error("v3 requires exactly four joblib jobs per fit")
    if args.bootstrap_replicates < 0:
        parser.error("bootstrap-replicates must be >= 0")
    for field in ("max_outer_splits", "max_targets", "max_deletions"):
        value = getattr(args, field)
        if value is not None and value < 1:
            parser.error(f"{field.replace('_', '-')} must be >= 1")
    if args.resume and args.overwrite:
        parser.error("resume and overwrite are mutually exclusive")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    math_v2.write_json(path, payload)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    math_v2.atomic_csv(frame, path)


def formal_settings_match(args: argparse.Namespace) -> bool:
    return bool(
        args.mode == "both"
        and args.outer_folds == 5
        and args.outer_repeats == 5
        and args.inner_folds == 4
        and args.n_estimators == 600
        and args.n_jobs == 4
        and args.bootstrap_replicates == 10_000
        and args.seed == DEFAULT_SEED
        and args.max_outer_splits is None
        and args.max_targets is None
        and args.max_deletions is None
        and args.only_target is None
        and not args.baseline_only
        and sha256_file(args.data_file) == EXPECTED_DATA_SHA256
        and all(
            os.environ.get(key) == value
            for key, value in REQUIRED_NATIVE_THREAD_ENV.items()
        )
    )


def identity_settings_match(args: argparse.Namespace) -> bool:
    """Whether this run must execute the all-target exact-identity gate."""

    return bool(
        args.mode in ("both", "source-deletion")
        and args.inner_folds == 4
        and args.n_estimators == 600
        and args.n_jobs == 4
        and args.seed == DEFAULT_SEED
        and args.max_targets is None
        and args.only_target is None
        and sha256_file(args.data_file) == EXPECTED_DATA_SHA256
    )


def expected_known_files() -> set[str]:
    return {
        *GENERIC_FILES.values(),
        *SOURCE_FILES.values(),
        *FINAL_FILES,
    }


def prepare_output_dir(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    name = output.name
    if "_v3" not in name or "_v1" in name or "_v2" in name:
        raise ValueError(
            "Version-3 output must have an explicit _v3 basename and must "
            "not contain _v1 or _v2"
        )
    if output.is_symlink():
        raise RuntimeError(f"Refusing symlinked output directory: {output}")
    markers = (
        "SUPERSEDED_DO_NOT_USE.md",
        "SUPERSEDED_FOR_PUBLICATION_DO_NOT_USE.md",
    )
    if any((output / marker).exists() for marker in markers):
        raise RuntimeError("Refusing an output directory with a superseded marker")
    output.mkdir(parents=True, exist_ok=True)

    entries = list(output.iterdir())
    symlinks = [path.name for path in entries if path.is_symlink()]
    directories = [path.name for path in entries if path.is_dir()]
    known = expected_known_files()
    unknown_files = [
        path.name
        for path in entries
        if path.is_file() and path.name not in known
    ]
    if symlinks or directories or unknown_files:
        raise RuntimeError(
            "Unsafe or unknown output entries: "
            f"symlinks={symlinks}, directories={directories}, "
            f"unknown_files={unknown_files}"
        )
    existing = [path for path in entries if path.is_file()]
    if existing and not (args.overwrite or args.resume):
        raise FileExistsError(
            f"Output contains known artifacts; use --resume or --overwrite: {output}"
        )
    if args.overwrite:
        for path in existing:
            path.unlink()
    if args.resume:
        for filename in ("run_manifest.json", "scientific_configuration.json"):
            path = output / filename
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("protocol_version") != PROTOCOL_VERSION:
                raise RuntimeError(f"Cross-version resume refused: {path}")
            if payload.get("response_dtype") != RESPONSE_DTYPE.name:
                raise RuntimeError(f"Non-float32 resume refused: {path}")
            contract = payload.get("native_thread_contract", {})
            if contract.get("required_environment") != REQUIRED_NATIVE_THREAD_ENV:
                raise RuntimeError(f"Native-thread contract mismatch: {path}")


def validate_frozen_inputs(args: argparse.Namespace) -> dict[str, str]:
    paths = {
        "data": args.data_file,
        "master_protocol": MASTER_PROTOCOL_DOCUMENT,
        "float32_correction": FLOAT32_CORRECTION_DOCUMENT,
        "v3_protocol": PROTOCOL_DOCUMENT,
        "thread_correction": THREAD_CORRECTION_DOCUMENT,
        "v2_math_runner": Path(math_v2.__file__).resolve(),
        "identity_qa": IDENTITY_QA_SCRIPT,
    }
    expected = {
        "data": EXPECTED_DATA_SHA256,
        "master_protocol": EXPECTED_MASTER_PROTOCOL_SHA256,
        "float32_correction": EXPECTED_FLOAT32_CORRECTION_SHA256,
        "v3_protocol": EXPECTED_PROTOCOL_SHA256,
        "thread_correction": EXPECTED_THREAD_CORRECTION_SHA256,
        "v2_math_runner": EXPECTED_V2_MATH_RUNNER_SHA256,
        "identity_qa": EXPECTED_IDENTITY_QA_SHA256,
    }
    observed: dict[str, str] = {}
    for label, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"{label} is missing or symlinked: {path}")
        observed[label] = sha256_file(path)
        if observed[label] != expected[label]:
            raise RuntimeError(
                f"{label} checksum mismatch: observed={observed[label]}, "
                f"expected={expected[label]}"
            )
    if identity_settings_match(args):
        for filename, expected_hash in EXPECTED_CONFIRMATORY_HASHES.items():
            path = args.identity_archive_dir / filename
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(
                    f"Confirmatory identity artifact missing or symlinked: {path}"
                )
            observed_hash = sha256_file(path)
            if observed_hash != expected_hash:
                raise RuntimeError(
                    f"Confirmatory identity artifact mismatch: {filename} "
                    f"observed={observed_hash}, expected={expected_hash}"
                )
            observed[f"confirmatory::{filename}"] = observed_hash
    return observed


def _normalized_threadpool_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pool in threadpool_info():
        records.append(
            {
                "user_api": str(pool.get("user_api", "")),
                "internal_api": str(pool.get("internal_api", "")),
                "num_threads": int(pool.get("num_threads", -1)),
                "prefix": str(pool.get("prefix", "")),
                "filepath": str(pool.get("filepath", "")),
                "version": (
                    None
                    if pool.get("version") is None
                    else str(pool.get("version"))
                ),
                "threading_layer": (
                    None
                    if pool.get("threading_layer") is None
                    else str(pool.get("threading_layer"))
                ),
                "architecture": (
                    None
                    if pool.get("architecture") is None
                    else str(pool.get("architecture"))
                ),
            }
        )
    return sorted(
        records,
        key=lambda row: (
            row["user_api"],
            row["internal_api"],
            row["prefix"],
            row["filepath"],
        ),
    )


def _cpu_affinity() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(int(value) for value in os.sched_getaffinity(0))
    return list(range(int(os.cpu_count() or 0)))


def capture_thread_snapshot(
    output_dir: Path,
    snapshots: list[dict[str, Any]],
    stage: str,
) -> None:
    pools = _normalized_threadpool_records()
    relevant = [
        pool for pool in pools if pool["user_api"] in {"openmp", "blas"}
    ]
    user_apis = {pool["user_api"] for pool in relevant}
    affinity = _cpu_affinity()
    issues: list[str] = []
    if user_apis != {"openmp", "blas"}:
        issues.append(f"required user APIs missing: observed={sorted(user_apis)}")
    for pool in relevant:
        if pool["num_threads"] != 24:
            issues.append(
                f"{pool['prefix']}:{pool['user_api']}="
                f"{pool['num_threads']} (expected 24)"
            )
    if len(affinity) < 24:
        issues.append(f"CPU affinity exposes {len(affinity)} CPUs; expected >=24")
    for key, expected in REQUIRED_NATIVE_THREAD_ENV.items():
        if os.environ.get(key) != expected:
            issues.append(
                f"{key}={os.environ.get(key)!r}; expected {expected!r}"
            )
    snapshot = {
        "stage": stage,
        "status": "PASS" if not issues else "FAIL",
        "captured_unix": time.time(),
        "required_environment": dict(REQUIRED_NATIVE_THREAD_ENV),
        "observed_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_DYNAMIC",
                "OMP_PROC_BIND",
                "OMP_PLACES",
            )
        },
        "cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "cpu_affinity_count": len(affinity),
        "threadpools": pools,
        "issues": issues,
    }
    snapshots.append(snapshot)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "contract_version": "scaffold_source_native_threads_v3.0",
        "required_environment": dict(REQUIRED_NATIVE_THREAD_ENV),
        "required_native_pool_threads": 24,
        "required_user_apis": ["blas", "openmp"],
        "threadpoolctl_version": importlib.metadata.version("threadpoolctl"),
        "status": (
            "PASS"
            if all(item["status"] == "PASS" for item in snapshots)
            else "FAIL"
        ),
        "snapshots": snapshots,
    }
    write_json(output_dir / THREAD_AUDIT_FILENAME, payload)
    if issues:
        raise RuntimeError(
            f"Native-thread contract failed at {stage}: " + "; ".join(issues)
        )


def environment_manifest(args: argparse.Namespace) -> dict[str, Any]:
    affinity = _cpu_affinity()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn_version,
        "rdkit": rdBase.rdkitVersion,
        "joblib": joblib.__version__,
        "threadpoolctl": importlib.metadata.version("threadpoolctl"),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "cpu_affinity_count": len(affinity),
        "n_jobs": int(args.n_jobs),
        "native_thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_DYNAMIC",
                "OMP_PROC_BIND",
                "OMP_PLACES",
            )
        },
    }


def enrich_source_domain_metrics(
    output_dir: Path,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Preserve fit provenance that v2's final aggregate table discarded."""

    path = output_dir / SOURCE_FILES["domain_metrics"]
    fit_metrics = pd.read_csv(path)
    split = pd.read_csv(output_dir / SOURCE_FILES["split_audit"])
    required_fit = {
        "deletion_source",
        "heldout_target",
        "model_id",
        "param_id",
        "n_train_rows",
        "n_test_rows",
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
    }
    if not required_fit.issubset(fit_metrics.columns):
        raise RuntimeError(
            "Source fit-domain metrics are missing v3 provenance fields: "
            f"{sorted(required_fit - set(fit_metrics.columns))}"
        )
    enriched_columns = [
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
    ]
    if tuple(fit_metrics.columns) == tuple(enriched_columns):
        expected_rows = (
            predictions[
                ["heldout_target", "deletion_source", "model_id"]
            ]
            .drop_duplicates()
            .shape[0]
        )
        if len(fit_metrics) != expected_rows:
            raise RuntimeError(
                "Previously enriched source-domain metrics are incomplete"
            )
        return fit_metrics
    split_columns = [
        "heldout_target",
        "deletion_source",
        "n_train_rows",
        "n_train_unique_smiles",
        "n_train_bemis_murcko_scaffolds",
        "n_test_rows",
        "n_test_unique_smiles",
        "n_test_bemis_murcko_scaffolds",
    ]
    if not set(split_columns).issubset(split.columns):
        raise RuntimeError("Source split audit is missing v3 count provenance")
    counts = split.loc[
        split["status"].astype(str) == "complete", split_columns
    ].copy()
    counts = counts.rename(
        columns={
            "n_train_bemis_murcko_scaffolds": "n_train_scaffolds",
            "n_test_bemis_murcko_scaffolds": "n_test_scaffolds",
        }
    )
    for column in ("n_train_rows", "n_test_rows"):
        fit_metrics = fit_metrics.drop(columns=[column])
    enriched = fit_metrics.merge(
        counts,
        on=["heldout_target", "deletion_source"],
        how="left",
        validate="many_to_one",
    )
    expected_rows = (
        predictions[
            ["heldout_target", "deletion_source", "model_id"]
        ]
        .drop_duplicates()
        .shape[0]
    )
    if (
        len(enriched) != expected_rows
        or enriched[
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
    ):
        raise RuntimeError("Enriched source-domain metrics are incomplete")
    enriched = enriched.loc[:, enriched_columns]
    atomic_csv(enriched, path)
    return enriched


def basic_all_target_prediction_identity(
    args: argparse.Namespace,
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    current = predictions[
        predictions["deletion_source"].astype(str) == BASELINE_CONDITION
    ].copy()
    archive = pd.read_csv(args.identity_archive_dir / "ood_predictions.csv")
    archive = archive[
        (archive["protocol"].astype(str) == "target_ood")
        & archive["model_id"].astype(str).isin(MODELS)
    ].rename(columns={"heldout_group": "heldout_target"})
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
    current = current.loc[:, columns].sort_values(
        sort_columns, kind="mergesort"
    ).reset_index(drop=True)
    archive = archive.loc[:, columns].sort_values(
        sort_columns, kind="mergesort"
    ).reset_index(drop=True)
    exact_columns = {
        column: bool(
            len(current) == len(archive)
            and np.array_equal(
                current[column].to_numpy(),
                archive[column].to_numpy(),
            )
        )
        for column in columns
    }
    if len(current) == len(archive):
        absolute = np.abs(
            current["y_pred"].to_numpy(dtype=np.float64)
            - archive["y_pred"].to_numpy(dtype=np.float64)
        )
        max_abs = float(absolute.max()) if len(absolute) else float("nan")
    else:
        max_abs = float("nan")
    target_ok = (
        set(current["heldout_target"].astype(str)) == set(TARGETS)
        and set(archive["heldout_target"].astype(str)) == set(TARGETS)
    )
    models_ok = set(current["model_id"].astype(str)) == set(MODELS)
    passed = bool(
        len(current) == len(archive) == 2_580
        and target_ok
        and models_ok
        and all(exact_columns.values())
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "n_current_rows": int(len(current)),
        "n_archive_rows": int(len(archive)),
        "expected_rows": 2_580,
        "target_coverage_exact": target_ok,
        "model_coverage_exact": models_ok,
        "exact_columns": exact_columns,
        "maximum_absolute_prediction_difference": max_abs,
    }


def add_check(
    checks: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    observed: Any,
    expected: Any,
) -> None:
    checks.append(
        {
            "check_id": check_id,
            "status": "PASS" if bool(passed) else "FAIL",
            "observed": observed,
            "expected": expected,
        }
    )


def rewrite_qa_payload(
    args: argparse.Namespace,
    base_payload: dict[str, Any],
    native_audit: dict[str, Any],
    basic_identity: dict[str, Any] | None,
    independent_identity: dict[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    checks = list(base_payload["checks"])
    add_check(
        checks,
        "native_threadpool_contract",
        native_audit.get("status") == "PASS",
        native_audit.get("status"),
        "PASS",
    )
    required_stages = {"startup"}
    if args.mode in ("both", "generic-scaffold"):
        required_stages.update({"before_generic", "after_generic"})
    if args.mode in ("both", "source-deletion"):
        required_stages.update({"before_source", "after_source"})
    observed_stages = {
        str(item.get("stage")) for item in native_audit.get("snapshots", [])
    }
    add_check(
        checks,
        "native_threadpool_stage_completeness",
        required_stages.issubset(observed_stages),
        sorted(observed_stages),
        sorted(required_stages),
    )
    if basic_identity is not None:
        add_check(
            checks,
            "all_target_baseline_prediction_identity",
            basic_identity.get("status") == "PASS",
            basic_identity,
            "PASS with 2,580 exact rows",
        )
    if independent_identity is not None:
        add_check(
            checks,
            "independent_all_target_identity_qa",
            independent_identity.get("status") == "PASS",
            {
                "status": independent_identity.get("status"),
                "n_pass": independent_identity.get("n_pass"),
                "n_fail": independent_identity.get("n_fail"),
            },
            "PASS",
        )
    passed = sum(check["status"] == "PASS" for check in checks)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "PASS" if passed == len(checks) else "FAIL",
        "formal_settings_match": formal_settings_match(args),
        "identity_settings_match": identity_settings_match(args),
        "n_checks": len(checks),
        "n_pass": passed,
        "n_fail": len(checks) - passed,
        "checks": checks,
    }
    lines = [
        "# QA summary",
        "",
        f"- Status: **{payload['status']}**",
        f"- Formal settings match: `{payload['formal_settings_match']}`",
        f"- All-target identity settings match: "
        f"`{payload['identity_settings_match']}`",
        f"- Checks: {passed}/{len(checks)} PASS",
        "",
        "| Check | Status | Observed | Expected |",
        "|---|---:|---:|---:|",
    ]
    for check in checks:
        lines.append(
            f"| {check['check_id']} | {check['status']} | "
            f"{check['observed']} | {check['expected']} |"
        )
    return payload, "\n".join(lines) + "\n"


def run_independent_identity_qa(args: argparse.Namespace) -> dict[str, Any]:
    if not IDENTITY_QA_SCRIPT.is_file() or IDENTITY_QA_SCRIPT.is_symlink():
        raise FileNotFoundError(
            f"Independent all-target identity QA is missing: {IDENTITY_QA_SCRIPT}"
        )
    command = [
        sys.executable,
        str(IDENTITY_QA_SCRIPT),
        "--current-dir",
        str(args.output_dir),
        "--archive-dir",
        str(args.identity_archive_dir),
    ]
    subprocess.run(command, check=True, cwd=PROJECT_DIR)
    payload = json.loads(
        (args.output_dir / IDENTITY_AUDIT_JSON).read_text(encoding="utf-8")
    )
    if payload.get("status") != "PASS":
        raise RuntimeError("Independent all-target identity QA failed")
    return payload


def main() -> None:
    args = parse_args()
    started = time.time()
    prepare_output_dir(args)
    frozen_hashes = validate_frozen_inputs(args)
    snapshots: list[dict[str, Any]] = []
    capture_thread_snapshot(args.output_dir, snapshots, "startup")

    rows, context_audit = math_v2.core.load_rows(args.data_file)
    if len(rows) != EXPECTED_N_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_N_ROWS} sanitized rows, observed {len(rows)}"
        )
    y = rows["pDC50"].to_numpy(dtype=np.float32)
    if y.dtype != RESPONSE_DTYPE:
        raise RuntimeError(
            f"Response dtype mismatch: {y.dtype}; expected {RESPONSE_DTYPE}"
        )
    bemis_ids = math_v2.core.build_scaffold_ids(rows["canonical_smiles"])
    generic_ids = math_v2.build_generic_scaffold_ids(rows["canonical_smiles"])
    group_summary = math_v2.generic_group_summary(rows, generic_ids, bemis_ids)
    record = group_summary.iloc[0]
    observed_groups = (
        int(record["n_generic_murcko_groups"]),
        int(record["n_singleton_generic_groups"]),
        int(record["largest_generic_group_rows"]),
    )
    expected_groups = (
        math_v2.EXPECTED_GENERIC_GROUPS,
        math_v2.EXPECTED_GENERIC_SINGLETONS,
        math_v2.EXPECTED_GENERIC_MAX_ROWS,
    )
    if observed_groups != expected_groups:
        raise RuntimeError(
            f"Generic scaffold identity changed: {observed_groups} != "
            f"{expected_groups}"
        )

    scientific_configuration: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "evidence_label": "post_hoc_sensitivity",
        "response_dtype": RESPONSE_DTYPE.name,
        "scientific_correction": (
            "native OpenMP/BLAS execution state locked and audited; "
            "mathematical implementation unchanged from hash-locked v2"
        ),
        "formal_settings_match": formal_settings_match(args),
        "identity_settings_match": identity_settings_match(args),
        "args": vars(args),
        "models": list(MODELS),
        "model_grid": {
            model: math_v2.core.model_param_grid(model) for model in MODELS
        },
        "source_deletion_tokens": list(SOURCE_DELETIONS),
        "target_ood_groups": list(TARGETS),
        "response_construction": (
            'rows["pDC50"].to_numpy(dtype=np.float32)'
        ),
        "native_thread_contract": {
            "contract_version": "scaffold_source_native_threads_v3.0",
            "required_environment": dict(REQUIRED_NATIVE_THREAD_ENV),
            "required_native_pool_threads": 24,
            "required_user_apis": ["blas", "openmp"],
            "required_cpu_affinity_count_minimum": 24,
            "audit_file": THREAD_AUDIT_FILENAME,
        },
        "mathematical_implementation": {
            "module": "run_post_hoc_scaffold_source_sensitivity_v2.py",
            "required_sha256": EXPECTED_V2_MATH_RUNNER_SHA256,
            "observed_sha256": frozen_hashes["v2_math_runner"],
            "reuse_scope": (
                "feature/split/tuning/fitting/metric/bootstrap functions only; "
                "no v2 result artifact is read"
            ),
        },
        "data_sha256": frozen_hashes["data"],
        "master_protocol_sha256": frozen_hashes["master_protocol"],
        "float32_correction_document_sha256": frozen_hashes[
            "float32_correction"
        ],
        "protocol_sha256": frozen_hashes["v3_protocol"],
        "thread_correction_document_sha256": frozen_hashes[
            "thread_correction"
        ],
        "this_script_sha256": sha256_file(Path(__file__).resolve()),
        "core_script_sha256": sha256_file(math_v2.CORE_SCRIPT),
        "ood_script_sha256": sha256_file(math_v2.OOD_SCRIPT),
        "identity_qa_script_sha256": (
            sha256_file(IDENTITY_QA_SCRIPT)
            if IDENTITY_QA_SCRIPT.is_file()
            else None
        ),
        "confirmatory_identity_artifact_sha256": {
            key.split("::", 1)[1]: value
            for key, value in frozen_hashes.items()
            if key.startswith("confirmatory::")
        },
        "environment": environment_manifest(args),
        "boundaries": [
            "Uses only the frozen 1,560-row core analysis table.",
            "Post-hoc sensitivity, not confirmatory or prospective evidence.",
            "No v1 or v2 result artifact is loaded or resumed.",
            "No data source other than the declared frozen core table is loaded.",
        ],
    }
    write_json(
        args.output_dir / "scientific_configuration.json",
        scientific_configuration,
    )
    manifest = {
        **scientific_configuration,
        "status": "running",
        "started_unix": started,
    }
    write_json(args.output_dir / "run_manifest.json", manifest)

    print("[FEATURE] building Morgan fingerprints", flush=True)
    morgan = math_v2.core.build_morgan_matrix(rows["canonical_smiles"])
    print("[FEATURE] building RDKit descriptors", flush=True)
    descriptors, descriptor_names, descriptor_failures = (
        math_v2.core.build_descriptor_matrix(rows["canonical_smiles"])
    )
    scientific_configuration["feature_manifest"] = math_v2.core.feature_manifest(
        descriptor_names,
        descriptor_failures,
    )
    write_json(
        args.output_dir / "scientific_configuration.json",
        scientific_configuration,
    )
    context_audit.to_csv(
        args.output_dir / "context_sanitization_audit.csv", index=False
    )
    group_summary.to_csv(
        args.output_dir / "generic_scaffold_group_summary.csv", index=False
    )
    analysis_index = rows[
        [
            "qc_id",
            "canonical_smiles",
            "source_database",
            "target_protein",
            "pDC50",
        ]
    ].copy()
    analysis_index["bemis_murcko_scaffold_id"] = bemis_ids
    analysis_index["generic_scaffold_id"] = generic_ids
    analysis_index.to_csv(args.output_dir / "analysis_index.csv", index=False)

    generic_predictions = pd.DataFrame()
    generic_bootstrap = pd.DataFrame()
    generic_averaged_metrics = pd.DataFrame()
    if args.mode in ("both", "generic-scaffold"):
        capture_thread_snapshot(args.output_dir, snapshots, "before_generic")
        generic_predictions = math_v2.run_generic_scaffold(
            args,
            rows,
            y,
            morgan,
            descriptors,
            generic_ids,
            bemis_ids,
        )
        repeat_metrics, averaged, generic_averaged_metrics = (
            math_v2.generic_summaries(generic_predictions)
        )
        atomic_csv(
            repeat_metrics,
            args.output_dir / "generic_scaffold_metrics_by_repeat.csv",
        )
        atomic_csv(
            averaged,
            args.output_dir / "generic_scaffold_repeat_averaged_predictions.csv",
        )
        atomic_csv(
            generic_averaged_metrics,
            args.output_dir / "generic_scaffold_repeat_averaged_metrics.csv",
        )
        generic_bootstrap = math_v2.generic_cluster_bootstrap(
            averaged, args.bootstrap_replicates, args.seed
        )
        if not generic_bootstrap.empty:
            atomic_csv(
                generic_bootstrap,
                args.output_dir / "generic_scaffold_paired_cluster_bootstrap.csv",
            )
        capture_thread_snapshot(args.output_dir, snapshots, "after_generic")

    source_predictions = pd.DataFrame()
    source_bootstrap = pd.DataFrame()
    source_macro = pd.DataFrame()
    basic_identity: dict[str, Any] | None = None
    if args.mode in ("both", "source-deletion"):
        capture_thread_snapshot(args.output_dir, snapshots, "before_source")
        source_predictions = math_v2.run_source_deletion(
            args,
            rows,
            y,
            morgan,
            descriptors,
            bemis_ids,
        )
        domain_metrics = enrich_source_domain_metrics(
            args.output_dir, source_predictions
        )
        source_macro, deltas = math_v2.source_macro_and_deltas(domain_metrics)
        atomic_csv(
            source_macro,
            args.output_dir / "source_deletion_equal_domain_macro.csv",
        )
        atomic_csv(
            deltas,
            args.output_dir / "source_deletion_domain_paired_deltas.csv",
        )
        source_bootstrap = math_v2.source_stratified_cluster_bootstrap(
            source_predictions, args.bootstrap_replicates, args.seed
        )
        if not source_bootstrap.empty:
            atomic_csv(
                source_bootstrap,
                args.output_dir / "source_deletion_paired_cluster_bootstrap.csv",
            )
        capture_thread_snapshot(args.output_dir, snapshots, "after_source")
        if identity_settings_match(args):
            # The frozen comparator is a CSV artifact.  Compare the atomically
            # emitted v3 CSV after the same round trip, not pre-serialization
            # Python floats whose sub-serialization tails are not part of
            # either publication artifact.
            emitted_source_predictions = pd.read_csv(
                args.output_dir / SOURCE_FILES["predictions"]
            )
            basic_identity = basic_all_target_prediction_identity(
                args, emitted_source_predictions
            )
            if basic_identity["status"] != "PASS":
                raise RuntimeError(
                    "All-target no-deletion predictions do not exactly match "
                    "the frozen confirmatory target-OOD baseline"
                )

    math_v2.write_results_brief(
        args,
        generic_averaged_metrics,
        generic_bootstrap,
        source_macro,
        source_bootstrap,
    )
    base_qa, _ = math_v2.qa_results(
        args,
        rows,
        y,
        generic_ids,
        generic_predictions,
        source_predictions,
        generic_bootstrap,
        source_bootstrap,
    )
    native_audit = json.loads(
        (args.output_dir / THREAD_AUDIT_FILENAME).read_text(encoding="utf-8")
    )
    preliminary_qa, preliminary_md = rewrite_qa_payload(
        args, base_qa, native_audit, basic_identity, None
    )
    write_json(args.output_dir / "qa_summary.json", preliminary_qa)
    (args.output_dir / "qa_summary.md").write_text(
        preliminary_md, encoding="utf-8"
    )

    independent_identity: dict[str, Any] | None = None
    if identity_settings_match(args):
        independent_identity = run_independent_identity_qa(args)
    qa_payload, qa_markdown = rewrite_qa_payload(
        args,
        base_qa,
        native_audit,
        basic_identity,
        independent_identity,
    )
    write_json(args.output_dir / "qa_summary.json", qa_payload)
    (args.output_dir / "qa_summary.md").write_text(
        qa_markdown, encoding="utf-8"
    )

    inventory = math_v2.artifact_inventory(args.output_dir)
    atomic_csv(inventory, args.output_dir / "artifact_sha256.csv")
    if (
        qa_payload["status"] == "PASS"
        and formal_settings_match(args)
        and independent_identity is not None
    ):
        status = "complete"
    elif (
        qa_payload["status"] == "PASS"
        and identity_settings_match(args)
        and args.baseline_only
        and independent_identity is not None
    ):
        status = "identity_preflight_complete"
    elif qa_payload["status"] == "PASS":
        status = "smoke_complete"
    else:
        status = "failed_qa"
    manifest.update(scientific_configuration)
    manifest.update(
        {
            "status": status,
            "qa_status": qa_payload["status"],
            "formal_settings_match": formal_settings_match(args),
            "identity_settings_match": identity_settings_match(args),
            "independent_identity_status": (
                None
                if independent_identity is None
                else independent_identity.get("status")
            ),
            "native_thread_status": native_audit.get("status"),
            "elapsed_seconds": float(time.time() - started),
            "artifact_count": int(len(inventory)),
            "artifact_inventory": "artifact_sha256.csv",
        }
    )
    write_json(args.output_dir / "run_manifest.json", manifest)
    if qa_payload["status"] != "PASS":
        raise RuntimeError("Post-hoc scaffold/source v3 QA failed")
    print(
        f"[DONE] status={status} elapsed={manifest['elapsed_seconds']:.1f}s "
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
