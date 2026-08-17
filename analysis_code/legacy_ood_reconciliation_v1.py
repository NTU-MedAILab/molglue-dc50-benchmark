#!/usr/bin/env python3
"""Strictly reconcile the legacy confirmatory OOD result directories.

This module does not alter either result directory.  It establishes whether
the later operational resume rewrite preserved the frozen scientific payload:

* primary artifacts must remain byte-identical;
* three explicitly named derived summary tables may differ only by finite
  numeric serialization noise at an absolute tolerance of 1e-12;
* the run manifests may differ only in three operational fields and in the
  hash/size inventory entries consequential to those three derived tables;
* the unavailable historical run-manifest digest must remain linked by both
  the strict frozen identity and the original artifact inventory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


RECONCILIATION_VERSION = (
    "confirmatory_ood_cpu_v1_legacy_reconciliation.v1"
)
RECONCILIATION_JSON_NAME = (
    "confirmatory_ood_cpu_v1_legacy_reconciliation_v1.json"
)
RECONCILIATION_MD_NAME = (
    "confirmatory_ood_cpu_v1_legacy_reconciliation_v1.md"
)
ANALYSIS_IDENTITY = "confirmatory_ood_cpu_v1.0"
ABSOLUTE_TOLERANCE = 1e-12
RELATIVE_TOLERANCE = 0.0

CURRENT_DIRECTORY = "reports/confirmatory_ood_cpu_v1"
REPRODUCTION_DIRECTORY = (
    "reports/publication_validation_v1/reproduction/"
    "confirmatory_ood_cpu_v1"
)
FROZEN_IDENTITY_PATH = (
    "docs/post_hoc_strict_domain_scaffold_ood_cpu_v1_frozen_identity.json"
)
ARTIFACT_INVENTORY_PATH = (
    "reports/confirmatory_ood_cpu_v1/artifact_sha256.csv"
)

PRIMARY_EXACT_FILES = (
    "ood_analysis_index.csv",
    "ood_checkpoint_inventory.json",
    "ood_context_sanitization_audit.csv",
    "ood_domain_metrics.csv",
    "ood_inner_split_audit.csv",
    "ood_inner_tuning_metrics.csv",
    "ood_predictions.csv",
    "ood_selected_hyperparameters.csv",
    "ood_split_audit.csv",
)
DERIVED_TOLERANCE_FILES = (
    "ood_aggregate_metrics.csv",
    "ood_domain_scaffold_bootstrap.csv",
    "ood_paired_scaffold_bootstrap.csv",
)
RUN_MANIFEST = "ood_run_manifest.json"
SHARED_FILES = tuple(
    sorted((*PRIMARY_EXACT_FILES, *DERIVED_TOLERANCE_FILES, RUN_MANIFEST))
)
CURRENT_ONLY_REPORT_FILES = (
    "artifact_sha256.csv",
    "qa_summary.md",
    "results_brief_zh.md",
)
MANIFEST_OPERATIONAL_FIELD_ALLOWLIST = (
    "elapsed_seconds",
    "resume_count",
    "run_session_started_unix",
)
MANIFEST_DERIVED_INVENTORY_FIELD_ALLOWLIST = tuple(
    sorted(
        f"artifact_inventory.{filename}.{field}"
        for filename in DERIVED_TOLERANCE_FILES
        for field in ("sha256", "size_bytes")
    )
)
MANIFEST_ALLOWED_DIFFERENCE_PATHS = tuple(
    sorted(
        (
            *MANIFEST_OPERATIONAL_FIELD_ALLOWLIST,
            *MANIFEST_DERIVED_INVENTORY_FIELD_ALLOWLIST,
        )
    )
)
ARTIFACT_INVENTORY_COLUMNS = (
    "relative_path",
    "sha256",
    "size_bytes",
    "kind",
    "n_rows",
    "n_columns",
    "verification",
)
OPERATIONAL_REWRITE_EXPLANATION = (
    "A completed-result resume rewrote elapsed_seconds, resume_count, and "
    "run_session_started_unix and reserialized three derived summary tables. "
    "Predictions, split assignments, tuning, selected hyperparameters, domain "
    "metrics, and the scientific configuration remained exact."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _directory_regular_files(path: Path) -> set[str]:
    observed: set[str] = set()
    for candidate in path.iterdir():
        if candidate.is_symlink():
            raise ValueError(f"symlink is not permitted: {candidate}")
        if candidate.is_file():
            observed.add(candidate.name)
        elif candidate.is_dir():
            raise ValueError(
                f"unexpected directory in legacy result root: {candidate}"
            )
        else:
            raise ValueError(
                f"unsupported filesystem entry in legacy result root: "
                f"{candidate}"
            )
    return observed


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _json_difference_paths(
    left: object,
    right: object,
    prefix: str = "",
) -> list[str]:
    if type(left) is not type(right):
        return [prefix or "<root>"]
    if isinstance(left, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(path)
            else:
                paths.extend(
                    _json_difference_paths(left[key], right[key], path)
                )
        return paths
    if isinstance(left, list):
        paths = []
        if len(left) != len(right):
            paths.append(f"{prefix}.length")
        for index, (left_item, right_item) in enumerate(
            zip(left, right, strict=False)
        ):
            paths.extend(
                _json_difference_paths(
                    left_item,
                    right_item,
                    f"{prefix}[{index}]",
                )
            )
        return paths
    return [] if left == right else [prefix or "<root>"]


def _csv_comparison(
    current_path: Path,
    reproduction_path: Path,
) -> dict[str, Any]:
    current = pd.read_csv(current_path)
    reproduction = pd.read_csv(reproduction_path)
    schema_exact = list(current.columns) == list(reproduction.columns)
    shape_exact = current.shape == reproduction.shape
    if not schema_exact or not shape_exact:
        return {
            "schema_exact": schema_exact,
            "shape": [int(current.shape[0]), int(current.shape[1])],
            "reproduction_shape": [
                int(reproduction.shape[0]),
                int(reproduction.shape[1]),
            ],
            "non_numeric_exact": False,
            "max_abs_numeric_difference": None,
            "numeric_mismatches_at_tolerance": None,
        }

    numeric_columns = list(
        current.select_dtypes(include=[np.number]).columns
    )
    if numeric_columns != list(
        reproduction.select_dtypes(include=[np.number]).columns
    ):
        return {
            "schema_exact": True,
            "shape": [int(current.shape[0]), int(current.shape[1])],
            "reproduction_shape": [
                int(reproduction.shape[0]),
                int(reproduction.shape[1]),
            ],
            "non_numeric_exact": False,
            "max_abs_numeric_difference": None,
            "numeric_mismatches_at_tolerance": None,
        }
    non_numeric_columns = [
        column for column in current.columns if column not in numeric_columns
    ]
    non_numeric_exact = current[non_numeric_columns].equals(
        reproduction[non_numeric_columns]
    )

    max_difference = 0.0
    mismatches = 0
    for column in numeric_columns:
        left = current[column].to_numpy(dtype=float)
        right = reproduction[column].to_numpy(dtype=float)
        if not np.array_equal(np.isnan(left), np.isnan(right)):
            mismatches += int(
                np.count_nonzero(np.isnan(left) != np.isnan(right))
            )
        if not np.array_equal(np.isposinf(left), np.isposinf(right)):
            mismatches += int(
                np.count_nonzero(np.isposinf(left) != np.isposinf(right))
            )
        if not np.array_equal(np.isneginf(left), np.isneginf(right)):
            mismatches += int(
                np.count_nonzero(np.isneginf(left) != np.isneginf(right))
            )
        finite = np.isfinite(left) & np.isfinite(right)
        if finite.any():
            max_difference = max(
                max_difference,
                float(np.max(np.abs(left[finite] - right[finite]))),
            )
        mismatches += int(
            np.count_nonzero(
                ~np.isclose(
                    left,
                    right,
                    atol=ABSOLUTE_TOLERANCE,
                    rtol=RELATIVE_TOLERANCE,
                    equal_nan=True,
                )
            )
        )
    return {
        "schema_exact": True,
        "shape": [int(current.shape[0]), int(current.shape[1])],
        "reproduction_shape": [
            int(reproduction.shape[0]),
            int(reproduction.shape[1]),
        ],
        "non_numeric_exact": bool(non_numeric_exact),
        "max_abs_numeric_difference": max_difference,
        "numeric_mismatches_at_tolerance": mismatches,
    }


def _file_record(
    current_directory: Path,
    reproduction_directory: Path,
    filename: str,
) -> dict[str, Any]:
    current_path = current_directory / filename
    reproduction_path = reproduction_directory / filename
    if not current_path.is_file() or not reproduction_path.is_file():
        raise FileNotFoundError(f"missing reconciliation input: {filename}")
    if current_path.is_symlink() or reproduction_path.is_symlink():
        raise ValueError(f"symlink is not permitted: {filename}")

    if filename in PRIMARY_EXACT_FILES:
        classification = "primary_exact"
    elif filename in DERIVED_TOLERANCE_FILES:
        classification = "derived_numeric_tolerance"
    elif filename == RUN_MANIFEST:
        classification = "operational_manifest"
    else:
        raise ValueError(f"unclassified reconciliation file: {filename}")

    record: dict[str, Any] = {
        "file": filename,
        "classification": classification,
        "current_sha256": sha256_file(current_path),
        "reproduction_sha256": sha256_file(reproduction_path),
        "current_size_bytes": current_path.stat().st_size,
        "reproduction_size_bytes": reproduction_path.stat().st_size,
        "byte_exact": current_path.read_bytes()
        == reproduction_path.read_bytes(),
        "schema_exact": None,
        "shape": None,
        "reproduction_shape": None,
        "non_numeric_exact": None,
        "max_abs_numeric_difference": None,
        "numeric_mismatches_at_tolerance": None,
    }
    if filename.endswith(".csv"):
        record.update(_csv_comparison(current_path, reproduction_path))
    return record


def _historical_manifest_chain(route_dir: Path) -> dict[str, Any]:
    frozen_path = route_dir / FROZEN_IDENTITY_PATH
    inventory_path = route_dir / ARTIFACT_INVENTORY_PATH
    frozen = _read_json_object(frozen_path)
    expected_historical_hash = (
        frozen.get("frozen_identity", {})
        .get("file_sha256", {})
        .get("ood_result_manifest")
    )
    if not isinstance(expected_historical_hash, str):
        raise ValueError("strict frozen identity lacks parent manifest hash")

    with inventory_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ARTIFACT_INVENTORY_COLUMNS:
            raise ValueError("legacy artifact inventory schema/order mismatch")
        rows = list(reader)
    matches = [
        row for row in rows if row.get("relative_path") == RUN_MANIFEST
    ]
    if len(matches) != 1:
        raise ValueError(
            "legacy artifact inventory must contain one run-manifest row"
        )
    row = matches[0]
    if (
        row.get("sha256") != expected_historical_hash
        or row.get("verification") != "PASS"
        or row.get("kind") != "json"
    ):
        raise ValueError("historical run-manifest hash chain is broken")

    current_manifest = route_dir / CURRENT_DIRECTORY / RUN_MANIFEST
    reproduction_manifest = (
        route_dir / REPRODUCTION_DIRECTORY / RUN_MANIFEST
    )
    historical_bytes_available = expected_historical_hash in {
        sha256_file(current_manifest),
        sha256_file(reproduction_manifest),
    }
    return {
        "frozen_identity_path": FROZEN_IDENTITY_PATH,
        "frozen_identity_sha256": sha256_file(frozen_path),
        "frozen_identity_key": (
            "frozen_identity.file_sha256.ood_result_manifest"
        ),
        "frozen_manifest_sha256": expected_historical_hash,
        "artifact_inventory_path": ARTIFACT_INVENTORY_PATH,
        "artifact_inventory_sha256": sha256_file(inventory_path),
        "artifact_inventory_row": {
            column: row[column] for column in ARTIFACT_INVENTORY_COLUMNS
        },
        "historical_manifest_bytes_available_in_compared_directories": (
            historical_bytes_available
        ),
        "current_manifest_is_historical_manifest": (
            sha256_file(current_manifest) == expected_historical_hash
        ),
    }


def build_reconciliation(route_dir: Path) -> dict[str, Any]:
    route_dir = route_dir.resolve()
    current_directory = route_dir / CURRENT_DIRECTORY
    reproduction_directory = route_dir / REPRODUCTION_DIRECTORY
    if not current_directory.is_dir() or not reproduction_directory.is_dir():
        raise FileNotFoundError("legacy OOD reconciliation directory missing")

    expected_current = set(SHARED_FILES) | set(CURRENT_ONLY_REPORT_FILES)
    observed_current = _directory_regular_files(current_directory)
    observed_reproduction = _directory_regular_files(reproduction_directory)
    if observed_current != expected_current:
        raise ValueError(
            "current legacy OOD directory file set changed: "
            f"missing={sorted(expected_current - observed_current)}, "
            f"extra={sorted(observed_current - expected_current)}"
        )
    if observed_reproduction != set(SHARED_FILES):
        raise ValueError(
            "reproduction legacy OOD directory file set changed: "
            f"missing={sorted(set(SHARED_FILES) - observed_reproduction)}, "
            f"extra={sorted(observed_reproduction - set(SHARED_FILES))}"
        )

    records = [
        _file_record(current_directory, reproduction_directory, filename)
        for filename in SHARED_FILES
    ]
    by_name = {record["file"]: record for record in records}
    if any(not by_name[name]["byte_exact"] for name in PRIMARY_EXACT_FILES):
        raise ValueError("a primary legacy OOD artifact changed")
    for name in DERIVED_TOLERANCE_FILES:
        record = by_name[name]
        if (
            record["schema_exact"] is not True
            or record["shape"] != record["reproduction_shape"]
            or record["non_numeric_exact"] is not True
            or record["numeric_mismatches_at_tolerance"] != 0
            or record["max_abs_numeric_difference"] is None
            or not math.isfinite(record["max_abs_numeric_difference"])
            or record["max_abs_numeric_difference"] > ABSOLUTE_TOLERANCE
        ):
            raise ValueError(
                f"derived legacy OOD table exceeds tolerance: {name}"
            )

    current_manifest = _read_json_object(current_directory / RUN_MANIFEST)
    reproduction_manifest = _read_json_object(
        reproduction_directory / RUN_MANIFEST
    )
    if (
        current_manifest.get("scientific_configuration")
        != reproduction_manifest.get("scientific_configuration")
    ):
        raise ValueError("legacy OOD scientific configuration changed")
    difference_paths = tuple(
        sorted(
            _json_difference_paths(
                current_manifest,
                reproduction_manifest,
            )
        )
    )
    if difference_paths != MANIFEST_ALLOWED_DIFFERENCE_PATHS:
        raise ValueError(
            "legacy OOD manifest contains non-whitelisted differences: "
            f"{difference_paths}"
        )
    scientific_configuration_sha256 = canonical_json_sha256(
        current_manifest["scientific_configuration"]
    )

    manifest_record = by_name[RUN_MANIFEST]
    return {
        "reconciliation_version": RECONCILIATION_VERSION,
        "status": "PASS",
        "analysis_identity": ANALYSIS_IDENTITY,
        "created_from_existing_artifacts_only": True,
        "restricted_collaborator_material_used": False,
        "comparison_policy": {
            "primary_exact_files": list(PRIMARY_EXACT_FILES),
            "derived_numeric_tolerance_files": list(
                DERIVED_TOLERANCE_FILES
            ),
            "absolute_tolerance": ABSOLUTE_TOLERANCE,
            "relative_tolerance": RELATIVE_TOLERANCE,
            "manifest_operational_field_allowlist": list(
                MANIFEST_OPERATIONAL_FIELD_ALLOWLIST
            ),
            "manifest_derived_inventory_field_allowlist": list(
                MANIFEST_DERIVED_INVENTORY_FIELD_ALLOWLIST
            ),
        },
        "directories": {
            "current": CURRENT_DIRECTORY,
            "reproduction": REPRODUCTION_DIRECTORY,
            "shared_file_count": len(SHARED_FILES),
            "current_only_report_files": list(CURRENT_ONLY_REPORT_FILES),
        },
        "historical_manifest_hash_chain": _historical_manifest_chain(
            route_dir
        ),
        "scientific_configuration": {
            "canonical_sha256": scientific_configuration_sha256,
            "exact_between_current_and_reproduction": True,
        },
        "shared_files": records,
        "manifest_comparison": {
            "current_sha256": manifest_record["current_sha256"],
            "reproduction_sha256": manifest_record[
                "reproduction_sha256"
            ],
            "current_size_bytes": manifest_record[
                "current_size_bytes"
            ],
            "reproduction_size_bytes": manifest_record[
                "reproduction_size_bytes"
            ],
            "scientific_configuration_exact": True,
            "allowed_difference_paths": list(difference_paths),
            "unexpected_difference_paths": [],
            "operational_values": {
                field: {
                    "current": current_manifest[field],
                    "reproduction": reproduction_manifest[field],
                }
                for field in MANIFEST_OPERATIONAL_FIELD_ALLOWLIST
            },
        },
        "conclusion": {
            "predictions_byte_exact": by_name[
                "ood_predictions.csv"
            ]["byte_exact"],
            "split_audit_byte_exact": by_name[
                "ood_split_audit.csv"
            ]["byte_exact"],
            "primary_exact_file_count": len(PRIMARY_EXACT_FILES),
            "derived_tables_within_tolerance": len(
                DERIVED_TOLERANCE_FILES
            ),
            "numeric_mismatches_at_tolerance": sum(
                int(
                    by_name[name][
                        "numeric_mismatches_at_tolerance"
                    ]
                )
                for name in DERIVED_TOLERANCE_FILES
            ),
            "scientific_equivalence": True,
            "operational_rewrite_explanation": (
                OPERATIONAL_REWRITE_EXPLANATION
            ),
        },
    }


def validate_reconciliation_document(
    route_dir: Path,
    document_path: Path | None = None,
) -> dict[str, Any]:
    expected = build_reconciliation(route_dir)
    path = (
        document_path
        if document_path is not None
        else route_dir / "docs" / RECONCILIATION_JSON_NAME
    )
    observed = _read_json_object(path)
    if observed != expected:
        differences = _json_difference_paths(observed, expected)
        raise ValueError(
            "legacy OOD reconciliation JSON is stale or altered: "
            + ", ".join(differences[:20])
        )
    return observed


def validate_reconciliation_markdown(
    route_dir: Path,
    payload: dict[str, Any] | None = None,
) -> None:
    record = (
        payload
        if payload is not None
        else validate_reconciliation_document(route_dir)
    )
    path = route_dir / "docs" / RECONCILIATION_MD_NAME
    if not path.is_file() or path.is_symlink():
        raise ValueError("legacy OOD reconciliation Markdown is missing")
    text = path.read_text(encoding="utf-8")
    required_tokens: Iterable[str] = (
        RECONCILIATION_VERSION,
        str(ABSOLUTE_TOLERANCE),
        record["historical_manifest_hash_chain"][
            "frozen_manifest_sha256"
        ],
        record["manifest_comparison"]["current_sha256"],
        record["manifest_comparison"]["reproduction_sha256"],
        record["scientific_configuration"]["canonical_sha256"],
        "ood_predictions.csv",
        "ood_split_audit.csv",
        "restricted collaborator material: **not used**",
    )
    missing = [token for token in required_tokens if token not in text]
    if missing:
        raise ValueError(
            "legacy OOD reconciliation Markdown lacks bound facts: "
            + ", ".join(missing)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--route-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = validate_reconciliation_document(args.route_dir)
    validate_reconciliation_markdown(args.route_dir, payload)
    print(
        "LEGACY_OOD_RECONCILIATION: PASS "
        f"(primary_exact={len(PRIMARY_EXACT_FILES)}, "
        f"derived_tolerance={len(DERIVED_TOLERANCE_FILES)}, "
        f"atol={ABSOLUTE_TOLERANCE:g})"
    )
    print("GPU_REQUIRED: NO")


if __name__ == "__main__":
    main()
