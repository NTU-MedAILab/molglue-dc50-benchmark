#!/usr/bin/env python3
"""Validate and atomically merge corrected v3 permutation-ID shards.

Run this only after the primary and shard workers have stopped.  The script
fails closed on scientific-implementation, configuration, coverage, schema,
or overlapping-result mismatches.  Existing primary tables are copied to a
recoverable snapshot directory before replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


SCRIPT_DIR = Path(__file__).resolve().parent
ROUTE_DIR = SCRIPT_DIR.parent
DEFAULT_PRIMARY = (
    ROUTE_DIR / "reports" / "post_hoc_null_applicability_censoring_v3"
)
EXPECTED_PARENT_SHA256 = (
    "41b99a803b0e090d1bd55950983441eee0ce8edc8b099f86f56387e2f29f3650"
)
EXPECTED_WRAPPER_SHA256 = (
    "5aa68bfdc69083b5f30fa1eaedb938f2b3af4ceb86d774c294d672373319637e"
)
EXPECTED_MASTER_SHA256 = (
    "7dee28f45e3ddf8d6299622a8e494c50492b3b2614f39ce8b2fdd2a1be754ff8"
)
EXPECTED_LOCAL_PROTOCOL_SHA256 = (
    "84178cde0edfc0ff31be3d7e12135351f95a2578b9ac69b61d9a5d517caa0dc5"
)
EXPECTED_FLOAT32_CORRECTION_SHA256 = (
    "75a8b0e4f894cead9d72039eab9e6162a89d2a20c79afceda8a2aae1ffd78f3f"
)
EXPECTED_APPLICABILITY_CORRECTION_SHA256 = (
    "9c2d8f90fa546480a37b6e7546cbd9e9a740916325e5dec973f9fa7f8ef899ce"
)
TABLES: dict[str, tuple[list[str], int]] = {
    "permutation_model_metrics.csv": (
        ["permutation_id", "model_id"],
        2,
    ),
    "permutation_contrast_metrics.csv": (
        ["permutation_id", "contrast_id"],
        1,
    ),
    "permutation_fold_diagnostics.csv": (
        ["permutation_id", "repeat", "outer_fold"],
        25,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-dir", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument(
        "--shard-dir",
        type=Path,
        action="append",
        required=True,
        help="Repeat once for every completed shard directory.",
    )
    parser.add_argument("--n-permutations", type=int, default=100)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.merge_tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.merge_tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sorted_frame(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return frame.sort_values(keys, kind="stable").reset_index(drop=True)


def assert_complete_ids(
    frame: pd.DataFrame,
    keys: list[str],
    rows_per_id: int,
    expected_ids: set[int],
    label: str,
) -> None:
    if frame.duplicated(keys).any():
        raise RuntimeError(f"{label}: duplicate key rows")
    observed_ids = set(frame["permutation_id"].astype(int))
    if observed_ids != expected_ids:
        raise RuntimeError(
            f"{label}: IDs {sorted(observed_ids)} != {sorted(expected_ids)}"
        )
    counts = frame.groupby("permutation_id", sort=False).size()
    if not counts.eq(rows_per_id).all():
        raise RuntimeError(
            f"{label}: rows per ID differ from {rows_per_id}: "
            f"{counts[counts != rows_per_id].to_dict()}"
        )


def exact_overlap_check(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: list[str],
    label: str,
) -> list[int]:
    overlap = sorted(
        set(left["permutation_id"].astype(int))
        & set(right["permutation_id"].astype(int))
    )
    for permutation_id in overlap:
        left_local = sorted_frame(
            left[left["permutation_id"].astype(int) == permutation_id],
            keys,
        )
        right_local = sorted_frame(
            right[right["permutation_id"].astype(int) == permutation_id],
            keys,
        )
        try:
            assert_frame_equal(
                left_local,
                right_local,
                check_exact=True,
                check_dtype=True,
            )
        except AssertionError as error:
            raise RuntimeError(
                f"{label}: overlapping permutation {permutation_id} "
                "is not exactly reproducible"
            ) from error
    return overlap


def validate_shard(
    path: Path,
    expected_feature_manifest_sha256: str,
) -> tuple[int, int, dict[str, Any]]:
    for marker_name in [
        "SUPERSEDED_DO_NOT_USE.md",
        "ABORTED_DO_NOT_RESUME.md",
    ]:
        marker = path / marker_name
        if marker.exists():
            raise RuntimeError(f"{path}: refusing marked shard: {marker}")
    if not path.name.endswith("_v3"):
        raise RuntimeError(f"{path}: corrected shard directory must end in '_v3'")
    manifest_path = path / "shard_manifest.json"
    qa_path = path / "shard_qa.json"
    if not manifest_path.is_file() or not qa_path.is_file():
        raise FileNotFoundError(f"Missing shard manifest or QA in {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    if manifest.get("scientific_implementation_sha256") != EXPECTED_PARENT_SHA256:
        raise RuntimeError(f"{path}: frozen parent SHA-256 mismatch")
    if manifest.get("master_protocol_sha256") != EXPECTED_MASTER_SHA256:
        raise RuntimeError(f"{path}: master protocol SHA-256 mismatch")
    if (
        manifest.get("local_protocol_sha256")
        != EXPECTED_LOCAL_PROTOCOL_SHA256
    ):
        raise RuntimeError(f"{path}: v3 local protocol SHA-256 mismatch")
    if (
        manifest.get("float32_correction_sha256")
        != EXPECTED_FLOAT32_CORRECTION_SHA256
    ):
        raise RuntimeError(f"{path}: float32 correction SHA-256 mismatch")
    if (
        manifest.get("applicability_correction_sha256")
        != EXPECTED_APPLICABILITY_CORRECTION_SHA256
    ):
        raise RuntimeError(
            f"{path}: applicability correction SHA-256 mismatch"
        )
    if manifest.get("protocol_version") != (
        "post_hoc_null_applicability_censoring_v3.0"
    ):
        raise RuntimeError(f"{path}: protocol version is not corrected v3")
    config = manifest.get("config", {})
    if (
        config.get("n_estimators") != 600
        or config.get("n_jobs") != 4
        or config.get("max_repeats") != 5
        or config.get("seed") != 260531
        or config.get("training_response_dtype") != "float32"
    ):
        raise RuntimeError(f"{path}: formal scientific configuration mismatch")
    if qa.get("overall_status") != "PASS":
        raise RuntimeError(f"{path}: shard QA is not PASS")
    feature_manifest_path = path / "feature_manifest.json"
    if (
        not feature_manifest_path.is_file()
        or sha256_file(feature_manifest_path)
        != expected_feature_manifest_sha256
    ):
        raise RuntimeError(f"{path}: feature manifest does not match primary")
    start_id = int(config["start_id"])
    end_id = int(config["end_id"])
    expected_ids = set(range(start_id, end_id + 1))
    for filename, (keys, rows_per_id) in TABLES.items():
        frame = pd.read_csv(path / filename)
        assert_complete_ids(
            frame,
            keys,
            rows_per_id,
            expected_ids,
            f"{path.name}/{filename}",
        )
    return start_id, end_id, manifest


def main() -> None:
    args = parse_args()
    primary = args.primary_dir.resolve()
    for marker_name in [
        "SUPERSEDED_DO_NOT_USE.md",
        "ABORTED_DO_NOT_RESUME.md",
    ]:
        marker = primary / marker_name
        if marker.exists():
            raise RuntimeError(f"Refusing marked primary: {marker}")
    if primary != DEFAULT_PRIMARY.resolve():
        raise RuntimeError(
            "This version is fail-closed to the frozen formal primary directory"
        )
    if args.n_permutations != 100:
        raise RuntimeError("Formal merge requires exactly 100 permutations")
    wrapper = SCRIPT_DIR / "run_post_hoc_permutation_shard_v3.py"
    parent = SCRIPT_DIR / "run_post_hoc_null_applicability_censoring_v3.py"
    if sha256_file(wrapper) != EXPECTED_WRAPPER_SHA256:
        raise RuntimeError("Permutation shard wrapper SHA-256 mismatch")
    if sha256_file(parent) != EXPECTED_PARENT_SHA256:
        raise RuntimeError("Frozen scientific implementation SHA-256 mismatch")
    primary_manifest_path = primary / "run_manifest.json"
    if not primary_manifest_path.is_file():
        raise FileNotFoundError(primary_manifest_path)
    primary_manifest = json.loads(
        primary_manifest_path.read_text(encoding="utf-8")
    )
    primary_config = primary_manifest.get("config", {})
    if (
        primary_manifest.get("protocol_version")
        != "post_hoc_null_applicability_censoring_v3.0"
        or primary_manifest.get("master_protocol_sha256")
        != EXPECTED_MASTER_SHA256
        or primary_manifest.get("local_protocol_sha256")
        != EXPECTED_LOCAL_PROTOCOL_SHA256
        or primary_manifest.get("float32_correction_sha256")
        != EXPECTED_FLOAT32_CORRECTION_SHA256
        or primary_manifest.get("applicability_correction_sha256")
        != EXPECTED_APPLICABILITY_CORRECTION_SHA256
        or primary_config.get("n_estimators") != 600
        or primary_config.get("n_jobs") != 4
        or primary_config.get("bootstrap_replicates") != 10_000
        or primary_config.get("max_repeats") != 5
        or primary_config.get("seed") != 260531
        or primary_config.get("training_response_dtype") != "float32"
    ):
        raise RuntimeError("Formal v3 primary manifest is not eligible")

    shard_dirs = [path.resolve() for path in args.shard_dir]
    if len(set(shard_dirs)) != len(shard_dirs):
        raise RuntimeError("Duplicate shard directory argument")
    primary_feature_manifest = primary / "feature_manifest.json"
    if not primary_feature_manifest.is_file():
        raise FileNotFoundError(primary_feature_manifest)
    feature_manifest_sha256 = sha256_file(primary_feature_manifest)
    shard_ranges: list[dict[str, Any]] = []
    for shard_dir in shard_dirs:
        start_id, end_id, _ = validate_shard(
            shard_dir,
            feature_manifest_sha256,
        )
        shard_ranges.append(
            {
                "path": str(shard_dir.relative_to(ROUTE_DIR)),
                "start_id": start_id,
                "end_id": end_id,
                "manifest_sha256": sha256_file(
                    shard_dir / "shard_manifest.json"
                ),
                "qa_sha256": sha256_file(shard_dir / "shard_qa.json"),
                "feature_manifest_sha256": sha256_file(
                    shard_dir / "feature_manifest.json"
                ),
            }
        )

    original_hashes = {
        filename: sha256_file(primary / filename) for filename in TABLES
    }
    merged_tables: dict[str, pd.DataFrame] = {}
    overlap_report: dict[str, list[int]] = {}
    expected_all = set(range(1, args.n_permutations + 1))
    for filename, (keys, rows_per_id) in TABLES.items():
        primary_frame = pd.read_csv(primary / filename)
        combined = primary_frame.copy()
        all_overlaps: list[int] = []
        for shard_dir in shard_dirs:
            shard_frame = pd.read_csv(shard_dir / filename)
            all_overlaps.extend(
                exact_overlap_check(
                    combined,
                    shard_frame,
                    keys,
                    f"{filename}/{shard_dir.name}",
                )
            )
            combined = pd.concat([combined, shard_frame], ignore_index=True)
            combined = combined.drop_duplicates(keys, keep="first")
        combined = sorted_frame(combined, keys)
        assert_complete_ids(
            combined,
            keys,
            rows_per_id,
            expected_all,
            f"merged/{filename}",
        )
        merged_tables[filename] = combined
        overlap_report[filename] = sorted(set(all_overlaps))

    diagnostics = merged_tables["permutation_fold_diagnostics.csv"]
    if not diagnostics["within_stratum_multiset_preserved"].astype(bool).all():
        raise RuntimeError("Merged diagnostic table contains a failed multiset check")
    metric_columns = ["spearman", "rmse", "mae"]
    if not np.isfinite(
        merged_tables["permutation_model_metrics.csv"][
            metric_columns
        ].to_numpy(dtype=float)
    ).all():
        raise RuntimeError("Merged model table contains non-finite metrics")
    delta_columns = ["delta_spearman", "delta_rmse", "delta_mae"]
    if not np.isfinite(
        merged_tables["permutation_contrast_metrics.csv"][
            delta_columns
        ].to_numpy(dtype=float)
    ).all():
        raise RuntimeError("Merged contrast table contains non-finite metrics")

    snapshot = primary / "premerge_primary_snapshot_v3"
    if snapshot.exists():
        raise FileExistsError(
            f"Recoverable snapshot already exists; refusing overwrite: {snapshot}"
        )
    snapshot.mkdir()
    for filename in TABLES:
        shutil.copy2(primary / filename, snapshot / filename)

    for filename, frame in merged_tables.items():
        atomic_write_csv(frame, primary / filename)

    manifest = {
        "status": "PASS",
        "role": "execution_only_deterministic_shard_merge",
        "n_permutations": args.n_permutations,
        "scientific_implementation_sha256": EXPECTED_PARENT_SHA256,
        "execution_wrapper_sha256": EXPECTED_WRAPPER_SHA256,
        "master_protocol_sha256": EXPECTED_MASTER_SHA256,
        "local_protocol_sha256": EXPECTED_LOCAL_PROTOCOL_SHA256,
        "float32_correction_sha256": EXPECTED_FLOAT32_CORRECTION_SHA256,
        "applicability_correction_sha256": (
            EXPECTED_APPLICABILITY_CORRECTION_SHA256
        ),
        "protocol_version": "post_hoc_null_applicability_censoring_v3.0",
        "training_response_dtype": "float32",
        "feature_manifest_sha256": feature_manifest_sha256,
        "shards": shard_ranges,
        "primary_premerge_sha256": original_hashes,
        "overlapping_ids_exactly_matched": overlap_report,
        "recoverable_snapshot": str(snapshot.relative_to(ROUTE_DIR)),
        "merged_sha256": {
            filename: sha256_file(primary / filename) for filename in TABLES
        },
        "scientific_change": False,
        "next_step": (
            "Run the frozen parent script with its original formal arguments "
            "and --resume to generate inference, QA, manifests and hashes."
        ),
    }
    atomic_write_json(
        manifest,
        primary / "permutation_shard_merge_manifest.json",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
