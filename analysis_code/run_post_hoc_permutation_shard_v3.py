#!/usr/bin/env python3
"""Compute a deterministic subset of the corrected v3 permutations.

This is an execution-only accelerator for
``run_post_hoc_null_applicability_censoring_v3.py``.  It imports the frozen
scientific implementation and changes only the order/location in which
independent permutation IDs are evaluated.  Results are written to an
isolated directory and must be identity-checked before merging.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_post_hoc_null_applicability_censoring_v3 as parent


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "reports" / "post_hoc_null_permutation_shard_v3"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-id", type=int, required=True)
    parser.add_argument("--end-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--max-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=parent.DEFAULT_SEED)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.start_id < 1 or args.end_id < args.start_id:
        parser.error("require 1 <= start-id <= end-id")
    if args.n_estimators != 600:
        parser.error("formal shard requires exactly 600 estimators")
    if not 1 <= args.n_jobs <= 4:
        parser.error("n-jobs must be between 1 and 4")
    if args.max_repeats != 5:
        parser.error("formal shard requires exactly 5 repeats")
    if args.seed != parent.DEFAULT_SEED:
        parser.error(f"formal shard requires seed {parent.DEFAULT_SEED}")
    return args


def prepare_output_dir(args: argparse.Namespace) -> None:
    path = args.output_dir
    for marker_name in [
        "SUPERSEDED_DO_NOT_USE.md",
        "ABORTED_DO_NOT_RESUME.md",
    ]:
        marker = path / marker_name
        if marker.exists():
            raise RuntimeError(f"Refusing marked output directory: {marker}")
    if not path.name.endswith("_v3"):
        raise ValueError(
            "Corrected shard outputs require a directory name ending in "
            "'_v3'; v1/v2 directories cannot be resumed or overwritten."
        )
    if path.exists() and any(path.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Output directory is non-empty: {path}; pass --resume"
        )
    if path.exists() and any(path.iterdir()) and args.resume:
        manifest_path = path / "shard_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(
                "Fail-closed v3 shard resume requires shard_manifest.json"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = manifest.get("config", {})
        expected = {
            "start_id": args.start_id,
            "end_id": args.end_id,
            "n_estimators": args.n_estimators,
            "n_jobs": args.n_jobs,
            "max_repeats": args.max_repeats,
            "seed": args.seed,
            "training_response_dtype": "float32",
        }
        if manifest.get("protocol_version") != parent.PROTOCOL_VERSION:
            raise RuntimeError("Existing shard protocol version is not v3")
        if (
            manifest.get("local_protocol_sha256")
            != parent.LOCAL_PROTOCOL_SHA256
            or manifest.get("float32_correction_sha256")
            != parent.FLOAT32_CORRECTION_NOTE_SHA256
            or manifest.get("applicability_correction_sha256")
            != parent.APPLICABILITY_CORRECTION_NOTE_SHA256
        ):
            raise RuntimeError("Existing shard correction identity mismatch")
        mismatches = {
            key: (config.get(key), value)
            for key, value in expected.items()
            if config.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"Existing shard scientific configuration mismatch: {mismatches}"
            )
    path.mkdir(parents=True, exist_ok=True)


def write_manifest(args: argparse.Namespace) -> None:
    payload = {
        "role": "execution_only_permutation_shard",
        "scientific_implementation": str(Path(parent.__file__).resolve()),
        "scientific_implementation_sha256": parent.sha256_file(
            Path(parent.__file__).resolve()
        ),
        "master_protocol_sha256": parent.MASTER_PROTOCOL_SHA256,
        "local_protocol_sha256": parent.LOCAL_PROTOCOL_SHA256,
        "float32_correction_sha256": (
            parent.FLOAT32_CORRECTION_NOTE_SHA256
        ),
        "applicability_correction_sha256": (
            parent.APPLICABILITY_CORRECTION_NOTE_SHA256
        ),
        "protocol_version": parent.PROTOCOL_VERSION,
        "config": {
            "start_id": args.start_id,
            "end_id": args.end_id,
            "n_estimators": args.n_estimators,
            "n_jobs": args.n_jobs,
            "max_repeats": args.max_repeats,
            "seed": args.seed,
            "models": parent.MODEL_IDS,
            "training_response_dtype": "float32",
            "permutation_strata": (
                "source_database × target_protein within each outer fit fold"
            ),
        },
        "scientific_change": False,
        "note": (
            "Permutation IDs, seeds, folds, features, selected "
            "hyperparameters, estimators and fit function are imported "
            "unchanged from the frozen parent implementation."
        ),
    }
    parent.atomic_write_json(args.output_dir / "shard_manifest.json", payload)


def main() -> None:
    args = parse_args()
    prepare_output_dir(args)
    parent.validate_frozen_identities()
    write_manifest(args)

    rows, _ = parent.core.load_rows(parent.CORE_DATA)
    if len(rows) != 1560:
        raise ValueError(f"Expected 1,560 core rows; observed {len(rows)}")
    _, _, splits, _ = parent.load_internal_design(rows, args.max_repeats)
    cached_splits, morgan, _, feature_manifest = parent.cache_fold_features(
        rows, splits
    )
    parent.atomic_write_json(
        args.output_dir / "feature_manifest.json", feature_manifest
    )
    y = rows["pDC50"].to_numpy(dtype=np.float32)

    models, contrasts, diagnostics, complete = parent.load_resume_tables(
        args.output_dir
    )
    requested = range(args.start_id, args.end_id + 1)
    missing = [permutation_id for permutation_id in requested if permutation_id not in complete]
    for position, permutation_id in enumerate(missing, start=1):
        started = time.time()
        model_add, contrast_add, diagnostic_add = parent.run_one_permutation(
            permutation_id=permutation_id,
            rows=rows,
            y=y,
            morgan=morgan,
            cached_splits=cached_splits,
            n_repeats=args.max_repeats,
            n_estimators=args.n_estimators,
            n_jobs=args.n_jobs,
            seed=args.seed,
        )
        models = parent.append_and_checkpoint(
            models,
            model_add,
            args.output_dir / "permutation_model_metrics.csv",
            ["permutation_id", "model_id"],
        )
        contrasts = parent.append_and_checkpoint(
            contrasts,
            contrast_add,
            args.output_dir / "permutation_contrast_metrics.csv",
            ["permutation_id", "contrast_id"],
        )
        diagnostics = parent.append_and_checkpoint(
            diagnostics,
            diagnostic_add,
            args.output_dir / "permutation_fold_diagnostics.csv",
            ["permutation_id", "repeat", "outer_fold"],
        )
        print(
            f"[SHARD] {position}/{len(missing)} id={permutation_id} "
            f"complete in {time.time() - started:.1f}s",
            flush=True,
        )

    expected = set(requested)
    for name, frame, expected_rows in [
        ("model", models, len(expected) * len(parent.MODEL_IDS)),
        ("contrast", contrasts, len(expected)),
        ("diagnostic", diagnostics, len(expected) * args.max_repeats * 5),
    ]:
        observed_ids = set(frame["permutation_id"].astype(int))
        if observed_ids != expected or len(frame) != expected_rows:
            raise RuntimeError(
                f"Incomplete {name} shard: ids={len(observed_ids)} "
                f"rows={len(frame)}, expected ids={len(expected)} "
                f"rows={expected_rows}"
            )

    qa = {
        "overall_status": "PASS",
        "n_permutation_ids": len(expected),
        "start_id": args.start_id,
        "end_id": args.end_id,
        "model_rows": len(models),
        "contrast_rows": len(contrasts),
        "diagnostic_rows": len(diagnostics),
        "all_metrics_finite": bool(
            np.isfinite(
                models[["spearman", "rmse", "mae"]].to_numpy(dtype=float)
            ).all()
            and np.isfinite(
                contrasts[
                    ["delta_spearman", "delta_rmse", "delta_mae"]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "all_multisets_preserved": bool(
            diagnostics["within_stratum_multiset_preserved"].astype(bool).all()
        ),
    }
    if not qa["all_metrics_finite"] or not qa["all_multisets_preserved"]:
        qa["overall_status"] = "FAIL"
    parent.atomic_write_json(args.output_dir / "shard_qa.json", qa)
    parent.write_artifact_hashes(args.output_dir)
    print(json.dumps(qa, ensure_ascii=False), flush=True)
    if qa["overall_status"] != "PASS":
        raise RuntimeError("Shard QA failed")


if __name__ == "__main__":
    main()
