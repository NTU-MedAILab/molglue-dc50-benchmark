#!/usr/bin/env python
"""CPU-only source- and target-OOD confirmatory DC50 benchmark.

Every OOD test domain is held out exactly once. Training also removes related
composite-domain rows and every occurrence of a test canonical SMILES. Model
selection is performed only inside the resulting training set by scaffold-
disjoint inner cross-validation.

This extension imports deterministic feature builders and core model routines
from ``run_confirmatory_cpu_v1.py`` but has independent split, checkpoint,
summary and bootstrap implementations because OOD test domains are not
repeated cross-validation folds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import joblib
import numpy as np
import pandas as pd
import scipy
from rdkit import rdBase
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import ExtraTreesRegressor

import run_confirmatory_cpu_v1 as core


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
CORE_SCRIPT = SCRIPT_DIR / "run_confirmatory_cpu_v1.py"
CORE_PROTOCOL_DOCUMENT = PROJECT_DIR / "docs" / "confirmatory_protocol_v1.md"
OOD_PROTOCOL_DOCUMENT = (
    PROJECT_DIR / "docs" / "confirmatory_ood_protocol_v1.md"
)
FROZEN_IDENTITY_FILE = (
    PROJECT_DIR / "docs" / "confirmatory_ood_cpu_v1_frozen_identity.json"
)
DATA_FILE = core.DATA_FILE
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "reports" / "confirmatory_ood_cpu_v1"

PROTOCOL_VERSION = "confirmatory_ood_cpu_v1.0"
FROZEN_IDENTITY_VERSION = "confirmatory_ood_cpu_v1.identity.v1"
CHECKPOINT_INVENTORY_VERSION = "confirmatory_ood_cpu_v1.checkpoint.v1"
DEFAULT_SEED = core.DEFAULT_SEED
FROZEN_DATA_SHA256 = core.FROZEN_DATA_SHA256
PRIMARY_METRIC = core.PRIMARY_METRIC
REFERENCE_MODEL = core.REFERENCE_MODEL
PROTOCOL_ORDER = ("source_ood", "target_ood")

SOURCE_OOD_GROUPS = (
    "MGTbind",
    "MGDB",
    "MolGlueDB",
    "TPDdb",
)
TARGET_OOD_GROUPS = (
    "VAV1",
    "CSNK1A1",
    "GSPT1",
    "WIZ",
    "CDK2",
    "CCNK+CDK12",
    "IKZF2",
    "IKZF1",
)
GROUPS_BY_PROTOCOL = {
    "source_ood": SOURCE_OOD_GROUPS,
    "target_ood": TARGET_OOD_GROUPS,
}
PROTOCOL_SEED_OFFSET = {
    "source_ood": 1_000_000,
    "target_ood": 2_000_000,
}

CONTEXT_EXTRA_TREES = "context_extra_trees"
LABEL_SHUFFLE_MODEL = "full_context_label_shuffle"
MODEL_ORDER = (
    "global_mean",
    "hierarchical_context_mean",
    "context_ridge",
    CONTEXT_EXTRA_TREES,
    "morgan_knn",
    "chemistry_extra_trees",
    "full_context_extra_trees",
    LABEL_SHUFFLE_MODEL,
)
TUNED_MODEL_ORDER = tuple(
    model_id for model_id in MODEL_ORDER if model_id != LABEL_SHUFFLE_MODEL
)

PRESPECIFIED_CONTRASTS = {
    "full_vs_context_ridge": (
        "full_context_extra_trees",
        "context_ridge",
    ),
    "full_vs_context_extra_trees_matched": (
        "full_context_extra_trees",
        CONTEXT_EXTRA_TREES,
    ),
    "chemistry_extra_trees_vs_morgan_knn": (
        "chemistry_extra_trees",
        "morgan_knn",
    ),
    "full_vs_chemistry_extra_trees": (
        "full_context_extra_trees",
        "chemistry_extra_trees",
    ),
    "full_vs_label_shuffle_sanity_control": (
        "full_context_extra_trees",
        LABEL_SHUFFLE_MODEL,
    ),
}

# Counts are checked after the sanitization performed by core.load_rows.
EXPECTED_SPLIT_COUNTS: dict[tuple[str, str], dict[str, int]] = {
    ("source_ood", "MGTbind"): {
        "n_test_rows": 755,
        "n_test_unique_smiles": 664,
        "n_test_scaffolds": 394,
        "n_related_rows_excluded": 33,
        "n_candidate_train_rows": 772,
        "n_compound_overlap_rows_excluded": 244,
        "n_train_rows": 528,
        "n_train_unique_smiles": 468,
        "n_train_scaffolds": 280,
        "n_test_rows_scaffold_seen": 16,
        "n_test_scaffolds_seen": 7,
    },
    ("source_ood", "MGDB"): {
        "n_test_rows": 357,
        "n_test_unique_smiles": 340,
        "n_test_scaffolds": 205,
        "n_related_rows_excluded": 8,
        "n_candidate_train_rows": 1195,
        "n_compound_overlap_rows_excluded": 76,
        "n_train_rows": 1119,
        "n_train_unique_smiles": 797,
        "n_train_scaffolds": 471,
        "n_test_rows_scaffold_seen": 24,
        "n_test_scaffolds_seen": 9,
    },
    ("source_ood", "MolGlueDB"): {
        "n_test_rows": 289,
        "n_test_unique_smiles": 247,
        "n_test_scaffolds": 144,
        "n_related_rows_excluded": 39,
        "n_candidate_train_rows": 1232,
        "n_compound_overlap_rows_excluded": 210,
        "n_train_rows": 1022,
        "n_train_unique_smiles": 885,
        "n_train_scaffolds": 539,
        "n_test_rows_scaffold_seen": 45,
        "n_test_scaffolds_seen": 16,
    },
    ("source_ood", "TPDdb"): {
        "n_test_rows": 119,
        "n_test_unique_smiles": 118,
        "n_test_scaffolds": 58,
        "n_related_rows_excluded": 0,
        "n_candidate_train_rows": 1441,
        "n_compound_overlap_rows_excluded": 81,
        "n_train_rows": 1360,
        "n_train_unique_smiles": 1019,
        "n_train_scaffolds": 621,
        "n_test_rows_scaffold_seen": 46,
        "n_test_scaffolds_seen": 12,
    },
    ("target_ood", "VAV1"): {
        "n_test_rows": 455,
        "n_test_unique_smiles": 404,
        "n_test_scaffolds": 230,
        "n_related_rows_excluded": 0,
        "n_candidate_train_rows": 1105,
        "n_compound_overlap_rows_excluded": 0,
        "n_train_rows": 1105,
        "n_train_unique_smiles": 733,
        "n_train_scaffolds": 437,
        "n_test_rows_scaffold_seen": 0,
        "n_test_scaffolds_seen": 0,
    },
    ("target_ood", "CSNK1A1"): {
        "n_test_rows": 159,
        "n_test_unique_smiles": 139,
        "n_test_scaffolds": 68,
        "n_related_rows_excluded": 30,
        "n_candidate_train_rows": 1371,
        "n_compound_overlap_rows_excluded": 101,
        "n_train_rows": 1270,
        "n_train_unique_smiles": 998,
        "n_train_scaffolds": 603,
        "n_test_rows_scaffold_seen": 32,
        "n_test_scaffolds_seen": 4,
    },
    ("target_ood", "GSPT1"): {
        "n_test_rows": 154,
        "n_test_unique_smiles": 106,
        "n_test_scaffolds": 74,
        "n_related_rows_excluded": 1,
        "n_candidate_train_rows": 1405,
        "n_compound_overlap_rows_excluded": 48,
        "n_train_rows": 1357,
        "n_train_unique_smiles": 1031,
        "n_train_scaffolds": 596,
        "n_test_rows_scaffold_seen": 13,
        "n_test_scaffolds_seen": 3,
    },
    ("target_ood", "WIZ"): {
        "n_test_rows": 150,
        "n_test_unique_smiles": 123,
        "n_test_scaffolds": 47,
        "n_related_rows_excluded": 0,
        "n_candidate_train_rows": 1410,
        "n_compound_overlap_rows_excluded": 1,
        "n_train_rows": 1409,
        "n_train_unique_smiles": 1014,
        "n_train_scaffolds": 620,
        "n_test_rows_scaffold_seen": 0,
        "n_test_scaffolds_seen": 0,
    },
    ("target_ood", "CDK2"): {
        "n_test_rows": 145,
        "n_test_unique_smiles": 145,
        "n_test_scaffolds": 102,
        "n_related_rows_excluded": 0,
        "n_candidate_train_rows": 1415,
        "n_compound_overlap_rows_excluded": 0,
        "n_train_rows": 1415,
        "n_train_unique_smiles": 992,
        "n_train_scaffolds": 565,
        "n_test_rows_scaffold_seen": 0,
        "n_test_scaffolds_seen": 0,
    },
    ("target_ood", "CCNK+CDK12"): {
        "n_test_rows": 86,
        "n_test_unique_smiles": 73,
        "n_test_scaffolds": 52,
        "n_related_rows_excluded": 2,
        "n_candidate_train_rows": 1472,
        "n_compound_overlap_rows_excluded": 0,
        "n_train_rows": 1472,
        "n_train_unique_smiles": 1063,
        "n_train_scaffolds": 614,
        "n_test_rows_scaffold_seen": 0,
        "n_test_scaffolds_seen": 0,
    },
    ("target_ood", "IKZF2"): {
        "n_test_rows": 76,
        "n_test_unique_smiles": 65,
        "n_test_scaffolds": 32,
        "n_related_rows_excluded": 21,
        "n_candidate_train_rows": 1463,
        "n_compound_overlap_rows_excluded": 94,
        "n_train_rows": 1369,
        "n_train_unique_smiles": 1072,
        "n_train_scaffolds": 635,
        "n_test_rows_scaffold_seen": 0,
        "n_test_scaffolds_seen": 0,
    },
    ("target_ood", "IKZF1"): {
        "n_test_rows": 65,
        "n_test_unique_smiles": 60,
        "n_test_scaffolds": 34,
        "n_related_rows_excluded": 11,
        "n_candidate_train_rows": 1484,
        "n_compound_overlap_rows_excluded": 92,
        "n_train_rows": 1392,
        "n_train_unique_smiles": 1077,
        "n_train_scaffolds": 643,
        "n_test_rows_scaffold_seen": 34,
        "n_test_scaffolds_seen": 10,
    },
}

PROGRESS_FILES = {
    "predictions": "ood_predictions.csv",
    "domain_metrics": "ood_domain_metrics.csv",
    "tuning": "ood_inner_tuning_metrics.csv",
    "selected": "ood_selected_hyperparameters.csv",
    "inner_audit": "ood_inner_split_audit.csv",
}
CHECKPOINT_INVENTORY_FILE = "ood_checkpoint_inventory.json"
HYPERPARAMETER_FIELDS = (
    "smoothing",
    "alpha",
    "min_samples_leaf",
    "max_features",
    "k",
    "power",
)


@dataclass(frozen=True)
class OODSplit:
    protocol: str
    heldout_group: str
    group_position: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    split_seed: int
    audit: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=list(PROTOCOL_ORDER),
        default=list(PROTOCOL_ORDER),
        help="OOD protocols to run.",
    )
    parser.add_argument("--data-file", type=Path, default=DATA_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help=(
            "Smoke-test helper: retain the first N frozen groups within each "
            "requested protocol and disable formal completeness status."
        ),
    )
    parser.add_argument(
        "--skip-label-shuffle",
        action="store_true",
        help="Skip the label-shuffle diagnostic.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only from a complete hash-matched scientific configuration.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite known OOD artifacts in an existing output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and verify data, features and frozen splits without fitting.",
    )
    args = parser.parse_args()
    if args.inner_folds < 2:
        parser.error("inner-folds must be >= 2")
    if args.n_estimators < 10:
        parser.error("n-estimators must be >= 10")
    if args.n_jobs < 1:
        parser.error("n-jobs must be >= 1")
    if args.bootstrap_replicates < 0:
        parser.error("bootstrap-replicates must be >= 0")
    if args.max_groups is not None and args.max_groups < 1:
        parser.error("max-groups must be >= 1")
    if len(args.protocols) != len(set(args.protocols)):
        parser.error("protocols must not contain duplicates")
    if args.resume and args.overwrite:
        parser.error("resume and overwrite are mutually exclusive")
    args.protocols = [
        protocol for protocol in PROTOCOL_ORDER if protocol in args.protocols
    ]
    return args


def tokenize_domain(value: Any) -> set[str]:
    return {
        token.strip()
        for token in str(value).split("+")
        if token.strip()
    }


def expected_counts_json() -> dict[str, dict[str, int]]:
    return {
        f"{protocol}::{group}": dict(counts)
        for (protocol, group), counts in EXPECTED_SPLIT_COUNTS.items()
    }


def canonical_json_sha256(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=core.json_ready,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def sequence_sha256(values: Iterable[Any]) -> str:
    return canonical_json_sha256(list(values))


def formal_argument_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "protocols": list(args.protocols),
        "inner_folds": int(args.inner_folds),
        "n_estimators": int(args.n_estimators),
        "n_jobs": int(args.n_jobs),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "seed": int(args.seed),
        "max_groups": args.max_groups,
        "skip_label_shuffle": bool(args.skip_label_shuffle),
        "dry_run": bool(args.dry_run),
    }


def frozen_formal_arguments() -> dict[str, Any]:
    return {
        "protocols": list(PROTOCOL_ORDER),
        "inner_folds": 4,
        "n_estimators": 600,
        "n_jobs": 10,
        "bootstrap_replicates": 10_000,
        "seed": DEFAULT_SEED,
        "max_groups": None,
        "skip_label_shuffle": False,
        "dry_run": False,
    }


def observed_frozen_identity(
    args: argparse.Namespace,
    *,
    data_sha256: str,
    script_sha256: str,
    protocol_sha256: str,
    core_script_sha256: str,
    core_protocol_sha256: str,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "file_sha256": {
            "data": data_sha256,
            "ood_script": script_sha256,
            "ood_protocol": protocol_sha256,
            "core_script": core_script_sha256,
            "core_protocol": core_protocol_sha256,
        },
        "formal_arguments": formal_argument_snapshot(args),
    }


def load_frozen_identity(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Frozen OOD identity must be a JSON object")
    if payload.get("identity_version") != FROZEN_IDENTITY_VERSION:
        raise ValueError(
            "Unexpected frozen OOD identity version: "
            f"{payload.get('identity_version')!r}"
        )
    frozen_identity = payload.get("frozen_identity")
    if not isinstance(frozen_identity, dict):
        raise ValueError("Frozen OOD identity has no frozen_identity object")
    required_identity_keys = {
        "protocol_version",
        "file_sha256",
        "formal_arguments",
    }
    if set(frozen_identity) != required_identity_keys:
        raise ValueError(
            "Frozen OOD identity keys changed: "
            f"{sorted(frozen_identity)}"
        )
    if set(frozen_identity["file_sha256"]) != {
        "data",
        "ood_script",
        "ood_protocol",
        "core_script",
        "core_protocol",
    }:
        raise ValueError("Frozen OOD file-hash inventory is incomplete")
    if frozen_identity["formal_arguments"] != frozen_formal_arguments():
        raise ValueError("Frozen OOD formal arguments changed inside lock file")
    return frozen_identity, core.sha256_file(path)


def formal_cli_requested(args: argparse.Namespace) -> bool:
    return formal_argument_snapshot(args) == frozen_formal_arguments()


def frozen_identity_mismatches(
    observed: dict[str, Any],
    frozen: dict[str, Any],
) -> list[str]:
    mismatches: list[str] = []
    if observed.get("protocol_version") != frozen.get("protocol_version"):
        mismatches.append("protocol_version")
    observed_hashes = observed.get("file_sha256", {})
    frozen_hashes = frozen.get("file_sha256", {})
    for key in (
        "data",
        "ood_script",
        "ood_protocol",
        "core_script",
        "core_protocol",
    ):
        if observed_hashes.get(key) != frozen_hashes.get(key):
            mismatches.append(f"file_sha256.{key}")
    if observed.get("formal_arguments") != frozen.get("formal_arguments"):
        mismatches.append("formal_arguments")
    return mismatches


def frozen_protocol_match(
    observed: dict[str, Any],
    frozen: dict[str, Any],
) -> bool:
    return not frozen_identity_mismatches(observed, frozen)


def value_is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def values_match(
    observed: Any,
    expected: Any,
    *,
    atol: float = 1e-10,
    rtol: float = 1e-10,
) -> bool:
    if value_is_missing(expected):
        return value_is_missing(observed)
    if value_is_missing(observed):
        return False
    if isinstance(expected, (bool, np.bool_)):
        if isinstance(observed, str):
            return observed.strip().lower() == str(bool(expected)).lower()
        return bool(observed) is bool(expected)
    if isinstance(expected, (int, float, np.integer, np.floating)):
        try:
            observed_float = float(observed)
            expected_float = float(expected)
        except (TypeError, ValueError):
            return False
        if math.isnan(expected_float):
            return math.isnan(observed_float)
        return bool(
            math.isclose(
                observed_float,
                expected_float,
                rel_tol=rtol,
                abs_tol=atol,
            )
        )
    return str(observed) == str(expected)


def file_inventory_entry(path: Path) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "filename": path.name,
        "sha256": core.sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        entry.update(
            {
                "kind": "csv",
                "n_rows": int(len(frame)),
                "n_columns": int(len(frame.columns)),
                "columns": frame.columns.astype(str).tolist(),
            }
        )
    else:
        entry["kind"] = path.suffix.lower().lstrip(".") or "file"
    return entry


def build_file_inventory(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        inventory[path.name] = file_inventory_entry(path)
    return inventory


def verify_file_inventory(
    output_dir: Path,
    inventory: dict[str, Any],
    expected_filenames: set[str],
) -> None:
    if set(inventory) != expected_filenames:
        raise RuntimeError(
            "Checkpoint inventory filenames changed: "
            f"observed={sorted(inventory)}, "
            f"expected={sorted(expected_filenames)}"
        )
    for filename in sorted(expected_filenames):
        path = output_dir / filename
        if not path.is_file():
            raise RuntimeError(f"Checkpoint file is missing: {path}")
        observed = file_inventory_entry(path)
        if observed != inventory[filename]:
            raise RuntimeError(
                "Checkpoint inventory mismatch for "
                f"{filename}: observed={observed}, "
                f"expected={inventory[filename]}"
            )


def verify_checkpoint_inventory(
    output_dir: Path,
    configuration_sha256: str,
) -> None:
    inventory_path = output_dir / CHECKPOINT_INVENTORY_FILE
    progress_paths = [
        output_dir / filename for filename in PROGRESS_FILES.values()
    ]
    existing_progress = [path for path in progress_paths if path.is_file()]
    if not existing_progress and not inventory_path.exists():
        return
    if len(existing_progress) != len(progress_paths):
        raise RuntimeError(
            "Resume refused: checkpoint progress files are incomplete"
        )
    if not inventory_path.is_file():
        raise RuntimeError(
            "Resume refused: checkpoint inventory is missing"
        )
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    if payload.get("inventory_version") != CHECKPOINT_INVENTORY_VERSION:
        raise RuntimeError("Resume refused: checkpoint inventory version changed")
    if payload.get("scientific_configuration_sha256") != configuration_sha256:
        raise RuntimeError(
            "Resume refused: checkpoint scientific configuration changed"
        )
    files = payload.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("Resume refused: checkpoint file inventory is invalid")
    verify_file_inventory(
        output_dir,
        files,
        set(PROGRESS_FILES.values()),
    )


def model_param_grid(model_id: str) -> list[dict[str, Any]]:
    if model_id == CONTEXT_EXTRA_TREES:
        return [
            dict(params)
            for params in core.model_param_grid(
                "full_context_extra_trees"
            )
        ]
    return [dict(params) for params in core.model_param_grid(model_id)]


def regularization_rank(
    model_id: str,
    params: dict[str, Any],
) -> tuple[float, ...]:
    if model_id == CONTEXT_EXTRA_TREES:
        max_features_rank = (
            1.0 if params["max_features"] == "sqrt" else 0.0
        )
        return (
            float(params["min_samples_leaf"]),
            max_features_rank,
        )
    return core.regularization_rank(model_id, params)


def output_model_ids(skip_label_shuffle: bool) -> list[str]:
    models = list(TUNED_MODEL_ORDER)
    if not skip_label_shuffle:
        models.append(LABEL_SHUFFLE_MODEL)
    return models


def fit_predict_model(
    model_id: str,
    params: dict[str, Any],
    rows: pd.DataFrame,
    y: np.ndarray,
    morgan: np.ndarray,
    fold_features: dict[str, tuple[np.ndarray, np.ndarray]],
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
    seed: int,
    n_estimators: int,
    n_jobs: int,
    *,
    override_train_y: np.ndarray | None = None,
) -> np.ndarray:
    if model_id != CONTEXT_EXTRA_TREES:
        return core.fit_predict_model(
            model_id,
            params,
            rows,
            y,
            morgan,
            fold_features,
            fit_idx,
            eval_idx,
            seed,
            n_estimators,
            n_jobs,
            override_train_y=override_train_y,
        )
    train_y = y[fit_idx] if override_train_y is None else override_train_y
    x_fit, x_eval = fold_features["context"]
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=params["max_features"],
        max_depth=None,
        bootstrap=False,
        random_state=seed,
        n_jobs=n_jobs,
    )
    model.fit(x_fit, train_y)
    return np.asarray(model.predict(x_eval), dtype=np.float32)


def build_ood_splits(
    rows: pd.DataFrame,
    scaffold_ids: np.ndarray,
    protocols: list[str],
    seed: int,
    max_groups: int | None,
) -> tuple[list[OODSplit], pd.DataFrame]:
    splits: list[OODSplit] = []
    audit_rows: list[dict[str, Any]] = []
    row_indices = np.arange(len(rows), dtype=np.int64)
    smiles = rows["canonical_smiles"].astype(str)

    for protocol in protocols:
        all_groups = GROUPS_BY_PROTOCOL[protocol]
        selected_groups = (
            all_groups
            if max_groups is None
            else all_groups[:max_groups]
        )
        domain_column = (
            "source_database"
            if protocol == "source_ood"
            else "target_protein"
        )
        domain_values = rows[domain_column].astype(str)
        for group_position, heldout_group in enumerate(
            all_groups,
            start=1,
        ):
            if heldout_group not in selected_groups:
                continue
            held_tokens = tokenize_domain(heldout_group)
            test_mask = domain_values.eq(heldout_group).to_numpy()
            member_mask = domain_values.map(
                lambda value: bool(
                    held_tokens.intersection(tokenize_domain(value))
                )
            ).to_numpy(dtype=bool)
            candidate_train_mask = ~member_mask
            test_smiles = set(smiles[test_mask])
            compound_overlap_mask = (
                candidate_train_mask
                & smiles.isin(test_smiles).to_numpy(dtype=bool)
            )
            train_mask = candidate_train_mask & ~compound_overlap_mask
            train_idx = row_indices[train_mask]
            test_idx = row_indices[test_mask]

            if len(train_idx) == 0 or len(test_idx) == 0:
                raise RuntimeError(
                    f"Empty OOD split: {protocol} {heldout_group}"
                )
            if set(train_idx).intersection(set(test_idx)):
                raise RuntimeError(
                    f"Row-index leakage: {protocol} {heldout_group}"
                )
            train_smiles = set(smiles.iloc[train_idx])
            if train_smiles.intersection(test_smiles):
                raise RuntimeError(
                    f"Compound leakage: {protocol} {heldout_group}"
                )
            train_domain_tokens: set[str] = set()
            for value in domain_values.iloc[train_idx]:
                train_domain_tokens.update(tokenize_domain(value))
            if held_tokens.intersection(train_domain_tokens):
                raise RuntimeError(
                    f"Domain-token leakage: {protocol} {heldout_group}"
                )

            test_scaffolds = set(scaffold_ids[test_idx])
            train_scaffolds = set(scaffold_ids[train_idx])
            observed_counts = {
                "n_test_rows": int(len(test_idx)),
                "n_test_unique_smiles": int(
                    rows.iloc[test_idx]["canonical_smiles"].nunique()
                ),
                "n_test_scaffolds": int(len(test_scaffolds)),
                "n_related_rows_excluded": int(
                    np.sum(member_mask & ~test_mask)
                ),
                "n_candidate_train_rows": int(
                    candidate_train_mask.sum()
                ),
                "n_compound_overlap_rows_excluded": int(
                    compound_overlap_mask.sum()
                ),
                "n_train_rows": int(len(train_idx)),
                "n_train_unique_smiles": int(
                    rows.iloc[train_idx]["canonical_smiles"].nunique()
                ),
                "n_train_scaffolds": int(len(train_scaffolds)),
                "n_test_rows_scaffold_seen": int(
                    np.isin(scaffold_ids[test_idx], list(train_scaffolds)).sum()
                ),
                "n_test_scaffolds_seen": int(
                    len(test_scaffolds.intersection(train_scaffolds))
                ),
            }
            expected = EXPECTED_SPLIT_COUNTS[(protocol, heldout_group)]
            if observed_counts != expected:
                raise RuntimeError(
                    "Frozen OOD split counts changed for "
                    f"{protocol} {heldout_group}: "
                    f"observed={observed_counts}, expected={expected}"
                )

            split_seed = (
                seed
                + PROTOCOL_SEED_OFFSET[protocol]
                + group_position * 10_000
            )
            audit = {
                "protocol": protocol,
                "heldout_group": heldout_group,
                "heldout_tokens": "|".join(sorted(held_tokens)),
                "domain_column": domain_column,
                "split_seed": int(split_seed),
                **observed_counts,
                "n_total_rows": int(len(rows)),
                "n_exact_domain_rows_removed": int(test_mask.sum()),
                "n_rows_unused_in_split": int(
                    len(rows) - len(train_idx) - len(test_idx)
                ),
                "train_test_row_overlap": 0,
                "train_test_smiles_overlap": 0,
                "heldout_token_seen_in_train": False,
            }
            splits.append(
                OODSplit(
                    protocol=protocol,
                    heldout_group=heldout_group,
                    group_position=group_position,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    split_seed=split_seed,
                    audit=audit,
                )
            )
            audit_rows.append(audit)
    return splits, pd.DataFrame(audit_rows)


def select_model_params(
    rows: pd.DataFrame,
    y: np.ndarray,
    morgan: np.ndarray,
    descriptors: np.ndarray,
    outer_train_idx: np.ndarray,
    scaffold_ids: np.ndarray,
    inner_folds: int,
    seed: int,
    n_estimators: int,
    n_jobs: int,
) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    inner_splits = core.balanced_group_splits(
        outer_train_idx,
        scaffold_ids,
        inner_folds,
        seed,
    )
    candidate_predictions: dict[tuple[str, str], np.ndarray] = {}
    local_position = {
        int(row_idx): position
        for position, row_idx in enumerate(outer_train_idx.tolist())
    }
    for model_id in TUNED_MODEL_ORDER:
        for params in model_param_grid(model_id):
            key = (model_id, str(params["param_id"]))
            candidate_predictions[key] = np.full(
                len(outer_train_idx),
                np.nan,
                dtype=np.float32,
            )

    inner_audit_rows: list[dict[str, Any]] = []
    for inner_fold, (fit_idx, eval_idx) in enumerate(
        inner_splits,
        start=1,
    ):
        fit_scaffolds = set(scaffold_ids[fit_idx])
        eval_scaffolds = set(scaffold_ids[eval_idx])
        overlap = fit_scaffolds.intersection(eval_scaffolds)
        if overlap:
            raise RuntimeError(
                f"Inner scaffold leakage in fold {inner_fold}"
            )
        inner_audit_rows.append(
            {
                "inner_fold": int(inner_fold),
                "inner_seed": int(seed),
                "fit_row_indices_sha256": sequence_sha256(
                    int(value) for value in fit_idx
                ),
                "eval_row_indices_sha256": sequence_sha256(
                    int(value) for value in eval_idx
                ),
                "fit_scaffold_ids_sha256": sequence_sha256(
                    sorted(str(value) for value in fit_scaffolds)
                ),
                "eval_scaffold_ids_sha256": sequence_sha256(
                    sorted(str(value) for value in eval_scaffolds)
                ),
                "n_fit_rows": int(len(fit_idx)),
                "n_eval_rows": int(len(eval_idx)),
                "n_fit_unique_smiles": int(
                    rows.iloc[fit_idx]["canonical_smiles"].nunique()
                ),
                "n_eval_unique_smiles": int(
                    rows.iloc[eval_idx]["canonical_smiles"].nunique()
                ),
                "n_fit_scaffolds": int(len(fit_scaffolds)),
                "n_eval_scaffolds": int(len(eval_scaffolds)),
                "n_scaffold_overlap": 0,
            }
        )
        fold_features, _ = core.build_fold_features(
            rows,
            morgan,
            descriptors,
            fit_idx,
            eval_idx,
        )
        eval_positions = np.asarray(
            [local_position[int(idx)] for idx in eval_idx],
            dtype=np.int64,
        )
        knn_precomputed = core.prepare_morgan_knn(
            rows,
            morgan,
            y,
            fit_idx,
            eval_idx,
        )
        for model_id in TUNED_MODEL_ORDER:
            for params in model_param_grid(model_id):
                key = (model_id, str(params["param_id"]))
                if model_id == "morgan_knn":
                    pred = core.morgan_knn_from_precomputed(
                        *knn_precomputed,
                        k=int(params["k"]),
                        power=float(params["power"]),
                    )
                else:
                    pred = fit_predict_model(
                        model_id,
                        params,
                        rows,
                        y,
                        morgan,
                        fold_features,
                        fit_idx,
                        eval_idx,
                        seed=seed + inner_fold * 101,
                        n_estimators=n_estimators,
                        n_jobs=n_jobs,
                    )
                if len(pred) != len(eval_idx) or not np.all(
                    np.isfinite(pred)
                ):
                    raise RuntimeError(
                        f"Invalid inner prediction for {model_id}"
                    )
                candidate_predictions[key][eval_positions] = pred

    selected: dict[str, dict[str, Any]] = {}
    tuning_rows: list[dict[str, Any]] = []
    inner_truth = y[outer_train_idx]
    for model_id in TUNED_MODEL_ORDER:
        scored: list[dict[str, Any]] = []
        for params in model_param_grid(model_id):
            param_id = str(params["param_id"])
            pred = candidate_predictions[(model_id, param_id)]
            if not np.all(np.isfinite(pred)):
                raise RuntimeError(
                    f"Incomplete inner OOF prediction: {model_id} {param_id}"
                )
            metrics = core.regression_metrics(inner_truth, pred)
            score = float(metrics[PRIMARY_METRIC])
            if not math.isfinite(score):
                score = -float("inf")
            scored.append(
                {
                    "score": score,
                    "rmse": float(metrics["rmse"]),
                    "param_id": param_id,
                    "params": params,
                    "regularization_rank": regularization_rank(
                        model_id,
                        params,
                    ),
                }
            )
            tuning_rows.append(
                {
                    "model_id": model_id,
                    "param_id": param_id,
                    "inner_n": int(len(inner_truth)),
                    **metrics,
                }
            )
        best_score = max(item["score"] for item in scored)
        eligible = [
            item
            for item in scored
            if item["score"] >= best_score - 0.005
        ]
        eligible.sort(
            key=lambda item: (
                tuple(
                    -value for value in item["regularization_rank"]
                ),
                item["rmse"],
                item["param_id"],
            )
        )
        selected[model_id] = dict(eligible[0]["params"])
    if len(tuning_rows) != sum(
        len(model_param_grid(model_id))
        for model_id in TUNED_MODEL_ORDER
    ):
        raise RuntimeError("Unexpected number of OOD tuning rows")
    return selected, tuning_rows, inner_audit_rows


def read_progress_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return pd.read_csv(path).to_dict("records")


def record_matches_split(
    record: dict[str, Any],
    split: OODSplit,
) -> bool:
    return bool(
        str(record.get("protocol")) == split.protocol
        and str(record.get("heldout_group")) == split.heldout_group
    )


def discard_split_records(
    records: list[dict[str, Any]],
    split: OODSplit,
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if not record_matches_split(record, split)
    ]


def parameter_grid_by_id(model_id: str) -> dict[str, dict[str, Any]]:
    return {
        str(params["param_id"]): dict(params)
        for params in model_param_grid(model_id)
    }


def validate_selected_and_tuning_records(
    split: OODSplit,
    local_selected: list[dict[str, Any]],
    local_tuning: list[dict[str, Any]],
) -> dict[str, str]:
    selected_by_model = {
        str(record["model_id"]): record for record in local_selected
    }
    tuning_by_model: dict[str, list[dict[str, Any]]] = {
        model_id: [] for model_id in TUNED_MODEL_ORDER
    }
    for record in local_tuning:
        model_id = str(record["model_id"])
        if model_id not in tuning_by_model:
            raise RuntimeError(
                f"Unexpected tuned model in checkpoint: {model_id}"
            )
        tuning_by_model[model_id].append(record)

    selected_param_ids: dict[str, str] = {}
    for model_id in TUNED_MODEL_ORDER:
        selected_record = selected_by_model.get(model_id)
        if selected_record is None:
            raise RuntimeError(
                f"Missing selected checkpoint row for {model_id}"
            )
        grid = parameter_grid_by_id(model_id)
        selected_param_id = str(selected_record.get("param_id"))
        if selected_param_id not in grid:
            raise RuntimeError(
                f"Off-grid selected parameter for {model_id}: "
                f"{selected_param_id}"
            )
        expected_params = grid[selected_param_id]
        for field in HYPERPARAMETER_FIELDS:
            expected_value = expected_params.get(field)
            observed_value = selected_record.get(field)
            if not values_match(observed_value, expected_value):
                raise RuntimeError(
                    "Selected parameter content mismatch for "
                    f"{model_id} {selected_param_id}: {field}"
                )

        tuning_records = tuning_by_model[model_id]
        tuning_keys = {
            str(record.get("param_id")) for record in tuning_records
        }
        if tuning_keys != set(grid) or len(tuning_records) != len(grid):
            raise RuntimeError(
                f"Tuning grid mismatch for {model_id}"
            )
        scored: list[dict[str, Any]] = []
        for record in tuning_records:
            param_id = str(record["param_id"])
            if not values_match(
                record.get("inner_n"),
                len(split.train_idx),
            ):
                raise RuntimeError(
                    f"Inner sample count mismatch for {model_id} {param_id}"
                )
            rmse = float(record.get("rmse", float("nan")))
            mae = float(record.get("mae", float("nan")))
            if (
                not math.isfinite(rmse)
                or rmse < 0.0
                or not math.isfinite(mae)
                or mae < 0.0
            ):
                raise RuntimeError(
                    f"Invalid tuning error metric for {model_id} {param_id}"
                )
            score = float(record.get(PRIMARY_METRIC, float("nan")))
            if not math.isfinite(score):
                score = -float("inf")
            params = grid[param_id]
            scored.append(
                {
                    "score": score,
                    "rmse": rmse,
                    "param_id": param_id,
                    "regularization_rank": regularization_rank(
                        model_id,
                        params,
                    ),
                }
            )
        best_score = max(item["score"] for item in scored)
        eligible = [
            item
            for item in scored
            if item["score"] >= best_score - 0.005
        ]
        eligible.sort(
            key=lambda item: (
                tuple(
                    -value for value in item["regularization_rank"]
                ),
                item["rmse"],
                item["param_id"],
            )
        )
        if selected_param_id != eligible[0]["param_id"]:
            raise RuntimeError(
                "Selected parameter is inconsistent with stored tuning "
                f"metrics for {model_id}: selected={selected_param_id}, "
                f"recomputed={eligible[0]['param_id']}"
            )
        selected_param_ids[model_id] = selected_param_id
    return selected_param_ids


def validate_inner_audit_records(
    split: OODSplit,
    local_inner: list[dict[str, Any]],
    rows: pd.DataFrame,
    scaffold_ids: np.ndarray,
    inner_folds: int,
) -> None:
    inner_seed = split.split_seed + 101
    expected_splits = core.balanced_group_splits(
        split.train_idx,
        scaffold_ids,
        inner_folds,
        inner_seed,
    )
    observed_by_fold = {
        int(record["inner_fold"]): record for record in local_inner
    }
    for inner_fold, (fit_idx, eval_idx) in enumerate(
        expected_splits,
        start=1,
    ):
        fit_scaffolds = set(scaffold_ids[fit_idx])
        eval_scaffolds = set(scaffold_ids[eval_idx])
        expected = {
            "inner_fold": int(inner_fold),
            "inner_seed": int(inner_seed),
            "fit_row_indices_sha256": sequence_sha256(
                int(value) for value in fit_idx
            ),
            "eval_row_indices_sha256": sequence_sha256(
                int(value) for value in eval_idx
            ),
            "fit_scaffold_ids_sha256": sequence_sha256(
                sorted(str(value) for value in fit_scaffolds)
            ),
            "eval_scaffold_ids_sha256": sequence_sha256(
                sorted(str(value) for value in eval_scaffolds)
            ),
            "n_fit_rows": int(len(fit_idx)),
            "n_eval_rows": int(len(eval_idx)),
            "n_fit_unique_smiles": int(
                rows.iloc[fit_idx]["canonical_smiles"].nunique()
            ),
            "n_eval_unique_smiles": int(
                rows.iloc[eval_idx]["canonical_smiles"].nunique()
            ),
            "n_fit_scaffolds": int(len(fit_scaffolds)),
            "n_eval_scaffolds": int(len(eval_scaffolds)),
            "n_scaffold_overlap": 0,
        }
        observed = observed_by_fold.get(inner_fold)
        if observed is None:
            raise RuntimeError(
                f"Missing inner audit fold {inner_fold}"
            )
        for field, expected_value in expected.items():
            if not values_match(observed.get(field), expected_value):
                raise RuntimeError(
                    "Inner split audit mismatch for "
                    f"{split.protocol} {split.heldout_group} "
                    f"fold={inner_fold} field={field}"
                )


def validate_prediction_records(
    split: OODSplit,
    pred_frame: pd.DataFrame,
    rows: pd.DataFrame,
    y: np.ndarray,
    scaffold_ids: np.ndarray,
    selected_param_ids: dict[str, str],
    skip_label_shuffle: bool,
) -> None:
    required_columns = {
        "protocol",
        "heldout_group",
        "row_index",
        "qc_id",
        "model_id",
        "param_id",
        "y_true",
        "y_pred",
        "canonical_smiles",
        "scaffold_id",
        "source_database",
        "recruiting_protein",
        "target_protein",
        "cell_line",
        "max_train_tanimoto",
        "compound_seen_in_train",
        "scaffold_seen_in_train",
        "target_train_rows",
        "target_train_unique_smiles",
        "recruiter_target_train_unique_smiles",
        "source_seen_in_train",
        "heldout_token_seen_in_train",
    }
    missing_columns = required_columns.difference(pred_frame.columns)
    if missing_columns:
        raise RuntimeError(
            f"Prediction checkpoint columns missing: {sorted(missing_columns)}"
        )
    train_rows = rows.iloc[split.train_idx]
    train_smiles = set(train_rows["canonical_smiles"].astype(str))
    train_scaffolds = set(scaffold_ids[split.train_idx])
    train_sources = set(train_rows["source_database"].astype(str))
    target_support_rows = train_rows["target_protein"].astype(str).value_counts()
    target_support_compounds = train_rows.groupby("target_protein")[
        "canonical_smiles"
    ].nunique()
    recruiter_target_support = (
        train_rows.assign(
            recruiter_target_key=core.key_series(
                train_rows,
                ("recruiting_protein", "target_protein"),
            ).to_numpy()
        )
        .groupby("recruiter_target_key")["canonical_smiles"]
        .nunique()
    )
    expected_models = output_model_ids(skip_label_shuffle)
    for model_id in expected_models:
        local = pred_frame[pred_frame["model_id"].astype(str) == model_id]
        expected_param_id = (
            f"shuffle::{selected_param_ids['full_context_extra_trees']}"
            if model_id == LABEL_SHUFFLE_MODEL
            else selected_param_ids[model_id]
        )
        for record in local.to_dict("records"):
            row_idx = int(record["row_index"])
            if row_idx not in set(split.test_idx.astype(int).tolist()):
                raise RuntimeError(
                    f"Unexpected OOD prediction row index: {row_idx}"
                )
            source_row = rows.iloc[row_idx]
            target_value = str(source_row["target_protein"])
            source_value = str(source_row["source_database"])
            recruiter_target_key = (
                f"{source_row['recruiting_protein']}||{target_value}"
            )
            expected_fields = {
                "protocol": split.protocol,
                "heldout_group": split.heldout_group,
                "qc_id": str(source_row["qc_id"]),
                "model_id": model_id,
                "param_id": expected_param_id,
                "y_true": float(y[row_idx]),
                "canonical_smiles": str(source_row["canonical_smiles"]),
                "scaffold_id": str(scaffold_ids[row_idx]),
                "source_database": source_value,
                "recruiting_protein": str(
                    source_row["recruiting_protein"]
                ),
                "target_protein": target_value,
                "cell_line": str(source_row["cell_line"]),
                "compound_seen_in_train": False,
                "scaffold_seen_in_train": bool(
                    scaffold_ids[row_idx] in train_scaffolds
                ),
                "target_train_rows": int(
                    target_support_rows.get(target_value, 0)
                ),
                "target_train_unique_smiles": int(
                    target_support_compounds.get(target_value, 0)
                ),
                "recruiter_target_train_unique_smiles": int(
                    recruiter_target_support.get(
                        recruiter_target_key,
                        0,
                    )
                ),
                "source_seen_in_train": bool(
                    source_value in train_sources
                ),
                "heldout_token_seen_in_train": False,
            }
            if str(source_row["canonical_smiles"]) in train_smiles:
                raise RuntimeError(
                    "Frozen split contains an exact compound overlap"
                )
            for field, expected_value in expected_fields.items():
                tolerance = 1e-7 if field == "y_true" else 1e-10
                if not values_match(
                    record.get(field),
                    expected_value,
                    atol=tolerance,
                    rtol=tolerance,
                ):
                    raise RuntimeError(
                        "Prediction checkpoint identity mismatch for "
                        f"{split.protocol} {split.heldout_group} "
                        f"model={model_id} row={row_idx} field={field}"
                    )
            y_pred = float(record.get("y_pred", float("nan")))
            max_tanimoto = float(
                record.get("max_train_tanimoto", float("nan"))
            )
            if not math.isfinite(y_pred):
                raise RuntimeError("Non-finite checkpoint prediction")
            if (
                not math.isfinite(max_tanimoto)
                or max_tanimoto < 0.0
                or max_tanimoto > 1.0 + 1e-6
            ):
                raise RuntimeError(
                    "Invalid checkpoint maximum Tanimoto similarity"
                )


def validate_domain_metric_records(
    split: OODSplit,
    pred_frame: pd.DataFrame,
    local_metrics: list[dict[str, Any]],
    rows: pd.DataFrame,
    scaffold_ids: np.ndarray,
    selected_param_ids: dict[str, str],
    skip_label_shuffle: bool,
) -> None:
    metric_by_model = {
        str(record["model_id"]): record for record in local_metrics
    }
    train_rows = rows.iloc[split.train_idx]
    expected_counts = {
        "n_train_rows": int(len(split.train_idx)),
        "n_train_unique_smiles": int(
            train_rows["canonical_smiles"].nunique()
        ),
        "n_train_scaffolds": int(
            len(np.unique(scaffold_ids[split.train_idx]))
        ),
        "n_test_rows": int(len(split.test_idx)),
        "n_test_unique_smiles": int(
            rows.iloc[split.test_idx]["canonical_smiles"].nunique()
        ),
        "n_test_scaffolds": int(
            len(np.unique(scaffold_ids[split.test_idx]))
        ),
    }
    for model_id in output_model_ids(skip_label_shuffle):
        record = metric_by_model.get(model_id)
        if record is None:
            raise RuntimeError(
                f"Missing domain metric checkpoint for {model_id}"
            )
        expected_param_id = (
            f"shuffle::{selected_param_ids['full_context_extra_trees']}"
            if model_id == LABEL_SHUFFLE_MODEL
            else selected_param_ids[model_id]
        )
        if not values_match(record.get("param_id"), expected_param_id):
            raise RuntimeError(
                f"Domain metric parameter linkage failed for {model_id}"
            )
        for field, expected_value in expected_counts.items():
            if not values_match(record.get(field), expected_value):
                raise RuntimeError(
                    f"Domain metric count mismatch for {model_id}: {field}"
                )
        local_predictions = pred_frame[
            pred_frame["model_id"].astype(str) == model_id
        ]
        recomputed = core.regression_metrics(
            local_predictions["y_true"].to_numpy(dtype=np.float32),
            local_predictions["y_pred"].to_numpy(dtype=np.float32),
        )
        for metric, expected_value in recomputed.items():
            if not values_match(
                record.get(metric),
                expected_value,
                atol=1e-9,
                rtol=1e-9,
            ):
                raise RuntimeError(
                    "Stored domain metric disagrees with predictions for "
                    f"{model_id}: {metric}"
                )


def checkpoint_is_complete(
    split: OODSplit,
    predictions: list[dict[str, Any]],
    domain_metrics: list[dict[str, Any]],
    tuning: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    inner_audit: list[dict[str, Any]],
    rows: pd.DataFrame,
    y: np.ndarray,
    scaffold_ids: np.ndarray,
    inner_folds: int,
    skip_label_shuffle: bool,
) -> bool:
    expected_models = set(output_model_ids(skip_label_shuffle))
    expected_tuned_models = set(TUNED_MODEL_ORDER)
    expected_rows = set(split.test_idx.astype(int).tolist())

    local_predictions = [
        row for row in predictions if record_matches_split(row, split)
    ]
    if len(local_predictions) != len(expected_rows) * len(expected_models):
        return False
    pred_frame = pd.DataFrame(local_predictions)
    if pred_frame.empty or bool(
        pred_frame.duplicated(["model_id", "row_index"]).any()
    ):
        return False
    if set(pred_frame["model_id"].astype(str)) != expected_models:
        return False
    if not np.all(
        np.isfinite(pd.to_numeric(pred_frame["y_pred"], errors="coerce"))
    ):
        return False
    for _, frame in pred_frame.groupby("model_id"):
        if set(frame["row_index"].astype(int)) != expected_rows:
            return False

    local_metrics = [
        row for row in domain_metrics if record_matches_split(row, split)
    ]
    if len(local_metrics) != len(expected_models):
        return False
    if {
        str(row.get("model_id")) for row in local_metrics
    } != expected_models:
        return False

    local_tuning = [
        row for row in tuning if record_matches_split(row, split)
    ]
    expected_tuning_keys = {
        (model_id, str(params["param_id"]))
        for model_id in TUNED_MODEL_ORDER
        for params in model_param_grid(model_id)
    }
    observed_tuning_keys = {
        (str(row.get("model_id")), str(row.get("param_id")))
        for row in local_tuning
    }
    if (
        len(local_tuning) != len(expected_tuning_keys)
        or observed_tuning_keys != expected_tuning_keys
    ):
        return False

    local_selected = [
        row for row in selected if record_matches_split(row, split)
    ]
    if len(local_selected) != len(expected_tuned_models):
        return False
    if {
        str(row.get("model_id")) for row in local_selected
    } != expected_tuned_models:
        return False

    local_inner = [
        row for row in inner_audit if record_matches_split(row, split)
    ]
    if len(local_inner) != inner_folds:
        return False
    if {
        int(row.get("inner_fold")) for row in local_inner
    } != set(range(1, inner_folds + 1)):
        return False
    if any(int(row.get("n_scaffold_overlap", -1)) != 0 for row in local_inner):
        return False
    selected_param_ids = validate_selected_and_tuning_records(
        split,
        local_selected,
        local_tuning,
    )
    validate_inner_audit_records(
        split,
        local_inner,
        rows,
        scaffold_ids,
        inner_folds,
    )
    validate_prediction_records(
        split,
        pred_frame,
        rows,
        y,
        scaffold_ids,
        selected_param_ids,
        skip_label_shuffle,
    )
    validate_domain_metric_records(
        split,
        pred_frame,
        local_metrics,
        rows,
        scaffold_ids,
        selected_param_ids,
        skip_label_shuffle,
    )
    return True


def write_progress_tables(
    output_dir: Path,
    predictions: list[dict[str, Any]],
    domain_metrics: list[dict[str, Any]],
    tuning: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    inner_audit: list[dict[str, Any]],
    configuration_sha256: str,
) -> None:
    tables = {
        "predictions": predictions,
        "domain_metrics": domain_metrics,
        "tuning": tuning,
        "selected": selected,
        "inner_audit": inner_audit,
    }
    for key, records in tables.items():
        core.atomic_to_csv(
            pd.DataFrame(records),
            output_dir / PROGRESS_FILES[key],
        )
    progress_paths = [
        output_dir / filename for filename in PROGRESS_FILES.values()
    ]
    checkpoint_inventory = {
        "inventory_version": CHECKPOINT_INVENTORY_VERSION,
        "scientific_configuration_sha256": configuration_sha256,
        "files": build_file_inventory(progress_paths),
    }
    core.write_json(
        output_dir / CHECKPOINT_INVENTORY_FILE,
        checkpoint_inventory,
    )


def run_splits(
    args: argparse.Namespace,
    rows: pd.DataFrame,
    y: np.ndarray,
    morgan: np.ndarray,
    descriptors: np.ndarray,
    scaffold_ids: np.ndarray,
    splits: list[OODSplit],
    output_dir: Path,
    configuration_sha256: str,
) -> pd.DataFrame:
    records = {
        key: (
            read_progress_records(output_dir / filename)
            if args.resume
            else []
        )
        for key, filename in PROGRESS_FILES.items()
    }
    start = time.perf_counter()
    for split_number, split in enumerate(splits, start=1):
        if args.resume and checkpoint_is_complete(
            split,
            records["predictions"],
            records["domain_metrics"],
            records["tuning"],
            records["selected"],
            records["inner_audit"],
            rows,
            y,
            scaffold_ids,
            args.inner_folds,
            args.skip_label_shuffle,
        ):
            print(
                (
                    f"[RESUME] protocol={split.protocol} "
                    f"group={split.heldout_group} complete; skipping"
                ),
                flush=True,
            )
            continue
        if args.resume:
            for key in records:
                records[key] = discard_split_records(
                    records[key],
                    split,
                )

        print(
            (
                f"[OOD] protocol={split.protocol} "
                f"group={split.heldout_group} "
                f"({split_number}/{len(splits)}) "
                f"train={len(split.train_idx)} test={len(split.test_idx)}"
            ),
            flush=True,
        )
        inner_seed = split.split_seed + 101
        selected, local_tuning, local_inner_audit = select_model_params(
            rows,
            y,
            morgan,
            descriptors,
            split.train_idx,
            scaffold_ids,
            args.inner_folds,
            inner_seed,
            args.n_estimators,
            args.n_jobs,
        )
        for row in local_tuning:
            records["tuning"].append(
                {
                    "protocol": split.protocol,
                    "heldout_group": split.heldout_group,
                    **row,
                }
            )
        for model_id, params in selected.items():
            records["selected"].append(
                {
                    "protocol": split.protocol,
                    "heldout_group": split.heldout_group,
                    "model_id": model_id,
                    **params,
                }
            )
        for row in local_inner_audit:
            records["inner_audit"].append(
                {
                    "protocol": split.protocol,
                    "heldout_group": split.heldout_group,
                    **row,
                }
            )

        fold_features, fold_feature_meta = core.build_fold_features(
            rows,
            morgan,
            descriptors,
            split.train_idx,
            split.test_idx,
        )
        similarities = core.tanimoto_matrix(
            morgan[split.test_idx],
            morgan[split.train_idx],
        )
        max_tanimoto = similarities.max(axis=1)
        train_rows = rows.iloc[split.train_idx]
        train_smiles = set(
            train_rows["canonical_smiles"].astype(str)
        )
        train_scaffolds = set(scaffold_ids[split.train_idx])
        train_sources = set(train_rows["source_database"].astype(str))
        train_targets = train_rows["target_protein"].astype(str)
        target_support_rows = train_targets.value_counts()
        target_support_compounds = train_rows.groupby("target_protein")[
            "canonical_smiles"
        ].nunique()
        recruiter_target_support_compounds = (
            train_rows.assign(
                recruiter_target_key=core.key_series(
                    train_rows,
                    ("recruiting_protein", "target_protein"),
                ).to_numpy()
            )
            .groupby("recruiter_target_key")["canonical_smiles"]
            .nunique()
        )
        held_tokens = tokenize_domain(split.heldout_group)
        domain_column = (
            "source_database"
            if split.protocol == "source_ood"
            else "target_protein"
        )
        train_domain_tokens: set[str] = set()
        for value in train_rows[domain_column].astype(str):
            train_domain_tokens.update(tokenize_domain(value))
        if held_tokens.intersection(train_domain_tokens):
            raise RuntimeError("Held-out token entered final training data")

        for model_position, model_id in enumerate(
            output_model_ids(args.skip_label_shuffle)
        ):
            model_seed = (
                split.split_seed + 1_000 + model_position
            )
            permutation_diagnostics: dict[str, float | int] = {}
            if model_id == LABEL_SHUFFLE_MODEL:
                params = selected["full_context_extra_trees"]
                shuffled_y, permutation_diagnostics = (
                    core.within_domain_label_permutation(
                        rows,
                        y[split.train_idx].copy(),
                        split.train_idx,
                        model_seed,
                    )
                )
                pred = fit_predict_model(
                    "full_context_extra_trees",
                    params,
                    rows,
                    y,
                    morgan,
                    fold_features,
                    split.train_idx,
                    split.test_idx,
                    model_seed,
                    args.n_estimators,
                    args.n_jobs,
                    override_train_y=shuffled_y,
                )
                param_id = f"shuffle::{params['param_id']}"
            else:
                params = selected[model_id]
                pred = fit_predict_model(
                    model_id,
                    params,
                    rows,
                    y,
                    morgan,
                    fold_features,
                    split.train_idx,
                    split.test_idx,
                    model_seed,
                    args.n_estimators,
                    args.n_jobs,
                )
                param_id = str(params["param_id"])
            if len(pred) != len(split.test_idx) or not np.all(
                np.isfinite(pred)
            ):
                raise RuntimeError(
                    f"Invalid final OOD predictions for {model_id}"
                )
            metrics = core.regression_metrics(
                y[split.test_idx],
                pred,
            )
            records["domain_metrics"].append(
                {
                    "protocol": split.protocol,
                    "heldout_group": split.heldout_group,
                    "model_id": model_id,
                    "param_id": param_id,
                    "n_train_rows": int(len(split.train_idx)),
                    "n_train_unique_smiles": int(
                        train_rows["canonical_smiles"].nunique()
                    ),
                    "n_train_scaffolds": int(
                        len(np.unique(scaffold_ids[split.train_idx]))
                    ),
                    "n_test_rows": int(len(split.test_idx)),
                    "n_test_unique_smiles": int(
                        rows.iloc[split.test_idx][
                            "canonical_smiles"
                        ].nunique()
                    ),
                    "n_test_scaffolds": int(
                        len(np.unique(scaffold_ids[split.test_idx]))
                    ),
                    **metrics,
                    **fold_feature_meta,
                    **permutation_diagnostics,
                }
            )
            for local_idx, row_idx in enumerate(split.test_idx):
                row = rows.iloc[row_idx]
                target_value = str(row["target_protein"])
                source_value = str(row["source_database"])
                recruiter_target_value = (
                    f"{row['recruiting_protein']}||{target_value}"
                )
                compound_seen = (
                    str(row["canonical_smiles"]) in train_smiles
                )
                if compound_seen:
                    raise RuntimeError(
                        "Exact test compound entered final training data"
                    )
                records["predictions"].append(
                    {
                        "protocol": split.protocol,
                        "heldout_group": split.heldout_group,
                        "row_index": int(row_idx),
                        "qc_id": str(row["qc_id"]),
                        "model_id": model_id,
                        "param_id": param_id,
                        "y_true": float(y[row_idx]),
                        "y_pred": float(pred[local_idx]),
                        "canonical_smiles": str(
                            row["canonical_smiles"]
                        ),
                        "scaffold_id": str(scaffold_ids[row_idx]),
                        "source_database": source_value,
                        "recruiting_protein": str(
                            row["recruiting_protein"]
                        ),
                        "target_protein": target_value,
                        "cell_line": str(row["cell_line"]),
                        "max_train_tanimoto": float(
                            max_tanimoto[local_idx]
                        ),
                        "compound_seen_in_train": False,
                        "scaffold_seen_in_train": bool(
                            scaffold_ids[row_idx] in train_scaffolds
                        ),
                        "target_train_rows": int(
                            target_support_rows.get(target_value, 0)
                        ),
                        "target_train_unique_smiles": int(
                            target_support_compounds.get(target_value, 0)
                        ),
                        "recruiter_target_train_unique_smiles": int(
                            recruiter_target_support_compounds.get(
                                recruiter_target_value,
                                0,
                            )
                        ),
                        "source_seen_in_train": bool(
                            source_value in train_sources
                        ),
                        "heldout_token_seen_in_train": False,
                    }
                )

        write_progress_tables(
            output_dir,
            records["predictions"],
            records["domain_metrics"],
            records["tuning"],
            records["selected"],
            records["inner_audit"],
            configuration_sha256,
        )
        if not checkpoint_is_complete(
            split,
            records["predictions"],
            records["domain_metrics"],
            records["tuning"],
            records["selected"],
            records["inner_audit"],
            rows,
            y,
            scaffold_ids,
            args.inner_folds,
            args.skip_label_shuffle,
        ):
            raise RuntimeError(
                f"Just-written checkpoint is incomplete: "
                f"{split.protocol} {split.heldout_group}"
            )
        elapsed = time.perf_counter() - start
        print(
            (
                f"[OOD] completed {split_number}/{len(splits)} "
                f"in {elapsed:.1f}s"
            ),
            flush=True,
        )
    return pd.DataFrame(records["predictions"])


def complete_predictions(
    predictions: pd.DataFrame,
    splits: list[OODSplit],
    rows: pd.DataFrame,
    y: np.ndarray,
    scaffold_ids: np.ndarray,
    skip_label_shuffle: bool,
) -> bool:
    if predictions.empty:
        return False
    expected_models = set(output_model_ids(skip_label_shuffle))
    if not np.all(np.isfinite(predictions["y_pred"])):
        return False
    if bool(
        predictions.duplicated(
            ["protocol", "model_id", "row_index"]
        ).any()
    ):
        return False
    split_by_key = {
        (split.protocol, split.heldout_group): split
        for split in splits
    }
    if set(
        zip(
            predictions["protocol"].astype(str),
            predictions["heldout_group"].astype(str),
        )
    ) != set(split_by_key):
        return False
    for split in splits:
        local = predictions[
            (predictions["protocol"] == split.protocol)
            & (
                predictions["heldout_group"]
                == split.heldout_group
            )
        ]
        if set(local["model_id"].astype(str)) != expected_models:
            return False
        expected_rows = set(split.test_idx.astype(int).tolist())
        for _, frame in local.groupby("model_id"):
            if set(frame["row_index"].astype(int)) != expected_rows:
                return False
        for record in local.to_dict("records"):
            row_idx = int(record["row_index"])
            source_row = rows.iloc[row_idx]
            expected_fields = {
                "protocol": split.protocol,
                "heldout_group": split.heldout_group,
                "qc_id": str(source_row["qc_id"]),
                "y_true": float(y[row_idx]),
                "canonical_smiles": str(source_row["canonical_smiles"]),
                "scaffold_id": str(scaffold_ids[row_idx]),
                "source_database": str(source_row["source_database"]),
                "recruiting_protein": str(
                    source_row["recruiting_protein"]
                ),
                "target_protein": str(source_row["target_protein"]),
                "cell_line": str(source_row["cell_line"]),
                "compound_seen_in_train": False,
                "heldout_token_seen_in_train": False,
            }
            for field, expected_value in expected_fields.items():
                tolerance = 1e-7 if field == "y_true" else 1e-10
                if not values_match(
                    record.get(field),
                    expected_value,
                    atol=tolerance,
                    rtol=tolerance,
                ):
                    return False
    y_consistency = predictions.groupby(
        ["protocol", "row_index"]
    )["y_true"].nunique()
    if bool((y_consistency != 1).any()):
        return False
    return True


def summarize_aggregate_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    metric_names = tuple(
        core.regression_metrics(
            np.asarray([0.0, 1.0, 2.0]),
            np.asarray([0.0, 1.0, 2.0]),
        )
    )
    for (protocol, model_id), frame in predictions.groupby(
        ["protocol", "model_id"],
        sort=False,
    ):
        groups = [
            group
            for group in GROUPS_BY_PROTOCOL[protocol]
            if group in set(frame["heldout_group"].astype(str))
        ]
        per_domain: dict[str, list[float]] = {
            metric: [] for metric in metric_names
        }
        for group in groups:
            local = frame[frame["heldout_group"] == group]
            metrics = core.regression_metrics(
                local["y_true"].to_numpy(dtype=np.float64),
                local["y_pred"].to_numpy(dtype=np.float64),
            )
            for metric in metric_names:
                per_domain[metric].append(float(metrics[metric]))
        pooled = core.regression_metrics(
            frame["y_true"].to_numpy(dtype=np.float64),
            frame["y_pred"].to_numpy(dtype=np.float64),
        )
        row: dict[str, Any] = {
            "protocol": protocol,
            "model_id": model_id,
            "n_domains": int(len(groups)),
            "n_rows": int(len(frame)),
            "n_unique_smiles": int(
                frame["canonical_smiles"].nunique()
            ),
            "n_scaffolds": int(frame["scaffold_id"].nunique()),
        }
        for metric in metric_names:
            values = np.asarray(per_domain[metric], dtype=np.float64)
            finite = values[np.isfinite(values)]
            row[f"domain_macro_{metric}"] = (
                float(np.mean(finite))
                if len(finite) == len(groups)
                else float("nan")
            )
            row[f"domain_macro_{metric}_finite_domains"] = int(
                len(finite)
            )
            row[f"pooled_{metric}"] = float(pooled[metric])
        if protocol == "source_ood":
            row.update(core.target_macro_metrics(frame))
            row["within_target_spearman"] = (
                core.within_target_spearman(frame)
            )
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def metric_bundle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    domains: np.ndarray,
    positions: np.ndarray,
    domain_order: tuple[str, ...],
) -> dict[str, Any]:
    selected_y = y_true[positions]
    selected_pred = y_pred[positions]
    selected_domains = domains[positions]
    per_domain: dict[str, dict[str, float]] = {}
    spearman_values: list[float] = []
    rmse_values: list[float] = []
    for domain in domain_order:
        local = np.where(selected_domains == domain)[0]
        if len(local) == 0:
            spearman = float("nan")
            rmse = float("nan")
        else:
            spearman = core.safe_spearman(
                selected_y[local],
                selected_pred[local],
            )
            rmse = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            selected_y[local] - selected_pred[local]
                        )
                    )
                )
            )
        per_domain[domain] = {
            "spearman": spearman,
            "rmse": rmse,
        }
        spearman_values.append(spearman)
        rmse_values.append(rmse)
    finite_spearman = [
        value for value in spearman_values if math.isfinite(value)
    ]
    finite_rmse = [
        value for value in rmse_values if math.isfinite(value)
    ]
    domain_macro_spearman = (
        float(np.mean(finite_spearman))
        if len(finite_spearman) == len(domain_order)
        else float("nan")
    )
    domain_macro_rmse = (
        float(np.mean(finite_rmse))
        if len(finite_rmse) == len(domain_order)
        else float("nan")
    )
    pooled_rmse = float(
        np.sqrt(np.mean(np.square(selected_y - selected_pred)))
    )
    return {
        "domain_macro_spearman": domain_macro_spearman,
        "domain_macro_rmse": domain_macro_rmse,
        "pooled_spearman": core.safe_spearman(
            selected_y,
            selected_pred,
        ),
        "pooled_rmse": pooled_rmse,
        "per_domain": per_domain,
    }


def finite_interval(
    values: Iterable[float],
) -> tuple[float, float, float, int]:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            0,
        )
    return (
        float(np.mean(finite)),
        float(np.percentile(finite, 2.5)),
        float(np.percentile(finite, 97.5)),
        int(len(finite)),
    )


def paired_scaffold_bootstrap(
    predictions: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_bootstrap <= 0:
        return pd.DataFrame(), pd.DataFrame()
    aggregate_rows: list[dict[str, Any]] = []
    domain_rows: list[dict[str, Any]] = []
    metric_names = (
        "domain_macro_spearman",
        "domain_macro_rmse",
        "pooled_spearman",
        "pooled_rmse",
    )
    for protocol in PROTOCOL_ORDER:
        protocol_df = predictions[
            predictions["protocol"] == protocol
        ].copy()
        if protocol_df.empty:
            continue
        models = [
            model_id
            for model_id in MODEL_ORDER
            if model_id in set(protocol_df["model_id"].astype(str))
        ]
        domain_order = tuple(
            group
            for group in GROUPS_BY_PROTOCOL[protocol]
            if group in set(
                protocol_df["heldout_group"].astype(str)
            )
        )
        reference_frame = (
            protocol_df[protocol_df["model_id"] == models[0]]
            .sort_values("row_index")
            .reset_index(drop=True)
        )
        row_indices = reference_frame["row_index"].to_numpy(
            dtype=np.int64
        )
        y_true = reference_frame["y_true"].to_numpy(
            dtype=np.float64
        )
        domains = reference_frame["heldout_group"].astype(str).to_numpy()
        clusters = reference_frame["scaffold_id"].astype(str).to_numpy()
        unique_clusters = np.unique(clusters)
        cluster_to_positions = {
            cluster: np.where(clusters == cluster)[0]
            for cluster in unique_clusters
        }
        model_predictions: dict[str, np.ndarray] = {}
        observed: dict[str, dict[str, Any]] = {}
        for model_id in models:
            frame = (
                protocol_df[protocol_df["model_id"] == model_id]
                .sort_values("row_index")
                .reset_index(drop=True)
            )
            if not np.array_equal(
                frame["row_index"].to_numpy(dtype=np.int64),
                row_indices,
            ):
                raise RuntimeError(
                    "Bootstrap requires model-aligned OOD rows"
                )
            if not np.array_equal(
                frame["y_true"].to_numpy(dtype=np.float64),
                y_true,
            ):
                raise RuntimeError(
                    "Bootstrap requires identical outcomes across models"
                )
            pred = frame["y_pred"].to_numpy(dtype=np.float64)
            model_predictions[model_id] = pred
            observed[model_id] = metric_bundle(
                y_true,
                pred,
                domains,
                np.arange(len(y_true), dtype=np.int64),
                domain_order,
            )

        distributions: dict[str, dict[str, list[float]]] = {
            model_id: {metric: [] for metric in metric_names}
            for model_id in models
        }
        per_domain_distributions: dict[
            str,
            dict[str, dict[str, list[float]]],
        ] = {
            model_id: {
                domain: {"spearman": [], "rmse": []}
                for domain in domain_order
            }
            for model_id in models
        }
        valid_contrasts = {
            contrast_id: pair
            for contrast_id, pair in PRESPECIFIED_CONTRASTS.items()
            if pair[0] in models and pair[1] in models
        }
        contrast_distributions: dict[
            str,
            dict[str, list[float]],
        ] = {
            contrast_id: {
                f"delta_{metric}": [] for metric in metric_names
            }
            for contrast_id in valid_contrasts
        }
        rng = np.random.default_rng(
            seed + PROTOCOL_SEED_OFFSET[protocol] + 9_000_000
        )
        for bootstrap_idx in range(n_bootstrap):
            sampled_clusters = rng.choice(
                unique_clusters,
                size=len(unique_clusters),
                replace=True,
            )
            sampled_positions = np.concatenate(
                [
                    cluster_to_positions[cluster]
                    for cluster in sampled_clusters
                ]
            )
            stats_by_model: dict[str, dict[str, Any]] = {}
            for model_id in models:
                stats = metric_bundle(
                    y_true,
                    model_predictions[model_id],
                    domains,
                    sampled_positions,
                    domain_order,
                )
                stats_by_model[model_id] = stats
                for metric in metric_names:
                    distributions[model_id][metric].append(
                        float(stats[metric])
                    )
                for domain in domain_order:
                    for metric in ("spearman", "rmse"):
                        per_domain_distributions[model_id][domain][
                            metric
                        ].append(
                            float(stats["per_domain"][domain][metric])
                        )
            for contrast_id, (
                first_model,
                comparator_model,
            ) in valid_contrasts.items():
                first = stats_by_model[first_model]
                comparator = stats_by_model[comparator_model]
                contrast_distributions[contrast_id][
                    "delta_domain_macro_spearman"
                ].append(
                    first["domain_macro_spearman"]
                    - comparator["domain_macro_spearman"]
                )
                contrast_distributions[contrast_id][
                    "delta_domain_macro_rmse"
                ].append(
                    comparator["domain_macro_rmse"]
                    - first["domain_macro_rmse"]
                )
                contrast_distributions[contrast_id][
                    "delta_pooled_spearman"
                ].append(
                    first["pooled_spearman"]
                    - comparator["pooled_spearman"]
                )
                contrast_distributions[contrast_id][
                    "delta_pooled_rmse"
                ].append(
                    comparator["pooled_rmse"]
                    - first["pooled_rmse"]
                )
            if (bootstrap_idx + 1) % max(1, n_bootstrap // 10) == 0:
                print(
                    (
                        f"[BOOT] {protocol}: "
                        f"{bootstrap_idx + 1}/{n_bootstrap}"
                    ),
                    flush=True,
                )

        for model_id in models:
            row: dict[str, Any] = {
                "record_type": "model",
                "protocol": protocol,
                "model_id": model_id,
                "cluster_unit": "global scaffold_id",
                "n_clusters": int(len(unique_clusters)),
                "n_domains": int(len(domain_order)),
                "n_bootstrap": int(n_bootstrap),
            }
            for metric in metric_names:
                mean, low, high, n_valid = finite_interval(
                    distributions[model_id][metric]
                )
                row[f"{metric}_observed"] = float(
                    observed[model_id][metric]
                )
                row[f"{metric}_bootstrap_mean"] = mean
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
                row[f"{metric}_n_valid"] = n_valid
            aggregate_rows.append(row)
            for domain in domain_order:
                domain_row: dict[str, Any] = {
                    "protocol": protocol,
                    "heldout_group": domain,
                    "model_id": model_id,
                    "cluster_unit": "global scaffold_id",
                    "n_bootstrap": int(n_bootstrap),
                }
                for metric in ("spearman", "rmse"):
                    mean, low, high, n_valid = finite_interval(
                        per_domain_distributions[model_id][domain][
                            metric
                        ]
                    )
                    domain_row[f"{metric}_observed"] = float(
                        observed[model_id]["per_domain"][domain][
                            metric
                        ]
                    )
                    domain_row[f"{metric}_bootstrap_mean"] = mean
                    domain_row[f"{metric}_ci_low"] = low
                    domain_row[f"{metric}_ci_high"] = high
                    domain_row[f"{metric}_n_valid"] = n_valid
                domain_rows.append(domain_row)

        for contrast_id, (
            first_model,
            comparator_model,
        ) in valid_contrasts.items():
            row = {
                "record_type": "contrast",
                "protocol": protocol,
                "contrast_id": contrast_id,
                "first_model": first_model,
                "comparator_model": comparator_model,
                "cluster_unit": "global scaffold_id",
                "n_clusters": int(len(unique_clusters)),
                "n_domains": int(len(domain_order)),
                "n_bootstrap": int(n_bootstrap),
                "delta_direction": (
                    "delta_spearman=first-comparator; "
                    "delta_rmse=comparator-first; positive favors first"
                ),
            }
            for metric in metric_names:
                delta_metric = f"delta_{metric}"
                first_value = float(observed[first_model][metric])
                comparator_value = float(
                    observed[comparator_model][metric]
                )
                observed_delta = (
                    comparator_value - first_value
                    if metric.endswith("rmse")
                    else first_value - comparator_value
                )
                mean, low, high, n_valid = finite_interval(
                    contrast_distributions[contrast_id][delta_metric]
                )
                row[f"{delta_metric}_observed"] = observed_delta
                row[f"{delta_metric}_bootstrap_mean"] = mean
                row[f"{delta_metric}_ci_low"] = low
                row[f"{delta_metric}_ci_high"] = high
                row[f"{delta_metric}_n_valid"] = n_valid
            aggregate_rows.append(row)
    return pd.DataFrame(aggregate_rows), pd.DataFrame(domain_rows)


def scientific_configuration(
    args: argparse.Namespace,
    data_sha256: str,
    script_sha256: str,
    protocol_sha256: str,
    core_script_sha256: str,
    core_protocol_sha256: str,
    frozen_identity_sha256: str,
    frozen_identity_matches: bool,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocols": list(args.protocols),
        "groups_by_protocol": {
            protocol: list(GROUPS_BY_PROTOCOL[protocol])
            for protocol in args.protocols
        },
        "expected_split_counts": expected_counts_json(),
        "data_file": str(args.data_file.resolve()),
        "data_sha256": data_sha256,
        "script_sha256": script_sha256,
        "protocol_document_sha256": protocol_sha256,
        "core_script_sha256": core_script_sha256,
        "core_protocol_document_sha256": core_protocol_sha256,
        "frozen_identity_file": str(FROZEN_IDENTITY_FILE.resolve()),
        "frozen_identity_file_sha256": frozen_identity_sha256,
        "frozen_identity_matches": bool(frozen_identity_matches),
        "inner_folds": int(args.inner_folds),
        "n_estimators": int(args.n_estimators),
        "n_jobs": int(args.n_jobs),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "seed": int(args.seed),
        "max_groups": args.max_groups,
        "skip_label_shuffle": bool(args.skip_label_shuffle),
        "dry_run": bool(args.dry_run),
        "model_order": list(MODEL_ORDER),
        "model_grids": {
            model_id: model_param_grid(model_id)
            for model_id in TUNED_MODEL_ORDER
        },
        "prespecified_contrasts": {
            key: list(value)
            for key, value in PRESPECIFIED_CONTRASTS.items()
        },
        "split_rules": {
            "test": "exact standardized domain label",
            "domain_deletion": (
                "remove training rows with held token membership or "
                "token intersection"
            ),
            "compound_deletion": (
                "remove every candidate-training row whose canonical_smiles "
                "occurs in the exact test domain"
            ),
            "inner_selection": (
                "four-fold scaffold-disjoint pooled inner-OOF Spearman "
                "for frozen formal run"
            ),
        },
        "runtime_versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn_version,
            "rdkit": rdBase.rdkitVersion,
            "joblib": joblib.__version__,
        },
    }


def environment_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn_version,
        "rdkit": rdBase.rdkitVersion,
        "joblib": joblib.__version__,
        "cpu_count_visible": os.cpu_count(),
        "n_jobs_used_per_fit": int(args.n_jobs),
        "gpu_required": False,
    }


def clear_known_artifacts(output_dir: Path) -> int:
    filenames = {
        "ood_run_manifest.json",
        "ood_analysis_index.csv",
        "ood_context_sanitization_audit.csv",
        "ood_split_audit.csv",
        "ood_aggregate_metrics.csv",
        "ood_paired_scaffold_bootstrap.csv",
        "ood_domain_scaffold_bootstrap.csv",
        CHECKPOINT_INVENTORY_FILE,
        *PROGRESS_FILES.values(),
    }
    removed = 0
    for filename in filenames:
        path = output_dir / filename
        temporary = path.with_name(f".{path.name}.tmp")
        for candidate in (path, temporary):
            if candidate.is_file():
                candidate.unlink()
                removed += 1
    return removed


def main() -> None:
    args = parse_args()
    data_path = args.data_file.resolve()
    required_files = (
        data_path,
        Path(__file__).resolve(),
        OOD_PROTOCOL_DOCUMENT,
        FROZEN_IDENTITY_FILE,
        CORE_SCRIPT,
        CORE_PROTOCOL_DOCUMENT,
    )
    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(path)

    data_sha256 = core.sha256_file(data_path)
    if data_sha256 != FROZEN_DATA_SHA256:
        raise ValueError(
            "OOD protocol refuses non-frozen input data: "
            f"observed={data_sha256}, expected={FROZEN_DATA_SHA256}"
        )
    script_sha256 = core.sha256_file(Path(__file__).resolve())
    protocol_sha256 = core.sha256_file(OOD_PROTOCOL_DOCUMENT)
    core_script_sha256 = core.sha256_file(CORE_SCRIPT)
    core_protocol_sha256 = core.sha256_file(CORE_PROTOCOL_DOCUMENT)
    frozen_identity, frozen_identity_sha256 = load_frozen_identity(
        FROZEN_IDENTITY_FILE
    )
    observed_identity = observed_frozen_identity(
        args,
        data_sha256=data_sha256,
        script_sha256=script_sha256,
        protocol_sha256=protocol_sha256,
        core_script_sha256=core_script_sha256,
        core_protocol_sha256=core_protocol_sha256,
    )
    matches_frozen = frozen_protocol_match(
        observed_identity,
        frozen_identity,
    )
    identity_mismatches = frozen_identity_mismatches(
        observed_identity,
        frozen_identity,
    )
    if formal_cli_requested(args) and not matches_frozen:
        raise RuntimeError(
            "Formal OOD arguments were requested, but the external frozen "
            "identity does not match. Mismatches: "
            + ", ".join(identity_mismatches)
        )
    configuration = scientific_configuration(
        args,
        data_sha256,
        script_sha256,
        protocol_sha256,
        core_script_sha256,
        core_protocol_sha256,
        frozen_identity_sha256,
        matches_frozen,
    )
    configuration_sha256 = canonical_json_sha256(configuration)

    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "ood_run_manifest.json"
    existing_manifest: dict[str, Any] | None = None
    output_is_nonempty = bool(
        output_dir.exists() and any(output_dir.iterdir())
    )
    if output_is_nonempty and args.resume:
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Cannot resume without {manifest_path}"
            )
        existing_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if existing_manifest.get("scientific_configuration") != configuration:
            raise ValueError(
                "Resume refused: data, OOD/core code, protocols, runtime "
                "versions, groups or scientific arguments changed."
            )
        if (
            existing_manifest.get("scientific_configuration_sha256")
            != configuration_sha256
        ):
            raise ValueError(
                "Resume refused: scientific configuration digest changed."
            )
        verify_checkpoint_inventory(
            output_dir,
            configuration_sha256,
        )
    elif output_is_nonempty and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use a new directory, --resume, or --overwrite."
        )
    elif output_is_nonempty and args.overwrite:
        removed = clear_known_artifacts(output_dir)
        print(
            f"[OVERWRITE] removed {removed} known OOD artifacts",
            flush=True,
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_started = time.time()
    rows, context_audit = core.load_rows(data_path)
    y = rows["pDC50"].to_numpy(dtype=np.float32)
    scaffold_ids = core.build_scaffold_ids(rows["canonical_smiles"])
    rows["confirmatory_scaffold_id"] = scaffold_ids
    splits, split_audit = build_ood_splits(
        rows,
        scaffold_ids,
        args.protocols,
        args.seed,
        args.max_groups,
    )
    if len(splits) == 0:
        raise RuntimeError("No OOD splits selected")

    print(
        (
            f"[DATA] rows={len(rows)} unique_smiles="
            f"{rows['canonical_smiles'].nunique()} scaffolds="
            f"{len(np.unique(scaffold_ids))} ood_groups={len(splits)}"
        ),
        flush=True,
    )
    print("[FEAT] building Morgan fingerprints", flush=True)
    morgan = core.build_morgan_matrix(rows["canonical_smiles"])
    print("[FEAT] building RDKit descriptors", flush=True)
    descriptors, descriptor_names, descriptor_failures = (
        core.build_descriptor_matrix(rows["canonical_smiles"])
    )

    core.atomic_to_csv(
        context_audit,
        output_dir / "ood_context_sanitization_audit.csv",
    )
    core.atomic_to_csv(
        split_audit,
        output_dir / "ood_split_audit.csv",
    )
    analysis_index = rows[
        [
            "qc_id",
            "canonical_smiles",
            "confirmatory_scaffold_id",
            "source_database",
            "recruiting_protein",
            "target_protein",
            "cell_line",
            "pDC50",
        ]
    ].copy()
    analysis_index.insert(
        0,
        "row_index",
        np.arange(len(analysis_index), dtype=np.int64),
    )
    core.atomic_to_csv(
        analysis_index,
        output_dir / "ood_analysis_index.csv",
    )

    analysis_mode = (
        "dry_run"
        if args.dry_run
        else (
            "smoke_test"
            if args.max_groups is not None
            else ("confirmatory" if matches_frozen else "custom")
        )
    )
    manifest: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "dry_run" if args.dry_run else "running",
        "analysis_mode": analysis_mode,
        "frozen_protocol_match": matches_frozen,
        "frozen_identity_file": str(FROZEN_IDENTITY_FILE.resolve()),
        "frozen_identity_file_sha256": frozen_identity_sha256,
        "frozen_identity_mismatches": identity_mismatches,
        "scientific_configuration": configuration,
        "scientific_configuration_sha256": configuration_sha256,
        "resume_count": (
            int(existing_manifest.get("resume_count", 0)) + 1
            if existing_manifest is not None
            else 0
        ),
        "run_session_started_unix": run_started,
        "n_rows": int(len(rows)),
        "n_unique_smiles": int(rows["canonical_smiles"].nunique()),
        "n_scaffolds": int(len(np.unique(scaffold_ids))),
        "n_ood_groups": int(len(splits)),
        "endpoint": "pDC50 = 9 - log10(DC50_nM)",
        "independent_unit_and_dependence": {
            "test_domains": (
                "four frozen sources or eight frozen exact target labels"
            ),
            "row_dependence": (
                "preserved by global Bemis-Murcko scaffold bootstrap"
            ),
            "folds_or_seeds_are_independent_replicates": False,
        },
        "models": list(MODEL_ORDER),
        "reference_model": REFERENCE_MODEL,
        "primary_metric": (
            "equal-domain-weighted macro Spearman"
        ),
        "feature_manifest": core.feature_manifest(
            descriptor_names,
            descriptor_failures,
        ),
        "environment": environment_manifest(args),
        "boundaries": [
            "Historical fixed train/test labels are ignored.",
            "All held-domain test compounds are removed from final training.",
            "Related composite source/target memberships are removed from training.",
            "No core-run hyperparameter selection is reused.",
            "Protein-language-model and target-encoding features are excluded.",
            "Label shuffle is a sanity control, not a permutation p-value.",
            "This is retrospective internal OOD validation, not prospective validation.",
        ],
    }
    core.write_json(manifest_path, manifest)
    if args.dry_run:
        print(
            f"[DONE] OOD dry-run artifacts written to {output_dir}",
            flush=True,
        )
        return

    try:
        predictions = run_splits(
            args,
            rows,
            y,
            morgan,
            descriptors,
            scaffold_ids,
            splits,
            output_dir,
            configuration_sha256,
        )
        verify_checkpoint_inventory(
            output_dir,
            configuration_sha256,
        )
        predictions_complete = complete_predictions(
            predictions,
            splits,
            rows,
            y,
            scaffold_ids,
            args.skip_label_shuffle,
        )
        manifest["completeness"] = {
            "predictions": predictions_complete,
            "n_prediction_rows": int(len(predictions)),
            "expected_prediction_rows": int(
                sum(len(split.test_idx) for split in splits)
                * len(output_model_ids(args.skip_label_shuffle))
            ),
        }
        if not predictions_complete:
            raise RuntimeError(
                "OOD prediction completeness checks failed"
            )

        aggregate_metrics = summarize_aggregate_metrics(predictions)
        core.atomic_to_csv(
            aggregate_metrics,
            output_dir / "ood_aggregate_metrics.csv",
        )
        aggregate_bootstrap = pd.DataFrame()
        domain_bootstrap = pd.DataFrame()
        if (
            args.max_groups is None
            and args.bootstrap_replicates > 0
        ):
            aggregate_bootstrap, domain_bootstrap = (
                paired_scaffold_bootstrap(
                    predictions,
                    args.bootstrap_replicates,
                    args.seed,
                )
            )
            core.atomic_to_csv(
                aggregate_bootstrap,
                output_dir / "ood_paired_scaffold_bootstrap.csv",
            )
            core.atomic_to_csv(
                domain_bootstrap,
                output_dir / "ood_domain_scaffold_bootstrap.csv",
            )
        elif args.bootstrap_replicates > 0:
            print(
                "[WARN] bootstrap skipped for max-groups smoke run",
                flush=True,
            )
        if matches_frozen and (
            aggregate_bootstrap.empty or domain_bootstrap.empty
        ):
            raise RuntimeError(
                "Frozen OOD run has no complete bootstrap artifacts"
            )

        artifacts = {
            "analysis_index": "ood_analysis_index.csv",
            "context_sanitization_audit": (
                "ood_context_sanitization_audit.csv"
            ),
            "split_audit": "ood_split_audit.csv",
            "predictions": PROGRESS_FILES["predictions"],
            "domain_metrics": PROGRESS_FILES["domain_metrics"],
            "inner_tuning_metrics": PROGRESS_FILES["tuning"],
            "selected_hyperparameters": PROGRESS_FILES["selected"],
            "inner_split_audit": PROGRESS_FILES["inner_audit"],
            "checkpoint_inventory": CHECKPOINT_INVENTORY_FILE,
            "aggregate_metrics": "ood_aggregate_metrics.csv",
            "paired_scaffold_bootstrap": (
                "ood_paired_scaffold_bootstrap.csv"
                if not aggregate_bootstrap.empty
                else None
            ),
            "domain_scaffold_bootstrap": (
                "ood_domain_scaffold_bootstrap.csv"
                if not domain_bootstrap.empty
                else None
            ),
        }
        if matches_frozen:
            expected_formal_rows = {
                PROGRESS_FILES["predictions"]: 22_480,
                PROGRESS_FILES["domain_metrics"]: 96,
                PROGRESS_FILES["tuning"]: 432,
                PROGRESS_FILES["selected"]: 84,
                PROGRESS_FILES["inner_audit"]: 48,
                "ood_aggregate_metrics.csv": 16,
                "ood_paired_scaffold_bootstrap.csv": 26,
                "ood_domain_scaffold_bootstrap.csv": 96,
            }
            for filename, expected_rows in expected_formal_rows.items():
                observed_rows = len(pd.read_csv(output_dir / filename))
                if observed_rows != expected_rows:
                    raise RuntimeError(
                        "Frozen OOD artifact row count changed for "
                        f"{filename}: observed={observed_rows}, "
                        f"expected={expected_rows}"
                    )
        artifact_paths = [
            output_dir / filename
            for filename in artifacts.values()
            if filename is not None
        ]
        artifact_inventory = build_file_inventory(artifact_paths)
        verify_file_inventory(
            output_dir,
            artifact_inventory,
            {path.name for path in artifact_paths},
        )
        manifest["artifacts"] = artifacts
        manifest["artifact_inventory"] = artifact_inventory
        manifest["status"] = "complete"
        manifest["elapsed_seconds"] = float(time.time() - run_started)
        core.write_json(manifest_path, manifest)
        print(
            (
                f"[DONE] protocols={args.protocols} "
                f"groups={len(splits)} elapsed="
                f"{manifest['elapsed_seconds']:.1f}s output={output_dir}"
            ),
            flush=True,
        )
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["elapsed_seconds"] = float(time.time() - run_started)
        manifest["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        core.write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
