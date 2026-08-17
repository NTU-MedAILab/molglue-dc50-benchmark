#!/usr/bin/env python
"""Post-hoc strict domain-plus-scaffold-cold OOD sensitivity analysis.

The frozen confirmatory OOD test domains and test rows are reproduced exactly.
Starting from every frozen domain-plus-compound-cold training set, all rows
whose Bemis--Murcko scaffold occurs in the test domain are additionally
deleted. Three matched ExtraTrees feature blocks are independently inner-tuned
and refit. This runner is always post-hoc sensitivity analysis; it never emits
a confirmatory analysis label.
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

import run_confirmatory_cpu_v1 as core
import run_confirmatory_ood_cpu_v1 as ood


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
CORE_SCRIPT = SCRIPT_DIR / "run_confirmatory_cpu_v1.py"
OOD_SCRIPT = SCRIPT_DIR / "run_confirmatory_ood_cpu_v1.py"
CORE_PROTOCOL = PROJECT_DIR / "docs" / "confirmatory_protocol_v1.md"
OOD_PROTOCOL = PROJECT_DIR / "docs" / "confirmatory_ood_protocol_v1.md"
OOD_IDENTITY = (
    PROJECT_DIR / "docs" / "confirmatory_ood_cpu_v1_frozen_identity.json"
)
OOD_RESULT_DIR = PROJECT_DIR / "reports" / "confirmatory_ood_cpu_v1"
OOD_RESULT_MANIFEST = OOD_RESULT_DIR / "ood_run_manifest.json"
OOD_RESULT_SPLIT_AUDIT = OOD_RESULT_DIR / "ood_split_audit.csv"
EXTENSION_PROTOCOL = (
    PROJECT_DIR / "docs" / "post_hoc_strict_domain_scaffold_ood_protocol_v1.md"
)
EXTENSION_IDENTITY = (
    PROJECT_DIR
    / "docs"
    / "post_hoc_strict_domain_scaffold_ood_cpu_v1_frozen_identity.json"
)
DATA_FILE = core.DATA_FILE
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "reports" / "post_hoc_strict_domain_scaffold_ood_cpu_v1"
)

PROTOCOL_VERSION = "post_hoc_strict_domain_scaffold_ood_cpu_v1.0"
IDENTITY_VERSION = (
    "post_hoc_strict_domain_scaffold_ood_cpu_v1.identity.v1"
)
CHECKPOINT_VERSION = (
    "post_hoc_strict_domain_scaffold_ood_cpu_v1.checkpoint.v1"
)
DEFAULT_SEED = 260531
PROTOCOL_ORDER = tuple(ood.PROTOCOL_ORDER)
GROUPS_BY_PROTOCOL = ood.GROUPS_BY_PROTOCOL
MODEL_ORDER = (
    ood.CONTEXT_EXTRA_TREES,
    "chemistry_extra_trees",
    "full_context_extra_trees",
)
PARENT_OOD_MODEL_POSITIONS = {
    model_id: int(ood.MODEL_ORDER.index(model_id))
    for model_id in MODEL_ORDER
}
if PARENT_OOD_MODEL_POSITIONS != {
    ood.CONTEXT_EXTRA_TREES: 3,
    "chemistry_extra_trees": 5,
    "full_context_extra_trees": 6,
}:
    raise RuntimeError("Frozen parent OOD model positions changed")
FULL_MODEL = "full_context_extra_trees"
CONTRASTS = {
    "full_vs_context_extra_trees_matched": (
        FULL_MODEL,
        ood.CONTEXT_EXTRA_TREES,
    ),
    "full_vs_chemistry_extra_trees": (
        FULL_MODEL,
        "chemistry_extra_trees",
    ),
}
PROTOCOL_SEED_OFFSET = dict(ood.PROTOCOL_SEED_OFFSET)
PRIMARY_METRIC = core.PRIMARY_METRIC

PROGRESS_FILES = {
    "predictions": "strict_ood_predictions.csv",
    "domain_metrics": "strict_ood_domain_metrics.csv",
    "tuning": "strict_ood_inner_tuning_metrics.csv",
    "selected": "strict_ood_selected_hyperparameters.csv",
    "inner_audit": "strict_ood_inner_split_audit.csv",
}
CHECKPOINT_FILE = "strict_ood_checkpoint_inventory.json"
FINAL_FILES = {
    "manifest": "strict_ood_run_manifest.json",
    "analysis_index": "strict_ood_analysis_index.csv",
    "context_audit": "strict_ood_context_sanitization_audit.csv",
    "split_audit": "strict_ood_split_audit.csv",
    "aggregate_metrics": "strict_ood_aggregate_metrics.csv",
    "bootstrap": "strict_ood_paired_global_scaffold_bootstrap.csv",
}
EXPECTED_FORMAL_ROWS = {
    PROGRESS_FILES["predictions"]: 8_430,
    PROGRESS_FILES["domain_metrics"]: 36,
    PROGRESS_FILES["tuning"]: 216,
    PROGRESS_FILES["selected"]: 36,
    PROGRESS_FILES["inner_audit"]: 48,
    FINAL_FILES["aggregate_metrics"]: 6,
    FINAL_FILES["bootstrap"]: 10,
}

IDENTITY_HASH_KEYS = {
    "data",
    "extension_script",
    "extension_protocol",
    "core_script",
    "core_protocol",
    "ood_script",
    "ood_protocol",
    "ood_identity",
    "ood_result_manifest",
    "ood_result_split_audit",
}
STRICT_EXPECTATION_FIELDS = {
    "n_test_rows",
    "n_test_unique_smiles",
    "n_test_scaffolds",
    "n_base_train_rows",
    "n_scaffold_overlap_rows_excluded",
    "n_train_rows",
    "n_train_unique_smiles",
    "n_train_scaffolds",
    "n_train_test_scaffold_overlap",
    "base_train_indices_sha256",
    "test_indices_sha256",
    "scaffold_excluded_indices_sha256",
    "strict_train_indices_sha256",
}


@dataclass(frozen=True)
class StrictOODSplit:
    protocol: str
    heldout_group: str
    group_position: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    scaffold_excluded_idx: np.ndarray
    split_seed: int
    audit: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=list(PROTOCOL_ORDER),
        default=list(PROTOCOL_ORDER),
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
        help="Smoke helper: first N groups in each requested protocol.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify identities, data, features and strict splits without fitting.",
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
        "dry_run": False,
    }


def load_extension_identity(
    path: Path,
) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Extension identity must be a JSON object")
    if payload.get("identity_version") != IDENTITY_VERSION:
        raise ValueError("Unexpected extension identity version")
    frozen = payload.get("frozen_identity")
    if not isinstance(frozen, dict):
        raise ValueError("Extension identity has no frozen_identity object")
    if set(frozen) != {
        "protocol_version",
        "file_sha256",
        "formal_arguments",
        "strict_split_expectations",
    }:
        raise ValueError("Extension frozen identity keys changed")
    if set(frozen["file_sha256"]) != IDENTITY_HASH_KEYS:
        raise ValueError("Extension file-hash inventory is incomplete")
    if frozen["formal_arguments"] != frozen_formal_arguments():
        raise ValueError("Frozen formal arguments changed inside identity")
    expected_keys = {
        f"{protocol}::{group}"
        for protocol in PROTOCOL_ORDER
        for group in GROUPS_BY_PROTOCOL[protocol]
    }
    expectations = frozen["strict_split_expectations"]
    if set(expectations) != expected_keys:
        raise ValueError("Frozen strict split domains changed")
    for key, record in expectations.items():
        if not isinstance(record, dict) or set(record) != STRICT_EXPECTATION_FIELDS:
            raise ValueError(f"Frozen strict fields changed for {key}")
    return frozen, core.sha256_file(path)


def observed_file_hashes(data_path: Path) -> dict[str, str]:
    return {
        "data": core.sha256_file(data_path),
        "extension_script": core.sha256_file(Path(__file__).resolve()),
        "extension_protocol": core.sha256_file(EXTENSION_PROTOCOL),
        "core_script": core.sha256_file(CORE_SCRIPT),
        "core_protocol": core.sha256_file(CORE_PROTOCOL),
        "ood_script": core.sha256_file(OOD_SCRIPT),
        "ood_protocol": core.sha256_file(OOD_PROTOCOL),
        "ood_identity": core.sha256_file(OOD_IDENTITY),
        "ood_result_manifest": core.sha256_file(OOD_RESULT_MANIFEST),
        "ood_result_split_audit": core.sha256_file(OOD_RESULT_SPLIT_AUDIT),
    }


def verify_parent_ood_result() -> None:
    manifest = json.loads(OOD_RESULT_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("Frozen parent OOD result is not complete")
    if manifest.get("analysis_mode") != "confirmatory":
        raise RuntimeError("Frozen parent OOD result is not confirmatory")
    if manifest.get("frozen_protocol_match") is not True:
        raise RuntimeError("Frozen parent OOD identity did not match")
    completeness = manifest.get("completeness", {})
    if completeness.get("predictions") is not True:
        raise RuntimeError("Frozen parent OOD predictions are incomplete")
    split_audit = pd.read_csv(OOD_RESULT_SPLIT_AUDIT)
    if len(split_audit) != 12:
        raise RuntimeError("Frozen parent OOD split audit must have 12 rows")
    keys = set(
        zip(
            split_audit["protocol"].astype(str),
            split_audit["heldout_group"].astype(str),
        )
    )
    expected_keys = {
        (protocol, group)
        for protocol in PROTOCOL_ORDER
        for group in GROUPS_BY_PROTOCOL[protocol]
    }
    if keys != expected_keys:
        raise RuntimeError("Frozen parent OOD split audit domains changed")


def verify_identity(
    args: argparse.Namespace,
    frozen: dict[str, Any],
    observed_hashes: dict[str, str],
) -> tuple[bool, list[str]]:
    mismatches: list[str] = []
    if frozen.get("protocol_version") != PROTOCOL_VERSION:
        mismatches.append("protocol_version")
    for key in sorted(IDENTITY_HASH_KEYS):
        if frozen["file_sha256"].get(key) != observed_hashes.get(key):
            mismatches.append(f"file_sha256.{key}")
    if mismatches:
        raise RuntimeError(
            "Post-hoc extension source/parent identity mismatch; failing "
            "closed: " + ", ".join(mismatches)
        )
    formal_match = (
        formal_argument_snapshot(args) == frozen["formal_arguments"]
    )
    if formal_argument_snapshot(args) == frozen_formal_arguments() and not formal_match:
        raise RuntimeError("Formal arguments do not match external identity")
    argument_mismatches = [] if formal_match else ["formal_arguments"]
    return formal_match, argument_mismatches


def strict_expected(
    frozen_identity: dict[str, Any],
    protocol: str,
    heldout_group: str,
) -> dict[str, Any]:
    return dict(
        frozen_identity["strict_split_expectations"][
            f"{protocol}::{heldout_group}"
        ]
    )


def build_strict_splits(
    rows: pd.DataFrame,
    scaffold_ids: np.ndarray,
    protocols: list[str],
    seed: int,
    max_groups: int | None,
    frozen_identity: dict[str, Any],
) -> tuple[list[StrictOODSplit], pd.DataFrame]:
    base_splits, _ = ood.build_ood_splits(
        rows,
        scaffold_ids,
        protocols,
        seed,
        max_groups,
    )
    strict_splits: list[StrictOODSplit] = []
    audit_rows: list[dict[str, Any]] = []
    smiles = rows["canonical_smiles"].astype(str)
    for base in base_splits:
        test_scaffolds = set(
            scaffold_ids[base.test_idx].astype(str).tolist()
        )
        exclude_mask = np.isin(
            scaffold_ids[base.train_idx].astype(str),
            list(test_scaffolds),
        )
        excluded_idx = np.asarray(
            base.train_idx[exclude_mask],
            dtype=np.int64,
        )
        strict_train_idx = np.asarray(
            base.train_idx[~exclude_mask],
            dtype=np.int64,
        )
        test_idx = np.asarray(base.test_idx, dtype=np.int64)
        if len(strict_train_idx) == 0:
            raise RuntimeError(
                f"Strict training set is empty: {base.protocol} "
                f"{base.heldout_group}"
            )
        if set(strict_train_idx).intersection(set(test_idx)):
            raise RuntimeError("Strict train/test row overlap")
        if set(smiles.iloc[strict_train_idx]).intersection(
            set(smiles.iloc[test_idx])
        ):
            raise RuntimeError("Strict train/test compound overlap")
        train_scaffolds = set(
            scaffold_ids[strict_train_idx].astype(str).tolist()
        )
        scaffold_overlap = train_scaffolds.intersection(test_scaffolds)
        if scaffold_overlap:
            raise RuntimeError("Strict train/test scaffold overlap")
        domain_column = (
            "source_database"
            if base.protocol == "source_ood"
            else "target_protein"
        )
        held_tokens = ood.tokenize_domain(base.heldout_group)
        train_tokens: set[str] = set()
        for value in rows.iloc[strict_train_idx][domain_column].astype(str):
            train_tokens.update(ood.tokenize_domain(value))
        if held_tokens.intersection(train_tokens):
            raise RuntimeError("Strict held-domain token overlap")

        observed = {
            "n_test_rows": int(len(test_idx)),
            "n_test_unique_smiles": int(
                rows.iloc[test_idx]["canonical_smiles"].nunique()
            ),
            "n_test_scaffolds": int(len(test_scaffolds)),
            "n_base_train_rows": int(len(base.train_idx)),
            "n_scaffold_overlap_rows_excluded": int(len(excluded_idx)),
            "n_train_rows": int(len(strict_train_idx)),
            "n_train_unique_smiles": int(
                rows.iloc[strict_train_idx]["canonical_smiles"].nunique()
            ),
            "n_train_scaffolds": int(len(train_scaffolds)),
            "n_train_test_scaffold_overlap": 0,
            "base_train_indices_sha256": sequence_sha256(
                int(value) for value in base.train_idx
            ),
            "test_indices_sha256": sequence_sha256(
                int(value) for value in test_idx
            ),
            "scaffold_excluded_indices_sha256": sequence_sha256(
                int(value) for value in excluded_idx
            ),
            "strict_train_indices_sha256": sequence_sha256(
                int(value) for value in strict_train_idx
            ),
        }
        expected = strict_expected(
            frozen_identity,
            base.protocol,
            base.heldout_group,
        )
        if observed != expected:
            raise RuntimeError(
                "Frozen strict split changed for "
                f"{base.protocol} {base.heldout_group}: "
                f"observed={observed}, expected={expected}"
            )
        audit = {
            "analysis_label": "post_hoc_sensitivity",
            "protocol": base.protocol,
            "heldout_group": base.heldout_group,
            "heldout_tokens": "|".join(sorted(held_tokens)),
            "domain_column": domain_column,
            "group_position": int(base.group_position),
            "split_seed": int(base.split_seed),
            **observed,
            "n_total_rows": int(len(rows)),
            "base_domain_related_rows_excluded": int(
                base.audit["n_related_rows_excluded"]
            ),
            "base_compound_overlap_rows_excluded": int(
                base.audit["n_compound_overlap_rows_excluded"]
            ),
            "train_test_row_overlap": 0,
            "train_test_smiles_overlap": 0,
            "train_test_scaffold_overlap": 0,
            "heldout_token_seen_in_train": False,
        }
        strict_splits.append(
            StrictOODSplit(
                protocol=base.protocol,
                heldout_group=base.heldout_group,
                group_position=base.group_position,
                train_idx=strict_train_idx,
                test_idx=test_idx,
                scaffold_excluded_idx=excluded_idx,
                split_seed=base.split_seed,
                audit=audit,
            )
        )
        audit_rows.append(audit)
    return strict_splits, pd.DataFrame(audit_rows)


def model_grid(model_id: str) -> list[dict[str, Any]]:
    grid = [dict(record) for record in ood.model_param_grid(model_id)]
    if len(grid) != 6:
        raise RuntimeError(f"{model_id} does not have the frozen six-grid")
    observed = {
        (
            int(params["min_samples_leaf"]),
            str(params["max_features"]),
        )
        for params in grid
    }
    expected = {
        (leaf, str(max_features))
        for leaf in (1, 5, 10)
        for max_features in ("sqrt", 0.3)
    }
    if observed != expected:
        raise RuntimeError(f"{model_id} ExtraTrees grid changed")
    return grid


def select_model_params(
    rows: pd.DataFrame,
    y: np.ndarray,
    morgan: np.ndarray,
    descriptors: np.ndarray,
    train_idx: np.ndarray,
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
        train_idx,
        scaffold_ids,
        inner_folds,
        seed,
    )
    local_position = {
        int(row_idx): position
        for position, row_idx in enumerate(train_idx.tolist())
    }
    candidate_predictions = {
        (model_id, str(params["param_id"])): np.full(
            len(train_idx),
            np.nan,
            dtype=np.float32,
        )
        for model_id in MODEL_ORDER
        for params in model_grid(model_id)
    }
    coverage = np.zeros(len(train_idx), dtype=np.int64)
    inner_audit: list[dict[str, Any]] = []
    for inner_fold, (fit_idx, eval_idx) in enumerate(
        inner_splits,
        start=1,
    ):
        fit_scaffolds = set(
            scaffold_ids[fit_idx].astype(str).tolist()
        )
        eval_scaffolds = set(
            scaffold_ids[eval_idx].astype(str).tolist()
        )
        if fit_scaffolds.intersection(eval_scaffolds):
            raise RuntimeError("Inner scaffold leakage")
        eval_positions = np.asarray(
            [local_position[int(idx)] for idx in eval_idx],
            dtype=np.int64,
        )
        coverage[eval_positions] += 1
        inner_audit.append(
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
                    sorted(fit_scaffolds)
                ),
                "eval_scaffold_ids_sha256": sequence_sha256(
                    sorted(eval_scaffolds)
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
        for model_id in MODEL_ORDER:
            for params in model_grid(model_id):
                pred = ood.fit_predict_model(
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
                candidate_predictions[
                    (model_id, str(params["param_id"]))
                ][eval_positions] = pred
    if not np.all(coverage == 1):
        raise RuntimeError("Inner OOF partitions do not cover training once")

    selected: dict[str, dict[str, Any]] = {}
    tuning_rows: list[dict[str, Any]] = []
    truth = y[train_idx]
    for model_id in MODEL_ORDER:
        scored: list[dict[str, Any]] = []
        for params in model_grid(model_id):
            param_id = str(params["param_id"])
            pred = candidate_predictions[(model_id, param_id)]
            if not np.all(np.isfinite(pred)):
                raise RuntimeError("Incomplete inner OOF prediction")
            metrics = core.regression_metrics(truth, pred)
            score = float(metrics[PRIMARY_METRIC])
            if not math.isfinite(score):
                score = -float("inf")
            rank = ood.regularization_rank(model_id, params)
            scored.append(
                {
                    "score": score,
                    "rmse": float(metrics["rmse"]),
                    "param_id": param_id,
                    "params": params,
                    "rank": rank,
                }
            )
            tuning_rows.append(
                {
                    "model_id": model_id,
                    "param_id": param_id,
                    "inner_n": int(len(truth)),
                    **metrics,
                }
            )
        best = max(record["score"] for record in scored)
        eligible = [
            record
            for record in scored
            if record["score"] >= best - 0.005
        ]
        eligible.sort(
            key=lambda record: (
                tuple(-value for value in record["rank"]),
                record["rmse"],
                record["param_id"],
            )
        )
        selected[model_id] = dict(eligible[0]["params"])
    if len(tuning_rows) != 18:
        raise RuntimeError("Expected 18 tuning rows per strict split")
    return selected, tuning_rows, inner_audit


def split_key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record["protocol"]), str(record["heldout_group"])


def records_for_split(
    records: list[dict[str, Any]],
    split: StrictOODSplit,
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if split_key(record) == (split.protocol, split.heldout_group)
    ]


def discard_split_records(
    records: list[dict[str, Any]],
    split: StrictOODSplit,
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if split_key(record) != (split.protocol, split.heldout_group)
    ]


def validate_split_checkpoint(
    split: StrictOODSplit,
    records: dict[str, list[dict[str, Any]]],
    rows: pd.DataFrame,
    y: np.ndarray,
    scaffold_ids: np.ndarray,
    inner_folds: int,
) -> bool:
    predictions = pd.DataFrame(
        records_for_split(records["predictions"], split)
    )
    metrics = pd.DataFrame(
        records_for_split(records["domain_metrics"], split)
    )
    tuning = pd.DataFrame(records_for_split(records["tuning"], split))
    selected = pd.DataFrame(
        records_for_split(records["selected"], split)
    )
    inner = pd.DataFrame(
        records_for_split(records["inner_audit"], split)
    )
    if any(frame.empty for frame in (predictions, metrics, tuning, selected, inner)):
        return False
    if len(predictions) != len(split.test_idx) * len(MODEL_ORDER):
        return False
    if predictions.duplicated(["model_id", "row_index"]).any():
        return False
    if set(predictions["model_id"].astype(str)) != set(MODEL_ORDER):
        return False
    if not np.all(np.isfinite(predictions["y_pred"].astype(float))):
        return False
    expected_rows = set(split.test_idx.astype(int).tolist())
    for model_id, frame in predictions.groupby("model_id"):
        if set(frame["row_index"].astype(int)) != expected_rows:
            return False
        expected_metric = core.regression_metrics(
            frame.sort_values("row_index")["y_true"].to_numpy(
                dtype=np.float32
            ),
            frame.sort_values("row_index")["y_pred"].to_numpy(
                dtype=np.float32
            ),
        )
        local_metric = metrics[metrics["model_id"] == model_id]
        if len(local_metric) != 1:
            return False
        for metric_name, value in expected_metric.items():
            if not ood.values_match(
                local_metric.iloc[0].get(metric_name),
                value,
                atol=1e-9,
                rtol=1e-9,
            ):
                return False
    for record in predictions.to_dict("records"):
        row_idx = int(record["row_index"])
        source = rows.iloc[row_idx]
        expected = {
            "qc_id": str(source["qc_id"]),
            "y_true": float(y[row_idx]),
            "canonical_smiles": str(source["canonical_smiles"]),
            "scaffold_id": str(scaffold_ids[row_idx]),
            "compound_seen_in_train": False,
            "scaffold_seen_in_train": False,
            "heldout_token_seen_in_train": False,
            "analysis_label": "post_hoc_sensitivity",
        }
        for field, value in expected.items():
            if not ood.values_match(
                record.get(field),
                value,
                atol=1e-7 if field == "y_true" else 1e-10,
                rtol=1e-7 if field == "y_true" else 1e-10,
            ):
                return False
    if len(metrics) != 3 or set(metrics["model_id"].astype(str)) != set(MODEL_ORDER):
        return False
    if len(tuning) != 18 or tuning.duplicated(["model_id", "param_id"]).any():
        return False
    if len(selected) != 3 or selected.duplicated(["model_id"]).any():
        return False
    if set(selected["model_id"].astype(str)) != set(MODEL_ORDER):
        return False
    for model_id in MODEL_ORDER:
        valid_ids = {
            str(params["param_id"]) for params in model_grid(model_id)
        }
        local_tuning = tuning[tuning["model_id"] == model_id]
        if set(local_tuning["param_id"].astype(str)) != valid_ids:
            return False
        local_selected = selected[selected["model_id"] == model_id]
        if str(local_selected.iloc[0]["param_id"]) not in valid_ids:
            return False
    if len(inner) != inner_folds:
        return False
    if set(inner["inner_fold"].astype(int)) != set(
        range(1, inner_folds + 1)
    ):
        return False
    if not (inner["n_scaffold_overlap"].astype(int) == 0).all():
        return False
    return True


def progress_frames(
    records: dict[str, list[dict[str, Any]]],
) -> dict[str, pd.DataFrame]:
    frames = {key: pd.DataFrame(value) for key, value in records.items()}
    sort_keys = {
        "predictions": [
            "protocol",
            "heldout_group",
            "model_id",
            "row_index",
        ],
        "domain_metrics": ["protocol", "heldout_group", "model_id"],
        "tuning": [
            "protocol",
            "heldout_group",
            "model_id",
            "param_id",
        ],
        "selected": ["protocol", "heldout_group", "model_id"],
        "inner_audit": [
            "protocol",
            "heldout_group",
            "inner_fold",
        ],
    }
    for key, frame in frames.items():
        if not frame.empty:
            frames[key] = frame.sort_values(sort_keys[key]).reset_index(
                drop=True
            )
    return frames


def write_progress(
    output_dir: Path,
    records: dict[str, list[dict[str, Any]]],
    configuration_sha256: str,
) -> None:
    frames = progress_frames(records)
    paths: list[Path] = []
    for key, filename in PROGRESS_FILES.items():
        path = output_dir / filename
        core.atomic_to_csv(frames[key], path)
        paths.append(path)
    inventory = {
        "inventory_version": CHECKPOINT_VERSION,
        "scientific_configuration_sha256": configuration_sha256,
        "files": ood.build_file_inventory(paths),
    }
    core.write_json(output_dir / CHECKPOINT_FILE, inventory)


def verify_checkpoint(
    output_dir: Path,
    configuration_sha256: str,
) -> None:
    checkpoint = output_dir / CHECKPOINT_FILE
    progress_paths = [
        output_dir / filename for filename in PROGRESS_FILES.values()
    ]
    existing = [path for path in progress_paths if path.is_file()]
    if not existing and not checkpoint.exists():
        return
    if len(existing) != len(progress_paths) or not checkpoint.is_file():
        raise RuntimeError("Resume refused: checkpoint set is incomplete")
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    if payload.get("inventory_version") != CHECKPOINT_VERSION:
        raise RuntimeError("Resume refused: checkpoint version changed")
    if payload.get("scientific_configuration_sha256") != configuration_sha256:
        raise RuntimeError("Resume refused: configuration digest changed")
    files = payload.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("Resume refused: invalid checkpoint inventory")
    ood.verify_file_inventory(
        output_dir,
        files,
        set(PROGRESS_FILES.values()),
    )


def read_records(output_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        key: (
            pd.read_csv(output_dir / filename).to_dict("records")
            if (output_dir / filename).is_file()
            else []
        )
        for key, filename in PROGRESS_FILES.items()
    }


def run_splits(
    args: argparse.Namespace,
    rows: pd.DataFrame,
    y: np.ndarray,
    morgan: np.ndarray,
    descriptors: np.ndarray,
    scaffold_ids: np.ndarray,
    splits: list[StrictOODSplit],
    output_dir: Path,
    configuration_sha256: str,
) -> pd.DataFrame:
    records = (
        read_records(output_dir)
        if args.resume
        else {key: [] for key in PROGRESS_FILES}
    )
    started = time.perf_counter()
    for split_number, split in enumerate(splits, start=1):
        if args.resume and validate_split_checkpoint(
            split,
            records,
            rows,
            y,
            scaffold_ids,
            args.inner_folds,
        ):
            print(
                f"[RESUME] {split.protocol} {split.heldout_group} complete",
                flush=True,
            )
            continue
        if args.resume:
            for key in records:
                records[key] = discard_split_records(records[key], split)
        print(
            (
                f"[STRICT-OOD] {split.protocol} {split.heldout_group} "
                f"({split_number}/{len(splits)}) train={len(split.train_idx)} "
                f"test={len(split.test_idx)} "
                f"scaffold_deleted={len(split.scaffold_excluded_idx)}"
            ),
            flush=True,
        )
        inner_seed = split.split_seed + 101
        selected, tuning_rows, inner_rows = select_model_params(
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
        for record in tuning_rows:
            records["tuning"].append(
                {
                    "analysis_label": "post_hoc_sensitivity",
                    "protocol": split.protocol,
                    "heldout_group": split.heldout_group,
                    **record,
                }
            )
        for model_id, params in selected.items():
            records["selected"].append(
                {
                    "analysis_label": "post_hoc_sensitivity",
                    "protocol": split.protocol,
                    "heldout_group": split.heldout_group,
                    "model_id": model_id,
                    **params,
                }
            )
        for record in inner_rows:
            records["inner_audit"].append(
                {
                    "analysis_label": "post_hoc_sensitivity",
                    "protocol": split.protocol,
                    "heldout_group": split.heldout_group,
                    **record,
                }
            )

        fold_features, feature_meta = core.build_fold_features(
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
        train_smiles = set(
            rows.iloc[split.train_idx]["canonical_smiles"].astype(str)
        )
        train_scaffolds = set(
            scaffold_ids[split.train_idx].astype(str).tolist()
        )
        held_tokens = ood.tokenize_domain(split.heldout_group)
        domain_column = (
            "source_database"
            if split.protocol == "source_ood"
            else "target_protein"
        )
        train_tokens: set[str] = set()
        for value in rows.iloc[split.train_idx][domain_column].astype(str):
            train_tokens.update(ood.tokenize_domain(value))
        if held_tokens.intersection(train_tokens):
            raise RuntimeError("Held-domain token entered final strict train")

        for model_id in MODEL_ORDER:
            params = selected[model_id]
            pred = ood.fit_predict_model(
                model_id,
                params,
                rows,
                y,
                morgan,
                fold_features,
                split.train_idx,
                split.test_idx,
                seed=(
                    split.split_seed
                    + 1_000
                    + PARENT_OOD_MODEL_POSITIONS[model_id]
                ),
                n_estimators=args.n_estimators,
                n_jobs=args.n_jobs,
            )
            if len(pred) != len(split.test_idx) or not np.all(
                np.isfinite(pred)
            ):
                raise RuntimeError(f"Invalid strict prediction for {model_id}")
            metrics = core.regression_metrics(y[split.test_idx], pred)
            records["domain_metrics"].append(
                {
                    "analysis_label": "post_hoc_sensitivity",
                    "protocol": split.protocol,
                    "heldout_group": split.heldout_group,
                    "model_id": model_id,
                    "param_id": str(params["param_id"]),
                    "n_train_rows": int(len(split.train_idx)),
                    "n_train_unique_smiles": int(
                        rows.iloc[split.train_idx][
                            "canonical_smiles"
                        ].nunique()
                    ),
                    "n_train_scaffolds": int(len(train_scaffolds)),
                    "n_test_rows": int(len(split.test_idx)),
                    "n_test_unique_smiles": int(
                        rows.iloc[split.test_idx][
                            "canonical_smiles"
                        ].nunique()
                    ),
                    "n_test_scaffolds": int(
                        len(
                            np.unique(
                                scaffold_ids[split.test_idx].astype(str)
                            )
                        )
                    ),
                    "n_scaffold_overlap_rows_excluded": int(
                        len(split.scaffold_excluded_idx)
                    ),
                    **metrics,
                    **feature_meta,
                }
            )
            for local_idx, row_idx in enumerate(split.test_idx):
                row = rows.iloc[row_idx]
                smiles_value = str(row["canonical_smiles"])
                scaffold_value = str(scaffold_ids[row_idx])
                if smiles_value in train_smiles:
                    raise RuntimeError("Test compound entered strict training")
                if scaffold_value in train_scaffolds:
                    raise RuntimeError("Test scaffold entered strict training")
                records["predictions"].append(
                    {
                        "analysis_label": "post_hoc_sensitivity",
                        "protocol": split.protocol,
                        "heldout_group": split.heldout_group,
                        "row_index": int(row_idx),
                        "qc_id": str(row["qc_id"]),
                        "model_id": model_id,
                        "param_id": str(params["param_id"]),
                        "y_true": float(y[row_idx]),
                        "y_pred": float(pred[local_idx]),
                        "canonical_smiles": smiles_value,
                        "scaffold_id": scaffold_value,
                        "source_database": str(row["source_database"]),
                        "recruiting_protein": str(
                            row["recruiting_protein"]
                        ),
                        "target_protein": str(row["target_protein"]),
                        "cell_line": str(row["cell_line"]),
                        "max_train_tanimoto": float(
                            max_tanimoto[local_idx]
                        ),
                        "compound_seen_in_train": False,
                        "scaffold_seen_in_train": False,
                        "heldout_token_seen_in_train": False,
                    }
                )
        write_progress(output_dir, records, configuration_sha256)
        if not validate_split_checkpoint(
            split,
            records,
            rows,
            y,
            scaffold_ids,
            args.inner_folds,
        ):
            raise RuntimeError("Just-written strict checkpoint is incomplete")
        print(
            (
                f"[STRICT-OOD] completed {split_number}/{len(splits)} "
                f"in {time.perf_counter() - started:.1f}s"
            ),
            flush=True,
        )
    return progress_frames(records)["predictions"]


def complete_predictions(
    predictions: pd.DataFrame,
    splits: list[StrictOODSplit],
) -> bool:
    if predictions.empty:
        return False
    if predictions.duplicated(
        ["protocol", "heldout_group", "model_id", "row_index"]
    ).any():
        return False
    if not np.all(np.isfinite(predictions["y_pred"].astype(float))):
        return False
    if set(predictions["analysis_label"].astype(str)) != {
        "post_hoc_sensitivity"
    }:
        return False
    for split in splits:
        local = predictions[
            (predictions["protocol"] == split.protocol)
            & (predictions["heldout_group"] == split.heldout_group)
        ]
        if len(local) != len(split.test_idx) * len(MODEL_ORDER):
            return False
        if set(local["model_id"].astype(str)) != set(MODEL_ORDER):
            return False
        for _, frame in local.groupby("model_id"):
            if set(frame["row_index"].astype(int)) != set(
                split.test_idx.astype(int).tolist()
            ):
                return False
    return True


def summarize_aggregate_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows_out: list[dict[str, Any]] = []
    for protocol in PROTOCOL_ORDER:
        protocol_df = predictions[predictions["protocol"] == protocol]
        if protocol_df.empty:
            continue
        domain_order = tuple(
            group
            for group in GROUPS_BY_PROTOCOL[protocol]
            if group in set(protocol_df["heldout_group"].astype(str))
        )
        for model_id in MODEL_ORDER:
            frame = (
                protocol_df[protocol_df["model_id"] == model_id]
                .sort_values("row_index")
                .reset_index(drop=True)
            )
            positions = np.arange(len(frame), dtype=np.int64)
            bundle = ood.metric_bundle(
                frame["y_true"].to_numpy(dtype=np.float64),
                frame["y_pred"].to_numpy(dtype=np.float64),
                frame["heldout_group"].astype(str).to_numpy(),
                positions,
                domain_order,
            )
            rows_out.append(
                {
                    "analysis_label": "post_hoc_sensitivity",
                    "protocol": protocol,
                    "model_id": model_id,
                    "n_rows": int(len(frame)),
                    "n_domains": int(len(domain_order)),
                    "n_global_scaffolds": int(
                        frame["scaffold_id"].astype(str).nunique()
                    ),
                    "domain_macro_spearman": float(
                        bundle["domain_macro_spearman"]
                    ),
                    "domain_macro_rmse": float(
                        bundle["domain_macro_rmse"]
                    ),
                    "pooled_spearman": float(bundle["pooled_spearman"]),
                    "pooled_rmse": float(bundle["pooled_rmse"]),
                }
            )
    return pd.DataFrame(rows_out)


def paired_global_scaffold_bootstrap(
    predictions: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    if n_bootstrap <= 0:
        return pd.DataFrame()
    output: list[dict[str, Any]] = []
    metric_names = (
        "domain_macro_spearman",
        "domain_macro_rmse",
        "pooled_spearman",
        "pooled_rmse",
    )
    for protocol in PROTOCOL_ORDER:
        protocol_df = predictions[predictions["protocol"] == protocol]
        if protocol_df.empty:
            continue
        domain_order = tuple(
            group
            for group in GROUPS_BY_PROTOCOL[protocol]
            if group in set(protocol_df["heldout_group"].astype(str))
        )
        reference = (
            protocol_df[
                protocol_df["model_id"] == MODEL_ORDER[0]
            ]
            .sort_values("row_index")
            .reset_index(drop=True)
        )
        if reference["row_index"].duplicated().any():
            raise RuntimeError("Bootstrap reference rows are duplicated")
        row_indices = reference["row_index"].to_numpy(dtype=np.int64)
        y_true = reference["y_true"].to_numpy(dtype=np.float64)
        domains = reference["heldout_group"].astype(str).to_numpy()
        clusters = reference["scaffold_id"].astype(str).to_numpy()
        unique_clusters = np.unique(clusters)
        cluster_positions = {
            cluster: np.where(clusters == cluster)[0]
            for cluster in unique_clusters
        }
        model_predictions: dict[str, np.ndarray] = {}
        observed: dict[str, dict[str, Any]] = {}
        for model_id in MODEL_ORDER:
            frame = (
                protocol_df[protocol_df["model_id"] == model_id]
                .sort_values("row_index")
                .reset_index(drop=True)
            )
            if not np.array_equal(
                frame["row_index"].to_numpy(dtype=np.int64),
                row_indices,
            ):
                raise RuntimeError("Bootstrap model rows are not aligned")
            if not np.array_equal(
                frame["y_true"].to_numpy(dtype=np.float64),
                y_true,
            ):
                raise RuntimeError("Bootstrap outcomes differ across models")
            pred = frame["y_pred"].to_numpy(dtype=np.float64)
            model_predictions[model_id] = pred
            observed[model_id] = ood.metric_bundle(
                y_true,
                pred,
                domains,
                np.arange(len(y_true), dtype=np.int64),
                domain_order,
            )
        model_distributions = {
            model_id: {metric: [] for metric in metric_names}
            for model_id in MODEL_ORDER
        }
        contrast_distributions = {
            contrast_id: {
                f"delta_{metric}": [] for metric in metric_names
            }
            for contrast_id in CONTRASTS
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
            positions = np.concatenate(
                [cluster_positions[cluster] for cluster in sampled_clusters]
            )
            stats: dict[str, dict[str, Any]] = {}
            for model_id in MODEL_ORDER:
                bundle = ood.metric_bundle(
                    y_true,
                    model_predictions[model_id],
                    domains,
                    positions,
                    domain_order,
                )
                stats[model_id] = bundle
                for metric in metric_names:
                    model_distributions[model_id][metric].append(
                        float(bundle[metric])
                    )
            for contrast_id, (first, comparator) in CONTRASTS.items():
                for metric in metric_names:
                    value = (
                        stats[comparator][metric] - stats[first][metric]
                        if metric.endswith("rmse")
                        else stats[first][metric] - stats[comparator][metric]
                    )
                    contrast_distributions[contrast_id][
                        f"delta_{metric}"
                    ].append(float(value))
            if (bootstrap_idx + 1) % max(1, n_bootstrap // 10) == 0:
                print(
                    f"[BOOT] {protocol} {bootstrap_idx + 1}/{n_bootstrap}",
                    flush=True,
                )
        for model_id in MODEL_ORDER:
            record: dict[str, Any] = {
                "analysis_label": "post_hoc_sensitivity",
                "record_type": "model",
                "protocol": protocol,
                "model_id": model_id,
                "cluster_unit": "global scaffold_id",
                "n_clusters": int(len(unique_clusters)),
                "n_domains": int(len(domain_order)),
                "n_bootstrap": int(n_bootstrap),
            }
            for metric in metric_names:
                mean, low, high, n_valid = ood.finite_interval(
                    model_distributions[model_id][metric]
                )
                record[f"{metric}_observed"] = float(
                    observed[model_id][metric]
                )
                record[f"{metric}_bootstrap_mean"] = mean
                record[f"{metric}_ci_low"] = low
                record[f"{metric}_ci_high"] = high
                record[f"{metric}_n_valid"] = n_valid
            output.append(record)
        for contrast_id, (first, comparator) in CONTRASTS.items():
            record = {
                "analysis_label": "post_hoc_sensitivity",
                "record_type": "contrast",
                "protocol": protocol,
                "contrast_id": contrast_id,
                "first_model": first,
                "comparator_model": comparator,
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
                delta_name = f"delta_{metric}"
                observed_delta = (
                    observed[comparator][metric] - observed[first][metric]
                    if metric.endswith("rmse")
                    else observed[first][metric] - observed[comparator][metric]
                )
                mean, low, high, n_valid = ood.finite_interval(
                    contrast_distributions[contrast_id][delta_name]
                )
                record[f"{delta_name}_observed"] = float(observed_delta)
                record[f"{delta_name}_bootstrap_mean"] = mean
                record[f"{delta_name}_ci_low"] = low
                record[f"{delta_name}_ci_high"] = high
                record[f"{delta_name}_n_valid"] = n_valid
            output.append(record)
    return pd.DataFrame(output)


def scientific_configuration(
    args: argparse.Namespace,
    observed_hashes: dict[str, str],
    identity_sha256: str,
    formal_match: bool,
    frozen_identity: dict[str, Any],
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "analysis_label": "post_hoc_sensitivity",
        "protocols": list(args.protocols),
        "groups_by_protocol": {
            protocol: list(GROUPS_BY_PROTOCOL[protocol])
            for protocol in args.protocols
        },
        "data_file": str(args.data_file.resolve()),
        "file_sha256": dict(observed_hashes),
        "extension_identity_file": str(EXTENSION_IDENTITY.resolve()),
        "extension_identity_file_sha256": identity_sha256,
        "external_identity_matches": True,
        "formal_arguments_match": bool(formal_match),
        "strict_split_expectations": frozen_identity[
            "strict_split_expectations"
        ],
        **formal_argument_snapshot(args),
        "models": list(MODEL_ORDER),
        "parent_ood_model_positions": dict(PARENT_OOD_MODEL_POSITIONS),
        "model_grids": {
            model_id: model_grid(model_id) for model_id in MODEL_ORDER
        },
        "contrasts": {
            key: list(value) for key, value in CONTRASTS.items()
        },
        "split_rule": (
            "frozen domain-plus-compound-cold OOD split, followed by "
            "deletion of every base-training row whose scaffold_id occurs "
            "in the unchanged test domain"
        ),
        "selection_rule": (
            "inner scaffold-disjoint pooled Spearman; within 0.005 of best "
            "choose larger leaf, then sqrt, then lower RMSE, then param_id"
        ),
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


def known_artifacts(output_dir: Path) -> list[Path]:
    names = {
        *PROGRESS_FILES.values(),
        CHECKPOINT_FILE,
        *FINAL_FILES.values(),
    }
    return [output_dir / name for name in sorted(names)]


def artifact_manifest_entries(
    include_bootstrap: bool,
) -> dict[str, str | None]:
    return {
        "analysis_index": FINAL_FILES["analysis_index"],
        "context_audit": FINAL_FILES["context_audit"],
        "split_audit": FINAL_FILES["split_audit"],
        **PROGRESS_FILES,
        "checkpoint": CHECKPOINT_FILE,
        "aggregate_metrics": FINAL_FILES["aggregate_metrics"],
        "paired_global_scaffold_bootstrap": (
            FINAL_FILES["bootstrap"] if include_bootstrap else None
        ),
    }


def result_artifact_paths(
    output_dir: Path,
    include_bootstrap: bool,
) -> list[Path]:
    paths = [
        output_dir / FINAL_FILES["analysis_index"],
        output_dir / FINAL_FILES["context_audit"],
        output_dir / FINAL_FILES["split_audit"],
        *[
            output_dir / filename
            for filename in PROGRESS_FILES.values()
        ],
        output_dir / CHECKPOINT_FILE,
        output_dir / FINAL_FILES["aggregate_metrics"],
    ]
    if include_bootstrap:
        paths.append(output_dir / FINAL_FILES["bootstrap"])
    return paths


def verify_completed_result(
    output_dir: Path,
    manifest: dict[str, Any],
    include_bootstrap: bool,
    formal_match: bool,
) -> None:
    inventory = manifest.get("artifact_inventory")
    if not isinstance(inventory, dict):
        raise RuntimeError("Completed manifest has no artifact inventory")
    expected_paths = result_artifact_paths(
        output_dir,
        include_bootstrap,
    )
    expected_filenames = {path.name for path in expected_paths}
    if set(inventory) != expected_filenames:
        raise RuntimeError(
            "Completed artifact inventory filenames changed: "
            f"observed={sorted(inventory)}, "
            f"expected={sorted(expected_filenames)}"
        )
    expected_entries = artifact_manifest_entries(include_bootstrap)
    if manifest.get("artifacts") != expected_entries:
        raise RuntimeError("Completed artifact map changed")
    ood.verify_file_inventory(
        output_dir,
        inventory,
        expected_filenames,
    )
    if formal_match:
        for filename, expected_rows in EXPECTED_FORMAL_ROWS.items():
            observed_rows = len(pd.read_csv(output_dir / filename))
            if observed_rows != expected_rows:
                raise RuntimeError(
                    f"Completed formal row count changed for {filename}: "
                    f"{observed_rows} != {expected_rows}"
                )


def output_is_in_frozen_parent(output_dir: Path) -> bool:
    resolved = output_dir.resolve()
    protected = (
        core.DEFAULT_OUTPUT_DIR.resolve(),
        OOD_RESULT_DIR.resolve(),
    )
    return any(
        resolved == parent or resolved.is_relative_to(parent)
        for parent in protected
    )


def clear_known_artifacts(output_dir: Path) -> int:
    removed = 0
    for path in known_artifacts(output_dir):
        temporary = path.with_name(f".{path.name}.tmp")
        for candidate in (path, temporary):
            if candidate.is_file():
                candidate.unlink()
                removed += 1
    return removed


def main() -> None:
    args = parse_args()
    required = (
        args.data_file.resolve(),
        Path(__file__).resolve(),
        EXTENSION_PROTOCOL,
        EXTENSION_IDENTITY,
        CORE_SCRIPT,
        CORE_PROTOCOL,
        OOD_SCRIPT,
        OOD_PROTOCOL,
        OOD_IDENTITY,
        OOD_RESULT_MANIFEST,
        OOD_RESULT_SPLIT_AUDIT,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    verify_parent_ood_result()
    frozen_identity, identity_sha256 = load_extension_identity(
        EXTENSION_IDENTITY
    )
    observed_hashes = observed_file_hashes(args.data_file.resolve())
    if observed_hashes["data"] != core.FROZEN_DATA_SHA256:
        raise RuntimeError("Input data do not match frozen core SHA-256")
    formal_match, argument_mismatches = verify_identity(
        args,
        frozen_identity,
        observed_hashes,
    )
    configuration = scientific_configuration(
        args,
        observed_hashes,
        identity_sha256,
        formal_match,
        frozen_identity,
    )
    configuration_sha256 = canonical_json_sha256(configuration)

    output_dir = args.output_dir.resolve()
    if output_is_in_frozen_parent(output_dir):
        raise RuntimeError(
            "Extension output cannot modify frozen core/OOD parent results"
        )
    manifest_path = output_dir / FINAL_FILES["manifest"]
    output_nonempty = bool(output_dir.exists() and any(output_dir.iterdir()))
    prior_manifest: dict[str, Any] | None = None
    if output_nonempty and args.resume:
        if not manifest_path.is_file():
            raise RuntimeError("Cannot resume without strict OOD manifest")
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior_manifest.get("scientific_configuration") != configuration:
            raise RuntimeError("Resume refused: scientific configuration changed")
        if (
            prior_manifest.get("scientific_configuration_sha256")
            != configuration_sha256
        ):
            raise RuntimeError("Resume refused: configuration digest changed")
        if prior_manifest.get("status") == "complete":
            include_bootstrap = (
                args.bootstrap_replicates > 0
                and args.max_groups is None
            )
            verify_completed_result(
                output_dir,
                prior_manifest,
                include_bootstrap,
                formal_match,
            )
            print("[DONE] completed strict OOD artifacts verified", flush=True)
            return
        verify_checkpoint(output_dir, configuration_sha256)
    elif output_nonempty and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; "
            "use --resume, --overwrite or a new directory"
        )
    elif output_nonempty and args.overwrite:
        removed = clear_known_artifacts(output_dir)
        print(f"[OVERWRITE] removed {removed} known artifacts", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_started = time.time()
    rows, context_audit = core.load_rows(args.data_file.resolve())
    y = rows["pDC50"].to_numpy(dtype=np.float32)
    scaffold_ids = core.build_scaffold_ids(rows["canonical_smiles"])
    rows["confirmatory_scaffold_id"] = scaffold_ids
    splits, split_audit = build_strict_splits(
        rows,
        scaffold_ids,
        args.protocols,
        args.seed,
        args.max_groups,
        frozen_identity,
    )
    if not splits:
        raise RuntimeError("No strict OOD splits selected")
    print(
        (
            f"[DATA] rows={len(rows)} unique_smiles="
            f"{rows['canonical_smiles'].nunique()} scaffolds="
            f"{len(np.unique(scaffold_ids))} strict_groups={len(splits)}"
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
        output_dir / FINAL_FILES["context_audit"],
    )
    core.atomic_to_csv(
        split_audit,
        output_dir / FINAL_FILES["split_audit"],
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
        output_dir / FINAL_FILES["analysis_index"],
    )
    analysis_mode = (
        "post_hoc_sensitivity"
        if formal_match
        else (
            "post_hoc_sensitivity_dry_run"
            if args.dry_run
            else "post_hoc_sensitivity_smoke"
        )
    )
    manifest: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "dry_run" if args.dry_run else "running",
        "analysis_mode": analysis_mode,
        "analysis_label": "post_hoc_sensitivity",
        "confirmatory": False,
        "external_identity_match": True,
        "formal_arguments_match": bool(formal_match),
        "identity_argument_mismatches": argument_mismatches,
        "scientific_configuration": configuration,
        "scientific_configuration_sha256": configuration_sha256,
        "resume_count": (
            int(prior_manifest.get("resume_count", 0)) + 1
            if prior_manifest is not None
            else 0
        ),
        "run_session_started_unix": run_started,
        "n_rows": int(len(rows)),
        "n_unique_smiles": int(rows["canonical_smiles"].nunique()),
        "n_scaffolds": int(len(np.unique(scaffold_ids))),
        "n_strict_ood_groups": int(len(splits)),
        "endpoint": "pDC50 = 9 - log10(DC50_nM)",
        "independent_unit_and_dependence": {
            "heldout_domains": (
                "four frozen source domains or eight frozen target domains"
            ),
            "bootstrap_cluster": "global Bemis-Murcko scaffold_id",
            "folds_or_seeds_are_independent_replicates": False,
        },
        "feature_manifest": core.feature_manifest(
            descriptor_names,
            descriptor_failures,
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_count_visible": os.cpu_count(),
            "n_jobs_used_per_fit": int(args.n_jobs),
            "gpu_required": False,
        },
        "boundaries": [
            "This is post-hoc sensitivity analysis, not confirmatory analysis.",
            "Frozen OOD test rows remain unchanged.",
            "All test compounds and test scaffolds are absent from training.",
            "All three ExtraTrees models are independently inner-tuned and refit.",
            "Bootstrap intervals are descriptive; no p values are produced.",
            "This is retrospective internal OOD validation.",
        ],
    }
    core.write_json(manifest_path, manifest)
    if args.dry_run:
        print(f"[DONE] strict OOD dry run: {output_dir}", flush=True)
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
        verify_checkpoint(output_dir, configuration_sha256)
        predictions_complete = complete_predictions(predictions, splits)
        manifest["completeness"] = {
            "predictions": bool(predictions_complete),
            "n_prediction_rows": int(len(predictions)),
            "expected_prediction_rows": int(
                sum(len(split.test_idx) for split in splits)
                * len(MODEL_ORDER)
            ),
        }
        if not predictions_complete:
            raise RuntimeError("Strict OOD prediction completeness failed")
        aggregate = summarize_aggregate_metrics(predictions)
        core.atomic_to_csv(
            aggregate,
            output_dir / FINAL_FILES["aggregate_metrics"],
        )
        bootstrap = pd.DataFrame()
        if args.bootstrap_replicates > 0 and args.max_groups is None:
            bootstrap = paired_global_scaffold_bootstrap(
                predictions,
                args.bootstrap_replicates,
                args.seed,
            )
            core.atomic_to_csv(
                bootstrap,
                output_dir / FINAL_FILES["bootstrap"],
            )
        elif args.bootstrap_replicates > 0:
            print("[WARN] bootstrap skipped for max-groups smoke", flush=True)
        if formal_match and bootstrap.empty:
            raise RuntimeError("Formal post-hoc run lacks bootstrap output")
        if formal_match:
            for filename, expected_rows in EXPECTED_FORMAL_ROWS.items():
                observed_rows = len(pd.read_csv(output_dir / filename))
                if observed_rows != expected_rows:
                    raise RuntimeError(
                        f"Formal row count changed for {filename}: "
                        f"{observed_rows} != {expected_rows}"
                    )
        artifact_paths = result_artifact_paths(
            output_dir,
            include_bootstrap=not bootstrap.empty,
        )
        inventory = ood.build_file_inventory(artifact_paths)
        ood.verify_file_inventory(
            output_dir,
            inventory,
            {path.name for path in artifact_paths},
        )
        manifest["artifacts"] = artifact_manifest_entries(
            include_bootstrap=not bootstrap.empty,
        )
        manifest["artifact_inventory"] = inventory
        manifest["status"] = "complete"
        manifest["elapsed_seconds"] = float(time.time() - run_started)
        core.write_json(manifest_path, manifest)
        print(
            (
                f"[DONE] post-hoc strict OOD groups={len(splits)} "
                f"elapsed={manifest['elapsed_seconds']:.1f}s "
                f"output={output_dir}"
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
