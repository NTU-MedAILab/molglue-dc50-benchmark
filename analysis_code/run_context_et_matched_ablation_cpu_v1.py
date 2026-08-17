#!/usr/bin/env python3
"""CPU-only matched ablation for the molecular-glue DC50 analysis.

This extension fits a context-only ExtraTrees model on exactly the outer
partitions used by ``confirmatory_cpu_v1.1``.  It leaves the frozen parent
script, protocol and result directory untouched, and compares its cross-fitted
predictions with the hash-verified saved full-model predictions.
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
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from rdkit import rdBase
from sklearn.ensemble import ExtraTreesRegressor


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
CORE_SCRIPT = SCRIPT_DIR / "run_confirmatory_cpu_v1.py"
EXTENSION_PROTOCOL = (
    PROJECT_DIR / "docs" / "context_et_matched_ablation_protocol_v1.md"
)
DEFAULT_CORE_RESULTS = PROJECT_DIR / "reports" / "confirmatory_cpu_v1"
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "reports" / "context_et_matched_ablation_cpu_v1"
)
DEFAULT_SMOKE_OUTPUT_DIR = (
    PROJECT_DIR / "reports" / "context_et_matched_ablation_cpu_v1_smoke"
)

sys.path.insert(0, str(SCRIPT_DIR))
import run_confirmatory_cpu_v1 as core  # noqa: E402


PROTOCOL_VERSION = "context_et_matched_ablation_v1"
PARENT_PROTOCOL_VERSION = "confirmatory_cpu_v1.1"
MODEL_ID = "context_extra_trees"
FULL_MODEL_ID = "full_context_extra_trees"
DEFAULT_SEED = 260531
FORMAL_OUTER_FOLDS = 5
FORMAL_OUTER_REPEATS = 5
FORMAL_INNER_FOLDS = 4
FORMAL_N_ESTIMATORS = 600
FORMAL_N_JOBS = 10
FORMAL_BOOTSTRAP_REPLICATES = 10_000
FULL_MODEL_POSITION = 5

FROZEN_DATA_SHA256 = (
    "e9494246bcbd198bc09d5b0d09721910c33f70f46ae859b76377a13775e3e4b2"
)
FROZEN_CORE_SCRIPT_SHA256 = (
    "ac1a57203b43d74023521b76cfaef28b958a803dbc6d9fb2866e7550f636995c"
)
FROZEN_CORE_PROTOCOL_SHA256 = (
    "7429b9a31f66b0ee7b1f75d977a1178615b8778bb3a507e06aa7f320eb11780b"
)
FROZEN_CORE_MANIFEST_SHA256 = (
    "3ce6e2e536e73f4ce2e8c79206e6b8b0602886dc08be65e46ee9625fcc3a3e46"
)
FROZEN_CORE_RAW_PREDICTIONS_SHA256 = (
    "c67fc7176195eade86b45930d2f766a7aa76942efa9e4f8ce11ed94703d80bd9"
)
FROZEN_CORE_AVERAGED_PREDICTIONS_SHA256 = (
    "183d6fa0047bdf47f1bb5f89679f5a3602344b09e4b0744792ea70ecce6f5e3b"
)
FROZEN_CORE_OUTER_METRICS_SHA256 = {
    "scaffold": (
        "cd22ea3af654180171876ba86d90c7ebb1a8cbc5c8f8bf5fdeaad8fcef9016d7"
    ),
    "compound": (
        "e56ab5b5b6424b80c03e712cf159ec5a4375686f900eddc692584390d2696965"
    ),
}
FROZEN_EXTENSION_PROTOCOL_SHA256 = (
    "84b3e40f7705a5c2b39b54ba3932572ef537fc8d68cf324b3418433c1744d63b"
)

PREDICTION_COLUMNS = [
    "protocol",
    "repeat",
    "outer_fold",
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
    "scaffold_seen_in_train",
    "target_train_rows",
    "target_train_unique_smiles",
    "recruiter_target_train_unique_smiles",
    "source_seen_in_train",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen context-only ExtraTrees matched ablation without "
            "modifying the parent result."
        )
    )
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=list(core.PROTOCOL_ORDER),
        default=list(core.PROTOCOL_ORDER),
    )
    parser.add_argument(
        "--core-results-dir",
        type=Path,
        default=DEFAULT_CORE_RESULTS,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-jobs", type=int, default=FORMAL_N_JOBS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run only the first formal scaffold outer split with 40 trees, "
            "two inner folds and no bootstrap."
        ),
    )
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.n_jobs < 1:
        parser.error("--n-jobs must be >= 1")
    args.protocols = [
        protocol
        for protocol in core.PROTOCOL_ORDER
        if protocol in set(args.protocols)
    ]
    if len(args.protocols) == 0:
        parser.error("at least one protocol is required")
    if args.smoke:
        if args.resume:
            parser.error("--smoke does not support --resume")
        args.protocols = ["scaffold"]
        args.inner_folds = 2
        args.n_estimators = 40
        args.bootstrap_replicates = 0
        args.max_outer_splits = 1
        if args.output_dir is None:
            args.output_dir = DEFAULT_SMOKE_OUTPUT_DIR
    else:
        if args.protocols != list(core.PROTOCOL_ORDER):
            parser.error(
                "the frozen formal run requires both scaffold and compound "
                "protocols"
            )
        if args.n_jobs != FORMAL_N_JOBS:
            parser.error(
                f"the frozen formal run requires --n-jobs {FORMAL_N_JOBS}"
            )
        args.inner_folds = FORMAL_INNER_FOLDS
        args.n_estimators = FORMAL_N_ESTIMATORS
        args.bootstrap_replicates = FORMAL_BOOTSTRAP_REPLICATES
        args.max_outer_splits = None
        if args.output_dir is None:
            args.output_dir = DEFAULT_OUTPUT_DIR
    args.outer_folds = FORMAL_OUTER_FOLDS
    args.outer_repeats = FORMAL_OUTER_REPEATS
    args.seed = DEFAULT_SEED
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def runtime_versions() -> dict[str, str]:
    return {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "rdkit": rdBase.rdkitVersion,
        "joblib": joblib.__version__,
    }


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"{label} SHA256 mismatch: expected {expected}, observed {observed}"
        )
    return observed


def verify_parent(
    core_results_dir: Path,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
    dict[str, str],
]:
    manifest_path = core_results_dir / "run_manifest.json"
    raw_path = core_results_dir / "all_cross_fitted_predictions.csv"
    averaged_path = (
        core_results_dir / "repeat_averaged_cross_fitted_predictions.csv"
    )
    hashes = {
        "parent_manifest_sha256": require_hash(
            manifest_path,
            FROZEN_CORE_MANIFEST_SHA256,
            "parent manifest",
        ),
        "parent_raw_predictions_sha256": require_hash(
            raw_path,
            FROZEN_CORE_RAW_PREDICTIONS_SHA256,
            "parent raw predictions",
        ),
        "parent_averaged_predictions_sha256": require_hash(
            averaged_path,
            FROZEN_CORE_AVERAGED_PREDICTIONS_SHA256,
            "parent averaged predictions",
        ),
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("Parent result manifest is not complete")
    if manifest.get("protocol_version") != PARENT_PROTOCOL_VERSION:
        raise RuntimeError("Unexpected parent protocol version")
    if not bool(manifest.get("frozen_protocol_match")):
        raise RuntimeError("Parent manifest does not match its frozen protocol")
    expected_parent_config = {
        "outer_folds": FORMAL_OUTER_FOLDS,
        "outer_repeats": FORMAL_OUTER_REPEATS,
        "inner_folds": FORMAL_INNER_FOLDS,
        "n_estimators": FORMAL_N_ESTIMATORS,
        "bootstrap_replicates": FORMAL_BOOTSTRAP_REPLICATES,
        "seed": DEFAULT_SEED,
    }
    parent_config = manifest.get("scientific_configuration", {})
    for key, expected in expected_parent_config.items():
        if parent_config.get(key) != expected:
            raise RuntimeError(
                f"Parent scientific configuration mismatch for {key}"
            )
    if manifest.get("n_rows") != 1560:
        raise RuntimeError("Parent row count is not 1,560")

    data_path = Path(manifest["data_file"])
    parent_script = Path(manifest["script_file"])
    parent_protocol = Path(manifest["protocol_document"])
    hashes.update(
        {
            "data_sha256": require_hash(
                data_path,
                FROZEN_DATA_SHA256,
                "frozen data",
            ),
            "parent_script_sha256": require_hash(
                parent_script,
                FROZEN_CORE_SCRIPT_SHA256,
                "parent script",
            ),
            "parent_protocol_sha256": require_hash(
                parent_protocol,
                FROZEN_CORE_PROTOCOL_SHA256,
                "parent protocol",
            ),
        }
    )
    if manifest.get("data_sha256") != hashes["data_sha256"]:
        raise RuntimeError("Parent manifest data hash is inconsistent")
    if manifest.get("script_sha256") != hashes["parent_script_sha256"]:
        raise RuntimeError("Parent manifest script hash is inconsistent")
    if (
        manifest.get("protocol_document_sha256")
        != hashes["parent_protocol_sha256"]
    ):
        raise RuntimeError("Parent manifest protocol hash is inconsistent")

    current_versions = runtime_versions()
    expected_versions = parent_config.get("runtime_versions", {})
    for key in (
        "numpy",
        "pandas",
        "scipy",
        "scikit_learn",
        "rdkit",
    ):
        if current_versions[key] != expected_versions.get(key):
            raise RuntimeError(
                f"Runtime version mismatch for {key}: "
                f"{current_versions[key]} != {expected_versions.get(key)}"
            )

    raw = pd.read_csv(raw_path)
    raw = raw[raw["model_id"] == FULL_MODEL_ID].copy()
    expected_raw_rows = (
        len(core.PROTOCOL_ORDER)
        * FORMAL_OUTER_REPEATS
        * int(manifest["n_rows"])
    )
    if len(raw) != expected_raw_rows:
        raise RuntimeError("Parent full-model raw prediction count is incomplete")
    raw_key = ["protocol", "repeat", "row_index"]
    if raw.duplicated(raw_key).any():
        raise RuntimeError("Parent full-model raw predictions have duplicate keys")
    if not np.all(np.isfinite(raw["y_true"])) or not np.all(
        np.isfinite(raw["y_pred"])
    ):
        raise RuntimeError("Parent full-model raw predictions are non-finite")
    if set(raw["protocol"].unique()) != set(core.PROTOCOL_ORDER):
        raise RuntimeError("Parent raw predictions have an unexpected protocol set")

    averaged = pd.read_csv(averaged_path)
    averaged = averaged[averaged["model_id"] == FULL_MODEL_ID].copy()
    expected_averaged_rows = (
        len(core.PROTOCOL_ORDER) * int(manifest["n_rows"])
    )
    if len(averaged) != expected_averaged_rows:
        raise RuntimeError("Parent full-model repeat averages are incomplete")
    if averaged.duplicated(["protocol", "row_index"]).any():
        raise RuntimeError("Parent full-model averages have duplicate keys")
    if not (averaged["n_oof_predictions"] == FORMAL_OUTER_REPEATS).all():
        raise RuntimeError("Parent full-model averages do not contain five OOFs")
    if not np.all(np.isfinite(averaged["y_pred"])):
        raise RuntimeError("Parent full-model averages are non-finite")

    outer_metrics: dict[str, pd.DataFrame] = {}
    for protocol in core.PROTOCOL_ORDER:
        path = core_results_dir / f"{protocol}_outer_fold_metrics.csv"
        outer_metrics_hash = require_hash(
            path,
            FROZEN_CORE_OUTER_METRICS_SHA256[protocol],
            f"parent {protocol} outer-fold metrics",
        )
        frame = pd.read_csv(path)
        frame = frame[frame["model_id"] == FULL_MODEL_ID].copy()
        if len(frame) != FORMAL_OUTER_FOLDS * FORMAL_OUTER_REPEATS:
            raise RuntimeError(
                f"Parent {protocol} full-model fold metrics are incomplete"
            )
        if frame.duplicated(["repeat", "outer_fold"]).any():
            raise RuntimeError(
                f"Parent {protocol} full-model fold metrics have duplicate keys"
            )
        outer_metrics[protocol] = frame
        hashes[f"parent_{protocol}_outer_metrics_sha256"] = (
            outer_metrics_hash
        )
    return manifest, raw, averaged, outer_metrics, hashes


def build_identity(
    args: argparse.Namespace,
    parent_hashes: dict[str, str],
) -> tuple[dict[str, Any], str]:
    protocol_hash = require_hash(
        EXTENSION_PROTOCOL,
        FROZEN_EXTENSION_PROTOCOL_SHA256,
        "extension protocol",
    )
    identity = {
        "protocol_version": PROTOCOL_VERSION,
        "mode": "smoke" if args.smoke else "formal",
        "protocols": list(args.protocols),
        "outer_folds": args.outer_folds,
        "outer_repeats": args.outer_repeats,
        "inner_folds": args.inner_folds,
        "n_estimators": args.n_estimators,
        "n_jobs": args.n_jobs,
        "bootstrap_replicates": args.bootstrap_replicates,
        "max_outer_splits": args.max_outer_splits,
        "seed": args.seed,
        "model_id": MODEL_ID,
        "extension_script_sha256": sha256_file(Path(__file__).resolve()),
        "extension_protocol_sha256": protocol_hash,
        "runtime_versions": runtime_versions(),
        **parent_hashes,
    }
    return identity, sha256_payload(identity)


def known_output_paths(output_dir: Path) -> list[Path]:
    names = {
        "run_manifest.json",
        "all_context_extra_trees_cross_fitted_predictions.csv",
        "context_extra_trees_repeat_averaged_predictions.csv",
        "context_extra_trees_metrics_by_repeat.csv",
        "context_extra_trees_summary_metrics.csv",
        "context_extra_trees_repeat_averaged_metrics.csv",
        "context_extra_trees_canonical_smiles_metrics.csv",
        "matched_repeat_averaged_metrics.csv",
        "paired_scaffold_bootstrap_full_vs_context_extra_trees.csv",
    }
    for protocol in core.PROTOCOL_ORDER:
        names.update(
            {
                f"{protocol}_context_extra_trees_cross_fitted_predictions.csv",
                f"{protocol}_context_extra_trees_outer_fold_metrics.csv",
                f"{protocol}_context_extra_trees_inner_tuning_metrics.csv",
                f"{protocol}_context_extra_trees_selected_hyperparameters.csv",
            }
        )
    return [output_dir / name for name in sorted(names)]


def initialize_output(
    args: argparse.Namespace,
    identity: dict[str, Any],
    identity_sha256: str,
) -> tuple[dict[str, Any], bool]:
    output_dir = args.output_dir.resolve()
    core_dir = args.core_results_dir.resolve()
    if output_dir == core_dir or output_dir.is_relative_to(core_dir):
        raise RuntimeError(
            "Extension output must not be the parent result directory or "
            "one of its descendants"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    if args.overwrite:
        for path in known_output_paths(output_dir):
            if path.exists():
                path.unlink()
    elif args.resume:
        if not manifest_path.exists():
            raise RuntimeError("Cannot resume without an extension manifest")
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("scientific_identity_sha256") != identity_sha256:
            raise RuntimeError("Resume identity mismatch; failing closed")
        if prior.get("status") in {"complete", "smoke_complete"}:
            for name, record in prior.get("artifacts", {}).items():
                path = output_dir / record["path"]
                if sha256_file(path) != record["sha256"]:
                    raise RuntimeError(
                        f"Completed artifact hash mismatch for {name}"
                    )
            print("[DONE] completed extension and artifact hashes verified")
            return prior, True
    elif any(output_dir.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {output_dir}; "
            "use --overwrite or --resume"
        )

    prior_resume_count = 0
    if args.resume and manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        prior_resume_count = int(prior.get("resume_count", 0)) + 1
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "running",
        "analysis_mode": "smoke" if args.smoke else "formal",
        "frozen_extension_match": not args.smoke,
        "scientific_identity": identity,
        "scientific_identity_sha256": identity_sha256,
        "parent_result_directory": str(core_dir),
        "output_directory": str(output_dir),
        "extension_script": str(Path(__file__).resolve()),
        "extension_protocol": str(EXTENSION_PROTOCOL.resolve()),
        "resume_count": prior_resume_count,
        "session_started_unix": time.time(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_count_visible": os.cpu_count(),
            "gpu_required": False,
            **runtime_versions(),
        },
        "boundaries": [
            "The parent result directory is read-only.",
            "Only the context feature block enters the new model.",
            "The saved full-model prediction is not refitted.",
            "This is retrospective internal validation.",
            "Folds and repeats are not independent replicates.",
        ],
    }
    atomic_write_json(manifest_path, manifest)
    return manifest, False


def validate_outer_assignments(
    rows: pd.DataFrame,
    scaffold_ids: np.ndarray,
    parent_full_raw: pd.DataFrame,
    protocols: list[str],
) -> dict[str, list[Any]]:
    split_map: dict[str, list[Any]] = {}
    for protocol in protocols:
        splits = core.make_outer_splits(
            protocol,
            rows,
            scaffold_ids,
            FORMAL_OUTER_FOLDS,
            FORMAL_OUTER_REPEATS,
            DEFAULT_SEED,
        )
        generated_rows: list[dict[str, int | str]] = []
        group_values = (
            rows["canonical_smiles"].to_numpy(dtype=object)
            if protocol == "compound"
            else scaffold_ids
        )
        for split in splits:
            if set(group_values[split.train_idx]).intersection(
                set(group_values[split.test_idx])
            ):
                raise RuntimeError(
                    f"Outer group leakage in {protocol} repeat "
                    f"{split.repeat} fold {split.fold}"
                )
            for row_index in split.test_idx:
                generated_rows.append(
                    {
                        "protocol": protocol,
                        "repeat": split.repeat,
                        "row_index": int(row_index),
                        "outer_fold": split.fold,
                    }
                )
        generated = pd.DataFrame(generated_rows).sort_values(
            ["protocol", "repeat", "row_index"]
        )
        parent = parent_full_raw[
            parent_full_raw["protocol"] == protocol
        ][["protocol", "repeat", "row_index", "outer_fold"]].sort_values(
            ["protocol", "repeat", "row_index"]
        )
        generated = generated.reset_index(drop=True)
        parent = parent.reset_index(drop=True)
        if len(generated) != FORMAL_OUTER_REPEATS * len(rows):
            raise RuntimeError("Generated outer assignment count is incomplete")
        if not generated.equals(parent.astype(generated.dtypes.to_dict())):
            raise RuntimeError(
                f"Generated {protocol} outer assignments do not exactly "
                "match the parent full-model OOF assignments"
            )
        split_map[protocol] = splits
    return split_map


def context_features(
    contexts: pd.DataFrame,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    encoder = core.make_one_hot_encoder()
    x_fit = encoder.fit_transform(contexts.iloc[fit_idx]).astype(np.float32)
    x_eval = encoder.transform(contexts.iloc[eval_idx]).astype(np.float32)
    if x_fit.shape[1] != x_eval.shape[1]:
        raise RuntimeError("Context feature dimensions differ within a fold")
    if not np.all(np.isfinite(x_fit)) or not np.all(np.isfinite(x_eval)):
        raise RuntimeError("Non-finite context features")
    return x_fit, x_eval


def et_grid() -> list[dict[str, Any]]:
    grid = core.model_param_grid(FULL_MODEL_ID)
    if len(grid) != 6:
        raise RuntimeError("Parent ExtraTrees grid does not contain six options")
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
        raise RuntimeError("Parent ExtraTrees grid differs from the extension")
    return [dict(params) for params in grid]


def fit_context_et(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_eval: np.ndarray,
    params: dict[str, Any],
    seed: int,
    n_estimators: int,
    n_jobs: int,
) -> np.ndarray:
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=params["max_features"],
        max_depth=None,
        bootstrap=False,
        random_state=seed,
        n_jobs=n_jobs,
    )
    model.fit(x_fit, y_fit)
    prediction = np.asarray(model.predict(x_eval), dtype=np.float32)
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("Context-only ExtraTrees produced non-finite values")
    return prediction


def select_context_et(
    rows: pd.DataFrame,
    contexts: pd.DataFrame,
    y: np.ndarray,
    outer_train_idx: np.ndarray,
    inner_groups: np.ndarray,
    inner_folds: int,
    seed: int,
    n_estimators: int,
    n_jobs: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    inner_splits = core.balanced_group_splits(
        outer_train_idx,
        inner_groups,
        inner_folds,
        seed,
    )
    eval_counter = {
        int(row_index): 0 for row_index in outer_train_idx.tolist()
    }
    for fit_idx, eval_idx in inner_splits:
        if set(inner_groups[fit_idx]).intersection(
            set(inner_groups[eval_idx])
        ):
            raise RuntimeError("Inner group leakage detected")
        for row_index in eval_idx:
            eval_counter[int(row_index)] += 1
    if set(eval_counter.values()) != {1}:
        raise RuntimeError("Inner OOF partitions do not cover training rows once")

    local_position = {
        int(row_index): position
        for position, row_index in enumerate(outer_train_idx.tolist())
    }
    grid = et_grid()
    predictions = {
        str(params["param_id"]): np.full(
            len(outer_train_idx),
            np.nan,
            dtype=np.float32,
        )
        for params in grid
    }
    for inner_fold, (fit_idx, eval_idx) in enumerate(inner_splits, start=1):
        x_fit, x_eval = context_features(contexts, fit_idx, eval_idx)
        eval_positions = np.asarray(
            [local_position[int(index)] for index in eval_idx],
            dtype=np.int64,
        )
        candidate_seed = seed + inner_fold * 101
        for params in grid:
            prediction = fit_context_et(
                x_fit,
                y[fit_idx],
                x_eval,
                params,
                seed=candidate_seed,
                n_estimators=n_estimators,
                n_jobs=n_jobs,
            )
            predictions[str(params["param_id"])][eval_positions] = prediction

    truth = y[outer_train_idx]
    scored: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    for params in grid:
        param_id = str(params["param_id"])
        prediction = predictions[param_id]
        if not np.all(np.isfinite(prediction)):
            raise RuntimeError("Incomplete inner OOF candidate predictions")
        metrics = core.regression_metrics(truth, prediction)
        score = float(metrics["spearman"])
        if not math.isfinite(score):
            score = -float("inf")
        scored.append(
            {
                "score": score,
                "rmse": float(metrics["rmse"]),
                "param_id": param_id,
                "params": params,
                "regularization_rank": core.regularization_rank(
                    FULL_MODEL_ID,
                    params,
                ),
            }
        )
        tuning_rows.append(
            {
                "model_id": MODEL_ID,
                "param_id": param_id,
                "inner_n": int(len(truth)),
                **metrics,
            }
        )
    best_score = max(record["score"] for record in scored)
    eligible = [
        record
        for record in scored
        if record["score"] >= best_score - 0.005
    ]
    eligible.sort(
        key=lambda record: (
            tuple(-value for value in record["regularization_rank"]),
            record["rmse"],
            record["param_id"],
        )
    )
    selected = dict(eligible[0]["params"])
    return selected, tuning_rows


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        return pd.read_csv(path).to_dict(orient="records")
    except pd.errors.EmptyDataError:
        return []


def record_matches(record: dict[str, Any], split: Any) -> bool:
    return bool(
        int(record["repeat"]) == split.repeat
        and int(record["outer_fold"]) == split.fold
    )


def discard_split(
    records: list[dict[str, Any]],
    split: Any,
) -> list[dict[str, Any]]:
    return [record for record in records if not record_matches(record, split)]


def split_checkpoint_complete(
    split: Any,
    prediction_rows: list[dict[str, Any]],
    fold_metric_rows: list[dict[str, Any]],
    tuning_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
) -> bool:
    local_predictions = [
        row for row in prediction_rows if record_matches(row, split)
    ]
    local_fold_metrics = [
        row for row in fold_metric_rows if record_matches(row, split)
    ]
    local_tuning = [
        row for row in tuning_rows if record_matches(row, split)
    ]
    local_selected = [
        row for row in selected_rows if record_matches(row, split)
    ]
    expected_rows = set(split.test_idx.astype(int).tolist())
    if len(local_predictions) != len(expected_rows):
        return False
    frame = pd.DataFrame(local_predictions)
    if frame["row_index"].astype(int).duplicated().any():
        return False
    if set(frame["row_index"].astype(int)) != expected_rows:
        return False
    if set(frame["model_id"].astype(str)) != {MODEL_ID}:
        return False
    if not np.all(
        np.isfinite(pd.to_numeric(frame["y_pred"], errors="coerce"))
    ):
        return False
    if len(local_fold_metrics) != 1 or len(local_selected) != 1:
        return False
    expected_param_ids = {
        str(params["param_id"]) for params in et_grid()
    }
    observed_param_ids = {
        str(row["param_id"]) for row in local_tuning
    }
    return bool(
        len(local_tuning) == len(expected_param_ids)
        and observed_param_ids == expected_param_ids
    )


def protocol_paths(output_dir: Path, protocol: str) -> dict[str, Path]:
    prefix = output_dir / f"{protocol}_context_extra_trees"
    return {
        "predictions": Path(f"{prefix}_cross_fitted_predictions.csv"),
        "fold_metrics": Path(f"{prefix}_outer_fold_metrics.csv"),
        "tuning": Path(f"{prefix}_inner_tuning_metrics.csv"),
        "selected": Path(f"{prefix}_selected_hyperparameters.csv"),
    }


def write_protocol_progress(
    paths: dict[str, Path],
    prediction_rows: list[dict[str, Any]],
    fold_metric_rows: list[dict[str, Any]],
    tuning_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
) -> None:
    atomic_to_csv(pd.DataFrame(prediction_rows), paths["predictions"])
    atomic_to_csv(pd.DataFrame(fold_metric_rows), paths["fold_metrics"])
    atomic_to_csv(pd.DataFrame(tuning_rows), paths["tuning"])
    atomic_to_csv(pd.DataFrame(selected_rows), paths["selected"])


def selected_param_from_tuning(tuning: pd.DataFrame) -> str:
    params_by_id = {
        str(params["param_id"]): params for params in et_grid()
    }
    scored: list[dict[str, Any]] = []
    for record in tuning.to_dict(orient="records"):
        param_id = str(record["param_id"])
        params = params_by_id[param_id]
        score = float(record["spearman"])
        if not math.isfinite(score):
            score = -float("inf")
        scored.append(
            {
                "score": score,
                "rmse": float(record["rmse"]),
                "param_id": param_id,
                "regularization_rank": core.regularization_rank(
                    FULL_MODEL_ID,
                    params,
                ),
            }
        )
    best_score = max(record["score"] for record in scored)
    eligible = [
        record
        for record in scored
        if record["score"] >= best_score - 0.005
    ]
    eligible.sort(
        key=lambda record: (
            tuple(-value for value in record["regularization_rank"]),
            record["rmse"],
            record["param_id"],
        )
    )
    return str(eligible[0]["param_id"])


def validate_protocol_artifacts(
    output_dir: Path,
    protocol: str,
    n_splits: int,
) -> None:
    paths = protocol_paths(output_dir, protocol)
    predictions = pd.read_csv(paths["predictions"])
    fold_metrics = pd.read_csv(paths["fold_metrics"])
    tuning = pd.read_csv(paths["tuning"])
    selected = pd.read_csv(paths["selected"])
    expected_prediction_rows = n_splits * 312
    if len(predictions) != expected_prediction_rows:
        raise RuntimeError(
            f"{protocol} checkpoint predictions are incomplete"
        )
    if len(fold_metrics) != n_splits or len(selected) != n_splits:
        raise RuntimeError(f"{protocol} fold/selection tables are incomplete")
    if len(tuning) != n_splits * len(et_grid()):
        raise RuntimeError(f"{protocol} tuning table is incomplete")
    if predictions.duplicated(
        ["repeat", "outer_fold", "row_index", "model_id"]
    ).any():
        raise RuntimeError(f"{protocol} prediction keys are duplicated")
    if fold_metrics.duplicated(["repeat", "outer_fold", "model_id"]).any():
        raise RuntimeError(f"{protocol} fold metric keys are duplicated")
    if selected.duplicated(["repeat", "outer_fold", "model_id"]).any():
        raise RuntimeError(f"{protocol} selected keys are duplicated")
    if tuning.duplicated(
        ["repeat", "outer_fold", "model_id", "param_id"]
    ).any():
        raise RuntimeError(f"{protocol} tuning keys are duplicated")
    if set(predictions["model_id"].astype(str)) != {MODEL_ID}:
        raise RuntimeError(f"{protocol} predictions have an unexpected model")
    if not np.all(np.isfinite(predictions["y_pred"])):
        raise RuntimeError(f"{protocol} predictions are non-finite")
    metric_columns = [
        "spearman",
        "pearson",
        "rmse",
        "mae",
        "r2",
        "calibration_intercept",
        "calibration_slope",
    ]
    if not np.all(np.isfinite(tuning[metric_columns].to_numpy(dtype=float))):
        raise RuntimeError(f"{protocol} tuning metrics are non-finite")
    if not np.all(
        np.isfinite(fold_metrics[metric_columns].to_numpy(dtype=float))
    ):
        raise RuntimeError(f"{protocol} outer metrics are non-finite")
    if not (fold_metrics["n_test_rows"] == 312).all():
        raise RuntimeError(f"{protocol} outer test size differs from 312")

    split_sets = [
        set(
            zip(
                frame["repeat"].astype(int),
                frame["outer_fold"].astype(int),
            )
        )
        for frame in (fold_metrics, selected)
    ]
    tuning_splits = set(
        zip(tuning["repeat"].astype(int), tuning["outer_fold"].astype(int))
    )
    if split_sets[0] != split_sets[1] or split_sets[0] != tuning_splits:
        raise RuntimeError(f"{protocol} split keys differ across tables")
    expected_param_ids = {
        str(params["param_id"]) for params in et_grid()
    }
    for (repeat, outer_fold), local_tuning in tuning.groupby(
        ["repeat", "outer_fold"]
    ):
        if set(local_tuning["param_id"].astype(str)) != expected_param_ids:
            raise RuntimeError(
                f"{protocol} repeat {repeat} fold {outer_fold} grid differs"
            )
        expected = selected_param_from_tuning(local_tuning)
        selected_row = selected[
            (selected["repeat"] == repeat)
            & (selected["outer_fold"] == outer_fold)
        ]
        if len(selected_row) != 1:
            raise RuntimeError("Selected hyperparameter key is not unique")
        observed = str(selected_row.iloc[0]["param_id"])
        if observed != expected:
            raise RuntimeError(
                f"{protocol} repeat {repeat} fold {outer_fold} selected "
                f"{observed}, expected {expected}"
            )


def run_protocol(
    protocol: str,
    args: argparse.Namespace,
    rows: pd.DataFrame,
    contexts: pd.DataFrame,
    y: np.ndarray,
    scaffold_ids: np.ndarray,
    splits: list[Any],
    parent_full_raw: pd.DataFrame,
    parent_outer_metrics: pd.DataFrame,
) -> pd.DataFrame:
    paths = protocol_paths(args.output_dir, protocol)
    prediction_rows = read_records(paths["predictions"]) if args.resume else []
    fold_metric_rows = (
        read_records(paths["fold_metrics"]) if args.resume else []
    )
    tuning_rows = read_records(paths["tuning"]) if args.resume else []
    selected_rows = read_records(paths["selected"]) if args.resume else []

    active_splits = splits
    if args.max_outer_splits is not None:
        active_splits = active_splits[: args.max_outer_splits]
    inner_groups = (
        rows["canonical_smiles"].to_numpy(dtype=object)
        if protocol == "compound"
        else scaffold_ids
    )
    parent_local = parent_full_raw[
        parent_full_raw["protocol"] == protocol
    ].copy()
    parent_dims = parent_outer_metrics.set_index(["repeat", "outer_fold"])

    started = time.perf_counter()
    for split_number, split in enumerate(active_splits, start=1):
        if args.resume and split_checkpoint_complete(
            split,
            prediction_rows,
            fold_metric_rows,
            tuning_rows,
            selected_rows,
        ):
            print(
                f"[RESUME] {protocol} repeat={split.repeat} "
                f"fold={split.fold} complete",
                flush=True,
            )
            continue
        if args.resume:
            prediction_rows = discard_split(prediction_rows, split)
            fold_metric_rows = discard_split(fold_metric_rows, split)
            tuning_rows = discard_split(tuning_rows, split)
            selected_rows = discard_split(selected_rows, split)

        print(
            f"[OUTER] {protocol} repeat={split.repeat} fold={split.fold} "
            f"({split_number}/{len(active_splits)})",
            flush=True,
        )
        selection_seed = (
            args.seed + split.repeat * 10_000 + split.fold * 100
        )
        selected, local_tuning = select_context_et(
            rows,
            contexts,
            y,
            split.train_idx,
            inner_groups,
            args.inner_folds,
            seed=selection_seed,
            n_estimators=args.n_estimators,
            n_jobs=args.n_jobs,
        )
        for record in local_tuning:
            tuning_rows.append(
                {
                    "protocol": protocol,
                    "repeat": split.repeat,
                    "outer_fold": split.fold,
                    **record,
                }
            )
        selected_rows.append(
            {
                "protocol": protocol,
                "repeat": split.repeat,
                "outer_fold": split.fold,
                "model_id": MODEL_ID,
                **selected,
            }
        )

        x_train, x_test = context_features(
            contexts,
            split.train_idx,
            split.test_idx,
        )
        expected_context_dim = int(
            parent_dims.loc[(split.repeat, split.fold), "context_dim"]
        )
        if x_train.shape[1] != expected_context_dim:
            raise RuntimeError(
                f"Context dimension mismatch for {protocol} repeat "
                f"{split.repeat} fold {split.fold}: {x_train.shape[1]} != "
                f"{expected_context_dim}"
            )
        outer_seed = (
            args.seed
            + split.repeat * 100_000
            + split.fold * 1_000
            + FULL_MODEL_POSITION
        )
        prediction = fit_context_et(
            x_train,
            y[split.train_idx],
            x_test,
            selected,
            seed=outer_seed,
            n_estimators=args.n_estimators,
            n_jobs=args.n_jobs,
        )
        metrics = core.regression_metrics(y[split.test_idx], prediction)
        fold_metric_rows.append(
            {
                "protocol": protocol,
                "repeat": split.repeat,
                "outer_fold": split.fold,
                "model_id": MODEL_ID,
                "param_id": str(selected["param_id"]),
                "n_train_rows": int(len(split.train_idx)),
                "n_test_rows": int(len(split.test_idx)),
                "n_test_unique_smiles": int(
                    rows.iloc[split.test_idx]["canonical_smiles"].nunique()
                ),
                "n_test_scaffolds": int(
                    len(np.unique(scaffold_ids[split.test_idx]))
                ),
                "context_dim": int(x_train.shape[1]),
                "matched_full_outer_seed": int(outer_seed),
                **metrics,
            }
        )

        parent_split = parent_local[
            (parent_local["repeat"] == split.repeat)
            & (parent_local["outer_fold"] == split.fold)
        ].set_index("row_index")
        if set(parent_split.index.astype(int)) != set(
            split.test_idx.astype(int)
        ):
            raise RuntimeError("Parent split metadata do not match test indices")
        for local_index, row_index in enumerate(split.test_idx):
            parent_record = parent_split.loc[int(row_index)].to_dict()
            if not math.isclose(
                float(parent_record["y_true"]),
                float(y[row_index]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError("Parent and extension y_true differ")
            parent_record.update(
                {
                    "protocol": protocol,
                    "repeat": split.repeat,
                    "outer_fold": split.fold,
                    "row_index": int(row_index),
                    "model_id": MODEL_ID,
                    "param_id": str(selected["param_id"]),
                    "y_pred": float(prediction[local_index]),
                }
            )
            prediction_rows.append(
                {column: parent_record[column] for column in PREDICTION_COLUMNS}
            )

        write_protocol_progress(
            paths,
            prediction_rows,
            fold_metric_rows,
            tuning_rows,
            selected_rows,
        )
        elapsed = time.perf_counter() - started
        print(
            f"[OUTER] completed {split_number}/{len(active_splits)} "
            f"in {elapsed:.1f}s",
            flush=True,
        )
    return pd.DataFrame(prediction_rows)


def validate_formal_predictions(
    predictions: pd.DataFrame,
    protocols: list[str],
    n_rows: int,
) -> None:
    expected = len(protocols) * FORMAL_OUTER_REPEATS * n_rows
    if len(predictions) != expected:
        raise RuntimeError(
            f"Formal prediction count {len(predictions)} != {expected}"
        )
    key = ["protocol", "repeat", "row_index", "model_id"]
    if predictions.duplicated(key).any():
        raise RuntimeError("Formal predictions have duplicate keys")
    if set(predictions["model_id"].unique()) != {MODEL_ID}:
        raise RuntimeError("Unexpected model ID in extension predictions")
    if not np.all(np.isfinite(predictions["y_pred"])):
        raise RuntimeError("Formal predictions contain non-finite values")
    expected_rows = set(range(n_rows))
    for protocol in protocols:
        local = predictions[predictions["protocol"] == protocol]
        if set(local["repeat"].astype(int)) != set(
            range(1, FORMAL_OUTER_REPEATS + 1)
        ):
            raise RuntimeError(f"Incomplete repeats for {protocol}")
        for _, frame in local.groupby("repeat"):
            if set(frame["row_index"].astype(int)) != expected_rows:
                raise RuntimeError(f"Incomplete row coverage for {protocol}")
        counts = local.groupby("row_index").size()
        if not (counts == FORMAL_OUTER_REPEATS).all():
            raise RuntimeError(f"Each {protocol} row must have five OOFs")


def validate_alignment(
    context_averaged: pd.DataFrame,
    parent_averaged: pd.DataFrame,
    protocols: list[str],
) -> pd.DataFrame:
    context_local = context_averaged[
        context_averaged["protocol"].isin(protocols)
    ].copy()
    parent_local = parent_averaged[
        parent_averaged["protocol"].isin(protocols)
    ].copy()
    identity_columns = [
        "protocol",
        "row_index",
        "qc_id",
        "y_true",
        "canonical_smiles",
        "scaffold_id",
        "source_database",
        "recruiting_protein",
        "target_protein",
        "cell_line",
    ]
    left = context_local.sort_values(["protocol", "row_index"]).reset_index(
        drop=True
    )
    right = parent_local.sort_values(["protocol", "row_index"]).reset_index(
        drop=True
    )
    if len(left) != len(right):
        raise RuntimeError("Matched repeat-average row counts differ")
    for column in identity_columns:
        if column == "y_true":
            if not np.allclose(
                left[column].to_numpy(dtype=float),
                right[column].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError("Matched repeat-average y_true values differ")
        elif not left[column].astype(str).equals(right[column].astype(str)):
            raise RuntimeError(
                f"Matched repeat-average identity differs for {column}"
            )
    return pd.concat([right, left], ignore_index=True)


def paired_scaffold_bootstrap(
    matched_averaged: pd.DataFrame,
    protocols: list[str],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seed_offsets = {"scaffold": 0, "compound": 1}
    for protocol in protocols:
        local = matched_averaged[
            matched_averaged["protocol"] == protocol
        ]
        full = local[local["model_id"] == FULL_MODEL_ID].sort_values(
            "row_index"
        )
        context = local[local["model_id"] == MODEL_ID].sort_values(
            "row_index"
        )
        if not np.array_equal(
            full["row_index"].to_numpy(dtype=int),
            context["row_index"].to_numpy(dtype=int),
        ):
            raise RuntimeError("Bootstrap model rows are not aligned")
        y_true = full["y_true"].to_numpy(dtype=np.float64)
        if not np.allclose(
            y_true,
            context["y_true"].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError("Bootstrap truth arrays are not aligned")
        full_pred = full["y_pred"].to_numpy(dtype=np.float64)
        context_pred = context["y_pred"].to_numpy(dtype=np.float64)
        clusters = full["scaffold_id"].astype(str).to_numpy()
        unique_clusters = np.unique(clusters)
        if len(unique_clusters) != 667:
            raise RuntimeError(
                f"Expected 667 bootstrap clusters, got {len(unique_clusters)}"
            )
        cluster_to_positions = {
            cluster: np.where(clusters == cluster)[0]
            for cluster in unique_clusters
        }
        rng = np.random.default_rng(seed + seed_offsets[protocol])
        delta_spearman: list[float] = []
        delta_rmse: list[float] = []
        for bootstrap_index in range(n_bootstrap):
            sampled_clusters = rng.choice(
                unique_clusters,
                size=len(unique_clusters),
                replace=True,
            )
            positions = np.concatenate(
                [
                    cluster_to_positions[cluster]
                    for cluster in sampled_clusters
                ]
            )
            full_s = core.safe_spearman(
                y_true[positions],
                full_pred[positions],
            )
            context_s = core.safe_spearman(
                y_true[positions],
                context_pred[positions],
            )
            full_rmse = float(
                np.sqrt(np.mean((y_true[positions] - full_pred[positions]) ** 2))
            )
            context_rmse = float(
                np.sqrt(
                    np.mean(
                        (y_true[positions] - context_pred[positions]) ** 2
                    )
                )
            )
            delta_spearman.append(full_s - context_s)
            delta_rmse.append(context_rmse - full_rmse)
            if (bootstrap_index + 1) % max(1, n_bootstrap // 10) == 0:
                print(
                    f"[BOOT] {protocol}: "
                    f"{bootstrap_index + 1}/{n_bootstrap}",
                    flush=True,
                )

        full_metrics = core.regression_metrics(y_true, full_pred)
        context_metrics = core.regression_metrics(y_true, context_pred)
        delta_s_mean, delta_s_low, delta_s_high = (
            core.finite_percentile_interval(delta_spearman)
        )
        delta_r_mean, delta_r_low, delta_r_high = (
            core.finite_percentile_interval(delta_rmse)
        )
        row = {
            "record_type": "contrast",
            "protocol": protocol,
            "contrast_id": "full_vs_context_extra_trees",
            "first_model": FULL_MODEL_ID,
            "comparator_model": MODEL_ID,
            "cluster_unit": "scaffold_id",
            "n_clusters": int(len(unique_clusters)),
            "n_bootstrap": int(n_bootstrap),
            "bootstrap_seed": int(seed + seed_offsets[protocol]),
            "delta_direction": (
                "delta_spearman=full-context; "
                "delta_rmse=context-full; positive favors full"
            ),
            "full_spearman_observed": full_metrics["spearman"],
            "context_spearman_observed": context_metrics["spearman"],
            "delta_spearman_observed": (
                full_metrics["spearman"] - context_metrics["spearman"]
            ),
            "delta_spearman_mean": delta_s_mean,
            "delta_spearman_ci_low": delta_s_low,
            "delta_spearman_ci_high": delta_s_high,
            "full_rmse_observed": full_metrics["rmse"],
            "context_rmse_observed": context_metrics["rmse"],
            "delta_rmse_observed": (
                context_metrics["rmse"] - full_metrics["rmse"]
            ),
            "delta_rmse_mean": delta_r_mean,
            "delta_rmse_ci_low": delta_r_low,
            "delta_rmse_ci_high": delta_r_high,
        }
        numeric_values = [
            value
            for key, value in row.items()
            if key.endswith("_observed")
            or key.endswith("_mean")
            or key.endswith("_low")
            or key.endswith("_high")
        ]
        if not np.all(np.isfinite(numeric_values)):
            raise RuntimeError("Bootstrap summary contains non-finite values")
        if not (
            row["delta_spearman_ci_low"]
            <= row["delta_spearman_ci_high"]
            and row["delta_rmse_ci_low"] <= row["delta_rmse_ci_high"]
        ):
            raise RuntimeError("Bootstrap interval endpoints are reversed")
        rows.append(row)
    return pd.DataFrame(rows)


def artifact_manifest(output_dir: Path) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(output_dir.glob("*.csv")):
        try:
            n_rows = int(len(pd.read_csv(path)))
        except pd.errors.EmptyDataError:
            n_rows = 0
        artifacts[path.stem] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "n_rows": n_rows,
            "size_bytes": int(path.stat().st_size),
        }
    return artifacts


def verify_parent_unchanged(
    core_results_dir: Path,
) -> None:
    require_hash(
        core_results_dir / "run_manifest.json",
        FROZEN_CORE_MANIFEST_SHA256,
        "parent manifest after extension",
    )
    require_hash(
        core_results_dir / "all_cross_fitted_predictions.csv",
        FROZEN_CORE_RAW_PREDICTIONS_SHA256,
        "parent raw predictions after extension",
    )
    require_hash(
        core_results_dir / "repeat_averaged_cross_fitted_predictions.csv",
        FROZEN_CORE_AVERAGED_PREDICTIONS_SHA256,
        "parent averaged predictions after extension",
    )
    for protocol in core.PROTOCOL_ORDER:
        require_hash(
            core_results_dir / f"{protocol}_outer_fold_metrics.csv",
            FROZEN_CORE_OUTER_METRICS_SHA256[protocol],
            f"parent {protocol} outer-fold metrics after extension",
        )


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if core.MODEL_ORDER.index(FULL_MODEL_ID) != FULL_MODEL_POSITION:
        raise RuntimeError("Parent full-model position differs from frozen seed")
    if CORE_SCRIPT.resolve() != Path(core.__file__).resolve():
        raise RuntimeError("Imported parent analysis module from an unexpected path")
    require_hash(
        CORE_SCRIPT,
        FROZEN_CORE_SCRIPT_SHA256,
        "imported parent analysis script",
    )

    (
        parent_manifest,
        parent_full_raw,
        parent_averaged,
        parent_outer_metrics,
        parent_hashes,
    ) = verify_parent(args.core_results_dir.resolve())
    identity, identity_sha256 = build_identity(args, parent_hashes)
    manifest, already_complete = initialize_output(
        args,
        identity,
        identity_sha256,
    )
    if already_complete:
        return
    args.output_dir = args.output_dir.resolve()

    rows, sanitization_audit = core.load_rows(
        Path(parent_manifest["data_file"])
    )
    # Match the parent core exactly: labels were converted to float32 before
    # every inner and outer fit and before being written to its OOF tables.
    y = rows["pDC50"].to_numpy(dtype=np.float32)
    scaffold_ids = core.build_scaffold_ids(rows["canonical_smiles"])
    if len(rows) != 1560 or len(np.unique(scaffold_ids)) != 667:
        raise RuntimeError("Extension data identity counts differ from parent")
    contexts = core.context_frame(rows)
    split_map = validate_outer_assignments(
        rows,
        scaffold_ids,
        parent_full_raw,
        args.protocols,
    )
    print(
        "[QA] regenerated outer assignments exactly match parent OOF folds",
        flush=True,
    )

    protocol_predictions: list[pd.DataFrame] = []
    for protocol in args.protocols:
        protocol_predictions.append(
            run_protocol(
                protocol,
                args,
                rows,
                contexts,
                y,
                scaffold_ids,
                split_map[protocol],
                parent_full_raw,
                parent_outer_metrics[protocol],
            )
        )
        validate_protocol_artifacts(
            args.output_dir,
            protocol,
            1
            if args.max_outer_splits is not None
            else FORMAL_OUTER_FOLDS * FORMAL_OUTER_REPEATS,
        )
    predictions = pd.concat(protocol_predictions, ignore_index=True)
    predictions = predictions.sort_values(
        ["protocol", "repeat", "outer_fold", "model_id", "row_index"]
    ).reset_index(drop=True)
    atomic_to_csv(
        predictions,
        args.output_dir
        / "all_context_extra_trees_cross_fitted_predictions.csv",
    )

    manifest["n_rows"] = int(len(rows))
    manifest["n_unique_smiles"] = int(rows["canonical_smiles"].nunique())
    manifest["n_scaffolds"] = int(len(np.unique(scaffold_ids)))
    manifest["n_sanitization_audit_records"] = int(len(sanitization_audit))
    manifest["model"] = {
        "model_id": MODEL_ID,
        "feature_block": "parent context main effects and interactions only",
        "grid": et_grid(),
        "selection": (
            "pooled inner-OOF Spearman; within 0.005 prefer larger leaf, "
            "then sqrt, lower RMSE and param_id"
        ),
        "n_estimators": args.n_estimators,
        "n_jobs": args.n_jobs,
    }

    if args.smoke:
        if len(predictions) != 312:
            raise RuntimeError("Smoke prediction count must be 312")
        if predictions.duplicated(
            ["protocol", "repeat", "row_index", "model_id"]
        ).any():
            raise RuntimeError("Smoke predictions have duplicate keys")
        if not np.all(np.isfinite(predictions["y_pred"])):
            raise RuntimeError("Smoke predictions contain non-finite values")
        verify_parent_unchanged(args.core_results_dir.resolve())
        manifest["status"] = "smoke_complete"
        manifest["elapsed_seconds"] = float(time.perf_counter() - started)
        manifest["artifacts"] = artifact_manifest(args.output_dir)
        atomic_write_json(args.output_dir / "run_manifest.json", manifest)
        print(
            f"[DONE] smoke completed in {manifest['elapsed_seconds']:.1f}s",
            flush=True,
        )
        return

    validate_formal_predictions(predictions, args.protocols, len(rows))
    averaged = core.average_cross_fitted_predictions(predictions)
    expected_averaged = len(args.protocols) * len(rows)
    if len(averaged) != expected_averaged:
        raise RuntimeError("Context-only repeat averages are incomplete")
    if not (
        averaged["n_oof_predictions"] == FORMAL_OUTER_REPEATS
    ).all() or not (averaged["n_repeats"] == FORMAL_OUTER_REPEATS).all():
        raise RuntimeError("Context-only repeat averages do not contain five OOFs")
    if not np.all(np.isfinite(averaged["y_pred"])) or not np.all(
        np.isfinite(averaged["y_pred_sd_across_repeats"])
    ):
        raise RuntimeError("Context-only repeat averages are non-finite")
    atomic_to_csv(
        averaged,
        args.output_dir / "context_extra_trees_repeat_averaged_predictions.csv",
    )

    repeat_metrics, summary_metrics = core.summarize_predictions(predictions)
    averaged_metrics = core.metrics_on_repeat_averaged_predictions(averaged)
    canonical_metrics = core.metrics_on_canonical_smiles_aggregates(averaged)
    atomic_to_csv(
        repeat_metrics,
        args.output_dir / "context_extra_trees_metrics_by_repeat.csv",
    )
    atomic_to_csv(
        summary_metrics,
        args.output_dir / "context_extra_trees_summary_metrics.csv",
    )
    atomic_to_csv(
        averaged_metrics,
        args.output_dir / "context_extra_trees_repeat_averaged_metrics.csv",
    )
    atomic_to_csv(
        canonical_metrics,
        args.output_dir / "context_extra_trees_canonical_smiles_metrics.csv",
    )

    matched_averaged = validate_alignment(
        averaged,
        parent_averaged,
        args.protocols,
    )
    matched_metrics = core.metrics_on_repeat_averaged_predictions(
        matched_averaged
    )
    atomic_to_csv(
        matched_metrics,
        args.output_dir / "matched_repeat_averaged_metrics.csv",
    )
    bootstrap = paired_scaffold_bootstrap(
        matched_averaged,
        args.protocols,
        args.bootstrap_replicates,
        args.seed,
    )
    if len(bootstrap) != len(args.protocols):
        raise RuntimeError("Paired bootstrap contrast table is incomplete")
    if not (bootstrap["n_clusters"] == 667).all() or not (
        bootstrap["n_bootstrap"] == FORMAL_BOOTSTRAP_REPLICATES
    ).all():
        raise RuntimeError("Paired bootstrap configuration is incomplete")
    atomic_to_csv(
        bootstrap,
        args.output_dir
        / "paired_scaffold_bootstrap_full_vs_context_extra_trees.csv",
    )

    verify_parent_unchanged(args.core_results_dir.resolve())
    manifest["status"] = "complete"
    manifest["elapsed_seconds"] = float(time.perf_counter() - started)
    manifest["completeness"] = {
        "raw_predictions": int(len(predictions)),
        "repeat_averaged_predictions": int(len(averaged)),
        "paired_bootstrap_contrasts": int(len(bootstrap)),
        "all_predictions_finite": True,
        "outer_assignments_match_parent": True,
        "parent_artifacts_unchanged": True,
    }
    manifest["artifacts"] = artifact_manifest(args.output_dir)
    atomic_write_json(args.output_dir / "run_manifest.json", manifest)
    print(
        f"[DONE] formal extension completed in "
        f"{manifest['elapsed_seconds']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
