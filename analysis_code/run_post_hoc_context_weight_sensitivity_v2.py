#!/usr/bin/env python
"""Corrected post-hoc CPU sensitivity: portable context and fit weights.

The runner is intentionally restricted to the frozen 1,560-row public core.
It imports the exact split and feature machinery from the frozen core and OOD
runners, verifies the formal baseline artifacts, and writes independent
post-hoc outputs. Version 2 aligns the training-response dtype with the frozen
confirmatory implementation. It never reads a prospective or
collaborator-restricted panel.
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
import warnings
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
from sklearn.exceptions import UndefinedMetricWarning


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_confirmatory_cpu_v1 as core  # noqa: E402
import run_confirmatory_ood_cpu_v1 as ood  # noqa: E402


PROTOCOL_VERSION = "post_hoc_context_weight_sensitivity_v2.0"
DATA_FILE = core.DATA_FILE
FROZEN_DATA_SHA256 = core.FROZEN_DATA_SHA256
MASTER_PROTOCOL = (
    PROJECT_DIR
    / "docs"
    / "post_hoc_computational_extension_master_protocol_v1.md"
)
MASTER_PROTOCOL_SHA256 = (
    "7dee28f45e3ddf8d6299622a8e494c50492b3b2614f39ce8b2fdd2a1be754ff8"
)
CORRECTION_DOCUMENT = (
    PROJECT_DIR
    / "docs"
    / "post_hoc_float32_response_correction_v2.md"
)
CORRECTION_DOCUMENT_SHA256 = (
    "75a8b0e4f894cead9d72039eab9e6162a89d2a20c79afceda8a2aae1ffd78f3f"
)
PROTOCOL_DOCUMENT = (
    PROJECT_DIR
    / "docs"
    / "post_hoc_context_weight_sensitivity_protocol_v2.md"
)
CORE_REPORT_DIR = PROJECT_DIR / "reports" / "confirmatory_cpu_v1"
OOD_REPORT_DIR = PROJECT_DIR / "reports" / "confirmatory_ood_cpu_v1"
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "reports" / "post_hoc_context_weight_sensitivity_v2"
)
SUPERSEDED_V1_OUTPUT_DIR = (
    PROJECT_DIR / "reports" / "post_hoc_context_weight_sensitivity_v1"
)
SUPERSEDED_MARKER = "SUPERSEDED_DO_NOT_USE.md"
SUPERSEDED_V1_RUNNER_SHA256 = (
    "e13b058c5958576a30c60ee8bc0f9d2ae593218bc2f4917244459a3d3e2a0eae"
)
TRAINING_RESPONSE_DTYPE = np.dtype(np.float32)

DEFAULT_SEED = core.DEFAULT_SEED
INTERNAL_BASE_MODELS = (
    "chemistry_extra_trees",
    "full_context_extra_trees",
)
WEIGHT_MODES = (
    "uniform",
    "compound_equal",
    "domain_balanced_clipped",
)
WEIGHTED_MODES = WEIGHT_MODES[1:]
INTERNAL_MODEL_ORDER = tuple(
    f"{base}__{mode}"
    for mode in WEIGHT_MODES
    for base in INTERNAL_BASE_MODELS
)
OOD_MODEL_ORDER = (
    "chemistry_extra_trees",
    "full_context_extra_trees",
    "portable_full_extra_trees",
)
INTERNAL_CONTRASTS = {
    "uniform_full_vs_chemistry": (
        "full_context_extra_trees__uniform",
        "chemistry_extra_trees__uniform",
    ),
    "compound_equal_full_vs_chemistry": (
        "full_context_extra_trees__compound_equal",
        "chemistry_extra_trees__compound_equal",
    ),
    "domain_balanced_full_vs_chemistry": (
        "full_context_extra_trees__domain_balanced_clipped",
        "chemistry_extra_trees__domain_balanced_clipped",
    ),
    "chemistry_compound_equal_vs_uniform": (
        "chemistry_extra_trees__compound_equal",
        "chemistry_extra_trees__uniform",
    ),
    "full_compound_equal_vs_uniform": (
        "full_context_extra_trees__compound_equal",
        "full_context_extra_trees__uniform",
    ),
    "chemistry_domain_balanced_vs_uniform": (
        "chemistry_extra_trees__domain_balanced_clipped",
        "chemistry_extra_trees__uniform",
    ),
    "full_domain_balanced_vs_uniform": (
        "full_context_extra_trees__domain_balanced_clipped",
        "full_context_extra_trees__uniform",
    ),
}
OOD_CONTRASTS = {
    "portable_vs_chemistry": (
        "portable_full_extra_trees",
        "chemistry_extra_trees",
    ),
    "full_vs_chemistry": (
        "full_context_extra_trees",
        "chemistry_extra_trees",
    ),
    "portable_vs_full": (
        "portable_full_extra_trees",
        "full_context_extra_trees",
    ),
}
TREE_GRID = tuple(
    dict(params)
    for params in core.model_param_grid("full_context_extra_trees")
)
DOMAIN_WEIGHT_CAP = 10.0

PORTABLE_CONTEXT_SPEC = {
    "source_ood": {
        "main": (
            "recruiting_protein",
            "target_protein",
            "cell_line",
            "assay_method",
            "activity_time",
            "mode_of_action",
        ),
        "interactions": {
            "recruiter_target": (
                "recruiting_protein",
                "target_protein",
            ),
            "target_cell": ("target_protein", "cell_line"),
            "assay_target": ("assay_method", "target_protein"),
        },
        "dropped_main": ("source_database",),
        "dropped_interactions": ("source_target",),
    },
    "target_ood": {
        "main": (
            "source_database",
            "recruiting_protein",
            "cell_line",
            "assay_method",
            "activity_time",
            "mode_of_action",
        ),
        "interactions": {},
        "dropped_main": ("target_protein",),
        "dropped_interactions": (
            "source_target",
            "recruiter_target",
            "target_cell",
            "assay_target",
        ),
    },
}

PROGRESS_FILES = (
    "configuration.json",
    "internal_predictions.csv",
    "internal_tuning_metrics.csv",
    "internal_selected_hyperparameters.csv",
    "internal_weight_diagnostics.csv",
    "internal_split_audit.csv",
    "ood_predictions.csv",
    "ood_portable_tuning_metrics.csv",
    "ood_portable_selected_hyperparameters.csv",
    "ood_portable_inner_split_audit.csv",
    "ood_portable_feature_audit.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-internal-splits",
        type=int,
        default=None,
        help="Smoke helper: keep the first N scaffold outer splits.",
    )
    parser.add_argument(
        "--max-ood-groups",
        type=int,
        default=None,
        help="Smoke helper: keep the first N groups per OOD axis.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only a configuration-matched partial run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove only known artifacts from the selected output directory.",
    )
    args = parser.parse_args()
    if args.n_estimators < 10:
        parser.error("n-estimators must be >= 10")
    if not 1 <= args.n_jobs <= 4:
        parser.error("n-jobs must be between 1 and 4")
    if args.inner_folds < 2:
        parser.error("inner-folds must be >= 2")
    if args.bootstrap_replicates < 0:
        parser.error("bootstrap-replicates must be >= 0")
    if (
        args.max_internal_splits is not None
        and args.max_internal_splits < 1
    ):
        parser.error("max-internal-splits must be >= 1")
    if args.max_ood_groups is not None and args.max_ood_groups < 1:
        parser.error("max-ood-groups must be >= 1")
    if args.resume and args.overwrite:
        parser.error("resume and overwrite are mutually exclusive")
    output_dir = args.output_dir.resolve()
    if output_dir == SUPERSEDED_V1_OUTPUT_DIR.resolve():
        parser.error("v2 refuses the superseded v1 output directory")
    if (output_dir / SUPERSEDED_MARKER).exists():
        parser.error(
            "v2 refuses any output directory carrying a superseded marker"
        )
    if "_v2" not in output_dir.name:
        parser.error("v2 output directory name must explicitly contain _v2")
    return args


def sha256_file(path: Path) -> str:
    return core.sha256_file(path)


def canonical_sha256(payload: Any) -> str:
    content = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=core.json_ready,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def sequence_sha256(values: Iterable[Any]) -> str:
    return canonical_sha256(list(values))


def formal_run(args: argparse.Namespace) -> bool:
    return bool(
        args.n_estimators == 600
        and args.n_jobs <= 4
        and args.inner_folds == 4
        and args.bootstrap_replicates == 10_000
        and args.seed == DEFAULT_SEED
        and args.max_internal_splits is None
        and args.max_ood_groups is None
    )


def runtime_environment(args: argparse.Namespace) -> dict[str, Any]:
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
        "n_jobs_per_fit": int(args.n_jobs),
        "gpu_required": False,
    }


def artifact_hash_from_inventory(
    inventory_path: Path,
    filename: str,
) -> str:
    inventory = pd.read_csv(inventory_path)
    name_col = (
        "filename" if "filename" in inventory.columns else "relative_path"
    )
    local = inventory[inventory[name_col].astype(str) == filename]
    if len(local) != 1:
        raise RuntimeError(
            f"Expected one inventory row for {filename}: {inventory_path}"
        )
    return str(local.iloc[0]["sha256"])


def verify_baseline_artifact(
    directory: Path,
    filename: str,
) -> dict[str, Any]:
    path = directory / filename
    inventory_path = directory / "artifact_sha256.csv"
    expected = artifact_hash_from_inventory(inventory_path, filename)
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"Formal baseline artifact hash mismatch: {path}"
        )
    return {
        "path": str(path.resolve()),
        "sha256": observed,
        "size_bytes": int(path.stat().st_size),
    }


def configuration_payload(
    args: argparse.Namespace,
    baseline_files: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "analysis_mode": (
            "formal_post_hoc_sensitivity"
            if formal_run(args)
            else "smoke_test_only"
        ),
        "data_file": str(DATA_FILE.resolve()),
        "data_sha256": sha256_file(DATA_FILE),
        "script_file": str(script_path),
        "script_sha256": sha256_file(script_path),
        "master_protocol": str(MASTER_PROTOCOL.resolve()),
        "master_protocol_sha256": sha256_file(MASTER_PROTOCOL),
        "correction_document": str(CORRECTION_DOCUMENT.resolve()),
        "correction_document_sha256": sha256_file(
            CORRECTION_DOCUMENT
        ),
        "child_protocol": str(PROTOCOL_DOCUMENT.resolve()),
        "child_protocol_sha256": sha256_file(PROTOCOL_DOCUMENT),
        "parent_implementations": {
            "confirmatory_core": {
                "path": str(Path(core.__file__).resolve()),
                "sha256": sha256_file(Path(core.__file__).resolve()),
            },
            "confirmatory_ood": {
                "path": str(Path(ood.__file__).resolve()),
                "sha256": sha256_file(Path(ood.__file__).resolve()),
            },
        },
        "superseded_v1_runner_sha256": (
            SUPERSEDED_V1_RUNNER_SHA256
        ),
        "training_response_dtype": TRAINING_RESPONSE_DTYPE.name,
        "metric_calculation": "unchanged; explicit float64 casts retained",
        "baseline_files": baseline_files,
        "outer_folds": 5,
        "outer_repeats": 5,
        "inner_folds": int(args.inner_folds),
        "n_estimators": int(args.n_estimators),
        "n_jobs": int(args.n_jobs),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "seed": int(args.seed),
        "max_internal_splits": args.max_internal_splits,
        "max_ood_groups": args.max_ood_groups,
        "tree_grid": list(TREE_GRID),
        "weight_modes": list(WEIGHT_MODES),
        "domain_definition": (
            "exact source_database || target_protein"
        ),
        "domain_weight_cap_after_first_normalization": (
            DOMAIN_WEIGHT_CAP
        ),
        "portable_context_spec": PORTABLE_CONTEXT_SPEC,
        "internal_contrasts": INTERNAL_CONTRASTS,
        "ood_contrasts": OOD_CONTRASTS,
        "environment": runtime_environment(args),
    }


def prepare_output_directory(
    output_dir: Path,
    args: argparse.Namespace,
    configuration: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "configuration.json"
    if args.resume:
        if not config_path.exists():
            raise RuntimeError("Resume requested without configuration.json")
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if canonical_sha256(previous) != canonical_sha256(configuration):
            raise RuntimeError("Resume configuration does not match")
        return
    known_outputs = {
        *PROGRESS_FILES,
        "context_sanitization_audit.csv",
        "internal_repeat_metrics.csv",
        "internal_repeat_averaged_predictions.csv",
        "internal_summary_metrics.csv",
        "internal_domain_metrics.csv",
        "internal_canonical_compound_metrics.csv",
        "internal_paired_scaffold_bootstrap.csv",
        "ood_domain_metrics.csv",
        "ood_aggregate_metrics.csv",
        "ood_paired_scaffold_bootstrap.csv",
        "ood_domain_scaffold_bootstrap.csv",
        "qa_summary.json",
        "qa_summary.md",
        "results_brief_zh.md",
        "run_manifest.json",
        "artifact_sha256.csv",
    }
    existing = [path for path in output_dir.iterdir()]
    if existing and not args.overwrite:
        raise RuntimeError(
            f"Output directory is not empty: {output_dir}; "
            "use --resume or --overwrite"
        )
    if args.overwrite:
        unexpected = [
            path.name for path in existing if path.name not in known_outputs
        ]
        if unexpected:
            raise RuntimeError(
                "Overwrite refused because unexpected files are present: "
                f"{unexpected}"
            )
        for path in existing:
            if path.is_file() or path.is_symlink():
                path.unlink()
            else:
                raise RuntimeError(
                    f"Overwrite refuses non-file artifact: {path}"
                )
    core.write_json(config_path, configuration)


def portable_context_frame(
    rows: pd.DataFrame,
    protocol: str,
) -> pd.DataFrame:
    spec = PORTABLE_CONTEXT_SPEC[protocol]
    frame = rows[list(spec["main"])].copy()
    for name, columns in spec["interactions"].items():
        frame[name] = (
            frame[list(columns)].astype(str).agg("||".join, axis=1)
        )
    expected = set(spec["main"]) | set(spec["interactions"])
    if set(frame.columns) != expected:
        raise RuntimeError("Portable context columns changed unexpectedly")
    prohibited = set(spec["dropped_main"]) | set(
        spec["dropped_interactions"]
    )
    if prohibited.intersection(frame.columns):
        raise RuntimeError("Held-axis feature entered portable context")
    return frame.fillna("NA").astype(str)


def build_portable_features(
    rows: pd.DataFrame,
    morgan: np.ndarray,
    descriptors: np.ndarray,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
    protocol: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rdkit_fit, rdkit_eval, rdkit_meta = core.prepare_rdkit_fold(
        descriptors,
        fit_idx,
        eval_idx,
    )
    chemistry_fit = np.hstack(
        [
            morgan[fit_idx].astype(np.float32),
            rdkit_fit,
        ]
    ).astype(np.float32)
    chemistry_eval = np.hstack(
        [
            morgan[eval_idx].astype(np.float32),
            rdkit_eval,
        ]
    ).astype(np.float32)
    contexts = portable_context_frame(rows, protocol)
    encoder = core.make_one_hot_encoder()
    context_fit = encoder.fit_transform(
        contexts.iloc[fit_idx]
    ).astype(np.float32)
    context_eval = encoder.transform(
        contexts.iloc[eval_idx]
    ).astype(np.float32)
    return (
        np.hstack([chemistry_fit, context_fit]).astype(np.float32),
        np.hstack([chemistry_eval, context_eval]).astype(np.float32),
        {
            **rdkit_meta,
            "chemistry_dim": int(chemistry_fit.shape[1]),
            "portable_context_dim": int(context_fit.shape[1]),
            "portable_full_dim": int(
                chemistry_fit.shape[1] + context_fit.shape[1]
            ),
        },
    )


def fit_extra_trees(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_eval: np.ndarray,
    params: dict[str, Any],
    seed: int,
    n_estimators: int,
    n_jobs: int,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    train_y = np.asarray(y_fit)
    if train_y.dtype != TRAINING_RESPONSE_DTYPE:
        raise RuntimeError(
            "Corrected v2 refit requires float32 training responses"
        )
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=params["max_features"],
        max_depth=None,
        bootstrap=False,
        random_state=int(seed),
        n_jobs=int(n_jobs),
    )
    model.fit(x_fit, train_y, sample_weight=sample_weight)
    pred = np.asarray(model.predict(x_eval), dtype=np.float32)
    if len(pred) != len(x_eval) or not np.all(np.isfinite(pred)):
        raise RuntimeError("ExtraTrees produced invalid predictions")
    return pred


def training_weights(
    rows: pd.DataFrame,
    fit_idx: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    fit_rows = rows.iloc[fit_idx]
    if mode == "uniform":
        values = np.ones(len(fit_idx), dtype=np.float64)
        pre_clip_weight_max = 1.0
        post_clip_prenorm_mean = 1.0
        post_clip_prenorm_max = 1.0
        clip_threshold = float("nan")
        group_count = 1
        clipping_fraction = 0.0
        compound_total_relative_range = float("nan")
    elif mode == "compound_equal":
        keys = fit_rows["canonical_smiles"].astype(str)
        counts = keys.value_counts()
        values = 1.0 / keys.map(counts).to_numpy(dtype=np.float64)
        values /= float(np.mean(values))
        pre_clip_weight_max = float(np.max(values))
        post_clip_prenorm_mean = float(np.mean(values))
        post_clip_prenorm_max = float(np.max(values))
        clip_threshold = float("nan")
        totals = (
            pd.DataFrame({"key": keys.to_numpy(), "weight": values})
            .groupby("key", sort=False)["weight"]
            .sum()
            .to_numpy(dtype=np.float64)
        )
        compound_total_relative_range = float(
            (np.max(totals) - np.min(totals)) / np.mean(totals)
        )
        if compound_total_relative_range > 1e-12:
            raise RuntimeError(
                "Compound-equal normalized totals are not equal"
            )
        group_count = int(keys.nunique())
        clipping_fraction = 0.0
    elif mode == "domain_balanced_clipped":
        keys = (
            fit_rows["source_database"].astype(str)
            + "||"
            + fit_rows["target_protein"].astype(str)
        )
        counts = keys.value_counts()
        values = 1.0 / keys.map(counts).to_numpy(dtype=np.float64)
        values /= float(np.mean(values))
        pre_clip_weight_max = float(np.max(values))
        clipped = values > DOMAIN_WEIGHT_CAP
        clipping_fraction = float(np.mean(clipped))
        values = np.minimum(values, DOMAIN_WEIGHT_CAP)
        post_clip_prenorm_mean = float(np.mean(values))
        post_clip_prenorm_max = float(np.max(values))
        clip_threshold = DOMAIN_WEIGHT_CAP
        values /= float(np.mean(values))
        group_count = int(keys.nunique())
        compound_total_relative_range = float("nan")
    else:
        raise ValueError(f"Unknown weight mode: {mode}")
    if (
        len(values) != len(fit_idx)
        or not np.all(np.isfinite(values))
        or np.any(values <= 0)
        or not math.isclose(
            float(np.mean(values)),
            1.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError(f"Invalid training weights for {mode}")
    effective_n = float(
        np.square(np.sum(values)) / np.sum(np.square(values))
    )
    return values, {
        "weight_mode": mode,
        "n_fit_rows": int(len(fit_idx)),
        "n_weight_groups": int(group_count),
        "weight_min": float(np.min(values)),
        "weight_max": float(np.max(values)),
        "weight_mean": float(np.mean(values)),
        "weight_sd": float(np.std(values, ddof=0)),
        "pre_clip_weight_max": pre_clip_weight_max,
        "clip_threshold": clip_threshold,
        "post_clip_prenorm_mean": post_clip_prenorm_mean,
        "post_clip_prenorm_max": post_clip_prenorm_max,
        "clipping_fraction": float(clipping_fraction),
        "effective_sample_size": effective_n,
        "effective_sample_fraction": float(effective_n / len(values)),
        "compound_total_relative_range": (
            compound_total_relative_range
        ),
    }


def select_tree_params(
    candidate_predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scored: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    for params in TREE_GRID:
        param_id = str(params["param_id"])
        pred = candidate_predictions[param_id]
        if not np.all(np.isfinite(pred)):
            raise RuntimeError(f"Incomplete inner prediction: {param_id}")
        metrics = core.regression_metrics(y_true, pred)
        score = float(metrics[core.PRIMARY_METRIC])
        if not math.isfinite(score):
            score = -float("inf")
        rank = core.regularization_rank(
            "full_context_extra_trees",
            params,
        )
        scored.append(
            {
                "score": score,
                "rmse": float(metrics["rmse"]),
                "param_id": param_id,
                "params": params,
                "regularization_rank": rank,
            }
        )
        tuning_rows.append(
            {
                "param_id": param_id,
                "inner_n": int(len(y_true)),
                **metrics,
            }
        )
    best_score = max(row["score"] for row in scored)
    eligible = [
        row for row in scored if row["score"] >= best_score - 0.005
    ]
    eligible.sort(
        key=lambda row: (
            tuple(-value for value in row["regularization_rank"]),
            row["rmse"],
            row["param_id"],
        )
    )
    return dict(eligible[0]["params"]), tuning_rows


def records_for_split(
    frame: pd.DataFrame,
    repeat: int,
    fold: int,
) -> pd.DataFrame:
    return frame[
        (frame["repeat"].astype(int) == int(repeat))
        & (frame["outer_fold"].astype(int) == int(fold))
    ].copy()


def read_records(path: Path, resume: bool) -> list[dict[str, Any]]:
    if not resume or not path.exists() or path.stat().st_size == 0:
        return []
    return pd.read_csv(path).to_dict("records")


def discard_outer_records(
    records: list[dict[str, Any]],
    repeat: int,
    fold: int,
) -> list[dict[str, Any]]:
    return [
        row
        for row in records
        if not (
            int(row.get("repeat", -1)) == repeat
            and int(row.get("outer_fold", -1)) == fold
        )
    ]


def internal_checkpoint_complete(
    split: core.OuterSplit,
    records: dict[str, list[dict[str, Any]]],
    inner_folds: int,
) -> bool:
    local = {
        key: [
            row
            for row in values
            if int(row.get("repeat", -1)) == split.repeat
            and int(row.get("outer_fold", -1)) == split.fold
        ]
        for key, values in records.items()
    }
    return bool(
        len(local["predictions"])
        == len(split.test_idx) * len(INTERNAL_MODEL_ORDER)
        and len(local["tuning"])
        == len(WEIGHTED_MODES)
        * len(INTERNAL_BASE_MODELS)
        * len(TREE_GRID)
        and len(local["selected"])
        == len(INTERNAL_MODEL_ORDER)
        and len(local["weights"])
        == len(WEIGHTED_MODES) * (inner_folds + 1)
        and len(local["audit"]) == 1
    )


def write_internal_progress(
    output_dir: Path,
    records: dict[str, list[dict[str, Any]]],
) -> None:
    mapping = {
        "predictions": "internal_predictions.csv",
        "tuning": "internal_tuning_metrics.csv",
        "selected": "internal_selected_hyperparameters.csv",
        "weights": "internal_weight_diagnostics.csv",
        "audit": "internal_split_audit.csv",
    }
    for key, filename in mapping.items():
        core.atomic_to_csv(pd.DataFrame(records[key]), output_dir / filename)


def validate_formal_internal_baseline(
    split: core.OuterSplit,
    baseline_predictions: pd.DataFrame,
    baseline_selected: pd.DataFrame,
    rows: pd.DataFrame,
    y: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    local_pred = records_for_split(
        baseline_predictions,
        split.repeat,
        split.fold,
    )
    local_pred = local_pred[
        local_pred["model_id"].isin(INTERNAL_BASE_MODELS)
    ].copy()
    expected_test = set(int(value) for value in split.test_idx)
    params_by_model: dict[str, dict[str, Any]] = {}
    for model_id in INTERNAL_BASE_MODELS:
        frame = local_pred[local_pred["model_id"] == model_id]
        if (
            len(frame) != len(split.test_idx)
            or set(frame["row_index"].astype(int)) != expected_test
            or frame["row_index"].duplicated().any()
        ):
            raise RuntimeError(
                f"Formal internal baseline row mismatch: {model_id}"
            )
        ordered = frame.set_index("row_index").loc[split.test_idx]
        if not np.allclose(
            ordered["y_true"].to_numpy(dtype=np.float64),
            y[split.test_idx],
            atol=1e-7,
            rtol=1e-7,
        ):
            raise RuntimeError("Formal internal outcomes changed")
        expected_qc = rows.iloc[split.test_idx]["qc_id"].astype(str).to_numpy()
        if not np.array_equal(
            ordered["qc_id"].astype(str).to_numpy(),
            expected_qc,
        ):
            raise RuntimeError("Formal internal qc_id alignment changed")
        selected = records_for_split(
            baseline_selected,
            split.repeat,
            split.fold,
        )
        selected = selected[selected["model_id"] == model_id]
        if len(selected) != 1:
            raise RuntimeError("Formal selected-parameter row missing")
        record = selected.iloc[0]
        param_id = str(record["param_id"])
        grid = {
            str(params["param_id"]): params for params in TREE_GRID
        }
        if param_id not in grid:
            raise RuntimeError("Formal selected parameter is off-grid")
        if set(frame["param_id"].astype(str)) != {param_id}:
            raise RuntimeError("Formal prediction parameter linkage changed")
        params_by_model[model_id] = dict(grid[param_id])
    return local_pred, params_by_model


def run_internal_sensitivity(
    args: argparse.Namespace,
    rows: pd.DataFrame,
    y: np.ndarray,
    morgan: np.ndarray,
    descriptors: np.ndarray,
    scaffold_ids: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:
    baseline_predictions = pd.read_csv(
        CORE_REPORT_DIR / "scaffold_cross_fitted_predictions.csv"
    )
    baseline_selected = pd.read_csv(
        CORE_REPORT_DIR / "scaffold_selected_hyperparameters.csv"
    )
    splits = core.make_outer_splits(
        "scaffold",
        rows,
        scaffold_ids,
        5,
        5,
        args.seed,
    )
    if args.max_internal_splits is not None:
        splits = splits[: args.max_internal_splits]

    records = {
        "predictions": read_records(
            output_dir / "internal_predictions.csv", args.resume
        ),
        "tuning": read_records(
            output_dir / "internal_tuning_metrics.csv", args.resume
        ),
        "selected": read_records(
            output_dir / "internal_selected_hyperparameters.csv",
            args.resume,
        ),
        "weights": read_records(
            output_dir / "internal_weight_diagnostics.csv",
            args.resume,
        ),
        "audit": read_records(
            output_dir / "internal_split_audit.csv", args.resume
        ),
    }

    started = time.perf_counter()
    for split_position, split in enumerate(splits, start=1):
        if args.resume and internal_checkpoint_complete(
            split,
            records,
            args.inner_folds,
        ):
            print(
                f"[INTERNAL RESUME] repeat={split.repeat} "
                f"fold={split.fold} complete",
                flush=True,
            )
            continue
        if args.resume:
            for key in records:
                records[key] = discard_outer_records(
                    records[key],
                    split.repeat,
                    split.fold,
                )
        baseline_local, uniform_params = (
            validate_formal_internal_baseline(
                split,
                baseline_predictions,
                baseline_selected,
                rows,
                y,
            )
        )
        train_scaffolds = set(scaffold_ids[split.train_idx])
        test_scaffolds = set(scaffold_ids[split.test_idx])
        overlap = train_scaffolds.intersection(test_scaffolds)
        if overlap:
            raise RuntimeError("Internal outer scaffold leakage")
        records["audit"].append(
            {
                "protocol": "scaffold",
                "repeat": split.repeat,
                "outer_fold": split.fold,
                "split_seed": split.split_seed,
                "n_train_rows": int(len(split.train_idx)),
                "n_test_rows": int(len(split.test_idx)),
                "n_train_scaffolds": int(len(train_scaffolds)),
                "n_test_scaffolds": int(len(test_scaffolds)),
                "n_scaffold_overlap": 0,
                "train_indices_sha256": sequence_sha256(
                    int(value) for value in split.train_idx
                ),
                "test_indices_sha256": sequence_sha256(
                    int(value) for value in split.test_idx
                ),
            }
        )
        for model_id in INTERNAL_BASE_MODELS:
            frame = (
                baseline_local[
                    baseline_local["model_id"] == model_id
                ]
                .sort_values("row_index")
                .copy()
            )
            for record in frame.to_dict("records"):
                records["predictions"].append(
                    {
                        **record,
                        "representation": model_id.replace(
                            "_extra_trees", ""
                        ),
                        "weight_mode": "uniform",
                        "model_id": f"{model_id}__uniform",
                        "baseline_provenance": (
                            "confirmatory_cpu_v1 formal prediction"
                        ),
                    }
                )
            records["selected"].append(
                {
                    "protocol": "scaffold",
                    "repeat": split.repeat,
                    "outer_fold": split.fold,
                    "representation": model_id.replace(
                        "_extra_trees", ""
                    ),
                    "weight_mode": "uniform",
                    "model_id": f"{model_id}__uniform",
                    "selection_source": (
                        "confirmatory_cpu_v1 formal nested selection"
                    ),
                    **uniform_params[model_id],
                }
            )

        inner_seed = (
            args.seed + split.repeat * 10_000 + split.fold * 100
        )
        inner_splits = core.balanced_group_splits(
            split.train_idx,
            scaffold_ids,
            args.inner_folds,
            inner_seed,
        )
        local_position = {
            int(row_idx): position
            for position, row_idx in enumerate(split.train_idx.tolist())
        }
        candidate_predictions = {
            (mode, model_id, str(params["param_id"])): np.full(
                len(split.train_idx),
                np.nan,
                dtype=np.float32,
            )
            for mode in WEIGHTED_MODES
            for model_id in INTERNAL_BASE_MODELS
            for params in TREE_GRID
        }
        for inner_fold, (fit_idx, eval_idx) in enumerate(
            inner_splits,
            start=1,
        ):
            if set(scaffold_ids[fit_idx]).intersection(
                set(scaffold_ids[eval_idx])
            ):
                raise RuntimeError("Internal inner scaffold leakage")
            features, _ = core.build_fold_features(
                rows,
                morgan,
                descriptors,
                fit_idx,
                eval_idx,
            )
            eval_positions = np.asarray(
                [local_position[int(value)] for value in eval_idx],
                dtype=np.int64,
            )
            for mode in WEIGHTED_MODES:
                weights, diagnostics = training_weights(
                    rows,
                    fit_idx,
                    mode,
                )
                records["weights"].append(
                    {
                        "protocol": "scaffold",
                        "repeat": split.repeat,
                        "outer_fold": split.fold,
                        "fit_stage": "inner",
                        "inner_fold": inner_fold,
                        "fit_indices_sha256": sequence_sha256(
                            int(value) for value in fit_idx
                        ),
                        "applies_to_models": "|".join(
                            INTERNAL_BASE_MODELS
                        ),
                        **diagnostics,
                    }
                )
                for model_id in INTERNAL_BASE_MODELS:
                    feature_key = (
                        "chemistry"
                        if model_id == "chemistry_extra_trees"
                        else "full"
                    )
                    x_fit, x_eval = features[feature_key]
                    for params in TREE_GRID:
                        pred = fit_extra_trees(
                            x_fit,
                            y[fit_idx],
                            x_eval,
                            params,
                            seed=inner_seed + inner_fold * 101,
                            n_estimators=args.n_estimators,
                            n_jobs=args.n_jobs,
                            sample_weight=weights,
                        )
                        candidate_predictions[
                            (mode, model_id, str(params["param_id"]))
                        ][eval_positions] = pred

        selected_weighted: dict[
            tuple[str, str], dict[str, Any]
        ] = {}
        inner_truth = y[split.train_idx]
        for mode in WEIGHTED_MODES:
            for model_id in INTERNAL_BASE_MODELS:
                candidates = {
                    str(params["param_id"]): candidate_predictions[
                        (mode, model_id, str(params["param_id"]))
                    ]
                    for params in TREE_GRID
                }
                selected, tuning = select_tree_params(
                    candidates,
                    inner_truth,
                )
                selected_weighted[(mode, model_id)] = selected
                for row in tuning:
                    records["tuning"].append(
                        {
                            "protocol": "scaffold",
                            "repeat": split.repeat,
                            "outer_fold": split.fold,
                            "representation": model_id.replace(
                                "_extra_trees", ""
                            ),
                            "weight_mode": mode,
                            "model_id": f"{model_id}__{mode}",
                            "selection_metric_weighting": "unweighted",
                            **row,
                        }
                    )
                records["selected"].append(
                    {
                        "protocol": "scaffold",
                        "repeat": split.repeat,
                        "outer_fold": split.fold,
                        "representation": model_id.replace(
                            "_extra_trees", ""
                        ),
                        "weight_mode": mode,
                        "model_id": f"{model_id}__{mode}",
                        "selection_source": (
                            "weighted inner fits; unweighted OOF Spearman"
                        ),
                        **selected,
                    }
                )

        final_features, _ = core.build_fold_features(
            rows,
            morgan,
            descriptors,
            split.train_idx,
            split.test_idx,
        )
        identity = (
            baseline_local[
                baseline_local["model_id"]
                == "chemistry_extra_trees"
            ]
            .set_index("row_index")
            .loc[split.test_idx]
            .reset_index()
        )
        for mode in WEIGHTED_MODES:
            weights, diagnostics = training_weights(
                rows,
                split.train_idx,
                mode,
            )
            records["weights"].append(
                {
                    "protocol": "scaffold",
                    "repeat": split.repeat,
                    "outer_fold": split.fold,
                    "fit_stage": "outer_final",
                    "inner_fold": np.nan,
                    "fit_indices_sha256": sequence_sha256(
                        int(value) for value in split.train_idx
                    ),
                    "applies_to_models": "|".join(
                        INTERNAL_BASE_MODELS
                    ),
                    **diagnostics,
                }
            )
            for model_id in INTERNAL_BASE_MODELS:
                params = selected_weighted[(mode, model_id)]
                feature_key = (
                    "chemistry"
                    if model_id == "chemistry_extra_trees"
                    else "full"
                )
                model_position = (
                    4 if model_id == "chemistry_extra_trees" else 5
                )
                model_seed = (
                    args.seed
                    + split.repeat * 100_000
                    + split.fold * 1_000
                    + model_position
                )
                pred = fit_extra_trees(
                    final_features[feature_key][0],
                    y[split.train_idx],
                    final_features[feature_key][1],
                    params,
                    seed=model_seed,
                    n_estimators=args.n_estimators,
                    n_jobs=args.n_jobs,
                    sample_weight=weights,
                )
                for local_idx, source in identity.iterrows():
                    records["predictions"].append(
                        {
                            **source.to_dict(),
                            "model_id": f"{model_id}__{mode}",
                            "param_id": str(params["param_id"]),
                            "y_pred": float(pred[local_idx]),
                            "representation": model_id.replace(
                                "_extra_trees", ""
                            ),
                            "weight_mode": mode,
                            "baseline_provenance": "new post-hoc refit",
                        }
                    )
        write_internal_progress(output_dir, records)
        if not internal_checkpoint_complete(
            split,
            records,
            args.inner_folds,
        ):
            raise RuntimeError("Internal split checkpoint is incomplete")
        elapsed = time.perf_counter() - started
        print(
            f"[INTERNAL] {split_position}/{len(splits)} "
            f"repeat={split.repeat} fold={split.fold} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
    return pd.DataFrame(records["predictions"])


def discard_ood_records(
    records: list[dict[str, Any]],
    protocol: str,
    heldout_group: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in records
        if not (
            str(row.get("protocol")) == protocol
            and str(row.get("heldout_group")) == heldout_group
        )
    ]


def ood_checkpoint_complete(
    split: ood.OODSplit,
    records: dict[str, list[dict[str, Any]]],
    inner_folds: int,
) -> bool:
    local = {
        key: [
            row
            for row in values
            if str(row.get("protocol")) == split.protocol
            and str(row.get("heldout_group")) == split.heldout_group
        ]
        for key, values in records.items()
    }
    return bool(
        len(local["predictions"])
        == len(split.test_idx) * len(OOD_MODEL_ORDER)
        and len(local["tuning"]) == len(TREE_GRID)
        and len(local["selected"]) == 1
        and len(local["inner_audit"]) == inner_folds
        and len(local["feature_audit"]) == 1
    )


def write_ood_progress(
    output_dir: Path,
    records: dict[str, list[dict[str, Any]]],
) -> None:
    mapping = {
        "predictions": "ood_predictions.csv",
        "tuning": "ood_portable_tuning_metrics.csv",
        "selected": "ood_portable_selected_hyperparameters.csv",
        "inner_audit": "ood_portable_inner_split_audit.csv",
        "feature_audit": "ood_portable_feature_audit.csv",
    }
    for key, filename in mapping.items():
        core.atomic_to_csv(pd.DataFrame(records[key]), output_dir / filename)


def validate_formal_ood_baseline(
    split: ood.OODSplit,
    baseline_predictions: pd.DataFrame,
    rows: pd.DataFrame,
    y: np.ndarray,
) -> pd.DataFrame:
    local = baseline_predictions[
        (baseline_predictions["protocol"].astype(str) == split.protocol)
        & (
            baseline_predictions["heldout_group"].astype(str)
            == split.heldout_group
        )
        & baseline_predictions["model_id"].isin(INTERNAL_BASE_MODELS)
    ].copy()
    expected_test = set(int(value) for value in split.test_idx)
    for model_id in INTERNAL_BASE_MODELS:
        frame = local[local["model_id"] == model_id]
        if (
            len(frame) != len(split.test_idx)
            or set(frame["row_index"].astype(int)) != expected_test
            or frame["row_index"].duplicated().any()
        ):
            raise RuntimeError(
                f"Formal OOD baseline row mismatch: {model_id}"
            )
        ordered = frame.set_index("row_index").loc[split.test_idx]
        if not np.allclose(
            ordered["y_true"].to_numpy(dtype=np.float64),
            y[split.test_idx],
            atol=1e-7,
            rtol=1e-7,
        ):
            raise RuntimeError("Formal OOD outcomes changed")
        expected_qc = rows.iloc[split.test_idx]["qc_id"].astype(str).to_numpy()
        if not np.array_equal(
            ordered["qc_id"].astype(str).to_numpy(),
            expected_qc,
        ):
            raise RuntimeError("Formal OOD qc_id alignment changed")
        if bool(ordered["compound_seen_in_train"].astype(bool).any()):
            raise RuntimeError("Formal OOD compound-overlap flag changed")
        if bool(ordered["heldout_token_seen_in_train"].astype(bool).any()):
            raise RuntimeError("Formal OOD held-token flag changed")
    return local


def select_portable_ood_params(
    args: argparse.Namespace,
    rows: pd.DataFrame,
    y: np.ndarray,
    morgan: np.ndarray,
    descriptors: np.ndarray,
    scaffold_ids: np.ndarray,
    split: ood.OODSplit,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    inner_seed = split.split_seed + 101
    inner_splits = core.balanced_group_splits(
        split.train_idx,
        scaffold_ids,
        args.inner_folds,
        inner_seed,
    )
    local_position = {
        int(row_idx): position
        for position, row_idx in enumerate(split.train_idx.tolist())
    }
    candidates = {
        str(params["param_id"]): np.full(
            len(split.train_idx),
            np.nan,
            dtype=np.float32,
        )
        for params in TREE_GRID
    }
    audit_rows: list[dict[str, Any]] = []
    for inner_fold, (fit_idx, eval_idx) in enumerate(
        inner_splits,
        start=1,
    ):
        fit_scaffolds = set(scaffold_ids[fit_idx])
        eval_scaffolds = set(scaffold_ids[eval_idx])
        if fit_scaffolds.intersection(eval_scaffolds):
            raise RuntimeError("Portable OOD inner scaffold leakage")
        x_fit, x_eval, metadata = build_portable_features(
            rows,
            morgan,
            descriptors,
            fit_idx,
            eval_idx,
            split.protocol,
        )
        eval_positions = np.asarray(
            [local_position[int(value)] for value in eval_idx],
            dtype=np.int64,
        )
        audit_rows.append(
            {
                "inner_fold": inner_fold,
                "inner_seed": inner_seed,
                "n_fit_rows": int(len(fit_idx)),
                "n_eval_rows": int(len(eval_idx)),
                "n_fit_scaffolds": int(len(fit_scaffolds)),
                "n_eval_scaffolds": int(len(eval_scaffolds)),
                "n_scaffold_overlap": 0,
                "fit_indices_sha256": sequence_sha256(
                    int(value) for value in fit_idx
                ),
                "eval_indices_sha256": sequence_sha256(
                    int(value) for value in eval_idx
                ),
                **metadata,
            }
        )
        for params in TREE_GRID:
            pred = fit_extra_trees(
                x_fit,
                y[fit_idx],
                x_eval,
                params,
                seed=inner_seed + inner_fold * 101,
                n_estimators=args.n_estimators,
                n_jobs=args.n_jobs,
            )
            candidates[str(params["param_id"])][eval_positions] = pred
    selected, tuning = select_tree_params(
        candidates,
        y[split.train_idx],
    )
    return selected, tuning, audit_rows


def run_ood_sensitivity(
    args: argparse.Namespace,
    rows: pd.DataFrame,
    y: np.ndarray,
    morgan: np.ndarray,
    descriptors: np.ndarray,
    scaffold_ids: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:
    baseline_predictions = pd.read_csv(
        OOD_REPORT_DIR / "ood_predictions.csv"
    )
    splits, _ = ood.build_ood_splits(
        rows,
        scaffold_ids,
        list(ood.PROTOCOL_ORDER),
        args.seed,
        args.max_ood_groups,
    )
    records = {
        "predictions": read_records(
            output_dir / "ood_predictions.csv", args.resume
        ),
        "tuning": read_records(
            output_dir / "ood_portable_tuning_metrics.csv",
            args.resume,
        ),
        "selected": read_records(
            output_dir / "ood_portable_selected_hyperparameters.csv",
            args.resume,
        ),
        "inner_audit": read_records(
            output_dir / "ood_portable_inner_split_audit.csv",
            args.resume,
        ),
        "feature_audit": read_records(
            output_dir / "ood_portable_feature_audit.csv",
            args.resume,
        ),
    }
    started = time.perf_counter()
    for split_position, split in enumerate(splits, start=1):
        if args.resume and ood_checkpoint_complete(
            split,
            records,
            args.inner_folds,
        ):
            print(
                f"[OOD RESUME] {split.protocol} "
                f"{split.heldout_group} complete",
                flush=True,
            )
            continue
        if args.resume:
            for key in records:
                records[key] = discard_ood_records(
                    records[key],
                    split.protocol,
                    split.heldout_group,
                )
        baseline_local = validate_formal_ood_baseline(
            split,
            baseline_predictions,
            rows,
            y,
        )
        for model_id in INTERNAL_BASE_MODELS:
            frame = (
                baseline_local[
                    baseline_local["model_id"] == model_id
                ]
                .sort_values("row_index")
                .copy()
            )
            for record in frame.to_dict("records"):
                records["predictions"].append(
                    {
                        **record,
                        "baseline_provenance": (
                            "confirmatory_ood_cpu_v1 formal prediction"
                        ),
                    }
                )

        selected, tuning, inner_audit = select_portable_ood_params(
            args,
            rows,
            y,
            morgan,
            descriptors,
            scaffold_ids,
            split,
        )
        for record in tuning:
            records["tuning"].append(
                {
                    "protocol": split.protocol,
                    "heldout_group": split.heldout_group,
                    "model_id": "portable_full_extra_trees",
                    "selection_metric_weighting": "unweighted",
                    **record,
                }
            )
        records["selected"].append(
            {
                "protocol": split.protocol,
                "heldout_group": split.heldout_group,
                "model_id": "portable_full_extra_trees",
                **selected,
            }
        )
        for record in inner_audit:
            records["inner_audit"].append(
                {
                    "protocol": split.protocol,
                    "heldout_group": split.heldout_group,
                    **record,
                }
            )

        x_fit, x_eval, metadata = build_portable_features(
            rows,
            morgan,
            descriptors,
            split.train_idx,
            split.test_idx,
            split.protocol,
        )
        spec = PORTABLE_CONTEXT_SPEC[split.protocol]
        retained_names = list(spec["main"]) + list(
            spec["interactions"]
        )
        dropped_names = list(spec["dropped_main"]) + list(
            spec["dropped_interactions"]
        )
        if set(retained_names).intersection(dropped_names):
            raise RuntimeError("Portable feature audit is inconsistent")
        records["feature_audit"].append(
            {
                "protocol": split.protocol,
                "heldout_group": split.heldout_group,
                "retained_features": "|".join(retained_names),
                "dropped_features": "|".join(dropped_names),
                "n_retained_context_fields": int(len(retained_names)),
                "n_dropped_context_fields": int(len(dropped_names)),
                "held_axis_feature_present": False,
                **metadata,
            }
        )
        model_seed = split.split_seed + 1_000 + 6
        pred = fit_extra_trees(
            x_fit,
            y[split.train_idx],
            x_eval,
            selected,
            seed=model_seed,
            n_estimators=args.n_estimators,
            n_jobs=args.n_jobs,
        )
        identity = (
            baseline_local[
                baseline_local["model_id"]
                == "chemistry_extra_trees"
            ]
            .set_index("row_index")
            .loc[split.test_idx]
            .reset_index()
        )
        for local_idx, source in identity.iterrows():
            records["predictions"].append(
                {
                    **source.to_dict(),
                    "model_id": "portable_full_extra_trees",
                    "param_id": str(selected["param_id"]),
                    "y_pred": float(pred[local_idx]),
                    "baseline_provenance": "new post-hoc refit",
                }
            )
        write_ood_progress(output_dir, records)
        if not ood_checkpoint_complete(
            split,
            records,
            args.inner_folds,
        ):
            raise RuntimeError("OOD split checkpoint is incomplete")
        elapsed = time.perf_counter() - started
        print(
            f"[OOD] {split_position}/{len(splits)} "
            f"{split.protocol} {split.heldout_group} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
    return pd.DataFrame(records["predictions"])


def average_internal_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    identity_columns = [
        "protocol",
        "model_id",
        "representation",
        "weight_mode",
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

    def joined_unique(values: pd.Series) -> str:
        return "|".join(sorted(set(values.astype(str))))

    return (
        predictions.groupby(
            identity_columns,
            sort=False,
            dropna=False,
        )
        .agg(
            n_oof_predictions=("y_pred", "size"),
            n_repeats=("repeat", "nunique"),
            selected_param_ids=("param_id", joined_unique),
            y_pred=("y_pred", "mean"),
            y_pred_sd_across_repeats=("y_pred", "std"),
        )
        .reset_index()
    )


def internal_repeat_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows_out: list[dict[str, Any]] = []
    for (model_id, repeat), frame in predictions.groupby(
        ["model_id", "repeat"],
        sort=False,
    ):
        rows_out.append(
            {
                "protocol": "scaffold",
                "model_id": model_id,
                "representation": str(
                    frame["representation"].iloc[0]
                ),
                "weight_mode": str(frame["weight_mode"].iloc[0]),
                "repeat": int(repeat),
                "n_rows": int(len(frame)),
                "n_unique_smiles": int(
                    frame["canonical_smiles"].nunique()
                ),
                "n_scaffolds": int(frame["scaffold_id"].nunique()),
                **core.regression_metrics(
                    frame["y_true"].to_numpy(dtype=np.float64),
                    frame["y_pred"].to_numpy(dtype=np.float64),
                ),
                **core.target_macro_metrics(frame),
                "within_target_spearman": core.within_target_spearman(
                    frame
                ),
            }
        )
    return pd.DataFrame(rows_out)


def internal_summary_metrics(
    averaged: pd.DataFrame,
) -> pd.DataFrame:
    rows_out: list[dict[str, Any]] = []
    for model_id, frame in averaged.groupby("model_id", sort=False):
        rows_out.append(
            {
                "protocol": "scaffold",
                "model_id": model_id,
                "representation": str(
                    frame["representation"].iloc[0]
                ),
                "weight_mode": str(frame["weight_mode"].iloc[0]),
                "n_rows": int(len(frame)),
                "n_unique_smiles": int(
                    frame["canonical_smiles"].nunique()
                ),
                "n_scaffolds": int(frame["scaffold_id"].nunique()),
                "n_oof_predictions_min": int(
                    frame["n_oof_predictions"].min()
                ),
                "n_oof_predictions_max": int(
                    frame["n_oof_predictions"].max()
                ),
                **core.regression_metrics(
                    frame["y_true"].to_numpy(dtype=np.float64),
                    frame["y_pred"].to_numpy(dtype=np.float64),
                ),
                **core.target_macro_metrics(frame),
                "within_target_spearman": core.within_target_spearman(
                    frame
                ),
            }
        )
    return pd.DataFrame(rows_out)


def internal_domain_metrics(
    averaged: pd.DataFrame,
) -> pd.DataFrame:
    work = averaged.copy()
    work["domain_id"] = (
        work["source_database"].astype(str)
        + "||"
        + work["target_protein"].astype(str)
    )
    rows_out: list[dict[str, Any]] = []
    for (model_id, domain_id), frame in work.groupby(
        ["model_id", "domain_id"],
        sort=False,
    ):
        rows_out.append(
            {
                "protocol": "scaffold",
                "model_id": model_id,
                "representation": str(
                    frame["representation"].iloc[0]
                ),
                "weight_mode": str(frame["weight_mode"].iloc[0]),
                "domain_id": domain_id,
                "source_database": str(
                    frame["source_database"].iloc[0]
                ),
                "target_protein": str(
                    frame["target_protein"].iloc[0]
                ),
                "n_rows": int(len(frame)),
                "n_unique_smiles": int(
                    frame["canonical_smiles"].nunique()
                ),
                "n_scaffolds": int(frame["scaffold_id"].nunique()),
                **core.regression_metrics(
                    frame["y_true"].to_numpy(dtype=np.float64),
                    frame["y_pred"].to_numpy(dtype=np.float64),
                ),
            }
        )
    return pd.DataFrame(rows_out)


def internal_canonical_metrics(
    averaged: pd.DataFrame,
) -> pd.DataFrame:
    rows_out: list[dict[str, Any]] = []
    for model_id, frame in averaged.groupby("model_id", sort=False):
        compounds = (
            frame.groupby("canonical_smiles", sort=False)
            .agg(
                y_true=("y_true", "median"),
                y_pred=("y_pred", "median"),
                n_source_rows=("row_index", "size"),
                scaffold_id=("scaffold_id", "first"),
            )
            .reset_index()
        )
        rows_out.append(
            {
                "protocol": "scaffold",
                "model_id": model_id,
                "representation": str(
                    frame["representation"].iloc[0]
                ),
                "weight_mode": str(frame["weight_mode"].iloc[0]),
                "aggregation": (
                    "median y_true and median repeat-averaged y_pred "
                    "within canonical_smiles"
                ),
                "n_unique_smiles": int(len(compounds)),
                "n_scaffolds": int(
                    compounds["scaffold_id"].nunique()
                ),
                "n_multirow_smiles": int(
                    np.sum(compounds["n_source_rows"] > 1)
                ),
                **core.regression_metrics(
                    compounds["y_true"].to_numpy(dtype=np.float64),
                    compounds["y_pred"].to_numpy(dtype=np.float64),
                ),
            }
        )
    return pd.DataFrame(rows_out)


def run_internal_bootstrap(
    averaged: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    old_order = core.MODEL_ORDER
    old_reference = core.REFERENCE_MODEL
    old_contrasts = core.PRESPECIFIED_CONTRASTS
    try:
        core.MODEL_ORDER = list(INTERNAL_MODEL_ORDER)
        core.REFERENCE_MODEL = (
            "full_context_extra_trees__uniform"
        )
        core.PRESPECIFIED_CONTRASTS = dict(INTERNAL_CONTRASTS)
        return core.cluster_bootstrap_summary(
            averaged,
            n_bootstrap,
            seed + 31_000_000,
            cluster_col="scaffold_id",
        )
    finally:
        core.MODEL_ORDER = old_order
        core.REFERENCE_MODEL = old_reference
        core.PRESPECIFIED_CONTRASTS = old_contrasts


def ood_point_metrics(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    domain_rows: list[dict[str, Any]] = []
    for (protocol, heldout_group, model_id), frame in predictions.groupby(
        ["protocol", "heldout_group", "model_id"],
        sort=False,
    ):
        domain_rows.append(
            {
                "protocol": protocol,
                "heldout_group": heldout_group,
                "model_id": model_id,
                "n_rows": int(len(frame)),
                "n_unique_smiles": int(
                    frame["canonical_smiles"].nunique()
                ),
                "n_scaffolds": int(frame["scaffold_id"].nunique()),
                **core.regression_metrics(
                    frame["y_true"].to_numpy(dtype=np.float64),
                    frame["y_pred"].to_numpy(dtype=np.float64),
                ),
            }
        )
    return (
        pd.DataFrame(domain_rows),
        ood.summarize_aggregate_metrics(predictions),
    )


def run_ood_bootstrap(
    predictions: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    old_order = ood.MODEL_ORDER
    old_contrasts = ood.PRESPECIFIED_CONTRASTS
    try:
        ood.MODEL_ORDER = tuple(OOD_MODEL_ORDER)
        ood.PRESPECIFIED_CONTRASTS = dict(OOD_CONTRASTS)
        return ood.paired_scaffold_bootstrap(
            predictions,
            n_bootstrap,
            seed + 41_000_000,
        )
    finally:
        ood.MODEL_ORDER = old_order
        ood.PRESPECIFIED_CONTRASTS = old_contrasts


def write_analysis_tables(
    args: argparse.Namespace,
    output_dir: Path,
    internal_predictions: pd.DataFrame,
    ood_predictions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    repeat_metrics = internal_repeat_metrics(internal_predictions)
    averaged = average_internal_predictions(internal_predictions)
    summary = internal_summary_metrics(averaged)
    domain = internal_domain_metrics(averaged)
    canonical = internal_canonical_metrics(averaged)
    internal_bootstrap = run_internal_bootstrap(
        averaged,
        args.bootstrap_replicates,
        args.seed,
    )
    ood_domain, ood_aggregate = ood_point_metrics(ood_predictions)
    ood_bootstrap, ood_domain_bootstrap = run_ood_bootstrap(
        ood_predictions,
        args.bootstrap_replicates,
        args.seed,
    )
    tables = {
        "internal_repeat_metrics.csv": repeat_metrics,
        "internal_repeat_averaged_predictions.csv": averaged,
        "internal_summary_metrics.csv": summary,
        "internal_domain_metrics.csv": domain,
        "internal_canonical_compound_metrics.csv": canonical,
        "internal_paired_scaffold_bootstrap.csv": internal_bootstrap,
        "ood_domain_metrics.csv": ood_domain,
        "ood_aggregate_metrics.csv": ood_aggregate,
        "ood_paired_scaffold_bootstrap.csv": ood_bootstrap,
        "ood_domain_scaffold_bootstrap.csv": ood_domain_bootstrap,
    }
    for filename, frame in tables.items():
        core.atomic_to_csv(frame, output_dir / filename)
    return tables


def qa_check(
    checks: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    detail: str,
) -> None:
    checks.append(
        {
            "check_id": check_id,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        }
    )


def build_qa_summary(
    args: argparse.Namespace,
    rows: pd.DataFrame,
    y: np.ndarray,
    internal_predictions: pd.DataFrame,
    ood_predictions: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    qa_check(
        checks,
        "frozen_data_sha256",
        sha256_file(DATA_FILE) == FROZEN_DATA_SHA256,
        sha256_file(DATA_FILE),
    )
    qa_check(
        checks,
        "master_protocol_sha256",
        sha256_file(MASTER_PROTOCOL) == MASTER_PROTOCOL_SHA256,
        sha256_file(MASTER_PROTOCOL),
    )
    qa_check(
        checks,
        "float32_correction_document_sha256",
        sha256_file(CORRECTION_DOCUMENT)
        == CORRECTION_DOCUMENT_SHA256,
        sha256_file(CORRECTION_DOCUMENT),
    )
    qa_check(
        checks,
        "training_response_dtype",
        y.dtype == TRAINING_RESPONSE_DTYPE,
        f"observed={y.dtype.name}, expected={TRAINING_RESPONSE_DTYPE.name}",
    )
    qa_check(
        checks,
        "core_row_identity",
        len(rows) == 1560
        and rows["qc_id"].nunique() == 1560
        and rows["canonical_smiles"].nunique() == 1137,
        (
            f"rows={len(rows)}, qc_ids={rows['qc_id'].nunique()}, "
            f"smiles={rows['canonical_smiles'].nunique()}"
        ),
    )
    qa_check(
        checks,
        "internal_prediction_finiteness",
        not internal_predictions.empty
        and np.all(
            np.isfinite(
                internal_predictions["y_pred"].to_numpy(dtype=np.float64)
            )
        ),
        f"rows={len(internal_predictions)}",
    )
    qa_check(
        checks,
        "internal_prediction_uniqueness",
        not internal_predictions.duplicated(
            ["repeat", "outer_fold", "model_id", "row_index"]
        ).any(),
        "keys=repeat,outer_fold,model_id,row_index",
    )
    expected_internal_splits = (
        args.max_internal_splits
        if args.max_internal_splits is not None
        else 25
    )
    audit = pd.read_csv(output_dir / "internal_split_audit.csv")
    qa_check(
        checks,
        "internal_split_count_and_scaffold_disjointness",
        len(audit) == expected_internal_splits
        and bool((audit["n_scaffold_overlap"] == 0).all()),
        (
            f"splits={len(audit)}/{expected_internal_splits}, "
            f"max_overlap={audit['n_scaffold_overlap'].max()}"
        ),
    )
    weight_diagnostics = pd.read_csv(
        output_dir / "internal_weight_diagnostics.csv"
    )
    expected_weight_rows = (
        expected_internal_splits
        * len(WEIGHTED_MODES)
        * (args.inner_folds + 1)
    )
    valid_weight_values = bool(
        np.all(np.isfinite(weight_diagnostics["weight_min"]))
        and np.all(weight_diagnostics["weight_min"] > 0)
        and np.allclose(
            weight_diagnostics["weight_mean"],
            1.0,
            atol=1e-12,
            rtol=1e-12,
        )
        and np.all(
            weight_diagnostics["effective_sample_size"] > 0
        )
        and np.all(
            (weight_diagnostics["clipping_fraction"] >= 0)
            & (weight_diagnostics["clipping_fraction"] <= 1)
        )
        and np.all(
            weight_diagnostics.loc[
                weight_diagnostics["weight_mode"]
                == "domain_balanced_clipped",
                "post_clip_prenorm_max",
            ]
            <= DOMAIN_WEIGHT_CAP + 1e-12
        )
    )
    qa_check(
        checks,
        "training_weight_diagnostics",
        len(weight_diagnostics) == expected_weight_rows
        and valid_weight_values,
        (
            f"rows={len(weight_diagnostics)}/{expected_weight_rows}, "
            f"max_clip_fraction="
            f"{weight_diagnostics['clipping_fraction'].max():.6g}, "
            f"min_ess_fraction="
            f"{weight_diagnostics['effective_sample_fraction'].min():.6g}"
        ),
    )
    compound_diag = weight_diagnostics[
        weight_diagnostics["weight_mode"] == "compound_equal"
    ]
    qa_check(
        checks,
        "compound_equal_total_weight",
        not compound_diag.empty
        and bool(
            (
                compound_diag["compound_total_relative_range"].fillna(
                    np.inf
                )
                <= 1e-12
            ).all()
        ),
        (
            "max_relative_range="
            f"{compound_diag['compound_total_relative_range'].max():.3g}"
        ),
    )
    tuning = pd.read_csv(output_dir / "internal_tuning_metrics.csv")
    expected_tuning = (
        expected_internal_splits
        * len(WEIGHTED_MODES)
        * len(INTERNAL_BASE_MODELS)
        * len(TREE_GRID)
    )
    qa_check(
        checks,
        "weighted_inner_grid_completeness",
        len(tuning) == expected_tuning
        and set(tuning["param_id"].astype(str))
        == {str(params["param_id"]) for params in TREE_GRID},
        f"rows={len(tuning)}/{expected_tuning}",
    )
    averaged = tables["internal_repeat_averaged_predictions.csv"]
    required_models = set(INTERNAL_MODEL_ORDER)
    qa_check(
        checks,
        "internal_model_completeness",
        set(averaged["model_id"].astype(str)) == required_models,
        (
            f"observed={sorted(set(averaged['model_id'].astype(str)))}, "
            f"expected={sorted(required_models)}"
        ),
    )
    if formal_run(args):
        full_average_complete = bool(
            len(averaged) == len(rows) * len(INTERNAL_MODEL_ORDER)
            and (averaged["n_oof_predictions"] == 5).all()
        )
    else:
        full_average_complete = bool(
            (averaged["n_oof_predictions"] >= 1).all()
        )
    qa_check(
        checks,
        "internal_repeat_average_completeness",
        full_average_complete,
        (
            f"rows={len(averaged)}, "
            f"prediction_count_range="
            f"{averaged['n_oof_predictions'].min()}-"
            f"{averaged['n_oof_predictions'].max()}"
        ),
    )
    qa_check(
        checks,
        "ood_prediction_finiteness",
        not ood_predictions.empty
        and np.all(
            np.isfinite(
                ood_predictions["y_pred"].to_numpy(dtype=np.float64)
            )
        ),
        f"rows={len(ood_predictions)}",
    )
    qa_check(
        checks,
        "ood_prediction_uniqueness",
        not ood_predictions.duplicated(
            ["protocol", "heldout_group", "model_id", "row_index"]
        ).any(),
        "keys=protocol,heldout_group,model_id,row_index",
    )
    expected_ood_groups = (
        2 * args.max_ood_groups
        if args.max_ood_groups is not None
        else 12
    )
    selected_ood = pd.read_csv(
        output_dir / "ood_portable_selected_hyperparameters.csv"
    )
    qa_check(
        checks,
        "portable_ood_group_completeness",
        len(selected_ood) == expected_ood_groups,
        f"groups={len(selected_ood)}/{expected_ood_groups}",
    )
    portable_audit = pd.read_csv(
        output_dir / "ood_portable_feature_audit.csv"
    )
    portable_feature_pass = (
        len(portable_audit) == expected_ood_groups
        and not portable_audit["held_axis_feature_present"].astype(bool).any()
    )
    for record in portable_audit.to_dict("records"):
        protocol = str(record["protocol"])
        dropped = set(str(record["dropped_features"]).split("|"))
        expected_dropped = set(
            PORTABLE_CONTEXT_SPEC[protocol]["dropped_main"]
        ) | set(
            PORTABLE_CONTEXT_SPEC[protocol]["dropped_interactions"]
        )
        portable_feature_pass &= dropped == expected_dropped
    qa_check(
        checks,
        "portable_held_axis_exclusion",
        bool(portable_feature_pass),
        f"audited_groups={len(portable_audit)}",
    )
    ood_tuning = pd.read_csv(
        output_dir / "ood_portable_tuning_metrics.csv"
    )
    qa_check(
        checks,
        "portable_inner_grid_completeness",
        len(ood_tuning) == expected_ood_groups * len(TREE_GRID),
        (
            f"rows={len(ood_tuning)}/"
            f"{expected_ood_groups * len(TREE_GRID)}"
        ),
    )
    qa_check(
        checks,
        "ood_no_compound_or_held_token_overlap",
        not ood_predictions["compound_seen_in_train"].astype(bool).any()
        and not ood_predictions[
            "heldout_token_seen_in_train"
        ].astype(bool).any(),
        "all stored flags are false",
    )
    internal_bootstrap = tables[
        "internal_paired_scaffold_bootstrap.csv"
    ]
    ood_bootstrap = tables["ood_paired_scaffold_bootstrap.csv"]
    if args.bootstrap_replicates > 0:
        bootstrap_pass = bool(
            not internal_bootstrap.empty
            and not ood_bootstrap.empty
            and (
                internal_bootstrap["n_bootstrap"]
                == args.bootstrap_replicates
            ).all()
            and (
                ood_bootstrap["n_bootstrap"]
                == args.bootstrap_replicates
            ).all()
        )
    else:
        bootstrap_pass = bool(
            internal_bootstrap.empty and ood_bootstrap.empty
        )
    qa_check(
        checks,
        "paired_scaffold_bootstrap",
        bootstrap_pass,
        f"replicates={args.bootstrap_replicates}",
    )
    allowed_input_paths = {
        str(DATA_FILE.resolve()),
        str(
            (CORE_REPORT_DIR / "scaffold_cross_fitted_predictions.csv")
            .resolve()
        ),
        str(
            (CORE_REPORT_DIR / "scaffold_selected_hyperparameters.csv")
            .resolve()
        ),
        str((OOD_REPORT_DIR / "ood_predictions.csv").resolve()),
        str(MASTER_PROTOCOL.resolve()),
        str(CORRECTION_DOCUMENT.resolve()),
        str(PROTOCOL_DOCUMENT.resolve()),
        str(Path(core.__file__).resolve()),
        str(Path(ood.__file__).resolve()),
        str(Path(__file__).resolve()),
    }
    configuration = json.loads(
        (output_dir / "configuration.json").read_text(encoding="utf-8")
    )
    configured_inputs = {
        str(configuration["data_file"]),
        str(configuration["script_file"]),
        str(configuration["master_protocol"]),
        str(configuration["correction_document"]),
        str(configuration["child_protocol"]),
        *(
            str(value["path"])
            for value in configuration[
                "parent_implementations"
            ].values()
        ),
        *(
            str(value["path"])
            for value in configuration["baseline_files"].values()
        ),
    }
    qa_check(
        checks,
        "publication_input_allowlist",
        configured_inputs == allowed_input_paths,
        (
            "only frozen core, formal baseline artifacts, frozen parent "
            "implementations, and protocols"
        ),
    )
    failed = [row for row in checks if row["status"] != "PASS"]
    return {
        "status": "PASS" if not failed else "FAIL",
        "analysis_mode": (
            "formal_post_hoc_sensitivity"
            if formal_run(args)
            else "smoke_test_only"
        ),
        "n_checks": len(checks),
        "n_pass": len(checks) - len(failed),
        "n_fail": len(failed),
        "checks": checks,
    }


def write_qa_files(
    qa: dict[str, Any],
    output_dir: Path,
) -> None:
    core.write_json(output_dir / "qa_summary.json", qa)
    lines = [
        "# Context/weight sensitivity QA",
        "",
        f"Overall status: **{qa['status']}**",
        "",
        f"Analysis mode: `{qa['analysis_mode']}`",
        "",
        "| Check | Status | Detail |",
        "|---|---:|---|",
    ]
    for row in qa["checks"]:
        detail = str(row["detail"]).replace("|", "/")
        lines.append(
            f"| `{row['check_id']}` | {row['status']} | {detail} |"
        )
    (output_dir / "qa_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def bootstrap_contrast_row(
    frame: pd.DataFrame,
    contrast_id: str,
) -> pd.Series:
    local = frame[
        (frame["record_type"] == "contrast")
        & (frame["contrast_id"].astype(str) == contrast_id)
    ]
    if len(local) != 1:
        raise RuntimeError(
            f"Expected one bootstrap contrast row: {contrast_id}"
        )
    return local.iloc[0]


def fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{number:.{digits}f}" if math.isfinite(number) else "NA"


def write_results_brief(
    args: argparse.Namespace,
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    internal_boot = tables[
        "internal_paired_scaffold_bootstrap.csv"
    ]
    ood_boot = tables["ood_paired_scaffold_bootstrap.csv"]
    lines = [
        "# 可迁移上下文与训练权重敏感性结果",
        "",
        (
            "分析身份：正式 post-hoc sensitivity。"
            if formal_run(args)
            else "分析身份：烟雾测试，仅用于验证代码，不构成论文证据。"
        ),
        "",
        "## 内部 scaffold-disjoint",
        "",
    ]
    for contrast_id in (
        "uniform_full_vs_chemistry",
        "compound_equal_full_vs_chemistry",
        "domain_balanced_full_vs_chemistry",
        "chemistry_compound_equal_vs_uniform",
        "full_compound_equal_vs_uniform",
        "chemistry_domain_balanced_vs_uniform",
        "full_domain_balanced_vs_uniform",
    ):
        row = bootstrap_contrast_row(internal_boot, contrast_id)
        lines.append(
            "- "
            f"`{contrast_id}`: ΔSpearman "
            f"{fmt(row['delta_spearman_observed'])} "
            f"[{fmt(row['delta_spearman_ci_low'])}, "
            f"{fmt(row['delta_spearman_ci_high'])}]; "
            f"ΔRMSE {fmt(row['delta_rmse_observed'])} "
            f"[{fmt(row['delta_rmse_ci_low'])}, "
            f"{fmt(row['delta_rmse_ci_high'])}]."
        )
    lines.extend(
        [
            "",
            "## OOD 可迁移上下文",
            "",
        ]
    )
    for protocol in ood.PROTOCOL_ORDER:
        local = ood_boot[ood_boot["protocol"] == protocol]
        for contrast_id in (
            "portable_vs_chemistry",
            "full_vs_chemistry",
            "portable_vs_full",
        ):
            row = bootstrap_contrast_row(local, contrast_id)
            lines.append(
                "- "
                f"`{protocol} / {contrast_id}`: "
                f"equal-domain ΔSpearman "
                f"{fmt(row['delta_domain_macro_spearman_observed'])} "
                f"[{fmt(row['delta_domain_macro_spearman_ci_low'])}, "
                f"{fmt(row['delta_domain_macro_spearman_ci_high'])}]; "
                f"equal-domain ΔRMSE "
                f"{fmt(row['delta_domain_macro_rmse_observed'])} "
                f"[{fmt(row['delta_domain_macro_rmse_ci_low'])}, "
                f"{fmt(row['delta_domain_macro_rmse_ci_high'])}]."
            )
    weight_diagnostics = pd.read_csv(
        output_dir / "internal_weight_diagnostics.csv"
    )
    domain_diag = weight_diagnostics[
        weight_diagnostics["weight_mode"]
        == "domain_balanced_clipped"
    ]
    lines.extend(
        [
            "",
            "## 诊断边界",
            "",
            (
                "- source-target domain balancing 的最大 clipping fraction "
                f"为 {fmt(domain_diag['clipping_fraction'].max(), 4)}；"
                "每个 fit-fold 的 ESS 已逐行保存。"
            ),
            (
                "- 该结果只说明现有回顾性数据中，结论对指定的特征删除和"
                "训练权重扰动是否稳健；不构成前瞻外部验证或因果解释。"
            ),
        ]
    )
    (output_dir / "results_brief_zh.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def build_artifact_inventory(output_dir: Path) -> pd.DataFrame:
    rows_out: list[dict[str, Any]] = []
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name == "artifact_sha256.csv":
            continue
        suffix = path.suffix.lower()
        record: dict[str, Any] = {
            "relative_path": path.name,
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
            "kind": suffix.lstrip(".") or "file",
            "n_rows": np.nan,
            "n_columns": np.nan,
            "verification": "PASS",
        }
        if suffix == ".csv":
            frame = pd.read_csv(path)
            record["n_rows"] = int(len(frame))
            record["n_columns"] = int(len(frame.columns))
        rows_out.append(record)
    return pd.DataFrame(rows_out)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if sha256_file(DATA_FILE) != FROZEN_DATA_SHA256:
        raise RuntimeError("Frozen core data SHA256 mismatch")
    if sha256_file(MASTER_PROTOCOL) != MASTER_PROTOCOL_SHA256:
        raise RuntimeError("Frozen master protocol SHA256 mismatch")
    if (
        sha256_file(CORRECTION_DOCUMENT)
        != CORRECTION_DOCUMENT_SHA256
    ):
        raise RuntimeError("Frozen float32 correction SHA256 mismatch")
    baseline_files = {
        "core_scaffold_predictions": verify_baseline_artifact(
            CORE_REPORT_DIR,
            "scaffold_cross_fitted_predictions.csv",
        ),
        "core_scaffold_selected": verify_baseline_artifact(
            CORE_REPORT_DIR,
            "scaffold_selected_hyperparameters.csv",
        ),
        "ood_predictions": verify_baseline_artifact(
            OOD_REPORT_DIR,
            "ood_predictions.csv",
        ),
    }
    configuration = configuration_payload(args, baseline_files)
    prepare_output_directory(args.output_dir, args, configuration)

    rows, sanitization_audit = core.load_rows(DATA_FILE)
    if len(rows) != 1560:
        raise RuntimeError(f"Expected 1,560 rows, observed {len(rows)}")
    core.atomic_to_csv(
        sanitization_audit,
        args.output_dir / "context_sanitization_audit.csv",
    )
    y = rows["pDC50"].to_numpy(dtype=np.float32)
    if y.dtype != TRAINING_RESPONSE_DTYPE:
        raise RuntimeError("Training response is not float32")
    scaffold_ids = core.build_scaffold_ids(rows["canonical_smiles"])
    if len(np.unique(scaffold_ids)) != 667:
        raise RuntimeError("Frozen Bemis--Murcko scaffold count changed")
    print("[FEATURES] Morgan fingerprints", flush=True)
    morgan = core.build_morgan_matrix(rows["canonical_smiles"])
    print("[FEATURES] RDKit descriptors", flush=True)
    descriptors, descriptor_names, descriptor_failures = (
        core.build_descriptor_matrix(rows["canonical_smiles"])
    )

    internal_predictions = run_internal_sensitivity(
        args,
        rows,
        y,
        morgan,
        descriptors,
        scaffold_ids,
        args.output_dir,
    )
    ood_predictions = run_ood_sensitivity(
        args,
        rows,
        y,
        morgan,
        descriptors,
        scaffold_ids,
        args.output_dir,
    )
    tables = write_analysis_tables(
        args,
        args.output_dir,
        internal_predictions,
        ood_predictions,
    )
    qa = build_qa_summary(
        args,
        rows,
        y,
        internal_predictions,
        ood_predictions,
        tables,
        args.output_dir,
    )
    write_qa_files(qa, args.output_dir)
    if qa["status"] != "PASS":
        raise RuntimeError("QA failed; see qa_summary.json")
    write_results_brief(args, tables, args.output_dir)

    elapsed = time.perf_counter() - started
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "complete",
        "analysis_mode": configuration["analysis_mode"],
        "scientific_configuration_sha256": canonical_sha256(
            configuration
        ),
        "configuration": configuration,
        "n_rows": int(len(rows)),
        "n_unique_smiles": int(
            rows["canonical_smiles"].nunique()
        ),
        "n_scaffolds": int(len(np.unique(scaffold_ids))),
        "endpoint": "pDC50 = 9 - log10(DC50_nM)",
        "internal_models": list(INTERNAL_MODEL_ORDER),
        "ood_models": list(OOD_MODEL_ORDER),
        "feature_manifest": core.feature_manifest(
            descriptor_names,
            descriptor_failures,
        ),
        "independent_resampling_unit": "global Bemis--Murcko scaffold",
        "gpu_required": False,
        "training_response_dtype": y.dtype.name,
        "metric_calculation": (
            "unchanged from v1; explicit float64 summary casts retained"
        ),
        "correction_document": {
            "path": str(CORRECTION_DOCUMENT.resolve()),
            "sha256": sha256_file(CORRECTION_DOCUMENT),
        },
        "supersedes": {
            "protocol_version": (
                "post_hoc_context_weight_sensitivity_v1.0"
            ),
            "runner_sha256": SUPERSEDED_V1_RUNNER_SHA256,
            "publication_eligible": False,
        },
        "publication_boundary": (
            "retrospective 1,560-row frozen core only; no restricted "
            "collaborator panel or derivative"
        ),
        "claim_boundary": (
            "post-hoc robustness/sensitivity; not preregistered, "
            "prospective, external, causal, or deployment evidence"
        ),
        "qa_status": qa["status"],
        "elapsed_seconds": elapsed,
        "artifact_inventory": (
            "artifact_sha256.csv; inventory is the sole checksum "
            "self-reference exception"
        ),
    }
    core.write_json(args.output_dir / "run_manifest.json", manifest)
    inventory = build_artifact_inventory(args.output_dir)
    core.atomic_to_csv(
        inventory,
        args.output_dir / "artifact_sha256.csv",
    )
    print(
        f"[COMPLETE] status={qa['status']} "
        f"mode={configuration['analysis_mode']} "
        f"elapsed={elapsed:.1f}s output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
