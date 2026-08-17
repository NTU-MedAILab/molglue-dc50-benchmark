#!/usr/bin/env python3
"""Run the frozen post-hoc internal scaffold learning-curve sensitivity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import numpy as np
import pandas as pd

import run_confirmatory_cpu_v1 as core


SCRIPT_DIR = Path(__file__).resolve().parent
ROUTE_DIR = SCRIPT_DIR.parent
PROTOCOL_PATH = (
    ROUTE_DIR / "docs" / "post_hoc_internal_learning_curve_protocol_v1.md"
)
DEFAULT_OUTPUT_DIR = (
    ROUTE_DIR / "reports" / "post_hoc_internal_learning_curve_v1"
)
ANALYSIS_IDENTITY = "post_hoc_internal_learning_curve_v1"
FRACTIONS = (0.25, 0.50, 0.75, 1.00)
MODEL_IDS = (
    "chemistry_extra_trees",
    "full_context_extra_trees",
)
MODEL_PARAMS = {
    "param_id": "leaf_1_maxfeat_sqrt_fixed",
    "min_samples_leaf": 1,
    "max_features": "sqrt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=core.DATA_FILE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=core.DEFAULT_SEED)
    parser.add_argument(
        "--max-outer-splits",
        type=int,
        default=None,
        help="Smoke-test helper; truncates outer splits and disables bootstrap.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.n_estimators < 10:
        parser.error("--n-estimators must be >= 10")
    if args.n_jobs < 1:
        parser.error("--n-jobs must be >= 1")
    if args.bootstrap_replicates < 0:
        parser.error("--bootstrap-replicates must be >= 0")
    if args.max_outer_splits is not None and args.max_outer_splits < 1:
        parser.error("--max-outer-splits must be >= 1")
    return args


def sha256_text(values: list[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {path}; use --overwrite"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ordered_training_scaffolds(
    scaffold_ids: np.ndarray,
    train_idx: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    unique = np.unique(scaffold_ids[train_idx])
    rng = np.random.default_rng(seed)
    return unique[rng.permutation(len(unique))]


def select_training_indices(
    train_idx: np.ndarray,
    scaffold_ids: np.ndarray,
    ordered_scaffolds: np.ndarray,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_selected = min(
        len(ordered_scaffolds),
        max(2, int(math.ceil(float(fraction) * len(ordered_scaffolds)))),
    )
    selected_scaffolds = ordered_scaffolds[:n_selected]
    mask = np.isin(scaffold_ids[train_idx], selected_scaffolds)
    selected_idx = train_idx[mask]
    if len(selected_idx) == 0:
        raise RuntimeError("Training subset is empty")
    return selected_idx, selected_scaffolds


def average_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["training_fraction", "model_id", "row_index"]
    averaged = (
        predictions.groupby(keys, as_index=False, sort=True)
        .agg(
            qc_id=("qc_id", "first"),
            y_true=("y_true", "first"),
            y_pred=("y_pred", "mean"),
            scaffold_id=("scaffold_id", "first"),
            n_repeat_predictions=("repeat", "nunique"),
        )
        .sort_values(keys)
        .reset_index(drop=True)
    )
    return averaged


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_learning_curve(
    averaged: pd.DataFrame,
    training_sizes: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fractions = list(FRACTIONS)
    models = list(MODEL_IDS)
    reference = (
        averaged[
            (averaged["training_fraction"] == fractions[0])
            & (averaged["model_id"] == models[0])
        ]
        .sort_values("row_index")
        .reset_index(drop=True)
    )
    if reference.empty:
        raise ValueError("No reference predictions for bootstrap")
    row_indices = reference["row_index"].to_numpy(dtype=np.int64)
    y_true = reference["y_true"].to_numpy(dtype=float)
    scaffolds = reference["scaffold_id"].astype(str).to_numpy()
    unique_scaffolds = np.unique(scaffolds)
    cluster_positions = [
        np.where(scaffolds == scaffold)[0] for scaffold in unique_scaffolds
    ]

    prediction_arrays: dict[tuple[float, str], np.ndarray] = {}
    for fraction in fractions:
        for model in models:
            frame = (
                averaged[
                    (averaged["training_fraction"] == fraction)
                    & (averaged["model_id"] == model)
                ]
                .sort_values("row_index")
                .reset_index(drop=True)
            )
            if not np.array_equal(
                frame["row_index"].to_numpy(dtype=np.int64),
                row_indices,
            ):
                raise ValueError(
                    f"Prediction row identity differs: fraction={fraction}, "
                    f"model={model}"
                )
            if not np.allclose(
                frame["y_true"].to_numpy(dtype=float),
                y_true,
                rtol=0.0,
                atol=1e-7,
            ):
                raise ValueError("Prediction truth values differ")
            prediction_arrays[(fraction, model)] = frame["y_pred"].to_numpy(
                dtype=float
            )

    point = np.full((len(fractions), len(models), 2), np.nan, dtype=float)
    for f_idx, fraction in enumerate(fractions):
        for m_idx, model in enumerate(models):
            pred = prediction_arrays[(fraction, model)]
            point[f_idx, m_idx, 0] = core.safe_spearman(y_true, pred)
            point[f_idx, m_idx, 1] = float(
                np.sqrt(np.mean(np.square(y_true - pred)))
            )

    draws = np.full(
        (replicates, len(fractions), len(models), 2),
        np.nan,
        dtype=np.float32,
    )
    if replicates:
        rng = np.random.default_rng(seed + 800_000)
        for draw in range(replicates):
            sampled = rng.integers(
                0,
                len(unique_scaffolds),
                size=len(unique_scaffolds),
            )
            positions = np.concatenate(
                [cluster_positions[index] for index in sampled]
            )
            truth = y_true[positions]
            for f_idx, fraction in enumerate(fractions):
                for m_idx, model in enumerate(models):
                    pred = prediction_arrays[(fraction, model)][positions]
                    draws[draw, f_idx, m_idx, 0] = core.safe_spearman(
                        truth,
                        pred,
                    )
                    draws[draw, f_idx, m_idx, 1] = float(
                        np.sqrt(np.mean(np.square(truth - pred)))
                    )

    size_summary = (
        training_sizes.groupby("training_fraction", sort=True)
        .agg(
            train_rows_median=("n_train_rows", "median"),
            train_rows_min=("n_train_rows", "min"),
            train_rows_max=("n_train_rows", "max"),
            train_compounds_median=("n_train_compounds", "median"),
            train_compounds_min=("n_train_compounds", "min"),
            train_compounds_max=("n_train_compounds", "max"),
            train_scaffolds_median=("n_train_scaffolds", "median"),
            train_scaffolds_min=("n_train_scaffolds", "min"),
            train_scaffolds_max=("n_train_scaffolds", "max"),
        )
        .reset_index()
        .set_index("training_fraction")
    )

    metric_rows: list[dict[str, Any]] = []
    for f_idx, fraction in enumerate(fractions):
        sizes = size_summary.loc[fraction]
        for m_idx, model in enumerate(models):
            for metric_idx, metric in enumerate(("spearman", "rmse")):
                low, high = percentile_interval(
                    draws[:, f_idx, m_idx, metric_idx]
                )
                metric_rows.append(
                    {
                        "training_fraction": fraction,
                        "model_id": model,
                        "metric": metric,
                        "estimate": float(point[f_idx, m_idx, metric_idx]),
                        "ci_low": low,
                        "ci_high": high,
                        "bootstrap_replicates": int(replicates),
                        "n_evaluation_rows": int(len(reference)),
                        "n_evaluation_scaffolds": int(
                            len(unique_scaffolds)
                        ),
                        **{
                            key: (
                                float(value)
                                if "median" in key
                                else int(value)
                            )
                            for key, value in sizes.items()
                        },
                    }
                )

    contrast_rows: list[dict[str, Any]] = []
    chem_idx = models.index("chemistry_extra_trees")
    full_idx = models.index("full_context_extra_trees")
    for f_idx, fraction in enumerate(fractions):
        definitions = (
            (
                "full_minus_chemistry_spearman",
                "spearman",
                point[f_idx, full_idx, 0] - point[f_idx, chem_idx, 0],
                draws[:, f_idx, full_idx, 0]
                - draws[:, f_idx, chem_idx, 0],
            ),
            (
                "chemistry_minus_full_rmse",
                "rmse_improvement",
                point[f_idx, chem_idx, 1] - point[f_idx, full_idx, 1],
                draws[:, f_idx, chem_idx, 1]
                - draws[:, f_idx, full_idx, 1],
            ),
        )
        for contrast_id, metric, estimate, values in definitions:
            low, high = percentile_interval(values)
            contrast_rows.append(
                {
                    "contrast_id": contrast_id,
                    "model_id": "paired_full_vs_chemistry",
                    "metric": metric,
                    "training_fraction_low": fraction,
                    "training_fraction_high": fraction,
                    "estimate": float(estimate),
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_replicates": int(replicates),
                }
            )

    f75 = fractions.index(0.75)
    f100 = fractions.index(1.00)
    for m_idx, model in enumerate(models):
        definitions = (
            (
                "fraction_100_minus_75_spearman",
                "spearman",
                point[f100, m_idx, 0] - point[f75, m_idx, 0],
                draws[:, f100, m_idx, 0] - draws[:, f75, m_idx, 0],
            ),
            (
                "fraction_75_minus_100_rmse",
                "rmse_improvement",
                point[f75, m_idx, 1] - point[f100, m_idx, 1],
                draws[:, f75, m_idx, 1] - draws[:, f100, m_idx, 1],
            ),
        )
        for contrast_id, metric, estimate, values in definitions:
            low, high = percentile_interval(values)
            contrast_rows.append(
                {
                    "contrast_id": contrast_id,
                    "model_id": model,
                    "metric": metric,
                    "training_fraction_low": 0.75,
                    "training_fraction_high": 1.00,
                    "estimate": float(estimate),
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_replicates": int(replicates),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(contrast_rows)


def artifact_inventory(output_dir: Path) -> pd.DataFrame:
    files = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file()
        and path.name not in {"artifact_sha256.csv", "run_manifest.json"}
    )
    rows = [
        {
            "artifact": path.name,
            "size_bytes": int(path.stat().st_size),
            "sha256": core.sha256_file(path),
        }
        for path in files
    ]
    rows.extend(
        [
            {
                "artifact": "protocol_document",
                "size_bytes": int(PROTOCOL_PATH.stat().st_size),
                "sha256": core.sha256_file(PROTOCOL_PATH),
            },
            {
                "artifact": "runner_script",
                "size_bytes": int(Path(__file__).stat().st_size),
                "sha256": core.sha256_file(Path(__file__)),
            },
        ]
    )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    started = time.time()
    output_dir = args.output_dir.resolve()
    prepare_output(output_dir, args.overwrite)
    data_path = args.data_file.resolve()
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if not PROTOCOL_PATH.is_file():
        raise FileNotFoundError(PROTOCOL_PATH)
    data_sha256 = core.sha256_file(data_path)
    if data_sha256 != core.FROZEN_DATA_SHA256:
        raise ValueError(
            "Input data hash does not match the confirmatory frozen identity"
        )

    rows, context_audit = core.load_rows(data_path)
    y = rows["pDC50"].to_numpy(dtype=np.float32)
    scaffold_ids = core.build_scaffold_ids(rows["canonical_smiles"])
    morgan = core.build_morgan_matrix(rows["canonical_smiles"])
    descriptors, descriptor_names, descriptor_failures = (
        core.build_descriptor_matrix(rows["canonical_smiles"])
    )
    splits = core.make_outer_splits(
        "scaffold",
        rows,
        scaffold_ids,
        n_folds=5,
        n_repeats=5,
        seed=args.seed,
    )
    if args.max_outer_splits is not None:
        splits = splits[: args.max_outer_splits]
        bootstrap_replicates = 0
        analysis_mode = "smoke_test"
    else:
        bootstrap_replicates = args.bootstrap_replicates
        analysis_mode = "formal_post_hoc"

    manifest: dict[str, Any] = {
        "analysis_identity": ANALYSIS_IDENTITY,
        "status": "running",
        "analysis_mode": analysis_mode,
        "evidence_identity": "post_hoc_sensitivity",
        "data_sha256": data_sha256,
        "protocol_sha256": core.sha256_file(PROTOCOL_PATH),
        "script_sha256": core.sha256_file(Path(__file__)),
        "fractions": list(FRACTIONS),
        "models": list(MODEL_IDS),
        "fixed_model_parameters": {
            **MODEL_PARAMS,
            "n_estimators": args.n_estimators,
            "bootstrap": False,
            "max_depth": None,
        },
        "outer_folds": 5,
        "outer_repeats": 5,
        "outer_splits_executed": len(splits),
        "bootstrap_replicates": bootstrap_replicates,
        "seed": args.seed,
        "n_rows": int(len(rows)),
        "n_compounds": int(rows["canonical_smiles"].nunique()),
        "n_scaffolds": int(len(np.unique(scaffold_ids))),
        "gpu_required": False,
        "boundaries": [
            "Post-hoc internal data-adequacy sensitivity.",
            "Fractions are nested by training scaffold, not by row.",
            "Folds and seeds are not independent replicates.",
            "No prospective, power, asymptotic, or sample-size claim.",
        ],
        "environment": core.environment_manifest(args),
    }
    write_json(output_dir / "run_manifest.json", manifest)
    context_audit.to_csv(
        output_dir / "context_sanitization_audit.csv",
        index=False,
    )

    prediction_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    training_size_rows: list[dict[str, Any]] = []
    for split_number, split in enumerate(splits, start=1):
        order_seed = args.seed + split.repeat * 10_000 + split.fold * 100
        ordered = ordered_training_scaffolds(
            scaffold_ids,
            split.train_idx,
            seed=order_seed,
        )
        for fraction_position, fraction in enumerate(FRACTIONS):
            train_idx, selected_scaffolds = select_training_indices(
                split.train_idx,
                scaffold_ids,
                ordered,
                fraction,
            )
            if set(scaffold_ids[train_idx]).intersection(
                set(scaffold_ids[split.test_idx])
            ):
                raise RuntimeError("Scaffold leakage in learning-curve subset")
            features, feature_meta = core.build_fold_features(
                rows,
                morgan,
                descriptors,
                train_idx,
                split.test_idx,
            )
            selection_sha256 = sha256_text(
                [str(value) for value in selected_scaffolds]
            )
            size_record = {
                "repeat": split.repeat,
                "outer_fold": split.fold,
                "training_fraction": fraction,
                "n_outer_train_scaffolds": int(len(ordered)),
                "n_train_rows": int(len(train_idx)),
                "n_train_compounds": int(
                    rows.iloc[train_idx]["canonical_smiles"].nunique()
                ),
                "n_train_scaffolds": int(len(selected_scaffolds)),
                "n_test_rows": int(len(split.test_idx)),
                "n_test_compounds": int(
                    rows.iloc[split.test_idx]["canonical_smiles"].nunique()
                ),
                "n_test_scaffolds": int(
                    len(np.unique(scaffold_ids[split.test_idx]))
                ),
                "selection_sha256": selection_sha256,
                **feature_meta,
            }
            training_size_rows.append(size_record)
            for model_position, model_id in enumerate(MODEL_IDS):
                model_seed = (
                    args.seed
                    + split.repeat * 100_000
                    + split.fold * 1_000
                    + fraction_position * 10
                    + model_position
                )
                pred = core.fit_predict_model(
                    model_id,
                    MODEL_PARAMS,
                    rows,
                    y,
                    morgan,
                    features,
                    train_idx,
                    split.test_idx,
                    seed=model_seed,
                    n_estimators=args.n_estimators,
                    n_jobs=args.n_jobs,
                )
                metrics = core.regression_metrics(y[split.test_idx], pred)
                fold_metric_rows.append(
                    {
                        "repeat": split.repeat,
                        "outer_fold": split.fold,
                        "training_fraction": fraction,
                        "model_id": model_id,
                        **size_record,
                        **metrics,
                    }
                )
                for local_idx, row_idx in enumerate(split.test_idx):
                    prediction_rows.append(
                        {
                            "repeat": split.repeat,
                            "outer_fold": split.fold,
                            "training_fraction": fraction,
                            "model_id": model_id,
                            "row_index": int(row_idx),
                            "qc_id": str(rows.iloc[row_idx]["qc_id"]),
                            "y_true": float(y[row_idx]),
                            "y_pred": float(pred[local_idx]),
                            "scaffold_id": str(scaffold_ids[row_idx]),
                        }
                    )
        print(
            f"[LEARNING_CURVE] split={split_number}/{len(splits)} complete",
            flush=True,
        )

    predictions = pd.DataFrame(prediction_rows)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    training_sizes = pd.DataFrame(training_size_rows)
    averaged = average_predictions(predictions)
    expected_repeats = (
        5 if args.max_outer_splits is None else predictions["repeat"].nunique()
    )
    if not (averaged["n_repeat_predictions"] == expected_repeats).all():
        if args.max_outer_splits is None:
            raise RuntimeError("Formal run lacks five predictions per row")

    core.atomic_to_csv(
        predictions,
        output_dir / "cross_fitted_predictions.csv",
    )
    core.atomic_to_csv(
        averaged,
        output_dir / "repeat_averaged_predictions.csv",
    )
    core.atomic_to_csv(
        fold_metrics,
        output_dir / "outer_fold_metrics.csv",
    )
    core.atomic_to_csv(
        training_sizes,
        output_dir / "training_sizes.csv",
    )
    metric_summary, contrasts = bootstrap_learning_curve(
        averaged,
        training_sizes,
        replicates=bootstrap_replicates,
        seed=args.seed,
    )
    core.atomic_to_csv(
        metric_summary,
        output_dir / "learning_curve_metrics.csv",
    )
    core.atomic_to_csv(
        contrasts,
        output_dir / "learning_curve_contrasts.csv",
    )
    descriptor_payload = {
        "descriptor_names": descriptor_names,
        "descriptor_failure_counts": descriptor_failures,
    }
    write_json(output_dir / "feature_manifest.json", descriptor_payload)
    inventory = artifact_inventory(output_dir)
    core.atomic_to_csv(inventory, output_dir / "artifact_sha256.csv")

    manifest.update(
        {
            "status": "complete",
            "runtime_seconds": float(time.time() - started),
            "prediction_rows": int(len(predictions)),
            "repeat_averaged_rows": int(len(averaged)),
            "fit_count": int(
                len(splits) * len(FRACTIONS) * len(MODEL_IDS)
            ),
            "artifact_count": int(len(inventory)),
            "artifact_sha256": {
                row["artifact"]: row["sha256"]
                for row in inventory.to_dict(orient="records")
            },
        }
    )
    write_json(output_dir / "run_manifest.json", manifest)
    print(
        (
            f"POST_HOC_INTERNAL_LEARNING_CURVE: PASS "
            f"(fits={manifest['fit_count']}, "
            f"predictions={manifest['prediction_rows']})"
        ),
        flush=True,
    )
    print("GPU_REQUIRED: NO", flush=True)


if __name__ == "__main__":
    main()
