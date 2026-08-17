#!/usr/bin/env python3
"""Run the frozen CPU HistGradientBoosting model-family sensitivity.

The protocol is post-hoc by design. It compares only chemistry and
chemistry-plus-context feature blocks under the frozen internal, OOD and
strict-OOD splits. No hyperparameter selection is performed.
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

import numpy as np
import pandas as pd
import rdkit
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor


SCRIPT_DIR = Path(__file__).resolve().parent
ROUTE_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_confirmatory_cpu_v1 as core  # noqa: E402
import run_confirmatory_ood_cpu_v1 as ood  # noqa: E402


ANALYSIS_LABEL = "post_hoc_model_family_sensitivity"
PROTOCOL_VERSION = "post_hoc_histgradientboosting_model_family_sensitivity_v1"
SEED = 260531
BOOTSTRAP_REPLICATES = 10_000

CHEM = "chemistry_hist_gradient_boosting"
FULL = "full_context_hist_gradient_boosting"
MODEL_ORDER = (CHEM, FULL)

DATA_FILE = (
    ROUTE_DIR
    / "data"
    / "processed"
    / "all_molglue_dc50_qc_train_test_standardized_context.csv"
)
PROTOCOL_FILE = (
    ROUTE_DIR
    / "docs"
    / "post_hoc_histgradientboosting_model_family_sensitivity_protocol_v1.md"
)
STRICT_AUDIT_FILE = (
    ROUTE_DIR
    / "reports"
    / "post_hoc_strict_domain_scaffold_ood_cpu_v1"
    / "strict_ood_split_audit.csv"
)
DEFAULT_OUTPUT = (
    ROUTE_DIR
    / "reports"
    / "post_hoc_histgradientboosting_model_family_sensitivity_v1"
)

HGB_PARAMS: dict[str, Any] = {
    "loss": "squared_error",
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "max_bins": 64,
    "early_stopping": False,
}


@dataclass(frozen=True)
class EvaluationSplit:
    split_regime: str
    protocol: str
    heldout_group: str
    split_order: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.ptp(y_true) == 0 or np.ptp(y_pred) == 0:
        return float("nan")
    return float(spearmanr(y_true, y_pred).statistic)


def metric_pair(frame: pd.DataFrame) -> tuple[float, float]:
    y_true = frame["y_true"].to_numpy(dtype=float)
    y_pred = frame["y_pred"].to_numpy(dtype=float)
    rho = safe_spearman(y_true, y_pred)
    rmse = float(np.sqrt(np.mean(np.square(y_true - y_pred))))
    return rho, rmse


def fit_predict(
    features: dict[str, tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    train_idx: np.ndarray,
    model_id: str,
    seed: int,
) -> np.ndarray:
    block = "chemistry" if model_id == CHEM else "full"
    x_train, x_test = features[block]
    model = HistGradientBoostingRegressor(
        **HGB_PARAMS,
        random_state=seed,
    )
    model.fit(x_train, y[train_idx])
    prediction = np.asarray(model.predict(x_test), dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Non-finite prediction from {model_id}")
    return prediction


def build_internal_splits(
    rows: pd.DataFrame, scaffold_ids: np.ndarray, smoke: bool
) -> list[EvaluationSplit]:
    core_splits = core.make_outer_splits(
        "scaffold",
        rows,
        scaffold_ids,
        n_folds=5,
        n_repeats=5,
        seed=SEED,
    )
    if smoke:
        core_splits = core_splits[:1]
    result: list[EvaluationSplit] = []
    for position, split in enumerate(core_splits, start=1):
        result.append(
            EvaluationSplit(
                split_regime="internal",
                protocol="scaffold",
                heldout_group=f"repeat_{split.repeat}_fold_{split.fold}",
                split_order=position,
                train_idx=np.asarray(split.train_idx, dtype=np.int64),
                test_idx=np.asarray(split.test_idx, dtype=np.int64),
                seed=SEED + split.repeat * 10_000 + split.fold * 100,
            )
        )
    return result


def build_ood_and_strict_splits(
    rows: pd.DataFrame, scaffold_ids: np.ndarray, smoke: bool
) -> tuple[list[EvaluationSplit], list[EvaluationSplit], pd.DataFrame]:
    max_groups = 1 if smoke else None
    base, base_audit = ood.build_ood_splits(
        rows,
        scaffold_ids,
        ["source_ood", "target_ood"],
        SEED,
        max_groups,
    )
    base_splits: list[EvaluationSplit] = []
    strict_splits: list[EvaluationSplit] = []
    strict_reference = pd.read_csv(STRICT_AUDIT_FILE)
    strict_audit_rows: list[dict[str, Any]] = []

    for position, split in enumerate(base, start=1):
        base_splits.append(
            EvaluationSplit(
                split_regime="domain_plus_compound_cold",
                protocol=split.protocol,
                heldout_group=split.heldout_group,
                split_order=position,
                train_idx=np.asarray(split.train_idx, dtype=np.int64),
                test_idx=np.asarray(split.test_idx, dtype=np.int64),
                seed=int(split.split_seed),
            )
        )

        test_scaffolds = set(
            scaffold_ids[split.test_idx].astype(str).tolist()
        )
        keep = ~np.isin(
            scaffold_ids[split.train_idx].astype(str),
            list(test_scaffolds),
        )
        strict_train = np.asarray(split.train_idx[keep], dtype=np.int64)
        strict_test = np.asarray(split.test_idx, dtype=np.int64)
        train_scaffolds = set(
            scaffold_ids[strict_train].astype(str).tolist()
        )
        train_smiles = set(
            rows.iloc[strict_train]["canonical_smiles"].astype(str)
        )
        test_smiles = set(
            rows.iloc[strict_test]["canonical_smiles"].astype(str)
        )
        if (
            set(strict_train).intersection(set(strict_test))
            or train_smiles.intersection(test_smiles)
            or train_scaffolds.intersection(test_scaffolds)
        ):
            raise RuntimeError(
                f"Strict leakage: {split.protocol} {split.heldout_group}"
            )

        reference = strict_reference[
            (strict_reference["protocol"] == split.protocol)
            & (strict_reference["heldout_group"] == split.heldout_group)
        ]
        if len(reference) != 1:
            raise RuntimeError("Strict reference split is not unique")
        reference_row = reference.iloc[0]
        observed = {
            "n_train_rows": len(strict_train),
            "n_test_rows": len(strict_test),
            "n_scaffold_overlap_rows_excluded": int((~keep).sum()),
            "train_test_row_overlap": 0,
            "train_test_smiles_overlap": 0,
            "train_test_scaffold_overlap": 0,
        }
        for key, value in observed.items():
            if int(reference_row[key]) != int(value):
                raise RuntimeError(
                    f"Strict split count mismatch for {split.protocol} "
                    f"{split.heldout_group}: {key}"
                )
        strict_audit_rows.append(
            {
                "analysis_label": ANALYSIS_LABEL,
                "protocol": split.protocol,
                "heldout_group": split.heldout_group,
                **observed,
                "heldout_token_seen_in_train": False,
            }
        )
        strict_splits.append(
            EvaluationSplit(
                split_regime="strict_domain_plus_scaffold_cold",
                protocol=split.protocol,
                heldout_group=split.heldout_group,
                split_order=position,
                train_idx=strict_train,
                test_idx=strict_test,
                seed=int(split.split_seed) + 500_000,
            )
        )
    return base_splits, strict_splits, pd.DataFrame(strict_audit_rows)


def run_splits(
    splits: list[EvaluationSplit],
    rows: pd.DataFrame,
    y: np.ndarray,
    morgan: np.ndarray,
    descriptors: np.ndarray,
    scaffold_ids: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    start = time.perf_counter()
    for split in splits:
        print(
            f"[FIT] regime={split.split_regime} protocol={split.protocol} "
            f"group={split.heldout_group} ({split.split_order}/{len(splits)}) "
            f"train={len(split.train_idx)} test={len(split.test_idx)}",
            flush=True,
        )
        features, metadata = core.build_fold_features(
            rows,
            morgan,
            descriptors,
            split.train_idx,
            split.test_idx,
        )
        for model_position, model_id in enumerate(MODEL_ORDER, start=1):
            prediction = fit_predict(
                features,
                y,
                split.train_idx,
                model_id,
                split.seed + model_position,
            )
            for local_position, row_idx in enumerate(split.test_idx):
                records.append(
                    {
                        "analysis_label": ANALYSIS_LABEL,
                        "split_regime": split.split_regime,
                        "protocol": split.protocol,
                        "heldout_group": split.heldout_group,
                        "split_order": split.split_order,
                        "row_index": int(row_idx),
                        "qc_id": str(rows.iloc[row_idx]["qc_id"]),
                        "model_id": model_id,
                        "y_true": float(y[row_idx]),
                        "y_pred": float(prediction[local_position]),
                        "canonical_smiles": str(
                            rows.iloc[row_idx]["canonical_smiles"]
                        ),
                        "scaffold_id": str(scaffold_ids[row_idx]),
                        "source_database": str(
                            rows.iloc[row_idx]["source_database"]
                        ),
                        "target_protein": str(
                            rows.iloc[row_idx]["target_protein"]
                        ),
                        "recruiting_protein": str(
                            rows.iloc[row_idx]["recruiting_protein"]
                        ),
                        "cell_line": str(rows.iloc[row_idx]["cell_line"]),
                        **metadata,
                    }
                )
        print(
            f"[FIT] completed={split.split_order}/{len(splits)} "
            f"elapsed={time.perf_counter() - start:.1f}s",
            flush=True,
        )
    return pd.DataFrame(records)


def summarize_internal(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    averaged = (
        predictions.groupby(
            [
                "model_id",
                "row_index",
                "qc_id",
                "canonical_smiles",
                "scaffold_id",
            ],
            sort=True,
            as_index=False,
        )
        .agg(
            y_true=("y_true", "first"),
            y_pred=("y_pred", "mean"),
            n_oof_predictions=("y_pred", "size"),
        )
    )
    metrics = []
    for model_id, group in averaged.groupby("model_id", sort=True):
        rho, error = metric_pair(group)
        metrics.append(
            {
                "analysis_label": ANALYSIS_LABEL,
                "split_regime": "internal",
                "protocol": "scaffold",
                "model_id": model_id,
                "n_rows": len(group),
                "n_scaffolds": group["scaffold_id"].nunique(),
                "spearman": rho,
                "rmse": error,
                "n_oof_predictions_min": group["n_oof_predictions"].min(),
                "n_oof_predictions_max": group["n_oof_predictions"].max(),
            }
        )
    return averaged, pd.DataFrame(metrics)


def domain_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["split_regime", "protocol", "heldout_group", "model_id"]
    for key, group in predictions.groupby(keys, sort=True):
        rho, error = metric_pair(group)
        rows.append(
            {
                "analysis_label": ANALYSIS_LABEL,
                "split_regime": key[0],
                "protocol": key[1],
                "heldout_group": key[2],
                "model_id": key[3],
                "n_rows": len(group),
                "n_unique_smiles": group["canonical_smiles"].nunique(),
                "n_scaffolds": group["scaffold_id"].nunique(),
                "spearman": rho,
                "rmse": error,
            }
        )
    return pd.DataFrame(rows)


def aggregate_ood(
    predictions: pd.DataFrame, local_domain_metrics: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    keys = ["split_regime", "protocol", "model_id"]
    for key, group in predictions.groupby(keys, sort=True):
        domains = local_domain_metrics[
            (local_domain_metrics["split_regime"] == key[0])
            & (local_domain_metrics["protocol"] == key[1])
            & (local_domain_metrics["model_id"] == key[2])
        ]
        finite = domains["spearman"].dropna()
        pooled_rho, pooled_error = metric_pair(group)
        rows.append(
            {
                "analysis_label": ANALYSIS_LABEL,
                "split_regime": key[0],
                "protocol": key[1],
                "model_id": key[2],
                "n_domains": domains["heldout_group"].nunique(),
                "n_rows": len(group),
                "n_scaffolds": group["scaffold_id"].nunique(),
                "domain_macro_spearman": (
                    finite.mean() if len(finite) == len(domains) else float("nan")
                ),
                "domain_macro_spearman_finite_domains": len(finite),
                "domain_macro_rmse": domains["rmse"].mean(),
                "pooled_spearman": pooled_rho,
                "pooled_rmse": pooled_error,
            }
        )
    return pd.DataFrame(rows)


def resample_by_cluster(
    frame: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    groups = {
        key: local.index.to_numpy(dtype=np.int64)
        for key, local in frame.groupby("scaffold_id", sort=True)
    }
    cluster_ids = np.asarray(list(groups), dtype=object)
    sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
    indices = np.concatenate([groups[key] for key in sampled])
    return frame.loc[indices].reset_index(drop=True)


def bootstrap_internal(
    averaged: pd.DataFrame, n_bootstrap: int, seed: int
) -> pd.DataFrame:
    wide = averaged.pivot(
        index=[
            "row_index",
            "qc_id",
            "canonical_smiles",
            "scaffold_id",
            "y_true",
        ],
        columns="model_id",
        values="y_pred",
    ).reset_index()
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {
        "chem_rho": [],
        "chem_rmse": [],
        "full_rho": [],
        "full_rmse": [],
        "delta_rho": [],
        "delta_rmse": [],
    }
    for replicate in range(n_bootstrap):
        boot = resample_by_cluster(wide, rng)
        truth = boot["y_true"].to_numpy(dtype=float)
        chem = boot[CHEM].to_numpy(dtype=float)
        full = boot[FULL].to_numpy(dtype=float)
        chem_rho = safe_spearman(truth, chem)
        full_rho = safe_spearman(truth, full)
        chem_rmse = float(np.sqrt(np.mean(np.square(truth - chem))))
        full_rmse = float(np.sqrt(np.mean(np.square(truth - full))))
        samples["chem_rho"].append(chem_rho)
        samples["chem_rmse"].append(chem_rmse)
        samples["full_rho"].append(full_rho)
        samples["full_rmse"].append(full_rmse)
        samples["delta_rho"].append(full_rho - chem_rho)
        samples["delta_rmse"].append(chem_rmse - full_rmse)
        if (replicate + 1) % 1000 == 0:
            print(f"[BOOT] internal {replicate + 1}/{n_bootstrap}", flush=True)

    observed = {}
    truth = wide["y_true"].to_numpy(dtype=float)
    for model_id in MODEL_ORDER:
        pred = wide[model_id].to_numpy(dtype=float)
        observed[model_id] = {
            "rho": safe_spearman(truth, pred),
            "rmse": float(np.sqrt(np.mean(np.square(truth - pred)))),
        }

    rows = []
    for model_id, rho_key, rmse_key in (
        (CHEM, "chem_rho", "chem_rmse"),
        (FULL, "full_rho", "full_rmse"),
    ):
        rho_values = np.asarray(samples[rho_key], dtype=float)
        rmse_values = np.asarray(samples[rmse_key], dtype=float)
        rows.append(
            {
                "analysis_label": ANALYSIS_LABEL,
                "record_type": "model",
                "split_regime": "internal",
                "protocol": "scaffold",
                "model_id": model_id,
                "n_clusters": wide["scaffold_id"].nunique(),
                "n_bootstrap": n_bootstrap,
                "spearman_observed": observed[model_id]["rho"],
                "spearman_ci_low": np.quantile(rho_values, 0.025),
                "spearman_ci_high": np.quantile(rho_values, 0.975),
                "rmse_observed": observed[model_id]["rmse"],
                "rmse_ci_low": np.quantile(rmse_values, 0.025),
                "rmse_ci_high": np.quantile(rmse_values, 0.975),
            }
        )
    rows.append(
        {
            "analysis_label": ANALYSIS_LABEL,
            "record_type": "contrast",
            "split_regime": "internal",
            "protocol": "scaffold",
            "contrast_id": "full_vs_chemistry_hist_gradient_boosting",
            "first_model": FULL,
            "comparator_model": CHEM,
            "n_clusters": wide["scaffold_id"].nunique(),
            "n_bootstrap": n_bootstrap,
            "delta_spearman_observed": (
                observed[FULL]["rho"] - observed[CHEM]["rho"]
            ),
            "delta_spearman_ci_low": np.quantile(
                samples["delta_rho"], 0.025
            ),
            "delta_spearman_ci_high": np.quantile(
                samples["delta_rho"], 0.975
            ),
            "delta_rmse_observed": (
                observed[CHEM]["rmse"] - observed[FULL]["rmse"]
            ),
            "delta_rmse_ci_low": np.quantile(
                samples["delta_rmse"], 0.025
            ),
            "delta_rmse_ci_high": np.quantile(
                samples["delta_rmse"], 0.975
            ),
        }
    )
    return pd.DataFrame(rows)


def domain_macro_for_model(
    frame: pd.DataFrame, model_id: str
) -> tuple[float, float, float, float]:
    model = frame[frame["model_id"] == model_id]
    domain_rho = []
    domain_rmse = []
    for _, group in model.groupby("heldout_group", sort=True):
        rho, error = metric_pair(group)
        domain_rho.append(rho)
        domain_rmse.append(error)
    macro_rho = (
        float(np.mean(domain_rho))
        if np.isfinite(domain_rho).all()
        else float("nan")
    )
    pooled_rho, pooled_rmse = metric_pair(model)
    return (
        macro_rho,
        float(np.mean(domain_rmse)),
        pooled_rho,
        pooled_rmse,
    )


def bootstrap_ood(
    predictions: pd.DataFrame,
    split_regime: str,
    protocol: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    frame = predictions[
        (predictions["split_regime"] == split_regime)
        & (predictions["protocol"] == protocol)
    ].reset_index(drop=True)
    if frame.empty:
        raise RuntimeError(f"No OOD predictions for {split_regime} {protocol}")
    rng = np.random.default_rng(seed)
    samples = {
        model_id: {
            "macro_rho": [],
            "macro_rmse": [],
            "pooled_rho": [],
            "pooled_rmse": [],
        }
        for model_id in MODEL_ORDER
    }
    delta_rho: list[float] = []
    delta_rmse: list[float] = []
    for replicate in range(n_bootstrap):
        boot = resample_by_cluster(frame, rng)
        local = {}
        for model_id in MODEL_ORDER:
            values = domain_macro_for_model(boot, model_id)
            local[model_id] = values
            for name, value in zip(
                ("macro_rho", "macro_rmse", "pooled_rho", "pooled_rmse"),
                values,
            ):
                samples[model_id][name].append(value)
        delta_rho.append(local[FULL][0] - local[CHEM][0])
        delta_rmse.append(local[CHEM][1] - local[FULL][1])
        if (replicate + 1) % 1000 == 0:
            print(
                f"[BOOT] {split_regime} {protocol} "
                f"{replicate + 1}/{n_bootstrap}",
                flush=True,
            )

    observed = {
        model_id: domain_macro_for_model(frame, model_id)
        for model_id in MODEL_ORDER
    }

    def interval(values: list[float]) -> tuple[float, float, int]:
        array = np.asarray(values, dtype=float)
        array = array[np.isfinite(array)]
        if len(array) == 0:
            return float("nan"), float("nan"), 0
        return (
            float(np.quantile(array, 0.025)),
            float(np.quantile(array, 0.975)),
            len(array),
        )

    rows = []
    for model_id in MODEL_ORDER:
        rho_low, rho_high, rho_n = interval(
            samples[model_id]["macro_rho"]
        )
        rmse_low, rmse_high, rmse_n = interval(
            samples[model_id]["macro_rmse"]
        )
        rows.append(
            {
                "analysis_label": ANALYSIS_LABEL,
                "record_type": "model",
                "split_regime": split_regime,
                "protocol": protocol,
                "model_id": model_id,
                "n_domains": frame["heldout_group"].nunique(),
                "n_clusters": frame["scaffold_id"].nunique(),
                "n_bootstrap": n_bootstrap,
                "domain_macro_spearman_observed": observed[model_id][0],
                "domain_macro_spearman_ci_low": rho_low,
                "domain_macro_spearman_ci_high": rho_high,
                "domain_macro_spearman_n_valid": rho_n,
                "domain_macro_rmse_observed": observed[model_id][1],
                "domain_macro_rmse_ci_low": rmse_low,
                "domain_macro_rmse_ci_high": rmse_high,
                "domain_macro_rmse_n_valid": rmse_n,
                "pooled_spearman_observed": observed[model_id][2],
                "pooled_rmse_observed": observed[model_id][3],
            }
        )
    delta_rho_low, delta_rho_high, delta_rho_n = interval(delta_rho)
    delta_rmse_low, delta_rmse_high, delta_rmse_n = interval(delta_rmse)
    rows.append(
        {
            "analysis_label": ANALYSIS_LABEL,
            "record_type": "contrast",
            "split_regime": split_regime,
            "protocol": protocol,
            "contrast_id": "full_vs_chemistry_hist_gradient_boosting",
            "first_model": FULL,
            "comparator_model": CHEM,
            "n_domains": frame["heldout_group"].nunique(),
            "n_clusters": frame["scaffold_id"].nunique(),
            "n_bootstrap": n_bootstrap,
            "delta_domain_macro_spearman_observed": (
                observed[FULL][0] - observed[CHEM][0]
            ),
            "delta_domain_macro_spearman_ci_low": delta_rho_low,
            "delta_domain_macro_spearman_ci_high": delta_rho_high,
            "delta_domain_macro_spearman_n_valid": delta_rho_n,
            "delta_domain_macro_rmse_observed": (
                observed[CHEM][1] - observed[FULL][1]
            ),
            "delta_domain_macro_rmse_ci_low": delta_rmse_low,
            "delta_domain_macro_rmse_ci_high": delta_rmse_high,
            "delta_domain_macro_rmse_n_valid": delta_rmse_n,
        }
    )
    return pd.DataFrame(rows)


def artifact_inventory(output_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name == "artifact_sha256.csv":
            continue
        rows.append(
            {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def build_results_brief(
    output_dir: Path,
    internal_metrics: pd.DataFrame,
    ood_aggregate: pd.DataFrame,
    bootstrap: pd.DataFrame,
    n_bootstrap: int,
    smoke: bool,
) -> None:
    internal_contrast = bootstrap[
        (bootstrap["record_type"] == "contrast")
        & (bootstrap["split_regime"] == "internal")
    ].iloc[0]
    ood_contrasts = bootstrap[
        (bootstrap["record_type"] == "contrast")
        & (bootstrap["split_regime"] != "internal")
    ].copy()
    base = ood_contrasts[
        ood_contrasts["split_regime"] == "domain_plus_compound_cold"
    ]
    qualitative_replication = bool(
        internal_contrast["delta_spearman_observed"] > 0
        and (base["delta_domain_macro_spearman_observed"] <= 0).all()
    )

    model_lines = []
    for row in internal_metrics.to_dict("records"):
        model_lines.append(
            f"| internal scaffold | {row['model_id']} | "
            f"{row['spearman']:.4f} | {row['rmse']:.4f} |"
        )
    for row in ood_aggregate.to_dict("records"):
        rho = (
            "NA"
            if not math.isfinite(float(row["domain_macro_spearman"]))
            else f"{row['domain_macro_spearman']:.4f}"
        )
        model_lines.append(
            f"| {row['split_regime']} / {row['protocol']} | "
            f"{row['model_id']} | {rho} | "
            f"{row['domain_macro_rmse']:.4f} |"
        )

    contrast_lines = [
        "| internal scaffold | "
        f"{internal_contrast['delta_spearman_observed']:.4f} "
        f"[{internal_contrast['delta_spearman_ci_low']:.4f}, "
        f"{internal_contrast['delta_spearman_ci_high']:.4f}] | "
        f"{internal_contrast['delta_rmse_observed']:.4f} "
        f"[{internal_contrast['delta_rmse_ci_low']:.4f}, "
        f"{internal_contrast['delta_rmse_ci_high']:.4f}] |"
    ]
    for row in ood_contrasts.to_dict("records"):
        rho = row["delta_domain_macro_spearman_observed"]
        rho_low = row["delta_domain_macro_spearman_ci_low"]
        rho_high = row["delta_domain_macro_spearman_ci_high"]
        rho_text = (
            "不可估"
            if not math.isfinite(float(rho))
            else f"{rho:.4f} [{rho_low:.4f}, {rho_high:.4f}]"
        )
        contrast_lines.append(
            f"| {row['split_regime']} / {row['protocol']} | {rho_text} | "
            f"{row['delta_domain_macro_rmse_observed']:.4f} "
            f"[{row['delta_domain_macro_rmse_ci_low']:.4f}, "
            f"{row['delta_domain_macro_rmse_ci_high']:.4f}] |"
        )

    if smoke:
        decision = (
            "这是代码与拆分 smoke test，不作科学判定；只有完整 25 个内部 "
            "outer splits、12 个 OOD 域及正式 bootstrap 完成后才应用冻结标准。"
        )
    else:
        decision = (
            "方向上跨学习器复现：内部 full 相对 chemistry-only 为正，而冻结的 "
            "source/target OOD 中 full 没有获得排序增益。"
            if qualitative_replication
            else "未满足预设的跨学习器方向复现标准；论文应把该模式收窄到 "
            "ExtraTrees pipeline，不得继续增加 learner 追逐结果。"
        )
    brief = f"""# HistGradientBoosting 模型家族后验敏感性结果

## 身份

- 分析：`{ANALYSIS_LABEL}`；
- 单一冻结 CPU learner，无超参数选择；
- 该结果不是 confirmatory，不报告 p 值；
- 所有区间为 {n_bootstrap:,} 次配对 global-scaffold percentile bootstrap；
- smoke test：`{str(smoke).lower()}`。

## 模型结果

| 场景 | 模型 | Spearman / domain-macro Spearman | RMSE / domain-macro RMSE |
|---|---|---:|---:|
{os.linesep.join(model_lines)}

## Full − chemistry-only 配对对比

正的 `ΔSpearman` 和 `ΔRMSE` 有利于 full。

| 场景 | ΔSpearman（95% CI） | ΔRMSE（95% CI） |
|---|---:|---:|
{os.linesep.join(contrast_lines)}

## 冻结判定

{decision}

## 边界

- 本分析只检验评价模式能否跨一个额外 learner 复现，不构成模型搜索；
- strict 结果仍是 post-hoc sensitivity；
- 固定的 4 个 source 域和 8 个 target 域不是从更大总体随机抽样；
- 相对模型对比不能建立绝对 unseen-target 泛化、校准、候选发现或机制主张；
- 全程 CPU-only，未使用 GPU。
"""
    (output_dir / "results_brief_zh.md").write_text(brief, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.smoke and args.output_dir == DEFAULT_OUTPUT:
        output_dir = Path(f"{DEFAULT_OUTPUT}_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.iterdir())
    if existing:
        raise RuntimeError(
            f"Output directory is not empty; refusing overwrite: {output_dir}"
        )

    data_hash = sha256_file(DATA_FILE)
    if data_hash != core.FROZEN_DATA_SHA256:
        raise RuntimeError("Frozen data hash mismatch")

    started = time.time()
    rows, context_audit = core.load_rows(DATA_FILE)
    y = rows["pDC50"].to_numpy(dtype=float)
    scaffold_ids = core.build_scaffold_ids(rows["canonical_smiles"])
    morgan = core.build_morgan_matrix(rows["canonical_smiles"])
    descriptors, descriptor_names, descriptor_failures = (
        core.build_descriptor_matrix(rows["canonical_smiles"])
    )
    internal_splits = build_internal_splits(rows, scaffold_ids, args.smoke)
    ood_splits, strict_splits, strict_audit = build_ood_and_strict_splits(
        rows, scaffold_ids, args.smoke
    )

    internal_predictions = run_splits(
        internal_splits,
        rows,
        y,
        morgan,
        descriptors,
        scaffold_ids,
    )
    ood_predictions = run_splits(
        [*ood_splits, *strict_splits],
        rows,
        y,
        morgan,
        descriptors,
        scaffold_ids,
    )
    all_predictions = pd.concat(
        [internal_predictions, ood_predictions], ignore_index=True
    )
    all_predictions.to_csv(output_dir / "predictions.csv", index=False)
    context_audit.to_csv(output_dir / "context_sanitization_audit.csv", index=False)
    strict_audit.to_csv(output_dir / "strict_split_audit.csv", index=False)

    averaged, internal_metrics = summarize_internal(internal_predictions)
    averaged.to_csv(
        output_dir / "internal_repeat_averaged_predictions.csv", index=False
    )
    internal_metrics.to_csv(output_dir / "internal_metrics.csv", index=False)
    local_domain_metrics = domain_metrics(ood_predictions)
    local_domain_metrics.to_csv(output_dir / "ood_domain_metrics.csv", index=False)
    ood_aggregate = aggregate_ood(ood_predictions, local_domain_metrics)
    ood_aggregate.to_csv(output_dir / "ood_aggregate_metrics.csv", index=False)

    n_bootstrap = 100 if args.smoke else BOOTSTRAP_REPLICATES
    internal_boot = bootstrap_internal(averaged, n_bootstrap, SEED + 700_000)
    boot_frames = [internal_boot]
    for regime_position, regime in enumerate(
        ("domain_plus_compound_cold", "strict_domain_plus_scaffold_cold"),
        start=1,
    ):
        for protocol_position, protocol in enumerate(
            ("source_ood", "target_ood"), start=1
        ):
            boot_frames.append(
                bootstrap_ood(
                    ood_predictions,
                    regime,
                    protocol,
                    n_bootstrap,
                    SEED
                    + 800_000
                    + regime_position * 10_000
                    + protocol_position * 1_000,
                )
            )
    bootstrap = pd.concat(boot_frames, ignore_index=True, sort=False)
    bootstrap.to_csv(
        output_dir / "paired_global_scaffold_bootstrap.csv", index=False
    )
    build_results_brief(
        output_dir,
        internal_metrics,
        ood_aggregate,
        bootstrap,
        n_bootstrap,
        bool(args.smoke),
    )

    expected_internal_predictions = sum(
        len(split.test_idx) * len(MODEL_ORDER)
        for split in internal_splits
    )
    expected_ood_predictions = sum(
        len(split.test_idx) * len(MODEL_ORDER)
        for split in [*ood_splits, *strict_splits]
    )
    qa = {
        "analysis_label": ANALYSIS_LABEL,
        "status": "PASS",
        "smoke": bool(args.smoke),
        "checks": {
            "data_hash_match": True,
            "all_predictions_finite": bool(
                np.isfinite(all_predictions["y_pred"]).all()
            ),
            "internal_prediction_count": (
                len(internal_predictions) == expected_internal_predictions
            ),
            "ood_prediction_count": (
                len(ood_predictions) == expected_ood_predictions
            ),
            "strict_zero_row_overlap": bool(
                (strict_audit["train_test_row_overlap"] == 0).all()
            ),
            "strict_zero_compound_overlap": bool(
                (strict_audit["train_test_smiles_overlap"] == 0).all()
            ),
            "strict_zero_scaffold_overlap": bool(
                (strict_audit["train_test_scaffold_overlap"] == 0).all()
            ),
        },
    }
    if not all(qa["checks"].values()):
        qa["status"] = "FAIL"
        write_json(output_dir / "qa_summary.json", qa)
        raise RuntimeError(f"QA failure: {qa['checks']}")
    write_json(output_dir / "qa_summary.json", qa)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "analysis_label": ANALYSIS_LABEL,
        "status": "complete",
        "confirmatory": False,
        "smoke": bool(args.smoke),
        "gpu_used": False,
        "data_file": str(DATA_FILE),
        "data_sha256": data_hash,
        "script_sha256": sha256_file(Path(__file__)),
        "protocol_document": str(PROTOCOL_FILE),
        "protocol_document_sha256": sha256_file(PROTOCOL_FILE),
        "n_rows": len(rows),
        "n_unique_smiles": rows["canonical_smiles"].nunique(),
        "n_scaffolds": len(np.unique(scaffold_ids)),
        "models": list(MODEL_ORDER),
        "learner": {
            "class": "sklearn.ensemble.HistGradientBoostingRegressor",
            "parameters": HGB_PARAMS,
            "no_hyperparameter_selection": True,
        },
        "splits": {
            "internal": len(internal_splits),
            "domain_plus_compound_cold": len(ood_splits),
            "strict_domain_plus_scaffold_cold": len(strict_splits),
        },
        "bootstrap_replicates": n_bootstrap,
        "seed": SEED,
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "rdkit": rdkit.__version__,
            "visible_cpu_count": os.cpu_count(),
        },
        "elapsed_seconds": time.time() - started,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    artifact_inventory(output_dir).to_csv(
        output_dir / "artifact_sha256.csv", index=False
    )
    print(
        f"[DONE] status=complete smoke={args.smoke} "
        f"elapsed={manifest['elapsed_seconds']:.1f}s output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
