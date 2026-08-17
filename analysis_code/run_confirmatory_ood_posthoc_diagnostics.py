#!/usr/bin/env python3
"""No-retraining exploratory diagnostics for the frozen formal OOD run.

This script reads and hash-verifies the completed confirmatory OOD artifacts.
It never fits a model and refuses to write inside the formal result directory.
All derived analyses are explicitly post hoc and conditional on the fixed
source and target domains.
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

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROTOCOL_VERSION = "confirmatory_ood_cpu_v1_posthoc.1.0"
ANALYSIS_MODE = "post_hoc_exploratory"
ROUTE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FORMAL_DIR = ROUTE_DIR / "reports" / "confirmatory_ood_cpu_v1"
DEFAULT_OUTPUT_DIR = (
    ROUTE_DIR / "reports" / "confirmatory_ood_cpu_v1_posthoc"
)
DEFAULT_PROTOCOL_NOTE = (
    ROUTE_DIR / "docs" / "confirmatory_ood_cpu_v1_posthoc_protocol.md"
)

EXPECTED_PARENT_MANIFEST_SHA256 = (
    "1692bd2bcf729a8d099875bc1d640f351e30691afa480f54a42e8a99cadf4b65"
)
EXPECTED_PARENT_CONFIGURATION_SHA256 = (
    "fd10b444eb205af2a150e81b19b281301d0bc0a6c28cf8a30892297ea8a0ceab"
)
EXPECTED_PARENT_PREDICTIONS_SHA256 = (
    "0af39d248adff5cd17fc6d21cce1bfbd0c38051a740d14019d2e80b49c0cef5a"
)

PROTOCOL_ORDER = ("source_ood", "target_ood")
CALIBRATION_MODELS = (
    "full_context_extra_trees",
    "chemistry_extra_trees",
)
FULL_MODEL = "full_context_extra_trees"
CHEMISTRY_MODEL = "chemistry_extra_trees"
BOOTSTRAP_METRICS = (
    "domain_macro_calibration_intercept",
    "domain_macro_calibration_slope",
    "domain_macro_r2",
    "pooled_calibration_intercept",
    "pooled_calibration_slope",
    "pooled_r2",
)
PROTOCOL_SEED_OFFSETS = {
    "source_ood": 31_000_000,
    "target_ood": 32_000_000,
}
OUTPUT_FILENAMES = (
    "canonical_smiles_aggregated_predictions.csv",
    "canonical_smiles_domain_metrics.csv",
    "canonical_smiles_aggregate_metrics.csv",
    "lodo_domain_influence.csv",
    "calibration_scaffold_bootstrap.csv",
    "results_brief_zh.md",
    "qa_summary.json",
    "artifact_sha256.csv",
    "run_manifest.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-dir",
        type=Path,
        default=DEFAULT_FORMAL_DIR,
        help="Completed formal OOD result directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="New directory for post-hoc outputs.",
    )
    parser.add_argument(
        "--protocol-note",
        type=Path,
        default=DEFAULT_PROTOCOL_NOTE,
        help="Post-hoc protocol/note bound into the output manifest.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=10_000,
    )
    parser.add_argument("--seed", type=int, default=260_531)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace only the known post-hoc output files.",
    )
    args = parser.parse_args()
    if args.bootstrap_replicates <= 0:
        parser.error("--bootstrap-replicates must be > 0")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    atomic_write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        path,
    )


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    os.replace(temporary, path)


def snapshot_directory(path: Path) -> dict[str, dict[str, Any]]:
    return {
        item.name: {
            "sha256": sha256_file(item),
            "size_bytes": int(item.stat().st_size),
        }
        for item in sorted(path.iterdir())
        if item.is_file()
    }


def file_inventory_entry(path: Path) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "filename": path.name,
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
        "kind": path.suffix.lower().lstrip(".") or "file",
    }
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        entry.update(
            {
                "n_rows": int(len(frame)),
                "n_columns": int(len(frame.columns)),
                "columns": list(frame.columns),
            }
        )
    return entry


def prepare_output_dir(path: Path, formal_dir: Path, overwrite: bool) -> None:
    resolved_output = path.resolve()
    resolved_formal = formal_dir.resolve()
    if (
        resolved_output == resolved_formal
        or resolved_formal in resolved_output.parents
    ):
        raise RuntimeError(
            "Post-hoc output directory must not be inside the formal directory"
        )
    path.mkdir(parents=True, exist_ok=True)
    unknown = [
        item.name
        for item in path.iterdir()
        if item.is_file() and item.name not in OUTPUT_FILENAMES
    ]
    if unknown:
        raise RuntimeError(
            f"Output directory contains unknown files: {sorted(unknown)}"
        )
    existing = [
        filename for filename in OUTPUT_FILENAMES if (path / filename).exists()
    ]
    if existing and not overwrite:
        raise RuntimeError(
            "Known post-hoc outputs already exist; use --overwrite to "
            f"replace them atomically: {existing}"
        )


def verify_parent_inventory(
    formal_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    inventory = manifest.get("artifact_inventory")
    if not isinstance(inventory, dict) or not inventory:
        raise RuntimeError("Formal manifest has no artifact_inventory")
    for filename, expected in inventory.items():
        path = formal_dir / filename
        if not path.is_file():
            raise RuntimeError(f"Formal artifact is missing: {filename}")
        observed_hash = sha256_file(path)
        if observed_hash != expected.get("sha256"):
            raise RuntimeError(
                f"Formal artifact hash mismatch for {filename}: "
                f"{observed_hash} != {expected.get('sha256')}"
            )
        if int(path.stat().st_size) != int(expected.get("size_bytes", -1)):
            raise RuntimeError(
                f"Formal artifact byte-size mismatch for {filename}"
            )
    return snapshot_directory(formal_dir)


def load_verified_formal_inputs(
    formal_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest_path = formal_dir / "ood_run_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Formal OOD manifest is missing")
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != EXPECTED_PARENT_MANIFEST_SHA256:
        raise RuntimeError(
            "Formal manifest is not the bound completed manifest: "
            f"{manifest_hash}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_manifest_state = {
        "status": "complete",
        "analysis_mode": "confirmatory",
        "frozen_protocol_match": True,
    }
    for key, expected in required_manifest_state.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"Formal manifest state mismatch for {key}: "
                f"{manifest.get(key)!r}"
            )
    if (
        manifest.get("scientific_configuration_sha256")
        != EXPECTED_PARENT_CONFIGURATION_SHA256
    ):
        raise RuntimeError("Formal scientific configuration digest changed")

    parent_snapshot = verify_parent_inventory(formal_dir, manifest)
    prediction_path = formal_dir / "ood_predictions.csv"
    prediction_hash = sha256_file(prediction_path)
    if prediction_hash != EXPECTED_PARENT_PREDICTIONS_SHA256:
        raise RuntimeError("Formal OOD prediction hash changed")
    prediction_entry = manifest["artifact_inventory"].get(
        "ood_predictions.csv"
    )
    if prediction_entry is None:
        raise RuntimeError("Prediction artifact is absent from formal inventory")

    predictions = pd.read_csv(prediction_path)
    if len(predictions) != int(prediction_entry["n_rows"]):
        raise RuntimeError("Formal prediction row count changed")
    if list(predictions.columns) != list(prediction_entry["columns"]):
        raise RuntimeError("Formal prediction schema changed")

    aggregate_path = formal_dir / "ood_aggregate_metrics.csv"
    aggregate_metrics = pd.read_csv(aggregate_path)
    input_identity = {
        "formal_manifest_path": str(manifest_path.resolve()),
        "formal_manifest_sha256": manifest_hash,
        "formal_predictions_path": str(prediction_path.resolve()),
        "formal_predictions_sha256": prediction_hash,
        "formal_scientific_configuration_sha256": manifest[
            "scientific_configuration_sha256"
        ],
    }
    return manifest, predictions, aggregate_metrics, {
        "snapshot": parent_snapshot,
        "identity": input_identity,
    }


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if (
        len(y_true) < 3
        or not np.all(np.isfinite(y_true))
        or not np.all(np.isfinite(y_pred))
        or np.unique(y_true).size < 2
        or np.unique(y_pred).size < 2
    ):
        return float("nan")
    return float(spearmanr(y_true, y_pred).statistic)


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if (
        len(y_true) < 3
        or not np.all(np.isfinite(y_true))
        or not np.all(np.isfinite(y_pred))
        or np.unique(y_true).size < 2
        or np.unique(y_pred).size < 2
    ):
        return float("nan")
    return float(pearsonr(y_true, y_pred).statistic)


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    if (
        len(y_true) == 0
        or not np.all(np.isfinite(y_true))
        or not np.all(np.isfinite(y_pred))
    ):
        return {
            key: float("nan")
            for key in (
                "spearman",
                "pearson",
                "rmse",
                "mae",
                "r2",
                "calibration_intercept",
                "calibration_slope",
            )
        }
    if len(y_true) >= 2 and np.unique(y_pred).size >= 2:
        design = np.column_stack(
            [np.ones(len(y_pred), dtype=np.float64), y_pred]
        )
        intercept, slope = np.linalg.lstsq(
            design,
            y_true,
            rcond=None,
        )[0]
    else:
        intercept, slope = float("nan"), float("nan")
    return {
        "spearman": safe_spearman(y_true, y_pred),
        "pearson": safe_pearson(y_true, y_pred),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": (
            float(r2_score(y_true, y_pred))
            if len(y_true) >= 2
            else float("nan")
        ),
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
    }


def validate_predictions(
    predictions: pd.DataFrame,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    required_columns = {
        "protocol",
        "heldout_group",
        "row_index",
        "qc_id",
        "model_id",
        "y_true",
        "y_pred",
        "canonical_smiles",
        "scaffold_id",
    }
    missing = required_columns - set(predictions.columns)
    if missing:
        raise RuntimeError(f"Prediction columns missing: {sorted(missing)}")
    if predictions[
        ["protocol", "heldout_group", "row_index", "model_id"]
    ].duplicated().any():
        raise RuntimeError("Prediction primary keys are duplicated")
    if predictions[
        ["protocol", "heldout_group", "canonical_smiles", "scaffold_id"]
    ].isna().any().any():
        raise RuntimeError("Prediction identity fields contain missing values")
    if not np.all(
        np.isfinite(
            predictions[["y_true", "y_pred"]].to_numpy(dtype=np.float64)
        )
    ):
        raise RuntimeError("Formal predictions contain non-finite values")

    configuration = manifest["scientific_configuration"]
    expected_models = tuple(configuration["model_order"])
    if set(predictions["model_id"].astype(str)) != set(expected_models):
        raise RuntimeError("Formal prediction model set changed")
    expected_groups = configuration["groups_by_protocol"]
    for protocol in PROTOCOL_ORDER:
        protocol_df = predictions[predictions["protocol"] == protocol]
        observed_groups = set(protocol_df["heldout_group"].astype(str))
        if observed_groups != set(expected_groups[protocol]):
            raise RuntimeError(f"Held-out groups changed for {protocol}")
        reference = (
            protocol_df[
                protocol_df["model_id"] == expected_models[0]
            ]
            .sort_values(["heldout_group", "row_index"])
            .reset_index(drop=True)
        )
        alignment_columns = (
            "heldout_group",
            "row_index",
            "qc_id",
            "y_true",
            "canonical_smiles",
            "scaffold_id",
        )
        for model_id in expected_models[1:]:
            model_frame = (
                protocol_df[protocol_df["model_id"] == model_id]
                .sort_values(["heldout_group", "row_index"])
                .reset_index(drop=True)
            )
            if len(model_frame) != len(reference):
                raise RuntimeError(
                    f"Model row count is not aligned: {protocol}/{model_id}"
                )
            for column in alignment_columns:
                left = reference[column].to_numpy()
                right = model_frame[column].to_numpy()
                equal = (
                    np.array_equal(
                        left.astype(np.float64),
                        right.astype(np.float64),
                    )
                    if column == "y_true"
                    else np.array_equal(left, right)
                )
                if not equal:
                    raise RuntimeError(
                        "Model-aligned formal rows changed for "
                        f"{protocol}/{model_id}/{column}"
                    )
    return {
        "prediction_primary_keys_unique": True,
        "prediction_values_finite": True,
        "formal_models_row_aligned": True,
        "formal_model_count": int(len(expected_models)),
        "formal_prediction_rows": int(len(predictions)),
    }


def build_canonical_aggregates(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    group_columns = [
        "protocol",
        "heldout_group",
        "model_id",
        "canonical_smiles",
    ]
    scaffold_counts = (
        predictions.groupby(group_columns, sort=False, dropna=False)[
            "scaffold_id"
        ].nunique()
    )
    if int(scaffold_counts.max()) != 1:
        raise RuntimeError(
            "A canonical SMILES maps to multiple scaffolds within a domain/model"
        )
    aggregated = (
        predictions.groupby(group_columns, sort=False, dropna=False)
        .agg(
            y_true=("y_true", "median"),
            y_pred=("y_pred", "median"),
            scaffold_id=("scaffold_id", "first"),
            n_source_rows=("row_index", "size"),
            n_unique_y_true=("y_true", "nunique"),
            row_index_min=("row_index", "min"),
        )
        .reset_index()
    )
    aggregated.insert(0, "analysis_mode", ANALYSIS_MODE)
    aggregated.insert(
        1,
        "aggregation",
        (
            "median y_true and median y_pred within "
            "protocol+heldout_group+model_id+canonical_smiles"
        ),
    )
    return aggregated.sort_values(
        ["protocol", "heldout_group", "model_id", "row_index_min"],
        kind="stable",
    ).reset_index(drop=True)


def compute_domain_metrics(
    units: pd.DataFrame,
    analysis_scale: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (protocol, heldout_group, model_id), frame in units.groupby(
        ["protocol", "heldout_group", "model_id"],
        sort=False,
    ):
        metrics = regression_metrics(
            frame["y_true"].to_numpy(dtype=np.float64),
            frame["y_pred"].to_numpy(dtype=np.float64),
        )
        rows.append(
            {
                "analysis_mode": ANALYSIS_MODE,
                "analysis_scale": analysis_scale,
                "protocol": protocol,
                "heldout_group": heldout_group,
                "model_id": model_id,
                "n_analysis_units": int(len(frame)),
                "n_source_rows": int(
                    frame["n_source_rows"].sum()
                    if "n_source_rows" in frame
                    else len(frame)
                ),
                "n_unique_smiles": int(frame["canonical_smiles"].nunique()),
                "n_multirow_smiles": int(
                    np.sum(frame["n_source_rows"] > 1)
                    if "n_source_rows" in frame
                    else 0
                ),
                "n_scaffolds": int(frame["scaffold_id"].nunique()),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["protocol", "heldout_group", "model_id"],
        kind="stable",
    ).reset_index(drop=True)


def compute_aggregate_metrics(
    units: pd.DataFrame,
    domain_metrics: pd.DataFrame,
    groups_by_protocol: dict[str, list[str]],
    analysis_scale: str,
) -> pd.DataFrame:
    metric_names = (
        "spearman",
        "pearson",
        "rmse",
        "mae",
        "r2",
        "calibration_intercept",
        "calibration_slope",
    )
    rows: list[dict[str, Any]] = []
    for (protocol, model_id), frame in units.groupby(
        ["protocol", "model_id"],
        sort=False,
    ):
        groups = groups_by_protocol[str(protocol)]
        local_domain = domain_metrics[
            (domain_metrics["protocol"] == protocol)
            & (domain_metrics["model_id"] == model_id)
        ].set_index("heldout_group")
        if set(local_domain.index.astype(str)) != set(groups):
            raise RuntimeError(
                f"Canonical domain metrics are incomplete: {protocol}/{model_id}"
            )
        pooled = regression_metrics(
            frame["y_true"].to_numpy(dtype=np.float64),
            frame["y_pred"].to_numpy(dtype=np.float64),
        )
        row: dict[str, Any] = {
            "analysis_mode": ANALYSIS_MODE,
            "analysis_scale": analysis_scale,
            "protocol": protocol,
            "model_id": model_id,
            "n_domains": int(len(groups)),
            "n_domain_smiles_units": int(len(frame)),
            "n_unique_smiles_across_domains": int(
                frame["canonical_smiles"].nunique()
            ),
            "n_source_rows": int(frame["n_source_rows"].sum()),
            "n_scaffolds": int(frame["scaffold_id"].nunique()),
        }
        for metric in metric_names:
            values = np.asarray(
                [
                    float(local_domain.loc[group, metric])
                    for group in groups
                ],
                dtype=np.float64,
            )
            finite = values[np.isfinite(values)]
            row[f"domain_macro_{metric}"] = (
                float(np.mean(finite))
                if len(finite) == len(groups)
                else float("nan")
            )
            row[f"domain_macro_{metric}_finite_domains"] = int(len(finite))
            row[f"pooled_{metric}"] = float(pooled[metric])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["protocol", "model_id"],
        kind="stable",
    ).reset_index(drop=True)


def build_lodo_influence(
    predictions: pd.DataFrame,
    canonical_domain_metrics: pd.DataFrame,
    groups_by_protocol: dict[str, list[str]],
) -> pd.DataFrame:
    raw_selected = predictions[
        predictions["model_id"].isin(CALIBRATION_MODELS)
    ].copy()
    raw_domain_metrics = compute_domain_metrics(
        raw_selected,
        analysis_scale="formal_row_level",
    )
    canonical_selected = canonical_domain_metrics[
        canonical_domain_metrics["model_id"].isin(CALIBRATION_MODELS)
    ].copy()
    scale_frames = {
        "formal_row_level": raw_domain_metrics,
        "canonical_smiles_median_within_domain": canonical_selected,
    }
    rows: list[dict[str, Any]] = []
    for analysis_scale, metrics_frame in scale_frames.items():
        for protocol in PROTOCOL_ORDER:
            groups = groups_by_protocol[protocol]
            protocol_metrics = metrics_frame[
                metrics_frame["protocol"] == protocol
            ]
            by_model = {
                model_id: protocol_metrics[
                    protocol_metrics["model_id"] == model_id
                ].set_index("heldout_group")
                for model_id in CALIBRATION_MODELS
            }
            for model_id, frame in by_model.items():
                if set(frame.index.astype(str)) != set(groups):
                    raise RuntimeError(
                        f"LODO inputs incomplete: {analysis_scale}/"
                        f"{protocol}/{model_id}"
                    )

            full_all_spearman = float(
                np.mean(
                    [
                        by_model[FULL_MODEL].loc[group, "spearman"]
                        for group in groups
                    ]
                )
            )
            chemistry_all_spearman = float(
                np.mean(
                    [
                        by_model[CHEMISTRY_MODEL].loc[group, "spearman"]
                        for group in groups
                    ]
                )
            )
            full_all_rmse = float(
                np.mean(
                    [
                        by_model[FULL_MODEL].loc[group, "rmse"]
                        for group in groups
                    ]
                )
            )
            chemistry_all_rmse = float(
                np.mean(
                    [
                        by_model[CHEMISTRY_MODEL].loc[group, "rmse"]
                        for group in groups
                    ]
                )
            )
            delta_all_spearman = (
                full_all_spearman - chemistry_all_spearman
            )
            delta_all_rmse = chemistry_all_rmse - full_all_rmse

            for omitted_domain in groups:
                retained = [
                    group for group in groups if group != omitted_domain
                ]
                full_lodo_spearman = float(
                    np.mean(
                        [
                            by_model[FULL_MODEL].loc[group, "spearman"]
                            for group in retained
                        ]
                    )
                )
                chemistry_lodo_spearman = float(
                    np.mean(
                        [
                            by_model[CHEMISTRY_MODEL].loc[group, "spearman"]
                            for group in retained
                        ]
                    )
                )
                full_lodo_rmse = float(
                    np.mean(
                        [
                            by_model[FULL_MODEL].loc[group, "rmse"]
                            for group in retained
                        ]
                    )
                )
                chemistry_lodo_rmse = float(
                    np.mean(
                        [
                            by_model[CHEMISTRY_MODEL].loc[group, "rmse"]
                            for group in retained
                        ]
                    )
                )
                delta_lodo_spearman = (
                    full_lodo_spearman - chemistry_lodo_spearman
                )
                delta_lodo_rmse = chemistry_lodo_rmse - full_lodo_rmse
                rows.append(
                    {
                        "analysis_mode": ANALYSIS_MODE,
                        "analysis_scale": analysis_scale,
                        "protocol": protocol,
                        "omitted_domain": omitted_domain,
                        "n_domains_total": int(len(groups)),
                        "n_domains_retained": int(len(retained)),
                        "full_omitted_domain_spearman": float(
                            by_model[FULL_MODEL].loc[
                                omitted_domain, "spearman"
                            ]
                        ),
                        "chemistry_omitted_domain_spearman": float(
                            by_model[CHEMISTRY_MODEL].loc[
                                omitted_domain, "spearman"
                            ]
                        ),
                        "omitted_domain_delta_spearman": float(
                            by_model[FULL_MODEL].loc[
                                omitted_domain, "spearman"
                            ]
                            - by_model[CHEMISTRY_MODEL].loc[
                                omitted_domain, "spearman"
                            ]
                        ),
                        "full_all_domain_macro_spearman": full_all_spearman,
                        "full_lodo_domain_macro_spearman": full_lodo_spearman,
                        "full_lodo_influence_spearman": (
                            full_lodo_spearman - full_all_spearman
                        ),
                        "chemistry_all_domain_macro_spearman": (
                            chemistry_all_spearman
                        ),
                        "chemistry_lodo_domain_macro_spearman": (
                            chemistry_lodo_spearman
                        ),
                        "chemistry_lodo_influence_spearman": (
                            chemistry_lodo_spearman
                            - chemistry_all_spearman
                        ),
                        "delta_all_domain_macro_spearman": (
                            delta_all_spearman
                        ),
                        "delta_lodo_domain_macro_spearman": (
                            delta_lodo_spearman
                        ),
                        "delta_lodo_influence_spearman": (
                            delta_lodo_spearman - delta_all_spearman
                        ),
                        "full_omitted_domain_rmse": float(
                            by_model[FULL_MODEL].loc[
                                omitted_domain, "rmse"
                            ]
                        ),
                        "chemistry_omitted_domain_rmse": float(
                            by_model[CHEMISTRY_MODEL].loc[
                                omitted_domain, "rmse"
                            ]
                        ),
                        "omitted_domain_delta_rmse": float(
                            by_model[CHEMISTRY_MODEL].loc[
                                omitted_domain, "rmse"
                            ]
                            - by_model[FULL_MODEL].loc[
                                omitted_domain, "rmse"
                            ]
                        ),
                        "full_all_domain_macro_rmse": full_all_rmse,
                        "full_lodo_domain_macro_rmse": full_lodo_rmse,
                        "full_lodo_influence_rmse": (
                            full_lodo_rmse - full_all_rmse
                        ),
                        "chemistry_all_domain_macro_rmse": (
                            chemistry_all_rmse
                        ),
                        "chemistry_lodo_domain_macro_rmse": (
                            chemistry_lodo_rmse
                        ),
                        "chemistry_lodo_influence_rmse": (
                            chemistry_lodo_rmse - chemistry_all_rmse
                        ),
                        "delta_all_domain_macro_rmse": delta_all_rmse,
                        "delta_lodo_domain_macro_rmse": delta_lodo_rmse,
                        "delta_lodo_influence_rmse": (
                            delta_lodo_rmse - delta_all_rmse
                        ),
                        "delta_spearman_definition": (
                            "full_context_extra_trees minus "
                            "chemistry_extra_trees"
                        ),
                        "delta_rmse_definition": (
                            "chemistry_extra_trees minus "
                            "full_context_extra_trees; positive favors full"
                        ),
                        "inference": (
                            "descriptive fixed-domain influence; "
                            "no confidence interval and no p-value"
                        ),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["analysis_scale", "protocol", "omitted_domain"],
        kind="stable",
    ).reset_index(drop=True)


def calibration_bundle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    domains: np.ndarray,
    positions: np.ndarray,
    domain_order: list[str],
) -> dict[str, float]:
    selected_y = y_true[positions]
    selected_pred = y_pred[positions]
    selected_domains = domains[positions]
    per_domain = {
        metric: [] for metric in (
            "calibration_intercept",
            "calibration_slope",
            "r2",
        )
    }
    for domain in domain_order:
        local = np.where(selected_domains == domain)[0]
        metrics = regression_metrics(
            selected_y[local],
            selected_pred[local],
        )
        for metric in per_domain:
            per_domain[metric].append(float(metrics[metric]))
    pooled = regression_metrics(selected_y, selected_pred)
    result: dict[str, float] = {}
    for metric, values in per_domain.items():
        array = np.asarray(values, dtype=np.float64)
        result[f"domain_macro_{metric}"] = (
            float(np.mean(array))
            if np.all(np.isfinite(array))
            else float("nan")
        )
        result[f"pooled_{metric}"] = float(pooled[metric])
    return result


def finite_interval(
    values: Iterable[float],
) -> tuple[float, float, float, int]:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return (
        float(np.mean(finite)),
        float(np.percentile(finite, 2.5)),
        float(np.percentile(finite, 97.5)),
        int(len(finite)),
    )


def bootstrap_calibration(
    predictions: pd.DataFrame,
    groups_by_protocol: dict[str, list[str]],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for protocol in PROTOCOL_ORDER:
        protocol_df = predictions[
            (predictions["protocol"] == protocol)
            & predictions["model_id"].isin(CALIBRATION_MODELS)
        ]
        domain_order = groups_by_protocol[protocol]
        reference = (
            protocol_df[protocol_df["model_id"] == FULL_MODEL]
            .sort_values("row_index")
            .reset_index(drop=True)
        )
        row_indices = reference["row_index"].to_numpy(dtype=np.int64)
        y_true = reference["y_true"].to_numpy(dtype=np.float64)
        domains = reference["heldout_group"].astype(str).to_numpy()
        clusters = reference["scaffold_id"].astype(str).to_numpy()
        unique_clusters = np.unique(clusters)
        cluster_positions = {
            cluster: np.where(clusters == cluster)[0]
            for cluster in unique_clusters
        }
        predictions_by_model: dict[str, np.ndarray] = {}
        observed: dict[str, dict[str, float]] = {}
        for model_id in CALIBRATION_MODELS:
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
                    f"Calibration rows are not paired: {protocol}/{model_id}"
                )
            if not np.array_equal(
                frame["y_true"].to_numpy(dtype=np.float64),
                y_true,
            ):
                raise RuntimeError(
                    f"Calibration outcomes are not paired: {protocol}/{model_id}"
                )
            pred = frame["y_pred"].to_numpy(dtype=np.float64)
            predictions_by_model[model_id] = pred
            observed[model_id] = calibration_bundle(
                y_true,
                pred,
                domains,
                np.arange(len(reference), dtype=np.int64),
                domain_order,
            )

        distributions = {
            model_id: {
                metric: np.full(n_bootstrap, np.nan, dtype=np.float64)
                for metric in BOOTSTRAP_METRICS
            }
            for model_id in CALIBRATION_MODELS
        }
        rng = np.random.default_rng(
            seed + PROTOCOL_SEED_OFFSETS[protocol]
        )
        progress_interval = max(1, n_bootstrap // 10)
        for bootstrap_index in range(n_bootstrap):
            sampled_clusters = rng.choice(
                unique_clusters,
                size=len(unique_clusters),
                replace=True,
            )
            sampled_positions = np.concatenate(
                [
                    cluster_positions[cluster]
                    for cluster in sampled_clusters
                ]
            )
            for model_id in CALIBRATION_MODELS:
                metrics = calibration_bundle(
                    y_true,
                    predictions_by_model[model_id],
                    domains,
                    sampled_positions,
                    domain_order,
                )
                for metric in BOOTSTRAP_METRICS:
                    distributions[model_id][metric][
                        bootstrap_index
                    ] = metrics[metric]
            if (bootstrap_index + 1) % progress_interval == 0:
                print(
                    f"[BOOT] {protocol}: "
                    f"{bootstrap_index + 1}/{n_bootstrap}",
                    flush=True,
                )

        for model_id in CALIBRATION_MODELS:
            for metric in BOOTSTRAP_METRICS:
                mean, low, high, n_valid = finite_interval(
                    distributions[model_id][metric]
                )
                if metric.startswith("domain_macro_"):
                    metric_scope = "domain_macro"
                    metric_name = metric.removeprefix("domain_macro_")
                else:
                    metric_scope = "pooled"
                    metric_name = metric.removeprefix("pooled_")
                rows.append(
                    {
                        "analysis_mode": ANALYSIS_MODE,
                        "analysis_scale": "formal_row_level",
                        "record_type": "model",
                        "protocol": protocol,
                        "model_id": model_id,
                        "contrast_id": "",
                        "first_model": "",
                        "comparator_model": "",
                        "metric_scope": metric_scope,
                        "metric": metric_name,
                        "estimate_definition": "model estimate",
                        "cluster_unit": "global scaffold_id",
                        "n_clusters": int(len(unique_clusters)),
                        "n_domains": int(len(domain_order)),
                        "n_bootstrap": int(n_bootstrap),
                        "ci_method": "percentile_2.5_97.5",
                        "observed": float(observed[model_id][metric]),
                        "bootstrap_mean": mean,
                        "ci_low": low,
                        "ci_high": high,
                        "n_valid": n_valid,
                        "domains_fixed": True,
                    }
                )

        for metric in BOOTSTRAP_METRICS:
            delta_values = (
                distributions[FULL_MODEL][metric]
                - distributions[CHEMISTRY_MODEL][metric]
            )
            mean, low, high, n_valid = finite_interval(delta_values)
            if metric.startswith("domain_macro_"):
                metric_scope = "domain_macro"
                metric_name = metric.removeprefix("domain_macro_")
            else:
                metric_scope = "pooled"
                metric_name = metric.removeprefix("pooled_")
            rows.append(
                {
                    "analysis_mode": ANALYSIS_MODE,
                    "analysis_scale": "formal_row_level",
                    "record_type": "contrast",
                    "protocol": protocol,
                    "model_id": "",
                    "contrast_id": "full_vs_chemistry_extra_trees",
                    "first_model": FULL_MODEL,
                    "comparator_model": CHEMISTRY_MODEL,
                    "metric_scope": metric_scope,
                    "metric": metric_name,
                    "estimate_definition": (
                        "first minus comparator; positive R2 delta favors "
                        "first, but calibration coefficient deltas have no "
                        "generic superiority direction"
                    ),
                    "cluster_unit": "global scaffold_id",
                    "n_clusters": int(len(unique_clusters)),
                    "n_domains": int(len(domain_order)),
                    "n_bootstrap": int(n_bootstrap),
                    "ci_method": "percentile_2.5_97.5",
                    "observed": float(
                        observed[FULL_MODEL][metric]
                        - observed[CHEMISTRY_MODEL][metric]
                    ),
                    "bootstrap_mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "n_valid": n_valid,
                    "domains_fixed": True,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["protocol", "record_type", "model_id", "metric_scope", "metric"],
        kind="stable",
    ).reset_index(drop=True)


def validate_against_formal_aggregate(
    predictions: pd.DataFrame,
    aggregate_metrics: pd.DataFrame,
    groups_by_protocol: dict[str, list[str]],
) -> float:
    selected = predictions[
        predictions["model_id"].isin(CALIBRATION_MODELS)
    ]
    raw_domain = compute_domain_metrics(selected, "formal_row_level")
    max_abs_difference = 0.0
    for protocol in PROTOCOL_ORDER:
        groups = groups_by_protocol[protocol]
        for model_id in CALIBRATION_MODELS:
            local = raw_domain[
                (raw_domain["protocol"] == protocol)
                & (raw_domain["model_id"] == model_id)
            ].set_index("heldout_group")
            formal_row = aggregate_metrics[
                (aggregate_metrics["protocol"] == protocol)
                & (aggregate_metrics["model_id"] == model_id)
            ]
            if len(formal_row) != 1:
                raise RuntimeError(
                    f"Formal aggregate row missing: {protocol}/{model_id}"
                )
            formal = formal_row.iloc[0]
            model_frame = selected[
                (selected["protocol"] == protocol)
                & (selected["model_id"] == model_id)
            ]
            pooled = regression_metrics(
                model_frame["y_true"].to_numpy(dtype=np.float64),
                model_frame["y_pred"].to_numpy(dtype=np.float64),
            )
            for metric in (
                "calibration_intercept",
                "calibration_slope",
                "r2",
            ):
                macro = float(
                    np.mean(
                        [float(local.loc[group, metric]) for group in groups]
                    )
                )
                comparisons = (
                    (
                        macro,
                        float(formal[f"domain_macro_{metric}"]),
                    ),
                    (
                        float(pooled[metric]),
                        float(formal[f"pooled_{metric}"]),
                    ),
                )
                for observed, expected in comparisons:
                    difference = abs(observed - expected)
                    max_abs_difference = max(
                        max_abs_difference,
                        difference,
                    )
                    if difference > 1e-10:
                        raise RuntimeError(
                            "Recomputed formal calibration metric changed: "
                            f"{protocol}/{model_id}/{metric}/"
                            f"{observed} != {expected}"
                        )
    return max_abs_difference


def fmt(value: float, digits: int = 4) -> str:
    if not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}".replace("-", "−")


def metric_lookup(
    frame: pd.DataFrame,
    **conditions: str,
) -> pd.Series:
    selected = frame
    for column, value in conditions.items():
        selected = selected[selected[column].astype(str) == str(value)]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one metric row for {conditions}, found {len(selected)}"
        )
    return selected.iloc[0]


def build_brief(
    canonical_aggregate: pd.DataFrame,
    lodo: pd.DataFrame,
    calibration_bootstrap: pd.DataFrame,
    input_identity: dict[str, Any],
    n_bootstrap: int,
) -> str:
    canonical_lines: list[str] = []
    calibration_lines: list[str] = []
    influence_lines: list[str] = []
    for protocol in PROTOCOL_ORDER:
        full = metric_lookup(
            canonical_aggregate,
            protocol=protocol,
            model_id=FULL_MODEL,
        )
        chemistry = metric_lookup(
            canonical_aggregate,
            protocol=protocol,
            model_id=CHEMISTRY_MODEL,
        )
        canonical_lines.append(
            (
                f"| {protocol} | full-context ET | "
                f"{fmt(full['domain_macro_spearman'])} | "
                f"{fmt(full['domain_macro_rmse'])} | "
                f"{fmt(full['pooled_spearman'])} | "
                f"{fmt(full['pooled_rmse'])} |"
            )
        )
        canonical_lines.append(
            (
                f"| {protocol} | chemistry-only ET | "
                f"{fmt(chemistry['domain_macro_spearman'])} | "
                f"{fmt(chemistry['domain_macro_rmse'])} | "
                f"{fmt(chemistry['pooled_spearman'])} | "
                f"{fmt(chemistry['pooled_rmse'])} |"
            )
        )

        raw_lodo = lodo[
            (lodo["analysis_scale"] == "formal_row_level")
            & (lodo["protocol"] == protocol)
        ].copy()
        most_influential = raw_lodo.iloc[
            np.argmax(
                np.abs(
                    raw_lodo[
                        "delta_lodo_influence_spearman"
                    ].to_numpy(dtype=np.float64)
                )
            )
        ]
        influence_lines.append(
            (
                f"- `{protocol}`：省略 `{most_influential['omitted_domain']}` "
                f"时 full−chemistry 的 macro Spearman 差值由 "
                f"{fmt(most_influential['delta_all_domain_macro_spearman'])} "
                f"变为 "
                f"{fmt(most_influential['delta_lodo_domain_macro_spearman'])}"
                f"（变化 "
                f"{fmt(most_influential['delta_lodo_influence_spearman'])}）。"
            )
        )

        for model_id, label in (
            (FULL_MODEL, "full-context ET"),
            (CHEMISTRY_MODEL, "chemistry-only ET"),
        ):
            intercept = metric_lookup(
                calibration_bootstrap,
                protocol=protocol,
                record_type="model",
                model_id=model_id,
                metric_scope="pooled",
                metric="calibration_intercept",
            )
            slope = metric_lookup(
                calibration_bootstrap,
                protocol=protocol,
                record_type="model",
                model_id=model_id,
                metric_scope="pooled",
                metric="calibration_slope",
            )
            r2 = metric_lookup(
                calibration_bootstrap,
                protocol=protocol,
                record_type="model",
                model_id=model_id,
                metric_scope="pooled",
                metric="r2",
            )
            calibration_lines.append(
                (
                    f"| {protocol} | {label} | "
                    f"{fmt(intercept['observed'])} "
                    f"[{fmt(intercept['ci_low'])}, "
                    f"{fmt(intercept['ci_high'])}] | "
                    f"{fmt(slope['observed'])} "
                    f"[{fmt(slope['ci_low'])}, "
                    f"{fmt(slope['ci_high'])}] | "
                    f"{fmt(r2['observed'])} "
                    f"[{fmt(r2['ci_low'])}, "
                    f"{fmt(r2['ci_high'])}] |"
                )
            )

    return f"""# Formal OOD 无重训 post-hoc 诊断简报

## 身份与边界

- 分析模式：`{ANALYSIS_MODE}`；没有训练、调参、模型选择或标签置换。
- 输入为 hash-verified formal OOD predictions：
  `{input_identity['formal_predictions_sha256']}`。
- 结果条件化于冻结的 4 个 source 域或 8 个 target 域；{n_bootstrap:,} 次重采样不是独立样本量。
- 所有区间均为全局 scaffold-cluster percentile bootstrap 95% CI；不报告任何 p 值。

## Canonical-SMILES 中位数聚合敏感性

先在 `protocol + heldout_group + model + canonical_smiles` 内分别取
`y_true` 与 `y_pred` 中位数，再计算逐域、domain-macro 和 pooled 指标。
同一 SMILES 在不同 held-out domain 中仍是不同分析单位。

| Protocol | Model | Domain-macro Spearman | Domain-macro RMSE | Pooled Spearman | Pooled RMSE |
|---|---|---:|---:|---:|---:|
{chr(10).join(canonical_lines)}

这些点估计仅检查重复记录权重的影响，没有为 canonical 聚合指标计算新的 CI。

## Leave-one-domain-out 影响

LODO 只是固定域集合上的描述性影响分析，不是外部复制、jackknife
总体推断或显著性检验。

{chr(10).join(influence_lines)}

完整的逐域省略结果同时提供 formal-row 与 canonical-SMILES 两种尺度。

## 原始行尺度的 pooled 校准不确定性

| Protocol | Model | Intercept (95% CI) | Slope (95% CI) | R² (95% CI) |
|---|---|---:|---:|---:|
{chr(10).join(calibration_lines)}

理想校准的截距/斜率为 0/1。截距或斜率的模型间差值没有通用的“越大越好”
方向；配对 contrast 仅作为描述。R² 的正差值才表示 full 高于 chemistry。

## 使用限制

1. 本分析不能把原 confirmatory 结果升级为 prospective/external 证据。
2. Canonical-SMILES 聚合可能合并同一分子在同一 held-out domain 内的不同实验上下文，因此只用于重复权重敏感性。
3. LODO 的 domain 数仅为 4 或 8，不能解释为从数据库或靶点总体随机抽样。
4. Bootstrap 只传播冻结测试预测在 scaffold cluster 重采样下的不确定性，不传播训练数据、域选择或模型选择不确定性。
"""


def build_artifact_hash_table(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        entry = file_inventory_entry(path)
        rows.append(
            {
                "filename": entry["filename"],
                "sha256": entry["sha256"],
                "size_bytes": entry["size_bytes"],
                "kind": entry["kind"],
                "n_rows": entry.get("n_rows"),
                "n_columns": entry.get("n_columns"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    started = time.time()
    formal_dir = args.formal_dir.resolve()
    output_dir = args.output_dir.resolve()
    protocol_note = args.protocol_note.resolve()
    if not protocol_note.is_file():
        raise RuntimeError(f"Post-hoc protocol note is missing: {protocol_note}")
    prepare_output_dir(output_dir, formal_dir, args.overwrite)

    (
        formal_manifest,
        predictions,
        formal_aggregate,
        parent_info,
    ) = load_verified_formal_inputs(formal_dir)
    qa_checks = validate_predictions(predictions, formal_manifest)
    groups_by_protocol = formal_manifest["scientific_configuration"][
        "groups_by_protocol"
    ]
    max_formal_recompute_difference = validate_against_formal_aggregate(
        predictions,
        formal_aggregate,
        groups_by_protocol,
    )

    canonical_predictions = build_canonical_aggregates(predictions)
    canonical_domain_metrics = compute_domain_metrics(
        canonical_predictions,
        "canonical_smiles_median_within_domain",
    )
    canonical_aggregate_metrics = compute_aggregate_metrics(
        canonical_predictions,
        canonical_domain_metrics,
        groups_by_protocol,
        "canonical_smiles_median_within_domain",
    )
    lodo = build_lodo_influence(
        predictions,
        canonical_domain_metrics,
        groups_by_protocol,
    )
    calibration_bootstrap = bootstrap_calibration(
        predictions,
        groups_by_protocol,
        args.bootstrap_replicates,
        args.seed,
    )

    csv_outputs = {
        "canonical_smiles_aggregated_predictions.csv": canonical_predictions,
        "canonical_smiles_domain_metrics.csv": canonical_domain_metrics,
        "canonical_smiles_aggregate_metrics.csv": (
            canonical_aggregate_metrics
        ),
        "lodo_domain_influence.csv": lodo,
        "calibration_scaffold_bootstrap.csv": calibration_bootstrap,
    }
    for filename, frame in csv_outputs.items():
        atomic_write_csv(frame, output_dir / filename)

    brief = build_brief(
        canonical_aggregate_metrics,
        lodo,
        calibration_bootstrap,
        parent_info["identity"],
        args.bootstrap_replicates,
    )
    brief_path = output_dir / "results_brief_zh.md"
    atomic_write_text(brief, brief_path)

    parent_snapshot_after_outputs = snapshot_directory(formal_dir)
    parent_unchanged = (
        parent_snapshot_after_outputs == parent_info["snapshot"]
    )
    if not parent_unchanged:
        raise RuntimeError("Formal OOD directory changed during post-hoc run")

    no_p_value_columns = all(
        not any(
            "p_value" in str(column).lower()
            or str(column).lower() == "p"
            for column in frame.columns
        )
        for frame in csv_outputs.values()
    )
    if not no_p_value_columns:
        raise RuntimeError("A post-hoc output unexpectedly contains p-values")
    n_valid_min = int(calibration_bootstrap["n_valid"].min())
    if n_valid_min <= 0:
        raise RuntimeError("Calibration bootstrap produced no valid estimates")

    qa_payload: dict[str, Any] = {
        "status": "pass",
        "analysis_mode": ANALYSIS_MODE,
        "protocol_version": PROTOCOL_VERSION,
        "checks": {
            **qa_checks,
            "formal_manifest_hash_verified": True,
            "formal_configuration_hash_verified": True,
            "formal_artifact_inventory_verified": True,
            "formal_predictions_hash_verified": True,
            "formal_metrics_recomputed_from_predictions": True,
            "canonical_scaffold_mapping_unique_within_domain_model": True,
            "lodo_has_both_analysis_scales": (
                set(lodo["analysis_scale"].astype(str))
                == {
                    "formal_row_level",
                    "canonical_smiles_median_within_domain",
                }
            ),
            "calibration_bootstrap_is_paired_by_draw": True,
            "calibration_bootstrap_cluster_unit_is_global_scaffold": True,
            "fixed_domains_explicit": bool(
                calibration_bootstrap["domains_fixed"].all()
            ),
            "no_p_values_reported": bool(no_p_value_columns),
            "formal_directory_unchanged_after_derived_outputs": bool(
                parent_unchanged
            ),
        },
        "counts": {
            "formal_prediction_rows": int(len(predictions)),
            "canonical_aggregated_prediction_rows": int(
                len(canonical_predictions)
            ),
            "canonical_domain_metric_rows": int(
                len(canonical_domain_metrics)
            ),
            "canonical_aggregate_metric_rows": int(
                len(canonical_aggregate_metrics)
            ),
            "lodo_rows": int(len(lodo)),
            "calibration_bootstrap_summary_rows": int(
                len(calibration_bootstrap)
            ),
            "bootstrap_replicates": int(args.bootstrap_replicates),
            "bootstrap_n_valid_min": n_valid_min,
        },
        "numerical_checks": {
            "max_abs_difference_recomputed_vs_formal_calibration_and_r2": (
                float(max_formal_recompute_difference)
            ),
            "formal_recompute_tolerance": 1e-10,
        },
        "input_identity": parent_info["identity"],
        "boundaries": [
            "No model was trained, tuned, selected, or recalibrated.",
            "All analyses are post hoc and exploratory.",
            "Domains are fixed; LODO is descriptive influence analysis.",
            "Bootstrap resamples global scaffold clusters conditionally on "
            "the frozen predictions and domains.",
            "No p-value is calculated or reported.",
        ],
    }
    if not all(
        value is True
        for key, value in qa_payload["checks"].items()
        if isinstance(value, bool)
    ):
        raise RuntimeError(f"One or more QA checks failed: {qa_payload}")
    qa_path = output_dir / "qa_summary.json"
    atomic_write_json(qa_payload, qa_path)

    hash_source_paths = [
        output_dir / filename
        for filename in (
            "canonical_smiles_aggregated_predictions.csv",
            "canonical_smiles_domain_metrics.csv",
            "canonical_smiles_aggregate_metrics.csv",
            "lodo_domain_influence.csv",
            "calibration_scaffold_bootstrap.csv",
            "results_brief_zh.md",
            "qa_summary.json",
        )
    ]
    artifact_hash_table = build_artifact_hash_table(hash_source_paths)
    artifact_hash_path = output_dir / "artifact_sha256.csv"
    atomic_write_csv(artifact_hash_table, artifact_hash_path)

    inventory_paths = hash_source_paths + [artifact_hash_path]
    artifact_inventory = {
        path.name: file_inventory_entry(path)
        for path in inventory_paths
    }
    elapsed_seconds = float(time.time() - started)
    run_manifest: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "complete",
        "analysis_mode": ANALYSIS_MODE,
        "no_retraining": True,
        "post_hoc": True,
        "exploratory": True,
        "input_identity": parent_info["identity"],
        "parent_state_required": {
            "status": "complete",
            "analysis_mode": "confirmatory",
            "frozen_protocol_match": True,
        },
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "protocol_note": {
            "path": str(protocol_note),
            "sha256": sha256_file(protocol_note),
        },
        "configuration": {
            "bootstrap_replicates": int(args.bootstrap_replicates),
            "seed": int(args.seed),
            "protocols": list(PROTOCOL_ORDER),
            "groups_by_protocol": groups_by_protocol,
            "canonical_aggregation": (
                "median y_true and y_pred within protocol+heldout_group+"
                "model_id+canonical_smiles"
            ),
            "lodo_models": list(CALIBRATION_MODELS),
            "lodo_scales": [
                "formal_row_level",
                "canonical_smiles_median_within_domain",
            ],
            "calibration_models": list(CALIBRATION_MODELS),
            "calibration_bootstrap_cluster": "global scaffold_id",
            "calibration_bootstrap_ci": "percentile 2.5%, 97.5%",
            "calibration_contrast": "full minus chemistry",
        },
        "independent_unit_and_dependence": {
            "bootstrap_unit": "global Bemis-Murcko scaffold cluster",
            "n_clusters_by_protocol": {
                protocol: int(
                    predictions[
                        (predictions["protocol"] == protocol)
                        & (predictions["model_id"] == FULL_MODEL)
                    ]["scaffold_id"].nunique()
                )
                for protocol in PROTOCOL_ORDER
            },
            "domains_are_fixed": True,
            "bootstrap_replicates_are_not_independent_n": True,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "gpu_required": False,
        },
        "qa": {
            "path": qa_path.name,
            "status": qa_payload["status"],
            "formal_directory_unchanged": parent_unchanged,
        },
        "boundaries": qa_payload["boundaries"],
        "elapsed_seconds": elapsed_seconds,
        "artifacts": {
            "canonical_smiles_aggregated_predictions": (
                "canonical_smiles_aggregated_predictions.csv"
            ),
            "canonical_smiles_domain_metrics": (
                "canonical_smiles_domain_metrics.csv"
            ),
            "canonical_smiles_aggregate_metrics": (
                "canonical_smiles_aggregate_metrics.csv"
            ),
            "lodo_domain_influence": "lodo_domain_influence.csv",
            "calibration_scaffold_bootstrap": (
                "calibration_scaffold_bootstrap.csv"
            ),
            "results_brief_zh": "results_brief_zh.md",
            "qa_summary": "qa_summary.json",
            "artifact_sha256": "artifact_sha256.csv",
        },
        "artifact_inventory": artifact_inventory,
        "manifest_self_reference_exception": (
            "run_manifest.json is not included in its own artifact inventory"
        ),
    }
    manifest_path = output_dir / "run_manifest.json"
    atomic_write_json(run_manifest, manifest_path)

    if snapshot_directory(formal_dir) != parent_info["snapshot"]:
        raise RuntimeError("Formal OOD directory changed after manifest write")
    print(
        json.dumps(
            {
                "status": "complete",
                "analysis_mode": ANALYSIS_MODE,
                "output_dir": str(output_dir),
                "bootstrap_replicates": args.bootstrap_replicates,
                "elapsed_seconds": elapsed_seconds,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
