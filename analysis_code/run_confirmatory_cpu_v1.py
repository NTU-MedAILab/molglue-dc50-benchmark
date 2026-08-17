#!/usr/bin/env python
"""CPU-only confirmatory benchmark for molecular-glue DC50 prediction.

This script intentionally does not reuse the historical fixed train/test split.
It produces repeated, nested, group-disjoint cross-fitted predictions for a
small frozen model set. Any operation that uses response values (model fitting,
hyperparameter selection, contextual means) is restricted to the relevant
training fold.

Primary protocol
----------------
* Five-fold Bemis--Murcko scaffold CV, repeated five times.
* Four-fold scaffold CV inside every outer-training fold for model selection.
* Pooled cross-fitted Spearman correlation is the primary metric.
* Paired uncertainty is estimated by scaffold-cluster bootstrap.

The first CPU release excludes target encoding, target-local routing and
protein language-model features. The historical target encoder uses each
training row's own response, and the current protein cache requires a separate
protein-to-UniProt mapping audit. Those modules can be added only after their
respective leakage/provenance issues are fixed.
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
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import pearsonr, spearmanr
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "all_molglue_dc50_qc_train_test_standardized_context.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "reports" / "confirmatory_cpu_v1"
PROTOCOL_DOCUMENT = PROJECT_DIR / "docs" / "confirmatory_protocol_v1.md"

PROTOCOL_VERSION = "confirmatory_cpu_v1.1"
DEFAULT_SEED = 260531
PRIMARY_METRIC = "spearman"
FROZEN_DATA_SHA256 = (
    "e9494246bcbd198bc09d5b0d09721910c33f70f46ae859b76377a13775e3e4b2"
)
PROTOCOL_ORDER = ("scaffold", "compound")

CONTEXT_COLUMNS = [
    "source_database",
    "recruiting_protein",
    "target_protein",
    "cell_line",
    "assay_method",
    "activity_time",
    "mode_of_action",
]
CONTEXT_INTERACTIONS = {
    "source_target": ("source_database", "target_protein"),
    "recruiter_target": ("recruiting_protein", "target_protein"),
    "target_cell": ("target_protein", "cell_line"),
    "assay_target": ("assay_method", "target_protein"),
}

MODEL_ORDER = [
    "global_mean",
    "hierarchical_context_mean",
    "context_ridge",
    "morgan_knn",
    "chemistry_extra_trees",
    "full_context_extra_trees",
    "full_context_label_shuffle",
]
REFERENCE_MODEL = "full_context_extra_trees"
PRESPECIFIED_CONTRASTS = {
    "full_vs_context_ridge": (
        "full_context_extra_trees",
        "context_ridge",
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
        "full_context_label_shuffle",
    ),
}

SUSPECT_TARGET_EXACT = {
    "293T",
    "HEK293",
    "22RV1cells",
    "eGFP(-SD36)",
    "SD40",
}
SUSPECT_CELL_SUBSTRINGS = (
    "dc50 =",
    "reporter",
    "prolabel",
    "-luc",
    " luc",
)


@dataclass(frozen=True)
class OuterSplit:
    protocol: str
    repeat: int
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    split_seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=["compound", "scaffold"],
        default=["scaffold"],
        help="Outer validation protocols to run.",
    )
    parser.add_argument("--data-file", type=Path, default=DATA_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--outer-repeats", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-outer-splits",
        type=int,
        default=None,
        help="Smoke-test helper. Truncates each protocol and disables bootstrap.",
    )
    parser.add_argument(
        "--skip-label-shuffle",
        action="store_true",
        help="Skip the prespecified within-domain label-shuffle negative control.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting files in an existing output directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a hash-matched interrupted run from complete outer folds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and audit data/features/splits without fitting models.",
    )
    args = parser.parse_args()
    if args.outer_folds < 2 or args.inner_folds < 2:
        parser.error("outer-folds and inner-folds must both be >= 2")
    if args.outer_repeats < 1:
        parser.error("outer-repeats must be >= 1")
    if args.n_estimators < 10:
        parser.error("n-estimators must be >= 10")
    if args.n_jobs < 1:
        parser.error("n-jobs must be >= 1")
    if args.bootstrap_replicates < 0:
        parser.error("bootstrap-replicates must be >= 0")
    if args.max_outer_splits is not None and args.max_outer_splits < 1:
        parser.error("max-outer-splits must be >= 1")
    if len(args.protocols) != len(set(args.protocols)):
        parser.error("protocols must not contain duplicates")
    if args.resume and args.overwrite:
        parser.error("resume and overwrite are mutually exclusive")
    args.protocols = [
        protocol for protocol in PROTOCOL_ORDER if protocol in args.protocols
    ]
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_ready) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3:
        return float("nan")
    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        return float("nan")
    if np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return float("nan")
    return float(spearmanr(y_true, y_pred).statistic)


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3:
        return float("nan")
    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        return float("nan")
    if np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return float("nan")
    return float(pearsonr(y_true, y_pred).statistic)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if not np.all(np.isfinite(y_pred)):
        return {
            "spearman": float("nan"),
            "pearson": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "calibration_intercept": float("nan"),
            "calibration_slope": float("nan"),
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
        "r2": float(r2_score(y_true, y_pred)),
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
    }


def load_rows(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.read_csv(path)
    required = {
        "qc_id",
        "canonical_smiles",
        "pDC50",
        "source_database",
        "recruiting_protein",
        "target_protein",
        "cell_line",
        "assay_method",
        "activity_time",
        "mode_of_action",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    rows = rows.copy().reset_index(drop=True)
    rows["qc_id"] = rows["qc_id"].astype(str)
    if rows["qc_id"].duplicated().any():
        raise ValueError("qc_id must be unique")
    rows["canonical_smiles"] = rows["canonical_smiles"].fillna("").astype(str)
    rows["pDC50"] = pd.to_numeric(rows["pDC50"], errors="coerce")
    invalid_mask = rows["canonical_smiles"].eq("") | ~np.isfinite(rows["pDC50"])
    if invalid_mask.any():
        bad = rows.loc[invalid_mask, ["qc_id", "canonical_smiles", "pDC50"]]
        raise ValueError(f"Invalid modeling rows detected:\n{bad.to_string(index=False)}")

    audit_rows: list[dict[str, Any]] = []
    for idx, value in rows["target_protein"].fillna("NA").astype(str).items():
        if value in SUSPECT_TARGET_EXACT:
            audit_rows.append(
                {
                    "row_index": idx,
                    "qc_id": rows.at[idx, "qc_id"],
                    "field": "target_protein",
                    "old_value": value,
                    "new_value": "NA_INVALID_TARGET",
                    "rule": "prespecified_obvious_nonprotein_target",
                }
            )
            rows.at[idx, "target_protein"] = "NA_INVALID_TARGET"

    for idx, value in rows["cell_line"].fillna("unspecified").astype(str).items():
        if any(token in value.lower() for token in SUSPECT_CELL_SUBSTRINGS):
            audit_rows.append(
                {
                    "row_index": idx,
                    "qc_id": rows.at[idx, "qc_id"],
                    "field": "cell_line",
                    "old_value": value,
                    "new_value": "unspecified",
                    "rule": "prespecified_noncell_context_text",
                }
            )
            rows.at[idx, "cell_line"] = "unspecified"

    for col in CONTEXT_COLUMNS:
        rows[col] = (
            rows[col]
            .fillna("NA" if col != "cell_line" else "unspecified")
            .astype(str)
            .replace({"": "NA", "nan": "NA", "None": "NA"})
        )
    audit = pd.DataFrame(
        audit_rows,
        columns=[
            "row_index",
            "qc_id",
            "field",
            "old_value",
            "new_value",
            "rule",
        ],
    )
    return rows, audit


def build_scaffold_ids(smiles_values: Iterable[str]) -> np.ndarray:
    scaffold_ids: list[str] = []
    for smiles in smiles_values:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            raise ValueError(f"RDKit could not parse canonical SMILES: {smiles}")
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol,
            includeChirality=False,
        )
        if not scaffold:
            scaffold = f"ACYCLIC::{Chem.MolToSmiles(mol, canonical=True)}"
        scaffold_ids.append(scaffold)
    return np.asarray(scaffold_ids, dtype=object)


def build_morgan_matrix(
    smiles_values: Iterable[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    smiles_list = list(smiles_values)
    matrix = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        includeChirality=True,
    )
    for row_idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            raise ValueError(f"RDKit could not parse canonical SMILES: {smiles}")
        fingerprint = generator.GetFingerprint(mol)
        arr = np.zeros(n_bits, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fingerprint, arr)
        matrix[row_idx] = arr.astype(np.uint8)
    return matrix


def build_descriptor_matrix(
    smiles_values: Iterable[str],
) -> tuple[np.ndarray, list[str], list[int]]:
    descriptor_specs = list(Descriptors._descList)
    names = [name for name, _ in descriptor_specs]
    smiles_list = list(smiles_values)
    matrix = np.full(
        (len(smiles_list), len(descriptor_specs)),
        np.nan,
        dtype=np.float64,
    )
    failure_counts = np.zeros(len(descriptor_specs), dtype=np.int64)
    for row_idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            raise ValueError(f"RDKit could not parse canonical SMILES: {smiles}")
        for col_idx, (_, function) in enumerate(descriptor_specs):
            try:
                value = float(function(mol))
            except Exception:
                failure_counts[col_idx] += 1
                continue
            if math.isfinite(value):
                matrix[row_idx, col_idx] = value
            else:
                failure_counts[col_idx] += 1
    return matrix, names, failure_counts.tolist()


def feature_manifest(
    descriptor_names: list[str],
    descriptor_failures: list[int],
) -> dict[str, Any]:
    return {
        "morgan": {
            "radius": 2,
            "n_bits": 2048,
            "use_chirality": True,
        },
        "murcko_scaffold": {"include_chirality": False},
        "rdkit_descriptors": {
            "names": descriptor_names,
            "n_descriptors": len(descriptor_names),
            "failure_counts": descriptor_failures,
            "imputation": "fit-fold median; all-missing columns set to zero",
            "fold_statistics_dtype": "float64",
            "zero_variance": "removed using fit fold only in float64",
        },
        "context_columns": CONTEXT_COLUMNS,
        "context_interactions": CONTEXT_INTERACTIONS,
    }


def balanced_group_splits(
    indices: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(indices) == 0:
        raise ValueError("Cannot split an empty index set")
    local_groups = groups[indices]
    unique_groups, inverse, counts = np.unique(
        local_groups,
        return_inverse=True,
        return_counts=True,
    )
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} groups, found {len(unique_groups)}"
        )
    rng = np.random.default_rng(seed)
    random_ties = rng.random(len(unique_groups))
    order = sorted(
        range(len(unique_groups)),
        key=lambda i: (-int(counts[i]), float(random_ties[i])),
    )
    fold_loads = np.zeros(n_splits, dtype=np.int64)
    group_to_fold: dict[Any, int] = {}
    for group_idx in order:
        minimum = int(fold_loads.min())
        candidates = np.where(fold_loads == minimum)[0]
        chosen_fold = int(rng.choice(candidates))
        group_to_fold[unique_groups[group_idx]] = chosen_fold
        fold_loads[chosen_fold] += int(counts[group_idx])

    assignments = np.asarray(
        [group_to_fold[group] for group in local_groups],
        dtype=np.int64,
    )
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_splits):
        eval_mask = assignments == fold
        eval_idx = indices[eval_mask]
        fit_idx = indices[~eval_mask]
        if len(eval_idx) == 0 or len(fit_idx) == 0:
            raise RuntimeError("Balanced split produced an empty fold")
        if set(groups[fit_idx]).intersection(set(groups[eval_idx])):
            raise RuntimeError("Group leakage detected while constructing folds")
        splits.append((fit_idx, eval_idx))
    return splits


def make_outer_splits(
    protocol: str,
    rows: pd.DataFrame,
    scaffold_ids: np.ndarray,
    n_folds: int,
    n_repeats: int,
    seed: int,
) -> list[OuterSplit]:
    if protocol == "compound":
        groups = rows["canonical_smiles"].to_numpy(dtype=object)
    elif protocol == "scaffold":
        groups = scaffold_ids
    else:
        raise ValueError(f"Unknown protocol: {protocol}")
    indices = np.arange(len(rows), dtype=np.int64)
    outer: list[OuterSplit] = []
    for repeat in range(n_repeats):
        split_seed = seed + repeat
        for fold, (train_idx, test_idx) in enumerate(
            balanced_group_splits(indices, groups, n_folds, split_seed),
            start=1,
        ):
            outer.append(
                OuterSplit(
                    protocol=protocol,
                    repeat=repeat + 1,
                    fold=fold,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    split_seed=split_seed,
                )
            )
    return outer


def context_frame(rows: pd.DataFrame) -> pd.DataFrame:
    frame = rows[CONTEXT_COLUMNS].copy()
    for name, columns in CONTEXT_INTERACTIONS.items():
        frame[name] = frame[list(columns)].astype(str).agg("||".join, axis=1)
    return frame.fillna("NA").astype(str)


def make_one_hot_encoder() -> OneHotEncoder:
    return OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=2,
        sparse_output=False,
        dtype=np.float32,
    )


def prepare_rdkit_fold(
    descriptor_matrix: np.ndarray,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train = descriptor_matrix[fit_idx].astype(np.float64, copy=True)
    eval_values = descriptor_matrix[eval_idx].astype(np.float64, copy=True)
    with np.errstate(all="ignore"):
        medians = np.nanmedian(train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float64)
    train_missing = ~np.isfinite(train)
    eval_missing = ~np.isfinite(eval_values)
    if train_missing.any():
        train[train_missing] = np.take(medians, np.where(train_missing)[1])
    if eval_missing.any():
        eval_values[eval_missing] = np.take(medians, np.where(eval_missing)[1])
    variances = np.var(train, axis=0)
    keep = np.isfinite(variances) & (variances > 0.0)
    if not np.any(keep):
        raise RuntimeError("All RDKit descriptors were removed as zero variance")
    return (
        train[:, keep],
        eval_values[:, keep],
        {
            "rdkit_kept": int(keep.sum()),
            "rdkit_total": int(len(keep)),
            "rdkit_train_missing": int(train_missing.sum()),
            "rdkit_eval_missing": int(eval_missing.sum()),
        },
    )


def build_fold_features(
    rows: pd.DataFrame,
    morgan: np.ndarray,
    descriptors: np.ndarray,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    rdkit_fit, rdkit_eval, rdkit_meta = prepare_rdkit_fold(
        descriptors,
        fit_idx,
        eval_idx,
    )
    morgan_fit = morgan[fit_idx].astype(np.float32)
    morgan_eval = morgan[eval_idx].astype(np.float32)
    chemistry_fit = np.hstack([morgan_fit, rdkit_fit]).astype(np.float32)
    chemistry_eval = np.hstack([morgan_eval, rdkit_eval]).astype(np.float32)

    encoder = make_one_hot_encoder()
    contexts = context_frame(rows)
    context_fit = encoder.fit_transform(contexts.iloc[fit_idx]).astype(np.float32)
    context_eval = encoder.transform(contexts.iloc[eval_idx]).astype(np.float32)
    features = {
        "chemistry": (chemistry_fit, chemistry_eval),
        "context": (context_fit, context_eval),
        "full": (
            np.hstack([chemistry_fit, context_fit]).astype(np.float32),
            np.hstack([chemistry_eval, context_eval]).astype(np.float32),
        ),
    }
    metadata = {
        **rdkit_meta,
        "context_dim": int(context_fit.shape[1]),
        "chemistry_dim": int(chemistry_fit.shape[1]),
        "full_dim": int(features["full"][0].shape[1]),
    }
    return features, metadata


def key_series(rows: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    return rows[list(columns)].fillna("NA").astype(str).agg("||".join, axis=1)


def hierarchical_context_predict(
    rows: pd.DataFrame,
    y: np.ndarray,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
    smoothing: float,
) -> np.ndarray:
    global_mean = float(np.mean(y[fit_idx]))
    fit_rows = rows.iloc[fit_idx]
    eval_rows = rows.iloc[eval_idx]
    keys = [
        ("target_protein",),
        ("recruiting_protein", "target_protein"),
        ("source_database", "target_protein"),
        ("cell_line", "target_protein"),
        ("assay_method", "target_protein"),
    ]
    estimates = [np.full(len(eval_idx), global_mean, dtype=np.float64)]
    weights = [np.ones(len(eval_idx), dtype=np.float64)]
    for columns in keys:
        fit_key = key_series(fit_rows, columns)
        eval_key = key_series(eval_rows, columns)
        stats = (
            pd.DataFrame({"key": fit_key.to_numpy(), "y": y[fit_idx]})
            .groupby("key")["y"]
            .agg(["mean", "count"])
        )
        smoothed = (
            stats["mean"] * stats["count"] + global_mean * smoothing
        ) / (stats["count"] + smoothing)
        pred = eval_key.map(smoothed).to_numpy(dtype=np.float64)
        count = eval_key.map(stats["count"]).to_numpy(dtype=np.float64)
        available = np.isfinite(pred) & np.isfinite(count)
        estimates.append(np.where(available, pred, 0.0))
        weights.append(np.where(available, np.log1p(count), 0.0))
    estimate_matrix = np.vstack(estimates)
    weight_matrix = np.vstack(weights)
    return (
        np.sum(estimate_matrix * weight_matrix, axis=0)
        / np.clip(np.sum(weight_matrix, axis=0), 1e-12, None)
    ).astype(np.float32)


def tanimoto_matrix(eval_fp: np.ndarray, fit_fp: np.ndarray) -> np.ndarray:
    eval_float = eval_fp.astype(np.float32, copy=False)
    fit_float = fit_fp.astype(np.float32, copy=False)
    intersection = eval_float @ fit_float.T
    denominator = (
        eval_float.sum(axis=1, keepdims=True)
        + fit_float.sum(axis=1, keepdims=True).T
        - intersection
    )
    return np.divide(
        intersection,
        denominator,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=denominator > 0,
    )


def prepare_morgan_knn(
    rows: pd.DataFrame,
    morgan: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fit_smiles = rows.iloc[fit_idx]["canonical_smiles"].to_numpy(dtype=object)
    _, first_positions, inverse = np.unique(
        fit_smiles,
        return_index=True,
        return_inverse=True,
    )
    unique_fit_idx = fit_idx[first_positions]
    compound_medians = np.asarray(
        [
            np.median(y[fit_idx][inverse == group])
            for group in range(len(first_positions))
        ],
        dtype=np.float32,
    )
    similarities = tanimoto_matrix(morgan[eval_idx], morgan[unique_fit_idx])
    return similarities, compound_medians


def morgan_knn_from_precomputed(
    similarities: np.ndarray,
    compound_medians: np.ndarray,
    k: int,
    power: float,
) -> np.ndarray:
    k_eff = min(int(k), similarities.shape[1])
    top = np.argpartition(-similarities, kth=k_eff - 1, axis=1)[:, :k_eff]
    top_sim = np.take_along_axis(similarities, top, axis=1)
    top_y = compound_medians[top]
    positive = top_sim > 0
    weights = np.where(positive, np.power(top_sim, power), 0.0)
    numerator = np.sum(weights * top_y, axis=1)
    denominator = np.sum(weights, axis=1)
    fallback = float(np.mean(compound_medians))
    return np.divide(
        numerator,
        denominator,
        out=np.full(similarities.shape[0], fallback, dtype=np.float64),
        where=denominator > 0,
    ).astype(np.float32)


def morgan_knn_predict(
    rows: pd.DataFrame,
    morgan: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
    k: int,
    power: float,
) -> np.ndarray:
    similarities, compound_medians = prepare_morgan_knn(
        rows,
        morgan,
        y,
        fit_idx,
        eval_idx,
    )
    return morgan_knn_from_precomputed(
        similarities,
        compound_medians,
        k,
        power,
    )


def model_param_grid(model_id: str) -> list[dict[str, Any]]:
    if model_id == "global_mean":
        return [{"param_id": "global"}]
    if model_id == "hierarchical_context_mean":
        return [
            {"param_id": f"smoothing_{value:g}", "smoothing": float(value)}
            for value in (2.0, 8.0, 32.0)
        ]
    if model_id == "context_ridge":
        return [
            {"param_id": f"alpha_{value:g}", "alpha": float(value)}
            for value in (0.1, 1.0, 10.0, 100.0)
        ]
    if model_id == "morgan_knn":
        return [
            {
                "param_id": f"k_{k}_power_{power:g}",
                "k": int(k),
                "power": float(power),
            }
            for k in (1, 3, 5, 10, 20)
            for power in (1.0, 2.0)
        ]
    if model_id in {
        "chemistry_extra_trees",
        "full_context_extra_trees",
    }:
        return [
            {
                "param_id": f"leaf_{leaf}_maxfeat_{str(max_features).replace('.', 'p')}",
                "min_samples_leaf": int(leaf),
                "max_features": max_features,
            }
            for leaf in (1, 5, 10)
            for max_features in ("sqrt", 0.3)
        ]
    raise ValueError(f"No parameter grid for model {model_id}")


def output_model_ids(skip_label_shuffle: bool) -> list[str]:
    models = list(MODEL_ORDER[:-1])
    if not skip_label_shuffle:
        models.append("full_context_label_shuffle")
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
    train_y = y[fit_idx] if override_train_y is None else override_train_y
    if model_id == "global_mean":
        return np.full(len(eval_idx), float(np.mean(train_y)), dtype=np.float32)
    if model_id == "hierarchical_context_mean":
        if override_train_y is not None:
            y_override = y.copy()
            y_override[fit_idx] = train_y
        else:
            y_override = y
        return hierarchical_context_predict(
            rows,
            y_override,
            fit_idx,
            eval_idx,
            smoothing=float(params["smoothing"]),
        )
    if model_id == "morgan_knn":
        if override_train_y is not None:
            y_override = y.copy()
            y_override[fit_idx] = train_y
        else:
            y_override = y
        return morgan_knn_predict(
            rows,
            morgan,
            y_override,
            fit_idx,
            eval_idx,
            k=int(params["k"]),
            power=float(params["power"]),
        )
    if model_id == "context_ridge":
        x_fit, x_eval = fold_features["context"]
        model = Ridge(alpha=float(params["alpha"]), solver="lsqr")
        model.fit(x_fit, train_y)
        return np.asarray(model.predict(x_eval), dtype=np.float32)
    if model_id in {
        "chemistry_extra_trees",
        "full_context_extra_trees",
    }:
        feature_key = (
            "chemistry" if model_id == "chemistry_extra_trees" else "full"
        )
        x_fit, x_eval = fold_features[feature_key]
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
    raise ValueError(f"Unknown model: {model_id}")


def within_domain_label_permutation(
    rows: pd.DataFrame,
    y_fit: np.ndarray,
    fit_idx: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    fit_rows = rows.iloc[fit_idx]
    strata = (
        fit_rows["source_database"].astype(str)
        + "||"
        + fit_rows["target_protein"].astype(str)
    ).to_numpy()
    shuffled = y_fit.copy()
    unique_strata, stratum_counts = np.unique(strata, return_counts=True)
    for stratum in unique_strata:
        local = np.where(strata == stratum)[0]
        if len(local) > 1:
            shuffled[local] = shuffled[rng.permutation(local)]
    diagnostics: dict[str, float | int] = {
        "permutation_n_strata": int(len(unique_strata)),
        "permutation_n_singleton_strata": int(np.sum(stratum_counts == 1)),
        "permutation_singleton_row_fraction": float(
            np.sum(stratum_counts[stratum_counts == 1]) / len(y_fit)
        ),
        "permutation_changed_label_fraction": float(
            np.mean(shuffled != y_fit)
        ),
        "permutation_label_pearson": safe_pearson(
            y_fit.astype(np.float64),
            shuffled.astype(np.float64),
        ),
    }
    return shuffled, diagnostics


def regularization_rank(
    model_id: str,
    params: dict[str, Any],
) -> tuple[float, ...]:
    if model_id == "global_mean":
        return (0.0,)
    if model_id == "hierarchical_context_mean":
        return (float(params["smoothing"]),)
    if model_id == "context_ridge":
        return (float(params["alpha"]),)
    if model_id == "morgan_knn":
        return (float(params["k"]), -float(params["power"]))
    if model_id in {"chemistry_extra_trees", "full_context_extra_trees"}:
        max_features_rank = 1.0 if params["max_features"] == "sqrt" else 0.0
        return (float(params["min_samples_leaf"]), max_features_rank)
    raise ValueError(f"No regularization rank for model {model_id}")


def select_model_params(
    model_ids: list[str],
    rows: pd.DataFrame,
    y: np.ndarray,
    morgan: np.ndarray,
    descriptors: np.ndarray,
    outer_train_idx: np.ndarray,
    inner_groups: np.ndarray,
    inner_folds: int,
    seed: int,
    n_estimators: int,
    n_jobs: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    inner_splits = balanced_group_splits(
        outer_train_idx,
        inner_groups,
        inner_folds,
        seed,
    )
    candidate_predictions: dict[tuple[str, str], np.ndarray] = {}
    candidate_params: dict[tuple[str, str], dict[str, Any]] = {}
    local_position = {
        int(row_idx): position
        for position, row_idx in enumerate(outer_train_idx.tolist())
    }
    for model_id in model_ids:
        for params in model_param_grid(model_id):
            key = (model_id, str(params["param_id"]))
            candidate_predictions[key] = np.full(
                len(outer_train_idx),
                np.nan,
                dtype=np.float32,
            )
            candidate_params[key] = params

    for inner_fold, (fit_idx, eval_idx) in enumerate(inner_splits, start=1):
        fold_features, _ = build_fold_features(
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
        knn_precomputed = (
            prepare_morgan_knn(rows, morgan, y, fit_idx, eval_idx)
            if "morgan_knn" in model_ids
            else None
        )
        for model_id in model_ids:
            for params in model_param_grid(model_id):
                key = (model_id, str(params["param_id"]))
                if model_id == "morgan_knn":
                    if knn_precomputed is None:
                        raise RuntimeError("Missing precomputed kNN similarities")
                    pred = morgan_knn_from_precomputed(
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
                candidate_predictions[key][eval_positions] = pred

    selected: dict[str, dict[str, Any]] = {}
    tuning_rows: list[dict[str, Any]] = []
    inner_truth = y[outer_train_idx]
    for model_id in model_ids:
        scored: list[dict[str, Any]] = []
        for params in model_param_grid(model_id):
            param_id = str(params["param_id"])
            pred = candidate_predictions[(model_id, param_id)]
            metrics = regression_metrics(inner_truth, pred)
            score = metrics[PRIMARY_METRIC]
            if not math.isfinite(score):
                score = -float("inf")
            scored.append(
                {
                    "score": float(score),
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
                tuple(-value for value in item["regularization_rank"]),
                item["rmse"],
                item["param_id"],
            )
        )
        selected[model_id] = dict(eligible[0]["params"])
    return selected, tuning_rows


def target_macro_metrics(frame: pd.DataFrame) -> dict[str, float]:
    correlations: list[float] = []
    eligible = 0
    for _, group in frame.groupby("target_protein", sort=False):
        if group["canonical_smiles"].nunique() < 20:
            continue
        eligible += 1
        value = safe_spearman(
            group["y_true"].to_numpy(dtype=np.float64),
            group["y_pred"].to_numpy(dtype=np.float64),
        )
        if math.isfinite(value):
            correlations.append(value)
    if not correlations:
        return {
            "target_macro_spearman": float("nan"),
            "target_macro_n_groups": int(eligible),
        }
    return {
        "target_macro_spearman": float(np.mean(correlations)),
        "target_macro_n_groups": int(len(correlations)),
    }


def within_target_spearman(frame: pd.DataFrame) -> float:
    centered_true = frame["y_true"] - frame.groupby("target_protein")[
        "y_true"
    ].transform("mean")
    centered_pred = frame["y_pred"] - frame.groupby("target_protein")[
        "y_pred"
    ].transform("mean")
    return safe_spearman(
        centered_true.to_numpy(dtype=np.float64),
        centered_pred.to_numpy(dtype=np.float64),
    )


def finite_distribution_summary(
    values: Iterable[float],
) -> tuple[float, float, float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return (float("nan"),) * 4
    standard_deviation = (
        float(np.std(finite, ddof=1)) if len(finite) >= 2 else float("nan")
    )
    return (
        float(np.mean(finite)),
        standard_deviation,
        float(np.min(finite)),
        float(np.max(finite)),
    )


def finite_percentile_interval(
    values: Iterable[float],
) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return (float("nan"),) * 3
    return (
        float(np.mean(finite)),
        float(np.percentile(finite, 2.5)),
        float(np.percentile(finite, 97.5)),
    )


def summarize_predictions(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    repeat_rows: list[dict[str, Any]] = []
    for (protocol, repeat, model_id), frame in predictions.groupby(
        ["protocol", "repeat", "model_id"],
        sort=False,
    ):
        metrics = regression_metrics(
            frame["y_true"].to_numpy(dtype=np.float64),
            frame["y_pred"].to_numpy(dtype=np.float64),
        )
        repeat_rows.append(
            {
                "protocol": protocol,
                "repeat": int(repeat),
                "model_id": model_id,
                "n_rows": int(len(frame)),
                "n_unique_smiles": int(frame["canonical_smiles"].nunique()),
                "n_scaffolds": int(frame["scaffold_id"].nunique()),
                **metrics,
                **target_macro_metrics(frame),
                "within_target_spearman": within_target_spearman(frame),
            }
        )
    repeat_metrics = pd.DataFrame(repeat_rows)
    summary_rows: list[dict[str, Any]] = []
    metric_columns = [
        "spearman",
        "pearson",
        "rmse",
        "mae",
        "r2",
        "calibration_intercept",
        "calibration_slope",
        "target_macro_spearman",
        "within_target_spearman",
    ]
    for (protocol, model_id), frame in repeat_metrics.groupby(
        ["protocol", "model_id"],
        sort=False,
    ):
        row: dict[str, Any] = {
            "protocol": protocol,
            "model_id": model_id,
            "n_repeats": int(frame["repeat"].nunique()),
            "n_rows_per_repeat": int(frame["n_rows"].max()),
            "n_unique_smiles": int(frame["n_unique_smiles"].max()),
            "n_scaffolds": int(frame["n_scaffolds"].max()),
        }
        for metric in metric_columns:
            values = frame[metric].to_numpy(dtype=np.float64)
            mean, standard_deviation, minimum, maximum = (
                finite_distribution_summary(values)
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd_across_splits"] = standard_deviation
            row[f"{metric}_min"] = minimum
            row[f"{metric}_max"] = maximum
        summary_rows.append(row)
    return repeat_metrics, pd.DataFrame(summary_rows)


def average_cross_fitted_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    identity_columns = [
        "protocol",
        "model_id",
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

    averaged = (
        predictions.groupby(identity_columns, sort=False, dropna=False)
        .agg(
            n_oof_predictions=("y_pred", "size"),
            n_repeats=("repeat", "nunique"),
            selected_param_ids=("param_id", joined_unique),
            y_pred=("y_pred", "mean"),
            y_pred_sd_across_repeats=("y_pred", "std"),
            max_train_tanimoto_mean=("max_train_tanimoto", "mean"),
            max_train_tanimoto_min=("max_train_tanimoto", "min"),
            scaffold_seen_fraction=("scaffold_seen_in_train", "mean"),
            target_train_rows_mean=("target_train_rows", "mean"),
            target_train_rows_min=("target_train_rows", "min"),
            target_train_unique_smiles_mean=(
                "target_train_unique_smiles",
                "mean",
            ),
            target_train_unique_smiles_min=(
                "target_train_unique_smiles",
                "min",
            ),
            recruiter_target_train_unique_smiles_mean=(
                "recruiter_target_train_unique_smiles",
                "mean",
            ),
            recruiter_target_train_unique_smiles_min=(
                "recruiter_target_train_unique_smiles",
                "min",
            ),
            source_seen_fraction=("source_seen_in_train", "mean"),
        )
        .reset_index()
    )
    return averaged


def metrics_on_repeat_averaged_predictions(
    averaged_predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (protocol, model_id), frame in averaged_predictions.groupby(
        ["protocol", "model_id"],
        sort=False,
    ):
        metrics = regression_metrics(
            frame["y_true"].to_numpy(dtype=np.float64),
            frame["y_pred"].to_numpy(dtype=np.float64),
        )
        rows.append(
            {
                "protocol": protocol,
                "model_id": model_id,
                "n_rows": int(len(frame)),
                "n_unique_smiles": int(frame["canonical_smiles"].nunique()),
                "n_scaffolds": int(frame["scaffold_id"].nunique()),
                "n_oof_predictions_per_row_min": int(
                    frame["n_oof_predictions"].min()
                ),
                "n_oof_predictions_per_row_max": int(
                    frame["n_oof_predictions"].max()
                ),
                **metrics,
                **target_macro_metrics(frame),
                "within_target_spearman": within_target_spearman(frame),
            }
        )
    return pd.DataFrame(rows)


def metrics_on_canonical_smiles_aggregates(
    averaged_predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (protocol, model_id), frame in averaged_predictions.groupby(
        ["protocol", "model_id"],
        sort=False,
    ):
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
        rows.append(
            {
                "protocol": protocol,
                "model_id": model_id,
                "aggregation": (
                    "median y_true and median repeat-averaged y_pred "
                    "within canonical_smiles"
                ),
                "n_unique_smiles": int(len(compounds)),
                "n_scaffolds": int(compounds["scaffold_id"].nunique()),
                "n_multirow_smiles": int(
                    np.sum(compounds["n_source_rows"] > 1)
                ),
                **regression_metrics(
                    compounds["y_true"].to_numpy(dtype=np.float64),
                    compounds["y_pred"].to_numpy(dtype=np.float64),
                ),
            }
        )
    return pd.DataFrame(rows)


def build_applicability_table(
    averaged_predictions: pd.DataFrame,
) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    detail_columns = [
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
        "n_oof_predictions",
        "n_repeats",
        "max_train_tanimoto_mean",
        "max_train_tanimoto_min",
        "scaffold_seen_fraction",
        "target_train_rows_mean",
        "target_train_rows_min",
        "target_train_unique_smiles_mean",
        "target_train_unique_smiles_min",
        "recruiter_target_train_unique_smiles_mean",
        "recruiter_target_train_unique_smiles_min",
        "source_seen_fraction",
    ]
    for protocol, frame in averaged_predictions.groupby("protocol", sort=False):
        reference = frame[
            frame["model_id"] == REFERENCE_MODEL
        ][detail_columns].copy()
        wide = frame.pivot(
            index="row_index",
            columns="model_id",
            values="y_pred",
        ).rename(columns=lambda name: f"prediction__{name}")
        reference = reference.merge(
            wide.reset_index(),
            on="row_index",
            how="left",
            validate="one_to_one",
        )
        knn_column = "prediction__morgan_knn"
        reference_column = f"prediction__{REFERENCE_MODEL}"
        context_column = "prediction__context_ridge"
        if knn_column in reference:
            reference["abs_full_minus_morgan_knn"] = np.abs(
                reference[reference_column] - reference[knn_column]
            )
        if context_column in reference:
            reference["abs_full_minus_context_ridge"] = np.abs(
                reference[reference_column] - reference[context_column]
            )
        tables.append(reference)
    return pd.concat(tables, ignore_index=True)


def risk_coverage_metrics(
    applicability: pd.DataFrame,
    model_ids: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    requested_coverages = (1.0, 0.8, 0.6, 0.4, 0.2)
    for protocol, frame in applicability.groupby("protocol", sort=False):
        ranked = frame.sort_values(
            [
                "max_train_tanimoto_mean",
                "target_train_unique_smiles_min",
                "row_index",
            ],
            ascending=[False, False, True],
            kind="mergesort",
        )
        for requested_coverage in requested_coverages:
            n_selected = max(
                1,
                int(math.ceil(len(ranked) * requested_coverage)),
            )
            selected = ranked.iloc[:n_selected]
            for model_id in model_ids:
                prediction_column = f"prediction__{model_id}"
                if prediction_column not in selected:
                    continue
                metrics = regression_metrics(
                    selected["y_true"].to_numpy(dtype=np.float64),
                    selected[prediction_column].to_numpy(dtype=np.float64),
                )
                rows.append(
                    {
                        "protocol": protocol,
                        "model_id": model_id,
                        "requested_coverage": requested_coverage,
                        "n_selected_rows": int(n_selected),
                        "n_selected_unique_smiles": int(
                            selected["canonical_smiles"].nunique()
                        ),
                        "n_selected_scaffolds": int(
                            selected["scaffold_id"].nunique()
                        ),
                        "realized_coverage": float(n_selected / len(ranked)),
                        "selection_rule": (
                            "max_train_tanimoto_mean descending; "
                            "target_train_unique_smiles_min descending; "
                            "row_index ascending; no evaluation labels"
                        ),
                        "minimum_selected_max_tanimoto": float(
                            selected["max_train_tanimoto_mean"].min()
                        ),
                        "mean_selected_max_tanimoto": float(
                            selected["max_train_tanimoto_mean"].mean()
                        ),
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def complete_repeats(
    predictions: pd.DataFrame,
    n_expected_rows: int,
    protocols: list[str],
    n_repeats: int,
    skip_label_shuffle: bool,
) -> bool:
    if predictions.empty:
        return False
    expected_models = set(output_model_ids(skip_label_shuffle))
    if set(predictions["protocol"].unique()) != set(protocols):
        return False
    if not np.all(np.isfinite(predictions["y_pred"])):
        return False
    key_columns = ["protocol", "repeat", "model_id", "row_index"]
    if bool(predictions.duplicated(key_columns).any()):
        return False
    expected_row_indices = set(range(n_expected_rows))
    for protocol in protocols:
        local = predictions[predictions["protocol"] == protocol]
        if set(local["repeat"].unique()) != set(range(1, n_repeats + 1)):
            return False
        if set(local["model_id"].unique()) != expected_models:
            return False
        for (_, _), frame in local.groupby(["repeat", "model_id"]):
            if len(frame) != n_expected_rows:
                return False
            if set(frame["row_index"].astype(int)) != expected_row_indices:
                return False
    return True


def complete_repeat_averages(
    averaged_predictions: pd.DataFrame,
    n_expected_rows: int,
    protocols: list[str],
    n_repeats: int,
    skip_label_shuffle: bool,
) -> bool:
    if averaged_predictions.empty:
        return False
    expected_models = set(output_model_ids(skip_label_shuffle))
    expected_row_indices = set(range(n_expected_rows))
    if set(averaged_predictions["protocol"].unique()) != set(protocols):
        return False
    if bool(
        averaged_predictions.duplicated(
            ["protocol", "model_id", "row_index"]
        ).any()
    ):
        return False
    if not bool(
        (averaged_predictions["n_oof_predictions"] == n_repeats).all()
    ):
        return False
    for protocol in protocols:
        local = averaged_predictions[
            averaged_predictions["protocol"] == protocol
        ]
        if set(local["model_id"].unique()) != expected_models:
            return False
        for _, frame in local.groupby("model_id"):
            if len(frame) != n_expected_rows:
                return False
            if set(frame["row_index"].astype(int)) != expected_row_indices:
                return False
    return True


def cluster_bootstrap_summary(
    averaged_predictions: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
    cluster_col: str = "scaffold_id",
) -> pd.DataFrame:
    if n_bootstrap <= 0:
        return pd.DataFrame()
    protocols = averaged_predictions["protocol"].unique().tolist()
    all_rows: list[dict[str, Any]] = []
    protocol_seed_offsets = {"scaffold": 0, "compound": 1}
    for protocol in protocols:
        protocol_df = averaged_predictions[
            averaged_predictions["protocol"] == protocol
        ].copy()
        models = [
            model for model in MODEL_ORDER if model in protocol_df["model_id"].unique()
        ]
        reference_frame = protocol_df[
            protocol_df["model_id"] == models[0]
        ].sort_values("row_index")
        row_indices = reference_frame["row_index"].to_numpy(dtype=np.int64)
        clusters = reference_frame[cluster_col].astype(str).to_numpy()
        unique_clusters = np.unique(clusters)
        cluster_to_positions = {
            cluster: np.where(clusters == cluster)[0]
            for cluster in unique_clusters
        }
        model_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for model_id in models:
            frame = protocol_df[
                protocol_df["model_id"] == model_id
            ].sort_values("row_index")
            if not np.array_equal(
                frame["row_index"].to_numpy(dtype=np.int64),
                row_indices,
            ):
                raise RuntimeError(
                    "Bootstrap requires aligned repeat-averaged predictions"
                )
            model_arrays[model_id] = (
                frame["y_true"].to_numpy(dtype=np.float64),
                frame["y_pred"].to_numpy(dtype=np.float64),
            )

        rng = np.random.default_rng(seed + protocol_seed_offsets[protocol])
        bootstrap_metrics: dict[str, dict[str, list[float]]] = {
            model: {"spearman": [], "rmse": []} for model in models
        }
        bootstrap_delta: dict[str, dict[str, list[float]]] = {
            model: {"delta_spearman": [], "delta_rmse": []}
            for model in models
            if model != REFERENCE_MODEL
        }
        valid_contrasts = {
            contrast_id: (first_model, comparator_model)
            for contrast_id, (first_model, comparator_model) in (
                PRESPECIFIED_CONTRASTS.items()
            )
            if first_model in models and comparator_model in models
        }
        bootstrap_contrasts: dict[str, dict[str, list[float]]] = {
            contrast_id: {"delta_spearman": [], "delta_rmse": []}
            for contrast_id in valid_contrasts
        }
        for bootstrap_idx in range(n_bootstrap):
            sampled_clusters = rng.choice(
                unique_clusters,
                size=len(unique_clusters),
                replace=True,
            )
            sampled_positions = np.concatenate(
                [cluster_to_positions[cluster] for cluster in sampled_clusters]
            )
            metrics_by_model: dict[str, dict[str, float]] = {}
            for model_id in models:
                y_true, y_pred = model_arrays[model_id]
                metrics_by_model[model_id] = regression_metrics(
                    y_true[sampled_positions],
                    y_pred[sampled_positions],
                )
                bootstrap_metrics[model_id]["spearman"].append(
                    metrics_by_model[model_id]["spearman"]
                )
                bootstrap_metrics[model_id]["rmse"].append(
                    metrics_by_model[model_id]["rmse"]
                )
            if REFERENCE_MODEL in metrics_by_model:
                reference = metrics_by_model[REFERENCE_MODEL]
                for model_id in bootstrap_delta:
                    current = metrics_by_model[model_id]
                    bootstrap_delta[model_id]["delta_spearman"].append(
                        reference["spearman"] - current["spearman"]
                    )
                    bootstrap_delta[model_id]["delta_rmse"].append(
                        current["rmse"] - reference["rmse"]
                    )
            for contrast_id, (
                first_model,
                comparator_model,
            ) in valid_contrasts.items():
                first = metrics_by_model[first_model]
                comparator = metrics_by_model[comparator_model]
                bootstrap_contrasts[contrast_id]["delta_spearman"].append(
                    first["spearman"] - comparator["spearman"]
                )
                bootstrap_contrasts[contrast_id]["delta_rmse"].append(
                    comparator["rmse"] - first["rmse"]
                )
            if (bootstrap_idx + 1) % max(1, n_bootstrap // 10) == 0:
                print(
                    f"[BOOT] {protocol}: {bootstrap_idx + 1}/{n_bootstrap}",
                    flush=True,
                )

        for model_id in models:
            observed_y, observed_pred = model_arrays[model_id]
            observed = regression_metrics(observed_y, observed_pred)
            (
                spearman_mean,
                spearman_low,
                spearman_high,
            ) = finite_percentile_interval(
                bootstrap_metrics[model_id]["spearman"]
            )
            rmse_mean, rmse_low, rmse_high = finite_percentile_interval(
                bootstrap_metrics[model_id]["rmse"]
            )
            row = {
                "record_type": "model",
                "protocol": protocol,
                "model_id": model_id,
                "cluster_unit": cluster_col,
                "n_clusters": int(len(unique_clusters)),
                "n_bootstrap": int(n_bootstrap),
                "spearman_observed": observed["spearman"],
                "spearman_bootstrap_mean": spearman_mean,
                "spearman_ci_low": spearman_low,
                "spearman_ci_high": spearman_high,
                "rmse_observed": observed["rmse"],
                "rmse_bootstrap_mean": rmse_mean,
                "rmse_ci_low": rmse_low,
                "rmse_ci_high": rmse_high,
                "reference_model": REFERENCE_MODEL,
                "delta_direction": (
                    "delta_spearman=reference-model; "
                    "delta_rmse=model-reference; positive favors reference"
                ),
            }
            if model_id in bootstrap_delta:
                delta_s = np.asarray(
                    bootstrap_delta[model_id]["delta_spearman"],
                    dtype=np.float64,
                )
                delta_r = np.asarray(
                    bootstrap_delta[model_id]["delta_rmse"],
                    dtype=np.float64,
                )
                delta_s_mean, delta_s_low, delta_s_high = (
                    finite_percentile_interval(delta_s)
                )
                delta_r_mean, delta_r_low, delta_r_high = (
                    finite_percentile_interval(delta_r)
                )
                reference_y, reference_pred = model_arrays[REFERENCE_MODEL]
                reference_observed = regression_metrics(
                    reference_y,
                    reference_pred,
                )
                row.update(
                    {
                        "delta_spearman_observed": (
                            reference_observed["spearman"]
                            - observed["spearman"]
                        ),
                        "delta_spearman_mean": delta_s_mean,
                        "delta_spearman_ci_low": delta_s_low,
                        "delta_spearman_ci_high": delta_s_high,
                        "delta_rmse_observed": (
                            observed["rmse"] - reference_observed["rmse"]
                        ),
                        "delta_rmse_mean": delta_r_mean,
                        "delta_rmse_ci_low": delta_r_low,
                        "delta_rmse_ci_high": delta_r_high,
                    }
                )
            all_rows.append(row)
        for contrast_id, (
            first_model,
            comparator_model,
        ) in valid_contrasts.items():
            first_y, first_pred = model_arrays[first_model]
            comparator_y, comparator_pred = model_arrays[comparator_model]
            first_observed = regression_metrics(first_y, first_pred)
            comparator_observed = regression_metrics(
                comparator_y,
                comparator_pred,
            )
            (
                delta_s_mean,
                delta_s_low,
                delta_s_high,
            ) = finite_percentile_interval(
                bootstrap_contrasts[contrast_id]["delta_spearman"]
            )
            (
                delta_r_mean,
                delta_r_low,
                delta_r_high,
            ) = finite_percentile_interval(
                bootstrap_contrasts[contrast_id]["delta_rmse"]
            )
            all_rows.append(
                {
                    "record_type": "contrast",
                    "protocol": protocol,
                    "contrast_id": contrast_id,
                    "first_model": first_model,
                    "comparator_model": comparator_model,
                    "cluster_unit": cluster_col,
                    "n_clusters": int(len(unique_clusters)),
                    "n_bootstrap": int(n_bootstrap),
                    "delta_direction": (
                        "delta_spearman=first-comparator; "
                        "delta_rmse=comparator-first; "
                        "positive favors first"
                    ),
                    "delta_spearman_observed": (
                        first_observed["spearman"]
                        - comparator_observed["spearman"]
                    ),
                    "delta_spearman_mean": delta_s_mean,
                    "delta_spearman_ci_low": delta_s_low,
                    "delta_spearman_ci_high": delta_s_high,
                    "delta_rmse_observed": (
                        comparator_observed["rmse"]
                        - first_observed["rmse"]
                    ),
                    "delta_rmse_mean": delta_r_mean,
                    "delta_rmse_ci_low": delta_r_low,
                    "delta_rmse_ci_high": delta_r_high,
                }
            )
    return pd.DataFrame(all_rows)


def atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def read_progress_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        return pd.read_csv(path).to_dict(orient="records")
    except pd.errors.EmptyDataError:
        return []


def record_matches_split(record: dict[str, Any], split: OuterSplit) -> bool:
    return bool(
        int(record["repeat"]) == split.repeat
        and int(record["outer_fold"]) == split.fold
    )


def checkpoint_is_complete(
    split: OuterSplit,
    prediction_rows: list[dict[str, Any]],
    fold_metric_rows: list[dict[str, Any]],
    tuning_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    skip_label_shuffle: bool,
) -> bool:
    local_predictions = [
        row for row in prediction_rows if record_matches_split(row, split)
    ]
    local_fold_metrics = [
        row for row in fold_metric_rows if record_matches_split(row, split)
    ]
    local_tuning = [
        row for row in tuning_rows if record_matches_split(row, split)
    ]
    local_selected = [
        row for row in selected_rows if record_matches_split(row, split)
    ]
    expected_models = output_model_ids(skip_label_shuffle)
    expected_tuned_models = list(MODEL_ORDER[:-1])
    expected_rows = set(split.test_idx.astype(int).tolist())
    if len(local_predictions) != len(expected_models) * len(expected_rows):
        return False
    prediction_frame = pd.DataFrame(local_predictions)
    if prediction_frame.duplicated(["model_id", "row_index"]).any():
        return False
    if not np.all(
        np.isfinite(pd.to_numeric(prediction_frame["y_pred"], errors="coerce"))
    ):
        return False
    for model_id in expected_models:
        model_rows = prediction_frame[
            prediction_frame["model_id"] == model_id
        ]
        if set(model_rows["row_index"].astype(int)) != expected_rows:
            return False
    if len(local_fold_metrics) != len(expected_models):
        return False
    if {
        str(row["model_id"]) for row in local_fold_metrics
    } != set(expected_models):
        return False
    if len(local_selected) != len(expected_tuned_models):
        return False
    if {
        str(row["model_id"]) for row in local_selected
    } != set(expected_tuned_models):
        return False
    expected_tuning = {
        (model_id, str(params["param_id"]))
        for model_id in expected_tuned_models
        for params in model_param_grid(model_id)
    }
    observed_tuning = {
        (str(row["model_id"]), str(row["param_id"]))
        for row in local_tuning
    }
    return bool(
        len(local_tuning) == len(expected_tuning)
        and observed_tuning == expected_tuning
    )


def discard_split_records(
    records: list[dict[str, Any]],
    split: OuterSplit,
) -> list[dict[str, Any]]:
    return [
        row for row in records if not record_matches_split(row, split)
    ]


def write_progress_tables(
    output_dir: Path,
    protocol: str,
    prediction_rows: list[dict[str, Any]],
    fold_metric_rows: list[dict[str, Any]],
    tuning_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
) -> None:
    atomic_to_csv(
        pd.DataFrame(prediction_rows),
        output_dir / f"{protocol}_cross_fitted_predictions.csv",
    )
    atomic_to_csv(
        pd.DataFrame(fold_metric_rows),
        output_dir / f"{protocol}_outer_fold_metrics.csv",
    )
    atomic_to_csv(
        pd.DataFrame(tuning_rows),
        output_dir / f"{protocol}_inner_tuning_metrics.csv",
    )
    atomic_to_csv(
        pd.DataFrame(selected_rows),
        output_dir / f"{protocol}_selected_hyperparameters.csv",
    )


def run_protocol(
    protocol: str,
    args: argparse.Namespace,
    rows: pd.DataFrame,
    y: np.ndarray,
    morgan: np.ndarray,
    descriptors: np.ndarray,
    scaffold_ids: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:
    outer_splits = make_outer_splits(
        protocol,
        rows,
        scaffold_ids,
        args.outer_folds,
        args.outer_repeats,
        args.seed,
    )
    if args.max_outer_splits is not None:
        outer_splits = outer_splits[: args.max_outer_splits]

    prediction_rows = (
        read_progress_records(
            output_dir / f"{protocol}_cross_fitted_predictions.csv"
        )
        if args.resume
        else []
    )
    fold_metric_rows = (
        read_progress_records(
            output_dir / f"{protocol}_outer_fold_metrics.csv"
        )
        if args.resume
        else []
    )
    tuning_rows = (
        read_progress_records(
            output_dir / f"{protocol}_inner_tuning_metrics.csv"
        )
        if args.resume
        else []
    )
    selected_rows = (
        read_progress_records(
            output_dir / f"{protocol}_selected_hyperparameters.csv"
        )
        if args.resume
        else []
    )
    model_ids = list(MODEL_ORDER[:-1])

    start = time.perf_counter()
    for outer_number, split in enumerate(outer_splits, start=1):
        if args.resume and checkpoint_is_complete(
            split,
            prediction_rows,
            fold_metric_rows,
            tuning_rows,
            selected_rows,
            args.skip_label_shuffle,
        ):
            print(
                (
                    f"[RESUME] protocol={protocol} repeat={split.repeat} "
                    f"fold={split.fold} checkpoint complete; skipping"
                ),
                flush=True,
            )
            continue
        if args.resume:
            prediction_rows = discard_split_records(
                prediction_rows,
                split,
            )
            fold_metric_rows = discard_split_records(
                fold_metric_rows,
                split,
            )
            tuning_rows = discard_split_records(tuning_rows, split)
            selected_rows = discard_split_records(selected_rows, split)
        print(
            (
                f"[OUTER] protocol={protocol} repeat={split.repeat} "
                f"fold={split.fold} ({outer_number}/{len(outer_splits)}) "
                f"train={len(split.train_idx)} test={len(split.test_idx)}"
            ),
            flush=True,
        )
        inner_groups = (
            rows["canonical_smiles"].to_numpy(dtype=object)
            if protocol == "compound"
            else scaffold_ids
        )
        selected, local_tuning = select_model_params(
            model_ids,
            rows,
            y,
            morgan,
            descriptors,
            split.train_idx,
            inner_groups,
            args.inner_folds,
            seed=args.seed + split.repeat * 10_000 + split.fold * 100,
            n_estimators=args.n_estimators,
            n_jobs=args.n_jobs,
        )
        for row in local_tuning:
            tuning_rows.append(
                {
                    "protocol": protocol,
                    "repeat": split.repeat,
                    "outer_fold": split.fold,
                    **row,
                }
            )
        for model_id, params in selected.items():
            selected_rows.append(
                {
                    "protocol": protocol,
                    "repeat": split.repeat,
                    "outer_fold": split.fold,
                    "model_id": model_id,
                    **params,
                }
            )

        fold_features, fold_feature_meta = build_fold_features(
            rows,
            morgan,
            descriptors,
            split.train_idx,
            split.test_idx,
        )
        similarities = tanimoto_matrix(
            morgan[split.test_idx],
            morgan[split.train_idx],
        )
        max_tanimoto = similarities.max(axis=1)
        train_rows = rows.iloc[split.train_idx]
        train_targets = train_rows["target_protein"]
        target_support_rows = train_targets.value_counts()
        target_support_compounds = train_rows.groupby("target_protein")[
            "canonical_smiles"
        ].nunique()
        recruiter_target_support_compounds = (
            train_rows.assign(
                recruiter_target_key=key_series(
                    train_rows,
                    ("recruiting_protein", "target_protein"),
                ).to_numpy()
            )
            .groupby("recruiter_target_key")["canonical_smiles"]
            .nunique()
        )
        train_sources = set(train_rows["source_database"].astype(str))
        train_scaffolds = set(scaffold_ids[split.train_idx])
        output_models = output_model_ids(args.skip_label_shuffle)

        for model_position, model_id in enumerate(output_models):
            permutation_diagnostics: dict[str, float | int] = {}
            model_seed = (
                args.seed
                + split.repeat * 100_000
                + split.fold * 1_000
                + model_position
            )
            if model_id == "full_context_label_shuffle":
                params = selected["full_context_extra_trees"]
                shuffled_y, permutation_diagnostics = (
                    within_domain_label_permutation(
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
                    seed=model_seed,
                    n_estimators=args.n_estimators,
                    n_jobs=args.n_jobs,
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
                    seed=model_seed,
                    n_estimators=args.n_estimators,
                    n_jobs=args.n_jobs,
                )
                param_id = str(params["param_id"])

            metrics = regression_metrics(y[split.test_idx], pred)
            fold_metric_rows.append(
                {
                    "protocol": protocol,
                    "repeat": split.repeat,
                    "outer_fold": split.fold,
                    "model_id": model_id,
                    "param_id": param_id,
                    "n_train_rows": int(len(split.train_idx)),
                    "n_test_rows": int(len(split.test_idx)),
                    "n_test_unique_smiles": int(
                        rows.iloc[split.test_idx]["canonical_smiles"].nunique()
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
                recruiter_target_value = (
                    f"{row['recruiting_protein']}||{target_value}"
                )
                source_value = str(row["source_database"])
                prediction_rows.append(
                    {
                        "protocol": protocol,
                        "repeat": split.repeat,
                        "outer_fold": split.fold,
                        "row_index": int(row_idx),
                        "qc_id": str(row["qc_id"]),
                        "model_id": model_id,
                        "param_id": param_id,
                        "y_true": float(y[row_idx]),
                        "y_pred": float(pred[local_idx]),
                        "canonical_smiles": str(row["canonical_smiles"]),
                        "scaffold_id": str(scaffold_ids[row_idx]),
                        "source_database": source_value,
                        "recruiting_protein": str(row["recruiting_protein"]),
                        "target_protein": target_value,
                        "cell_line": str(row["cell_line"]),
                        "max_train_tanimoto": float(max_tanimoto[local_idx]),
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
                    }
                )

        write_progress_tables(
            output_dir,
            protocol,
            prediction_rows,
            fold_metric_rows,
            tuning_rows,
            selected_rows,
        )
        elapsed = time.perf_counter() - start
        print(
            f"[OUTER] completed {outer_number}/{len(outer_splits)} in {elapsed:.1f}s",
            flush=True,
        )

    predictions = pd.DataFrame(prediction_rows)
    return predictions


def environment_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
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


def scientific_configuration(
    args: argparse.Namespace,
    data_sha256: str,
    script_sha256: str,
    protocol_document_sha256: str,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocols": list(args.protocols),
        "data_sha256": data_sha256,
        "script_sha256": script_sha256,
        "protocol_document_sha256": protocol_document_sha256,
        "outer_folds": int(args.outer_folds),
        "outer_repeats": int(args.outer_repeats),
        "inner_folds": int(args.inner_folds),
        "n_estimators": int(args.n_estimators),
        "n_jobs": int(args.n_jobs),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "seed": int(args.seed),
        "max_outer_splits": args.max_outer_splits,
        "skip_label_shuffle": bool(args.skip_label_shuffle),
        "dry_run": bool(args.dry_run),
        "runtime_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn_version,
            "rdkit": rdBase.rdkitVersion,
        },
    }


def frozen_protocol_match(
    args: argparse.Namespace,
    data_sha256: str,
) -> bool:
    return bool(
        args.protocols == list(PROTOCOL_ORDER)
        and data_sha256 == FROZEN_DATA_SHA256
        and args.outer_folds == 5
        and args.outer_repeats == 5
        and args.inner_folds == 4
        and args.n_estimators == 600
        and args.n_jobs == 10
        and args.bootstrap_replicates == 10_000
        and args.seed == DEFAULT_SEED
        and args.max_outer_splits is None
        and not args.skip_label_shuffle
        and not args.dry_run
    )


def clear_known_run_artifacts(output_dir: Path) -> int:
    filenames = {
        "run_manifest.json",
        "analysis_index.csv",
        "context_sanitization_audit.csv",
        "all_cross_fitted_predictions.csv",
        "repeat_averaged_cross_fitted_predictions.csv",
        "metrics_by_repeat.csv",
        "summary_metrics.csv",
        "metrics_on_repeat_averaged_predictions.csv",
        "metrics_on_canonical_smiles_aggregates.csv",
        "applicability_by_row.csv",
        "risk_coverage_metrics.csv",
        "paired_cluster_bootstrap.csv",
    }
    for protocol in PROTOCOL_ORDER:
        filenames.update(
            {
                f"{protocol}_cross_fitted_predictions.csv",
                f"{protocol}_outer_fold_metrics.csv",
                f"{protocol}_inner_tuning_metrics.csv",
                f"{protocol}_selected_hyperparameters.csv",
            }
        )
    removed = 0
    for filename in filenames:
        path = output_dir / filename
        temporary_path = path.with_name(f".{path.name}.tmp")
        for candidate in (path, temporary_path):
            if candidate.is_file():
                candidate.unlink()
                removed += 1
    return removed


def main() -> None:
    args = parse_args()
    data_path = args.data_file.resolve()
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    if not PROTOCOL_DOCUMENT.exists():
        raise FileNotFoundError(PROTOCOL_DOCUMENT)
    data_sha256 = sha256_file(data_path)
    script_sha256 = sha256_file(Path(__file__).resolve())
    protocol_document_sha256 = sha256_file(PROTOCOL_DOCUMENT)
    configuration = scientific_configuration(
        args,
        data_sha256,
        script_sha256,
        protocol_document_sha256,
    )

    output_dir = args.output_dir.resolve()
    existing_manifest: dict[str, Any] | None = None
    output_is_nonempty = bool(
        output_dir.exists() and any(output_dir.iterdir())
    )
    if output_is_nonempty and args.resume:
        manifest_path = output_dir / "run_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Cannot resume without {manifest_path}"
            )
        existing_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if existing_manifest.get("scientific_configuration") != configuration:
            raise ValueError(
                "Resume refused: data, code, protocol document, runtime "
                "versions, or scientific arguments differ from the checkpoint."
            )
    elif output_is_nonempty and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use a new directory, --resume, or --overwrite."
        )
    elif output_is_nonempty and args.overwrite:
        removed = clear_known_run_artifacts(output_dir)
        print(
            f"[OVERWRITE] removed {removed} known prior-run artifacts",
            flush=True,
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_started = time.time()
    rows, context_audit = load_rows(data_path)
    y = rows["pDC50"].to_numpy(dtype=np.float32)
    scaffold_ids = build_scaffold_ids(rows["canonical_smiles"])
    rows["confirmatory_scaffold_id"] = scaffold_ids
    print(
        (
            f"[DATA] rows={len(rows)} unique_smiles="
            f"{rows['canonical_smiles'].nunique()} scaffolds="
            f"{len(np.unique(scaffold_ids))}"
        ),
        flush=True,
    )
    print("[FEAT] building Morgan fingerprints", flush=True)
    morgan = build_morgan_matrix(rows["canonical_smiles"])
    print("[FEAT] building RDKit descriptors", flush=True)
    descriptors, descriptor_names, descriptor_failures = build_descriptor_matrix(
        rows["canonical_smiles"]
    )
    context_audit.to_csv(output_dir / "context_sanitization_audit.csv", index=False)
    matches_frozen_protocol = frozen_protocol_match(args, data_sha256)
    analysis_mode = (
        "dry_run"
        if args.dry_run
        else (
            "smoke_test"
            if args.max_outer_splits is not None
            else ("confirmatory" if matches_frozen_protocol else "custom")
        )
    )
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "dry_run" if args.dry_run else "running",
        "analysis_mode": analysis_mode,
        "frozen_protocol_match": matches_frozen_protocol,
        "data_file": str(data_path),
        "data_sha256": data_sha256,
        "frozen_data_sha256": FROZEN_DATA_SHA256,
        "frozen_data_hash_match": data_sha256 == FROZEN_DATA_SHA256,
        "script_file": str(Path(__file__).resolve()),
        "script_sha256": script_sha256,
        "protocol_document": str(PROTOCOL_DOCUMENT),
        "protocol_document_sha256": protocol_document_sha256,
        "scientific_configuration": configuration,
        "resume_count": (
            int(existing_manifest.get("resume_count", 0)) + 1
            if existing_manifest is not None
            else 0
        ),
        "run_session_started_unix": run_started,
        "n_rows": int(len(rows)),
        "n_unique_smiles": int(rows["canonical_smiles"].nunique()),
        "n_scaffolds": int(len(np.unique(scaffold_ids))),
        "endpoint": "pDC50 = 9 - log10(DC50_nM)",
        "independent_units": {
            "compound_protocol": "canonical_smiles group",
            "primary_scaffold_protocol": "Bemis-Murcko scaffold group",
            "folds_and_seeds_are_independent_replicates": False,
        },
        "primary_metric": PRIMARY_METRIC,
        "reference_model": REFERENCE_MODEL,
        "models": {
            model_id: model_param_grid(model_id)
            for model_id in MODEL_ORDER
            if model_id
            not in {"full_context_label_shuffle"}
        },
        "label_shuffle_control": not args.skip_label_shuffle,
        "feature_manifest": feature_manifest(
            descriptor_names,
            descriptor_failures,
        ),
        "args": vars(args),
        "environment": environment_manifest(args),
        "boundaries": [
            "Historical fixed-test labels are not used.",
            "No target encoding is used in this CPU v1 core.",
            "No protein-language-model feature is used pending sequence mapping audit.",
            "The stratified label shuffle is a sanity control, not a formal permutation test.",
            "This is retrospective internal validation, not prospective external validation.",
        ],
    }
    write_json(output_dir / "run_manifest.json", manifest)
    rows[
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
    ].to_csv(output_dir / "analysis_index.csv", index=False)
    if args.dry_run:
        print(f"[DONE] dry run artifacts written to {output_dir}", flush=True)
        return

    all_predictions: list[pd.DataFrame] = []
    for protocol in args.protocols:
        protocol_predictions = run_protocol(
            protocol,
            args,
            rows,
            y,
            morgan,
            descriptors,
            scaffold_ids,
            output_dir,
        )
        all_predictions.append(protocol_predictions)
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(
        output_dir / "all_cross_fitted_predictions.csv",
        index=False,
    )
    repeat_metrics, summary = summarize_predictions(predictions)
    repeat_metrics.to_csv(output_dir / "metrics_by_repeat.csv", index=False)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    averaged_predictions = average_cross_fitted_predictions(predictions)
    averaged_predictions.to_csv(
        output_dir / "repeat_averaged_cross_fitted_predictions.csv",
        index=False,
    )
    averaged_metrics = metrics_on_repeat_averaged_predictions(
        averaged_predictions
    )
    averaged_metrics.to_csv(
        output_dir / "metrics_on_repeat_averaged_predictions.csv",
        index=False,
    )
    compound_aggregate_metrics = metrics_on_canonical_smiles_aggregates(
        averaged_predictions
    )
    compound_aggregate_metrics.to_csv(
        output_dir / "metrics_on_canonical_smiles_aggregates.csv",
        index=False,
    )
    applicability = build_applicability_table(averaged_predictions)
    applicability.to_csv(
        output_dir / "applicability_by_row.csv",
        index=False,
    )
    coverage_metrics = risk_coverage_metrics(
        applicability,
        output_model_ids(args.skip_label_shuffle),
    )
    coverage_metrics.to_csv(
        output_dir / "risk_coverage_metrics.csv",
        index=False,
    )

    raw_predictions_complete = complete_repeats(
        predictions,
        len(rows),
        args.protocols,
        args.outer_repeats,
        args.skip_label_shuffle,
    )
    averaged_predictions_complete = complete_repeat_averages(
        averaged_predictions,
        len(rows),
        args.protocols,
        args.outer_repeats,
        args.skip_label_shuffle,
    )
    manifest["completeness"] = {
        "raw_cross_fitted_predictions": raw_predictions_complete,
        "repeat_averaged_predictions": averaged_predictions_complete,
    }
    if matches_frozen_protocol and not (
        raw_predictions_complete and averaged_predictions_complete
    ):
        manifest["status"] = "failed_incomplete"
        write_json(output_dir / "run_manifest.json", manifest)
        raise RuntimeError(
            "Frozen confirmatory run is incomplete; refusing to mark it complete."
        )

    bootstrap = pd.DataFrame()
    if (
        args.max_outer_splits is None
        and args.bootstrap_replicates > 0
        and raw_predictions_complete
        and averaged_predictions_complete
    ):
        bootstrap = cluster_bootstrap_summary(
            averaged_predictions,
            args.bootstrap_replicates,
            args.seed,
            cluster_col="scaffold_id",
        )
        bootstrap.to_csv(output_dir / "paired_cluster_bootstrap.csv", index=False)
    elif args.bootstrap_replicates > 0:
        print(
            "[WARN] bootstrap skipped because cross-fitted repeats are incomplete",
            flush=True,
        )
    if matches_frozen_protocol and bootstrap.empty:
        manifest["status"] = "failed_missing_bootstrap"
        write_json(output_dir / "run_manifest.json", manifest)
        raise RuntimeError(
            "Frozen confirmatory run has no bootstrap summary; "
            "refusing to mark it complete."
        )

    manifest["status"] = "complete"
    manifest["elapsed_seconds"] = float(time.time() - run_started)
    manifest["artifacts"] = {
        "analysis_index": "analysis_index.csv",
        "predictions": "all_cross_fitted_predictions.csv",
        "repeat_averaged_predictions": (
            "repeat_averaged_cross_fitted_predictions.csv"
        ),
        "metrics_by_repeat": "metrics_by_repeat.csv",
        "summary_metrics": "summary_metrics.csv",
        "metrics_on_repeat_averaged_predictions": (
            "metrics_on_repeat_averaged_predictions.csv"
        ),
        "metrics_on_canonical_smiles_aggregates": (
            "metrics_on_canonical_smiles_aggregates.csv"
        ),
        "applicability_by_row": "applicability_by_row.csv",
        "risk_coverage_metrics": "risk_coverage_metrics.csv",
        "paired_cluster_bootstrap": (
            "paired_cluster_bootstrap.csv" if not bootstrap.empty else None
        ),
        "context_sanitization_audit": "context_sanitization_audit.csv",
    }
    write_json(output_dir / "run_manifest.json", manifest)
    print(
        (
            f"[DONE] protocols={args.protocols} elapsed="
            f"{manifest['elapsed_seconds']:.1f}s output={output_dir}"
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
