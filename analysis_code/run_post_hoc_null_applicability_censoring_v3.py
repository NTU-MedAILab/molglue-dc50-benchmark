#!/usr/bin/env python
"""Run the corrected v3 post-hoc null, applicability, and censoring audits.

This CPU-only script is deliberately isolated from the formal confirmatory
runner. It reads the saved public-scope core predictions and split identities,
refits only the two matched ExtraTrees models for a conditional label-
permutation null, stratifies existing predictions by fixed chemical-similarity
bins, and produces descriptive censoring/selection summaries. Version 3 keeps
the float32 response correction and fixes the OOD applicability bootstrap to
retain a metric-specific fixed eligible-domain set in every replicate.
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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import numpy as np
import pandas as pd
import scipy
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn import __version__ as sklearn_version

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_confirmatory_cpu_v1 as core  # noqa: E402


PROTOCOL_VERSION = "post_hoc_null_applicability_censoring_v3.0"
MASTER_PROTOCOL = (
    PROJECT_DIR / "docs" / "post_hoc_computational_extension_master_protocol_v1.md"
)
MASTER_PROTOCOL_SHA256 = (
    "7dee28f45e3ddf8d6299622a8e494c50492b3b2614f39ce8b2fdd2a1be754ff8"
)
LOCAL_PROTOCOL = (
    PROJECT_DIR / "docs" / "post_hoc_null_applicability_censoring_protocol_v3.md"
)
LOCAL_PROTOCOL_SHA256 = (
    "84178cde0edfc0ff31be3d7e12135351f95a2578b9ac69b61d9a5d517caa0dc5"
)
FLOAT32_CORRECTION_NOTE = (
    PROJECT_DIR / "docs" / "post_hoc_float32_response_correction_v2.md"
)
FLOAT32_CORRECTION_NOTE_SHA256 = (
    "75a8b0e4f894cead9d72039eab9e6162a89d2a20c79afceda8a2aae1ffd78f3f"
)
APPLICABILITY_CORRECTION_NOTE = (
    PROJECT_DIR
    / "docs"
    / "post_hoc_applicability_fixed_domain_correction_v3.md"
)
APPLICABILITY_CORRECTION_NOTE_SHA256 = (
    "9c2d8f90fa546480a37b6e7546cbd9e9a740916325e5dec973f9fa7f8ef899ce"
)
EXPECTED_CORE_IMPLEMENTATION_SHA256 = (
    "ac1a57203b43d74023521b76cfaef28b958a803dbc6d9fb2866e7550f636995c"
)

CORE_DATA = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "all_molglue_dc50_qc_train_test_standardized_context.csv"
)
PARSED_DATA = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "all_molglue_dc50_parsed_records.csv"
)
INTERNAL_DIR = PROJECT_DIR / "reports" / "confirmatory_cpu_v1"
INTERNAL_PREDICTIONS = INTERNAL_DIR / "scaffold_cross_fitted_predictions.csv"
INTERNAL_PARAMETERS = INTERNAL_DIR / "scaffold_selected_hyperparameters.csv"
INTERNAL_APPLICABILITY = INTERNAL_DIR / "applicability_by_row.csv"
OOD_PREDICTIONS = (
    PROJECT_DIR
    / "reports"
    / "confirmatory_ood_cpu_v1"
    / "ood_predictions.csv"
)
STRICT_OOD_PREDICTIONS = (
    PROJECT_DIR
    / "reports"
    / "post_hoc_strict_domain_scaffold_ood_cpu_v1"
    / "strict_ood_predictions.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "reports" / "post_hoc_null_applicability_censoring_v3"
)

MODEL_IDS = ("chemistry_extra_trees", "full_context_extra_trees")
SIMILARITY_EDGES = (0.0, 0.4, 0.6, 0.8, 1.000001)
SIMILARITY_LABELS = (
    "0.0_to_lt_0.4",
    "0.4_to_lt_0.6",
    "0.6_to_lt_0.8",
    "0.8_to_1.000001",
)
EXPECTED_CORE_SHA256 = (
    "e9494246bcbd198bc09d5b0d09721910c33f70f46ae859b76377a13775e3e4b2"
)
EXPECTED_PARSED_SHA256 = (
    "ae1a16613b41ce84f92f539dbd31f16ada272ac5e401250acc7e97fd5e8bd970"
)
DEFAULT_SEED = 260531
METRICS = ("spearman", "rmse", "mae")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--max-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume complete permutation IDs in a matching output directory.",
    )
    args = parser.parse_args()
    if args.n_permutations < 1:
        parser.error("n-permutations must be >= 1")
    if args.n_estimators < 10:
        parser.error("n-estimators must be >= 10")
    if not 1 <= args.n_jobs <= 4:
        parser.error("n-jobs must be between 1 and 4")
    if args.bootstrap_replicates < 1:
        parser.error("bootstrap-replicates must be >= 1")
    if not 1 <= args.max_repeats <= 5:
        parser.error("max-repeats must be between 1 and 5")
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


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_ready)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def finite_interval(values: Iterable[float]) -> tuple[float, float, float, float, int]:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return (float("nan"),) * 4 + (0,)
    sd = float(np.std(finite, ddof=1)) if len(finite) >= 2 else float("nan")
    return (
        float(np.mean(finite)),
        sd,
        float(np.percentile(finite, 2.5)),
        float(np.percentile(finite, 97.5)),
        int(len(finite)),
    )


def requested_input_paths() -> list[Path]:
    return [
        CORE_DATA,
        PARSED_DATA,
        INTERNAL_PREDICTIONS,
        INTERNAL_PARAMETERS,
        INTERNAL_APPLICABILITY,
        OOD_PREDICTIONS,
        STRICT_OOD_PREDICTIONS,
        MASTER_PROTOCOL,
        LOCAL_PROTOCOL,
        FLOAT32_CORRECTION_NOTE,
        APPLICABILITY_CORRECTION_NOTE,
        Path(__file__).resolve(),
        SCRIPT_DIR / "run_confirmatory_cpu_v1.py",
    ]


def prepare_output_directory(args: argparse.Namespace) -> None:
    forbidden_markers = [
        args.output_dir / "SUPERSEDED_DO_NOT_USE.md",
        args.output_dir / "ABORTED_DO_NOT_RESUME.md",
    ]
    for marker in forbidden_markers:
        if marker.exists():
            raise RuntimeError(f"Refusing marked output directory: {marker}")
    if not args.output_dir.name.endswith("_v3"):
        raise ValueError(
            "Corrected outputs require a directory name ending in '_v3'; "
            "v1/v2 directories cannot be resumed or overwritten."
        )
    if args.output_dir.exists():
        existing = list(args.output_dir.iterdir())
        if existing and not args.resume:
            raise FileExistsError(
                f"Output directory is non-empty: {args.output_dir}. "
                "Use a new directory or --resume."
            )
        if existing and args.resume:
            validate_resume_identity(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)


def validate_resume_identity(args: argparse.Namespace) -> None:
    manifest_path = args.output_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            "Fail-closed v3 resume requires an existing run_manifest.json"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("Existing output protocol version is not v3")
    if manifest.get("master_protocol_sha256") != MASTER_PROTOCOL_SHA256:
        raise RuntimeError("Existing output master-protocol identity mismatch")
    if manifest.get("local_protocol_sha256") != LOCAL_PROTOCOL_SHA256:
        raise RuntimeError("Existing output local-protocol identity mismatch")
    if (
        manifest.get("float32_correction_sha256")
        != FLOAT32_CORRECTION_NOTE_SHA256
    ):
        raise RuntimeError("Existing output float32-correction mismatch")
    if (
        manifest.get("applicability_correction_sha256")
        != APPLICABILITY_CORRECTION_NOTE_SHA256
    ):
        raise RuntimeError("Existing output applicability-correction mismatch")
    config = manifest.get("config", {})
    expected = {
        "n_estimators": args.n_estimators,
        "n_jobs": args.n_jobs,
        "bootstrap_replicates": args.bootstrap_replicates,
        "max_repeats": args.max_repeats,
        "seed": args.seed,
        "training_response_dtype": "float32",
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Existing output scientific configuration mismatch: {mismatches}"
        )
    existing_n = int(config.get("n_permutations", -1))
    if existing_n == args.n_permutations:
        return
    merge_path = args.output_dir / "permutation_shard_merge_manifest.json"
    if args.n_permutations != 100 or not merge_path.is_file():
        raise RuntimeError(
            "Changing n_permutations on resume requires a validated formal "
            "v3 100-ID shard merge"
        )
    merge = json.loads(merge_path.read_text(encoding="utf-8"))
    if (
        merge.get("status") != "PASS"
        or merge.get("protocol_version") != PROTOCOL_VERSION
        or int(merge.get("n_permutations", -1)) != 100
        or merge.get("local_protocol_sha256") != LOCAL_PROTOCOL_SHA256
        or merge.get("applicability_correction_sha256")
        != APPLICABILITY_CORRECTION_NOTE_SHA256
    ):
        raise RuntimeError("Formal v3 shard-merge manifest is not eligible")


def software_versions() -> dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn_version,
        "rdkit": rdBase.rdkitVersion,
    }


def validate_frozen_identities() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for name, path, expected in [
        ("core_data", CORE_DATA, EXPECTED_CORE_SHA256),
        ("parsed_data", PARSED_DATA, EXPECTED_PARSED_SHA256),
        ("master_protocol", MASTER_PROTOCOL, MASTER_PROTOCOL_SHA256),
        ("local_protocol_v3", LOCAL_PROTOCOL, LOCAL_PROTOCOL_SHA256),
        (
            "float32_correction_v2",
            FLOAT32_CORRECTION_NOTE,
            FLOAT32_CORRECTION_NOTE_SHA256,
        ),
        (
            "applicability_correction_v3",
            APPLICABILITY_CORRECTION_NOTE,
            APPLICABILITY_CORRECTION_NOTE_SHA256,
        ),
        (
            "confirmatory_implementation",
            SCRIPT_DIR / "run_confirmatory_cpu_v1.py",
            EXPECTED_CORE_IMPLEMENTATION_SHA256,
        ),
    ]:
        observed = sha256_file(path)
        checks.append(
            {
                "check": f"{name}_sha256",
                "status": "PASS" if observed == expected else "FAIL",
                "detail": f"observed={observed}; expected={expected}",
            }
        )
        if observed != expected:
            raise RuntimeError(
                f"Frozen identity mismatch for {name}: {observed} != {expected}"
            )
    return checks


def load_internal_design(
    rows: pd.DataFrame,
    max_repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], np.ndarray]:
    predictions = pd.read_csv(INTERNAL_PREDICTIONS)
    parameters = pd.read_csv(INTERNAL_PARAMETERS)
    predictions = predictions[
        (predictions["protocol"] == "scaffold")
        & predictions["model_id"].isin(MODEL_IDS)
        & (predictions["repeat"] <= max_repeats)
    ].copy()
    parameters = parameters[
        (parameters["protocol"] == "scaffold")
        & parameters["model_id"].isin(MODEL_IDS)
        & (parameters["repeat"] <= max_repeats)
    ].copy()
    if predictions.empty or parameters.empty:
        raise ValueError("Saved internal predictions or parameters are empty")

    assignment = predictions[
        predictions["model_id"] == MODEL_IDS[0]
    ][
        [
            "repeat",
            "outer_fold",
            "row_index",
            "scaffold_id",
        ]
    ].copy()
    if assignment.duplicated(["repeat", "row_index"]).any():
        raise ValueError("Duplicate saved split assignment")
    expected_rows = set(range(len(rows)))
    splits: list[dict[str, Any]] = []
    all_indices = np.arange(len(rows), dtype=np.int64)
    scaffold_by_row = (
        assignment.drop_duplicates("row_index")
        .set_index("row_index")["scaffold_id"]
        .reindex(all_indices)
    )
    if scaffold_by_row.isna().any():
        raise ValueError("Saved predictions do not identify every row scaffold")
    scaffold_ids = scaffold_by_row.astype(str).to_numpy(dtype=object)

    for repeat in range(1, max_repeats + 1):
        local_repeat = assignment[assignment["repeat"] == repeat]
        if set(local_repeat["row_index"].astype(int)) != expected_rows:
            raise ValueError(f"Repeat {repeat} does not cover every row once")
        for fold in sorted(local_repeat["outer_fold"].unique()):
            test_idx = np.sort(
                local_repeat.loc[
                    local_repeat["outer_fold"] == fold, "row_index"
                ].to_numpy(dtype=np.int64)
            )
            train_idx = np.setdiff1d(all_indices, test_idx, assume_unique=True)
            overlap = set(scaffold_ids[train_idx]).intersection(
                set(scaffold_ids[test_idx])
            )
            if overlap:
                raise ValueError(
                    f"Scaffold overlap in repeat={repeat}, fold={fold}"
                )
            split_params: dict[str, dict[str, Any]] = {}
            for model_id in MODEL_IDS:
                match = parameters[
                    (parameters["repeat"] == repeat)
                    & (parameters["outer_fold"] == fold)
                    & (parameters["model_id"] == model_id)
                ]
                if len(match) != 1:
                    raise ValueError(
                        f"Expected one parameter row for {repeat}/{fold}/{model_id}"
                    )
                record = match.iloc[0]
                raw_max_features = str(record["max_features"])
                max_features: str | float
                if raw_max_features == "sqrt":
                    max_features = "sqrt"
                else:
                    max_features = float(raw_max_features)
                split_params[model_id] = {
                    "param_id": str(record["param_id"]),
                    "min_samples_leaf": int(record["min_samples_leaf"]),
                    "max_features": max_features,
                }
            splits.append(
                {
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                    "params": split_params,
                }
            )
    return predictions, parameters, splits, scaffold_ids


def observed_internal_metrics(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    averaged = (
        predictions.groupby(["model_id", "row_index"], sort=False)
        .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
        .reset_index()
    )
    model_rows: list[dict[str, Any]] = []
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model_id in MODEL_IDS:
        frame = averaged[averaged["model_id"] == model_id].sort_values(
            "row_index"
        )
        arrays[model_id] = (
            frame["y_true"].to_numpy(dtype=np.float64),
            frame["y_pred"].to_numpy(dtype=np.float64),
        )
        metrics = core.regression_metrics(*arrays[model_id])
        model_rows.append(
            {
                "record_type": "model",
                "model_id": model_id,
                "n_rows": int(len(frame)),
                "spearman": metrics["spearman"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
            }
        )
    chem = dict(zip(("y", "pred"), arrays[MODEL_IDS[0]]))
    full = dict(zip(("y", "pred"), arrays[MODEL_IDS[1]]))
    if not np.array_equal(chem["y"], full["y"]):
        raise ValueError("Observed model truths are not aligned")
    chem_metrics = core.regression_metrics(chem["y"], chem["pred"])
    full_metrics = core.regression_metrics(full["y"], full["pred"])
    contrast = pd.DataFrame(
        [
            {
                "record_type": "contrast",
                "contrast_id": "full_minus_chemistry",
                "n_rows": int(len(chem["y"])),
                "delta_spearman": (
                    full_metrics["spearman"] - chem_metrics["spearman"]
                ),
                "delta_rmse": chem_metrics["rmse"] - full_metrics["rmse"],
                "delta_mae": chem_metrics["mae"] - full_metrics["mae"],
                "direction": "positive values favor full for every metric",
            }
        ]
    )
    return pd.DataFrame(model_rows), contrast


def multiset_by_stratum(
    rows: pd.DataFrame,
    fit_idx: np.ndarray,
    values: np.ndarray,
) -> dict[str, tuple[float, ...]]:
    local = rows.iloc[fit_idx]
    strata = (
        local["source_database"].astype(str)
        + "||"
        + local["target_protein"].astype(str)
    ).to_numpy(dtype=object)
    result: dict[str, tuple[float, ...]] = {}
    for stratum in np.unique(strata):
        positions = np.where(strata == stratum)[0]
        result[str(stratum)] = tuple(
            np.sort(values[positions].astype(np.float64)).tolist()
        )
    return result


def cache_fold_features(
    rows: pd.DataFrame,
    splits: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, Any]]:
    print("[FEATURE] building Morgan fingerprints and RDKit descriptors", flush=True)
    morgan = core.build_morgan_matrix(rows["canonical_smiles"])
    descriptors, descriptor_names, descriptor_failures = (
        core.build_descriptor_matrix(rows["canonical_smiles"])
    )
    cached: list[dict[str, Any]] = []
    for position, split in enumerate(splits, start=1):
        features, metadata = core.build_fold_features(
            rows,
            morgan,
            descriptors,
            split["train_idx"],
            split["test_idx"],
        )
        cached.append(
            {
                **split,
                "features": {
                    "chemistry": features["chemistry"],
                    "full": features["full"],
                },
                "feature_metadata": metadata,
            }
        )
        print(
            f"[FEATURE] cached fold {position}/{len(splits)} "
            f"(repeat={split['repeat']}, fold={split['fold']})",
            flush=True,
        )
    manifest = core.feature_manifest(descriptor_names, descriptor_failures)
    return cached, morgan, descriptors, manifest


def run_one_permutation(
    permutation_id: int,
    rows: pd.DataFrame,
    y: np.ndarray,
    morgan: np.ndarray,
    cached_splits: list[dict[str, Any]],
    n_repeats: int,
    n_estimators: int,
    n_jobs: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if y.dtype != np.dtype(np.float32):
        raise TypeError(
            f"Corrected v3 training response must be float32; observed {y.dtype}"
        )
    n_rows = len(rows)
    prediction_sums = {
        model_id: np.zeros(n_rows, dtype=np.float64) for model_id in MODEL_IDS
    }
    prediction_counts = {
        model_id: np.zeros(n_rows, dtype=np.int64) for model_id in MODEL_IDS
    }
    fold_rows: list[dict[str, Any]] = []
    for split in cached_splits:
        repeat = int(split["repeat"])
        fold = int(split["fold"])
        train_idx = split["train_idx"]
        test_idx = split["test_idx"]
        permutation_seed = (
            seed
            + permutation_id * 1_000_003
            + repeat * 10_009
            + fold * 101
        )
        original_train_y = y[train_idx]
        permuted_train_y, diagnostics = core.within_domain_label_permutation(
            rows,
            original_train_y,
            train_idx,
            permutation_seed,
        )
        if (
            original_train_y.dtype != np.dtype(np.float32)
            or permuted_train_y.dtype != np.dtype(np.float32)
        ):
            raise TypeError(
                "Corrected v3 original and permuted training responses "
                "must both remain float32"
            )
        before = multiset_by_stratum(
            rows, train_idx, original_train_y
        )
        after = multiset_by_stratum(rows, train_idx, permuted_train_y)
        if before != after:
            raise RuntimeError("Within-stratum label multiset changed")
        for model_id in MODEL_IDS:
            pred = core.fit_predict_model(
                model_id=model_id,
                params=split["params"][model_id],
                rows=rows,
                y=y,
                morgan=morgan,
                fold_features=split["features"],
                fit_idx=train_idx,
                eval_idx=test_idx,
                seed=permutation_seed,
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                override_train_y=permuted_train_y,
            )
            if len(pred) != len(test_idx) or not np.all(np.isfinite(pred)):
                raise RuntimeError("Non-finite or incomplete permutation prediction")
            prediction_sums[model_id][test_idx] += pred
            prediction_counts[model_id][test_idx] += 1
        fold_rows.append(
            {
                "permutation_id": int(permutation_id),
                "repeat": repeat,
                "outer_fold": fold,
                "permutation_seed": int(permutation_seed),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                **diagnostics,
                "within_stratum_multiset_preserved": True,
            }
        )

    model_rows: list[dict[str, Any]] = []
    model_metrics: dict[str, dict[str, float]] = {}
    for model_id in MODEL_IDS:
        counts = prediction_counts[model_id]
        if not np.all(counts == n_repeats):
            raise RuntimeError(
                f"Permutation {permutation_id} has incomplete row coverage "
                f"for {model_id}: {np.unique(counts, return_counts=True)}"
            )
        averaged = prediction_sums[model_id] / counts
        metrics = core.regression_metrics(y, averaged)
        model_metrics[model_id] = metrics
        model_rows.append(
            {
                "permutation_id": int(permutation_id),
                "model_id": model_id,
                "n_rows": int(n_rows),
                "n_repeats": int(n_repeats),
                "spearman": metrics["spearman"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
            }
        )
    chemistry = model_metrics[MODEL_IDS[0]]
    full = model_metrics[MODEL_IDS[1]]
    contrast = pd.DataFrame(
        [
            {
                "permutation_id": int(permutation_id),
                "contrast_id": "full_minus_chemistry",
                "n_rows": int(n_rows),
                "n_repeats": int(n_repeats),
                "delta_spearman": full["spearman"] - chemistry["spearman"],
                "delta_rmse": chemistry["rmse"] - full["rmse"],
                "delta_mae": chemistry["mae"] - full["mae"],
            }
        ]
    )
    return pd.DataFrame(model_rows), contrast, pd.DataFrame(fold_rows)


def load_resume_tables(
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[int]]:
    model_path = output_dir / "permutation_model_metrics.csv"
    contrast_path = output_dir / "permutation_contrast_metrics.csv"
    diagnostic_path = output_dir / "permutation_fold_diagnostics.csv"
    model = pd.read_csv(model_path) if model_path.exists() else pd.DataFrame()
    contrast = (
        pd.read_csv(contrast_path) if contrast_path.exists() else pd.DataFrame()
    )
    diagnostics = (
        pd.read_csv(diagnostic_path)
        if diagnostic_path.exists()
        else pd.DataFrame()
    )
    complete: set[int] = set()
    if not model.empty and not contrast.empty and not diagnostics.empty:
        for permutation_id in sorted(
            set(model["permutation_id"].astype(int))
            & set(contrast["permutation_id"].astype(int))
            & set(diagnostics["permutation_id"].astype(int))
        ):
            if (
                set(
                    model.loc[
                        model["permutation_id"] == permutation_id, "model_id"
                    ].astype(str)
                )
                == set(MODEL_IDS)
                and len(
                    contrast[
                        contrast["permutation_id"] == permutation_id
                    ]
                )
                == 1
            ):
                complete.add(int(permutation_id))
    return model, contrast, diagnostics, complete


def append_and_checkpoint(
    existing: pd.DataFrame,
    addition: pd.DataFrame,
    path: Path,
    sort_columns: list[str],
) -> pd.DataFrame:
    combined = pd.concat([existing, addition], ignore_index=True)
    combined = combined.drop_duplicates(sort_columns, keep="last")
    combined = combined.sort_values(sort_columns, kind="mergesort").reset_index(
        drop=True
    )
    atomic_write_csv(combined, path)
    return combined


def empirical_tail_p(
    observed: float,
    null_values: np.ndarray,
    direction: str,
) -> float:
    finite = null_values[np.isfinite(null_values)]
    if not math.isfinite(observed) or len(finite) == 0:
        return float("nan")
    if direction == "greater":
        extreme = int(np.sum(finite >= observed))
    elif direction == "smaller":
        extreme = int(np.sum(finite <= observed))
    else:
        raise ValueError(direction)
    return float((1 + extreme) / (len(finite) + 1))


def build_null_inference(
    observed_models: pd.DataFrame,
    observed_contrast: pd.DataFrame,
    null_models: pd.DataFrame,
    null_contrasts: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, observed_row in observed_models.iterrows():
        model_id = str(observed_row["model_id"])
        null = null_models[null_models["model_id"] == model_id]
        for metric, direction in [
            ("spearman", "greater"),
            ("rmse", "smaller"),
            ("mae", "smaller"),
        ]:
            values = null[metric].to_numpy(dtype=np.float64)
            mean, sd, low, high, n_finite = finite_interval(values)
            observed = float(observed_row[metric])
            rows.append(
                {
                    "record_type": "model",
                    "estimand_id": model_id,
                    "metric": metric,
                    "observed": observed,
                    "null_mean": mean,
                    "null_sd": sd,
                    "null_percentile_2p5": low,
                    "null_percentile_97p5": high,
                    "n_permutations_finite": n_finite,
                    "empirical_tail": direction,
                    "empirical_p": empirical_tail_p(
                        observed, values, direction
                    ),
                }
            )
    observed = observed_contrast.iloc[0]
    for metric, direction in [
        ("delta_spearman", "greater"),
        ("delta_rmse", "greater"),
        ("delta_mae", "greater"),
    ]:
        values = null_contrasts[metric].to_numpy(dtype=np.float64)
        mean, sd, low, high, n_finite = finite_interval(values)
        observed_value = float(observed[metric])
        rows.append(
            {
                "record_type": "contrast",
                "estimand_id": "full_minus_chemistry",
                "metric": metric,
                "observed": observed_value,
                "null_mean": mean,
                "null_sd": sd,
                "null_percentile_2p5": low,
                "null_percentile_97p5": high,
                "n_permutations_finite": n_finite,
                "empirical_tail": direction,
                "empirical_p": empirical_tail_p(
                    observed_value, values, direction
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_pair_columns(
    chemistry: pd.DataFrame,
    full: pd.DataFrame,
    key_columns: list[str],
) -> pd.DataFrame:
    chemistry = chemistry.sort_values(key_columns, kind="mergesort")
    full = full.sort_values(key_columns, kind="mergesort")
    if chemistry.duplicated(key_columns).any() or full.duplicated(
        key_columns
    ).any():
        raise ValueError("Duplicate applicability pair key")
    merged = chemistry.merge(
        full,
        on=key_columns,
        how="outer",
        suffixes=("_chemistry", "_full"),
        validate="one_to_one",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError("Chemistry/full applicability rows are not matched")
    for column in [
        "y_true",
        "canonical_smiles",
        "scaffold_id",
        "max_train_tanimoto",
    ]:
        left = merged[f"{column}_chemistry"]
        right = merged[f"{column}_full"]
        if pd.api.types.is_numeric_dtype(left):
            equal = np.isclose(
                left.to_numpy(dtype=np.float64),
                right.to_numpy(dtype=np.float64),
                equal_nan=True,
            )
        else:
            equal = left.fillna("NA").astype(str).to_numpy() == right.fillna(
                "NA"
            ).astype(str).to_numpy()
        if not bool(np.all(equal)):
            raise ValueError(f"Paired applicability mismatch in {column}")
    return pd.DataFrame(
        {
            **{column: merged[column] for column in key_columns},
            "y_true": merged["y_true_chemistry"],
            "canonical_smiles": merged["canonical_smiles_chemistry"],
            "scaffold_id": merged["scaffold_id_chemistry"],
            "max_train_tanimoto": merged[
                "max_train_tanimoto_chemistry"
            ],
            "prediction_chemistry": merged["y_pred_chemistry"],
            "prediction_full": merged["y_pred_full"],
        }
    )


def load_applicability_pairs() -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    internal = pd.read_csv(INTERNAL_APPLICABILITY)
    internal = internal[internal["protocol"] == "scaffold"].copy()
    required_internal = [
        "row_index",
        "y_true",
        "canonical_smiles",
        "scaffold_id",
        "max_train_tanimoto_mean",
        "prediction__chemistry_extra_trees",
        "prediction__full_context_extra_trees",
    ]
    missing = sorted(set(required_internal).difference(internal.columns))
    if missing:
        raise ValueError(f"Internal applicability missing columns: {missing}")
    tables.append(
        pd.DataFrame(
            {
                "regime": "internal_scaffold_disjoint",
                "protocol": "scaffold",
                "heldout_group": "pooled",
                "row_index": internal["row_index"],
                "y_true": internal["y_true"],
                "canonical_smiles": internal["canonical_smiles"],
                "scaffold_id": internal["scaffold_id"],
                "max_train_tanimoto": internal[
                    "max_train_tanimoto_mean"
                ],
                "prediction_chemistry": internal[
                    "prediction__chemistry_extra_trees"
                ],
                "prediction_full": internal[
                    "prediction__full_context_extra_trees"
                ],
            }
        )
    )

    for regime, path in [
        ("frozen_ood", OOD_PREDICTIONS),
        ("strict_domain_scaffold_ood", STRICT_OOD_PREDICTIONS),
    ]:
        predictions = pd.read_csv(path)
        predictions = predictions[predictions["model_id"].isin(MODEL_IDS)].copy()
        common = [
            "protocol",
            "heldout_group",
            "row_index",
            "y_true",
            "canonical_smiles",
            "scaffold_id",
            "max_train_tanimoto",
            "y_pred",
        ]
        missing = sorted(set(common).difference(predictions.columns))
        if missing:
            raise ValueError(f"{regime} predictions missing columns: {missing}")
        chemistry = predictions[predictions["model_id"] == MODEL_IDS[0]][
            common
        ]
        full = predictions[predictions["model_id"] == MODEL_IDS[1]][common]
        paired = validate_pair_columns(
            chemistry,
            full,
            ["protocol", "heldout_group", "row_index"],
        )
        paired.insert(0, "regime", regime)
        tables.append(paired)
    combined = pd.concat(tables, ignore_index=True)
    similarities = combined["max_train_tanimoto"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(similarities)):
        raise ValueError("Non-finite max-train Tanimoto")
    if np.any(similarities < 0.0) or np.any(similarities > 1.000001):
        raise ValueError("Max-train Tanimoto outside frozen bin range")
    combined["similarity_bin"] = pd.cut(
        combined["max_train_tanimoto"],
        bins=SIMILARITY_EDGES,
        labels=SIMILARITY_LABELS,
        right=False,
        include_lowest=True,
    )
    if combined["similarity_bin"].isna().any():
        raise ValueError("A similarity value was not assigned to a frozen bin")
    return combined


def requested_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[dict[str, float], dict[str, str]]:
    metrics = {
        "spearman": float("nan"),
        "rmse": float("nan"),
        "mae": float("nan"),
    }
    reasons = {"spearman": "", "rmse": "", "mae": ""}
    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        for metric in reasons:
            reasons[metric] = "non_finite_value"
        return metrics, reasons
    if len(y_true) == 0:
        for metric in reasons:
            reasons[metric] = "zero_rows"
        return metrics, reasons
    residual = y_true - y_pred
    metrics["rmse"] = float(np.sqrt(np.mean(np.square(residual))))
    metrics["mae"] = float(np.mean(np.abs(residual)))
    spearman_reasons: list[str] = []
    if len(y_true) < 3:
        spearman_reasons.append("fewer_than_3_rows")
    if np.unique(y_true).size < 2:
        spearman_reasons.append("constant_y_true")
    if np.unique(y_pred).size < 2:
        spearman_reasons.append("constant_prediction")
    if spearman_reasons:
        reasons["spearman"] = "|".join(spearman_reasons)
    else:
        metrics["spearman"] = core.safe_spearman(y_true, y_pred)
    return metrics, reasons


def combined_reasons(reasons: dict[str, str]) -> str:
    return "|".join(
        f"{metric}:{reason}"
        for metric, reason in reasons.items()
        if reason
    )


def stable_domain_sequence(domains: Iterable[str]) -> tuple[str, str]:
    normalized = sorted({str(domain) for domain in domains})
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return payload, digest


def oriented_advantage(
    metric: str,
    chemistry_value: float,
    full_value: float,
) -> float:
    if not (
        math.isfinite(chemistry_value) and math.isfinite(full_value)
    ):
        return float("nan")
    if metric == "spearman":
        return float(full_value - chemistry_value)
    if metric in {"rmse", "mae"}:
        return float(chemistry_value - full_value)
    raise ValueError(f"Unsupported metric: {metric}")


def build_macro_domain_eligibility(
    frame: pd.DataFrame,
    *,
    include_domain_macro: bool,
) -> pd.DataFrame:
    columns = [
        "heldout_group",
        "metric",
        "n_rows",
        "n_unique_scaffolds",
        "chemistry_metric",
        "full_metric",
        "observed_delta",
        "eligible_for_observed_macro",
        "eligible_domain_sparse_for_bootstrap",
        "non_estimable_reason",
    ]
    if not include_domain_macro:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for heldout_group, domain_frame in frame.groupby(
        "heldout_group", sort=True, dropna=False
    ):
        domain_id = str(heldout_group)
        chemistry_metrics, chemistry_reasons = requested_metrics(
            domain_frame["y_true"].to_numpy(dtype=np.float64),
            domain_frame["prediction_chemistry"].to_numpy(dtype=np.float64),
        )
        full_metrics, full_reasons = requested_metrics(
            domain_frame["y_true"].to_numpy(dtype=np.float64),
            domain_frame["prediction_full"].to_numpy(dtype=np.float64),
        )
        n_unique_scaffolds = int(
            domain_frame["scaffold_id"].fillna("NA").astype(str).nunique()
        )
        for metric in METRICS:
            delta = oriented_advantage(
                metric,
                chemistry_metrics[metric],
                full_metrics[metric],
            )
            reason_parts: list[str] = []
            if chemistry_reasons[metric]:
                reason_parts.append(
                    f"chemistry:{chemistry_reasons[metric]}"
                )
            if full_reasons[metric]:
                reason_parts.append(f"full:{full_reasons[metric]}")
            if not math.isfinite(delta) and not reason_parts:
                reason_parts.append("non_finite_paired_delta")
            eligible = bool(math.isfinite(delta))
            rows.append(
                {
                    "heldout_group": domain_id,
                    "metric": metric,
                    "n_rows": int(len(domain_frame)),
                    "n_unique_scaffolds": n_unique_scaffolds,
                    "chemistry_metric": chemistry_metrics[metric],
                    "full_metric": full_metrics[metric],
                    "observed_delta": delta,
                    "eligible_for_observed_macro": eligible,
                    "eligible_domain_sparse_for_bootstrap": bool(
                        eligible and n_unique_scaffolds < 2
                    ),
                    "non_estimable_reason": "|".join(reason_parts),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def fixed_domain_macro_metric(
    y_true: np.ndarray,
    chemistry: np.ndarray,
    full: np.ndarray,
    domains: np.ndarray,
    metric: str,
    eligible_domains: tuple[str, ...],
) -> tuple[float, str]:
    if not eligible_domains:
        return float("nan"), "no_observed_eligible_domains"
    represented = set(np.unique(domains).astype(str))
    if not set(eligible_domains).issubset(represented):
        return float("nan"), "missing_fixed_eligible_domain"
    values: list[float] = []
    for domain in eligible_domains:
        selected = domains == domain
        chemistry_metrics, _ = requested_metrics(
            y_true[selected], chemistry[selected]
        )
        full_metrics, _ = requested_metrics(
            y_true[selected], full[selected]
        )
        delta = oriented_advantage(
            metric,
            chemistry_metrics[metric],
            full_metrics[metric],
        )
        if not math.isfinite(delta):
            return (
                float("nan"),
                "fixed_eligible_domain_metric_nonestimable",
            )
        values.append(delta)
    return float(np.mean(np.asarray(values, dtype=np.float64))), "valid"


def paired_advantage_metrics(
    y_true: np.ndarray,
    chemistry: np.ndarray,
    full: np.ndarray,
    domains: np.ndarray,
    *,
    domain_macro: bool,
) -> tuple[dict[str, float], dict[str, int], int]:
    if not domain_macro:
        chemistry_metrics, _ = requested_metrics(y_true, chemistry)
        full_metrics, _ = requested_metrics(y_true, full)
        deltas = {
            "spearman": (
                full_metrics["spearman"] - chemistry_metrics["spearman"]
            ),
            "rmse": chemistry_metrics["rmse"] - full_metrics["rmse"],
            "mae": chemistry_metrics["mae"] - full_metrics["mae"],
        }
        finite_counts = {
            metric: int(math.isfinite(value))
            for metric, value in deltas.items()
        }
        return deltas, finite_counts, 1

    unique_domains = np.unique(domains)
    domain_deltas = {
        metric: [] for metric in ("spearman", "rmse", "mae")
    }
    for domain in unique_domains:
        selected = domains == domain
        chemistry_metrics, _ = requested_metrics(
            y_true[selected], chemistry[selected]
        )
        full_metrics, _ = requested_metrics(y_true[selected], full[selected])
        values = {
            "spearman": (
                full_metrics["spearman"] - chemistry_metrics["spearman"]
            ),
            "rmse": chemistry_metrics["rmse"] - full_metrics["rmse"],
            "mae": chemistry_metrics["mae"] - full_metrics["mae"],
        }
        for metric, value in values.items():
            if math.isfinite(value):
                domain_deltas[metric].append(value)
    deltas = {
        metric: (
            float(np.mean(values)) if values else float("nan")
        )
        for metric, values in domain_deltas.items()
    }
    finite_counts = {
        metric: int(len(values)) for metric, values in domain_deltas.items()
    }
    return deltas, finite_counts, int(len(unique_domains))


def bootstrap_applicability_delta(
    frame: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
    include_domain_macro: bool,
    domain_eligibility: pd.DataFrame,
) -> dict[str, Any]:
    y = frame["y_true"].to_numpy(dtype=np.float64)
    chemistry = frame["prediction_chemistry"].to_numpy(dtype=np.float64)
    full = frame["prediction_full"].to_numpy(dtype=np.float64)
    domains = frame["heldout_group"].fillna("NA").astype(str).to_numpy()
    pooled, _, _ = paired_advantage_metrics(
        y, chemistry, full, domains, domain_macro=False
    )
    observed = {
        f"delta_{metric}": value for metric, value in pooled.items()
    }
    represented_domains = tuple(sorted(set(domains.astype(str))))
    represented_json, represented_sha256 = stable_domain_sequence(
        represented_domains
    )
    observed["n_domains_total"] = (
        len(represented_domains) if include_domain_macro else 1
    )
    observed["represented_domains_json"] = (
        represented_json if include_domain_macro else '["pooled"]'
    )
    observed["represented_domains_sha256"] = (
        represented_sha256
        if include_domain_macro
        else stable_domain_sequence(["pooled"])[1]
    )
    eligible_by_metric: dict[str, tuple[str, ...]] = {}
    sparse_by_metric: dict[str, tuple[str, ...]] = {}
    for metric in METRICS:
        local = domain_eligibility[
            domain_eligibility["metric"].astype(str) == metric
        ]
        eligible = tuple(
            sorted(
                local.loc[
                    boolean_mask(local["eligible_for_observed_macro"]),
                    "heldout_group",
                ]
                .astype(str)
                .tolist()
            )
        )
        sparse = tuple(
            sorted(
                local.loc[
                    boolean_mask(
                        local["eligible_domain_sparse_for_bootstrap"]
                    ),
                    "heldout_group",
                ]
                .astype(str)
                .tolist()
            )
        )
        eligible_by_metric[metric] = eligible
        sparse_by_metric[metric] = sparse
        eligible_json, eligible_sha256 = stable_domain_sequence(eligible)
        sparse_json, sparse_sha256 = stable_domain_sequence(sparse)
        observed_values = local.loc[
            boolean_mask(local["eligible_for_observed_macro"]),
            "observed_delta",
        ].to_numpy(dtype=np.float64)
        observed_macro = (
            float(np.mean(observed_values))
            if include_domain_macro and len(observed_values)
            else float("nan")
        )
        prefix = f"delta_domain_macro_{metric}"
        observed[prefix] = observed_macro
        observed[f"{prefix}_finite_domains"] = (
            len(eligible) if include_domain_macro else 0
        )
        observed[f"{prefix}_n_domains_eligible"] = (
            len(eligible) if include_domain_macro else 0
        )
        observed[f"{prefix}_eligible_domains_json"] = (
            eligible_json if include_domain_macro else "[]"
        )
        observed[f"{prefix}_eligible_domains_sha256"] = (
            eligible_sha256 if include_domain_macro else ""
        )
        observed[f"{prefix}_n_sparse_eligible_domains"] = (
            len(sparse) if include_domain_macro else 0
        )
        observed[f"{prefix}_sparse_eligible_domains_json"] = (
            sparse_json if include_domain_macro else "[]"
        )
        observed[f"{prefix}_sparse_eligible_domains_sha256"] = (
            sparse_sha256 if include_domain_macro else ""
        )
    clusters = frame["scaffold_id"].fillna("NA").astype(str).to_numpy()
    unique_clusters = np.unique(clusters)
    positions = {
        cluster: np.where(clusters == cluster)[0] for cluster in unique_clusters
    }
    rng = np.random.default_rng(seed)
    values = {
        metric: []
        for metric in (
            "delta_spearman",
            "delta_rmse",
            "delta_mae",
            "delta_domain_macro_spearman",
            "delta_domain_macro_rmse",
            "delta_domain_macro_mae",
        )
    }
    macro_counts: dict[str, dict[str, int]] = {
        metric: {
            "valid": 0,
            "missing": 0,
            "nonestimable": 0,
            "not_attempted": 0,
        }
        for metric in METRICS
    }
    macro_structural_reason: dict[str, str] = {}
    for metric in METRICS:
        if not include_domain_macro:
            macro_structural_reason[metric] = "not_applicable_internal_pooled"
        elif not eligible_by_metric[metric]:
            macro_structural_reason[metric] = "no_observed_eligible_domains"
            macro_counts[metric]["not_attempted"] = n_bootstrap
        elif sparse_by_metric[metric]:
            macro_structural_reason[metric] = (
                "fixed_eligible_domain_fewer_than_2_scaffold_clusters"
            )
            macro_counts[metric]["not_attempted"] = n_bootstrap
        elif len(unique_clusters) < 2:
            macro_structural_reason[metric] = (
                "fewer_than_2_global_scaffold_clusters"
            )
            macro_counts[metric]["not_attempted"] = n_bootstrap
        else:
            macro_structural_reason[metric] = ""

    if len(unique_clusters) >= 2:
        for _ in range(n_bootstrap):
            sampled_clusters = rng.choice(
                unique_clusters, size=len(unique_clusters), replace=True
            )
            sampled = np.concatenate(
                [positions[cluster] for cluster in sampled_clusters]
            )
            pooled_boot, _, _ = paired_advantage_metrics(
                y[sampled],
                chemistry[sampled],
                full[sampled],
                domains[sampled],
                domain_macro=False,
            )
            for metric in METRICS:
                values[f"delta_{metric}"].append(pooled_boot[metric])
            if include_domain_macro:
                for metric in METRICS:
                    if macro_structural_reason[metric]:
                        continue
                    macro_value, status = fixed_domain_macro_metric(
                        y[sampled],
                        chemistry[sampled],
                        full[sampled],
                        domains[sampled],
                        metric,
                        eligible_by_metric[metric],
                    )
                    if status == "valid":
                        macro_counts[metric]["valid"] += 1
                        values[
                            f"delta_domain_macro_{metric}"
                        ].append(macro_value)
                    elif status == "missing_fixed_eligible_domain":
                        macro_counts[metric]["missing"] += 1
                        values[
                            f"delta_domain_macro_{metric}"
                        ].append(float("nan"))
                    elif (
                        status
                        == "fixed_eligible_domain_metric_nonestimable"
                    ):
                        macro_counts[metric]["nonestimable"] += 1
                        values[
                            f"delta_domain_macro_{metric}"
                        ].append(float("nan"))
                    else:
                        raise RuntimeError(
                            f"Unexpected fixed-domain bootstrap status: {status}"
                        )
    output: dict[str, Any] = {**observed}
    insufficient: list[str] = []
    minimum_finite = max(100, int(math.ceil(n_bootstrap * 0.5)))
    for metric in METRICS:
        key = f"delta_{metric}"
        metric_values = values[key]
        array = np.asarray(metric_values, dtype=np.float64)
        finite = array[np.isfinite(array)]
        output[f"{key}_n_bootstrap_finite"] = int(len(finite))
        if len(finite) < minimum_finite:
            output[f"{key}_ci_low"] = float("nan")
            output[f"{key}_ci_high"] = float("nan")
            reason = (
                "fewer_than_2_scaffold_clusters"
                if len(unique_clusters) < 2
                else f"too_few_finite_{key}_bootstrap"
            )
            output[f"{key}_non_estimable_reason"] = reason
            insufficient.append(reason)
        else:
            output[f"{key}_ci_low"] = float(np.percentile(finite, 2.5))
            output[f"{key}_ci_high"] = float(np.percentile(finite, 97.5))
            output[f"{key}_non_estimable_reason"] = ""

    for metric in METRICS:
        key = f"delta_domain_macro_{metric}"
        counts = macro_counts[metric]
        metric_values = np.asarray(values[key], dtype=np.float64)
        finite = metric_values[np.isfinite(metric_values)]
        if counts["valid"] != len(finite):
            raise RuntimeError(
                f"{key}: valid-count and finite-value count disagree"
            )
        output[f"{key}_n_bootstrap_requested"] = int(n_bootstrap)
        output[f"{key}_n_bootstrap_finite"] = int(len(finite))
        output[f"{key}_n_bootstrap_valid"] = int(counts["valid"])
        output[f"{key}_n_bootstrap_invalid_missing_domain"] = int(
            counts["missing"]
        )
        output[
            f"{key}_n_bootstrap_invalid_metric_nonestimable"
        ] = int(counts["nonestimable"])
        output[
            f"{key}_n_bootstrap_not_attempted_structural"
        ] = int(counts["not_attempted"])
        output[f"{key}_minimum_valid_required"] = int(minimum_finite)
        if not include_domain_macro:
            output[f"{key}_ci_low"] = float("nan")
            output[f"{key}_ci_high"] = float("nan")
            output[f"{key}_ci_status"] = "NOT_APPLICABLE_INTERNAL_POOLED"
            output[f"{key}_non_estimable_reason"] = (
                "not_applicable_internal_pooled"
            )
            continue
        total_classified = sum(counts.values())
        if total_classified != n_bootstrap:
            raise RuntimeError(
                f"{key}: classified bootstrap count {total_classified} "
                f"!= requested {n_bootstrap}"
            )
        structural_reason = macro_structural_reason[metric]
        if structural_reason:
            output[f"{key}_ci_low"] = float("nan")
            output[f"{key}_ci_high"] = float("nan")
            output[f"{key}_ci_status"] = "NON_ESTIMABLE_STRUCTURAL"
            output[f"{key}_non_estimable_reason"] = structural_reason
            insufficient.append(f"{key}:{structural_reason}")
        elif len(finite) < minimum_finite:
            output[f"{key}_ci_low"] = float("nan")
            output[f"{key}_ci_high"] = float("nan")
            output[f"{key}_ci_status"] = "NON_ESTIMABLE_TOO_FEW_VALID"
            output[f"{key}_non_estimable_reason"] = (
                "too_few_complete_fixed_domain_bootstrap_replicates"
            )
            insufficient.append(
                f"{key}:too_few_complete_fixed_domain_bootstrap_replicates"
            )
        else:
            output[f"{key}_ci_low"] = float(np.percentile(finite, 2.5))
            output[f"{key}_ci_high"] = float(np.percentile(finite, 97.5))
            output[f"{key}_ci_status"] = "ESTIMATED"
            output[f"{key}_non_estimable_reason"] = ""
    output["bootstrap_non_estimable_reason"] = "|".join(
        sorted(set(insufficient))
    )
    return output


def run_applicability_analysis(
    n_bootstrap: int,
    seed: int,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    paired = load_applicability_pairs()
    combinations = [
        ("internal_scaffold_disjoint", "scaffold"),
        ("frozen_ood", "source_ood"),
        ("frozen_ood", "target_ood"),
        ("strict_domain_scaffold_ood", "source_ood"),
        ("strict_domain_scaffold_ood", "target_ood"),
    ]
    model_path = (
        checkpoint_dir / "applicability_model_metrics.csv"
        if checkpoint_dir is not None
        else None
    )
    delta_path = (
        checkpoint_dir / "applicability_paired_deltas.csv"
        if checkpoint_dir is not None
        else None
    )
    domain_path = (
        checkpoint_dir / "applicability_domain_metrics.csv"
        if checkpoint_dir is not None
        else None
    )
    eligibility_path = (
        checkpoint_dir / "applicability_macro_domain_eligibility.csv"
        if checkpoint_dir is not None
        else None
    )
    existing_models = (
        pd.read_csv(model_path)
        if resume and model_path is not None and model_path.exists()
        else pd.DataFrame()
    )
    existing_deltas = (
        pd.read_csv(delta_path)
        if resume and delta_path is not None and delta_path.exists()
        else pd.DataFrame()
    )
    existing_domains = (
        pd.read_csv(domain_path)
        if resume and domain_path is not None and domain_path.exists()
        else pd.DataFrame()
    )
    existing_eligibility = (
        pd.read_csv(eligibility_path)
        if (
            resume
            and eligibility_path is not None
            and eligibility_path.exists()
        )
        else pd.DataFrame()
    )
    model_rows: list[dict[str, Any]] = existing_models.to_dict("records")
    delta_rows: list[dict[str, Any]] = existing_deltas.to_dict("records")
    domain_rows: list[dict[str, Any]] = existing_domains.to_dict("records")
    eligibility_rows: list[dict[str, Any]] = (
        existing_eligibility.to_dict("records")
    )
    combination_index = 0
    for regime, protocol in combinations:
        local = paired[
            (paired["regime"] == regime)
            & (paired["protocol"] == protocol)
        ]
        for similarity_bin in SIMILARITY_LABELS:
            combination_index += 1
            key_mask_delta = (
                (
                    existing_deltas["regime"].astype(str) == regime
                )
                & (
                    existing_deltas["protocol"].astype(str) == protocol
                )
                & (
                    existing_deltas["similarity_bin"].astype(str)
                    == similarity_bin
                )
                & (
                    existing_deltas["n_bootstrap_requested"].astype(int)
                    == n_bootstrap
                )
                if not existing_deltas.empty
                else pd.Series(dtype=bool)
            )
            key_mask_model = (
                (
                    existing_models["regime"].astype(str) == regime
                )
                & (
                    existing_models["protocol"].astype(str) == protocol
                )
                & (
                    existing_models["similarity_bin"].astype(str)
                    == similarity_bin
                )
                if not existing_models.empty
                else pd.Series(dtype=bool)
            )
            key_mask_domain = (
                (
                    existing_domains["regime"].astype(str) == regime
                )
                & (
                    existing_domains["protocol"].astype(str) == protocol
                )
                & (
                    existing_domains["similarity_bin"].astype(str)
                    == similarity_bin
                )
                if not existing_domains.empty
                else pd.Series(dtype=bool)
            )
            key_mask_eligibility = (
                (
                    existing_eligibility["regime"].astype(str) == regime
                )
                & (
                    existing_eligibility["protocol"].astype(str) == protocol
                )
                & (
                    existing_eligibility["similarity_bin"].astype(str)
                    == similarity_bin
                )
                if not existing_eligibility.empty
                else pd.Series(dtype=bool)
            )
            if (
                int(key_mask_delta.sum()) == 1
                and int(key_mask_model.sum()) == len(MODEL_IDS)
                and int(key_mask_domain.sum()) > 0
                and (
                    protocol == "scaffold"
                    or int(key_mask_eligibility.sum()) > 0
                )
            ):
                print(
                    f"[AD] reuse {regime}/{protocol}/{similarity_bin}",
                    flush=True,
                )
                continue
            frame = local[
                local["similarity_bin"].astype(str) == similarity_bin
            ].copy()
            counts = {
                "n_rows": int(len(frame)),
                "n_unique_smiles": int(frame["canonical_smiles"].nunique()),
                "n_scaffolds": int(frame["scaffold_id"].nunique()),
                "n_heldout_groups": int(frame["heldout_group"].nunique()),
            }
            expected_domains = (
                1
                if protocol == "scaffold"
                else (4 if protocol == "source_ood" else 8)
            )
            include_domain_macro = protocol != "scaffold"
            current_model_rows: list[dict[str, Any]] = []
            current_domain_rows: list[dict[str, Any]] = []
            local_eligibility = build_macro_domain_eligibility(
                frame,
                include_domain_macro=include_domain_macro,
            )
            if not local_eligibility.empty:
                local_eligibility.insert(0, "similarity_bin", similarity_bin)
                local_eligibility.insert(0, "protocol", protocol)
                local_eligibility.insert(0, "regime", regime)
            current_eligibility_rows = local_eligibility.to_dict("records")
            for model_id, prediction_column in [
                (MODEL_IDS[0], "prediction_chemistry"),
                (MODEL_IDS[1], "prediction_full"),
            ]:
                metrics, reasons = requested_metrics(
                    frame["y_true"].to_numpy(dtype=np.float64),
                    frame[prediction_column].to_numpy(dtype=np.float64),
                )
                per_domain_metrics: dict[str, list[float]] = {
                    metric: [] for metric in ("spearman", "rmse", "mae")
                }
                for heldout_group, domain_frame in frame.groupby(
                    "heldout_group", sort=True, dropna=False
                ):
                    domain_metrics, domain_reasons = requested_metrics(
                        domain_frame["y_true"].to_numpy(dtype=np.float64),
                        domain_frame[prediction_column].to_numpy(
                            dtype=np.float64
                        ),
                    )
                    current_domain_rows.append(
                        {
                            "regime": regime,
                            "protocol": protocol,
                            "similarity_bin": similarity_bin,
                            "heldout_group": heldout_group,
                            "model_id": model_id,
                            "n_rows": int(len(domain_frame)),
                            "n_unique_smiles": int(
                                domain_frame["canonical_smiles"].nunique()
                            ),
                            "n_scaffolds": int(
                                domain_frame["scaffold_id"].nunique()
                            ),
                            **domain_metrics,
                            "spearman_non_estimable_reason": domain_reasons[
                                "spearman"
                            ],
                            "rmse_non_estimable_reason": domain_reasons["rmse"],
                            "mae_non_estimable_reason": domain_reasons["mae"],
                        }
                    )
                    for metric, value in domain_metrics.items():
                        if math.isfinite(value):
                            per_domain_metrics[metric].append(value)
                macro = {
                    metric: (
                        float(np.mean(values))
                        if include_domain_macro and values
                        else float("nan")
                    )
                    for metric, values in per_domain_metrics.items()
                }
                current_model_rows.append(
                    {
                        "regime": regime,
                        "protocol": protocol,
                        "similarity_bin": similarity_bin,
                        "similarity_bin_left": SIMILARITY_EDGES[
                            SIMILARITY_LABELS.index(similarity_bin)
                        ],
                        "similarity_bin_right_exclusive": SIMILARITY_EDGES[
                            SIMILARITY_LABELS.index(similarity_bin) + 1
                        ],
                        "model_id": model_id,
                        **counts,
                        "n_domains_expected": int(expected_domains),
                        "n_domains_total": int(
                            frame["heldout_group"].nunique()
                        ),
                        **metrics,
                        "spearman_non_estimable_reason": reasons["spearman"],
                        "rmse_non_estimable_reason": reasons["rmse"],
                        "mae_non_estimable_reason": reasons["mae"],
                        "domain_macro_spearman": macro["spearman"],
                        "domain_macro_spearman_finite_domains": (
                            len(per_domain_metrics["spearman"])
                            if include_domain_macro
                            else 0
                        ),
                        "domain_macro_rmse": macro["rmse"],
                        "domain_macro_rmse_finite_domains": (
                            len(per_domain_metrics["rmse"])
                            if include_domain_macro
                            else 0
                        ),
                        "domain_macro_mae": macro["mae"],
                        "domain_macro_mae_finite_domains": (
                            len(per_domain_metrics["mae"])
                            if include_domain_macro
                            else 0
                        ),
                        "domain_macro_non_estimable_reason": (
                            ""
                            if include_domain_macro
                            else "not_applicable_internal_pooled"
                        ),
                    }
                )
            bootstrap = bootstrap_applicability_delta(
                frame,
                n_bootstrap,
                seed + combination_index * 1009,
                include_domain_macro,
                local_eligibility,
            )
            current_delta_row = (
                {
                    "regime": regime,
                    "protocol": protocol,
                    "similarity_bin": similarity_bin,
                    "contrast_id": "full_minus_chemistry",
                    **counts,
                    "n_domains_expected": int(expected_domains),
                    "n_bootstrap_requested": int(n_bootstrap),
                    "delta_definition": (
                        "Spearman: full - chemistry; RMSE/MAE: "
                        "chemistry - full"
                    ),
                    "direction": "positive values favor full for every metric",
                    **bootstrap,
                }
            )
            model_rows = [
                row
                for row in model_rows
                if not (
                    str(row["regime"]) == regime
                    and str(row["protocol"]) == protocol
                    and str(row["similarity_bin"]) == similarity_bin
                )
            ] + current_model_rows
            delta_rows = [
                row
                for row in delta_rows
                if not (
                    str(row["regime"]) == regime
                    and str(row["protocol"]) == protocol
                    and str(row["similarity_bin"]) == similarity_bin
                )
            ] + [current_delta_row]
            domain_rows = [
                row
                for row in domain_rows
                if not (
                    str(row["regime"]) == regime
                    and str(row["protocol"]) == protocol
                    and str(row["similarity_bin"]) == similarity_bin
                )
            ] + current_domain_rows
            eligibility_rows = [
                row
                for row in eligibility_rows
                if not (
                    str(row["regime"]) == regime
                    and str(row["protocol"]) == protocol
                    and str(row["similarity_bin"]) == similarity_bin
                )
            ] + current_eligibility_rows
            if checkpoint_dir is not None:
                atomic_write_csv(
                    pd.DataFrame(model_rows).sort_values(
                        ["regime", "protocol", "similarity_bin", "model_id"],
                        kind="mergesort",
                    ),
                    model_path,
                )
                atomic_write_csv(
                    pd.DataFrame(delta_rows).sort_values(
                        ["regime", "protocol", "similarity_bin"],
                        kind="mergesort",
                    ),
                    delta_path,
                )
                atomic_write_csv(
                    pd.DataFrame(domain_rows).sort_values(
                        [
                            "regime",
                            "protocol",
                            "similarity_bin",
                            "heldout_group",
                            "model_id",
                        ],
                        kind="mergesort",
                    ),
                    domain_path,
                )
                if eligibility_rows:
                    atomic_write_csv(
                        pd.DataFrame(eligibility_rows).sort_values(
                            [
                                "regime",
                                "protocol",
                                "similarity_bin",
                                "metric",
                                "heldout_group",
                            ],
                            kind="mergesort",
                        ),
                        eligibility_path,
                    )
            print(
                f"[AD] complete {combination_index}/"
                f"{len(combinations) * len(SIMILARITY_LABELS)} "
                f"{regime}/{protocol}/{similarity_bin}",
                flush=True,
            )
    return (
        paired,
        pd.DataFrame(model_rows),
        pd.DataFrame(delta_rows),
        pd.DataFrame(domain_rows),
        pd.DataFrame(eligibility_rows),
    )


def boolean_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def same_finite_or_nan(left: float, right: float) -> bool:
    if math.isnan(left) and math.isnan(right):
        return True
    return bool(
        math.isfinite(left)
        and math.isfinite(right)
        and math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
    )


def build_macro_bootstrap_audit(
    applicability_deltas: pd.DataFrame,
    domain_eligibility: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ood = applicability_deltas[
        applicability_deltas["protocol"].isin(["source_ood", "target_ood"])
    ]
    for _, delta_row in ood.sort_values(
        ["regime", "protocol", "similarity_bin"],
        kind="mergesort",
    ).iterrows():
        regime = str(delta_row["regime"])
        protocol = str(delta_row["protocol"])
        similarity_bin = str(delta_row["similarity_bin"])
        stratum = domain_eligibility[
            (domain_eligibility["regime"].astype(str) == regime)
            & (domain_eligibility["protocol"].astype(str) == protocol)
            & (
                domain_eligibility["similarity_bin"].astype(str)
                == similarity_bin
            )
        ]
        for metric in METRICS:
            local = stratum[stratum["metric"].astype(str) == metric].copy()
            if local.duplicated(["heldout_group"]).any():
                raise RuntimeError(
                    "Duplicate macro-domain eligibility identity for "
                    f"{regime}/{protocol}/{similarity_bin}/{metric}"
                )
            represented = tuple(
                sorted(local["heldout_group"].astype(str).tolist())
            )
            eligible_mask = boolean_mask(
                local["eligible_for_observed_macro"]
            )
            sparse_mask = boolean_mask(
                local["eligible_domain_sparse_for_bootstrap"]
            )
            eligible = tuple(
                sorted(
                    local.loc[eligible_mask, "heldout_group"]
                    .astype(str)
                    .tolist()
                )
            )
            sparse = tuple(
                sorted(
                    local.loc[sparse_mask, "heldout_group"]
                    .astype(str)
                    .tolist()
                )
            )
            represented_json, represented_hash = stable_domain_sequence(
                represented
            )
            eligible_json, eligible_hash = stable_domain_sequence(eligible)
            sparse_json, sparse_hash = stable_domain_sequence(sparse)
            observed_values = local.loc[
                eligible_mask, "observed_delta"
            ].to_numpy(dtype=np.float64)
            reconstructed = (
                float(np.mean(observed_values))
                if len(observed_values)
                else float("nan")
            )
            key = f"delta_domain_macro_{metric}"
            observed = float(delta_row[key])
            requested = int(delta_row[f"{key}_n_bootstrap_requested"])
            valid = int(delta_row[f"{key}_n_bootstrap_valid"])
            invalid_missing = int(
                delta_row[
                    f"{key}_n_bootstrap_invalid_missing_domain"
                ]
            )
            invalid_nonestimable = int(
                delta_row[
                    f"{key}_n_bootstrap_invalid_metric_nonestimable"
                ]
            )
            not_attempted = int(
                delta_row[
                    f"{key}_n_bootstrap_not_attempted_structural"
                ]
            )
            minimum = int(delta_row[f"{key}_minimum_valid_required"])
            status = str(delta_row[f"{key}_ci_status"])
            ci_low = float(delta_row[f"{key}_ci_low"])
            ci_high = float(delta_row[f"{key}_ci_high"])
            expected_structural = bool(not eligible or sparse)
            expected_status = (
                "NON_ESTIMABLE_STRUCTURAL"
                if expected_structural
                else (
                    "ESTIMATED"
                    if valid >= minimum
                    else "NON_ESTIMABLE_TOO_FEW_VALID"
                )
            )
            checks = {
                "represented_identity_match": (
                    str(delta_row["represented_domains_json"])
                    == represented_json
                    and str(delta_row["represented_domains_sha256"])
                    == represented_hash
                ),
                "eligible_identity_match": (
                    str(delta_row[f"{key}_eligible_domains_json"])
                    == eligible_json
                    and str(
                        delta_row[f"{key}_eligible_domains_sha256"]
                    )
                    == eligible_hash
                    and int(delta_row[f"{key}_n_domains_eligible"])
                    == len(eligible)
                ),
                "sparse_identity_match": (
                    str(
                        delta_row[f"{key}_sparse_eligible_domains_json"]
                    )
                    == sparse_json
                    and str(
                        delta_row[
                            f"{key}_sparse_eligible_domains_sha256"
                        ]
                    )
                    == sparse_hash
                    and int(
                        delta_row[
                            f"{key}_n_sparse_eligible_domains"
                        ]
                    )
                    == len(sparse)
                ),
                "observed_macro_exact_fixed_set_mean": same_finite_or_nan(
                    observed, reconstructed
                ),
                "counts_reconcile": (
                    valid
                    + invalid_missing
                    + invalid_nonestimable
                    + not_attempted
                    == requested
                ),
                "status_matches_rule": status == expected_status,
                "structural_skip_matches_rule": (
                    (not_attempted == requested)
                    if expected_structural
                    else (not_attempted == 0)
                ),
                "interval_presence_matches_rule": (
                    (
                        math.isfinite(ci_low)
                        and math.isfinite(ci_high)
                        and ci_low <= ci_high
                    )
                    if expected_status == "ESTIMATED"
                    else (math.isnan(ci_low) and math.isnan(ci_high))
                ),
                "sparse_domain_interval_fail_safe": (
                    not sparse
                    or (
                        status == "NON_ESTIMABLE_STRUCTURAL"
                        and not_attempted == requested
                        and math.isnan(ci_low)
                        and math.isnan(ci_high)
                    )
                ),
            }
            rows.append(
                {
                    "regime": regime,
                    "protocol": protocol,
                    "similarity_bin": similarity_bin,
                    "metric": metric,
                    "represented_domains_json": represented_json,
                    "represented_domains_sha256": represented_hash,
                    "eligible_domains_json": eligible_json,
                    "eligible_domains_sha256": eligible_hash,
                    "sparse_eligible_domains_json": sparse_json,
                    "sparse_eligible_domains_sha256": sparse_hash,
                    "n_represented_domains": len(represented),
                    "n_eligible_domains": len(eligible),
                    "n_sparse_eligible_domains": len(sparse),
                    "reported_observed_macro": observed,
                    "reconstructed_observed_macro": reconstructed,
                    "n_bootstrap_requested": requested,
                    "n_bootstrap_valid": valid,
                    "n_bootstrap_invalid_missing_domain": invalid_missing,
                    "n_bootstrap_invalid_metric_nonestimable": (
                        invalid_nonestimable
                    ),
                    "n_bootstrap_not_attempted_structural": not_attempted,
                    "minimum_valid_required": minimum,
                    "ci_status": status,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    **checks,
                    "overall_status": (
                        "PASS" if all(checks.values()) else "FAIL"
                    ),
                }
            )
    return pd.DataFrame(rows)


def relation_class(value: Any) -> str:
    if pd.isna(value):
        return "missing_or_unparsed"
    operator = str(value).strip()
    if operator == "=":
        return "equality"
    if operator == "range":
        return "range"
    if operator in {"<", "<=", ">", ">="}:
        return "one_sided"
    return "missing_or_unparsed"


def relation_summary(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    classes = ("equality", "range", "one_sided", "missing_or_unparsed")
    if not group_columns:
        groups = [((), frame)]
    elif len(group_columns) == 1:
        groups = frame.groupby(group_columns[0], sort=True, dropna=False)
    else:
        groups = frame.groupby(group_columns, sort=True, dropna=False)
    rows: list[dict[str, Any]] = []
    for key, local in groups:
        if not isinstance(key, tuple):
            key = (key,)
        base = {
            column: value for column, value in zip(group_columns, key)
        }
        total = len(local)
        for class_id in classes:
            count = int(np.sum(local["relation_class"] == class_id))
            rows.append(
                {
                    **base,
                    "relation_class": class_id,
                    "n_records": count,
                    "group_total_records": int(total),
                    "fraction_within_group": (
                        float(count / total) if total else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def safe_scaffold(smiles: Any) -> str | None:
    if pd.isna(smiles) or not str(smiles).strip():
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=False
    )
    if not scaffold:
        scaffold = f"ACYCLIC::{Chem.MolToSmiles(mol, canonical=True)}"
    return scaffold


def run_censoring_audit(
    core_rows: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    parsed = pd.read_csv(PARSED_DATA)
    parsed = parsed.copy()
    parsed["relation_class"] = parsed["dc50_relation"].map(relation_class)
    parsed["relation_operator"] = (
        parsed["dc50_relation"].fillna("MISSING").astype(str)
    )
    parsed["source_database"] = (
        parsed["source_database"].fillna("NA").astype(str)
    )
    parsed["target_protein"] = (
        parsed["target_protein"].fillna("NA").astype(str)
    )

    overall = relation_summary(parsed, [])
    by_source = relation_summary(parsed, ["source_database"])
    by_target = relation_summary(parsed, ["target_protein"])
    operator = (
        parsed.groupby(["relation_operator", "relation_class"], dropna=False)
        .size()
        .rename("n_records")
        .reset_index()
        .sort_values(["relation_class", "relation_operator"])
    )
    operator["fraction_all_records"] = operator["n_records"] / len(parsed)

    core_smiles = set(core_rows["canonical_smiles"].astype(str))
    core_scaffold = set(core.build_scaffold_ids(core_smiles).astype(str))
    parsed["derived_scaffold_id"] = parsed["canonical_smiles"].map(
        safe_scaffold
    )
    parsed["structure_derivable"] = parsed["derived_scaffold_id"].notna()
    parsed["compound_in_core"] = (
        parsed["canonical_smiles"].fillna("").astype(str).isin(core_smiles)
        & parsed["structure_derivable"]
    )
    parsed["scaffold_in_core"] = (
        parsed["derived_scaffold_id"].fillna("").astype(str).isin(core_scaffold)
        & parsed["structure_derivable"]
    )
    overlap_rows: list[dict[str, Any]] = []
    for class_id in (
        "equality",
        "range",
        "one_sided",
        "missing_or_unparsed",
        "all",
    ):
        local = (
            parsed
            if class_id == "all"
            else parsed[parsed["relation_class"] == class_id]
        )
        derivable = local[local["structure_derivable"]]
        unique_smiles = set(derivable["canonical_smiles"].astype(str))
        unique_scaffolds = set(derivable["derived_scaffold_id"].astype(str))
        overlap_rows.append(
            {
                "relation_class": class_id,
                "n_records": int(len(local)),
                "n_structure_derivable_records": int(len(derivable)),
                "n_missing_or_invalid_structure_records": int(
                    len(local) - len(derivable)
                ),
                "n_unique_compounds": int(len(unique_smiles)),
                "n_unique_scaffolds": int(len(unique_scaffolds)),
                "n_records_with_compound_in_core": int(
                    local["compound_in_core"].sum()
                ),
                "fraction_records_with_compound_in_core": (
                    float(local["compound_in_core"].mean())
                    if len(local)
                    else float("nan")
                ),
                "n_unique_compounds_in_core": int(
                    len(unique_smiles.intersection(core_smiles))
                ),
                "fraction_unique_compounds_in_core": (
                    float(
                        len(unique_smiles.intersection(core_smiles))
                        / len(unique_smiles)
                    )
                    if unique_smiles
                    else float("nan")
                ),
                "n_records_with_scaffold_in_core": int(
                    local["scaffold_in_core"].sum()
                ),
                "fraction_records_with_scaffold_in_core": (
                    float(local["scaffold_in_core"].mean())
                    if len(local)
                    else float("nan")
                ),
                "n_unique_scaffolds_in_core": int(
                    len(unique_scaffolds.intersection(core_scaffold))
                ),
                "fraction_unique_scaffolds_in_core": (
                    float(
                        len(unique_scaffolds.intersection(core_scaffold))
                        / len(unique_scaffolds)
                    )
                    if unique_scaffolds
                    else float("nan")
                ),
            }
        )
    overlap = pd.DataFrame(overlap_rows)

    parsed_ids = set(parsed["record_id"].astype(str))
    core_component_ids: list[str] = []
    for record_ids in core_rows["record_id"].astype(str):
        core_component_ids.extend(record_ids.split("|"))
    core_unique_component_ids = set(core_component_ids)
    if not core_unique_component_ids.issubset(parsed_ids):
        raise ValueError("Core record components are not contained in parsed data")
    numeric_endpoint = (
        pd.to_numeric(parsed["dc50_nM"], errors="coerce").notna()
        & pd.to_numeric(parsed["pDC50"], errors="coerce").notna()
    )
    flow = pd.DataFrame(
        [
            {
                "stage_order": 1,
                "stage": "all_parsed_records",
                "n": int(len(parsed)),
                "unit": "record_rows",
                "note": "all parsed source records",
            },
            {
                "stage_order": 2,
                "stage": "finite_numeric_endpoint_records",
                "n": int(numeric_endpoint.sum()),
                "unit": "record_rows",
                "note": "finite dc50_nM and pDC50; relation not restricted",
            },
            {
                "stage_order": 3,
                "stage": "equality_relation_records",
                "n": int(np.sum(parsed["relation_class"] == "equality")),
                "unit": "record_rows",
                "note": "recorded relation is equality",
            },
            {
                "stage_order": 4,
                "stage": "equality_unique_record_ids",
                "n": int(
                    parsed.loc[
                        parsed["relation_class"] == "equality", "record_id"
                    ]
                    .astype(str)
                    .nunique()
                ),
                "unit": "unique_record_ids",
                "note": "unique equality-record identifiers in parsed data",
            },
            {
                "stage_order": 5,
                "stage": "core_component_record_occurrences",
                "n": int(len(core_component_ids)),
                "unit": "component_record_occurrences",
                "note": "pipe-delimited source-record components after aggregation",
            },
            {
                "stage_order": 6,
                "stage": "core_unique_component_record_ids",
                "n": int(len(core_unique_component_ids)),
                "unit": "unique_record_ids",
                "note": "unique parsed source records represented in core",
            },
            {
                "stage_order": 7,
                "stage": "final_aggregated_modeling_rows",
                "n": int(len(core_rows)),
                "unit": "aggregated_model_rows",
                "note": "frozen strict-exact modeling observations",
            },
        ]
    )

    source_rows: list[dict[str, Any]] = []
    sources = sorted(set(parsed["source_database"].astype(str)))
    parsed_source_by_id = (
        parsed[["record_id", "source_database"]]
        .drop_duplicates()
        .assign(record_id=lambda x: x["record_id"].astype(str))
    )
    core_token_sets = core_rows["source_database"].fillna("NA").astype(str).map(
        lambda value: set(value.split("+"))
    )
    for source in sources:
        local_parsed = parsed[parsed["source_database"] == source]
        source_parsed_ids = set(
            parsed_source_by_id.loc[
                parsed_source_by_id["source_database"].astype(str) == source,
                "record_id",
            ]
        )
        source_rows.append(
            {
                "source_database": source,
                "n_parsed_records": int(len(local_parsed)),
                "n_equality_records": int(
                    np.sum(local_parsed["relation_class"] == "equality")
                ),
                "n_range_records": int(
                    np.sum(local_parsed["relation_class"] == "range")
                ),
                "n_one_sided_records": int(
                    np.sum(local_parsed["relation_class"] == "one_sided")
                ),
                "n_missing_or_unparsed_records": int(
                    np.sum(
                        local_parsed["relation_class"]
                        == "missing_or_unparsed"
                    )
                ),
                "n_parsed_unique_record_ids": int(len(source_parsed_ids)),
                "n_core_unique_component_record_ids": int(
                    len(source_parsed_ids.intersection(core_unique_component_ids))
                ),
                "n_core_aggregated_rows_with_source_token": int(
                    sum(source in tokens for tokens in core_token_sets)
                ),
                "note": (
                    "counts use different declared units; no inclusion "
                    "probability is implied"
                ),
            }
        )
    core_target = (
        core_rows.groupby("target_protein", dropna=False)
        .agg(
            n_core_modeling_rows=("qc_id", "size"),
            n_core_unique_compounds=("canonical_smiles", "nunique"),
        )
        .reset_index()
        .sort_values("n_core_modeling_rows", ascending=False)
    )
    return {
        "censoring_relation_overall.csv": overall,
        "censoring_relation_by_source.csv": by_source,
        "censoring_relation_by_target.csv": by_target,
        "censoring_relation_operator_counts.csv": operator,
        "censoring_overlap_with_core.csv": overlap,
        "selection_flow_overall.csv": flow,
        "selection_representation_by_source.csv": pd.DataFrame(source_rows),
        "core_modeling_distribution_by_target.csv": core_target,
    }


def build_qa(
    args: argparse.Namespace,
    identity_checks: list[dict[str, Any]],
    rows: pd.DataFrame,
    splits: list[dict[str, Any]],
    null_models: pd.DataFrame,
    null_contrasts: pd.DataFrame,
    null_diagnostics: pd.DataFrame,
    applicability_models: pd.DataFrame,
    applicability_deltas: pd.DataFrame,
    applicability_domains: pd.DataFrame,
    applicability_eligibility: pd.DataFrame,
    applicability_macro_audit: pd.DataFrame,
    censoring_tables: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    checks = list(identity_checks)

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append(
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
        )

    add("core_row_count", len(rows) == 1560, f"n={len(rows)}")
    add(
        "split_count",
        len(splits) == args.max_repeats * 5,
        f"observed={len(splits)}; expected={args.max_repeats * 5}",
    )
    expected_permutations = set(range(1, args.n_permutations + 1))
    observed_model_permutations = set(
        null_models["permutation_id"].astype(int)
    )
    observed_contrast_permutations = set(
        null_contrasts["permutation_id"].astype(int)
    )
    observed_diagnostic_permutations = set(
        null_diagnostics["permutation_id"].astype(int)
    )
    add(
        "permutation_ids_complete",
        (
            observed_model_permutations
            == observed_contrast_permutations
            == observed_diagnostic_permutations
            == expected_permutations
        ),
        (
            f"models={len(observed_model_permutations)}; "
            f"contrasts={len(observed_contrast_permutations)}; "
            f"diagnostics={len(observed_diagnostic_permutations)}; "
            f"expected={args.n_permutations}"
        ),
    )
    add(
        "permutation_model_rows_complete",
        len(null_models) == args.n_permutations * len(MODEL_IDS),
        f"n={len(null_models)}",
    )
    add(
        "permutation_fold_rows_complete",
        len(null_diagnostics)
        == args.n_permutations * args.max_repeats * 5,
        f"n={len(null_diagnostics)}",
    )
    add(
        "permutation_multisets_preserved",
        bool(
            null_diagnostics["within_stratum_multiset_preserved"]
            .fillna(False)
            .all()
        ),
        "all fit-fold within-stratum multisets preserved",
    )
    add(
        "permutation_predictions_finite",
        bool(
            np.isfinite(
                null_models[["spearman", "rmse", "mae"]].to_numpy(
                    dtype=np.float64
                )
            ).all()
        ),
        "all permutation-level metrics finite",
    )
    expected_applicability_strata = 5 * len(SIMILARITY_LABELS)
    add(
        "applicability_model_strata_complete",
        len(applicability_models)
        == expected_applicability_strata * len(MODEL_IDS),
        f"n={len(applicability_models)}",
    )
    add(
        "applicability_delta_strata_complete",
        len(applicability_deltas) == expected_applicability_strata,
        f"n={len(applicability_deltas)}",
    )
    ood_models = applicability_models[
        applicability_models["protocol"].isin(["source_ood", "target_ood"])
    ]
    add(
        "applicability_domain_macro_model_schema",
        all(
            column in applicability_models.columns
            for column in [
                "domain_macro_spearman",
                "domain_macro_spearman_finite_domains",
                "domain_macro_rmse",
                "domain_macro_rmse_finite_domains",
                "domain_macro_mae",
                "domain_macro_mae_finite_domains",
            ]
        )
        and bool(ood_models["domain_macro_rmse"].notna().all())
        and bool(ood_models["domain_macro_mae"].notna().all()),
        f"ood_model_rows={len(ood_models)}",
    )
    ood_deltas = applicability_deltas[
        applicability_deltas["protocol"].isin(["source_ood", "target_ood"])
    ]
    add(
        "applicability_domain_macro_delta_schema",
        all(
            column in applicability_deltas.columns
            for column in [
                "delta_domain_macro_spearman",
                "delta_domain_macro_spearman_ci_low",
                "delta_domain_macro_spearman_ci_high",
                "delta_domain_macro_rmse",
                "delta_domain_macro_rmse_ci_low",
                "delta_domain_macro_rmse_ci_high",
                "delta_domain_macro_mae",
                "delta_domain_macro_mae_ci_low",
                "delta_domain_macro_mae_ci_high",
            ]
        )
        and bool(ood_deltas["delta_domain_macro_rmse"].notna().all())
        and bool(ood_deltas["delta_domain_macro_mae"].notna().all()),
        f"ood_delta_rows={len(ood_deltas)}",
    )
    add(
        "applicability_domain_rows_present",
        len(applicability_domains) > 0,
        f"n={len(applicability_domains)}",
    )
    add(
        "applicability_macro_eligibility_rows_unique",
        (
            len(applicability_eligibility) > 0
            and not applicability_eligibility.duplicated(
                [
                    "regime",
                    "protocol",
                    "similarity_bin",
                    "metric",
                    "heldout_group",
                ]
            ).any()
        ),
        f"n={len(applicability_eligibility)}",
    )
    expected_macro_audit_rows = (
        4 * len(SIMILARITY_LABELS) * len(METRICS)
    )
    add(
        "applicability_fixed_domain_macro_audit_complete",
        (
            len(applicability_macro_audit) == expected_macro_audit_rows
            and applicability_macro_audit["overall_status"].eq("PASS").all()
        ),
        (
            f"n={len(applicability_macro_audit)}; "
            f"expected={expected_macro_audit_rows}; "
            f"failed={int(applicability_macro_audit['overall_status'].ne('PASS').sum())}"
        ),
    )
    low_target_audit = applicability_macro_audit[
        (applicability_macro_audit["protocol"] == "target_ood")
        & (
            applicability_macro_audit["similarity_bin"]
            == "0.0_to_lt_0.4"
        )
    ]
    add(
        "applicability_low_target_sparse_domain_fail_safe",
        (
            len(low_target_audit)
            == 2 * len(METRICS)
            and low_target_audit["n_sparse_eligible_domains"].ge(1).all()
            and low_target_audit["ci_status"]
            .eq("NON_ESTIMABLE_STRUCTURAL")
            .all()
            and low_target_audit["n_bootstrap_valid"].eq(0).all()
            and (
                low_target_audit[
                    "n_bootstrap_not_attempted_structural"
                ]
                == low_target_audit["n_bootstrap_requested"]
            ).all()
        ),
        (
            f"rows={len(low_target_audit)}; "
            f"regimes={sorted(low_target_audit['regime'].astype(str).unique())}"
        ),
    )
    overall = censoring_tables["censoring_relation_overall.csv"]
    add(
        "censoring_classes_sum_to_parsed_total",
        int(overall["n_records"].sum()) == 3117,
        f"sum={int(overall['n_records'].sum())}",
    )
    operator = censoring_tables["censoring_relation_operator_counts.csv"]
    expected_operators = {"=", "range", "<", "<=", ">", ">=", "MISSING"}
    observed_operators = set(operator["relation_operator"].astype(str))
    add(
        "relation_operators_explicit",
        observed_operators == expected_operators,
        f"observed={sorted(observed_operators)}",
    )
    formal = (
        args.n_permutations >= 100
        and args.n_estimators == 600
        and args.max_repeats == 5
        and args.bootstrap_replicates == 10_000
    )
    add(
        "formal_permutation_minimum",
        (not formal) or args.n_permutations >= 100,
        f"run_kind={'formal' if formal else 'smoke_or_development'}",
    )
    n_pass = sum(check["status"] == "PASS" for check in checks)
    n_fail = sum(check["status"] == "FAIL" for check in checks)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "run_kind": "formal" if formal else "smoke_or_development",
        "overall_status": "PASS" if n_fail == 0 else "FAIL",
        "n_checks": len(checks),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "checks": checks,
    }


def results_brief(
    qa: dict[str, Any],
    inference: pd.DataFrame,
    applicability_deltas: pd.DataFrame,
    censoring_overall: pd.DataFrame,
) -> str:
    full_s = inference[
        (inference["estimand_id"] == "full_context_extra_trees")
        & (inference["metric"] == "spearman")
    ].iloc[0]
    chem_s = inference[
        (inference["estimand_id"] == "chemistry_extra_trees")
        & (inference["metric"] == "spearman")
    ].iloc[0]
    contrast_s = inference[
        (inference["estimand_id"] == "full_minus_chemistry")
        & (inference["metric"] == "delta_spearman")
    ].iloc[0]
    class_counts = dict(
        zip(
            censoring_overall["relation_class"],
            censoring_overall["n_records"],
        )
    )
    estimable = int(
        applicability_deltas["bootstrap_non_estimable_reason"].fillna("").eq("").sum()
    )
    total = len(applicability_deltas)
    return (
        "# 后验置乱、适用域与删失审计结果简报\n\n"
        f"- QA：**{qa['overall_status']}**（{qa['n_pass']}/{qa['n_checks']} 项通过）。\n"
        f"- 条件置乱对照：chemistry 的实测 Spearman={chem_s['observed']:.4f}，"
        f"单侧经验 p={chem_s['empirical_p']:.4g}；full 的实测 "
        f"Spearman={full_s['observed']:.4f}，单侧经验 "
        f"p={full_s['empirical_p']:.4g}。\n"
        f"- full−chemistry 的实测 ΔSpearman={contrast_s['observed']:.4f}，"
        f"条件置乱单侧经验 p={contrast_s['empirical_p']:.4g}。该检验是固定折、"
        "固定原始数据超参数下的 repeated negative control，不是重调参随机化检验。\n"
        f"- 适用域：固定 4 档 × 5 个验证组合共 {total} 个分层；"
        f"{estimable} 个分层获得全部配对 bootstrap 区间。未估计分层保留原因，"
        "未合并阈值。\n"
        f"- 3,117 条解析记录中，等号 {class_counts.get('equality', 0)} 条，"
        f"range {class_counts.get('range', 0)} 条，单侧界 "
        f"{class_counts.get('one_sided', 0)} 条，缺失/未解析 "
        f"{class_counts.get('missing_or_unparsed', 0)} 条。后 3 类均未进入本次"
        "预测模型。\n"
    )


def write_artifact_hashes(output_dir: Path) -> None:
    files = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "artifact_sha256.csv"
    )
    rows = [
        {
            "relative_path": path.name,
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        for path in files
    ]
    atomic_write_csv(pd.DataFrame(rows), output_dir / "artifact_sha256.csv")


def main() -> None:
    args = parse_args()
    started = time.time()
    prepare_output_directory(args)
    for path in requested_input_paths():
        if not path.exists():
            raise FileNotFoundError(path)
    identity_checks = validate_frozen_identities()
    input_hashes = pd.DataFrame(
        [
            {
                "relative_path": str(path.relative_to(PROJECT_DIR)),
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
            for path in requested_input_paths()
        ]
    )
    atomic_write_csv(input_hashes, args.output_dir / "input_sha256.csv")

    rows, sanitization_audit = core.load_rows(CORE_DATA)
    if len(rows) != 1560:
        raise ValueError(f"Expected 1,560 core rows; observed {len(rows)}")
    internal_predictions, _, splits, _ = load_internal_design(
        rows, args.max_repeats
    )
    observed_models, observed_contrast = observed_internal_metrics(
        internal_predictions
    )
    atomic_write_csv(
        observed_models, args.output_dir / "permutation_observed_model_metrics.csv"
    )
    atomic_write_csv(
        observed_contrast,
        args.output_dir / "permutation_observed_contrast_metrics.csv",
    )
    atomic_write_csv(
        sanitization_audit,
        args.output_dir / "context_sanitization_audit.csv",
    )

    (
        paired,
        applicability_models,
        applicability_deltas,
        applicability_domains,
        applicability_eligibility,
    ) = (
        run_applicability_analysis(
            args.bootstrap_replicates,
            args.seed + 700_000,
            checkpoint_dir=args.output_dir,
            resume=args.resume,
        )
    )
    atomic_write_csv(
        paired, args.output_dir / "applicability_paired_source_data.csv"
    )
    atomic_write_csv(
        applicability_models,
        args.output_dir / "applicability_model_metrics.csv",
    )
    atomic_write_csv(
        applicability_deltas,
        args.output_dir / "applicability_paired_deltas.csv",
    )
    atomic_write_csv(
        applicability_domains,
        args.output_dir / "applicability_domain_metrics.csv",
    )
    atomic_write_csv(
        applicability_eligibility,
        args.output_dir / "applicability_macro_domain_eligibility.csv",
    )
    applicability_macro_audit = build_macro_bootstrap_audit(
        applicability_deltas,
        applicability_eligibility,
    )
    if not applicability_macro_audit["overall_status"].eq("PASS").all():
        failed = applicability_macro_audit[
            applicability_macro_audit["overall_status"] != "PASS"
        ]
        raise RuntimeError(
            "Fixed-domain applicability audit failed before model fitting: "
            f"{len(failed)} rows"
        )
    atomic_write_csv(
        applicability_macro_audit,
        args.output_dir / "applicability_macro_bootstrap_audit.csv",
    )

    censoring_tables = run_censoring_audit(rows)
    for filename, frame in censoring_tables.items():
        atomic_write_csv(frame, args.output_dir / filename)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "analysis_identity": (
            "post-hoc sensitivity and descriptive audit; not confirmatory"
        ),
        "master_protocol_sha256": MASTER_PROTOCOL_SHA256,
        "local_protocol_sha256": LOCAL_PROTOCOL_SHA256,
        "float32_correction_sha256": FLOAT32_CORRECTION_NOTE_SHA256,
        "applicability_correction_sha256": (
            APPLICABILITY_CORRECTION_NOTE_SHA256
        ),
        "supersedes_protocol_versions": [
            "post_hoc_null_applicability_censoring_v1.0",
            "post_hoc_null_applicability_censoring_v2.0",
        ],
        "config": {
            "n_permutations": args.n_permutations,
            "n_estimators": args.n_estimators,
            "n_jobs": args.n_jobs,
            "bootstrap_replicates": args.bootstrap_replicates,
            "max_repeats": args.max_repeats,
            "outer_folds_per_repeat": 5,
            "seed": args.seed,
            "similarity_edges": SIMILARITY_EDGES,
            "models": MODEL_IDS,
            "training_response_dtype": "float32",
            "applicability_domain_macro_bootstrap": (
                "global scaffold clusters; metric-specific frozen eligible "
                "domain set; incomplete fixed-domain replicates are NA"
            ),
            "applicability_minimum_valid_bootstrap": (
                "max(100, ceil(0.50 * requested_replicates))"
            ),
            "applicability_sparse_domain_rule": (
                "domain-macro CI is non-estimable when any fixed eligible "
                "domain has fewer than two unique global scaffolds"
            ),
            "label_permutation_strata": (
                "source_database × target_protein within each outer fit fold"
            ),
            "hyperparameters": (
                "reuse corresponding original-data outer-fold selections; "
                "no retuning"
            ),
        },
        "publication_boundary": (
            "1,560-row frozen core for fitting; public-source parsed table "
            "and the same core for aggregate descriptive audit; "
            "collaborator-restricted records and derivatives prohibited"
        ),
        "software": software_versions(),
        "started_unix": started,
    }
    atomic_write_json(args.output_dir / "run_manifest.json", manifest)

    null_models, null_contrasts, null_diagnostics, complete = load_resume_tables(
        args.output_dir
    )
    missing_permutations = [
        permutation_id
        for permutation_id in range(1, args.n_permutations + 1)
        if permutation_id not in complete
    ]
    if missing_permutations:
        cached_splits, morgan, _, feature_manifest = cache_fold_features(
            rows, splits
        )
        atomic_write_json(
            args.output_dir / "feature_manifest.json", feature_manifest
        )
        y = rows["pDC50"].to_numpy(dtype=np.float32)
        for run_position, permutation_id in enumerate(
            missing_permutations, start=1
        ):
            permutation_started = time.time()
            model_add, contrast_add, diagnostic_add = run_one_permutation(
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
            null_models = append_and_checkpoint(
                null_models,
                model_add,
                args.output_dir / "permutation_model_metrics.csv",
                ["permutation_id", "model_id"],
            )
            null_contrasts = append_and_checkpoint(
                null_contrasts,
                contrast_add,
                args.output_dir / "permutation_contrast_metrics.csv",
                ["permutation_id", "contrast_id"],
            )
            null_diagnostics = append_and_checkpoint(
                null_diagnostics,
                diagnostic_add,
                args.output_dir / "permutation_fold_diagnostics.csv",
                ["permutation_id", "repeat", "outer_fold"],
            )
            elapsed = time.time() - permutation_started
            print(
                f"[PERM] {run_position}/{len(missing_permutations)} "
                f"id={permutation_id} complete in {elapsed:.1f}s",
                flush=True,
            )
    else:
        print("[PERM] all requested permutation IDs already complete", flush=True)

    null_models = null_models[
        null_models["permutation_id"].astype(int)
        <= args.n_permutations
    ].copy()
    null_contrasts = null_contrasts[
        null_contrasts["permutation_id"].astype(int)
        <= args.n_permutations
    ].copy()
    null_diagnostics = null_diagnostics[
        null_diagnostics["permutation_id"].astype(int)
        <= args.n_permutations
    ].copy()
    inference = build_null_inference(
        observed_models,
        observed_contrast,
        null_models,
        null_contrasts,
    )
    atomic_write_csv(
        inference, args.output_dir / "permutation_null_inference.csv"
    )

    qa = build_qa(
        args=args,
        identity_checks=identity_checks,
        rows=rows,
        splits=splits,
        null_models=null_models,
        null_contrasts=null_contrasts,
        null_diagnostics=null_diagnostics,
        applicability_models=applicability_models,
        applicability_deltas=applicability_deltas,
        applicability_domains=applicability_domains,
        applicability_eligibility=applicability_eligibility,
        applicability_macro_audit=applicability_macro_audit,
        censoring_tables=censoring_tables,
    )
    atomic_write_json(args.output_dir / "qa_summary.json", qa)
    qa_lines = [
        "# QA summary",
        "",
        f"Overall: **{qa['overall_status']}** "
        f"({qa['n_pass']}/{qa['n_checks']} checks passed)",
        "",
        "| Check | Status | Detail |",
        "|---|---:|---|",
    ]
    for check in qa["checks"]:
        detail = str(check["detail"]).replace("|", "/")
        qa_lines.append(
            f"| {check['check']} | {check['status']} | {detail} |"
        )
    atomic_write_text(
        args.output_dir / "qa_summary.md", "\n".join(qa_lines) + "\n"
    )
    atomic_write_text(
        args.output_dir / "results_brief_zh.md",
        results_brief(
            qa,
            inference,
            applicability_deltas,
            censoring_tables["censoring_relation_overall.csv"],
        ),
    )
    manifest["completed_unix"] = time.time()
    manifest["runtime_seconds"] = manifest["completed_unix"] - started
    manifest["qa_status"] = qa["overall_status"]
    atomic_write_json(args.output_dir / "run_manifest.json", manifest)
    write_artifact_hashes(args.output_dir)
    print(
        f"[DONE] {qa['overall_status']} in "
        f"{manifest['runtime_seconds']:.1f}s -> {args.output_dir}",
        flush=True,
    )
    if qa["overall_status"] != "PASS":
        raise RuntimeError("QA failed; inspect qa_summary.json")


if __name__ == "__main__":
    main()
