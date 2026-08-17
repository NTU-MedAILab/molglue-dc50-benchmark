#!/usr/bin/env python
"""Post-hoc generic-scaffold and target-OOD source-deletion sensitivities v2.

This CPU-only runner is deliberately isolated from the confirmatory outputs.
It imports the frozen data sanitation, feature construction, model grid and
target-OOD split construction, but writes only to its own v2 report directory.

Version 2 makes one scientific implementation correction relative to v1:
the training response is explicitly constructed as float32, matching the
frozen confirmatory implementation. The v1 script and output remain unchanged
as a superseded audit trail.
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

import joblib
import numpy as np
import pandas as pd
import scipy
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn import __version__ as sklearn_version

import run_confirmatory_cpu_v1 as core
import run_confirmatory_ood_cpu_v1 as ood


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_FILE = core.DATA_FILE
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "reports" / "post_hoc_scaffold_source_sensitivity_v2"
)
PROTOCOL_DOCUMENT = (
    PROJECT_DIR / "docs" / "post_hoc_scaffold_source_sensitivity_protocol_v2.md"
)
CORRECTION_DOCUMENT = (
    PROJECT_DIR / "docs" / "post_hoc_float32_response_correction_v2.md"
)
EXPECTED_CORRECTION_DOCUMENT_SHA256 = (
    "75a8b0e4f894cead9d72039eab9e6162a89d2a20c79afceda8a2aae1ffd78f3f"
)
MASTER_PROTOCOL_DOCUMENT = (
    PROJECT_DIR / "docs" / "post_hoc_computational_extension_master_protocol_v1.md"
)
EXPECTED_MASTER_PROTOCOL_SHA256 = (
    "7dee28f45e3ddf8d6299622a8e494c50492b3b2614f39ce8b2fdd2a1be754ff8"
)
CORE_SCRIPT = SCRIPT_DIR / "run_confirmatory_cpu_v1.py"
OOD_SCRIPT = SCRIPT_DIR / "run_confirmatory_ood_cpu_v1.py"

PROTOCOL_VERSION = "post_hoc_scaffold_source_sensitivity_v2.0"
RESPONSE_DTYPE = np.dtype(np.float32)
EXPECTED_DATA_SHA256 = core.FROZEN_DATA_SHA256
EXPECTED_N_ROWS = 1_560
EXPECTED_GENERIC_GROUPS = 376
EXPECTED_GENERIC_SINGLETONS = 183
EXPECTED_GENERIC_MAX_ROWS = 76
DEFAULT_SEED = core.DEFAULT_SEED
MODELS = ("chemistry_extra_trees", "full_context_extra_trees")
SOURCE_DELETIONS = ("MGTbind", "MGDB", "MolGlueDB", "TPDdb")
BASELINE_CONDITION = "none"
TARGETS = ood.TARGET_OOD_GROUPS

GENERIC_FILES = {
    "predictions": "generic_scaffold_predictions.csv",
    "fold_metrics": "generic_scaffold_outer_fold_metrics.csv",
    "tuning": "generic_scaffold_inner_tuning_metrics.csv",
    "selected": "generic_scaffold_selected_hyperparameters.csv",
    "outer_audit": "generic_scaffold_outer_split_audit.csv",
    "inner_audit": "generic_scaffold_inner_split_audit.csv",
}
SOURCE_FILES = {
    "predictions": "source_deletion_predictions.csv",
    "domain_metrics": "source_deletion_domain_metrics.csv",
    "tuning": "source_deletion_inner_tuning_metrics.csv",
    "selected": "source_deletion_selected_hyperparameters.csv",
    "split_audit": "source_deletion_split_audit.csv",
    "inner_audit": "source_deletion_inner_split_audit.csv",
}
FINAL_FILES = (
    "analysis_index.csv",
    "context_sanitization_audit.csv",
    "generic_scaffold_group_summary.csv",
    "generic_scaffold_metrics_by_repeat.csv",
    "generic_scaffold_repeat_averaged_predictions.csv",
    "generic_scaffold_repeat_averaged_metrics.csv",
    "generic_scaffold_paired_cluster_bootstrap.csv",
    "source_deletion_equal_domain_macro.csv",
    "source_deletion_domain_paired_deltas.csv",
    "source_deletion_paired_cluster_bootstrap.csv",
    "results_brief_zh.md",
    "qa_summary.json",
    "qa_summary.md",
    "artifact_sha256.csv",
    "run_manifest.json",
    "scientific_configuration.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("both", "generic-scaffold", "source-deletion"),
        default="both",
    )
    parser.add_argument("--data-file", type=Path, default=DATA_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--outer-repeats", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-outer-splits",
        type=int,
        default=None,
        help="Smoke helper: truncate the generic-scaffold outer splits.",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help="Smoke helper: truncate the ordered held-target list.",
    )
    parser.add_argument(
        "--max-deletions",
        type=int,
        default=None,
        help="Smoke helper: keep baseline plus this many source deletions.",
    )
    parser.add_argument(
        "--only-target",
        choices=TARGETS,
        default=None,
        help="Audit helper: run only this held-target stratum.",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Audit helper: run only the no-source-deletion condition.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.outer_folds < 2 or args.inner_folds < 2:
        parser.error("outer-folds and inner-folds must be >= 2")
    if args.outer_repeats < 1:
        parser.error("outer-repeats must be >= 1")
    if args.n_estimators < 10:
        parser.error("n-estimators must be >= 10")
    if not 1 <= args.n_jobs <= 4:
        parser.error("n-jobs must be between 1 and 4")
    if args.bootstrap_replicates < 0:
        parser.error("bootstrap-replicates must be >= 0")
    for field in ("max_outer_splits", "max_targets", "max_deletions"):
        value = getattr(args, field)
        if value is not None and value < 1:
            parser.error(f"{field.replace('_', '-')} must be >= 1")
    if args.resume and args.overwrite:
        parser.error("resume and overwrite are mutually exclusive")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(values: Iterable[Any]) -> str:
    payload = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_ready)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    return pd.read_csv(path).to_dict("records")


def prepare_output_dir(args: argparse.Namespace) -> None:
    output_name = args.output_dir.resolve().name
    if "_v2" not in output_name or "_v1" in output_name:
        raise ValueError(
            "Version-2 output must use an explicitly identified v2 directory "
            "and must not target a v1 directory"
        )
    superseded_marker = args.output_dir / "SUPERSEDED_DO_NOT_USE.md"
    if superseded_marker.exists():
        raise RuntimeError(
            "Refusing an output directory carrying a superseded-output marker: "
            f"{superseded_marker}"
        )
    if args.resume:
        identity_files = (
            args.output_dir / "run_manifest.json",
            args.output_dir / "scientific_configuration.json",
        )
        for identity_file in identity_files:
            if not identity_file.is_file():
                continue
            identity = json.loads(identity_file.read_text(encoding="utf-8"))
            if identity.get("protocol_version") != PROTOCOL_VERSION:
                raise RuntimeError(
                    "Refusing cross-version resume from "
                    f"{identity_file}: protocol_version="
                    f"{identity.get('protocol_version')!r}"
                )
            if identity.get("response_dtype") != RESPONSE_DTYPE.name:
                raise RuntimeError(
                    "Refusing resume without response_dtype=float32 in "
                    f"{identity_file}"
                )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    known = {
        *GENERIC_FILES.values(),
        *SOURCE_FILES.values(),
        *FINAL_FILES,
    }
    existing = [args.output_dir / name for name in known if (args.output_dir / name).exists()]
    if existing and not (args.overwrite or args.resume):
        raise FileExistsError(
            f"Output directory contains known artifacts; use --resume or "
            f"--overwrite: {args.output_dir}"
        )
    if args.overwrite:
        for path in existing:
            path.unlink()


def formal_settings_match(args: argparse.Namespace) -> bool:
    return bool(
        args.mode == "both"
        and args.outer_folds == 5
        and args.outer_repeats == 5
        and args.inner_folds == 4
        and args.n_estimators == 600
        and args.n_jobs <= 4
        and args.bootstrap_replicates == 10_000
        and args.seed == DEFAULT_SEED
        and args.max_outer_splits is None
        and args.max_targets is None
        and args.max_deletions is None
        and args.only_target is None
        and not args.baseline_only
        and sha256_file(args.data_file) == EXPECTED_DATA_SHA256
    )


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
        "cpu_count": os.cpu_count(),
        "n_jobs_cap": int(args.n_jobs),
    }


def build_generic_scaffold_ids(smiles_values: Iterable[str]) -> np.ndarray:
    identifiers: list[str] = []
    for smiles in smiles_values:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            raise ValueError(f"RDKit could not parse canonical SMILES: {smiles}")
        framework = MurckoScaffold.GetScaffoldForMol(mol)
        generic = MurckoScaffold.MakeScaffoldGeneric(framework)
        identifier = Chem.MolToSmiles(
            generic,
            canonical=True,
            isomericSmiles=False,
        )
        if not identifier:
            identifier = (
                "ACYCLIC::"
                + Chem.MolToSmiles(
                    mol,
                    canonical=True,
                    isomericSmiles=False,
                )
            )
        identifiers.append(identifier)
    return np.asarray(identifiers, dtype=object)


def generic_group_summary(
    rows: pd.DataFrame,
    generic_ids: np.ndarray,
    bemis_ids: np.ndarray,
) -> pd.DataFrame:
    unique, counts = np.unique(generic_ids, return_counts=True)
    return pd.DataFrame(
        [
            {
                "n_rows": int(len(rows)),
                "n_unique_smiles": int(rows["canonical_smiles"].nunique()),
                "n_bemis_murcko_groups": int(len(np.unique(bemis_ids))),
                "n_generic_murcko_groups": int(len(unique)),
                "n_singleton_generic_groups": int(np.sum(counts == 1)),
                "largest_generic_group_rows": int(counts.max()),
                "median_generic_group_rows": float(np.median(counts)),
                "generic_definition": (
                    "RDKit MurckoScaffold.GetScaffoldForMol followed by "
                    "MakeScaffoldGeneric; compound-specific acyclic fallback"
                ),
            }
        ]
    )


def inner_split_audit(
    rows: pd.DataFrame,
    train_idx: np.ndarray,
    group_ids: np.ndarray,
    n_folds: int,
    seed: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for fold, (fit_idx, eval_idx) in enumerate(
        core.balanced_group_splits(train_idx, group_ids, n_folds, seed),
        start=1,
    ):
        fit_groups = set(group_ids[fit_idx].astype(str))
        eval_groups = set(group_ids[eval_idx].astype(str))
        overlap = fit_groups.intersection(eval_groups)
        records.append(
            {
                "inner_fold": int(fold),
                "inner_seed": int(seed),
                "n_fit_rows": int(len(fit_idx)),
                "n_eval_rows": int(len(eval_idx)),
                "n_fit_unique_smiles": int(
                    rows.iloc[fit_idx]["canonical_smiles"].nunique()
                ),
                "n_eval_unique_smiles": int(
                    rows.iloc[eval_idx]["canonical_smiles"].nunique()
                ),
                "n_fit_groups": int(len(fit_groups)),
                "n_eval_groups": int(len(eval_groups)),
                "n_group_overlap": int(len(overlap)),
                "fit_row_indices_sha256": sequence_sha256(fit_idx),
                "eval_row_indices_sha256": sequence_sha256(eval_idx),
                "fit_group_ids_sha256": sequence_sha256(sorted(fit_groups)),
                "eval_group_ids_sha256": sequence_sha256(sorted(eval_groups)),
            }
        )
    return records


def selected_params(
    rows: pd.DataFrame,
    y: np.ndarray,
    morgan: np.ndarray,
    descriptors: np.ndarray,
    train_idx: np.ndarray,
    groups: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    return core.select_model_params(
        list(MODELS),
        rows,
        y,
        morgan,
        descriptors,
        train_idx,
        groups,
        args.inner_folds,
        seed,
        args.n_estimators,
        args.n_jobs,
    )


def fit_two_models(
    rows: pd.DataFrame,
    y: np.ndarray,
    morgan: np.ndarray,
    descriptors: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    selected: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    seed: int,
    model_seeds: dict[str, int] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    fold_features, metadata = core.build_fold_features(
        rows,
        morgan,
        descriptors,
        train_idx,
        test_idx,
    )
    predictions: dict[str, np.ndarray] = {}
    for position, model_id in enumerate(MODELS):
        predictions[model_id] = core.fit_predict_model(
            model_id,
            selected[model_id],
            rows,
            y,
            morgan,
            fold_features,
            train_idx,
            test_idx,
            (
                int(model_seeds[model_id])
                if model_seeds is not None
                else seed + position
            ),
            args.n_estimators,
            args.n_jobs,
        )
        if not np.all(np.isfinite(predictions[model_id])):
            raise RuntimeError(f"Non-finite prediction from {model_id}")
    return predictions, metadata


def save_record_sets(
    output_dir: Path,
    filenames: dict[str, str],
    records: dict[str, list[dict[str, Any]]],
) -> None:
    for key, filename in filenames.items():
        atomic_csv(pd.DataFrame(records[key]), output_dir / filename)


def generic_split_complete(
    split: core.OuterSplit,
    records: dict[str, list[dict[str, Any]]],
    expected_tuning: int,
    inner_folds: int,
) -> bool:
    key = (split.repeat, split.fold)
    local_predictions = [
        row
        for row in records["predictions"]
        if (int(row["repeat"]), int(row["outer_fold"])) == key
    ]
    local_tuning = [
        row
        for row in records["tuning"]
        if (int(row["repeat"]), int(row["outer_fold"])) == key
    ]
    local_selected = [
        row
        for row in records["selected"]
        if (int(row["repeat"]), int(row["outer_fold"])) == key
    ]
    local_outer = [
        row
        for row in records["outer_audit"]
        if (int(row["repeat"]), int(row["outer_fold"])) == key
    ]
    local_inner = [
        row
        for row in records["inner_audit"]
        if (int(row["repeat"]), int(row["outer_fold"])) == key
    ]
    return bool(
        len(local_predictions) == len(split.test_idx) * len(MODELS)
        and len(local_tuning) == expected_tuning
        and len(local_selected) == len(MODELS)
        and len(local_outer) == 1
        and len(local_inner) == inner_folds
    )


def discard_generic_split(
    split: core.OuterSplit,
    records: dict[str, list[dict[str, Any]]],
) -> None:
    key = (split.repeat, split.fold)
    for name in records:
        records[name] = [
            row
            for row in records[name]
            if (int(row["repeat"]), int(row["outer_fold"])) != key
        ]


def run_generic_scaffold(
    args: argparse.Namespace,
    rows: pd.DataFrame,
    y: np.ndarray,
    morgan: np.ndarray,
    descriptors: np.ndarray,
    generic_ids: np.ndarray,
    bemis_ids: np.ndarray,
) -> pd.DataFrame:
    output_dir = args.output_dir
    records = {
        key: read_records(output_dir / filename) if args.resume else []
        for key, filename in GENERIC_FILES.items()
    }
    splits = core.make_outer_splits(
        "scaffold",
        rows,
        generic_ids,
        args.outer_folds,
        args.outer_repeats,
        args.seed,
    )
    if args.max_outer_splits is not None:
        splits = splits[: args.max_outer_splits]
    expected_tuning = sum(len(core.model_param_grid(model)) for model in MODELS)
    started = time.perf_counter()
    for split_number, split in enumerate(splits, start=1):
        if args.resume and generic_split_complete(
            split,
            records,
            expected_tuning,
            args.inner_folds,
        ):
            print(
                f"[GENERIC RESUME] repeat={split.repeat} fold={split.fold}",
                flush=True,
            )
            continue
        discard_generic_split(split, records)
        train_groups = set(generic_ids[split.train_idx].astype(str))
        test_groups = set(generic_ids[split.test_idx].astype(str))
        overlap = train_groups.intersection(test_groups)
        if overlap:
            raise RuntimeError("Outer generic-scaffold leakage")
        inner_seed = args.seed + split.repeat * 10_000 + split.fold * 100
        local_inner = inner_split_audit(
            rows,
            split.train_idx,
            generic_ids,
            args.inner_folds,
            inner_seed,
        )
        if any(row["n_group_overlap"] != 0 for row in local_inner):
            raise RuntimeError("Inner generic-scaffold leakage")
        selected, tuning = selected_params(
            rows,
            y,
            morgan,
            descriptors,
            split.train_idx,
            generic_ids,
            args,
            inner_seed,
        )
        for row in tuning:
            records["tuning"].append(
                {
                    "repeat": int(split.repeat),
                    "outer_fold": int(split.fold),
                    **row,
                }
            )
        for model_id, params in selected.items():
            records["selected"].append(
                {
                    "repeat": int(split.repeat),
                    "outer_fold": int(split.fold),
                    "model_id": model_id,
                    **params,
                }
            )
        for row in local_inner:
            records["inner_audit"].append(
                {
                    "repeat": int(split.repeat),
                    "outer_fold": int(split.fold),
                    **row,
                }
            )
        records["outer_audit"].append(
            {
                "repeat": int(split.repeat),
                "outer_fold": int(split.fold),
                "split_seed": int(split.split_seed),
                "n_train_rows": int(len(split.train_idx)),
                "n_test_rows": int(len(split.test_idx)),
                "n_train_groups": int(len(train_groups)),
                "n_test_groups": int(len(test_groups)),
                "n_group_overlap": int(len(overlap)),
                "n_train_unique_smiles": int(
                    rows.iloc[split.train_idx]["canonical_smiles"].nunique()
                ),
                "n_test_unique_smiles": int(
                    rows.iloc[split.test_idx]["canonical_smiles"].nunique()
                ),
                "train_row_indices_sha256": sequence_sha256(split.train_idx),
                "test_row_indices_sha256": sequence_sha256(split.test_idx),
            }
        )
        prediction_map, feature_meta = fit_two_models(
            rows,
            y,
            morgan,
            descriptors,
            split.train_idx,
            split.test_idx,
            selected,
            args,
            args.seed
            + split.repeat * 100_000
            + split.fold * 1_000,
            model_seeds={
                "chemistry_extra_trees": (
                    args.seed
                    + split.repeat * 100_000
                    + split.fold * 1_000
                    + 5
                ),
                "full_context_extra_trees": (
                    args.seed
                    + split.repeat * 100_000
                    + split.fold * 1_000
                    + 6
                ),
            },
        )
        for model_id, prediction in prediction_map.items():
            metrics = core.regression_metrics(y[split.test_idx], prediction)
            records["fold_metrics"].append(
                {
                    "repeat": int(split.repeat),
                    "outer_fold": int(split.fold),
                    "model_id": model_id,
                    "param_id": selected[model_id]["param_id"],
                    "n_train_rows": int(len(split.train_idx)),
                    "n_test_rows": int(len(split.test_idx)),
                    "n_test_unique_smiles": int(
                        rows.iloc[split.test_idx]["canonical_smiles"].nunique()
                    ),
                    "n_test_generic_scaffolds": int(len(test_groups)),
                    **metrics,
                    **feature_meta,
                }
            )
            for local_position, row_idx in enumerate(split.test_idx):
                source_row = rows.iloc[row_idx]
                records["predictions"].append(
                    {
                        "protocol": "generic_scaffold",
                        "repeat": int(split.repeat),
                        "outer_fold": int(split.fold),
                        "row_index": int(row_idx),
                        "qc_id": str(source_row["qc_id"]),
                        "model_id": model_id,
                        "param_id": selected[model_id]["param_id"],
                        "y_true": float(y[row_idx]),
                        "y_pred": float(prediction[local_position]),
                        "canonical_smiles": str(source_row["canonical_smiles"]),
                        "generic_scaffold_id": str(generic_ids[row_idx]),
                        "bemis_murcko_scaffold_id": str(bemis_ids[row_idx]),
                        "source_database": str(source_row["source_database"]),
                        "target_protein": str(source_row["target_protein"]),
                    }
                )
        save_record_sets(output_dir, GENERIC_FILES, records)
        elapsed = time.perf_counter() - started
        print(
            (
                f"[GENERIC] {split_number}/{len(splits)} "
                f"repeat={split.repeat} fold={split.fold} elapsed={elapsed:.1f}s"
            ),
            flush=True,
        )
    predictions = pd.DataFrame(records["predictions"])
    return predictions


def generic_summaries(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repeat_rows: list[dict[str, Any]] = []
    for (repeat, model_id), frame in predictions.groupby(
        ["repeat", "model_id"],
        sort=False,
    ):
        repeat_rows.append(
            {
                "repeat": int(repeat),
                "model_id": model_id,
                "n_rows": int(len(frame)),
                "n_unique_smiles": int(frame["canonical_smiles"].nunique()),
                "n_generic_scaffolds": int(
                    frame["generic_scaffold_id"].nunique()
                ),
                **core.regression_metrics(
                    frame["y_true"].to_numpy(dtype=np.float64),
                    frame["y_pred"].to_numpy(dtype=np.float64),
                ),
            }
        )
    repeat_metrics = pd.DataFrame(repeat_rows)
    averaged = (
        predictions.groupby(
            [
                "model_id",
                "row_index",
                "qc_id",
                "y_true",
                "canonical_smiles",
                "generic_scaffold_id",
                "bemis_murcko_scaffold_id",
                "source_database",
                "target_protein",
            ],
            sort=False,
            dropna=False,
        )
        .agg(
            n_oof_predictions=("y_pred", "size"),
            n_repeats=("repeat", "nunique"),
            y_pred=("y_pred", "mean"),
            y_pred_sd_across_repeats=("y_pred", "std"),
        )
        .reset_index()
    )
    averaged_rows: list[dict[str, Any]] = []
    for model_id, frame in averaged.groupby("model_id", sort=False):
        averaged_rows.append(
            {
                "model_id": model_id,
                "n_rows": int(len(frame)),
                "n_unique_smiles": int(frame["canonical_smiles"].nunique()),
                "n_generic_scaffolds": int(
                    frame["generic_scaffold_id"].nunique()
                ),
                "n_oof_predictions_per_row_min": int(
                    frame["n_oof_predictions"].min()
                ),
                "n_oof_predictions_per_row_max": int(
                    frame["n_oof_predictions"].max()
                ),
                **core.regression_metrics(
                    frame["y_true"].to_numpy(dtype=np.float64),
                    frame["y_pred"].to_numpy(dtype=np.float64),
                ),
            }
        )
    return repeat_metrics, averaged, pd.DataFrame(averaged_rows)


def finite_interval(values: Iterable[float]) -> tuple[float, float, float, int]:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    return (
        float(np.mean(finite)),
        float(np.percentile(finite, 2.5)),
        float(np.percentile(finite, 97.5)),
        int(len(finite)),
    )


def generic_cluster_bootstrap(
    averaged: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    if n_bootstrap <= 0 or averaged.empty:
        return pd.DataFrame()
    reference = (
        averaged[averaged["model_id"] == MODELS[0]]
        .sort_values("row_index")
        .reset_index(drop=True)
    )
    row_indices = reference["row_index"].to_numpy(dtype=np.int64)
    y_true = reference["y_true"].to_numpy(dtype=np.float64)
    clusters = reference["generic_scaffold_id"].astype(str).to_numpy()
    unique_clusters = np.unique(clusters)
    cluster_positions = {
        cluster: np.where(clusters == cluster)[0] for cluster in unique_clusters
    }
    model_predictions: dict[str, np.ndarray] = {}
    observed: dict[str, dict[str, float]] = {}
    for model_id in MODELS:
        frame = (
            averaged[averaged["model_id"] == model_id]
            .sort_values("row_index")
            .reset_index(drop=True)
        )
        if not np.array_equal(
            frame["row_index"].to_numpy(dtype=np.int64),
            row_indices,
        ):
            raise RuntimeError("Generic bootstrap requires aligned model rows")
        prediction = frame["y_pred"].to_numpy(dtype=np.float64)
        model_predictions[model_id] = prediction
        observed[model_id] = core.regression_metrics(y_true, prediction)
    distribution = {
        model: {"spearman": [], "rmse": []} for model in MODELS
    }
    delta_distribution = {"spearman": [], "rmse": []}
    rng = np.random.default_rng(seed + 7_100_000)
    for bootstrap_idx in range(n_bootstrap):
        sampled = rng.choice(
            unique_clusters,
            size=len(unique_clusters),
            replace=True,
        )
        positions = np.concatenate([cluster_positions[value] for value in sampled])
        local_metrics = {}
        for model_id in MODELS:
            metrics = core.regression_metrics(
                y_true[positions],
                model_predictions[model_id][positions],
            )
            local_metrics[model_id] = metrics
            distribution[model_id]["spearman"].append(metrics["spearman"])
            distribution[model_id]["rmse"].append(metrics["rmse"])
        delta_distribution["spearman"].append(
            local_metrics["full_context_extra_trees"]["spearman"]
            - local_metrics["chemistry_extra_trees"]["spearman"]
        )
        delta_distribution["rmse"].append(
            local_metrics["chemistry_extra_trees"]["rmse"]
            - local_metrics["full_context_extra_trees"]["rmse"]
        )
        if (bootstrap_idx + 1) % max(1, n_bootstrap // 10) == 0:
            print(
                f"[GENERIC BOOT] {bootstrap_idx + 1}/{n_bootstrap}",
                flush=True,
            )
    result: list[dict[str, Any]] = []
    for model_id in MODELS:
        for metric in ("spearman", "rmse"):
            mean, low, high, valid = finite_interval(
                distribution[model_id][metric]
            )
            result.append(
                {
                    "record_type": "model",
                    "model_id": model_id,
                    "contrast_id": "",
                    "metric": metric,
                    "observed": float(observed[model_id][metric]),
                    "bootstrap_mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "n_valid": valid,
                    "n_bootstrap": int(n_bootstrap),
                    "cluster_unit": "generic_murcko_scaffold",
                    "n_clusters": int(len(unique_clusters)),
                    "positive_direction": (
                        "higher is better" if metric == "spearman" else "lower is better"
                    ),
                }
            )
    full_observed = observed["full_context_extra_trees"]
    chemistry_observed = observed["chemistry_extra_trees"]
    for metric in ("spearman", "rmse"):
        mean, low, high, valid = finite_interval(delta_distribution[metric])
        observed_delta = (
            full_observed["spearman"] - chemistry_observed["spearman"]
            if metric == "spearman"
            else chemistry_observed["rmse"] - full_observed["rmse"]
        )
        result.append(
            {
                "record_type": "contrast",
                "model_id": "",
                "contrast_id": "full_vs_chemistry",
                "metric": f"delta_{metric}",
                "observed": float(observed_delta),
                "bootstrap_mean": mean,
                "ci_low": low,
                "ci_high": high,
                "n_valid": valid,
                "n_bootstrap": int(n_bootstrap),
                "cluster_unit": "generic_murcko_scaffold",
                "n_clusters": int(len(unique_clusters)),
                "positive_direction": "positive favors full-context",
            }
        )
    return pd.DataFrame(result)


def selected_targets(args: argparse.Namespace) -> tuple[str, ...]:
    if args.only_target is not None:
        return (str(args.only_target),)
    return TARGETS if args.max_targets is None else TARGETS[: args.max_targets]


def selected_deletion_conditions(args: argparse.Namespace) -> tuple[str, ...]:
    if args.baseline_only:
        return (BASELINE_CONDITION,)
    deletions = (
        SOURCE_DELETIONS
        if args.max_deletions is None
        else SOURCE_DELETIONS[: args.max_deletions]
    )
    return (BASELINE_CONDITION, *deletions)


def source_condition_complete(
    heldout_target: str,
    condition: str,
    test_size: int,
    records: dict[str, list[dict[str, Any]]],
    expected_tuning: int,
    inner_folds: int,
) -> bool:
    def local(name: str) -> list[dict[str, Any]]:
        return [
            row
            for row in records[name]
            if str(row["heldout_target"]) == heldout_target
            and str(row["deletion_source"]) == condition
        ]

    return bool(
        len(local("predictions")) == test_size * len(MODELS)
        and len(local("domain_metrics")) == len(MODELS)
        and len(local("tuning")) == expected_tuning
        and len(local("selected")) == len(MODELS)
        and len(local("split_audit")) == 1
        and len(local("inner_audit")) == inner_folds
    )


def discard_source_condition(
    heldout_target: str,
    condition: str,
    records: dict[str, list[dict[str, Any]]],
) -> None:
    for name in records:
        records[name] = [
            row
            for row in records[name]
            if not (
                str(row["heldout_target"]) == heldout_target
                and str(row["deletion_source"]) == condition
            )
        ]


def run_source_deletion(
    args: argparse.Namespace,
    rows: pd.DataFrame,
    y: np.ndarray,
    morgan: np.ndarray,
    descriptors: np.ndarray,
    bemis_ids: np.ndarray,
) -> pd.DataFrame:
    output_dir = args.output_dir
    records = {
        key: read_records(output_dir / filename) if args.resume else []
        for key, filename in SOURCE_FILES.items()
    }
    baseline_splits, _ = ood.build_ood_splits(
        rows,
        bemis_ids,
        ["target_ood"],
        args.seed,
        args.max_targets,
    )
    split_by_target = {split.heldout_group: split for split in baseline_splits}
    targets = selected_targets(args)
    conditions = selected_deletion_conditions(args)
    expected_tuning = sum(len(core.model_param_grid(model)) for model in MODELS)
    total = len(targets) * len(conditions)
    completed = 0
    started = time.perf_counter()
    source_values = rows["source_database"].astype(str)
    for target_position, heldout_target in enumerate(targets, start=1):
        split = split_by_target[heldout_target]
        for condition_position, condition in enumerate(conditions):
            completed += 1
            if args.resume and source_condition_complete(
                heldout_target,
                condition,
                len(split.test_idx),
                records,
                expected_tuning,
                args.inner_folds,
            ):
                print(
                    f"[SOURCE RESUME] target={heldout_target} delete={condition}",
                    flush=True,
                )
                continue
            discard_source_condition(heldout_target, condition, records)
            baseline_train_idx = split.train_idx
            if condition == BASELINE_CONDITION:
                keep = np.ones(len(baseline_train_idx), dtype=bool)
            else:
                keep = np.asarray(
                    [
                        condition not in ood.tokenize_domain(
                            source_values.iloc[row_idx]
                        )
                        for row_idx in baseline_train_idx
                    ],
                    dtype=bool,
                )
            train_idx = baseline_train_idx[keep]
            test_idx = split.test_idx
            n_train_scaffolds = int(len(np.unique(bemis_ids[train_idx])))
            if len(train_idx) == 0 or n_train_scaffolds < args.inner_folds:
                records["split_audit"].append(
                    {
                        "heldout_target": heldout_target,
                        "deletion_source": condition,
                        "status": "failed_infeasible",
                        "failure_reason": (
                            "empty training set"
                            if len(train_idx) == 0
                            else "fewer training scaffolds than inner folds"
                        ),
                        "n_baseline_train_rows": int(len(baseline_train_idx)),
                        "n_source_rows_removed": int(
                            len(baseline_train_idx) - len(train_idx)
                        ),
                        "n_train_rows": int(len(train_idx)),
                        "n_train_bemis_murcko_scaffolds": n_train_scaffolds,
                        "n_test_rows": int(len(test_idx)),
                    }
                )
                save_record_sets(output_dir, SOURCE_FILES, records)
                print(
                    (
                        f"[SOURCE FAILED] target={heldout_target} "
                        f"delete={condition}: infeasible training stratum"
                    ),
                    flush=True,
                )
                continue
            train_smiles = set(rows.iloc[train_idx]["canonical_smiles"].astype(str))
            test_smiles = set(rows.iloc[test_idx]["canonical_smiles"].astype(str))
            compound_overlap = train_smiles.intersection(test_smiles)
            held_tokens = ood.tokenize_domain(heldout_target)
            train_target_tokens: set[str] = set()
            for value in rows.iloc[train_idx]["target_protein"].astype(str):
                train_target_tokens.update(ood.tokenize_domain(value))
            source_tokens_after: set[str] = set()
            for value in rows.iloc[train_idx]["source_database"].astype(str):
                source_tokens_after.update(ood.tokenize_domain(value))
            if compound_overlap:
                raise RuntimeError("Exact-compound overlap after source deletion")
            if held_tokens.intersection(train_target_tokens):
                raise RuntimeError("Held-target token entered training set")
            if (
                condition != BASELINE_CONDITION
                and condition in source_tokens_after
            ):
                raise RuntimeError("Deleted source token remained in training set")
            inner_seed = (
                split.split_seed
                + 101
                + condition_position * 1_000_000
            )
            local_inner = inner_split_audit(
                rows,
                train_idx,
                bemis_ids,
                args.inner_folds,
                inner_seed,
            )
            if any(row["n_group_overlap"] != 0 for row in local_inner):
                raise RuntimeError("Inner BM-scaffold leakage")
            selected, tuning = selected_params(
                rows,
                y,
                morgan,
                descriptors,
                train_idx,
                bemis_ids,
                args,
                inner_seed,
            )
            for row in tuning:
                records["tuning"].append(
                    {
                        "heldout_target": heldout_target,
                        "deletion_source": condition,
                        **row,
                    }
                )
            for model_id, params in selected.items():
                records["selected"].append(
                    {
                        "heldout_target": heldout_target,
                        "deletion_source": condition,
                        "model_id": model_id,
                        **params,
                    }
                )
            for row in local_inner:
                records["inner_audit"].append(
                    {
                        "heldout_target": heldout_target,
                        "deletion_source": condition,
                        **row,
                    }
                )
            baseline_source_rows = int(len(baseline_train_idx))
            records["split_audit"].append(
                {
                    "heldout_target": heldout_target,
                    "deletion_source": condition,
                    "status": "complete",
                    "failure_reason": "",
                    "deletion_rule": (
                        "none"
                        if condition == BASELINE_CONDITION
                        else "remove row when plus-tokenized provenance contains source"
                    ),
                    "split_seed": int(split.split_seed),
                    "inner_seed": int(inner_seed),
                    "n_baseline_train_rows": baseline_source_rows,
                    "n_source_rows_removed": int(baseline_source_rows - len(train_idx)),
                    "n_train_rows": int(len(train_idx)),
                    "n_train_unique_smiles": int(
                        rows.iloc[train_idx]["canonical_smiles"].nunique()
                    ),
                    "n_train_bemis_murcko_scaffolds": int(
                        len(np.unique(bemis_ids[train_idx]))
                    ),
                    "n_test_rows": int(len(test_idx)),
                    "n_test_unique_smiles": int(
                        rows.iloc[test_idx]["canonical_smiles"].nunique()
                    ),
                    "n_test_bemis_murcko_scaffolds": int(
                        len(np.unique(bemis_ids[test_idx]))
                    ),
                    "train_test_row_overlap": int(
                        len(set(train_idx).intersection(set(test_idx)))
                    ),
                    "train_test_smiles_overlap": int(len(compound_overlap)),
                    "heldout_target_token_seen_in_train": bool(
                        held_tokens.intersection(train_target_tokens)
                    ),
                    "deleted_source_token_seen_in_train": bool(
                        condition != BASELINE_CONDITION
                        and condition in source_tokens_after
                    ),
                    "train_row_indices_sha256": sequence_sha256(train_idx),
                    "test_row_indices_sha256": sequence_sha256(test_idx),
                }
            )
            prediction_map, feature_meta = fit_two_models(
                rows,
                y,
                morgan,
                descriptors,
                train_idx,
                test_idx,
                selected,
                args,
                split.split_seed
                + 5_000_000
                + condition_position * 10_000,
                model_seeds={
                    "chemistry_extra_trees": (
                        split.split_seed + 1_005
                        if condition == BASELINE_CONDITION
                        else split.split_seed
                        + 5_000_000
                        + condition_position * 10_000
                        + 5
                    ),
                    "full_context_extra_trees": (
                        split.split_seed + 1_006
                        if condition == BASELINE_CONDITION
                        else split.split_seed
                        + 5_000_000
                        + condition_position * 10_000
                        + 6
                    ),
                },
            )
            for model_id, prediction in prediction_map.items():
                metrics = core.regression_metrics(y[test_idx], prediction)
                records["domain_metrics"].append(
                    {
                        "heldout_target": heldout_target,
                        "deletion_source": condition,
                        "model_id": model_id,
                        "param_id": selected[model_id]["param_id"],
                        "n_train_rows": int(len(train_idx)),
                        "n_test_rows": int(len(test_idx)),
                        **metrics,
                        **feature_meta,
                    }
                )
                for local_position, row_idx in enumerate(test_idx):
                    source_row = rows.iloc[row_idx]
                    records["predictions"].append(
                        {
                            "heldout_target": heldout_target,
                            "deletion_source": condition,
                            "row_index": int(row_idx),
                            "qc_id": str(source_row["qc_id"]),
                            "model_id": model_id,
                            "param_id": selected[model_id]["param_id"],
                            "y_true": float(y[row_idx]),
                            "y_pred": float(prediction[local_position]),
                            "canonical_smiles": str(
                                source_row["canonical_smiles"]
                            ),
                            "bemis_murcko_scaffold_id": str(
                                bemis_ids[row_idx]
                            ),
                            "source_database": str(
                                source_row["source_database"]
                            ),
                            "target_protein": str(
                                source_row["target_protein"]
                            ),
                        }
                    )
            save_record_sets(output_dir, SOURCE_FILES, records)
            elapsed = time.perf_counter() - started
            print(
                (
                    f"[SOURCE] {completed}/{total} target={heldout_target} "
                    f"delete={condition} train={len(train_idx)} "
                    f"elapsed={elapsed:.1f}s"
                ),
                flush=True,
            )
    return pd.DataFrame(records["predictions"])


def source_domain_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (condition, target, model_id), frame in predictions.groupby(
        ["deletion_source", "heldout_target", "model_id"],
        sort=False,
    ):
        records.append(
            {
                "deletion_source": condition,
                "heldout_target": target,
                "model_id": model_id,
                "n_rows": int(len(frame)),
                "n_unique_smiles": int(frame["canonical_smiles"].nunique()),
                "n_bemis_murcko_scaffolds": int(
                    frame["bemis_murcko_scaffold_id"].nunique()
                ),
                **core.regression_metrics(
                    frame["y_true"].to_numpy(dtype=np.float64),
                    frame["y_pred"].to_numpy(dtype=np.float64),
                ),
            }
        )
    return pd.DataFrame(records)


def source_macro_and_deltas(
    domain_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    macro_records: list[dict[str, Any]] = []
    for (condition, model_id), frame in domain_metrics.groupby(
        ["deletion_source", "model_id"],
        sort=False,
    ):
        record: dict[str, Any] = {
            "record_type": "model",
            "deletion_source": condition,
            "model_id": model_id,
            "contrast_id": "",
            "n_domains": int(frame["heldout_target"].nunique()),
        }
        for metric in ("spearman", "rmse", "mae", "r2"):
            values = frame[metric].to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            record[f"domain_macro_{metric}"] = (
                float(np.mean(finite))
                if len(finite) == len(values)
                else float("nan")
            )
            record[f"finite_domains_{metric}"] = int(len(finite))
        macro_records.append(record)
    delta_records: list[dict[str, Any]] = []
    by_key = {
        (
            str(row.deletion_source),
            str(row.heldout_target),
            str(row.model_id),
        ): row
        for row in domain_metrics.itertuples(index=False)
    }
    conditions = tuple(
        dict.fromkeys(domain_metrics["deletion_source"].astype(str))
    )
    targets = tuple(dict.fromkeys(domain_metrics["heldout_target"].astype(str)))
    for condition in conditions:
        for target in targets:
            chemistry = by_key[(condition, target, MODELS[0])]
            full = by_key[(condition, target, MODELS[1])]
            delta_records.append(
                {
                    "contrast_type": "full_vs_chemistry",
                    "deletion_source": condition,
                    "heldout_target": target,
                    "model_id": "",
                    "delta_spearman": float(
                        full.spearman - chemistry.spearman
                    ),
                    "delta_rmse": float(chemistry.rmse - full.rmse),
                    "positive_direction": "positive favors full-context",
                }
            )
    for condition in conditions:
        if condition == BASELINE_CONDITION:
            continue
        for target in targets:
            for model_id in MODELS:
                baseline = by_key[(BASELINE_CONDITION, target, model_id)]
                deletion = by_key[(condition, target, model_id)]
                delta_records.append(
                    {
                        "contrast_type": "deletion_vs_baseline",
                        "deletion_source": condition,
                        "heldout_target": target,
                        "model_id": model_id,
                        "delta_spearman": float(
                            deletion.spearman - baseline.spearman
                        ),
                        "delta_rmse": float(baseline.rmse - deletion.rmse),
                        "positive_direction": "positive favors source deletion",
                    }
                )
    deltas = pd.DataFrame(delta_records)
    for (contrast_type, condition, model_id), frame in deltas.groupby(
        ["contrast_type", "deletion_source", "model_id"],
        sort=False,
    ):
        macro_records.append(
            {
                "record_type": "contrast",
                "deletion_source": condition,
                "model_id": model_id,
                "contrast_id": contrast_type,
                "n_domains": int(frame["heldout_target"].nunique()),
                "domain_macro_spearman": float(
                    frame["delta_spearman"].mean()
                ),
                "finite_domains_spearman": int(
                    np.isfinite(frame["delta_spearman"]).sum()
                ),
                "domain_macro_rmse": float(frame["delta_rmse"].mean()),
                "finite_domains_rmse": int(
                    np.isfinite(frame["delta_rmse"]).sum()
                ),
                "domain_macro_mae": float("nan"),
                "finite_domains_mae": 0,
                "domain_macro_r2": float("nan"),
                "finite_domains_r2": 0,
            }
        )
    return pd.DataFrame(macro_records), deltas


def source_stratified_cluster_bootstrap(
    predictions: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    if n_bootstrap <= 0 or predictions.empty:
        return pd.DataFrame()
    conditions = tuple(
        dict.fromkeys(predictions["deletion_source"].astype(str))
    )
    targets = tuple(dict.fromkeys(predictions["heldout_target"].astype(str)))
    reference = (
        predictions[
            (predictions["deletion_source"] == BASELINE_CONDITION)
            & (predictions["model_id"] == MODELS[0])
        ]
        .sort_values(["heldout_target", "row_index"])
        .reset_index(drop=True)
    )
    reference_keys = list(
        zip(
            reference["heldout_target"].astype(str),
            reference["row_index"].astype(int),
        )
    )
    y_true = reference["y_true"].to_numpy(dtype=np.float64)
    target_values = reference["heldout_target"].astype(str).to_numpy()
    scaffold_values = reference["bemis_murcko_scaffold_id"].astype(str).to_numpy()
    prediction_arrays: dict[tuple[str, str], np.ndarray] = {}
    for condition in conditions:
        for model_id in MODELS:
            frame = (
                predictions[
                    (predictions["deletion_source"] == condition)
                    & (predictions["model_id"] == model_id)
                ]
                .sort_values(["heldout_target", "row_index"])
                .reset_index(drop=True)
            )
            keys = list(
                zip(
                    frame["heldout_target"].astype(str),
                    frame["row_index"].astype(int),
                )
            )
            if keys != reference_keys:
                raise RuntimeError(
                    "Source-deletion bootstrap requires aligned test rows"
                )
            if not np.array_equal(
                frame["y_true"].to_numpy(dtype=np.float64),
                y_true,
            ):
                raise RuntimeError("Outcome mismatch across deletion conditions")
            prediction_arrays[(condition, model_id)] = frame[
                "y_pred"
            ].to_numpy(dtype=np.float64)
    unique_global_scaffolds = np.unique(scaffold_values)
    global_cluster_positions = {
        scaffold: np.where(scaffold_values == scaffold)[0]
        for scaffold in unique_global_scaffolds
    }

    performance_distribution: dict[
        tuple[str, str, str], list[float]
    ] = {}
    contrast_distribution: dict[tuple[str, str, str, str], list[float]] = {}
    for condition in conditions:
        for model_id in MODELS:
            for metric in ("spearman", "rmse"):
                performance_distribution[(condition, model_id, metric)] = []
        for metric in ("spearman", "rmse"):
            contrast_distribution[
                ("full_vs_chemistry", condition, "", metric)
            ] = []
            for target in targets:
                contrast_distribution[
                    ("full_vs_chemistry_domain", condition, target, metric)
                ] = []
    for condition in conditions:
        if condition == BASELINE_CONDITION:
            continue
        for model_id in MODELS:
            for metric in ("spearman", "rmse"):
                contrast_distribution[
                    ("deletion_vs_baseline", condition, model_id, metric)
                ] = []
                for target in targets:
                    contrast_distribution[
                        (
                            "deletion_vs_baseline_domain",
                            condition,
                            f"{model_id}||{target}",
                            metric,
                        )
                    ] = []

    rng = np.random.default_rng(seed + 8_200_000)
    for bootstrap_idx in range(n_bootstrap):
        sampled_clusters = rng.choice(
            unique_global_scaffolds,
            size=len(unique_global_scaffolds),
            replace=True,
        )
        sampled_positions = np.concatenate(
            [global_cluster_positions[cluster] for cluster in sampled_clusters]
        )
        sampled_by_target = {
            target: sampled_positions[
                target_values[sampled_positions] == target
            ]
            for target in targets
        }
        domain_metrics: dict[
            tuple[str, str, str], dict[str, float]
        ] = {}
        for condition in conditions:
            for model_id in MODELS:
                for target in targets:
                    positions = sampled_by_target[target]
                    if len(positions) == 0:
                        domain_metrics[(condition, model_id, target)] = {
                            "spearman": float("nan"),
                            "rmse": float("nan"),
                        }
                    else:
                        domain_metrics[(condition, model_id, target)] = core.regression_metrics(
                            y_true[positions],
                            prediction_arrays[(condition, model_id)][positions],
                        )
                for metric in ("spearman", "rmse"):
                    values = [
                        domain_metrics[(condition, model_id, target)][metric]
                        for target in targets
                    ]
                    performance_distribution[
                        (condition, model_id, metric)
                    ].append(
                        float(np.mean(values))
                        if np.all(np.isfinite(values))
                        else float("nan")
                    )
            for metric in ("spearman", "rmse"):
                direction = 1.0 if metric == "spearman" else -1.0
                domain_deltas = []
                for target in targets:
                    delta = direction * (
                        domain_metrics[(condition, MODELS[1], target)][metric]
                        - domain_metrics[(condition, MODELS[0], target)][metric]
                    )
                    domain_deltas.append(delta)
                    contrast_distribution[
                        ("full_vs_chemistry_domain", condition, target, metric)
                    ].append(float(delta))
                contrast_distribution[
                    ("full_vs_chemistry", condition, "", metric)
                ].append(
                    float(np.mean(domain_deltas))
                    if np.all(np.isfinite(domain_deltas))
                    else float("nan")
                )
        for condition in conditions:
            if condition == BASELINE_CONDITION:
                continue
            for model_id in MODELS:
                for metric in ("spearman", "rmse"):
                    direction = 1.0 if metric == "spearman" else -1.0
                    domain_deltas = []
                    for target in targets:
                        delta = direction * (
                            domain_metrics[(condition, model_id, target)][metric]
                            - domain_metrics[
                                (BASELINE_CONDITION, model_id, target)
                            ][metric]
                        )
                        domain_deltas.append(delta)
                        contrast_distribution[
                            (
                                "deletion_vs_baseline_domain",
                                condition,
                                f"{model_id}||{target}",
                                metric,
                            )
                        ].append(float(delta))
                    contrast_distribution[
                        ("deletion_vs_baseline", condition, model_id, metric)
                    ].append(
                        float(np.mean(domain_deltas))
                        if np.all(np.isfinite(domain_deltas))
                        else float("nan")
                    )
        if (bootstrap_idx + 1) % max(1, n_bootstrap // 10) == 0:
            print(
                f"[SOURCE BOOT] {bootstrap_idx + 1}/{n_bootstrap}",
                flush=True,
            )

    observed_domain = source_domain_metrics(predictions)
    observed_lookup = {
        (
            str(row.deletion_source),
            str(row.model_id),
            str(row.heldout_target),
        ): row
        for row in observed_domain.itertuples(index=False)
    }
    output: list[dict[str, Any]] = []
    for (condition, model_id, metric), values in performance_distribution.items():
        observed_values = [
            float(getattr(observed_lookup[(condition, model_id, target)], metric))
            for target in targets
        ]
        mean, low, high, valid = finite_interval(values)
        output.append(
            {
                "record_type": "model_equal_domain_macro",
                "contrast_type": "",
                "deletion_source": condition,
                "model_id": model_id,
                "heldout_target": "",
                "metric": metric,
                "observed": float(np.mean(observed_values)),
                "bootstrap_mean": mean,
                "ci_low": low,
                "ci_high": high,
                "n_valid": valid,
                "n_bootstrap": int(n_bootstrap),
                "cluster_unit": (
                    "global Bemis-Murcko scaffold cluster across fixed targets"
                ),
                "n_global_scaffold_clusters": int(
                    len(unique_global_scaffolds)
                ),
                "positive_direction": (
                    "higher is better"
                    if metric == "spearman"
                    else "lower is better"
                ),
            }
        )
    for (contrast_type, condition, identity, metric), values in (
        contrast_distribution.items()
    ):
        if contrast_type.startswith("full_vs_chemistry"):
            model_id = ""
            target = identity
            if target:
                full_value = float(
                    getattr(
                        observed_lookup[(condition, MODELS[1], target)],
                        metric,
                    )
                )
                chemistry_value = float(
                    getattr(
                        observed_lookup[(condition, MODELS[0], target)],
                        metric,
                    )
                )
                observed_value = (
                    full_value - chemistry_value
                    if metric == "spearman"
                    else chemistry_value - full_value
                )
            else:
                domain_values = []
                for local_target in targets:
                    full_value = float(
                        getattr(
                            observed_lookup[
                                (condition, MODELS[1], local_target)
                            ],
                            metric,
                        )
                    )
                    chemistry_value = float(
                        getattr(
                            observed_lookup[
                                (condition, MODELS[0], local_target)
                            ],
                            metric,
                        )
                    )
                    domain_values.append(
                        full_value - chemistry_value
                        if metric == "spearman"
                        else chemistry_value - full_value
                    )
                observed_value = float(np.mean(domain_values))
            direction_label = "positive favors full-context"
        else:
            if contrast_type.endswith("_domain"):
                model_id, target = identity.split("||", maxsplit=1)
                deletion_value = float(
                    getattr(
                        observed_lookup[(condition, model_id, target)],
                        metric,
                    )
                )
                baseline_value = float(
                    getattr(
                        observed_lookup[
                            (BASELINE_CONDITION, model_id, target)
                        ],
                        metric,
                    )
                )
                observed_value = (
                    deletion_value - baseline_value
                    if metric == "spearman"
                    else baseline_value - deletion_value
                )
            else:
                model_id = identity
                target = ""
                domain_values = []
                for local_target in targets:
                    deletion_value = float(
                        getattr(
                            observed_lookup[
                                (condition, model_id, local_target)
                            ],
                            metric,
                        )
                    )
                    baseline_value = float(
                        getattr(
                            observed_lookup[
                                (
                                    BASELINE_CONDITION,
                                    model_id,
                                    local_target,
                                )
                            ],
                            metric,
                        )
                    )
                    domain_values.append(
                        deletion_value - baseline_value
                        if metric == "spearman"
                        else baseline_value - deletion_value
                    )
                observed_value = float(np.mean(domain_values))
            direction_label = "positive favors source deletion"
        mean, low, high, valid = finite_interval(values)
        output.append(
            {
                "record_type": contrast_type,
                "contrast_type": contrast_type.replace("_domain", ""),
                "deletion_source": condition,
                "model_id": model_id,
                "heldout_target": target,
                "metric": f"delta_{metric}",
                "observed": observed_value,
                "bootstrap_mean": mean,
                "ci_low": low,
                "ci_high": high,
                "n_valid": valid,
                "n_bootstrap": int(n_bootstrap),
                "cluster_unit": (
                    "global Bemis-Murcko scaffold cluster across fixed targets"
                ),
                "n_global_scaffold_clusters": int(
                    len(unique_global_scaffolds)
                ),
                "positive_direction": direction_label,
            }
        )
    return pd.DataFrame(output)


def add_check(
    checks: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    observed: Any,
    expected: Any,
) -> None:
    checks.append(
        {
            "check_id": check_id,
            "status": "PASS" if passed else "FAIL",
            "observed": observed,
            "expected": expected,
        }
    )


def qa_results(
    args: argparse.Namespace,
    rows: pd.DataFrame,
    y: np.ndarray,
    generic_ids: np.ndarray,
    generic_predictions: pd.DataFrame,
    source_predictions: pd.DataFrame,
    generic_bootstrap: pd.DataFrame,
    source_bootstrap: pd.DataFrame,
) -> tuple[dict[str, Any], str]:
    checks: list[dict[str, Any]] = []
    add_check(
        checks,
        "training_response_dtype_float32",
        y.dtype == RESPONSE_DTYPE,
        str(y.dtype),
        RESPONSE_DTYPE.name,
    )
    add_check(
        checks,
        "frozen_data_sha256",
        sha256_file(args.data_file) == EXPECTED_DATA_SHA256,
        sha256_file(args.data_file),
        EXPECTED_DATA_SHA256,
    )
    add_check(
        checks,
        "analysis_row_count",
        len(rows) == EXPECTED_N_ROWS,
        len(rows),
        EXPECTED_N_ROWS,
    )
    unique, counts = np.unique(generic_ids, return_counts=True)
    add_check(
        checks,
        "generic_group_count",
        len(unique) == EXPECTED_GENERIC_GROUPS,
        len(unique),
        EXPECTED_GENERIC_GROUPS,
    )
    add_check(
        checks,
        "generic_singleton_count",
        int(np.sum(counts == 1)) == EXPECTED_GENERIC_SINGLETONS,
        int(np.sum(counts == 1)),
        EXPECTED_GENERIC_SINGLETONS,
    )
    add_check(
        checks,
        "generic_largest_group",
        int(counts.max()) == EXPECTED_GENERIC_MAX_ROWS,
        int(counts.max()),
        EXPECTED_GENERIC_MAX_ROWS,
    )
    if args.mode in ("both", "generic-scaffold"):
        outer_audit = pd.read_csv(
            args.output_dir / GENERIC_FILES["outer_audit"]
        )
        inner_audit = pd.read_csv(
            args.output_dir / GENERIC_FILES["inner_audit"]
        )
        add_check(
            checks,
            "generic_outer_group_overlap_zero",
            bool((outer_audit["n_group_overlap"] == 0).all()),
            int(outer_audit["n_group_overlap"].max()),
            0,
        )
        add_check(
            checks,
            "generic_inner_group_overlap_zero",
            bool((inner_audit["n_group_overlap"] == 0).all()),
            int(inner_audit["n_group_overlap"].max()),
            0,
        )
        add_check(
            checks,
            "generic_predictions_finite",
            bool(
                np.isfinite(
                    generic_predictions[["y_true", "y_pred"]].to_numpy(
                        dtype=np.float64
                    )
                ).all()
            ),
            int(
                np.isfinite(
                    generic_predictions[["y_true", "y_pred"]].to_numpy(
                        dtype=np.float64
                    )
                ).sum()
            ),
            int(len(generic_predictions) * 2),
        )
        duplicate_keys = int(
            generic_predictions.duplicated(
                ["repeat", "outer_fold", "row_index", "model_id"]
            ).sum()
        )
        add_check(
            checks,
            "generic_prediction_keys_unique",
            duplicate_keys == 0,
            duplicate_keys,
            0,
        )
        if formal_settings_match(args):
            expected = EXPECTED_N_ROWS * args.outer_repeats * len(MODELS)
            add_check(
                checks,
                "generic_formal_prediction_completeness",
                len(generic_predictions) == expected,
                len(generic_predictions),
                expected,
            )
            repeat_counts = generic_predictions.groupby(
                ["model_id", "row_index"]
            )["repeat"].nunique()
            add_check(
                checks,
                "generic_five_predictions_per_row_model",
                bool((repeat_counts == args.outer_repeats).all()),
                (
                    int(repeat_counts.min()),
                    int(repeat_counts.max()),
                ),
                args.outer_repeats,
            )
    if args.mode in ("both", "source-deletion"):
        split_audit = pd.read_csv(
            args.output_dir / SOURCE_FILES["split_audit"]
        )
        failed_strata = int(
            (split_audit["status"].astype(str) != "complete").sum()
        )
        add_check(
            checks,
            "source_failed_strata_explicit_and_absent",
            failed_strata == 0,
            failed_strata,
            0,
        )
        complete_split_audit = split_audit[
            split_audit["status"].astype(str) == "complete"
        ]
        inner_audit = pd.read_csv(
            args.output_dir / SOURCE_FILES["inner_audit"]
        )
        add_check(
            checks,
            "source_inner_scaffold_overlap_zero",
            bool((inner_audit["n_group_overlap"] == 0).all()),
            int(inner_audit["n_group_overlap"].max()),
            0,
        )
        leakage_columns = [
            "train_test_row_overlap",
            "train_test_smiles_overlap",
            "heldout_target_token_seen_in_train",
            "deleted_source_token_seen_in_train",
        ]
        leakage_sum = int(
            complete_split_audit[leakage_columns]
            .astype(int)
            .to_numpy(dtype=np.int64)
            .sum()
        )
        add_check(
            checks,
            "source_deletion_leakage_checks",
            leakage_sum == 0,
            leakage_sum,
            0,
        )
        add_check(
            checks,
            "source_predictions_finite",
            bool(
                np.isfinite(
                    source_predictions[["y_true", "y_pred"]].to_numpy(
                        dtype=np.float64
                    )
                ).all()
            ),
            int(
                np.isfinite(
                    source_predictions[["y_true", "y_pred"]].to_numpy(
                        dtype=np.float64
                    )
                ).sum()
            ),
            int(len(source_predictions) * 2),
        )
        duplicate_keys = int(
            source_predictions.duplicated(
                [
                    "heldout_target",
                    "deletion_source",
                    "row_index",
                    "model_id",
                ]
            ).sum()
        )
        add_check(
            checks,
            "source_prediction_keys_unique",
            duplicate_keys == 0,
            duplicate_keys,
            0,
        )
        aligned = True
        reference_keys: set[tuple[str, int]] | None = None
        for (_, _), frame in source_predictions.groupby(
            ["deletion_source", "model_id"],
            sort=False,
        ):
            keys = set(
                zip(
                    frame["heldout_target"].astype(str),
                    frame["row_index"].astype(int),
                )
            )
            if reference_keys is None:
                reference_keys = keys
            elif keys != reference_keys:
                aligned = False
        add_check(
            checks,
            "source_test_rows_aligned",
            aligned,
            aligned,
            True,
        )
        if formal_settings_match(args):
            observed_pairs = set(
                zip(
                    split_audit["heldout_target"].astype(str),
                    split_audit["deletion_source"].astype(str),
                )
            )
            expected_pairs = {
                (target, condition)
                for target in TARGETS
                for condition in (BASELINE_CONDITION, *SOURCE_DELETIONS)
            }
            add_check(
                checks,
                "source_formal_condition_completeness",
                observed_pairs == expected_pairs,
                len(observed_pairs),
                len(expected_pairs),
            )
            expected_rows = int(
                sum(
                    ood.EXPECTED_SPLIT_COUNTS[
                        ("target_ood", target)
                    ]["n_test_rows"]
                    for target in TARGETS
                )
                * (1 + len(SOURCE_DELETIONS))
                * len(MODELS)
            )
            add_check(
                checks,
                "source_formal_prediction_completeness",
                len(source_predictions) == expected_rows,
                len(source_predictions),
                expected_rows,
            )
    if args.bootstrap_replicates > 0:
        if args.mode in ("both", "generic-scaffold"):
            generic_valid = (
                not generic_bootstrap.empty
                and bool(
                    (
                        generic_bootstrap["n_bootstrap"]
                        == args.bootstrap_replicates
                    ).all()
                )
            )
            add_check(
                checks,
                "generic_bootstrap_replicates",
                generic_valid,
                (
                    0
                    if generic_bootstrap.empty
                    else int(generic_bootstrap["n_bootstrap"].min())
                ),
                args.bootstrap_replicates,
            )
        if args.mode in ("both", "source-deletion"):
            source_valid = (
                not source_bootstrap.empty
                and bool(
                    (
                        source_bootstrap["n_bootstrap"]
                        == args.bootstrap_replicates
                    ).all()
                )
            )
            add_check(
                checks,
                "source_bootstrap_replicates",
                source_valid,
                (
                    0
                    if source_bootstrap.empty
                    else int(source_bootstrap["n_bootstrap"].min())
                ),
                args.bootstrap_replicates,
            )
    passed = sum(row["status"] == "PASS" for row in checks)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "PASS" if passed == len(checks) else "FAIL",
        "formal_settings_match": formal_settings_match(args),
        "n_checks": len(checks),
        "n_pass": passed,
        "n_fail": len(checks) - passed,
        "checks": checks,
    }
    markdown_lines = [
        "# QA summary",
        "",
        f"- Status: **{payload['status']}**",
        f"- Formal settings match: `{payload['formal_settings_match']}`",
        f"- Checks: {passed}/{len(checks)} PASS",
        "",
        "| Check | Status | Observed | Expected |",
        "|---|---:|---:|---:|",
    ]
    for row in checks:
        markdown_lines.append(
            f"| {row['check_id']} | {row['status']} | "
            f"{row['observed']} | {row['expected']} |"
        )
    return payload, "\n".join(markdown_lines) + "\n"


def artifact_inventory(output_dir: Path) -> pd.DataFrame:
    excluded = {"artifact_sha256.csv", "run_manifest.json"}
    records = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name in excluded:
            continue
        records.append(
            {
                "file": path.name,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(records)


def format_estimate_interval(row: pd.Series) -> str:
    return (
        f"{float(row['observed']):.3f} "
        f"[{float(row['ci_low']):.3f}, {float(row['ci_high']):.3f}]"
    )


def write_results_brief(
    args: argparse.Namespace,
    generic_averaged_metrics: pd.DataFrame,
    generic_bootstrap: pd.DataFrame,
    source_macro: pd.DataFrame,
    source_bootstrap: pd.DataFrame,
) -> None:
    lines = [
        "# 替代骨架与训练来源删除敏感性分析简报",
        "",
        "- 证据层级：post-hoc sensitivity；不改变原 confirmatory 结论。",
        "- 数据范围：仅冻结的 1,560 条 strict-exact 核心记录。",
        "- 计算：CPU-only ExtraTrees；无 GPU。",
        "",
    ]
    if not generic_averaged_metrics.empty:
        lines.extend(
            [
                "## Generic Murcko scaffold",
                "",
                "5×5 完整外层运行的 repeat-averaged 结果："
                if formal_settings_match(args)
                else "Smoke 配置结果（不可作为正式效应量）：",
                "",
                "| 模型 | Spearman | RMSE |",
                "|---|---:|---:|",
            ]
        )
        for row in generic_averaged_metrics.itertuples(index=False):
            lines.append(
                f"| {row.model_id} | {float(row.spearman):.3f} | "
                f"{float(row.rmse):.3f} |"
            )
        if not generic_bootstrap.empty:
            for metric in ("delta_spearman", "delta_rmse"):
                match = generic_bootstrap[
                    (generic_bootstrap["record_type"] == "contrast")
                    & (generic_bootstrap["metric"] == metric)
                ]
                if not match.empty:
                    label = (
                        "full−chemistry Spearman"
                        if metric == "delta_spearman"
                        else "chemistry−full RMSE"
                    )
                    lines.append(
                        f"- {label}（正值支持 full）："
                        f"{format_estimate_interval(match.iloc[0])}。"
                    )
        lines.append("")
    if not source_macro.empty:
        lines.extend(
            [
                "## Target-OOD training-source deletion",
                "",
                "以下为固定目标集合上的 equal-domain macro；区间来自跨目标轴的"
                "全局 Bemis–Murcko scaffold 成簇 bootstrap。",
                "",
                "| 删除来源 | 模型/对比 | Macro Spearman | Macro RMSE |",
                "|---|---|---:|---:|",
            ]
        )
        for row in source_macro.itertuples(index=False):
            identity = (
                row.model_id
                if row.record_type == "model"
                else (
                    f"{row.contrast_id}"
                    + (f"::{row.model_id}" if str(row.model_id) else "")
                )
            )
            lines.append(
                f"| {row.deletion_source} | {identity} | "
                f"{float(row.domain_macro_spearman):.3f} | "
                f"{float(row.domain_macro_rmse):.3f} |"
            )
        if not source_bootstrap.empty:
            lines.extend(["", "正式 paired contrasts（estimate [95% CI]）：", ""])
            contrast_rows = source_bootstrap[
                source_bootstrap["record_type"].isin(
                    ["full_vs_chemistry", "deletion_vs_baseline"]
                )
            ]
            for row in contrast_rows.itertuples(index=False):
                identity = str(row.contrast_type)
                if str(row.model_id):
                    identity += f"::{row.model_id}"
                lines.append(
                    f"- {row.deletion_source} / {identity} / {row.metric}: "
                    f"{float(row.observed):.3f} "
                    f"[{float(row.ci_low):.3f}, {float(row.ci_high):.3f}]。"
                )
        lines.append("")
    lines.extend(
        [
            "## 解读边界",
            "",
            "这些结果只用于判断主结论是否依赖骨架分组或某一训练数据库。"
            "它们不是前瞻外部验证、机制验证或部署校准证据；区间为描述性"
            "敏感性区间，不作多重性校正后的显著性声明。",
            "",
        ]
    )
    (args.output_dir / "results_brief_zh.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    run_started = time.time()
    prepare_output_dir(args)
    if not args.data_file.is_file():
        raise FileNotFoundError(args.data_file)
    if not PROTOCOL_DOCUMENT.is_file():
        raise FileNotFoundError(PROTOCOL_DOCUMENT)
    if not CORRECTION_DOCUMENT.is_file():
        raise FileNotFoundError(CORRECTION_DOCUMENT)
    observed_correction_hash = sha256_file(CORRECTION_DOCUMENT)
    if observed_correction_hash != EXPECTED_CORRECTION_DOCUMENT_SHA256:
        raise RuntimeError(
            "Frozen float32 correction document checksum mismatch: "
            f"{observed_correction_hash}"
        )
    if not MASTER_PROTOCOL_DOCUMENT.is_file():
        raise FileNotFoundError(MASTER_PROTOCOL_DOCUMENT)
    observed_master_protocol_hash = sha256_file(MASTER_PROTOCOL_DOCUMENT)
    if observed_master_protocol_hash != EXPECTED_MASTER_PROTOCOL_SHA256:
        raise RuntimeError(
            "Frozen master protocol checksum mismatch: "
            f"{observed_master_protocol_hash}"
        )
    observed_data_hash = sha256_file(args.data_file)
    if observed_data_hash != EXPECTED_DATA_SHA256:
        raise RuntimeError(
            f"Frozen core data checksum mismatch: {observed_data_hash}"
        )
    rows, context_audit = core.load_rows(args.data_file)
    if len(rows) != EXPECTED_N_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_N_ROWS} sanitized rows, observed {len(rows)}"
        )
    y = rows["pDC50"].to_numpy(dtype=np.float32)
    if y.dtype != RESPONSE_DTYPE:
        raise RuntimeError(
            f"Training response dtype mismatch: {y.dtype}; expected float32"
        )
    bemis_ids = core.build_scaffold_ids(rows["canonical_smiles"])
    generic_ids = build_generic_scaffold_ids(rows["canonical_smiles"])
    group_summary = generic_group_summary(rows, generic_ids, bemis_ids)
    group_record = group_summary.iloc[0]
    expected_group_values = (
        EXPECTED_GENERIC_GROUPS,
        EXPECTED_GENERIC_SINGLETONS,
        EXPECTED_GENERIC_MAX_ROWS,
    )
    observed_group_values = (
        int(group_record["n_generic_murcko_groups"]),
        int(group_record["n_singleton_generic_groups"]),
        int(group_record["largest_generic_group_rows"]),
    )
    if observed_group_values != expected_group_values:
        raise RuntimeError(
            "Generic scaffold identity changed: "
            f"observed={observed_group_values}, expected={expected_group_values}"
        )
    scientific_configuration = {
        "protocol_version": PROTOCOL_VERSION,
        "evidence_label": "post_hoc_sensitivity",
        "response_dtype": RESPONSE_DTYPE.name,
        "scientific_correction": (
            "training response aligned to frozen confirmatory float32 "
            "implementation; v1 float64 refits are superseded"
        ),
        "formal_settings_match": formal_settings_match(args),
        "args": vars(args),
        "models": list(MODELS),
        "model_grid": {
            model: core.model_param_grid(model) for model in MODELS
        },
        "source_deletion_tokens": list(SOURCE_DELETIONS),
        "target_ood_groups": list(TARGETS),
        "source_deletion_rule": (
            "remove every training row whose plus-tokenized source provenance "
            "contains the deletion token"
        ),
        "generic_scaffold_definition": (
            "GetScaffoldForMol then MakeScaffoldGeneric; compound-specific "
            "acyclic fallback"
        ),
        "data_sha256": observed_data_hash,
        "master_protocol_sha256": observed_master_protocol_hash,
        "protocol_sha256": sha256_file(PROTOCOL_DOCUMENT),
        "float32_correction_document_sha256": observed_correction_hash,
        "core_script_sha256": sha256_file(CORE_SCRIPT),
        "ood_script_sha256": sha256_file(OOD_SCRIPT),
        "this_script_sha256": sha256_file(Path(__file__).resolve()),
        "environment": environment_manifest(args),
        "boundaries": [
            "Uses only the frozen 1,560-row core analysis table.",
            "Post-hoc sensitivity, not confirmatory or prospective evidence.",
            "No target encoding or protein language-model feature.",
            "No data source other than the declared frozen core table is loaded.",
            "The v1 scaffold/source refits are retained but superseded.",
        ],
    }
    write_json(
        args.output_dir / "scientific_configuration.json",
        scientific_configuration,
    )
    manifest = {
        **scientific_configuration,
        "status": "running",
        "started_unix": run_started,
    }
    write_json(args.output_dir / "run_manifest.json", manifest)

    print("[FEATURE] building Morgan fingerprints", flush=True)
    morgan = core.build_morgan_matrix(rows["canonical_smiles"])
    print("[FEATURE] building RDKit descriptors", flush=True)
    descriptors, descriptor_names, descriptor_failures = (
        core.build_descriptor_matrix(rows["canonical_smiles"])
    )
    scientific_configuration["feature_manifest"] = core.feature_manifest(
        descriptor_names,
        descriptor_failures,
    )
    write_json(
        args.output_dir / "scientific_configuration.json",
        scientific_configuration,
    )
    context_audit.to_csv(
        args.output_dir / "context_sanitization_audit.csv",
        index=False,
    )
    group_summary.to_csv(
        args.output_dir / "generic_scaffold_group_summary.csv",
        index=False,
    )
    analysis_index = rows[
        [
            "qc_id",
            "canonical_smiles",
            "source_database",
            "target_protein",
            "pDC50",
        ]
    ].copy()
    analysis_index["bemis_murcko_scaffold_id"] = bemis_ids
    analysis_index["generic_scaffold_id"] = generic_ids
    analysis_index.to_csv(args.output_dir / "analysis_index.csv", index=False)

    generic_predictions = pd.DataFrame()
    generic_bootstrap = pd.DataFrame()
    generic_averaged_metrics = pd.DataFrame()
    if args.mode in ("both", "generic-scaffold"):
        generic_predictions = run_generic_scaffold(
            args,
            rows,
            y,
            morgan,
            descriptors,
            generic_ids,
            bemis_ids,
        )
        repeat_metrics, averaged, generic_averaged_metrics = generic_summaries(
            generic_predictions
        )
        atomic_csv(
            repeat_metrics,
            args.output_dir / "generic_scaffold_metrics_by_repeat.csv",
        )
        atomic_csv(
            averaged,
            args.output_dir
            / "generic_scaffold_repeat_averaged_predictions.csv",
        )
        atomic_csv(
            generic_averaged_metrics,
            args.output_dir
            / "generic_scaffold_repeat_averaged_metrics.csv",
        )
        generic_bootstrap = generic_cluster_bootstrap(
            averaged,
            args.bootstrap_replicates,
            args.seed,
        )
        if not generic_bootstrap.empty:
            atomic_csv(
                generic_bootstrap,
                args.output_dir
                / "generic_scaffold_paired_cluster_bootstrap.csv",
            )

    source_predictions = pd.DataFrame()
    source_bootstrap = pd.DataFrame()
    source_macro = pd.DataFrame()
    if args.mode in ("both", "source-deletion"):
        source_predictions = run_source_deletion(
            args,
            rows,
            y,
            morgan,
            descriptors,
            bemis_ids,
        )
        domain_metrics = source_domain_metrics(source_predictions)
        atomic_csv(
            domain_metrics,
            args.output_dir / SOURCE_FILES["domain_metrics"],
        )
        source_macro, deltas = source_macro_and_deltas(domain_metrics)
        atomic_csv(
            source_macro,
            args.output_dir / "source_deletion_equal_domain_macro.csv",
        )
        atomic_csv(
            deltas,
            args.output_dir / "source_deletion_domain_paired_deltas.csv",
        )
        source_bootstrap = source_stratified_cluster_bootstrap(
            source_predictions,
            args.bootstrap_replicates,
            args.seed,
        )
        if not source_bootstrap.empty:
            atomic_csv(
                source_bootstrap,
                args.output_dir
                / "source_deletion_paired_cluster_bootstrap.csv",
            )

    write_results_brief(
        args,
        generic_averaged_metrics,
        generic_bootstrap,
        source_macro,
        source_bootstrap,
    )

    qa_payload, qa_markdown = qa_results(
        args,
        rows,
        y,
        generic_ids,
        generic_predictions,
        source_predictions,
        generic_bootstrap,
        source_bootstrap,
    )
    write_json(args.output_dir / "qa_summary.json", qa_payload)
    (args.output_dir / "qa_summary.md").write_text(
        qa_markdown,
        encoding="utf-8",
    )
    inventory = artifact_inventory(args.output_dir)
    atomic_csv(inventory, args.output_dir / "artifact_sha256.csv")
    manifest.update(
        {
            "status": (
                "complete"
                if qa_payload["status"] == "PASS"
                and formal_settings_match(args)
                else (
                    "smoke_complete"
                    if qa_payload["status"] == "PASS"
                    else "failed_qa"
                )
            ),
            "qa_status": qa_payload["status"],
            "formal_settings_match": formal_settings_match(args),
            "elapsed_seconds": float(time.time() - run_started),
            "artifact_count": int(len(inventory)),
            "artifact_inventory": "artifact_sha256.csv",
        }
    )
    write_json(args.output_dir / "run_manifest.json", manifest)
    if qa_payload["status"] != "PASS":
        raise RuntimeError("Post-hoc sensitivity QA failed")
    print(
        (
            f"[DONE] status={manifest['status']} "
            f"elapsed={manifest['elapsed_seconds']:.1f}s "
            f"output={args.output_dir}"
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
