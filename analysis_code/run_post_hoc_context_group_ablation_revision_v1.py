#!/usr/bin/env python
"""Reviewer-driven grouped-context ExtraTrees sensitivity (CPU only).

The analysis reuses the frozen strict-exact core, scaffold splits and OOD
deletion rules. It adds provenance, biological/assay and missingness context
blocks to chemistry, with held-axis-portable definitions where specified in
the reviewer revision protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_confirmatory_cpu_v1 as core  # noqa: E402
import run_confirmatory_ood_cpu_v1 as ood  # noqa: E402
import run_post_hoc_context_weight_sensitivity_v2 as sensitivity  # noqa: E402


PROTOCOL_VERSION = "post_hoc_context_group_ablation_revision_v1.0"
PROTOCOL_FILE = (
    PROJECT_DIR / "docs" / "post_hoc_reviewer_revision_v1_protocol.md"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "reports" / "post_hoc_context_group_ablation_revision_v1"
)
INTERNAL_BASELINE = (
    PROJECT_DIR
    / "reports"
    / "confirmatory_cpu_v1"
    / "scaffold_cross_fitted_predictions.csv"
)
OOD_BASELINE = (
    PROJECT_DIR / "reports" / "confirmatory_ood_cpu_v1" / "ood_predictions.csv"
)
TREE_GRID = tuple(dict(x) for x in core.model_param_grid("full_context_extra_trees"))
BASELINE_MODELS = ("chemistry_extra_trees", "full_context_extra_trees")
NEW_MODELS = (
    "chemistry_plus_provenance",
    "chemistry_plus_biological_assay",
    "chemistry_plus_missingness",
)
ALL_MODELS = BASELINE_MODELS + NEW_MODELS
MODEL_POSITION = {
    "chemistry_plus_provenance": 31,
    "chemistry_plus_biological_assay": 32,
    "chemistry_plus_missingness": 33,
}
MISSING_COLUMNS = (
    "recruiting_protein",
    "target_protein",
    "cell_line",
    "assay_method",
    "activity_time",
    "mode_of_action",
)
KNOWN_OUTPUTS = {
    "configuration.json",
    "context_sanitization_audit.csv",
    "feature_manifest.json",
    "internal_predictions.csv",
    "internal_selected_hyperparameters.csv",
    "internal_tuning_metrics.csv",
    "internal_split_audit.csv",
    "internal_repeat_averaged_predictions.csv",
    "internal_summary_metrics.csv",
    "ood_predictions.csv",
    "ood_selected_hyperparameters.csv",
    "ood_tuning_metrics.csv",
    "ood_split_audit.csv",
    "ood_aggregate_metrics.csv",
    "ood_domain_metrics.csv",
    "qa_summary.json",
    "qa_summary.md",
    "run_manifest.json",
    "component_execution_provenance.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=core.DEFAULT_SEED)
    parser.add_argument("--max-internal-splits", type=int, default=None)
    parser.add_argument("--max-ood-groups", type=int, default=None)
    parser.add_argument("--skip-internal", action="store_true")
    parser.add_argument("--skip-ood", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.n_estimators < 10:
        parser.error("n-estimators must be >= 10")
    if not 1 <= args.n_jobs <= 8:
        parser.error("n-jobs must be between 1 and 8")
    if args.inner_folds < 2:
        parser.error("inner-folds must be >= 2")
    if args.resume and args.overwrite:
        parser.error("resume and overwrite are mutually exclusive")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def missing_flag(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    text = str(value).strip().lower()
    return (
        "missing"
        if text in {"", "nan", "none", "na", "n/a", "unspecified", "unknown"}
        else "observed"
    )


def context_definition(
    model_id: str,
    protocol: str | None,
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    if model_id == "chemistry_plus_provenance":
        if protocol == "source_ood":
            return (), {}
        if protocol == "target_ood":
            return ("source_database",), {}
        return (
            ("source_database",),
            {"source_target": ("source_database", "target_protein")},
        )
    if model_id == "chemistry_plus_biological_assay":
        if protocol == "target_ood":
            return (
                (
                    "recruiting_protein",
                    "cell_line",
                    "assay_method",
                    "activity_time",
                    "mode_of_action",
                ),
                {},
            )
        return (
            (
                "recruiting_protein",
                "target_protein",
                "cell_line",
                "assay_method",
                "activity_time",
                "mode_of_action",
            ),
            {
                "recruiter_target": ("recruiting_protein", "target_protein"),
                "target_cell": ("target_protein", "cell_line"),
                "assay_target": ("assay_method", "target_protein"),
            },
        )
    if model_id == "chemistry_plus_missingness":
        return tuple(f"missing__{name}" for name in MISSING_COLUMNS), {}
    raise ValueError(f"Unknown grouped-context model: {model_id}")


def context_frame(
    rows: pd.DataFrame,
    raw_rows: pd.DataFrame,
    model_id: str,
    protocol: str | None,
) -> pd.DataFrame:
    main, interactions = context_definition(model_id, protocol)
    if model_id == "chemistry_plus_missingness":
        frame = pd.DataFrame(index=rows.index)
        for name in MISSING_COLUMNS:
            frame[f"missing__{name}"] = raw_rows[name].map(missing_flag)
        return frame.astype(str)
    frame = rows[list(main)].copy()
    for name, columns in interactions.items():
        frame[name] = rows[list(columns)].astype(str).agg("||".join, axis=1)
    return frame.fillna("NA").astype(str)


def build_features(
    rows: pd.DataFrame,
    raw_rows: pd.DataFrame,
    morgan: np.ndarray,
    descriptors: np.ndarray,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
    model_id: str,
    protocol: str | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rdkit_fit, rdkit_eval, metadata = core.prepare_rdkit_fold(
        descriptors, fit_idx, eval_idx
    )
    chemistry_fit = np.hstack(
        [morgan[fit_idx].astype(np.float32), rdkit_fit]
    ).astype(np.float32)
    chemistry_eval = np.hstack(
        [morgan[eval_idx].astype(np.float32), rdkit_eval]
    ).astype(np.float32)
    contexts = context_frame(rows, raw_rows, model_id, protocol)
    if contexts.shape[1] == 0:
        raise RuntimeError(
            f"Context block {model_id} is not portable for {protocol}"
        )
    encoder = core.make_one_hot_encoder()
    context_fit = encoder.fit_transform(contexts.iloc[fit_idx]).astype(np.float32)
    context_eval = encoder.transform(contexts.iloc[eval_idx]).astype(np.float32)
    return (
        np.hstack([chemistry_fit, context_fit]).astype(np.float32),
        np.hstack([chemistry_eval, context_eval]).astype(np.float32),
        {
            **metadata,
            "chemistry_dim": int(chemistry_fit.shape[1]),
            "context_dim": int(context_fit.shape[1]),
            "full_dim": int(chemistry_fit.shape[1] + context_fit.shape[1]),
            "context_fields": "|".join(contexts.columns.astype(str)),
        },
    )


def fit_model(
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
        random_state=int(seed),
        n_jobs=int(n_jobs),
    )
    model.fit(x_fit, y_fit.astype(np.float32))
    return np.asarray(model.predict(x_eval), dtype=np.float32)


def select_params(
    rows: pd.DataFrame,
    raw_rows: pd.DataFrame,
    y: np.ndarray,
    morgan: np.ndarray,
    descriptors: np.ndarray,
    scaffold_ids: np.ndarray,
    train_idx: np.ndarray,
    model_id: str,
    protocol: str | None,
    inner_folds: int,
    seed: int,
    n_estimators: int,
    n_jobs: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    splits = core.balanced_group_splits(
        train_idx, scaffold_ids, inner_folds, seed
    )
    local_position = {
        int(value): position for position, value in enumerate(train_idx.tolist())
    }
    candidates = {
        str(params["param_id"]): np.full(len(train_idx), np.nan, dtype=np.float32)
        for params in TREE_GRID
    }
    audit: list[dict[str, Any]] = []
    for inner_fold, (fit_idx, eval_idx) in enumerate(splits, start=1):
        if set(scaffold_ids[fit_idx]).intersection(scaffold_ids[eval_idx]):
            raise RuntimeError("Inner scaffold leakage")
        x_fit, x_eval, metadata = build_features(
            rows,
            raw_rows,
            morgan,
            descriptors,
            fit_idx,
            eval_idx,
            model_id,
            protocol,
        )
        positions = np.asarray(
            [local_position[int(value)] for value in eval_idx], dtype=np.int64
        )
        for params in TREE_GRID:
            pred = fit_model(
                x_fit,
                y[fit_idx],
                x_eval,
                params,
                seed + inner_fold * 101 + MODEL_POSITION[model_id],
                n_estimators,
                n_jobs,
            )
            candidates[str(params["param_id"])][positions] = pred
        audit.append(
            {
                "inner_fold": inner_fold,
                "n_fit_rows": len(fit_idx),
                "n_eval_rows": len(eval_idx),
                "n_scaffold_overlap": 0,
                **metadata,
            }
        )
    if not all(np.all(np.isfinite(value)) for value in candidates.values()):
        raise RuntimeError("Incomplete inner predictions")
    selected, tuning = sensitivity.select_tree_params(candidates, y[train_idx])
    return selected, tuning, audit


def read_csv_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        return pd.read_csv(path).to_dict("records")
    except pd.errors.EmptyDataError:
        return []


def write_records(records: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    filenames = {
        "internal_predictions": "internal_predictions.csv",
        "internal_selected": "internal_selected_hyperparameters.csv",
        "internal_tuning": "internal_tuning_metrics.csv",
        "internal_audit": "internal_split_audit.csv",
        "ood_predictions": "ood_predictions.csv",
        "ood_selected": "ood_selected_hyperparameters.csv",
        "ood_tuning": "ood_tuning_metrics.csv",
        "ood_audit": "ood_split_audit.csv",
    }
    for key, filename in filenames.items():
        pd.DataFrame(records[key]).to_csv(output_dir / filename, index=False)


def internal_complete(
    records: dict[str, list[dict[str, Any]]], repeat: int, fold: int, n_test: int
) -> bool:
    frame = pd.DataFrame(records["internal_predictions"])
    if frame.empty:
        return False
    local = frame[(frame["repeat"] == repeat) & (frame["outer_fold"] == fold)]
    return (
        set(local["model_id"].astype(str)) == set(ALL_MODELS)
        and all(len(group) == n_test for _, group in local.groupby("model_id"))
    )


def run_internal(
    args: argparse.Namespace,
    records: dict[str, list[dict[str, Any]]],
    rows: pd.DataFrame,
    raw_rows: pd.DataFrame,
    y: np.ndarray,
    morgan: np.ndarray,
    descriptors: np.ndarray,
    scaffold_ids: np.ndarray,
    output_dir: Path,
) -> None:
    baseline = pd.read_csv(INTERNAL_BASELINE)
    splits = core.make_outer_splits(
        "scaffold", rows, scaffold_ids, 5, 5, args.seed
    )
    if args.max_internal_splits is not None:
        splits = splits[: args.max_internal_splits]
    started = time.perf_counter()
    for position, split in enumerate(splits, start=1):
        if args.resume and internal_complete(
            records, split.repeat, split.fold, len(split.test_idx)
        ):
            continue
        train_scaffolds = set(scaffold_ids[split.train_idx])
        test_scaffolds = set(scaffold_ids[split.test_idx])
        if train_scaffolds.intersection(test_scaffolds):
            raise RuntimeError("Outer scaffold leakage")
        base = baseline[
            (baseline["repeat"] == split.repeat)
            & (baseline["outer_fold"] == split.fold)
            & baseline["model_id"].isin(BASELINE_MODELS)
        ].copy()
        for record in base.to_dict("records"):
            records["internal_predictions"].append(
                {**record, "context_block": "baseline"}
            )
        records["internal_audit"].append(
            {
                "repeat": split.repeat,
                "outer_fold": split.fold,
                "n_train_rows": len(split.train_idx),
                "n_test_rows": len(split.test_idx),
                "n_train_scaffolds": len(train_scaffolds),
                "n_test_scaffolds": len(test_scaffolds),
                "n_scaffold_overlap": 0,
            }
        )
        identity = (
            base[base["model_id"] == "chemistry_extra_trees"]
            .set_index("row_index")
            .loc[split.test_idx]
            .reset_index()
        )
        inner_seed = args.seed + split.repeat * 10_000 + split.fold * 100
        for model_id in NEW_MODELS:
            selected, tuning, feature_audit = select_params(
                rows,
                raw_rows,
                y,
                morgan,
                descriptors,
                scaffold_ids,
                split.train_idx,
                model_id,
                None,
                args.inner_folds,
                inner_seed,
                args.n_estimators,
                args.n_jobs,
            )
            records["internal_selected"].append(
                {
                    "repeat": split.repeat,
                    "outer_fold": split.fold,
                    "model_id": model_id,
                    **selected,
                }
            )
            for row in tuning:
                records["internal_tuning"].append(
                    {
                        "repeat": split.repeat,
                        "outer_fold": split.fold,
                        "model_id": model_id,
                        **row,
                    }
                )
            x_fit, x_eval, metadata = build_features(
                rows,
                raw_rows,
                morgan,
                descriptors,
                split.train_idx,
                split.test_idx,
                model_id,
                None,
            )
            pred = fit_model(
                x_fit,
                y[split.train_idx],
                x_eval,
                selected,
                inner_seed + 1_000 + MODEL_POSITION[model_id],
                args.n_estimators,
                args.n_jobs,
            )
            for local_index, source in identity.iterrows():
                records["internal_predictions"].append(
                    {
                        **source.to_dict(),
                        "model_id": model_id,
                        "param_id": str(selected["param_id"]),
                        "y_pred": float(pred[local_index]),
                        "context_block": model_id.replace("chemistry_plus_", ""),
                    }
                )
            records["internal_audit"].append(
                {
                    "repeat": split.repeat,
                    "outer_fold": split.fold,
                    "model_id": model_id,
                    "fit_stage": "outer_final_features",
                    "n_inner_feature_audits": len(feature_audit),
                    **metadata,
                }
            )
        write_records(records, output_dir)
        if not internal_complete(
            records, split.repeat, split.fold, len(split.test_idx)
        ):
            raise RuntimeError("Internal checkpoint incomplete")
        print(
            f"[INTERNAL] {position}/{len(splits)} elapsed="
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )


def ood_expected_models(protocol: str) -> tuple[str, ...]:
    if protocol == "source_ood":
        return (
            *BASELINE_MODELS,
            "chemistry_plus_biological_assay",
            "chemistry_plus_missingness",
        )
    return ALL_MODELS


def ood_complete(
    records: dict[str, list[dict[str, Any]]],
    protocol: str,
    group: str,
    n_test: int,
) -> bool:
    frame = pd.DataFrame(records["ood_predictions"])
    if frame.empty:
        return False
    local = frame[
        (frame["protocol"] == protocol) & (frame["heldout_group"] == group)
    ]
    return (
        set(local["model_id"].astype(str)) == set(ood_expected_models(protocol))
        and all(len(x) == n_test for _, x in local.groupby("model_id"))
    )


def run_ood(
    args: argparse.Namespace,
    records: dict[str, list[dict[str, Any]]],
    rows: pd.DataFrame,
    raw_rows: pd.DataFrame,
    y: np.ndarray,
    morgan: np.ndarray,
    descriptors: np.ndarray,
    scaffold_ids: np.ndarray,
    output_dir: Path,
) -> None:
    baseline = pd.read_csv(OOD_BASELINE)
    splits, _ = ood.build_ood_splits(
        rows,
        scaffold_ids,
        list(ood.PROTOCOL_ORDER),
        args.seed,
        args.max_ood_groups,
    )
    started = time.perf_counter()
    for position, split in enumerate(splits, start=1):
        if args.resume and ood_complete(
            records,
            split.protocol,
            split.heldout_group,
            len(split.test_idx),
        ):
            continue
        base = baseline[
            (baseline["protocol"] == split.protocol)
            & (baseline["heldout_group"] == split.heldout_group)
            & baseline["model_id"].isin(BASELINE_MODELS)
        ].copy()
        for record in base.to_dict("records"):
            records["ood_predictions"].append(
                {**record, "context_block": "baseline"}
            )
        identity = (
            base[base["model_id"] == "chemistry_extra_trees"]
            .set_index("row_index")
            .loc[split.test_idx]
            .reset_index()
        )
        train_smiles = set(rows.iloc[split.train_idx]["canonical_smiles"].astype(str))
        test_smiles = set(rows.iloc[split.test_idx]["canonical_smiles"].astype(str))
        if train_smiles.intersection(test_smiles):
            raise RuntimeError("OOD test compound entered training")
        records["ood_audit"].append(
            {
                "protocol": split.protocol,
                "heldout_group": split.heldout_group,
                "n_train_rows": len(split.train_idx),
                "n_test_rows": len(split.test_idx),
                "n_compound_overlap": 0,
            }
        )
        models = [
            model_id
            for model_id in NEW_MODELS
            if not (
                split.protocol == "source_ood"
                and model_id == "chemistry_plus_provenance"
            )
        ]
        inner_seed = split.split_seed + 101
        for model_id in models:
            selected, tuning, feature_audit = select_params(
                rows,
                raw_rows,
                y,
                morgan,
                descriptors,
                scaffold_ids,
                split.train_idx,
                model_id,
                split.protocol,
                args.inner_folds,
                inner_seed,
                args.n_estimators,
                args.n_jobs,
            )
            records["ood_selected"].append(
                {
                    "protocol": split.protocol,
                    "heldout_group": split.heldout_group,
                    "model_id": model_id,
                    **selected,
                }
            )
            for row in tuning:
                records["ood_tuning"].append(
                    {
                        "protocol": split.protocol,
                        "heldout_group": split.heldout_group,
                        "model_id": model_id,
                        **row,
                    }
                )
            x_fit, x_eval, metadata = build_features(
                rows,
                raw_rows,
                morgan,
                descriptors,
                split.train_idx,
                split.test_idx,
                model_id,
                split.protocol,
            )
            pred = fit_model(
                x_fit,
                y[split.train_idx],
                x_eval,
                selected,
                inner_seed + 1_000 + MODEL_POSITION[model_id],
                args.n_estimators,
                args.n_jobs,
            )
            for local_index, source in identity.iterrows():
                records["ood_predictions"].append(
                    {
                        **source.to_dict(),
                        "model_id": model_id,
                        "param_id": str(selected["param_id"]),
                        "y_pred": float(pred[local_index]),
                        "context_block": model_id.replace("chemistry_plus_", ""),
                    }
                )
            main, interactions = context_definition(model_id, split.protocol)
            records["ood_audit"].append(
                {
                    "protocol": split.protocol,
                    "heldout_group": split.heldout_group,
                    "model_id": model_id,
                    "fit_stage": "final_features",
                    "retained_context_fields": "|".join(main),
                    "retained_interactions": "|".join(interactions),
                    "n_inner_feature_audits": len(feature_audit),
                    **metadata,
                }
            )
        write_records(records, output_dir)
        if not ood_complete(
            records,
            split.protocol,
            split.heldout_group,
            len(split.test_idx),
        ):
            raise RuntimeError("OOD checkpoint incomplete")
        print(
            f"[OOD] {position}/{len(splits)} {split.protocol} "
            f"{split.heldout_group} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )


def summarize(output_dir: Path, records: dict[str, list[dict[str, Any]]]) -> None:
    internal = pd.DataFrame(records["internal_predictions"])
    averaged_rows: list[dict[str, Any]] = []
    if not internal.empty:
        identity = [
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
            "context_block",
        ]
        for keys, frame in internal.groupby(identity, dropna=False, sort=False):
            row = dict(zip(identity, keys))
            row["n_repeats"] = int(frame["repeat"].nunique())
            row["y_pred"] = float(frame["y_pred"].mean())
            averaged_rows.append(row)
    averaged = pd.DataFrame(averaged_rows)
    averaged.to_csv(
        output_dir / "internal_repeat_averaged_predictions.csv", index=False
    )
    internal_summary: list[dict[str, Any]] = []
    if not averaged.empty:
        for model_id, frame in averaged.groupby("model_id", sort=False):
            internal_summary.append(
                {
                    "model_id": model_id,
                    "n_rows": len(frame),
                    "n_compounds": frame["canonical_smiles"].nunique(),
                    "n_scaffolds": frame["scaffold_id"].nunique(),
                    **core.regression_metrics(
                        frame["y_true"].to_numpy(dtype=float),
                        frame["y_pred"].to_numpy(dtype=float),
                    ),
                    **core.target_macro_metrics(frame),
                    "within_target_spearman": core.within_target_spearman(frame),
                }
            )
    pd.DataFrame(internal_summary).to_csv(
        output_dir / "internal_summary_metrics.csv", index=False
    )

    ood_predictions = pd.DataFrame(records["ood_predictions"])
    domain_rows: list[dict[str, Any]] = []
    if not ood_predictions.empty:
        for keys, frame in ood_predictions.groupby(
            ["protocol", "heldout_group", "model_id"], sort=False
        ):
            domain_rows.append(
                {
                    "protocol": keys[0],
                    "heldout_group": keys[1],
                    "model_id": keys[2],
                    "n_rows": len(frame),
                    "n_compounds": frame["canonical_smiles"].nunique(),
                    "n_scaffolds": frame["scaffold_id"].nunique(),
                    **core.regression_metrics(
                        frame["y_true"].to_numpy(dtype=float),
                        frame["y_pred"].to_numpy(dtype=float),
                    ),
                }
            )
    pd.DataFrame(domain_rows).to_csv(
        output_dir / "ood_domain_metrics.csv", index=False
    )
    aggregate_rows: list[dict[str, Any]] = []
    if not ood_predictions.empty:
        for (protocol, model_id), frame in ood_predictions.groupby(
            ["protocol", "model_id"], sort=False
        ):
            fixed = ood.GROUPS_BY_PROTOCOL[protocol]
            values = []
            errors = []
            for group in fixed:
                local = frame[frame["heldout_group"] == group]
                if local.empty:
                    continue
                metrics = core.regression_metrics(
                    local["y_true"].to_numpy(dtype=float),
                    local["y_pred"].to_numpy(dtype=float),
                )
                values.append(metrics["spearman"])
                errors.append(metrics["rmse"])
            aggregate_rows.append(
                {
                    "protocol": protocol,
                    "model_id": model_id,
                    "n_domains": len(values),
                    "domain_macro_spearman": (
                        float(np.mean(values)) if np.all(np.isfinite(values)) else np.nan
                    ),
                    "domain_macro_rmse": (
                        float(np.mean(errors)) if np.all(np.isfinite(errors)) else np.nan
                    ),
                    **{
                        f"pooled_{key}": value
                        for key, value in core.regression_metrics(
                            frame["y_true"].to_numpy(dtype=float),
                            frame["y_pred"].to_numpy(dtype=float),
                        ).items()
                    },
                }
            )
    pd.DataFrame(aggregate_rows).to_csv(
        output_dir / "ood_aggregate_metrics.csv", index=False
    )


def build_qa(
    args: argparse.Namespace,
    output_dir: Path,
    records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    internal = pd.DataFrame(records["internal_predictions"])
    if not internal.empty:
        duplicate = internal.duplicated(
            ["repeat", "outer_fold", "model_id", "row_index"]
        ).any()
        checks.append(
            {
                "check": "internal_prediction_keys_unique",
                "status": "PASS" if not duplicate else "FAIL",
                "detail": f"n_rows={len(internal)}",
            }
        )
    ood_predictions = pd.DataFrame(records["ood_predictions"])
    if not ood_predictions.empty:
        duplicate = ood_predictions.duplicated(
            ["protocol", "heldout_group", "model_id", "row_index"]
        ).any()
        checks.append(
            {
                "check": "ood_prediction_keys_unique",
                "status": "PASS" if not duplicate else "FAIL",
                "detail": f"n_rows={len(ood_predictions)}",
            }
        )
    finite = True
    for key in ("internal_predictions", "ood_predictions"):
        frame = pd.DataFrame(records[key])
        if not frame.empty:
            finite &= bool(np.isfinite(pd.to_numeric(frame["y_pred"])).all())
    checks.append(
        {
            "check": "predictions_finite",
            "status": "PASS" if finite else "FAIL",
            "detail": "all saved predictions checked",
        }
    )
    status = "PASS" if all(x["status"] == "PASS" for x in checks) else "FAIL"
    qa = {"status": status, "n_checks": len(checks), "checks": checks}
    (output_dir / "qa_summary.json").write_text(
        json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = ["# QA summary", "", f"Overall status: **{status}**", ""]
    lines.extend(
        f"- {x['status']}: {x['check']} — {x['detail']}" for x in checks
    )
    (output_dir / "qa_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return qa


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        unexpected = [x.name for x in output_dir.iterdir() if x.name not in KNOWN_OUTPUTS]
        if unexpected:
            raise RuntimeError(f"Refusing overwrite; unexpected files: {unexpected}")
        for path in output_dir.iterdir():
            if path.is_file():
                path.unlink()
    elif any(output_dir.iterdir()) and not args.resume:
        raise RuntimeError("Output directory is not empty; use --resume or --overwrite")

    raw_rows = pd.read_csv(core.DATA_FILE).reset_index(drop=True)
    rows, audit = core.load_rows(core.DATA_FILE)
    y = rows["pDC50"].to_numpy(dtype=np.float32)
    scaffold_ids = core.build_scaffold_ids(rows["canonical_smiles"])
    morgan = core.build_morgan_matrix(rows["canonical_smiles"])
    descriptors, descriptor_names, descriptor_failures = core.build_descriptor_matrix(
        rows["canonical_smiles"]
    )
    audit.to_csv(output_dir / "context_sanitization_audit.csv", index=False)
    manifest = {
        "context_blocks": {
            model: {
                "internal": context_definition(model, None),
                "source_ood": (
                    "not_portable"
                    if model == "chemistry_plus_provenance"
                    else context_definition(model, "source_ood")
                ),
                "target_ood": context_definition(model, "target_ood"),
            }
            for model in NEW_MODELS
        },
        "descriptor_names": descriptor_names,
        "descriptor_failure_counts": descriptor_failures,
    }
    (output_dir / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=list) + "\n", encoding="utf-8"
    )
    configuration = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": sha256_file(PROTOCOL_FILE),
        "data_sha256": sha256_file(core.DATA_FILE),
        "internal_baseline_sha256": sha256_file(INTERNAL_BASELINE),
        "ood_baseline_sha256": sha256_file(OOD_BASELINE),
        "n_estimators": args.n_estimators,
        "n_jobs": args.n_jobs,
        "inner_folds": args.inner_folds,
        "seed": args.seed,
        "max_internal_splits": args.max_internal_splits,
        "max_ood_groups": args.max_ood_groups,
        "skip_internal": args.skip_internal,
        "skip_ood": args.skip_ood,
        "tree_grid": TREE_GRID,
    }
    (output_dir / "configuration.json").write_text(
        json.dumps(configuration, indent=2, default=list) + "\n",
        encoding="utf-8",
    )
    records = {
        "internal_predictions": read_csv_records(
            output_dir / "internal_predictions.csv"
        ) if args.resume else [],
        "internal_selected": read_csv_records(
            output_dir / "internal_selected_hyperparameters.csv"
        ) if args.resume else [],
        "internal_tuning": read_csv_records(
            output_dir / "internal_tuning_metrics.csv"
        ) if args.resume else [],
        "internal_audit": read_csv_records(
            output_dir / "internal_split_audit.csv"
        ) if args.resume else [],
        "ood_predictions": read_csv_records(
            output_dir / "ood_predictions.csv"
        ) if args.resume else [],
        "ood_selected": read_csv_records(
            output_dir / "ood_selected_hyperparameters.csv"
        ) if args.resume else [],
        "ood_tuning": read_csv_records(
            output_dir / "ood_tuning_metrics.csv"
        ) if args.resume else [],
        "ood_audit": read_csv_records(
            output_dir / "ood_split_audit.csv"
        ) if args.resume else [],
    }
    if not args.skip_internal:
        run_internal(
            args,
            records,
            rows,
            raw_rows,
            y,
            morgan,
            descriptors,
            scaffold_ids,
            output_dir,
        )
    if not args.skip_ood:
        run_ood(
            args,
            records,
            rows,
            raw_rows,
            y,
            morgan,
            descriptors,
            scaffold_ids,
            output_dir,
        )
    summarize(output_dir, records)
    qa = build_qa(args, output_dir, records)
    run_manifest = {
        "status": qa["status"],
        "configuration_sha256": sha256_file(output_dir / "configuration.json"),
        "n_internal_predictions": len(records["internal_predictions"]),
        "n_ood_predictions": len(records["ood_predictions"]),
        "component_execution_provenance_sha256": (
            sha256_file(output_dir / "component_execution_provenance.json")
            if (output_dir / "component_execution_provenance.json").is_file()
            else None
        ),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
