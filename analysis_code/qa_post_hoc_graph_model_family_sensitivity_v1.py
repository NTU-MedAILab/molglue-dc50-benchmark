#!/usr/bin/env python3
"""Independent read-only QA for the frozen molecular-graph sensitivity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SCRIPT_DIR = Path(__file__).resolve().parent
ROUTE_DIR = SCRIPT_DIR.parent
RESULT_DIR = (
    ROUTE_DIR
    / "reports"
    / "post_hoc_graph_model_family_sensitivity_v1"
)
OUTPUT_FILE = (
    ROUTE_DIR
    / "reports"
    / "publication_validation_v1"
    / "post_hoc_graph_model_family_sensitivity_v1_independent_qa.json"
)
DATA_FILE = (
    ROUTE_DIR
    / "data"
    / "processed"
    / "all_molglue_dc50_qc_train_test_standardized_context.csv"
)
PROTOCOL_FILE = (
    ROUTE_DIR
    / "docs"
    / "post_hoc_graph_model_family_sensitivity_protocol_v1.md"
)
RUNNER_FILE = (
    SCRIPT_DIR / "run_post_hoc_graph_model_family_sensitivity_v1.py"
)

CHEM = "chemistry_graph_mpn"
FULL = "full_context_graph_mpn"
MODELS = {CHEM, FULL}
EXPECTED_OOD_COUNTS = {
    ("domain_plus_compound_cold", "source_ood"): 3_040,
    ("domain_plus_compound_cold", "target_ood"): 2_580,
    ("strict_domain_plus_scaffold_cold", "source_ood"): 3_040,
    ("strict_domain_plus_scaffold_cold", "target_ood"): 2_580,
}
FORMAL_CONFIG = {
    "hidden_dim": 128,
    "graph_dim": 128,
    "context_dim": 64,
    "n_layers": 3,
    "dropout": 0.15,
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "max_epochs": 120,
    "patience": 15,
    "smooth_l1_beta": 0.5,
    "gradient_clip_norm": 5.0,
    "inner_folds": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--output-file", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--expect-smoke", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.ptp(y_true) == 0 or np.ptp(y_pred) == 0:
        return float("nan")
    result = spearmanr(y_true, y_pred)
    value = (
        result.statistic
        if hasattr(result, "statistic")
        else result.correlation
    )
    return float(value)


def metrics(frame: pd.DataFrame) -> tuple[float, float]:
    truth = frame["y_true"].to_numpy(dtype=float)
    prediction = frame["y_pred"].to_numpy(dtype=float)
    return (
        safe_spearman(truth, prediction),
        float(np.sqrt(np.mean(np.square(truth - prediction)))),
    )


def close(first: Any, second: Any, tolerance: float = 1e-10) -> bool:
    a = float(first)
    b = float(second)
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance)


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    output_file = args.output_file.resolve()
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append(
            {
                "name": name,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
        )

    manifest = json.loads((result_dir / "run_manifest.json").read_text())
    expected_smoke = bool(args.expect_smoke)
    add(
        "manifest_identity",
        manifest.get("status") == "complete"
        and manifest.get("analysis_label")
        == "post_hoc_graph_model_family_sensitivity"
        and manifest.get("protocol_version")
        == "post_hoc_graph_model_family_sensitivity_v1"
        and manifest.get("confirmatory") is False
        and manifest.get("smoke") is expected_smoke
        and (expected_smoke or manifest.get("gpu_used") is True),
        "complete, post-hoc graph identity; formal execution must use GPU",
    )
    add(
        "identity_hashes",
        manifest.get("data_sha256") == sha256_file(DATA_FILE)
        and manifest.get("script_sha256") == sha256_file(RUNNER_FILE)
        and manifest.get("protocol_document_sha256")
        == sha256_file(PROTOCOL_FILE),
        "data, runner and protocol hashes match current frozen files",
    )
    add(
        "formal_configuration",
        manifest.get("formal_training_config") == FORMAL_CONFIG
        and manifest.get("no_hyperparameter_selection") is True,
        "frozen graph architecture and training settings match",
    )

    inventory = pd.read_csv(result_dir / "artifact_sha256.csv")
    inventory_pass = True
    for record in inventory.to_dict("records"):
        path = result_dir / str(record["filename"])
        inventory_pass &= (
            path.is_file()
            and path.stat().st_size == int(record["size_bytes"])
            and sha256_file(path) == str(record["sha256"])
        )
    add(
        "artifact_inventory",
        bool(inventory_pass),
        f"{len(inventory)}/{len(inventory)} listed artifacts hash-checked",
    )

    predictions = pd.read_csv(result_dir / "predictions.csv")
    add(
        "prediction_models_and_finiteness",
        set(predictions["model_id"]) == MODELS
        and np.isfinite(predictions["y_true"]).all()
        and np.isfinite(predictions["y_pred"]).all(),
        "exactly two paired graph models and finite values",
    )
    key_columns = [
        "split_regime",
        "protocol",
        "heldout_group",
        "model_id",
        "row_index",
    ]
    add(
        "prediction_key_uniqueness",
        not predictions.duplicated(key_columns).any(),
        "one prediction per split/model/row key",
    )
    truth_counts = predictions.groupby("row_index")["y_true"].nunique()
    add(
        "truth_stability",
        bool((truth_counts == 1).all()),
        "each original row has one invariant endpoint",
    )

    internal = predictions[predictions["split_regime"] == "internal"]
    internal_counts = internal.groupby(["model_id", "row_index"]).size()
    if expected_smoke:
        internal_pass = (
            len(internal_counts) > 0
            and internal_counts.min() == 1
            and internal_counts.max() == 1
        )
        internal_detail = "smoke contains one scaffold outer split"
    else:
        internal_pass = (
            len(predictions) == 26_840
            and len(internal) == 15_600
            and len(internal_counts) == 3_120
            and internal_counts.min() == 5
            and internal_counts.max() == 5
        )
        internal_detail = "1,560 rows x 2 models x 5 OOF predictions"
    add("internal_cross_fitting", bool(internal_pass), internal_detail)

    ood_counts = (
        predictions[predictions["split_regime"] != "internal"]
        .groupby(["split_regime", "protocol"])
        .size()
        .to_dict()
    )
    add(
        "ood_prediction_counts",
        bool(ood_counts) if expected_smoke else ood_counts == EXPECTED_OOD_COUNTS,
        str(ood_counts),
    )

    strict = pd.read_csv(result_dir / "strict_split_audit.csv")
    strict_pass = (
        len(strict) == (2 if expected_smoke else 12)
        and (strict["train_test_row_overlap"] == 0).all()
        and (strict["train_test_smiles_overlap"] == 0).all()
        and (strict["train_test_scaffold_overlap"] == 0).all()
        and (~strict["heldout_token_seen_in_train"].astype(bool)).all()
    )
    add(
        "strict_leakage_audit",
        bool(strict_pass),
        "strict splits have zero row, compound, scaffold and domain overlap",
    )

    training = pd.read_csv(result_dir / "training_audit.csv")
    paired_seed_counts = (
        training.groupby(
            ["split_regime", "protocol", "heldout_group"]
        )["seed"].nunique()
    )
    model_counts = training.groupby(
        ["split_regime", "protocol", "heldout_group"]
    )["model_id"].nunique()
    training_pass = (
        len(training) == (10 if expected_smoke else 98)
        and (paired_seed_counts == 1).all()
        and (model_counts == 2).all()
        and (training["inner_fit_validation_row_overlap"] == 0).all()
        and (training["inner_train_test_row_overlap"] == 0).all()
        and (training["inner_fit_validation_scaffold_overlap"] == 0).all()
        and (training["best_epoch"] >= 1).all()
        and (training["best_epoch"] <= training["epochs_trained"]).all()
        and (
            training.loc[training["model_id"] == CHEM, "context_dim"] == 0
        ).all()
        and (
            training.loc[training["model_id"] == FULL, "context_dim"] > 0
        ).all()
    )
    add(
        "training_audit",
        bool(training_pass),
        "paired seeds/models, fold-fitted context, and zero inner leakage",
    )

    graph = pd.read_csv(result_dir / "graph_construction_audit.csv")
    graph_pass = (
        len(graph) == 1_560
        and graph["row_index"].nunique() == 1_560
        and (graph["n_atoms"] > 0).all()
        and (graph["n_undirected_bonds"] >= 0).all()
        and graph["atom_feature_dim"].nunique() == 1
        and int(graph["atom_feature_dim"].iloc[0]) == 43
    )
    add(
        "graph_construction",
        bool(graph_pass),
        "1,560 non-empty molecular graphs with the frozen 43 atom features",
    )

    averaged = (
        internal.groupby(
            [
                "model_id",
                "row_index",
                "qc_id",
                "canonical_smiles",
                "scaffold_id",
            ],
            as_index=False,
        )
        .agg(
            y_true=("y_true", "first"),
            y_pred=("y_pred", "mean"),
            n_oof_predictions=("y_pred", "size"),
        )
    )
    stored_averaged = pd.read_csv(
        result_dir / "internal_repeat_averaged_predictions.csv"
    )
    merged_average = averaged.merge(
        stored_averaged,
        on=["model_id", "row_index"],
        suffixes=("_calc", "_stored"),
        validate="one_to_one",
    )
    average_pass = (
        len(merged_average) == len(averaged) == len(stored_averaged)
        and np.allclose(
            merged_average["y_pred_calc"],
            merged_average["y_pred_stored"],
            rtol=0,
            atol=1e-12,
        )
    )
    add(
        "repeat_average_recalculation",
        bool(average_pass),
        "repeat-averaged predictions independently reproduced",
    )

    stored_internal = pd.read_csv(result_dir / "internal_metrics.csv")
    internal_metric_pass = True
    for model_id, group in averaged.groupby("model_id"):
        rho, rmse = metrics(group)
        stored = stored_internal[stored_internal["model_id"] == model_id]
        internal_metric_pass &= (
            len(stored) == 1
            and close(rho, stored.iloc[0]["spearman"])
            and close(rmse, stored.iloc[0]["rmse"])
        )
    add(
        "internal_metric_recalculation",
        bool(internal_metric_pass),
        "internal Spearman and RMSE independently reproduced",
    )

    ood = predictions[predictions["split_regime"] != "internal"]
    stored_domain = pd.read_csv(result_dir / "ood_domain_metrics.csv")
    domain_pass = True
    group_columns = [
        "split_regime",
        "protocol",
        "heldout_group",
        "model_id",
    ]
    for key, group in ood.groupby(group_columns):
        rho, rmse = metrics(group)
        stored = stored_domain.copy()
        for column, value in zip(group_columns, key):
            stored = stored[stored[column] == value]
        domain_pass &= (
            len(stored) == 1
            and close(rho, stored.iloc[0]["spearman"])
            and close(rmse, stored.iloc[0]["rmse"])
        )
    add(
        "ood_domain_metric_recalculation",
        bool(domain_pass),
        "all OOD domain metrics independently reproduced",
    )

    stored_aggregate = pd.read_csv(result_dir / "ood_aggregate_metrics.csv")
    aggregate_pass = True
    for key, group in stored_domain.groupby(
        ["split_regime", "protocol", "model_id"]
    ):
        stored = stored_aggregate.copy()
        for column, value in zip(
            ["split_regime", "protocol", "model_id"], key
        ):
            stored = stored[stored[column] == value]
        finite_rho = group["spearman"].dropna()
        expected_rho = (
            finite_rho.mean()
            if len(finite_rho) == len(group)
            else float("nan")
        )
        aggregate_pass &= (
            len(stored) == 1
            and close(expected_rho, stored.iloc[0]["domain_macro_spearman"])
            and close(
                group["rmse"].mean(),
                stored.iloc[0]["domain_macro_rmse"],
            )
        )
    add(
        "ood_aggregate_recalculation",
        bool(aggregate_pass),
        "equal-domain macro metrics independently reproduced",
    )

    bootstrap = pd.read_csv(
        result_dir / "paired_global_scaffold_bootstrap.csv"
    )
    expected_bootstrap = 100 if expected_smoke else 10_000
    bootstrap_pass = (
        len(bootstrap) == 15
        and set(bootstrap["record_type"]) == {"model", "contrast"}
        and set(bootstrap.loc[bootstrap["record_type"] == "contrast", "contrast_id"])
        == {"full_vs_chemistry_graph_mpn"}
        and set(bootstrap["n_bootstrap"]) == {expected_bootstrap}
    )
    add(
        "bootstrap_contract",
        bool(bootstrap_pass),
        f"five regimes/axes x paired model/contrast records; n={expected_bootstrap}",
    )

    passed = sum(check["status"] == "PASS" for check in checks)
    payload = {
        "analysis_label": "post_hoc_graph_model_family_sensitivity",
        "status": "PASS" if passed == len(checks) else "FAIL",
        "expect_smoke": expected_smoke,
        "n_checks": len(checks),
        "n_pass": passed,
        "n_fail": len(checks) - passed,
        "checks": checks,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
