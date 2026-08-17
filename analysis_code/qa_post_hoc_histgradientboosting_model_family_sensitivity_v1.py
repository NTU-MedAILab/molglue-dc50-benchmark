#!/usr/bin/env python3
"""Independent read-only QA for the frozen HistGradientBoosting sensitivity."""

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
    / "post_hoc_histgradientboosting_model_family_sensitivity_v1"
)
OUTPUT_DIR = ROUTE_DIR / "reports" / "publication_validation_v1"
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
RUNNER_FILE = (
    SCRIPT_DIR
    / "run_post_hoc_histgradientboosting_model_family_sensitivity_v1.py"
)

CHEM = "chemistry_hist_gradient_boosting"
FULL = "full_context_hist_gradient_boosting"
MODELS = {CHEM, FULL}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
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
    return float(spearmanr(y_true, y_pred).statistic)


def metrics(frame: pd.DataFrame) -> tuple[float, float]:
    y_true = frame["y_true"].to_numpy(dtype=float)
    y_pred = frame["y_pred"].to_numpy(dtype=float)
    return (
        safe_spearman(y_true, y_pred),
        float(np.sqrt(np.mean(np.square(y_true - y_pred)))),
    )


def close(a: Any, b: Any, tolerance: float = 1e-12) -> bool:
    a_float = float(a)
    b_float = float(b)
    if math.isnan(a_float) and math.isnan(b_float):
        return True
    return math.isclose(a_float, b_float, rel_tol=0.0, abs_tol=tolerance)


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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
    add(
        "manifest_identity",
        manifest.get("status") == "complete"
        and manifest.get("analysis_label")
        == "post_hoc_model_family_sensitivity"
        and manifest.get("confirmatory") is False
        and manifest.get("smoke") is False
        and manifest.get("gpu_used") is False,
        "complete post-hoc, non-smoke, CPU-only manifest",
    )
    add(
        "identity_hashes",
        manifest.get("data_sha256") == sha256_file(DATA_FILE)
        and manifest.get("script_sha256") == sha256_file(RUNNER_FILE)
        and manifest.get("protocol_document_sha256")
        == sha256_file(PROTOCOL_FILE),
        "data, runner and frozen protocol hashes match",
    )

    inventory = pd.read_csv(result_dir / "artifact_sha256.csv")
    inventory_pass = True
    for row in inventory.to_dict("records"):
        path = result_dir / str(row["filename"])
        inventory_pass &= (
            path.is_file()
            and path.stat().st_size == int(row["size_bytes"])
            and sha256_file(path) == str(row["sha256"])
        )
    add(
        "artifact_inventory",
        bool(inventory_pass),
        f"{len(inventory)}/{len(inventory)} listed artifacts hash-checked",
    )

    predictions = pd.read_csv(result_dir / "predictions.csv")
    add(
        "prediction_shape",
        len(predictions) == 26_840,
        f"observed={len(predictions)} expected=26840",
    )
    add(
        "prediction_models_and_finiteness",
        set(predictions["model_id"]) == MODELS
        and np.isfinite(predictions["y_true"]).all()
        and np.isfinite(predictions["y_pred"]).all(),
        "two frozen models; all endpoints and predictions finite",
    )
    duplicate_keys = [
        "split_regime",
        "protocol",
        "heldout_group",
        "model_id",
        "row_index",
    ]
    add(
        "prediction_key_uniqueness",
        not predictions.duplicated(duplicate_keys).any(),
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
    add(
        "internal_cross_fitting",
        len(internal) == 15_600
        and len(internal_counts) == 3_120
        and internal_counts.min() == 5
        and internal_counts.max() == 5,
        "1,560 rows × 2 models × 5 OOF predictions",
    )
    ood_counts = (
        predictions[predictions["split_regime"] != "internal"]
        .groupby(["split_regime", "protocol"])
        .size()
        .to_dict()
    )
    expected_ood_counts = {
        ("domain_plus_compound_cold", "source_ood"): 3_040,
        ("domain_plus_compound_cold", "target_ood"): 2_580,
        ("strict_domain_plus_scaffold_cold", "source_ood"): 3_040,
        ("strict_domain_plus_scaffold_cold", "target_ood"): 2_580,
    }
    add(
        "ood_prediction_counts",
        ood_counts == expected_ood_counts,
        str(ood_counts),
    )

    strict_audit = pd.read_csv(result_dir / "strict_split_audit.csv")
    strict_pass = (
        len(strict_audit) == 12
        and (strict_audit["train_test_row_overlap"] == 0).all()
        and (strict_audit["train_test_smiles_overlap"] == 0).all()
        and (strict_audit["train_test_scaffold_overlap"] == 0).all()
        and (~strict_audit["heldout_token_seen_in_train"].astype(bool)).all()
    )
    add(
        "strict_leakage_audit",
        bool(strict_pass),
        "12/12 strict splits have zero row, compound, scaffold and domain overlap",
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
            sort=True,
        )
        .agg(
            y_true=("y_true", "first"),
            y_pred=("y_pred", "mean"),
        )
    )
    internal_reported = pd.read_csv(result_dir / "internal_metrics.csv")
    internal_metric_pass = True
    for model_id in MODELS:
        rho, error = metrics(averaged[averaged["model_id"] == model_id])
        row = internal_reported[internal_reported["model_id"] == model_id].iloc[0]
        internal_metric_pass &= close(rho, row["spearman"]) and close(
            error, row["rmse"]
        )
    add(
        "internal_metric_recalculation",
        bool(internal_metric_pass),
        "independent recalculation agrees within 1e-12",
    )

    noninternal = predictions[predictions["split_regime"] != "internal"]
    reported_domains = pd.read_csv(result_dir / "ood_domain_metrics.csv")
    domain_pass = True
    recalculated_domains: list[dict[str, Any]] = []
    keys = ["split_regime", "protocol", "heldout_group", "model_id"]
    for key, group in noninternal.groupby(keys, sort=True):
        rho, error = metrics(group)
        reported = reported_domains.copy()
        for column, value in zip(keys, key):
            reported = reported[reported[column] == value]
        if len(reported) != 1:
            domain_pass = False
            continue
        row = reported.iloc[0]
        domain_pass &= close(rho, row["spearman"]) and close(error, row["rmse"])
        recalculated_domains.append(
            {
                "split_regime": key[0],
                "protocol": key[1],
                "heldout_group": key[2],
                "model_id": key[3],
                "spearman": rho,
                "rmse": error,
            }
        )
    add(
        "domain_metric_recalculation",
        bool(domain_pass) and len(recalculated_domains) == 48,
        "48/48 domain-model metric cells independently recalculated",
    )

    domain_frame = pd.DataFrame(recalculated_domains)
    reported_aggregate = pd.read_csv(result_dir / "ood_aggregate_metrics.csv")
    aggregate_pass = True
    for key, group in domain_frame.groupby(
        ["split_regime", "protocol", "model_id"], sort=True
    ):
        finite_rho = group["spearman"].dropna()
        macro_rho = (
            finite_rho.mean()
            if len(finite_rho) == len(group)
            else float("nan")
        )
        macro_rmse = group["rmse"].mean()
        reported = reported_aggregate.copy()
        for column, value in zip(
            ["split_regime", "protocol", "model_id"], key
        ):
            reported = reported[reported[column] == value]
        if len(reported) != 1:
            aggregate_pass = False
            continue
        row = reported.iloc[0]
        aggregate_pass &= close(
            macro_rho, row["domain_macro_spearman"]
        ) and close(macro_rmse, row["domain_macro_rmse"])
    add(
        "aggregate_metric_recalculation",
        bool(aggregate_pass),
        "8/8 regime-protocol-model macro cells independently recalculated",
    )

    bootstrap = pd.read_csv(
        result_dir / "paired_global_scaffold_bootstrap.csv"
    )
    ordered = True
    interval_pairs = [
        ("spearman_ci_low", "spearman_ci_high"),
        ("rmse_ci_low", "rmse_ci_high"),
        ("delta_spearman_ci_low", "delta_spearman_ci_high"),
        ("delta_rmse_ci_low", "delta_rmse_ci_high"),
        (
            "domain_macro_spearman_ci_low",
            "domain_macro_spearman_ci_high",
        ),
        ("domain_macro_rmse_ci_low", "domain_macro_rmse_ci_high"),
        (
            "delta_domain_macro_spearman_ci_low",
            "delta_domain_macro_spearman_ci_high",
        ),
        (
            "delta_domain_macro_rmse_ci_low",
            "delta_domain_macro_rmse_ci_high",
        ),
    ]
    for low, high in interval_pairs:
        if low not in bootstrap or high not in bootstrap:
            continue
        finite = bootstrap[[low, high]].dropna()
        ordered &= bool((finite[low] <= finite[high]).all())
    add(
        "bootstrap_structure",
        len(bootstrap) == 15
        and set(bootstrap["n_bootstrap"].dropna().astype(int)) == {10_000}
        and ordered,
        "15 summary rows; all finite interval bounds ordered; 10,000 replicates",
    )

    qa_pass = all(row["status"] == "PASS" for row in checks)
    payload = {
        "analysis": "independent_read_only_qa",
        "target_analysis": "post_hoc_model_family_sensitivity",
        "status": "PASS" if qa_pass else "FAIL",
        "n_checks": len(checks),
        "n_passed": sum(row["status"] == "PASS" for row in checks),
        "n_failed": sum(row["status"] == "FAIL" for row in checks),
        "checks": checks,
    }
    json_path = output_dir / "hgb_independent_qa_summary.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# HistGradientBoosting sensitivity independent QA",
        "",
        f"Overall: **{payload['status']}**",
        "",
        f"- Checks: {payload['n_passed']}/{payload['n_checks']} passed;",
        "- Independent recalculation tolerance: absolute 1e-12;",
        "- This audit does not promote the post-hoc analysis to confirmatory.",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| {row['name']} | {row['status']} | "
        f"{str(row['detail']).replace('|', '/')} |"
        for row in checks
    )
    (output_dir / "hgb_independent_qa_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        f"[QA] status={payload['status']} "
        f"passed={payload['n_passed']}/{payload['n_checks']}"
    )
    if not qa_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
