#!/usr/bin/env python3
"""Independent fail-closed QA for the internal learning-curve sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_confirmatory_cpu_v1 as core
import run_post_hoc_internal_learning_curve_v1 as learning


SCRIPT_DIR = Path(__file__).resolve().parent
ROUTE_DIR = SCRIPT_DIR.parent
DEFAULT_RUN_DIR = (
    ROUTE_DIR / "reports" / "post_hoc_internal_learning_curve_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def add(
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


def load_csv(run_dir: Path, name: str) -> pd.DataFrame:
    path = run_dir / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    predictions = load_csv(run_dir, "cross_fitted_predictions.csv")
    averaged = load_csv(run_dir, "repeat_averaged_predictions.csv")
    fold_metrics = load_csv(run_dir, "outer_fold_metrics.csv")
    training_sizes = load_csv(run_dir, "training_sizes.csv")
    metrics = load_csv(run_dir, "learning_curve_metrics.csv")
    contrasts = load_csv(run_dir, "learning_curve_contrasts.csv")
    inventory = load_csv(run_dir, "artifact_sha256.csv")

    formal = manifest.get("analysis_mode") == "formal_post_hoc"
    checks: list[dict[str, Any]] = []
    add(
        checks,
        "manifest_complete",
        manifest.get("status") == "complete",
        str(manifest.get("status")),
    )
    add(
        checks,
        "analysis_identity",
        manifest.get("analysis_identity") == learning.ANALYSIS_IDENTITY
        and manifest.get("evidence_identity") == "post_hoc_sensitivity",
        (
            f"{manifest.get('analysis_identity')}; "
            f"{manifest.get('evidence_identity')}"
        ),
    )
    add(
        checks,
        "data_hash",
        manifest.get("data_sha256") == core.FROZEN_DATA_SHA256,
        str(manifest.get("data_sha256")),
    )
    add(
        checks,
        "protocol_and_script_hashes",
        manifest.get("protocol_sha256")
        == core.sha256_file(learning.PROTOCOL_PATH)
        and manifest.get("script_sha256")
        == core.sha256_file(Path(learning.__file__)),
        "current protocol and runner match manifest",
    )
    add(
        checks,
        "frozen_fractions_and_models",
        manifest.get("fractions") == list(learning.FRACTIONS)
        and manifest.get("models") == list(learning.MODEL_IDS),
        f"fractions={manifest.get('fractions')}; models={manifest.get('models')}",
    )
    params = manifest.get("fixed_model_parameters", {})
    estimator_count_ok = (
        params.get("n_estimators") == 600
        if formal
        else int(params.get("n_estimators", 0)) >= 10
    )
    add(
        checks,
        "fixed_model_configuration",
        params.get("min_samples_leaf") == 1
        and params.get("max_features") == "sqrt"
        and estimator_count_ok
        and params.get("bootstrap") is False,
        json.dumps(params, sort_keys=True),
    )
    add(
        checks,
        "gpu_not_required",
        manifest.get("gpu_required") is False,
        str(manifest.get("gpu_required")),
    )

    rows, _ = core.load_rows(core.DATA_FILE)
    y = rows["pDC50"].to_numpy(dtype=float)
    scaffold_ids = core.build_scaffold_ids(rows["canonical_smiles"])
    splits = core.make_outer_splits(
        "scaffold",
        rows,
        scaffold_ids,
        n_folds=5,
        n_repeats=5,
        seed=int(manifest["seed"]),
    )
    if not formal:
        splits = splits[: int(manifest["outer_splits_executed"])]

    selection_ok = True
    leakage_ok = True
    nesting_ok = True
    size_ok = True
    for split in splits:
        order_seed = (
            int(manifest["seed"])
            + split.repeat * 10_000
            + split.fold * 100
        )
        ordered = learning.ordered_training_scaffolds(
            scaffold_ids,
            split.train_idx,
            seed=order_seed,
        )
        prior: set[str] = set()
        for fraction in learning.FRACTIONS:
            selected_idx, selected = learning.select_training_indices(
                split.train_idx,
                scaffold_ids,
                ordered,
                fraction,
            )
            selected_set = set(map(str, selected))
            nesting_ok &= prior.issubset(selected_set)
            prior = selected_set
            leakage_ok &= not bool(
                selected_set.intersection(
                    set(map(str, scaffold_ids[split.test_idx]))
                )
            )
            expected_hash = learning.sha256_text(list(selected_set))
            observed = training_sizes[
                (training_sizes["repeat"] == split.repeat)
                & (training_sizes["outer_fold"] == split.fold)
                & np.isclose(
                    training_sizes["training_fraction"],
                    fraction,
                )
            ]
            if len(observed) != 1:
                selection_ok = False
                size_ok = False
                continue
            record = observed.iloc[0]
            selection_ok &= str(record["selection_sha256"]) == expected_hash
            size_ok &= (
                int(record["n_train_rows"]) == len(selected_idx)
                and int(record["n_train_scaffolds"]) == len(selected)
                and int(record["n_test_rows"]) == len(split.test_idx)
            )
    add(checks, "deterministic_selection_identity", selection_ok, "recomputed")
    add(checks, "nested_training_scaffolds", nesting_ok, "recomputed")
    add(checks, "no_train_test_scaffold_leakage", leakage_ok, "recomputed")
    add(checks, "training_and_test_counts", size_ok, "recomputed")

    expected_prediction_rows = (
        sum(len(split.test_idx) for split in splits)
        * len(learning.FRACTIONS)
        * len(learning.MODEL_IDS)
    )
    if formal:
        expected_prediction_rows = (
            5
            * len(rows)
            * len(learning.FRACTIONS)
            * len(learning.MODEL_IDS)
        )
    add(
        checks,
        "prediction_row_count",
        len(predictions) == expected_prediction_rows,
        f"observed={len(predictions)}; expected={expected_prediction_rows}",
    )
    duplicate_keys = [
        "repeat",
        "outer_fold",
        "training_fraction",
        "model_id",
        "row_index",
    ]
    add(
        checks,
        "prediction_keys_unique",
        not predictions.duplicated(duplicate_keys).any(),
        f"duplicates={int(predictions.duplicated(duplicate_keys).sum())}",
    )
    truth_ok = np.allclose(
        predictions["y_true"].to_numpy(dtype=float),
        y[predictions["row_index"].to_numpy(dtype=int)],
        rtol=0.0,
        atol=1e-6,
    )
    add(checks, "prediction_truth_identity", truth_ok, "matched frozen endpoint")

    if formal:
        coverage = (
            predictions.groupby(
                ["training_fraction", "model_id", "row_index"]
            )["repeat"]
            .nunique()
            .to_numpy()
        )
        coverage_ok = (
            len(averaged)
            == len(rows) * len(learning.FRACTIONS) * len(learning.MODEL_IDS)
            and np.all(coverage == 5)
            and np.all(averaged["n_repeat_predictions"] == 5)
        )
    else:
        coverage_ok = len(averaged) > 0
    add(checks, "repeat_averaged_coverage", coverage_ok, f"rows={len(averaged)}")

    point_ok = True
    for record in metrics.to_dict(orient="records"):
        frame = averaged[
            np.isclose(
                averaged["training_fraction"],
                float(record["training_fraction"]),
            )
            & (averaged["model_id"] == record["model_id"])
        ]
        if record["metric"] == "spearman":
            expected = core.safe_spearman(
                frame["y_true"].to_numpy(dtype=float),
                frame["y_pred"].to_numpy(dtype=float),
            )
        else:
            expected = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            frame["y_true"].to_numpy(dtype=float)
                            - frame["y_pred"].to_numpy(dtype=float)
                        )
                    )
                )
            )
        point_ok &= np.isclose(
            float(record["estimate"]),
            expected,
            rtol=0.0,
            atol=1e-9,
        )
    add(checks, "metric_point_estimates", point_ok, "independently recomputed")

    contrast_ok = True
    metric_lookup = {
        (
            float(row["training_fraction"]),
            str(row["model_id"]),
            str(row["metric"]),
        ): float(row["estimate"])
        for row in metrics.to_dict(orient="records")
    }
    for row in contrasts.to_dict(orient="records"):
        low = float(row["training_fraction_low"])
        high = float(row["training_fraction_high"])
        model = str(row["model_id"])
        if row["contrast_id"] == "full_minus_chemistry_spearman":
            expected = (
                metric_lookup[(low, "full_context_extra_trees", "spearman")]
                - metric_lookup[(low, "chemistry_extra_trees", "spearman")]
            )
        elif row["contrast_id"] == "chemistry_minus_full_rmse":
            expected = (
                metric_lookup[(low, "chemistry_extra_trees", "rmse")]
                - metric_lookup[(low, "full_context_extra_trees", "rmse")]
            )
        elif row["contrast_id"] == "fraction_100_minus_75_spearman":
            expected = (
                metric_lookup[(high, model, "spearman")]
                - metric_lookup[(low, model, "spearman")]
            )
        elif row["contrast_id"] == "fraction_75_minus_100_rmse":
            expected = (
                metric_lookup[(low, model, "rmse")]
                - metric_lookup[(high, model, "rmse")]
            )
        else:
            contrast_ok = False
            continue
        contrast_ok &= np.isclose(
            float(row["estimate"]),
            expected,
            rtol=0.0,
            atol=1e-9,
        )
    add(checks, "paired_contrast_point_estimates", contrast_ok, "recomputed")

    if formal:
        intervals = pd.concat(
            [
                metrics[["estimate", "ci_low", "ci_high"]],
                contrasts[["estimate", "ci_low", "ci_high"]],
            ],
            ignore_index=True,
        )
        interval_ok = (
            np.isfinite(intervals.to_numpy(dtype=float)).all()
            and (intervals["ci_low"] <= intervals["ci_high"]).all()
            and (metrics["bootstrap_replicates"] == 10_000).all()
            and (contrasts["bootstrap_replicates"] == 10_000).all()
        )
    else:
        interval_ok = (
            metrics["ci_low"].isna().all()
            and contrasts["ci_low"].isna().all()
        )
    add(checks, "bootstrap_interval_contract", interval_ok, "formal or smoke contract")

    inventory_ok = True
    for row in inventory.to_dict(orient="records"):
        artifact = str(row["artifact"])
        if artifact == "protocol_document":
            path = learning.PROTOCOL_PATH
        elif artifact == "runner_script":
            path = Path(learning.__file__)
        else:
            path = run_dir / artifact
        inventory_ok &= (
            path.is_file()
            and int(row["size_bytes"]) == path.stat().st_size
            and str(row["sha256"]) == core.sha256_file(path)
        )
    add(checks, "artifact_hash_inventory", inventory_ok, f"files={len(inventory)}")

    expected_fit_count = (
        len(splits) * len(learning.FRACTIONS) * len(learning.MODEL_IDS)
    )
    add(
        checks,
        "manifest_counts",
        int(manifest.get("fit_count", -1)) == expected_fit_count
        and int(manifest.get("prediction_rows", -1)) == len(predictions),
        (
            f"fits={manifest.get('fit_count')}; "
            f"prediction_rows={manifest.get('prediction_rows')}"
        ),
    )

    failed = [check for check in checks if check["status"] != "PASS"]
    payload = {
        "qa_identity": "post_hoc_internal_learning_curve_v1_independent_qa",
        "status": "PASS" if not failed else "FAIL",
        "run_directory": str(run_dir),
        "formal": formal,
        "checks_passed": len(checks) - len(failed),
        "checks_total": len(checks),
        "checks": checks,
        "gpu_required": False,
    }
    output_json = (
        args.output_json.resolve()
        if args.output_json is not None
        else run_dir / "independent_qa.json"
    )
    output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        (
            f"POST_HOC_INTERNAL_LEARNING_CURVE_QA: {payload['status']} "
            f"({payload['checks_passed']}/{payload['checks_total']})"
        ),
        flush=True,
    )
    print("GPU_REQUIRED: NO", flush=True)
    if failed:
        for check in failed:
            print(
                f"FAIL: {check['check_id']}: {check['detail']}",
                flush=True,
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
