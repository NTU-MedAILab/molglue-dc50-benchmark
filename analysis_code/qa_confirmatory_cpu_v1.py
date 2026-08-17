#!/usr/bin/env python
"""Fail-closed QA for the frozen confirmatory CPU v1 result bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_REPORT_DIR = PROJECT_DIR / "reports" / "confirmatory_cpu_v1"
EXPECTED_PROTOCOLS = {"scaffold", "compound"}
EXPECTED_MODELS = {
    "global_mean",
    "hierarchical_context_mean",
    "context_ridge",
    "morgan_knn",
    "chemistry_extra_trees",
    "full_context_extra_trees",
    "full_context_label_shuffle",
}
EXPECTED_REPEATS = set(range(1, 6))
EXPECTED_FOLDS = set(range(1, 6))
N_ROWS = 1560
N_UNIQUE_SMILES = 1137
N_SCAFFOLDS = 667


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    def format_value(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            if not math.isfinite(float(value)):
                return "NA"
            return f"{float(value):.6f}"
        return str(value).replace("|", "\\|")

    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(
            "| " + " | ".join(format_value(value) for value in row) + " |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir.resolve()
    manifest_path = report_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw = pd.read_csv(report_dir / "all_cross_fitted_predictions.csv")
    averaged = pd.read_csv(
        report_dir / "repeat_averaged_cross_fitted_predictions.csv"
    )
    metrics = pd.read_csv(
        report_dir / "metrics_on_repeat_averaged_predictions.csv"
    )
    bootstrap = pd.read_csv(report_dir / "paired_cluster_bootstrap.csv")
    coverage = pd.read_csv(report_dir / "risk_coverage_metrics.csv")
    audit = pd.read_csv(report_dir / "context_sanitization_audit.csv")

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "detail": detail,
            }
        )

    check(
        "manifest_identity",
        (
            manifest.get("status") == "complete"
            and manifest.get("analysis_mode") == "confirmatory"
            and manifest.get("frozen_protocol_match") is True
        ),
        (
            f"status={manifest.get('status')}; "
            f"mode={manifest.get('analysis_mode')}; "
            f"frozen={manifest.get('frozen_protocol_match')}"
        ),
    )
    check(
        "manifest_counts",
        (
            manifest.get("n_rows") == N_ROWS
            and manifest.get("n_unique_smiles") == N_UNIQUE_SMILES
            and manifest.get("n_scaffolds") == N_SCAFFOLDS
        ),
        (
            f"rows={manifest.get('n_rows')}; "
            f"smiles={manifest.get('n_unique_smiles')}; "
            f"scaffolds={manifest.get('n_scaffolds')}"
        ),
    )
    identity_paths = {
        "data": Path(manifest["data_file"]),
        "script": Path(manifest["script_file"]),
        "protocol_document": Path(manifest["protocol_document"]),
    }
    identity_hashes = {
        name: sha256_file(path) for name, path in identity_paths.items()
    }
    check(
        "identity_hashes",
        (
            identity_hashes["data"] == manifest["data_sha256"]
            and identity_hashes["script"] == manifest["script_sha256"]
            and identity_hashes["protocol_document"]
            == manifest["protocol_document_sha256"]
        ),
        "; ".join(
            f"{name}={digest}" for name, digest in identity_hashes.items()
        ),
    )

    expected_raw_rows = (
        len(EXPECTED_PROTOCOLS)
        * len(EXPECTED_REPEATS)
        * len(EXPECTED_MODELS)
        * N_ROWS
    )
    raw_key = ["protocol", "repeat", "model_id", "row_index"]
    check(
        "raw_shape_and_unique_keys",
        (
            len(raw) == expected_raw_rows
            and not raw.duplicated(raw_key).any()
        ),
        f"observed={len(raw)}; expected={expected_raw_rows}",
    )
    check(
        "raw_sets",
        (
            set(raw["protocol"]) == EXPECTED_PROTOCOLS
            and set(raw["repeat"]) == EXPECTED_REPEATS
            and set(raw["model_id"]) == EXPECTED_MODELS
        ),
        (
            f"protocols={sorted(raw['protocol'].unique())}; "
            f"repeats={sorted(raw['repeat'].unique())}; "
            f"models={sorted(raw['model_id'].unique())}"
        ),
    )
    raw_group_sizes = raw.groupby(
        ["protocol", "repeat", "model_id"]
    ).size()
    check(
        "raw_complete_cross_fitting",
        bool((raw_group_sizes == N_ROWS).all()),
        (
            f"groups={len(raw_group_sizes)}; "
            f"min={raw_group_sizes.min()}; max={raw_group_sizes.max()}"
        ),
    )
    numeric_prediction = pd.to_numeric(raw["y_pred"], errors="coerce")
    check(
        "raw_predictions_finite",
        bool(np.isfinite(numeric_prediction).all()),
        (
            f"min={numeric_prediction.min():.6g}; "
            f"max={numeric_prediction.max():.6g}"
        ),
    )
    check(
        "truth_is_row_stable",
        bool((raw.groupby("row_index")["y_true"].nunique() == 1).all()),
        "Each original row has one invariant observed endpoint.",
    )
    check(
        "scaffold_disjoint_flag",
        bool(
            ~raw.loc[
                raw["protocol"] == "scaffold",
                "scaffold_seen_in_train",
            ].astype(bool).any()
        ),
        "No scaffold-protocol evaluation row reports a seen scaffold.",
    )
    similarity = raw["max_train_tanimoto"].to_numpy(dtype=np.float64)
    check(
        "applicability_similarity_range",
        bool(np.isfinite(similarity).all() and ((0 <= similarity) & (similarity <= 1)).all()),
        f"min={similarity.min():.6g}; max={similarity.max():.6g}",
    )

    expected_average_rows = (
        len(EXPECTED_PROTOCOLS) * len(EXPECTED_MODELS) * N_ROWS
    )
    averaged_key = ["protocol", "model_id", "row_index"]
    check(
        "repeat_average_shape_and_keys",
        (
            len(averaged) == expected_average_rows
            and not averaged.duplicated(averaged_key).any()
            and (averaged["n_oof_predictions"] == 5).all()
            and (averaged["n_repeats"] == 5).all()
        ),
        f"observed={len(averaged)}; expected={expected_average_rows}",
    )

    for protocol in EXPECTED_PROTOCOLS:
        fold_metrics = pd.read_csv(
            report_dir / f"{protocol}_outer_fold_metrics.csv"
        )
        selected = pd.read_csv(
            report_dir / f"{protocol}_selected_hyperparameters.csv"
        )
        tuning = pd.read_csv(
            report_dir / f"{protocol}_inner_tuning_metrics.csv"
        )
        check(
            f"{protocol}_fold_tables",
            (
                len(fold_metrics) == 25 * len(EXPECTED_MODELS)
                and len(selected) == 25 * 6
                and len(tuning) == 25 * 30
                and set(fold_metrics["outer_fold"]) == EXPECTED_FOLDS
                and set(fold_metrics["repeat"]) == EXPECTED_REPEATS
            ),
            (
                f"fold_metrics={len(fold_metrics)}; "
                f"selected={len(selected)}; tuning={len(tuning)}"
            ),
        )
        non_global_tuning = tuning[
            tuning["model_id"] != "global_mean"
        ]
        check(
            f"{protocol}_tuning_metrics_finite",
            bool(
                np.isfinite(
                    non_global_tuning[
                        ["spearman", "rmse", "mae"]
                    ].to_numpy(dtype=np.float64)
                ).all()
            ),
            "All non-constant inner-OOF tuning metrics are finite.",
        )

    check(
        "primary_metric_table",
        (
            len(metrics) == len(EXPECTED_PROTOCOLS) * len(EXPECTED_MODELS)
            and set(metrics["protocol"]) == EXPECTED_PROTOCOLS
            and set(metrics["model_id"]) == EXPECTED_MODELS
        ),
        f"rows={len(metrics)}",
    )
    substantive_metrics = metrics[
        metrics["model_id"] != "global_mean"
    ][["spearman", "pearson", "rmse", "mae", "r2"]]
    check(
        "substantive_metrics_finite",
        bool(
            np.isfinite(
                substantive_metrics.to_numpy(dtype=np.float64)
            ).all()
        ),
        "All non-constant primary metrics are finite.",
    )

    model_bootstrap = bootstrap[bootstrap["record_type"] == "model"]
    contrast_bootstrap = bootstrap[
        bootstrap["record_type"] == "contrast"
    ]
    check(
        "bootstrap_shape",
        (
            len(model_bootstrap)
            == len(EXPECTED_PROTOCOLS) * len(EXPECTED_MODELS)
            and len(contrast_bootstrap) == len(EXPECTED_PROTOCOLS) * 4
            and (bootstrap["n_bootstrap"] == 10_000).all()
        ),
        (
            f"model_rows={len(model_bootstrap)}; "
            f"contrast_rows={len(contrast_bootstrap)}"
        ),
    )
    interval_columns = [
        "delta_spearman_ci_low",
        "delta_spearman_ci_high",
        "delta_rmse_ci_low",
        "delta_rmse_ci_high",
    ]
    check(
        "contrast_intervals_finite_and_ordered",
        bool(
            np.isfinite(
                contrast_bootstrap[interval_columns].to_numpy(dtype=np.float64)
            ).all()
            and (
                contrast_bootstrap["delta_spearman_ci_low"]
                <= contrast_bootstrap["delta_spearman_ci_high"]
            ).all()
            and (
                contrast_bootstrap["delta_rmse_ci_low"]
                <= contrast_bootstrap["delta_rmse_ci_high"]
            ).all()
        ),
        "All paired contrast interval bounds are finite and ordered.",
    )
    check(
        "risk_coverage_shape",
        (
            len(coverage)
            == len(EXPECTED_PROTOCOLS) * len(EXPECTED_MODELS) * 5
            and set(coverage["requested_coverage"])
            == {1.0, 0.8, 0.6, 0.4, 0.2}
            and (coverage["n_selected_unique_smiles"] > 0).all()
            and (coverage["n_selected_scaffolds"] > 0).all()
        ),
        f"rows={len(coverage)}",
    )
    check(
        "context_sanitization_audit",
        len(audit) == 18,
        (
            f"rows={len(audit)}; "
            f"fields={audit.groupby('field').size().to_dict()}"
        ),
    )
    check(
        "manifest_completeness",
        manifest.get("completeness")
        == {
            "raw_cross_fitted_predictions": True,
            "repeat_averaged_predictions": True,
        },
        json.dumps(manifest.get("completeness"), sort_keys=True),
    )

    artifact_rows: list[dict[str, Any]] = []
    qa_output_names = {
        "qa_summary.json",
        "qa_summary.md",
        "artifact_sha256.csv",
    }
    for path in sorted(report_dir.iterdir()):
        if path.is_file() and path.name not in qa_output_names:
            artifact_rows.append(
                {
                    "filename": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    pd.DataFrame(artifact_rows).to_csv(
        report_dir / "artifact_sha256.csv",
        index=False,
    )

    passed = bool(all(item["passed"] for item in checks))
    payload = {
        "qa_passed": passed,
        "n_checks": len(checks),
        "n_failed": sum(not item["passed"] for item in checks),
        "checks": checks,
        "identity_hashes": identity_hashes,
        "artifact_inventory": "artifact_sha256.csv",
    }
    (report_dir / "qa_summary.json").write_text(
        json.dumps(payload, indent=2, default=json_ready) + "\n",
        encoding="utf-8",
    )

    primary = metrics[
        metrics["model_id"] == "full_context_extra_trees"
    ][["protocol", "spearman", "rmse", "calibration_slope"]]
    primary_contrast = contrast_bootstrap[
        contrast_bootstrap["contrast_id"] == "full_vs_context_ridge"
    ][
        [
            "protocol",
            "delta_spearman_observed",
            "delta_spearman_ci_low",
            "delta_spearman_ci_high",
            "delta_rmse_observed",
            "delta_rmse_ci_low",
            "delta_rmse_ci_high",
        ]
    ]
    markdown_lines = [
        "# Confirmatory CPU v1 QA",
        "",
        f"- QA passed: **{passed}**",
        f"- Checks: {len(checks)}; failed: {payload['n_failed']}",
        f"- Rows / unique SMILES / scaffolds: {N_ROWS} / {N_UNIQUE_SMILES} / {N_SCAFFOLDS}",
        f"- Manifest status: `{manifest['status']}`; frozen match: `{manifest['frozen_protocol_match']}`",
        "",
        "## Full-model point estimates",
        "",
        dataframe_to_markdown(primary),
        "",
        "## Primary paired contrast",
        "",
        dataframe_to_markdown(primary_contrast),
        "",
        "## Checks",
        "",
    ]
    for item in checks:
        marker = "PASS" if item["passed"] else "FAIL"
        markdown_lines.append(
            f"- **{marker} — {item['name']}**: {item['detail']}"
        )
    (report_dir / "qa_summary.md").write_text(
        "\n".join(markdown_lines) + "\n",
        encoding="utf-8",
    )
    print(
        f"QA {'PASS' if passed else 'FAIL'}: "
        f"{len(checks) - payload['n_failed']}/{len(checks)} checks"
    )
    if not passed:
        failed_names = [
            item["name"] for item in checks if not item["passed"]
        ]
        raise SystemExit(f"Failed checks: {failed_names}")


if __name__ == "__main__":
    main()
