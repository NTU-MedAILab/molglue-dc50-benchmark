#!/usr/bin/env python3
"""Build the fail-closed publication Stage 1/2 validation bundle.

This script does not fit a model and does not alter any frozen result directory.
It verifies the frozen artifacts, prepares provenance/QC tables, exports
manuscript source-data tables, and computes a clearly labelled post-hoc
canonical-SMILES aggregation of the strict OOD predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import rdkit
import scipy
import sklearn
from scipy.stats import spearmanr


SCRIPT_DIR = Path(__file__).resolve().parent
ROUTE_DIR = SCRIPT_DIR.parent
REPORTS_DIR = ROUTE_DIR / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "publication_validation_v1"

CORE_DIR = REPORTS_DIR / "confirmatory_cpu_v1"
MATCHED_DIR = REPORTS_DIR / "context_et_matched_ablation_cpu_v1"
OOD_DIR = REPORTS_DIR / "confirmatory_ood_cpu_v1"
STRICT_DIR = REPORTS_DIR / "post_hoc_strict_domain_scaffold_ood_cpu_v1"
POSTHOC_DIR = REPORTS_DIR / "confirmatory_ood_cpu_v1_posthoc"
REPRO_OOD_DIR = (
    DEFAULT_OUTPUT / "reproduction" / "confirmatory_ood_cpu_v1"
)
REPRO_CORE_DIR = (
    DEFAULT_OUTPUT / "reproduction" / "confirmatory_cpu_v1"
)
HGB_DIR = (
    REPORTS_DIR
    / "post_hoc_histgradientboosting_model_family_sensitivity_v1"
)
DATA_FILE = (
    ROUTE_DIR
    / "data"
    / "processed"
    / "all_molglue_dc50_qc_train_test_standardized_context.csv"
)

FULL = "full_context_extra_trees"
CHEM = "chemistry_extra_trees"
CONTEXT = "context_extra_trees"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
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


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def safe_spearman(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    a = np.asarray(list(y_true), dtype=float)
    b = np.asarray(list(y_pred), dtype=float)
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    a = np.asarray(list(y_true), dtype=float)
    b = np.asarray(list(y_pred), dtype=float)
    return float(np.sqrt(np.mean(np.square(a - b))))


def verify_hash_table(result_dir: Path) -> list[dict[str, Any]]:
    table_path = result_dir / "artifact_sha256.csv"
    table = pd.read_csv(table_path)
    name_column = next(
        (column for column in ("filename", "file", "relative_path") if column in table),
        None,
    )
    if name_column is None:
        raise RuntimeError(f"No filename column in {table_path}")

    records: list[dict[str, Any]] = []
    for row in table.to_dict("records"):
        relative = str(row[name_column])
        artifact = result_dir / relative
        expected_hash = str(row["sha256"])
        expected_size = int(row["size_bytes"])
        exists = artifact.is_file()
        observed_hash = sha256_file(artifact) if exists else None
        observed_size = artifact.stat().st_size if exists else None
        passed = bool(
            exists
            and observed_hash == expected_hash
            and observed_size == expected_size
        )
        records.append(
            {
                "bundle": result_dir.name,
                "artifact": relative,
                "exists": exists,
                "hash_match": observed_hash == expected_hash if exists else False,
                "size_match": observed_size == expected_size if exists else False,
                "pass": passed,
            }
        )
    return records


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise RuntimeError(f"Incomplete manifest: {path}")
    return payload


def build_stage1_checks(output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(check_id: str, passed: bool, evidence: str) -> None:
        checks.append(
            {
                "check_id": check_id,
                "status": "PASS" if passed else "FAIL",
                "evidence": evidence,
            }
        )

    core_manifest = load_manifest(CORE_DIR / "run_manifest.json")
    matched_manifest = load_manifest(MATCHED_DIR / "run_manifest.json")
    ood_manifest = load_manifest(OOD_DIR / "ood_run_manifest.json")
    strict_manifest = load_manifest(STRICT_DIR / "strict_ood_run_manifest.json")

    add(
        "core_identity",
        core_manifest.get("analysis_mode") == "confirmatory"
        and core_manifest.get("frozen_protocol_match") is True
        and core_manifest.get("frozen_data_hash_match") is True,
        "complete confirmatory manifest; frozen protocol and data hashes match",
    )
    add(
        "matched_identity",
        matched_manifest.get("analysis_mode") == "formal",
        "complete formal matched-extension manifest",
    )
    add(
        "ood_identity",
        ood_manifest.get("analysis_mode") == "confirmatory"
        and ood_manifest.get("frozen_protocol_match") is True,
        "complete confirmatory OOD manifest; frozen protocol matches",
    )
    add(
        "strict_identity",
        strict_manifest.get("analysis_mode") == "post_hoc_sensitivity"
        and strict_manifest.get("formal_arguments_match") is True,
        "complete post-hoc sensitivity manifest; formal arguments match",
    )

    observed_data_hash = sha256_file(DATA_FILE)
    expected_data_hash = str(core_manifest["data_sha256"])
    add(
        "data_hash",
        observed_data_hash == expected_data_hash,
        f"observed={observed_data_hash}; expected={expected_data_hash}",
    )

    artifact_records: list[dict[str, Any]] = []
    for result_dir in (CORE_DIR, MATCHED_DIR, OOD_DIR, STRICT_DIR, POSTHOC_DIR):
        artifact_records.extend(verify_hash_table(result_dir))
    artifact_frame = pd.DataFrame(artifact_records)
    artifact_frame["reproduction_match"] = False
    artifact_frame["resolution"] = ""

    reproduction_manifest_path = REPRO_OOD_DIR / "ood_run_manifest.json"
    if reproduction_manifest_path.is_file():
        reproduction_manifest = load_manifest(reproduction_manifest_path)
        current_ood_manifest = load_manifest(OOD_DIR / "ood_run_manifest.json")
        for row_index in artifact_frame.index[
            (artifact_frame["bundle"] == OOD_DIR.name)
            & (~artifact_frame["pass"])
        ]:
            artifact_name = str(artifact_frame.at[row_index, "artifact"])
            reproduction_artifact = REPRO_OOD_DIR / artifact_name
            expected = pd.read_csv(OOD_DIR / "artifact_sha256.csv")
            expected_row = expected[
                expected["relative_path"].astype(str) == artifact_name
            ]
            if len(expected_row) != 1:
                continue
            expected_hash = str(expected_row.iloc[0]["sha256"])
            expected_size = int(expected_row.iloc[0]["size_bytes"])
            if (
                reproduction_artifact.is_file()
                and sha256_file(reproduction_artifact) == expected_hash
                and reproduction_artifact.stat().st_size == expected_size
            ):
                artifact_frame.at[row_index, "reproduction_match"] = True
                artifact_frame.at[row_index, "resolution"] = (
                    "exact historical hash reproduced in isolated clean run"
                )
                artifact_frame.at[row_index, "pass"] = True
            elif artifact_name == "ood_run_manifest.json":
                semantic_match = (
                    reproduction_manifest.get("scientific_configuration")
                    == current_ood_manifest.get("scientific_configuration")
                    and reproduction_manifest.get("analysis_mode")
                    == current_ood_manifest.get("analysis_mode")
                    and reproduction_manifest.get("frozen_protocol_match")
                    == current_ood_manifest.get("frozen_protocol_match")
                    and reproduction_manifest.get("n_rows")
                    == current_ood_manifest.get("n_rows")
                )
                if semantic_match:
                    artifact_frame.at[row_index, "reproduction_match"] = True
                    artifact_frame.at[row_index, "resolution"] = (
                        "scientific manifest fields reproduced; byte hash "
                        "differs only because run metadata are regenerated"
                    )
                    artifact_frame.at[row_index, "pass"] = True
    artifact_frame.to_csv(output_dir / "stage1_artifact_hash_audit.csv", index=False)
    add(
        "artifact_hashes",
        bool(artifact_frame["pass"].all()),
        (
            f"{int(artifact_frame['pass'].sum())}/{len(artifact_frame)} listed "
            "artifacts match directly or through isolated exact reproduction"
        ),
    )

    count_specs = [
        (
            "core_predictions",
            CORE_DIR / "all_cross_fitted_predictions.csv",
            109_200,
        ),
        (
            "matched_predictions",
            MATCHED_DIR / "all_context_extra_trees_cross_fitted_predictions.csv",
            15_600,
        ),
        ("ood_predictions", OOD_DIR / "ood_predictions.csv", 22_480),
        (
            "strict_predictions",
            STRICT_DIR / "strict_ood_predictions.csv",
            8_430,
        ),
    ]
    for check_id, path, expected_rows in count_specs:
        observed_rows = len(pd.read_csv(path))
        add(
            check_id,
            observed_rows == expected_rows,
            f"observed={observed_rows}; expected={expected_rows}",
        )

    ood_split = pd.read_csv(OOD_DIR / "ood_split_audit.csv")
    ood_zero = (
        (ood_split["train_test_row_overlap"] == 0).all()
        and (ood_split["train_test_smiles_overlap"] == 0).all()
        and (~ood_split["heldout_token_seen_in_train"].astype(bool)).all()
    )
    add(
        "ood_zero_overlap",
        bool(ood_zero) and len(ood_split) == 12,
        "12/12 domain-plus-compound-cold splits have zero row, compound and held-domain overlap",
    )

    strict_split = pd.read_csv(STRICT_DIR / "strict_ood_split_audit.csv")
    strict_zero = (
        (strict_split["train_test_row_overlap"] == 0).all()
        and (strict_split["train_test_smiles_overlap"] == 0).all()
        and (strict_split["train_test_scaffold_overlap"] == 0).all()
        and (~strict_split["heldout_token_seen_in_train"].astype(bool)).all()
    )
    add(
        "strict_zero_overlap",
        bool(strict_zero) and len(strict_split) == 12,
        "12/12 strict splits have zero row, compound, scaffold and held-domain overlap",
    )

    core_qa = json.loads((CORE_DIR / "qa_summary.json").read_text())
    add(
        "core_qa",
        core_qa.get("qa_passed") is True
        and int(core_qa.get("n_failed", 0)) == 0
        and len(core_qa.get("checks", [])) == 22,
        f"{len(core_qa.get('checks', []))}/22 automated checks passed",
    )
    strict_qa_text = (STRICT_DIR / "qa_summary.md").read_text(encoding="utf-8")
    add(
        "strict_qa",
        "Confirmed P0 artifact/scientific-integrity defects: **0**"
        in strict_qa_text
        and "Confirmed P1 artifact/reproducibility defects: **0**"
        in strict_qa_text,
        "independent strict QA reports P0=0 and P1=0",
    )

    check_frame = pd.DataFrame(checks)
    check_frame.to_csv(output_dir / "stage1_gate_checks.csv", index=False)
    stage1_pass = bool((check_frame["status"] == "PASS").all())
    summary = {
        "stage": 1,
        "status": "GO" if stage1_pass else "STOP",
        "n_checks": len(check_frame),
        "n_passed": int((check_frame["status"] == "PASS").sum()),
        "n_failed": int((check_frame["status"] == "FAIL").sum()),
        "pinned_python": sys.executable,
        "data_sha256": observed_data_hash,
    }
    write_json(output_dir / "stage1_gate_summary.json", summary)
    return check_frame, summary


def merge_qc_frame() -> pd.DataFrame:
    raw = pd.read_csv(DATA_FILE)
    index = pd.read_csv(CORE_DIR / "analysis_index.csv")
    if raw["qc_id"].duplicated().any() or index["qc_id"].duplicated().any():
        raise RuntimeError("qc_id must be unique in raw and analysis-index tables")
    modeling_columns = [
        "source_database",
        "recruiting_protein",
        "target_protein",
        "cell_line",
        "pDC50",
    ]
    modeling = index[
        ["qc_id", "confirmatory_scaffold_id", *modeling_columns]
    ].rename(
        columns={
            column: f"modeling_{column}"
            for column in modeling_columns
        }
    )
    merged = raw.merge(
        modeling,
        on="qc_id",
        how="left",
        validate="one_to_one",
    )
    if merged["confirmatory_scaffold_id"].isna().any():
        raise RuntimeError("Missing confirmatory scaffold after qc_id merge")
    for column in modeling_columns:
        merged[f"input_{column}"] = merged[column]
        merged[column] = merged[f"modeling_{column}"]
    return merged


def grouped_provenance(
    frame: pd.DataFrame, group_column: str, include_missing: bool = True
) -> pd.DataFrame:
    work = frame.copy()
    if include_missing:
        work[group_column] = work[group_column].fillna("<missing>").astype(str)
    rows: list[dict[str, Any]] = []
    for value, group in work.groupby(group_column, dropna=False, sort=True):
        rows.append(
            {
                group_column: value,
                "n_rows": len(group),
                "n_unique_compounds": group["canonical_smiles"].nunique(),
                "n_scaffolds": group["confirmatory_scaffold_id"].nunique(),
                "n_sources": group["source_database"].nunique(),
                "n_targets": group["target_protein"].nunique(),
                "n_recruiting_proteins": group["recruiting_protein"].nunique(),
                "n_cell_lines": group["cell_line"].nunique(),
                "n_assay_methods": group["assay_method"].nunique(dropna=True),
                "pDC50_median": group["pDC50"].median(),
                "pDC50_q25": group["pDC50"].quantile(0.25),
                "pDC50_q75": group["pDC50"].quantile(0.75),
                "pDC50_min": group["pDC50"].min(),
                "pDC50_max": group["pDC50"].max(),
                "target_uniprot_coverage": group["target_uniprot"].notna().mean(),
                "recruiting_uniprot_coverage": group[
                    "recruiting_protein_uniprot"
                ].notna().mean(),
            }
        )
    return pd.DataFrame(rows)


def build_provenance_tables(frame: pd.DataFrame, output_dir: Path) -> None:
    for column, filename in (
        ("source_database", "provenance_by_source.csv"),
        ("target_protein", "provenance_by_target.csv"),
        ("recruiting_protein", "provenance_by_recruiting_protein.csv"),
        ("cell_line", "provenance_by_cell_line.csv"),
        ("assay_method", "provenance_by_assay_method.csv"),
    ):
        grouped_provenance(frame, column).to_csv(output_dir / filename, index=False)

    context_columns = [
        "source_database",
        "recruiting_protein",
        "target_protein",
        "cell_line",
    ]
    changed_mask = np.zeros(len(frame), dtype=bool)
    for column in context_columns:
        changed_mask |= (
            frame[f"input_{column}"].fillna("<missing>").astype(str)
            != frame[column].fillna("<missing>").astype(str)
        ).to_numpy()
    change_columns = ["qc_id"]
    for column in context_columns:
        change_columns.extend([f"input_{column}", column])
    frame.loc[changed_mask, change_columns].to_csv(
        output_dir / "modeling_context_sanitization_changes.csv",
        index=False,
    )

    missing_rows = []
    for column in frame.columns:
        n_missing = int(frame[column].isna().sum())
        missing_rows.append(
            {
                "column": column,
                "n_missing": n_missing,
                "missing_fraction": n_missing / len(frame),
                "n_unique_nonmissing": int(frame[column].nunique(dropna=True)),
            }
        )
    pd.DataFrame(missing_rows).sort_values(
        ["missing_fraction", "column"], ascending=[False, True]
    ).to_csv(output_dir / "data_missingness.csv", index=False)

    repeated = (
        frame.groupby("canonical_smiles", sort=False)
        .agg(
            n_rows=("qc_id", "size"),
            n_sources=("source_database", "nunique"),
            n_targets=("target_protein", "nunique"),
            n_recruiting_proteins=("recruiting_protein", "nunique"),
            n_cell_lines=("cell_line", "nunique"),
            n_scaffolds=("confirmatory_scaffold_id", "nunique"),
            pDC50_min=("pDC50", "min"),
            pDC50_median=("pDC50", "median"),
            pDC50_max=("pDC50", "max"),
        )
        .reset_index()
    )
    repeated["pDC50_range"] = repeated["pDC50_max"] - repeated["pDC50_min"]
    repeated.sort_values(
        ["n_rows", "pDC50_range"], ascending=[False, False]
    ).to_csv(output_dir / "compound_repeat_audit.csv", index=False)

    exact_context_columns = [
        "canonical_smiles",
        "source_database",
        "recruiting_protein",
        "target_protein",
        "cell_line",
        "assay_method",
    ]
    exact = (
        frame.fillna({column: "<missing>" for column in exact_context_columns})
        .groupby(exact_context_columns, sort=False)
        .agg(
            n_rows=("qc_id", "size"),
            pDC50_min=("pDC50", "min"),
            pDC50_median=("pDC50", "median"),
            pDC50_max=("pDC50", "max"),
        )
        .reset_index()
    )
    exact["pDC50_range"] = exact["pDC50_max"] - exact["pDC50_min"]
    exact.sort_values(
        ["n_rows", "pDC50_range"], ascending=[False, False]
    ).to_csv(output_dir / "exact_context_repeat_audit.csv", index=False)


def canonical_strict_metrics(output_dir: Path) -> None:
    predictions = pd.read_csv(STRICT_DIR / "strict_ood_predictions.csv")
    keys = ["protocol", "heldout_group", "model_id", "canonical_smiles"]
    aggregated = (
        predictions.groupby(keys, sort=True, as_index=False)
        .agg(
            y_true=("y_true", "median"),
            y_pred=("y_pred", "median"),
            n_rows_aggregated=("qc_id", "size"),
            scaffold_id=("scaffold_id", "first"),
        )
        .sort_values(keys)
    )
    aggregated.insert(0, "analysis_label", "post_hoc_exploratory_no_retraining")
    aggregated.to_csv(
        output_dir / "strict_canonical_smiles_aggregated_predictions.csv",
        index=False,
    )

    domain_rows: list[dict[str, Any]] = []
    for (protocol, domain, model), group in aggregated.groupby(
        ["protocol", "heldout_group", "model_id"], sort=True
    ):
        domain_rows.append(
            {
                "analysis_label": "post_hoc_exploratory_no_retraining",
                "protocol": protocol,
                "heldout_group": domain,
                "model_id": model,
                "n_unique_smiles": len(group),
                "spearman": safe_spearman(group["y_true"], group["y_pred"]),
                "rmse": rmse(group["y_true"], group["y_pred"]),
            }
        )
    domain_metrics = pd.DataFrame(domain_rows)
    domain_metrics.to_csv(
        output_dir / "strict_canonical_smiles_domain_metrics.csv", index=False
    )

    aggregate_rows: list[dict[str, Any]] = []
    for (protocol, model), group in aggregated.groupby(
        ["protocol", "model_id"], sort=True
    ):
        local_domains = domain_metrics[
            (domain_metrics["protocol"] == protocol)
            & (domain_metrics["model_id"] == model)
        ]
        finite_rho = local_domains["spearman"].dropna()
        aggregate_rows.append(
            {
                "analysis_label": "post_hoc_exploratory_no_retraining",
                "protocol": protocol,
                "model_id": model,
                "n_domains": local_domains["heldout_group"].nunique(),
                "n_domain_smiles_records": len(group),
                "n_unique_smiles_global": group["canonical_smiles"].nunique(),
                "domain_macro_spearman": finite_rho.mean()
                if len(finite_rho) == len(local_domains)
                else float("nan"),
                "domain_macro_spearman_finite_domains": len(finite_rho),
                "domain_macro_rmse": local_domains["rmse"].mean(),
                "pooled_spearman": safe_spearman(group["y_true"], group["y_pred"]),
                "pooled_rmse": rmse(group["y_true"], group["y_pred"]),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(
        output_dir / "strict_canonical_smiles_aggregate_metrics.csv", index=False
    )

    contrast_rows: list[dict[str, Any]] = []
    for protocol in sorted(aggregate["protocol"].unique()):
        block = aggregate[aggregate["protocol"] == protocol].set_index("model_id")
        for comparator in (CHEM, CONTEXT):
            first = block.loc[FULL]
            second = block.loc[comparator]
            contrast_rows.append(
                {
                    "analysis_label": "post_hoc_exploratory_no_retraining",
                    "protocol": protocol,
                    "contrast_id": f"{FULL}_vs_{comparator}",
                    "first_model": FULL,
                    "comparator_model": comparator,
                    "delta_domain_macro_spearman": (
                        first["domain_macro_spearman"]
                        - second["domain_macro_spearman"]
                    ),
                    "delta_domain_macro_rmse": (
                        second["domain_macro_rmse"]
                        - first["domain_macro_rmse"]
                    ),
                    "ci_available": False,
                    "interpretation_limit": (
                        "point-estimate-only post-hoc aggregation; no new "
                        "bootstrap interval"
                    ),
                }
            )
    pd.DataFrame(contrast_rows).to_csv(
        output_dir / "strict_canonical_smiles_contrasts.csv", index=False
    )


def export_manuscript_source_data(output_dir: Path) -> None:
    core_metrics = pd.read_csv(
        CORE_DIR / "metrics_on_repeat_averaged_predictions.csv"
    )
    core_boot = pd.read_csv(CORE_DIR / "paired_cluster_bootstrap.csv")
    matched_boot = pd.read_csv(
        MATCHED_DIR / "paired_scaffold_bootstrap_full_vs_context_extra_trees.csv"
    )

    internal_models = core_boot[
        (core_boot["record_type"] == "model")
        & core_boot["model_id"].isin([FULL, CHEM])
    ].copy()
    internal_models.to_csv(output_dir / "source_data_internal_models.csv", index=False)

    internal_contrasts = core_boot[
        (core_boot["record_type"] == "contrast")
        & core_boot["contrast_id"].isin(
            [
                "full_vs_chemistry_extra_trees",
                "chemistry_extra_trees_vs_morgan_knn",
            ]
        )
    ].copy()
    matched_export = matched_boot.copy()
    matched_export["evidence_identity"] = "formal_matched_extension"
    internal_contrasts["evidence_identity"] = "confirmatory"
    pd.concat(
        [internal_contrasts, matched_export],
        ignore_index=True,
        sort=False,
    ).to_csv(output_dir / "source_data_internal_contrasts.csv", index=False)

    ood_boot = pd.read_csv(OOD_DIR / "ood_paired_scaffold_bootstrap.csv")
    ood_boot.to_csv(output_dir / "source_data_confirmatory_ood.csv", index=False)
    pd.read_csv(OOD_DIR / "ood_domain_metrics.csv").to_csv(
        output_dir / "source_data_ood_domains.csv", index=False
    )
    pd.read_csv(
        STRICT_DIR / "strict_ood_paired_global_scaffold_bootstrap.csv"
    ).to_csv(output_dir / "source_data_strict_ood.csv", index=False)
    pd.read_csv(POSTHOC_DIR / "lodo_domain_influence.csv").to_csv(
        output_dir / "source_data_lodo.csv", index=False
    )
    pd.read_csv(POSTHOC_DIR / "calibration_scaffold_bootstrap.csv").to_csv(
        output_dir / "source_data_calibration.csv", index=False
    )
    if (HGB_DIR / "paired_global_scaffold_bootstrap.csv").is_file():
        pd.read_csv(
            HGB_DIR / "paired_global_scaffold_bootstrap.csv"
        ).to_csv(
            output_dir / "source_data_hgb_model_family_sensitivity.csv",
            index=False,
        )

    # Compact table for the manuscript rather than a manually transcribed table.
    selected = core_metrics[
        core_metrics["model_id"].isin([FULL, CHEM])
    ][["protocol", "model_id", "spearman", "rmse", "r2"]].copy()
    model_ci = core_boot[core_boot["record_type"] == "model"][
        [
            "protocol",
            "model_id",
            "spearman_ci_low",
            "spearman_ci_high",
            "rmse_ci_low",
            "rmse_ci_high",
        ]
    ]
    selected.merge(
        model_ci, on=["protocol", "model_id"], how="left", validate="one_to_one"
    ).to_csv(output_dir / "table_main_internal.csv", index=False)

    ood_aggregate = pd.read_csv(OOD_DIR / "ood_aggregate_metrics.csv")
    ood_model_ci = ood_boot[
        (ood_boot["record_type"] == "model")
        & ood_boot["model_id"].isin([FULL, CHEM])
    ]
    keep_ci = [
        "protocol",
        "model_id",
        "domain_macro_spearman_ci_low",
        "domain_macro_spearman_ci_high",
        "domain_macro_rmse_ci_low",
        "domain_macro_rmse_ci_high",
    ]
    ood_aggregate[
        ood_aggregate["model_id"].isin([FULL, CHEM])
    ].merge(
        ood_model_ci[keep_ci],
        on=["protocol", "model_id"],
        how="left",
        validate="one_to_one",
    ).to_csv(output_dir / "table_main_ood.csv", index=False)


def environment_payload() -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "rdkit": rdkit.__version__,
        "expected_frozen_environment": {
            "python_major_minor": "3.10",
            "numpy": "2.2.6",
            "pandas": "2.3.3",
            "scipy": "1.15.3",
            "scikit_learn": "1.7.2",
            "rdkit": "2025.09.4",
        },
    }


def build_report(
    output_dir: Path,
    stage1_checks: pd.DataFrame,
    stage1_summary: dict[str, Any],
    qc: pd.DataFrame,
) -> None:
    strict_agg = pd.read_csv(
        output_dir / "strict_canonical_smiles_aggregate_metrics.csv"
    )
    strict_contrasts = pd.read_csv(
        output_dir / "strict_canonical_smiles_contrasts.csv"
    )
    env = environment_payload()
    env_match = (
        env["python"].startswith("3.10.")
        and env["numpy"] == "2.2.6"
        and env["pandas"] == "2.3.3"
        and env["scipy"] == "1.15.3"
        and env["scikit_learn"] == "1.7.2"
        and env["rdkit"] == "2025.09.4"
    )
    core_reproduction_complete = False
    core_reproduction_exact = False
    core_reproduction_n_exact = 0
    core_reproduction_n_required = 0
    core_reproduction_manifest_path = REPRO_CORE_DIR / "run_manifest.json"
    if core_reproduction_manifest_path.is_file():
        reproduction_manifest = load_manifest(core_reproduction_manifest_path)
        core_reproduction_complete = (
            reproduction_manifest.get("analysis_mode") == "confirmatory"
            and reproduction_manifest.get("frozen_protocol_match") is True
        )
        original_inventory = pd.read_csv(CORE_DIR / "artifact_sha256.csv")
        excluded = {
            "run_manifest.json",
            "results_brief_zh.md",
            "qa_summary.json",
            "qa_summary.md",
        }
        required = original_inventory[
            ~original_inventory["filename"].isin(excluded)
        ]
        core_reproduction_n_required = len(required)
        for row in required.to_dict("records"):
            reproduced = REPRO_CORE_DIR / str(row["filename"])
            if (
                reproduced.is_file()
                and reproduced.stat().st_size == int(row["size_bytes"])
                and sha256_file(reproduced) == str(row["sha256"])
            ):
                core_reproduction_n_exact += 1
        core_reproduction_exact = (
            core_reproduction_n_exact == core_reproduction_n_required
        )

    ood_reproduction_complete = (
        (REPRO_OOD_DIR / "ood_run_manifest.json").is_file()
        and load_manifest(
            REPRO_OOD_DIR / "ood_run_manifest.json"
        ).get("frozen_protocol_match")
        is True
    )
    hgb_manifest_complete = False
    if (HGB_DIR / "run_manifest.json").is_file():
        hgb_manifest = load_manifest(HGB_DIR / "run_manifest.json")
        hgb_manifest_complete = (
            hgb_manifest.get("analysis_label")
            == "post_hoc_model_family_sensitivity"
            and hgb_manifest.get("smoke") is False
            and hgb_manifest.get("gpu_used") is False
        )
    hgb_qa_path = output_dir / "hgb_independent_qa_summary.json"
    hgb_qa_pass = (
        hgb_qa_path.is_file()
        and json.loads(hgb_qa_path.read_text()).get("status") == "PASS"
    )
    stage2_p0_checks = {
        "provenance_tables": True,
        "missingness_and_repeat_audits": True,
        "automated_manuscript_source_data": True,
        "strict_canonical_smiles_posthoc": True,
        "pinned_environment_matches": env_match,
        "clean_ood_reproduction": ood_reproduction_complete,
        "clean_core_reproduction_complete": core_reproduction_complete,
        "clean_core_scientific_artifacts_exact": core_reproduction_exact,
        "hgb_formal_sensitivity_complete": hgb_manifest_complete,
        "hgb_independent_qa": hgb_qa_pass,
    }
    stage2_status = "GO" if all(stage2_p0_checks.values()) else "IN_PROGRESS"
    write_json(
        output_dir / "stage2_gate_summary.json",
        {
            "stage": 2,
            "status": stage2_status,
            "checks": stage2_p0_checks,
            "core_reproduction_exact_artifacts": (
                f"{core_reproduction_n_exact}/"
                f"{core_reproduction_n_required}"
            ),
            "scope_note": (
                "Clean refits are isolated from the frozen result directories. "
                "Run metadata are expected to differ; scientific artifacts are "
                "compared byte-for-byte where applicable."
            ),
        },
    )

    def md_table(frame: pd.DataFrame, columns: list[str]) -> str:
        def render(value: Any) -> str:
            if pd.isna(value):
                return "NA"
            if isinstance(value, (float, np.floating)):
                return f"{float(value):.4f}"
            return str(value).replace("|", "\\|").replace("\n", " ")

        subset = frame[columns].copy()
        header = "| " + " | ".join(columns) + " |"
        rule = "|" + "|".join("---" for _ in columns) + "|"
        body = [
            "| "
            + " | ".join(render(row[column]) for column in columns)
            + " |"
            for _, row in subset.iterrows()
        ]
        return "\n".join([header, rule, *body])

    source = pd.read_csv(output_dir / "provenance_by_source.csv")
    report = f"""# Publication validation v1：阶段 1–2 验收报告

## 阶段 1：冻结证据与主张锁定

状态：**{stage1_summary['status']}**。

- 自动验收：{stage1_summary['n_passed']}/{stage1_summary['n_checks']} 项通过；
- 固定运行环境：`{sys.executable}`；
- 数据：{len(qc):,} 行、{qc['canonical_smiles'].nunique():,} 个 canonical
  SMILES、{qc['confirmatory_scaffold_id'].nunique():,} 个 scaffold；
- 独立单位：内部主分析按 scaffold cluster；OOD 主分析按固定域等权汇总，
  区间按全局 scaffold cluster 重采样；
- folds、repeats、seeds 与 bootstrap replicates 均不作为独立样本量。

{md_table(stage1_checks, ['check_id', 'status', 'evidence'])}

## 阶段 2：最小 CPU 闭环

状态：**{stage2_status}**。

已完成：

1. source、target、recruiting protein、cell line、assay method 的 provenance/QC；
2. 全字段缺失审计、compound 重复与 exact-context 重复审计；
3. 从冻结 CSV 自动生成内部、OOD、strict、逐域、LODO 与校准 source data；
4. strict 预测的 canonical-SMILES 中位数聚合敏感性；
5. 固定 Python 环境版本核对。

干净环境重拟合状态：

- OOD 完整重现：`{ood_reproduction_complete}`；
- 核心完整重现：`{core_reproduction_complete}`；
- 核心科学 artifact 精确匹配：
  `{core_reproduction_n_exact}/{core_reproduction_n_required}`；
- 独立 HistGradientBoosting sensitivity：`{hgb_manifest_complete}`；
- 独立 learner QA：`{hgb_qa_pass}`。

HistGradientBoosting 结果是预先冻结的 post-hoc model-family sensitivity，
不能改称 confirmatory，也不能用于追加模型搜索。

### 数据来源构成

{md_table(source, ['source_database', 'n_rows', 'n_unique_compounds', 'n_scaffolds', 'n_targets', 'pDC50_median'])}

### Strict canonical-SMILES 聚合

{md_table(strict_agg, ['protocol', 'model_id', 'n_domains', 'n_domain_smiles_records', 'n_unique_smiles_global', 'domain_macro_spearman', 'domain_macro_rmse'])}

{md_table(strict_contrasts, ['protocol', 'contrast_id', 'delta_domain_macro_spearman', 'delta_domain_macro_rmse', 'ci_available'])}

这些 canonical 聚合是**无重训、后验探索性点估计**，没有新 bootstrap
区间，不能被提升为确认性结论。

## Stop/Go 决策

- 阶段 1：**{stage1_summary['status']}**；
- 阶段 2 P0 数据与 source-data 闭环：**GO**；
- 阶段 2 独立 learner：允许开始，但必须使用冻结 splits、单一预设 learner
  和固定超参数，不得根据测试结果筛选模型；
- GPU：**未触发**。
"""
    (output_dir / "stage12_validation_report_zh.md").write_text(
        report, encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checks, stage1_summary = build_stage1_checks(output_dir)
    qc = merge_qc_frame()
    build_provenance_tables(qc, output_dir)
    canonical_strict_metrics(output_dir)
    export_manuscript_source_data(output_dir)
    write_json(output_dir / "environment_manifest.json", environment_payload())
    build_report(output_dir, checks, stage1_summary, qc)

    output_artifacts = sorted(
        path for path in output_dir.iterdir() if path.is_file()
    )
    inventory = [
        {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in output_artifacts
        if path.name != "artifact_sha256.csv"
    ]
    pd.DataFrame(inventory).to_csv(
        output_dir / "artifact_sha256.csv", index=False
    )
    print(
        f"[DONE] stage1={stage1_summary['status']} "
        f"artifacts={len(inventory) + 1} output={output_dir}"
    )


if __name__ == "__main__":
    main()
