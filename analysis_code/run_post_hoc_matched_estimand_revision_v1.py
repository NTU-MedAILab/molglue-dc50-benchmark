#!/usr/bin/env python
"""Matched-estimand post-hoc analysis for reviewer-driven revision.

The script compares internal scaffold-disjoint and held-domain predictions on
identical rows, domains, weights and paired global-scaffold resamples. It does
not fit models and does not read prospective or collaborator-restricted data.
"""

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
PROJECT_DIR = SCRIPT_DIR.parent
INTERNAL_FILE = (
    PROJECT_DIR
    / "reports"
    / "confirmatory_cpu_v1"
    / "repeat_averaged_cross_fitted_predictions.csv"
)
OOD_FILE = (
    PROJECT_DIR
    / "reports"
    / "confirmatory_ood_cpu_v1"
    / "ood_predictions.csv"
)
PROTOCOL_FILE = (
    PROJECT_DIR / "docs" / "post_hoc_reviewer_revision_v1_protocol.md"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "reports" / "post_hoc_reviewer_revision_v1"

MODELS = ("chemistry_extra_trees", "full_context_extra_trees")
DOMAINS = {
    "source": ("MGTbind", "MGDB", "MolGlueDB", "TPDdb"),
    "target": (
        "VAV1",
        "CSNK1A1",
        "GSPT1",
        "WIZ",
        "CDK2",
        "CCNK+CDK12",
        "IKZF2",
        "IKZF1",
    ),
}
EXPECTED = {
    "source": {"n_rows": 1520, "n_compounds": 1132, "n_scaffolds": 667},
    "target": {"n_rows": 1290, "n_compounds": 1032, "n_scaffolds": 601},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        parser.error("bootstrap-replicates must be >= 100")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_spearman(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) < 3 or np.unique(y).size < 2 or np.unique(pred).size < 2:
        return float("nan")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(pred)):
        return float("nan")
    return float(spearmanr(y, pred).statistic)


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y - pred))))


def percentile(values: list[float]) -> tuple[float, float, float, int]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return (
        float(np.mean(finite)),
        float(np.percentile(finite, 2.5)),
        float(np.percentile(finite, 97.5)),
        int(len(finite)),
    )


def metric_bundle(
    y: np.ndarray,
    pred: np.ndarray,
    domain: np.ndarray,
    positions: np.ndarray,
    domain_order: tuple[str, ...],
) -> dict[str, Any]:
    yy = y[positions]
    pp = pred[positions]
    dd = domain[positions]
    per_domain: dict[str, dict[str, float]] = {}
    domain_spearman: list[float] = []
    domain_rmse: list[float] = []
    for name in domain_order:
        local = np.where(dd == name)[0]
        if len(local) == 0:
            rho = float("nan")
            err = float("nan")
        else:
            rho = safe_spearman(yy[local], pp[local])
            err = rmse(yy[local], pp[local])
        per_domain[name] = {"spearman": rho, "rmse": err}
        domain_spearman.append(rho)
        domain_rmse.append(err)
    macro_rho = (
        float(np.mean(domain_spearman))
        if np.all(np.isfinite(domain_spearman))
        else float("nan")
    )
    macro_rmse = (
        float(np.mean(domain_rmse))
        if np.all(np.isfinite(domain_rmse))
        else float("nan")
    )
    return {
        "domain_macro_spearman": macro_rho,
        "domain_macro_rmse": macro_rmse,
        "pooled_spearman": safe_spearman(yy, pp),
        "pooled_rmse": rmse(yy, pp),
        "per_domain": per_domain,
    }


def aligned_axis_frame(
    internal: pd.DataFrame,
    ood: pd.DataFrame,
    axis: str,
) -> pd.DataFrame:
    domain_col = "source_database" if axis == "source" else "target_protein"
    protocol = "source_ood" if axis == "source" else "target_ood"
    domain_order = DOMAINS[axis]

    internal_local = internal[
        (internal["protocol"] == "scaffold")
        & internal["model_id"].isin(MODELS)
        & internal[domain_col].astype(str).isin(domain_order)
    ].copy()
    ood_local = ood[
        (ood["protocol"] == protocol)
        & ood["model_id"].isin(MODELS)
        & ood["heldout_group"].astype(str).isin(domain_order)
    ].copy()

    keys = ["row_index"]
    identity_cols = [
        "row_index",
        "qc_id",
        "y_true",
        "canonical_smiles",
        "scaffold_id",
        "source_database",
        "target_protein",
    ]
    reference = (
        internal_local[
            internal_local["model_id"] == "chemistry_extra_trees"
        ][identity_cols]
        .drop_duplicates("row_index")
        .sort_values("row_index")
    )
    if reference["row_index"].duplicated().any():
        raise RuntimeError("Internal reference rows are duplicated")
    merged = reference.copy()
    merged["domain"] = merged[domain_col].astype(str)

    for regime, frame in (("internal", internal_local), ("ood", ood_local)):
        for model in MODELS:
            model_frame = frame[frame["model_id"] == model][
                ["row_index", "y_true", "y_pred"]
            ].copy()
            if model_frame["row_index"].duplicated().any():
                raise RuntimeError(f"Duplicate {regime} {model} rows")
            suffix = "chemistry" if model == MODELS[0] else "full"
            model_frame = model_frame.rename(
                columns={
                    "y_true": f"y_true_{regime}_{suffix}",
                    "y_pred": f"pred_{regime}_{suffix}",
                }
            )
            merged = merged.merge(model_frame, on=keys, how="inner")

    y_columns = [column for column in merged if column.startswith("y_true_")]
    for column in y_columns:
        if not np.allclose(
            merged["y_true"].to_numpy(dtype=float),
            merged[column].to_numpy(dtype=float),
            rtol=0,
            atol=1e-7,
        ):
            raise RuntimeError(f"Outcome mismatch in {column}")
    merged = merged.drop(columns=y_columns)
    if tuple(sorted(merged["domain"].unique())) != tuple(sorted(domain_order)):
        raise RuntimeError(f"Unexpected {axis} domain identity")
    return merged.sort_values("row_index").reset_index(drop=True)


def analyse_axis(
    frame: pd.DataFrame,
    axis: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    domains = frame["domain"].astype(str).to_numpy()
    y = frame["y_true"].to_numpy(dtype=np.float64)
    scaffolds = frame["scaffold_id"].astype(str).to_numpy()
    unique_scaffolds = np.unique(scaffolds)
    positions_by_scaffold = {
        value: np.where(scaffolds == value)[0] for value in unique_scaffolds
    }
    predictions = {
        (regime, model): frame[f"pred_{regime}_{model}"].to_numpy(dtype=float)
        for regime in ("internal", "ood")
        for model in ("chemistry", "full")
    }
    all_positions = np.arange(len(frame), dtype=np.int64)
    domain_order = DOMAINS[axis]
    observed = {
        key: metric_bundle(y, pred, domains, all_positions, domain_order)
        for key, pred in predictions.items()
    }

    model_distributions = {
        key: {
            metric: []
            for metric in (
                "domain_macro_spearman",
                "domain_macro_rmse",
                "pooled_spearman",
                "pooled_rmse",
            )
        }
        for key in predictions
    }
    contrast_distributions = {
        regime: {
            "delta_domain_macro_spearman": [],
            "delta_domain_macro_rmse": [],
            "delta_pooled_spearman": [],
            "delta_pooled_rmse": [],
        }
        for regime in ("internal", "ood")
    }
    regime_change = {
        "delta_domain_macro_spearman": [],
        "delta_domain_macro_rmse": [],
        "delta_pooled_spearman": [],
        "delta_pooled_rmse": [],
    }
    domain_distributions = {
        (regime, domain): []
        for regime in ("internal", "ood")
        for domain in domain_order
    }
    rng = np.random.default_rng(seed + (0 if axis == "source" else 1_000_000))
    for _ in range(n_bootstrap):
        sampled = rng.choice(
            unique_scaffolds, size=len(unique_scaffolds), replace=True
        )
        positions = np.concatenate([positions_by_scaffold[x] for x in sampled])
        stats = {
            key: metric_bundle(y, pred, domains, positions, domain_order)
            for key, pred in predictions.items()
        }
        for key, bundle in stats.items():
            for metric in model_distributions[key]:
                model_distributions[key][metric].append(float(bundle[metric]))
        bootstrap_contrasts: dict[str, dict[str, float]] = {}
        for regime in ("internal", "ood"):
            chemistry = stats[(regime, "chemistry")]
            full = stats[(regime, "full")]
            values = {
                "delta_domain_macro_spearman": (
                    full["domain_macro_spearman"]
                    - chemistry["domain_macro_spearman"]
                ),
                "delta_domain_macro_rmse": (
                    chemistry["domain_macro_rmse"]
                    - full["domain_macro_rmse"]
                ),
                "delta_pooled_spearman": (
                    full["pooled_spearman"] - chemistry["pooled_spearman"]
                ),
                "delta_pooled_rmse": (
                    chemistry["pooled_rmse"] - full["pooled_rmse"]
                ),
            }
            bootstrap_contrasts[regime] = values
            for metric, value in values.items():
                contrast_distributions[regime][metric].append(float(value))
            for domain in domain_order:
                domain_distributions[(regime, domain)].append(
                    float(
                        full["per_domain"][domain]["spearman"]
                        - chemistry["per_domain"][domain]["spearman"]
                    )
                )
        for metric in regime_change:
            regime_change[metric].append(
                bootstrap_contrasts["ood"][metric]
                - bootstrap_contrasts["internal"][metric]
            )

    model_rows: list[dict[str, Any]] = []
    for (regime, model), bundle in observed.items():
        row: dict[str, Any] = {
            "axis": axis,
            "validation_regime": regime,
            "model": model,
            "n_domains": len(domain_order),
            "n_rows": len(frame),
            "n_compounds": frame["canonical_smiles"].nunique(),
            "n_scaffolds": len(unique_scaffolds),
        }
        for metric in model_distributions[(regime, model)]:
            mean, low, high, valid = percentile(
                model_distributions[(regime, model)][metric]
            )
            row[f"{metric}_observed"] = float(bundle[metric])
            row[f"{metric}_bootstrap_mean"] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
            row[f"{metric}_n_valid"] = valid
        model_rows.append(row)

    contrast_rows: list[dict[str, Any]] = []
    for regime in ("internal", "ood"):
        chemistry = observed[(regime, "chemistry")]
        full = observed[(regime, "full")]
        observed_values = {
            "delta_domain_macro_spearman": (
                full["domain_macro_spearman"]
                - chemistry["domain_macro_spearman"]
            ),
            "delta_domain_macro_rmse": (
                chemistry["domain_macro_rmse"] - full["domain_macro_rmse"]
            ),
            "delta_pooled_spearman": (
                full["pooled_spearman"] - chemistry["pooled_spearman"]
            ),
            "delta_pooled_rmse": (
                chemistry["pooled_rmse"] - full["pooled_rmse"]
            ),
        }
        row = {
            "axis": axis,
            "contrast_type": "within_regime_full_minus_chemistry",
            "validation_regime": regime,
            "n_domains": len(domain_order),
            "n_scaffolds": len(unique_scaffolds),
        }
        for metric, value in observed_values.items():
            mean, low, high, valid = percentile(
                contrast_distributions[regime][metric]
            )
            row[f"{metric}_observed"] = float(value)
            row[f"{metric}_bootstrap_mean"] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
            row[f"{metric}_n_valid"] = valid
        contrast_rows.append(row)

    change_row: dict[str, Any] = {
        "axis": axis,
        "contrast_type": "ood_minus_internal_context_increment",
        "validation_regime": "paired_regime_change",
        "n_domains": len(domain_order),
        "n_scaffolds": len(unique_scaffolds),
    }
    for metric, values in regime_change.items():
        internal_value = contrast_rows[0][f"{metric}_observed"]
        ood_value = contrast_rows[1][f"{metric}_observed"]
        mean, low, high, valid = percentile(values)
        change_row[f"{metric}_observed"] = ood_value - internal_value
        change_row[f"{metric}_bootstrap_mean"] = mean
        change_row[f"{metric}_ci_low"] = low
        change_row[f"{metric}_ci_high"] = high
        change_row[f"{metric}_n_valid"] = valid
    contrast_rows.append(change_row)

    domain_rows: list[dict[str, Any]] = []
    for regime in ("internal", "ood"):
        for domain in domain_order:
            local = frame[frame["domain"] == domain]
            chem = observed[(regime, "chemistry")]["per_domain"][domain]
            full = observed[(regime, "full")]["per_domain"][domain]
            mean, low, high, valid = percentile(
                domain_distributions[(regime, domain)]
            )
            domain_rows.append(
                {
                    "axis": axis,
                    "validation_regime": regime,
                    "domain": domain,
                    "n_rows": len(local),
                    "n_compounds": local["canonical_smiles"].nunique(),
                    "n_scaffolds": local["scaffold_id"].nunique(),
                    "chemistry_spearman": chem["spearman"],
                    "full_spearman": full["spearman"],
                    "delta_spearman_observed": (
                        full["spearman"] - chem["spearman"]
                    ),
                    "delta_spearman_bootstrap_mean": mean,
                    "delta_spearman_ci_low": low,
                    "delta_spearman_ci_high": high,
                    "delta_spearman_n_valid": valid,
                    "chemistry_rmse": chem["rmse"],
                    "full_rmse": full["rmse"],
                    "delta_rmse_observed": chem["rmse"] - full["rmse"],
                }
            )

    audit_rows = [
        {
            "axis": axis,
            "check": "fixed_domain_identity",
            "status": "PASS",
            "detail": "|".join(domain_order),
        },
        {
            "axis": axis,
            "check": "common_row_universe",
            "status": "PASS",
            "detail": f"n_rows={len(frame)}",
        },
        {
            "axis": axis,
            "check": "global_scaffold_resampling",
            "status": "PASS",
            "detail": f"n_scaffolds={len(unique_scaffolds)}; n_bootstrap={n_bootstrap}",
        },
    ]
    return (
        pd.DataFrame(model_rows),
        pd.DataFrame(contrast_rows),
        pd.DataFrame(domain_rows),
        pd.DataFrame(audit_rows),
    )


def write_brief(
    output_dir: Path,
    contrast_table: pd.DataFrame,
    domain_table: pd.DataFrame,
    lodo_table: pd.DataFrame,
) -> None:
    lines = [
        "# Matched-estimand reviewer revision results",
        "",
        "All intervals use 10,000 paired global-scaffold bootstrap resamples.",
        "",
    ]
    for axis in ("source", "target"):
        lines.append(f"## {axis.capitalize()} axis")
        for regime in ("internal", "ood", "paired_regime_change"):
            row = contrast_table[
                (contrast_table["axis"] == axis)
                & (contrast_table["validation_regime"] == regime)
            ].iloc[0]
            metric = "delta_domain_macro_spearman"
            lines.append(
                f"- {regime}: {row[f'{metric}_observed']:.3f} "
                f"({row[f'{metric}_ci_low']:.3f} to "
                f"{row[f'{metric}_ci_high']:.3f})."
            )
        local = domain_table[
            (domain_table["axis"] == axis)
            & (domain_table["validation_regime"] == "ood")
        ].copy()
        local["abs_delta"] = local["delta_spearman_observed"].abs()
        driver = local.sort_values("abs_delta", ascending=False).iloc[0]
        lines.append(
            f"- Largest absolute OOD domain contrast: {driver['domain']} "
            f"({driver['delta_spearman_observed']:.3f})."
        )
        influence = lodo_table[lodo_table["axis"] == axis].copy()
        influence["abs_change"] = (
            influence["paired_regime_change_after_omission"]
            - influence["paired_regime_change_all_domains"]
        ).abs()
        strongest = influence.sort_values("abs_change", ascending=False).iloc[0]
        lines.append(
            f"- Largest matched leave-one-domain influence: omit "
            f"{strongest['omitted_domain']} -> "
            f"{strongest['paired_regime_change_after_omission']:.3f}."
        )
        lines.append("")
    (output_dir / "results_brief.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in output_dir.iterdir():
            if path.is_file():
                path.unlink()
            else:
                raise RuntimeError(f"Refusing to overwrite directory: {path}")

    internal = pd.read_csv(INTERNAL_FILE)
    ood = pd.read_csv(OOD_FILE)
    all_models: list[pd.DataFrame] = []
    all_contrasts: list[pd.DataFrame] = []
    all_domains: list[pd.DataFrame] = []
    all_audits: list[pd.DataFrame] = []
    aligned_frames: dict[str, pd.DataFrame] = {}
    for axis in ("source", "target"):
        frame = aligned_axis_frame(internal, ood, axis)
        aligned_frames[axis] = frame
        expected = EXPECTED[axis]
        observed = {
            "n_rows": len(frame),
            "n_compounds": frame["canonical_smiles"].nunique(),
            "n_scaffolds": frame["scaffold_id"].nunique(),
        }
        if observed != expected:
            raise RuntimeError(
                f"Frozen {axis} identity mismatch: {observed} != {expected}"
            )
        models, contrasts, domains, audit = analyse_axis(
            frame,
            axis,
            args.bootstrap_replicates,
            args.seed,
        )
        all_models.append(models)
        all_contrasts.append(contrasts)
        all_domains.append(domains)
        all_audits.append(audit)

    model_table = pd.concat(all_models, ignore_index=True)
    contrast_table = pd.concat(all_contrasts, ignore_index=True)
    domain_table = pd.concat(all_domains, ignore_index=True)
    audit_table = pd.concat(all_audits, ignore_index=True)
    pivot = domain_table.pivot_table(
        index=["axis", "domain"],
        columns="validation_regime",
        values="delta_spearman_observed",
    ).reset_index()
    pivot["domain_regime_change"] = pivot["ood"] - pivot["internal"]
    lodo_rows: list[dict[str, Any]] = []
    for axis, local in pivot.groupby("axis", sort=False):
        all_domains_change = float(local["domain_regime_change"].mean())
        for omitted in DOMAINS[axis]:
            retained = local[local["domain"] != omitted]
            lodo_rows.append(
                {
                    "axis": axis,
                    "omitted_domain": omitted,
                    "n_domains_retained": len(retained),
                    "paired_regime_change_all_domains": all_domains_change,
                    "paired_regime_change_after_omission": float(
                        retained["domain_regime_change"].mean()
                    ),
                }
            )
    lodo_table = pd.DataFrame(lodo_rows)
    model_table.to_csv(output_dir / "matched_estimand_models.csv", index=False)
    contrast_table.to_csv(
        output_dir / "matched_estimand_contrasts.csv", index=False
    )
    domain_table.to_csv(
        output_dir / "matched_domain_contrasts.csv", index=False
    )
    lodo_table.to_csv(
        output_dir / "matched_regime_change_lodo.csv", index=False
    )
    audit_table.to_csv(output_dir / "bootstrap_audit.csv", index=False)

    configuration = {
        "analysis": "post_hoc_reviewer_revision_matched_estimand_v1",
        "protocol_file": str(PROTOCOL_FILE.resolve()),
        "protocol_sha256": sha256_file(PROTOCOL_FILE),
        "internal_file": str(INTERNAL_FILE.resolve()),
        "internal_sha256": sha256_file(INTERNAL_FILE),
        "ood_file": str(OOD_FILE.resolve()),
        "ood_sha256": sha256_file(OOD_FILE),
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "fixed_domains": DOMAINS,
        "expected_identity": EXPECTED,
    }
    (output_dir / "configuration.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_brief(output_dir, contrast_table, domain_table, lodo_table)
    for axis in ("source", "target"):
        observed = float(
            contrast_table[
                (contrast_table["axis"] == axis)
                & (
                    contrast_table["contrast_type"]
                    == "ood_minus_internal_context_increment"
                )
            ]["delta_domain_macro_spearman_observed"].iloc[0]
        )
        reconstructed = float(
            lodo_table[lodo_table["axis"] == axis][
                "paired_regime_change_all_domains"
            ].iloc[0]
        )
        all_audits.append(
            pd.DataFrame(
                [
                    {
                        "axis": axis,
                        "check": "lodo_reconstructs_paired_regime_change",
                        "status": (
                            "PASS"
                            if math.isclose(observed, reconstructed, abs_tol=1e-12)
                            else "FAIL"
                        ),
                        "detail": f"observed={observed}; reconstructed={reconstructed}",
                    }
                ]
            )
        )
    audit_table = pd.concat(all_audits, ignore_index=True)
    audit_table.to_csv(output_dir / "bootstrap_audit.csv", index=False)
    failures = audit_table[audit_table["status"] != "PASS"]
    qa = {
        "status": "PASS" if failures.empty else "FAIL",
        "n_checks": int(len(audit_table)),
        "n_failures": int(len(failures)),
        "checks": audit_table.to_dict("records"),
    }
    (output_dir / "qa_summary.json").write_text(
        json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(qa, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
