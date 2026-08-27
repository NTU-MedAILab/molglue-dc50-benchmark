#!/usr/bin/env python3
"""Bootstrap grouped-context contrasts on matched fixed-domain test universes."""

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


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "reports" / "post_hoc_context_group_ablation_revision_v1"
DEFAULT_OUTPUT = ROOT / "reports" / "post_hoc_context_group_ablation_revision_v1_summary"
DOMAINS = {
    "source": ("MGTbind", "MGDB", "MolGlueDB", "TPDdb"),
    "target": ("VAV1", "CSNK1A1", "GSPT1", "WIZ", "CDK2", "CCNK+CDK12", "IKZF2", "IKZF1"),
}
MODEL_ORDER = (
    "chemistry_extra_trees",
    "chemistry_plus_provenance",
    "chemistry_plus_biological_assay",
    "chemistry_plus_missingness",
    "full_context_extra_trees",
)
LABELS = {
    "chemistry_extra_trees": "Chemistry",
    "chemistry_plus_provenance": "Provenance",
    "chemistry_plus_biological_assay": "Biological/assay",
    "chemistry_plus_missingness": "Missingness",
    "full_context_extra_trees": "Full context",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
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


def rho(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) < 3 or np.unique(y).size < 2 or np.unique(pred).size < 2:
        return float("nan")
    return float(spearmanr(y, pred).statistic)


def domain_macro(
    y: np.ndarray,
    pred: np.ndarray,
    domain: np.ndarray,
    positions: np.ndarray,
    order: tuple[str, ...],
) -> tuple[float, dict[str, float]]:
    values: dict[str, float] = {}
    for name in order:
        local = positions[domain[positions] == name]
        values[name] = rho(y[local], pred[local]) if len(local) else float("nan")
    sequence = np.asarray([values[name] for name in order], dtype=float)
    return (
        float(np.mean(sequence)) if np.all(np.isfinite(sequence)) else float("nan"),
        values,
    )


def align(
    internal: pd.DataFrame,
    ood: pd.DataFrame,
    axis: str,
    regime: str,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    order = DOMAINS[axis]
    domain_col = "source_database" if axis == "source" else "target_protein"
    if regime == "internal":
        local = internal[internal[domain_col].astype(str).isin(order)].copy()
    else:
        protocol = f"{axis}_ood"
        local = ood[
            (ood["protocol"] == protocol)
            & ood["heldout_group"].astype(str).isin(order)
        ].copy()
    models = tuple(model for model in MODEL_ORDER if model in set(local["model_id"].astype(str)))
    if models[0] != "chemistry_extra_trees":
        raise RuntimeError("Chemistry baseline is absent")
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
        local[local.model_id == "chemistry_extra_trees"][identity_cols]
        .drop_duplicates("row_index")
        .sort_values("row_index")
    )
    if reference.row_index.duplicated().any():
        raise RuntimeError("Reference row keys are duplicated")
    merged = reference.copy()
    merged["domain"] = merged[domain_col].astype(str)
    for model in models:
        model_frame = local[local.model_id == model][["row_index", "y_true", "y_pred"]].copy()
        if model_frame.row_index.duplicated().any():
            raise RuntimeError(f"Duplicate row keys for {axis}/{regime}/{model}")
        model_frame = model_frame.rename(
            columns={"y_true": f"y_true__{model}", "y_pred": f"pred__{model}"}
        )
        merged = merged.merge(model_frame, on="row_index", how="inner")
    if set(merged.domain.astype(str)) != set(order):
        raise RuntimeError(f"Unexpected domains for {axis}/{regime}")
    if len(merged) != len(reference):
        raise RuntimeError(f"Model predictions are incomplete for {axis}/{regime}")
    for model in models:
        if not np.allclose(merged.y_true, merged[f"y_true__{model}"], rtol=0, atol=1e-7):
            raise RuntimeError(f"Outcome mismatch for {axis}/{regime}/{model}")
    return merged.drop(columns=[f"y_true__{model}" for model in models]), models


def analyse(
    frame: pd.DataFrame,
    models: tuple[str, ...],
    axis: str,
    regime: str,
    replicates: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    order = DOMAINS[axis]
    y = frame.y_true.to_numpy(dtype=float)
    domains = frame.domain.astype(str).to_numpy()
    scaffolds = frame.scaffold_id.astype(str).to_numpy()
    unique_scaffolds = np.unique(scaffolds)
    positions_by_scaffold = {value: np.where(scaffolds == value)[0] for value in unique_scaffolds}
    predictions = {model: frame[f"pred__{model}"].to_numpy(dtype=float) for model in models}
    positions = np.arange(len(frame), dtype=np.int64)
    observed = {model: domain_macro(y, pred, domains, positions, order) for model, pred in predictions.items()}
    baseline = observed["chemistry_extra_trees"]
    distributions = {model: [] for model in models if model != "chemistry_extra_trees"}
    domain_distributions = {
        (model, domain): []
        for model in distributions
        for domain in order
    }
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        sampled = rng.choice(unique_scaffolds, len(unique_scaffolds), replace=True)
        boot_positions = np.concatenate([positions_by_scaffold[value] for value in sampled])
        stats = {
            model: domain_macro(y, pred, domains, boot_positions, order)
            for model, pred in predictions.items()
        }
        for model in distributions:
            distributions[model].append(stats[model][0] - stats["chemistry_extra_trees"][0])
            for domain in order:
                domain_distributions[(model, domain)].append(
                    stats[model][1][domain] - stats["chemistry_extra_trees"][1][domain]
                )

    rows: list[dict[str, Any]] = []
    domain_rows: list[dict[str, Any]] = []
    for model, values in distributions.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        rows.append(
            {
                "axis": axis,
                "validation_regime": regime,
                "model_id": model,
                "context_group": LABELS[model],
                "n_domains": len(order),
                "n_rows": len(frame),
                "n_scaffolds": len(unique_scaffolds),
                "delta_domain_macro_spearman_observed": observed[model][0] - baseline[0],
                "delta_domain_macro_spearman_bootstrap_mean": float(np.mean(finite)),
                "delta_domain_macro_spearman_ci_low": float(np.percentile(finite, 2.5)),
                "delta_domain_macro_spearman_ci_high": float(np.percentile(finite, 97.5)),
                "delta_domain_macro_spearman_n_valid": len(finite),
            }
        )
        for domain in order:
            local = frame[frame.domain == domain]
            values_domain = np.asarray(domain_distributions[(model, domain)], dtype=float)
            values_domain = values_domain[np.isfinite(values_domain)]
            domain_rows.append(
                {
                    "axis": axis,
                    "validation_regime": regime,
                    "domain": domain,
                    "model_id": model,
                    "context_group": LABELS[model],
                    "n_rows": len(local),
                    "n_compounds": local.canonical_smiles.nunique(),
                    "n_scaffolds": local.scaffold_id.nunique(),
                    "delta_spearman_observed": observed[model][1][domain] - baseline[1][domain],
                    "delta_spearman_ci_low": float(np.percentile(values_domain, 2.5)),
                    "delta_spearman_ci_high": float(np.percentile(values_domain, 97.5)),
                    "delta_spearman_n_valid": len(values_domain),
                }
            )
    return rows, domain_rows, len(unique_scaffolds)


def main() -> None:
    args = parse_args()
    source = args.input_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()) and not args.overwrite:
        raise RuntimeError("Output directory is not empty; use --overwrite")
    if args.overwrite:
        for path in output.iterdir():
            if path.is_file():
                path.unlink()
    required = (
        source / "internal_repeat_averaged_predictions.csv",
        source / "ood_predictions.csv",
        source / "configuration.json",
        source / "qa_summary.json",
        source / "run_manifest.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Formal grouped-ablation run is incomplete: {missing}")
    configuration = json.loads((source / "configuration.json").read_text(encoding="utf-8"))
    run_qa = json.loads((source / "qa_summary.json").read_text(encoding="utf-8"))
    if run_qa.get("status") != "PASS" or int(configuration.get("n_estimators", 0)) != 600:
        raise RuntimeError("Grouped-ablation input is not the formal 600-tree PASS run")
    internal = pd.read_csv(source / "internal_repeat_averaged_predictions.csv")
    ood = pd.read_csv(source / "ood_predictions.csv")
    rows: list[dict[str, Any]] = []
    domain_rows: list[dict[str, Any]] = []
    scaffold_counts: dict[str, int] = {}
    for axis_index, axis in enumerate(("source", "target")):
        for regime in ("internal", "ood"):
            frame, models = align(internal, ood, axis, regime)
            local_rows, local_domains, n_scaffolds = analyse(
                frame,
                models,
                axis,
                regime,
                args.bootstrap_replicates,
                args.seed + axis_index * 1_000_000,
            )
            rows.extend(local_rows)
            domain_rows.extend(local_domains)
            scaffold_counts[f"{axis}_{regime}"] = n_scaffolds
    summary = pd.DataFrame(rows)
    per_domain = pd.DataFrame(domain_rows)
    summary.to_csv(output / "matched_group_contrasts.csv", index=False)
    per_domain.to_csv(output / "matched_group_domain_contrasts.csv", index=False)
    checks = [
        {"check": "formal_fit_qa_pass", "status": "PASS" if run_qa.get("status") == "PASS" else "FAIL"},
        {"check": "internal_repeat_average_complete", "status": "PASS" if (pd.to_numeric(internal["n_repeats"]) == 5).all() else "FAIL"},
        {"check": "four_axis_regime_estimands", "status": "PASS" if summary.groupby(["axis", "validation_regime"]).ngroups == 4 else "FAIL"},
        {"check": "bootstrap_complete", "status": "PASS" if (summary.delta_domain_macro_spearman_n_valid == args.bootstrap_replicates).all() else "FAIL"},
        {"check": "source_provenance_not_claimed_portable", "status": "PASS" if summary[(summary.axis == "source") & (summary.validation_regime == "ood") & (summary.model_id == "chemistry_plus_provenance")].empty else "FAIL"},
        {"check": "expected_scaffold_counts", "status": "PASS" if scaffold_counts == {"source_internal": 667, "source_ood": 667, "target_internal": 601, "target_ood": 601} else "FAIL"},
        {"check": "all_effects_finite", "status": "PASS" if np.isfinite(summary.filter(regex="observed|ci_low|ci_high").to_numpy(dtype=float)).all() else "FAIL"},
    ]
    status = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
    qa = {"status": status, "n_checks": len(checks), "checks": checks}
    (output / "qa_summary.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "status": status,
        "protocol": "post_hoc_context_group_ablation_revision_v1_summary.0",
        "input_configuration_sha256": sha256_file(source / "configuration.json"),
        "input_run_manifest_sha256": sha256_file(source / "run_manifest.json"),
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "n_contrasts": len(summary),
        "n_domain_contrasts": len(per_domain),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
