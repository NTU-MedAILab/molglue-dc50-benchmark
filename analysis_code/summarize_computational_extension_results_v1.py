#!/usr/bin/env python3
"""Create a claim-safe numerical brief from aggregate extension source data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from computational_extension_lineage_v1 import (
    ANALYSIS_IDENTITY,
    PUBLIC_PROSPECTIVE_DATASET_STATEMENT,
    aggregate_bundle_hashes,
    downstream_software_bindings,
    require_publication_lineage_frozen,
    results_generation_id,
    validate_results_payload_from_aggregates,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ROUTE_DIR = SCRIPT_DIR.parent
DEFAULT_SOURCE_DIR = ROUTE_DIR / "reports" / "publication_validation_v1"
DEFAULT_JSON = DEFAULT_SOURCE_DIR / "computational_extension_results_v1.json"
DEFAULT_MD = DEFAULT_SOURCE_DIR / "computational_extension_results_v1.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    return parser.parse_args()


def read(source_dir: Path, stem: str) -> pd.DataFrame:
    path = source_dir / f"source_data_extension_{stem}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def scalar(row: pd.Series, name: str) -> float:
    value = float(row[name])
    if not np.isfinite(value):
        raise ValueError(f"Non-finite {name} in selected result row")
    return value


def one(frame: pd.DataFrame, label: str) -> pd.Series:
    if len(frame) != 1:
        raise ValueError(f"{label}: expected one row, observed {len(frame)}")
    return frame.iloc[0]


def interval_direction(low: float, high: float) -> str:
    if low > 0:
        return "favourable_interval_above_zero"
    if high < 0:
        return "unfavourable_interval_below_zero"
    return "interval_includes_zero"


def effect_record(
    analysis: str,
    condition: str,
    metric: str,
    estimate: float,
    low: float,
    high: float,
    direction: str,
    n_resamples: int,
) -> dict[str, Any]:
    return {
        "analysis": analysis,
        "condition": condition,
        "metric": metric,
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "interval_direction": interval_direction(low, high),
        "positive_direction": direction,
        "n_resamples": n_resamples,
        "evidence_identity": "post_hoc_sensitivity",
    }


def collect_portable(source_dir: Path) -> list[dict[str, Any]]:
    frame = read(source_dir, "portable_context")
    records: list[dict[str, Any]] = []
    for protocol in ("source_ood", "target_ood"):
        for contrast in (
            "full_vs_chemistry",
            "portable_vs_chemistry",
            "portable_vs_full",
        ):
            row = one(
                frame[
                    (frame["record_type"] == "contrast")
                    & (frame["protocol"] == protocol)
                    & (frame["contrast_id"] == contrast)
                ],
                f"{protocol}/{contrast}",
            )
            for metric in ("spearman", "rmse"):
                prefix = f"delta_domain_macro_{metric}"
                records.append(
                    effect_record(
                        "held_axis_portable_context",
                        f"{protocol}:{contrast}",
                        f"delta_domain_macro_{metric}",
                        scalar(row, f"{prefix}_observed"),
                        scalar(row, f"{prefix}_ci_low"),
                        scalar(row, f"{prefix}_ci_high"),
                        "positive_favours_first_named_model",
                        int(row["n_bootstrap"]),
                    )
                )
    return records


def collect_weights(source_dir: Path) -> list[dict[str, Any]]:
    frame = read(source_dir, "weighting_bootstrap")
    records: list[dict[str, Any]] = []
    contrasts = (
        "uniform_full_vs_chemistry",
        "compound_equal_full_vs_chemistry",
        "domain_balanced_full_vs_chemistry",
    )
    for contrast in contrasts:
        row = one(
            frame[
                (frame["record_type"] == "contrast")
                & (frame["contrast_id"] == contrast)
            ],
            contrast,
        )
        for metric in ("spearman", "rmse"):
            prefix = f"delta_{metric}"
            records.append(
                effect_record(
                    "internal_fit_weight",
                    contrast,
                    prefix,
                    scalar(row, f"{prefix}_observed"),
                    scalar(row, f"{prefix}_ci_low"),
                    scalar(row, f"{prefix}_ci_high"),
                    "positive_favours_full_context",
                    int(row["n_bootstrap"]),
                )
            )
    return records


def collect_weight_diagnostics(source_dir: Path) -> list[dict[str, Any]]:
    frame = read(source_dir, "weighting_diagnostics")
    records: list[dict[str, Any]] = []
    for weight_mode, group in frame.groupby("weight_mode", sort=False):
        clipping = pd.to_numeric(
            group["clipping_fraction"], errors="coerce"
        ).dropna()
        effective = pd.to_numeric(
            group["effective_sample_fraction"], errors="coerce"
        ).dropna()
        if clipping.empty or effective.empty:
            raise ValueError(f"Incomplete weight diagnostics for {weight_mode}")
        records.append(
            {
                "weight_mode": str(weight_mode),
                "n_fit_records": int(len(group)),
                "fit_stage_counts": {
                    str(key): int(value)
                    for key, value in group["fit_stage"].value_counts().items()
                },
                "n_fit_rows_min": int(group["n_fit_rows"].min()),
                "n_fit_rows_max": int(group["n_fit_rows"].max()),
                "clipping_fraction_mean": float(clipping.mean()),
                "clipping_fraction_max": float(clipping.max()),
                "effective_sample_fraction_min": float(effective.min()),
                "effective_sample_fraction_median": float(effective.median()),
                "effective_sample_fraction_max": float(effective.max()),
            }
        )
    return records


def collect_generic(source_dir: Path) -> list[dict[str, Any]]:
    frame = read(source_dir, "generic_scaffold")
    records: list[dict[str, Any]] = []
    for metric in ("delta_spearman", "delta_rmse"):
        row = one(
            frame[
                (frame["record_type"] == "contrast")
                & (frame["contrast_id"] == "full_vs_chemistry")
                & (frame["metric"] == metric)
            ],
            f"generic/{metric}",
        )
        records.append(
            effect_record(
                "generic_murcko_scaffold",
                "full_vs_chemistry",
                metric,
                scalar(row, "observed"),
                scalar(row, "ci_low"),
                scalar(row, "ci_high"),
                "positive_favours_full_context",
                int(row["n_bootstrap"]),
            )
        )
    return records


def collect_generic_group_summary(source_dir: Path) -> dict[str, Any]:
    row = one(read(source_dir, "generic_scaffold_groups"), "generic groups")
    return {
        "n_rows": int(row["n_rows"]),
        "n_unique_smiles": int(row["n_unique_smiles"]),
        "n_bemis_murcko_groups": int(row["n_bemis_murcko_groups"]),
        "n_generic_murcko_groups": int(row["n_generic_murcko_groups"]),
        "n_singleton_generic_groups": int(row["n_singleton_generic_groups"]),
        "largest_generic_group_rows": int(row["largest_generic_group_rows"]),
    }


def collect_source_deletion(source_dir: Path) -> list[dict[str, Any]]:
    frame = read(source_dir, "source_deletion")
    records: list[dict[str, Any]] = []
    for deletion in ("none", "MGTbind", "MGDB", "MolGlueDB", "TPDdb"):
        for metric in ("delta_spearman", "delta_rmse"):
            row = one(
                frame[
                    (frame["record_type"] == "full_vs_chemistry")
                    & (frame["contrast_type"] == "full_vs_chemistry")
                    & (frame["deletion_source"] == deletion)
                    & (frame["heldout_target"].isna())
                    & (frame["metric"] == metric)
                ],
                f"source_deletion/{deletion}/{metric}",
            )
            records.append(
                effect_record(
                    "target_ood_training_source_deletion",
                    deletion,
                    metric,
                    scalar(row, "observed"),
                    scalar(row, "ci_low"),
                    scalar(row, "ci_high"),
                    "positive_favours_full_context",
                    int(row["n_bootstrap"]),
                )
            )
    for deletion in ("MGTbind", "MGDB", "MolGlueDB", "TPDdb"):
        for model_id in (
            "chemistry_extra_trees",
            "full_context_extra_trees",
        ):
            for metric in ("delta_spearman", "delta_rmse"):
                row = one(
                    frame[
                        (frame["record_type"] == "deletion_vs_baseline")
                        & (frame["contrast_type"] == "deletion_vs_baseline")
                        & (frame["deletion_source"] == deletion)
                        & (frame["model_id"] == model_id)
                        & (frame["heldout_target"].isna())
                        & (frame["metric"] == metric)
                    ],
                    (
                        "source_deletion_vs_baseline/"
                        f"{deletion}/{model_id}/{metric}"
                    ),
                )
                records.append(
                    effect_record(
                        "target_ood_deletion_vs_within_script_baseline",
                        f"{deletion}:{model_id}",
                        metric,
                        scalar(row, "observed"),
                        scalar(row, "ci_low"),
                        scalar(row, "ci_high"),
                        "positive_favours_deletion_condition",
                        int(row["n_bootstrap"]),
                    )
                )
    return records


def collect_applicability(source_dir: Path) -> list[dict[str, Any]]:
    frame = read(source_dir, "applicability_contrasts")
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        internal = (
            row["regime"] == "internal_scaffold_disjoint"
            and row["protocol"] == "scaffold"
        )
        for metric in ("spearman", "rmse"):
            prefix = (
                f"delta_{metric}"
                if internal
                else f"delta_domain_macro_{metric}"
            )
            estimate = pd.to_numeric(
                pd.Series([row.get(prefix)]), errors="coerce"
            ).iloc[0]
            low = pd.to_numeric(
                pd.Series([row.get(f"{prefix}_ci_low")]), errors="coerce"
            ).iloc[0]
            high = pd.to_numeric(
                pd.Series([row.get(f"{prefix}_ci_high")]), errors="coerce"
            ).iloc[0]
            records.append(
                {
                    "analysis": "chemical_novelty_applicability",
                    "condition": (
                        f"{row['regime']}:{row['protocol']}:"
                        f"{row['similarity_bin']}"
                    ),
                    "metric": prefix,
                    "estimate": None if pd.isna(estimate) else float(estimate),
                    "ci_low": None if pd.isna(low) else float(low),
                    "ci_high": None if pd.isna(high) else float(high),
                    "interval_direction": (
                        "not_estimable"
                        if pd.isna(low) or pd.isna(high)
                        else interval_direction(float(low), float(high))
                    ),
                    "positive_direction": "positive_favours_full_context",
                    "n_resamples": int(row["n_bootstrap_requested"]),
                    "n_rows": int(row["n_rows"]),
                    "n_scaffolds": int(row["n_scaffolds"]),
                    "n_domains_expected": int(row["n_domains_expected"]),
                    "evidence_identity": "post_hoc_diagnostic",
                }
            )
    return records


def collect_permutation(source_dir: Path) -> list[dict[str, Any]]:
    inference = read(source_dir, "permutation_inference")
    records: list[dict[str, Any]] = []
    for _, row in inference.iterrows():
        records.append(
            {
                "analysis": "conditional_label_randomization",
                "estimand_id": row["estimand_id"],
                "metric": row["metric"],
                "observed": scalar(row, "observed"),
                "null_mean": scalar(row, "null_mean"),
                "null_percentile_2p5": scalar(row, "null_percentile_2p5"),
                "null_percentile_97p5": scalar(row, "null_percentile_97p5"),
                "empirical_tail": row["empirical_tail"],
                "empirical_p": scalar(row, "empirical_p"),
                "n_permutations": int(row["n_permutations_finite"]),
                "evidence_identity": "post_hoc_repeated_negative_control",
            }
        )
    return records


def collect_censoring(source_dir: Path) -> dict[str, Any]:
    overall = read(source_dir, "censoring_overall")
    diagnostics = read(source_dir, "permutation_diagnostics")
    selection = read(source_dir, "censoring_selection_flow").drop(
        columns=["analysis_identity"]
    )
    return {
        "parsed_total": int(overall["n_records"].sum()),
        "relation_counts": {
            str(row["relation_class"]): int(row["n_records"])
            for _, row in overall.iterrows()
        },
        "relation_fractions": {
            str(row["relation_class"]): float(row["fraction_within_group"])
            for _, row in overall.iterrows()
        },
        "permutation_singleton_row_fraction_mean": float(
            diagnostics["permutation_singleton_row_fraction"].mean()
        ),
        "permutation_changed_label_fraction_mean": float(
            diagnostics["permutation_changed_label_fraction"].mean()
        ),
        "selection_flow": selection.to_dict(orient="records"),
        "evidence_identity": "post_hoc_descriptive_audit",
    }


def fmt_effect(record: dict[str, Any]) -> str:
    return (
        f"{record['estimate']:+.3f} "
        f"[{record['ci_low']:+.3f}, {record['ci_high']:+.3f}]"
    )


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    require_publication_lineage_frozen()
    source_data_provenance = aggregate_bundle_hashes(args.source_dir)
    effects = (
        collect_portable(args.source_dir)
        + collect_weights(args.source_dir)
        + collect_generic(args.source_dir)
        + collect_source_deletion(args.source_dir)
        + collect_applicability(args.source_dir)
    )
    weight_diagnostics = collect_weight_diagnostics(args.source_dir)
    generic_groups = collect_generic_group_summary(args.source_dir)
    permutation = collect_permutation(args.source_dir)
    censoring = collect_censoring(args.source_dir)
    software_bindings = downstream_software_bindings()
    payload = {
        "analysis_identity": ANALYSIS_IDENTITY,
        "source_data_provenance": source_data_provenance,
        "software_bindings": software_bindings,
        "results_generation_id": results_generation_id(
            source_data_provenance,
            software_bindings,
        ),
        "effect_records": effects,
        "weight_diagnostics": weight_diagnostics,
        "generic_scaffold_group_summary": generic_groups,
        "permutation_records": permutation,
        "censoring_and_randomization_diagnostics": censoring,
        "claim_boundary": [
            "All analyses are post-hoc sensitivities, diagnostics or repeated negative controls.",
            "Intervals are conditional on the frozen observations and fixed OOD domains.",
            "No result is prospective validation, calibrated absolute prediction or mechanistic evidence.",
            PUBLIC_PROSPECTIVE_DATASET_STATEMENT,
        ],
    }
    validate_results_payload_from_aggregates(
        payload,
        args.source_dir,
        source_data_provenance,
    )
    atomic_text(
        args.output_json,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )

    lines = [
        "# Computational extension results v1",
        "",
        f"Analysis identity: `{ANALYSIS_IDENTITY}`  ",
        "Source-data generation ID: "
        f"`{source_data_provenance['source_data_generation_id']}`",
        "",
        "Evidence identity: **post-hoc**. Positive paired contrasts favour the "
        "first named model; no multiplicity-adjusted significance claim is made.",
        "",
        "## Primary robustness effects",
        "",
        "| Analysis | Condition | Metric | Estimate [95% interval] | Interval relation to zero |",
        "|---|---|---|---:|---|",
    ]
    primary_analyses = {
        "held_axis_portable_context",
        "internal_fit_weight",
        "generic_murcko_scaffold",
        "target_ood_training_source_deletion",
        "target_ood_deletion_vs_within_script_baseline",
    }
    for record in effects:
        if record["analysis"] not in primary_analyses:
            continue
        lines.append(
            f"| {record['analysis']} | {record['condition']} | "
            f"{record['metric']} | {fmt_effect(record)} | "
            f"{record['interval_direction']} |"
        )
    lines.extend(
        [
            "",
            "Generic Murcko grouping reduced "
            f"{generic_groups['n_bemis_murcko_groups']:,} Bemis–Murcko groups "
            f"to {generic_groups['n_generic_murcko_groups']:,} generic groups; "
            f"{generic_groups['n_singleton_generic_groups']:,} were singletons "
            "and the largest contained "
            f"{generic_groups['largest_generic_group_rows']:,} rows.",
            "",
            "## Fit-weight diagnostics",
            "",
            "| Weight mode | Fit records | Clipped fraction, mean–max | Effective-sample fraction, min–median–max |",
            "|---|---:|---:|---:|",
        ]
    )
    for record in weight_diagnostics:
        lines.append(
            f"| {record['weight_mode']} | {record['n_fit_records']} | "
            f"{record['clipping_fraction_mean']:.3f}–"
            f"{record['clipping_fraction_max']:.3f} | "
            f"{record['effective_sample_fraction_min']:.3f}–"
            f"{record['effective_sample_fraction_median']:.3f}–"
            f"{record['effective_sample_fraction_max']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Chemical-novelty applicability contrasts",
            "",
            "| Regime, protocol and Tanimoto bin | Metric | Estimate [95% interval] | Interval relation to zero | Rows | Scaffolds |",
            "|---|---|---:|---|---:|---:|",
        ]
    )
    for record in effects:
        if record["analysis"] != "chemical_novelty_applicability":
            continue
        if record["estimate"] is None:
            estimate = "not estimable"
        elif record["ci_low"] is None or record["ci_high"] is None:
            estimate = f"{record['estimate']:+.3f} [interval not estimable]"
        else:
            estimate = (
                f"{record['estimate']:+.3f} "
                f"[{record['ci_low']:+.3f}, {record['ci_high']:+.3f}]"
            )
        lines.append(
            f"| {record['condition']} | {record['metric']} | {estimate} | "
            f"{record['interval_direction']} | {record['n_rows']} | "
            f"{record['n_scaffolds']} |"
        )
    lines.extend(
        [
            "",
            "## Conditional label-randomization controls",
            "",
            "| Estimand | Metric | Tail | Observed | Null mean | Null 95% range | Empirical p |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for record in permutation:
        lines.append(
            f"| {record['estimand_id']} | {record['metric']} | "
            f"{record['empirical_tail']} | "
            f"{record['observed']:.3f} | {record['null_mean']:.3f} | "
            f"[{record['null_percentile_2p5']:.3f}, "
            f"{record['null_percentile_97p5']:.3f}] | "
            f"{record['empirical_p']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Endpoint-selection boundary",
            "",
            f"- Parsed records: {censoring['parsed_total']:,}.",
            "- Relation counts: "
            + ", ".join(
                f"{key}={value:,}"
                for key, value in censoring["relation_counts"].items()
            )
            + ".",
            "- Mean singleton-row fraction in conditional permutation fit folds: "
            f"{censoring['permutation_singleton_row_fraction_mean']:.3%}.",
            "- Mean changed-label fraction in conditional permutation fit folds: "
            f"{censoring['permutation_changed_label_fraction_mean']:.3%}.",
            "",
            "## Claim boundary",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload["claim_boundary"])
    lines.append("")
    atomic_text(args.output_md, "\n".join(lines))
    print(
        "COMPUTATIONAL_EXTENSION_SUMMARY: PASS "
        f"({len(effects)} effect records; {len(permutation)} null records)"
    )


if __name__ == "__main__":
    main()
