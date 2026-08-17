#!/usr/bin/env python3
"""Fail-closed consistency checks between manuscript prose and source data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from computational_extension_lineage_v1 import (
    ANALYSIS_IDENTITY as EXTENSION_ANALYSIS_IDENTITY,
    FLOAT32_CORRECTION_SHA256,
    PROVENANCE_FILENAME as EXTENSION_PROVENANCE_FILENAME,
    WORKFLOW_IDENTITIES,
    aggregate_bundle_hashes,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ROUTE_DIR = SCRIPT_DIR.parent
SOURCE_DIR = ROUTE_DIR / "reports" / "publication_validation_v1"
MANUSCRIPT = ROUTE_DIR / "manuscript" / "manuscript_draft_v1.md"
SUPPLEMENT = ROUTE_DIR / "manuscript" / "supplementary_information_v1.md"
REFERENCES = ROUTE_DIR / "manuscript" / "references_v1.bib"
EXTENSION_RESULTS = SOURCE_DIR / "computational_extension_results_v1.json"
EXTENSION_PROVENANCE = SOURCE_DIR / EXTENSION_PROVENANCE_FILENAME
ACTIVE_EXTENSION_INTEGRATION_CONTRACT = (
    ROUTE_DIR / "docs" / "manuscript_v4_extension_integration_contract.md"
)
ACTIVE_EXTENSION_INTEGRATION_CONTRACT_SHA256 = (
    "eddcc263cc9a72c27f4f0ead5aee4e91f928d432b7d251ed906542be8777a2db"
)
SUPERSEDED_EXTENSION_INTEGRATION_CONTRACT = (
    ROUTE_DIR / "docs" / "manuscript_v3_extension_integration_contract.md"
)
FIXED_DOMAIN_CORRECTION_SHA256 = str(
    WORKFLOW_IDENTITIES["null_applicability_censoring"][
        "fixed_domain_correction_sha256"
    ]
)
MIXED_LINEAGE_WORKFLOWS = {
    workflow: ROUTE_DIR / "reports" / str(identity["directory_name"])
    for workflow, identity in WORKFLOW_IDENTITIES.items()
}

RESULT_BEGIN = "<!-- COMPUTATIONAL_EXTENSION_V4_RESULTS_BEGIN -->"
RESULT_END = "<!-- COMPUTATIONAL_EXTENSION_V4_RESULTS_END -->"
RESULT_PENDING = "COMPUTATIONAL_EXTENSION_V4_RESULTS_PENDING"
DISCUSSION_PENDING = "COMPUTATIONAL_EXTENSION_V4_DISCUSSION_PENDING"
SI_RESULT_BEGIN = "<!-- SI_COMPUTATIONAL_EXTENSION_V4_RESULTS_BEGIN -->"
SI_RESULT_END = "<!-- SI_COMPUTATIONAL_EXTENSION_V4_RESULTS_END -->"
SI_RESULT_PENDING = "SI_COMPUTATIONAL_EXTENSION_V4_RESULTS_PENDING"

FULL = "full_context_extra_trees"
CHEM = "chemistry_extra_trees"
MAIN_EXTENSION_EFFECT_KEYS = (
    (
        "held_axis_portable_context",
        "source_ood:portable_vs_chemistry",
        "delta_domain_macro_spearman",
    ),
    (
        "held_axis_portable_context",
        "target_ood:portable_vs_chemistry",
        "delta_domain_macro_spearman",
    ),
    (
        "internal_fit_weight",
        "compound_equal_full_vs_chemistry",
        "delta_spearman",
    ),
    (
        "internal_fit_weight",
        "domain_balanced_full_vs_chemistry",
        "delta_spearman",
    ),
    (
        "generic_murcko_scaffold",
        "full_vs_chemistry",
        "delta_spearman",
    ),
)
SI_PERMUTATION_KEYS = (
    ("chemistry_extra_trees", "spearman"),
    ("full_context_extra_trees", "spearman"),
    ("full_minus_chemistry", "delta_spearman"),
)
DECIMAL_CLAIM_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.])[-+]?(?:\d+\.\d+|\.\d+)(?![A-Za-z0-9_.])"
)


def normalized(text: str) -> str:
    text = text.replace("-\n", "-").replace("−", "-")
    return re.sub(r"\s+", " ", text).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-pending-extension",
        action="store_true",
        help=(
            "Development-only: allow absent mixed-lineage v4 aggregate results "
            "when all explicit manuscript/SI pending sentinels are present. "
            "Omit this flag for submission/release QA."
        ),
    )
    return parser.parse_args()


def section(text: str, start: str, end: str) -> str:
    if start not in text or end not in text:
        return ""
    return text.split(start, 1)[1].split(end, 1)[0]


def word_count(text: str) -> int:
    return len(
        re.findall(
            r"[A-Za-z0-9]+(?:[–-][A-Za-z0-9]+)*",
            re.sub(r"\[[^\]]+\]", " ", text),
        )
    )


def sentence_count(text: str) -> int:
    prose = normalized(text)
    if not prose:
        return 0
    return len(re.split(r"(?<=[.!?])\s+(?=[A-Z])", prose))


def json_status(payload: dict[str, Any]) -> str:
    return str(
        payload.get(
            "overall_status",
            payload.get("status", payload.get("qa_status", "")),
        )
    ).strip().upper()


def three_decimal_pattern(value: float) -> str:
    token = f"{float(value):.3f}"
    if token.startswith("-"):
        return re.escape(token)
    return rf"\+?{re.escape(token)}"


def effect_triplet_present(block: str, record: dict[str, Any]) -> bool:
    prose = normalized(block)
    try:
        estimate = three_decimal_pattern(float(record["estimate"]))
        low = three_decimal_pattern(float(record["ci_low"]))
        high = three_decimal_pattern(float(record["ci_high"]))
    except (KeyError, TypeError, ValueError):
        return False
    return (
        re.search(
            rf"(?<![\d.]){estimate}(?![\d.]).{{0,180}}"
            rf"(?<![\d.]){low}(?![\d.]).{{0,80}}"
            rf"(?<![\d.]){high}(?![\d.])",
            prose,
        )
        is not None
    )


def permutation_record_present(block: str, record: dict[str, Any]) -> bool:
    prose = normalized(block)
    try:
        observed = three_decimal_pattern(float(record["observed"]))
        low = three_decimal_pattern(float(record["null_percentile_2p5"]))
        high = three_decimal_pattern(float(record["null_percentile_97p5"]))
        p_value = float(record["empirical_p"])
    except (KeyError, TypeError, ValueError):
        return False
    p_patterns = {
        f"{p_value:.3f}",
        f"{p_value:.4f}",
        f"{p_value:.5f}",
        f"{p_value:.6f}",
    }
    matched = re.search(
        rf"(?<![\d.]){observed}(?![\d.]).{{0,220}}"
        rf"(?<![\d.]){low}(?![\d.]).{{0,80}}"
        rf"(?<![\d.]){high}(?![\d.]).{{0,180}}",
        prose,
    )
    return matched is not None and any(
        token in matched.group(0) for token in p_patterns
    )


def decimal_claim_tokens(block: str) -> list[str]:
    """Return display-sensitive decimal claims from a bounded result block."""

    return [
        match.group(0).lstrip("+")
        for match in DECIMAL_CLAIM_PATTERN.finditer(
            block.replace("−", "-")
        )
    ]


def formatted_value_tokens(
    value: object,
    precisions: tuple[int, ...],
) -> set[str]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return set()
    if not math.isfinite(numeric):
        return set()
    return {
        f"{numeric:.{precision}f}".lstrip("+")
        for precision in precisions
    }


def aggregate_decimal_allowlist(
    extension: dict[str, Any],
    *,
    main_block: bool,
) -> set[str]:
    """Build the decimal display allowlist from the bound aggregate payload.

    Main-text decimals are restricted to the five contracted primary records.
    The SI may use any aggregate effect/permutation value because it is the
    detailed reporting location. Prespecified decimal bin and percentile
    labels are derived from aggregate conditions/schema rather than admitted
    as arbitrary prose constants.
    """

    allowed: set[str] = set()
    effects = [
        record
        for record in extension.get("effect_records", [])
        if isinstance(record, dict)
    ]
    if main_block:
        keys = set(MAIN_EXTENSION_EFFECT_KEYS)
        effects = [
            record
            for record in effects
            if (
                record.get("analysis"),
                record.get("condition"),
                record.get("metric"),
            )
            in keys
        ]
    for record in effects:
        for field in ("estimate", "ci_low", "ci_high"):
            allowed.update(formatted_value_tokens(record.get(field), (3,)))
        if not main_block:
            for raw_boundary in re.findall(
                r"\d+\.\d+",
                str(record.get("condition", "")),
            ):
                allowed.add(raw_boundary)
                numeric = float(raw_boundary)
                allowed.update(formatted_value_tokens(numeric, (1, 3, 6)))

    if not main_block:
        for record in extension.get("permutation_records", []):
            if not isinstance(record, dict):
                continue
            for field in (
                "observed",
                "null_mean",
                "null_percentile_2p5",
                "null_percentile_97p5",
            ):
                allowed.update(
                    formatted_value_tokens(record.get(field), (3,))
                )
            allowed.update(
                formatted_value_tokens(
                    record.get("empirical_p"),
                    (3, 4, 5, 6),
                )
            )
        allowed.update({"2.5", "97.5"})
    return allowed


def unbound_decimal_claims(
    block: str,
    allowed: set[str],
) -> list[str]:
    return sorted(
        {
            token
            for token in decimal_claim_tokens(block)
            if token not in allowed
        }
    )


def main() -> None:
    args = parse_args()
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    supplement = SUPPLEMENT.read_text(encoding="utf-8")
    prose = normalized(manuscript)
    combined = normalized(f"{manuscript}\n{supplement}")
    checks: list[dict[str, Any]] = []

    def add(check_id: str, passed: bool, evidence: str) -> None:
        checks.append(
            {
                "check_id": check_id,
                "status": "PASS" if passed else "FAIL",
                "evidence": evidence,
            }
        )

    def has(check_id: str, snippet: str) -> None:
        target = normalized(snippet)
        add(check_id, target in prose, target)

    def has_combined(check_id: str, snippet: str) -> None:
        target = normalized(snippet)
        add(check_id, target in combined, target)

    active_contract_hash = (
        sha256_file(ACTIVE_EXTENSION_INTEGRATION_CONTRACT)
        if ACTIVE_EXTENSION_INTEGRATION_CONTRACT.is_file()
        else ""
    )
    add(
        "active_v4_extension_integration_contract",
        (
            active_contract_hash
            == ACTIVE_EXTENSION_INTEGRATION_CONTRACT_SHA256
        ),
        (
            f"expected={ACTIVE_EXTENSION_INTEGRATION_CONTRACT_SHA256}; "
            f"observed={active_contract_hash or 'missing'}"
        ),
    )
    superseded_contract_text = (
        SUPERSEDED_EXTENSION_INTEGRATION_CONTRACT.read_text(encoding="utf-8")
        if SUPERSEDED_EXTENSION_INTEGRATION_CONTRACT.is_file()
        else ""
    )
    add(
        "v3_extension_integration_contract_superseded",
        (
            "SUPERSEDED" in superseded_contract_text[:200]
            and "non-authoritative" in superseded_contract_text[:500]
            and "manuscript_v4_extension_integration_contract.md"
            in superseded_contract_text[:800]
        ),
        "v3 contract is visibly superseded and points to the active v4 contract",
    )

    internal = pd.read_csv(SOURCE_DIR / "source_data_internal_models.csv")
    internal_contrasts = pd.read_csv(
        SOURCE_DIR / "source_data_internal_contrasts.csv"
    )
    ood = pd.read_csv(SOURCE_DIR / "source_data_confirmatory_ood.csv")
    strict = pd.read_csv(SOURCE_DIR / "source_data_strict_ood.csv")
    hgb = pd.read_csv(
        SOURCE_DIR / "source_data_hgb_model_family_sensitivity.csv"
    )
    graph = pd.read_csv(
        SOURCE_DIR / "source_data_graph_model_family_sensitivity.csv"
    )
    learning = pd.read_csv(
        SOURCE_DIR / "source_data_internal_learning_curve.csv"
    )
    learning_contrasts = pd.read_csv(
        SOURCE_DIR / "source_data_internal_learning_curve_contrasts.csv"
    )

    def internal_model(model_id: str) -> pd.Series:
        return internal[
            (internal["protocol"] == "scaffold")
            & (internal["model_id"] == model_id)
        ].iloc[0]

    full = internal_model(FULL)
    chem = internal_model(CHEM)
    has(
        "internal_full",
        (
            f"Spearman correlation of {full.spearman_observed:.3f} "
            f"(95% paired scaffold-bootstrap interval, "
            f"{full.spearman_ci_low:.3f}–{full.spearman_ci_high:.3f}) "
            f"and an RMSE of {full.rmse_observed:.3f} pDC50 units "
            f"({full.rmse_ci_low:.3f}–{full.rmse_ci_high:.3f}"
        ),
    )
    has(
        "internal_chemistry",
        (
            f"chemistry-only ExtraTrees model achieved a Spearman correlation "
            f"of {chem.spearman_observed:.3f} "
            f"({chem.spearman_ci_low:.3f}–{chem.spearman_ci_high:.3f}) "
            f"and an RMSE of {chem.rmse_observed:.3f} "
            f"({chem.rmse_ci_low:.3f}–{chem.rmse_ci_high:.3f})"
        ),
    )

    contrast = internal_contrasts[
        (internal_contrasts["protocol"] == "scaffold")
        & (
            internal_contrasts["contrast_id"]
            == "full_vs_chemistry_extra_trees"
        )
    ].iloc[0]
    has(
        "internal_full_minus_chemistry",
        (
            f"matched full-minus-chemistry contrast was therefore "
            f"{contrast.delta_spearman_observed:.3f} for Spearman "
            f"({contrast.delta_spearman_ci_low:.3f}–"
            f"{contrast.delta_spearman_ci_high:.3f}) and "
            f"{contrast.delta_rmse_observed:.3f} for RMSE improvement "
            f"({contrast.delta_rmse_ci_low:.3f}–"
            f"{contrast.delta_rmse_ci_high:.3f}"
        ),
    )

    def learning_metric(
        fraction: float, model_id: str, metric: str
    ) -> pd.Series:
        return learning[
            (learning["training_fraction"] == fraction)
            & (learning["model_id"] == model_id)
            & (learning["metric"] == metric)
        ].iloc[0]

    chem_25 = learning_metric(0.25, CHEM, "spearman")
    chem_75 = learning_metric(0.75, CHEM, "spearman")
    full_25 = learning_metric(0.25, FULL, "spearman")
    full_75 = learning_metric(0.75, FULL, "spearman")
    has(
        "learning_curve_training_scaffolds",
        "median number of training scaffolds was 134, 267, 401 and 534",
    )
    has(
        "learning_curve_chemistry",
        (
            f"Chemistry-only Spearman correlation rose from "
            f"{chem_25.estimate:.3f} "
            f"({chem_25.ci_low:.3f}–{chem_25.ci_high:.3f}) at 25% to "
            f"{chem_75.estimate:.3f} "
            f"({chem_75.ci_low:.3f}–{chem_75.ci_high:.3f}) at 75%"
        ),
    )
    has(
        "learning_curve_full",
        (
            f"full model rose from {full_25.estimate:.3f} "
            f"({full_25.ci_low:.3f}–{full_25.ci_high:.3f}) to "
            f"{full_75.estimate:.3f} "
            f"({full_75.ci_low:.3f}–{full_75.ci_high:.3f})"
        ),
    )
    final_spearman = {
        row.model_id: row
        for _, row in learning_contrasts[
            learning_contrasts["contrast_id"]
            == "fraction_100_minus_75_spearman"
        ].iterrows()
    }
    for model_id, label in ((CHEM, "chemistry"), (FULL, "full")):
        row = final_spearman[model_id]
        has(
            f"learning_curve_final_spearman_{label}",
            (
                f"{row.estimate:.3f} "
                f"({row.ci_low:.3f} to {row.ci_high:.3f})"
            ),
        )
    has_combined(
        "learning_curve_claim_boundary",
        (
            "does not establish saturation, statistical power or a required "
            "future sample size"
        ),
    )
    learning_qa = json.loads(
        (
            SOURCE_DIR
            / "post_hoc_internal_learning_curve_v1_independent_qa.json"
        ).read_text(encoding="utf-8")
    )
    add(
        "learning_curve_independent_qa",
        (
            learning_qa.get("status") == "PASS"
            and learning_qa.get("checks_passed") == 20
            and learning_qa.get("checks_total") == 20
        ),
        (
            f"status={learning_qa.get('status')}; "
            f"checks={learning_qa.get('checks_passed')}/"
            f"{learning_qa.get('checks_total')}"
        ),
    )

    def ood_model(protocol: str, model_id: str) -> pd.Series:
        return ood[
            (ood["record_type"] == "model")
            & (ood["protocol"] == protocol)
            & (ood["model_id"] == model_id)
        ].iloc[0]

    for protocol, prefix in (
        ("source_ood", "source"),
        ("target_ood", "target"),
    ):
        row = ood_model(protocol, CHEM)
        has(
            f"{prefix}_ood_chemistry",
            (
                f"macro Spearman correlation of "
                f"{row.domain_macro_spearman_observed:.3f} "
                f"({row.domain_macro_spearman_ci_low:.3f}–"
                f"{row.domain_macro_spearman_ci_high:.3f}) and a macro RMSE "
                f"of {row.domain_macro_rmse_observed:.3f} "
                f"({row.domain_macro_rmse_ci_low:.3f}–"
                f"{row.domain_macro_rmse_ci_high:.3f}"
            ),
        )

    for protocol, prefix in (
        ("source_ood", "source"),
        ("target_ood", "target"),
    ):
        row = strict[
            (strict["record_type"] == "contrast")
            & (strict["protocol"] == protocol)
            & (
                strict["contrast_id"]
                == "full_vs_chemistry_extra_trees"
            )
        ].iloc[0]
        has(
            f"strict_{prefix}_contrast",
            (
                f"{row.delta_domain_macro_spearman_observed:.3f} "
                f"({row.delta_domain_macro_spearman_ci_low:.3f} to "
                f"{row.delta_domain_macro_spearman_ci_high:.3f})"
            ),
        )

    hgb_internal = hgb[
        (hgb["record_type"] == "contrast")
        & (hgb["split_regime"] == "internal")
    ].iloc[0]
    has(
        "hgb_internal_contrast",
        (
            f"improved over chemistry-only by "
            f"{hgb_internal.delta_spearman_observed:.3f} in Spearman "
            f"({hgb_internal.delta_spearman_ci_low:.3f}–"
            f"{hgb_internal.delta_spearman_ci_high:.3f})"
        ),
    )

    graph_internal = graph[
        (graph["record_type"] == "contrast")
        & (graph["split_regime"] == "internal")
    ].iloc[0]
    has(
        "graph_internal_contrast",
        (
            f"adding context to the graph model increased Spearman by "
            f"{graph_internal.delta_spearman_observed:.3f} "
            f"({graph_internal.delta_spearman_ci_low:.3f}–"
            f"{graph_internal.delta_spearman_ci_high:.3f})"
        ),
    )
    for protocol, prefix in (
        ("source_ood", "source"),
        ("target_ood", "target"),
    ):
        row = graph[
            (graph["record_type"] == "contrast")
            & (graph["split_regime"] == "domain_plus_compound_cold")
            & (graph["protocol"] == protocol)
        ].iloc[0]
        has(
            f"graph_{prefix}_ood_contrast",
            (
                f"{row.delta_domain_macro_spearman_observed:.3f} "
                f"({row.delta_domain_macro_spearman_ci_low:.3f} to "
                f"{row.delta_domain_macro_spearman_ci_high:.3f})"
            ),
        )

    has(
        "data_counts",
        "1,560 observations, 1,137 canonical compounds and 667 Bemis–Murcko scaffold groups",
    )
    has(
        "fixed_domains",
        "four source and eight target domains",
    )
    has(
        "claim_boundary",
        "It does not yet support prospective utility, calibrated absolute DC50 prediction, candidate-discovery efficacy or mechanistic inference.",
    )
    add(
        "evidence_identity_terms",
        all(
            term in manuscript
            for term in (
                "confirmatory",
                "formal extension",
                "post-hoc sensitivity",
                "exploratory",
            )
        ),
        "all four evidence identities appear",
    )
    add(
        "no_citation_placeholders",
        "[CITATION NEEDED" not in manuscript,
        "no unresolved citation placeholder in manuscript",
    )

    abstract = section(
        manuscript,
        "## Abstract",
        "### Scientific Contribution",
    )
    contribution = section(
        manuscript,
        "### Scientific Contribution",
        "**Keywords:**",
    )
    abstract_words = word_count(abstract)
    contribution_sentences = sentence_count(contribution)
    add(
        "abstract_word_limit",
        0 < abstract_words <= 350,
        f"abstract_words={abstract_words}; limit=350",
    )
    add(
        "scientific_contribution_sentence_limit",
        0 < contribution_sentences <= 3,
        f"sentences={contribution_sentences}; limit=3",
    )
    has_combined(
        "float32_training_response_disclosed",
        "ExtraTrees training-response arrays were constructed explicitly as NumPy `float32`",
    )
    has_combined(
        "atomic_v1_supersession_disclosed",
        "The correction was deliberately atomic",
    )
    has_combined(
        "cross_version_mixing_prohibited",
        "the complete null/applicability/censoring package remains version 3 and may not import any version-1 or version-2 table.",
    )
    has_combined(
        "float32_correction_identity_disclosed",
        FLOAT32_CORRECTION_SHA256,
    )
    has_combined(
        "fixed_domain_correction_identity_disclosed",
        FIXED_DOMAIN_CORRECTION_SHA256,
    )
    has_combined(
        "mixed_lineage_disclosed",
        "The publication lineage is therefore deliberately mixed and is identified as aggregate v4: the corrected held-axis/fit-weight package remains version 2, the complete scaffold/source-deletion package is regenerated as version 3, and the complete null/applicability/censoring package remains version 3",
    )
    has_combined(
        "changing_domain_bootstrap_defect_disclosed",
        "if a small represented domain was absent from a replicate it averaged over the remaining domain labels",
    )
    has_combined(
        "metric_specific_fixed_eligible_domains",
        "version 3 freezes the represented-domain set and then the metric-specific eligible subset",
    )
    has_combined(
        "missing_fixed_domain_repeat_is_na",
        "An absent eligible domain is retained as `NA` with reason `missing_fixed_eligible_domain`",
    )
    has_combined(
        "single_scaffold_macro_interval_structural_na",
        "If any eligible domain contains one scaffold, the descriptive point estimate remains visible, all domain-macro replicates for that metric are structurally not attempted, and the interval is `NA`.",
    )
    has_combined(
        "fixed_domain_unaffected_outputs",
        "The fixed-domain correction leaves every saved prediction, observed pooled or domain-level metric, observed OOD domain-macro point estimate, pooled-row applicability bootstrap, permutation fit and inference result, and censoring/selection audit unchanged.",
    )
    has_combined(
        "unresolved_p0_p1_blocks_aggregation",
        "Any additional publication-blocking (P0) or high-priority (P1) finding would stop aggregation rather than be waived.",
    )
    has(
        "main_prospective_boundary",
        "no publication-eligible prospective validation data were available for this study",
    )
    add(
        "si_prospective_boundary",
        (
            normalized(
                "No publication-eligible prospective validation data were "
                "available or used."
            )
            in normalized(supplement)
        ),
        "SI states that no publication-eligible prospective data were used",
    )
    add(
        "no_restricted_panel_disclosure",
        (
            re.search(r"\bcollaborat(?:or|ion|ive)", combined, flags=re.I)
            is None
            and re.search(
                r"\b96\b.{0,80}\b(?:compound|candidate|panel)s?\b",
                combined,
                flags=re.I,
            )
            is None
        ),
        "no collaboration-specific panel count or composition in manuscript/SI",
    )
    has_combined(
        "selection_parsed_rows_vs_ids",
        "The parsed public-source table contained 3,117 record rows but 3,082 unique `record_id` values.",
    )
    has_combined(
        "selection_component_identifier_unit",
        "The 1,727 component identifiers are neither 1,727 unique parsed rows nor 1,727 modelling observations.",
    )
    has_combined(
        "selection_not_inclusion_probability",
        "are not row-wise attrition rates, inclusion probabilities or evidence that the strict-exact subset is unbiased.",
    )
    has_combined(
        "permutation_finalization_runtime_boundary",
        "The runtime stored by the resumed parent process starts at post-merge finalization and therefore is not the total permutation wall time.",
    )

    result_block = section(manuscript, RESULT_BEGIN, RESULT_END)
    si_result_block = section(supplement, SI_RESULT_BEGIN, SI_RESULT_END)
    add(
        "extension_result_block_layout",
        (
            manuscript.count(RESULT_BEGIN) == 1
            and manuscript.count(RESULT_END) == 1
            and supplement.count(SI_RESULT_BEGIN) == 1
            and supplement.count(SI_RESULT_END) == 1
            and bool(result_block)
            and bool(si_result_block)
        ),
        (
            "unique main/SI blocks: "
            f"main_begin={manuscript.count(RESULT_BEGIN)}, "
            f"main_end={manuscript.count(RESULT_END)}, "
            f"si_begin={supplement.count(SI_RESULT_BEGIN)}, "
            f"si_end={supplement.count(SI_RESULT_END)}"
        ),
    )

    if not EXTENSION_RESULTS.is_file():
        pending_ok = all(
            count == 1
            for count in (
                manuscript.count(RESULT_PENDING),
                manuscript.count(DISCUSSION_PENDING),
                supplement.count(SI_RESULT_PENDING),
            )
        )
        pending_blocks_have_no_old_values = (
            re.search(r"(?<![A-Za-z])[-+−]?\d+\.\d+", result_block)
            is None
            and re.search(
                r"(?<![A-Za-z])[-+−]?\d+\.\d+", si_result_block
            )
            is None
        )
        add(
            "pending_blocks_exclude_numerical_results",
            pending_blocks_have_no_old_values,
            (
                "no decimal result claim occurs in either pending v4 block; "
                "superseded v1 or null-v2 estimates cannot be staged there"
            ),
        )
        add(
            "extension_results_state",
            (
                pending_ok
                and pending_blocks_have_no_old_values
                and args.allow_pending_extension
            ),
            (
                "development-only pending state explicitly allowed; "
                "mixed-lineage v4 aggregate result file absent and v4 sentinels "
                "preserved exactly once"
                if pending_ok
                else "aggregate absent and one or more v4 sentinels missing"
            ),
        )
    else:
        extension = json.loads(
            EXTENSION_RESULTS.read_text(encoding="utf-8")
        )
        try:
            aggregate_binding = aggregate_bundle_hashes(SOURCE_DIR)
            aggregate_binding_error = ""
        except (FileNotFoundError, OSError, TypeError, ValueError) as error:
            aggregate_binding = {}
            aggregate_binding_error = str(error)
        add(
            "extension_v4_aggregate_identity",
            (
                extension.get("analysis_identity")
                == EXTENSION_ANALYSIS_IDENTITY
                and extension.get("source_data_provenance")
                == aggregate_binding
                and EXTENSION_PROVENANCE.is_file()
                and not aggregate_binding_error
            ),
            (
                f"expected={EXTENSION_ANALYSIS_IDENTITY}; "
                f"observed={extension.get('analysis_identity')}; "
                f"binding_error={aggregate_binding_error or 'none'}"
            ),
        )
        add(
            "extension_pending_sentinels_removed",
            all(
                marker not in f"{manuscript}\n{supplement}"
                for marker in (
                    RESULT_PENDING,
                    DISCUSSION_PENDING,
                    SI_RESULT_PENDING,
                )
            ),
            "all pending sentinels removed after aggregate integration",
        )

        for workflow, directory in MIXED_LINEAGE_WORKFLOWS.items():
            identity = WORKFLOW_IDENTITIES[workflow]
            qa_path = directory / "qa_summary.json"
            manifest_path = directory / "run_manifest.json"
            package_ok = qa_path.is_file() and manifest_path.is_file()
            detail = f"qa={qa_path.is_file()} manifest={manifest_path.is_file()}"
            if package_ok:
                qa = json.loads(qa_path.read_text(encoding="utf-8"))
                manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                serialized = json.dumps(manifest, ensure_ascii=False)
                package_ok = (
                    json_status(qa) == "PASS"
                    and manifest.get("protocol_version")
                    == identity["protocol_version"]
                    and "float32" in serialized
                    and FLOAT32_CORRECTION_SHA256 in serialized
                    and not (directory / "SUPERSEDED_DO_NOT_USE.md").exists()
                    and not (directory / "ABORTED_DO_NOT_RESUME.md").exists()
                )
                if workflow == "null_applicability_censoring":
                    package_ok = package_ok and (
                        int(manifest.get("config", {}).get(
                            "n_permutations", 0
                        ))
                        == 100
                        and FIXED_DOMAIN_CORRECTION_SHA256 in serialized
                    )
                detail = (
                    f"qa_status={json_status(qa)}; "
                    f"protocol={manifest.get('protocol_version')}; "
                    f"expected_protocol={identity['protocol_version']}; "
                    "float32 and workflow-specific correction identities required"
                )
            add(
                f"extension_mixed_lineage_package_{workflow}",
                package_ok,
                detail,
            )

        effects = extension.get("effect_records", [])
        permutations = extension.get("permutation_records", [])

        def effect(
            analysis: str, condition: str, metric: str
        ) -> dict[str, Any] | None:
            matches = [
                record
                for record in effects
                if record.get("analysis") == analysis
                and record.get("condition") == condition
                and record.get("metric") == metric
            ]
            return matches[0] if len(matches) == 1 else None

        for analysis, condition, metric in MAIN_EXTENSION_EFFECT_KEYS:
            record = effect(analysis, condition, metric)
            add(
                "extension_main_"
                + re.sub(
                    r"[^a-z0-9]+",
                    "_",
                    f"{analysis}_{condition}_{metric}".lower(),
                ).strip("_"),
                (
                    record is not None
                    and effect_triplet_present(result_block, record)
                ),
                (
                    "one exact aggregate record and its rounded "
                    f"estimate/interval required: {analysis}/{condition}/{metric}"
                ),
            )

        for deletion in ("MGTbind", "MGDB", "MolGlueDB", "TPDdb"):
            record = effect(
                "target_ood_training_source_deletion",
                deletion,
                "delta_spearman",
            )
            add(
                f"extension_si_source_deletion_{deletion.lower()}",
                (
                    record is not None
                    and effect_triplet_present(si_result_block, record)
                ),
                (
                    "SI exact aggregate estimate/interval required for "
                    f"training-source deletion={deletion}"
                ),
            )

        for estimand_id, metric in SI_PERMUTATION_KEYS:
            matches = [
                record
                for record in permutations
                if record.get("estimand_id") == estimand_id
                and record.get("metric") == metric
            ]
            record = matches[0] if len(matches) == 1 else None
            add(
                f"extension_si_permutation_{estimand_id}_{metric}",
                (
                    record is not None
                    and permutation_record_present(si_result_block, record)
                ),
                (
                    "SI observed/null interval/empirical-p record required: "
                    f"{estimand_id}/{metric}"
                ),
            )

        main_decimal_allowlist = aggregate_decimal_allowlist(
            extension,
            main_block=True,
        )
        si_decimal_allowlist = aggregate_decimal_allowlist(
            extension,
            main_block=False,
        )
        unbound_main_decimals = unbound_decimal_claims(
            result_block,
            main_decimal_allowlist,
        )
        unbound_si_decimals = unbound_decimal_claims(
            si_result_block,
            si_decimal_allowlist,
        )
        add(
            "extension_main_decimal_claims_bound",
            not unbound_main_decimals,
            (
                f"unbound={unbound_main_decimals}; "
                f"allowed_token_count={len(main_decimal_allowlist)}"
            ),
        )
        add(
            "extension_si_decimal_claims_bound",
            not unbound_si_decimals,
            (
                f"unbound={unbound_si_decimals}; "
                f"allowed_token_count={len(si_decimal_allowlist)}"
            ),
        )

        censoring = extension.get(
            "censoring_and_randomization_diagnostics", {}
        )
        flow = {
            str(record.get("stage")): int(record.get("n"))
            for record in censoring.get("selection_flow", [])
            if record.get("stage") is not None
            and record.get("n") is not None
        }
        add(
            "extension_censoring_aggregate_identity",
            (
                int(censoring.get("parsed_total", -1)) == 3_117
                and flow.get("all_parsed_records") == 3_117
                and flow.get("core_unique_component_record_ids") == 1_727
                and flow.get("final_aggregated_modeling_rows") == 1_560
            ),
            (
                f"parsed_total={censoring.get('parsed_total')}; "
                f"selection_flow={flow}"
            ),
        )

    bib = REFERENCES.read_text(encoding="utf-8")
    dois = re.findall(r"doi\s*=\s*\{([^}]+)\}", bib, flags=re.I)
    add(
        "reference_count",
        bib.count("@article{") == 19 and len(dois) == 19,
        f"articles={bib.count('@article{')} dois={len(dois)}",
    )
    add(
        "reference_doi_unique",
        len(dois) == len({doi.lower() for doi in dois}),
        f"unique_dois={len(set(doi.lower() for doi in dois))}",
    )
    add(
        "bibtex_braces",
        bib.count("{") == bib.count("}"),
        f"open={bib.count('{')} close={bib.count('}')}",
    )

    failed = [row for row in checks if row["status"] == "FAIL"]
    development_pending = (
        not EXTENSION_RESULTS.is_file()
        and args.allow_pending_extension
        and not failed
    )
    overall_status = (
        "FAIL"
        if failed
        else "DEVELOPMENT_PASS"
        if development_pending
        else "PASS"
    )
    summary = {
        "status": overall_status,
        "submission_ready": overall_status == "PASS",
        "development_pending_extension": development_pending,
        "n_checks": len(checks),
        "n_passed": len(checks) - len(failed),
        "n_failed": len(failed),
        "checks": checks,
    }
    json_path = SOURCE_DIR / "manuscript_evidence_qa_summary.json"
    md_path = SOURCE_DIR / "manuscript_evidence_qa_summary.md"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Manuscript evidence QA",
        "",
        f"Status: **{summary['status']}** "
        f"({summary['n_passed']}/{summary['n_checks']} checks passed).",
        f"Submission ready: **{str(summary['submission_ready']).upper()}**.",
        "",
        "| Check | Status | Evidence |",
        "|---|---|---|",
    ]
    for row in checks:
        evidence = str(row["evidence"]).replace("|", "\\|")
        lines.append(
            f"| {row['check_id']} | {row['status']} | {evidence} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"[{summary['status']}] {summary['n_passed']}/"
        f"{summary['n_checks']} manuscript checks"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
