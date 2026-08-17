# Manuscript integration contract for mixed-lineage computational extension v4

Status: **active, frozen integration contract**.

Analysis identity: `post_hoc_extension_float32_v4`  
Contributing lineage: context/weight v2 + scaffold/source-deletion v3 +
null/applicability/censoring v3

This contract is the only authoritative manuscript-integration contract for
the corrected extension. `manuscript_v3_extension_integration_contract.md` and
all v1/v2 extension integration instructions are superseded audit records.

## Publication gate

Numerical extension results may replace the pending sentinels in
`manuscript/manuscript_draft_v1.md` and
`manuscript/supplementary_information_v1.md` only after all of the following
conditions hold:

1. context/weight v2, scaffold/source-deletion v3 and
   null/applicability/censoring v3 workflow QA summaries report `PASS`;
2. each formal manifest records its exact protocol, required native-thread or
   fixed-domain execution correction where applicable, a `float32` response
   and global float32-correction SHA-256
   `75a8b0e4f894cead9d72039eab9e6162a89d2a20c79afceda8a2aae1ffd78f3f`;
3. the null/applicability/censoring v3 manifest additionally records
   fixed-domain correction SHA-256
   `9c2d8f90fa546480a37b6e7546cbd9e9a740916325e5dec973f9fa7f8ef899ce`;
4. the aggregate publication source-data builder and independent
   computational-extension cross-check both pass with their exact required
   check-ID sets and frozen script identities; and
5. `computational_extension_results_v1.json` is regenerated from that exact
   aggregate bundle and reports `post_hoc_extension_float32_v4` with matching
   source-data and software-generation bindings.

Any additional publication-blocking (P0) or high-priority (P1) finding stops
aggregation rather than being waived. No result, bootstrap, inference table,
QA record or manifest may be copied from context/weight v1,
scaffold/source-deletion v1 or v2, or null/applicability/censoring v1 or v2.
The compatibility result filename does not relax this boundary.

## Correction and interpretation boundary

All three formal descendants inherit the float32 response correction.
Scaffold/source-deletion v3 additionally binds the frozen native-thread
execution identity. Null/applicability/censoring v3 freezes the represented
and metric-specific eligible domains before its fixed-domain bootstrap:
missing eligible domains make a replicate `NA`, and a one-scaffold eligible
domain leaves the descriptive point visible while making the domain-macro
interval structurally non-estimable.

These are post-hoc sensitivities, diagnostics and repeated negative controls.
They are not prospective validation, calibrated acceptance criteria,
population sampling of databases, causal source effects or mechanistic
evidence.

## Main-text result block

Keep exactly one pair of active boundary markers:

- `COMPUTATIONAL_EXTENSION_V4_RESULTS_BEGIN`
- `COMPUTATIONAL_EXTENSION_V4_RESULTS_END`

Remove `COMPUTATIONAL_EXTENSION_V4_RESULTS_PENDING` only after the bounded
block contains, at minimum, the exact three-decimal estimate and 95% interval
for each aggregate record below:

| Analysis | Condition | Metric |
|---|---|---|
| `held_axis_portable_context` | `source_ood:portable_vs_chemistry` | `delta_domain_macro_spearman` |
| `held_axis_portable_context` | `target_ood:portable_vs_chemistry` | `delta_domain_macro_spearman` |
| `internal_fit_weight` | `compound_equal_full_vs_chemistry` | `delta_spearman` |
| `internal_fit_weight` | `domain_balanced_full_vs_chemistry` | `delta_spearman` |
| `generic_murcko_scaffold` | `full_vs_chemistry` | `delta_spearman` |

Replace `COMPUTATIONAL_EXTENSION_V4_DISCUSSION_PENDING` only with an
aggregate-grounded, bounded interpretation. The main text may describe all
four source-deletion conditions without repeating their values.

## Supplementary result block

Keep exactly one pair of active boundary markers:

- `SI_COMPUTATIONAL_EXTENSION_V4_RESULTS_BEGIN`
- `SI_COMPUTATIONAL_EXTENSION_V4_RESULTS_END`

Remove `SI_COMPUTATIONAL_EXTENSION_V4_RESULTS_PENDING` only after the block
reports:

1. the exact three-decimal estimate and 95% interval for target-OOD
   full-minus-chemistry `delta_spearman` after deleting MGTbind, MGDB,
   MolGlueDB and TPDdb separately;
2. the observed value, conditional-null 2.5th–97.5th percentiles and empirical
   tail probability for chemistry-only Spearman, full-model Spearman and
   full-minus-chemistry `delta_spearman`;
3. represented and metric-specific eligible domains for every applicability
   stratum discussed numerically;
4. all discussed non-estimable fixed-bin results with their recorded reason,
   including missing-fixed-domain repeats and structural single-scaffold
   intervals; and
5. an explicit statement that fixed-bin applicability is heterogeneous
   descriptive evidence, not a calibrated acceptance domain.

## Decimal-claim allowlist

The active main and Supplementary Results blocks are fail-closed numeric
zones. Every decimal token in either block must resolve to a value in the
bound v4 aggregate result record at its declared display precision. The only
non-estimate decimal labels permitted are prespecified bin boundaries and
percentile labels derivable from the same aggregate schema. A rounded number
copied from a superseded result, an invented value, or an extra/conflicting
decimal claim that cannot be resolved to the bound aggregate is a hard QA
failure. Required record-specific estimate/interval and permutation checks
remain mandatory; the allowlist is an additional guard, not a substitute for
them.

While the aggregate result file is absent, the development state remains
valid only when the three v4 pending sentinels occur exactly once and neither
bounded result block contains a decimal result claim.

## Endpoint-selection and runtime units

The selection flow changes units across parsed rows, unique `record_id`
values, component-identifier occurrences, unique component identifiers and
aggregated modelling rows. Ratios between stages are not attrition rates or
inclusion probabilities. A reused identifier does not imply that a censored
endpoint entered model fitting.

For sharded conditional permutations, resumed-parent runtime measures
post-merge finalization only; it is not total wall time, an inferential sample
size or a performance benchmark.

## Claim and disclosure boundary

Use only: “No publication-eligible prospective validation data were available
or used.” Do not disclose the count, identity, composition, prediction,
summary or derived statistic of publication-ineligible collaborator material.

## QA commands

During integration development, while the v4 aggregate result file is absent:

```bash
../../env/bin/python scripts/qa_manuscript_evidence_consistency_v1.py \
  --allow-pending-extension
```

For final manuscript, release and submission QA, omit the development flag:

```bash
../../env/bin/python scripts/qa_manuscript_evidence_consistency_v1.py
```

The final command fails closed on an absent or mismatched v4 aggregate,
remaining sentinel, mixed-lineage package failure, missing required rounded
record, or any unbound decimal token in either active result block.
