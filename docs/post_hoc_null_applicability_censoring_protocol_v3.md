# Post-hoc null, applicability, and censoring audit protocol v3

**Status:** frozen corrected post-hoc sensitivity and descriptive audit  
**Frozen on:** 2026-07-30 (Asia/Shanghai)  
**Compute:** CPU only; no GPU; at most four fitting threads per process  
**Publication scope:** the 1,560-row public-scope frozen core and the
public-source parsed-record tables only. Collaborator-restricted records and
all derivatives of those records are out of scope.

This analysis-specific protocol implements E4, E6, and E7 of
`post_hoc_computational_extension_master_protocol_v1.md`, frozen with SHA-256
`7dee28f45e3ddf8d6299622a8e494c50492b3b2614f39ce8b2fdd2a1be754ff8`.
If wording differs, the master protocol controls scientific scope and this
document controls only additional execution detail.

V3 incorporates two frozen corrections:

1. `post_hoc_float32_response_correction_v2.md`, SHA-256
   `75a8b0e4f894cead9d72039eab9e6162a89d2a20c79afceda8a2aae1ffd78f3f`,
   which aligns the ExtraTrees fitting response with the confirmatory
   `np.float32` implementation; and
2. `post_hoc_applicability_fixed_domain_correction_v3.md`, SHA-256
   `9c2d8f90fa546480a37b6e7546cbd9e9a740916325e5dec973f9fa7f8ef899ce`,
   which keeps the OOD domain-macro bootstrap estimand fixed across global-
   scaffold replicates and refuses an interval when any fixed eligible domain
   has fewer than two unique global scaffold clusters.

All v1 and v2 outputs from this package family are superseded and prohibited
from manuscript, figure, source-data, statistical-audit and release use. They
remain unchanged on disk as audit trails. V3 is regenerated atomically in
independent directories and may not import any v1 or v2 result table.

## Purpose and claim boundary

This protocol adds three reviewer-facing controls without changing the
confirmatory benchmark or its claims:

1. a repeated, full-refit label-permutation null for two matched ExtraTrees
   models;
2. a fixed-bin applicability analysis of the matched context increment; and
3. a descriptive audit of censoring and selection in the source compilation.

All analyses are post hoc. The permutation experiment tests whether observed
cross-fitted scores exceed a conditional no-signal distribution under the
stated exchangeability restriction. It is not evidence of prospective
validity. The applicability analysis describes performance conditional on
training-set chemical similarity; it does not define a calibrated acceptance
domain. The censoring audit is descriptive and does not estimate an
interval-censored outcome model.

## Frozen inputs

- `data/processed/all_molglue_dc50_qc_train_test_standardized_context.csv`
- `data/processed/all_molglue_dc50_parsed_records.csv`
- `reports/confirmatory_cpu_v1/scaffold_cross_fitted_predictions.csv`
- `reports/confirmatory_cpu_v1/scaffold_selected_hyperparameters.csv`
- `reports/confirmatory_cpu_v1/applicability_by_row.csv`
- `reports/confirmatory_ood_cpu_v1/ood_predictions.csv`
- `reports/post_hoc_strict_domain_scaffold_ood_cpu_v1/strict_ood_predictions.csv`

The run manifest records SHA-256 checksums for every input. Original
confirmatory predictions, models and output directories are read-only inputs.

## Analysis A: repeated conditional label-permutation null

### Validation and model fitting

- Reuse the exact 25 outer scaffold-disjoint assignments from the formal
  internal benchmark (five repeats by five folds).
- Reuse selected hyperparameters for `chemistry_extra_trees` and
  `full_context_extra_trees` separately for every outer fold. Do not retune on
  permuted labels.
- Rebuild fold-local chemistry and context matrices with the same frozen
  feature code and sanitization as the confirmatory run.
- In each outer-training fold and permutation, independently permute training
  `pDC50` within exact `source_database × target_protein` strata.
- Construct the complete response once as
  `rows["pDC50"].to_numpy(dtype=np.float32)`. The within-stratum permutation
  must preserve `np.float32`, and the same permuted array must be passed to
  both paired fits.
- Test labels are never permuted or used for fitting.
- Refit both ExtraTrees models, predict the untouched outer test fold and
  average each row's five cross-fitted predictions across repeats before
  permutation-level metrics are computed.

The formal run uses 100 independent permutations, 600 trees per fit, seed
`260531`, five complete repeats and no more than four fitting threads per
process. A smoke run may use one or more permutations, one complete outer
repeat, 20 trees and fewer bootstrap replicates; smoke outputs are
non-inferential and must live in a separate `_v3` directory.

Before the formal run, permutation ID 1 must be computed independently by the
parent-v3 runner and shard-v3 wrapper in the locked project environment using
600 trees, five repeats, four fitting threads and seed `260531`. Model,
contrast and fold-diagnostic tables must match exactly after deterministic key
sorting, including dtypes. This identity check is execution validation, not an
inferential replicate.

### Metrics and inference

For each model and permutation report pooled Spearman correlation, RMSE and
MAE. Positive paired deltas always favor the full model:

- `delta_spearman = Spearman(full) - Spearman(chemistry)`;
- `delta_rmse = RMSE(chemistry) - RMSE(full)`;
- `delta_mae = MAE(chemistry) - MAE(full)`.

For each observed statistic, report null mean, standard deviation and
2.5th/97.5th percentiles. Empirical one-sided p values use
`(1 + count as-or-more-extreme)/(B + 1)`: greater is better for model
Spearman, smaller is better for model RMSE/MAE, and greater is better for each
oriented paired delta. The minimum attainable p value at `B = 100` is `1/101`.
The within-stratum label multiset must be preserved; singleton strata remain
unchanged and their frequency is reported.

## Analysis B: fixed-bin applicability sensitivity

No model is selected or refit. Use existing or exactly reconstructed maximum
test-to-training Morgan Tanimoto values. Outcome-independent bins remain fixed
and are never merged:

1. `[0.0, 0.4)`
2. `[0.4, 0.6)`
3. `[0.6, 0.8)`
4. `[0.8, 1.000001]`

The analysis covers five validation combinations:

- internal repeated scaffold-disjoint CV with repeat-averaged predictions;
- frozen source OOD;
- frozen target OOD;
- strict source-plus-scaffold OOD; and
- strict target-plus-scaffold OOD.

For every regime, protocol and bin, report rows, unique canonical SMILES,
global scaffolds and represented held-out domains. Report Spearman, RMSE and
MAE for the chemistry-only and full-context ExtraTrees models, plus paired
positive-favors-full contrasts.

Internal scaffold-disjoint values are pooled across repeat-averaged rows.
For every original or strict source/target OOD bin, pooled-row metrics remain
a sensitivity and the equal-heldout-domain macro contrast is primary.

### Frozen domain identities

For each OOD `regime × protocol × bin`, freeze the sorted represented-domain
identity sequence before resampling. For each metric, freeze the eligible
subset as represented domains whose observed paired domain contrast is finite.
The observed macro must equal the arithmetic mean of the observed domain
contrasts over exactly that metric-specific eligible set. Empty sets remain
explicitly non-estimable.

Write a long-form eligibility table containing every represented
`stratum × metric × heldout_group`, observed chemistry/full metrics, observed
paired contrast, eligibility flag, unique-global-scaffold count and exclusion
reason. Also write stable compact-JSON identity sequences and their SHA-256
hashes in the paired-delta and bootstrap-audit tables.

### Corrected global-scaffold bootstrap

Paired intervals use 10,000 global-scaffold cluster bootstrap replicates. In
each replicate, sample the observed unique global scaffold labels with
replacement and include all rows belonging to every sampled label. This
retains cross-domain dependence carried by a scaffold.

For pooled-row contrasts, retain the existing paired bootstrap.

For each OOD domain-macro metric:

1. If the metric-specific eligible set is empty, report the interval as `NA`.
2. If any fixed eligible domain has fewer than two observed unique global
   scaffold clusters, retain the descriptive point estimate but report the
   interval as `NA`; all requested macro replicates are structurally not
   attempted for that metric.
3. Otherwise, a replicate is valid only if all fixed eligible domains are
   present and each yields a finite paired contrast for that metric.
4. If an eligible domain is absent, classify the metric replicate as
   `missing_fixed_eligible_domain`.
5. If all eligible domains are present but any eligible-domain contrast is
   non-finite, classify it as
   `fixed_eligible_domain_metric_nonestimable`.
6. Average valid replicate contrasts over exactly the fixed eligible-domain
   set; never over the observed subset in that replicate.

For each metric, requested replicates must equal the sum of valid,
missing-domain-invalid, metric-nonestimable-invalid and
structural-not-attempted replicates. Report all four counts.

The interval is the 2.5th and 97.5th percentiles of complete valid replicates
only. At least `max(100, ceil(0.50 × B))` valid replicates are required; for
the formal run, at least 5,000 of 10,000. Below the threshold the interval is
`NA` with an explicit reason. Thresholds may not be changed after result
inspection.

Sparse or constant bins remain present with explicit reasons. The analysis is
a stratified performance description, not a causal interaction test. No
multiplicity-adjusted significance claim is made from an individual bin.

## Analysis C: censoring and selection-bias description

Classify every row in `all_molglue_dc50_parsed_records.csv` as equality,
range, one-sided or missing/unparsed from the recorded relation field. Report
counts and fractions overall, by declared source and by target. Report a
transparent processing flow from parsed records to equality records and final
aggregated strict-exact modeling rows. Because final rows aggregate repeated
and cross-source records, ratios across changing units are descriptive
representation summaries, not inclusion probabilities.

Where canonical structures parse reproducibly, report compound and global
Bemis--Murcko-scaffold overlap between each relation class and the strict-exact
core. Invalid or missing structures remain in denominators and are reported.
Censored/range/missing rows are never added to fitting or evaluation. No
midpoint substitution, bound substitution or interval-censored regression is
performed.

## Required QA and provenance

The run must fail if:

- the complete or permuted training response is not `np.float32`;
- any core row or split key is duplicated unexpectedly;
- formal split assignments disagree with saved predictions;
- a train/test scaffold overlap is detected;
- selected hyperparameters are missing or duplicated;
- paired applicability rows disagree in truth, scaffold or similarity;
- a similarity value lies outside `[0, 1.000001]`;
- a represented/eligible domain identity hash is inconsistent;
- any observed domain macro differs from the fixed eligibility-table mean;
- bootstrap counts do not reconcile to the requested `B`;
- an OOD macro replicate averages a changing domain set;
- a domain-macro interval is emitted with fewer than the frozen minimum valid
  replicates;
- a domain-macro interval is emitted when any eligible domain has fewer than
  two unique global scaffold clusters;
- a relation operator is silently dropped;
- a formal run contains fewer than 100 complete permutations;
- any requested prediction is non-finite without an explicit reason; or
- `n_jobs` exceeds four.

Outputs include configuration, software versions, runtime, input and artifact
checksums, machine-readable QA, eligibility and bootstrap-audit tables, and a
concise results brief.

## Version and execution boundary

V3 parent and shard output directories must end in `_v3`. They reject
`SUPERSEDED_DO_NOT_USE.md`, `ABORTED_DO_NOT_RESUME.md` and any existing
manifest with another protocol version. V3 resume is allowed only when the
existing manifest exactly matches protocol identities and full scientific
configuration. The merge helper accepts only the frozen formal v3 primary
directory and complete v3 shards whose parent, wrapper, protocol, corrections,
feature manifest and configuration hashes match.

V3 must not resume, merge, overwrite or copy any permutation or applicability
table from v1 or v2. No existing formal result, manuscript, figure, release
file or superseded audit artifact is modified by the v3 run.
