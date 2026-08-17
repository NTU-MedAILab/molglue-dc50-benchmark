# Applicability fixed-domain bootstrap correction v3

**Status:** frozen statistical implementation correction  
**Frozen on:** 2026-07-30 (Asia/Shanghai)  
**Scope:** OOD domain-macro confidence intervals in the fixed-bin
applicability analysis only  
**Compute:** CPU only; no GPU

## Reason for correction

The computational-extension master protocol freezes the held-out-domain set
for OOD aggregation and specifies global-scaffold cluster resampling. The v2
applicability implementation sampled global scaffold clusters correctly, but
then recomputed each bootstrap domain macro over only the domain labels that
happened to remain in that replicate. If a small represented domain received
no sampled scaffold, the replicate silently averaged a smaller domain set.
That changed the estimand across replicates and made the resulting
domain-macro percentile interval ineligible for publication use.

This is a P1 interval-estimation defect, not a response-dtype defect and not a
model-fitting defect. It does not change:

- any saved prediction;
- any observed pooled or domain-level metric;
- any observed OOD domain-macro point estimate;
- any pooled-row applicability bootstrap;
- any permutation fit, permutation point estimate, null distribution or
  empirical p value;
- the censoring/selection audit; or
- the public-scope input boundary.

## Corrected estimand

For each frozen
`regime × protocol × similarity_bin × metric` combination:

1. Freeze the observed **represented-domain set** before resampling.
2. Freeze the metric-specific **eligible-domain set** as the represented
   domains for which the observed paired domain contrast is finite. The paired
   contrasts are `full - chemistry` for Spearman and
   `chemistry - full` for RMSE and MAE.
3. Define the observed domain-macro contrast as the equal-weight arithmetic
   mean over exactly that metric-specific eligible-domain set.
4. Draw global scaffold clusters with replacement, using the same number of
   draws as observed unique global scaffolds. A sampled scaffold contributes
   all of its rows across all held-out domains, retaining dependence induced
   by a scaffold shared across domains.
5. For a given metric, accept a bootstrap replicate only if every member of
   the frozen eligible-domain set is represented and yields a finite paired
   domain contrast in that replicate. Average over exactly the frozen eligible
   set.
6. Before resampling, require every fixed eligible domain to contain at least
   two unique global scaffold clusters. If any eligible domain contains fewer
   than two clusters, retain the descriptive point estimate but report the
   domain-macro interval as non-estimable. Do not construct a conditional
   interval from the subset of replicates in which its single scaffold happens
   to be sampled.
7. If any eligible domain is absent, record the metric replicate as `NA` with
   reason `missing_fixed_eligible_domain`. If all eligible domains are present
   but any paired domain contrast is not estimable, record it as `NA` with
   reason `fixed_eligible_domain_metric_nonestimable`.
8. Construct the percentile interval from complete finite replicates only.
   Report requested, valid, missing-domain-invalid and
   metric-nonestimable-invalid replicate counts separately. When the
   fewer-than-two-scaffold rule applies, report all requested macro replicates
   as structurally not attempted in a separate count.

The invalid-reason counts are mutually exclusive: a replicate with a missing
eligible domain is counted as missing-domain invalid before metric
estimability is assessed. Requested replicates reconcile to valid,
missing-domain invalid, metric-nonestimable invalid and
structural-not-attempted counts.

## Minimum-valid rule

The minimum number of complete replicates is frozen as

`max(100, ceil(0.50 × B))`,

where `B` is the requested number of bootstrap replicates. For the formal
10,000-replicate analysis, at least 5,000 complete replicates are required.
Below this threshold, the point estimate and all diagnostics remain visible,
but the affected interval is reported as `NA` with an explicit reason. The
threshold must not be changed after inspecting results.

## Why domain-stratified resampling is not the primary correction

Sampling independently within every held-out domain would guarantee domain
presence, but it would break the joint resampling of a global scaffold that
appears in more than one domain. That would change the dependence model stated
in the master protocol and could understate or otherwise alter uncertainty.
The corrected primary analysis therefore keeps the global-scaffold bootstrap
and treats incomplete fixed-domain replicates as invalid. Domain-stratified
resampling is not used as a fallback.

## Audit outputs and fail-closed checks

The v3 package must emit:

- one eligibility row for every observed represented
  `stratum × metric × heldout_group`, including observed chemistry and full
  metrics, paired contrast, eligibility status, unique-scaffold count,
  sparse-domain flag and non-estimability reason;
- stable JSON sequences and SHA-256 identities for each represented-domain
  set and metric-specific eligible-domain set;
- per-metric counts of requested, valid, missing-domain-invalid and
  metric-nonestimable-invalid bootstrap replicates, plus structurally
  not-attempted replicates;
- the identities of any eligible domains with fewer than two unique global
  scaffold clusters;
- the frozen minimum-valid threshold and interval eligibility;
- a per-stratum machine-readable audit proving that the reported observed
  macro is the equal-weight mean over exactly the frozen eligible set and that
  all bootstrap counts reconcile to `B`.

The run must fail if an identity hash is inconsistent, an observed macro does
not reproduce from the frozen eligibility table, invalid-reason counts do not
sum to `B`, an interval is emitted below the minimum-valid threshold, or an
eligible-domain set changes within bootstrap resampling. It must also fail if
an interval is emitted when any fixed eligible domain has fewer than two
unique global scaffold clusters.

## Version migration

The following v2 implementation identities are superseded for this package
family:

| Artifact | Superseded SHA-256 |
|---|---|
| parent runner | `02a94d3219fec67fad09d26bd5fee9b43de7bed56379b58adc1ab333b52e4b3e` |
| permutation shard wrapper | `2bb41cc10c43633ad241a19d8fa3171d56ac07efcb490819016a838821c920c8` |
| merge helper | `22e1754d8f82f27bad630ad38dbc1e9f2f1459667f9ec84cd556d4491e55022f` |
| local protocol v2 | `786a5c532ea1431db8d5db38dd99bb3dfa00f3221ac06786333ced4f6ff1f6f9` |

The interrupted formal v2 parent contained 19 of 20 applicability strata and
no completed primary permutation IDs. Each of the three interrupted formal v2
shards contained four permutation IDs. The parent directory and all three
shards are retained with `SUPERSEDED_DO_NOT_USE.md` markers. They must not be
resumed, merged, copied into v3 or used in manuscript, figures, source data,
statistical audit or release material.

The complete replacement is independently versioned as
`post_hoc_null_applicability_censoring_v3.0`. V3 parent, shard and merge tools
accept only directory names ending in `_v3` and reject superseded or aborted
markers. No v1 or v2 table may be imported into a v3 output.

## Publication and data boundary

The correction uses only the frozen 1,560-row public-scope modeling core,
existing public-scope saved predictions and the public-source parsed-record
table permitted by the master protocol. Collaborator-restricted material and
all derivatives remain prohibited and are not accessed.
