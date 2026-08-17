# Post-hoc computational extension master protocol v1

Status: **FROZEN BEFORE FORMAL EXTENSION RUNS**

Frozen at: 2026-07-29T18:43:40+08:00

Analysis identity: post-hoc robustness, sensitivity and diagnostic extension.
None of the analyses below is confirmatory or preregistered. The original
confirmatory estimands and results remain unchanged.

## Publication boundary

All model refits use only:

`data/processed/all_molglue_dc50_qc_train_test_standardized_context.csv`

Expected SHA-256:

`e9494246bcbd198bc09d5b0d09721910c33f70f46ae859b76377a13775e3e4b2`

This table contains 1,560 strict-exact observations. Collaborator-supplied
candidate material without publication authorization, including every
derivative of that material, is prohibited as input, calibration material,
model-selection material, evidence, source data or figure content.

The descriptive censoring audit may additionally read:

`data/processed/all_molglue_dc50_parsed_records.csv`

Expected SHA-256:

`ae1a16613b41ce84f92f539dbd31f16ada272ac5e401250acc7e97fd5e8bd970`

It must report aggregate counts only and must not introduce censored records
into a predictive model.

## Scientific question

The extension tests whether the observed internal context increment and the
chemistry-only transfer advantage are robust to:

1. removal of context fields that cannot be carried across the held axis;
2. unequal representation of repeated compounds and source-target domains;
3. a coarser scaffold definition;
4. chemical novelty within the existing validation regimes;
5. deletion of individual training databases; and
6. repeated outcome-label randomization.

It also quantifies how endpoint censoring changes the composition of the
available evidence base.

## Common estimands and reporting

- Response: pDC50.
- Primary ranking metric: Spearman correlation.
- Error metrics: RMSE and MAE in pDC50 units.
- Internal independent resampling unit: global scaffold.
- OOD aggregate: equal-weight mean across the four frozen source domains or
  eight frozen exact-target domains.
- OOD resampling unit: global scaffold, retaining the fixed domain set.
- Intervals: paired 95% percentile scaffold-cluster bootstrap intervals with
  10,000 replicates unless an analysis-specific protocol states otherwise.
- Model comparisons use matched predictions on identical evaluation rows.
- Exact estimates and intervals are reported; no star notation is used.
- Post-hoc comparison families are descriptive. No multiplicity-adjusted
  claim of statistical significance will be made.
- Rows, folds, repeats, domains and bootstrap/permutation replicates are not
  described as biological replicates.

## E1 — Held-axis-portable context

Frozen source-OOD portable context retains recruiter, target, cell line, assay
method, activity time, mode of action and all interactions that do not contain
source identity. It removes the source main effect and source-target
interaction.

Frozen target-OOD portable context retains source, recruiter, cell line, assay
method, activity time and mode of action. It removes the target main effect and
 every interaction containing target identity.

The frozen held-domain splits, compound deletion, ExtraTrees parameter grid,
inner group-disjoint tuning, seeds and test rows are reused. The prespecified
comparisons are portable full versus chemistry-only, original full versus
chemistry-only, and portable full versus original full. A matched portable
context-only ExtraTrees result may be reported as a diagnostic but does not
change the comparison hierarchy.

## E2 — Training-weight sensitivities

The original uniform-row fit is the reference.

### Compound-equal fit

Within every fit fold, each canonical compound receives total weight one:
each row weight is the reciprocal of that compound's number of fit-fold rows.
Weights are then normalized to mean one. Validation metrics remain unweighted
to preserve the original estimand; canonical-compound-aggregated metrics are
reported separately.

### Source-target-domain-balanced fit

The training domain is the exact pair
`source_database × target_protein`. Each row initially receives the reciprocal
of its fit-fold domain size. After normalization to mean one, weights above ten
times the mean are clipped and all weights are renormalized to mean one. The
clipping fraction and effective sample size are reported for every fit.

Only chemistry-only and full ExtraTrees are compared. Hyperparameter selection
uses the same group-disjoint inner folds and the original unweighted Spearman
selection estimand; sample weights affect model fitting only.

## E3 — Alternative scaffold definition

The sensitivity converts each non-empty Bemis-Murcko scaffold to its generic
Murcko form (atom types replaced by carbon and bond types by single bonds).
Acyclic compounds retain a canonical-compound-specific fallback group.

The original repeated 5 × 5 outer and four-fold inner design, seeds, feature
blocks, ExtraTrees grid and metrics are reused with generic scaffold groups.
Train-test generic-scaffold overlap must be zero in every inner and outer
split. Chemistry-only and full ExtraTrees are evaluated. Group count,
singleton count, maximum group size and fold balance are reported.

## E4 — Applicability and chemical novelty

No model is selected or refit for this diagnostic. Existing frozen predictions
are stratified by maximum train-set Morgan Tanimoto similarity using fixed
bins:

- `[0.0, 0.4)`;
- `[0.4, 0.6)`;
- `[0.6, 0.8)`;
- `[0.8, 1.000001]`.

The bins are applied without data-dependent merging to internal
scaffold-disjoint, original source/target OOD and strict source/target OOD
predictions. Sparse or constant bins remain present with an explicit
not-estimable reason. Each bin reports rows, unique compounds, global
scaffolds, chemistry-only and full metrics, and paired full-minus-chemistry
contrasts with scaffold-bootstrap intervals.

## E5 — Training-source deletion in target OOD

For each of the eight frozen target-OOD test domains, training starts from the
original target-OOD training set and then deletes every row whose tokenized
source provenance contains one of the four frozen source labels. Composite
source labels are therefore removed when they contain the deleted token.

Test rows and the held-target/compound-deletion rule remain unchanged.
Chemistry-only and full ExtraTrees are retuned inside the remaining training
set using the original grid and group-disjoint inner logic. Results are
aggregated across the same eight fixed target domains. Any infeasible fit is
retained as an explicit failed stratum rather than silently omitted.

## E6 — Repeated conditional label-randomization control

For each of 100 deterministic seeds, training pDC50 values are independently
permuted within exact `source_database × target_protein` strata inside each
outer training fold. Test labels remain untouched. Singleton strata therefore
remain unchanged and their fraction is reported.

Chemistry-only and full ExtraTrees are refit and predict every frozen internal
scaffold-disjoint test fold. Outer splits and the hyperparameter selected for
the corresponding original-data outer fold are held fixed; the result is a
conditional repeated negative control, not a full retuning-based randomization
test. The observed Spearman, RMSE and full-minus-chemistry contrast are compared
with the 100-member null distributions. Directional empirical tail
probabilities use `(1 + number at least as extreme)/(100 + 1)`.

## E7 — Endpoint-censoring and selection audit

All parsed records are classified as exact, interval/range, one-sided or
missing according to recorded relation fields. Aggregate counts and
proportions are reported overall and by declared source and target, together
with compound/scaffold overlap with the strict-exact core where these
identifiers can be derived reproducibly.

This analysis is descriptive. It does not impute point labels, fit an
interval-censored predictor or claim that the strict-exact subset is an
unbiased sample.

## Excluded scope

The extension deliberately excludes:

- the unauthorized collaborator panel and all derivatives;
- prospective or external-validation claims;
- temporal validation, because publication year is absent for most frozen
  rows and the identifiable subset is strongly source-confounded;
- conformal coverage claims under held-domain exchangeability failure;
- unaudited protein-sequence or protein-language-model features;
- GNN/Transformer/model-zoo expansion that does not test a current reviewer
  risk; and
- interval-censored predictive modelling without a separately audited endpoint
  and target-alias protocol.

These exclusions are scientific scope controls, not evidence that the omitted
analyses would be negative.

## Execution and QA

- CPU only; no GPU is required.
- The project environment is used and package versions are captured.
- Smoke tests precede formal runs.
- Each formal output directory contains configuration, input/script/protocol
  checksums, predictions or sufficient source data, metrics, QA checks and a
  concise results brief.
- Failed or non-estimable strata remain visible.
- Formal outputs are not added to the manuscript or release package until
  independent consistency, restriction-fingerprint and figure-source audits
  pass.
