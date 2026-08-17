# Molecular-glue DC50 confirmatory protocol v1 — public redacted copy

Status: public redacted copy of the frozen CPU core protocol  
Redaction scope: two non-computational references to confidential third-party
material outside the study were removed; the original frozen protocol SHA-256
and this release-copy SHA-256 are recorded in `SCIENTIFIC_IDENTITY.json`.  
Protocol version: `confirmatory_cpu_v1.1`  
Primary analysis date: 2026-07-28

## Scientific question

Does molecular structure add reproducible ranking information beyond database,
target and assay-context priors, and under which chemical domains does that
information remain useful?

The protocol does not aim to maximize a single retrospective test-set score.
It estimates performance, uncertainty and applicability boundaries under
predefined distribution shifts.

## Data

- Input:
  `data/processed/all_molglue_dc50_qc_train_test_standardized_context.csv`
- Frozen input SHA256:
  `e9494246bcbd198bc09d5b0d09721910c33f70f46ae859b76377a13775e3e4b2`
- Endpoint: `pDC50 = 9 - log10(DC50_nM)`.
- Primary data scope: 1,560 strict-exact QC rows.
- Historical `split=train/test` labels are ignored.
- The historical 308/305-row fixed test is treated only as retrospective
  exploratory evidence because it has been repeatedly inspected.
- ChEMBL candidates and weak/censored labels are outside the CPU v1 primary
  analysis. No non-study confidential third-party material is used.

No row may be excluded on the basis of model error. Obvious non-protein target
tokens and non-cell reporter text are sanitized by fixed rules and written to a
row-level audit file; the source table remains unchanged.

## Independent units

- Compound protocol: canonical SMILES group.
- Primary scaffold protocol: Bemis--Murcko scaffold group.
- Folds, repeats and model seeds are not independent experimental units.
- Every result reports rows, unique canonical SMILES and scaffolds.

## Validation

### Primary: scaffold-disjoint nested cross-validation

- Outer evaluation: five folds, repeated five times.
- Inner model selection: four scaffold-disjoint folds inside each
  outer-training set.
- Scaffold: RDKit Bemis--Murcko scaffold, chirality excluded.
- Acyclic molecules receive a structure-specific acyclic group rather than one
  shared empty-scaffold group.
- Folds are balanced using group sizes and random tie-breaking only; response
  values are not used to construct folds.
- Every row receives one outer-fold prediction per repeat. The five
  cross-fitted predictions are averaged row-wise before calculating the
  confirmatory point estimate or confidence interval.

### Secondary: compound-disjoint nested cross-validation

- Outer evaluation: five folds, repeated five times.
- Inner model selection: four canonical-SMILES-disjoint folds.
- This estimates interpolation to unseen compounds that may share a scaffold
  with training compounds.

Source- and target-OOD protocols will be added after the primary CPU run passes
QA. Temporal OOD is not part of v1 because `activity_time` records assay
exposure duration rather than publication year.

## Frozen CPU model set

1. Global training-fold mean.
2. Hierarchically smoothed context means using target, recruiter-target,
   source-target, cell-target and assay-target groups.
3. Context-only one-hot Ridge regression.
4. Morgan-Tanimoto k-nearest-neighbour regression. Training labels are first
   aggregated to the median within canonical SMILES, so repeated measurements
   cannot occupy multiple neighbour slots.
5. Chemistry-only ExtraTrees using Morgan fingerprints and RDKit 2D
   descriptors.
6. Chemistry-plus-context ExtraTrees.
7. Within-source-target label-shuffle sanity control using the selected full
   ExtraTrees configuration.

Context encoding and descriptor imputation are fitted separately inside every
training fold. Target encoding is excluded from the CPU core because the
historical implementation encodes each training row using statistics that
include its own label.

### Hyperparameters

- Hierarchical mean smoothing: `{2, 8, 32}`.
- Context Ridge alpha: `{0.1, 1, 10, 100}`.
- Morgan kNN: `k={1,3,5,10,20}`, similarity power `{1,2}`.
- ExtraTrees:
  - `n_estimators=600`;
  - `min_samples_leaf={1,5,10}`;
  - `max_features={"sqrt",0.3}`;
  - uniform sample weights;
  - no bootstrap.

The inner pooled Spearman correlation selects each model configuration. Among
configurations within 0.005 of the best Spearman value, the more regularized
configuration is selected; remaining ties use lower RMSE and then parameter
ID. Regularization order is larger smoothing, larger Ridge alpha, larger k
then lower similarity power, and larger ExtraTrees leaf size then `sqrt`
feature sampling. No outer-fold response is used for model selection.

The label-shuffle model reuses hyperparameters selected on the unshuffled
training labels. It is a diagnostic sanity control, not a formal permutation
null distribution or p-value. Each outer-fold report records the number of
permutation strata, singleton fraction, changed-label fraction and correlation
between original and permuted labels.

## Features

- Morgan fingerprint: radius 2, 2,048 bits, chirality included.
- RDKit 2D descriptors:
  - descriptor names and RDKit version are recorded;
  - non-finite descriptor values remain missing until a fold is fitted;
  - fit-fold median imputation in float64;
  - fit-fold float64 variance calculation and zero-variance removal.
- Context main effects:
  source, recruiter, target, cell line, assay, activity time and mode of action.
- Context interactions:
  source-target, recruiter-target, target-cell and assay-target.

Protein-language-model features are excluded pending a target/recruiter to
UniProt mapping audit. Reusing existing embeddings does not require a GPU, but
rebuilding corrected embeddings may benefit from one.

## Outcomes and uncertainty

- Primary outcome: pooled scaffold-OOF Spearman correlation.
- Prespecified primary contrast:
  full chemistry-plus-context model versus context-only Ridge.
- Secondary outcomes:
  Pearson correlation, RMSE, MAE, R2, calibration slope/intercept,
  target-macro Spearman and within-target Spearman. Calibration is reported as
  `observed = intercept + slope × predicted`, with ideal intercept 0 and slope
  1.
- Sensitivity outcome:
  metrics after taking the median observed and predicted value within each
  canonical SMILES. This is a sensitivity analysis, not the primary unit of
  analysis.
- Uncertainty:
  paired 10,000-replicate scaffold-cluster percentile bootstrap applied to the
  row-wise mean of the five OOF predictions.
- Effect direction:
  positive `delta_spearman = full - comparator` and positive
  `delta_rmse = comparator - full` favour the full model.
- Prespecified paired contrasts:
  1. full chemistry-plus-context ExtraTrees versus context Ridge (primary);
  2. chemistry-only ExtraTrees versus Morgan kNN;
  3. full chemistry-plus-context ExtraTrees versus chemistry-only ExtraTrees;
  4. full chemistry-plus-context ExtraTrees versus the label-shuffle sanity
     control.
  For every contrast, positive `delta_spearman = first - comparator` and
  positive `delta_rmse = comparator - first` favour the first model.
- The analysis emphasizes effect sizes and confidence intervals rather than
  treating a p-value threshold as the scientific conclusion.

## Applicability information

Every cross-fitted prediction records:

- maximum Morgan Tanimoto similarity to its outer-training fold;
- whether its scaffold occurs in the outer-training fold;
- target support in the outer-training fold;
- whether its source occurs in the outer-training fold.

Risk-coverage analysis and source/target OOD extensions will use these
training-only quantities. Reliability thresholds may not be selected using
outer-test labels.

The CPU v1 run additionally freezes a Tanimoto-based risk-coverage analysis at
100%, 80%, 60%, 40% and 20% coverage. Rows are ranked without evaluation
labels by mean outer-training maximum Tanimoto similarity, then minimum target
support and row index. The per-row table also reports
`|full ExtraTrees - Morgan kNN|` and
`|full ExtraTrees - context Ridge|` as model-disagreement diagnostics; these
diagnostics do not select the reported coverage subsets.

## Claim boundary

The analysis is a frozen-protocol retrospective internal validation. It is not
prospective external validation, and no non-study confidential third-party
material or derivative contributes to its estimates or claims.

The run manifest binds results to the input-data SHA256, analysis-script
SHA256, protocol-document SHA256, package versions and scientific arguments.
Interrupted runs may use `--resume`; a checkpoint is reused only when these
identities match and all predictions, metrics and tuning records for an outer
fold are complete.

## Execution

Smoke test:

```bash
env/bin/python \
  experiments/260531_molglue_dc50_route/scripts/run_confirmatory_cpu_v1.py \
  --protocols scaffold \
  --outer-folds 3 \
  --outer-repeats 1 \
  --inner-folds 2 \
  --n-estimators 40 \
  --bootstrap-replicates 0 \
  --max-outer-splits 1 \
  --n-jobs 8 \
  --output-dir experiments/260531_molglue_dc50_route/reports/confirmatory_cpu_v1_smoke
```

Frozen primary CPU run:

```bash
env/bin/python \
  experiments/260531_molglue_dc50_route/scripts/run_confirmatory_cpu_v1.py \
  --protocols scaffold compound \
  --outer-folds 5 \
  --outer-repeats 5 \
  --inner-folds 4 \
  --n-estimators 600 \
  --bootstrap-replicates 10000 \
  --n-jobs 10 \
  --output-dir experiments/260531_molglue_dc50_route/reports/confirmatory_cpu_v1
```

If the formal run is interrupted, rerun the same command with `--resume`.
Changing data, code, this protocol document, runtime package versions or any
scientific argument causes resume to fail closed.

This CPU core does not require a GPU. A GPU is considered only if the audited
protein-to-UniProt map is later used to regenerate protein embeddings or to
fine-tune a neural representation model.
