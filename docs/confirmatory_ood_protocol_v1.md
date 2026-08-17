# Molecular-glue DC50 confirmatory OOD protocol v1

Status: frozen CPU extension protocol

Protocol version: `confirmatory_ood_cpu_v1.0`

Primary analysis date: 2026-07-28

## Scientific questions

This extension asks two deliberately strict questions.

1. **Source OOD:** how well do models rank and calibrate measurements found
   exclusively in a held-out database after removing every training record
   attributable to that database and every training occurrence of a test
   compound?
2. **Target OOD:** how well do models rank and calibrate measurements for a
   held-out target label after removing related composite target labels and
   every training occurrence of a test compound?

Both estimands are therefore domain-plus-compound-cold retrospective internal
validation. They are stricter than source-only or target-only transfer and are
not prospective external validation.

## Frozen data and preprocessing

- Input:
  `data/processed/all_molglue_dc50_qc_train_test_standardized_context.csv`
- Frozen SHA256:
  `e9494246bcbd198bc09d5b0d09721910c33f70f46ae859b76377a13775e3e4b2`
- Endpoint: `pDC50 = 9 - log10(DC50_nM)`.
- Rows: 1,560 strict-exact QC observations.
- Historical `split=train/test` labels are ignored.
- The context sanitization rules and molecular features are imported from
  `scripts/run_confirmatory_cpu_v1.py`.
- Formal-run identity is anchored by the external lock
  `docs/confirmatory_ood_cpu_v1_frozen_identity.json`. The lock is generated
  only after the OOD runner and this protocol are final. It records the frozen
  data, OOD runner, OOD protocol, imported core runner and core protocol
  SHA256 values together with the exact formal arguments. The lock does not
  contain its own hash; the run manifest records that hash at execution time.

Morgan fingerprints and raw RDKit descriptors are molecule-local,
label-independent transformations and may be generated for all rows.
Descriptor imputation, zero-variance filtering and context one-hot encoding
are fitted separately within every inner or final training fold.

## Test-domain definitions and deletion rules

No response value is used to select a domain, construct a split or remove a
row. A plus-delimited source or target label is treated as a set of component
tokens only for training-deletion checks. Test groups always require an exact
standardized label.

### Source OOD

The four test domains are frozen as:

`MGTbind`, `MGDB`, `MolGlueDB`, `TPDdb`.

For held source \(s\):

1. test rows have `source_database == s`;
2. training candidates exclude every row for which
   `s in source_database.split("+")`;
3. final training rows additionally exclude every canonical SMILES present in
   the test rows.

Composite-source rows never serve as test rows. They may be used for training
only when they do not contain the held source and do not contain a test
compound.

| Held source | Test rows | Test SMILES | Test scaffolds | Related composite rows excluded | Candidate-train compound rows excluded | Final train rows | Final train SMILES | Final train scaffolds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MGTbind | 755 | 664 | 394 | 33 | 244 | 528 | 468 | 280 |
| MGDB | 357 | 340 | 205 | 8 | 76 | 1,119 | 797 | 471 |
| MolGlueDB | 289 | 247 | 144 | 39 | 210 | 1,022 | 885 | 539 |
| TPDdb | 119 | 118 | 58 | 0 | 81 | 1,360 | 1,019 | 621 |

The compound purge is essential: before purging, the respective source test
sets contain 166, 28, 108 and 70 rows whose canonical SMILES, target,
recruiter and pDC50 are all reproduced in the remaining databases.

The four exact-source test sets are mutually disjoint and contain 1,520 rows,
1,132 canonical SMILES and 667 scaffolds in total. Forty composite-source rows
are not source-OOD test observations.

### Target OOD

The eight test domains are the frozen sanitized exact labels with at least 60
rows:

`VAV1`, `CSNK1A1`, `GSPT1`, `WIZ`, `CDK2`, `CCNK+CDK12`, `IKZF2`, `IKZF1`.

For held target label \(t\):

1. test rows have `target_protein == t`;
2. let the held tokens be `set(t.split("+"))`;
3. training candidates exclude every row whose target-token set intersects
   the held-token set;
4. final training rows additionally exclude every canonical SMILES present in
   the test rows.

| Held target | Test rows | Test SMILES | Test scaffolds | Related target rows excluded | Candidate-train compound rows excluded | Final train rows | Final train SMILES | Final train scaffolds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VAV1 | 455 | 404 | 230 | 0 | 0 | 1,105 | 733 | 437 |
| CSNK1A1 | 159 | 139 | 68 | 30 | 101 | 1,270 | 998 | 603 |
| GSPT1 | 154 | 106 | 74 | 1 | 48 | 1,357 | 1,031 | 596 |
| WIZ | 150 | 123 | 47 | 0 | 1 | 1,409 | 1,014 | 620 |
| CDK2 | 145 | 145 | 102 | 0 | 0 | 1,415 | 992 | 565 |
| CCNK+CDK12 | 86 | 73 | 52 | 2 | 0 | 1,472 | 1,063 | 614 |
| IKZF2 | 76 | 65 | 32 | 21 | 94 | 1,369 | 1,072 | 635 |
| IKZF1 | 65 | 60 | 34 | 11 | 92 | 1,392 | 1,077 | 643 |

These exact-target test sets are mutually disjoint and contain 1,290 rows,
1,032 canonical SMILES and 601 scaffolds in total.

The implementation fails closed if any frozen count, group label, data hash,
domain-token exclusion or compound-disjointness assertion differs.

## Nested model selection

Each held-out domain is evaluated once. Outer repetitions are not applicable,
and inner folds, tree seeds or model seeds are not independent experimental
units.

Within each final OOD training set:

- hyperparameters are selected by four-fold Bemis--Murcko
  scaffold-disjoint cross-validation;
- folds are balanced using scaffold sizes and random tie breaking only;
- the pooled inner cross-fitted Spearman correlation is the selection metric;
- candidates within 0.005 of the best Spearman use the same regularization,
  RMSE and parameter-ID tie rules as the CPU core;
- all feature preprocessing is fitted using the relevant inner-training fold;
- no hyperparameter selected by the completed core run is reused, because
  those selections were not trained under the relevant OOD deletion rule.

## Frozen CPU model set

1. Global training mean.
2. Hierarchically smoothed context mean.
3. Context-only Ridge.
4. **Context-only ExtraTrees matched learner.**
5. Morgan-Tanimoto k-nearest-neighbour regression.
6. Chemistry-only ExtraTrees.
7. Chemistry-plus-context ExtraTrees.
8. Within-source-target label-shuffled chemistry-plus-context ExtraTrees.

The context-only ExtraTrees model uses exactly the same context matrix,
estimator family, tree count and tree hyperparameter grid as the full
ExtraTrees model. It separates the value of chemistry from a change in learner
capacity.

Frozen grids:

- hierarchical smoothing: `{2, 8, 32}`;
- context Ridge alpha: `{0.1, 1, 10, 100}`;
- Morgan kNN: `k={1,3,5,10,20}`, similarity power `{1,2}`;
- each ExtraTrees learner:
  `n_estimators=600`, `min_samples_leaf={1,5,10}`,
  `max_features={"sqrt",0.3}`, no bootstrap and uniform row weights.

The label-shuffle control reuses the full model configuration selected on
unshuffled OOD-training labels. It remains a diagnostic rather than a formal
permutation p-value.

## Outcomes and contrasts

The primary outcome for each OOD protocol is equal-domain-weighted macro
Spearman:

- four equally weighted source-specific correlations for source OOD;
- eight equally weighted within-target correlations for target OOD.

The primary contrast remains full chemistry-plus-context ExtraTrees versus
context Ridge. The matched-learner contrast, full ExtraTrees versus
context-only ExtraTrees, is prespecified separately.

Other secondary outcomes are:

- equal-domain-weighted macro RMSE;
- pooled Spearman, Pearson, RMSE, MAE, R2 and calibration;
- per-domain metrics;
- source-OOD target-macro and within-target Spearman;
- paired core contrasts against Morgan kNN, chemistry-only ExtraTrees and the
  label-shuffle diagnostic.

Positive `delta_spearman = first - comparator` and positive
`delta_rmse = comparator - first` favour the first model.

A constant prediction has undefined Spearman correlation. Undefined
correlations remain missing and are accompanied by the number of finite
domains; they are never changed to zero. This is expected for some mean-based
models in target OOD.

## Uncertainty

Paired 10,000-replicate percentile confidence intervals are generated by a
global scaffold-cluster bootstrap separately for source OOD and target OOD.

For each bootstrap replicate:

1. unique scaffold IDs in the combined, mutually disjoint test universe are
   sampled with replacement;
2. all rows belonging to each sampled scaffold are included, including rows
   from different held-out domains;
3. the identical sampled row positions are used for every model;
4. per-domain metrics are calculated first and then equally averaged;
5. pooled metrics and all prespecified paired deltas are calculated from the
   same sample.

Global scaffold resampling preserves chemical dependence when a scaffold
appears in more than one held-out domain. Confidence intervals are conditional
on the four frozen sources or eight frozen targets. Per-domain intervals are
exploratory. No p-value is reported, and folds or model seeds are not treated
as `n`.

## Applicability and claim boundary

Every prediction records maximum Morgan similarity to final training data,
scaffold overlap, target support, recruiter-target support and source support.
Exact compound overlap is required to be zero. Scaffold overlap is retained
because the estimand is domain-plus-compound cold, not necessarily
domain-plus-scaffold cold.

Results support only retrospective performance under the frozen internal
domains and deletion rules. They do not establish general performance for all
future databases, targets, scaffolds or prospective experiments.

## Reproducibility and checkpointing

The run manifest binds:

- frozen input hash;
- OOD script and protocol hashes;
- imported core script and core protocol hashes;
- runtime package versions;
- exact domain lists and expected split counts;
- model grids, contrasts, seed schedule and scientific arguments.

Resume is allowed only when this complete scientific configuration matches.
Each domain checkpoint must contain all expected predictions, domain metrics,
36 tuning records, seven selected configurations and four inner-split audit
records. Inner audit rows include deterministic fit/evaluation row and
scaffold digests.

The five checkpoint CSVs are covered by an atomically written inventory that
records the scientific-configuration digest, SHA256, byte size, row count and
column schema of each table. Resume first verifies that inventory and then
validates each complete domain against the frozen rows and reconstructed
split: row identity, response, scaffold and held-domain fields; compound and
domain deletion flags; selected parameters and tuning linkage; prediction and
domain-metric parameter IDs; deterministic inner splits; and domain metrics
recomputed from stored predictions. Missing, modified or scientifically
inconsistent checkpoints fail closed.

Before `status=complete`, every output artifact is reread and inventoried in
the manifest with SHA256, byte size and, for CSV files, row count and column
schema. The manifest is the sole self-referential exception and therefore does
not hash itself.

A run is labelled `confirmatory` only when both its formal arguments and the
observed hashes of the data, OOD runner, OOD protocol, core runner and core
protocol exactly match the external frozen identity. Requesting the formal
arguments with any identity mismatch aborts before model fitting. Custom and
smoke arguments remain executable but cannot receive confirmatory status.

## Execution

Smoke test:

```bash
env/bin/python \
  experiments/260531_molglue_dc50_route/scripts/run_confirmatory_ood_cpu_v1.py \
  --protocols source_ood \
  --inner-folds 2 \
  --n-estimators 10 \
  --bootstrap-replicates 0 \
  --max-groups 1 \
  --n-jobs 4 \
  --output-dir /tmp/confirmatory_ood_cpu_v1_smoke
```

Frozen formal CPU run:

```bash
env/bin/python \
  experiments/260531_molglue_dc50_route/scripts/run_confirmatory_ood_cpu_v1.py \
  --protocols source_ood target_ood \
  --inner-folds 4 \
  --n-estimators 600 \
  --bootstrap-replicates 10000 \
  --n-jobs 10 \
  --output-dir \
  experiments/260531_molglue_dc50_route/reports/confirmatory_ood_cpu_v1
```

If interrupted, rerun the identical command with `--resume`.

After freezing, do not edit the OOD runner, this protocol, either imported core
file or the identity JSON. Any intended scientific change requires a new
protocol version and a newly generated external identity after code review.

This protocol is CPU-only and does not require a GPU.
