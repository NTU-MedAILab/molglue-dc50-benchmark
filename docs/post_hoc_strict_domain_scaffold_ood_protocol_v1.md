# Post-hoc strict domain-plus-scaffold-cold OOD sensitivity protocol v1

## Status and scope

This document defines a **post-hoc sensitivity analysis**, not a confirmatory
analysis and not an amendment to the frozen confirmatory OOD protocol. It asks
whether the matched ExtraTrees feature-block comparisons remain directionally
similar after imposing a stricter training exclusion:

1. reproduce each frozen domain-plus-compound-cold OOD split;
2. retain its exact test rows;
3. additionally remove every training row whose Bemis--Murcko scaffold occurs
   anywhere in that held-out test domain.

The frozen core runner, frozen OOD runner, their protocols, identity files and
result directories are read-only parents. This extension writes only to its
own output directory.

## Frozen parents and endpoint

- Input dataset:
  `data/processed/all_molglue_dc50_qc_train_test_standardized_context.csv`
- Endpoint: `pDC50 = 9 - log10(DC50_nM)`.
- Frozen core protocol: `confirmatory_cpu_v1.1`.
- Frozen OOD protocol: `confirmatory_ood_cpu_v1.0`.
- Source-OOD domains, in frozen order:
  `MGTbind`, `MGDB`, `MolGlueDB`, `TPDdb`.
- Target-OOD domains, in frozen order:
  `VAV1`, `CSNK1A1`, `GSPT1`, `WIZ`, `CDK2`, `CCNK+CDK12`,
  `IKZF2`, `IKZF1`.
- Global seed: `260531`.

An external JSON identity file freezes the data, this script and protocol, the
core and OOD scripts and protocols, the frozen OOD identity file, the strict
split expectations and the formal arguments. A hash mismatch fails closed.

## Split construction

For each of the same 12 held-out domains:

1. call the frozen OOD split builder, which:
   - selects the exact standardized held-out domain as test;
   - deletes from training rows with held-domain token membership;
   - deletes from training every occurrence of a test canonical SMILES;
   - verifies the frozen domain-plus-compound-cold counts;
2. collect all `scaffold_id` values in the unchanged test set;
3. delete from the frozen base training set every row whose `scaffold_id`
   belongs to that test-scaffold set;
4. assert zero row, canonical-SMILES, held-domain-token and scaffold overlap.

The scaffold identifier is the same deterministic Bemis--Murcko identifier
used by the frozen core/OOD runners. The exact base-train, test, scaffold-
excluded and strict-train index sequences are frozen by SHA-256.

### Frozen strict split counts

| Axis | Held-out domain | Test rows | Test scaffolds | Base train rows | Rows deleted by scaffold | Strict train rows | Strict train unique SMILES | Strict train scaffolds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| source | MGTbind | 755 | 394 | 528 | 17 | 511 | 458 | 273 |
| source | MGDB | 357 | 205 | 1,119 | 37 | 1,082 | 770 | 462 |
| source | MolGlueDB | 289 | 144 | 1,022 | 65 | 957 | 850 | 523 |
| source | TPDdb | 119 | 58 | 1,360 | 92 | 1,268 | 943 | 609 |
| target | VAV1 | 455 | 230 | 1,105 | 0 | 1,105 | 733 | 437 |
| target | CSNK1A1 | 159 | 68 | 1,270 | 17 | 1,253 | 988 | 599 |
| target | GSPT1 | 154 | 74 | 1,357 | 27 | 1,330 | 1,021 | 593 |
| target | WIZ | 150 | 47 | 1,409 | 0 | 1,409 | 1,014 | 620 |
| target | CDK2 | 145 | 102 | 1,415 | 0 | 1,415 | 992 | 565 |
| target | CCNK+CDK12 | 86 | 52 | 1,472 | 0 | 1,472 | 1,063 | 614 |
| target | IKZF2 | 76 | 32 | 1,369 | 0 | 1,369 | 1,072 | 635 |
| target | IKZF1 | 65 | 34 | 1,392 | 37 | 1,355 | 1,050 | 633 |

Five domains already have zero test-scaffold occurrence in the frozen base
training set. Their strict and base training sets are therefore identical;
they remain in the analysis to keep the domain set unchanged.

## Matched models and features

Exactly three `ExtraTreesRegressor` models are independently selected and
refit in every strict split:

1. `context_extra_trees`: fold-fitted one-hot experimental context only;
2. `chemistry_extra_trees`: Morgan fingerprint plus fold-prepared RDKit
   descriptors;
3. `full_context_extra_trees`: the same chemistry block plus the same context
   block.

Common settings:

- `n_estimators = 600`;
- `max_depth = None`;
- `bootstrap = False`;
- `n_jobs = 10`;
- six candidates: `min_samples_leaf` in `{1, 5, 10}` crossed with
  `max_features` in `{"sqrt", 0.3}`.
- final-refit random seeds retain the frozen parent OOD model positions:
  `context_extra_trees = 3`, `chemistry_extra_trees = 5`, and
  `full_context_extra_trees = 6`. Consequently, a domain whose strict
  training set is unchanged has the same inner selection and final-refit seed
  as its parent OOD counterpart; scaffold deletion is the only intended
  perturbation.

No target encoding, protein-language-model embedding, target-local routing,
saved fitted model or response-derived parent prediction is used.

## Independent inner selection

Each strict training set receives a new four-fold scaffold-disjoint inner
partition. All six candidates for each of the three models produce pooled
inner out-of-fold predictions.

- Selection metric: pooled Spearman correlation.
- Regularization rule: retain candidates within `0.005` of the best Spearman;
  choose the more regularized candidate (larger leaf, then `sqrt`), then lower
  RMSE, then lexicographic parameter ID.
- The test rows and test outcomes never enter selection.
- Every inner training row must receive exactly one prediction per candidate.

Folds and seeds are computational partitions, not independent sample size.

## Estimands and uncertainty

Metrics are reported for every held-out domain and aggregated separately over
the source-OOD and target-OOD axes:

- equal-domain-weighted macro Spearman;
- equal-domain-weighted macro RMSE;
- pooled Spearman;
- pooled RMSE.

Two feature-block contrasts are evaluated:

1. `full_vs_context_extra_trees_matched`;
2. `full_vs_chemistry_extra_trees`.

For Spearman, delta is `full - comparator`; for RMSE, delta is
`comparator - full`. Positive values favor the full model.

Paired uncertainty uses 10,000 resamples of the union of test
Bemis--Murcko scaffolds within each axis (source or target). A sampled scaffold
contributes all of its rows, including rows in different held-out domains.
The same sampled positions are applied to all three models. Percentile 95%
intervals are descriptive sensitivity intervals.

Because the analysis and its two contrasts were specified after inspection of
the core project, no p values or confirmatory significance claims are made.
All model and contrast rows are reported; no result-dependent filtering is
allowed.

## Reproducibility, checkpointing and completeness

The formal command is:

```bash
env/bin/python \
  experiments/260531_molglue_dc50_route/scripts/run_post_hoc_strict_domain_scaffold_ood_cpu_v1.py \
  --protocols source_ood target_ood \
  --inner-folds 4 \
  --n-estimators 600 \
  --n-jobs 10 \
  --bootstrap-replicates 10000 \
  --seed 260531 \
  --output-dir experiments/260531_molglue_dc50_route/reports/post_hoc_strict_domain_scaffold_ood_cpu_v1
```

After every completed domain, the five progress tables are atomically written
and hashed in a checkpoint inventory. `--resume` first verifies the manifest,
scientific-configuration digest, file hashes, schemas and split completeness.
A partially represented domain is discarded and recomputed; altered
checkpoints are rejected.

Formal expected rows:

- predictions: `8,430` (`2,810` test rows × 3 models);
- domain metrics: `36`;
- inner tuning: `216` (`12 × 3 × 6`);
- selected hyperparameters: `36`;
- inner split audit: `48`;
- aggregate metrics: `6`;
- paired global-scaffold bootstrap: `10`
  (three model rows plus two contrast rows for each axis).

The completed manifest must always label the run
`post_hoc_sensitivity`; it must never use `confirmatory`.

## Interpretation boundaries

- This is retrospective internal OOD sensitivity analysis, not external or
  prospective validation.
- The strict deletion evaluates extrapolation beyond test-domain scaffolds,
  but does not prove generalization to all unseen chemical series.
- Source and target axes contain only four and eight held-out domains,
  respectively; domain-macro estimates may be imprecise.
- Multiple rows can share compounds, scaffolds and experimental contexts;
  the global-scaffold bootstrap addresses scaffold clustering only.
- The five unchanged domains are not evidence that the stricter rule has no
  effect generally; they simply had no scaffold overlap to delete.
- GPU hardware is neither used nor required.
