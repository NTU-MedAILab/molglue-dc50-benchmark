# Post-hoc internal learning-curve protocol v1

## Status and question

This protocol was frozen on 31 July 2026 after completion of the confirmatory
analyses. It is a post-hoc data-adequacy sensitivity and cannot change the
confirmatory evidence identity.

The question is whether internal scaffold-disjoint prediction continues to
improve as the number of training scaffolds increases. The analysis does not
estimate prospective utility, learning under a future acquisition process, or
the sample size required for a target performance.

## Frozen input and outer evaluation

- Input: the same 1,560-row strict-exact pDC50 analysis table used by the
  confirmatory CPU benchmark.
- Group: the same achiral Bemis-Murcko scaffold identifier, including
  structure-specific identifiers for acyclic molecules.
- Outer evaluation: five scaffold-disjoint folds repeated five times, using
  the confirmatory split generator and seed 260531.
- Test folds are identical across all training fractions and models.
- Test rows and test scaffolds never enter training-subset construction.

## Nested training subsets

Within each outer-training fold, its unique scaffolds are placed in one
deterministic random order using a seed derived only from the frozen base seed,
repeat and outer-fold number. Nested prefixes select:

- 25% of available training scaffolds;
- 50%;
- 75%;
- 100%.

The selected number is the ceiling of fraction times available training
scaffolds, bounded to at least two scaffolds. All rows belonging to a selected
scaffold are retained. The subsets are therefore nested by scaffold but their
row and compound counts are allowed to vary.

## Models

Two matched ExtraTrees regressors are evaluated:

1. `chemistry_extra_trees`: radius-2 2,048-bit Morgan fingerprints with
   chirality plus fold-fitted RDKit two-dimensional descriptors;
2. `full_context_extra_trees`: the same chemistry block plus fold-fitted
   one-hot context main effects and prespecified interactions.

Both use 600 trees, `min_samples_leaf=1`, `max_features="sqrt"`, no bootstrap,
no maximum depth, and a deterministic split/model seed. Hyperparameters remain
fixed at every fraction; no subset-specific or full-data tuning is performed.
Descriptor imputation, zero-variance filtering and context encoding are fitted
only on the selected training subset.

## Estimands and uncertainty

Predictions are averaged over the five repeats for every model, fraction and
row before inference. The independent resampling unit is the scaffold, not the
row, fold, seed or prediction.

For each model and fraction:

- pooled Spearman correlation;
- RMSE on the pDC50 scale;
- median and range of the actual training row, compound and scaffold counts
  across the 25 outer folds.

Paired contrasts are:

- full minus chemistry for Spearman;
- chemistry RMSE minus full RMSE, so positive values favour the full model;
- 100% minus 75% Spearman within each model;
- 75% RMSE minus 100% RMSE within each model, so positive values indicate
  improvement at the largest training size.

All intervals are 95% percentile intervals from 10,000 paired
scaffold-cluster bootstrap resamples. No p values, significance labels,
independent-fold standard errors, extrapolated asymptotes or power calculations
are reported. The four fractions and two metrics form a descriptive post-hoc
profile; no multiplicity-controlled discovery claim is made.

## Outputs and boundaries

The formal output must include a hash-bound run manifest, prediction table,
outer-fold metrics, training-size table, bootstrap summaries, paired
contrasts, an artifact checksum inventory and an independent QA report.

Supported wording is limited to whether the fixed retrospective internal
estimate was still increasing between the observed training fractions.
Plateau, saturation, future-data performance and required sample size must not
be claimed from a confidence interval that includes zero or from visual
inspection alone.

No publication-ineligible prospective or collaborator-restricted material may
be read, described or included.
