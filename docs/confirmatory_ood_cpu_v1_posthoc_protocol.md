# Confirmatory OOD CPU v1 post-hoc diagnostics protocol

Status: exploratory, no-retraining sensitivity analysis

Protocol version: `confirmatory_ood_cpu_v1_posthoc.1.0`

## Purpose and identity

This package derives descriptive sensitivity analyses from the completed,
frozen formal OOD predictions. It does not fit, tune, select, recalibrate or
otherwise change a model.

Bound formal inputs:

- formal manifest SHA256:
  `1692bd2bcf729a8d099875bc1d640f351e30691afa480f54a42e8a99cadf4b65`;
- formal scientific-configuration SHA256:
  `fd10b444eb205af2a150e81b19b281301d0bc0a6c28cf8a30892297ea8a0ceab`;
- `ood_predictions.csv` SHA256:
  `0af39d248adff5cd17fc6d21cce1bfbd0c38051a740d14019d2e80b49c0cef5a`.

The standalone runner verifies these identities, the completed confirmatory
state and every formal artifact in the parent manifest before reading the
predictions. It snapshots all files in the formal result directory and
requires the snapshot to remain unchanged after output generation. It refuses
to write inside the formal result directory.

## Analysis 1: canonical-SMILES aggregation

Within each `protocol + heldout_group + model_id + canonical_smiles` group:

- `y_true` is replaced by its median;
- `y_pred` is replaced by its median;
- the molecule-local scaffold ID must be unique;
- the number of contributing source rows is retained.

Metrics are first calculated separately within every frozen held-out domain.
Domain-macro metrics are equal-weight means of the domain estimates and are
reported only when all frozen domains have finite estimates. Pooled metrics
are calculated over the domain-specific aggregated units. A canonical SMILES
appearing in different held-out domains remains a separate analysis unit in
each domain.

This analysis tests sensitivity to repeated-record weighting. It may combine
different experimental contexts for the same molecule within one held-out
domain, so it is not a replacement estimand for the formal row-level result.
No new confidence interval is calculated for these aggregated point estimates.

## Analysis 2: leave-one-domain-out influence

For full-context ExtraTrees and chemistry-only ExtraTrees, domain-macro
Spearman and RMSE are recomputed after omitting each frozen domain in turn.
Results are produced on:

1. the formal row scale;
2. the canonical-SMILES median scale.

The Spearman contrast is `full - chemistry`. The RMSE contrast is
`chemistry - full`, so a positive RMSE contrast favours full. Influence is the
leave-one-domain-out value minus the all-domain value.

This is a descriptive influence analysis for the fixed four source or eight
target domains. It is not a population jackknife, external replication,
confidence interval or significance test.

## Analysis 3: paired global-scaffold bootstrap for calibration

Calibration uncertainty is calculated on the formal row-level predictions for
full-context ExtraTrees and chemistry-only ExtraTrees.

Separately for source OOD and target OOD:

1. the unique scaffold IDs in the combined frozen test universe are sampled
   with replacement;
2. every row belonging to a sampled scaffold is included, including rows from
   different domains;
3. identical sampled row positions are used for both models;
4. within-domain calibration intercept, calibration slope and R2 are
   calculated and equally averaged for domain-macro estimates;
5. pooled estimates are calculated over all sampled rows;
6. the model estimates and paired `full - chemistry` contrasts receive
   2.5th and 97.5th percentile intervals from 10,000 replicates.

Calibration is defined by:

`observed pDC50 = intercept + slope * predicted pDC50`.

The ideal intercept and slope are 0 and 1. A signed difference in intercept or
slope has no generic superiority direction; the contrast is descriptive.
Positive R2 contrast favours full.

## Inference boundary

- The analyses are explicitly `post_hoc_exploratory`.
- The four source and eight target domains are fixed, not random population
  samples.
- Bootstrap replicates are Monte Carlo draws, not independent `n`.
- The bootstrap propagates uncertainty conditional on the frozen test
  predictions and scaffold clusters; it does not propagate uncertainty from
  training data, domain choice, hyperparameter selection or model selection.
- No p-value is calculated or reported.
- These analyses do not create prospective, external, strict
  domain-plus-scaffold-cold or mechanistic evidence.

## Outputs and QA

The new directory
`reports/confirmatory_ood_cpu_v1_posthoc/` contains:

- aggregated prediction units and their domain/aggregate metrics;
- formal-row and canonical-scale LODO influence tables;
- paired calibration bootstrap summaries;
- a Chinese results brief;
- machine-readable QA;
- an artifact hash table and run manifest.

The output manifest binds the script, this note, input identities, exact
configuration, software versions, QA status and output artifact hashes. The
manifest itself is the sole self-referential hash exception.
