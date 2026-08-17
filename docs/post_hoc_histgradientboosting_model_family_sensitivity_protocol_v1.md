# Frozen post-hoc HistGradientBoosting model-family sensitivity protocol v1

Freeze date: 2026-07-28 (Asia/Shanghai)  
Analysis identity: `post_hoc_model_family_sensitivity`  
Compute: CPU only; no GPU  
Endpoint: `pDC50 = 9 - log10(DC50_nM)`

## Question

Does the qualitative evaluation pattern observed with ExtraTrees—small internal
benefit from adding context, but no corresponding benefit under source/target
OOD and strict scaffold purging—persist with one independently specified CPU
learner?

This analysis is not confirmatory and cannot change the identity of the
previously frozen results. It is a single model-family sensitivity analysis,
not a model search.

## Frozen data and splits

- Use the same 1,560-row frozen QC table and the same sanitized modeling rows.
- Internal analysis uses only the primary five-times-repeated five-fold
  scaffold-disjoint splits reconstructed by the frozen core splitter.
- Source and target OOD use the 4 and 8 frozen domain-plus-compound-cold
  splits, respectively.
- Strict OOD keeps the same test domains and rows, then removes from training
  every row whose Bemis–Murcko scaffold occurs in that test domain.
- The runner must reproduce the published split counts and verify zero
  row/compound/held-domain overlap; strict splits must additionally have zero
  scaffold overlap.

## Frozen feature blocks

- `chemistry_hist_gradient_boosting`: Morgan radius 2, 2,048 bits with
  chirality, plus RDKit descriptors.
- `full_context_hist_gradient_boosting`: the identical chemistry block plus
  the frozen categorical context columns and interactions.
- RDKit missing-value imputation and zero-variance filtering are learned from
  each training fold only.
- The one-hot context encoder is fitted on each training fold only.

No protein-language-model embedding is included.

## Frozen learner

`sklearn.ensemble.HistGradientBoostingRegressor` with exactly:

- `loss="squared_error"`
- `learning_rate=0.05`
- `max_iter=200`
- `max_leaf_nodes=15`
- `min_samples_leaf=20`
- `l2_regularization=1.0`
- `max_bins=64`
- `early_stopping=False`
- `random_state=260531 + split-specific deterministic offset`

There is no hyperparameter grid, inner selection, ensemble, calibration, or
test-dependent model choice.

## Estimands

Primary sensitivity contrast:

- `ΔSpearman = full − chemistry-only`
- `ΔRMSE = RMSE(chemistry-only) − RMSE(full)`

Positive values favor the full model.

Internal point estimates are calculated after averaging each row's five
out-of-fold predictions. OOD estimates are equal-domain-weight
domain-macro Spearman and RMSE; pooled metrics are secondary.

## Uncertainty

- Internal: 10,000 paired global scaffold-cluster bootstrap replicates.
- Source and target OOD, including strict variants: 10,000 paired global
  scaffold-cluster bootstrap replicates per axis and split regime.
- Intervals are percentile 95% intervals.
- Folds, repeats, seeds, domains and bootstrap replicates are not described as
  additional independent observations.
- No p-values are reported.

## Frozen interpretation

- If internal `ΔSpearman` is positive while both OOD axes are non-positive or
  compatible with no benefit, the evaluation pattern is considered
  qualitatively replicated across two learner families.
- If the direction is not reproduced, the manuscript remains viable but must
  describe the context-transfer result as specific to the matched ExtraTrees
  pipeline.
- Absolute unseen-target generalization is not established by a relative
  full-versus-chemistry contrast.
- No mechanistic, causal, prospective, external-validation or deployment
  claim is permitted.

## Stop conditions

- Stop on any split leakage, frozen count mismatch, non-finite prediction, or
  data/script identity mismatch.
- Do not add a second learner after inspecting the result.
- Do not change learner parameters, feature blocks, split rules, metrics or
  bootstrap seeds after the first formal execution.
- GPU use is prohibited for this protocol.
