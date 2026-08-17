# Corrected post-hoc portable-context and training-weight sensitivity protocol v2

Freeze date: 2026-07-29 (Asia/Shanghai)  
Analysis identity: `post_hoc_context_weight_sensitivity_v2`  
Compute: CPU only; no GPU  
Endpoint: `pDC50 = 9 - log10(DC50_nM)`

Master protocol:
`docs/post_hoc_computational_extension_master_protocol_v1.md`, frozen SHA256
`7dee28f45e3ddf8d6299622a8e494c50492b3b2614f39ce8b2fdd2a1be754ff8`.
This child protocol is subordinate to that frozen master identity.

Float32 implementation correction:
`docs/post_hoc_float32_response_correction_v2.md`, frozen SHA256
`75a8b0e4f894cead9d72039eab9e6162a89d2a20c79afceda8a2aae1ffd78f3f`.

## Status and scope

This is a post-hoc robustness analysis of the already frozen 1,560-row
strict-exact molecular-glue DC50 benchmark. It is not preregistered,
prospective, external, or confirmatory. It cannot replace or silently modify
the frozen core or OOD analyses.

Only the publication-eligible core table with SHA256
`e9494246bcbd198bc09d5b0d09721910c33f70f46ae859b76377a13775e3e4b2`
may be read. No collaborator-restricted prospective panel or derivative is
permitted as input.

This v2 protocol supersedes the complete
`post_hoc_context_weight_sensitivity_v1` output package. The v1 runner
constructed the model-training response as `float64`, while the frozen
confirmatory ExtraTrees implementation constructs it as `float32`. All v1
outputs are retained unchanged as an audit trail but are publication
ineligible and must not be used, resumed, overwritten, merged, plotted, or
combined with v2.

The sole scientific implementation correction from v1 is:

```python
y = rows["pDC50"].to_numpy(dtype=np.float32)
```

All metric definitions and explicit float64 summary calculations remain
unchanged.

## Questions

1. Is the poor transfer of the full context model driven mainly by one-hot
   categories that are structurally absent in the held-out domain, or does it
   remain after those held-axis features are removed?
2. Does the internal scaffold-disjoint advantage of adding context persist
   when repeated measurements of the same canonical compound do not receive
   extra total training weight?
3. Does it persist when large source-target domains are prevented from
   dominating ExtraTrees impurity calculations?

## Shared frozen components

- The sanitized rows, canonical SMILES, `pDC50`, Morgan radius-2 2,048-bit
  fingerprints with chirality, RDKit descriptor block, Bemis--Murcko
  scaffolds, fold-local descriptor processing, fold-local one-hot encoding,
  ExtraTrees parameter grid, 0.005 selection tolerance, and deterministic
  split builders are imported from the frozen CPU runners.
- ExtraTrees uses 600 trees, no bootstrap, no depth limit,
  `min_samples_leaf={1,5,10}`, `max_features={"sqrt",0.3}`, and at most four
  CPU workers.
- Internal validation reconstructs the same five-times-repeated five-fold
  scaffold-disjoint outer splits.
- OOD validation reconstructs the same four source and eight target
  domain-plus-compound-cold splits and must reproduce every frozen count.
- The existing formal predictions and selected-parameter tables are used only
  after data, row, split, response, and file-integrity checks.
- Every new ExtraTrees refit must receive an `np.float32` training-response
  array. The runner fails closed on any other response dtype.

## Portable-context OOD sensitivity

Three matched ExtraTrees representations are compared on each frozen OOD
axis:

1. `chemistry_extra_trees`: unchanged chemistry-only formal prediction;
2. `full_context_extra_trees`: unchanged full formal prediction;
3. `portable_full_extra_trees`: chemistry plus a context block that
   excludes the identity of the held axis and every interaction containing
   it.

For source OOD, portable context drops the `source_database` main effect and
the `source_target` interaction. It retains recruiter, target, cell line,
assay method, activity time, mode of action, recruiter-target, target-cell,
and assay-target features.

For target OOD, portable context drops the `target_protein` main effect and
all four target-containing interactions: source-target, recruiter-target,
target-cell, and assay-target. It retains source, recruiter, cell line, assay
method, activity time, and mode of action.

The portable model is independently selected within each OOD training set
using the same four-fold scaffold-disjoint inner splits, six-point ExtraTrees
grid, pooled inner cross-fitted Spearman, 0.005 tolerance, and tie rules as the
frozen OOD analysis. Test outcomes are never used for selection.

Primary portable-context contrasts are:

- portable minus chemistry-only;
- portable minus full context;
- full context minus chemistry-only.

Positive `delta_spearman = first - comparator` and positive
`delta_rmse = RMSE(comparator) - RMSE(first)` favour the first model.

## Internal training-weight sensitivity

Only chemistry-only and full-context ExtraTrees are refitted. Within every
frozen outer-training set, each weighted model is selected using the same
four-fold scaffold-disjoint inner splits, six-point ExtraTrees grid, pooled
unweighted inner cross-fitted Spearman, 0.005 tolerance, and tie rules as the
frozen core. Sample weights affect inner and final model fitting only; inner
validation metrics remain unweighted.

Three training-weight modes are compared:

1. `uniform`: the unchanged formal cross-fitted prediction;
2. `compound_equal`: each canonical compound has total raw weight one within
   the training fold, followed by normalization to mean row weight one;
3. `domain_balanced_clipped`: define a domain as the exact
   `source_database || target_protein` pair. Each row first receives reciprocal
   domain-size weight, those weights are normalized to mean one, values above
   ten times the mean are clipped, and all weights are renormalized to mean
   one.

Weights affect model fitting only. Test rows and all evaluation metrics remain
unweighted. Integer `min_samples_leaf` remains a row-count constraint in
scikit-learn; sample weights affect impurity and leaf-value calculations.

Primary internal contrasts are:

- full minus chemistry within each weight mode;
- each weighted refit minus its corresponding uniform model.

## Estimands and uncertainty

Internal point estimates are computed after averaging each row's five
cross-fitted predictions. Report pooled Spearman and RMSE, target-macro
Spearman, within-target Spearman, canonical-compound aggregate metrics, and
exact source-target-domain metrics. The main internal estimand is pooled
Spearman.

OOD point estimates include per-held-domain metrics, equal-domain macro
Spearman and RMSE, and pooled metrics. The main OOD estimand is equal-domain
macro Spearman. Undefined domain correlations remain missing and are never
changed to zero.

Paired 10,000-replicate percentile intervals use global Bemis--Murcko
scaffold-cluster resampling, separately for the internal, source-OOD, and
target-OOD universes. The same sampled row positions are used for every model
in a contrast. OOD domain-macro metrics are recomputed inside each replicate.
The intervals are conditional on the frozen dataset and held-domain set; no
p-values are reported.

## Quality controls

The formal run fails closed unless all of the following hold:

- input SHA256 and row count match the frozen core;
- the global float32 correction document has frozen SHA256
  `75a8b0e4f894cead9d72039eab9e6162a89d2a20c79afceda8a2aae1ffd78f3f`;
- the response passed to every new ExtraTrees fit is `np.float32`;
- the output directory is explicitly identified as v2, is not the v1 output
  directory, and contains no `SUPERSEDED_DO_NOT_USE.md` marker;
- formal baseline artifacts exist and their own artifact inventories verify;
- reconstructed outer and OOD split identities match stored predictions;
- internal training and test scaffolds are disjoint;
- OOD training and test canonical compounds are disjoint and the held-domain
  token is absent from training;
- portable feature audits prove the required columns and interactions were
  removed;
- all weights are finite and positive, mean row weight is one, and
  compound-equal totals are equal within numerical tolerance;
- predictions are finite and model-aligned;
- bootstrap and summary tables can be recomputed from stored row predictions;
- every emitted artifact has a SHA256 checksum, byte count, row count, and
  column schema where applicable.

Smoke tests may reduce splits, held groups, trees, and bootstrap replicates,
but their outputs must be written to a separately named directory and cannot
be described as formal evidence.

The formal v2 output directory is
`reports/post_hoc_context_weight_sensitivity_v2`. It must be created fresh.
No v1 checkpoint or result table may be copied into or resumed by v2.

## Frozen v2 implementation identity

- v2 runner:
  `scripts/run_post_hoc_context_weight_sensitivity_v2.py`
- v2 runner SHA256:
  `a007246e01cbfefd68286a506c9031a0d33c643a3f80164b963c99798d86e701`
- superseded v1 runner SHA256:
  `e13b058c5958576a30c60ee8bc0f9d2ae593218bc2f4917244459a3d3e2a0eae`
- frozen confirmatory core runner SHA256:
  `ac1a57203b43d74023521b76cfaef28b958a803dbc6d9fb2866e7550f636995c`
- frozen confirmatory OOD runner SHA256:
  `ad7d95843007eb13596c5e8171d954d37fe1050fe9880eb1c7b64ffc91107757`
- locked v2 verification test:
  `tests/test_post_hoc_context_weight_sensitivity_v2.py`
- locked v2 verification test SHA256:
  `f02e3ca67f00e1d6915b80d81442ca28e0c82390bc9972cfb3f27dbee0bda03b`

## Interpretation boundary

If portable context still fails to outperform chemistry-only OOD, the result
supports the interpretation that the transfer problem is not explained only
by all-zero unseen one-hot categories. It does not prove that every assay or
biological context variable is non-transferable.

If weighting preserves the internal full-minus-chemistry direction, repeated
compounds or large source-target domains are less plausible as sole
explanations for that contrast. If the direction changes, the manuscript must
describe the internal context gain as weight-sensitive.

No mechanistic, causal, prospective, external-validation, calibrated absolute
prediction, or deployment claim is permitted.
