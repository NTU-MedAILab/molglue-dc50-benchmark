# Float32 training-response alignment correction v2

**Status:** frozen implementation correction  
**Frozen on:** 2026-07-29 (Asia/Shanghai)  
**Scope:** all post-hoc ExtraTrees refit workflows that reuse the frozen
confirmatory implementation: context/weight sensitivity, generic-scaffold and
training-source-deletion sensitivity, and conditional permutation null  
**Compute:** CPU only; no GPU

## Correction decision

The formal confirmatory runner constructs the modeling response as
`rows["pDC50"].to_numpy(dtype=np.float32)`. Three v1 post-hoc refit families
instead constructed that response as `np.float64`:

1. context-axis and compound/domain-weight sensitivity;
2. generic-Murcko scaffold and training-source-deletion sensitivity; and
3. the conditional permutation-null parent and execution-only shards.

Although response values, split identities, seeds, features, selected
hyperparameters and model settings were otherwise unchanged within each
protocol, this dtype difference means those v1 refits were not implementation
identical to the frozen confirmatory refits.

Version 2 therefore makes one scientific implementation correction:

```python
y = rows["pDC50"].to_numpy(dtype=np.float32)
```

The within-stratum permutation operates on a copy of this array and must
preserve `np.float32` through model fitting. Metric accumulation and metric
calculation remain unchanged; the correction concerns the training response,
not the precision used to summarize predictions.

## Supersession and publication rule

All outputs produced by the following v1 refit families are **superseded and
must not be used in the manuscript, figures, source-data package, statistical
audit or public release**:

- `run_post_hoc_context_weight_sensitivity_v1.py`;
- `run_post_hoc_scaffold_source_sensitivity_v1.py`; and
- `run_post_hoc_null_applicability_censoring_v1.py` together with every
  `run_post_hoc_permutation_shard_v1.py` shard and v1 merge product.

They are retained unchanged as an audit trail and must not be deleted,
overwritten, merged into v2, or resumed by v2.

Some tables within these packages do not refit a model—for example
applicability summaries based on saved predictions and the descriptive
censoring audit—and are not numerically affected by the response dtype.
Nevertheless, every affected v1 output package is superseded atomically so
that publication evidence cannot mix artifacts from two implementation
identities. Each v2 workflow regenerates its complete package in a separate
directory.

Cross-version mixing is prohibited at every level: a v2 aggregator, plotter or
release builder must not combine v2 tables with any model result, bootstrap,
inference table, QA file, manifest or checksum table from an affected v1
package. A publication result is eligible only after the complete contributing
v2 workflow has passed its own QA.

## Invariants

The correction does not change:

- the 1,560-row public-scope frozen modeling core;
- the parsed public-source table used only for descriptive censoring audit;
- outer folds, five repeats, seeds or conditional permutation strata;
- the two model identities, fold-specific selected hyperparameters, number of
  trees, thread ceiling or feature construction;
- the number of formal permutations or empirical-tail definition;
- fixed applicability bins, paired bootstrap procedure or censoring rules; or
- the post-hoc, non-prospective claim boundary.

No collaborator-restricted record or derivative is read or produced.

## Required v2 controls

Every corrected refit workflow must:

1. use an output directory whose name ends in `_v2` (or an explicitly
   identified v2 shard directory);
2. refuse to resume or merge any v1 output;
3. construct the response explicitly as `np.float32`;
4. verify that both the original and permuted training responses remain
   `np.float32`;
5. use the same permuted response for the paired chemistry-only and
   full-context fits;
6. record the parent implementation, protocol and input hashes; and
7. use workflow-specific v2 protocols and manifests to lock the final v2
   executable identities; and
8. for the permutation family, pass a locked-environment exact-identity test
   for permutation ID 1 between the parent-v2 runner and shard-v2 runner before
   the formal 100-ID run.

## Frozen source identities

| Artifact | SHA-256 |
|---|---|
| v1 context/weight runner (superseded) | `e13b058c5958576a30c60ee8bc0f9d2ae593218bc2f4917244459a3d3e2a0eae` |
| v1 scaffold/source runner (superseded) | `94b281d7c8069b2f3ed82b124af9fb848b18c808d7a49bbccd6a55b7d4b8c60e` |
| v1 post-hoc parent (superseded) | `789b5814d28db01c5b704dbfd71f01ec9a44f9a73cda6e127e7d60a7d4d2d88b` |
| v1 shard wrapper (superseded) | `134e3042808f7212c6f5a67ff1d7170aeda3ead22025377e9fa604562b6b9f6d` |
| v1 merge helper (superseded) | `503bcb52a38cdcff93391f95997b25c92585670873eb234861e1db41248aec9c` |
| frozen confirmatory implementation | `ac1a57203b43d74023521b76cfaef28b958a803dbc6d9fb2866e7550f636995c` |
| computational-extension master protocol v1 | `7dee28f45e3ddf8d6299622a8e494c50492b3b2614f39ce8b2fdd2a1be754ff8` |
| frozen 1,560-row modeling core | `e9494246bcbd198bc09d5b0d09721910c33f70f46ae859b76377a13775e3e4b2` |
| public-source parsed-record table | `ae1a16613b41ce84f92f539dbd31f16ada272ac5e401250acc7e97fd5e8bd970` |

## V2 implementation identity registry

This global correction note is frozen before the three corrected workflows
are finalized, so it deliberately uses non-hash placeholders rather than
creating circular document/script dependencies. Each final hash must be
recorded in the workflow-specific v2 protocol and/or fail-closed runtime
manifest before formal execution.

| Corrected workflow | V2 implementation identity location |
|---|---|
| context/weight sensitivity | `post_hoc_context_weight_sensitivity_protocol_v2.md` and v2 run manifest |
| generic scaffold/source deletion | `post_hoc_scaffold_source_sensitivity_protocol_v2.md` and v2 run manifest |
| null/applicability/censoring parent | `post_hoc_null_applicability_censoring_protocol_v2.md`, v2 input checksum table and shard manifest |
| permutation shard wrapper | hard-locked by `merge_post_hoc_permutation_shards_v2.py` and recorded in each shard manifest |
| permutation shard merge helper | correction verification record and operator run log |

Each run additionally writes input and artifact checksum tables, so results
remain bound to the exact executable files and inputs used.
