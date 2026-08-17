# Post-hoc scaffold and training-source sensitivity protocol v2

Status: frozen implementation-correction protocol on 2026-07-29  
Evidence label: post-hoc sensitivity analysis  
Execution: CPU only

Governing master identity:
`docs/post_hoc_computational_extension_master_protocol_v1.md`, SHA-256
`7dee28f45e3ddf8d6299622a8e494c50492b3b2614f39ce8b2fdd2a1be754ff8`.
Where wording differs, the frozen master protocol governs.

Float32 correction identity:
`docs/post_hoc_float32_response_correction_v2.md`, SHA-256
`75a8b0e4f894cead9d72039eab9e6162a89d2a20c79afceda8a2aae1ffd78f3f`.
The runner fails closed if this identity changes.
The global correction decision governs model-refit response precision.

## Version-2 correction and supersession

The frozen confirmatory runners construct the response with
`rows["pDC50"].to_numpy(dtype=np.float32)`. Version 1 of this scaffold/source
runner instead constructed it as `np.float64`, so its refits were not
implementation-identical to the confirmatory model even though the data,
split indices, feature matrices, tuning procedure, selected parameters,
seeds and tree count were unchanged.

Version 2 makes exactly one scientific implementation correction:

```python
y = rows["pDC50"].to_numpy(dtype=np.float32)
```

Metric calculations and stored predictions retain their existing numerical
precision; only the response supplied to tuning and fitting changes. The v1
script, protocol and reports are retained unchanged as a provenance trail but
are superseded for publication use. No v1 output may be resumed into, copied
into or used as a baseline for v2.

## Scope and data identity

This analysis uses only
`data/processed/all_molglue_dc50_qc_train_test_standardized_context.csv`,
whose required SHA-256 is
`e9494246bcbd198bc09d5b0d09721910c33f70f46ae859b76377a13775e3e4b2`.
The expected analysis set is 1,560 rows after the already frozen sanitation
implemented by `run_confirmatory_cpu_v1.py`. No other observation table or
supplemental panel is loaded.

The analysis answers two reviewer-facing robustness questions:

1. Does the matched full-versus-chemistry conclusion depend on the exact
   Bemis--Murcko grouping used by the primary scaffold-disjoint validation?
2. In target-OOD evaluation, does the conclusion depend on retaining any one
   source database in the training set?

Neither analysis is confirmatory, prospective, or an estimate of absolute
DC50 calibration in deployment.

## Frozen models and features

Only the two matched ExtraTrees regressors are evaluated:

- `chemistry_extra_trees`: Morgan radius-2, 2,048-bit chiral fingerprint plus
  the frozen RDKit descriptor block;
- `full_context_extra_trees`: the same chemistry block plus the frozen
  one-hot context main effects and interactions.

The parameter grid is inherited unchanged from the primary CPU protocol:
`min_samples_leaf` in `{1, 5, 10}` crossed with `max_features` in
`{"sqrt", 0.3}`. Selection maximizes pooled inner out-of-fold Spearman
correlation. Candidates within 0.005 of the best value use the same
regularization-first tie break as the primary protocol.

Formal settings are 600 trees, four scaffold-disjoint inner folds, seed
260531, and at most four CPU jobs per fit.

The response supplied to every tuning fold and final ExtraTrees refit must
have NumPy dtype `float32`. The scientific configuration records
`response_dtype=float32`, and the QA report must fail if the in-memory
response dtype is not exactly `float32`.

## Analysis A: generic Murcko scaffold sensitivity

Each canonical molecule is converted to its Bemis--Murcko framework and then
to RDKit's generic framework using `MakeScaffoldGeneric`; chirality and atom
identity are removed by that operation. An acyclic molecule receives the
compound-specific identifier `ACYCLIC::<canonical non-isomeric SMILES>` so
unrelated acyclic molecules are not collapsed into one group.

The expected grouping is 376 generic scaffold groups, including 183 singleton
groups, with a largest group of 76 rows. A mismatch stops the run.

The validation design otherwise matches the primary scaffold protocol:
five outer folds repeated five times, with four generic-scaffold-disjoint
inner folds inside every outer training set. Both inner and outer
train/evaluation group overlap must be zero. All 1,560 observations receive
one out-of-fold prediction per repeat and model.

Reported endpoints are pooled Spearman correlation, RMSE and the remaining
frozen regression metrics by repeat and after averaging each observation's
five out-of-fold predictions. The primary matched contrast is
`full_context_extra_trees - chemistry_extra_trees` for Spearman correlation
and the reverse subtraction for RMSE, so positive values favor the full model.
Uncertainty uses 10,000 paired percentile bootstrap resamples of generic
scaffold clusters.

## Analysis B: target-OOD training-source deletion

The eight exact held-target test sets and their leakage exclusions are
inherited unchanged from `run_confirmatory_ood_cpu_v1.py`:
VAV1, CSNK1A1, GSPT1, WIZ, CDK2, CCNK+CDK12, IKZF2 and IKZF1.
Test rows never change across deletion conditions.

Five training conditions are evaluated for every held target:

- no source deletion (within-script baseline);
- delete MGTbind;
- delete MGDB;
- delete MolGlueDB;
- delete TPDdb.

Source provenance is tokenized on `+`. Deleting source `S` removes every
training row whose provenance token set contains `S`, including composite
provenance strings. The corresponding source token must be absent after
deletion. The inherited held-target token exclusion, exact-compound
train/test disjointness, and row disjointness must remain valid.

For every target and deletion condition, both models are retuned from scratch
using four Bemis--Murcko-scaffold-disjoint inner folds and then refitted on the
remaining training set. Evaluation is never reweighted. Results include:

- metrics for each target, deletion condition and model;
- the paired full-versus-chemistry delta within every target;
- deletion-versus-baseline deltas within each model;
- the equal-target-domain macro mean across all eight held targets.

An infeasible training condition is written as an explicit failed stratum
with its reason and counts; it is never omitted or replaced by an
available-case macro estimate.

Uncertainty uses 10,000 paired global Bemis--Murcko scaffold-cluster
bootstrap replicates, matching the formal combined-axis OOD analysis. Global
scaffolds are sampled with replacement across the combined eight-target test
axis; every occurrence of a sampled scaffold across all targets is retained
together. The same global resample is applied to every model and deletion
condition, after which target-level metrics are recomputed and combined with
equal weight over the fixed eight-domain set. A replicate in which a required
domain metric is not estimable remains non-estimable rather than silently
dropping that domain. Percentile 95% intervals are reported. A contrast is
interpreted as descriptive sensitivity evidence; no multiplicity-adjusted
null-hypothesis test is claimed.

## Smoke tests, completeness and provenance

The smoke run may reduce trees, outer splits, targets, deletion conditions and
bootstrap replicates, or use the audit-only `--only-target` and
`--baseline-only` controls, and must be written to a directory distinct from
the formal output. These two audit controls only select a subset of frozen
strata; they do not change any selected stratum's data, fitting or evaluation
logic. The formal run is complete only if:

- all 25 generic-scaffold outer splits are present;
- every row has five predictions from each model in Analysis A;
- all 8 targets x 5 deletion conditions x 2 models are present in Analysis B;
- all finite-prediction, alignment and zero-overlap checks pass;
- the requested 10,000 bootstrap replicates are recorded for both analyses.

The output manifest records the command-line configuration, environment
versions, input and protocol checksums, imported script checksums, elapsed
time, completeness status and an artifact checksum inventory.

Every v2 run writes to an explicitly identified v2 directory. The runner
refuses any output-directory basename containing `_v1` or lacking `_v2`.
It also refuses a directory carrying `SUPERSEDED_DO_NOT_USE.md` and, on
resume, requires any existing manifest/configuration identity to report this
exact v2 protocol version and `response_dtype=float32`.
The manifest also binds the frozen float32-correction document checksum and
records the response dtype. A locked-environment baseline reconstruction is
required before the formal v2 run: at least one no-deletion target must be
refit, and CDK2 must be compared row by row with the archived confirmatory
target-OOD predictions using identical ordering, selected hyperparameters and
seeds.

## Interpretation boundary

These experiments diagnose dependence on grouping and source composition.
They cannot establish prospective transportability, recover unmeasured assay
metadata, validate a mechanistic hypothesis, or authorize use of any data
outside the frozen core table.

No collaborator-restricted observation or derivative may be loaded, tested,
summarized or emitted by this workflow.
