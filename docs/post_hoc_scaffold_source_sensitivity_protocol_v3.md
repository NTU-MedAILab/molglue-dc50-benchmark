# Post-hoc scaffold and training-source sensitivity protocol v3

Status: frozen execution-state correction protocol on 2026-07-30  
Evidence label: post-hoc sensitivity analysis  
Execution: CPU only

Governing master identity:
`docs/post_hoc_computational_extension_master_protocol_v1.md`, SHA-256
`7dee28f45e3ddf8d6299622a8e494c50492b3b2614f39ce8b2fdd2a1be754ff8`.
Where wording differs, the frozen master protocol governs.

Float32 correction identity:
`docs/post_hoc_float32_response_correction_v2.md`, SHA-256
`75a8b0e4f894cead9d72039eab9e6162a89d2a20c79afceda8a2aae1ffd78f3f`.
The response supplied to every tuning fold and final ExtraTrees refit remains
NumPy `float32`.

Thread-execution correction identity:
`docs/post_hoc_scaffold_source_thread_execution_correction_v3.md`.
The v3 runner fails closed if this document, the frozen v2 mathematical
implementation, the master protocol, or the float32 correction changes.

## Version-3 correction and v2 publication boundary

Version 2 fixed the response dtype but did not freeze or record the native
OpenMP and BLAS thread state. Independent reconstruction showed that its formal
predictions were generated with one OpenMP thread, whereas the archived
confirmatory target-OOD predictions and the earlier single-target identity run
were generated with 24 OpenMP threads. The data, ordered split indices, inner
folds, selected hyperparameters, feature matrices, model seeds and tree count
were unchanged. The prediction mismatch was reproduced solely by changing the
OpenMP execution state.

Version 3 changes no estimand, split, feature, model, hyperparameter grid,
selection rule, seed, metric or bootstrap rule. It makes the native execution
state part of the frozen computational identity:

- the process must be launched with `OMP_NUM_THREADS=24`;
- the process must be launched with `OPENBLAS_NUM_THREADS=24`;
- after NumPy, SciPy and scikit-learn are imported, every active OpenMP and
  BLAS pool reported by `threadpoolctl` must report 24 threads;
- the CPU affinity available to the process must include at least 24 logical
  CPUs;
- the native-pool state is checked at startup and around both analysis blocks,
  and every snapshot is written to `native_threadpool_audit.json`.

The v3 runner reuses the frozen v2 mathematical implementation only after
verifying its exact SHA-256. This reuse is an explicit code dependency, not a
reuse of v2 results. A formal v3 run starts from an empty v3 directory and may
not copy or resume any v1 or v2 artifact.

The v1 and v2 result packages remain unchanged as audit records. Neither is
eligible for manuscript, figure, source-data or release use after v3 has
passed. A publication-facing claim must resolve only to the completed v3
package.

## Scope and data identity

The only observation table loaded is
`data/processed/all_molglue_dc50_qc_train_test_standardized_context.csv`,
whose required SHA-256 is
`e9494246bcbd198bc09d5b0d09721910c33f70f46ae859b76377a13775e3e4b2`.
The expected sanitized analysis set is 1,560 rows.

The analyses answer two reviewer-facing robustness questions:

1. Does the matched full-versus-chemistry conclusion depend on the exact
   Bemis--Murcko grouping used by the primary scaffold-disjoint validation?
2. In target-OOD evaluation, does the conclusion depend on retaining any one
   source database in the training set?

Neither analysis is confirmatory, prospective, mechanistic, or an estimate of
absolute deployment calibration.

## Frozen models and features

Only the two matched ExtraTrees regressors are evaluated:

- `chemistry_extra_trees`: Morgan radius-2, 2,048-bit chiral fingerprint plus
  the frozen RDKit descriptor block;
- `full_context_extra_trees`: the same chemistry block plus the frozen one-hot
  context main effects and interactions.

The parameter grid is inherited unchanged:
`min_samples_leaf` in `{1, 5, 10}` crossed with `max_features` in
`{"sqrt", 0.3}`. Selection maximizes pooled inner out-of-fold Spearman
correlation. Candidates within 0.005 of the best value use the same
regularization-first tie break as the confirmatory runner.

Formal settings are 600 trees, four scaffold-disjoint inner folds, seed
260531, four joblib jobs per fit, OpenMP 24 and BLAS 24. The modeling response
is exactly `float32`; metrics and serialized predictions retain their existing
precision.

## Analysis A: generic Murcko scaffold sensitivity

Each canonical molecule is converted to its Bemis--Murcko framework and then
to RDKit's generic framework with `MakeScaffoldGeneric`. An acyclic molecule
receives the compound-specific identifier
`ACYCLIC::<canonical non-isomeric SMILES>` so unrelated acyclic molecules are
not collapsed into one group.

The frozen grouping contains 376 generic scaffold groups, including 183
singletons, and the largest group contains 76 rows. A mismatch stops the run.

The validation design is five outer folds repeated five times, with four
generic-scaffold-disjoint inner folds in every outer training set. Both inner
and outer train/evaluation group overlap must be zero. Every observation must
receive one out-of-fold prediction per repeat and model.

Reported endpoints are pooled Spearman correlation, RMSE and the remaining
frozen regression metrics by repeat and after averaging each observation's
five out-of-fold predictions. The primary matched contrast is
`full_context_extra_trees - chemistry_extra_trees` for Spearman and the reverse
subtraction for RMSE, so positive values favor the full model. Uncertainty uses
10,000 paired percentile bootstrap resamples of generic scaffold clusters.

## Analysis B: target-OOD training-source deletion

The eight exact held-target test sets and leakage exclusions are inherited
unchanged from `run_confirmatory_ood_cpu_v1.py`: VAV1, CSNK1A1, GSPT1, WIZ,
CDK2, CCNK+CDK12, IKZF2 and IKZF1. Test rows never change across deletion
conditions.

Five training conditions are evaluated for every held target:

- no source deletion;
- delete MGTbind;
- delete MGDB;
- delete MolGlueDB;
- delete TPDdb.

Source provenance is tokenized on `+`. Deleting source `S` removes every
training row whose provenance token set contains `S`, including composite
provenance strings. The corresponding source token must be absent after
deletion. Held-target exclusion, exact-compound train/test disjointness and
row disjointness must remain valid.

For every target and deletion condition, both models are retuned from scratch
using four Bemis--Murcko-scaffold-disjoint inner folds and refitted on the
remaining training set. Evaluation is not reweighted. An infeasible condition
is written as an explicit failed stratum and may not be silently omitted.

Uncertainty uses 10,000 paired global Bemis--Murcko scaffold-cluster bootstrap
replicates across the combined eight-target test axis. The same resample is
applied to every model and deletion condition, and target metrics are combined
with equal weight over the fixed eight-domain set. A replicate in which a
required domain metric is not estimable remains non-estimable.

## Mandatory all-target baseline identity gate

Before a multi-hour formal run, an all-target, no-deletion preflight must be
executed with the formal model and native-thread settings. The same independent
identity QA is rerun on the no-deletion rows inside the completed formal
package. Across all eight targets and both models it requires:

- exact prediction row keys, ordering, `qc_id`, response, prediction and
  selected parameter identity against the archived confirmatory target-OOD
  results;
- exact selected-hyperparameter and inner-tuning fields;
- exact reconstructed outer train/test index sequences and all four inner
  split sequences per target;
- exact retained RDKit, context, chemistry and full feature dimensions; and
- complete coverage of the fixed target set and both matched models.

Any mismatch is a hard failure. A single-target identity run is not sufficient
for v3 publication eligibility.

## Smoke, preflight, formal completion and provenance

A smoke run may reduce trees, splits, targets, deletion conditions and
bootstrap replicates, but must use a separate v3 directory. It checks plumbing
only and is never scientific evidence.

The identity preflight uses all eight targets, the no-deletion condition, 600
trees, four inner folds, seed 260531, four joblib jobs, OpenMP 24 and BLAS 24.
It uses zero bootstrap replicates and is written outside the formal directory.

The formal run uses `--mode both`, all 25 generic-scaffold outer splits, all
eight targets and all five source conditions, 600 trees and 10,000 bootstrap
replicates. It is complete only if the runner QA and independent all-target
identity QA both pass.

The formal manifest records command configuration, environment versions,
input/protocol/correction/imported-script checksums, response dtype, joblib
jobs, required environment variables, CPU affinity, normalized
`threadpoolctl` snapshots, elapsed time, completeness status and an artifact
checksum inventory.

Every v3 run writes to an explicitly identified v3 directory. The runner
refuses a basename containing `_v1` or `_v2`, a basename lacking `_v3`, or a
directory carrying a superseded-output marker. Resume is allowed only within
the same v3 protocol, response dtype and native-thread contract. Formal v3
computation must start from an empty directory.

## Interpretation and data boundary

These experiments diagnose dependence on grouping and source composition.
They cannot establish prospective transportability, recover unmeasured assay
metadata, validate a mechanism, or authorize data outside the frozen core
table.

No collaborator-restricted observation or derivative may be loaded, tested,
summarized or emitted by this workflow.
