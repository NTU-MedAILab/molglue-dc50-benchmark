# Scaffold/source native-thread execution correction v3

Status: frozen implementation and provenance correction on 2026-07-30  
Scope: post-hoc generic-scaffold and target-OOD source-deletion ExtraTrees
refits only

## Finding

The scaffold/source v2 runner fixed the response dtype but recorded only
`n_jobs=4`. It neither froze `OMP_NUM_THREADS` nor captured the native thread
pools active inside scikit-learn.

An all-target comparison subsequently showed that the no-source-deletion rows
of the completed v2 package did not exactly reproduce the archived
confirmatory target-OOD ExtraTrees predictions. The mismatch affected
predictions, while row identity, response, selected parameter identity and
quality-control identifiers remained aligned.

Independent reconstruction isolated the execution-state cause:

- with `OMP_NUM_THREADS=1`, the same frozen code, data, split sequences,
  feature matrices, hyperparameters and model seeds reproduced the completed
  v2 predictions to CSV serialization precision;
- with the default 24-thread OpenMP state, the same reconstruction reproduced
  the archived confirmatory predictions exactly; and
- a deliberately altered model-seed mapping produced substantially larger
  differences and did not reproduce v2, ruling out the suspected seed-path
  explanation.

The difference is therefore an ExtraTrees native-thread numerical execution
effect, not a change in data, leakage control, split construction, response
precision, feature definition, hyperparameter selection rule or intended
random seed.

## Impact assessment

The source-deletion analysis uses the no-deletion condition as the paired
reference. Replacing only that reference would mix execution states within a
paired analysis and is prohibited. All source conditions and their paired
bootstrap summaries must be recomputed under one frozen thread contract.

The generic-scaffold and source-deletion calculations were emitted as one
formal v2 package, and all ExtraTrees fits in that process shared the
unrecorded native execution state. Retaining only one component would require
a split publication lineage and a retroactive component-level provenance
exception. To minimize publication ambiguity, v3 recomputes the complete
combined package.

The context-weight workflow and the null/applicability/censoring workflow are
separate implementations and are outside this correction.

## Corrective decision

The v3 scientific estimands and mathematics are unchanged. The following
execution state becomes part of the frozen identity:

1. launch the Python process with `OMP_NUM_THREADS=24`;
2. launch the Python process with `OPENBLAS_NUM_THREADS=24`;
3. require at least 24 logical CPUs in the process affinity mask;
4. after third-party imports, require every active OpenMP and BLAS pool
   reported by `threadpoolctl` to expose exactly 24 threads;
5. repeat the runtime check before and after the generic-scaffold and
   source-deletion analysis blocks;
6. write the raw environment values, CPU topology/affinity and normalized pool
   records to `native_threadpool_audit.json`; and
7. require an independent all-eight-target exact-identity audit before the
   formal output can be publication eligible.

The v3 runner may import the v2 mathematical implementation only if the v2
runner SHA-256 is exactly
`449d8821ff31e9a5392a227a86ce78fc09c1660f293449a1fbc850a3ee8ffb34`.
It reads no v2 result artifact and refuses cross-version resume.

## Required identity evidence

The all-target preflight and the completed formal package are each compared
with `reports/confirmatory_ood_cpu_v1`. The independent QA must cover the fixed
eight target domains and both matched ExtraTrees models, and must require:

- exact no-deletion prediction vectors after deterministic key sorting;
- exact `row_index`, `qc_id`, `y_true`, `param_id` and model identity;
- exact selected hyperparameters and inner-tuning records;
- exact reconstructed outer train/test sequences;
- exact reconstructed four-fold inner split sequences; and
- exact RDKit-retained, context, chemistry and full feature dimensions.

This identity comparison is an implementation audit. It is not a new
estimand, validation cohort or inferential result.

## Version and supersession rule

The v1 and v2 scripts, protocols and outputs are preserved unchanged for
forensic traceability. No v1 or v2 artifact may be copied, resumed or inserted
into the v3 output. After the formal v3 package and its independent QA pass,
downstream publication gates must accept only
`post_hoc_scaffold_source_sensitivity_v3.0`.

Until that point, scaffold/source downstream publication generation remains
blocked. This correction does not authorize changes to any other completed
workflow.

## Interpretation and data boundary

The correction restores an exact execution identity between the within-script
no-deletion reference and the frozen confirmatory target-OOD ExtraTrees
baseline. It does not make the post-hoc analysis confirmatory or prospective,
and it does not establish deployment transportability.

Only the frozen public-scope core table is permitted. No
collaborator-restricted observation or derivative may be accessed or emitted.
