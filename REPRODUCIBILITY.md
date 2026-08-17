# Reproducibility protocol v2

## Reproduction unit

The reproducible unit is a clean private work directory created from this
public payload. Source exports, the rebuilt 1,560-row table and every row-level
prediction stay in that directory and are not copied back into the repository.

## Frozen source and dataset identity

`data_builder/config/source_manifest.json` records the official landing page,
exact endpoint, access-era filename, byte size and SHA-256 for every required
source. `data_builder/config/expected_outputs.json` records the six rebuilt
table identities. The standardized analysis input must have 1,560 rows, 30
columns, 1,137 canonical compounds, 667 Bemis–Murcko scaffolds, byte size
664,847 and SHA-256
`e9494246bcbd198bc09d5b0d09721910c33f70f46ae859b76377a13775e3e4b2`.

A changed owner export is not coerced into the frozen identity. It is a new
temporal-validation snapshot and must receive a new manifest and analysis.

## Formal CPU order

1. verify the builder and analysis environments;
2. acquire or copy owner-hosted source files and verify every hash;
3. build and exactly verify all six local data tables;
4. rerun scaffold- and compound-disjoint confirmatory validation;
5. rerun the matched context-only ExtraTrees ablation;
6. rerun source- and target-OOD validation and no-refit diagnostics;
7. rerun strict domain-plus-scaffold-cold OOD sensitivity;
8. rerun HistGradientBoosting and internal learning-curve sensitivities;
9. rerun portable-context/fit-weight and scaffold/source sensitivities;
10. run permutation IDs 1–100 in four isolated deterministic CPU workers,
    merge them, and finalize applicability/censoring inference; and
11. compare regenerated CPU CSV artifacts byte-for-byte with the independent
    ledgers in `reference/cpu_artifact_ledgers/`.

The state file supports stage-level restart. Individual scientific scripts
also retain their original fold/permutation checkpoints. Logs are written once
per stage; real-time tracing is unnecessary.

## Identity policy

- source files, data tables, split/prediction/result CSVs on the locked Linux
  environments: exact byte identity;
- integer counts, category membership and identifiers: exact;
- runtime manifests: excluded from portable byte identity because paths,
  timestamps and runtime duration legitimately vary;
- public aggregate evidence: exact deposited identity;
- regenerated figures: source-data and dimension equivalence; renderer bytes
  may differ; and
- GPU graph sensitivity: deposited aggregate reference only in the CPU route.

The exact comparison is implemented independently in
`scripts/verify_cpu_results_v2.py`; it does not call the modeling metric
helpers.

## Resource expectation

No GPU is used. The formal chain is long-running and checkpointed. The two
largest CPU sensitivities and the 100-permutation analysis can take multiple
hours on a 24-core workstation. The scaffold/source runner intentionally locks
`OMP_NUM_THREADS=24`, `OPENBLAS_NUM_THREADS=24` and `n_jobs=4` to preserve its
reported execution identity.

