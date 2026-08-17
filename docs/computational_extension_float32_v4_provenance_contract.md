# Computational extension mixed-lineage provenance contract v4

Status: **frozen downstream contract**  
Analysis identity: `post_hoc_extension_float32_v4`  
Release label: `post_hoc_computational_extension_float32_v4`

## 1. Publication-eligible workflow lineage

The only permitted mixed lineage is:

| Workflow | Formal output | Protocol version |
|---|---|---|
| context-weight sensitivity | `reports/post_hoc_context_weight_sensitivity_v2` | `post_hoc_context_weight_sensitivity_v2.0` |
| scaffold/source sensitivity | `reports/post_hoc_scaffold_source_sensitivity_v3` | `post_hoc_scaffold_source_sensitivity_v3.0` |
| null/applicability/censoring | `reports/post_hoc_null_applicability_censoring_v3` | `post_hoc_null_applicability_censoring_v3.0` |

The float32 correction SHA-256 is
`75a8b0e4f894cead9d72039eab9e6162a89d2a20c79afceda8a2aae1ffd78f3f`.
The fixed-domain applicability correction SHA-256 is
`9c2d8f90fa546480a37b6e7546cbd9e9a740916325e5dec973f9fa7f8ef899ce`.

The frozen scaffold/source v3 identities are:

- protocol: `5e67fe833b3b5d3f87d38ac3b28b6b7ac2600aebb2756676e031aed7bd2aab83`;
- runner: `6000af04d0abcc5e3f97adf55c8be8876f9abe1bc0cfc8363d173f30aa979f0a`;
- independent all-target identity QA:
  `3636760961c858716d0eb087a4be5d8491058da45a61bc1a9821f73fb262b298`;
- native-thread correction:
  `733b56540fb3917c6c2e3189b7c4d8c4bc33118a5f1fd2db3f132e46e3210456`;
- imported mathematical implementation:
  `449d8821ff31e9a5392a227a86ce78fc09c1660f293449a1fbc850a3ee8ffb34`.

The frozen null/applicability/censoring v3 identities are:

- protocol: `84178cde0edfc0ff31be3d7e12135351f95a2578b9ac69b61d9a5d517caa0dc5`;
- parent runner: `41b99a803b0e090d1bd55950983441eee0ce8edc8b099f86f56387e2f29f3650`;
- shard wrapper: `5aa68bfdc69083b5f30fa1eaedb938f2b3af4ceb86d774c294d672373319637e`;
- merge helper: `95a5dbc4927e53f0a22eba824076026b371261e4f9a8fdfc4afb577f1ff9be43`;
- independent fixed-domain QA:
  `7e51d44a9f0b352475b564047ad99c14e8c81583b4f0ed963abbbde6ab839e83`.

Scaffold/source v1 or v2 results, null v1 or v2 results, symlinked workflow
directories, superseded markers, smoke or identity-preflight directories,
cross-version substitutions and hash mismatches are publication ineligible.

## 2. Scaffold/source v3 execution-identity gate

The formal scaffold/source manifest must have `status=complete`,
`qa_status=PASS`, `native_thread_status=PASS` and
`independent_identity_status=PASS`. Both its formal-settings and all-target
identity-settings flags must be true.

Publication eligibility additionally requires one clean formal execution:
the argument records in `scientific_configuration.json` and
`run_manifest.json` must be exactly equal, must include `overwrite=true` and
`resume=false`, and must contain the full untruncated formal argument set.
A resumed or incrementally completed package remains useful for recovery but
is not publication eligible.

`native_threadpool_audit.json` must report PASS for startup, before and after
the generic-scaffold block, and before and after the source-deletion block.
Every active OpenMP and BLAS pool must expose exactly 24 threads, and every
snapshot must expose at least 24 logical CPUs.

`all_target_exact_identity_audit.json` must report PASS with zero failed
checks for the `formal_complete_package` profile. It binds the exact
no-deletion prediction, parameter, tuning, metric, feature-dimension and split
identity against the frozen confirmatory target-OOD archive.

The enriched `source_deletion_domain_metrics.csv` schema is exactly:

`deletion_source, heldout_target, model_id, param_id, n_train_rows,
n_train_unique_smiles, n_train_scaffolds, n_test_rows, n_test_unique_smiles,
n_test_scaffolds, spearman, pearson, rmse, mae, r2, calibration_intercept,
calibration_slope, rdkit_kept, rdkit_total, rdkit_train_missing,
rdkit_eval_missing, context_dim, chemistry_dim, full_dim`.

## 3. Upstream artefact inventories

Every workflow must contain its exact expected top-level file set.
`artifact_sha256.csv` is verified row by row for safe basename, uniqueness,
size, SHA-256 and exact row set. Context inventories additionally bind file
kind, CSV row/column counts and `verification=PASS`.

`artifact_sha256.csv` is the sole checksum self-reference exception for
context and null. The scaffold/source inventory excludes only itself and the
final `run_manifest.json`; that manifest is bound separately. Its thread and
identity audit files must be present in the exact directory set and in the
inventory. No other omission is allowed.

## 4. Aggregate source-data boundary

The builder accepts exactly 20 named aggregate tables. Every source CSV must
match its ordered column allowlist exactly; extra columns are rejected.
The publication projection removes split/fold identifiers and fit-index hashes
from weighting diagnostics. Row-level predictions, molecular structures,
identifiers, split assignments, unexpected NaN/Inf, non-canonical boolean
tokens and incomplete replicate counts are rejected.

After removing the added `analysis_identity` column, every emitted aggregate
must equal its selected ordered upstream columns cell for cell, including
missing-value positions and parsed dtypes. Matching only the upstream file
hash, schema, row count or checksum inventory is insufficient.

All no-deletion target-OOD prediction sets for both matched ExtraTrees models
must be exactly identical to the frozen confirmatory target-OOD predictions.
Equal-domain source-deletion macros require the complete frozen target-domain
set with finite Spearman values; missing domains may not be silently dropped.

The aggregate index contains:

`table_id, output_file, source_file, n_rows, n_columns, source_size_bytes,
source_sha256, source_inventory_sha256, evidence_identity, protocol_version,
response_dtype, float32_correction_sha256,
fixed_domain_correction_sha256, contains_row_level_records,
contains_molecular_structures`.

Null-derived rows carry the fixed-domain correction hash; other rows carry
`not_applicable`.

## 5. Immutable downstream bindings

`source_data_extension_provenance_v4.json` binds the three workflow
inventories, protocols, runners, QA summaries and run manifests. The
scaffold/source entry additionally binds the native-thread audit, independent
identity audit, independent-QA script, execution correction and imported
mathematical implementation.

The provenance also binds the exact SHA-256 values of the source-data builder,
result summarizer and independent cross-check scripts. It records the complete
ordered independent-check identifier list, its count and a canonical-list
SHA-256. These software identities participate in the source-data generation
ID. Downstream acceptance independently reconstructs every effect record,
permutation record, fit-weight diagnostic, scaffold-group summary and
censoring/randomization diagnostic from the aggregate CSVs.

`source_data_extension_sha256.csv` covers every emitted aggregate table, the
index and the provenance JSON, excluding only itself. Numerical-result and
independent-cross-check JSON records bind the index, checksum inventory,
provenance file, deterministic generation ID and final analysis identity.

The figure bundle uses `reports/publication_figures_v4` and contains exactly
8 stems × 4 formats with unique current size/SHA-256 records. Formal extension
figures bind the current aggregate identity. Figure provenance and formal
figure QA bind each other, the figure manifest and the source-data generation
ID.

Release synchronisation constructs a clean temporary staging tree from an
exact whole-tree allowlist, reconstructs the complete evidence `csv_files`
contract from the staged CSVs, rebuilds and checks both checksum manifests,
and runs the portable verifier before atomically replacing the prior release.
The portable verifier rejects every unexpected file even if checksum
manifests were rebuilt after that file was added.

## 6. Independent prospective-data boundary

No publication-eligible independent prospective dataset was available.
Public narrative outputs may not disclose the count, composition or identity
of any unavailable dataset.
