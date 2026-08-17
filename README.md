# Molecular-glue DC50 benchmark — GitHub v2

This package is designed to reconstruct and verify the complete **CPU**
analysis chain for the associated retrospective molecular-glue DC50 benchmark. It combines the
four-source deterministic data builder, frozen analysis implementations,
formal protocols, independent exact-output verifier, aggregate manuscript
source data and submission figure bundle.

## Rights-safe design

The database owners did not reply to the authors' redistribution-permission
requests. This release therefore does not rely on permission being granted and
does not redistribute source exports or row-level derivatives. Each reproducer
downloads the files from the four database owners (or supplies an exact local
cache obtained from those owner endpoints), verifies the frozen SHA-256
identities and builds the 1,560-row table in an ignored clean workspace.

The GitHub payload contains no raw database export, harmonized row-level table,
SMILES-level audit, row-level prediction or fitted model. Do not add these
files to a public fork. Public downloadability is not treated as a licence to
republish database content.

## Public-payload verification

```bash
python scripts/reproduce_release_v2.py verify-public
```

This standard-library gate scans the payload and fails if known raw,
harmonized or prediction artifacts are present.

## Current validation status

The public payload, four-source builder, exact six-table reconstruction and
formal confirmatory CPU core have passed. The first clean full-chain run then
stopped at the matched-context stage because that historical script binds a
parent runtime-manifest byte hash containing non-portable execution metadata.
The portable-v2 downstream identity repair and remaining formal refit are
therefore still in progress. Do not interpret this repository state as a
completed end-to-end acceptance run; see `GITHUB_V2_BUILD_REPORT.md`.

## Complete clean CPU reproduction

Two exact environments are used deliberately:

- builder: Python 3.11, NumPy 1.26.4, pandas 2.3.3, RDKit 2023.09.6 and
  scikit-learn 1.3.0 (`data_builder/pyproject.toml`);
- analysis: Python 3.10.19 and the versions in `environment.yml` and
  `requirements-lock.txt`.

From an empty work directory, using owner-hosted downloads:

```bash
python scripts/reproduce_release_v2.py all \
  --work-dir /private/work/molglue-dc50-v2 \
  --builder-python /path/to/builder-python \
  --analysis-python /path/to/analysis-python
```

If the exact official files have already been downloaded, they may be reused
without a network request:

```bash
python scripts/reproduce_release_v2.py all \
  --work-dir /private/work/molglue-dc50-v2 \
  --source-cache /private/official-source-cache \
  --builder-python /path/to/builder-python \
  --analysis-python /path/to/analysis-python
```

Interrupted computations can be continued with the same arguments plus
`--resume`. Runtime logs, rebuilt data and row-level predictions remain under
the private work directory. The final gate compares every declared scientific
CSV artifact with frozen independent SHA-256 ledgers and must ultimately
report a complete PASS.

## CPU scope and GPU boundary

The chain reruns internal scaffold/compound validation, matched context
ablation, source/target OOD validation, OOD diagnostics, strict
domain-plus-scaffold OOD, HistGradientBoosting sensitivity, learning curve,
context/weight sensitivity, scaffold/source sensitivity, 100 conditional
permutations, applicability analysis and censoring analysis. The four
permutation ranges are isolated and executed in parallel on CPU.

The molecular-graph sensitivity is not silently substituted by a CPU model.
Its previously generated aggregate evidence is included as a clearly labelled
`GPU_REFERENCE_ONLY` sensitivity. No GPU is requested by this v2 reproduction.

## Directory map

| Path | Purpose |
|---|---|
| `data_builder/` | official-source acquisition, hash validation and six-table reconstruction |
| `analysis_code/` | byte-preserved scientific implementations and QA code |
| `docs/` | frozen protocols and scientific identity documents |
| `reference/cpu_artifact_ledgers/` | independent expected hashes for rebuilt CPU CSV artifacts |
| `evidence/source_data/` | publication-safe aggregate evidence only |
| `figures/` | editable and submission-format figure bundle |
| `scripts/` | clean-workspace orchestrator, public-boundary gate and result verifier |

## Remaining release decisions

Before the immutable journal release, the authors must choose a licence for
author-owned aggregate evidence and figures, complete the portable-v2
downstream refit, create the immutable tag and archive that exact tag. The code
is MIT licensed and its repository URL is fixed. See `RELEASE_BLOCKERS.md`.
