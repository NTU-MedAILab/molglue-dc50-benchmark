# MolGlue DC50 reproducible dataset builder

This repository reconstructs the analysis dataset used in the associated
molecular-glue DC50 study from four owner-hosted databases: MGTbind, MGDB,
MolGlueDB and TPDdb. It downloads the frozen official source snapshots,
validates them byte-for-byte, parses DC50 evidence, converts concentration
units to nM, applies predeclared quality-control rules, aggregates duplicate
contexts, creates the deterministic molecule-grouped train/test split and
standardizes assay-context fields.

The result is an **author-curated harmonized analysis dataset**, not a claim
that the authors own the underlying third-party database records.

## Data-rights boundary

No source export or row-level derivative is included in this repository.
`data/raw/` and `data/processed/` are git-ignored. Public accessibility of a
download is not the same as permission to redistribute it. Before running the
download command, read [THIRD_PARTY_DATA.md](docs/THIRD_PARTY_DATA.md). The
code, manifests, hashes, schemas and aggregate counts can be public; the raw
and harmonized row-level files must remain local unless the relevant database
owner grants permission or the journal confirms another lawful route.

## Exact environment

Python 3.11 is required. The exact study environment is pinned in
`pyproject.toml`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
pytest
python scripts/check_public_release.py
```

## One-command exact reconstruction

```bash
python scripts/reproduce.py all --acknowledge-third-party-notice
```

The command performs five fail-closed stages:

1. download only from the official endpoints in `config/source_manifest.json`;
2. validate every required source by size and SHA-256;
3. build all six analysis tables locally;
4. record parser, exclusion, standardization and software audits; and
5. require exact output row counts, column counts and SHA-256 identities.

The final line must report:

```text
PASS: 6 output tables match the frozen row, column, and SHA-256 identities.
```

## Existing local source files

If the official endpoints are temporarily unavailable, place lawfully
obtained files in `data/raw/` using the canonical filenames listed in the
manifest, then run:

```bash
python scripts/reproduce.py verify-sources
python scripts/reproduce.py build
python scripts/reproduce.py verify
```

The builder will not accept a different snapshot under an old filename. If an
official database has changed, preserve the new access date and checksum and
treat the run as a temporal-validation dataset; do not compare its output hash
to the frozen study identity.

## Outputs

All outputs are written locally to `data/processed/`:

- `all_molglue_dc50_parsed_records.csv`: every parsed candidate DC50 record;
- `all_molglue_dc50_strict_exact_records.csv`: exact, positive, structure- and
  context-valid observations before aggregation;
- `all_molglue_dc50_qc_train_test.csv`: median-aggregated QC table and split;
- `all_molglue_dc50_qc_train_test_standardized_context.csv`: frozen modeling
  input with raw and normalized context columns;
- `all_molglue_dc50_train.csv` and `all_molglue_dc50_test.csv`; and
- `build_report.json` and `verification_report.json`.

See [DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) for field definitions and
the manifest for source versions, official URLs, citations and checksums.

## Reproducibility identity

The exact historical result contains 3,117 parsed observations, 1,758 strict
observations and 1,560 aggregated QC rows (1,252 train; 308 test). Expected
identities live in `config/expected_outputs.json`; they contain no row-level
third-party data.

## Citation and authorship

Use `CITATION.cff` for software citation and also cite all four source database
articles listed in the source manifest. Jinsong Shao developed the code,
executed the experiments, collected the data and drafted the manuscript. Li
Wang conceived the experimental ideas, provided facilities, funding and
equipment, validated the experimental accuracy and revised the manuscript.

The author-owned code is released under the repository MIT License. The MIT
License does not apply to third-party database exports or their row-level
derivatives, which are excluded from the repository.

GitHub Actions runs the unit tests and the public-release boundary check on
every push and pull request. The full end-to-end reconstruction is intentionally
not run in public CI because it would download and process third-party exports
on every event.
