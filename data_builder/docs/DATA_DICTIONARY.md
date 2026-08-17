# Data dictionary and transformation contract

## Unit and activity contract

Only DC50 measurements are retained. Concentrations are converted to nM using
`1 pM = 0.001 nM`, `1 uM/μM/µM = 1,000 nM`, `1 mM = 1,000,000 nM` and
`1 M = 1,000,000,000 nM`. Exact relations (`=` and approximate values parsed
as exact in the historical contract) may enter strict QC. Censored relations,
ranges, missing/non-positive values and percentage endpoints do not.

`pDC50 = 9 - log10(DC50_nM)`.

## Parsed-record fields

| Field | Meaning |
|---|---|
| `record_id` | Deterministic parser-level record identifier |
| `source_database` | MGTbind, MGDB, MolGlueDB or TPDdb |
| `source_id` | Source database record identifier |
| `molecular_glue_name` | Source-reported compound name |
| `smiles` | Source-reported SMILES |
| `canonical_smiles` | RDKit isomeric canonical SMILES |
| `structure_qc_error` | Missing/invalid structure reason, otherwise empty |
| `recruiting_protein` | Parsed recruiter or ligase context |
| `recruiting_protein_uniprot` | Source-reported recruiter accession where available |
| `target_protein` | Parsed degraded target |
| `target_uniprot` | Source-reported target accession where available |
| `target_assignment` | Provenance for target/recruiter assignment logic |
| `cell_line` | Source-reported or text-extracted cellular context |
| `activity_time` | Source-reported treatment or assay time |
| `dc50_relation` | Exact, censored or range relation |
| `dc50_raw_value`, `dc50_raw_unit` | Parsed source-scale value and unit |
| `dc50_nM` | Normalized concentration in nM |
| `pDC50` | Negative base-10 molar logarithm |
| `assay_method` | Source assay descriptor |
| `mode_of_action` | Source mode/subtype descriptor |
| `raw_activity_text` | Source evidence string used by the parser |
| `source_url` | DOI, PubMed or source-provenance URL |
| `source_note` | Parser note or exclusion-relevant annotation |

## Strict QC

A row must have an exact positive DC50, a valid canonical structure, a
non-placeholder recruiter and target, and distinct recruiter/target identities.
MGDB rows whose recruiter was only a fallback or remained missing are excluded.

## Aggregation and split

Strict rows are grouped by canonical SMILES, recruiting protein, target
protein and cell line. DC50 is the median; minimum, maximum and measurement
count are retained, and provenance values are sorted and joined. The test set
is a deterministic 20% `GroupShuffleSplit` on canonical SMILES with random
state `260531`, so no canonical molecule appears in both partitions.

## Standardized modeling view

Seven context columns are normalized: source database, recruiter, target,
cell line, assay method, activity time and mode of action. Their original
values are preserved as `raw_*` columns. `species` is currently
`unspecified`. The exact mapping rules are executable and reviewable in
`src/molglue_dc50/context.py`; `build_report.json` records changed-row and
unique-category counts for each column.

