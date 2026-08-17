# Data dictionary and field-level release boundary

## Frozen private analysis table

The model input has 1,560 rows and 30 columns. It is identified by checksum
but is not included because it combines row-level fields from four databases
whose redistribution terms are unresolved.

Empty textual values were normalized to an explicit missing category for
modeling where required; empty numeric values remain missing rather than
zero. `pDC50` is defined as:

`pDC50 = 9 - log10(DC50 in nM)`.

| Column | Type | Definition | Release boundary |
|---|---|---|---|
| `qc_id` | string | Project QC row identifier | Restricted row-level field |
| `canonical_smiles` | string | RDKit-canonicalized structure used for compound grouping | Restricted structure derivative |
| `recruiting_protein` | category | Standardized recruited ligase/adaptor context | Restricted row-level normalized context |
| `target_protein` | category | Standardized degraded target | Restricted row-level normalized context |
| `cell_line` | category | Standardized assay cell line; unspecified when unavailable | Restricted row-level normalized context |
| `record_id` | string | Project/source record linkage identifier | Restricted row-level identifier |
| `source_database` | category | Harmonized source membership; combined values preserve multi-source provenance | Restricted at row level; aggregate counts released |
| `source_id` | string | Original database record identifier | Restricted third-party identifier |
| `molecular_glue_name` | string | Reported compound/name label | Restricted third-party row value |
| `smiles` | string | Source-reported or normalized structure string | Restricted third-party structure |
| `dc50_nM` | float | Harmonized degradation concentration in nanomolar | Restricted row-level activity |
| `n_measurements` | integer | Number of source measurements summarized in the row | Restricted row-level provenance |
| `dc50_nM_min` | float | Minimum contributing DC50 in nanomolar | Restricted row-level activity |
| `dc50_nM_max` | float | Maximum contributing DC50 in nanomolar | Restricted row-level activity |
| `target_uniprot` | string | Target UniProt accession when mapped | Restricted row-level annotation |
| `recruiting_protein_uniprot` | string | Recruiting-protein UniProt accession when mapped | Restricted row-level annotation |
| `activity_time` | string/category | Harmonized exposure/time context | Restricted row-level context |
| `assay_method` | category | Harmonized assay/readout method | Restricted row-level context |
| `mode_of_action` | category/string | Harmonized mechanism/context label where reported | Restricted row-level context |
| `source_url` | string | Source record or provenance URL | Restricted row-level source linkage |
| `pDC50` | float | Log-transformed endpoint defined above | Restricted row-level derived activity |
| `split` | category | Historical train/test routing label retained for provenance; not the nested-validation fold assignment | Restricted row-level project metadata |
| `raw_source_database` | string | Pre-sanitization source label | Restricted row-level audit field |
| `raw_recruiting_protein` | string | Pre-sanitization recruiting-protein label | Restricted row-level audit field |
| `raw_target_protein` | string | Pre-sanitization target label | Restricted row-level audit field |
| `raw_cell_line` | string | Pre-sanitization cell-line label | Restricted row-level audit field |
| `raw_assay_method` | string | Pre-sanitization assay-method label | Restricted row-level audit field |
| `raw_activity_time` | string | Pre-sanitization activity-time label | Restricted row-level audit field |
| `raw_mode_of_action` | string | Pre-sanitization mechanism label | Restricted row-level audit field |
| `species` | category/string | Assay/source species where available | Restricted row-level context |

## Released aggregate evidence

All CSV files under `evidence/source_data/` are non-row-level aggregate
tables. Their exact schemas and row counts are machine-checked against
`EVIDENCE_CONTRACT.json`.

### Common identifiers

| Field | Definition |
|---|---|
| `protocol` | `scaffold`, `compound`, `source_ood` or `target_ood` evaluation axis |
| `model_id` | Frozen learner/feature-block identifier |
| `record_type` | Absolute `model` estimate or ordered `contrast` |
| `heldout_group` | Source or target used as the held domain |
| `analysis_label` / `analysis_mode` | Confirmatory, formal-extension, sensitivity or exploratory evidence status |
| `contrast_id` | Named ordered model comparison |
| `first_model`, `comparator_model` | Models defining the ordered contrast |
| `weight_mode` | Uniform-row, compound-equal or clipped source–target-balanced model fitting |
| `deletion_source` | Source token removed from a target-OOD training set; `none` is the within-script baseline |
| `similarity_bin` | Prespecified interval of maximum train-set Morgan Tanimoto similarity |
| `permutation_id` | One independently seeded conditional label-randomization run; not a sample |
| `relation_class` | Exact equality, range, one-sided or missing/unparsed endpoint relation |

### Counts and independent units

| Field/prefix | Definition |
|---|---|
| `n_rows`, `n_test_rows` | Database observations; not independent compound counts |
| `n_unique_compounds`, `n_test_unique_smiles` | Unique canonical compounds |
| `n_scaffolds`, `n_test_scaffolds` | Unique Bemis–Murcko scaffold groups |
| `n_domains` | Equally weighted fixed held-out domains |
| `n_clusters` | Global scaffold clusters available to the paired bootstrap |
| `n_bootstrap` | Number of paired bootstrap draws, normally 10,000; not sample size |
| `n_bootstrap_requested` | Requested paired bootstrap draws for applicability contrasts |
| `n_permutations_finite` | Conditional-randomization runs yielding a finite statistic |
| `*_n_valid` | Bootstrap draws with a finite statistic |

Folds, repeats, model seeds and bootstrap draws are computational devices and
are not independent sample sizes.

### Metrics

| Field/prefix | Definition and unit |
|---|---|
| `spearman*` | Spearman rank correlation, dimensionless; missing when undefined |
| `pearson*` | Pearson correlation, dimensionless |
| `rmse*` | Root mean squared error in pDC50 units |
| `mae*` | Mean absolute error in pDC50 units |
| `r2*` | Coefficient of determination; may be negative |
| `calibration_intercept*` | Intercept in observed = intercept + slope × predicted, pDC50 units |
| `calibration_slope*` | Slope in observed = intercept + slope × predicted |
| `*_observed` | Point estimate in the frozen observed data |
| `*_bootstrap_mean` | Mean across valid bootstrap draws |
| `*_ci_low`, `*_ci_high` | 2.5th and 97.5th percentile bounds |
| `domain_macro_*` | Equal-weight mean over the fixed held domains; undefined domains are not replaced by zero |
| `delta_*` | Ordered paired model contrast with its direction declared in the table |
| `empirical_p` | Finite-permutation-corrected one-sided tail fraction; resolution is limited by 100 runs |

Unless a table states otherwise:

- Spearman contrast is `first model - comparator`;
- RMSE contrast is `comparator - first model`; and
- positive values favour the first model for both metrics.

Empty CSV fields mean not applicable or undefined for that record type. They
are never imputed to numerical zero in the deposited evidence.

## Excluded source-data tables

The following manuscript-building tables are deliberately absent because
they expose compound- or row-level source derivatives:

- `modeling_context_sanitization_changes.csv`
- `compound_repeat_audit.csv`
- `exact_context_repeat_audit.csv`
- every `strict_canonical_smiles_*` table
- row-level prediction, split-index and paired-applicability tables

GitHub v2 includes `source_data_repeat_summary.csv`, a one-row table containing
only the seven counts required by Fig. 1. It replaces the two excluded repeat
tables for plotting and enables a pixel-identical PNG render without exposing
SMILES or contexts. Figures 1–5 and Supplementary Figs. S1–S3 therefore have
all required redistribution-safe aggregate source tables. Supplementary Fig.
S4 is provided separately from aggregate learning-curve tables.
