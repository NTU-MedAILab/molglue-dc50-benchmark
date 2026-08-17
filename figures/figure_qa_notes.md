# Figure QA notes — mixed-lineage v4 bundle

- Backend: Python/matplotlib exclusively.
- Bundle: eight figure stems, each exported as editable SVG, PDF, 600 dpi RGB
  LZW TIFF and 300 dpi PNG (32 quantitative figure files).
- Fig. 1–4 are byte-preserved from the publication v1 bundle; Fig. 2 and
  Fig. 3 include the independently QA-verified molecular-graph sensitivity.
  Fig. 5 and
  Supplementary Figs. S1–S3 are regenerated from aggregate,
  publication-safe mixed-lineage v4 source data.
- Extension width: exactly 170 mm. Contracted heights are 140.970 mm,
  188.468 mm, 135.890 mm and 149.860 mm, respectively.
- No tight bounding box is used; physical canvas dimensions are preserved.
- No synthetic or simulated scientific data are plotted. Conditional
  permutations are declared negative controls and all 100 outcomes are shown.
- Colour is never the only encoding; marker fill/shape, hatching and line
  direction provide redundant encodings.
- Confidence intervals use the estimand-specific global-scaffold bootstrap.
  Fixed-domain applicability intervals preserve the frozen eligible-domain
  set, including explicit non-estimable states.
- No star notation, multiplicity-adjusted significance claim, monotonic-trend
  claim, causal source claim or prospective-validation claim is made.
- The machine-readable manifest binds file size and SHA-256 for every export.
  Formal automated QA is stored separately in
  `computational_extension_figure_qa_summary.json`.

## Manual visual review

- PASS (2026-07-31): all eight PNG previews were inspected at full
  resolution; no panel, title, axis label, tick label, interval, legend or
  annotation was cropped or materially overlapped.
- PASS: primary and supplementary panels remain legible at the declared
  canvas sizes; open/filled markers and patterned censoring classes remain
  distinguishable without colour.

## Figure-to-source-data map

- Fig. 1: frozen v1 provenance, missingness and repeat-audit source tables.
- Fig. 2: publication v1 internal model, contrast, HistGradientBoosting and
  molecular-graph sensitivity tables.
- Fig. 3: publication v1 confirmatory OOD, strict OOD,
  HistGradientBoosting and molecular-graph sensitivity tables.
- Fig. 4: frozen v1 domain, leave-one-domain-out and calibration tables.
- Fig. 5: `source_data_extension_portable_context.csv`,
  `source_data_extension_weighting_bootstrap.csv`,
  `source_data_extension_generic_scaffold.csv` and
  `source_data_extension_source_deletion.csv`.
- Supplementary Fig. S1:
  `source_data_extension_applicability_contrasts.csv`.
- Supplementary Fig. S2:
  `source_data_extension_permutation_models.csv`,
  `source_data_extension_permutation_contrasts.csv` and
  `source_data_extension_permutation_inference.csv`.
- Supplementary Fig. S3: aggregate censoring-overall, by-source, by-target and
  overlap source tables in the computational-extension source-data bundle.
