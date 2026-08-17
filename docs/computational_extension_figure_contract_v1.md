# Computational extension figure contract v1

Status: frozen layout and provenance contract; numerical claims remain
conditional on completed mixed-lineage float32-v4 formal QA. Superseded v1
calculations, null v2 calculations, smoke runs and identity-test runs are
forbidden as scientific figure inputs.

Analysis identity: `post_hoc_extension_float32_v4`  
Formal directory: `reports/publication_figures_v4`

Backend: Python/matplotlib only  
Archetype: asymmetric quantitative grid  
Target: Journal of Cheminformatics  
Export: editable SVG and PDF, 600 dpi RGB TIFF, PNG preview  
Final size: full-page width, 170 mm; figure plus legend must remain below
225 mm total height

Canvas heights:

- Fig. 5: 141.0 mm;
- Supplementary Fig. S1: 188.5 mm;
- Supplementary Fig. S2: 135.9 mm;
- Supplementary Fig. S3: 149.9 mm.

No export uses a tight bounding box, so the SVG/PDF physical canvas and the
600-dpi TIFF/300-dpi PNG raster dimensions retain the declared 170 mm width.

## Global conclusion

The post-hoc stress tests quantify how the validation-regime feature-block
contrast changes under non-portable context removal, representation
reweighting, scaffold redefinition, chemical-novelty strata, training-source
deletion and conditional label randomization.

The final title and conclusion sentence must be rewritten to match the observed
results. No panel may imply that robustness was established before the
analysis was run.

## Non-negotiable source-data gate

Before any figure file is written, the plotting program must verify
`source_data_extension_index.csv`,
`source_data_extension_provenance_v4.json`,
`source_data_extension_sha256.csv` and the frozen confirmatory artifact
checksum table. The central mixed-lineage provenance gate validates all 20
aggregate tables, even when a particular figure uses only a subset. Every
extension table must:

- have evidence and analysis identity
  `post_hoc_extension_float32_v4`;
- declare `response_dtype=float32`;
- carry correction-protocol SHA-256
  `75a8b0e4f894cead9d72039eab9e6162a89d2a20c79afceda8a2aae1ffd78f3f`;
- for null-derived tables, carry the fixed-domain applicability correction
  SHA-256
  `9c2d8f90fa546480a37b6e7546cbd9e9a740916325e5dec973f9fa7f8ef899ce`;
  context and scaffold rows must instead say `not_applicable`;
- resolve to one of exactly three formal parents:
  `post_hoc_context_weight_sensitivity_v2`,
  `post_hoc_scaffold_source_sensitivity_v3` or
  `post_hoc_null_applicability_censoring_v3`;
- match both the indexed formal-source hash and the aggregate source-data
  checksum;
- contain no row-level prediction records or molecular structures.

The provenance JSON must additionally bind the frozen mixed-lineage contract
SHA-256
`a35871eb9657700fc0b8c973449e001398a431830cdacd4db83598019752390d`,
the exact workflow inventories, protocols and runners, and the deterministic
source-data generation ID. Any v1 workflow, scaffold/source v2 result, null v2
workflow, smoke, standalone identity-preflight output, symlink, cross-version
substitution or
`SUPERSEDED_DO_NOT_USE` lineage is a hard failure. The
`publication_validation_v1` directory name is the frozen submission-package
version; it is acceptable only when its index and v4 provenance record prove
the mixed lineage above.

No collaborator-restricted data, prediction, identifier, summary or derived
statistic may enter the aggregate source data, figures, manifest, QA record or
release.

## Main Fig. 5 — robustness map

Core question: Which reviewer-facing perturbations change the internal context
increment or the chemistry-only transfer advantage?

Panel map:

- **a, hero:** held-source and held-target full-minus-chemistry Spearman
  contrasts for original full context and held-axis-portable context, with a
  zero reference and paired global-scaffold intervals;
- **b:** internal full-minus-chemistry contrasts under uniform-row,
  compound-equal and source-target-domain-balanced fitting;
- **c:** internal full and chemistry-only performance under original and
  generic Murcko scaffold grouping, visually separating absolute performance
  from the paired feature-block increment;
- **d:** target-OOD full-minus-chemistry contrast after deleting each training
  source, alongside the undeleted reference.

Evidence hierarchy:

- hero evidence: portable-context OOD decomposition;
- validation evidence: representation-weight sensitivities;
- extrapolation stress test: generic scaffold grouping;
- influence evidence: training-source deletion.

Statistics:

- points are observed estimates;
- horizontal lines are 95% percentile global-scaffold-bootstrap intervals from
  10,000 resamples; feature-block contrasts are paired, whereas the absolute
  chemistry-only and full-model intervals in panel c are model-specific;
- positive Spearman contrasts favour the full/portable-full model;
- fixed-domain source-deletion estimates are post-hoc sensitivities and are not
  a population sample;
- no star notation or multiplicity-adjusted significance claim.

Reviewer risks:

- portable context is a restricted feature block, not an external dataset;
- weighting changes the fit, not the original evaluation estimand;
- generic Murcko is one coarser scaffold definition, not a complete chemical
  extrapolation proof;
- source-deletion effects cannot identify a causal database-quality mechanism.
- panels a, b and d use related but distinct estimands; their common horizontal
  scale aids visual calibration and does not make them exchangeable.

## Supplementary Fig. S1 — applicability profile

Core question: How do ranking, error and the full-minus-chemistry difference
change across prespecified maximum-train-Tanimoto bins?

Panel map:

- internal scaffold-disjoint bins;
- source OOD bins;
- target OOD bins;
- strict OOD bins or a compact risk-coverage inset.

All observations are retained. Empty, sparse or constant bins remain visible
as not estimable. Bin counts and global-scaffold counts are printed below every
bin. Spearman is dimensionless; RMSE differences are reported in pDC50 units.
Positive values favour the context-inclusive model for both displayed
contrasts.

For every OOD metric and bin, the eligible held-domain set is frozen before
the global-scaffold bootstrap. A replicate is valid only when every eligible
domain is represented and its metric is finite. Formal intervals use 10,000
requested resamples and require at least 5,000 valid replicates. The eligible
domain set must never change across replicates and no domain-stratified
fallback may replace this estimand.

Interval status is encoded without inventing information:

- `ESTIMATED`: draw the observed point and its paired 95% interval;
- `NON_ESTIMABLE_STRUCTURAL`: when any fixed eligible domain has fewer than
  two global scaffolds, retain the finite descriptive point as an open diamond,
  omit the interval line and label it `CI NE*`;
- `NON_ESTIMABLE_TOO_FEW_VALID`: retain a finite descriptive point as an open
  diamond, omit the interval line and label it `CI NE`;
- if the descriptive point itself is non-estimable, draw a grey `x` at the
  zero reference solely as a locator and label it `point NE`.

The asterisk is defined as “structural: a fixed eligible domain has fewer than
two scaffolds.” All descriptive point estimates, including point-only states,
must contribute to the plotted axis range. Bins are discrete: no line connects
them and no monotonic-trend test or monotonicity claim is made.

Reviewer risks:

- a bin-wise interval is descriptive and does not support a multiple-testing
  or monotonic-trend claim;
- maximum-train Tanimoto is one applicability coordinate, not a proof of
  chemical extrapolation;
- an equal-domain macro interval can be structurally non-estimable even when
  its descriptive point remains finite.

## Supplementary Fig. S2 — repeated negative controls

Core question: Do the observed chemistry-only, full-model and paired-contrast
scores exceed their within-source-target conditional permutation
distributions?

Panel map:

- chemistry-only Spearman null distribution;
- full-model Spearman null distribution;
- full-minus-chemistry Spearman null distribution;
- optional RMSE contrast null distribution.

The observed value is a dark vertical line; the 100 permutation outcomes are
shown in full without downsampling. Empirical tail probabilities are printed
as exact fractions and decimals.

The legend must state that randomization is conditional within source × target
strata and that the finite one-sided p-value floor is `1/(100+1)=0.0099`.
The shaded region is the empirical 2.5th--97.5th percentile range, not a
parametric confidence interval. Failure to cross that range alone is not
described as a multiplicity-adjusted test.

## Supplementary Fig. S3 — endpoint selection boundary

Core question: Which sources and targets contribute exact, interval,
one-sided and missing DC50 records?

Panel map:

- overall relation-type composition;
- source-stratified 100% stacked composition;
- all target categories shown as total parsed-record count versus exact-record
  fraction, with labels restricted to a small declared set of largest or
  extreme categories while no target point is omitted;
- compound/scaffold overlap between censored candidates and the strict-exact
  core where reproducibly available.

This is descriptive and must not visually imply that censored observations
were modelled as exact labels.

The count unit must be explicit: 3,117 denotes parsed-record table rows, not
3,117 unique record IDs. Panel c shows every target category and labels only a
declared subset; it does not omit unlabelled points. The selection audit
changes units across its stages, so neither this figure nor its legend may
describe the sequence as a single row-wise attrition rate or inclusion
probability.

## Visual vocabulary

- chemistry-only: muted blue, open circle;
- original full: deep navy, filled circle;
- portable full: violet, diamond;
- uniform/undeleted reference: dark navy;
- compound-equal: teal;
- domain-balanced: ochre;
- generic scaffold: grey outline;
- favourable delta: restrained green;
- unfavourable delta: restrained red;
- null permutations: pale grey-blue with observed value in deep navy;
- censoring classes: exact in navy, range in teal, one-sided in muted ochre,
  missing in light grey, with solid/forward-hatch/backward-hatch/dot patterns
  as redundant encoding in composition bars.

Color is never the only encoding. Marker shape, fill and line style are
redundant encodings. The same model or evidence status keeps the same visual
identity in every panel.

## Legend and source-data minimum

Every quantitative legend defines:

- the evaluation regime and feature blocks;
- what `n` denotes;
- row, compound, scaffold and fixed-domain counts where applicable;
- point and interval definitions;
- bootstrap or permutation replicate count;
- post-hoc analysis identity;
- exact source-data file.

Metric notation is fixed:

- `Spearman ρ` for rank correlation;
- `ΔSpearman ρ = full − chemistry`;
- `ΔRMSE = chemistry − full`, in pDC50 units;
- `domain macro` means equal weight across the prespecified held domains.

Legends end with: `Source data are provided as a Source Data file.`

Caption-ready minimums:

- **Fig. 5:** define panels a--d, the full, portable-full and chemistry-only
  feature blocks, internal versus held-axis protocols, points, 95% intervals,
  pairing, 10,000 bootstrap resamples and the post-hoc sensitivity status.
- **Supplementary Fig. S1:** define the four fixed maximum-train-Tanimoto bins,
  `n` as rows, `S` as global scaffolds, `D` as prespecified held domains,
  positive contrast direction, the fixed eligible-domain bootstrap,
  `CI NE` as interval non-estimable, `CI NE*` as structurally interval
  non-estimable with its descriptive point retained, and `point NE` as a
  non-estimable descriptive point. State that bins are unconnected and no
  trend test was performed.
- **Supplementary Fig. S2:** define the two absolute scores and two paired
  contrasts, all 100 conditional permutations, the null percentile band,
  observed line and finite one-sided empirical p-value formula.
- **Supplementary Fig. S3:** define the relation classes, row count rather than
  unique-record count, target-label rule, unique compound/scaffold overlap and
  the fact that censored values were not converted to point labels for fitting.

## Image-integrity and QA

These are generated quantitative figures; no microscopy or photographic image
adjustment applies. Plotting uses all eligible observations or an explicitly
declared aggregate. The formal bundle contains exactly eight named figure
stems, each in SVG, PDF, TIFF and PNG (32 figure files total), plus its manifest,
figure provenance and QA record. Each submitted figure file must remain below
10 MB. Source preflight, editable-text checks, embedded-font checks,
physical-size rendering, TIFF/PNG DPI checks, clipping/overlap inspection and
colour-vision/grayscale checks are required before manuscript insertion.

The plotting program builds the entire bundle in a validated sibling staging
directory. Publication is a whole-directory rename. If a current v4 bundle
exists it is first renamed to a unique recovery backup; a failed publish must
restore that backup. Per-file updates into a live formal directory are
forbidden.

The size, file and legend limits above were checked against the Journal of
Cheminformatics submission guidelines on 2026-07-29.

## QA release gate

Formal delivery requires:

- syntax compilation and the Nature-figure source preflight with zero FAIL;
- exactly these eight stems, with no duplicate or additional figure filename:
  `fig1_validation_gradient`, `fig2_internal_evidence`,
  `fig3_domain_transfer`, `fig4_boundary_and_calibration`,
  `fig5_robustness_map`, `supp_fig_s1_applicability_profile`,
  `supp_fig_s2_permutation_controls` and
  `supp_fig_s3_endpoint_selection_boundary`;
- exactly four formats per stem (`.svg`, `.pdf`, `.tiff`, `.png`), hence 32/32
  non-symlinked, non-empty exports;
- a unique manifest row, current size and current SHA-256 for every one of the
  32 files;
- exact 170 mm SVG/PDF width and contracted height for Fig. 5 and
  Supplementary Figs. S1--S3;
- selectable SVG text and embedded PDF fonts for all eight figures; extension
  canvases must match the declared dimensions and frozen Fig. 1--4 formats
  must agree with their own SVG canvases;
- 600-dpi LZW-compressed RGB TIFF and 300-dpi PNG for all eight figures;
- every file below 10 MB;
- required panel labels, axes, interval definitions and censoring-boundary text
  present in the rendered SVG;
- colour and grayscale visual inspection at final size;
- a machine-readable figure manifest and mixed-lineage float32-v4 provenance
  record;
- figure provenance that binds the current source index SHA-256, source
  checksum-inventory SHA-256, source provenance SHA-256, source-data generation
  ID, figure manifest SHA-256, plotting-script SHA-256, frozen figure-contract
  SHA-256, frozen figure-QA-script SHA-256, and the canonical SHA-256 and count
  of the exact formal required check-ID set. The figure generation ID covers
  every one of these identities;
- formal QA JSON that independently binds those same four source-data
  identities plus the current figure-manifest and figure-provenance SHA-256
  values, the frozen figure contract and QA script, and the exact ordered
  required check-ID set, its count and canonical SHA-256. Every required check
  must occur exactly once and report `PASS`; a shortened, duplicated,
  additional, `WARN` or stale-binding record is invalid;
- independent enforcement of the frozen contract SHA-256, QA-script SHA-256
  and exact formal required check-ID identity by the figure QA, release-sync
  gate and portable release verifier. A top-level `PASS` alone is never
  sufficient.

A schema/rendering smoke test may use temporary non-scientific inputs only
under `/tmp`; its status is `SMOKE_TEST_ONLY` and can never satisfy this formal
release gate. Smoke, v1, null v2 and superseded paths are rejected in formal
mode. The null-v3 verification note is intentionally not hash-locked by the
figure gate because it is updated after the formal run; immutable protocol,
runner, wrapper, merge-helper, independent-QA and correction hashes provide
the executable lineage lock.
