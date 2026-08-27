# Post-hoc reviewer-driven revision protocol v1.1

## Status and scope

This protocol was written before inspecting any outputs from the analyses
defined below. All analyses are explicitly post-hoc and use only the frozen
1,560-row strict-exact molecular-glue DC50 core and its saved predictions.
No prospective or collaborator-restricted observations are used.

Version 1.1 clarifies the grouped-context labels to match the already frozen
executable definitions before any grouped-ablation effect estimates were
inspected. It does not change rows, splits, features, models, tuning, seeds,
metrics or decision rules.

## Analysis 1: matched-estimand validation-regime comparison

The internal scaffold-disjoint and held-domain predictions will be compared on
the same row universe, with the same exact domains, equal-domain aggregation
and paired global-scaffold bootstrap samples.

- Source axis: MGTbind, MGDB, MolGlueDB and TPDdb exact-source rows.
- Target axis: VAV1, CSNK1A1, GSPT1, WIZ, CDK2, CCNK+CDK12, IKZF2 and IKZF1
  exact-target-label rows.
- Models: chemistry-only ExtraTrees and chemistry-plus-context ExtraTrees.
- Primary metric: equal-domain macro Spearman correlation.
- Secondary metrics: pooled Spearman correlation and equal-domain macro RMSE.
- Primary contrast: full minus chemistry for Spearman; chemistry minus full
  for RMSE, so positive values favour the context-inclusive model.
- Regime-change contrast: held-domain context contrast minus internal context
  contrast. Negative values indicate attenuation of the context increment
  under held-domain validation.
- Uncertainty: 10,000 paired resamples of the global Bemis–Murcko scaffold
  clusters over the common row universe within each axis. A macro replicate is
  valid only when every fixed domain has a finite metric.

Domain-level model metrics and paired contrasts will retain all fixed domains,
sample sizes, unique compounds, scaffold counts and valid-bootstrap counts.
A descriptive matched leave-one-domain calculation will report how the paired
regime-change average changes after omitting each fixed domain; it will not be
assigned a population-domain confidence interval.

## Analysis 2: within-domain ranking readout

The domain-level results from Analysis 1 will be used to state explicitly what
the ranking estimand represents. The eight fixed target-label domains and four
fixed source domains will be reported without selective omission. No
multiplicity-adjusted per-domain significance claim will be made.

## Analysis 3: grouped context ablation

Matched chemistry-plus-context ExtraTrees models will be compared using the
same frozen internal and held-domain splits and the same six-member tree grid.
The add-on context blocks are:

1. provenance: source database and the source–target interaction internally;
2. biological/assay: recruiting protein, target protein, cell line, assay
   method, activity time and mode of action, plus recruiter–target,
   target–cell and assay–target interactions internally;
3. missingness: binary indicators derived from the pre-sanitization missing or
   unspecified state of the six non-source context fields; and
4. full context: the original main effects and interactions (existing frozen
   predictions).

For held-source analysis, provenance is not treated as portable and no
provenance model is fitted. For held-target analysis, the provenance model
retains only the source main effect; the biological/assay model removes target
protein and every target-containing interaction. All other block definitions
are fixed before results are inspected.

The grouped-ablation analyses are post-hoc explanatory sensitivities. They do
not convert associations into causal biological effects.

## Decision rules for manuscript claims

1. The manuscript may retain a validation-regime attenuation claim only if the
   matched equal-domain context contrast is more favourable internally than in
   held-domain validation.
2. If a fixed-domain average is strongly influenced by one or two domains, the
   abstract and conclusion must report heterogeneity and avoid universal
   transfer language.
3. If provenance or missingness accounts for most of the internal increment,
   the interpretation will be restricted to recorded-data shortcuts.
4. If a portable biological or assay block transfers favourably, the
   conclusion will be component-specific rather than a claim about context as
   a whole.

## Reporting boundary

These analyses remain restricted to directly reported point DC50 records.
Range and one-sided endpoints are not converted to point labels, and the
strict-exact subset is not represented as an unbiased sample of all source
records.
