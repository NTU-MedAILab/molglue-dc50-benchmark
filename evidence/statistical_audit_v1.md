# Statistical audit v1

Status: internal pre-submission audit  
Scope: frozen retrospective analyses, two independently specified post-hoc
learner-family sensitivities, and the frozen post-hoc computational extension;
no prospective data

## Overall assessment

The current benchmark/evaluation claim is statistically defensible if the
manuscript keeps its independent units, estimands and evidence identities
explicit. The work does not yet support predictive-utility, calibration or
mechanistic claims. The molecular-graph sensitivity is GPU-dependent in its
formal execution identity but is not required to reproduce the confirmatory
ExtraTrees results.

The computational extension addresses held-axis feature portability,
representation weighting, generic-scaffold grouping, training-source
deletion, chemical-novelty strata, repeated conditional label randomization
and endpoint selection/censoring. These analyses increase robustness and
failure-mode coverage but remain post-hoc; they do not convert the historical
benchmark into prospective evidence.

## Design and independent units

| Item | Audit result |
|---|---|
| Analysis rows | 1,560 strict-exact observations |
| Chemical units | 1,137 canonical compounds and 667 Bemis–Murcko scaffolds |
| Primary internal split | repeated nested scaffold-disjoint cross-validation |
| Secondary internal split | repeated nested compound-disjoint cross-validation |
| OOD domains | 4 frozen sources and 8 frozen exact target labels |
| OOD deletion | held domain plus exact test-compound removal |
| Strict sensitivity | additional test-scaffold removal; explicitly post-hoc |
| Independent unit | scaffold cluster for retrospective uncertainty; fixed domain for macro averaging |
| Not independent | rows sharing chemistry/context, folds, repeats, random seeds, technical replicates and bootstrap draws |

The manuscript must never report outer folds, model seeds or 10,000 bootstrap
replicates as sample size. The repeated outer predictions are averaged
row-wise before the internal point estimate and interval.

## Estimands and uncertainty

- The primary internal ranking estimand is pooled scaffold-OOF Spearman
  correlation.
- The primary OOD ranking estimand is equal-domain-weighted macro Spearman,
  conditional on the four frozen sources or eight frozen targets.
- RMSE is reported on pDC50. Its contrasts use `comparator − first`, so positive
  values favour the first model; Spearman contrasts use `first − comparator`.
- Internal confidence intervals use 10,000 paired scaffold-cluster percentile
  bootstrap samples.
- OOD confidence intervals use 10,000 paired global-scaffold bootstrap samples
  over the combined test universe so that scaffolds shared across domains are
  resampled together.
- Model comparisons use identical resamples. This paired design is appropriate
  and should be stated in every quantitative legend.
- Confidence intervals are descriptive uncertainty intervals, not evidence
  that rows or domains were sampled independently from a well-defined
  population of all future targets/databases.

Percentile intervals are acceptable for the current effect-size emphasis, but
the manuscript should avoid claiming exact frequentist coverage under the
small fixed-domain design.

## Evidence identity

| Analysis | Permitted label |
|---|---|
| Frozen core scaffold/compound validation | confirmatory |
| Frozen held-source/held-target analysis | confirmatory |
| Same-learner context-only ExtraTrees comparison | formal matched extension |
| Calibration, LODO and strict canonical aggregation | post-hoc exploratory |
| Strict domain-plus-scaffold exclusion | post-hoc sensitivity |
| HistGradientBoosting learner-family check | post-hoc sensitivity |
| Molecular-graph learner and representation check | post-hoc sensitivity |
| Portable-context, fit-weight, generic-scaffold and source-deletion checks | post-hoc sensitivity |
| Fixed-bin applicability profile and endpoint-selection audit | post-hoc diagnostic |
| Repeated conditional label randomization | post-hoc repeated negative control |

No post-hoc analysis may be relabelled confirmatory. HistGradientBoosting and
molecular-graph checks are two frozen sensitivities, not a learner search or
independent external replication.

Training-source-deletion effects use the no-deletion model refit inside the
same script as their paired comparator. They must not be calculated by mixing
the new deletion fits with the older archived OOD prediction table.

## Main statistical conclusions supported

1. The full ExtraTrees model has moderate retrospective internal
   scaffold-disjoint ranking signal: Spearman 0.621
   (95% scaffold-bootstrap interval 0.541–0.689) and RMSE 0.792
   (0.721–0.859).
2. In matched internal ExtraTrees comparisons, context adds a smaller but
   positive increment over chemistry-only: delta Spearman 0.055
   (0.023–0.095); chemistry adds a larger increment over context-only:
   delta Spearman 0.193 (0.126–0.270).
3. Under frozen held domains, chemistry-only has source macro Spearman 0.237
   and target macro Spearman 0.118. Adding context worsens the matched
   Spearman contrast by −0.086 for source and −0.066 for target.
4. Strict scaffold removal does not restore a context benefit and attenuates
   the target-axis absolute chemistry-only signal.
5. A HistGradientBoosting sensitivity reproduces the qualitative
   internal-versus-transfer pattern: small positive internal context
   increment, no positive held-domain increment.
6. The molecular-graph sensitivity independently reproduces and accentuates
   the same directional boundary: delta Spearman is 0.104
   (0.041–0.168) internally, −0.224 (−0.357 to −0.091) for source OOD and
   −0.119 (−0.222 to −0.002) for target OOD. This is not evidence that the
   graph learner is superior.

These statements should report exact intervals from the Source Data tables
where space permits and must retain the retrospective/fixed-domain boundary.

## Multiplicity and interpretation

- The manuscript emphasizes prespecified contrasts, effect sizes and
  confidence intervals rather than dichotomizing results by p values.
- Per-domain estimates, calibration analyses, LODO influence diagnostics,
  strict canonical aggregation and learner-family sensitivities are not
  multiplicity-controlled. They are descriptive and must not be presented as
  multiple independent discoveries.
- The historical single label-shuffle model remains one diagnostic
  realization. Separately, the post-hoc extension uses 100 independently
  seeded within-source–target training-label permutations. Its empirical
  one-sided tail fractions use the finite-permutation correction, have a
  minimum attainable value of 1/101, and are not multiplicity-adjusted.
- Undefined Spearman correlations from constant predictions remain missing;
  they must not be replaced with zero.
- OOD macro estimates contain only 4 or 8 domains. LODO changes can diagnose
  influence but do not estimate population-level domain uncertainty.

## Numerical reconstruction boundary

A diagnostic reconstruction of the VAV1 no-deletion target-OOD condition
matched the archived train/test order, four inner splits, model seeds,
selected parameters and feature dimensions, but did not reproduce the
historical prediction CSV bit for bit. Maximum absolute prediction
differences were 0.011870 for chemistry-only and 0.018532 pDC50 units for the
full model; aggregate metric differences were small. The discrepancy remains
unresolved and is not attributed to thread count.

This does not enter the source-deletion estimand because every deletion
condition and its comparator are generated within one script/environment and
use identical test rows and paired resamples. It does remain a documented
boundary for any claim of full historical byte-level reconstruction. See
`docs/post_hoc_scaffold_source_baseline_reconciliation_v1.md`.

## Calibration and use boundary

The OOD calibration results do not support absolute-DC50 deployment. Slopes
are non-ideal and target-axis domain-macro R² is negative for both chemistry-
only and full models. A positive rank correlation must not be translated into
claims of accurate concentration prediction, hit thresholding or assay
replacement.

Any decision-use statement requires a separate prespecified utility threshold,
prospective external data and calibration assessed on observations not used
for adaptation.

## Missing-data, repeats and provenance

- The row-level missingness table and modelling-context sanitization audit must
  be included as Source Data/Supplementary Data.
- Repeated compound/context observations are measurements, not independent
  compounds. Primary cluster resampling addresses shared scaffold dependence
  but does not model all source, target and assay correlation simultaneously.
- Median compound aggregation is a sensitivity analysis rather than the
  primary estimand.
- Source labels and sanitized modelling fields must be distinguished from raw
  input text so that obvious non-protein/non-cell tokens are not counted as
  biological categories.

## Reporting checklist

- Report exact `n` rows, compounds, scaffolds and fixed domains for every
  regime.
- Define pDC50 with units.
- State the split/deletion rule before each performance result.
- State model/feature block and whether hyperparameters were selected inside
  training data only.
- Report point estimate, 95% interval, resampling cluster and number of
  bootstrap draws.
- Report effect direction for every contrast.
- For source-deletion analyses, state that `none` is the within-script paired
  baseline and avoid causal database-quality interpretations.
- Identify exploratory/post-hoc results in text, figures and legends.
- Report negative, undefined and heterogeneous domain results; do not select
  only favourable domains.
- State that no formal power calculation was used for the retrospective data,
  whose available sample size was fixed.
- For prospective Stage 4, justify biological replication and report the
  number of independent compounds rather than wells.

## Gate

**GO for a retrospective benchmark/evaluation manuscript**, conditional on
successful clean reproduction, complete source-data packaging and explicit
claim boundaries.

**NO-GO for external predictive utility, calibrated absolute DC50 prediction,
candidate-discovery efficacy or mechanism** until the blinded prospective
protocol is executed.
