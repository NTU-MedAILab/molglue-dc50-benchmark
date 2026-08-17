# Main figure legends v1

**Fig. 1 | Data structure and validation gradient.** a, The three evaluation
regimes progressively separate chemical and domain context; familiar context
can recur internally but is unavailable or shifted in a held domain. b, Rows, unique
canonical compounds and Bemis–Murcko scaffolds across standardized source
labels. c, Sample-size imbalance across modeling target categories; dark bars
denote the eight exact-target domains entering the frozen target-OOD analysis.
d, Missingness of prespecified provenance/context fields and the number of
repeated compound and exact-context groups. Percentages refer to the 1,560-row
frozen QC table. Source data are provided as a Source Data file.

**Fig. 2 | Internal scaffold-disjoint prediction and matched feature-block
evidence.** a,b, Repeat-averaged internal Spearman correlation and RMSE for
chemistry-only and chemistry-plus-context ExtraTrees under scaffold- and
compound-disjoint validation. c,d, Paired scaffold-cluster contrasts for
ranking and error; filled markers denote confirmatory comparisons, grey open
markers the formal matched context-only extension, teal squares the post-hoc
HistGradientBoosting sensitivity, and purple diamonds the post-hoc molecular-
graph sensitivity. Points show estimates
and lines show 95% percentile intervals from 10,000 paired scaffold-cluster
bootstrap replicates (667 global scaffolds). Folds, repeats and bootstrap
replicates are not independent samples. Positive contrasts favour the first
named model. Source data are provided as a Source Data file.

**Fig. 3 | Context gain does not persist under held-domain transfer.**
a,c, Equal-domain-weight Spearman correlation and RMSE for chemistry-only and
full ExtraTrees in four held-source and eight held-target domains. b,d, Paired
full-minus-chemistry contrasts under the frozen domain-plus-compound-cold
protocols, strict post-hoc scaffold purging and the frozen post-hoc
HistGradientBoosting and molecular-graph sensitivities. Positive contrasts
favour the full model.
Points show estimates and lines show 95% percentile intervals from 10,000
paired global-scaffold bootstrap replicates (667 source-axis and 601
target-axis scaffolds). Source data are provided as a Source Data file.

**Fig. 4 | Domain heterogeneity and calibration define the supported claim
boundary.** a, Within-domain Spearman correlations for chemistry-only and full
ExtraTrees across all frozen source and target domains. b, Change in the
full-minus-chemistry domain-macro Spearman contrast after omitting one fixed
domain; this is descriptive and has no population-level confidence interval.
c, Pooled R² with 95% scaffold-bootstrap intervals for frozen OOD predictions.
d, Claims supported and not supported by the current retrospective evidence.
Domain-level and calibration panels are post-hoc exploratory analyses; no
per-domain multiplicity-adjusted significance claim is made. Source data are
provided as a Source Data file.

**Fig. 5 | Post-hoc robustness map of the matched context contrast.**
a, Held-source and held-target domain-macro Spearman contrasts for the
original full model and a portable full model that removes held-axis
categories. b, Internal full-minus-chemistry Spearman contrasts under
uniform-row, compound-equal and source–target-balanced fitting; weighting
changes model fitting while evaluation remains unweighted. c, Internal
chemistry-only and full-model Spearman estimates under original
Bemis–Murcko and generic Murcko scaffold grouping, with their paired
full-minus-chemistry contrasts. d, Target-OOD full-minus-chemistry contrasts
without deletion and after deleting each training source. Points are observed
estimates and lines are 95% percentile intervals from 10,000 paired
global-scaffold bootstrap resamples. Positive contrasts favour the
context-inclusive model. All panels are post-hoc sensitivities and source
deletions do not identify causal database effects. Source data are provided
as a Source Data file.

**Supplementary Fig. S1 | Chemical-novelty applicability profile.**
Full-minus-chemistry Spearman and RMSE contrasts are shown in four
prespecified maximum-train-Tanimoto bins for a, internal scaffold-disjoint
validation; b, held-source OOD; c, held-target OOD; d, strict held-source plus
scaffold OOD; and e, strict held-target plus scaffold OOD. Positive values
favour the full model. `n` denotes rows, `S` global scaffolds and `D`
prespecified held domains. Lines are paired 95% global-scaffold-bootstrap
intervals from 10,000 requested resamples. OOD intervals freeze the
represented and metric-specific eligible domains before resampling; a repeat
is valid only when every eligible domain is represented with a finite metric.
`CI NE` denotes too few complete fixed-domain repeats, and `CI NE*` denotes a
structurally non-estimable interval caused by an eligible domain containing
one scaffold; in both cases the descriptive point is retained. `point NE`
would denote a non-estimable descriptive point. Bins are unconnected, none
was omitted or merged, and no monotonic-trend test was performed. Source data
are provided as a Source Data file.

**Supplementary Fig. S2 | Repeated conditional label-randomization
controls.** a,b, Conditional null distributions for chemistry-only and
full-model Spearman. c,d, Conditional null distributions for the paired
full-minus-chemistry Spearman increment and RMSE reduction. Labels were
randomized within source × target strata. All 100 outcomes are displayed;
shading denotes their empirical 2.5th–97.5th percentile range and the dark
line the observed value. Printed one-sided empirical probabilities use
`(1 + number at least as extreme)/(100 + 1)`, with a finite floor of
`1/101`. The shaded range is not a parametric confidence interval and the
controls are not multiplicity-adjusted confirmatory tests. Source data are
provided as a Source Data file.

**Supplementary Fig. S3 | Endpoint-selection and censoring boundary.**
a, Relation composition across all 3,117 parsed-record table rows. b,
Within-source relation composition. c, Exact-label fraction versus
parsed-record row count for all target categories; labels identify only the
declared largest and lowest-exact-fraction categories. d, Fractions of unique
compounds and scaffolds represented in the strict-exact modelling core for
each relation class. Exact, interval/range, one-sided and missing/unparsed
relations are redundantly encoded by colour and hatch or marker shape.
Counts refer to record-table rows rather than unique record identifiers.
Censored observations were not converted to point labels for fitting, and
the selection stages are not a row-wise inclusion probability. Source data
are provided as a Source Data file.
