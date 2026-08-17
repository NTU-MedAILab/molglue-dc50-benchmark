# Context-only ExtraTrees matched-ablation protocol v1

Status: frozen CPU extension protocol  
Protocol version: `context_et_matched_ablation_v1`  
Parent protocol: `confirmatory_cpu_v1.1`  
Analysis role: prespecified follow-up to resolve model-family confounding

## Scientific question

Does adding molecular-structure features improve cross-fitted prediction when
the context-only comparator uses the same nonlinear ExtraTrees learner,
hyperparameter grid, nested model-selection rule and outer data partitions as
the frozen full chemistry-plus-context model?

The primary contrast is:

`full_context_extra_trees - context_extra_trees`

Positive `delta_spearman = full - context` and positive
`delta_rmse = context - full` favour the full chemistry-plus-context model.
This extension tests incremental predictive information; it does not establish
mechanism or prospective external validity.

## Frozen upstream identities

The extension reads, but must not modify, the completed parent result:

- Parent result directory: `reports/confirmatory_cpu_v1`
- Parent manifest SHA256:
  `3ce6e2e536e73f4ce2e8c79206e6b8b0602886dc08be65e46ee9625fcc3a3e46`
- Parent raw OOF predictions SHA256:
  `c67fc7176195eade86b45930d2f766a7aa76942efa9e4f8ce11ed94703d80bd9`
- Parent repeat-averaged predictions SHA256:
  `183d6fa0047bdf47f1bb5f89679f5a3602344b09e4b0744792ea70ecce6f5e3b`
- Parent scaffold outer-fold metrics SHA256:
  `cd22ea3af654180171876ba86d90c7ebb1a8cbc5c8f8bf5fdeaad8fcef9016d7`
- Parent compound outer-fold metrics SHA256:
  `e56ab5b5b6424b80c03e712cf159ec5a4375686f900eddc692584390d2696965`
- Frozen input-data SHA256:
  `e9494246bcbd198bc09d5b0d09721910c33f70f46ae859b76377a13775e3e4b2`
- Parent analysis-script SHA256:
  `ac1a57203b43d74023521b76cfaef28b958a803dbc6d9fb2866e7550f636995c`
- Parent protocol-document SHA256:
  `7429b9a31f66b0ee7b1f75d977a1178615b8778bb3a507e06aa7f320eb11780b`

Execution fails closed if these hashes, the parent manifest status, package
versions or parent scientific configuration do not match.

## Data and splits

- The same 1,560 sanitized modeling rows and endpoint are loaded through the
  parent analysis module.
- Primary protocol: five-times-repeated five-fold scaffold-disjoint nested
  cross-validation.
- Secondary protocol: five-times-repeated five-fold
  canonical-SMILES-disjoint nested cross-validation.
- The extension regenerates all parent outer assignments with the same split
  function and seed (`260531`) and requires exact equality, row by row, with
  the saved full-model OOF fold assignments before fitting.
- Inner selection uses four group-disjoint folds within each outer-training
  set and exactly the parent seed formula.
- Response values are not used to construct outer or inner folds.

## Context-only features

Only the parent context block is used:

- main effects: source database, recruiting protein, target protein, cell
  line, assay method, activity time and mode of action;
- interactions: source-target, recruiter-target, target-cell and
  assay-target.

The one-hot encoder is fitted independently in every inner- and outer-training
fold with the parent settings: `handle_unknown="ignore"`,
`min_frequency=2`, dense `float32`. No Morgan fingerprints, RDKit
descriptors, target encoding or protein-language-model features enter this
ablation.

## Learner and nested selection

The context-only comparator uses `ExtraTreesRegressor` with:

- `n_estimators=600`;
- `min_samples_leaf={1,5,10}`;
- `max_features={"sqrt",0.3}`;
- `max_depth=None`;
- `bootstrap=False`;
- uniform sample weights;
- `n_jobs=10`.

Each model configuration receives pooled inner-OOF predictions. Inner pooled
Spearman is the selection metric. Configurations within `0.005` of the best
Spearman are eligible; the more regularized configuration is selected, with
larger leaf size preferred and then `"sqrt"` preferred over `0.3`. Remaining
ties use lower RMSE and parameter ID. The comparator selects its own
hyperparameters; it does not reuse the full model's selected configuration.

For stochastic matching, every inner candidate uses the same random-state
formula as the corresponding parent ExtraTrees candidate. The selected
outer-fold context-only fit uses the saved full-model seed position, so the
feature block—not a deliberately different random seed—is the intended
difference.

## Outcomes and uncertainty

The five context-only OOF predictions are averaged row-wise. The saved,
hash-verified full-model repeat averages are joined one-to-one by protocol and
row index. The primary outcomes and calibration definitions are inherited
from the parent protocol.

The paired uncertainty analysis uses 10,000 scaffold-cluster percentile
bootstrap replicates for both validation protocols:

- cluster unit: the 667 parent Bemis-Murcko scaffold groups;
- scaffold RNG seed: `260531`;
- compound RNG seed: `260532`;
- the same sampled clusters are used for both models within every replicate;
- `delta_spearman = full - context`;
- `delta_rmse = context - full`.

A confidence interval wholly above zero supports incremental performance of
the full chemistry-plus-context model over a matched nonlinear context-only
learner. An interval crossing zero requires retaining the narrower parent
claim that the full model outperformed context Ridge.

## Completeness and provenance

The formal run must assert:

1. exact parent and extension hashes and runtime versions;
2. exact outer-fold identity with the saved full OOF assignments;
3. no outer- or inner-group overlap;
4. six tuning records and one selected configuration per outer split;
5. 15,600 unique raw context-only predictions;
6. five OOF predictions for every protocol-row pair;
7. finite predictions and primary metrics;
8. exact one-to-one alignment with the saved full repeat averages;
9. two paired-bootstrap contrast records, each with 667 clusters and 10,000
   replicates;
10. hashes and row counts for every final extension artifact.

Folds and repeats are not independent replicates. The bootstrap is applied to
the row-wise mean of the five cross-fitted predictions and is conditional on
the fitted cross-validation ensemble.

## Execution

Smoke test (first formal scaffold outer split only; 40 trees, two inner folds,
no bootstrap):

```bash
env/bin/python \
  experiments/260531_molglue_dc50_route/scripts/run_context_et_matched_ablation_cpu_v1.py \
  --smoke \
  --overwrite \
  --output-dir experiments/260531_molglue_dc50_route/reports/context_et_matched_ablation_cpu_v1_smoke
```

Frozen formal run:

```bash
env/bin/python \
  experiments/260531_molglue_dc50_route/scripts/run_context_et_matched_ablation_cpu_v1.py \
  --n-jobs 10 \
  --output-dir experiments/260531_molglue_dc50_route/reports/context_et_matched_ablation_cpu_v1
```

Resume an interrupted formal run with the identical command plus `--resume`.
Resume fails closed if any scientific identity differs.

This extension is CPU-only. It does not require or use a GPU.
