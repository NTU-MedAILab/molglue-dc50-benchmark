# Frozen post-hoc molecular-graph model-family sensitivity protocol v1

Freeze date: 2026-07-31 (Asia/Shanghai)  
Analysis identity: `post_hoc_graph_model_family_sensitivity_v1`  
Compute: one NVIDIA GPU; formal host alias `3080`  
Endpoint: `pDC50 = 9 - log10(DC50_nM)`

## Question

Does the qualitative full-context-versus-chemistry-only comparison persist when
the molecular representation and learner are replaced by an independently
specified molecular graph neural network?

This is a post-hoc model-family sensitivity analysis. It is neither a
confirmatory analysis nor a model search and cannot change the identity of the
previously frozen results.

## Frozen data and evaluation splits

- Use only the frozen 1,560-row publication-eligible QC table after the same
  prespecified context sanitization.
- Do not use any restricted, collaborator-owned, prospective, external, or
  otherwise publication-ineligible records.
- Internal evaluation uses the primary five-times-repeated five-fold
  scaffold-disjoint outer splits.
- Source and target OOD use the frozen 4 and 8
  domain-plus-compound-cold splits, respectively.
- Strict OOD keeps the same test domains and rows and removes every training
  row whose Bemis--Murcko scaffold occurs in the corresponding test domain.
- Every outer training set receives one deterministic five-way
  scaffold-disjoint split; fold 1 is used only for early stopping and is never
  used to fit model parameters.
- The runner must verify zero train/test row and compound overlap, held-domain
  exclusion, and zero train/test scaffold overlap for internal and strict OOD
  evaluation.

## Frozen matched models

- `chemistry_graph_mpn`: molecular graph only.
- `full_context_graph_mpn`: the identical graph network plus the frozen
  categorical context columns and interactions.
- The context one-hot encoder is fitted on the inner fitting subset only,
  uses `min_frequency=2`, and ignores unseen categories.
- No Morgan fingerprint, RDKit descriptor, protein-language-model embedding,
  pretrained molecular representation, calibration, routing, or test-time
  augmentation is used.

## Frozen molecular graph

Atoms use fixed one-hot features for:

- atomic number (H, B, C, N, O, F, Si, P, S, Cl, Se, Br, I, plus unknown);
- degree 0--5 plus unknown;
- total hydrogens 0--4 plus unknown;
- formal charge -2--2 plus unknown;
- hybridization (S, SP, SP2, SP3, SP3D, SP3D2, plus unknown);
- aromaticity, ring membership, and possible chirality.

Edges are undirected and separated into five fixed channels: single, double,
triple, aromatic, and other. No conformer or three-dimensional information is
used.

## Frozen learner and training

The graph encoder is a three-layer bond-channel-aware message-passing network:

- atom projection width 128;
- separate learned neighbor transform for each bond channel in every layer;
- degree-normalized aggregation, residual connection, layer normalization,
  GELU, and dropout 0.15;
- concatenated masked mean and max pooling followed by a 128-dimensional graph
  projection.

For the full model, the fold-fitted context vector is projected to 64
dimensions and concatenated with the graph vector. The regression head is
128 hidden units, then 64 hidden units, then one output, with GELU and dropout
0.15.

Training is fixed as follows:

- one deterministic seed per outer split, paired across the two models;
- AdamW, learning rate `5e-4`, weight decay `1e-4`;
- Smooth L1 loss with beta `0.5`;
- batch size 64;
- maximum 120 epochs;
- validation every epoch and patience 15;
- checkpoint selection by higher inner-validation Spearman, with lower
  validation RMSE only as a deterministic tie-break;
- target mean and standard deviation fitted on the inner fitting subset only;
- gradient norm clipped at 5;
- no hyperparameter grid, seed selection, ensembling, or refitting after
  early stopping.

The single fixed seed is intentional: the repeated outer design supplies five
out-of-fold predictions per internal row without treating stochastic restarts
as independent evidence.

## Estimands and uncertainty

Primary sensitivity contrasts:

- `Delta Spearman = full - chemistry-only`;
- `Delta RMSE = RMSE(chemistry-only) - RMSE(full)`.

Positive values favor the full model. Internal point estimates are computed
after averaging each row's five out-of-fold predictions. OOD point estimates
use equal-domain-weight domain-macro Spearman and RMSE; pooled metrics are
secondary.

- Internal: 10,000 paired global-scaffold cluster-bootstrap replicates.
- Each source/target OOD regime, including strict variants: 10,000 paired
  global-scaffold cluster-bootstrap replicates.
- Report percentile 95% intervals and no p-values.
- Folds, repeats, domains, seeds, epochs, and bootstrap replicates are not
  additional independent observations.

## Frozen interpretation

- The analysis asks whether the direction of the matched context contrast is
  robust to an independent graph learner; it is not a leaderboard comparison.
- A positive internal contrast with absent or non-positive OOD contrast is
  qualitative support for the previously observed evaluation pattern.
- Any other result narrows, rather than invalidates, the prior conclusion and
  must be reported without adding or tuning another graph model.
- No absolute unseen-target generalization, mechanism, causal effect,
  prospective validation, candidate discovery, or deployment claim is
  permitted.

## Stop conditions

- Stop on any data hash mismatch, invalid molecule, non-finite prediction,
  split-count mismatch, train/test leakage, inner-fit/validation scaffold
  overlap, or result identity mismatch.
- Smoke runs may reduce splits, epochs, patience, and bootstrap replicates only
  to test code paths; they are not scientific evidence.
- Formal architecture, features, loss, optimizer, batch size, early-stopping
  rule, splits, metrics, and bootstrap seeds may not change after the first
  formal execution.
