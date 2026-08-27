# GitHub v2 release gates

## Closed — row-level redistribution dependency

The publication route does not redistribute any of the four raw exports or
row-level derivatives. Deterministic reconstruction starts from owner-hosted
official downloads, so row-level redistribution is not an open engineering
gate. Source access conditions remain a journal/editorial boundary that is
described transparently.

## Closed — raw-to-analysis reconstruction

`data_builder/` downloads from the four official routes, validates nine
required file identities (including the MolGlueDB archive and extracted CSV),
parses 3,117 candidates, retains 1,758 strict observations, aggregates 1,560 QC
rows, makes the frozen 1,252/308 split and reproduces all six output hashes.

## In progress — complete clean CPU refit

The v2 orchestrator and independent exact-output verifier are implemented. The
clean run reproduced the data and formal confirmatory core, then stopped at
the matched-context stage because the historical extension binds a parent
runtime-manifest byte hash. Portable-v2 must replace that operational binding
with the already declared data, configuration and artifact identities, then
resume every remaining CPU stage. This gate is computational, not legal, and
uses no GPU.

## Remaining decisions before immutable journal release

1. select a licence for author-owned aggregate tables and figures, while
   expressly excluding third-party database rows;
2. complete and accept the clean portable-v2 downstream CPU refit;
3. create and freeze release tag `v2.0.0` (or another author-selected tag);
4. archive that exact tag in a DOI-granting repository and insert the DOI; and
5. run the public-boundary check and regenerate `MANIFEST.tsv` and
   `SHA256SUMS` after the final metadata edits.

The author-owned code licence is MIT and the repository URL is
`https://github.com/NTU-MedAILab/molglue-dc50-benchmark`.

## Separate manuscript figure status

The author-redrawn study-overview figure has replaced the earlier layout
draft. Final PDF-size inspection remains a manuscript-production check and is
unrelated to CPU reproducibility.
