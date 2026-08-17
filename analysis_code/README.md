# Analysis-code snapshot

These files are byte-identical snapshots of the author-created study scripts.
They are included for inspection and scientific provenance.

The snapshot includes the frozen post-hoc computational extension: portable
held-axis context, fit-weight sensitivity, generic-scaffold validation,
training-source deletion, applicability profiling, conditional
label-randomization, endpoint-censoring audit, aggregate-source construction,
independent cross-checking and figure generation. The permutation shard and
merge utilities alter execution order only; their exact-match gate and
execution log are included for audit.

The snapshot also includes the independently frozen, post-hoc molecular-graph
runner and its read-only QA implementation. That sensitivity uses a
CUDA-capable GPU for its formal execution identity and remains separate from
the mixed-lineage ExtraTrees extension.

They are not the release entry point. Several downstream scripts intentionally
retain historical fail-closed checks against parent result manifests. GitHub
v2 supplies the exact protocol and builder needed by those scripts and runs
them only through the checkpointed clean-workspace orchestrator.

Use `../scripts/reproduce_release_v2.py` for public-boundary verification and
the complete CPU chain. Do not interpret a manually run subset as a
reproduction of the complete manuscript analysis.

The v2 acceptance gate preserves the scientific arguments and identity digests
recorded in `../SCIENTIFIC_IDENTITY.json` and compares regenerated scientific
CSV artifacts with independent frozen SHA-256 ledgers. Runtime manifests are
not used as portable scientific identity because they contain paths and times.
