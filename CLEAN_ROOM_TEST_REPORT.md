# Clean-room release verification report

Date: 2026-07-31  
Scope: portable deposited code, aggregate evidence and figure artifacts  
GPU: not used or required

## Procedure

1. Generated `MANIFEST.tsv` and `SHA256SUMS` from the release payload.
2. Copied the release directory to a newly created temporary directory.
3. Changed the working directory to the temporary directory outside the
   project tree.
4. Ran `sha256sum -c SHA256SUMS`.
5. Ran the release verifier with the system Python:
   `python3 release_jcheminf_v1/scripts/reproduce_release_v1.py verify --json`.
6. Independently preflighted the private frozen input in place.
7. Invoked `reconstruct` with the valid input and confirmed that it exited
   with the documented blocked code and created no output directory.
8. Created a new Python 3.11 virtual environment outside the project, installed
   every exact requirement from `requirements-lock.txt`, and ran `pip check`.
9. Copied the release into a second clean directory and reran the portable
   verifier with bytecode writing disabled.
10. Ran an eight-fit internal-learning-curve scientific smoke analysis in the
    fresh environment and executed its independent QA.

## Results

- SHA-256 ledger: **PASS**, every listed payload plus `MANIFEST.tsv`.
- Exact manifest/inventory: **PASS**.
- Restricted-row and private-path exclusion: **PASS**.
- Portable scientific identity: **PASS**.
- Aggregate evidence contract: **PASS**.
- Source-rights/checksum registries: **PASS**.
- Environment-record consistency: **PASS**.
- Four publication figure bundles: **PASS**.
- Frozen-input preflight: **PASS** for 664,847 bytes, 1,560 rows, 1,137
  canonical compounds and SHA-256
  `e9494246bcbd198bc09d5b0d09721910c33f70f46ae859b76377a13775e3e4b2`.
- Fail-closed reconstruction: **PASS**; exit code 3 and no output created.
- Fresh locked installation: **PASS**; exact requirements installed without
  substitution and `pip check` reported no broken requirements.
- Locked scientific-stack imports: **PASS** (NumPy 2.2.6, pandas 2.3.3,
  SciPy 1.15.3, scikit-learn 1.7.2, RDKit 2025.09.4 and matplotlib 3.10.8).
- Second clean-copy release verification: **PASS**; zero unexpected bytecode
  files.
- Scientific smoke execution: **PASS**; eight fits, 2,496 predictions and
  20/20 independent QA checks.

## Qualification

The clean-install and clean-directory execution check is complete. This is not
a full portable-v2 refit: row-level source redistribution rights and the
rights-dependent reconstruction chain remain blocked as stated in
`RELEASE_BLOCKERS.md`.
