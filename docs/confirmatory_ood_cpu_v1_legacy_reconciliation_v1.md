# Legacy confirmatory OOD reconciliation

Reconciliation identity: `confirmatory_ood_cpu_v1_legacy_reconciliation.v1`.

## Decision

The current `reports/confirmatory_ood_cpu_v1` directory and the clean
publication-validation reproduction preserve the same scientific result.
Nine primary artifacts are byte-identical, including
`ood_predictions.csv` and `ood_split_audit.csv`. The scientific
configuration is also exact; its canonical SHA-256 is
`fd10b444eb205af2a150e81b19b281301d0bc0a6c28cf8a30892297ea8a0ceab`.

Three derived summary tables were reserialized:

- `ood_aggregate_metrics.csv`: maximum absolute numeric difference
  `2.1316282072803006e-14`;
- `ood_domain_scaffold_bootstrap.csv`: maximum absolute numeric difference
  `6.661338147750939e-16`;
- `ood_paired_scaffold_bootstrap.csv`: maximum absolute numeric difference
  `2.220446049250313e-16`.

Their schemas, shapes, missing-value locations, and non-numeric fields are
exact. Every numeric comparison passes the predeclared absolute tolerance
`1e-12` with relative tolerance zero.

## Historical hash chain

The strict frozen identity and the original
`reports/confirmatory_ood_cpu_v1/artifact_sha256.csv` independently retain
the historical parent-manifest SHA-256
`1692bd2bcf729a8d099875bc1d640f351e30691afa480f54a42e8a99cadf4b65`.
The historical manifest bytes are no longer present in either compared
directory, so that digest is provenance evidence rather than an executable
whole-file identity.

The current operational manifest SHA-256 is
`34c56be73c362f85a662d5f411e0140fe8c6f27d242d8563542c0460d3009c17`;
the reproduction manifest SHA-256 is
`121a61a6f04f2d1c43908f7fcbe328c1f85c99d9396e0cce5c33d33be973f221`.
Their only differences are `elapsed_seconds`, `resume_count`,
`run_session_started_unix`, and the hash/size inventory entries for the
three tolerance-qualified derived tables. No scientific-configuration or
primary-artifact field differs.

## Cause and boundary

A completed-result resume rewrote operational timing/session fields and
reserialized three derived summaries. It did not change predictions, split
assignments, tuning results, selected hyperparameters, domain metrics, or
the scientific configuration. Any future difference outside the exact
allowlists above fails closed.

This reconciliation used existing authorized artifacts only; restricted collaborator material: **not used**. It does not relax the zero-exception
artifact inventories required for the new computational-extension
workflows.
