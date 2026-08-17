# Null/applicability/censoring v3 correction verification

**Verification date:** 2026-07-30 (Asia/Shanghai)  
**Status:** PASS for protocol, implementation, smoke, fixed-domain regression,
sparse-domain fail-safe, fail-closed migration and locked permutation-ID-1
identity  
**Formal 100-permutation run:** NOT STARTED  
**Compute used:** CPU only; no GPU

## Frozen implementation identities

| Artifact | SHA-256 |
|---|---|
| `run_post_hoc_null_applicability_censoring_v3.py` | `41b99a803b0e090d1bd55950983441eee0ce8edc8b099f86f56387e2f29f3650` |
| `run_post_hoc_permutation_shard_v3.py` | `5aa68bfdc69083b5f30fa1eaedb938f2b3af4ceb86d774c294d672373319637e` |
| `merge_post_hoc_permutation_shards_v3.py` | `95a5dbc4927e53f0a22eba824076026b371261e4f9a8fdfc4afb577f1ff9be43` |
| `qa_post_hoc_applicability_fixed_domain_v3.py` | `7e51d44a9f0b352475b564047ad99c14e8c81583b4f0ed963abbbde6ab839e83` |
| `post_hoc_null_applicability_censoring_protocol_v3.md` | `84178cde0edfc0ff31be3d7e12135351f95a2578b9ac69b61d9a5d517caa0dc5` |
| `post_hoc_applicability_fixed_domain_correction_v3.md` | `9c2d8f90fa546480a37b6e7546cbd9e9a740916325e5dec973f9fa7f8ef899ce` |
| float32 response correction v2 | `75a8b0e4f894cead9d72039eab9e6162a89d2a20c79afceda8a2aae1ffd78f3f` |
| frozen confirmatory implementation | `ac1a57203b43d74023521b76cfaef28b958a803dbc6d9fb2866e7550f636995c` |
| computational-extension master protocol v1 | `7dee28f45e3ddf8d6299622a8e494c50492b3b2614f39ce8b2fdd2a1be754ff8` |
| frozen 1,560-row modeling core | `e9494246bcbd198bc09d5b0d09721910c33f70f46ae859b76377a13775e3e4b2` |
| public-source parsed-record table | `ae1a16613b41ce84f92f539dbd31f16ada272ac5e401250acc7e97fd5e8bd970` |

The merge helper hard-locks the parent and wrapper hashes above. Parent and
wrapper manifests record both correction identities, the local/master
protocol identities and the float32 response invariant.

## Statistical correction verified

For each OOD `regime × protocol × bin × metric`, v3 freezes a sorted
metric-specific eligible-domain identity before resampling. Global scaffold
clusters remain the bootstrap unit, so a sampled scaffold contributes all rows
across every domain in which it occurs.

A macro replicate is valid only when every fixed eligible domain is present
and yields a finite paired domain contrast. Missing-domain and
metric-nonestimable failures are mutually exclusive and are counted
separately. Intervals require at least
`max(100, ceil(0.50 × B))` complete replicates.

If any fixed eligible domain contains fewer than two unique global scaffolds,
the descriptive macro point estimate remains visible but all requested macro
replicates are recorded as structurally not attempted and the interval is
`NA`. No domain-stratified fallback is used.

The output package adds:

- `applicability_macro_domain_eligibility.csv`;
- `applicability_macro_bootstrap_audit.csv`;
- stable represented/eligible/sparse domain JSON sequences and SHA-256
  identities in the paired-delta and audit tables; and
- per-metric valid, missing-domain-invalid,
  metric-nonestimable-invalid and structural-not-attempted counts.

## Verification results

1. `py_compile`: PASS for parent, shard wrapper, merge helper and regression-QA
   script.
2. Development smoke: PASS, 25/25 QA checks. It used one permutation, one
   complete five-fold repeat, 20 trees, four threads and 100 bootstrap
   replicates. It is non-inferential and was written only under `/tmp`.
3. Float32 guard: PASS. A direct float64 call was rejected before fitting.
4. Synthetic fixed-domain regression: PASS. With two eligible domains and
   1,000 deterministic global-scaffold replicates, an independent calculation
   found 135 replicates missing at least one fixed domain. V3 recorded exactly
   135 missing-domain invalid replicates and used only the remaining 865 for
   the RMSE macro interval. The former changing-domain implementation would
   have averaged the observed subset in those 135 replicates.
5. Synthetic sparse-domain regression: PASS. A finite descriptive point
   estimate with one fixed eligible domain represented by one scaffold was
   retained, while all three macro intervals were refused and all requested
   replicates were structurally not attempted.
6. Real lowest-bin regression: PASS. In both frozen and strict target-OOD
   `[0.0, 0.4)` strata, the same eligible held-out domain contains four rows
   but only one unique global scaffold. V3 records that domain identity and
   reports Spearman, RMSE and MAE macro intervals as structurally
   non-estimable.
7. Observed point-estimate invariance: PASS. For all 19 interrupted v2
   applicability strata available for audit, pooled and domain-macro
   Spearman/RMSE/MAE paired point estimates matched v3 exactly
   (`atol = 0`, including matching `NA`).
8. Fixed-domain audit table: PASS, 48/48 OOD
   `stratum × metric` rows. Every reported macro reproduced from exactly the
   hashed eligible-domain set, every bootstrap count reconciled, and interval
   presence matched the frozen threshold and sparse-domain rule.
9. A non-inferential 1,000-replicate stability audit produced 39 estimated
   intervals, six structural `NA` intervals and three additional Spearman
   `NA` intervals below the 50% complete-replicate threshold. Every
   non-structural RMSE/MAE stratum exceeded the threshold. This pilot did not
   change the protocol; the formal 10,000-replicate run determines final
   interval status.
10. Fail-closed migration: PASS. V3 parent, wrapper and merge helper rejected
    marked v2 directories. The v2 parent also rejected its newly marked formal
    directory. V3 resume accepted an identical completed smoke configuration
    and rejected a change in permutation count without an eligible formal v3
    shard-merge manifest.
11. Locked parent-v3 permutation ID 1: PASS with 600 trees, five repeats, four
    fitting threads and seed `260531`.
12. Locked shard-v3 permutation ID 1: PASS under the identical scientific
    configuration; shard QA reported two model rows, one contrast row, 25 fold
    diagnostics, finite metrics and preserved within-stratum label multisets.
13. Parent/shard exact identity: PASS with pandas
    `assert_frame_equal(check_exact=True, check_dtype=True)` after stable key
    sorting. Raw CSV files and the feature manifest were byte-identical.
14. Merge-helper shard validation: PASS on the locked ID-1 shard, including
    parent/protocol/correction/configuration and feature-manifest identities.

### Locked ID-1 exact artifacts

| Artifact | Rows | Shared SHA-256 |
|---|---:|---|
| `permutation_model_metrics.csv` | 2 | `5fe53a1200ab6ebb6b5ddd2f1f2ff1253f6bb25f35dec9d9e1c5c9463c3b99c5` |
| `permutation_contrast_metrics.csv` | 1 | `929727402be846be7e6a1c158a4c39d82c67d60163a8979765b5241010b6ded6` |
| `permutation_fold_diagnostics.csv` | 25 | `ba4072ee27fedcb278a0c77f2919840bf8a327fd5acb299ccce6b8b6184b9da2` |
| `feature_manifest.json` | — | `6b393388b50bf42313943dc9df9d7cf654f3e1ed6983df0382ced9b1f06c4d6f` |

The verification outputs were
`/tmp/molglue_null_id1_parent_mLNOAF_v3` and
`/tmp/molglue_null_id1_shard_7xLywq_v3`. They are not publication evidence.

## Superseded v2 formal directories

The interrupted v2 primary and three shards contain
`SUPERSEDED_DO_NOT_USE.md`:

| Directory | Frozen incomplete state |
|---|---|
| `reports/post_hoc_null_applicability_censoring_v2` | 19/20 applicability strata; no completed primary permutation ID |
| `reports/post_hoc_null_permutation_shard_026_050_v2` | IDs 26–29 only |
| `reports/post_hoc_null_permutation_shard_051_075_v2` | IDs 51–54 only |
| `reports/post_hoc_null_permutation_shard_076_100_v2` | IDs 76–79 only |

The primary marker SHA-256 is
`28d6d517a4c319e1d88c55445b06886df4020f11b238aa250d09b81309c2d9fb`;
the identical shard marker SHA-256 is
`bb7afb0c83127635f44f99ad86f3cbd25f69cd7a9c5efc2e100068633592e7c7`.
The directories remain as audit trails and may not be resumed, merged,
released or used as publication evidence.

## Authorized formal deterministic-shard route

Run from the repository root. The parent first computes IDs 1–25:

```bash
MPLCONFIGDIR=/tmp/mplconfig env/bin/python \
  experiments/260531_molglue_dc50_route/scripts/run_post_hoc_null_applicability_censoring_v3.py \
  --output-dir experiments/260531_molglue_dc50_route/reports/post_hoc_null_applicability_censoring_v3 \
  --n-permutations 25 --n-estimators 600 --n-jobs 4 \
  --bootstrap-replicates 10000 --max-repeats 5 --seed 260531
```

Three non-overlapping workers compute the remaining IDs:

```bash
MPLCONFIGDIR=/tmp/mplconfig env/bin/python \
  experiments/260531_molglue_dc50_route/scripts/run_post_hoc_permutation_shard_v3.py \
  --start-id 26 --end-id 50 \
  --output-dir experiments/260531_molglue_dc50_route/reports/post_hoc_null_permutation_shard_026_050_v3 \
  --n-estimators 600 --n-jobs 4 --max-repeats 5 --seed 260531

MPLCONFIGDIR=/tmp/mplconfig env/bin/python \
  experiments/260531_molglue_dc50_route/scripts/run_post_hoc_permutation_shard_v3.py \
  --start-id 51 --end-id 75 \
  --output-dir experiments/260531_molglue_dc50_route/reports/post_hoc_null_permutation_shard_051_075_v3 \
  --n-estimators 600 --n-jobs 4 --max-repeats 5 --seed 260531

MPLCONFIGDIR=/tmp/mplconfig env/bin/python \
  experiments/260531_molglue_dc50_route/scripts/run_post_hoc_permutation_shard_v3.py \
  --start-id 76 --end-id 100 \
  --output-dir experiments/260531_molglue_dc50_route/reports/post_hoc_null_permutation_shard_076_100_v3 \
  --n-estimators 600 --n-jobs 4 --max-repeats 5 --seed 260531
```

Only after every worker has stopped and all shard QA files are PASS:

```bash
env/bin/python \
  experiments/260531_molglue_dc50_route/scripts/merge_post_hoc_permutation_shards_v3.py \
  --shard-dir experiments/260531_molglue_dc50_route/reports/post_hoc_null_permutation_shard_026_050_v3 \
  --shard-dir experiments/260531_molglue_dc50_route/reports/post_hoc_null_permutation_shard_051_075_v3 \
  --shard-dir experiments/260531_molglue_dc50_route/reports/post_hoc_null_permutation_shard_076_100_v3 \
  --n-permutations 100

MPLCONFIGDIR=/tmp/mplconfig env/bin/python \
  experiments/260531_molglue_dc50_route/scripts/run_post_hoc_null_applicability_censoring_v3.py \
  --output-dir experiments/260531_molglue_dc50_route/reports/post_hoc_null_applicability_censoring_v3 \
  --n-permutations 100 --n-estimators 600 --n-jobs 4 \
  --bootstrap-replicates 10000 --max-repeats 5 --seed 260531 --resume
```

The merge creates
`reports/post_hoc_null_applicability_censoring_v3/premerge_primary_snapshot_v3`.
Do not launch a single-process formal run against the same primary directory
while the shard route is active.
