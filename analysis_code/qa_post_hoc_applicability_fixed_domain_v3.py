#!/usr/bin/env python3
"""Regression tests for the v3 fixed-domain applicability bootstrap.

The tests use a synthetic two-domain frame plus the two public-scope target-OOD
lowest-similarity bins. They do not fit a model, access collaborator-restricted
material, or write a publication result.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_post_hoc_null_applicability_censoring_v3 as parent  # noqa: E402


def synthetic_frame(*, sparse_second_domain: bool = False) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    definitions = {
        "domain_A": [
            ("A_scaffold_1", 4.0, 4.3, 4.1),
            ("A_scaffold_1", 5.0, 5.2, 5.1),
            ("A_scaffold_2", 6.0, 5.4, 5.8),
        ],
        "domain_B": [
            ("B_scaffold_1", 5.0, 5.4, 5.1),
            ("B_scaffold_1", 6.0, 6.3, 6.1),
            (
                "B_scaffold_1" if sparse_second_domain else "B_scaffold_2",
                7.0,
                6.2,
                6.8,
            ),
        ],
    }
    row_index = 0
    for domain, local_rows in definitions.items():
        for scaffold, y_true, chemistry, full in local_rows:
            rows.append(
                {
                    "heldout_group": domain,
                    "row_index": row_index,
                    "y_true": y_true,
                    "prediction_chemistry": chemistry,
                    "prediction_full": full,
                    "scaffold_id": scaffold,
                    "canonical_smiles": f"synthetic_{row_index}",
                }
            )
            row_index += 1
    return pd.DataFrame(rows)


def independent_missing_domain_count(
    frame: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
    eligible_domains: set[str],
) -> int:
    clusters = frame["scaffold_id"].astype(str).to_numpy()
    domains = frame["heldout_group"].astype(str).to_numpy()
    unique_clusters = np.unique(clusters)
    cluster_domains = {
        cluster: set(domains[clusters == cluster])
        for cluster in unique_clusters
    }
    rng = np.random.default_rng(seed)
    missing = 0
    for _ in range(n_bootstrap):
        sampled = rng.choice(
            unique_clusters,
            size=len(unique_clusters),
            replace=True,
        )
        represented: set[str] = set()
        for cluster in sampled:
            represented.update(cluster_domains[str(cluster)])
        if not eligible_domains.issubset(represented):
            missing += 1
    return missing


def run_synthetic_complete_domain_test() -> dict[str, Any]:
    frame = synthetic_frame()
    eligibility = parent.build_macro_domain_eligibility(
        frame,
        include_domain_macro=True,
    )
    n_bootstrap = 1_000
    seed = 917_331
    output = parent.bootstrap_applicability_delta(
        frame,
        n_bootstrap,
        seed,
        True,
        eligibility,
    )
    expected_domains = {"domain_A", "domain_B"}
    expected_missing = independent_missing_domain_count(
        frame,
        n_bootstrap,
        seed,
        expected_domains,
    )
    key = "delta_domain_macro_rmse"
    valid = int(output[f"{key}_n_bootstrap_valid"])
    missing = int(
        output[f"{key}_n_bootstrap_invalid_missing_domain"]
    )
    nonestimable = int(
        output[f"{key}_n_bootstrap_invalid_metric_nonestimable"]
    )
    not_attempted = int(
        output[f"{key}_n_bootstrap_not_attempted_structural"]
    )
    checks = {
        "eligible_domains_frozen": (
            json.loads(output[f"{key}_eligible_domains_json"])
            == sorted(expected_domains)
        ),
        "missing_domains_are_invalid_not_dropped": (
            missing == expected_missing and missing > 0
        ),
        "complete_count_reconciliation": (
            valid + missing + nonestimable + not_attempted == n_bootstrap
        ),
        "rmse_replicates_otherwise_estimable": (
            valid == n_bootstrap - missing
            and nonestimable == 0
            and not_attempted == 0
        ),
        "valid_threshold_met": (
            valid >= int(output[f"{key}_minimum_valid_required"])
            and output[f"{key}_ci_status"] == "ESTIMATED"
            and math.isfinite(float(output[f"{key}_ci_low"]))
            and math.isfinite(float(output[f"{key}_ci_high"]))
        ),
    }

    omitted_b = frame[frame["heldout_group"] == "domain_A"]
    value, status = parent.fixed_domain_macro_metric(
        omitted_b["y_true"].to_numpy(dtype=np.float64),
        omitted_b["prediction_chemistry"].to_numpy(dtype=np.float64),
        omitted_b["prediction_full"].to_numpy(dtype=np.float64),
        omitted_b["heldout_group"].astype(str).to_numpy(),
        "rmse",
        ("domain_A", "domain_B"),
    )
    checks["direct_missing_domain_guard"] = (
        math.isnan(value) and status == "missing_fixed_eligible_domain"
    )
    return {
        "test": "synthetic_fixed_domain_missing_replicates",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "n_bootstrap": n_bootstrap,
        "expected_missing": expected_missing,
        "observed_missing": missing,
        "observed_valid": valid,
        "checks": checks,
    }


def run_synthetic_sparse_domain_test() -> dict[str, Any]:
    frame = synthetic_frame(sparse_second_domain=True)
    eligibility = parent.build_macro_domain_eligibility(
        frame,
        include_domain_macro=True,
    )
    n_bootstrap = 100
    output = parent.bootstrap_applicability_delta(
        frame,
        n_bootstrap,
        440_091,
        True,
        eligibility,
    )
    checks: dict[str, bool] = {}
    for metric in parent.METRICS:
        key = f"delta_domain_macro_{metric}"
        checks[f"{metric}_sparse_identity"] = (
            json.loads(
                output[f"{key}_sparse_eligible_domains_json"]
            )
            == ["domain_B"]
        )
        checks[f"{metric}_interval_refused"] = (
            output[f"{key}_ci_status"] == "NON_ESTIMABLE_STRUCTURAL"
            and output[f"{key}_non_estimable_reason"]
            == "fixed_eligible_domain_fewer_than_2_scaffold_clusters"
            and int(output[f"{key}_n_bootstrap_valid"]) == 0
            and int(
                output[
                    f"{key}_n_bootstrap_not_attempted_structural"
                ]
            )
            == n_bootstrap
            and math.isnan(float(output[f"{key}_ci_low"]))
            and math.isnan(float(output[f"{key}_ci_high"]))
            and math.isfinite(float(output[key]))
        )
    return {
        "test": "synthetic_sparse_eligible_domain_fail_safe",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }


def run_real_low_bin_test() -> dict[str, Any]:
    paired = parent.load_applicability_pairs()
    checks: dict[str, bool] = {}
    observations: list[dict[str, Any]] = []
    for regime in ["frozen_ood", "strict_domain_scaffold_ood"]:
        frame = paired[
            (paired["regime"] == regime)
            & (paired["protocol"] == "target_ood")
            & (
                paired["similarity_bin"].astype(str)
                == "0.0_to_lt_0.4"
            )
        ].copy()
        eligibility = parent.build_macro_domain_eligibility(
            frame,
            include_domain_macro=True,
        )
        output = parent.bootstrap_applicability_delta(
            frame,
            100,
            771_000 + len(observations),
            True,
            eligibility,
        )
        for metric in parent.METRICS:
            key = f"delta_domain_macro_{metric}"
            sparse = json.loads(
                output[f"{key}_sparse_eligible_domains_json"]
            )
            local = eligibility[
                (eligibility["metric"] == metric)
                & (
                    eligibility[
                        "eligible_domain_sparse_for_bootstrap"
                    ]
                )
            ]
            label = f"{regime}_{metric}"
            checks[f"{label}_one_sparse_domain"] = (
                sparse == ["IKZF2"]
                and len(local) == 1
                and str(local.iloc[0]["heldout_group"]) == "IKZF2"
                and int(local.iloc[0]["n_rows"]) == 4
                and int(local.iloc[0]["n_unique_scaffolds"]) == 1
            )
            checks[f"{label}_interval_refused"] = (
                output[f"{key}_ci_status"]
                == "NON_ESTIMABLE_STRUCTURAL"
                and int(output[f"{key}_n_bootstrap_valid"]) == 0
                and int(
                    output[
                        f"{key}_n_bootstrap_not_attempted_structural"
                    ]
                )
                == 100
                and math.isnan(float(output[f"{key}_ci_low"]))
                and math.isnan(float(output[f"{key}_ci_high"]))
            )
            observations.append(
                {
                    "regime": regime,
                    "metric": metric,
                    "sparse_domains": sparse,
                    "point_estimate": float(output[key]),
                    "ci_status": output[f"{key}_ci_status"],
                }
            )
    return {
        "test": "real_target_ood_low_bin_sparse_domain_regression",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "observations": observations,
        "checks": checks,
    }


def run_float32_guard_test() -> dict[str, Any]:
    rejected = False
    message = ""
    try:
        parent.run_one_permutation(
            permutation_id=1,
            rows=pd.DataFrame(),
            y=np.asarray([1.0], dtype=np.float64),
            morgan=np.empty((0, 0), dtype=np.float32),
            cached_splits=[],
            n_repeats=1,
            n_estimators=10,
            n_jobs=1,
            seed=parent.DEFAULT_SEED,
        )
    except TypeError as error:
        rejected = True
        message = str(error)
    return {
        "test": "float32_response_guard",
        "status": "PASS" if rejected and "float32" in message else "FAIL",
        "message": message,
    }


def main() -> None:
    results = [
        run_synthetic_complete_domain_test(),
        run_synthetic_sparse_domain_test(),
        run_real_low_bin_test(),
        run_float32_guard_test(),
    ]
    overall = "PASS" if all(
        result["status"] == "PASS" for result in results
    ) else "FAIL"
    payload = {
        "protocol_version": parent.PROTOCOL_VERSION,
        "overall_status": overall,
        "n_tests": len(results),
        "n_pass": sum(result["status"] == "PASS" for result in results),
        "tests": results,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if overall != "PASS":
        raise RuntimeError("Fixed-domain v3 regression QA failed")


if __name__ == "__main__":
    main()
