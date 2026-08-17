from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_confirmatory_cpu_v1 as core  # noqa: E402
import run_post_hoc_strict_domain_scaffold_ood_cpu_v1 as strict  # noqa: E402


@pytest.fixture(scope="module")
def strict_inputs():
    frozen, identity_sha = strict.load_extension_identity(
        strict.EXTENSION_IDENTITY
    )
    rows, _ = core.load_rows(core.DATA_FILE)
    scaffolds = core.build_scaffold_ids(rows["canonical_smiles"])
    splits, audit = strict.build_strict_splits(
        rows,
        scaffolds,
        list(strict.PROTOCOL_ORDER),
        strict.DEFAULT_SEED,
        None,
        frozen,
    )
    return frozen, identity_sha, rows, scaffolds, splits, audit


def test_external_identity_and_parent_hashes_are_fail_closed(strict_inputs):
    frozen, _, _, _, _, _ = strict_inputs
    observed = strict.observed_file_hashes(core.DATA_FILE)
    assert observed == frozen["file_sha256"]
    assert frozen["formal_arguments"] == strict.frozen_formal_arguments()
    assert (
        frozen["protocol_version"]
        == "post_hoc_strict_domain_scaffold_ood_cpu_v1.0"
    )


def test_strict_splits_preserve_tests_and_remove_all_training_scaffolds(
    strict_inputs,
):
    frozen, _, rows, scaffolds, splits, audit = strict_inputs
    assert len(splits) == 12
    assert len(audit) == 12
    for split in splits:
        train_scaffolds = set(scaffolds[split.train_idx])
        test_scaffolds = set(scaffolds[split.test_idx])
        train_smiles = set(
            rows.iloc[split.train_idx]["canonical_smiles"].astype(str)
        )
        test_smiles = set(
            rows.iloc[split.test_idx]["canonical_smiles"].astype(str)
        )
        assert train_scaffolds.isdisjoint(test_scaffolds)
        assert train_smiles.isdisjoint(test_smiles)
        expected = strict.strict_expected(
            frozen,
            split.protocol,
            split.heldout_group,
        )
        assert len(split.train_idx) == expected["n_train_rows"]
        assert len(split.test_idx) == expected["n_test_rows"]
        assert (
            len(split.scaffold_excluded_idx)
            == expected["n_scaffold_overlap_rows_excluded"]
        )
        assert strict.sequence_sha256(
            int(value) for value in split.train_idx
        ) == expected["strict_train_indices_sha256"]
        assert strict.sequence_sha256(
            int(value) for value in split.test_idx
        ) == expected["test_indices_sha256"]


def test_strict_test_rows_exactly_match_parent_ood_predictions(strict_inputs):
    _, _, _, _, splits, _ = strict_inputs
    parent_predictions = pd.read_csv(
        strict.OOD_RESULT_DIR / "ood_predictions.csv"
    )
    reference_model = strict.ood.MODEL_ORDER[0]
    parent_predictions = parent_predictions[
        parent_predictions["model_id"] == reference_model
    ]
    for split in splits:
        parent_rows = (
            parent_predictions[
                (parent_predictions["protocol"] == split.protocol)
                & (
                    parent_predictions["heldout_group"]
                    == split.heldout_group
                )
            ]
            .sort_values("row_index")["row_index"]
            .to_numpy(dtype=np.int64)
        )
        assert np.array_equal(
            parent_rows,
            np.sort(split.test_idx.astype(np.int64)),
        )


def test_exact_three_matched_extra_trees_six_grids():
    assert strict.MODEL_ORDER == (
        "context_extra_trees",
        "chemistry_extra_trees",
        "full_context_extra_trees",
    )
    for model_id in strict.MODEL_ORDER:
        grid = strict.model_grid(model_id)
        assert len(grid) == 6
        assert {
            (
                int(params["min_samples_leaf"]),
                str(params["max_features"]),
            )
            for params in grid
        } == {
            (leaf, str(max_features))
            for leaf in (1, 5, 10)
            for max_features in ("sqrt", 0.3)
        }


def test_final_refit_seeds_reuse_frozen_parent_model_positions():
    assert strict.PARENT_OOD_MODEL_POSITIONS == {
        "context_extra_trees": 3,
        "chemistry_extra_trees": 5,
        "full_context_extra_trees": 6,
    }
    for model_id in strict.MODEL_ORDER:
        assert (
            strict.PARENT_OOD_MODEL_POSITIONS[model_id]
            == strict.ood.MODEL_ORDER.index(model_id)
        )


def test_global_scaffold_bootstrap_is_paired_and_post_hoc():
    records = []
    domains = strict.GROUPS_BY_PROTOCOL["source_ood"]
    row_index = 0
    for scaffold_position, scaffold in enumerate(("s1", "s2", "s3")):
        for domain_position, domain in enumerate(domains):
            truth = float(scaffold_position + domain_position / 10)
            for model_id, error in (
                ("context_extra_trees", 0.5),
                ("chemistry_extra_trees", 0.3),
                ("full_context_extra_trees", 0.1),
            ):
                records.append(
                    {
                        "analysis_label": "post_hoc_sensitivity",
                        "protocol": "source_ood",
                        "heldout_group": domain,
                        "row_index": row_index,
                        "scaffold_id": scaffold,
                        "model_id": model_id,
                        "y_true": truth,
                        "y_pred": truth + error * (-1) ** domain_position,
                    }
                )
            row_index += 1
    result = strict.paired_global_scaffold_bootstrap(
        pd.DataFrame(records),
        n_bootstrap=50,
        seed=strict.DEFAULT_SEED,
    )
    assert len(result) == 5
    assert set(result["analysis_label"]) == {"post_hoc_sensitivity"}
    contrasts = result[result["record_type"] == "contrast"]
    assert set(contrasts["contrast_id"].dropna()) == set(strict.CONTRASTS)
    assert (contrasts["delta_pooled_rmse_observed"] > 0).all()
    assert set(result["cluster_unit"]) == {"global scaffold_id"}
    assert set(result["n_bootstrap"]) == {50}


def test_checkpoint_inventory_rejects_tampering(tmp_path):
    records = {
        "predictions": [
            {
                "protocol": "source_ood",
                "heldout_group": "MGTbind",
                "model_id": strict.MODEL_ORDER[0],
                "row_index": 0,
            }
        ],
        "domain_metrics": [
            {
                "protocol": "source_ood",
                "heldout_group": "MGTbind",
                "model_id": strict.MODEL_ORDER[0],
            }
        ],
        "tuning": [
            {
                "protocol": "source_ood",
                "heldout_group": "MGTbind",
                "model_id": strict.MODEL_ORDER[0],
                "param_id": "leaf_1_maxfeat_sqrt",
            }
        ],
        "selected": [
            {
                "protocol": "source_ood",
                "heldout_group": "MGTbind",
                "model_id": strict.MODEL_ORDER[0],
            }
        ],
        "inner_audit": [
            {
                "protocol": "source_ood",
                "heldout_group": "MGTbind",
                "inner_fold": 1,
            }
        ],
    }
    digest = strict.canonical_json_sha256({"test": True})
    strict.write_progress(tmp_path, records, digest)
    strict.verify_checkpoint(tmp_path, digest)

    prediction_path = tmp_path / strict.PROGRESS_FILES["predictions"]
    frame = pd.read_csv(prediction_path)
    frame.loc[0, "row_index"] = 999
    frame.to_csv(prediction_path, index=False)
    with pytest.raises(RuntimeError, match="inventory mismatch"):
        strict.verify_checkpoint(tmp_path, digest)


def test_completed_resume_rejects_vacuous_artifact_inventory(tmp_path):
    manifest = {
        "artifact_inventory": {},
        "artifacts": strict.artifact_manifest_entries(
            include_bootstrap=True
        ),
    }
    with pytest.raises(
        RuntimeError,
        match="artifact inventory filenames changed",
    ):
        strict.verify_completed_result(
            tmp_path,
            manifest,
            include_bootstrap=True,
            formal_match=True,
        )


def test_both_frozen_parent_result_directories_are_protected():
    assert strict.output_is_in_frozen_parent(core.DEFAULT_OUTPUT_DIR)
    assert strict.output_is_in_frozen_parent(
        core.DEFAULT_OUTPUT_DIR / "strict_child"
    )
    assert strict.output_is_in_frozen_parent(strict.OOD_RESULT_DIR)
    assert strict.output_is_in_frozen_parent(
        strict.OOD_RESULT_DIR / "strict_child"
    )
    assert not strict.output_is_in_frozen_parent(strict.DEFAULT_OUTPUT_DIR)


def test_manifest_labels_cannot_be_confirmatory():
    source = Path(strict.__file__).read_text(encoding="utf-8")
    assert '"confirmatory": False' in source
    assert '"analysis_label": "post_hoc_sensitivity"' in source
    protocol = strict.EXTENSION_PROTOCOL.read_text(encoding="utf-8")
    assert "post-hoc sensitivity analysis" in protocol.lower()
    identity = json.loads(
        strict.EXTENSION_IDENTITY.read_text(encoding="utf-8")
    )
    assert identity["frozen_identity"]["protocol_version"].startswith(
        "post_hoc_"
    )
