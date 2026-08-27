#!/usr/bin/env python3
"""Merge independently executed internal and OOD grouped-ablation components."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INTERNAL = ROOT / "reports" / "post_hoc_context_group_ablation_revision_v1"
DEFAULT_OOD = ROOT / "reports" / "post_hoc_context_group_ablation_revision_v1_ood_component"
OOD_FILES = (
    "ood_predictions.csv",
    "ood_selected_hyperparameters.csv",
    "ood_tuning_metrics.csv",
    "ood_split_audit.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-dir", type=Path, default=DEFAULT_INTERNAL)
    parser.add_argument("--ood-dir", type=Path, default=DEFAULT_OOD)
    return parser.parse_args()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    internal_dir = args.internal_dir.resolve()
    ood_dir = args.ood_dir.resolve()
    internal_config = json.loads((internal_dir / "configuration.json").read_text(encoding="utf-8"))
    ood_config = json.loads((ood_dir / "configuration.json").read_text(encoding="utf-8"))
    identity_fields = (
        "data_sha256",
        "internal_baseline_sha256",
        "ood_baseline_sha256",
        "n_estimators",
        "n_jobs",
        "inner_folds",
        "seed",
        "tree_grid",
    )
    mismatches = [field for field in identity_fields if internal_config.get(field) != ood_config.get(field)]
    if mismatches:
        raise RuntimeError(f"Component configuration mismatch: {mismatches}")
    if internal_config.get("n_estimators") != 600 or ood_config.get("max_internal_splits") != 0:
        raise RuntimeError("Components are not the formal internal/OOD execution pair")

    internal = pd.read_csv(internal_dir / "internal_predictions.csv")
    if len(internal) != 39_000:
        raise RuntimeError(f"Expected 39,000 internal predictions, observed {len(internal)}")
    if internal.duplicated(["repeat", "outer_fold", "model_id", "row_index"]).any():
        raise RuntimeError("Internal prediction keys are duplicated")
    split_models = internal.groupby(["repeat", "outer_fold"])["model_id"].nunique()
    if len(split_models) != 25 or not (split_models == 5).all():
        raise RuntimeError("Internal component does not contain 25 complete five-model splits")

    ood = pd.read_csv(ood_dir / "ood_predictions.csv")
    if len(ood) != 12_530:
        raise RuntimeError(f"Expected 12,530 OOD predictions, observed {len(ood)}")
    if ood.duplicated(["protocol", "heldout_group", "model_id", "row_index"]).any():
        raise RuntimeError("OOD prediction keys are duplicated")
    group_models = ood.groupby(["protocol", "heldout_group"])["model_id"].nunique()
    expected_models = {"source_ood": 4, "target_ood": 5}
    if len(group_models) != 12 or any(
        count != expected_models[protocol]
        for (protocol, _), count in group_models.items()
    ):
        raise RuntimeError("OOD component does not contain 12 complete protocol groups")
    ood_qa = json.loads((ood_dir / "qa_summary.json").read_text(encoding="utf-8"))
    if ood_qa.get("status") != "PASS":
        raise RuntimeError("OOD component QA did not pass")

    source_hashes = {filename: sha(ood_dir / filename) for filename in OOD_FILES}
    for filename in OOD_FILES:
        shutil.copy2(ood_dir / filename, internal_dir / filename)
    copied_hashes = {filename: sha(internal_dir / filename) for filename in OOD_FILES}
    if copied_hashes != source_hashes:
        raise RuntimeError("Copied OOD component hashes do not match")
    provenance = {
        "status": "PASS",
        "execution_design": "independent internal and OOD CPU components with identical formal model settings",
        "internal_configuration_sha256": sha(internal_dir / "configuration.json"),
        "ood_configuration_sha256": sha(ood_dir / "configuration.json"),
        "protocol_sha256_internal_component": internal_config.get("protocol_sha256"),
        "protocol_sha256_ood_component": ood_config.get("protocol_sha256"),
        "note": (
            "Protocol v1.1 corrected grouped-context labels before effect estimates were inspected; "
            "the executable feature definitions and all computational settings were unchanged."
        ),
        "n_internal_predictions": len(internal),
        "n_ood_predictions": len(ood),
        "ood_file_sha256": copied_hashes,
    }
    (internal_dir / "component_execution_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, sort_keys=True))


if __name__ == "__main__":
    main()
