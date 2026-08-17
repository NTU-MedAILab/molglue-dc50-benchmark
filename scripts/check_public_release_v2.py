#!/usr/bin/env python3
"""Fail closed if the v2 public payload contains row-level study data."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_DIRS = {"reports", "runtime", "runtime_logs"}
FORBIDDEN_BASENAMES = {
    "MolGlueDB_full.csv",
    "all_molglue_dc50_parsed_records.csv",
    "all_molglue_dc50_qc_train_test.csv",
    "all_molglue_dc50_qc_train_test_standardized_context.csv",
    "all_molglue_dc50_strict_exact_records.csv",
    "all_molglue_dc50_train.csv",
    "all_molglue_dc50_test.csv",
    "mgtbind_compounds_260531.csv",
    "mgtbind_complexes_260531.csv",
    "mgtbind_citations_260531.csv",
    "mgdb_MG_Compound_260531.csv",
    "mgdb_Activity_Data_260531.csv",
    "tpddb_MG_main_table_260531.txt",
    "tpddb_MG_activity_260531.txt",
    "all_cross_fitted_predictions.csv",
    "ood_predictions.csv",
    "strict_ood_predictions.csv",
    "predictions.csv",
    "internal_predictions.csv",
    "source_deletion_predictions.csv",
    "generic_scaffold_predictions.csv",
    "applicability_paired_source_data.csv",
}
ALLOWED_PLACEHOLDERS = {
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data_builder/data/raw/.gitkeep",
    "data_builder/data/processed/.gitkeep",
}


def public_files() -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
        )
        return [ROOT / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]
    except (FileNotFoundError, subprocess.CalledProcessError):
        return [path for path in ROOT.rglob("*") if path.is_file()]


def main() -> int:
    failures: list[str] = []
    files = public_files()
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        parts = set(Path(relative).parts)
        if relative in ALLOWED_PLACEHOLDERS:
            continue
        if parts & FORBIDDEN_DIRS:
            failures.append(f"runtime result included: {relative}")
        if path.name in FORBIDDEN_BASENAMES:
            failures.append(f"row-level artifact included: {relative}")
        if relative.startswith(("data/raw/", "data/processed/", "data_builder/data/raw/", "data_builder/data/processed/")):
            failures.append(f"local data included: {relative}")
    for relative in (
        "data_builder/config/source_manifest.json",
        "data_builder/config/expected_outputs.json",
        "SCIENTIFIC_IDENTITY.json",
    ):
        try:
            json.loads((ROOT / relative).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            failures.append(f"invalid required JSON {relative}: {error}")
    if failures:
        print("PUBLIC RELEASE CHECK FAILED", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(f"PASS: v2 public boundary checked across {len(files)} files; no row-level source or prediction data found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

