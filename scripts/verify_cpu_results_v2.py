#!/usr/bin/env python3
"""Compare a rebuilt CPU result tree with the frozen exact CSV identities."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = RELEASE_ROOT / "reference" / "cpu_artifact_ledgers"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_ledger(path: Path) -> list[tuple[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"empty ledger: {path}")
        path_column = next(
            (name for name in ("filename", "file", "relative_path", "artifact") if name in reader.fieldnames),
            None,
        )
        if path_column is None or "sha256" not in reader.fieldnames:
            raise ValueError(f"unsupported ledger schema: {path}")
        rows = []
        for row in reader:
            relative = row[path_column]
            # Exact CSV identity is the portable scientific gate. Runtime JSON,
            # Markdown and protocol/script ledger entries may contain paths,
            # timestamps or platform metadata and are audited separately.
            if relative.endswith(".csv") and relative != "artifact_sha256.csv":
                rows.append((relative, row["sha256"]))
        return rows


def verify(work_root: Path) -> dict[str, object]:
    checks: list[dict[str, object]] = []
    for ledger in sorted(REFERENCE_DIR.glob("*.csv")):
        report_dir = work_root / "reports" / ledger.stem
        for relative, expected in parse_ledger(ledger):
            candidate = report_dir / relative
            observed = sha256_file(candidate) if candidate.is_file() else None
            checks.append(
                {
                    "workflow": ledger.stem,
                    "artifact": relative,
                    "expected_sha256": expected,
                    "observed_sha256": observed,
                    "passed": observed == expected,
                }
            )
    failures = [check for check in checks if not check["passed"]]
    return {
        "schema_version": "2.0",
        "comparison_scope": "exact byte identity for frozen CPU CSV artifacts",
        "work_root": str(work_root.resolve()),
        "checks": checks,
        "n_checks": len(checks),
        "n_passed": len(checks) - len(failures),
        "n_failed": len(failures),
        "passed": not failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.work_root)
    output = args.output or args.work_root / "cpu_reproduction_acceptance_v2.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if report["passed"]:
        print(f"PASS: {report['n_passed']}/{report['n_checks']} frozen CPU CSV identities match exactly.")
        print(f"REPORT: {output}")
        return 0
    print(f"FAIL: {report['n_failed']}/{report['n_checks']} CPU CSV identities differ or are missing.")
    print(f"REPORT: {output}")
    for check in report["checks"]:
        if not check["passed"]:
            print(f"- {check['workflow']}/{check['artifact']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

