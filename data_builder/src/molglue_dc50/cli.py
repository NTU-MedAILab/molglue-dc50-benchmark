"""Command-line interface for acquisition, construction, and verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .build import build_dataset
from .io import SnapshotMismatch, download_sources, load_json, sha256_file, verify_required_sources


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def verify_outputs(output_dir: Path, expected: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    for filename, identity in expected["outputs"].items():
        path = output_dir / filename
        if not path.is_file():
            failures.append(f"missing output: {filename}")
            continue
        frame = pd.read_csv(path)
        actual = {"rows": int(len(frame)), "columns": int(len(frame.columns)), "sha256": sha256_file(path)}
        passed = all(actual[key] == identity[key] for key in ["rows", "columns", "sha256"])
        checks.append({"filename": filename, "expected": identity, "actual": actual, "passed": passed})
        if not passed:
            failures.append(f"identity mismatch: {filename}")
    with (output_dir / "verification_report.json").open("w", encoding="utf-8") as handle:
        json.dump({"passed": not failures, "checks": checks, "failures": failures}, handle, indent=2)
        handle.write("\n")
    if failures:
        raise RuntimeError("Output verification failed:\n" + "\n".join(failures))
    return checks


def build_parser() -> argparse.ArgumentParser:
    root = repository_root()
    parser = argparse.ArgumentParser(
        prog="molglue-dc50",
        description="Reconstruct the frozen four-source molecular-glue DC50 analysis tables.",
    )
    parser.add_argument("command", choices=["download", "verify-sources", "build", "verify", "all"])
    parser.add_argument("--raw-dir", type=Path, default=root / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "processed")
    parser.add_argument("--manifest", type=Path, default=root / "config" / "source_manifest.json")
    parser.add_argument("--expected", type=Path, default=root / "config" / "expected_outputs.json")
    parser.add_argument("--include-optional", action="store_true", help="Also acquire source-reference audit files.")
    parser.add_argument("--force", action="store_true", help="Re-download existing source files and re-extract archives.")
    parser.add_argument(
        "--acknowledge-third-party-notice",
        action="store_true",
        help="Confirm that official downloads remain third-party data and will not be redistributed from this repository.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_json(args.manifest)
    expected = load_json(args.expected)
    try:
        if args.command in {"download", "all"}:
            if not args.acknowledge_third_party_notice:
                raise RuntimeError(
                    "Downloading requires --acknowledge-third-party-notice. "
                    "Access does not grant permission to republish source exports or row-level derivatives."
                )
            log = download_sources(
                args.raw_dir, manifest, include_optional=args.include_optional, force=args.force
            )
            print(f"Verified {len(log)} downloaded/existing source artifacts.")
        if args.command in {"verify-sources", "build", "all"}:
            source_checks = verify_required_sources(args.raw_dir, manifest)
            print(f"Verified {len(source_checks)} required source artifacts against frozen SHA-256 values.")
        if args.command in {"build", "all"}:
            report = build_dataset(args.raw_dir, args.output_dir)
            print(
                f"Built {report['exclusion_audit']['qc_rows_after_aggregation']} QC rows "
                f"in {args.output_dir}."
            )
        if args.command in {"verify", "all"}:
            checks = verify_outputs(args.output_dir, expected)
            print(f"PASS: {len(checks)} output tables match the frozen row, column, and SHA-256 identities.")
        return 0
    except (FileNotFoundError, RuntimeError, SnapshotMismatch) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

