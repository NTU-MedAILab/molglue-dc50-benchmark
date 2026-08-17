#!/usr/bin/env python
"""Fail if a public commit contains ignored third-party or derived row-level data."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PREFIXES = ("data/raw/", "data/processed/")
ALLOWED_PLACEHOLDERS = {"data/raw/.gitkeep", "data/processed/.gitkeep"}
MAX_TRACKED_BYTES = 5 * 1024 * 1024


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    )
    return [item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def main() -> int:
    try:
        files = tracked_files()
    except (FileNotFoundError, subprocess.CalledProcessError):
        files = [str(path.relative_to(ROOT)) for path in ROOT.rglob("*") if path.is_file()]

    failures: list[str] = []
    for relative in files:
        path = ROOT / relative
        if relative.startswith(FORBIDDEN_PREFIXES) and relative not in ALLOWED_PLACEHOLDERS:
            failures.append(f"forbidden row-level data path: {relative}")
        if path.is_file() and path.stat().st_size > MAX_TRACKED_BYTES:
            failures.append(f"unexpected tracked file above 5 MiB: {relative}")

    manifest = ROOT / "config" / "source_manifest.json"
    expected = ROOT / "config" / "expected_outputs.json"
    for required_json in [manifest, expected]:
        try:
            with required_json.open("r", encoding="utf-8") as handle:
                json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            failures.append(f"invalid required JSON {required_json.name}: {error}")

    if failures:
        print("PUBLIC RELEASE CHECK FAILED", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(f"PASS: public release boundary checked across {len(files)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

