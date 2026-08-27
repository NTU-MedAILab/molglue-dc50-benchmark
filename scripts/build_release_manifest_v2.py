#!/usr/bin/env python3
"""Build or check the exact GitHub-v2 public inventory using only stdlib."""

from __future__ import annotations

import argparse
import hashlib
import mimetypes
from pathlib import Path
from typing import Iterable


RELEASE_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "MANIFEST.tsv"
SUMS_NAME = "SHA256SUMS"
EXCLUDED_FILES = {MANIFEST_NAME, SUMS_NAME}
EXCLUDED_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "reports",
    "runtime",
    "runtime_logs",
}
MANIFEST_COLUMNS = (
    "path",
    "role",
    "size_bytes",
    "media_type",
    "sha256",
    "evidence_identity",
    "licence_or_access_route",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_files(root: Path = RELEASE_ROOT) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if any(
            part == ".eggs" or part.endswith(".egg-info")
            for part in relative.parts
        ):
            continue
        if path.is_symlink():
            raise ValueError(f"Symlink is not permitted in release: {relative}")
        if not path.is_file():
            continue
        if relative.as_posix() in EXCLUDED_FILES:
            continue
        if path.suffix in {".pyc", ".tmp"}:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def media_type(path: Path) -> str:
    overrides = {
        ".cff": "application/yaml",
        ".csv": "text/csv",
        ".json": "application/json",
        ".md": "text/markdown",
        ".pdf": "application/pdf",
        ".py": "text/x-python",
        ".svg": "image/svg+xml",
        ".tiff": "image/tiff",
        ".tsv": "text/tab-separated-values",
        ".txt": "text/plain",
        ".yml": "application/yaml",
        ".yaml": "application/yaml",
    }
    if path.name.endswith(".cff.template"):
        return "application/yaml"
    return overrides.get(
        path.suffix.lower(),
        mimetypes.guess_type(path.name)[0] or "application/octet-stream",
    )


def role_for(relative: str) -> str:
    name = Path(relative).name
    if relative.startswith("analysis_code/"):
        return "analysis_code_snapshot"
    if relative.startswith("data_builder/"):
        return "owner_source_reconstruction_code_or_metadata"
    if relative.startswith("reference/cpu_artifact_ledgers/"):
        return "frozen_cpu_artifact_identity_ledger"
    if relative.startswith("tests_snapshot/"):
        return "historical_test_snapshot"
    if relative.startswith("scripts/"):
        return "portable_release_tool"
    if relative.startswith("docs/"):
        return "frozen_protocol_or_identity"
    if relative.startswith("evidence/source_data/"):
        return "aggregate_manuscript_source_data"
    if relative.startswith("evidence/"):
        return "scientific_audit"
    if relative.startswith("figures/"):
        return "publication_figure_or_figure_documentation"
    if name == "THIRD_PARTY_SOURCES.tsv":
        return "third_party_rights_registry"
    if name == "RAW_INPUT_CHECKSUMS.tsv":
        return "restricted_input_identity_registry"
    if name in {
        "SCIENTIFIC_IDENTITY.json",
        "EVIDENCE_CONTRACT.json",
        "FIGURE_CONTRACT.json",
        "FIGURE_SOURCE_MAP.tsv",
    }:
        return "release_identity_or_contract"
    if name in {
        "environment.yml",
        "requirements-lock.txt",
        "environment-lock-linux-64.json",
    }:
        return "environment_specification"
    if name == "CITATION.cff.template":
        return "citation_metadata_template"
    return "release_documentation"


def evidence_identity_for(relative: str) -> str:
    name = Path(relative).name
    if name.startswith("source_data_extension_"):
        return "post_hoc_computational_extension"
    if name.startswith("computational_extension_results_"):
        return "post_hoc_computational_extension_summary"
    if name.startswith("computational_extension_crosscheck_"):
        return "independent_post_hoc_extension_audit"
    identities = {
        "provenance_by_source.csv": "QC_descriptive",
        "provenance_by_target.csv": "QC_descriptive",
        "provenance_by_recruiting_protein.csv": "QC_descriptive",
        "provenance_by_cell_line.csv": "QC_descriptive",
        "provenance_by_assay_method.csv": "QC_descriptive",
        "data_missingness.csv": "QC_descriptive",
        "source_data_internal_models.csv": "confirmatory",
        "source_data_internal_contrasts.csv": (
            "confirmatory_and_formal_matched_extension"
        ),
        "source_data_confirmatory_ood.csv": "confirmatory",
        "source_data_ood_domains.csv": (
            "confirmatory_estimates_exploratory_domain_interpretation"
        ),
        "source_data_strict_ood.csv": "post_hoc_sensitivity",
        "source_data_lodo.csv": "post_hoc_exploratory",
        "source_data_calibration.csv": "post_hoc_exploratory",
        "source_data_matched_estimand_models.csv": "post_hoc_matched_estimand",
        "source_data_matched_estimand_contrasts.csv": "post_hoc_matched_estimand",
        "source_data_matched_domain_contrasts.csv": "post_hoc_matched_estimand_domain_descriptive",
        "source_data_matched_regime_change_lodo.csv": "post_hoc_matched_estimand_fixed_domain_influence",
        "source_data_grouped_context_contrasts.csv": "post_hoc_grouped_context_ablation",
        "source_data_grouped_context_domain_contrasts.csv": "post_hoc_grouped_context_fixed_domain_descriptive",
        "matched_estimand_qa_summary.json": "post_hoc_matched_estimand_QA",
        "grouped_context_qa_summary.json": "post_hoc_grouped_context_QA",
        "source_data_hgb_model_family_sensitivity.csv": (
            "post_hoc_model_family_sensitivity"
        ),
        "table_main_internal.csv": "mixed_as_identified_by_source_row",
        "table_main_ood.csv": "mixed_as_identified_by_source_row",
        "statistical_audit_v1.md": "statistical_audit",
        "reproducibility_deviation_log_v1.md": "reproducibility_audit",
    }
    if name in identities:
        return identities[name]
    if relative.startswith("figures/"):
        return "mixed_as_mapped_in_FIGURE_SOURCE_MAP"
    if relative.startswith("analysis_code/"):
        return "software_snapshot"
    if relative.startswith("docs/"):
        return "protocol_or_frozen_identity"
    if relative.startswith("tests_snapshot/"):
        return "software_test_snapshot"
    if relative.startswith("scripts/"):
        return "release_QA"
    return "release_metadata"


def licence_for(relative: str) -> str:
    name = Path(relative).name
    if name == "LICENSE":
        return "MIT_LICENSE_TEXT"
    if name in {"THIRD_PARTY_SOURCES.tsv", "RAW_INPUT_CHECKSUMS.tsv"}:
        return "metadata_only_no_third_party_rows"
    if relative.startswith(("analysis_code/", "data_builder/", "scripts/", "tests_snapshot/")):
        return "MIT"
    if relative.startswith(("evidence/", "figures/", "docs/")):
        return "PENDING_AUTHOR_CONTENT_LICENSE"
    if name in {
        "DATA_LICENSE.md",
        "CODE_LICENSE_STATUS.md",
        "CITATION.cff.template",
    }:
        return "STATUS_OR_TEMPLATE_NOT_A_LICENCE"
    return "PENDING_AUTHOR_CONTENT_LICENSE"


def manifest_rows(root: Path = RELEASE_ROOT) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for path in payload_files(root):
        relative = path.relative_to(root).as_posix()
        rows.append(
            (
                relative,
                role_for(relative),
                str(path.stat().st_size),
                media_type(path),
                sha256_file(path),
                evidence_identity_for(relative),
                licence_for(relative),
            )
        )
    return rows


def render_manifest(root: Path = RELEASE_ROOT) -> str:
    lines = ["\t".join(MANIFEST_COLUMNS)]
    lines.extend("\t".join(row) for row in manifest_rows(root))
    return "\n".join(lines) + "\n"


def render_sha256sums(
    manifest_text: str,
    root: Path = RELEASE_ROOT,
) -> str:
    entries = [
        (sha256_file(path), path.relative_to(root).as_posix())
        for path in payload_files(root)
    ]
    entries.append(
        (hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(), MANIFEST_NAME)
    )
    entries.sort(key=lambda item: item[1])
    return "".join(f"{digest}  {relative}\n" for digest, relative in entries)


def expected_contents(root: Path = RELEASE_ROOT) -> tuple[str, str]:
    manifest = render_manifest(root)
    return manifest, render_sha256sums(manifest, root)


def check_current(root: Path = RELEASE_ROOT) -> list[str]:
    expected_manifest, expected_sums = expected_contents(root)
    errors: list[str] = []
    manifest_path = root / MANIFEST_NAME
    sums_path = root / SUMS_NAME
    if not manifest_path.is_file():
        errors.append(f"missing {MANIFEST_NAME}")
    elif manifest_path.read_text(encoding="utf-8") != expected_manifest:
        errors.append(f"{MANIFEST_NAME} does not match the current payload tree")
    if not sums_path.is_file():
        errors.append(f"missing {SUMS_NAME}")
    elif sums_path.read_text(encoding="utf-8") != expected_sums:
        errors.append(f"{SUMS_NAME} does not match the current payload tree")
    return errors


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_current(root: Path = RELEASE_ROOT) -> tuple[int, str]:
    manifest, sums = expected_contents(root)
    atomic_write(root / MANIFEST_NAME, manifest)
    atomic_write(root / SUMS_NAME, sums)
    return len(manifest_rows(root)), hashlib.sha256(
        manifest.encode("utf-8")
    ).hexdigest()


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="check existing files instead of regenerating them",
    )
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] | None = None) -> int:
    args = parse_args(arguments)
    if args.check:
        errors = check_current()
        if errors:
            for error in errors:
                print(f"FAIL: {error}")
            return 1
        print("PASS: manifest and SHA256SUMS match the exact payload tree")
        return 0
    count, manifest_hash = write_current()
    print(f"WROTE: {MANIFEST_NAME} with {count} payload files")
    print(f"WROTE: {SUMS_NAME}")
    print(f"MANIFEST_SHA256: {manifest_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
