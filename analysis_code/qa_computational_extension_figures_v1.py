#!/usr/bin/env python3
"""Static and rendered-output QA for Fig. 5 and Supplementary Figs. S1--S3.

Formal mode re-runs the mixed-lineage float32-v4 source gate before checking the
figure bundle.  ``--non-scientific-smoke`` is deliberately restricted to
``/tmp`` and checks rendering mechanics only; it cannot certify scientific
provenance or create a submission-ready PASS.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from PIL import Image

import plot_computational_extension_figures_v1 as plot


DEFAULT_REPORT_NAME = plot.FIGURE_QA_FILENAME
FORMATS = plot.lineage.FIGURE_FORMATS
FIGURE_LIKE_SUFFIXES = frozenset(
    {*FORMATS, ".tif", ".eps", ".jpg", ".jpeg", ".webp"}
)
EXPECTED_TEXT = {
    "fig5_robustness_map": (
        "Held-axis context portability",
        "Scaffold-definition sensitivity",
        "Training-source deletion",
        "95% CI",
        "10,000 resamples",
    ),
    "supp_fig_s1_applicability_profile": (
        "Maximum train-set Tanimoto bin",
        "RMSE",
        "resamples",
        "No lines connect bins",
        "CI NE",
    ),
    "supp_fig_s2_permutation_controls": (
        "Permutation count",
        "one-sided",
        "All 100",
        "within-source",
    ),
    "supp_fig_s3_endpoint_selection_boundary": (
        "parsed-record rows",
        "not unique record IDs",
        "not fitted as point labels",
        "target categories are shown",
    ),
}
EXPECTED_PANEL_LABELS = {
    "fig5_robustness_map": set("abcd"),
    "supp_fig_s1_applicability_profile": set("abcde"),
    "supp_fig_s2_permutation_controls": set("abcd"),
    "supp_fig_s3_endpoint_selection_boundary": set("abcd"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=plot.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-dir", type=Path, default=plot.DEFAULT_SOURCE_DIR)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument(
        "--extension-only",
        action="store_true",
        help="Do not require the copied Fig. 1--4 files.",
    )
    parser.add_argument(
        "--non-scientific-smoke",
        action="store_true",
        help="Render-only QA under /tmp; never a formal provenance PASS.",
    )
    return parser.parse_args()


def parse_physical_length(value: str) -> float:
    match = re.fullmatch(
        r"\s*([0-9]+(?:\.[0-9]+)?)\s*(mm|cm|in|pt)\s*",
        value,
    )
    if match is None:
        raise ValueError(f"Unsupported SVG physical length: {value!r}")
    magnitude = float(match.group(1))
    unit = match.group(2)
    factors = {"mm": 1.0, "cm": 10.0, "in": 25.4, "pt": 25.4 / 72.0}
    return magnitude * factors[unit]


class Audit:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.checks: list[dict[str, object]] = []

    def add(
        self,
        check_id: str,
        passed: bool,
        message: str,
        evidence: object = None,
        *,
        warning: bool = False,
    ) -> None:
        if any(item["check_id"] == check_id for item in self.checks):
            raise RuntimeError(f"Duplicate figure-QA check ID: {check_id}")
        status = "PASS" if passed else ("WARN" if warning else "FAIL")
        self.checks.append(
            {
                "check_id": check_id,
                "status": status,
                "message": message,
                "evidence": evidence,
            }
        )

    @property
    def failed(self) -> bool:
        return any(item["status"] == "FAIL" for item in self.checks)

    def payload(
        self,
        bindings: dict[str, object] | None = None,
    ) -> dict[str, object]:
        status = "FAIL" if self.failed else (
            "SMOKE_TEST_ONLY" if self.mode == "smoke" else "PASS"
        )
        payload: dict[str, object] = {
            "overall_status": status,
            "mode": self.mode,
            "formal_scientific_certification": self.mode == "formal" and not self.failed,
            "checks": self.checks,
            "counts": {
                label: sum(item["status"] == label for item in self.checks)
                for label in ("PASS", "WARN", "FAIL")
            },
        }
        if bindings:
            payload.update(bindings)
        return payload


def svg_text_and_geometry(path: Path) -> tuple[str, float, float, int]:
    raw_svg = path.read_text(encoding="utf-8")
    root = ET.parse(path).getroot()
    width_mm = parse_physical_length(root.attrib["width"])
    height_mm = parse_physical_length(root.attrib["height"])
    texts = [
        "".join(element.itertext())
        for element in root.iter()
        if element.tag.rsplit("}", 1)[-1] == "text"
    ]
    searchable_content = "\n".join([*texts, raw_svg])
    return searchable_content, width_mm, height_mm, len(texts)


def pdf_geometry_and_fonts(path: Path) -> tuple[float, float, list[str]]:
    info = subprocess.run(
        ["pdfinfo", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    match = re.search(
        r"^Page size:\s+([0-9.]+)\s+x\s+([0-9.]+)\s+pts",
        info,
        flags=re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(f"Could not parse PDF page size: {path}")
    width_mm = float(match.group(1)) * 25.4 / 72.0
    height_mm = float(match.group(2)) * 25.4 / 72.0

    font_output = subprocess.run(
        ["pdffonts", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    font_rows = [
        line.split()
        for line in font_output
        if line.strip() and not line.startswith(("name", "-"))
    ]
    failures: list[str] = []
    if not font_rows:
        failures.append("no fonts detected")
    for fields in font_rows:
        if len(fields) < 8 or fields[-5].lower() != "yes":
            failures.append(" ".join(fields))
    return width_mm, height_mm, failures


def check_rendered_figure(
    audit: Audit,
    figure_dir: Path,
    stem: str,
) -> None:
    is_extension = stem in plot.FIGURE_HEIGHT_MM_BY_STEM
    paths = {suffix: figure_dir / f"{stem}{suffix}" for suffix in FORMATS}
    for suffix, path in paths.items():
        audit.add(
            f"{stem}:exists:{suffix}",
            path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
            f"{path.name} exists, is non-symlinked and is non-empty",
            int(path.stat().st_size) if path.is_file() else None,
        )
    if any(not path.is_file() or path.is_symlink() for path in paths.values()):
        return

    for path in paths.values():
        audit.add(
            f"{stem}:size:{path.suffix}",
            path.stat().st_size < 10_000_000,
            f"{path.name} is below 10 MB",
            int(path.stat().st_size),
        )

    expected_width: float | None = None
    expected_height: float | None = None
    try:
        svg_text, svg_width, svg_height, text_count = svg_text_and_geometry(
            paths[".svg"]
        )
        expected_width = (
            plot.FIGURE_WIDTH_MM if is_extension else svg_width
        )
        expected_height = (
            plot.FIGURE_HEIGHT_MM_BY_STEM[stem] if is_extension else svg_height
        )
        audit.add(
            f"{stem}:svg-width",
            (
                math.isclose(svg_width, plot.FIGURE_WIDTH_MM, abs_tol=0.02)
                if is_extension
                else 0 < svg_width <= plot.FIGURE_WIDTH_MM
            ),
            (
                "SVG physical width is 170 mm"
                if is_extension
                else "Frozen SVG width is positive and no wider than 170 mm"
            ),
            svg_width,
        )
        audit.add(
            f"{stem}:svg-height",
            (
                math.isclose(svg_height, expected_height, abs_tol=0.02)
                if is_extension
                else 0 < svg_height < 225.0
            ),
            (
                "SVG physical height matches the figure contract"
                if is_extension
                else "Frozen SVG height is within the journal page budget"
            ),
            svg_height,
        )
        audit.add(
            f"{stem}:svg-editable-text",
            text_count >= 10,
            "SVG retains editable text elements",
            text_count,
        )
        for phrase in EXPECTED_TEXT.get(stem, ()):
            audit.add(
                f"{stem}:text:{phrase}",
                phrase in svg_text,
                f"Rendered figure contains required explanatory text: {phrase}",
            )
        if stem in EXPECTED_PANEL_LABELS:
            observed_panels = {
                value.strip()
                for value in svg_text.splitlines()
                if value.strip() in set("abcdef")
            }
            audit.add(
                f"{stem}:panel-labels",
                EXPECTED_PANEL_LABELS[stem].issubset(observed_panels),
                "Expected lowercase panel labels are rendered",
                sorted(observed_panels),
            )
    except Exception as exc:
        audit.add(
            f"{stem}:svg-parse",
            False,
            "SVG geometry/editable-text audit completed",
            repr(exc),
        )

    try:
        pdf_width, pdf_height, font_failures = pdf_geometry_and_fonts(
            paths[".pdf"]
        )
        audit.add(
            f"{stem}:pdf-width",
            expected_width is not None
            and math.isclose(pdf_width, expected_width, abs_tol=0.12),
            "PDF physical width matches the SVG canvas",
            pdf_width,
        )
        audit.add(
            f"{stem}:pdf-height",
            expected_height is not None
            and math.isclose(pdf_height, expected_height, abs_tol=0.12),
            "PDF physical height matches the SVG/figure contract",
            pdf_height,
        )
        audit.add(
            f"{stem}:pdf-fonts",
            not font_failures,
            "PDF fonts are detected and embedded",
            font_failures,
        )
    except Exception as exc:
        audit.add(
            f"{stem}:pdf-audit",
            False,
            "PDF geometry/font audit completed",
            repr(exc),
        )

    if expected_width is None or expected_height is None:
        return
    expected_pixels = {
        ".tiff": (
            round(expected_width / 25.4 * 600),
            round(expected_height / 25.4 * 600),
            600.0,
        ),
        ".png": (
            round(expected_width / 25.4 * 300),
            round(expected_height / 25.4 * 300),
            300.0,
        ),
    }
    for suffix, (expected_pixel_width, expected_height_px, expected_dpi) in (
        expected_pixels.items()
    ):
        path = paths[suffix]
        try:
            with Image.open(path) as image:
                dpi = image.info.get("dpi", (math.nan, math.nan))
                pixel_tolerance = 1 if is_extension else (3 if suffix == ".tiff" else 2)
                audit.add(
                    f"{stem}:{suffix}:pixels",
                    abs(image.width - expected_pixel_width) <= pixel_tolerance
                    and abs(image.height - expected_height_px) <= pixel_tolerance,
                    f"{suffix} pixel dimensions match the physical canvas",
                    [image.width, image.height],
                )
                audit.add(
                    f"{stem}:{suffix}:dpi",
                    math.isclose(float(dpi[0]), expected_dpi, abs_tol=1.0)
                    and math.isclose(float(dpi[1]), expected_dpi, abs_tol=1.0),
                    f"{suffix} resolution matches the export contract",
                    [float(dpi[0]), float(dpi[1])],
                )
                if suffix == ".tiff":
                    audit.add(
                        f"{stem}:tiff-rgb",
                        image.mode == "RGB",
                        "TIFF is flattened RGB",
                        image.mode,
                    )
                    audit.add(
                        f"{stem}:tiff-lzw",
                        str(image.info.get("compression", "")).lower()
                        == "tiff_lzw",
                        "TIFF uses LZW compression",
                        image.info.get("compression"),
                    )
        except Exception as exc:
            audit.add(
                f"{stem}:{suffix}:raster-audit",
                False,
                f"{suffix} raster audit completed",
                repr(exc),
            )


def check_formal_manifest(
    audit: Audit,
    figure_dir: Path,
    provenance: dict[str, object],
) -> dict[str, object]:
    manifest_path = figure_dir / "figure_bundle_manifest.csv"
    provenance_path = figure_dir / plot.FIGURE_PROVENANCE_FILENAME
    bindings: dict[str, object] = {}
    audit.add(
        "manifest:exists",
        manifest_path.is_file() and not manifest_path.is_symlink(),
        "Figure bundle manifest exists and is non-symlinked",
    )
    audit.add(
        "provenance-record:exists",
        provenance_path.is_file() and not provenance_path.is_symlink(),
        "Figure provenance record exists and is non-symlinked",
    )
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or not provenance_path.is_file()
        or provenance_path.is_symlink()
    ):
        return bindings

    bindings["figure_bundle_manifest_sha256"] = plot.sha256_file(manifest_path)
    bindings["figure_provenance_sha256"] = plot.sha256_file(provenance_path)
    try:
        manifest = pd.read_csv(manifest_path, keep_default_na=False)
        if tuple(manifest.columns) != plot.FIGURE_MANIFEST_COLUMNS:
            raise ValueError(
                "Figure manifest does not match the ordered 20-column schema"
            )
    except Exception as exc:
        audit.add(
            "manifest:schema",
            False,
            "Figure manifest has the required v3 binding schema",
            repr(exc),
        )
        return bindings

    expected_names = set(plot.EXPECTED_FIGURE_NAMES)
    manifest_names = list(manifest["file"].astype(str))
    observed_names = {
        path.name
        for path in figure_dir.iterdir()
        if path.suffix.lower() in FIGURE_LIKE_SUFFIXES
    }
    symlinked_names = sorted(
        path.name
        for path in figure_dir.iterdir()
        if path.suffix.lower() in FIGURE_LIKE_SUFFIXES and path.is_symlink()
    )
    exact_set = (
        len(manifest_names) == len(expected_names)
        and len(manifest_names) == len(set(manifest_names))
        and set(manifest_names) == expected_names
        and observed_names == expected_names
        and not symlinked_names
    )
    audit.add(
        "manifest:exact-figure-set",
        exact_set,
        "Manifest and directory contain exactly 8 stems x 4 formats",
        {
            "manifest_rows": len(manifest_names),
            "directory_files": len(observed_names),
            "symlinks": symlinked_names,
        },
    )
    hashes = list(manifest["sha256"].astype(str))
    audit.add(
        "manifest:unique-sha256",
        len(hashes) == len(expected_names)
        and len(hashes) == len(set(hashes))
        and all(re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes),
        "All 32 figure rows have unique, canonical SHA-256 values",
    )

    extension_stems = set(plot.FIGURE_HEIGHT_MM_BY_STEM)
    bound_keys = (
        "source_data_extension_index_sha256",
        "source_data_extension_checksums_sha256",
        "source_data_extension_provenance_sha256",
        "source_data_generation_id",
    )
    for _, row in manifest.iterrows():
        filename = str(row["file"])
        path = figure_dir / filename
        stem = Path(filename).stem
        suffix = Path(filename).suffix.lower()
        is_extension = stem in extension_stems
        expected_fixed = (
            plot.EXPECTED_FIXED_DOMAIN_CORRECTION_SHA256
            if stem in plot.NULL_DERIVED_FIGURE_STEMS
            else "not_applicable"
        )
        try:
            common_matches = (
                Path(filename).name == filename
                and filename in expected_names
                and suffix == str(row["extension"])
                and suffix in FORMATS
                and path.is_file()
                and not path.is_symlink()
                and int(row["size_bytes"]) == path.stat().st_size
                and str(row["sha256"]) == plot.sha256_file(path)
            )
            raster_columns = (
                "pixel_width",
                "pixel_height",
                "image_mode",
                "dpi_x",
                "dpi_y",
            )
            if suffix in {".png", ".tiff"} and path.is_file():
                with Image.open(path) as image:
                    dpi = image.info.get("dpi", (math.nan, math.nan))
                    raster_matches = (
                        int(float(row["pixel_width"])) == image.width
                        and int(float(row["pixel_height"])) == image.height
                        and str(row["image_mode"]) == image.mode
                        and math.isclose(
                            float(row["dpi_x"]), float(dpi[0]), abs_tol=0.01
                        )
                        and math.isclose(
                            float(row["dpi_y"]), float(dpi[1]), abs_tol=0.01
                        )
                    )
            else:
                raster_matches = all(str(row[key]) == "" for key in raster_columns)
            if is_extension:
                lineage_matches = (
                    str(row["source"]) == plot.EXPECTED_EXTENSION_IDENTITY
                    and str(row["scientific_status"]) == "FORMAL"
                    and str(row["response_dtype"])
                    == plot.EXPECTED_RESPONSE_DTYPE
                    and str(row["float32_correction_sha256"])
                    == plot.EXPECTED_FLOAT32_CORRECTION_SHA256
                    and str(row["fixed_domain_correction_sha256"])
                    == expected_fixed
                    and all(
                        str(row[key]) == str(provenance[key])
                        for key in bound_keys
                    )
                    and math.isclose(
                        float(row["width_mm"]),
                        plot.FIGURE_WIDTH_MM,
                        abs_tol=1e-6,
                    )
                    and math.isclose(
                        float(row["height_mm"]),
                        plot.FIGURE_HEIGHT_MM_BY_STEM[stem],
                        abs_tol=1e-6,
                    )
                )
            else:
                lineage_matches = (
                    stem in plot.EXISTING_FIGURE_STEMS
                    and str(row["source"]) == "existing_publication_figure_v1"
                    and str(row["scientific_status"]) == "FROZEN_EXISTING"
                    and str(row["response_dtype"]) == ""
                    and str(row["float32_correction_sha256"]) == ""
                    and str(row["fixed_domain_correction_sha256"])
                    == "not_applicable"
                    and all(str(row[key]) == "" for key in bound_keys)
                )
            matches = common_matches and raster_matches and lineage_matches
        except Exception:
            matches = False
        audit.add(
            f"manifest:{filename}",
            matches,
            "Manifest row matches its file and mixed-lineage v4 provenance",
        )

    try:
        record = json.loads(provenance_path.read_text(encoding="utf-8"))
    except Exception as exc:
        audit.add(
            "provenance-record:json",
            False,
            "Figure provenance is a readable JSON object",
            repr(exc),
        )
        return bindings
    source_binding = dict(provenance["source_data_provenance"])
    manifest_sha256 = str(bindings["figure_bundle_manifest_sha256"])
    expected_plot_sha256 = plot.sha256_file(Path(plot.__file__).resolve())
    expected_generation_id = plot.lineage.figure_generation_id(
        source_data_generation_id=str(
            provenance["source_data_generation_id"]
        ),
        figure_bundle_manifest_sha256=manifest_sha256,
        plot_script_sha256=expected_plot_sha256,
    )
    provenance_matches = (
        isinstance(record, dict)
        and record.get("status") == "FORMAL"
        and record.get("figure_provenance_version")
        == "computational_extension_figure_float32_v4.0"
        and record.get("provenance_version") == plot.lineage.PROVENANCE_VERSION
        and record.get("analysis_identity") == plot.EXPECTED_EXTENSION_IDENTITY
        and record.get("response_dtype") == plot.EXPECTED_RESPONSE_DTYPE
        and record.get("float32_correction_sha256")
        == plot.EXPECTED_FLOAT32_CORRECTION_SHA256
        and record.get("fixed_domain_correction_sha256")
        == plot.EXPECTED_FIXED_DOMAIN_CORRECTION_SHA256
        and record.get("provenance_contract_sha256")
        == plot.lineage.PROVENANCE_CONTRACT_SHA256
        and record.get("source_data_provenance") == source_binding
        and record.get(
            "publication_eligible_independent_prospective_dataset_available"
        )
        is False
        and record.get("prospective_dataset_statement")
        == plot.lineage.PUBLIC_PROSPECTIVE_DATASET_STATEMENT
        and all(record.get(key) == provenance[key] for key in bound_keys)
        and record.get("figure_bundle_manifest_filename")
        == manifest_path.name
        and record.get("figure_bundle_manifest_sha256") == manifest_sha256
        and record.get("plot_script_sha256") == expected_plot_sha256
        and record.get("figure_generation_id") == expected_generation_id
        and record.get("figure_contract_filename")
        == plot.lineage.FIGURE_CONTRACT_FILENAME
        and record.get("figure_contract_sha256")
        == plot.lineage.FIGURE_CONTRACT_SHA256
        and record.get("figure_qa_script_sha256")
        == plot.lineage.FIGURE_QA_SCRIPT_SHA256
        and record.get("figure_qa_required_check_ids")
        == list(plot.lineage.FIGURE_QA_REQUIRED_CHECK_IDS)
        and record.get("figure_qa_required_check_count")
        == plot.lineage.FIGURE_QA_REQUIRED_CHECK_COUNT
        and record.get("figure_qa_required_check_ids_sha256")
        == plot.lineage.FIGURE_QA_REQUIRED_CHECK_IDS_SHA256
        and record.get("figure_file_count") == len(expected_names)
        and record.get("figure_stems") == list(plot.lineage.FIGURE_STEMS)
        and record.get("figure_formats") == list(plot.lineage.FIGURE_FORMATS)
        and record.get("figure_qa_filename") == plot.FIGURE_QA_FILENAME
        and record.get("figure_qa_must_bind_this_record_sha256") is True
    )
    audit.add(
        "provenance-record:identity",
        provenance_matches,
        "Figure provenance binds the manifest, source generation and v4 lineage",
    )
    if isinstance(record, dict):
        bindings["figure_generation_id"] = record.get("figure_generation_id")
    return bindings


def main() -> None:
    args = parse_args()
    mode = "smoke" if args.non_scientific_smoke else "formal"
    audit = Audit(mode)
    figure_dir = args.figure_dir.resolve()
    if mode == "formal":
        if args.extension_only:
            raise RuntimeError("--extension-only is permitted only for smoke QA")
        if (
            args.figure_dir.is_symlink()
            or figure_dir != plot.DEFAULT_OUTPUT_DIR.resolve()
        ):
            raise RuntimeError(
                "Formal QA accepts only reports/publication_figures_v4"
            )
        if (figure_dir / "SUPERSEDED_DO_NOT_USE.md").exists():
            raise RuntimeError(
                f"Formal figure directory is superseded: {figure_dir}"
            )

    provenance: dict[str, object] | None = None
    figure_contract_sha256 = ""
    figure_qa_script_sha256 = ""
    if args.non_scientific_smoke:
        try:
            figure_dir.relative_to(Path("/tmp").resolve())
        except ValueError as exc:
            raise RuntimeError(
                "--non-scientific-smoke is restricted to /tmp"
            ) from exc
        audit.add(
            "smoke:boundary",
            True,
            "Smoke QA is isolated under /tmp and cannot certify provenance",
        )
    else:
        figure_contract_path = (
            plot.ROUTE_DIR / "docs" / plot.lineage.FIGURE_CONTRACT_FILENAME
        )
        figure_contract_sha256 = (
            plot.sha256_file(figure_contract_path)
            if figure_contract_path.is_file()
            and not figure_contract_path.is_symlink()
            else ""
        )
        figure_qa_script_path = Path(__file__).resolve()
        figure_qa_script_sha256 = plot.sha256_file(figure_qa_script_path)
        audit.add(
            "formal:figure-contract-sha256",
            (
                figure_contract_sha256
                == plot.lineage.FIGURE_CONTRACT_SHA256
            ),
            "Formal figure contract is present, non-symlinked and hash-frozen",
            {
                "expected": plot.lineage.FIGURE_CONTRACT_SHA256,
                "observed": figure_contract_sha256 or "missing",
            },
        )
        audit.add(
            "formal:figure-qa-script-sha256",
            (
                figure_qa_script_sha256
                == plot.lineage.FIGURE_QA_SCRIPT_SHA256
            ),
            "Formal figure-QA implementation matches its frozen identity",
            {
                "expected": plot.lineage.FIGURE_QA_SCRIPT_SHA256,
                "observed": figure_qa_script_sha256,
            },
        )
        try:
            provenance = plot.validate_source_data_provenance(args.source_dir)
            audit.add(
                "formal:source-provenance",
                True,
                "All aggregate tables resolve to the frozen mixed-lineage v4 sources",
                provenance["verified_table_count"],
            )
        except Exception as exc:
            audit.add(
                "formal:source-provenance",
                False,
                "All aggregate tables resolve to the frozen mixed-lineage v4 sources",
                repr(exc),
            )

    stems = (
        tuple(plot.FIGURE_HEIGHT_MM_BY_STEM)
        if mode == "smoke"
        else tuple(plot.lineage.FIGURE_STEMS)
    )
    for stem in stems:
        check_rendered_figure(audit, figure_dir, stem)

    if mode == "formal":
        existing_names = {
            f"{stem}{suffix}"
            for stem in plot.EXISTING_FIGURE_STEMS
            for suffix in FORMATS
        }
        existing = (
            {
                path.name
                for path in figure_dir.iterdir()
                if path.name in existing_names
                and path.is_file()
                and not path.is_symlink()
            }
            if figure_dir.is_dir()
            else set()
        )
        audit.add(
            "existing-figures:complete",
            existing == existing_names,
            "The exact frozen Fig. 1--4 export set is present",
            len(existing),
        )

    figure_bindings: dict[str, object] = {}
    if mode == "formal" and provenance is not None:
        figure_bindings = check_formal_manifest(audit, figure_dir, provenance)

    if mode == "formal":
        expected_ids = set(plot.lineage.FIGURE_QA_REQUIRED_CHECK_IDS)
        gate_id = "formal:required-check-id-set"
        expected_before_gate = expected_ids - {gate_id}
        observed_before_gate = [
            str(item["check_id"]) for item in audit.checks
        ]
        observed_set = set(observed_before_gate)
        exact_required_set = (
            gate_id in expected_ids
            and len(observed_before_gate) == len(observed_set)
            and len(observed_before_gate) == len(expected_before_gate)
            and observed_set == expected_before_gate
        )
        audit.add(
            gate_id,
            exact_required_set,
            "Formal QA executed the complete, unique required check-ID set",
            {
                "required_count": (
                    plot.lineage.FIGURE_QA_REQUIRED_CHECK_COUNT
                ),
                "observed_before_gate": len(observed_before_gate),
                "missing": sorted(expected_before_gate - observed_set),
                "unexpected": sorted(observed_set - expected_before_gate),
            },
        )

    report_path = (
        args.report_json
        if args.report_json is not None
        else figure_dir / DEFAULT_REPORT_NAME
    )
    if mode == "smoke":
        try:
            report_path.resolve().relative_to(Path("/tmp").resolve())
        except ValueError as exc:
            raise RuntimeError("Smoke QA reports must remain under /tmp") from exc
    else:
        expected_report = figure_dir / plot.FIGURE_QA_FILENAME
        if report_path.resolve() != expected_report.resolve():
            raise RuntimeError(
                "Formal QA must write the bound figure QA record in the v4 bundle"
            )
    if report_path.is_symlink():
        raise RuntimeError(f"Refusing to replace symlinked QA report: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    bindings: dict[str, object] = {}
    if mode == "formal" and provenance is not None:
        source_binding = dict(provenance["source_data_provenance"])
        bindings = {
            **source_binding,
            "source_data_provenance": source_binding,
            (
                "publication_eligible_independent_prospective_"
                "dataset_available"
            ): False,
            "prospective_dataset_statement": (
                plot.lineage.PUBLIC_PROSPECTIVE_DATASET_STATEMENT
            ),
            "response_dtype": provenance["response_dtype"],
            "float32_correction_sha256": provenance[
                "float32_correction_sha256"
            ],
            "fixed_domain_correction_sha256": provenance[
                "fixed_domain_correction_sha256"
            ],
            "provenance_contract_sha256": provenance[
                "provenance_contract_sha256"
            ],
            "figure_contract_filename": (
                plot.lineage.FIGURE_CONTRACT_FILENAME
            ),
            "figure_contract_sha256": figure_contract_sha256,
            "figure_qa_script_sha256": figure_qa_script_sha256,
            "figure_qa_required_check_ids": list(
                plot.lineage.FIGURE_QA_REQUIRED_CHECK_IDS
            ),
            "figure_qa_required_check_count": (
                plot.lineage.FIGURE_QA_REQUIRED_CHECK_COUNT
            ),
            "figure_qa_required_check_ids_sha256": (
                plot.lineage.FIGURE_QA_REQUIRED_CHECK_IDS_SHA256
            ),
            **figure_bindings,
        }
    payload = audit.payload(bindings)
    temporary = report_path.with_name(f".{report_path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(report_path)
    print(
        "COMPUTATIONAL_EXTENSION_FIGURE_QA: "
        f"{payload['overall_status']} "
        f"(PASS={payload['counts']['PASS']}; "
        f"WARN={payload['counts']['WARN']}; "
        f"FAIL={payload['counts']['FAIL']}; report={report_path})"
    )
    if audit.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
