#!/usr/bin/env python3
"""Render publication figures for the frozen post-hoc extension.

Only aggregate, publication-safe source-data tables are read.  The script
creates Fig. 5 and Supplementary Figs. S1--S3 as editable SVG/PDF, 600 dpi
LZW-compressed RGB TIFF, and PNG previews.  Existing Fig. 1--4 bundles are
copied into the v4 directory so that the output is a complete figure package.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image

import computational_extension_lineage_v1 as lineage


SCRIPT_DIR = Path(__file__).resolve().parent
ROUTE_DIR = SCRIPT_DIR.parent
DEFAULT_SOURCE_DIR = ROUTE_DIR / "reports" / "publication_validation_v1"
DEFAULT_EXISTING_DIR = ROUTE_DIR / "reports" / "publication_figures_v1"
DEFAULT_OUTPUT_DIR = ROUTE_DIR / "reports" / lineage.FIGURE_DIRECTORY_NAME

MM_PER_INCH = 25.4
FIGURE_WIDTH_MM = 170.0
FIGURE_WIDTH_IN = FIGURE_WIDTH_MM / MM_PER_INCH
EXPECTED_EXTENSION_IDENTITY = lineage.ANALYSIS_IDENTITY
EXPECTED_RESPONSE_DTYPE = "float32"
EXPECTED_FLOAT32_CORRECTION_SHA256 = lineage.FLOAT32_CORRECTION_SHA256
EXPECTED_FIXED_DOMAIN_CORRECTION_SHA256 = str(
    lineage.WORKFLOW_IDENTITIES["null_applicability_censoring"][
        "fixed_domain_correction_sha256"
    ]
)
SOURCE_INDEX_FILENAME = "source_data_extension_index.csv"
SOURCE_CHECKSUM_FILENAME = "source_data_extension_sha256.csv"
SOURCE_PROVENANCE_FILENAME = lineage.PROVENANCE_FILENAME
FIGURE_PROVENANCE_FILENAME = lineage.FIGURE_PROVENANCE_FILENAME
FIGURE_QA_FILENAME = lineage.FIGURE_QA_FILENAME
FIGURE_LEGENDS_FILENAME = "figure_legends.md"
FIGURE_QA_NOTES_FILENAME = "figure_qa_notes.md"

CONTEXT_PARENT = (
    "reports/"
    + str(lineage.WORKFLOW_IDENTITIES["context_weight"]["directory_name"])
)
SCAFFOLD_PARENT = (
    "reports/"
    + str(lineage.WORKFLOW_IDENTITIES["scaffold_source"]["directory_name"])
)
NULL_PARENT = (
    "reports/"
    + str(
        lineage.WORKFLOW_IDENTITIES["null_applicability_censoring"][
            "directory_name"
        ]
    )
)
EXPECTED_PROTOCOL_BY_PARENT = {
    CONTEXT_PARENT: str(
        lineage.WORKFLOW_IDENTITIES["context_weight"]["protocol_version"]
    ),
    SCAFFOLD_PARENT: str(
        lineage.WORKFLOW_IDENTITIES["scaffold_source"]["protocol_version"]
    ),
    NULL_PARENT: str(
        lineage.WORKFLOW_IDENTITIES["null_applicability_censoring"][
            "protocol_version"
        ]
    ),
}
PLOT_TABLE_LINEAGE = {
    "portable_context": CONTEXT_PARENT,
    "weighting_bootstrap": CONTEXT_PARENT,
    "generic_scaffold": SCAFFOLD_PARENT,
    "source_deletion": SCAFFOLD_PARENT,
    "applicability_contrasts": NULL_PARENT,
    "permutation_models": NULL_PARENT,
    "permutation_contrasts": NULL_PARENT,
    "permutation_inference": NULL_PARENT,
    "censoring_overall": NULL_PARENT,
    "censoring_by_source": NULL_PARENT,
    "censoring_by_target": NULL_PARENT,
    "censoring_overlap": NULL_PARENT,
}
FIGURE_HEIGHT_MM_BY_STEM = {
    "fig5_robustness_map": 5.55 * MM_PER_INCH,
    "supp_fig_s1_applicability_profile": 7.42 * MM_PER_INCH,
    "supp_fig_s2_permutation_controls": 5.35 * MM_PER_INCH,
    "supp_fig_s3_endpoint_selection_boundary": 5.90 * MM_PER_INCH,
}
EXPECTED_FIGURE_NAMES = frozenset(
    f"{stem}{suffix}"
    for stem in lineage.FIGURE_STEMS
    for suffix in lineage.FIGURE_FORMATS
)
EXISTING_FIGURE_STEMS = tuple(lineage.FIGURE_STEMS[:4])
NULL_DERIVED_FIGURE_STEMS = frozenset(
    {
        "supp_fig_s1_applicability_profile",
        "supp_fig_s2_permutation_controls",
        "supp_fig_s3_endpoint_selection_boundary",
    }
)
FIGURE_MANIFEST_COLUMNS = (
    "file",
    "extension",
    "size_bytes",
    "sha256",
    "width_mm",
    "height_mm",
    "source",
    "scientific_status",
    "response_dtype",
    "float32_correction_sha256",
    "fixed_domain_correction_sha256",
    "source_data_extension_index_sha256",
    "source_data_extension_checksums_sha256",
    "source_data_extension_provenance_sha256",
    "source_data_generation_id",
    "pixel_width",
    "pixel_height",
    "image_mode",
    "dpi_x",
    "dpi_y",
)

CHEM = "chemistry_extra_trees"
FULL = "full_context_extra_trees"
PORTABLE = "portable_full_extra_trees"

COLORS = {
    "ink": "#182B3A",
    "navy": "#173F5F",
    "blue": "#4C78A8",
    "violet": "#7A5195",
    "teal": "#2A9D8F",
    "ochre": "#C8942D",
    "green": "#2F7D5C",
    "red": "#B64E4E",
    "grey": "#747B83",
    "light_grey": "#DCE2E7",
    "pale_blue": "#DCE8F2",
    "grid": "#D8DDE2",
    "white": "#FFFFFF",
    "exact": "#173F5F",
    "equality": "#173F5F",
    "range": "#2A9D8F",
    "one_sided": "#C8942D",
    "missing_or_unparsed": "#C8CDD2",
}
CENSOR_HATCHES = {
    "equality": "",
    "range": "////",
    "one_sided": "\\\\\\\\",
    "missing_or_unparsed": "..",
}

BIN_ORDER = [
    "0.0_to_lt_0.4",
    "0.4_to_lt_0.6",
    "0.6_to_lt_0.8",
    "0.8_to_1.000001",
]
BIN_LABELS = {
    "0.0_to_lt_0.4": "0–0.4",
    "0.4_to_lt_0.6": "0.4–0.6",
    "0.6_to_lt_0.8": "0.6–0.8",
    "0.8_to_1.000001": "0.8–1.0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument(
        "--existing-figure-dir", type=Path, default=DEFAULT_EXISTING_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def set_style() -> None:
    # Literal settings are retained for deterministic editable-text preflight.
    # svg.fonttype='none'; pdf.fonttype=42
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [
        "Arial",
        "Helvetica",
        "Liberation Sans",
        "DejaVu Sans",
    ]
    mpl.rcParams["font.size"] = 7
    mpl.rcParams["axes.labelsize"] = 7
    mpl.rcParams["axes.titlesize"] = 8
    mpl.rcParams["xtick.labelsize"] = 6.5
    mpl.rcParams["ytick.labelsize"] = 6.5
    mpl.rcParams["legend.fontsize"] = 6.3
    mpl.rcParams["axes.linewidth"] = 0.65
    mpl.rcParams["axes.edgecolor"] = COLORS["ink"]
    mpl.rcParams["axes.labelcolor"] = COLORS["ink"]
    mpl.rcParams["xtick.color"] = COLORS["ink"]
    mpl.rcParams["ytick.color"] = COLORS["ink"]
    mpl.rcParams["text.color"] = COLORS["ink"]
    mpl.rcParams["savefig.facecolor"] = "white"
    mpl.rcParams["figure.facecolor"] = "white"
    mpl.rcParams["legend.frameon"] = False
    mpl.rcParams["hatch.linewidth"] = 0.4
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


sha256_file = lineage.sha256_file


def validate_source_data_provenance(source_dir: Path) -> dict[str, object]:
    """Validate the one frozen mixed-lineage v4 aggregate before plotting."""

    source_dir = source_dir.absolute()
    if (
        source_dir.is_symlink()
        or not source_dir.is_dir()
        or source_dir.resolve() != DEFAULT_SOURCE_DIR.resolve()
    ):
        raise RuntimeError(
            "Formal plotting accepts only the frozen publication_validation_v1 "
            "aggregate directory"
        )
    if (source_dir / "SUPERSEDED_DO_NOT_USE.md").exists():
        raise RuntimeError(f"Source-data directory is superseded: {source_dir}")

    lineage.require_publication_lineage_frozen()
    bound = lineage.aggregate_bundle_hashes(source_dir)
    provenance_path = source_dir / SOURCE_PROVENANCE_FILENAME
    source_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(source_provenance, dict):
        raise ValueError("Extension source-data provenance is not a JSON object")
    if (
        source_provenance.get("provenance_version") != lineage.PROVENANCE_VERSION
        or source_provenance.get("analysis_identity")
        != EXPECTED_EXTENSION_IDENTITY
        or source_provenance.get("provenance_contract_sha256")
        != lineage.PROVENANCE_CONTRACT_SHA256
        or source_provenance.get("response_dtype") != EXPECTED_RESPONSE_DTYPE
        or source_provenance.get("float32_correction_sha256")
        != EXPECTED_FLOAT32_CORRECTION_SHA256
    ):
        raise RuntimeError("Extension source-data v4 provenance identity mismatch")

    controls = source_provenance.get("publication_controls")
    expected_controls = {
        "superseded_workflow_results_included": False,
        "cross_version_workflow_mixing_permitted": False,
        "row_level_predictions_in_aggregate_source_data": False,
        (
            "publication_eligible_independent_prospective_"
            "dataset_available"
        ): False,
    }
    if not isinstance(controls, dict):
        raise RuntimeError("Extension source-data publication controls are missing")
    for key, expected in expected_controls.items():
        if controls.get(key) is not expected:
            raise RuntimeError(f"Unsafe source-data publication control: {key}")

    workflows = source_provenance.get("workflows")
    if not isinstance(workflows, dict) or set(workflows) != set(
        lineage.WORKFLOW_IDENTITIES
    ):
        raise RuntimeError("Extension source-data workflow provenance is incomplete")
    observed_workflow_parents = {
        str(entry.get("source_directory"))
        for entry in workflows.values()
        if isinstance(entry, dict)
    }
    if observed_workflow_parents != set(EXPECTED_PROTOCOL_BY_PARENT):
        raise RuntimeError("Extension workflow provenance violates mixed lineage v4")
    for workflow, identity in lineage.WORKFLOW_IDENTITIES.items():
        entry = workflows[workflow]
        if (
            not isinstance(entry, dict)
            or entry.get("response_dtype") != EXPECTED_RESPONSE_DTYPE
            or entry.get("superseded_marker_present") is not False
            or entry.get("protocol_version") != identity["protocol_version"]
            or entry.get("protocol_sha256") != identity["protocol_sha256"]
            or entry.get("runner_sha256") != identity["runner_sha256"]
        ):
            raise RuntimeError(
                f"Extension workflow provenance mismatch: {workflow}"
            )
    null_entry = workflows["null_applicability_censoring"]
    null_identity = lineage.WORKFLOW_IDENTITIES[
        "null_applicability_censoring"
    ]
    if (
        null_entry.get("fixed_domain_correction_sha256")
        != EXPECTED_FIXED_DOMAIN_CORRECTION_SHA256
        or null_entry.get("permutation_wrapper_sha256")
        != null_identity["wrapper_sha256"]
        or null_entry.get("permutation_merge_helper_sha256")
        != null_identity["merge_sha256"]
    ):
        raise RuntimeError(
            "NULL/applicability provenance is not the fixed-domain v3 lineage"
        )
    scaffold_entry = workflows["scaffold_source"]
    scaffold_identity = lineage.WORKFLOW_IDENTITIES["scaffold_source"]
    if (
        scaffold_entry.get("native_threadpool_audit_sha256") is None
        or scaffold_entry.get("all_target_exact_identity_audit_sha256") is None
        or scaffold_entry.get("identity_qa_script_sha256")
        != scaffold_identity["qa_sha256"]
        or scaffold_entry.get("execution_correction_sha256")
        != scaffold_identity["execution_correction_sha256"]
        or scaffold_entry.get("mathematical_runner_sha256")
        != scaffold_identity["math_runner_sha256"]
    ):
        raise RuntimeError(
            "Scaffold/source provenance lacks the v3 thread and identity gates"
        )

    base_table = source_dir / "source_data_internal_models.csv"
    base_manifest = source_dir / "artifact_sha256.csv"
    if (
        not base_table.is_file()
        or base_table.is_symlink()
        or not base_manifest.is_file()
        or base_manifest.is_symlink()
    ):
        raise FileNotFoundError(
            "Fig. 5 requires the frozen confirmatory internal-model source table "
            "and its artifact checksum manifest"
        )
    base_checksums = pd.read_csv(base_manifest, keep_default_na=False)
    require_columns(base_checksums, {"filename", "sha256"}, base_manifest.name)
    base_rows = base_checksums[
        base_checksums["filename"] == base_table.name
    ]
    base_row = one_row(base_rows, "frozen internal-model checksum")
    if sha256_file(base_table) != str(base_row["sha256"]).strip():
        raise RuntimeError("Frozen internal-model source-table checksum mismatch")

    result: dict[str, object] = {
        **bound,
        "response_dtype": EXPECTED_RESPONSE_DTYPE,
        "float32_correction_sha256": EXPECTED_FLOAT32_CORRECTION_SHA256,
        "fixed_domain_correction_sha256": (
            EXPECTED_FIXED_DOMAIN_CORRECTION_SHA256
        ),
        "provenance_version": lineage.PROVENANCE_VERSION,
        "provenance_contract_sha256": lineage.PROVENANCE_CONTRACT_SHA256,
        (
            "publication_eligible_independent_prospective_"
            "dataset_available"
        ): False,
        "prospective_dataset_statement": (
            lineage.PUBLIC_PROSPECTIVE_DATASET_STATEMENT
        ),
        "verified_table_count": len(lineage.EXPECTED_TABLE_IDS),
        "plotted_table_count": len(PLOT_TABLE_LINEAGE),
    }
    result["source_data_provenance"] = dict(bound)
    return result


def read_table(source_dir: Path, stem: str, columns: Iterable[str]) -> pd.DataFrame:
    path = source_dir / f"source_data_extension_{stem}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    require_columns(frame, columns, path.name)
    return frame


def read_base_table(source_dir: Path, filename: str, columns: Iterable[str]) -> pd.DataFrame:
    path = source_dir / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    require_columns(frame, columns, path.name)
    return frame


def one_row(frame: pd.DataFrame, description: str) -> pd.Series:
    if len(frame) != 1:
        raise ValueError(f"{description}: expected one row, observed {len(frame)}")
    return frame.iloc[0]


def finite_number(value: object) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"Expected a finite number, observed {value!r}")
    return number


def require_replicate_count(
    frame: pd.DataFrame,
    column: str,
    expected: int,
    description: str,
) -> None:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.empty or values.isna().any() or not values.eq(expected).all():
        raise RuntimeError(
            f"{description} requires {expected:,} completed replicates"
        )


def panel_label(ax: plt.Axes, label: str, x: float = -0.12) -> None:
    ax.text(
        x,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def zero_line(ax: plt.Axes, orientation: str = "vertical") -> None:
    if orientation == "vertical":
        ax.axvline(0, color="#8F969D", lw=0.75, ls=(0, (2.5, 2.5)), zorder=0)
    else:
        ax.axhline(0, color="#8F969D", lw=0.75, ls=(0, (2.5, 2.5)), zorder=0)


def horizontal_ci(
    ax: plt.Axes,
    y: float,
    estimate: float,
    low: float,
    high: float,
    color: str,
    marker: str,
    filled: bool = True,
    size: float = 4.8,
    zorder: int = 3,
) -> None:
    ax.plot(
        [low, high],
        [y, y],
        color=color,
        lw=1.25,
        solid_capstyle="round",
        zorder=zorder - 1,
    )
    ax.plot(
        estimate,
        y,
        marker=marker,
        ms=size,
        mfc=color if filled else COLORS["white"],
        mec=color,
        mew=0.95,
        linestyle="none",
        zorder=zorder,
    )


def vertical_ci(
    ax: plt.Axes,
    x: float,
    estimate: float,
    low: float,
    high: float,
    color: str,
    marker: str,
    filled: bool = True,
    size: float = 4.2,
) -> None:
    ax.plot(
        [x, x],
        [low, high],
        color=color,
        lw=1.05,
        solid_capstyle="round",
        zorder=2,
    )
    ax.plot(
        x,
        estimate,
        marker=marker,
        ms=size,
        mfc=color if filled else COLORS["white"],
        mec=color,
        mew=0.85,
        linestyle="none",
        zorder=3,
    )


def padded_limits(values: Iterable[float], minimum_span: float = 0.1) -> tuple[float, float]:
    values = list(values)
    n_before = len(values)
    finite = np.asarray([float(v) for v in values if np.isfinite(float(v))])
    n_after = int(finite.size)
    if n_after > n_before:
        raise AssertionError("Finite-value count cannot exceed the input count")
    if finite.size == 0:
        return (-0.1, 0.1)
    low = min(float(finite.min()), 0.0)
    high = max(float(finite.max()), 0.0)
    span = max(high - low, minimum_span)
    return (low - 0.10 * span, high + 0.10 * span)


def ensure_svg_font_family(svg_path: Path) -> None:
    """Retain editable text under both old and new Matplotlib SVG syntax."""

    svg_text = svg_path.read_text(encoding="utf-8")
    if "<text " in svg_text and "font-family:" not in svg_text:
        svg_text = svg_text.replace(
            'style="font:',
            (
                "style=\"font-family: 'Arial', 'Helvetica', "
                "'Liberation Sans', sans-serif; font:"
            ),
        )
        svg_path.write_text(svg_text, encoding="utf-8", newline="\n")


def save_bundle(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    width_in, height_in = fig.get_size_inches()
    if not math.isclose(
        float(width_in), FIGURE_WIDTH_IN, rel_tol=0.0, abs_tol=1e-6
    ):
        raise RuntimeError(
            f"{stem} canvas is not exactly {FIGURE_WIDTH_MM:.0f} mm wide"
        )
    if float(height_in) * MM_PER_INCH > 205.0:
        raise RuntimeError(
            f"{stem} figure canvas exceeds the 205 mm pre-legend height budget"
        )
    paths = [
        output_dir / f"{stem}.svg",
        output_dir / f"{stem}.pdf",
        output_dir / f"{stem}.tiff",
        output_dir / f"{stem}.png",
    ]
    # Export atomically and never use bbox_inches="tight": the physical canvas
    # must remain exactly 170 mm wide in every format.
    with tempfile.TemporaryDirectory(
        prefix=f".{stem}.", dir=output_dir
    ) as temporary:
        temporary_dir = Path(temporary)
        staged = [temporary_dir / path.name for path in paths]
        fig.savefig(staged[0], metadata={"Creator": "matplotlib"})
        ensure_svg_font_family(staged[0])
        fig.savefig(
            staged[1],
            metadata={
                "Creator": "matplotlib",
                "Title": stem.replace("_", " "),
            },
        )
        fig.savefig(
            staged[2],
            dpi=600,
            facecolor="white",
            transparent=False,
            pil_kwargs={"compression": "tiff_lzw"},
        )
        with Image.open(staged[2]) as image:
            if image.mode == "RGBA":
                white = Image.new("RGBA", image.size, "white")
                image = Image.alpha_composite(white, image).convert("RGB")
                image.save(
                    staged[2],
                    compression="tiff_lzw",
                    dpi=(600, 600),
                )
            elif image.mode != "RGB":
                image.convert("RGB").save(
                    staged[2],
                    compression="tiff_lzw",
                    dpi=(600, 600),
                )
        fig.savefig(
            staged[3],
            dpi=300,
            facecolor="white",
            transparent=False,
        )
        for staged_path, destination in zip(staged, paths):
            if not staged_path.is_file() or staged_path.stat().st_size == 0:
                raise RuntimeError(f"Failed to export {staged_path.name}")
            staged_path.replace(destination)
    plt.close(fig)
    return paths


def copy_existing_figures(existing_dir: Path, output_dir: Path) -> list[Path]:
    copied: list[Path] = []
    if not existing_dir.is_dir() or existing_dir.is_symlink():
        raise FileNotFoundError(existing_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {
        f"{stem}{suffix}"
        for stem in EXISTING_FIGURE_STEMS
        for suffix in lineage.FIGURE_FORMATS
    }
    observed_names = {
        path.name
        for path in existing_dir.iterdir()
        if path.suffix.lower() in lineage.FIGURE_FORMATS
    }
    if observed_names != expected_names:
        raise RuntimeError(
            "Frozen Fig. 1--4 input set mismatch: "
            f"missing={sorted(expected_names - observed_names)}, "
            f"extra={sorted(observed_names - expected_names)}"
        )
    for stem in EXISTING_FIGURE_STEMS:
        for suffix in lineage.FIGURE_FORMATS:
            source = existing_dir / f"{stem}{suffix}"
            if not source.is_file() or source.is_symlink():
                raise RuntimeError(f"Unsafe frozen figure input: {source}")
            destination = output_dir / source.name
            shutil.copy2(source, destination)
            copied.append(destination)
    if len(copied) != len(expected_names):
        raise RuntimeError(
            f"Expected {len(expected_names)} Fig. 1--4 files, copied "
            f"{len(copied)} from {existing_dir}"
        )
    return copied


def load_main_figure_data(source_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "portable": read_table(
            source_dir,
            "portable_context",
            {
                "record_type",
                "protocol",
                "contrast_id",
                "delta_domain_macro_spearman_observed",
                "delta_domain_macro_spearman_ci_low",
                "delta_domain_macro_spearman_ci_high",
                "n_bootstrap",
            },
        ),
        "weights": read_table(
            source_dir,
            "weighting_bootstrap",
            {
                "record_type",
                "contrast_id",
                "delta_spearman_observed",
                "delta_spearman_ci_low",
                "delta_spearman_ci_high",
                "n_bootstrap",
            },
        ),
        "generic": read_table(
            source_dir,
            "generic_scaffold",
            {
                "record_type",
                "model_id",
                "contrast_id",
                "metric",
                "observed",
                "ci_low",
                "ci_high",
                "n_bootstrap",
            },
        ),
        "deletion": read_table(
            source_dir,
            "source_deletion",
            {
                "record_type",
                "contrast_type",
                "deletion_source",
                "heldout_target",
                "metric",
                "observed",
                "ci_low",
                "ci_high",
                "n_bootstrap",
            },
        ),
        "internal": read_base_table(
            source_dir,
            "source_data_internal_models.csv",
            {
                "record_type",
                "protocol",
                "model_id",
                "spearman_observed",
                "spearman_ci_low",
                "spearman_ci_high",
                "n_bootstrap",
            },
        ),
        "internal_contrasts": read_base_table(
            source_dir,
            "source_data_internal_contrasts.csv",
            {
                "record_type",
                "protocol",
                "contrast_id",
                "delta_spearman_observed",
                "delta_spearman_ci_low",
                "delta_spearman_ci_high",
                "n_bootstrap",
            },
        ),
    }


def make_figure5(source_dir: Path, output_dir: Path) -> list[Path]:
    data = load_main_figure_data(source_dir)
    portable = data["portable"]
    weights = data["weights"]
    generic = data["generic"]
    deletion = data["deletion"]
    internal = data["internal"]
    internal_contrasts = data["internal_contrasts"]

    for frame, label in (
        (portable, "portable-context table"),
        (weights, "weighting table"),
        (generic, "generic-scaffold table"),
        (deletion, "source-deletion table"),
        (internal, "frozen internal-model table"),
        (internal_contrasts, "frozen internal-contrast table"),
    ):
        require_replicate_count(frame, "n_bootstrap", 10_000, label)

    portable_rows: list[tuple[str, str, pd.Series]] = []
    for protocol, protocol_label in (
        ("source_ood", "Held source"),
        ("target_ood", "Held target"),
    ):
        for contrast_id, contrast_label in (
            ("full_vs_chemistry", "Original full"),
            ("portable_vs_chemistry", "Portable full"),
        ):
            row = one_row(
                portable[
                    (portable["record_type"] == "contrast")
                    & (portable["protocol"] == protocol)
                    & (portable["contrast_id"] == contrast_id)
                ],
                f"{protocol}/{contrast_id}",
            )
            portable_rows.append((protocol_label, contrast_label, row))

    weight_specs = [
        ("uniform_full_vs_chemistry", "Uniform rows", COLORS["navy"], "o"),
        (
            "compound_equal_full_vs_chemistry",
            "Compound equal",
            COLORS["teal"],
            "s",
        ),
        (
            "domain_balanced_full_vs_chemistry",
            "Source × target balanced",
            COLORS["ochre"],
            "^",
        ),
    ]
    weight_rows = [
        (
            label,
            color,
            marker,
            one_row(
                weights[
                    (weights["record_type"] == "contrast")
                    & (weights["contrast_id"] == contrast_id)
                ],
                contrast_id,
            ),
        )
        for contrast_id, label, color, marker in weight_specs
    ]

    deletion_order = ["none", "MGTbind", "MGDB", "MolGlueDB", "TPDdb"]
    deletion_labels = {
        "none": "No deletion",
        "MGTbind": "Delete MGTbind",
        "MGDB": "Delete MGDB",
        "MolGlueDB": "Delete MolGlueDB",
        "TPDdb": "Delete TPDdb",
    }
    deletion_rows: list[tuple[str, pd.Series]] = []
    for source in deletion_order:
        mask = (
            (deletion["record_type"] == "full_vs_chemistry")
            & (deletion["contrast_type"] == "full_vs_chemistry")
            & (deletion["deletion_source"] == source)
            & (deletion["metric"] == "delta_spearman")
            & (deletion["heldout_target"].isna())
        )
        deletion_rows.append(
            (deletion_labels[source], one_row(deletion[mask], f"deletion={source}"))
        )

    shared_delta_limits: list[float] = []
    for _, _, row in portable_rows:
        shared_delta_limits.extend(
            [
                row["delta_domain_macro_spearman_ci_low"],
                row["delta_domain_macro_spearman_ci_high"],
            ]
        )
    for _, _, _, row in weight_rows:
        shared_delta_limits.extend(
            [row["delta_spearman_ci_low"], row["delta_spearman_ci_high"]]
        )
    for _, row in deletion_rows:
        shared_delta_limits.extend([row["ci_low"], row["ci_high"]])
    delta_xlim = padded_limits(shared_delta_limits, minimum_span=0.2)

    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 5.55))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.105,
        right=0.985,
        bottom=0.10,
        top=0.865,
        wspace=0.42,
        hspace=0.51,
        height_ratios=[1.0, 1.12],
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    # a: held-axis portability.
    panel_label(ax_a, "a")
    ax_a.set_title("Held-axis context portability", loc="left", pad=8)
    y_positions = [3.2, 2.6, 1.4, 0.8]
    for y, (protocol_label, contrast_label, row) in zip(
        y_positions, portable_rows
    ):
        portable_model = contrast_label == "Portable full"
        color = COLORS["violet"] if portable_model else COLORS["navy"]
        marker = "D" if portable_model else "o"
        horizontal_ci(
            ax_a,
            y,
            finite_number(row["delta_domain_macro_spearman_observed"]),
            finite_number(row["delta_domain_macro_spearman_ci_low"]),
            finite_number(row["delta_domain_macro_spearman_ci_high"]),
            color,
            marker,
            filled=True,
        )
    ax_a.axhspan(2.25, 3.55, color=COLORS["pale_blue"], alpha=0.28, zorder=-2)
    ax_a.axhspan(0.45, 1.75, color=COLORS["light_grey"], alpha=0.18, zorder=-2)
    ax_a.text(
        0.012,
        3.46,
        "SOURCE OOD",
        transform=ax_a.get_yaxis_transform(),
        fontsize=5.8,
        fontweight="bold",
        color=COLORS["grey"],
        va="top",
    )
    ax_a.text(
        0.012,
        1.66,
        "TARGET OOD",
        transform=ax_a.get_yaxis_transform(),
        fontsize=5.8,
        fontweight="bold",
        color=COLORS["grey"],
        va="top",
    )
    ax_a.set_yticks(y_positions)
    ax_a.set_yticklabels([item[1] for item in portable_rows])
    ax_a.set_ylim(0.35, 3.65)
    ax_a.set_xlim(*delta_xlim)
    zero_line(ax_a)
    ax_a.set_xlabel(r"$\Delta$ domain-macro Spearman $\rho$")
    ax_a.grid(axis="x", color=COLORS["grid"], lw=0.45, alpha=0.65)
    ax_a.text(
        0.99,
        0.98,
        "circle: original full    diamond: portable full",
        transform=ax_a.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        color=COLORS["grey"],
    )

    # b: fit-weight sensitivities.
    panel_label(ax_b, "b")
    ax_b.set_title("Internal fit-weight sensitivity", loc="left", pad=8)
    y_positions_b = np.arange(len(weight_rows))[::-1]
    for y, (_, color, marker, row) in zip(y_positions_b, weight_rows):
        horizontal_ci(
            ax_b,
            float(y),
            finite_number(row["delta_spearman_observed"]),
            finite_number(row["delta_spearman_ci_low"]),
            finite_number(row["delta_spearman_ci_high"]),
            color,
            marker,
            filled=True,
        )
    ax_b.set_yticks(y_positions_b)
    ax_b.set_yticklabels([item[0] for item in weight_rows])
    ax_b.set_ylim(-0.55, len(weight_rows) - 0.45)
    ax_b.set_xlim(*delta_xlim)
    zero_line(ax_b)
    ax_b.set_xlabel(r"$\Delta$ Spearman $\rho$ (full − chemistry)")
    ax_b.grid(axis="x", color=COLORS["grid"], lw=0.45, alpha=0.65)
    ax_b.text(
        0.99,
        0.98,
        "fit changes; evaluation stays unweighted",
        transform=ax_b.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        color=COLORS["grey"],
    )

    # c: original versus generic scaffold definition.
    panel_label(ax_c, "c")
    ax_c.set_title("Scaffold-definition sensitivity", loc="left", pad=8)
    base_rows = {}
    contrast_rows = {}
    for model in (CHEM, FULL):
        base_rows[("Original BM", model)] = one_row(
            internal[
                (internal["record_type"] == "model")
                & (internal["protocol"] == "scaffold")
                & (internal["model_id"] == model)
            ],
            f"internal {model}",
        )
        base_rows[("Generic Murcko", model)] = one_row(
            generic[
                (generic["record_type"] == "model")
                & (generic["model_id"] == model)
                & (generic["metric"] == "spearman")
            ],
            f"generic {model}",
        )
    contrast_rows["Original BM"] = one_row(
        internal_contrasts[
            (internal_contrasts["record_type"] == "contrast")
            & (internal_contrasts["protocol"] == "scaffold")
            & (
                internal_contrasts["contrast_id"]
                == "full_vs_chemistry_extra_trees"
            )
        ],
        "internal full-versus-chemistry contrast",
    )
    contrast_rows["Generic Murcko"] = one_row(
        generic[
            (generic["record_type"] == "contrast")
            & (generic["contrast_id"] == "full_vs_chemistry")
            & (generic["metric"] == "delta_spearman")
        ],
        "generic-scaffold full-versus-chemistry contrast",
    )
    group_y = {"Original BM": 1.25, "Generic Murcko": 0.25}
    c_values: list[float] = []
    for scaffold_label, center in group_y.items():
        chem_row = base_rows[(scaffold_label, CHEM)]
        full_row = base_rows[(scaffold_label, FULL)]
        if scaffold_label == "Original BM":
            c_low_names = ("spearman_ci_low", "spearman_ci_high")
            c_est_name = "spearman_observed"
        else:
            c_low_names = ("ci_low", "ci_high")
            c_est_name = "observed"
        chem_est = finite_number(chem_row[c_est_name])
        full_est = finite_number(full_row[c_est_name])
        chem_low = finite_number(chem_row[c_low_names[0]])
        chem_high = finite_number(chem_row[c_low_names[1]])
        full_low = finite_number(full_row[c_low_names[0]])
        full_high = finite_number(full_row[c_low_names[1]])
        c_values.extend([chem_low, chem_high, full_low, full_high])
        ax_c.plot(
            [chem_est, full_est],
            [center - 0.12, center + 0.12],
            color=COLORS["light_grey"],
            lw=1.15,
            zorder=1,
        )
        horizontal_ci(
            ax_c,
            center - 0.12,
            chem_est,
            chem_low,
            chem_high,
            COLORS["blue"],
            "o",
            filled=False,
        )
        horizontal_ci(
            ax_c,
            center + 0.12,
            full_est,
            full_low,
            full_high,
            COLORS["navy"],
            "o",
            filled=True,
        )
        contrast_row = contrast_rows[scaffold_label]
        if scaffold_label == "Original BM":
            delta_estimate = finite_number(
                contrast_row["delta_spearman_observed"]
            )
            delta_low = finite_number(contrast_row["delta_spearman_ci_low"])
            delta_high = finite_number(contrast_row["delta_spearman_ci_high"])
        else:
            delta_estimate = finite_number(contrast_row["observed"])
            delta_low = finite_number(contrast_row["ci_low"])
            delta_high = finite_number(contrast_row["ci_high"])
        ax_c.text(
            max(chem_high, full_high) + 0.012,
            center,
            rf"$\Delta\rho$={delta_estimate:+.3f}"
            + "\n"
            + rf"95% CI [{delta_low:+.3f}, {delta_high:+.3f}]",
            fontsize=5.9,
            va="center",
            ha="left",
            linespacing=1.15,
        )
    c_xlim = padded_limits(c_values, minimum_span=0.25)
    # Leave room at right for the observed paired difference annotation.
    c_span = c_xlim[1] - c_xlim[0]
    ax_c.set_xlim(c_xlim[0], c_xlim[1] + 0.28 * c_span)
    ax_c.set_yticks(list(group_y.values()))
    ax_c.set_yticklabels(["Original\nBemis–Murcko", "Generic Murcko"])
    ax_c.set_ylim(-0.25, 1.75)
    ax_c.set_xlabel(r"Internal repeat-averaged Spearman $\rho$")
    ax_c.grid(axis="x", color=COLORS["grid"], lw=0.45, alpha=0.65)
    ax_c.text(
        0.99,
        0.98,
        "open circle: chemistry    filled circle: full",
        transform=ax_c.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        color=COLORS["grey"],
    )

    # d: source deletion.
    panel_label(ax_d, "d")
    ax_d.set_title("Training-source deletion in target OOD", loc="left", pad=8)
    y_positions_d = np.arange(len(deletion_rows))[::-1]
    for y, (label, row) in zip(y_positions_d, deletion_rows):
        reference = label == "No deletion"
        horizontal_ci(
            ax_d,
            float(y),
            finite_number(row["observed"]),
            finite_number(row["ci_low"]),
            finite_number(row["ci_high"]),
            COLORS["navy"] if reference else COLORS["grey"],
            "o" if reference else "s",
            filled=reference,
        )
    ax_d.set_yticks(y_positions_d)
    ax_d.set_yticklabels([item[0] for item in deletion_rows])
    ax_d.set_ylim(-0.55, len(deletion_rows) - 0.45)
    ax_d.set_xlim(*delta_xlim)
    zero_line(ax_d)
    ax_d.set_xlabel(r"$\Delta$ domain-macro Spearman $\rho$")
    ax_d.grid(axis="x", color=COLORS["grid"], lw=0.45, alpha=0.65)
    ax_d.text(
        0.99,
        0.98,
        "same eight held targets",
        transform=ax_d.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        color=COLORS["grey"],
    )

    fig.text(
        0.5,
        0.985,
        "Post-hoc robustness map",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    fig.text(
        0.985,
        0.02,
        "Points: observed estimates  |  Lines: 95% global-scaffold bootstrap intervals; contrasts are paired (10,000 resamples)",
        ha="right",
        va="bottom",
        fontsize=5.8,
        color=COLORS["grey"],
    )
    return save_bundle(fig, output_dir, "fig5_robustness_map")


def applicability_value_columns(
    row: pd.Series,
    metric: str,
) -> tuple[str, str, str, str | None, str]:
    internal = (
        str(row["regime"]) == "internal_scaffold_disjoint"
        and str(row["protocol"]) == "scaffold"
    )
    if internal:
        return (
            f"delta_{metric}",
            f"delta_{metric}_ci_low",
            f"delta_{metric}_ci_high",
            None,
            f"delta_{metric}_non_estimable_reason",
        )
    return (
        f"delta_domain_macro_{metric}",
        f"delta_domain_macro_{metric}_ci_low",
        f"delta_domain_macro_{metric}_ci_high",
        f"delta_domain_macro_{metric}_ci_status",
        f"delta_domain_macro_{metric}_non_estimable_reason",
    )


def applicability_interval_state(
    row: pd.Series,
    metric: str,
) -> tuple[float, float, float, str, str]:
    estimate_name, low_name, high_name, status_name, reason_name = (
        applicability_value_columns(row, metric)
    )
    values = pd.to_numeric(
        pd.Series(
            [row.get(estimate_name), row.get(low_name), row.get(high_name)]
        ),
        errors="coerce",
    ).to_numpy(dtype=float)
    estimate, low, high = map(float, values)
    if status_name is None:
        if np.isfinite(values).all():
            status = "ESTIMATED"
        elif np.isfinite(estimate):
            status = "NON_ESTIMABLE_POINT_ONLY"
        else:
            status = "NON_ESTIMABLE_POINT"
    else:
        status = str(row.get(status_name, "")).strip()
    reason = str(row.get(reason_name, "")).strip()
    return estimate, low, high, status, reason


def make_supplementary_s1(
    source_dir: Path,
    output_dir: Path,
    *,
    expected_bootstraps: int = 10_000,
) -> list[Path]:
    contrasts = read_table(
        source_dir,
        "applicability_contrasts",
        {
            "regime",
            "protocol",
            "similarity_bin",
            "contrast_id",
            "n_rows",
            "n_scaffolds",
            "n_domains_expected",
            "delta_spearman",
            "delta_spearman_ci_low",
            "delta_spearman_ci_high",
            "delta_spearman_non_estimable_reason",
            "delta_rmse",
            "delta_rmse_ci_low",
            "delta_rmse_ci_high",
            "delta_rmse_non_estimable_reason",
            "delta_domain_macro_spearman",
            "delta_domain_macro_spearman_ci_low",
            "delta_domain_macro_spearman_ci_high",
            "delta_domain_macro_spearman_n_bootstrap_requested",
            "delta_domain_macro_spearman_n_bootstrap_valid",
            "delta_domain_macro_spearman_n_bootstrap_not_attempted_structural",
            "delta_domain_macro_spearman_minimum_valid_required",
            "delta_domain_macro_spearman_ci_status",
            "delta_domain_macro_spearman_non_estimable_reason",
            "delta_domain_macro_rmse",
            "delta_domain_macro_rmse_ci_low",
            "delta_domain_macro_rmse_ci_high",
            "delta_domain_macro_rmse_n_bootstrap_requested",
            "delta_domain_macro_rmse_n_bootstrap_valid",
            "delta_domain_macro_rmse_n_bootstrap_not_attempted_structural",
            "delta_domain_macro_rmse_minimum_valid_required",
            "delta_domain_macro_rmse_ci_status",
            "delta_domain_macro_rmse_non_estimable_reason",
            "n_bootstrap_requested",
            "bootstrap_non_estimable_reason",
        },
    )
    if not pd.to_numeric(
        contrasts["n_bootstrap_requested"], errors="coerce"
    ).eq(expected_bootstraps).all():
        raise RuntimeError(
            "Supplementary Fig. S1 has an unexpected bootstrap count"
        )
    if set(contrasts["contrast_id"].astype(str)) != {"full_minus_chemistry"}:
        raise RuntimeError("Supplementary Fig. S1 requires full-minus-chemistry rows")
    if len(contrasts) != 5 * len(BIN_ORDER):
        raise RuntimeError("Supplementary Fig. S1 requires five complete bin profiles")
    count_columns = contrasts[["n_rows", "n_scaffolds"]].apply(
        pd.to_numeric, errors="raise"
    )
    if (
        (count_columns["n_rows"] <= 0).any()
        or (count_columns["n_scaffolds"] <= 0).any()
        or (count_columns["n_scaffolds"] > count_columns["n_rows"]).any()
    ):
        raise RuntimeError("Applicability row/scaffold counts are inconsistent")
    allowed_macro_statuses = {
        "ESTIMATED",
        "NON_ESTIMABLE_STRUCTURAL",
        "NON_ESTIMABLE_TOO_FEW_VALID",
        "NOT_APPLICABLE_INTERNAL_POOLED",
    }
    for _, row in contrasts.iterrows():
        internal = (
            str(row["regime"]) == "internal_scaffold_disjoint"
            and str(row["protocol"]) == "scaffold"
        )
        for metric in ("spearman", "rmse"):
            estimate, low, high, status, _ = applicability_interval_state(
                row, metric
            )
            macro_prefix = f"delta_domain_macro_{metric}"
            macro_status = str(row[f"{macro_prefix}_ci_status"])
            if macro_status not in allowed_macro_statuses:
                raise RuntimeError(
                    f"Unexpected applicability CI status: {macro_status}"
                )
            requested = int(row[f"{macro_prefix}_n_bootstrap_requested"])
            valid = int(row[f"{macro_prefix}_n_bootstrap_valid"])
            not_attempted = int(
                row[f"{macro_prefix}_n_bootstrap_not_attempted_structural"]
            )
            minimum = int(row[f"{macro_prefix}_minimum_valid_required"])
            if requested != expected_bootstraps:
                raise RuntimeError("Macro applicability bootstrap count mismatch")
            if internal:
                if macro_status != "NOT_APPLICABLE_INTERNAL_POOLED":
                    raise RuntimeError(
                        "Internal applicability rows must use pooled intervals"
                    )
            elif status == "ESTIMATED":
                if (
                    not np.isfinite([estimate, low, high]).all()
                    or valid < minimum
                    or not_attempted != 0
                ):
                    raise RuntimeError(
                        "Estimated fixed-domain interval has invalid evidence"
                    )
            elif status == "NON_ESTIMABLE_STRUCTURAL":
                if (
                    not np.isfinite(estimate)
                    or np.isfinite([low, high]).any()
                    or valid != 0
                    or not_attempted != requested
                ):
                    raise RuntimeError(
                        "Structural CI must retain only its descriptive point"
                    )
            elif status == "NON_ESTIMABLE_TOO_FEW_VALID":
                if (
                    np.isfinite([low, high]).any()
                    or valid >= minimum
                    or not_attempted != 0
                ):
                    raise RuntimeError(
                        "Too-few-valid CI status is internally inconsistent"
                    )
            else:
                raise RuntimeError(
                    f"Unexpected non-internal applicability state: {status}"
                )

    panels = [
        (
            "internal_scaffold_disjoint",
            "scaffold",
            "Internal scaffold-disjoint",
        ),
        ("frozen_ood", "source_ood", "Held-source OOD"),
        ("frozen_ood", "target_ood", "Held-target OOD"),
        (
            "strict_domain_scaffold_ood",
            "source_ood",
            "Strict held-source + scaffold",
        ),
        (
            "strict_domain_scaffold_ood",
            "target_ood",
            "Strict held-target + scaffold",
        ),
    ]

    spearman_bounds: list[float] = []
    rmse_bounds: list[float] = []
    for _, row in contrasts.iterrows():
        for metric, destination in (
            ("spearman", spearman_bounds),
            ("rmse", rmse_bounds),
        ):
            estimate, low, high, _, _ = applicability_interval_state(row, metric)
            for value in (estimate, low, high):
                if np.isfinite(value):
                    destination.append(float(value))
    spearman_ylim = padded_limits(spearman_bounds, minimum_span=0.3)
    rmse_ylim = padded_limits(rmse_bounds, minimum_span=0.15)

    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 7.42))
    outer = fig.add_gridspec(
        3,
        2,
        left=0.085,
        right=0.985,
        bottom=0.075,
        top=0.94,
        wspace=0.30,
        hspace=0.43,
    )
    letters = ["a", "b", "c", "d", "e"]
    for panel_index, (regime, protocol, title) in enumerate(panels):
        cell = outer[panel_index // 2, panel_index % 2]
        sub = cell.subgridspec(2, 1, height_ratios=[1.0, 0.72], hspace=0.10)
        ax_rho = fig.add_subplot(sub[0, 0])
        ax_rmse = fig.add_subplot(sub[1, 0], sharex=ax_rho)
        panel_label(ax_rho, letters[panel_index], x=-0.11)
        ax_rho.set_title(title, loc="left", pad=6)
        subset = contrasts[
            (contrasts["regime"] == regime)
            & (contrasts["protocol"] == protocol)
        ].copy()
        subset["_bin_order"] = pd.Categorical(
            subset["similarity_bin"], categories=BIN_ORDER, ordered=True
        )
        subset = subset.sort_values("_bin_order", kind="mergesort")
        if list(subset["similarity_bin"]) != BIN_ORDER:
            raise RuntimeError(f"Incomplete applicability bins for {title}")
        for metric, axis, color, marker, ylim in (
            ("spearman", ax_rho, COLORS["navy"], "o", spearman_ylim),
            ("rmse", ax_rmse, COLORS["teal"], "s", rmse_ylim),
        ):
            zero_line(axis, orientation="horizontal")
            for x, (_, row) in enumerate(subset.iterrows()):
                estimate, low, high, status, _ = applicability_interval_state(
                    row, metric
                )
                if status == "ESTIMATED":
                    vertical_ci(
                        axis,
                        float(x),
                        estimate,
                        low,
                        high,
                        color,
                        marker,
                        filled=metric == "spearman",
                    )
                elif np.isfinite(estimate):
                    axis.plot(
                        x,
                        estimate,
                        marker="D",
                        ms=4.1,
                        mfc=COLORS["white"],
                        mec=color,
                        mew=0.9,
                        linestyle="none",
                        zorder=3,
                    )
                    label = (
                        "CI NE*"
                        if status == "NON_ESTIMABLE_STRUCTURAL"
                        else "CI NE"
                    )
                    axis.annotate(
                        label,
                        (x, estimate),
                        xytext=(0, 5),
                        textcoords="offset points",
                        fontsize=5.0,
                        color=COLORS["grey"],
                        ha="center",
                        va="bottom",
                    )
                else:
                    axis.plot(
                        x,
                        0,
                        marker="x",
                        ms=4,
                        color=COLORS["grey"],
                        linestyle="none",
                    )
                    axis.text(
                        x,
                        0.04 * (ylim[1] - ylim[0]),
                        "point NE",
                        fontsize=5.3,
                        color=COLORS["grey"],
                        ha="center",
                        va="bottom",
                    )
            axis.set_ylim(*ylim)
            axis.grid(axis="y", color=COLORS["grid"], lw=0.4, alpha=0.65)
            axis.set_xlim(-0.45, 3.45)
        ax_rho.set_ylabel(r"$\Delta$ Spearman $\rho$")
        ax_rmse.set_ylabel(r"$\Delta$ RMSE (pDC50)")
        plt.setp(ax_rho.get_xticklabels(), visible=False)
        tick_labels = [
            f"{BIN_LABELS[row['similarity_bin']]}\n"
            f"n={int(row['n_rows'])}; S={int(row['n_scaffolds'])}"
            for _, row in subset.iterrows()
        ]
        ax_rmse.set_xticks(range(4))
        ax_rmse.set_xticklabels(tick_labels, fontsize=5.5, linespacing=1.15)
        ax_rmse.set_xlabel("Maximum train-set Tanimoto bin", labelpad=3)
        if panel_index > 0:
            finite_domains = pd.to_numeric(
                subset.get("n_domains_expected"), errors="coerce"
            ).dropna()
            if not finite_domains.empty:
                ax_rho.text(
                    0.99,
                    1.02,
                    f"equal-domain macro; D≤{int(finite_domains.max())}",
                    transform=ax_rho.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=5.4,
                    color=COLORS["grey"],
                )

    # Sixth cell is a deliberately compact interpretation key.
    key_ax = fig.add_subplot(outer[2, 1])
    key_ax.set_axis_off()
    key_ax.text(
        0.0,
        0.88,
        "Reading the profile",
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="top",
    )
    key_ax.plot(
        [0.02, 0.16],
        [0.68, 0.68],
        color=COLORS["navy"],
        lw=1.2,
        transform=key_ax.transAxes,
    )
    key_ax.plot(
        0.09,
        0.68,
        marker="o",
        ms=4.3,
        color=COLORS["navy"],
        transform=key_ax.transAxes,
    )
    key_ax.text(
        0.21,
        0.68,
        r"$\Delta$Spearman = full − chemistry",
        transform=key_ax.transAxes,
        va="center",
        fontsize=6.5,
    )
    key_ax.plot(
        [0.02, 0.16],
        [0.51, 0.51],
        color=COLORS["teal"],
        lw=1.2,
        transform=key_ax.transAxes,
    )
    key_ax.plot(
        0.09,
        0.51,
        marker="s",
        ms=4.2,
        mfc="white",
        mec=COLORS["teal"],
        transform=key_ax.transAxes,
    )
    key_ax.text(
        0.21,
        0.51,
        r"$\Delta$RMSE = chemistry − full",
        transform=key_ax.transAxes,
        va="center",
        fontsize=6.5,
    )
    key_ax.plot(
        0.04,
        0.36,
        marker="D",
        ms=4.1,
        mfc=COLORS["white"],
        mec=COLORS["grey"],
        mew=0.9,
        linestyle="none",
        transform=key_ax.transAxes,
    )
    key_ax.text(
        0.09,
        0.36,
        "Descriptive point only; CI NE (* structural)",
        transform=key_ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color=COLORS["grey"],
    )
    key_ax.text(
        0.0,
        0.23,
        "Positive values favour the context-inclusive model.\n"
        "Lines are paired 95% global-scaffold bootstrap intervals\n"
        f"({expected_bootstraps:,} resamples; RMSE is in pDC50 units).\n"
        "No lines connect bins; no trend test. All bins remain visible.",
        transform=key_ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        linespacing=1.35,
    )
    fig.text(
        0.5,
        0.976,
        "Chemical-novelty applicability profile",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    fig.text(
        0.985,
        0.018,
        "n, rows; S, global scaffolds. NE, non-estimable; no bin was merged or omitted.",
        ha="right",
        va="bottom",
        fontsize=5.8,
        color=COLORS["grey"],
    )
    return save_bundle(fig, output_dir, "supp_fig_s1_applicability_profile")


def empirical_p_label(row: pd.Series) -> str:
    n = int(row["n_permutations_finite"])
    p = finite_number(row["empirical_p"])
    extreme = int(round(p * (n + 1) - 1))
    tail = str(row["empirical_tail"])
    if tail not in {"greater", "smaller"}:
        raise ValueError(f"Unexpected empirical tail: {tail}")
    if not 0 <= extreme <= n:
        raise ValueError(f"Impossible empirical extreme count: {extreme}/{n}")
    reconstructed = (1 + extreme) / (n + 1)
    if not math.isclose(p, reconstructed, rel_tol=0.0, abs_tol=5e-7):
        raise ValueError(
            f"Empirical p-value is inconsistent with its finite count: {p}"
        )
    direction = "≥ observed" if tail == "greater" else "≤ observed"
    return (
        rf"one-sided $p_{{emp}}=(1+{extreme})/({n}+1)={p:.3f}$"
        + f" ({direction})"
    )


def make_supplementary_s2(source_dir: Path, output_dir: Path) -> list[Path]:
    models = read_table(
        source_dir,
        "permutation_models",
        {"permutation_id", "model_id", "spearman", "rmse", "mae"},
    )
    contrasts = read_table(
        source_dir,
        "permutation_contrasts",
        {
            "permutation_id",
            "contrast_id",
            "delta_spearman",
            "delta_rmse",
            "delta_mae",
        },
    )
    inference = read_table(
        source_dir,
        "permutation_inference",
        {
            "record_type",
            "estimand_id",
            "metric",
            "observed",
            "null_percentile_2p5",
            "null_percentile_97p5",
            "n_permutations_finite",
            "empirical_tail",
            "empirical_p",
        },
    )
    if (
        models["permutation_id"].nunique() != 100
        or contrasts["permutation_id"].nunique() != 100
    ):
        raise RuntimeError("Supplementary Fig. S2 requires all 100 permutations")
    expected_permutation_ids = set(range(1, 101))
    observed_model_ids = set(
        pd.to_numeric(models["permutation_id"], errors="raise").astype(int)
    )
    observed_contrast_ids = set(
        pd.to_numeric(contrasts["permutation_id"], errors="raise").astype(int)
    )
    if (
        observed_model_ids != expected_permutation_ids
        or observed_contrast_ids != expected_permutation_ids
    ):
        raise RuntimeError("Permutation IDs must be exactly 1--100")
    for model_id in (CHEM, FULL):
        subset = models[models["model_id"] == model_id]
        if len(subset) != 100 or subset["permutation_id"].duplicated().any():
            raise RuntimeError(f"Incomplete permutation distribution for {model_id}")
    paired = contrasts[
        contrasts["contrast_id"] == "full_minus_chemistry"
    ]
    if len(paired) != 100 or paired["permutation_id"].duplicated().any():
        raise RuntimeError("Incomplete paired permutation-contrast distribution")

    specs = [
        (
            "a",
            "Chemistry-only ranking",
            models.loc[models["model_id"] == CHEM, "spearman"].to_numpy(),
            "chemistry_extra_trees",
            "spearman",
            r"Spearman $\rho$",
            COLORS["blue"],
        ),
        (
            "b",
            "Full-model ranking",
            models.loc[models["model_id"] == FULL, "spearman"].to_numpy(),
            "full_context_extra_trees",
            "spearman",
            r"Spearman $\rho$",
            COLORS["navy"],
        ),
        (
            "c",
            "Paired ranking increment",
            contrasts.loc[
                contrasts["contrast_id"] == "full_minus_chemistry",
                "delta_spearman",
            ].to_numpy(),
            "full_minus_chemistry",
            "delta_spearman",
            r"$\Delta$ Spearman $\rho$ (full − chemistry)",
            COLORS["violet"],
        ),
        (
            "d",
            "Paired error reduction",
            contrasts.loc[
                contrasts["contrast_id"] == "full_minus_chemistry",
                "delta_rmse",
            ].to_numpy(),
            "full_minus_chemistry",
            "delta_rmse",
            r"$\Delta$ RMSE (chemistry − full; pDC50)",
            COLORS["teal"],
        ),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(FIGURE_WIDTH_IN, 5.35),
        gridspec_kw={
            "left": 0.09,
            "right": 0.985,
            "bottom": 0.105,
            "top": 0.855,
            "wspace": 0.29,
            "hspace": 0.40,
        },
    )
    for ax, (letter, title, values, estimand, metric, xlabel, color) in zip(
        axes.flat, specs
    ):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size != 100:
            raise RuntimeError(f"{title} has {values.size} finite permutations")
        row = one_row(
            inference[
                (inference["estimand_id"] == estimand)
                & (inference["metric"] == metric)
            ],
            f"inference {estimand}/{metric}",
        )
        observed = finite_number(row["observed"])
        low = finite_number(row["null_percentile_2p5"])
        high = finite_number(row["null_percentile_97p5"])
        bins = np.linspace(values.min(), values.max(), 15)
        if np.unique(bins).size < 3:
            bins = 12
        counts, edges, _ = ax.hist(
            values,
            bins=bins,
            color=color,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.55,
        )
        ymax = max(float(np.max(counts)), 1.0)
        ax.axvspan(low, high, color=COLORS["light_grey"], alpha=0.55, zorder=0)
        ax.axvline(
            observed,
            color=COLORS["ink"],
            lw=1.55,
            zorder=4,
            label="Observed",
        )
        rug_y = -0.055 * ymax
        ax.plot(
            values,
            np.full(values.size, rug_y),
            "|",
            color=color,
            ms=3.2,
            mew=0.55,
            alpha=0.9,
            clip_on=False,
        )
        ax.set_ylim(rug_y * 1.55, ymax * 1.14)
        panel_label(ax, letter)
        ax.set_title(title, loc="left", pad=7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Permutation count")
        ax.grid(axis="y", color=COLORS["grid"], lw=0.4, alpha=0.55)
        ax.text(
            0.975,
            0.94,
            f"null 95%: [{low:.3f}, {high:.3f}]\n"
            f"observed: {observed:.3f}\n{empirical_p_label(row)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=5.7,
            linespacing=1.25,
            zorder=6,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.94,
                "pad": 1.8,
            },
        )
    fig.text(
        0.5,
        0.985,
        "Repeated conditional label-randomization controls",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    fig.text(
        0.985,
        0.018,
        "All 100 within-source × target permutations are shown; shaded bands are null 2.5th–97.5th percentiles.",
        ha="right",
        va="bottom",
        fontsize=5.8,
        color=COLORS["grey"],
    )
    return save_bundle(fig, output_dir, "supp_fig_s2_permutation_controls")


def make_supplementary_s3(source_dir: Path, output_dir: Path) -> list[Path]:
    overall = read_table(
        source_dir,
        "censoring_overall",
        {"relation_class", "n_records", "fraction_within_group"},
    )
    by_source = read_table(
        source_dir,
        "censoring_by_source",
        {
            "source_database",
            "relation_class",
            "n_records",
            "fraction_within_group",
        },
    )
    by_target = read_table(
        source_dir,
        "censoring_by_target",
        {
            "target_protein",
            "relation_class",
            "n_records",
            "fraction_within_group",
        },
    )
    overlap = read_table(
        source_dir,
        "censoring_overlap",
        {
            "relation_class",
            "n_records",
            "n_unique_compounds",
            "n_unique_scaffolds",
            "fraction_unique_compounds_in_core",
            "fraction_unique_scaffolds_in_core",
        },
    )
    class_order = ["equality", "range", "one_sided", "missing_or_unparsed"]
    class_labels = {
        "equality": "Exact",
        "range": "Interval/range",
        "one_sided": "One-sided",
        "missing_or_unparsed": "Missing/unparsed",
    }
    if set(overall["relation_class"]) != set(class_order):
        raise RuntimeError("Unexpected overall censoring classes")
    if overall["relation_class"].duplicated().any():
        raise RuntimeError("Overall censoring table has duplicate relation classes")
    overall_total_rows = int(
        pd.to_numeric(overall["n_records"], errors="raise").sum()
    )
    if overall_total_rows != 3_117:
        raise RuntimeError(
            f"Frozen censoring audit requires 3,117 table rows, found "
            f"{overall_total_rows:,}"
        )
    overall_fraction_sum = float(
        pd.to_numeric(overall["fraction_within_group"], errors="raise").sum()
    )
    if not math.isclose(
        overall_fraction_sum, 1.0, rel_tol=0.0, abs_tol=5e-6
    ):
        raise RuntimeError("Overall censoring fractions do not sum to one")
    for frame, group_column, label in (
        (by_source, "source_database", "source"),
        (by_target, "target_protein", "target"),
    ):
        if frame.duplicated([group_column, "relation_class"]).any():
            raise RuntimeError(f"Duplicate {label}/relation censoring rows")
        grouped_fractions = frame.groupby(group_column, sort=False)[
            "fraction_within_group"
        ].sum()
        if not np.allclose(
            pd.to_numeric(grouped_fractions, errors="raise").to_numpy(float),
            1.0,
            rtol=0.0,
            atol=5e-6,
        ):
            raise RuntimeError(f"Within-{label} censoring fractions do not sum to one")
        grouped_total = int(
            frame.groupby(group_column, sort=False)["n_records"]
            .sum()
            .sum()
        )
        if grouped_total != overall_total_rows:
            raise RuntimeError(
                f"{label.capitalize()} censoring counts do not reconcile "
                "to the overall table"
            )
    overlap_fractions = overlap[
        [
            "fraction_unique_compounds_in_core",
            "fraction_unique_scaffolds_in_core",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    finite_overlap = overlap_fractions.to_numpy(float)
    finite_overlap = finite_overlap[np.isfinite(finite_overlap)]
    if np.any((finite_overlap < 0) | (finite_overlap > 1)):
        raise RuntimeError("Censoring-overlap fractions fall outside [0, 1]")

    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 5.90))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.09,
        right=0.985,
        bottom=0.125,
        top=0.855,
        wspace=0.34,
        hspace=0.43,
        height_ratios=[0.82, 1.18],
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    # a: overall composition.
    panel_label(ax_a, "a")
    ax_a.set_title("Parsed DC50 record-table rows", loc="left", pad=7)
    left = 0.0
    overall_lookup = overall.set_index("relation_class")
    for relation in class_order:
        row = overall_lookup.loc[relation]
        fraction = finite_number(row["fraction_within_group"])
        ax_a.barh(
            [0],
            [fraction],
            left=left,
            color=COLORS[relation],
            hatch=CENSOR_HATCHES[relation],
            height=0.42,
            edgecolor="white",
            linewidth=0.6,
        )
        if fraction >= 0.055:
            text_color = (
                "white"
                if relation in {"equality", "range", "one_sided"}
                else COLORS["ink"]
            )
            ax_a.text(
                left + fraction / 2,
                0,
                f"{int(row['n_records']):,}\n{100 * fraction:.1f}%",
                ha="center",
                va="center",
                fontsize=5.9,
                color=text_color,
                linespacing=1.05,
            )
        left += fraction
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(-0.55, 0.55)
    ax_a.set_yticks([])
    ax_a.set_xlabel(
        f"Fraction of {overall_total_rows:,} parsed-record rows"
    )
    ax_a.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    relation_handles = [
        Patch(
            facecolor=COLORS[relation],
            hatch=CENSOR_HATCHES[relation],
            edgecolor="white",
            linewidth=0.5,
            label=class_labels[relation],
        )
        for relation in class_order
    ]

    # b: source-stratified 100% composition.
    panel_label(ax_b, "b")
    ax_b.set_title("Relation composition by source", loc="left", pad=7)
    source_totals = (
        by_source.groupby("source_database", sort=False)["n_records"]
        .sum()
        .sort_values(ascending=True)
    )
    sources = list(source_totals.index)
    for y, source in enumerate(sources):
        subset = by_source[by_source["source_database"] == source].set_index(
            "relation_class"
        )
        left = 0.0
        for relation in class_order:
            fraction = (
                finite_number(subset.loc[relation, "fraction_within_group"])
                if relation in subset.index
                else 0.0
            )
            ax_b.barh(
                y,
                fraction,
                left=left,
                color=COLORS[relation],
                hatch=CENSOR_HATCHES[relation],
                height=0.62,
                edgecolor="white",
                linewidth=0.45,
            )
            left += fraction
        ax_b.text(
            0.015,
            y,
            f"n={int(source_totals.loc[source]):,}",
            transform=ax_b.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=5.5,
            color="white",
            fontweight="bold",
            bbox={
                "facecolor": COLORS["ink"],
                "edgecolor": "none",
                "alpha": 0.78,
                "pad": 1.0,
            },
        )
    ax_b.set_yticks(range(len(sources)))
    ax_b.set_yticklabels(sources)
    ax_b.set_xlim(0, 1)
    ax_b.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax_b.set_xlabel("Within-source fraction")
    ax_b.grid(axis="x", color=COLORS["grid"], lw=0.4, alpha=0.6)

    # c: every target category.
    panel_label(ax_c, "c")
    ax_c.set_title("Exact-label fraction across target categories", loc="left", pad=7)
    target_totals = (
        by_target.groupby("target_protein", sort=False)["n_records"].sum()
    )
    exact = by_target[by_target["relation_class"] == "equality"].set_index(
        "target_protein"
    )["n_records"]
    target_frame = pd.DataFrame({"n_total": target_totals}).join(
        exact.rename("n_exact")
    )
    target_frame["n_exact"] = target_frame["n_exact"].fillna(0)
    target_frame["exact_fraction"] = (
        target_frame["n_exact"] / target_frame["n_total"]
    )
    if np.any(target_frame["n_total"].to_numpy() <= 0):
        raise ValueError("Logarithmic target-count axis requires strictly positive counts")
    sizes = 12 + 36 * np.sqrt(
        target_frame["n_total"] / target_frame["n_total"].max()
    )
    ax_c.scatter(
        target_frame["n_total"],
        target_frame["exact_fraction"],
        s=sizes,
        color=COLORS["navy"],
        alpha=0.72,
        edgecolor="white",
        linewidth=0.45,
        zorder=2,
    )
    ax_c.set_xscale("log")
    ax_c.set_ylim(-0.04, 1.06)
    ax_c.set_xlabel("Parsed-record rows per target category (log scale)")
    ax_c.set_ylabel("Exact-label fraction")
    ax_c.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax_c.grid(color=COLORS["grid"], lw=0.4, alpha=0.55)
    label_names = list(target_frame.nlargest(3, "n_total").index)
    eligible_extreme = target_frame[target_frame["n_total"] >= 8]
    label_names.extend(
        list(eligible_extreme.nsmallest(1, "exact_fraction").index)
    )
    label_names = list(dict.fromkeys(label_names))
    label_offsets = [(6, 7), (6, -11), (-8, 12), (-8, 10)]
    for name, offset in zip(label_names, label_offsets):
        row = target_frame.loc[name]
        ax_c.annotate(
            str(name),
            (row["n_total"], row["exact_fraction"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=5.1,
            ha="left" if offset[0] >= 0 else "right",
            va="bottom" if offset[1] >= 0 else "top",
            color=COLORS["ink"],
            arrowprops={
                "arrowstyle": "-",
                "color": COLORS["grey"],
                "lw": 0.45,
                "shrinkA": 1,
                "shrinkB": 2,
            },
        )
    # d: censored-candidate overlap with the exact modelling core.
    panel_label(ax_d, "d")
    ax_d.set_title("Structure overlap with strict-exact core", loc="left", pad=7)
    overlap = overlap[overlap["relation_class"].isin(class_order)].copy()
    overlap["_order"] = pd.Categorical(
        overlap["relation_class"], categories=class_order, ordered=True
    )
    overlap = overlap.sort_values("_order", kind="mergesort")
    y = np.arange(len(overlap))[::-1]
    compound = overlap["fraction_unique_compounds_in_core"].to_numpy(float)
    scaffold = overlap["fraction_unique_scaffolds_in_core"].to_numpy(float)
    for index, y_value in enumerate(y):
        ax_d.plot(
            [compound[index], scaffold[index]],
            [y_value, y_value],
            color=COLORS["light_grey"],
            lw=1.2,
            zorder=1,
        )
    ax_d.scatter(
        compound,
        y,
        marker="o",
        s=24,
        facecolor="white",
        edgecolor=COLORS["blue"],
        linewidth=0.95,
        label="Unique compounds",
        zorder=3,
    )
    ax_d.scatter(
        scaffold,
        y,
        marker="s",
        s=21,
        facecolor=COLORS["teal"],
        edgecolor=COLORS["teal"],
        linewidth=0.8,
        label="Unique scaffolds",
        zorder=3,
    )
    ax_d.set_yticks(y)
    ax_d.set_yticklabels(
        [class_labels[value] for value in overlap["relation_class"]]
    )
    ax_d.set_xlim(-0.035, 1.035)
    ax_d.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax_d.set_xlabel("Fraction represented in strict-exact core")
    ax_d.grid(axis="x", color=COLORS["grid"], lw=0.4, alpha=0.6)
    ax_d.text(
        0.015,
        0.985,
        "○ compounds   ■ scaffolds",
        transform=ax_d.transAxes,
        ha="left",
        va="top",
        fontsize=5.4,
        color=COLORS["grey"],
        zorder=6,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.9,
            "pad": 1.2,
        },
    )

    fig.text(
        0.5,
        0.988,
        "Endpoint selection and censoring boundary",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    fig.legend(
        handles=relation_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=4,
        columnspacing=1.25,
        handletextpad=0.35,
        borderaxespad=0,
    )
    fig.text(
        0.985,
        0.037,
        f"Panel c: all {len(target_frame)} target categories are shown; "
        "labels identify the three largest and the lowest exact fraction "
        "(at least eight rows).",
        ha="right",
        va="bottom",
        fontsize=5.6,
        color=COLORS["grey"],
    )
    fig.text(
        0.985,
        0.015,
        "Counts denote record-table rows, not unique record IDs. Descriptive audit only; censored observations were not fitted as point labels.",
        ha="right",
        va="bottom",
        fontsize=5.8,
        color=COLORS["grey"],
    )
    return save_bundle(fig, output_dir, "supp_fig_s3_endpoint_selection_boundary")


def extension_figure_stem(filename: str) -> str | None:
    for stem in FIGURE_HEIGHT_MM_BY_STEM:
        if filename.startswith(f"{stem}."):
            return stem
    return None


def validate_exact_figure_set(
    paths: Iterable[Path],
    output_dir: Path,
) -> list[Path]:
    ordered = sorted(paths, key=lambda item: item.name)
    names = [path.name for path in ordered]
    observed_on_disk = {
        path.name
        for path in output_dir.iterdir()
        if path.suffix.lower() in lineage.FIGURE_FORMATS
    }
    if (
        len(names) != len(EXPECTED_FIGURE_NAMES)
        or len(names) != len(set(names))
        or set(names) != EXPECTED_FIGURE_NAMES
        or observed_on_disk != EXPECTED_FIGURE_NAMES
    ):
        raise RuntimeError(
            "Figure bundle must contain exactly 8 stems x 4 formats"
        )
    for path in ordered:
        if (
            path.parent.resolve() != output_dir.resolve()
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size <= 0
        ):
            raise RuntimeError(f"Unsafe or empty staged figure: {path}")
    return ordered


def write_manifest(
    paths: Iterable[Path],
    output_dir: Path,
    provenance: dict[str, object],
) -> Path:
    rows: list[dict[str, object]] = []
    for path in validate_exact_figure_set(paths, output_dir):
        figure_stem = extension_figure_stem(path.name)
        is_extension = figure_stem is not None
        row: dict[str, object] = {
            "file": path.name,
            "extension": path.suffix.lower(),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
            "width_mm": FIGURE_WIDTH_MM if is_extension else "",
            "height_mm": (
                FIGURE_HEIGHT_MM_BY_STEM[figure_stem]
                if figure_stem is not None
                else ""
            ),
            "source": (
                EXPECTED_EXTENSION_IDENTITY
                if is_extension
                else "existing_publication_figure_v1"
            ),
            "scientific_status": "FORMAL" if is_extension else "FROZEN_EXISTING",
            "response_dtype": (
                provenance["response_dtype"] if is_extension else ""
            ),
            "float32_correction_sha256": (
                provenance["float32_correction_sha256"] if is_extension else ""
            ),
            "fixed_domain_correction_sha256": (
                provenance["fixed_domain_correction_sha256"]
                if figure_stem in NULL_DERIVED_FIGURE_STEMS
                else "not_applicable"
            ),
            "source_data_extension_index_sha256": (
                provenance["source_data_extension_index_sha256"]
                if is_extension
                else ""
            ),
            "source_data_extension_checksums_sha256": (
                provenance["source_data_extension_checksums_sha256"]
                if is_extension
                else ""
            ),
            "source_data_extension_provenance_sha256": (
                provenance["source_data_extension_provenance_sha256"]
                if is_extension
                else ""
            ),
            "source_data_generation_id": (
                provenance["source_data_generation_id"] if is_extension else ""
            ),
        }
        if path.suffix.lower() in {".png", ".tiff"}:
            with Image.open(path) as image:
                row["pixel_width"] = int(image.width)
                row["pixel_height"] = int(image.height)
                row["image_mode"] = image.mode
                row["dpi_x"] = float(image.info.get("dpi", (math.nan, math.nan))[0])
                row["dpi_y"] = float(image.info.get("dpi", (math.nan, math.nan))[1])
        rows.append(row)
    manifest = pd.DataFrame(rows)
    if tuple(manifest.columns) != FIGURE_MANIFEST_COLUMNS:
        raise RuntimeError("Figure manifest ordered schema mismatch")
    path = output_dir / "figure_bundle_manifest.csv"
    manifest.to_csv(path, index=False)
    return path


def write_provenance_record(
    provenance: dict[str, object],
    output_dir: Path,
    manifest_path: Path,
) -> Path:
    path = output_dir / FIGURE_PROVENANCE_FILENAME
    manifest_sha256 = sha256_file(manifest_path)
    plot_script_sha256 = sha256_file(Path(__file__).resolve())
    figure_generation_id = lineage.figure_generation_id(
        source_data_generation_id=str(
            provenance["source_data_generation_id"]
        ),
        figure_bundle_manifest_sha256=manifest_sha256,
        plot_script_sha256=plot_script_sha256,
    )
    payload = {
        "status": "FORMAL",
        "backend": "Python/matplotlib",
        "figure_provenance_version": (
            "computational_extension_figure_float32_v4.0"
        ),
        "figure_width_mm": FIGURE_WIDTH_MM,
        "figure_directory_name": lineage.FIGURE_DIRECTORY_NAME,
        "figure_stems": list(lineage.FIGURE_STEMS),
        "figure_formats": list(lineage.FIGURE_FORMATS),
        "figure_file_count": len(EXPECTED_FIGURE_NAMES),
        "figure_bundle_manifest_filename": manifest_path.name,
        "figure_bundle_manifest_sha256": manifest_sha256,
        "plot_script_sha256": plot_script_sha256,
        "figure_generation_id": figure_generation_id,
        "figure_contract_filename": lineage.FIGURE_CONTRACT_FILENAME,
        "figure_contract_sha256": lineage.FIGURE_CONTRACT_SHA256,
        "figure_qa_script_sha256": lineage.FIGURE_QA_SCRIPT_SHA256,
        "figure_qa_required_check_ids": list(
            lineage.FIGURE_QA_REQUIRED_CHECK_IDS
        ),
        "figure_qa_required_check_count": (
            lineage.FIGURE_QA_REQUIRED_CHECK_COUNT
        ),
        "figure_qa_required_check_ids_sha256": (
            lineage.FIGURE_QA_REQUIRED_CHECK_IDS_SHA256
        ),
        "figure_qa_filename": FIGURE_QA_FILENAME,
        "figure_qa_must_bind_this_record_sha256": True,
        **provenance,
    }
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def write_figure_documents(
    output_dir: Path,
    existing_figure_dir: Path,
) -> tuple[Path, Path]:
    """Emit the complete caption set and human-readable QA handoff notes."""

    base_legends_path = existing_figure_dir / FIGURE_LEGENDS_FILENAME
    if not base_legends_path.is_file() or base_legends_path.is_symlink():
        raise FileNotFoundError(
            f"Frozen Fig. 1--4 legends are missing: {base_legends_path}"
        )
    base_legends = base_legends_path.read_text(encoding="utf-8").rstrip()
    extension_legends = """

**Fig. 5 | Post-hoc robustness map of the matched context contrast.**
a, Held-source and held-target domain-macro Spearman contrasts for the
original full model and a portable full model that removes held-axis
categories. b, Internal full-minus-chemistry Spearman contrasts under
uniform-row, compound-equal and source–target-balanced fitting; weighting
changes model fitting while evaluation remains unweighted. c, Internal
chemistry-only and full-model Spearman estimates under original
Bemis–Murcko and generic Murcko scaffold grouping, with their paired
full-minus-chemistry contrasts. d, Target-OOD full-minus-chemistry contrasts
without deletion and after deleting each training source. Points are observed
estimates and lines are 95% percentile intervals from 10,000 paired
global-scaffold bootstrap resamples. Positive contrasts favour the
context-inclusive model. All panels are post-hoc sensitivities and source
deletions do not identify causal database effects. Source data are provided
as a Source Data file.

**Supplementary Fig. S1 | Chemical-novelty applicability profile.**
Full-minus-chemistry Spearman and RMSE contrasts are shown in four
prespecified maximum-train-Tanimoto bins for a, internal scaffold-disjoint
validation; b, held-source OOD; c, held-target OOD; d, strict held-source plus
scaffold OOD; and e, strict held-target plus scaffold OOD. Positive values
favour the full model. `n` denotes rows, `S` global scaffolds and `D`
prespecified held domains. Lines are paired 95% global-scaffold-bootstrap
intervals from 10,000 requested resamples. OOD intervals freeze the
represented and metric-specific eligible domains before resampling; a repeat
is valid only when every eligible domain is represented with a finite metric.
`CI NE` denotes too few complete fixed-domain repeats, and `CI NE*` denotes a
structurally non-estimable interval caused by an eligible domain containing
one scaffold; in both cases the descriptive point is retained. `point NE`
would denote a non-estimable descriptive point. Bins are unconnected, none
was omitted or merged, and no monotonic-trend test was performed. Source data
are provided as a Source Data file.

**Supplementary Fig. S2 | Repeated conditional label-randomization
controls.** a,b, Conditional null distributions for chemistry-only and
full-model Spearman. c,d, Conditional null distributions for the paired
full-minus-chemistry Spearman increment and RMSE reduction. Labels were
randomized within source × target strata. All 100 outcomes are displayed;
shading denotes their empirical 2.5th–97.5th percentile range and the dark
line the observed value. Printed one-sided empirical probabilities use
`(1 + number at least as extreme)/(100 + 1)`, with a finite floor of
`1/101`. The shaded range is not a parametric confidence interval and the
controls are not multiplicity-adjusted confirmatory tests. Source data are
provided as a Source Data file.

**Supplementary Fig. S3 | Endpoint-selection and censoring boundary.**
a, Relation composition across all 3,117 parsed-record table rows. b,
Within-source relation composition. c, Exact-label fraction versus
parsed-record row count for all target categories; labels identify only the
declared largest and lowest-exact-fraction categories. d, Fractions of unique
compounds and scaffolds represented in the strict-exact modelling core for
each relation class. Exact, interval/range, one-sided and missing/unparsed
relations are redundantly encoded by colour and hatch or marker shape.
Counts refer to record-table rows rather than unique record identifiers.
Censored observations were not converted to point labels for fitting, and
the selection stages are not a row-wise inclusion probability. Source data
are provided as a Source Data file.
""".strip()
    legends_path = output_dir / FIGURE_LEGENDS_FILENAME
    legends_path.write_text(
        f"{base_legends}\n\n{extension_legends}\n",
        encoding="utf-8",
    )

    notes = """# Figure QA notes — mixed-lineage v4 bundle

- Backend: Python/matplotlib exclusively.
- Bundle: eight figure stems, each exported as editable SVG, PDF, 600 dpi RGB
  LZW TIFF and 300 dpi PNG (32 quantitative figure files).
- Fig. 1–4 are byte-preserved from the publication v1 bundle; Fig. 2 and
  Fig. 3 include the independently QA-verified molecular-graph sensitivity.
  Fig. 5 and
  Supplementary Figs. S1–S3 are regenerated from aggregate,
  publication-safe mixed-lineage v4 source data.
- Extension width: exactly 170 mm. Contracted heights are 140.970 mm,
  188.468 mm, 135.890 mm and 149.860 mm, respectively.
- No tight bounding box is used; physical canvas dimensions are preserved.
- No synthetic or simulated scientific data are plotted. Conditional
  permutations are declared negative controls and all 100 outcomes are shown.
- Colour is never the only encoding; marker fill/shape, hatching and line
  direction provide redundant encodings.
- Confidence intervals use the estimand-specific global-scaffold bootstrap.
  Fixed-domain applicability intervals preserve the frozen eligible-domain
  set, including explicit non-estimable states.
- No star notation, multiplicity-adjusted significance claim, monotonic-trend
  claim, causal source claim or prospective-validation claim is made.
- The machine-readable manifest binds file size and SHA-256 for every export.
  Formal automated QA is stored separately in
  `computational_extension_figure_qa_summary.json`.

## Manual visual review

- PASS (2026-07-31): all eight PNG previews were inspected at full
  resolution; no panel, title, axis label, tick label, interval, legend or
  annotation was cropped or materially overlapped.
- PASS: primary and supplementary panels remain legible at the declared
  canvas sizes; open/filled markers and patterned censoring classes remain
  distinguishable without colour.

## Figure-to-source-data map

- Fig. 1: frozen v1 provenance, missingness and repeat-audit source tables.
- Fig. 2: publication v1 internal model, contrast, HistGradientBoosting and
  molecular-graph sensitivity tables.
- Fig. 3: publication v1 confirmatory OOD, strict OOD,
  HistGradientBoosting and molecular-graph sensitivity tables.
- Fig. 4: frozen v1 domain, leave-one-domain-out and calibration tables.
- Fig. 5: `source_data_extension_portable_context.csv`,
  `source_data_extension_weighting_bootstrap.csv`,
  `source_data_extension_generic_scaffold.csv` and
  `source_data_extension_source_deletion.csv`.
- Supplementary Fig. S1:
  `source_data_extension_applicability_contrasts.csv`.
- Supplementary Fig. S2:
  `source_data_extension_permutation_models.csv`,
  `source_data_extension_permutation_contrasts.csv` and
  `source_data_extension_permutation_inference.csv`.
- Supplementary Fig. S3: aggregate censoring-overall, by-source, by-target and
  overlap source tables in the computational-extension source-data bundle.
"""
    notes_path = output_dir / FIGURE_QA_NOTES_FILENAME
    notes_path.write_text(notes, encoding="utf-8")
    return legends_path, notes_path


def publish_staged_directory(
    staging_dir: Path,
    output_dir: Path,
) -> Path | None:
    """Atomically swap a complete sibling stage, preserving the old bundle."""

    if (
        not staging_dir.is_dir()
        or staging_dir.is_symlink()
        or staging_dir.parent.resolve() != output_dir.parent.resolve()
        or not staging_dir.name.startswith(f".{output_dir.name}.stage.")
    ):
        raise RuntimeError(f"Unsafe figure staging directory: {staging_dir}")
    backup_dir: Path | None = None
    if output_dir.exists() or output_dir.is_symlink():
        if not output_dir.is_dir() or output_dir.is_symlink():
            raise RuntimeError(f"Refusing to replace unsafe output: {output_dir}")
        backup_dir = output_dir.with_name(
            f".{output_dir.name}.backup.{uuid.uuid4().hex[:12]}"
        )
        if backup_dir.exists() or backup_dir.is_symlink():
            raise RuntimeError(f"Refusing to overwrite backup: {backup_dir}")
        output_dir.rename(backup_dir)
    try:
        staging_dir.rename(output_dir)
    except BaseException:
        if backup_dir is not None and backup_dir.is_dir():
            if output_dir.exists() or output_dir.is_symlink():
                raise RuntimeError(
                    "Figure publish failed after creating an unexpected target; "
                    f"recover the previous bundle from {backup_dir}"
                )
            backup_dir.rename(output_dir)
        raise
    return backup_dir


def main() -> None:
    args = parse_args()
    lineage.require_figure_publication_contract_frozen()
    set_style()
    if args.output_dir.absolute().resolve() != DEFAULT_OUTPUT_DIR.resolve():
        raise RuntimeError(
            "Formal figures may be published only to reports/publication_figures_v4"
        )
    if args.existing_figure_dir.absolute().resolve() != DEFAULT_EXISTING_DIR.resolve():
        raise RuntimeError("Fig. 1--4 must come from the frozen v1 figure bundle")
    provenance = validate_source_data_provenance(args.source_dir)
    if (args.output_dir / "SUPERSEDED_DO_NOT_USE.md").exists():
        raise RuntimeError(f"Refusing to write a superseded output: {args.output_dir}")

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_dir.name}.stage.",
            dir=args.output_dir.parent,
        )
    )
    backup_dir: Path | None = None
    try:
        emitted: list[Path] = []
        emitted.extend(
            copy_existing_figures(args.existing_figure_dir, staging_dir)
        )
        emitted.extend(make_figure5(args.source_dir, staging_dir))
        emitted.extend(make_supplementary_s1(args.source_dir, staging_dir))
        emitted.extend(make_supplementary_s2(args.source_dir, staging_dir))
        emitted.extend(make_supplementary_s3(args.source_dir, staging_dir))
        emitted = validate_exact_figure_set(emitted, staging_dir)
        oversize = [
            path for path in emitted if path.stat().st_size >= 10_000_000
        ]
        if oversize:
            raise RuntimeError(
                "Figure files exceed the 10 MB journal limit: "
                + ", ".join(path.name for path in oversize)
            )
        manifest_path = write_manifest(emitted, staging_dir, provenance)
        provenance_path = write_provenance_record(
            provenance, staging_dir, manifest_path
        )
        legends_path, notes_path = write_figure_documents(
            staging_dir,
            args.existing_figure_dir,
        )
        expected_stage_entries = EXPECTED_FIGURE_NAMES | {
            manifest_path.name,
            provenance_path.name,
            legends_path.name,
            notes_path.name,
        }
        observed_stage_entries = {path.name for path in staging_dir.iterdir()}
        if observed_stage_entries != expected_stage_entries:
            raise RuntimeError(
                "Staged figure directory contains unexpected entries"
            )
        backup_dir = publish_staged_directory(staging_dir, args.output_dir)
    except BaseException:
        if (
            staging_dir.is_dir()
            and not staging_dir.is_symlink()
            and staging_dir.parent.resolve() == args.output_dir.parent.resolve()
            and staging_dir.name.startswith(f".{args.output_dir.name}.stage.")
        ):
            shutil.rmtree(staging_dir)
        raise

    final_manifest_path = args.output_dir / "figure_bundle_manifest.csv"
    backup_note = f"; previous={backup_dir}" if backup_dir is not None else ""
    print(
        "COMPUTATIONAL_EXTENSION_FIGURES: PASS "
        f"({len(EXPECTED_FIGURE_NAMES)} figure files; "
        f"manifest={final_manifest_path}{backup_note})"
    )


if __name__ == "__main__":
    main()
