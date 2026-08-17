#!/usr/bin/env python3
"""Render the four publication figures from frozen source-data tables."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROUTE_DIR = SCRIPT_DIR.parent
SOURCE_DIR = ROUTE_DIR / "reports" / "publication_validation_v1"
DEFAULT_OUTPUT = ROUTE_DIR / "reports" / "publication_figures_v1"

# A 5.77-inch canvas leaves room for tight-bounding-box labels while keeping
# the exported bundles at or below the journal's 170 mm full-page width.
JOURNAL_CANVAS_WIDTH_IN = 5.77
JOURNAL_SCALE = JOURNAL_CANVAS_WIDTH_IN / 7.2047

FULL = "full_context_extra_trees"
CHEM = "chemistry_extra_trees"
FULL_HGB = "full_context_hist_gradient_boosting"
CHEM_HGB = "chemistry_hist_gradient_boosting"

COLORS = {
    "full": "#173F5F",
    "chem": "#4C78A8",
    "context": "#8C8C8C",
    "hgb": "#2A9D8F",
    "positive": "#2E8B57",
    "negative": "#C44E52",
    "neutral": "#666666",
    "light": "#D9E2EA",
    "confirmatory": "#173F5F",
    "formal": "#8C8C8C",
    "posthoc": "#2A9D8F",
}

TARGET_OOD = {
    "VAV1",
    "CSNK1A1",
    "GSPT1",
    "WIZ",
    "CDK2",
    "CCNK+CDK12",
    "IKZF2",
    "IKZF1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def set_style() -> None:
    # Mandatory editable-text settings retained in this literal form so that
    # deterministic source preflight can verify them.
    # svg.fonttype='none'; pdf.fonttype=42
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['pdf.fonttype'] = 42
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [
        "Arial",
        "DejaVu Sans",
        "Liberation Sans",
    ]
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.labelsize"] = 7
    plt.rcParams["axes.titlesize"] = 8
    plt.rcParams["xtick.labelsize"] = 6.5
    plt.rcParams["ytick.labelsize"] = 6.5
    plt.rcParams["legend.fontsize"] = 6.5
    plt.rcParams["axes.linewidth"] = 0.7
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["legend.frameon"] = False


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def save_bundle(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(
        output_dir / f"{stem}.tiff",
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def errorbar_h(
    ax: plt.Axes,
    y: float,
    estimate: float,
    low: float,
    high: float,
    color: str,
    marker: str = "o",
    filled: bool = True,
) -> None:
    face = color if filled else "white"
    ax.plot([low, high], [y, y], color=color, lw=1.3, solid_capstyle="round")
    ax.plot(
        estimate,
        y,
        marker=marker,
        ms=4.5,
        mfc=face,
        mec=color,
        mew=1,
        linestyle="none",
        zorder=3,
    )


def add_reference(ax: plt.Axes, value: float = 0.0) -> None:
    ax.axvline(value, color="#999999", lw=0.8, ls="--", zorder=0)


def figure1(source_dir: Path, output_dir: Path) -> None:
    source = pd.read_csv(source_dir / "provenance_by_source.csv")
    target = pd.read_csv(source_dir / "provenance_by_target.csv")
    missing = pd.read_csv(source_dir / "data_missingness.csv")
    compound_repeat = pd.read_csv(
        source_dir / "compound_repeat_audit.csv"
    )
    exact_context_repeat = pd.read_csv(
        source_dir / "exact_context_repeat_audit.csv"
    )

    fig = plt.figure(
        figsize=(JOURNAL_CANVAS_WIDTH_IN, 6.2992 * JOURNAL_SCALE)
    )
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.08, 1.0],
        height_ratios=[0.82, 1.18, 0.9],
        wspace=0.34,
        hspace=0.53,
    )
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    ax_d = fig.add_subplot(grid[2, :])

    ax_a.set_axis_off()
    panel_label(ax_a, "a")
    stages = [
        (
            "Internal\nscaffold-disjoint",
            "5 × 5 nested CV\nno train/test scaffold overlap\nClaim: internal ranking",
            "#DCE8F2",
        ),
        (
            "Held-domain\n+ compound-cold",
            "4 sources; 8 exact targets\nno compound or domain overlap\nClaim: fixed-domain transfer",
            "#C8D8E8",
        ),
        (
            "Strict domain\n+ scaffold-cold",
            "same 12 test domains\nall test scaffolds purged\nClaim: post-hoc extrapolation",
            "#B4C8DD",
        ),
    ]
    x_positions = [0.04, 0.365, 0.69]
    for (title, subtitle, face), x in zip(stages, x_positions):
        box = FancyBboxPatch(
            (x, 0.23),
            0.265,
            0.55,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            transform=ax_a.transAxes,
            facecolor=face,
            edgecolor=COLORS["full"],
            lw=0.9,
        )
        ax_a.add_patch(box)
        ax_a.text(
            x + 0.1325,
            0.66,
            title,
            transform=ax_a.transAxes,
            ha="center",
            va="center",
            fontsize=7.1,
            fontweight="bold",
            linespacing=1.05,
        )
        ax_a.text(
            x + 0.1325,
            0.39,
            subtitle,
            transform=ax_a.transAxes,
            ha="center",
            va="center",
            fontsize=6.2,
            linespacing=1.25,
        )
    for left, right in zip(x_positions[:-1], x_positions[1:]):
        ax_a.add_patch(
            FancyArrowPatch(
                (left + 0.267, 0.505),
                (right - 0.008, 0.505),
                transform=ax_a.transAxes,
                arrowstyle="-|>",
                mutation_scale=10,
                color=COLORS["neutral"],
                lw=1,
            )
        )
    ax_a.text(
        0.5,
        0.05,
        "Increasing separation of chemistry and domain context → narrower, more defensible claims",
        transform=ax_a.transAxes,
        ha="center",
        color=COLORS["neutral"],
        fontsize=6.5,
    )

    panel_label(ax_b, "b")
    plot_source = source.sort_values("n_rows", ascending=True)
    y = np.arange(len(plot_source))
    ax_b.barh(
        y,
        plot_source["n_rows"],
        color=COLORS["light"],
        edgecolor=COLORS["full"],
        lw=0.6,
        label="Rows",
    )
    ax_b.scatter(
        plot_source["n_unique_compounds"],
        y,
        color=COLORS["chem"],
        s=14,
        label="Unique compounds",
        zorder=3,
    )
    ax_b.scatter(
        plot_source["n_scaffolds"],
        y,
        facecolor="white",
        edgecolor=COLORS["full"],
        s=14,
        label="Scaffolds",
        zorder=3,
    )
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(plot_source["source_database"])
    ax_b.set_xlabel("Count")
    ax_b.set_title("Multi-source composition", loc="left")
    ax_b.legend(
        ncol=1,
        loc="lower right",
        handlelength=1.2,
        labelspacing=0.25,
    )

    panel_label(ax_c, "c")
    target_sorted = target.sort_values("n_rows", ascending=False).reset_index(
        drop=True
    )
    if (target_sorted["n_rows"] <= 0).any():
        raise ValueError("Log-scale target counts must be strictly positive")
    colors = [
        COLORS["full"] if value in TARGET_OOD else "#C9C9C9"
        for value in target_sorted["target_protein"]
    ]
    ax_c.bar(
        np.arange(len(target_sorted)),
        target_sorted["n_rows"],
        color=colors,
        width=0.86,
        linewidth=0,
    )
    ax_c.set_yscale("log")
    ax_c.set_xlabel("Modeling target categories ranked by sample size")
    ax_c.set_ylabel("Rows (log scale)")
    ax_c.set_title("Target imbalance and OOD coverage", loc="left")
    ax_c.text(
        0.98,
        0.93,
        f"{len(TARGET_OOD)}/{len(target_sorted)} modeling categories\nentered frozen target OOD",
        transform=ax_c.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color=COLORS["full"],
    )
    selected_missing = missing[
        missing["column"].isin(
            [
                "activity_time",
                "source_url",
                "target_uniprot",
                "recruiting_protein_uniprot",
                "assay_method",
            ]
        )
    ].copy()
    selected_missing = selected_missing.sort_values(
        "missing_fraction", ascending=True
    )
    y = np.arange(len(selected_missing))
    ax_d.barh(
        y,
        selected_missing["missing_fraction"] * 100,
        color=COLORS["light"],
        edgecolor=COLORS["full"],
        lw=0.6,
    )
    ax_d.set_yticks(y)
    ax_d.set_yticklabels(
        [
            value.replace("_", " ")
            for value in selected_missing["column"]
        ]
    )
    ax_d.set_xlabel("Missing observations (%)")
    ax_d.set_xlim(0, 100)
    ax_d.set_title("Missingness and repeat structure", loc="left")
    ax_d.grid(axis="x", color="#E6E6E6", lw=0.5)
    panel_label(ax_d, "d")
    repeated_compounds = int((compound_repeat["n_rows"] > 1).sum())
    repeated_rows = int(
        compound_repeat.loc[
            compound_repeat["n_rows"] > 1, "n_rows"
        ].sum()
    )
    wide_range = int((compound_repeat["pDC50_range"] >= 1).sum())
    repeated_contexts = int(
        (exact_context_repeat["n_rows"] > 1).sum()
    )
    ax_d.text(
        0.99,
        0.04,
        (
            f"{repeated_compounds}/1,137 compounds repeated "
            f"({repeated_rows}/1,560 rows); "
            f"{wide_range} span ≥1 pDC50 unit\n"
            f"{repeated_contexts}/{len(exact_context_repeat):,} "
            "exact-context groups repeated"
        ),
        transform=ax_d.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color=COLORS["neutral"],
    )
    save_bundle(fig, output_dir, "fig1_validation_gradient")


def figure2(source_dir: Path, output_dir: Path) -> None:
    models = pd.read_csv(source_dir / "source_data_internal_models.csv")
    contrasts = pd.read_csv(source_dir / "source_data_internal_contrasts.csv")
    hgb = pd.read_csv(
        source_dir / "source_data_hgb_model_family_sensitivity.csv"
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(JOURNAL_CANVAS_WIDTH_IN, 5.1969 * JOURNAL_SCALE),
        gridspec_kw={"wspace": 0.52, "hspace": 0.48},
    )
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    labels = [
        "Scaffold · full",
        "Scaffold · chemistry",
        "Compound · full",
        "Compound · chemistry",
    ]
    selections = [
        ("scaffold", FULL),
        ("scaffold", CHEM),
        ("compound", FULL),
        ("compound", CHEM),
    ]
    for axis, metric, low, high, xlabel in (
        (
            ax_a,
            "spearman_observed",
            "spearman_ci_low",
            "spearman_ci_high",
            "Spearman ρ",
        ),
        (
            ax_b,
            "rmse_observed",
            "rmse_ci_low",
            "rmse_ci_high",
            "RMSE (pDC50 units)",
        ),
    ):
        for y, (protocol, model_id) in enumerate(selections[::-1]):
            row = models[
                (models["protocol"] == protocol)
                & (models["model_id"] == model_id)
            ].iloc[0]
            color = COLORS["full"] if model_id == FULL else COLORS["chem"]
            errorbar_h(
                axis,
                y,
                row[metric],
                row[low],
                row[high],
                color,
                filled=model_id == FULL,
            )
        axis.set_yticks(np.arange(4))
        axis.set_yticklabels(labels[::-1])
        axis.set_xlabel(xlabel)
        axis.grid(axis="x", color="#E6E6E6", lw=0.5)
    ax_a.set_title("Internal rank ordering", loc="left")
    ax_b.set_title("Internal absolute error", loc="left")
    panel_label(ax_a, "a")
    panel_label(ax_b, "b")

    scaffold_contrasts = contrasts[
        contrasts["protocol"].eq("scaffold")
        & contrasts["contrast_id"].isin(
            [
                "full_vs_chemistry_extra_trees",
                "full_vs_context_extra_trees",
                "chemistry_extra_trees_vs_morgan_knn",
            ]
        )
    ].copy()
    label_map = {
        "full_vs_chemistry_extra_trees": "Full − chem (ET)",
        "full_vs_context_extra_trees": "Full − context (ET)",
        "chemistry_extra_trees_vs_morgan_knn": "Chem ET − Morgan kNN",
    }
    rows = []
    for row in scaffold_contrasts.to_dict("records"):
        rows.append(
            {
                "label": label_map[row["contrast_id"]],
                "rho": row["delta_spearman_observed"],
                "rho_low": row["delta_spearman_ci_low"],
                "rho_high": row["delta_spearman_ci_high"],
                "rmse": row["delta_rmse_observed"],
                "rmse_low": row["delta_rmse_ci_low"],
                "rmse_high": row["delta_rmse_ci_high"],
                "identity": row["evidence_identity"],
            }
        )
    hgb_internal = hgb[
        (hgb["record_type"] == "contrast")
        & (hgb["split_regime"] == "internal")
    ].iloc[0]
    rows.append(
        {
            "label": "Full − chem (HGB)",
            "rho": hgb_internal["delta_spearman_observed"],
            "rho_low": hgb_internal["delta_spearman_ci_low"],
            "rho_high": hgb_internal["delta_spearman_ci_high"],
            "rmse": hgb_internal["delta_rmse_observed"],
            "rmse_low": hgb_internal["delta_rmse_ci_low"],
            "rmse_high": hgb_internal["delta_rmse_ci_high"],
            "identity": "post_hoc",
        }
    )
    contrast_frame = pd.DataFrame(rows)
    for axis, metric, low, high, xlabel in (
        (ax_c, "rho", "rho_low", "rho_high", "ΔSpearman"),
        (ax_d, "rmse", "rmse_low", "rmse_high", "ΔRMSE"),
    ):
        add_reference(axis)
        for y, row in enumerate(
            contrast_frame.iloc[::-1].to_dict("records")
        ):
            identity = row["identity"]
            color = (
                COLORS["formal"]
                if identity == "formal_matched_extension"
                else COLORS["posthoc"]
                if identity == "post_hoc"
                else COLORS["confirmatory"]
            )
            errorbar_h(
                axis,
                y,
                row[metric],
                row[low],
                row[high],
                color,
                marker="s" if identity == "post_hoc" else "o",
                filled=identity == "confirmatory",
            )
        axis.set_yticks(np.arange(len(contrast_frame)))
        axis.set_yticklabels(contrast_frame["label"].iloc[::-1])
        axis.set_xlabel(xlabel + " (positive favors first model)")
        axis.grid(axis="x", color="#E6E6E6", lw=0.5)
    ax_c.set_title("Matched ranking contrasts", loc="left")
    ax_d.set_title("Matched error contrasts", loc="left")
    panel_label(ax_c, "c")
    panel_label(ax_d, "d")
    save_bundle(fig, output_dir, "fig2_internal_evidence")


def select_ood_model_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        (frame["record_type"] == "model")
        & frame["model_id"].isin([FULL, CHEM])
    ].copy()


def figure3(source_dir: Path, output_dir: Path) -> None:
    ood_frame = pd.read_csv(source_dir / "source_data_confirmatory_ood.csv")
    strict = pd.read_csv(source_dir / "source_data_strict_ood.csv")
    hgb = pd.read_csv(
        source_dir / "source_data_hgb_model_family_sensitivity.csv"
    )
    models = select_ood_model_rows(ood_frame)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(JOURNAL_CANVAS_WIDTH_IN, 5.1969 * JOURNAL_SCALE),
        gridspec_kw={"wspace": 0.54, "hspace": 0.5},
    )
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    selections = [
        ("source_ood", FULL),
        ("source_ood", CHEM),
        ("target_ood", FULL),
        ("target_ood", CHEM),
    ]
    labels = [
        "Source OOD · full",
        "Source OOD · chemistry",
        "Target OOD · full",
        "Target OOD · chemistry",
    ]
    for axis, metric, low, high, xlabel in (
        (
            ax_a,
            "domain_macro_spearman_observed",
            "domain_macro_spearman_ci_low",
            "domain_macro_spearman_ci_high",
            "Domain-macro Spearman ρ",
        ),
        (
            ax_c,
            "domain_macro_rmse_observed",
            "domain_macro_rmse_ci_low",
            "domain_macro_rmse_ci_high",
            "Domain-macro RMSE",
        ),
    ):
        if metric.startswith("domain_macro_spearman"):
            add_reference(axis)
        for y, (protocol, model_id) in enumerate(selections[::-1]):
            row = models[
                (models["protocol"] == protocol)
                & (models["model_id"] == model_id)
            ].iloc[0]
            color = COLORS["full"] if model_id == FULL else COLORS["chem"]
            errorbar_h(
                axis,
                y,
                row[metric],
                row[low],
                row[high],
                color,
                filled=model_id == FULL,
            )
        axis.set_yticks(np.arange(4))
        axis.set_yticklabels(labels[::-1])
        axis.set_xlabel(xlabel)
        axis.grid(axis="x", color="#E6E6E6", lw=0.5)
    ax_a.set_title("Held-domain ranking", loc="left")
    ax_c.set_title("Held-domain error", loc="left")
    panel_label(ax_a, "a")
    panel_label(ax_c, "c")

    contrast_rows = []
    confirmatory_contrasts = ood_frame[
        (ood_frame["record_type"] == "contrast")
        & (ood_frame["contrast_id"] == "full_vs_chemistry_extra_trees")
    ]
    strict_contrasts = strict[
        (strict["record_type"] == "contrast")
        & (strict["contrast_id"] == "full_vs_chemistry_extra_trees")
    ]
    for frame, regime, identity in (
        (
            confirmatory_contrasts,
            "compound-cold",
            "confirmatory",
        ),
        (strict_contrasts, "strict scaffold-cold", "post_hoc"),
    ):
        for row in frame.to_dict("records"):
            contrast_rows.append(
                {
                    "label": (
                        f"{'Source' if row['protocol'] == 'source_ood' else 'Target'} · "
                        f"{'cold' if regime == 'compound-cold' else 'strict'}"
                    ),
                    "rho": row["delta_domain_macro_spearman_observed"],
                    "rho_low": row["delta_domain_macro_spearman_ci_low"],
                    "rho_high": row["delta_domain_macro_spearman_ci_high"],
                    "rmse": row["delta_domain_macro_rmse_observed"],
                    "rmse_low": row["delta_domain_macro_rmse_ci_low"],
                    "rmse_high": row["delta_domain_macro_rmse_ci_high"],
                    "identity": identity,
                }
            )
    hgb_contrasts = hgb[
        (hgb["record_type"] == "contrast")
        & (hgb["split_regime"] != "internal")
    ]
    for row in hgb_contrasts.to_dict("records"):
        contrast_rows.append(
            {
                "label": (
                    f"{'Source' if row['protocol'] == 'source_ood' else 'Target'} · "
                    f"HGB {'strict' if row['split_regime'].startswith('strict') else 'cold'}"
                ),
                "rho": row["delta_domain_macro_spearman_observed"],
                "rho_low": row["delta_domain_macro_spearman_ci_low"],
                "rho_high": row["delta_domain_macro_spearman_ci_high"],
                "rmse": row["delta_domain_macro_rmse_observed"],
                "rmse_low": row["delta_domain_macro_rmse_ci_low"],
                "rmse_high": row["delta_domain_macro_rmse_ci_high"],
                "identity": "post_hoc_hgb",
            }
        )
    contrasts = pd.DataFrame(contrast_rows)
    for axis, metric, low, high, xlabel in (
        (ax_b, "rho", "rho_low", "rho_high", "Δ domain-macro Spearman"),
        (ax_d, "rmse", "rmse_low", "rmse_high", "Δ domain-macro RMSE"),
    ):
        add_reference(axis)
        for y, row in enumerate(contrasts.iloc[::-1].to_dict("records")):
            identity = row["identity"]
            color = (
                COLORS["confirmatory"]
                if identity == "confirmatory"
                else COLORS["posthoc"]
                if identity == "post_hoc_hgb"
                else COLORS["formal"]
            )
            errorbar_h(
                axis,
                y,
                row[metric],
                row[low],
                row[high],
                color,
                marker="s" if identity == "post_hoc_hgb" else "o",
                filled=identity == "confirmatory",
            )
        axis.set_yticks(np.arange(len(contrasts)))
        axis.set_yticklabels(contrasts["label"].iloc[::-1])
        axis.set_xlabel(xlabel + "\n(positive favors full)")
        axis.grid(axis="x", color="#E6E6E6", lw=0.5)
    ax_b.set_title("Ranking transfer contrasts", loc="left")
    ax_d.set_title("Error transfer contrasts", loc="left")
    panel_label(ax_b, "b")
    panel_label(ax_d, "d")
    save_bundle(fig, output_dir, "fig3_domain_transfer")


def figure4(source_dir: Path, output_dir: Path) -> None:
    domains = pd.read_csv(source_dir / "source_data_ood_domains.csv")
    lodo = pd.read_csv(source_dir / "source_data_lodo.csv")
    calibration = pd.read_csv(source_dir / "source_data_calibration.csv")

    fig = plt.figure(
        figsize=(JOURNAL_CANVAS_WIDTH_IN, 5.3150 * JOURNAL_SCALE)
    )
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.12, 0.88],
        wspace=0.52,
        hspace=0.5,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[0, 1])
    ax_d = fig.add_subplot(grid[1, 1])

    domain_data = domains[
        domains["model_id"].isin([FULL, CHEM])
    ][["protocol", "heldout_group", "model_id", "spearman"]]
    pivot = domain_data.pivot(
        index=["protocol", "heldout_group"],
        columns="model_id",
        values="spearman",
    ).reset_index()
    pivot["sort_key"] = (
        pivot["protocol"].map({"source_ood": 0, "target_ood": 1})
        * 10
        + np.arange(len(pivot)) / 100
    )
    pivot = pivot.sort_values(["protocol", FULL], ascending=[True, True])
    y = np.arange(len(pivot))
    add_reference(ax_a)
    for yi, row in zip(y, pivot.to_dict("records")):
        ax_a.plot(
            [row[CHEM], row[FULL]],
            [yi, yi],
            color="#BBBBBB",
            lw=0.8,
        )
        ax_a.plot(
            row[CHEM],
            yi,
            "o",
            ms=3.8,
            mfc="white",
            mec=COLORS["chem"],
            mew=0.9,
        )
        ax_a.plot(
            row[FULL],
            yi,
            "o",
            ms=3.8,
            color=COLORS["full"],
        )
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(
        [
            f"{row.protocol.replace('_ood', '')} · {row.heldout_group}"
            for row in pivot.itertuples()
        ]
    )
    ax_a.set_xlabel("Within-domain Spearman ρ")
    ax_a.set_title("Domain heterogeneity", loc="left")
    ax_a.grid(axis="x", color="#E6E6E6", lw=0.5)
    ax_a.legend(
        handles=[
            mpl.lines.Line2D(
                [], [], marker="o", mfc="white", mec=COLORS["chem"],
                color="none", label="Chemistry-only"
            ),
            mpl.lines.Line2D(
                [], [], marker="o", color=COLORS["full"],
                linestyle="none", label="Full"
            ),
        ],
        loc="upper left",
        ncol=1,
        borderaxespad=0.35,
        handletextpad=0.45,
        labelspacing=0.25,
    )
    panel_label(ax_a, "a")

    lodo_plot = lodo[
        (lodo["analysis_scale"] == "formal_row_level")
    ].copy()
    lodo_plot = lodo_plot.sort_values(
        ["protocol", "delta_lodo_influence_spearman"]
    )
    y = np.arange(len(lodo_plot))
    colors = [
        COLORS["negative"] if value > 0 else COLORS["positive"]
        for value in lodo_plot["delta_lodo_influence_spearman"]
    ]
    ax_b.barh(
        y,
        lodo_plot["delta_lodo_influence_spearman"],
        color=colors,
        alpha=0.78,
    )
    ax_b.axvline(0, color="#999999", lw=0.8)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(
        [
            f"{row.protocol.replace('_ood', '')} · omit {row.omitted_domain}"
            for row in lodo_plot.itertuples()
        ]
    )
    ax_b.set_xlabel("Change in full − chemistry macro-Spearman")
    ax_b.set_title("Fixed-domain leave-one-out influence", loc="left")
    panel_label(ax_b, "b")

    calibration_plot = calibration[
        (calibration["analysis_scale"] == "formal_row_level")
        & (calibration["record_type"] == "model")
        & (calibration["metric_scope"] == "pooled")
        & (calibration["metric"] == "r2")
        & calibration["model_id"].isin([FULL, CHEM])
    ].copy()
    selection_order = [
        ("source_ood", FULL),
        ("source_ood", CHEM),
        ("target_ood", FULL),
        ("target_ood", CHEM),
    ]
    labels = [
        "Source · full",
        "Source · chemistry",
        "Target · full",
        "Target · chemistry",
    ]
    add_reference(ax_c)
    for y, (protocol, model_id) in enumerate(selection_order[::-1]):
        row = calibration_plot[
            (calibration_plot["protocol"] == protocol)
            & (calibration_plot["model_id"] == model_id)
        ].iloc[0]
        color = COLORS["full"] if model_id == FULL else COLORS["chem"]
        errorbar_h(
            ax_c,
            y,
            row["observed"],
            row["ci_low"],
            row["ci_high"],
            color,
            filled=model_id == FULL,
        )
    ax_c.set_yticks(np.arange(4))
    ax_c.set_yticklabels(labels[::-1])
    ax_c.set_xlabel("Pooled R² with 95% scaffold-bootstrap CI")
    ax_c.set_title("Absolute prediction remains weak", loc="left")
    ax_c.grid(axis="x", color="#E6E6E6", lw=0.5)
    panel_label(ax_c, "c")

    ax_d.set_axis_off()
    panel_label(ax_d, "d")
    ax_d.set_title("Claim boundary", loc="left")
    supported = (
        "SUPPORTED\n"
        "• internal scaffold-disjoint ranking\n"
        "• matched internal feature-block increments\n"
        "• modest, heterogeneous chemistry transfer\n"
        "• no OOD context advantage"
    )
    unsupported = (
        "NOT SUPPORTED\n"
        "• broad unseen-target prediction\n"
        "• calibrated absolute DC50\n"
        "• candidate-selection utility\n"
        "• causal or mechanistic interpretation"
    )
    for y0, text, face, edge in (
        (0.52, supported, "#E5F1EA", COLORS["positive"]),
        (0.02, unsupported, "#F5E3E3", COLORS["negative"]),
    ):
        patch = FancyBboxPatch(
            (0.02, y0),
            0.96,
            0.43,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            transform=ax_d.transAxes,
            facecolor=face,
            edgecolor=edge,
            lw=0.8,
        )
        ax_d.add_patch(patch)
        ax_d.text(
            0.06,
            y0 + 0.39,
            text,
            transform=ax_d.transAxes,
            ha="left",
            va="top",
            fontsize=5.7,
            linespacing=1.22,
        )
    save_bundle(fig, output_dir, "fig4_boundary_and_calibration")


def write_legends(output_dir: Path) -> None:
    text = """# Main figure legends v1

**Fig. 1 | Data structure and validation gradient.** a, The three evaluation
regimes progressively separate chemical and domain context; familiar context
can recur internally but is unavailable or shifted in a held domain. b, Rows, unique
canonical compounds and Bemis–Murcko scaffolds across standardized source
labels. c, Sample-size imbalance across modeling target categories; dark bars
denote the eight exact-target domains entering the frozen target-OOD analysis.
d, Missingness of prespecified provenance/context fields and the number of
repeated compound and exact-context groups. Percentages refer to the 1,560-row
frozen QC table. Source data are provided as a Source Data file.

**Fig. 2 | Internal scaffold-disjoint prediction and matched feature-block
evidence.** a,b, Repeat-averaged internal Spearman correlation and RMSE for
chemistry-only and chemistry-plus-context ExtraTrees under scaffold- and
compound-disjoint validation. c,d, Paired scaffold-cluster contrasts for
ranking and error; filled markers denote confirmatory comparisons, grey open
markers the formal matched context-only extension, and teal squares the single
post-hoc HistGradientBoosting model-family sensitivity. Points show estimates
and lines show 95% percentile intervals from 10,000 paired scaffold-cluster
bootstrap replicates (667 global scaffolds). Folds, repeats and bootstrap
replicates are not independent samples. Positive contrasts favour the first
named model. Source data are provided as a Source Data file.

**Fig. 3 | Context gain does not persist under held-domain transfer.**
a,c, Equal-domain-weight Spearman correlation and RMSE for chemistry-only and
full ExtraTrees in four held-source and eight held-target domains. b,d, Paired
full-minus-chemistry contrasts under the frozen domain-plus-compound-cold
protocols, strict post-hoc scaffold purging and the frozen post-hoc
HistGradientBoosting sensitivity. Positive contrasts favour the full model.
Points show estimates and lines show 95% percentile intervals from 10,000
paired global-scaffold bootstrap replicates (667 source-axis and 601
target-axis scaffolds). Source data are provided as a Source Data file.

**Fig. 4 | Domain heterogeneity and calibration define the supported claim
boundary.** a, Within-domain Spearman correlations for chemistry-only and full
ExtraTrees across all frozen source and target domains. b, Change in the
full-minus-chemistry domain-macro Spearman contrast after omitting one fixed
domain; this is descriptive and has no population-level confidence interval.
c, Pooled R² with 95% scaffold-bootstrap intervals for frozen OOD predictions.
d, Claims supported and not supported by the current retrospective evidence.
Domain-level and calibration panels are post-hoc exploratory analyses; no
per-domain multiplicity-adjusted significance claim is made. Source data are
provided as a Source Data file.
"""
    (output_dir / "figure_legends.md").write_text(text, encoding="utf-8")


def write_qa_notes(source_dir: Path, output_dir: Path) -> None:
    required = [
        "provenance_by_source.csv",
        "provenance_by_target.csv",
        "data_missingness.csv",
        "compound_repeat_audit.csv",
        "exact_context_repeat_audit.csv",
        "source_data_internal_models.csv",
        "source_data_internal_contrasts.csv",
        "source_data_confirmatory_ood.csv",
        "source_data_ood_domains.csv",
        "source_data_strict_ood.csv",
        "source_data_lodo.csv",
        "source_data_calibration.csv",
        "source_data_hgb_model_family_sensitivity.csv",
    ]
    notes = [
        "# Figure QA notes",
        "",
        "- Backend: Python/matplotlib exclusively.",
        "- Final width: at or below 170 mm; editable SVG and PDF plus 600 dpi "
        "TIFF.",
        "- No synthetic or simulated data.",
        "- No row was excluded from the source analysis; plotted subsets are "
        "scientifically declared model/metric/evidence subsets.",
        "- Confirmatory, formal-extension and post-hoc identities are encoded "
        "by marker fill/shape and stated in legends.",
        "- Confidence intervals are 10,000-replicate paired global-scaffold "
        "percentile intervals; no p values or star notation.",
        "- OOD domain-macro metrics are primary; pooled R² appears only as a "
        "calibration failure-mode panel.",
        "",
        "## Source files",
        "",
    ]
    for filename in required:
        path = source_dir / filename
        notes.append(
            f"- `{filename}`: {'PASS' if path.is_file() else 'MISSING'}"
        )
    (output_dir / "figure_qa_notes.md").write_text(
        "\n".join(notes) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    set_style()
    figure1(source_dir, output_dir)
    figure2(source_dir, output_dir)
    figure3(source_dir, output_dir)
    figure4(source_dir, output_dir)
    write_legends(output_dir)
    write_qa_notes(source_dir, output_dir)
    print(f"[DONE] four figure bundles written to {output_dir}")


if __name__ == "__main__":
    main()
