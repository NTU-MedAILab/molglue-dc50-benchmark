#!/usr/bin/env python3
"""Plot grouped-context increments for matched internal and OOD estimands."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "reports" / "post_hoc_context_group_ablation_revision_v1_summary"
DEFAULT_OUTPUT = ROOT / "reports" / "revision_figures_v2"
MODEL_ORDER = (
    "chemistry_plus_provenance",
    "chemistry_plus_biological_assay",
    "chemistry_plus_missingness",
    "full_context_extra_trees",
)
LABELS = {
    "chemistry_plus_provenance": "Provenance",
    "chemistry_plus_biological_assay": "Biological/assay",
    "chemistry_plus_missingness": "Missingness",
    "full_context_extra_trees": "All recorded context",
}
COLORS = {
    "chemistry_plus_provenance": "#3E6D9C",
    "chemistry_plus_biological_assay": "#4D8070",
    "chemistry_plus_missingness": "#B07A3F",
    "full_context_extra_trees": "#76558F",
}
NEUTRAL = "#707780"
GRID = "#D9DDE2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.9,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.10, label, transform=ax.transAxes, fontsize=9.2, fontweight="bold", va="top")


def draw_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    axis: str,
    regime: str,
    title: str,
    xlimits: tuple[float, float],
) -> None:
    local = data[(data.axis == axis) & (data.validation_regime == regime)].set_index("model_id")
    y_positions = np.arange(len(MODEL_ORDER))[::-1]
    for y, model in zip(y_positions, MODEL_ORDER):
        if model not in local.index:
            ax.text(
                xlimits[0] + 0.025 * (xlimits[1] - xlimits[0]),
                y,
                "not portable",
                color=NEUTRAL,
                va="center",
                fontsize=6.8,
                fontstyle="italic",
            )
            continue
        row = local.loc[model]
        color = COLORS[model]
        ax.plot(
            [row.delta_domain_macro_spearman_ci_low, row.delta_domain_macro_spearman_ci_high],
            [y, y],
            color=color,
            lw=1.6,
            solid_capstyle="round",
        )
        ax.scatter(
            [row.delta_domain_macro_spearman_observed],
            [y],
            s=29,
            marker="D" if model == "full_context_extra_trees" else "o",
            facecolor=color,
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
    ax.axvline(0, color=NEUTRAL, lw=0.8, ls="--", zorder=0)
    ax.set_yticks(y_positions, [LABELS[model] for model in MODEL_ORDER])
    ax.set_xlim(*xlimits)
    ax.grid(axis="x", color=GRID, lw=0.55)
    ax.set_title(title, loc="left", pad=7)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(input_dir / "matched_group_contrasts.csv")
    low = float(data.delta_domain_macro_spearman_ci_low.min())
    high = float(data.delta_domain_macro_spearman_ci_high.max())
    span = max(high - low, 0.2)
    xlimits = (min(low - 0.08 * span, -0.05), max(high + 0.08 * span, 0.05))

    style()
    fig, axes = plt.subplots(2, 2, figsize=(6.68, 5.25), sharex=True)
    fig.subplots_adjust(left=0.20, right=0.985, bottom=0.17, top=0.92, wspace=0.55, hspace=0.52)
    specifications = (
        (axes[0, 0], "source", "internal", "Source-axis rows · internal"),
        (axes[0, 1], "source", "ood", "Held-source evaluation"),
        (axes[1, 0], "target", "internal", "Target-axis rows · internal"),
        (axes[1, 1], "target", "ood", "Held exact-target-label evaluation"),
    )
    for label, (ax, axis, regime, title) in zip("abcd", specifications):
        draw_panel(ax, data, axis, regime, title, xlimits)
        panel_label(ax, label)
    fig.supxlabel(
        "Context group − chemistry domain-macro Spearman",
        x=0.59,
        y=0.065,
        fontsize=7.5,
    )
    fig.text(
        0.5,
        0.018,
        "Paired 95% global-scaffold bootstrap intervals; target OOD excludes held-target fields.",
        ha="center",
        va="bottom",
        fontsize=6.7,
        color="#4F555B",
    )
    prefix = output_dir / "fig6_context_group_ablation_revision_v1"
    fig.savefig(prefix.with_suffix(".svg"))
    fig.savefig(prefix.with_suffix(".pdf"))
    fig.savefig(prefix.with_suffix(".png"), dpi=400)
    fig.savefig(
        prefix.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    with Image.open(prefix.with_suffix(".tiff")) as rendered:
        rendered.convert("RGB").save(
            prefix.with_suffix(".tiff"),
            compression="tiff_lzw",
            dpi=(600, 600),
        )
    data.to_csv(output_dir / "fig6_context_group_ablation_source_data.csv", index=False)
    print(prefix)


if __name__ == "__main__":
    main()
