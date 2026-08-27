#!/usr/bin/env python
"""Plot the matched-estimand validation-regime and domain-heterogeneity figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT = (
    PROJECT_DIR / "reports" / "post_hoc_reviewer_revision_v1"
)
DEFAULT_OUTPUT = PROJECT_DIR / "reports" / "revision_figures_v2"

BLUE = "#3E6D9C"
ORANGE = "#C47A44"
PURPLE = "#76558F"
NEGATIVE = "#A55252"
POSITIVE = "#4D8070"
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
            "font.size": 7.4,
            "axes.titlesize": 8.4,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def interval_point(
    ax: plt.Axes,
    y: float,
    value: float,
    low: float,
    high: float,
    color: str,
    marker: str = "o",
    filled: bool = True,
    size: float = 28,
) -> None:
    ax.plot([low, high], [y, y], color=color, lw=1.5, solid_capstyle="round")
    ax.scatter(
        [value],
        [y],
        s=size,
        marker=marker,
        facecolor=color if filled else "white",
        edgecolor=color,
        linewidth=1.1,
        zorder=3,
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    contrasts = pd.read_csv(input_dir / "matched_estimand_contrasts.csv")
    domains = pd.read_csv(input_dir / "matched_domain_contrasts.csv")
    domains_ood = domains[domains["validation_regime"] == "ood"].copy()

    style()
    fig = plt.figure(figsize=(6.68, 6.35), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.82, 1.42],
        width_ratios=[0.86, 1.14],
        left=0.19,
        right=0.985,
        bottom=0.095,
        top=0.945,
        wspace=0.58,
        hspace=0.52,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    # a: matched context increments under the same equal-domain estimand.
    ordering = [
        ("source", "internal", "Source · internal", BLUE),
        ("source", "ood", "Source · held domain", ORANGE),
        ("target", "internal", "Target · internal", BLUE),
        ("target", "ood", "Target · held label", ORANGE),
    ]
    y_positions = np.arange(len(ordering))[::-1]
    for y, (axis, regime, label, color) in zip(y_positions, ordering):
        row = contrasts[
            (contrasts["axis"] == axis)
            & (contrasts["validation_regime"] == regime)
        ].iloc[0]
        interval_point(
            ax_a,
            y,
            row["delta_domain_macro_spearman_observed"],
            row["delta_domain_macro_spearman_ci_low"],
            row["delta_domain_macro_spearman_ci_high"],
            color,
            filled=regime == "ood",
        )
    ax_a.axvline(0, color=NEUTRAL, lw=0.8, ls="--", zorder=0)
    ax_a.set_yticks(y_positions, [item[2] for item in ordering])
    ax_a.set_xlabel("Full − chemistry domain-macro Spearman")
    ax_a.set_title("Matched context increments", loc="left", pad=7)
    ax_a.grid(axis="x", color=GRID, lw=0.55)
    panel_label(ax_a, "a")

    # b: paired change in the context increment from internal to held domain.
    change = contrasts[
        contrasts["validation_regime"] == "paired_regime_change"
    ].set_index("axis")
    change_order = [("source", "Source axis"), ("target", "Target axis")]
    y_change = np.arange(2)[::-1]
    for y, (axis, label) in zip(y_change, change_order):
        row = change.loc[axis]
        interval_point(
            ax_b,
            y,
            row["delta_domain_macro_spearman_observed"],
            row["delta_domain_macro_spearman_ci_low"],
            row["delta_domain_macro_spearman_ci_high"],
            PURPLE,
            marker="D",
        )
        ax_b.text(
            row["delta_domain_macro_spearman_ci_high"] + 0.006,
            y,
            f"{row['delta_domain_macro_spearman_observed']:+.3f}",
            va="center",
            fontsize=6.8,
            color=PURPLE,
        )
    ax_b.axvline(0, color=NEUTRAL, lw=0.8, ls="--", zorder=0)
    ax_b.set_yticks(y_change, [item[1] for item in change_order])
    ax_b.set_xlabel("Held-domain minus internal context increment")
    ax_b.set_title("Paired validation-regime attenuation", loc="left", pad=7)
    ax_b.xaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
    ax_b.grid(axis="x", color=GRID, lw=0.55)
    panel_label(ax_b, "b")

    # c/d: domain-level paired forest plots.
    for ax, axis, title in (
        (ax_c, "source", "Held-source heterogeneity"),
        (ax_d, "target", "Held-target-label heterogeneity"),
    ):
        local = domains_ood[domains_ood["axis"] == axis].copy()
        local = local.sort_values("delta_spearman_observed", ascending=True)
        ypos = np.arange(len(local))
        for y, (_, row) in zip(ypos, local.iterrows()):
            value = row["delta_spearman_observed"]
            color = NEGATIVE if value < 0 else POSITIVE
            interval_point(
                ax,
                y,
                value,
                row["delta_spearman_ci_low"],
                row["delta_spearman_ci_high"],
                color,
                size=23,
            )
        labels = [
            f"{row.domain}  {int(row.n_rows)}/{int(row.n_scaffolds)}"
            for row in local.itertuples()
        ]
        ax.set_yticks(ypos, labels)
        ax.axvline(0, color=NEUTRAL, lw=0.8, ls="--", zorder=0)
        ax.grid(axis="x", color=GRID, lw=0.55)
        ax.set_xlabel("Full − chemistry Spearman")
        ax.set_title(title, loc="left", pad=7)
    panel_label(ax_c, "c")
    panel_label(ax_d, "d")

    fig.text(
        0.5,
        0.022,
        "Paired 95% global-scaffold bootstrap intervals; labels give rows/scaffolds.",
        ha="center",
        va="bottom",
        fontsize=6.6,
        color="#4F555B",
    )

    prefix = output_dir / "fig4_matched_estimand_revision_v2"
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

    source = domains_ood.copy()
    source.to_csv(output_dir / "fig4_matched_estimand_source_data.csv", index=False)
    contrasts.to_csv(
        output_dir / "fig4_matched_estimand_contrast_source_data.csv", index=False
    )
    print(prefix)


if __name__ == "__main__":
    main()
