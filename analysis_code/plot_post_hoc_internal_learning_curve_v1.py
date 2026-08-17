#!/usr/bin/env python3
"""Render Supplementary Fig. S4 from the frozen learning-curve aggregates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
ROUTE_DIR = SCRIPT_DIR.parent
DEFAULT_SOURCE_DIR = (
    ROUTE_DIR / "reports" / "post_hoc_internal_learning_curve_v1"
)
DEFAULT_OUTPUT_DIR = DEFAULT_SOURCE_DIR / "publication_figure"
STEM = "supp_fig_s4_internal_learning_curve"
FORMATS = (".svg", ".pdf", ".tiff", ".png")
WIDTH_MM = 170.0
HEIGHT_MM = 88.0
MM_PER_INCH = 25.4
CHEM = "chemistry_extra_trees"
FULL = "full_context_extra_trees"
COLORS = {
    "chem": "#4C78A8",
    "full": "#173F5F",
    "grid": "#D8DDE2",
    "text": "#182B3A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_style() -> None:
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
    mpl.rcParams["legend.fontsize"] = 6.5
    mpl.rcParams["axes.linewidth"] = 0.65
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["legend.frameon"] = False
    mpl.rcParams["text.color"] = COLORS["text"]
    mpl.rcParams["axes.labelcolor"] = COLORS["text"]
    mpl.rcParams["xtick.color"] = COLORS["text"]
    mpl.rcParams["ytick.color"] = COLORS["text"]


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.13,
        1.05,
        label,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def validate_source(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_path = source_dir / "run_manifest.json"
    qa_path = source_dir / "independent_qa.json"
    metrics_path = source_dir / "learning_curve_metrics.csv"
    contrasts_path = source_dir / "learning_curve_contrasts.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or manifest.get("analysis_mode") != "formal_post_hoc"
        or manifest.get("bootstrap_replicates") != 10_000
        or manifest.get("fit_count") != 200
        or manifest.get("gpu_required") is not False
    ):
        raise RuntimeError("Learning-curve formal manifest is not complete")
    if (
        qa.get("status") != "PASS"
        or qa.get("checks_passed") != 20
        or qa.get("checks_total") != 20
    ):
        raise RuntimeError("Learning-curve independent QA is not PASS 20/20")
    metrics = pd.read_csv(metrics_path)
    contrasts = pd.read_csv(contrasts_path)
    expected_metric_keys = {
        (fraction, model, metric)
        for fraction in (0.25, 0.50, 0.75, 1.00)
        for model in (CHEM, FULL)
        for metric in ("spearman", "rmse")
    }
    observed_metric_keys = set(
        zip(
            metrics["training_fraction"],
            metrics["model_id"],
            metrics["metric"],
        )
    )
    if observed_metric_keys != expected_metric_keys:
        raise RuntimeError("Learning-curve metric grid is incomplete")
    required = ["estimate", "ci_low", "ci_high"]
    if not np.isfinite(metrics[required].to_numpy(dtype=float)).all():
        raise RuntimeError("Learning-curve metrics contain non-finite values")
    return metrics, contrasts


def make_figure(
    metrics: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> plt.Figure:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(WIDTH_MM / MM_PER_INCH, HEIGHT_MM / MM_PER_INCH),
    )
    model_specs = (
        (CHEM, "Chemistry only", COLORS["chem"], "o", False),
        (FULL, "Chemistry + context", COLORS["full"], "s", True),
    )
    for axis, metric, ylabel in (
        (axes[0], "spearman", "Pooled Spearman correlation"),
        (axes[1], "rmse", "RMSE (pDC50 units)"),
    ):
        for model, label, color, marker, filled in model_specs:
            frame = (
                metrics[
                    (metrics["model_id"] == model)
                    & (metrics["metric"] == metric)
                ]
                .sort_values("training_fraction")
                .reset_index(drop=True)
            )
            x = frame["train_scaffolds_median"].to_numpy(dtype=float)
            y = frame["estimate"].to_numpy(dtype=float)
            low = frame["ci_low"].to_numpy(dtype=float)
            high = frame["ci_high"].to_numpy(dtype=float)
            axis.plot(x, y, color=color, lw=1.2, zorder=2)
            axis.errorbar(
                x,
                y,
                yerr=np.vstack([y - low, high - y]),
                fmt=marker,
                ms=4.8,
                mfc=color if filled else "white",
                mec=color,
                mew=0.9,
                color=color,
                ecolor=color,
                elinewidth=0.9,
                capsize=2.2,
                label=label,
                zorder=3,
            )
        axis.set_xlabel("Median training scaffolds across outer folds")
        axis.set_ylabel(ylabel)
        axis.set_xticks([134, 267, 401, 534])
        axis.set_xticklabels(["134\n(25%)", "267\n(50%)", "401\n(75%)", "534\n(100%)"])
        axis.grid(axis="y", color=COLORS["grid"], lw=0.5, zorder=0)
        axis.legend(loc="best")
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    axes[0].set_ylim(0.36, 0.72)
    axes[1].set_ylim(0.74, 0.97)
    axes[1].invert_yaxis()

    final_rows = contrasts[
        contrasts["contrast_id"].isin(
            [
                "fraction_100_minus_75_spearman",
                "fraction_75_minus_100_rmse",
            ]
        )
    ]
    chem_rho = final_rows[
        (final_rows["model_id"] == CHEM)
        & (final_rows["metric"] == "spearman")
    ].iloc[0]
    full_rho = final_rows[
        (final_rows["model_id"] == FULL)
        & (final_rows["metric"] == "spearman")
    ].iloc[0]
    figure.text(
        0.5,
        0.01,
        (
            "Observed 75%→100% Spearman change: "
            f"chemistry {chem_rho['estimate']:+.003f} "
            f"[{chem_rho['ci_low']:+.003f}, {chem_rho['ci_high']:+.003f}]; "
            f"full {full_rho['estimate']:+.003f} "
            f"[{full_rho['ci_low']:+.003f}, {full_rho['ci_high']:+.003f}]. "
            "Intervals are paired 95% scaffold-bootstrap intervals."
        ),
        ha="center",
        va="bottom",
        fontsize=6.2,
        color=COLORS["text"],
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.92,
        bottom=0.24,
        wspace=0.28,
    )
    return figure


def save_bundle(figure: plt.Figure, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{STEM}{suffix}" for suffix in FORMATS]
    figure.savefig(paths[0], bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    figure.savefig(
        paths[2],
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    with Image.open(paths[2]) as image:
        if image.mode == "RGBA":
            canvas = Image.new("RGBA", image.size, "white")
            Image.alpha_composite(canvas, image).convert("RGB").save(
                paths[2],
                dpi=(600, 600),
                compression="tiff_lzw",
            )
    figure.savefig(
        paths[3],
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    return paths


def main() -> None:
    args = parse_args()
    set_style()
    metrics, contrasts = validate_source(args.source_dir.resolve())
    figure = make_figure(metrics, contrasts)
    paths = save_bundle(figure, args.output_dir.resolve())
    plt.close(figure)

    source_data_path = args.output_dir / f"{STEM}_source_data.csv"
    metrics.to_csv(source_data_path, index=False)
    manifest_rows = []
    for path in [*paths, source_data_path]:
        row = {
            "file": path.name,
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        if path.suffix.lower() in {".png", ".tiff"}:
            with Image.open(path) as image:
                row.update(
                    {
                        "pixel_width": image.width,
                        "pixel_height": image.height,
                        "mode": image.mode,
                        "dpi_x": float(
                            image.info.get("dpi", (math.nan, math.nan))[0]
                        ),
                        "dpi_y": float(
                            image.info.get("dpi", (math.nan, math.nan))[1]
                        ),
                    }
                )
        manifest_rows.append(row)
    pd.DataFrame(manifest_rows).to_csv(
        args.output_dir / f"{STEM}_manifest.csv",
        index=False,
    )
    (args.output_dir / f"{STEM}_legend.md").write_text(
        (
            "**Supplementary Fig. S4 | Internal scaffold learning curve.** "
            "**a,b**, Pooled Spearman correlation and RMSE for fixed "
            "chemistry-only and chemistry-plus-context ExtraTrees models as "
            "nested prefixes retained 25%, 50%, 75% or 100% of the available "
            "outer-training scaffolds. Points use repeat-averaged predictions; "
            "error bars are paired 95% intervals from 10,000 scaffold-cluster "
            "bootstrap resamples. Values below each x-axis count give the "
            "prespecified scaffold fraction. The 75%-to-100% contrast is "
            "post-hoc and does not establish an asymptote or required sample "
            "size.\n"
        ),
        encoding="utf-8",
    )
    print(
        f"POST_HOC_INTERNAL_LEARNING_CURVE_FIGURE: PASS ({len(paths)} files)"
    )
    print("GPU_REQUIRED: NO")


if __name__ == "__main__":
    main()
