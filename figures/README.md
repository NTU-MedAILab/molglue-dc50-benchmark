# Figure bundle

Each figure is supplied as editable SVG, editable-text PDF, 600 dpi RGB TIFF
and a PNG review render. `FIGURE_CONTRACT.json` at the release root records
the exact review/TIFF dimensions, and the release verifier checks the file
formats, dimensions, resolution, embedded/vector text and checksums without
using private figure-QA tools.

Figures 1–5 and Supplementary Figs. S1–S3 have redistribution-safe source
tables under `evidence/source_data/`; Figs. 2 and 3 include the independently
QA-verified molecular-graph sensitivity. GitHub v2 replaces the two excluded
SMILES/context-level repeat tables used only for four displayed counts with
`source_data_repeat_summary.csv`, a one-row aggregate sufficient for the same
render. `analysis_code/plot_publication_main_figures_v2.py` reproduces the Fig.
1 PNG pixel-for-pixel in the locked environment without exposing a structure,
record identifier or row-level context.
