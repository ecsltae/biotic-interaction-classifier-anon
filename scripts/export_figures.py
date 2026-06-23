#!/usr/bin/env python3
"""
Export slide figures as PNGs — styled to match the Madrid Beamer presentation.
Colors, row shading, and table structure mirror the LaTeX source exactly.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

OUT = Path("/path/to/MetaP/slides_export")
OUT.mkdir(exist_ok=True)

# ── Exact colors from classifier_slides.tex ───────────────────────────────────
GOOD      = "#228B22"   # \definecolor{good}
BAD       = "#B22222"   # \definecolor{bad}
WARN      = "#D28C00"   # \definecolor{warn}
NEUTRAL   = "#3C5AA0"   # \definecolor{neutral}  (Madrid header blue)
HIGHLIGHT = "#0064B4"   # \definecolor{highlight}

# Row tint variants (15–20% opacity equivalent)
GOOD15    = "#D6EDD6"
BAD15     = "#F5CECE"
WARN15    = "#FAF0CE"
GOOD20    = "#C8E6C8"
LGREY     = "#F2F2F2"   # alternating row
WHITE     = "#FFFFFF"

DPI = 180

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          12,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.titlesize":     14,
    "axes.titleweight":   "bold",
    "figure.dpi":         DPI,
})


# ── Helper: render a table the way Beamer does ────────────────────────────────
def render_table(ax, rows, cols, row_colors,
                 col_widths=None, font_size=11, row_height=0.65):
    """
    row_colors: list of hex colors, one per data row (not header).
    col_widths: relative widths; None = equal.
    """
    ax.axis("off")
    n_cols = len(cols)
    n_rows = len(rows)
    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols

    # normalise to sum=1
    total = sum(col_widths)
    col_widths = [w / total for w in col_widths]

    total_h = (n_rows + 1) * row_height   # +1 for header
    total_w = 1.0

    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)

    def draw_row(row_idx, values, bg_color, text_color="black",
                 bold=False, fontsize=None):
        if fontsize is None:
            fontsize = font_size
        y_bot = total_h - (row_idx + 1) * row_height
        x = 0.0
        for c, (val, cw) in enumerate(zip(values, col_widths)):
            ax.add_patch(plt.Rectangle((x, y_bot), cw, row_height,
                                       facecolor=bg_color, edgecolor=WHITE, lw=1.5))
            ax.text(x + cw / 2, y_bot + row_height / 2, str(val),
                    ha="center", va="center", fontsize=fontsize,
                    color=text_color,
                    fontweight="bold" if bold else "normal",
                    wrap=True)
            x += cw

    # Header row
    draw_row(0, cols, NEUTRAL, text_color=WHITE, bold=True)
    # Data rows
    for i, (row, bg) in enumerate(zip(rows, row_colors)):
        draw_row(i + 1, row, bg)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 01 — Dataset version F1 bar chart
# ══════════════════════════════════════════════════════════════════════════════
def fig_version_f1():
    # Per-source breakdown: Template-trained BiomedBERT vs Multi-task warm-start champion.
    # Computed on the final 500-sentence test set (eval-100/BioTx-random deduplicated
    # and merged into one 104-sentence group; gen-set-150-extension added to restore 500).
    sources    = ["EP-A\n(n=99)", "EP-passage\n(n=100)", "eval-100 /\nBioTx-random\n(n=104)",
                  "gen-set-100\n(n=100)", "gen-set-ext.\n(n=97)"]
    template_f1 = [0.781, 0.805, 0.933, 0.909, 0.983]
    champion_f1 = [0.840, 0.818, 0.933, 0.838, 0.983]

    x = np.arange(len(sources))
    width = 0.32

    fig, ax = plt.subplots(figsize=(11, 5.8))
    b1 = ax.bar(x - width / 2, template_f1, width, color="#888888",
                edgecolor=WHITE, linewidth=1.2, zorder=3, label="BiomedBERT (template-trained)")
    b2 = ax.bar(x + width / 2, champion_f1, width, color=GOOD,
                edgecolor=WHITE, linewidth=1.2, zorder=3, label="Multi-task champion (warm-start)")

    for bar, val in zip(b1, template_f1):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}",
                ha="center", va="bottom", fontsize=10, color="#333333")
    for bar, val in zip(b2, champion_f1):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1a6b1a")

    ax.set_xticks(x)
    ax.set_xticklabels(sources, fontsize=11)
    ax.set_ylim(0.60, 1.05)
    ax.set_ylabel("F1 on this source", fontsize=12, color="#444444")
    ax.set_title("Per-Source F1: Template-trained BiomedBERT vs Multi-task Champion\n"
                 "(500-sentence test set, 5 sources)", pad=14, fontsize=12)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, framealpha=0.9, loc="upper center",
              bbox_to_anchor=(0.5, -0.20), ncol=2)

    fig.tight_layout()
    fig.savefig(OUT / "fig_01_version_f1.png", bbox_inches="tight")
    plt.close()
    print("fig_01_version_f1.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 02 — Model evolution F1 (matches slide 8)
# ══════════════════════════════════════════════════════════════════════════════
def fig_model_evolution():
    # Ordered to show distillation + warm-start story:
    # hard CE → cold-start multitask → distilled → ensemble → warm-start (champion) → template-trained
    models = ["Hard CE\n(same arch.)", "Multi-task\ncold-start", "Distilled\n(student)",
              "Ensemble\nBERT × T5", "Multi-task\nwarm-start (champion)", "BiomedBERT\n(template-trained)"]
    f1s    = [0.673, 0.835, 0.858, 0.850, 0.874, 0.875]
    # Champion: green; our other configs: blue; reference/baseline: grey
    colors = [NEUTRAL, NEUTRAL, NEUTRAL, "#888888", GOOD, "#888888"]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(models, f1s, color=colors, edgecolor=WHITE,
                  linewidth=1.4, width=0.6, zorder=3)

    for bar, val, col in zip(bars, f1s, colors):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=11.5, fontweight="bold",
                color=GOOD if col == GOOD else "#333333")

    ax.set_ylim(0.60, 0.92)
    ax.set_ylabel("Test set F1", fontsize=12, color="#444444")
    ax.set_title("Model Comparison on 500-sentence Test Set", pad=14)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Soft-label gain annotation (hard CE -> cold-start multitask)
    xm = 0.5
    y  = 0.850
    ax.annotate("", xy=(1 - 0.25, y), xytext=(0 + 0.25, y),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.5))
    ax.text(xm, y + 0.008, "+0.162\nsoft labels", ha="center", fontsize=8.5,
            color="#333333", fontweight="bold")

    # Warm-start gain annotation (cold-start multitask -> warm-start champion)
    xm2 = 3.0
    y2  = 0.882
    ax.annotate("", xy=(4 - 0.25, y2), xytext=(1 + 0.25, y2),
                arrowprops=dict(arrowstyle="-|>", color=GOOD, lw=1.5,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(xm2, y2 + 0.018, "+0.040 warm-start,\nzero NER pre-train", ha="center", fontsize=8.5,
            color=GOOD, fontweight="bold")

    # Annotation: champion ties template-trained BiomedBERT, surpasses ensemble
    ax.annotate("ns vs. template-\ntrained BiomedBERT\n(p = 0.760)", xy=(4, 0.874), xytext=(4.55, 0.70),
                fontsize=8.5, color="#333333", ha="center",
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.0, alpha=0.7))

    # Bracket for reference models
    ax.axvline(3.5, color="#AAAAAA", linestyle="--", linewidth=1.0, alpha=0.5)

    patches = [
        mpatches.Patch(color=GOOD,    label="Proposed champion (warm-start multi-task)"),
        mpatches.Patch(color=NEUTRAL, label="Our pipeline (other configs)"),
        mpatches.Patch(color="#888888", label="Reference models"),
    ]
    ax.legend(handles=patches, fontsize=9.5, framealpha=0.9,
              loc="upper left", bbox_to_anchor=(0.01, 0.99))

    fig.tight_layout()
    fig.savefig(OUT / "fig_02_model_evolution.png", bbox_inches="tight")
    plt.close()
    print("fig_02_model_evolution.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 03 — Distillation table (matches slide 10)
# ══════════════════════════════════════════════════════════════════════════════
def fig_distillation_table():
    # Updated to 500-sentence test set (manuscript Appendix A)
    cols = ["Student model", "T", "α", "Test F1"]
    rows = [
        ["v1  BiomedBERT",   "4",   "0.7", "0.800"],
        ["v2  BiomedBERT ★", "2",   "0.5", "0.858"],
        ["v3  BiomedBERT",   "4",   "0.9", "0.704"],
        ["v4  DistilBERT",   "2",   "0.5", "0.785"],
        ["v5  SciBERT",      "2",   "0.5", "0.799"],
        ["v6  BiomedBERT",   "1.5", "0.5", "0.783"],
    ]
    row_colors = [LGREY, GOOD20, WHITE, LGREY, WHITE, LGREY]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("Knowledge Distillation — Hyperparameter Search\n"
                 "(500-sentence test set)", pad=14)
    render_table(ax, rows, cols, row_colors,
                 col_widths=[2.8, 0.6, 0.6, 1.0], font_size=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_03_distillation_table.png", bbox_inches="tight")
    plt.close()
    print("fig_03_distillation_table.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 04 — Full ablation table (matches slide 13)
# ══════════════════════════════════════════════════════════════════════════════
def fig_ablation_table():
    # Updated to 500-sentence test set (manuscript Appendix B, Table "ablation_full")
    cols = ["Config", "NER scheme", "α", "Init", "NER\npre-train", "Test F1", "P", "R"]
    rows = [
        ["basic_a05",                 "basic",      "0.5", "cold", "0 ep", "0.777", "0.854", "0.713"],
        ["typed_a05",                 "typed",      "0.5", "cold", "0 ep", "0.815", "0.815", "0.815"],
        ["full_a03",                  "full",       "0.3", "cold", "0 ep", "0.821", "0.828", "0.815"],
        ["full_typed_a05",            "full_typed", "0.5", "cold", "0 ep", "0.792", "0.813", "0.772"],
        ["full_a05_ner2",             "full",       "0.5", "cold", "2 ep", "0.781", "0.859", "0.717"],
        ["full_typed_a03_ner2",       "full_typed", "0.3", "cold", "2 ep", "0.802", "0.834", "0.772"],
        ["full_typed_a05_ner2",       "full_typed", "0.5", "cold", "2 ep", "0.835", "0.885", "0.790"],
        ["full_typed (5 joint ep)",   "full_typed", "0.5", "cold", "2+5",  "0.798", "0.836", "0.764"],
        ["full_typed_a05 ★ champion", "full_typed", "0.5", "warm", "0 ep", "0.874", "0.925", "0.829"],
        ["full_typed_a05 (warm)",     "full_typed", "0.5", "warm", "1 ep", "0.860", "0.881", "0.840"],
        ["full_typed_a05 (warm)",     "full_typed", "0.5", "warm", "2 ep", "0.822", "0.902", "0.754"],
        ["── hard CE (no distill.)",  "full_typed", "0.5", "cold", "2 ep", "0.673", "0.920", "0.530"],
        ["── distilled student",      "—",          "—",   "cold", "—",    "0.858", "0.877", "0.840"],
        ["── Ensemble BERT×T5",       "—",          "—",   "—",    "—",    "0.850", "0.925", "0.787"],
    ]
    row_colors = [
        LGREY, WHITE, LGREY, WHITE,
        LGREY, WHITE,
        LGREY,            # cold-start best (full_typed_a05_ner2)
        WHITE,
        GOOD20,           # champion (warm-start, 0 NER ep)
        LGREY, WHITE,
        WARN15,           # hard CE baseline
        WARN15, WARN15,
    ]

    fig, ax = plt.subplots(figsize=(16, 7.5))
    ax.set_title("Multi-task Ablation — Full Results, 500-sentence Test Set\n"
                 "★ = champion (warm-start, 0 NER pre-train epochs)    ── rows = reference baselines",
                 pad=14)
    render_table(ax, rows, cols, row_colors,
                 col_widths=[3.2, 1.8, 0.5, 0.7, 1.0, 0.9, 0.8, 0.8],
                 font_size=10, row_height=0.58)
    fig.tight_layout()
    fig.savefig(OUT / "fig_04_ablation_table.png", bbox_inches="tight")
    plt.close()
    print("fig_04_ablation_table.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 05 — Bootstrap CI (matches slide 14)
# ══════════════════════════════════════════════════════════════════════════════
def fig_bootstrap_ci():
    models    = ["BiomedBERT\n(template-trained)", "Distilled v2", "Multi-task (best)"]
    ep_f1     = [0.785, 0.808, 0.868]
    ep_lo     = [0.693, 0.717, 0.791]
    ep_hi     = [0.864, 0.884, 0.930]
    ev_f1     = [0.831, 0.768, 0.776]
    ev_lo     = [0.797, 0.726, 0.735]
    ev_hi     = [0.863, 0.805, 0.814]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Model Comparison — 95% Bootstrap Confidence Intervals (n = 10,000)",
                 fontsize=14, fontweight="bold", y=1.01)

    datasets = [
        (axes[0], ep_f1, ep_lo, ep_hi, "EP-relax  (n = 99, 48 pos)",  2),
        (axes[1], ev_f1, ev_lo, ev_hi, "eval_sets  (n = 599, 7 files)", 0),
    ]

    for ax, f1s, los, his, title, winner in datasets:
        x = np.arange(len(models))
        bar_colors = [GOOD if i == winner else NEUTRAL for i in range(len(models))]
        yerr_lo = [f - l for f, l in zip(f1s, los)]
        yerr_hi = [h - f for f, h in zip(f1s, his)]

        bars = ax.bar(x, f1s, width=0.55, color=bar_colors,
                      edgecolor=WHITE, linewidth=1.2, zorder=3)
        ax.errorbar(x, f1s, yerr=[yerr_lo, yerr_hi],
                    fmt="none", ecolor="#333333", capsize=7,
                    capthick=1.5, linewidth=1.5, zorder=4)

        for i, (bar, val) in enumerate(zip(bars, f1s)):
            ypos = val + yerr_hi[i] + 0.008
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{val:.3f}",
                    ha="center", va="bottom",
                    fontsize=11,
                    fontweight="bold" if i == winner else "normal",
                    color=GOOD if i == winner else "#333333")

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10.5)
        ax.set_ylim(0.62, 0.98)
        ax.set_ylabel("F1", fontsize=12)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.yaxis.grid(True, linestyle=":", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)

    # McNemar annotations below bars
    axes[0].text(0.5, -0.16,
                 "MT vs BiomedBERT v7: p = 0.016 ✓  |  MT vs distilled: p = 0.149 (n.s.)",
                 transform=axes[0].transAxes, ha="center", fontsize=9.5,
                 color="#555555", style="italic")
    axes[1].text(0.5, -0.16,
                 "v7 vs MT: p = 0.009 ✓  |  v7 vs distilled: p = 0.003 ✓",
                 transform=axes[1].transAxes, ha="center", fontsize=9.5,
                 color="#555555", style="italic")

    fig.tight_layout()
    fig.savefig(OUT / "fig_05_bootstrap_ci.png", bbox_inches="tight")
    plt.close()
    print("fig_05_bootstrap_ci.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 06 — Performance ceiling bar chart (matches slide 16)
# ══════════════════════════════════════════════════════════════════════════════
def fig_ceiling():
    types  = ["endoparasiteOf\n32%", "hasHost\n14%", "preysOn\n12%",
              "pathogenOf\n22%", "other\n20%"]
    shares = [32, 14, 12, 22, 20]
    colors = [BAD, BAD, BAD, GOOD, NEUTRAL]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(types, shares, color=colors, edgecolor=WHITE,
                  linewidth=1.4, width=0.62, zorder=3)

    for bar, val, col in zip(bars, shares, colors):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                f"{val}%", ha="center", va="bottom",
                fontsize=12, fontweight="bold",
                color=BAD if col == BAD else "#333333")

    ax.set_ylabel("Share of EP-relax positives", fontsize=12)
    ax.set_ylim(0, 42)
    ax.set_title("EP-relax Positive Distribution\n"
                 "58% of positives are ecological types absent from PubMed",
                 pad=12)
    ax.yaxis.grid(True, linestyle=":", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    patches = [
        mpatches.Patch(color=BAD,     label="Absent from PubMed training corpus"),
        mpatches.Patch(color=GOOD,    label="Present in training corpus"),
        mpatches.Patch(color=NEUTRAL, label="Partially represented"),
    ]
    ax.legend(handles=patches, fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_06_ceiling.png", bbox_inches="tight")
    plt.close()
    print("fig_06_ceiling.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 07 — NER scheme table (matches slide 13)
# ══════════════════════════════════════════════════════════════════════════════
def fig_ner_schemes():
    # Updated to 500-sentence test set (manuscript Table 2, cold-start F1 for
    # like-for-like comparison; full_typed reaches 0.874 with warm-start)
    cols = ["Scheme", "Labels included", "What is tagged", "# labels", "Cold-start F1"]
    rows = [
        ["basic",       "O, B/I-SPECIES",
         "Any species mention\n(no role distinction)", "3", "0.777"],
        ["typed",       "O, B/I-HOST, B/I-PATHOGEN",
         "Role-distinguished species\n(host vs pathogen)", "5", "0.815"],
        ["full",        "O, B/I-SPECIES, B/I-INT",
         "Any species\n+ interaction verbs", "5", "0.824"],
        ["full_typed ★","O, B/I-HOST, B/I-PATHOGEN, B/I-SPECIES, B/I-INT",
         "Role-distinguished species\n+ interaction verbs", "9", "0.835"],
    ]
    row_colors = [LGREY, WHITE, LGREY, GOOD20]

    fig, ax = plt.subplots(figsize=(13, 4.0))
    ax.set_title("NER Label Schemes  —  B = Beginning of span, I = Inside span\n"
                 "★ = winning scheme (cold-start F1; reaches 0.874 with warm-start init)", pad=12)
    render_table(ax, rows, cols, row_colors,
                 col_widths=[1.5, 3.8, 2.8, 0.9, 1.0],
                 font_size=10.5, row_height=0.72)
    fig.tight_layout()
    fig.savefig(OUT / "fig_07_ner_schemes.png", bbox_inches="tight")
    plt.close()
    print("fig_07_ner_schemes.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 08 — Dataset version history table (matches slide 6)
# ══════════════════════════════════════════════════════════════════════════════
def fig_version_table():
    cols = ["Version", "Samples", "Positives", "EP F1", "Key change"]
    rows = [
        ["v1",                "40K",   "17.8K",  "—",     "Initial GloBI templates"],
        ["v2",                "45K",   "13.4K",  "—",     "Hard negatives added"],
        ["v3–v6",             "~32K",  "~6.5K",  "—",     "Template refinement & diversity"],
        ["v7 ★",              "25K",   "7.3K",   "0.788", "Claude API validation on every positive"],
        ["v8",                "26K",   "—",      "0.695", "SIBiLS harvest, 92% pathogen bias"],
        ["v9",                "26K",   "—",      "0.644", "Regex labels — noise"],
        ["v10",               "27K",   "8.1K",   "0.722", "Real PMC sentences, pathogen-biased"],
        ["v11–v12",           "28K",   "8.1K",   "0.729", "Diversity push + score>0 filter"],
        ["v14",               "35K",   "11.4K",  "0.706", "Filter repeated — regression"],
        ["v15–v19 (R.A.)",    "26–35K","8–11K",  "0.828↓","Research agent experiments, all regressed"],
        ["Qwen+soft labels ★","44K",   "4.1K",   "0.868", "Qwen YES/NO + ensemble soft labels — BEST"],
    ]
    row_colors = [
        LGREY, WHITE, LGREY,
        GOOD15,             # v7
        BAD15, BAD15, BAD15,
        WARN15,             # v12
        BAD15,              # v14
        WARN15,             # v15–v19
        GOOD20,             # Qwen best
    ]

    fig, ax = plt.subplots(figsize=(15, 6.5))
    ax.set_title("Training Dataset Version History\n"
                 "★ = key milestones    R.A. = Research Agent experiments",
                 pad=12)
    render_table(ax, rows, cols, row_colors,
                 col_widths=[2.2, 1.2, 1.2, 0.9, 4.5],
                 font_size=10.5, row_height=0.62)
    fig.tight_layout()
    fig.savefig(OUT / "fig_08_version_table.png", bbox_inches="tight")
    plt.close()
    print("fig_08_version_table.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 09 — Multi-task architecture diagram (matches slide 11)
# ══════════════════════════════════════════════════════════════════════════════
def fig_architecture():
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)

    def box(x, y, w, h, line1, line2="", fc="#F2F2F2", ec="#555555", lw=1.5,
            bold=False, fontsize=12):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                    facecolor=fc, edgecolor=ec, linewidth=lw))
        y_mid = y + h / 2
        if line2:
            ax.text(x + w/2, y_mid + 0.17, line1, ha="center", va="center",
                    fontsize=fontsize, fontweight="bold" if bold else "normal")
            ax.text(x + w/2, y_mid - 0.22, line2, ha="center", va="center",
                    fontsize=fontsize - 1.5, color="#555555")
        else:
            ax.text(x + w/2, y_mid, line1, ha="center", va="center",
                    fontsize=fontsize, fontweight="bold" if bold else "normal")

    def arr(x1, y1, x2, y2, label="", label_side="right"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.6))
        if label:
            xm = (x1+x2)/2 + (0.18 if label_side == "right" else -0.18)
            ym = (y1+y2)/2
            ax.text(xm, ym, label, ha="left" if label_side == "right" else "right",
                    va="center", fontsize=9.5, color="#555555")

    # Input box
    box(3.0, 7.8, 4.0, 0.85, "Input sentence  (tokenised)",
        fc="#E8F4FD", ec=HIGHLIGHT, lw=1.8)

    # Encoder box
    box(2.2, 5.6, 5.6, 1.8,
        "BiomedBERT Encoder",
        "12 transformer layers  —  one 768-dim vector per token",
        fc="#D6E8F8", ec=NEUTRAL, lw=2.0, bold=True)

    # Shared weights label
    ax.text(5.0, 5.45, "← shared weights for both tasks →",
            ha="center", fontsize=9.5, color=NEUTRAL, style="italic")

    # CLS head
    box(0.4, 3.0, 3.8, 1.6,
        "[CLS] classification head",
        "Linear(768 → 2)  →  P(interaction)",
        fc="#EAF5EA", ec=GOOD, lw=1.8)

    # NER head
    box(5.8, 3.0, 3.8, 1.6,
        "NER head  (per token)",
        "Linear(768 → 9)  →  BIO label",
        fc="#FFF3E0", ec=WARN, lw=1.8)

    # Output boxes
    box(0.7, 1.0, 3.2, 0.9, "YES / NO  + confidence score",
        fc="#EAF5EA", ec=GOOD)
    box(6.1, 1.0, 3.2, 0.9, "HOST / PATHOGEN / INT / O …",
        fc="#FFF3E0", ec=WARN)

    # Arrows
    arr(5.0, 7.8, 5.0, 7.4)
    arr(4.6, 5.6, 2.6, 4.6, "[CLS] token", "right")
    arr(5.4, 5.6, 7.4, 4.6, "All tokens", "right")
    arr(2.3, 3.0, 2.3, 1.9)
    arr(7.7, 3.0, 7.7, 1.9)

    ax.set_title("Multi-task BiomedBERT — Architecture",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(OUT / "fig_09_architecture.png", bbox_inches="tight")
    plt.close()
    print("fig_09_architecture.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 10 — Two-phase training diagram (matches slide 12)
# ══════════════════════════════════════════════════════════════════════════════
def fig_training_phases():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Two-Phase Training Strategy", fontsize=14, fontweight="bold", y=1.01)

    for ax in axes:
        ax.set_xlim(0, 5); ax.set_ylim(0, 5.5)
        ax.axis("off")

    def pbox(ax, x, y, w, h, line1, line2="", fc=LGREY, ec="#555555",
             frozen=False, lw=1.5):
        ls = "--" if frozen else "-"
        ec2 = BAD if frozen else ec
        lw2 = 2.2 if frozen else lw
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                    facecolor=fc, edgecolor=ec2, lw=lw2,
                                    linestyle=ls))
        ym = y + h / 2
        if line2:
            ax.text(x+w/2, ym+0.15, line1, ha="center", va="center", fontsize=10.5)
            ax.text(x+w/2, ym-0.18, line2, ha="center", va="center",
                    fontsize=9, color="#555555")
        else:
            ax.text(x+w/2, ym, line1, ha="center", va="center", fontsize=10.5)

    def parr(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.4))

    # ── Phase 1 ──────────────────────────────────────────────────
    ax = axes[0]
    ax.set_title("Phase 1 — NER pre-train (2 epochs)",
                 fontsize=12, fontweight="bold", color=NEUTRAL, pad=8)
    pbox(ax, 1.1, 4.2, 2.8, 0.9, "BiomedBERT encoder", fc="#D6E8F8", ec=NEUTRAL, lw=1.8)
    pbox(ax, 0.2, 2.5, 1.8, 1.1,
         "[CLS] head", "FROZEN", fc="#EEEEEE", frozen=True)
    pbox(ax, 3.0, 2.5, 1.8, 1.1,
         "NER head", "updated", fc="#FFF3E0", ec=WARN, lw=1.8)
    pbox(ax, 3.0, 0.8, 1.8, 0.9,
         "NER cross-entropy loss", fc="#FFE0B2", ec=WARN)
    parr(ax, 2.5, 4.2, 1.3, 3.6)
    parr(ax, 2.5, 4.2, 3.7, 3.6)
    parr(ax, 3.9, 2.5, 3.9, 1.7)
    ax.text(1.1, 3.0, "✗  no gradient", ha="center", fontsize=9,
            color=BAD, fontweight="bold")

    # ── Phase 2 ──────────────────────────────────────────────────
    ax = axes[1]
    ax.set_title("Phase 2 — Joint training (3 epochs)",
                 fontsize=12, fontweight="bold", color=GOOD, pad=8)
    pbox(ax, 1.1, 4.2, 2.8, 0.9, "BiomedBERT encoder", fc="#D6E8F8", ec=NEUTRAL, lw=1.8)
    pbox(ax, 0.2, 2.5, 1.8, 1.1,
         "[CLS] head", "updated", fc="#EAF5EA", ec=GOOD, lw=1.8)
    pbox(ax, 3.0, 2.5, 1.8, 1.1,
         "NER head", "updated", fc="#FFF3E0", ec=WARN, lw=1.8)
    pbox(ax, 0.2, 0.8, 1.8, 0.9,
         "KL soft loss  (α = 0.5)", fc="#C8E6C9", ec=GOOD)
    pbox(ax, 3.0, 0.8, 1.8, 0.9,
         "NER CE loss  (1−α = 0.5)", fc="#FFE0B2", ec=WARN)
    parr(ax, 2.5, 4.2, 1.3, 3.6)
    parr(ax, 2.5, 4.2, 3.7, 3.6)
    parr(ax, 1.1, 2.5, 1.1, 1.7)
    parr(ax, 3.9, 2.5, 3.9, 1.7)
    ax.text(2.5, 0.25, "ℒ = α · KL + (1−α) · CE_NER",
            ha="center", fontsize=10, color="#333333")

    fig.tight_layout()
    fig.savefig(OUT / "fig_10_training_phases.png", bbox_inches="tight")
    plt.close()
    print("fig_10_training_phases.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG: Manuscript CI — single panel, 3 models (paper figure)
# ══════════════════════════════════════════════════════════════════════════════
def fig_manuscript_ci():
    models  = ["Hard CE\n(same arch.)", "Multi-task\ncold-start", "Multi-task\nwarm-start (champion)",
               "BiomedBERT\n(template-trained)"]
    f1s     = [0.673, 0.835, 0.874, 0.875]
    ci_lo   = [0.621, 0.799, 0.843, 0.844]
    ci_hi   = [0.721, 0.867, 0.902, 0.903]

    x = np.arange(len(models))
    bar_colors = [NEUTRAL, NEUTRAL, GOOD, "#888888"]
    yerr_lo = [f - l for f, l in zip(f1s, ci_lo)]
    yerr_hi = [h - f for f, h in zip(f1s, ci_hi)]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(x, f1s, width=0.55, color=bar_colors,
                  edgecolor=WHITE, linewidth=1.2, zorder=3)
    ax.errorbar(x, f1s, yerr=[yerr_lo, yerr_hi],
                fmt="none", ecolor="#333333", capsize=7,
                capthick=1.5, linewidth=1.5, zorder=4)

    for i, (bar, val) in enumerate(zip(bars, f1s)):
        ypos = val + yerr_hi[i] + 0.008
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=11.5,
                fontweight="bold",
                color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10.5)
    ax.set_ylim(0.60, 0.96)
    ax.set_ylabel("F1", fontsize=12)
    ax.set_title("95% Bootstrap CIs — 500-sentence test set (n = 10,000 resamples)",
                 fontsize=12, pad=10)
    ax.yaxis.grid(True, linestyle=":", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    ax.text(0.5, -0.16,
            "Champion vs hard CE: p < 0.001 ✓   |   Champion vs cold-start: p = 0.002 ✓   |   "
            "Champion vs BiomedBERT (template-trained): p = 0.760 (n.s.)",
            transform=ax.transAxes, ha="center", fontsize=9,
            color="#555555", style="italic")

    fig.tight_layout()
    fig.savefig(OUT / "fig_manuscript_ci.png", bbox_inches="tight")
    plt.close()
    print("fig_manuscript_ci.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG: Threshold curve — champion on 499-sentence test set (paper figure)
# ══════════════════════════════════════════════════════════════════════════════
def fig_threshold_curve():
    # Hardcoded from champion (full_typed_a05_ner2) on held-out val set:
    # 300 positive + 700 negative sentences from v14 training data, not seen during training.
    # This is the curve on which τ=0.090 was selected.
    thresholds = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1, 0.105, 0.11, 0.115, 0.12, 0.125, 0.13, 0.135, 0.14, 0.145, 0.15, 0.155, 0.16, 0.165, 0.17, 0.175, 0.18, 0.185, 0.19, 0.195, 0.2, 0.205, 0.21, 0.215, 0.22, 0.225, 0.23, 0.235, 0.24, 0.245, 0.25, 0.255, 0.26, 0.265, 0.27, 0.275, 0.28, 0.285, 0.29, 0.295, 0.3, 0.305, 0.31, 0.315, 0.32, 0.325, 0.33, 0.335, 0.34, 0.345, 0.35, 0.355, 0.36, 0.365, 0.37, 0.375, 0.38, 0.385, 0.39, 0.395, 0.4, 0.405, 0.41, 0.415, 0.42, 0.425, 0.43, 0.435, 0.44, 0.445, 0.45, 0.455, 0.46, 0.465, 0.47, 0.475, 0.48, 0.485, 0.49, 0.495, 0.5, 0.505, 0.51, 0.515, 0.52, 0.525, 0.53, 0.535, 0.54, 0.545, 0.55, 0.555, 0.56, 0.565, 0.57, 0.575, 0.58, 0.585, 0.59, 0.595, 0.6, 0.605, 0.61, 0.615, 0.62, 0.625, 0.63, 0.635, 0.64, 0.645, 0.65, 0.655, 0.66, 0.665, 0.67, 0.675, 0.68, 0.685, 0.69, 0.695, 0.7, 0.705, 0.71, 0.715, 0.72, 0.725, 0.73, 0.735, 0.74, 0.745, 0.75, 0.755, 0.76, 0.765, 0.77, 0.775, 0.78, 0.785, 0.79, 0.795, 0.8, 0.805, 0.81, 0.815, 0.82, 0.825, 0.83, 0.835, 0.84, 0.845, 0.85, 0.855, 0.86, 0.865, 0.87, 0.875, 0.88, 0.885, 0.89, 0.895, 0.9, 0.905, 0.91, 0.915, 0.92, 0.925, 0.93, 0.935, 0.94, 0.945]
    precisions = [0.7971, 0.8254, 0.8274, 0.8338, 0.8349, 0.84, 0.8395, 0.8406, 0.8433, 0.8459, 0.8508, 0.8558, 0.8585, 0.8641, 0.8725, 0.8783, 0.8812, 0.8841, 0.887, 0.8867, 0.8863, 0.8855, 0.8881, 0.8942, 0.8942, 0.8973, 0.8969, 0.8966, 0.8966, 0.8997, 0.8997, 0.8997, 0.8997, 0.9028, 0.9028, 0.9028, 0.9028, 0.9024, 0.9024, 0.9056, 0.9088, 0.912, 0.9149, 0.9149, 0.9149, 0.9146, 0.9146, 0.9146, 0.9146, 0.9146, 0.9146, 0.9179, 0.9179, 0.9179, 0.9179, 0.9179, 0.9211, 0.9245, 0.9242, 0.9242, 0.9275, 0.9275, 0.9275, 0.9275, 0.9275, 0.9275, 0.9275, 0.9309, 0.9309, 0.9309, 0.9309, 0.9309, 0.9309, 0.9309, 0.9309, 0.9309, 0.9309, 0.9309, 0.9307, 0.9307, 0.9307, 0.9341, 0.9341, 0.9341, 0.9341, 0.9341, 0.9341, 0.9341, 0.9341, 0.9341, 0.9341, 0.9341, 0.9341, 0.9341, 0.9341, 0.9341, 0.9341, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375, 0.941, 0.941, 0.941, 0.941, 0.941, 0.9405, 0.9405, 0.9405, 0.9405, 0.9405, 0.9405, 0.9405, 0.944, 0.944, 0.944, 0.944, 0.944, 0.944, 0.944, 0.944, 0.9438, 0.9438, 0.9474, 0.9474, 0.9474, 0.9474, 0.9474, 0.9474, 0.9472, 0.9472, 0.9472, 0.947, 0.9506, 0.9506, 0.9506, 0.9506, 0.954, 0.954, 0.954, 0.954, 0.954, 0.954, 0.954, 0.954, 0.954, 0.954, 0.954, 0.954, 0.954, 0.954, 0.954, 0.954, 0.9577, 0.9575, 0.9575, 0.9575, 0.9575, 0.9575, 0.9574, 0.9611, 0.9611, 0.9611, 0.9611, 0.9611, 0.9611, 0.9611, 0.9611, 0.9611, 0.9611, 0.9611, 0.9611, 0.9611, 0.9611, 0.9611, 0.9611, 0.9611, 0.9611, 0.9611, 0.9648, 0.9648, 0.9648, 0.9648, 0.9648, 0.9647, 0.9685, 0.9684]
    recalls    = [0.93, 0.93, 0.9267, 0.92, 0.91, 0.91, 0.9067, 0.8967, 0.8967, 0.8967, 0.8933, 0.89, 0.89, 0.89, 0.89, 0.89, 0.89, 0.89, 0.89, 0.8867, 0.8833, 0.8767, 0.8733, 0.8733, 0.8733, 0.8733, 0.87, 0.8667, 0.8667, 0.8667, 0.8667, 0.8667, 0.8667, 0.8667, 0.8667, 0.8667, 0.8667, 0.8633, 0.8633, 0.8633, 0.8633, 0.8633, 0.86, 0.86, 0.86, 0.8567, 0.8567, 0.8567, 0.8567, 0.8567, 0.8567, 0.8567, 0.8567, 0.8567, 0.8567, 0.8567, 0.8567, 0.8567, 0.8533, 0.8533, 0.8533, 0.8533, 0.8533, 0.8533, 0.8533, 0.8533, 0.8533, 0.8533, 0.8533, 0.8533, 0.8533, 0.8533, 0.8533, 0.8533, 0.8533, 0.8533, 0.8533, 0.8533, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.8433, 0.8433, 0.8433, 0.8433, 0.8433, 0.8433, 0.8433, 0.8433, 0.8433, 0.8433, 0.8433, 0.8433, 0.8433, 0.8433, 0.8433, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.8367, 0.8367, 0.8367, 0.8333, 0.8333, 0.8333, 0.8333, 0.8333, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83, 0.8267, 0.8267, 0.8267, 0.8267, 0.8267, 0.8233, 0.8233, 0.8233, 0.8233, 0.8233, 0.8233, 0.8233, 0.8233, 0.8233, 0.8233, 0.8233, 0.8233, 0.8233, 0.8233, 0.8233, 0.8233, 0.8233, 0.8233, 0.8233, 0.8233, 0.8233, 0.8233, 0.8233, 0.8233, 0.8233, 0.82, 0.82, 0.8167]
    f1_scores  = [0.8585, 0.8746, 0.8742, 0.8748, 0.8708, 0.8736, 0.8718, 0.8677, 0.8691, 0.8706, 0.8715, 0.8725, 0.874, 0.8768, 0.8812, 0.8841, 0.8856, 0.887, 0.8885, 0.8867, 0.8848, 0.8811, 0.8807, 0.8836, 0.8836, 0.8851, 0.8832, 0.8814, 0.8814, 0.8829, 0.8829, 0.8829, 0.8829, 0.8844, 0.8844, 0.8844, 0.8844, 0.8825, 0.8825, 0.884, 0.8855, 0.887, 0.8866, 0.8866, 0.8866, 0.8847, 0.8847, 0.8847, 0.8847, 0.8847, 0.8847, 0.8862, 0.8862, 0.8862, 0.8862, 0.8862, 0.8877, 0.8893, 0.8873, 0.8873, 0.8889, 0.8889, 0.8889, 0.8889, 0.8889, 0.8889, 0.8889, 0.8904, 0.8904, 0.8904, 0.8904, 0.8904, 0.8904, 0.8904, 0.8904, 0.8904, 0.8904, 0.8904, 0.8885, 0.8885, 0.8885, 0.8901, 0.8901, 0.8901, 0.8901, 0.8901, 0.8901, 0.8901, 0.8901, 0.8901, 0.8901, 0.8901, 0.8901, 0.8901, 0.8901, 0.8901, 0.8901, 0.8916, 0.8916, 0.8916, 0.8916, 0.8916, 0.8916, 0.8932, 0.8932, 0.8932, 0.8932, 0.8932, 0.8893, 0.8893, 0.8893, 0.8893, 0.8893, 0.8893, 0.8893, 0.8908, 0.8908, 0.8908, 0.8908, 0.8908, 0.8908, 0.8908, 0.8908, 0.8889, 0.8889, 0.8905, 0.8905, 0.8905, 0.8905, 0.8905, 0.8905, 0.8885, 0.8885, 0.8885, 0.8865, 0.8881, 0.8881, 0.8881, 0.8881, 0.8877, 0.8877, 0.8877, 0.8877, 0.8877, 0.8877, 0.8877, 0.8877, 0.8877, 0.8877, 0.8877, 0.8877, 0.8877, 0.8877, 0.8877, 0.8877, 0.8893, 0.8873, 0.8873, 0.8873, 0.8873, 0.8873, 0.8853, 0.8869, 0.8869, 0.8869, 0.8869, 0.8869, 0.8869, 0.8869, 0.8869, 0.8869, 0.8869, 0.8869, 0.8869, 0.8869, 0.8869, 0.8869, 0.8869, 0.8869, 0.8869, 0.8869, 0.8885, 0.8885, 0.8885, 0.8885, 0.8885, 0.8865, 0.8881, 0.8861]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, precisions, color=NEUTRAL, linewidth=2.0, label="Precision")
    ax.plot(thresholds, recalls,    color=BAD,     linewidth=2.0, label="Recall")
    ax.plot(thresholds, f1_scores,  color=GOOD,    linewidth=2.5, label="F1",
            linestyle="-", zorder=5)

    # Annotate deployed threshold τ=0.090
    ax.axvline(0.090, color="#333333", linestyle="--", linewidth=1.4, alpha=0.8)
    ax.text(0.090 + 0.012, 0.775, "τ = 0.090\n(deployed)\nval F1 = 0.887\ntest F1 = 0.807",
            fontsize=8.5, color="#333333")

    ax.set_xlim(0.0, 0.96)
    ax.set_ylim(0.78, 1.00)
    ax.set_xlabel("Decision threshold τ", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision, Recall, F1 vs Decision Threshold\n"
                 "Champion model — validation set (300 pos + 700 neg, held out from training)",
                 fontsize=12, pad=10)
    ax.legend(fontsize=11, framealpha=0.9, loc="lower left")
    ax.yaxis.grid(True, linestyle=":", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUT / "fig_threshold_curve.png", bbox_inches="tight")
    plt.close()
    print("fig_threshold_curve.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG: Distribution mismatch — training vs full test set (paper figure)
# ══════════════════════════════════════════════════════════════════════════════
def fig_distribution_mismatch():
    # Same 6 categories, same order, on both panels for direct visual comparison.
    categories   = ["Predation", "Herbivory", "Pollination", "Parasitism/\nHost", "Pathogen/\nInfection", "Other"]
    # Training positives: 4,065 Qwen-labeled PMC-harvested sentences
    train_pcts   = [42, 22, 22, 10, 3, 1]
    # Full corrected test set positives: n=223, classified via GloBI interaction_term
    # (EP-A, EP-passage, BioTx-random/eval-100) and gen-set-100 category field.
    test_pcts    = [8.5, 3.6, 4.3, 41.6, 26.3, 15.7]
    # Color reflects training coverage: green = well-represented in training,
    # red = under-represented, grey = other/miscellaneous.
    cat_colors   = [GOOD, GOOD, GOOD, BAD, BAD, NEUTRAL]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Distribution Mismatch: Training vs Full Test-Set Interaction Types",
                 fontsize=13, fontweight="bold", y=1.01)

    # Training distribution
    b1 = ax1.bar(categories, train_pcts, color=cat_colors, edgecolor=WHITE,
                 linewidth=1.2, width=0.65, zorder=3)
    for bar, val in zip(b1, train_pcts):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.4,
                 f"{val}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 52)
    ax1.set_ylabel("Share of positives (%)", fontsize=11)
    ax1.set_title(f"Training positives\n(n = 4,065 Qwen-labeled sentences)", fontsize=11)
    ax1.yaxis.grid(True, linestyle=":", alpha=0.4, zorder=0)
    ax1.set_axisbelow(True)

    # Full test set distribution (same category order/colors)
    b2 = ax2.bar(categories, test_pcts, color=cat_colors, edgecolor=WHITE,
                 linewidth=1.2, width=0.65, zorder=3)
    for bar, val in zip(b2, test_pcts):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.4,
                 f"{val}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 52)
    ax2.set_title(f"Full test set positives\n(n = 281, all five sources)",
                  fontsize=11)
    ax2.yaxis.grid(True, linestyle=":", alpha=0.4, zorder=0)
    ax2.set_axisbelow(True)

    patches = [
        mpatches.Patch(color=GOOD,    label="Present in training corpus"),
        mpatches.Patch(color=BAD,     label="Under-represented in training"),
        mpatches.Patch(color=NEUTRAL, label="Partially represented"),
    ]
    fig.legend(handles=patches, fontsize=10, framealpha=0.9,
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))

    fig.tight_layout()
    fig.savefig(OUT / "fig_distribution_mismatch.png", bbox_inches="tight")
    plt.close()
    print("fig_distribution_mismatch.png")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures...")
    fig_version_f1()
    fig_model_evolution()
    fig_distillation_table()
    fig_ablation_table()
    fig_bootstrap_ci()
    fig_ceiling()
    fig_ner_schemes()
    fig_version_table()
    fig_architecture()
    fig_training_phases()
    # Manuscript-specific figures
    fig_manuscript_ci()
    fig_threshold_curve()
    fig_distribution_mismatch()

    print(f"\nAll saved to {OUT}/")
    for f in sorted(OUT.glob("*.png")):
        print(f"  {f.name}")
