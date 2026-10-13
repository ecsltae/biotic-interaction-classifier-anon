#!/usr/bin/env python3
r"""Redrawn figures for the pair-conditioned verification paper.

    python paper_figures.py fig3                     # Figure 3, from the paper's Tables 4 and 5
    python paper_figures.py fig2 --scores rows.csv   # Figure 2, from your per-row scores
    python paper_figures.py fig2 --demo              # Figure 2 layout preview on synthetic scores

rows.csv: one row per benchmark candidate (the 437 rows), with columns
    gold       1 if the expert grade is positive, else 0
    sentence   sentence-level score (three-checkpoint ensemble)
    pair       pair-conditioned score, order-free (max over the two argument orders)
    triple     triple-conditioned score, order-free

Every figure is sized for one ACL column (7.7 cm) and saved as PDF (fonts embedded
as TrueType, which ACL requires) plus a 300-dpi PNG preview. In LaTeX:
    \includegraphics[width=\columnwidth]{fig3_gain_by_taxa.pdf}

Series colours are the paper's existing Okabe-Ito hues. They pass a colour-blind
check on every pair on a white page (worst deuteranopia dE 11.0, all >= 3:1 contrast),
and each series also has its own marker and line style, so the figures survive
greyscale printing.
"""
import argparse
import csv
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

COL_W = 7.7 / 2.54  # ACL column width, inches

SENT, PAIR, TRIP = "#d55e00", "#0072b2", "#009e73"  # sentence / pair / triple
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"  # text: primary / secondary / muted
GRID, AXIS, GAIN = "#e1e0d9", "#c3c2b7", "#bdbbb3"  # hairline grid / axis / gain bar

STYLE = {
    "sentence": dict(color=SENT, ls=(0, (4.5, 1.8)), marker="s", mfc="white", mec=SENT, mew=1.1, ms=4.4),
    "pair":     dict(color=PAIR, ls="-",              marker="o", mfc=PAIR,    mec="white", mew=0.7, ms=5.0),
    "triple":   dict(color=TRIP, ls=(0, (1.2, 1.4)),  marker="D", mfc=TRIP,    mec="white", mew=0.6, ms=4.0),
}
NAMES = {"sentence": "sentence-level", "pair": "pair-conditioned", "triple": "triple-conditioned"}

from matplotlib import font_manager  # noqa: E402
for _f in font_manager.findSystemFonts(fontext="ttf"):  # register Liberation Serif (TrueType; TeX Gyre is CFF)
    if "LiberationSerif" in _f:
        font_manager.fontManager.addfont(_f)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "TeX Gyre Termes", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7.6,
    "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
    "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300,
})


def save(fig, stem, pdf=True):
    if pdf:
        fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png")
    print("wrote", f"{stem}.pdf and .png" if pdf else f"{stem}.png")


def legend_handles(keys):
    return [Line2D([], [], **{k: v for k, v in STYLE[key].items()}, lw=1.4) for key in keys], \
           [NAMES[key] for key in keys]


# --------------------------------------------------------------------------------------
# Figure 3: AUPRC by number of entities in the passage
# Values are three-checkpoint ensembles: Table 4 (biodiversity) and Table 5 (BioRED).
# "pos" is optional: the share of positives in the row. Fill it in from your per-row
# table and each label shows it, so readers can judge AUPRC against the row's base rate.
# --------------------------------------------------------------------------------------
FIG3_ROWS = [
    dict(kind="head", label="TaxaPairs: taxa named in the passage"),
    dict(kind="group", label="2 or fewer", n=161, sentence=0.896, pair=0.939, triple=0.930, pos=None),
    dict(kind="sub", label="both arguments are taxa", n=33, sentence=0.958, pair=0.984, triple=0.976, pos=None),
    dict(kind="sub", label="an argument is not a taxon", n=128, sentence=0.876, pair=0.925, triple=0.915, pos=None),
    dict(kind="group", label="3 or more", n=276, sentence=0.857, pair=0.955, triple=0.963, pos=None),
    dict(kind="sub", label="both arguments are taxa", n=227, sentence=0.891, pair=0.975, triple=0.982, pos=None),
    dict(kind="sub", label="an argument is not a taxon", n=49, sentence=0.628, pair=0.808, triple=0.839, pos=None),
    dict(kind="head", label="BioRED: concepts named in the sentence"),
    dict(kind="group", label="2", n=538, sentence=0.909, pair=0.902, triple=None, pos=None),
    dict(kind="group", label="3 or more", n=16068, sentence=0.727, pair=0.847, triple=None, pos=None),
]


STATS_KEYS = ["le2_taxa", "both_taxa_le2", "not_a_taxon_le2", "ge3_taxa", "both_taxa_ge3",
              "not_a_taxon_ge3", "2_concepts", "ge3_concepts"]  # data rows of FIG3_ROWS, in order


def load_fig3_stats(path):
    """pos and bootstrap 95% intervals from paperA/fig/fig3_stats.py; point values must match."""
    import json
    st = {r["key"]: r for r in json.load(open(path))["rows"]}
    for r, k in zip([r for r in FIG3_ROWS if r["kind"] != "head"], STATS_KEYS):
        assert st[k]["n"] == r["n"], k
        for arm in ("sentence", "pair", "triple"):
            if r[arm] is not None:
                assert round(st[k]["auprc"][arm], 3) == r[arm], (k, arm)
        r["pos"], r["ci"] = st[k]["pos"], st[k]["ci95"]


def fig3(stem="fig3_gain_by_taxa", stats=None):
    if stats:
        load_fig3_stats(stats)
    XLO = 0.4 if stats else 0.6   # room for the widest interval (0.45 on 49 rows)
    fig = plt.figure(figsize=(COL_W, 2.5))
    ax = fig.add_axes([0.5, 0.13, 0.37, 0.795])  # x-extent is fitted to the labels below
    lab = blended_transform_factory(fig.transFigure, ax.transData)

    # vertical layout: one unit per row, extra air before the second header
    ys, y = [], 0.0
    for i, r in enumerate(FIG3_ROWS):
        if r["kind"] == "head" and i > 0:
            y += 0.45
        ys.append(y)
        y += 1.0

    # hairline grid only behind data rows (headers stay clean)
    spans, start = [], None
    for r, yy in zip(FIG3_ROWS + [dict(kind="head")], ys + [y]):
        if r["kind"] == "head":
            if start is not None:
                spans.append((start - 0.55, prev + 0.55))
            start = None
        else:
            start = yy if start is None else start
            prev = yy
    for lo, hi in spans:
        ax.vlines(np.arange(XLO, 1.01, 0.2), lo, hi, color=GRID, lw=0.5, zorder=0)

    dodge = 0.27  # triple marker sits just below its row so it never hides the pair marker
    labels, deltas = [], []
    for r, yy in zip(FIG3_ROWS, ys):
        if r["kind"] == "head":
            fig.text(0.012, yy, r["label"], transform=lab, va="center", ha="left",
                     fontsize=8, fontweight="bold")
            continue
        s, p, t = r["sentence"], r["pair"], r["triple"]
        ci = r.get("ci") or {}
        for arm, v, off in (("sentence", s, -0.27), ("pair", p, 0.0), ("triple", t, dodge)):
            if v is not None and arm in ci and r["n"] < 100:   # intervals drawn for the small rows
                lo, hi = ci[arm]
                ax.plot([lo, hi], [yy + off] * 2, color=STYLE[arm]["color"], lw=0.7, zorder=2,
                        solid_capstyle="butt")
        ax.plot([s, p], [yy, yy], color=GAIN, lw=2.6, solid_capstyle="butt", zorder=1)
        st = {k: v for k, v in STYLE["sentence"].items() if k not in ("ls", "color")}
        ax.plot(s, yy, ls="none", zorder=3, **st)
        st = {k: v for k, v in STYLE["pair"].items() if k not in ("ls", "color")}
        ax.plot(p, yy, ls="none", zorder=4, **st)
        if t is not None:
            st = {k: v for k, v in STYLE["triple"].items() if k not in ("ls", "color")}
            ax.plot(t, yy + dodge, ls="none", zorder=4, **st)

        extra = f", {r['pos']:.0%}" if r.get("pos") is not None else ""
        text = f"{r['label']} ({r['n']:,}{extra})"
        group = r["kind"] == "group"
        labels.append(fig.text(0.03 if group else 0.058, yy, text, transform=lab, va="center",
                               ha="left", fontsize=7.9 if group else 7.4,
                               color=INK if group else INK2))
        d = p - s
        deltas.append(fig.text(0.992, yy, f"{d:+.3f}".replace("-", "−"), transform=lab,
                               va="center", ha="right", fontsize=7.6 if group else 7.3,
                               color=INK if group else INK2))

    fig.text(0.992, ys[0], "Δ", transform=lab, va="center", ha="right", fontsize=8, color=INK2)

    # fit the plot between the longest row label and the delta column
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    w = fig.bbox.width
    left = max(t.get_window_extent(rend).x1 for t in labels) / w + 0.03
    right = min(t.get_window_extent(rend).x0 for t in deltas) / w - 0.035
    pos = ax.get_position()
    ax.set_position([left, pos.y0, right - left, pos.height])

    ax.set_xlim(XLO, 1.0)
    ax.set_ylim(y - 0.4, -0.6)
    ax.set_xticks(np.arange(XLO, 1.01, 0.2))
    ax.set_xticklabels([f"{v:.1f}" for v in np.arange(XLO, 1.01, 0.2)])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("AUPRC", labelpad=2)

    h, l = legend_handles(["sentence", "pair", "triple"])
    for handle in h:
        handle.set_linestyle("none")
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False,
               handletextpad=0.25, columnspacing=1.1, borderaxespad=0.2)
    save(fig, stem)
    return fig


# --------------------------------------------------------------------------------------
# Figure 2: operating curves that do not depend on calibration
#   (a) precision against recall, markers at the Table 1 operating points
#   (b) F1 against the share of candidates accepted, i.e. at equal review workload
# --------------------------------------------------------------------------------------
# Table 1 operating points (block-held-out thresholds): recall, precision
OPERATING = {"sentence": (0.833, 0.752), "pair": (0.896, 0.869), "triple": (0.892, 0.842)}


def curves(y, s):
    o = np.argsort(-s, kind="mergesort")
    y, s = y[o], s[o]
    tp = np.cumsum(y).astype(float)
    k = np.arange(1, len(y) + 1, dtype=float)
    last = np.r_[s[1:] != s[:-1], True]  # treat tied scores as one cut
    tp, k = tp[last], k[last]
    pos = y.sum()
    return dict(precision=tp / k, recall=tp / pos, f1=2 * tp / (k + pos), share=k / len(y))


def load_scores(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    need = {"gold", "sentence", "pair", "triple"}
    missing = need - set(rows[0])
    if missing:
        sys.exit(f"{path}: missing columns {sorted(missing)}")
    gold = np.array([int(float(r["gold"])) for r in rows])
    return gold, {m: np.array([float(r[m]) for r in rows]) for m in ("sentence", "pair", "triple")}


def synthetic_scores(seed=7):
    """Made-up scores with roughly the benchmark's size and base rate. Layout only."""
    rng = np.random.default_rng(seed)
    n, n_pos = 437, 251
    gold = np.r_[np.ones(n_pos, int), np.zeros(n - n_pos, int)]
    rng.shuffle(gold)

    def model(sep, bias, temp):
        z = rng.normal(0, 1, n) + sep * (2 * gold - 1)
        return 1 / (1 + np.exp(-(z - bias) / temp))

    return gold, {"sentence": model(0.95, 0.0, 1.0), "pair": model(1.6, 1.8, 0.8),
                  "triple": model(1.55, 1.5, 0.7)}


def best_f1_point(c):
    i = int(np.argmax(c["f1"]))
    return c["recall"][i], c["precision"][i]


def fig2(gold, scores, stem, demo=False):
    fig = plt.figure(figsize=(COL_W, 4.05))
    axa = fig.add_axes([0.155, 0.555, 0.81, 0.36])
    axb = fig.add_axes([0.155, 0.085, 0.81, 0.36])
    base = gold.mean()
    order = ["sentence", "pair", "triple"]
    cur = {m: curves(gold, scores[m]) for m in order}

    for m in order:
        st = STYLE[m]
        lw = 1.5 if m == "pair" else 1.3
        axa.plot(cur[m]["recall"], cur[m]["precision"], color=st["color"], ls=st["ls"], lw=lw,
                 zorder=3 if m == "pair" else 2)
        axb.plot(cur[m]["share"], cur[m]["f1"], color=st["color"], ls=st["ls"], lw=lw,
                 zorder=3 if m == "pair" else 2)
        mk = {k: v for k, v in st.items() if k not in ("ls", "color")}
        # Table 1 operating points use block-held-out thresholds, so they can sit slightly
        # off the pooled curve. The demo has no such points; it marks each curve's best F1.
        rec, prec = best_f1_point(cur[m]) if demo else OPERATING[m]
        f1 = 2 * prec * rec / (prec + rec)
        share = rec * base / prec  # candidates accepted / all candidates
        axa.plot(rec, prec, ls="none", zorder=5, **mk)
        axb.plot(share, f1, ls="none", zorder=5, **mk)

    # (a) precision-recall
    axa.axhline(base, color=MUTED, lw=0.9, ls=(0, (1, 1.5)), zorder=1)
    axa.text(0.01, base + 0.012, "accept everything", color=INK2, fontsize=7.2, va="bottom")
    axa.text(0.01, 0.835, "database ingestion", color=MUTED, fontsize=7.2, style="italic")
    axa.text(0.88, 0.645, "expert review", color=MUTED, fontsize=7.2, style="italic", ha="right")
    axa.set_xlim(0, 1.0)
    axa.set_ylim(0.5, 1.01)
    axa.set_xlabel("recall", labelpad=1.5)
    axa.set_ylabel("precision", labelpad=2)

    # (b) F1 at equal workload
    axb.axvline(base, color=MUTED, lw=0.9, ls=(0, (1, 1.5)), zorder=1)
    axb.text(base + 0.012, 0.05, "share of positives", color=INK2, fontsize=7.2, va="bottom")
    axb.set_xlim(0, 1.0)
    axb.set_ylim(0, 1.0)
    axb.set_xlabel("share of candidates accepted", labelpad=1.5)
    axb.set_ylabel("F1", labelpad=2)

    for ax in (axa, axb):
        ax.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_xticks(np.arange(0, 1.01, 0.2))
    fig.text(0.012, 0.918, "(a)", fontweight="bold", fontsize=8, va="bottom")
    fig.text(0.012, 0.448, "(b)", fontweight="bold", fontsize=8, va="bottom")

    h, l = legend_handles(order)
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
               fontsize=7.2, handlelength=1.9, handletextpad=0.3, columnspacing=0.75,
               borderaxespad=0.2)

    if demo:
        fig.text(0.5, 0.5, "SYNTHETIC SCORES\nlayout preview only", rotation=28, fontsize=19,
                 fontweight="bold", color="#9c9a92", alpha=0.35, ha="center", va="center",
                 zorder=20, linespacing=1.2)
    save(fig, stem, pdf=not demo)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figure", choices=["fig2", "fig3"])
    ap.add_argument("--scores", help="per-row CSV for fig2 (see header)")
    ap.add_argument("--demo", action="store_true", help="fig2 on synthetic scores, watermarked")
    ap.add_argument("--stats", help="fig3: JSON from paperA/fig/fig3_stats.py (pos, 95% intervals)")
    ap.add_argument("--stem", help="output path without extension")
    a = ap.parse_args()
    if a.figure == "fig3":
        fig3(a.stem or "fig3_gain_by_taxa", a.stats)
    elif a.demo:
        fig2(*synthetic_scores(), stem="fig2_LAYOUT_PREVIEW_synthetic", demo=True)
    elif a.scores:
        fig2(*load_scores(a.scores), stem="fig2_operating_curves")
    else:
        ap.error("fig2 needs --scores rows.csv (or --demo)")


if __name__ == "__main__":
    main()
