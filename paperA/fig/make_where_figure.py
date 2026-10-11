#!/usr/bin/env python3
"""Figure 3 (fig:where): where the gain lives.

AUPRC of the three-checkpoint ensembles by how many entities the passage names, as a horizontal
dumbbell. Biodiversity benchmark (437 expert-graded candidates): distinct taxa that TaxoNERD finds
in the passage (2 or fewer, 3 or more), each split by whether both candidate arguments are taxa
TaxoNERD recognises there. BioRED (16,606 candidates): gold-annotated concepts in the sentence
(2, 3 or more); BioRED has no relation term, hence no triple-conditioned model. Open vermillion
square: sentence-level; filled blue circle: pair-conditioned; green diamond: triple-conditioned,
drawn above the circle; grey bar: the gain from asking about the pair. No value labels: the
numbers are in the appendix tables (tab:where, tab:biored) and in the JSON written next to the
figure.

Inputs (read only)
  * src/eval/core.py clean_benchmark(--labels): the 437 clean rows, hash-checked
  * <results-dir>/S_biodiv_{sentence,pair,triple}.npy: ensemble scores (scripts/paperA_tables.py)
  * results/paperA_rebuild_2026-09-28/n_taxa_per_row.csv and results/paperA_v2/arg_recognition.csv:
    TaxoNERD counts, row-aligned with the 437 clean rows (label-independent)
  * <results-dir>/derived_biodiv.json strata.*.{n, auprc}: the biodiversity values plotted
  * <results-dir>/tables_biored.json, else results/paperA_v2/tables_biored.json (BioRED does not
    depend on the biodiversity labels): {sentence,pair}.auprc_{2,ge3}_concepts, n_2_concepts,
    n_ge3_concepts

Checks (each failure exits non-zero)
  * the six biodiversity strata are recomputed from the scores, the labels and the TaxoNERD counts
    and must equal derived_biodiv.json to 1e-12;
  * under the current labels, every plotted value must round to the value the paper prints (three
    decimals) and every n must equal the printed n;
  * every piece of text must lie inside the figure, at 6 pt or more, and no two may overlap.
The serif face is the first of Times New Roman, Liberation Serif, Nimbus Roman, TeX Gyre Termes
and DejaVu Serif that has TrueType outlines: a CFF-outline face (Nimbus Roman and TeX Gyre Termes
ship as .otf) is written under pdf.fonttype 42 as a font whose embedded file does not match its
declared type, which PDF readers flag.

Outputs: <out-dir>/fig_where.pdf, fig_where.png (300 dpi preview) and fig_where.json (the plotted
values under the keys of the source files, and the gain each grey bar spans).

Usage
  python3 paperA/fig/make_where_figure.py
  python3 paperA/fig/make_where_figure.py --labels pre_review_2026-10-06 \\
      --results-dir results/paperA_v2/regression_2026-10-06_pre_review --out-dir <dir>
(the comparison with the printed values is made under the current labels only)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from fontTools.ttLib import TTFont  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from eval.core import LABEL_VERSIONS, benchmark_provenance, clean_benchmark  # noqa: E402

RESULTS = REPO / "results/paperA_v2"
FIGDIR = Path(__file__).resolve().parent
N_TAXA = REPO / "results/paperA_rebuild_2026-09-28/n_taxa_per_row.csv"
ARG_RECOGNITION = REPO / "results/paperA_v2/arg_recognition.csv"
STEM = "fig_where"
ARMS = ("sentence", "pair", "triple")
NAMES = {"sentence": "sentence-level", "pair": "pair-conditioned", "triple": "triple-conditioned"}

# Style: Okabe-Ito as in make_threshold_figure.py (blueprint section 4)
VERM, BLUE, GREEN = "#D55E00", "#0072B2", "#009E73"   # sentence-level, pair-, triple-conditioned
GREY, MUTED, INK, GRID = "#9a9a95", "#55554f", "#1a1a19", "#e8e8e4"
SERIF = ("Times New Roman", "Liberation Serif", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif")
TEXT_PT, SUB_PT, TICK_PT = 7.5, 7.0, 6.5               # headers and rows; sub-rows and legend; ticks
W, H = 3.03, 2.5           # inches; included at \columnwidth (3.03 in), so nothing is rescaled
PAD_L, PAD_R = 0.02, 0.09  # inches: left edge of the labels; right of the axes (half a tick label)
INDENT, GAP = 0.10, 0.07   # inches: sub-row indent; labels to axes
BOTTOM, TOP = 0.50, 0.02   # inches below and above the axes (ticks, axis label, legend)
GROUP_GAP = 0.35           # extra space between the two benchmarks, in row units
XTICKS = (0.6, 0.7, 0.8, 0.9, 1.0)
BAR_LW = 2.2
PNG_DPI = 300
MARK = {  # open square, filled circle, small diamond above the circle; white rings where they meet
    "sentence": dict(marker="s", ms=4.2, mfc="white", mec=VERM, mew=1.0, zorder=3),
    "pair": dict(marker="o", ms=4.8, mfc=BLUE, mec="white", mew=0.6, zorder=4),
    "triple": dict(marker="D", ms=3.4, mfc=GREEN, mec="white", mew=0.6, zorder=5),
}

# Rows, top to bottom
HEADERS = ("Biodiversity: taxa named in the passage", "BioRED: concepts named in the sentence")
BIODIV_ROWS = (("le2_taxa", "2 or fewer", 0),             # (derived_biodiv.json stratum, label, level)
               ("both_taxa_le2", "both arguments are taxa", 1),
               ("not_a_taxon_le2", "an argument is not a taxon", 1),
               ("ge3_taxa", "3 or more", 0),
               ("both_taxa_ge3", "both arguments are taxa", 1),
               ("not_a_taxon_ge3", "an argument is not a taxon", 1))
BIORED_ROWS = (("2_concepts", "2"), ("ge3_concepts", "3 or more"))  # tables_biored.json suffix, label

# What the paper prints (tab:where, tab:biored; blueprint 4.3): n, then sentence, pair, triple AUPRC.
PRINTED = {
    "le2_taxa": (161, "0.896", "0.939", "0.930"),
    "both_taxa_le2": (33, "0.958", "0.984", "0.976"),
    "not_a_taxon_le2": (128, "0.876", "0.925", "0.915"),
    "ge3_taxa": (276, "0.857", "0.955", "0.963"),
    "both_taxa_ge3": (227, "0.891", "0.975", "0.982"),
    "not_a_taxon_ge3": (49, "0.628", "0.808", "0.839"),
    "2_concepts": (538, "0.909", "0.902", None),
    "ge3_concepts": (16068, "0.727", "0.847", None),
}


def fail(msg: str) -> None:
    sys.exit(f"make_where_figure: {msg}")


def rel(p: Path) -> str:
    """A path relative to the classifier directory, so no absolute path reaches the JSON."""
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return p.name


# Data
def recompute_strata(labels: str, res_dir: Path) -> dict:
    """Ensemble AUPRC per stratum from the scores, the labels and the TaxoNERD counts, as
    scripts/paperA_derived_numbers.py computes it."""
    d = clean_benchmark(labels)
    y = d.label.to_numpy()
    nt, ar = pd.read_csv(N_TAXA), pd.read_csv(ARG_RECOGNITION)
    if not (len(nt) == len(ar) == len(d)) or not (nt.species1.to_numpy() == d.species1.to_numpy()).all() \
            or not (ar.n_taxa.to_numpy() == nt.n_taxa.to_numpy()).all() or ar.both_args_taxa.dtype != bool:
        fail("the TaxoNERD counts are not row-aligned with the clean benchmark")
    multi = nt.n_taxa.to_numpy() >= 3
    both = ar.both_args_taxa.to_numpy()
    masks = {"le2_taxa": ~multi, "ge3_taxa": multi,
             "both_taxa_le2": both & ~multi, "both_taxa_ge3": both & multi,
             "not_a_taxon_le2": ~both & ~multi, "not_a_taxon_ge3": ~both & multi}
    S = {arm: np.load(res_dir / f"S_biodiv_{arm}.npy") for arm in ARMS}
    if any(len(s) != len(y) for s in S.values()):
        fail(f"score vectors in {rel(res_dir)} do not cover the {len(y)} clean rows")
    return {key: {"n": int(m.sum()),
                  "auprc": {arm: float(average_precision_score(y[m], S[arm][m])) for arm in ARMS}}
            for key, m in masks.items()}


def load(labels: str, res_dir: Path) -> tuple[list[dict], dict]:
    """The plotted rows, and the JSON record of where they come from."""
    prov = benchmark_provenance(labels)
    derived_path = res_dir / "derived_biodiv.json"
    derived = json.loads(derived_path.read_text())
    if derived["benchmark"]["sha256"] != prov["sha256"]:
        fail(f"{rel(derived_path)} was not computed on the {labels} labels; run "
             f"scripts/paperA_derived_numbers.py --labels {labels} --results-dir {rel(res_dir)} first")
    strata = {key: derived["strata"][key] for key, _, _ in BIODIV_ROWS}
    again = recompute_strata(labels, res_dir)
    for key, got in again.items():
        ref = strata[key]
        if got["n"] != ref["n"] or any(abs(got["auprc"][a] - ref["auprc"][a]) > 1e-12 for a in ARMS):
            fail(f"stratum {key}: recomputed {got} differs from {rel(derived_path)} {ref}")

    biored_path = res_dir / "tables_biored.json"
    if not biored_path.exists():
        biored_path = RESULTS / "tables_biored.json"
    br = json.loads(biored_path.read_text())
    if br["n_2_concepts"] + br["n_ge3_concepts"] != br["n"]:
        fail(f"{rel(biored_path)}: concept strata do not partition the {br['n']} candidates")

    rows = []
    for key, text, level in BIODIV_ROWS:
        s = strata[key]
        rows.append({"group": "biodiversity", "key": key, "level": level, "n": int(s["n"]),
                     "label": f"{text} ({s['n']:,})", **{arm: float(s["auprc"][arm]) for arm in ARMS}})
    for key, text in BIORED_ROWS:
        n = int(br[f"n_{key}"])
        rows.append({"group": "biored", "key": key, "level": 0, "n": n, "label": f"{text} ({n:,})",
                     "sentence": float(br["sentence"][f"auprc_{key}"]),
                     "pair": float(br["pair"][f"auprc_{key}"]), "triple": None})
    for r in rows:
        r["gain_pair_vs_sentence"] = r["pair"] - r["sentence"]

    record = {
        "figure": f"{STEM}.pdf", "label": "fig:where", "script": "paperA/fig/make_where_figure.py",
        "what": "AUPRC of the three-checkpoint ensembles by how many entities the passage names; "
                "grey bar from the sentence-level to the pair-conditioned value",
        "benchmark": prov,
        "sources": {"biodiversity": f"{rel(derived_path)} strata.*.{{n, auprc}}",
                    "biodiversity_recomputed_from": [f"{rel(res_dir)}/S_biodiv_{{{','.join(ARMS)}}}.npy",
                                                     "clean_benchmark() labels", rel(N_TAXA),
                                                     rel(ARG_RECOGNITION)],
                    "biored": f"{rel(biored_path)} {{sentence,pair}}.auprc_{{2,ge3}}_concepts, "
                              "n_2_concepts, n_ge3_concepts"},
        "strata": {key: {"n": int(s["n"]), "auprc": {arm: float(s["auprc"][arm]) for arm in ARMS},
                         "gain_pair_vs_sentence": float(s["auprc"]["pair"] - s["auprc"]["sentence"])}
                   for key, s in strata.items()},
        "biored": {"n": int(br["n"]), "n_2_concepts": int(br["n_2_concepts"]),
                   "n_ge3_concepts": int(br["n_ge3_concepts"]),
                   **{arm: {f"auprc_{k}": float(br[arm][f"auprc_{k}"]) for k, _ in BIORED_ROWS}
                      for arm in ("sentence", "pair")},
                   "gain_pair_vs_sentence": {k: float(br["pair"][f"auprc_{k}"] - br["sentence"][f"auprc_{k}"])
                                             for k, _ in BIORED_ROWS}},
        "rows_top_to_bottom": [{k: r[k] for k in ("group", "key", "label", "n", *ARMS)} for r in rows],
    }
    return rows, record


def check_printed(rows: list[dict]) -> None:
    """Every plotted value must round to what the paper prints; every n must be the printed n."""
    bad = []
    for r in rows:
        n, *want = PRINTED[r["key"]]
        if r["n"] != n:
            bad.append(f"{r['key']}: n {r['n']} against printed {n}")
        for arm, w in zip(ARMS, want):
            got = r[arm]
            if (w is None) != (got is None) or (w is not None and f"{got:.3f}" != w):
                bad.append(f"{r['key']} {arm}: {got} against printed {w}")
    if bad:
        fail("plotted values differ from the paper's:\n  " + "\n  ".join(bad))


# Drawing
def truetype_serif() -> list[str]:
    """SERIF, keeping only the families whose regular face has TrueType outlines (a font cache
    built before Liberation Serif was installed does not list it, so it is added here)."""
    fm = font_manager.fontManager
    if "Liberation Serif" not in {f.name for f in fm.ttflist}:
        for p in font_manager.findSystemFonts(fontext="ttf"):
            if Path(p).name.startswith("LiberationSerif-") and Path(p).suffix.lower() == ".ttf":
                fm.addfont(p)
    keep = []
    for fam in SERIF:
        try:
            path = font_manager.findfont(font_manager.FontProperties(family=fam), fallback_to_default=False)
        except ValueError:
            continue
        try:
            if "glyf" in TTFont(path, lazy=True):
                keep.append(fam)
        except Exception:  # an unreadable or collection file: not usable here
            continue
    return keep


def style() -> str:
    serif = truetype_serif()
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,   # TrueType, not Type 3: the ACL checker flags Type 3
        "font.family": "serif", "font.serif": serif, "font.size": TEXT_PT,
        "axes.linewidth": 0.6, "axes.edgecolor": MUTED, "figure.dpi": 100,
        "savefig.facecolor": "white",
    })
    return serif[0]


def positions() -> dict:
    """y of each header and row, top to bottom: one unit per line, a short gap between groups."""
    ys, y = {}, 0.0
    ys["h0"] = y
    for key, _, _ in BIODIV_ROWS:
        y -= 1
        ys[key] = y
    y -= 1 + GROUP_GAP
    ys["h1"] = y
    for key, _ in BIORED_ROWS:
        y -= 1
        ys[key] = y
    return ys


def draw(rows: list[dict]):
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0.5, BOTTOM / H, 0.4, (H - BOTTOM - TOP) / H))   # left edge set below
    ys = positions()
    y_bottom = ys[BIORED_ROWS[-1][0]] - 0.65
    ax.set_ylim(y_bottom, ys["h0"] + 0.45)
    ax.set_xlim(XTICKS[0], XTICKS[-1])

    # labels: x in figure inches from the left edge, y on the rows
    on_rows = blended_transform_factory(fig.transFigure, ax.transData)
    for i, text in enumerate(HEADERS):
        ax.text(PAD_L / W, ys[f"h{i}"], text, transform=on_rows, ha="left", va="center",
                fontsize=TEXT_PT, fontweight="bold", color=INK)
    row_labels = []
    for r in rows:
        sub = r["level"] == 1
        row_labels.append(ax.text((PAD_L + (INDENT if sub else 0)) / W, ys[r["key"]], r["label"],
                                  transform=on_rows, ha="left", va="center",
                                  fontsize=SUB_PT if sub else TEXT_PT, color=MUTED if sub else INK))
    fig.canvas.draw()
    right = max(t.get_window_extent().x1 for t in row_labels) / fig.dpi        # inches
    left = right + GAP
    ax.set_position((left / W, BOTTOM / H, (W - PAD_R - left) / W, (H - BOTTOM - TOP) / H))

    # light vertical grid, interrupted at the group headers
    first, last = BIODIV_ROWS[0][0], BIODIV_ROWS[-1][0]
    ax.vlines(XTICKS, ys[last] - 0.5, ys[first] + 0.5, color=GRID, lw=0.55, zorder=0)
    ax.vlines(XTICKS, y_bottom, ys[BIORED_ROWS[0][0]] + 0.5, color=GRID, lw=0.55, zorder=0)

    for r in rows:
        y = ys[r["key"]]
        ax.plot([r["sentence"], r["pair"]], [y, y], color=GREY, lw=BAR_LW, solid_capstyle="butt",
                zorder=2, gid=f"gain:{r['key']}")
        for arm in ARMS:
            if r[arm] is not None:
                ax.plot([r[arm]], [y], ls="none", gid=f"{arm}:{r['key']}", **MARK[arm])

    ax.set_xticks(XTICKS)
    ax.set_xticklabels([f"{t:.1f}" for t in XTICKS])
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", labelsize=TICK_PT, length=2.5, width=0.6, color=MUTED, labelcolor=INK, pad=1.5)
    ax.set_xlabel("AUPRC", fontsize=TEXT_PT, color=INK, labelpad=2)

    handles = [Line2D([], [], ls="none", **{k: v for k, v in MARK[arm].items() if k != "zorder"})
               for arm in ARMS]
    fig.legend(handles, [NAMES[arm] for arm in ARMS], loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=3, frameon=False, fontsize=SUB_PT, handlelength=1.0, handletextpad=0.35,
               columnspacing=1.4, borderaxespad=0.15, labelcolor=INK)
    return fig, ax, row_labels


def check_layout(fig, ax, row_labels) -> None:
    """All text inside the figure, at 6 pt or more, no two pieces overlapping; row labels left of
    the axes; the legend clear of the axis text."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fb = fig.bbox
    legend = fig.legends[0]
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    bad, boxes = [], []
    for t in texts:
        bb = t.get_window_extent(renderer)
        if bb.x0 < fb.x0 - 0.5 or bb.y0 < fb.y0 - 0.5 or bb.x1 > fb.x1 + 0.5 or bb.y1 > fb.y1 + 0.5:
            bad.append(f"cut off: {t.get_text()!r}")
        if t.get_fontsize() < 6:
            bad.append(f"below 6 pt: {t.get_text()!r}")
        if t not in legend.get_texts():
            boxes.append((repr(t.get_text()), bb))
    boxes.append(("the legend", legend.get_window_extent(renderer)))
    for i, (a, ba) in enumerate(boxes):
        for b, bb in boxes[i + 1:]:
            if ba.overlaps(bb):
                bad.append(f"overlap: {a} and {b}")
    axes_left = ax.get_window_extent(renderer).x0
    if max(t.get_window_extent(renderer).x1 for t in row_labels) >= axes_left:
        bad.append("a row label reaches into the axes")
    if bad:
        fail("layout:\n  " + "\n  ".join(bad))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", choices=sorted(LABEL_VERSIONS), default="current",
                    help="label version of the benchmark (src/eval/core.py LABEL_VERSIONS)")
    ap.add_argument("--results-dir", default=str(RESULTS),
                    help="where derived_biodiv.json and the score vectors are")
    ap.add_argument("--out-dir", default=str(FIGDIR), help="where the figure and its JSON are written")
    a = ap.parse_args()
    res_dir = Path(a.results_dir)
    if not res_dir.is_absolute():
        res_dir = REPO / res_dir
    out_dir = Path(a.out_dir)
    if a.labels != "current" and out_dir.resolve() == FIGDIR:
        fail("the paper's figure uses the current labels; give --out-dir for another label version")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, record = load(a.labels, res_dir)
    if a.labels == "current":
        check_printed(rows)
    record["checked_against_printed_values"] = a.labels == "current"

    face = style()
    fig, ax, row_labels = draw(rows)
    check_layout(fig, ax, row_labels)
    fig.savefig(out_dir / f"{STEM}.pdf",
                metadata={"Creator": None, "Producer": None, "CreationDate": None, "Author": None, "Title": None})
    fig.savefig(out_dir / f"{STEM}.png", dpi=PNG_DPI, metadata={"Software": None})
    plt.close(fig)
    (out_dir / f"{STEM}.json").write_text(json.dumps(record, indent=1) + "\n")

    for r in rows:
        tri = "  n/a" if r["triple"] is None else f"{r['triple']:.3f}"
        print(f"{r['group']:12s} {r['label']:34s} sentence {r['sentence']:.3f}  pair {r['pair']:.3f}  "
              f"triple {tri}  gain {r['gain_pair_vs_sentence']:+.3f}")
    print(f"{'checked against the printed values' if a.labels == 'current' else 'not checked against the paper'};"
          f" serif face: {face}")
    print(f"wrote {out_dir / STEM}.{{pdf,png,json}}")


if __name__ == "__main__":
    main()
