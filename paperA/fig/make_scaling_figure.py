#!/usr/bin/env python3
"""Figure 4 (fig:scaling): the same gap in zero-shot language models.

AUPRC of zero-shot Qwen models, by size, asked the sentence question or the pair question, on
(a) the 437 expert-graded candidates and (b) the 3,000-candidate BioRED sample, with the trained
110M models (three-checkpoint ensembles) on the same rows and the base rate. Built to the
restructure blueprint, section 4.4; style as make_threshold_figure.py (Okabe-Ito, serif, one
column, TrueType fonts).

Inputs, all read only:
  results/paperA_v2/tables_biodiv.json    llm["biodiv_{m}_{q}"].auprc; sentence.auprc_ensemble,
                                          pair.auprc_ensemble; base_rate
  results/paperA_v2/tables_biored.json    llm["biored_{m}_{q}"].auprc and .trained_arms_same_rows
  results/paperA_v2/derived_biodiv.json   scaling_sentence_auprc, scaling_pair_minus_sentence
  results/paperA_v2/llm/*.csv             per-candidate p_yes; BioRED sample labels (base rates)
  results/paperA_v2/S_{biodiv,biored}_{sentence,pair}.npy   trained ensembles' scores
  src/eval/core.py clean_benchmark()      biodiversity labels (hash-checked)

Every plotted value is taken from the tables JSONs, rescored from the per-candidate files (they
must agree to 1e-9) and checked against the three-decimal values the paper prints (tab:scaling);
any mismatch exits non-zero. Outputs, next to this script unless --out-dir says otherwise:
fig_scaling.pdf (3.03 x 3.0 in, TrueType fonts, no metadata), fig_scaling.png (300 dpi preview)
and fig_scaling.json (every value drawn or stated in the caption, under the blueprint's keys).

Usage
  python3 paperA/fig/make_scaling_figure.py
  python3 paperA/fig/make_scaling_figure.py --labels pre_review_2026-10-06 --out-dir <dir>
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.legend_handler import HandlerBase  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, MultipleLocator  # noqa: E402
from sklearn.metrics import average_precision_score as ap  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from eval.core import LABEL_VERSIONS, benchmark_provenance, clean_benchmark  # noqa: E402

RESULTS = REPO / "results/paperA_v2"
FIGDIR = Path(__file__).resolve().parent
STEM = "fig_scaling"

# Okabe-Ito, as in make_threshold_figure.py; greys and ink as in the blueprint (section 4)
BLUE, VERM, GREY, DARK, INK, GRID = "#0072B2", "#D55E00", "#9a9a95", "#55554f", "#1a1a19", "#e8e8e4"
GREEN = "#009E73"
DASHED, DOTTED = (0, (4, 2)), (0, (1, 1.2))

DAGGER = "\u2020"
# The paper's rows of tab:scaling, in its order, and no others (medgemma-27b, qwen3-32b-v035 and
# qwen3.8-27b-v035 are in the result files but not in the paper). Tick label in (a), in (b).
MODELS = [
    ("qwen3-0.6b", "0.6B", "0.6B"),
    ("qwen3-1.7b", "1.7B", "1.7B"),
    ("qwen3-4b-q4_K_M", "4B", "4B"),
    ("qwen3-8b", "8B", "8B"),
    ("qwen3-14b", "14B", "14B"),
    ("qwen3-30b-a3b-q4_K_M", "30B-A3B*", "30B-A3B*"),
    ("qwen3-32b", "32B\nteacher", "32B\nteacher"),
    ("qwen3.5-122b", "122B*", "122B*"),
]
KEYS = [m for m, _, _ in MODELS]
N_QWEN3 = 7          # lines join the seven Qwen3 models; Qwen3.5-122B, a later release, stands alone
TEACHER = "qwen3-32b"
SMALLEST = "qwen3-0.6b"
LATER = "qwen3.5-122b"

# What the paper prints (tab:scaling and its caption, 3 decimals). The script exits non-zero
# unless the result files give exactly these values.
PRINTED = {
    "biodiv": {
        "sentence": [0.676, 0.769, 0.761, 0.819, 0.818, 0.797, 0.804, 0.788],
        "pair": [0.658, 0.808, 0.885, 0.909, 0.929, 0.927, 0.960, 0.985],
        "triple": [0.663, 0.844, 0.921, 0.947, 0.959, 0.956, 0.964, 0.984],
        "trained": {"sentence": 0.871, "pair": 0.947},
        "base_rate": 0.574,
    },
    "biored": {
        "sentence": [0.522, 0.533, 0.546, 0.524, 0.555, 0.571, 0.566, 0.519],
        "pair": [0.523, 0.648, 0.745, 0.800, 0.805, 0.792, 0.813, 0.820],
        "trained": {"sentence": 0.757, "pair": 0.860},
        "base_rate": 0.476,
    },
    # text and appendix: the pair question's lead from 1.7B on, and the sentence question's range
    "gap_from_1.7B_2dp": {"biodiv": [0.04, 0.20], "biored": [0.11, 0.30]},
    "sentence_range_2dp": {"biodiv_from_4B": [0.76, 0.82], "biored_all": [0.52, 0.57]},
}
TOL = 1e-9


class Mismatch(SystemExit):
    pass


def check(cond, msg):
    if not cond:
        raise Mismatch(f"MISMATCH: {msg}")


def r3(v):
    return float(f"{v:.3f}")


# loading and checking

def biodiv_values(labels, res):
    """Panel (a): from tables_biodiv.json, rescored from llm/biodiv_*.csv and the trained
    ensembles' npy against clean_benchmark(labels). With labels other than "current" the
    rescored values are used (the tables JSON holds the current labels' numbers)."""
    tab = json.loads((res / "tables_biodiv.json").read_text())
    d = clean_benchmark(labels)
    y = d.label.to_numpy()
    prov = benchmark_provenance(labels)
    check(len(d) == prov["n"] == 437 and int(y.sum()) == prov["positives"], "clean benchmark size")
    current = labels == "current"
    if current:
        check(tab["benchmark_sha256"] == prov["sha256"], "tables_biodiv.json not scored on current labels")

    rescored = {}
    for m in KEYS:
        for q in ("sentence", "pair", "triple"):
            f = res / "llm" / f"biodiv_{m}_{q}.csv"
            o = pd.read_csv(f)
            idx = o.row.to_numpy() if "row" in o else np.arange(len(o))
            for c in ("species1", "species2", "sentence"):       # the CSV's label column is stale
                check((o[c].astype(str).to_numpy() == d[c].astype(str).to_numpy()[idx]).all(),
                      f"{f.name}: column {c} not row-aligned with the benchmark")
            rescored[f"biodiv_{m}_{q}"] = {"n": int(len(o)), "auprc": float(ap(y[idx], o.p_yes.to_numpy()))}
    trained_rescored = {arm: float(ap(y, np.load(res / f"S_biodiv_{arm}.npy"))) for arm in ("sentence", "pair")}

    if current:
        llm = {}
        for k, v in rescored.items():
            t = tab["llm"][k]
            check(t["n"] == v["n"] == 437, f"{k}: n")
            check(abs(t["auprc"] - v["auprc"]) < TOL, f"{k}: JSON {t['auprc']} vs rescored {v['auprc']}")
            llm[k] = {"n": t["n"], "auprc": t["auprc"]}
        trained = {arm: tab[arm]["auprc_ensemble"] for arm in ("sentence", "pair")}
        for arm in trained:
            check(abs(trained[arm] - trained_rescored[arm]) < TOL, f"trained {arm}: JSON vs npy")
            for m in KEYS:                                   # the same 437 rows for every model
                for q in ("sentence", "pair"):
                    same = tab["llm"][f"biodiv_{m}_{q}"]["trained_arms_same_rows"][arm]
                    check(abs(same - trained[arm]) < TOL, f"biodiv_{m}_{q} trained_arms_same_rows.{arm}")
        base = tab["base_rate"]
        check(abs(base - y.mean()) < TOL, "base_rate vs clean_benchmark labels")
        # cross-check: derived_biodiv.json
        der = json.loads((res / "derived_biodiv.json").read_text())
        for m in KEYS:
            s, p = llm[f"biodiv_{m}_sentence"]["auprc"], llm[f"biodiv_{m}_pair"]["auprc"]
            check(abs(der["scaling_sentence_auprc"][m] - s) < TOL, f"derived scaling_sentence_auprc {m}")
            check(abs(der["scaling_pair_minus_sentence"][m] - (p - s)) < TOL,
                  f"derived scaling_pair_minus_sentence {m}")
    else:
        llm, trained, base = rescored, trained_rescored, float(y.mean())
    return {"n": int(len(d)), "positives": int(y.sum()), "labels": labels, "benchmark_sha256": prov["sha256"],
            "base_rate": base, "sentence.auprc_ensemble": trained["sentence"],
            "pair.auprc_ensemble": trained["pair"], "llm": llm}


def biored_values(res):
    """Panel (b): from tables_biored.json, rescored from llm/biored_*.csv (labels and p_yes) and
    the trained ensembles' npy; base rates are the CSVs' label means."""
    tab = json.loads((res / "tables_biored.json").read_text())
    S = {arm: np.load(res / f"S_biored_{arm}.npy") for arm in ("sentence", "pair")}
    ref = pd.read_csv(res / "llm" / f"biored_{TEACHER}_pair.csv")
    llm, same_rows = {}, {}
    for m in KEYS:
        for q in ("sentence", "pair"):
            k = f"biored_{m}_{q}"
            o = pd.read_csv(res / "llm" / f"{k}.csv")
            n = len(o)
            check(n == 3000, f"{k}: n = {n}")
            check((o[["row", "label"]].to_numpy() == ref[["row", "label"]].to_numpy()[:n]).all(),
                  f"{k}: rows are not the sample's first {n}")
            t = tab["llm"][k]
            check(t["n"] == n, f"{k}: JSON n")
            a = float(ap(o.label.to_numpy(), o.p_yes.to_numpy()))
            check(abs(t["auprc"] - a) < TOL, f"{k}: JSON {t['auprc']} vs rescored {a}")
            llm[k] = {"n": n, "auprc": t["auprc"]}
            for arm in ("sentence", "pair"):
                a = float(ap(o.label.to_numpy(), S[arm][o.row.to_numpy()]))
                check(abs(t["trained_arms_same_rows"][arm] - a) < TOL, f"{k}: trained_arms_same_rows.{arm}")
            if q == "pair" and m == TEACHER:
                same_rows[k] = {arm: t["trained_arms_same_rows"][arm] for arm in ("sentence", "pair")}
            if True:                                         # one set of rows for every model
                for arm in ("sentence", "pair"):
                    check(abs(t["trained_arms_same_rows"][arm]
                              - tab["llm"][f"biored_{TEACHER}_pair"]["trained_arms_same_rows"][arm]) < TOL,
                          f"{k}: not on the teacher's rows")
    lab_3000 = ref.label.to_numpy()
    return {"n_sample": int(len(lab_3000)), "base_rate_sample": float(lab_3000.mean()),
            "llm": llm, "trained_arms_same_rows": same_rows}


def series(vals, bench, q):
    return [vals["llm"][f"{bench}_{m}_{q}"]["auprc"] for m in KEYS]


def assert_printed(bd, br):
    """The values the paper prints, at its precision."""
    P = PRINTED
    for q in ("sentence", "pair", "triple"):
        got = [r3(v) for v in series(bd, "biodiv", q)]
        check(got == P["biodiv"][q], f"biodiv {q}: {got} vs printed {P['biodiv'][q]}")
    for q in ("sentence", "pair"):
        got = [r3(v) for v in series(br, "biored", q)]
        check(got == P["biored"][q], f"biored {q}: {got} vs printed {P['biored'][q]}")
    check(r3(bd["sentence.auprc_ensemble"]) == P["biodiv"]["trained"]["sentence"], "biodiv trained sentence")
    check(r3(bd["pair.auprc_ensemble"]) == P["biodiv"]["trained"]["pair"], "biodiv trained pair")
    check(r3(bd["base_rate"]) == P["biodiv"]["base_rate"], "biodiv base rate")
    t32 = br["trained_arms_same_rows"][f"biored_{TEACHER}_pair"]
    check({a: r3(v) for a, v in t32.items()} == P["biored"]["trained"], f"biored trained {t32}")
    check(r3(br["base_rate_sample"]) == P["biored"]["base_rate"], "biored base rate")


def claims(bd, br):
    """The caption's and the text's claims about the plotted values, computed and checked."""
    out = {}
    for bench, vals in (("biodiv", bd), ("biored", br)):
        s, p = np.array(series(vals, bench, "sentence")), np.array(series(vals, bench, "pair"))
        gap = p - s
        out[f"pair_minus_sentence_{bench}"] = dict(zip(KEYS, map(float, gap)))
        out[f"pair_leads_from_1.7B_{bench}"] = bool((gap[1:] > 0).all())
        out[f"gap_range_from_1.7B_{bench}"] = [float(gap[1:].min()), float(gap[1:].max())]
    out["pair_trails_at_0.6B_biodiv"] = bool(out["pair_minus_sentence_biodiv"][SMALLEST] < 0)
    sb = np.array(series(bd, "biodiv", "sentence"))
    sr = np.array(series(br, "biored", "sentence"))
    out["sentence_range_biodiv_from_4B"] = [float(sb[2:].min()), float(sb[2:].max())]
    out["sentence_range_biored_all"] = [float(sr.min()), float(sr.max())]
    tr = np.array(series(bd, "biodiv", "triple")) - np.array(series(bd, "biodiv", "pair"))
    out["triple_minus_pair_biodiv"] = dict(zip(KEYS, map(float, tr)))
    out["triple_ahead_of_pair_qwen3_0.6B_to_30B"] = bool((tr[:KEYS.index(TEACHER)] > 0).all())
    return out


def assert_claims(c):
    """The claims hold for the labels the paper reports."""
    check(c["pair_leads_from_1.7B_biodiv"] and c["pair_leads_from_1.7B_biored"],
          "the pair question does not lead at every size from 1.7B")
    check(c["pair_trails_at_0.6B_biodiv"], "0.6B exception (pair trails on ours)")
    # caption: asked the triple question, the models track or exceed the pair question
    # (ahead from 0.6B to 30B-A3B, +0.004 at 32B, 0.984 against 0.985 at 122B)
    tr = c["triple_minus_pair_biodiv"]
    check(c["triple_ahead_of_pair_qwen3_0.6B_to_30B"], "triple question not ahead from 0.6B to 30B-A3B")
    check(r3(tr[TEACHER]) == 0.004, f"triple minus pair at 32B: {tr[TEACHER]}")
    for bench in ("biodiv", "biored"):
        lo, hi = c[f"gap_range_from_1.7B_{bench}"]
        check([round(lo, 2), round(hi, 2)] == PRINTED["gap_from_1.7B_2dp"][bench], f"gap range {bench}")
    for k, v in (("biodiv_from_4B", c["sentence_range_biodiv_from_4B"]),
                 ("biored_all", c["sentence_range_biored_all"])):
        check([round(x, 2) for x in v] == PRINTED["sentence_range_2dp"][k], f"sentence range {k}")


# drawing

def serif_family():
    """First TrueType face among Times New Roman, Liberation Serif (its metric clone), STIXGeneral
    (bundled with matplotlib) and DejaVu Serif. CFF-based OpenType faces (Nimbus Roman, TeX Gyre
    Termes) are skipped: matplotlib embeds them as a mislabelled CID font, not as TrueType."""
    fm = font_manager.fontManager
    known = {Path(f.fname).name for f in fm.ttflist}
    for path in font_manager.findSystemFonts(fontext="ttf"):    # the font cache may predate them
        name = Path(path).name
        if name.lower().endswith(".ttf") and name.lower().startswith(("liberationserif", "times")) \
                and name not in known:
            try:
                fm.addfont(path)
            except (OSError, RuntimeError, ValueError):
                pass
    for fam in ("Times New Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"):
        try:
            f = font_manager.findfont(font_manager.FontProperties(family=fam), fallback_to_default=False)
        except ValueError:
            continue
        if f.lower().endswith(".ttf"):
            return fam
    return "DejaVu Serif"


class TrainedHandle:
    """Legend key for the trained models: two dotted lines in the two arm colours."""


class TrainedHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        x = [-xdescent, width - xdescent]
        return [Line2D(x, [height * f - ydescent] * 2, color=c, lw=1.0, ls=DOTTED, transform=trans)
                for f, c in ((0.85, BLUE), (0.15, VERM))]


W_IN, H_IN = 3.03, 3.0
# vertical layout, in points from the top: legend, title (a), axes (a), ticks (a), gap,
# title (b), axes (b), ticks (b), x label. The two axes share what is left in proportion to their
# y ranges, so 0.1 AUPRC is the same height in both panels.
L_LEFT, L_RIGHT = 27.0, 3.5
L_TOP, L_TITLE, L_TICKS, L_GAP, L_XLABEL, L_BOTTOM = 21.0, 10.5, 19.5, 3.5, 9.5, 1.5
YLIM_A = (0.55, 1.02)          # 0.02 of headroom above 1.0, so the 0.985 marker is not clipped
YLIM_B = (0.40, 0.90)
XLIM = (-1.15, 7.45)           # room at the left for the inline labels of the reference lines
SEP = N_QWEN3 - 0.5            # dotted separator before Qwen3.5-122B
LONE = (N_QWEN3 - 0.32, N_QWEN3 + 0.32)   # short reference segments at the 122B position


def draw(bd, br, serif):
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,      # TrueType, not Type 3
        "font.family": "serif", "font.serif": [serif, "DejaVu Serif"],
        "font.size": 7.0, "axes.titlesize": 7.0, "axes.labelsize": 7.0, "legend.fontsize": 7.0,
        "xtick.labelsize": 6.3, "ytick.labelsize": 6.5,
        "axes.linewidth": 0.6, "axes.edgecolor": DARK, "axes.labelcolor": INK, "text.color": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "xtick.major.pad": 1.6, "ytick.major.pad": 1.6,
        "xtick.color": DARK, "ytick.color": DARK, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    })
    fig = plt.figure(figsize=(W_IN, H_IN))
    W, H = W_IN * 72, H_IN * 72
    both = H - L_TOP - 2 * L_TITLE - 2 * L_TICKS - L_GAP - L_XLABEL - L_BOTTOM
    span_a, span_b = YLIM_A[1] - YLIM_A[0], YLIM_B[1] - YLIM_B[0]
    h_a, h_b = both * span_a / (span_a + span_b), both * span_b / (span_a + span_b)
    top_a = H - L_TOP - L_TITLE
    top_b = top_a - h_a - L_TICKS - L_GAP - L_TITLE
    rect = lambda top, h: [L_LEFT / W, (top - h) / H, (W - L_LEFT - L_RIGHT) / W, h / H]  # noqa: E731
    ax_a, ax_b = fig.add_axes(rect(top_a, h_a)), fig.add_axes(rect(top_b, h_b))

    x = np.arange(len(KEYS), dtype=float)
    q3 = slice(0, N_QWEN3)
    t32 = br["trained_arms_same_rows"][f"biored_{TEACHER}_pair"]
    panels = [
        dict(ax=ax_a, bench="biodiv", vals=bd, ylim=YLIM_A,
             title="(a) TaxaPairs, 437 expert-graded candidates",
             ticks=[a for _, a, _ in MODELS],
             trained={"pair": bd["pair.auprc_ensemble"], "sentence": bd["sentence.auprc_ensemble"]},
             trained_lone=None, base=bd["base_rate"], base_lone=None,
             # label side for each reference line: +1 above, -1 below
             side={"pair": +1, "sentence": +1, "base": +1}),
        dict(ax=ax_b, bench="biored", vals=br, ylim=YLIM_B,
             title="(b) BioRED, 3,000-candidate sample",
             ticks=[b for _, _, b in MODELS],
             trained={"pair": t32["pair"], "sentence": t32["sentence"]},
             trained_lone=None, base=br["base_rate_sample"], base_lone=None,
             side={"pair": -1, "sentence": -1, "base": -1}),
    ]
    plotted, labels_drawn = {}, []
    for pn, p in zip("ab", panels):
        ax, bench = p["ax"], p["bench"]
        s = np.array(series(p["vals"], bench, "sentence"))
        pr = np.array(series(p["vals"], bench, "pair"))
        # the pair question's lead over the seven Qwen3 models (only where it leads)
        ax.fill_between(x[q3], s[q3], pr[q3], where=pr[q3] >= s[q3], interpolate=True,
                        color=BLUE, alpha=0.10, lw=0, zorder=1)
        ax.plot(x[q3], s[q3], color=VERM, lw=1.2, ls=DASHED, zorder=3)
        ax.plot(x[q3], pr[q3], color=BLUE, lw=1.5, zorder=3.2)
        ax.plot(x, s, ls="none", marker="s", ms=3.5, mfc="white", mec=VERM, mew=0.9, zorder=4)
        ax.plot(x, pr, ls="none", marker="o", ms=3.9, mfc=BLUE, mec="white", mew=0.45, zorder=4.2)
        if bench == "biodiv":                        # the triple question: thin line, small diamonds
            tq = np.array(series(p["vals"], bench, "triple"))
            ax.plot(x[q3], tq[q3], color=GREEN, lw=0.7, zorder=2.9)
            ax.plot(x, tq, ls="none", marker="D", ms=2.2, mfc=GREEN, mec="none", zorder=4.1)
        ax.axvline(SEP, color=GREY, lw=0.6, ls=(0, (1, 2)), zorder=1.5)

        # trained 110M models (dotted, arm colours) and the base rate (grey dotted)
        x_end = XLIM[1] if p["trained_lone"] is None else SEP - 0.12
        refs = {"pair": (p["trained"]["pair"], BLUE, "110M pair-conditioned"),
                "sentence": (p["trained"]["sentence"], VERM, "110M sentence-level")}
        for arm, (yv, col, text) in refs.items():
            own = ax.plot([XLIM[0], x_end], [yv, yv], color=col, lw=0.9, ls=DOTTED, zorder=2)[0]
            labels_drawn.append((inline(ax, XLIM[0], yv, text, INK, p["side"][arm]), own))
            if p["trained_lone"] is not None:
                ax.plot(LONE, [p["trained_lone"][arm]] * 2, color=col, lw=0.9, ls=DOTTED, zorder=2)
        own = ax.plot([XLIM[0], x_end], [p["base"]] * 2, color=GREY, lw=0.7, ls=(0, (1, 1.6)), zorder=1.6)[0]
        labels_drawn.append((inline(ax, XLIM[0], p["base"], "base rate", DARK, p["side"]["base"]), own))
        if p["base_lone"] is not None:
            ax.plot(LONE, [p["base_lone"]] * 2, color=GREY, lw=0.7, ls=(0, (1, 1.6)), zorder=1.6)

        ax.set_xlim(*XLIM)
        ax.set_ylim(*p["ylim"])
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.set_xticks(x, p["ticks"])
        ax.tick_params(axis="x", length=2.5)
        ax.set_ylabel("AUPRC", labelpad=2.5)
        ax.set_title(p["title"], loc="left", pad=3.0)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        plotted[pn] = {
            "x_positions": dict(zip(KEYS, map(float, x))),
            "sentence_question": dict(zip(KEYS, map(float, s))),
            "pair_question": dict(zip(KEYS, map(float, pr))),
            "lines_join": KEYS[:N_QWEN3],
            "trained_110m": {"pair": float(p["trained"]["pair"]), "sentence": float(p["trained"]["sentence"]),
                             "x_span": [XLIM[0], float(x_end)]},
            "base_rate": {"y": float(p["base"]), "x_span": [XLIM[0], float(x_end)]},
        }
        if bench == "biodiv":
            plotted[pn]["triple_question"] = dict(zip(KEYS, map(float, tq)))
        if p["trained_lone"] is not None:
            plotted[pn]["trained_110m_at_122B"] = {"pair": float(p["trained_lone"]["pair"]),
                                                   "sentence": float(p["trained_lone"]["sentence"]),
                                                   "x_span": list(LONE)}
            plotted[pn]["base_rate_at_122B"] = {"y": float(p["base_lone"]), "x_span": list(LONE)}
        plotted[pn]["ylim"] = list(p["ylim"])
        plotted[pn]["separator_x"] = SEP

    ax_b.set_xlabel("zero-shot Qwen model (total parameters)", labelpad=2.0)
    # one row at 7 pt is wider than the column: two columns, filled column-wise, the zero-shot
    # questions on the left and the trained models on the right
    handles = [
        Line2D([], [], color=BLUE, lw=1.5, marker="o", ms=3.9, mfc=BLUE, mec="white", mew=0.45),
        Line2D([], [], color=VERM, lw=1.2, ls=DASHED, marker="s", ms=3.5, mfc="white", mec=VERM, mew=0.9),
        Line2D([], [], color=GREEN, lw=0.7, marker="D", ms=2.2, mfc=GREEN, mec="none"),
        TrainedHandle(),
    ]
    leg = fig.legend(handles, ["pair question", "sentence question", "triple question (a)",
                               "trained 110M (same rows)"],
                     handler_map={TrainedHandle: TrainedHandler()}, loc="upper center",
                     bbox_to_anchor=((L_LEFT + (W - L_LEFT - L_RIGHT) / 2) / W, 1.0), ncol=2,
                     frameon=False, handlelength=2.6, handletextpad=0.5, columnspacing=1.8,
                     labelspacing=0.25, borderaxespad=0.2, borderpad=0.15)
    return fig, (ax_a, ax_b), plotted, labels_drawn, leg


def inline(ax, x0, y, text, color, side):
    """A label at the left end of a horizontal reference line, just above (+1) or below (-1) it."""
    # a white backing interrupts the light grid behind the text; the clearance check keeps every
    # line and marker outside it
    return ax.annotate(text, (x0, y), xytext=(1.5, 1.2 * side), textcoords="offset points",
                       ha="left", va="bottom" if side > 0 else "top", fontsize=6.0, color=color, zorder=5,
                       bbox=dict(boxstyle="square,pad=0.06", fc="white", ec="none"))


def layout_checks(fig, axes, labels_drawn, leg, strict=True):
    """Nothing outside the canvas; each inline label clear of the other text and of every line
    and marker in its panel (its own reference line excepted). Fatal for the paper's figure; for
    other label versions (regression copies) a problem is printed as a warning."""
    problems = []

    def report(cond, msg):
        if not cond:
            problems.append(msg)

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    pt = fig.dpi / 72
    texts = [t for t, _ in labels_drawn]
    fixed = [leg, *[t for ax in axes for t in ax.get_xticklabels() + ax.get_yticklabels()],
             *[ax.title for ax in axes], *[ax.xaxis.label for ax in axes], *[ax.yaxis.label for ax in axes]]
    fixed = [a for a in fixed if not hasattr(a, "get_text") or a.get_text()]
    for a in [*fixed, *axes, *texts]:
        bb = a.get_window_extent(r)
        report(bb.x0 >= -0.5 and bb.y0 >= -0.5 and bb.x1 <= W + 0.5 and bb.y1 <= H + 0.5,
               f"{type(a).__name__} {getattr(a, 'get_text', lambda: '')()!r} is cut off: {bb}")
    boxes = [t.get_window_extent(r).padded(0.5 * pt) for t in texts]     # font ascent to descent
    for i, b in enumerate(boxes):
        for j, o in enumerate([*boxes, *[a.get_window_extent(r) for a in fixed]]):
            other = texts[j] if j < len(texts) else fixed[j - len(texts)]
            report(not (j != i and b.overlaps(o) and (j >= len(texts) or texts[i].axes is texts[j].axes)),
                   f"label {texts[i].get_text()!r} overlaps {other!r}")
    f = np.linspace(0, 1, 120)
    for (t, own), b in zip(labels_drawn, boxes):
        ax = t.axes
        for ln in ax.lines:
            if ln is own:
                continue
            xs, ys = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
            if ln.get_transform() is not ax.transData:   # axvline: x in data, y in axes units
                xs, ys = xs[:1].repeat(2), np.array(ax.get_ylim())
            pts, rad = [], []
            if ln.get_linestyle() not in ("None", "none", "", " ") and len(xs) > 1:
                for k in range(len(xs) - 1):              # dense samples along each segment
                    pts.append(np.c_[xs[k] + f * (xs[k + 1] - xs[k]), ys[k] + f * (ys[k + 1] - ys[k])])
                    rad.append(np.full(len(f), ln.get_linewidth() / 2 * pt))
            if ln.get_marker() not in ("None", "none", "", " ", None):
                pts.append(np.c_[xs, ys])
                rad.append(np.full(len(xs), (ln.get_markersize() / 2 + ln.get_markeredgewidth()) * pt))
            if not pts:
                continue
            P, R = ax.transData.transform(np.vstack(pts)), np.concatenate(rad)
            hit = (P[:, 0] > b.x0 - R) & (P[:, 0] < b.x1 + R) & (P[:, 1] > b.y0 - R) & (P[:, 1] < b.y1 + R)
            report(not hit.any(), f"label {t.get_text()!r} touches a line or marker near "
                                  f"{ax.transData.inverted().transform(P[hit][:1])}")
    if problems and strict:
        raise Mismatch("LAYOUT: " + "; ".join(problems))
    for msg in problems:
        print(f"warning (layout): {msg}", file=sys.stderr)


def save(fig, out_dir, stem):
    pdf, png = out_dir / f"{stem}.pdf", out_dir / f"{stem}.png"
    fig.savefig(pdf, metadata={"Creator": None, "Producer": None, "CreationDate": None,
                               "Author": None, "Title": None})
    fig.savefig(png, dpi=300, metadata={"Software": None})
    raw = pdf.read_bytes()
    check(b"/Type3" not in raw, "Type 3 font in the PDF")
    check(b"/FontFile3" not in raw and b"/FontFile2" in raw, "a non-TrueType font program in the PDF")
    for key in (b"/Creator", b"/Producer", b"/Author", b"/Title", b"/CreationDate"):
        check(key not in raw, f"PDF metadata {key.decode()} present")
    return pdf, png


def main():
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--labels", choices=sorted(LABEL_VERSIONS), default="current",
                     help="label version of the biodiversity benchmark; other than 'current', panel (a) "
                          "is rescored and written with the version as a suffix, unchecked against the paper")
    ap_.add_argument("--results-dir", default=str(RESULTS), help="where the tables, npy and llm/ CSVs are")
    ap_.add_argument("--out-dir", default=str(FIGDIR), help="where the figure and its JSON are written")
    a = ap_.parse_args()
    res, out_dir = Path(a.results_dir), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = STEM if a.labels == "current" else f"{STEM}_{a.labels}"

    bd = biodiv_values(a.labels, res)
    br = biored_values(res)
    if a.labels == "current":
        assert_printed(bd, br)
    c = claims(bd, br)
    if a.labels == "current":
        assert_claims(c)

    serif = serif_family()
    fig, axes, plotted, labels_drawn, leg = draw(bd, br, serif)
    layout_checks(fig, axes, labels_drawn, leg, strict=a.labels == "current")
    pdf, png = save(fig, out_dir, stem)

    record = {
        "figure": stem, "latex_label": "fig:scaling", "script": Path(__file__).name,
        "labels": a.labels, "benchmark_sha256": bd["benchmark_sha256"], "font": serif,
        "size_in": [W_IN, H_IN],
        "models": KEYS, "tick_labels": {"a": [t for _, t, _ in MODELS], "b": [t for _, _, t in MODELS]},
        "tables_biodiv.json": {k: bd[k] for k in ("n", "positives", "base_rate", "sentence.auprc_ensemble",
                                                  "pair.auprc_ensemble", "llm")},
        "tables_biored.json": {"llm": br["llm"], "trained_arms_same_rows": br["trained_arms_same_rows"]},
        "llm/biored_qwen3-32b_pair.csv": {"n": br["n_sample"], "label_mean": br["base_rate_sample"]},
        "plotted": plotted,
        "caption": {
            "n_biodiv": bd["n"], "n_biored_sample": br["n_sample"],
            "trained_model_size": "110M", "teacher": TEACHER, "pair_leads_from": "qwen3-1.7b",
            "base_rate_biodiv": bd["base_rate"], "base_rate_biored_sample": br["base_rate_sample"],
            "mixture_of_experts": ["qwen3-30b-a3b-q4_K_M", "qwen3.5-122b"],
        },
        "claims": c,
        "printed_3dp": PRINTED if a.labels == "current" else None,
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"font {serif}; wrote {pdf}, {png} and {out_dir / (stem + '.json')}")


if __name__ == "__main__":
    main()
