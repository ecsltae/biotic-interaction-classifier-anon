#!/usr/bin/env python3
"""Figure 2 of Paper A, operating curves (labels fig:curves and fig:threshold).

(a) Precision against recall of the three trained models (three-checkpoint ensembles) on the 437
clean benchmark rows: step curves from precision_recall_curve, each model's point at tau = 0.5, and
the base rate (accepting every candidate). (b) Precision, recall and F1 of the pair-conditioned model
against the decision threshold tau on the 99-point grid 0.01..0.99, with the sentence-level model's F1
and a band over the pair model's three block-held-out thresholds. The triple-conditioned model's F1 is
never drawn against tau: its scores are calibrated differently, so an equal tau is not an equal
operating point (at equal tau its F1 is above the pair model's at 75 of 99 thresholds).

Inputs, read only:
  results/paperA_v2/S_biodiv_{sentence,pair,triple}.npy   ensemble scores on the 437 clean rows
  results/paperA_v2/tables_biodiv.json                    AUPRC, curve entries, thresholds, base rate
  results/paperA_v2/derived_biodiv.json                   fig4: the F1 comparison over 99 thresholds
  src/eval/core.py clean_benchmark(labels)                gold labels, hash-checked

Outputs, in --out-dir (default: this directory):
  fig_curves.pdf    the figure, 3.03 x 3.3 in (column width), TrueType fonts, empty metadata
  fig_curves.png    300 dpi preview
  fig_curves.json   every value the caption or text states, under the key paths of the result
                    files (base_rate, {arm}.auprc_ensemble, curve, pair.thresholds, fig4), plus the
                    precision-recall comparison behind the caption, the plotted series and the
                    SHA-256 of every input

Every plotted value is recomputed from the score vectors and checked against the result files and,
for --labels current, against the values the paper prints. Any mismatch exits with status 1 before a
file is written. fig4_threshold.* and make_threshold_figure.py are left unchanged.

Usage:
  python3 paperA/fig/make_curves_figure.py
  python3 paperA/fig/make_curves_figure.py --labels current --out-dir paperA/fig
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

FIGDIR = Path(__file__).resolve().parent
REPO = FIGDIR.parents[1]
sys.path.insert(0, str(REPO / "src"))
from eval.core import LABEL_VERSIONS, clean_benchmark, sha256  # noqa: E402

RESULTS = REPO / "results/paperA_v2"
STEM = "fig_curves"
ARMS = ("sentence", "pair", "triple")
WIDTH_IN, HEIGHT_IN = 3.03, 3.3      # \columnwidth = 7.7 cm; included with width=\columnwidth
TAU_MARK = 0.5
BAND_PAD = 0.004                     # half-width added on each side so the band is visible
TXT = 7.0                            # every text item; tick labels are 6.5 pt

# Okabe-Ito, as in make_threshold_figure.py
VERM, BLUE, GREEN = "#D55E00", "#0072B2", "#009E73"
GREY, DARK, INK, GRIDC = "#9a9a95", "#55554f", "#1a1a19", "#e8e8e4"

# One encoding per formulation, repeated by colour, line style and marker so it survives greyscale.
ARM_STYLE = {
    "sentence": {"color": VERM, "ls": (0, (4, 2)), "lw": 1.1, "z": 2, "label": "sentence-level",
                 "marker": {"marker": "s", "ms": 4.2, "mfc": "white", "mec": VERM, "mew": 1.0}},
    "pair": {"color": BLUE, "ls": "-", "lw": 1.5, "z": 3, "label": "pair-conditioned",
             "marker": {"marker": "o", "ms": 4.8, "mfc": BLUE, "mec": "white", "mew": 0.6}},
    "triple": {"color": GREEN, "ls": (0, (1, 1.2)), "lw": 1.3, "z": 4, "label": "triple-conditioned",
               "marker": {"marker": "D", "ms": 4.2, "mfc": GREEN, "mec": "white", "mew": 0.6}},
}

# Preferred serif families, Times-like first. pdf.fonttype 42 writes the font program as TrueType,
# so a family is used only if its file has TrueType outlines: Nimbus Roman and TeX Gyre Termes ship
# as CFF OpenType here, which the PDF backend would put in a TrueType slot (poppler then reports a
# font-type mismatch). Liberation Serif has the metrics of Times New Roman.
SERIF = ("Times New Roman", "Liberation Serif", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif")

# Values the paper prints (labels "current"); the figure must agree with them.
PAPER = {
    "n": 437, "positives": 251, "base_rate": "0.574",
    "auprc_ensemble": {"sentence": "0.8705", "pair": "0.9468", "triple": "0.9480"},
    "tau_0.5": {"sentence": ("0.886", "0.526"), "pair": ("0.940", "0.689"), "triple": ("0.942", "0.777")},
    "pair_thresholds": {"biotx100": "0.17", "reject50": "0.14", "test299": "0.15"},
    "fig4": {"thresholds": 99, "pair_ahead": 99, "min_gap": "0.0545", "min_gap_3dp": "0.054",
             "min_gap_tau": "0.01"},
    # Appendix table tab:curve (P, R, F1 at six thresholds), the values behind panel (b)
    "curve": {
        "sentence": {"0.01": (".709", ".920", ".801"), "0.04": (".782", ".817", ".799"),
                     "0.10": (".819", ".737", ".776"), "0.25": (".861", ".693", ".768"),
                     "0.50": (".886", ".526", ".660"), "0.90": (".930", ".159", ".272")},
        "pair": {"0.01": (".761", ".976", ".855"), "0.04": (".815", ".948", ".877"),
                 "0.10": (".854", ".908", ".880"), "0.25": (".911", ".853", ".881"),
                 "0.50": (".940", ".689", ".795"), "0.90": (".981", ".422", ".591")},
        "triple": {"0.01": (".740", ".940", ".828"), "0.04": (".811", ".924", ".864"),
                   "0.10": (".845", ".912", ".877"), "0.25": (".903", ".849", ".875"),
                   "0.50": (".942", ".777", ".852"), "0.90": (".982", ".637", ".773")},
    },
}

# Caption sentence "ahead only in the high-precision region and behind at the highest recall":
# where the triple model's interpolated precision is above the pair model's, both stay above this
# precision, and the triple model is behind at every recall from HIGH_RECALL on.
HIGH_PRECISION = 0.91
HIGH_RECALL = 0.85


class Checks:
    """Collects failed checks so that every mismatch is reported before the script exits."""

    def __init__(self) -> None:
        self.failed: list[str] = []
        self.passed = 0

    def ok(self, cond: bool, what: str) -> None:
        """Record one check.

        Args:
            cond: whether the check holds.
            what: description printed if it does not.
        """
        if cond:
            self.passed += 1
        else:
            self.failed.append(what)

    def close(self, got: float, want: float, what: str, tol: float = 1e-12) -> None:
        """Check two floats agree to within ``tol``."""
        self.ok(abs(float(got) - float(want)) <= tol, f"{what}: got {got!r}, expected {want!r}")

    def printed(self, got: float, want: str, what: str) -> None:
        """Check a value rounds to the string the paper prints (same number of decimals)."""
        decimals = len(want.split(".")[1])
        txt = f"{float(got):.{decimals}f}"
        if want.startswith("."):
            txt = txt[1:] if txt.startswith("0.") else txt
        self.ok(txt == want, f"{what}: {float(got)!r} prints as {txt}, the paper prints {want}")


def truetype_serif() -> str:
    """Return the first family in ``SERIF`` that resolves to a TrueType-outline font file.

    Fonts installed after matplotlib built its cache (often Liberation) are registered first.

    Returns:
        The family name to use for all text, including mathtext.
    """
    known = {f.name for f in font_manager.fontManager.ttflist}
    if not {"Times New Roman", "Liberation Serif"} <= known:
        for path in font_manager.findSystemFonts(fontext="ttf"):
            if Path(path).name.lower().startswith(("times", "liberationserif")):
                try:
                    font_manager.fontManager.addfont(path)
                except (OSError, RuntimeError, ValueError):
                    pass
    for family in SERIF:
        try:
            path = font_manager.findfont(font_manager.FontProperties(family=family),
                                         fallback_to_default=False)
        except ValueError:
            continue
        with open(path, "rb") as fh:
            if fh.read(4) in (b"\x00\x01\x00\x00", b"true"):
                return family
    raise SystemExit("no TrueType serif font found (tried: " + ", ".join(SERIF) + ")")


def set_style(family: str) -> None:
    """Apply the shared figure style (blueprint section 4): serif 7 to 7.5 pt, ticks 6.5 pt."""
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif", "font.serif": [family, "DejaVu Serif"],
        "mathtext.fontset": "custom", "mathtext.rm": family, "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold", "mathtext.cal": family, "mathtext.sf": family,
        "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 7.0, "axes.linewidth": 0.6, "axes.edgecolor": DARK,
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": DARK, "ytick.color": DARK,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.pad": 2.0, "ytick.major.pad": 2.0, "axes.labelpad": 2.5,
        "lines.solid_capstyle": "butt", "lines.dash_capstyle": "butt",
    })


def rel(path: Path) -> str:
    """Path relative to the repository root, or the bare file name: never an absolute path."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return path.name


def at_tau(y: np.ndarray, s: np.ndarray, tau: float) -> dict:
    """Precision, recall, F1 and number accepted when accepting every score >= ``tau``."""
    yhat = (s >= tau).astype(int)
    return {"tau": float(tau), "P": float(precision_score(y, yhat, zero_division=0)),
            "R": float(recall_score(y, yhat, zero_division=0)),
            "F1": float(f1_score(y, yhat, zero_division=0)), "accepts": int(yhat.sum())}


def interpolated_precision(y: np.ndarray, s: np.ndarray, recall_grid: np.ndarray) -> np.ndarray:
    """Best precision at recall >= r, for each r in ``recall_grid``."""
    p, r, _ = precision_recall_curve(y, s)
    return np.array([p[r >= x - 1e-12].max() for x in recall_grid])


def compute(labels: str, results: Path) -> tuple[dict, Checks]:
    """Load the inputs, recompute every plotted value and check it.

    Args:
        labels: benchmark label version (``eval.core.LABEL_VERSIONS``).
        results: directory holding the score vectors and the two result tables.

    Returns:
        The values to plot and write, and the checks with their outcome.
    """
    c = Checks()
    bench = clean_benchmark(labels)          # raises if the benchmark file's SHA-256 has moved
    y = bench.label.to_numpy().astype(int)
    tb_path, dv_path = results / "tables_biodiv.json", results / "derived_biodiv.json"
    tb = json.loads(tb_path.read_text())
    dv = json.loads(dv_path.read_text())
    npy = {a: results / f"S_biodiv_{a}.npy" for a in ARMS}
    S = {a: np.load(p).astype(float) for a, p in npy.items()}

    # the result files were scored against these labels
    c.ok(tb.get("labels") == labels, f"tables_biodiv.json labels {tb.get('labels')!r} != {labels!r}")
    c.ok(tb.get("benchmark_sha256") == LABEL_VERSIONS[labels]["sha256"],
         "tables_biodiv.json benchmark_sha256 differs from the label version's")
    c.ok(dv.get("benchmark", {}).get("sha256") == LABEL_VERSIONS[labels]["sha256"]
         and dv["benchmark"].get("labels") == labels,
         "derived_biodiv.json benchmark labels or sha256 differ from the label version's")
    c.ok(len(y) == tb["n"] and int(y.sum()) == tb["positives"],
         f"benchmark has {len(y)} rows / {int(y.sum())} positive, tables say {tb['n']} / {tb['positives']}")
    for a in ARMS:
        c.ok(S[a].shape == (len(y),), f"S_biodiv_{a}.npy has shape {S[a].shape}, expected ({len(y)},)")
    if c.failed:
        return {}, c
    base_rate = float(y.mean())
    c.close(base_rate, tb["base_rate"], "base rate vs tables_biodiv.json base_rate")

    # (a) precision-recall step curves, AUPRC and the tau = 0.5 points
    pr, auprc, tau05 = {}, {}, {}
    for a in ARMS:
        p, r, thr = precision_recall_curve(y, S[a])
        pr[a] = {"recall": r.tolist(), "precision": p.tolist()}
        auprc[a] = float(average_precision_score(y, S[a]))
        c.close(auprc[a], tb[a]["auprc_ensemble"], f"{a}: average precision vs {a}.auprc_ensemble")
        tau05[a] = at_tau(y, S[a], TAU_MARK)
        on_curve = np.any(np.isclose(r, tau05[a]["R"], atol=1e-12) & np.isclose(p, tau05[a]["P"], atol=1e-12))
        c.ok(bool(on_curve), f"{a}: tau = 0.5 point is not a vertex of its precision-recall curve")

    # the six-threshold curve entries (tab:curve), recomputed from the scores
    curve = {}
    for a in ARMS:
        curve[a] = []
        for e in tb["curve"][a]:
            got = at_tau(y, S[a], e["tau"])
            for k in ("P", "R", "F1"):
                c.close(got[k], e[k], f"{a}: {k} at tau = {e['tau']} vs tables_biodiv.json curve")
            c.ok(got["accepts"] == e["accepts"], f"{a}: accepts at tau = {e['tau']} vs curve")
            curve[a].append({k: e[k] for k in ("tau", "P", "R", "F1", "accepts")})
        e05 = [e for e in tb["curve"][a] if abs(e["tau"] - TAU_MARK) < 1e-9]
        c.ok(len(e05) == 1, f"{a}: no single curve entry with tau == 0.5")
        if e05:
            c.close(tau05[a]["P"], e05[0]["P"], f"{a}: P at tau = 0.5 vs curve entry")
            c.close(tau05[a]["R"], e05[0]["R"], f"{a}: R at tau = 0.5 vs curve entry")

    # (b) the 99-threshold grid
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    pair_rows = [at_tau(y, S["pair"], t) for t in grid]
    sent_rows = [at_tau(y, S["sentence"], t) for t in grid]
    pair_P = np.array([r["P"] for r in pair_rows])
    pair_R = np.array([r["R"] for r in pair_rows])
    pair_F1 = np.array([r["F1"] for r in pair_rows])
    sent_F1 = np.array([r["F1"] for r in sent_rows])
    gap = pair_F1 - sent_F1
    ahead = int((gap > 0).sum())
    min_gap, min_gap_tau = float(gap.min()), float(grid[int(np.argmin(gap))])
    f4 = dv["fig4"]
    c.ok(len(grid) == f4["thresholds"], f"grid has {len(grid)} thresholds, fig4 says {f4['thresholds']}")
    c.ok(ahead == f4["pair_ahead"], f"pair F1 ahead at {ahead}, fig4 says {f4['pair_ahead']}")
    c.close(min_gap, f4["min_gap"], "smallest F1 gap vs fig4.min_gap", tol=1e-9)
    c.close(min_gap_tau, f4["min_gap_tau"], "tau of the smallest gap vs fig4.min_gap_tau", tol=1e-9)
    # the plotted series pass through the tab:curve entries
    for a, rows in (("pair", pair_rows), ("sentence", sent_rows)):
        for e in tb["curve"][a]:
            i = int(np.argmin(np.abs(grid - e["tau"])))
            if abs(grid[i] - e["tau"]) < 1e-9:
                for k in ("P", "R", "F1"):
                    c.close(rows[i][k], e[k], f"{a}: plotted {k} at tau = {e['tau']} vs curve entry")

    thr = {k: float(v) for k, v in tb["pair"]["thresholds"].items()}
    c.ok(set(thr) == {"biotx100", "reject50", "test299"}, f"pair.thresholds has blocks {sorted(thr)}")
    checkpoints = {a: len(tb[a]["auprc_per_seed"]) for a in ARMS}
    c.ok(len(set(checkpoints.values())) == 1, f"arms have different numbers of checkpoints {checkpoints}")
    band = [min(thr.values()) - BAND_PAD, max(thr.values()) + BAND_PAD]

    # the caption's precision-recall claim (interpolated precision on a 0.01 recall grid)
    rg = np.round(np.arange(0.0, 1.0001, 0.01), 2)
    ip = {a: interpolated_precision(y, S[a], rg) for a in ARMS}
    d_tp = ip["triple"] - ip["pair"]
    lead, trail = rg[d_tp > 1e-12], rg[d_tp < -1e-12]
    lead_minp = float(min(ip["triple"][d_tp > 1e-12].min(), ip["pair"][d_tp > 1e-12].min()))
    c.ok(lead.size > 0 and lead_minp > HIGH_PRECISION,
         f"triple ahead somewhere below precision {HIGH_PRECISION} (min {lead_minp:.4f})")
    c.ok(bool(np.all(d_tp[rg >= HIGH_RECALL - 1e-9] < -1e-12)),
         f"triple not behind at every recall from {HIGH_RECALL}")
    d_sp = ip["sentence"] - ip["pair"]
    sent_above = rg[d_sp > 1e-12]

    out = {
        "y": y, "S": S, "pr": pr, "grid": grid, "pair_P": pair_P, "pair_R": pair_R,
        "pair_F1": pair_F1, "sent_F1": sent_F1, "band": band, "tau05": tau05,
        "json": {
            "figure": f"{STEM}.pdf", "latex_labels": ["fig:curves", "fig:threshold"],
            "width_in": WIDTH_IN, "height_in": HEIGHT_IN,
            "labels": labels, "benchmark_sha256": LABEL_VERSIONS[labels]["sha256"],
            "n": len(y), "positives": int(y.sum()), "base_rate": base_rate,
            "sentence": {"auprc_ensemble": auprc["sentence"]},
            "pair": {"auprc_ensemble": auprc["pair"], "thresholds": thr},
            "triple": {"auprc_ensemble": auprc["triple"]},
            "curve": curve,
            "fig4": {"thresholds": len(grid), "pair_ahead": ahead, "min_gap": min_gap,
                     "min_gap_tau": min_gap_tau},
            "caption": {
                "candidates": len(y), "ensemble_checkpoints": checkpoints["pair"],
                "marker_tau": TAU_MARK, "block_held_out_thresholds": len(thr),
                "thresholds": len(grid), "pair_ahead": ahead, "min_gap_3dp": round(min_gap, 3),
                "supported_by": {
                    "the 437 expert-graded candidates": "n, positives, benchmark_sha256",
                    "three-checkpoint ensembles": "{arm}.auprc_ensemble; per-seed counts in tables_biodiv.json",
                    "markers at tau = 0.5": "panel_a.markers (equal to curve entries with tau 0.5)",
                    "dotted line, accepting every candidate": "base_rate",
                    "triple-conditioned ahead only in the high-precision region, behind at the "
                    "highest recall": "panel_a.precision_recall_comparison",
                    "the shaded band spans the three block-held-out thresholds": "pair.thresholds, panel_b.band",
                    "ahead at every one of 99 thresholds": "fig4.thresholds, fig4.pair_ahead",
                    "by at least 0.054 F1": "fig4.min_gap, fig4.min_gap_tau",
                },
            },
            "panel_a": {
                "markers": {a: {k: tau05[a][k] for k in ("tau", "P", "R", "F1", "accepts")} for a in ARMS},
                "base_rate_line": base_rate,
                "precision_recall_comparison": {
                    "method": "interpolated precision (best precision at recall >= r), r = 0.00..1.00 by 0.01",
                    "triple_ahead_of_pair_at_recall": lead.tolist(),
                    "triple_behind_pair_at_recall": trail.tolist(),
                    "triple_ahead_from": float(lead.min()), "triple_ahead_to": float(lead.max()),
                    "min_precision_of_either_where_triple_ahead": lead_minp,
                    "triple_behind_at_every_recall_from": HIGH_RECALL,
                    "sentence_above_pair_at_recall": sent_above.tolist(),
                    "precision_at_recall_1": {a: float(ip[a][-1]) for a in ARMS},
                },
                "curves": pr,
            },
            "panel_b": {
                "band": band, "band_pad": BAND_PAD,
                "tau": grid.tolist(), "pair_P": pair_P.tolist(), "pair_R": pair_R.tolist(),
                "pair_F1": pair_F1.tolist(), "sentence_F1": sent_F1.tolist(),
            },
            "inputs": {rel(p): sha256(p) for p in (*npy.values(), tb_path, dv_path)},
            "script": rel(Path(__file__)),
        },
    }

    if labels == "current":                  # the values the paper prints
        c.ok(len(y) == PAPER["n"] and int(y.sum()) == PAPER["positives"], "paper: 437 rows, 251 positive")
        c.printed(base_rate, PAPER["base_rate"], "paper: base rate")
        for a in ARMS:
            c.printed(auprc[a], PAPER["auprc_ensemble"][a], f"paper: {a} ensemble AUPRC")
            c.printed(tau05[a]["P"], PAPER["tau_0.5"][a][0], f"paper: {a} P at tau = 0.5")
            c.printed(tau05[a]["R"], PAPER["tau_0.5"][a][1], f"paper: {a} R at tau = 0.5")
            for e in curve[a]:
                want = PAPER["curve"][a][f"{e['tau']:.2f}"]
                for k, w in zip(("P", "R", "F1"), want):
                    c.printed(e[k], w, f"paper tab:curve: {a} {k} at tau = {e['tau']:.2f}")
        for k, w in PAPER["pair_thresholds"].items():
            c.printed(thr[k], w, f"paper: pair block-held-out threshold {k}")
        c.ok(len(grid) == PAPER["fig4"]["thresholds"] and ahead == PAPER["fig4"]["pair_ahead"],
             "paper: pair ahead at 99 of 99 thresholds")
        c.printed(min_gap, PAPER["fig4"]["min_gap"], "paper: smallest gap (4 dp)")
        c.printed(min_gap, PAPER["fig4"]["min_gap_3dp"], "paper: smallest gap (3 dp)")
        c.printed(min_gap_tau, PAPER["fig4"]["min_gap_tau"], "paper: tau of the smallest gap")
    return out, c


def panel_letter(fig: plt.Figure, ax: plt.Axes, letter: str) -> None:
    """Write the panel letter at the figure's left edge, level with the top of ``ax``."""
    top = ax.get_position().y1
    fig.text(0.006, top + 0.004, letter, ha="left", va="top", fontsize=7.5, fontweight="bold")


def draw(v: dict) -> plt.Figure:
    """Draw both panels from the checked values.

    Args:
        v: output of ``compute``.

    Returns:
        The figure, WIDTH_IN x HEIGHT_IN.
    """
    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN))
    gs = fig.add_gridspec(2, 1, left=0.135, right=0.972, top=0.905, bottom=0.092, hspace=0.36)
    axa, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # (a) precision against recall
    for a in ARMS:
        st = ARM_STYLE[a]
        axa.plot(v["pr"][a]["recall"], v["pr"][a]["precision"], drawstyle="steps-post",
                 color=st["color"], ls=st["ls"], lw=st["lw"], zorder=st["z"])
    for a in ARMS:
        axa.plot([v["tau05"][a]["R"]], [v["tau05"][a]["P"]], ls="none", zorder=6,
                 **ARM_STYLE[a]["marker"])
    base = v["json"]["base_rate"]
    axa.axhline(base, color=GREY, lw=0.8, ls=(0, (1, 1.5)), zorder=1)
    axa.text(0.012, base + 0.007, "accept everything", fontsize=TXT, color=DARK, ha="left", va="bottom")
    axa.text(0.652, 0.906, r"$\tau = 0.5$", fontsize=TXT, color=DARK, ha="center", va="center")
    axa.text(0.012, 0.872, "database ingestion", fontsize=TXT, color=GREY, style="italic",
             ha="left", va="center")
    axa.text(0.905, 0.635, "expert review", fontsize=TXT, color=GREY, style="italic",
             ha="right", va="center")
    axa.set_xlim(0, 1.01)
    axa.set_ylim(0.5, 1.012)
    axa.set_xticks(np.round(np.arange(0, 1.01, 0.2), 1))
    axa.set_yticks(np.round(np.arange(0.5, 1.01, 0.1), 1))
    axa.set_xlabel("recall")
    axa.set_ylabel("precision")
    axa.grid(color=GRIDC, lw=0.55)

    # (b) against the decision threshold
    lo, hi = v["band"]
    axb.axvspan(lo, hi, color=GREY, alpha=0.22, lw=0, zorder=0)
    axb.text(hi + 0.012, 1.003, "block-held-out", fontsize=TXT, color=DARK, ha="left", va="top")
    g = v["grid"]
    axb.plot(g, v["pair_P"], color=DARK, lw=0.8, ls="-", zorder=2)
    axb.plot(g, v["pair_R"], color=DARK, lw=0.8, ls="-.", zorder=2)
    st_s, st_p = ARM_STYLE["sentence"], ARM_STYLE["pair"]
    axb.plot(g, v["sent_F1"], color=st_s["color"], lw=1.2, ls=st_s["ls"], zorder=3)
    axb.plot(g, v["pair_F1"], color=st_p["color"], lw=1.6, ls=st_p["ls"], zorder=4)
    axb.text(0.60, 0.918, "precision", fontsize=TXT, color=DARK, ha="center", va="top")
    axb.text(0.70, 0.636, "recall", fontsize=TXT, color=DARK, ha="center", va="baseline")
    axb.text(0.80, 0.722, "F1", fontsize=TXT, color=st_p["color"], ha="center", va="bottom")
    axb.text(0.72, 0.470, "F1", fontsize=TXT, color=st_s["color"], ha="center", va="top")
    axb.set_xlim(0, 1)
    axb.set_ylim(0.15, 1.02)
    axb.set_xticks(np.round(np.arange(0, 1.01, 0.2), 1))
    axb.set_yticks(np.round(np.arange(0.2, 1.01, 0.2), 1))
    axb.set_xlabel(r"decision threshold $\tau$")
    axb.grid(axis="y", color=GRIDC, lw=0.55)

    for ax in (axa, axb):
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    panel_letter(fig, axa, "(a)")
    panel_letter(fig, axb, "(b)")

    handles = [Line2D([0], [0], color=ARM_STYLE[a]["color"], ls=ARM_STYLE[a]["ls"],
                      lw=ARM_STYLE[a]["lw"], **ARM_STYLE[a]["marker"]) for a in ARMS]
    fig.legend(handles, [ARM_STYLE[a]["label"] for a in ARMS], loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handlelength=2.1,
               handletextpad=0.4, columnspacing=0.9, borderaxespad=0.25)
    return fig


def text_inside(fig: plt.Figure, margin_pt: float = 1.0) -> list[str]:
    """Names of text items (and the legend) that reach outside the figure, i.e. would be cut off.

    Args:
        fig: the drawn figure.
        margin_pt: required clearance from the figure edge, in points.

    Returns:
        A description of every offending item; empty when everything fits.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    m = margin_pt * fig.dpi / 72
    W, H = fig.bbox.width, fig.bbox.height
    bad = []
    items = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    items += [lg for lg in fig.legends]
    for t in items:
        bb = t.get_window_extent(r)
        if bb.x0 < m or bb.y0 < m or bb.x1 > W - m or bb.y1 > H - m:
            name = t.get_text() if isinstance(t, Text) else "legend"
            bad.append(f"{name!r} at ({bb.x0:.1f}, {bb.y0:.1f}, {bb.x1:.1f}, {bb.y1:.1f}) px, figure {W:.0f} x {H:.0f}")
    return bad


def jsonable(x):
    """Convert numpy scalars inside the output dict for json.dump."""
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    raise TypeError(type(x))


def main() -> int:
    """Build Figure 2 and its JSON, or exit non-zero if a check fails."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", choices=sorted(LABEL_VERSIONS), default="current",
                    help="benchmark label version (default: current, the one the paper reports)")
    ap.add_argument("--results-dir", default=str(RESULTS), help="score vectors and result tables")
    ap.add_argument("--out-dir", default=str(FIGDIR), help="where the figure, preview and JSON go")
    args = ap.parse_args()

    v, checks = compute(args.labels, Path(args.results_dir))
    if checks.failed:
        print(f"{len(checks.failed)} check(s) failed, nothing written:", file=sys.stderr)
        for f in checks.failed:
            print("  " + f, file=sys.stderr)
        return 1
    print(f"{checks.passed} checks passed ({'with' if args.labels == 'current' else 'without'} "
          f"the paper's printed values)")

    family = truetype_serif()
    set_style(family)
    fig = draw(v)
    cut = text_inside(fig)
    if cut:
        print("text outside the figure, nothing written:", *cut, sep="\n  ", file=sys.stderr)
        return 1
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{STEM}.pdf", metadata={"Creator": None, "Producer": None, "CreationDate": None,
                                               "Author": None, "Title": None})
    fig.savefig(out / f"{STEM}.png", dpi=300, metadata={"Software": None})
    plt.close(fig)
    v["json"]["font"] = family
    (out / f"{STEM}.json").write_text(json.dumps(v["json"], indent=1, default=jsonable) + "\n")
    j = v["json"]
    print(f"font {family}; AUPRC (ensembles) " + ", ".join(f"{a} {j[a]['auprc_ensemble']:.4f}" for a in ARMS))
    print(f"pair F1 ahead at {j['fig4']['pair_ahead']} of {j['fig4']['thresholds']} thresholds, "
          f"smallest gap {j['fig4']['min_gap']:.4f} at tau = {j['fig4']['min_gap_tau']:.2f}")
    print(f"wrote {out / STEM}.{{pdf,png,json}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
