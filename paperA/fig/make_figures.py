#!/usr/bin/env python3
"""Figures for Paper A (query-conditioned verification of literature-mined
biotic-interaction triples).

Style contract, unchanged from the previous revision: Okabe-Ito palette (CVD-safe),
serif, ACL single-column width 3.3in, no top/right spines, and every series carries a
distinct marker, hatch or linestyle so identity survives greyscale printing.

All input data are embedded as literals so the figures are reproducible from this file
alone.  Provenance for each block is given at its definition; every value was recomputed
on 2026-09-22 against the FORM query (the representation the model is trained on), which
is what paper_numbers.md reports.  Nothing here comes from the pre-bugfix TERM harness.

    python3 make_figures.py          # writes fig1..fig4 as .pdf and .png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent
BLUE, VERM, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#6b7873"
INK = "#3d4a45"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "grid.linewidth": 0.4, "grid.alpha": 0.35, "lines.linewidth": 1.4,
    "figure.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


# ════════════════════════════════════════════════════════════════════════════
# DATA
# ════════════════════════════════════════════════════════════════════════════

# ── Biotx100 ────────────────────────────────────────────────────────────────
# BX_S : verifier (V3) score per candidate, form query, mean of 3 seeds.
#        source: scratchpad/S_form_v3.npy
# BX_Y : expert grade, `triples_ok_full` in scratchpad/biotx100_scored.csv (65 supported).
# Sweeping BX_S reproduces the operating-curve table in paper_numbers.md exactly
# (thr .001/.014/.025/.100/.219/.850 -> keep 87/80/76/66/61/54).
BX_S = np.array([
    0.717283, 0.045450, 0.198612, 0.976780, 0.998690, 0.004810, 0.971527, 0.994303,
    0.998586, 0.726536, 0.994528, 0.998609, 0.987875, 0.807779, 0.998601, 0.227256,
    0.998554, 0.991426, 0.001152, 0.996583, 0.031554, 0.998348, 0.001548, 0.998642,
    0.998209, 0.014725, 0.000528, 0.998353, 0.980673, 0.997548, 0.002884, 0.996940,
    0.000806, 0.998629, 0.998806, 0.998525, 0.998760, 0.053278, 0.416976, 0.998239,
    0.991715, 0.998503, 0.998608, 0.996163, 0.998695, 0.995259, 0.998587, 0.998614,
    0.182210, 0.991100, 0.900598, 0.000843, 0.140848, 0.000329, 0.998217, 0.026791,
    0.000427, 0.998282, 0.996160, 0.995292, 0.083974, 0.157431, 0.000611, 0.998488,
    0.042685, 0.969636, 0.998701, 0.024148, 0.002298, 0.616215, 0.998590, 0.984305,
    0.982702, 0.024804, 0.000681, 0.998561, 0.000375, 0.064613, 0.107070, 0.002080,
    0.000736, 0.998053, 0.995268, 0.053754, 0.998740, 0.995759, 0.998833, 0.000325,
    0.993359, 0.098374, 0.001467, 0.040514, 0.905646, 0.995643, 0.016830, 0.998007,
    0.720177, 0.000361, 0.000675, 0.000366,
])
BX_Y = np.array([
    1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1,
    1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1,
    1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1,
    0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0,
])

# V1 = the deployed sentence classifier's own accept/reject column in the benchmark file
# (`classifier` in biotx100_scored.csv): 91 accepts, 65 TP / 26 FP / 0 FN / 9 TN.
V1_PREC, V1_REC, V1_KEEP = 65 / 91, 1.0, 91
BASE_PREC = 0.65                     # accept everything: 65/100
THR_OP = 0.219                       # the reported operating point

# ── Biotx100 false-positive mechanism decomposition ─────────────────────────
# V1's 26 false positives hand-classified in scratchpad/fp_analysis.csv (`mech`);
# the V3 column is how many of each class survive the verifier at THR_OP, recovered
# by joining that file to BX_S on `id`.
MECH = [                             # (label, V1 count, still accepted by V3 @0.219)
    ("pair-binding",          11, 1),
    ("predicate misbinding",   8, 1),
    ("entity",                 7, 2),
]

# ── Reject50 ───────────────────────────────────────────────────────────────
# 50 curated pipeline REJECTIONS; 11 are true positives the pipeline threw away.
# RJ_S : verifier score, form query.   RJ_Y : curated gold.
# source: scratchpad/reject50_scored.csv (`p_v3_form`, `gold_label`).
# Reproduces paper_numbers.md at both reported thresholds: 10/11 recovered at 0.10,
# 8/11 at 0.219 with 5/39 wrongly re-accepted.
RJ_S = np.array([
    0.000263, 0.000420, 0.183172, 0.000337, 0.019026, 0.985451, 0.001817, 0.942302,
    0.001499, 0.000846, 0.521718, 0.000254, 0.000269, 0.002612, 0.004266, 0.884816,
    0.011900, 0.000301, 0.000260, 0.000309, 0.000329, 0.975581, 0.000281, 0.000332,
    0.025144, 0.000265, 0.965953, 0.000482, 0.000223, 0.692157, 0.002249, 0.000270,
    0.154825, 0.996732, 0.982022, 0.852161, 0.000268, 0.000248, 0.000274, 0.000265,
    0.000285, 0.000517, 0.004608, 0.333076, 0.000424, 0.282061, 0.971000, 0.018299,
    0.000380, 0.007351,
])
RJ_Y = np.array([
    0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
])

# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════
def pr_curve(scores, labels):
    """Precision and recall of the kept set at every achievable threshold."""
    order = np.argsort(-scores)
    y = labels[order]
    tp = np.cumsum(y)
    kept = np.arange(1, len(y) + 1)
    return tp / kept, tp / labels.sum(), scores[order]


def sweep(scores, labels, grid):
    """(precision-of-kept, recall, n_kept) on an explicit threshold grid."""
    prec, rec, keep = [], [], []
    pos = labels.sum()
    for t in grid:
        k = scores >= t
        n = int(k.sum())
        tp = int((k & (labels == 1)).sum())
        prec.append(tp / n if n else np.nan)
        rec.append(tp / pos)
        keep.append(n)
    return np.array(prec), np.array(rec), np.array(keep)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — the deployed filter is one point on the verifier's own curve
# ════════════════════════════════════════════════════════════════════════════
# ARGUMENT: the deployed classifier does not sit below the verifier, it sits ON it —
# at matched acceptance the verifier reproduces V1 exactly — so the contribution is a
# tunable calibrated score where there was a single fixed hard decision.
fig, ax = plt.subplots(figsize=(3.3, 2.45))

# per-component oracle bound: 16 of the 35 unsupported triples have all three
# components individually correct, so a component-wise checker admits all 16 at
# every operating point.  Source: biotx100_audit.csv component grades.
_r = np.linspace(0.44, 1.0, 200)
ax.plot(_r, 65 * _r / (65 * _r + 16), color=GREY, lw=1.0, ls=(0, (5, 2)),
        zorder=2, label="per-component oracle bound")
ax.fill_between(_r, 0.60, 65 * _r / (65 * _r + 16), color=GREY, alpha=0.07,
                lw=0, zorder=1)

prec, rec, srt = pr_curve(BX_S, BX_Y)
ax.plot(rec, prec, color=BLUE, lw=1.5, solid_capstyle="round", zorder=3,
        label="query-conditioned verifier (all thresholds)")

# the three points that matter, all on or at the end of that curve
ax.scatter([1.0], [BASE_PREC], s=30, color="white", marker="o", zorder=5,
           edgecolor=GREY, linewidth=1.1, label="accept every candidate")
ax.scatter([V1_REC], [V1_PREC], s=42, color=VERM, marker="s", zorder=6,
           edgecolor="white", linewidth=0.8,
           label="deployed filter; verifier @ 91 kept")
ax.scatter([0.8769], [0.9344], s=46, color=BLUE, marker="^", zorder=6,
           edgecolor="white", linewidth=0.8, label=r"verifier @ $\tau=0.219$")

# the move the paper is actually claiming
ax.annotate("", xy=(0.8830, 0.9180), xytext=(0.9930, 0.7320),
            arrowprops=dict(arrowstyle="->", color=INK, lw=0.85,
                            shrinkA=3, shrinkB=6,
                            connectionstyle="arc3,rad=-0.32"), zorder=4)
ax.text(0.8760, 0.7950, "$-$22 FP\n$-$8 TP", color=INK, fontsize=6.8,
        ha="right", va="center", linespacing=1.15)
ax.text(0.9930, 0.7000, "91 kept", color=VERM, fontsize=6.4, ha="right", va="top")

ax.set_xlabel("recall of supported triples")
ax.set_ylabel("precision of the kept set")
ax.set_xlim(0.44, 1.035)
ax.set_ylim(0.62, 1.015)
ax.set_yticks([0.65, 0.75, 0.85, 0.95])
ax.grid(True, axis="both")
ax.legend(loc="lower center", bbox_to_anchor=(0.46, -0.56), ncol=1,
          frameon=False, handletextpad=0.5, borderpad=0.15, labelspacing=0.28)
fig.savefig(OUT / "fig1_operating_curve.pdf")
fig.savefig(OUT / "fig1_operating_curve.png")
plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — which failure class the reformulation repairs
# ════════════════════════════════════════════════════════════════════════════
# ARGUMENT: the reformulation repairs the two ARGUMENT-BINDING classes almost
# completely (10/11 and 7/8) and entity failures only partly (5/7) — it fixes exactly
# the errors a sentence-level decision cannot see, and nothing else.
fig, ax = plt.subplots(figsize=(2.95, 1.85))

labels = [m[0] for m in MECH]
tot = np.array([m[1] for m in MECH], float)
left = np.array([m[2] for m in MECH], float)
fixed = tot - left
ypos = np.arange(len(MECH))[::-1]

b1 = ax.barh(ypos, fixed, 0.58, color=BLUE, edgecolor="white", linewidth=0.8,
             label="repaired")
b2 = ax.barh(ypos, left, 0.58, left=fixed, color="white", edgecolor=VERM,
             linewidth=0.9, hatch="////", label="still accepted @ $\\tau{=}0.219$")

for y, f, l, t in zip(ypos, fixed, left, tot):
    ax.text(f / 2, y, f"{int(f)}", ha="center", va="center", fontsize=7.5,
            color="white", fontweight="bold")
    ax.text(f + l + 0.28, y, f"{int(t)}\u2009$\\to$\u2009{int(l)}", ha="left",
            va="center", fontsize=7, color=INK)

ax.set_yticks(ypos)
ax.set_yticklabels(labels)
ax.set_xlabel("false positives of the deployed filter (of 26)")
ax.set_xlim(0, 12.6)
ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
ax.set_xlabel("false positives of the deployed filter (of 26)", labelpad=2)
ax.set_ylim(-0.62, len(MECH) - 0.38)
ax.grid(True, axis="x")
ax.set_axisbelow(True)
ax.legend(loc="lower center", bbox_to_anchor=(0.46, -0.52), ncol=2, frameon=False,
          handletextpad=0.5, handlelength=1.5, columnspacing=0.8, borderpad=0.15)
fig.savefig(OUT / "fig2_mechanism.pdf")
fig.savefig(OUT / "fig2_mechanism.png")
plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — one threshold moves both sides
# ════════════════════════════════════════════════════════════════════════════
# ARGUMENT: the same calibrated score improves the kept side and the discarded side at
# once — at tau = 0.219 it clears 22 of the deployed filter's 26 false positives while
# recovering 8 of the 11 true positives the pipeline had already thrown away.
fig, ax = plt.subplots(figsize=(3.3, 2.35))

grid = np.logspace(np.log10(2e-4), np.log10(0.999), 400)
bx_prec, _, _ = sweep(BX_S, BX_Y, grid)
rj_rec = np.array([((RJ_S >= t) & (RJ_Y == 1)).sum() for t in grid]) / 11.0
rj_bad = np.array([((RJ_S >= t) & (RJ_Y == 0)).sum() for t in grid]) / 39.0

ax.plot(grid, bx_prec, color=BLUE, ls="-", lw=1.5, marker="^", markevery=45,
        ms=4, markeredgecolor="white", markeredgewidth=0.6, zorder=4,
        label="Biotx100: precision of kept")
ax.plot(grid, rj_rec, color=GREEN, ls="--", lw=1.4, marker="o", markevery=(20, 45),
        ms=3.8, markeredgecolor="white", markeredgewidth=0.6, zorder=4,
        label="Reject50: true positives recovered (/11)")
ax.plot(grid, rj_bad, color=VERM, ls="-.", lw=1.3, marker="s", markevery=(33, 45),
        ms=3.4, markeredgecolor="white", markeredgewidth=0.6, zorder=4,
        label="Reject50: correct rejections undone (/39)")

ax.axhline(V1_PREC, color=GREY, lw=0.9, ls=":", zorder=2)
ax.text(1.1e-3, V1_PREC - 0.045, "deployed filter precision 0.714",
        color=GREY, fontsize=6.4, ha="left", va="top")
ax.axvline(THR_OP, color=INK, lw=0.7, ls=(0, (4, 2)), zorder=2)
ax.text(THR_OP * 0.86, 0.235, r"$\tau=0.219$", color=INK, fontsize=6.6,
        ha="right", va="bottom", rotation=90)

ax.set_xscale("log")
ax.set_xlabel(r"verifier threshold $\tau$")
ax.set_ylabel("fraction")
ax.set_xlim(2e-4, 1.0)
ax.set_ylim(-0.02, 1.04)
ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.grid(True, axis="y")
ax.set_axisbelow(True)
ax.legend(loc="lower center", bbox_to_anchor=(0.46, -0.62), ncol=1, frameon=False,
          handletextpad=0.5, handlelength=2.2, labelspacing=0.28, borderpad=0.15)
fig.savefig(OUT / "fig3_two_sided.pdf")
fig.savefig(OUT / "fig3_two_sided.png")
plt.close(fig)


print("wrote fig1_operating_curve, fig2_mechanism, fig3_two_sided "
      "(.pdf and .png each)")
