#!/usr/bin/env python3
"""Figure 1: swap consistency of the unconstrained arm against the corpus prior pi.

Okabe-Ito palette (CVD-safe), serif, two-column width. The palette was validated against the
lightness band, chroma floor, adjacent and all-pairs CVD separation, normal-vision floor and
contrast on white; worst CVD deltaE 11.0 (deutan), worst normal-vision deltaE 18.7.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, VERM, GREEN, GREY, INK = "#0072B2", "#D55E00", "#009E73", "#9a9a95", "#1a1a19"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.edgecolor": "#55554f", "figure.dpi": 200,
})

# corpus, pi, per-seed unconstrained swap consistency (order-sensitive input)
SEEDS = {"Biodiversity": (0.708, [0.473]),
         "SemEval-2010": (0.544, [0.9588, 0.9066, 0.9496]),
         "BioRED":       (0.486, [0.9809, 0.9694, 0.9569])}
PTS = [(k, pi, sum(v) / len(v)) for k, (pi, v) in SEEDS.items()]

fig, ax = plt.subplots(figsize=(3.4, 2.6))

# ours: exact at any pi
ax.axhline(1.0, color=BLUE, lw=1.6, zorder=3)
ax.text(0.742, 1.008, "ours: exact, any $\\pi$", color=BLUE, fontsize=6.8,
        ha="right", va="bottom")

ax.axhline(0.5, color=GREY, lw=0.8, ls=(0, (3, 2)), zorder=1)
ax.text(0.455, 0.512, "chance", color=GREY, fontsize=6.4, va="bottom")

xs = [p[1] for p in PTS]; ys = [p[2] for p in PTS]
# seed spread, where more than one seed was run
for name, pi, mu in PTS:
    v = SEEDS[name][1]
    if len(v) > 1:
        ax.plot([pi, pi], [min(v), max(v)], color=VERM, lw=1.0, solid_capstyle="butt", zorder=4)
ax.scatter(xs, ys, s=42, color=VERM, edgecolor="white", linewidth=0.9, zorder=5)

LAB = {  # (dx pt, dy pt, ha, va)
    "BioRED":       (  0, -13, "center", "top"),
    "SemEval-2010": ( 10,  -8, "left",   "top"),
    "Biodiversity": ( -6,  10, "right",  "bottom"),
}
for name, x, y in PTS:
    dx, dy, ha, va = LAB[name]
    ax.annotate(f"{name}\n$\\pi={x:.3f}$", (x, y), xytext=(dx, dy), va=va,
                textcoords="offset points", ha=ha, fontsize=6.6, color=INK,
                linespacing=1.25)

# the released system, which has no single pi we control
ax.scatter([0.486], [0.700], s=46, marker="s", color=GREY, edgecolor="white",
           linewidth=0.8, zorder=5)
ax.annotate("BioREDirect\n(released)", (0.486, 0.700), xytext=(14, -2),
            textcoords="offset points", ha="left", va="center",
            fontsize=6.6, color=GREY)

ax.set_xlabel("$\\pi$   (P[first-listed argument is the subject], training set)")
ax.set_ylabel("swap consistency")
# bars span min-max over three seeds; biodiversity is a single run
ax.set_xlim(0.452, 0.748)
ax.set_ylim(0.40, 1.06)
ax.set_yticks(np.arange(0.4, 1.01, 0.1))
ax.grid(axis="y", color="#e8e8e4", lw=0.55, zorder=0)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout(pad=0.35)
for ext in ("pdf", "png"):
    fig.savefig(Path(__file__).parent / f"fig1_pi_consistency.{ext}", bbox_inches="tight")
print("wrote fig1_pi_consistency.{pdf,png}")
