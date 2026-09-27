#!/usr/bin/env python3
"""Figure 1: swap consistency against the positional prior pi.

Two series on one axis. The orange points are three corpora as they come, which is
observational. The blue curve is SemEval resampled to a target pi by changing only the order
in which the two arguments are presented -- same passages, pairs, relations and test set --
which is the controlled version of the same claim.

Okabe-Ito palette (CVD-safe), serif, single column. Validated against the lightness band,
chroma floor, adjacent and all-pairs CVD separation, normal-vision floor and contrast on
white; worst CVD deltaE 11.0 (deutan), worst normal-vision deltaE 18.7.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, VERM, GREEN, GREY, INK = "#0072B2", "#D55E00", "#009E73", "#9a9a95", "#1a1a19"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 6.8, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.edgecolor": "#55554f", "figure.dpi": 200,
})

HERE = Path(__file__).parent
SWEEP = Path.home() / "biored_baseline" / "PI_SWEEP.json"

# three corpora as found: name -> (pi, per-seed unconstrained swap consistency)
SEEDS = {"Biodiversity": (0.708, [0.473]),
         "SemEval-2010": (0.544, [0.9588, 0.9066, 0.9496]),
         "BioRED":       (0.486, [0.9809, 0.9694, 0.9569])}
OBS = [(k, pi, float(np.mean(v))) for k, (pi, v) in SEEDS.items()]

fig, ax = plt.subplots(figsize=(3.4, 2.7))

ax.axhline(1.0, color=GREEN, lw=1.5, zorder=3)
ax.text(0.995, 1.010, "constrained arm: exact at every $\\pi$", color=GREEN, fontsize=6.6,
        ha="right", va="bottom")
ax.axhline(0.5, color=GREY, lw=0.8, ls=(0, (3, 2)), zorder=1)
ax.text(0.409, 0.515, "chance", color=GREY, fontsize=6.4, va="bottom")

# controlled sweep
if SWEEP.exists():
    rows = [r for r in json.load(open(SWEEP)) if r["arm"] == "text_uncon"]
    rows.sort(key=lambda r: r["pi"])
    px = [r["pi"] for r in rows]; py = [r["swap"] for r in rows]
    pe = [r["swap_sd"] or 0.0 for r in rows]
    ax.errorbar(px, py, yerr=pe, color=BLUE, lw=1.3, marker="o", ms=3.6,
                capsize=1.8, elinewidth=0.8, mec="white", mew=0.7, zorder=6,
                label="SemEval, $\\pi$ set by resampling order")

ax.scatter([p[1] for p in OBS], [p[2] for p in OBS], s=44, color=VERM, marker="D",
           edgecolor="white", linewidth=0.9, zorder=7, label="corpora as found")

# pi is readable from the x position, so the labels carry only the corpus name
LAB = {"BioRED":       (-7,   0, "right",  "center"),
       "SemEval-2010": ( 0, -11, "center", "top"),
       "Biodiversity": ( 0,  10, "center", "bottom")}
for name, x, y in OBS:
    dx, dy, ha, va = LAB[name]
    ax.annotate(name, (x, y), xytext=(dx, dy), va=va,
                textcoords="offset points", ha=ha, fontsize=6.6, color=INK)

ax.set_xlabel("$\\pi$   (P[first-listed argument is the subject], training set)")
ax.set_ylabel("swap consistency")
ax.set_xlim(0.405, 1.02)
ax.set_ylim(-0.03, 1.09)
ax.set_yticks(np.arange(0.0, 1.01, 0.2))
ax.set_xticks(np.arange(0.5, 1.01, 0.1))
ax.grid(axis="y", color="#e8e8e4", lw=0.55, zorder=0)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(loc="lower left", frameon=False, handlelength=1.6, borderpad=0.2,
          labelspacing=0.35, bbox_to_anchor=(0.015, 0.045))
fig.tight_layout(pad=0.35)
for ext in ("pdf", "png"):
    fig.savefig(HERE / f"fig1_pi_consistency.{ext}", bbox_inches="tight")
print("wrote fig1_pi_consistency.{pdf,png}")
