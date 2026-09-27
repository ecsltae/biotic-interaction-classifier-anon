#!/usr/bin/env python3
"""Figure 2: the controlled pi sweep -- accuracy and swap consistency diverge.

SemEval training pairs resampled to a target pi by changing only the order in which the two
arguments are presented. Same passages, pairs, relations and test set throughout, so pi is the
only thing that varies. Two seeds per point.

Okabe-Ito palette (CVD-safe), serif, single column.
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
    "legend.fontsize": 6.9, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.edgecolor": "#55554f", "figure.dpi": 200,
})

HERE = Path(__file__).parent
ALL = json.load(open(Path.home() / "biored_baseline" / "PI_SWEEP.json"))
rows = sorted([r for r in ALL if r["arm"] == "text_uncon"], key=lambda r: r["pi"])
brow = sorted([r for r in ALL if r["arm"] == "biored_text_uncon"], key=lambda r: r["pi"])
crow = sorted([r for r in ALL if r["arm"] == "canonical_sym"], key=lambda r: r["pi"])
x = np.array([r["pi"] for r in rows])
sw = np.array([r["swap"] for r in rows]); sw_e = np.array([r["swap_sd"] or 0 for r in rows])
ac = np.array([r["acc"] for r in rows]);  ac_e = np.array([r["acc_sd"] or 0 for r in rows])

fig, ax = plt.subplots(figsize=(3.4, 2.5))

# what the positional shortcut alone is worth on the training distribution
ax.plot(x, np.maximum(x, 1 - x), color=GREY, lw=0.9, ls=(0, (3, 2)), zorder=2)
ax.text(0.735, 0.665, "value of the\npositional shortcut", color=GREY, fontsize=6.3,
        ha="center", va="top", linespacing=1.2)

h_acc = ax.errorbar(x, ac, yerr=ac_e, color=VERM, lw=1.3, marker="s", ms=3.4, capsize=1.8,
            elinewidth=0.8, mec="white", mew=0.7, zorder=5, label="direction accuracy, SemEval")
h_sw = ax.errorbar(x, sw, yerr=sw_e, color=BLUE, lw=1.3, marker="o", ms=3.6, capsize=1.8,
            elinewidth=0.8, mec="white", mew=0.7, zorder=6, label="swap consistency, SemEval")
# the same intervention on a second corpus
bx = [r["pi"] for r in brow]; by = [r["swap"] for r in brow]
h_b, = ax.plot(bx, by, color=BLUE, lw=1.0, ls=(0, (4, 2)), marker="^", ms=3.2, mfc="white",
        mec=BLUE, mew=0.9, zorder=5, label="swap consistency, BioRED")
# the constrained arm, unaffected at either extreme
cx = [r["pi"] for r in crow]; cy = [r["swap"] for r in crow]
h_c, = ax.plot(cx, cy, color=GREEN, lw=1.4, marker="D", ms=3.4, mec="white", mew=0.7,
        zorder=7, label="constrained arm, both metrics")

ax.axhline(0.5, color=GREY, lw=0.7, ls=(0, (1, 2)), zorder=1)

ax.set_xlabel("$\\pi$ imposed on the training set")
ax.set_ylabel("direction accuracy / swap consistency", fontsize=7.4)
ax.set_xlim(0.47, 1.03)
ax.set_ylim(-0.03, 1.06)
ax.set_yticks(np.arange(0.0, 1.01, 0.2))
ax.set_xticks(np.arange(0.5, 1.01, 0.1))
ax.grid(axis="y", color="#e8e8e4", lw=0.55, zorder=0)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(handles=[h_sw, h_b, h_acc, h_c], loc="lower left", frameon=False,
          handlelength=1.7, borderpad=0.2, labelspacing=0.35, bbox_to_anchor=(0.0, 0.02))
fig.tight_layout(pad=0.35)
for ext in ("pdf", "png"):
    fig.savefig(HERE / f"fig2_pi_sweep.{ext}", bbox_inches="tight")
print("wrote fig2_pi_sweep.{pdf,png}")
