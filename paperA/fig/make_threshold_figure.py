#!/usr/bin/env python3
"""Precision / recall / F1 against the decision threshold, with V1 as the reference.

Style contract matches make_figures.py: Okabe-Ito (CVD-safe), serif, two-column width.
Palette validated against the six checks (lightness band, chroma floor, adjacent AND
all-pairs CVD separation, normal-vision floor, contrast on white): worst CVD deltaE 11.0
(deutan), worst normal-vision deltaE 18.7, all contrasts >= 3.0 on a white page.

Series are direct-labelled at the curve ends, so identity never rests on colour alone.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

BLUE, VERM, GREEN, GREY, INK = "#0072B2", "#D55E00", "#009E73", "#9a9a95", "#1a1a19"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.edgecolor": "#55554f", "figure.dpi": 200,
})


def load():
    d = pd.read_csv(REPO / "data/evaluation/unified_test_set.csv")
    d["maxj"] = pd.read_csv(REPO / "results/test_contamination_scan.csv").maxj.values
    leak = ((d.maxj >= 0.5) | d.in_train).to_numpy()
    keep = (~leak)[~d.in_train.to_numpy()]
    y = d[~leak].label.to_numpy()
    V = np.load(REPO / "results/v1_decisions_449.npy")[~leak]
    S = np.load(REPO / "results/dirhead/joint_a05_s1_binary_scores.npy")[keep]
    return y, V, S


def curves(y, S, grid):
    P = np.array([precision_score(y, (S >= t).astype(int), zero_division=0) for t in grid])
    R = np.array([recall_score(y, (S >= t).astype(int), zero_division=0) for t in grid])
    F = np.array([f1_score(y, (S >= t).astype(int), zero_division=0) for t in grid])
    return P, R, F


def main(out_stem="fig4_threshold", wide=False):
    y, V, S = load()
    grid = np.arange(0.01, 1.00, 0.01)
    P, R, F = curves(y, S, grid)
    vP, vR, vF = precision_score(y, V), recall_score(y, V), f1_score(y, V)

    i50 = int(np.argmin(np.abs(grid - 0.50)))
    # Recommended operating point, under the project's stated policy (precision first,
    # F1 secondary): the highest precision that still keeps recall >= 0.90 -- i.e. a clear
    # recall margin over V1's 0.785 -- and F1 within 2 points of its maximum.
    ok = np.where((R >= 0.90) & (F >= F.max() - 0.020))[0]
    irec = ok[np.argmax(P[ok])]
    trec = grid[irec]

    fig, ax = plt.subplots(figsize=(6.6, 3.6) if wide else (3.3, 2.5))
    ax.plot(grid, P, color=BLUE, lw=1.6, solid_capstyle="round", zorder=4)
    ax.plot(grid, R, color=VERM, lw=1.6, solid_capstyle="round", zorder=4)
    ax.plot(grid, F, color=GREEN, lw=1.6, solid_capstyle="round", zorder=4)

    # shade the region that satisfies the policy, so the choice is visible not asserted
    lo_t, hi_t = grid[ok.min()], grid[ok.max()]
    ax.axvspan(lo_t, hi_t, color=BLUE, alpha=0.055, lw=0, zorder=0)
    ax.axhline(vP, color=GREY, lw=0.9, ls=(0, (4, 2)), zorder=1)
    ax.axhline(vR, color=GREY, lw=0.9, ls=(0, (1, 2)), zorder=1)
    # reference key in the dead space bottom-left, so the lines are identified without
    # dropping text on top of a curve
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=GREY, lw=0.9, ls=(0, (4, 2)),
                              label=f"V1 precision {vP:.3f}"),
                       Line2D([], [], color=GREY, lw=0.9, ls=(0, (1, 2)),
                              label=f"V1 recall {vR:.3f}")],
              loc="lower left", frameon=False, fontsize=6.3, handlelength=2.0,
              labelcolor=GREY, borderaxespad=0.3, handletextpad=0.5,
              borderpad=0.1, labelspacing=0.25)

    for t, lab, col in ((0.50, "shipped\n0.50", INK), (trec, f"proposed\n{trec:.2f}", BLUE)):
        ax.axvline(t, color=col, lw=0.8, ls=(0, (2, 2)), alpha=0.55, zorder=2)
    ax.scatter([0.50] * 3, [P[i50], R[i50], F[i50]], s=17, zorder=6,
               c=[BLUE, VERM, GREEN], edgecolor="white", linewidth=0.7)
    ax.scatter([trec] * 3, [P[irec], R[irec], F[irec]], s=30, marker="D", zorder=6,
               c=[BLUE, VERM, GREEN], edgecolor="white", linewidth=0.7)

    xt = grid[-1] + 0.012
    for val, name, col in ((P[-1], "precision", BLUE), (R[-1], "recall", VERM), (F[-1], "F1", GREEN)):
        ax.text(xt, val, name, color=col, fontsize=7, va="center", ha="left")

    ax.text(0.50, 1.000, "shipped 0.50", color=INK, fontsize=6.4, ha="center", va="bottom")
    ax.text(trec, 1.000, f"proposed {trec:.2f}", color=BLUE, fontsize=6.4, ha="center", va="bottom")

    ax.set_xlabel("decision threshold")
    ax.set_ylabel("score")
    ax.set_xlim(0, 1.18 if not wide else 1.12)
    ax.set_ylim(0.70, 1.002)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0.70, 1.01, 0.05))
    ax.grid(axis="y", color="#e8e8e4", lw=0.55, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(Path(__file__).parent / f"{out_stem}.{ext}", bbox_inches="tight")
    plt.close(fig)

    print(f"{'threshold':>10} {'precision':>10} {'recall':>8} {'F1':>8}")
    for t in (0.50, trec, 0.90, 0.95, 0.99):
        i = int(np.argmin(np.abs(grid - t)))
        tag = "  <- shipped" if abs(t - 0.50) < 1e-9 else ("  <- proposed" if abs(t - trec) < 1e-9 else "")
        print(f"{grid[i]:10.2f} {P[i]:10.4f} {R[i]:8.4f} {F[i]:8.4f}{tag}")
    print(f"{'V1':>10} {vP:10.4f} {vR:8.4f} {vF:8.4f}")
    print(f"\nproposed tau={trec:.2f}: precision {P[irec]:.4f} (+{P[irec]-P[i50]:.4f} vs shipped), "
          f"F1 {F[irec]:.4f} ({F[irec]-F[i50]:+.4f})")
    return trec


if __name__ == "__main__":
    main(wide="--wide" in sys.argv,
         out_stem="fig4_threshold_wide" if "--wide" in sys.argv else "fig4_threshold")
