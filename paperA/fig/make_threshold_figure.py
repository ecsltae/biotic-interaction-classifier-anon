#!/usr/bin/env python3
"""Figure 4: precision, recall and F1 against the decision threshold.

Built from the rebuilt score vector (scripts/rebuild_paperA_numbers.py) for the model the paper
actually reports, on the 437 clean rows with the benchmark's species-level labels. An earlier
version of this script scored `dirhead/joint_a05_s1`, a different checkpoint, which is why its
output never matched the paper's numbers.

The point of the figure is that the verifier is a curve where a filter is a single point, and
that conditioning on the pair dominates the sentence-level formulation trained on the same data at
every threshold. Labels come from src/eval/core.py's clean_benchmark() (hash-checked), for the
label version given by --labels; the run prints how many of the 99 thresholds put the pair model
ahead and the smallest F1 gap.

Okabe-Ito palette (CVD-safe), serif, single column.

Usage
  python3 paperA/fig/make_threshold_figure.py
  python3 paperA/fig/make_threshold_figure.py --labels pre_review_2026-10-06 --results-dir <dir> --out-dir <dir>
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42   # TrueType, not Type 3: the ACL checker flags Type 3
plt.rcParams["ps.fonttype"] = 42  # noqa: E402
from sklearn.metrics import f1_score, precision_score, recall_score  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from eval.core import LABEL_VERSIONS, clean_benchmark  # noqa: E402

RESULTS = REPO / "results/paperA_v2"     # S_biodiv_pair.npy (pair-conditioned, order-free),
FIGDIR = Path(__file__).resolve().parent  # S_biodiv_sentence.npy and tables_biodiv.json

BLUE, VERM, GREEN, GREY, INK = "#0072B2", "#D55E00", "#009E73", "#9a9a95", "#1a1a19"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 6.9, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.edgecolor": "#55554f", "figure.dpi": 200,
})


def load(labels="current", results=RESULTS):
    d = clean_benchmark(labels)
    return d.label.to_numpy(), np.load(results / "S_biodiv_sentence.npy"), np.load(results / "S_biodiv_pair.npy")


def main(out_stem="fig4_threshold", wide=False, labels="current", results=RESULTS, out_dir=FIGDIR):
    y, SB, S = load(labels, results)
    grid = np.arange(0.01, 1.00, 0.01)
    P = np.array([precision_score(y, (S >= t).astype(int), zero_division=0) for t in grid])
    R = np.array([recall_score(y, (S >= t).astype(int), zero_division=0) for t in grid])
    F = np.array([f1_score(y, (S >= t).astype(int), zero_division=0) for t in grid])
    FB = np.array([f1_score(y, (SB >= t).astype(int), zero_division=0) for t in grid])
    gap = F - FB
    print(f"pair F1 above sentence F1 at {int((gap > 0).sum())} of {len(grid)} thresholds; "
          f"smallest gap {gap.min():.4f} at tau = {grid[int(np.argmin(gap))]:.2f}")

    fig, ax = plt.subplots(figsize=(6.6, 2.5) if wide else (3.35, 2.6))
    ax.plot(grid, P, color=BLUE, lw=1.4, label="precision")
    ax.plot(grid, R, color=VERM, lw=1.4, label="recall")
    ax.plot(grid, F, color=GREEN, lw=1.6, label="F1, pair-conditioned")

    # the controlled baseline: same data and encoder, passage only
    ax.plot(grid, FB, color=GREEN, lw=1.1, ls=(0, (4, 2)), zorder=2, label="F1, sentence-only baseline")

    # the reported operating points: one threshold per source block, each fitted on the other
    # two. They are not one number, so the figure shows their range rather than a single marker.
    import json
    thr = json.loads((results / "tables_biodiv.json").read_text())["pair"]["thresholds"]
    lo, hi = min(thr.values()), max(thr.values())
    ax.axvspan(lo - 0.004, hi + 0.004, color=GREY, alpha=0.18, lw=0, zorder=0)
    ax.text((lo + hi) / 2, 0.985, "block-held-out", color=INK, fontsize=6.3, ha="center", va="top")

    ax.set_xlabel("decision threshold $\\tau$")
    ax.set_ylabel("")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.15, 1.02)
    ax.set_yticks(np.arange(0.2, 1.01, 0.1))
    ax.grid(axis="y", color="#e8e8e4", lw=0.55, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower left", frameon=False, handlelength=1.6, labelspacing=0.3,
              bbox_to_anchor=(0.02, 0.02))
    fig.tight_layout(pad=0.35)
    for ext in ("pdf", "png"):
        fig.savefig(Path(out_dir) / f"{out_stem}.{ext}", bbox_inches="tight")
    print(f"wrote {Path(out_dir) / out_stem}.{{pdf,png}}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", choices=sorted(LABEL_VERSIONS), default="current")
    ap.add_argument("--results-dir", default=str(RESULTS), help="where the score vectors and tables_biodiv.json are")
    ap.add_argument("--out-dir", default=str(FIGDIR), help="where the figures are written")
    a = ap.parse_args()
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    main(labels=a.labels, results=Path(a.results_dir), out_dir=a.out_dir)
    main("fig4_threshold_wide", wide=True, labels=a.labels, results=Path(a.results_dir), out_dir=a.out_dir)
