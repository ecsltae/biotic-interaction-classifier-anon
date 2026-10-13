#!/usr/bin/env python3
"""F1 at equal share of candidates accepted (Paper A, Figure 2b), recomputed from rows.csv.

For each model and each k = 1..437 the model accepts its k highest-scoring candidates; share = k/437
and F1 = 2 TP / (k + P), with P = 251 positives. Scores are the three-checkpoint ensembles of
rows.csv (the file Figure 2 is drawn from; sentence-level, pair-conditioned order-free,
triple-conditioned order-free). rows.csv rounds the scores to six decimals, which creates a few ties
(none among the pair-conditioned scores); inside a tied group TP is its expected value when the
group is ordered at random. As a cross-check the same curves are computed from the full-precision
scores (results/paperA_v2/S_biodiv_*.npy, no ties) and with a stable sort of rows.csv.

Output: results/paperA_v2/fig2_equal_share.json (curves, pair-minus-sentence gaps and the ranges the
paper quotes).

Usage: python3 scripts/paperA_fig2_equal_share.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROWS = REPO / "paperA/fig/user_2026-10-07/rows.csv"
NPY = REPO / "results/paperA_v2/S_biodiv_{}.npy"
OUT = REPO / "results/paperA_v2/fig2_equal_share.json"
ARMS = ("sentence", "pair", "triple")
# tab:curve (Appendix): share accepted at each printed threshold, and F1 there
TAB3 = {
    "sentence": {0.01: (.746, .801), 0.04: (.600, .799), 0.10: (.517, .776), 0.25: (.462, .768),
                 0.50: (.341, .660), 0.90: (.098, .272)},
    "pair": {0.01: (.737, .855), 0.04: (.668, .877), 0.10: (.611, .880), 0.25: (.538, .881),
             0.50: (.421, .795), 0.90: (.247, .591)},
}


def topk_tp(y: np.ndarray, s: np.ndarray, ties: str = "expected") -> np.ndarray:
    """True positives among the k highest scores, for k = 1..n.

    Args:
        y: gold labels (0/1).
        s: scores.
        ties: "expected" (random order inside a tied group, expected TP) or "stable"
            (tied rows keep their file order).

    Returns:
        Array of length n with TP at k = 1..n (float when ties are averaged).
    """
    o = np.argsort(-s, kind="mergesort")
    ys, ss = y[o].astype(float), s[o]
    if ties == "stable":
        return np.cumsum(ys)
    tp = np.cumsum(ys)
    i = 0
    n = len(ss)
    while i < n:
        j = i
        while j + 1 < n and ss[j + 1] == ss[i]:
            j += 1
        if j > i:  # tied group i..j: expected TP grows linearly across the group
            before = tp[i - 1] if i > 0 else 0.0
            rate = ys[i:j + 1].mean()
            tp[i:j + 1] = before + rate * np.arange(1, j - i + 2)
        i = j + 1
    return tp


def f1_curve(y: np.ndarray, s: np.ndarray, ties: str = "expected") -> np.ndarray:
    """F1 when the model accepts its top k candidates, k = 1..n."""
    k = np.arange(1, len(y) + 1, dtype=float)
    return 2 * topk_tp(y, s, ties) / (k + y.sum())


def runs(mask: np.ndarray, k: np.ndarray, gap: np.ndarray, n: int) -> list[dict]:
    """Contiguous runs of k where mask holds, with their share range and gap extremes."""
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            g = gap[i:j + 1]
            out.append({"k_lo": int(k[i]), "k_hi": int(k[j]), "share_lo": k[i] / n,
                        "share_hi": k[j] / n, "n_k": int(j - i + 1), "gap_min": float(g.min()),
                        "gap_max": float(g.max())})
            i = j + 1
        else:
            i += 1
    return out


def summarise(y: np.ndarray, scores: dict[str, np.ndarray], ties: str) -> dict:
    """Pair-minus-sentence gap summaries over k/n from 1% to 100%."""
    n = len(y)
    k = np.arange(1, n + 1)
    f1 = {m: f1_curve(y, scores[m], ties) for m in ARMS}
    gap = f1["pair"] - f1["sentence"]
    grid = k / n >= 0.01
    kk, gg = k[grid], gap[grid]
    band = (k / n >= 0.50) & (k / n <= 0.70)
    neg = gg < 0
    out = {
        "k_range": [int(kk[0]), int(kk[-1])],
        "gap_min": float(gg.min()), "gap_min_k": int(kk[gg.argmin()]),
        "gap_max": float(gg.max()), "gap_max_k": int(kk[gg.argmax()]),
        "n_k": int(grid.sum()), "n_k_pair_behind": int(neg.sum()),
        "n_k_level": int((gg == 0).sum()), "n_k_pair_ahead": int((gg > 0).sum()),
        "pair_ahead_runs": runs(gg > 0, kk, gg, n),
        "pair_behind_runs": runs(neg, kk, gg, n),
        "pair_behind_or_level_runs": runs(gg <= 0, kk, gg, n),
        "within_0.01_runs": runs(np.abs(gg) < 0.01, kk, gg, n),
        "band_50_70": {"k_lo": int(k[band][0]), "k_hi": int(k[band][-1]),
                       "gap_min": float(gap[band].min()),
                       "gap_min_share": float(k[band][gap[band].argmin()] / n),
                       "gap_max": float(gap[band].max()),
                       "gap_max_share": float(k[band][gap[band].argmax()] / n),
                       "gap_min_2dp": round(float(gap[band].min()), 2),
                       "gap_max_2dp": round(float(gap[band].max()), 2)},
        "triple_minus_sentence_min": float((f1["triple"] - f1["sentence"])[grid].min()),
    }
    return out, f1, gap


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=Path, default=ROWS)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    r = pd.read_csv(a.rows)
    y = r.gold.to_numpy().astype(int)
    n, pos = len(y), int(y.sum())
    assert (n, pos) == (437, 251), (n, pos)
    scores = {m: r[m].to_numpy(dtype=float) for m in ARMS}

    main_summary, f1, gap = summarise(y, scores, "expected")
    stable_summary, _, gap_stable = summarise(y, scores, "stable")
    full = {m: np.load(str(NPY).format(m)) for m in ARMS}
    for m in ARMS:
        assert np.abs(full[m] - scores[m]).max() < 5.1e-7, m  # rows.csv = 6-decimal rounding
    full_summary, _, gap_full = summarise(y, full, "expected")

    k = np.arange(1, n + 1)
    pct_grid = []
    for p in range(1, 101):
        kp = int(round(p / 100 * n))
        pct_grid.append({"pct": p, "k": kp, "sentence": float(f1["sentence"][kp - 1]),
                         "pair": float(f1["pair"][kp - 1]), "triple": float(f1["triple"][kp - 1]),
                         "pair_minus_sentence": float(gap[kp - 1])})

    # The Table-3 reading: each model's own thresholds give different shares, so pairing rows
    # compares different workloads. Report the equal-share gap at each printed pair-model share.
    tab3 = []
    for tau, (sh_p, f_p) in TAB3["pair"].items():
        sh_s, f_s = TAB3["sentence"][tau]
        kp = int(round(sh_p * n))
        tab3.append({"tau": tau, "pair_share": sh_p, "pair_f1": f_p, "sentence_share": sh_s,
                     "sentence_f1": f_s, "table_gap_same_tau": round(f_p - f_s, 3),
                     "equal_share_gap_at_pair_share": float(gap[kp - 1])})

    # The statements the paper makes about Figure 2b, checked on every k with k/n >= 1%.
    sh = k / n
    on = sh >= 0.01
    within = on & (np.abs(gap) < 0.01)
    behind = on & (gap < 0)
    b5254 = (sh >= 0.52) & (sh <= 0.54)
    claims = {
        "intro_pair_higher_at_every_share_4_to_89pct": bool((gap[(sh >= 0.04) & (sh <= 0.89)] > 0).all()),
        "caption_ahead_or_level_except_93_to_96pct": bool(
            behind.any() and round(sh[behind].min(), 2) == 0.93 and round(sh[behind].max(), 2) == 0.96),
        "caption_trails_by_at_most_0.003": bool(round(float(-gap[behind].min()), 3) == 0.003),
        "sec41_band_50_70_gap_2dp": [main_summary["band_50_70"]["gap_min_2dp"],
                                     main_summary["band_50_70"]["gap_max_2dp"]],
        "sec41_within_0.01_only_below_10pct_or_above_85pct": bool(
            ((sh[within] < 0.10) | (sh[within] > 0.85)).all()),
        "min_gap_10_to_85pct": float(gap[(sh >= 0.10) & (sh <= 0.85)].min()),
        "old_text_within_0.01_only_below_5pct_or_above_89pct": bool(
            ((sh[within] < 0.05) | (sh[within] > 0.89)).all()),
        "old_text_counterexample_shares": [float(x) for x in sh[within & (sh >= 0.05) & (sh <= 0.89)]],
        "equal_share_gap_52_to_54pct": {"k_lo": int(k[b5254][0]), "k_hi": int(k[b5254][-1]),
                                        "gap_min": float(gap[b5254].min()),
                                        "gap_max": float(gap[b5254].max())},
    }

    out = {
        "claims_checked": claims,
        "description": "F1 of each model when it accepts its top-k candidates, k/437 from 1% to 100%; "
                       "three-checkpoint ensemble scores from rows.csv (Figure 2).",
        "source": str(a.rows.relative_to(REPO)),
        "script": "scripts/paperA_fig2_equal_share.py",
        "n": n, "positives": pos,
        "tie_handling": "rows.csv rounds scores to 6 decimals: ties in sentence (2 groups of 2), triple "
                        "(4 groups), none in pair; inside a tied group TP is its expected value under "
                        "random order. Cross-checks: stable sort, and full-precision S_biodiv_*.npy.",
        "pair_minus_sentence": main_summary,
        "crosscheck_stable_sort": stable_summary,
        "crosscheck_full_precision_npy": full_summary,
        "max_abs_gap_difference_vs_stable": float(np.abs(gap - gap_stable).max()),
        "max_abs_gap_difference_vs_full_precision": float(np.abs(gap - gap_full).max()),
        "table3_reading": tab3,
        "percent_grid": pct_grid,
        "curves": {"k": k.tolist(), "share": (k / n).tolist(),
                   "f1": {m: f1[m].tolist() for m in ARMS},
                   "pair_minus_sentence": gap.tolist()},
    }
    a.out.write_text(json.dumps(out, indent=1))
    s = main_summary
    print(f"wrote {a.out}")
    print(f"k {s['k_range']}: gap min {s['gap_min']:+.4f} at k={s['gap_min_k']} "
          f"({s['gap_min_k'] / n:.3f}), max {s['gap_max']:+.4f} at k={s['gap_max_k']} "
          f"({s['gap_max_k'] / n:.3f})")
    print("pair behind:", [(r['k_lo'], r['k_hi'], f"{r['share_lo']:.4f}-{r['share_hi']:.4f}",
                            f"{r['gap_min']:+.4f}") for r in s["pair_behind_runs"]])
    print("pair behind or level:", [(r['k_lo'], r['k_hi'], f"{r['share_lo']:.4f}-{r['share_hi']:.4f}")
                                    for r in s["pair_behind_or_level_runs"]])
    print("within 0.01:", [(r['k_lo'], r['k_hi'], f"{r['share_lo']:.4f}-{r['share_hi']:.4f}")
                           for r in s["within_0.01_runs"]])
    print("band 50-70:", s["band_50_70"])
    for name, t in (("stable", stable_summary), ("full", full_summary)):
        print(name, f"min {t['gap_min']:+.4f}", "behind:", [(r['k_lo'], r['k_hi'], f"{r['gap_min']:+.4f}")
                                                         for r in t["pair_behind_runs"]],
              "band:", t["band_50_70"]["gap_min_2dp"], t["band_50_70"]["gap_max_2dp"])
    print("max |gap diff| stable / full:", out["max_abs_gap_difference_vs_stable"],
          out["max_abs_gap_difference_vs_full_precision"])
    print("pair ahead:", [(r['k_lo'], r['k_hi'], f"{r['share_lo']:.4f}-{r['share_hi']:.4f}")
                          for r in s["pair_ahead_runs"]])
    for t in tab3:
        print(t)
    for key, v in claims.items():
        print(key, v)


if __name__ == "__main__":
    main()
