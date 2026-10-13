#!/usr/bin/env python3
"""Shares added to two appendix tables of Paper A (tab:curve, tab:where).

  * tab:curve: at each printed threshold, the share of the 437 candidates each three-checkpoint
    ensemble accepts (score >= tau); P, R and F1 are recomputed and must round to the printed values.
  * tab:where: the share of positives in each row (TaxoNERD strata and source blocks).

Inputs (read only): clean_benchmark() labels, results/paperA_v2/S_biodiv_{sentence,pair,triple}.npy,
results/paperA_rebuild_2026-09-28/n_taxa_per_row.csv, results/paperA_v2/arg_recognition.csv.
Output: results/paperA_v2/table_shares.json

Usage: python3 scripts/paperA_table_shares.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from eval.core import clean_benchmark  # noqa: E402

RES = REPO / "results/paperA_v2"
TAUS = (0.01, 0.04, 0.10, 0.25, 0.50, 0.90)
PRINTED = {  # tab:curve, P R F1 per arm per tau
    "sentence": [(.709, .920, .801), (.782, .817, .799), (.819, .737, .776), (.861, .693, .768),
                 (.886, .526, .660), (.930, .159, .272)],
    "pair": [(.761, .976, .855), (.815, .948, .877), (.854, .908, .880), (.911, .853, .881),
             (.940, .689, .795), (.981, .422, .591)],
    "triple": [(.740, .940, .828), (.811, .924, .864), (.845, .912, .877), (.903, .849, .875),
               (.942, .777, .852), (.982, .637, .773)],
}


def main() -> None:
    d = clean_benchmark()
    y = d.label.to_numpy()
    out = {"n": int(len(y)), "positives": int(y.sum()), "curve": {}, "where_pos": {}}
    for arm in ("sentence", "pair", "triple"):
        s = np.load(RES / f"S_biodiv_{arm}.npy")
        rows = []
        for tau, (P0, R0, F0) in zip(TAUS, PRINTED[arm]):
            a = s >= tau
            tp = int((a & (y == 1)).sum())
            P, R = tp / a.sum(), tp / y.sum()
            F = 2 * P * R / (P + R)
            assert (round(P, 3), round(R, 3), round(F, 3)) == (P0, R0, F0), (arm, tau, P, R, F)
            rows.append({"tau": tau, "accepted": int(a.sum()), "share_accepted": float(a.mean()),
                         "P": P, "R": R, "F1": F})
        out["curve"][arm] = rows
    nt = pd.read_csv(REPO / "results/paperA_rebuild_2026-09-28/n_taxa_per_row.csv")
    ar = pd.read_csv(RES / "arg_recognition.csv")
    assert len(nt) == len(ar) == len(d) and (nt.species1.to_numpy() == d.species1.to_numpy()).all()
    multi, both = nt.n_taxa.to_numpy() >= 3, ar.both_args_taxa.to_numpy().astype(bool)
    src = d.source.to_numpy()
    masks = {"le2_taxa": ~multi, "ge3_taxa": multi, "both_taxa_le2": both & ~multi,
             "both_taxa_ge3": both & multi, "not_a_taxon_le2": ~both & ~multi,
             "not_a_taxon_ge3": ~both & multi, "test299": src == "test299",
             "biotx100": src == "biotx100", "reject50": src == "reject50"}
    for k, m in masks.items():
        out["where_pos"][k] = {"n": int(m.sum()), "positives": int(y[m].sum()), "share": float(y[m].mean())}
    (RES / "table_shares.json").write_text(json.dumps(out, indent=1))
    for arm, rows in out["curve"].items():
        print(arm, [f"{r['share_accepted']:.3f}" for r in rows])
    for k, v in out["where_pos"].items():
        print(k, v["n"], f"{v['share']:.3f}")


if __name__ == "__main__":
    main()
