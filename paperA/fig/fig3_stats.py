#!/usr/bin/env python3
"""Values behind Figure 3 (fig:where): per row, n, share of positives, ensemble AUPRC and a
percentile bootstrap 95% interval per arm (candidates resampled with replacement within the row,
2,000 resamples, numpy default_rng seed 0; a resample with no positive or no negative is redrawn).

Inputs (read only): clean_benchmark() labels, results/paperA_v2/S_biodiv_{sentence,pair,triple}.npy,
the TaxoNERD counts used by make_where_figure.py, data/benchmarks/biored_bc8/test.csv and
results/paperA_v2/S_biored_{sentence,pair}.npy. Each point estimate is checked against
fig_where.json (biodiversity strata and BioRED rows) to 1e-12.

Output: paperA/fig/fig3_gain_by_taxa.json (read by user_2026-10-07/figures/paper_figures.py fig3).

Usage: python3 paperA/fig/fig3_stats.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from eval.core import clean_benchmark  # noqa: E402

RES = REPO / "results/paperA_v2"
FIG = Path(__file__).resolve().parent
B, SEED = 2000, 0


def boot(y: np.ndarray, s: dict, rng: np.random.Generator) -> dict:
    n, out = len(y), {k: [] for k in s}
    while len(out[next(iter(s))]) < B:
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        for k, v in s.items():
            out[k].append(ap(y[i], v[i]))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}


def main() -> None:
    ref = json.loads((FIG / "fig_where.json").read_text())
    rows = []
    d = clean_benchmark()
    y = d.label.to_numpy()
    nt = pd.read_csv(REPO / "results/paperA_rebuild_2026-09-28/n_taxa_per_row.csv")
    ar = pd.read_csv(RES / "arg_recognition.csv")
    assert len(nt) == len(ar) == len(d) and (nt.species1.to_numpy() == d.species1.to_numpy()).all()
    multi, both = nt.n_taxa.to_numpy() >= 3, ar.both_args_taxa.to_numpy().astype(bool)
    masks = {"le2_taxa": ~multi, "both_taxa_le2": both & ~multi, "not_a_taxon_le2": ~both & ~multi,
             "ge3_taxa": multi, "both_taxa_ge3": both & multi, "not_a_taxon_ge3": ~both & multi}
    S = {a: np.load(RES / f"S_biodiv_{a}.npy") for a in ("sentence", "pair", "triple")}
    for key, m in masks.items():
        rng = np.random.default_rng(SEED)
        sc = {a: v[m] for a, v in S.items()}
        auprc = {a: float(ap(y[m], v)) for a, v in sc.items()}
        for a in auprc:
            assert abs(auprc[a] - ref["strata"][key]["auprc"][a]) < 1e-12, (key, a)
        rows.append({"bench": "biodiversity", "key": key, "n": int(m.sum()), "positives": int(y[m].sum()),
                     "pos": float(y[m].mean()), "auprc": auprc, "ci95": boot(y[m], sc, rng)})
        print(rows[-1]["key"], rows[-1]["n"], round(rows[-1]["pos"], 3), rows[-1]["ci95"], flush=True)
    t = pd.read_csv(REPO / "data/benchmarks/biored_bc8/test.csv")
    yb, mb = t.label.to_numpy(), t.n_concepts.to_numpy() >= 3
    Sb = {a: np.load(RES / f"S_biored_{a}.npy") for a in ("sentence", "pair")}
    for key, m in (("2_concepts", ~mb), ("ge3_concepts", mb)):
        rng = np.random.default_rng(SEED)
        sc = {a: v[m] for a, v in Sb.items()}
        auprc = {a: float(ap(yb[m], v)) for a, v in sc.items()}
        for a in auprc:
            assert abs(auprc[a] - ref["biored"][a][f"auprc_{key}"]) < 1e-12, (key, a)
        rows.append({"bench": "biored", "key": key, "n": int(m.sum()), "positives": int(yb[m].sum()),
                     "pos": float(yb[m].mean()), "auprc": auprc, "ci95": boot(yb[m], sc, rng)})
        print(rows[-1]["key"], rows[-1]["n"], round(rows[-1]["pos"], 3), rows[-1]["ci95"], flush=True)
    (FIG / "fig3_gain_by_taxa.json").write_text(json.dumps(
        {"figure": "fig3_gain_by_taxa.pdf", "label": "fig:where", "script": "paperA/fig/fig3_stats.py",
         "bootstrap": {"resamples": B, "seed": SEED, "rng": "numpy default_rng",
                       "unit": "candidate, within the row", "interval": "percentile 2.5-97.5"},
         "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
