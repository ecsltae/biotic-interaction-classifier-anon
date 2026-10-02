#!/usr/bin/env python3
"""Every number in Paper A's results tables, from trained checkpoints, in one place.

Decisions fixed in advance, not chosen on the reporting set:

  * Label semantics are pair-level and order-free ("do these two interact?"), so every
    order-sensitive model (the pair and triple arms) is scored as the max over its two argument
    orders: the probability that the pair interacts in at least one direction. The
    sentence-level arm has no query and is order-free already.
  * Biodiversity operating points are block-held-out: a threshold is fitted on two of the three
    source blocks and applied to the third. BioRED operating points are pre-specified: the mean
    of the three checkpoints' own development thresholds.
  * The three arms share an encoder (BiomedBERT-base), a corpus and a recipe; only the input
    differs: passage only / pair + passage / pair + relation + passage.

Usage
  python3 scripts/paperA_tables.py --bench biodiv
  python3 scripts/paperA_tables.py --bench biored
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2
from sklearn.metrics import average_precision_score as ap
from sklearn.metrics import f1_score, precision_score, recall_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from eval_unified import model_format, score  # noqa: E402
import candidate_rules as CR  # noqa: E402

OUT = REPO / "results/paperA_v2"
GRID = np.arange(0.01, 1.00, 0.01)
ARMS_BIODIV = {"sentence": "models/sentence_baseline/xenc_s{}", "pair": "models/pair_baseline/xenc_s{}",
               "triple": "models/student_v3/xenc_s{}"}
ARMS_BIORED = {"sentence": "models/biored_verify/sentence_s{}", "pair": "models/biored_verify/pair_s{}"}


def mcnemar(a_pred, b_pred, y):
    """Continuity-corrected McNemar; b relative to a: (p, b fixes, b breaks)."""
    ca, cb = a_pred == y, b_pred == y
    fix, brk = int((~ca & cb).sum()), int((ca & ~cb).sum())
    if fix + brk == 0:
        return 1.0, fix, brk
    return float(chi2.sf((abs(fix - brk) - 1) ** 2 / (fix + brk), 1)), fix, brk


def prf(y, p):
    return {"P": float(precision_score(y, p, zero_division=0)),
            "R": float(recall_score(y, p, zero_division=0)),
            "F1": float(f1_score(y, p, zero_division=0)), "accepts": int(p.sum())}


def order_free(md, d, dev):
    """Max over the two argument orders for a query-reading model; a single pass otherwise."""
    s = score(md, d, dev)
    if model_format(md) == "sentence":
        return s
    sw = d.copy()
    sw["species1"], sw["species2"] = d.species2.values, d.species1.values
    return np.maximum(s, score(md, sw, dev))


def block_held_out(y, s, blocks):
    pred, thr = np.zeros(len(y), int), {}
    for blk in sorted(set(blocks)):
        te = blocks == blk
        t = max(GRID, key=lambda t: f1_score(y[~te], (s[~te] >= t).astype(int), zero_division=0))
        thr[blk] = float(t)
        pred[te] = (s[te] >= t).astype(int)
    return pred, thr


def per_block_pred(s, blocks, thr):
    return np.array([int(v >= thr[b]) for v, b in zip(s, blocks)])


def biodiv(dev):
    d = pd.read_csv(REPO / "data/evaluation/unified_test_set.csv")
    scan = pd.read_csv(REPO / "results/test_contamination_scan.csv")
    d = d[~(d.in_train.to_numpy() | (scan.maxj.to_numpy() > 0.5))].reset_index(drop=True)
    assert len(d) == 437 and int(d.label.sum()) == 246, "clean benchmark changed"
    y, blocks = d.label.to_numpy(), d.source.to_numpy()
    nt = pd.read_csv(REPO / "results/paperA_rebuild_2026-09-28/n_taxa_per_row.csv")
    assert (nt.species1.values == d.species1.values).all(), "n_taxa table misaligned"
    multi = nt.n_taxa.to_numpy() >= 3
    out, S, PRED = {"n": 437, "positives": 246}, {}, {}

    for arm, pat in ARMS_BIODIV.items():
        per = [order_free(REPO / pat.format(s), d, dev) for s in (1, 2, 3)]
        S[arm] = np.mean(per, axis=0)
        pred, thr = block_held_out(y, S[arm], blocks)
        PRED[arm] = pred
        aps = [ap(y, p) for p in per]
        out[arm] = {"auprc_per_seed": aps, "auprc_mean": float(np.mean(aps)),
                    "auprc_sd": float(np.std(aps, ddof=1)), "auprc_ensemble": float(ap(y, S[arm])),
                    "thresholds": thr, **prf(y, pred),
                    "auprc_le2_taxa": float(ap(y[~multi], S[arm][~multi])),
                    "auprc_ge3_taxa": float(ap(y[multi], S[arm][multi])),
                    "auprc_by_block": {b: float(ap(y[blocks == b], S[arm][blocks == b])) for b in sorted(set(blocks))}}
        np.save(OUT / f"S_biodiv_{arm}.npy", S[arm])
    out["n_le2_taxa"], out["n_ge3_taxa"] = int((~multi).sum()), int(multi.sum())
    out["n_ge3_taxa_positives"] = int(y[multi].sum())
    out["mcnemar"] = {f"{b}_vs_{a}": dict(zip(("p", "fixes", "breaks"), mcnemar(PRED[a], PRED[b], y)))
                      for a, b in (("sentence", "pair"), ("sentence", "triple"), ("pair", "triple"))}

    # operating curves
    out["curve"] = {arm: [{"tau": float(t), **prf(y, (S[arm] >= t).astype(int))}
                          for t in (0.01, 0.04, 0.10, 0.25, 0.50, 0.90)] for arm in S}

    # recall side: the discarded candidates
    m = blocks == "reject50"
    out["reject50"] = {arm: {"recovered": int(((PRED[arm] == 1) & (y == 1) & m).sum()),
                             "of_positives": int((y[m] == 1).sum()),
                             "readmitted": int(((PRED[arm] == 1) & (y == 0) & m).sum()),
                             "of_negatives": int((y[m] == 0).sum()),
                             "F1": float(f1_score(y[m], PRED[arm][m], zero_division=0))} for arm in S}

    # candidate rules on top of the triple arm
    base = PRED["triple"]
    for tag, rules in (("rules7", CR.RULES), ("rules8", CR.RULES_WITH_COLIST)):
        rej = np.array([any(fn(str(p), a, r, b) for _, fn in rules)
                        for p, a, r, b in zip(d.sentence, d.species1, d.relation, d.species2)])
        pr = base * ~rej
        p, fx, br = mcnemar(base, pr, y)
        out[tag] = {**prf(y, pr), "mcnemar_p": p, "fixed": fx, "broken": br,
                    "per_block_F1": {b: float(f1_score(y[blocks == b], pr[blocks == b])) for b in sorted(set(blocks))}}

    # query ablation of the triple arm, at the triple arm's own per-block held-out thresholds
    thr = out["triple"]["thresholds"]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(d))
    blank = [""] * len(d)
    variants = {"retrieved": d,
                "generic": d.assign(relation="interacts with"),
                "shuffled": d.assign(relation=d.relation.iloc[perm].values),
                "rel_only": d.assign(species1=blank, species2=blank),
                "empty": d.assign(species1=blank, relation=blank, species2=blank)}
    out["ablation"] = {}
    for name, v in variants.items():
        per = [order_free(REPO / ARMS_BIODIV["triple"].format(s), v, dev) for s in (1, 2, 3)]
        sv = np.mean(per, axis=0)
        out["ablation"][name] = {"auprc": float(ap(y, sv)), "auprc_sd": float(np.std([ap(y, p) for p in per], ddof=1)),
                                 **prf(y, per_block_pred(sv, blocks, thr))}
    return out


def biored(dev):
    d = pd.read_csv(REPO / "data/benchmarks/biored_verify/test.csv")
    d = d.rename(columns={"source_species": "species1", "target_species": "species2",
                          "interaction_type": "relation", "text": "sentence"})
    d["relation"] = d.relation.fillna("")
    y, multi = d.label.to_numpy(), d.n_concepts.to_numpy() >= 3
    out, S, PRED = {"n": int(len(d)), "positives": int(y.sum()),
                    "n_2_concepts": int((~multi).sum()), "n_ge3_concepts": int(multi.sum())}, {}, {}
    for arm, pat in ARMS_BIORED.items():
        mds = [REPO / pat.format(s) for s in (1, 2, 3)]
        per = [order_free(md, d, dev) for md in mds]
        S[arm] = np.mean(per, axis=0)
        thr = float(np.mean([json.loads((md / "student_config.json").read_text())["threshold_dev"] for md in mds]))
        PRED[arm] = (S[arm] >= thr).astype(int)
        aps = [ap(y, p) for p in per]
        out[arm] = {"auprc_per_seed": aps, "auprc_mean": float(np.mean(aps)), "auprc_sd": float(np.std(aps, ddof=1)),
                    "auprc_ensemble": float(ap(y, S[arm])), "threshold_prespecified": thr, **prf(y, PRED[arm]),
                    "auprc_2_concepts": float(ap(y[~multi], S[arm][~multi])),
                    "auprc_ge3_concepts": float(ap(y[multi], S[arm][multi])),
                    "auprc_2_concepts_per_seed": [float(ap(y[~multi], p[~multi])) for p in per],
                    "auprc_ge3_concepts_per_seed": [float(ap(y[multi], p[multi])) for p in per]}
        np.save(OUT / f"S_biored_{arm}.npy", S[arm])
    out["mcnemar_pair_vs_sentence"] = dict(zip(("p", "fixes", "breaks"), mcnemar(PRED["sentence"], PRED["pair"], y)))
    out["base_rate"] = float(y.mean())
    return out


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--bench", choices=("biodiv", "biored"), required=True)
    a = ap_.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    res = biodiv(dev) if a.bench == "biodiv" else biored(dev)
    (OUT / f"tables_{a.bench}.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
