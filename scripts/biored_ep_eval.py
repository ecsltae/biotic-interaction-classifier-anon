#!/usr/bin/env python3
"""BioRED entity-pair (EP) F1 -- the standard document-level metric -- for Paper A's verifiers.

EP asks, per abstract, which concept pairs BioRED relates (any relation type); it is the first
column of BioREDirect's Table 2 (Lai et al.), whose protocol the BC8 splits follow (train on
BioRED train+dev, select on the BioRED test split, report on the BioCreative VIII test set).

Two ways to reach a document-level decision:
  sentence-scope  (data/benchmarks/biored_bc8)      a pair's score is the max over the sentences
                  that co-mention it; a pair never co-mentioned in one sentence is never predicted.
  document-scope  (data/benchmarks/biored_bc8_doc)  one score per pair from the whole abstract.

Scoring is in gold-relation space: a prediction is correct when it covers a gold relation of its
abstract (concept IDs from the PubTator annotations), every gold relation not covered is a miss.
The decision threshold is the F1-optimal one on the development split, per checkpoint; nothing
is fitted on the test split.

Usage
  python3 scripts/biored_ep_eval.py --src data/raw/bioredirect
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from convert_biored_verify import read_pubtator  # noqa: E402
from eval_unified import score  # noqa: E402
from paperA_tables import order_free  # noqa: E402

OUT = REPO / "results/paperA_v2"
GRID = np.arange(0.01, 1.00, 0.01)
ARMS = {"sentence": ("biored_bc8", "models/biored_bc8/sentence_s{}"),
        "pair": ("biored_bc8", "models/biored_bc8/pair_s{}"),
        "pair_mark": ("biored_bc8", "models/biored_bc8/pair_mark_s{}"),
        "mark_canon": ("biored_bc8", "models/biored_bc8/mark_canon_s{}"),
        "doc_pair": ("biored_bc8_doc", "models/biored_bc8_doc/pair_s{}"),
        # BiomedBERT-large (lr 2e-5, 3 epochs), chosen on development data by the 2026-10-03 sweep
        "sentence_large": ("biored_bc8", "models/large/biored_sentence_s{}"),
        "pair_large": ("biored_bc8", "models/large/biored_pair_s{}"),
        # BioLinkBERT-base (lr 2e-5, 3 epochs), chosen on development data by the session-2 screen
        "sentence_linkbert": ("biored_bc8", "models/linkbert/biored_sentence_s{}"),
        "pair_linkbert": ("biored_bc8", "models/linkbert/biored_pair_s{}")}
PUBLISHED_BC8_EP = {"PubMedBERT": 74.07, "BioREx": 74.75, "BioREDirect": 75.34,
                    "GPT-4 zero-shot": 42.93, "GPT-3.5 fine-tuned": 70.18, "Llama3.2-11B fine-tuned": 68.78}


def gold_relations(docs) -> set:
    return {(d["pmid"], r) for d in docs for r in d["rels"] if len(r) == 2}


def load(bench: str, split: str) -> pd.DataFrame:
    d = pd.read_csv(REPO / f"data/benchmarks/{bench}/{split}.csv", dtype={"pmid": str, "cid1": str, "cid2": str})
    return d.rename(columns={"source_species": "species1", "target_species": "species2",
                             "interaction_type": "relation", "text": "sentence"}).assign(relation="")


def doc_scores(d: pd.DataFrame, s: np.ndarray) -> pd.DataFrame:
    """One row per (abstract, concept pair): the max score over its candidate rows."""
    k = d.assign(score=s, a=[min(x, y) for x, y in zip(d.cid1, d.cid2)],
                 b=[max(x, y) for x, y in zip(d.cid1, d.cid2)])
    return k.groupby(["pmid", "a", "b"], as_index=False).score.max()


def ep(pairs: pd.DataFrame, gold: set, thr: float) -> dict:
    pred = pairs[pairs.score >= thr]
    covered, wrong = set(), 0
    for pm, a, b in zip(pred.pmid, pred.a, pred.b):
        hit = {(pm, frozenset((x, y))) for x in a.split("|") for y in b.split("|")} & gold
        covered |= hit
        wrong += not hit
    n_pred = len(pred)
    P = (n_pred - wrong) / n_pred if n_pred else 0.0
    R = len(covered) / len(gold)
    return {"P": P, "R": R, "F1": 2 * P * R / (P + R) if P + R else 0.0, "predicted": n_pred}


def ceiling(pairs: pd.DataFrame, gold: set) -> float:
    """Recall if every pair the benchmark offers were accepted."""
    return ep(pairs.assign(score=1.0), gold, 0.5)["R"]


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--src", default="data/raw/bioredirect")
    ap_.add_argument("--arms", nargs="*", default=list(ARMS))
    a = ap_.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    src = Path(a.src)
    gold = {"dev": gold_relations(read_pubtator(src / "bioredirect_test.pubtator")),
            "test": gold_relations(read_pubtator(src / "bioredirect_bc8_test.pubtator"))}
    out = {"gold_relations": {k: len(v) for k, v in gold.items()}, "published_bc8_ep_f1": PUBLISHED_BC8_EP}
    for arm in a.arms:
        bench, pat = ARMS[arm]
        mds = [REPO / pat.format(s) for s in (1, 2, 3)]
        if not all((m / "student_config.json").exists() for m in mds):
            print(f"skip {arm}: checkpoints missing"); continue
        D = {sp: load(bench, sp) for sp in ("dev", "test")}
        per, ens = [], {}
        for md in mds:
            # document-scope rows carry their markers in the text, in first-mention order, so they
            # are scored once; swapping the query would contradict the markers
            fn = score if bench.endswith("_doc") else order_free
            pairs = {sp: doc_scores(D[sp], fn(md, D[sp], dev)) for sp in D}
            thr = float(max(GRID, key=lambda t: ep(pairs["dev"], gold["dev"], t)["F1"]))
            per.append({"threshold_dev": thr, "dev": ep(pairs["dev"], gold["dev"], thr),
                        "test": ep(pairs["test"], gold["test"], thr)})
            for sp in D:
                ens.setdefault(sp, []).append(pairs[sp].score.to_numpy())
        f1 = [p["test"]["F1"] for p in per]
        out[arm] = {"per_seed": per, "test_F1_mean": float(np.mean(f1)), "test_F1_sd": float(np.std(f1, ddof=1)),
                    "test_P_mean": float(np.mean([p["test"]["P"] for p in per])),
                    "test_R_mean": float(np.mean([p["test"]["R"] for p in per])),
                    "recall_ceiling_test": ceiling(pairs["test"], gold["test"])}
        print(f"{arm:10} EP F1 {100 * np.mean(f1):.2f} ± {100 * np.std(f1, ddof=1):.2f} "
              f"(P {100 * out[arm]['test_P_mean']:.1f}, R {100 * out[arm]['test_R_mean']:.1f}; "
              f"recall ceiling {100 * out[arm]['recall_ceiling_test']:.1f})", flush=True)
    prev = OUT / "biored_ep.json"
    if prev.exists():                                     # keep arms scored in earlier calls
        out = {**json.loads(prev.read_text()), **out}
    prev.write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
