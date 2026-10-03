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
  * BioRED follows the BioREDirect/BC8 protocol: train on BioRED train+dev, select epochs and
    thresholds on the BioRED test split, report on the BioCreative VIII test set (400 abstracts).
  * The three arms share an encoder (BiomedBERT-base), a corpus and a recipe; only the input
    differs: passage only / pair + passage / pair + relation + passage.

Usage
  python3 scripts/paperA_tables.py --bench biodiv
  python3 scripts/paperA_tables.py --bench biored
  python3 scripts/paperA_tables.py --bench biodiv --arms large   # adds *_large keys, keeps the rest
  python3 scripts/paperA_tables.py --bench biodiv --arms soft    # adds *_soft keys, keeps the rest
  python3 scripts/paperA_tables.py --bench biored --arms linkbert  # adds *_linkbert keys
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
ARMS_BIORED = {"sentence": "models/biored_bc8/sentence_s{}", "pair": "models/biored_bc8/pair_s{}",
               "pair_mark": "models/biored_bc8/pair_mark_s{}", "mark_canon": "models/biored_bc8/mark_canon_s{}"}
# BiomedBERT-large, lr 2e-5, 3 epochs: the recipe the 2026-10-03 sweep chose on development data
# (BioRED pair dev AUPRC and the biodiversity pair arm's internal dev split). --arms large scores
# these and writes every key with a "_large" suffix next to the base-encoder keys.
ARMS_BIODIV_LARGE = {"sentence": "models/large/biodiv_sentence_s{}", "pair": "models/large/biodiv_pair_s{}",
                     "triple": "models/large/biodiv_triple_s{}"}
ARMS_BIORED_LARGE = {"sentence": "models/large/biored_sentence_s{}", "pair": "models/large/biored_pair_s{}"}
# Soft-label distillation: the same three arms and recipe, trained on the teacher's P(yes)
# (scripts/soft_relabel.py, --soft-col soft_label) instead of its verdict. Biodiversity only: the
# teacher labelled the biodiversity corpus. --arms soft writes "_soft" keys.
ARMS_BIODIV_SOFT = {"sentence": "models/soft_students/sentence_s{}", "pair": "models/soft_students/pair_s{}",
                    "triple": "models/soft_students/triple_s{}"}
# BioLinkBERT-base, lr 2e-5, 3 epochs (base recipe, other encoder): screened on development data in
# session 2 of 2026-10-03 (scripts/linkbert_screen_2026-10-03.sh). --arms linkbert: "_linkbert" keys.
ARMS_BIODIV_LINKBERT = {"sentence": "models/linkbert/biodiv_sentence_s{}", "pair": "models/linkbert/biodiv_pair_s{}",
                        "triple": "models/linkbert/biodiv_triple_s{}"}
ARMS_BIORED_LINKBERT = {"sentence": "models/linkbert/biored_sentence_s{}", "pair": "models/linkbert/biored_pair_s{}"}
SUFFIXES = ("_large", "_soft", "_linkbert")
LLM_DIR = REPO / "results/paperA_v2/llm"


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


def biodiv(dev, arms=ARMS_BIODIV, sfx=""):
    d = pd.read_csv(REPO / "data/evaluation/unified_test_set.csv")
    scan = pd.read_csv(REPO / "results/test_contamination_scan.csv")
    d = d[~(d.in_train.to_numpy() | (scan.maxj.to_numpy() > 0.5))].reset_index(drop=True)
    assert len(d) == 437 and int(d.label.sum()) == 246, "clean benchmark changed"
    y, blocks = d.label.to_numpy(), d.source.to_numpy()
    nt = pd.read_csv(REPO / "results/paperA_rebuild_2026-09-28/n_taxa_per_row.csv")
    assert (nt.species1.values == d.species1.values).all(), "n_taxa table misaligned"
    multi = nt.n_taxa.to_numpy() >= 3
    out, S, PRED = {"n": 437, "positives": 246}, {}, {}
    arms = complete(arms)

    for arm, pat in arms.items():
        per = [order_free(REPO / pat.format(s), d, dev) for s in (1, 2, 3)]
        S[arm] = np.mean(per, axis=0)
        pred, thr = block_held_out(y, S[arm], blocks)
        PRED[arm] = pred
        aps = [ap(y, p) for p in per]
        out[arm + sfx] = {"auprc_per_seed": aps, "auprc_mean": float(np.mean(aps)),
                    "auprc_sd": float(np.std(aps, ddof=1)), "auprc_ensemble": float(ap(y, S[arm])),
                    "thresholds": thr, **prf(y, pred),
                    "auprc_le2_taxa": float(ap(y[~multi], S[arm][~multi])),
                    "auprc_ge3_taxa": float(ap(y[multi], S[arm][multi])),
                    "auprc_by_block": {b: float(ap(y[blocks == b], S[arm][blocks == b])) for b in sorted(set(blocks))}}
        np.save(OUT / f"S_biodiv_{arm}{sfx}.npy", S[arm])
    out["n_le2_taxa"], out["n_ge3_taxa"] = int((~multi).sum()), int(multi.sum())
    out["n_ge3_taxa_positives"] = int(y[multi].sum())
    out["mcnemar" + sfx] = {f"{b}_vs_{a}": dict(zip(("p", "fixes", "breaks"), mcnemar(PRED[a], PRED[b], y)))
                            for a, b in (("sentence", "pair"), ("sentence", "triple"), ("pair", "triple"))
                            if a in PRED and b in PRED}
    if not sfx:            # the LLM rows of a suffixed run are refreshed with --llm-only (all saved S_*.npy)
        out["llm"] = llm_rows("biodiv", y, S, multi, blocks)
    else:                  # each arm against the base arm of the same format, at its saved thresholds
        base = json.loads((OUT / "tables_biodiv.json").read_text())
        out["mcnemar_vs_base" + sfx] = {}
        for arm in PRED:
            bp = per_block_pred(np.load(OUT / f"S_biodiv_{arm}.npy"), blocks, base[arm]["thresholds"])
            assert abs(prf(y, bp)["F1"] - base[arm]["F1"]) < 1e-9, f"base {arm} predictions not reproduced"
            out["mcnemar_vs_base" + sfx][arm] = dict(zip(("p", "fixes", "breaks"), mcnemar(bp, PRED[arm], y)))

    # operating curves
    out["curve" + sfx] = {arm: [{"tau": float(t), **prf(y, (S[arm] >= t).astype(int))}
                          for t in (0.01, 0.04, 0.10, 0.25, 0.50, 0.90)] for arm in S}

    # recall side: the discarded candidates
    m = blocks == "reject50"
    out["reject50" + sfx] = {arm: {"recovered": int(((PRED[arm] == 1) & (y == 1) & m).sum()),
                             "of_positives": int((y[m] == 1).sum()),
                             "readmitted": int(((PRED[arm] == 1) & (y == 0) & m).sum()),
                             "of_negatives": int((y[m] == 0).sum()),
                             "F1": float(f1_score(y[m], PRED[arm][m], zero_division=0))} for arm in S}

    if "triple" not in PRED:             # a partial --arms large run: the triple-arm analyses wait
        return out
    # candidate rules on top of the triple arm
    base = PRED["triple"]
    for tag, rules in (("rules7", CR.RULES), ("rules8", CR.RULES_WITH_COLIST)):
        rej = np.array([any(fn(str(p), a, r, b) for _, fn in rules)
                        for p, a, r, b in zip(d.sentence, d.species1, d.relation, d.species2)])
        pr = base * ~rej
        p, fx, br = mcnemar(base, pr, y)
        out[tag + sfx] = {**prf(y, pr), "mcnemar_p": p, "fixed": fx, "broken": br,
                    "per_block_F1": {b: float(f1_score(y[blocks == b], pr[blocks == b])) for b in sorted(set(blocks))}}

    # query ablation of the triple arm, at the triple arm's own per-block held-out thresholds
    thr = out["triple" + sfx]["thresholds"]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(d))
    blank = [""] * len(d)
    variants = {"retrieved": d,
                "generic": d.assign(relation="interacts with"),
                "shuffled": d.assign(relation=d.relation.iloc[perm].values),
                "rel_only": d.assign(species1=blank, species2=blank),
                "empty": d.assign(species1=blank, relation=blank, species2=blank)}
    out["ablation" + sfx] = {}
    for name, v in variants.items():
        per = [order_free(REPO / arms["triple"].format(s), v, dev) for s in (1, 2, 3)]
        sv = np.mean(per, axis=0)
        out["ablation" + sfx][name] = {"auprc": float(ap(y, sv)), "auprc_sd": float(np.std([ap(y, p) for p in per], ddof=1)),
                                 **prf(y, per_block_pred(sv, blocks, thr))}
    return out


def complete(arms: dict) -> dict:
    """The arms whose three checkpoints all exist (the base arms always do; a partial run of the
    large arms scores what has finished and says what it skipped)."""
    ok = {a: p for a, p in arms.items() if all((REPO / p.format(s) / "student_config.json").exists() for s in (1, 2, 3))}
    for a in set(arms) - set(ok):
        print(f"skip {a}: checkpoints missing", file=sys.stderr)
    return ok


def biored(dev, arms=ARMS_BIORED, sfx=""):
    d = pd.read_csv(REPO / "data/benchmarks/biored_bc8/test.csv")
    d = d.rename(columns={"source_species": "species1", "target_species": "species2",
                          "interaction_type": "relation", "text": "sentence"})
    d["relation"] = d.relation.fillna("")
    y, multi = d.label.to_numpy(), d.n_concepts.to_numpy() >= 3
    out, S, PRED = {"n": int(len(d)), "positives": int(y.sum()),
                    "n_2_concepts": int((~multi).sum()), "n_ge3_concepts": int(multi.sum())}, {}, {}
    for arm, pat in complete(arms).items():
        mds = [REPO / pat.format(s) for s in (1, 2, 3)]
        per = [order_free(md, d, dev) for md in mds]
        S[arm] = np.mean(per, axis=0)
        thr = float(np.mean([json.loads((md / "student_config.json").read_text())["threshold_dev"] for md in mds]))
        PRED[arm] = (S[arm] >= thr).astype(int)
        aps = [ap(y, p) for p in per]
        out[arm + sfx] = {"auprc_per_seed": aps, "auprc_mean": float(np.mean(aps)), "auprc_sd": float(np.std(aps, ddof=1)),
                    "auprc_ensemble": float(ap(y, S[arm])), "threshold_prespecified": thr, **prf(y, PRED[arm]),
                    "auprc_2_concepts": float(ap(y[~multi], S[arm][~multi])),
                    "auprc_ge3_concepts": float(ap(y[multi], S[arm][multi])),
                    "auprc_2_concepts_per_seed": [float(ap(y[~multi], p[~multi])) for p in per],
                    "auprc_ge3_concepts_per_seed": [float(ap(y[multi], p[multi])) for p in per]}
        np.save(OUT / f"S_biored_{arm}{sfx}.npy", S[arm])
    if "sentence" in PRED and "pair" in PRED:
        out["mcnemar_pair_vs_sentence" + sfx] = dict(zip(("p", "fixes", "breaks"), mcnemar(PRED["sentence"], PRED["pair"], y)))
    for arm in ("pair_mark", "mark_canon"):
        if arm in PRED and "pair" in PRED:
            out[f"mcnemar_{arm}_vs_pair{sfx}"] = dict(zip(("p", "fixes", "breaks"), mcnemar(PRED["pair"], PRED[arm], y)))
    if sfx:                # each arm against the base arm of the same format, at its pre-specified threshold
        base = json.loads((OUT / "tables_biored.json").read_text())
        out["mcnemar_vs_base" + sfx] = {}
        for arm in PRED:
            bp = (np.load(OUT / f"S_biored_{arm}.npy") >= base[arm]["threshold_prespecified"]).astype(int)
            assert abs(prf(y, bp)["F1"] - base[arm]["F1"]) < 1e-9, f"base {arm} predictions not reproduced"
            out["mcnemar_vs_base" + sfx][arm] = dict(zip(("p", "fixes", "breaks"), mcnemar(bp, PRED[arm], y)))
    out["base_rate"] = float(y.mean())
    if not sfx:
        out["llm"] = llm_rows("biored", y, S, multi)
    return out


def llm_rows(bench, y, S, multi=None, blocks=None):
    """Zero-shot local-LLM rows (scripts/llm_baseline.py), with the trained arms re-scored on
    exactly the LLM's rows so every number in a comparison shares its items."""
    out = {}
    for f in sorted(LLM_DIR.glob(f"{bench}_*.csv")):
        o = pd.read_csv(f)
        idx = o.row.to_numpy() if "row" in o else np.arange(len(o))
        assert (o.label.to_numpy() == y[idx]).all(), f"{f.name}: rows misaligned"
        r = {"n": int(len(o)), "auprc": float(ap(o.label, o.p_yes)),
             "greedy": prf(o.label.to_numpy(), o.verdict.to_numpy())}
        if blocks is not None:                 # the trained arms' protocol: block-held-out thresholds
            pred, thr = block_held_out(o.label.to_numpy(), o.p_yes.to_numpy(), blocks[idx])
            r["block_held_out"] = {**prf(o.label.to_numpy(), pred), "thresholds": thr}
        if multi is not None:
            m = multi[idx]
            r["auprc_ge3"] = float(ap(o.label[m], o.p_yes[m]))
            r["auprc_le2"] = float(ap(o.label[~m], o.p_yes[~m])) if (~m).sum() and o.label[~m].nunique() > 1 else None
        r["trained_arms_same_rows"] = {arm: float(ap(y[idx], S[arm][idx])) for arm in S}
        out[f.stem] = r
    return out


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--bench", choices=("biodiv", "biored"), required=True)
    ap_.add_argument("--arms", choices=("base", "large", "soft", "linkbert"), default="base",
                     help="base: the BiomedBERT-base arms (writes the unsuffixed keys); large: the "
                          "BiomedBERT-large arms, merged into the existing JSON under *_large keys; "
                          "soft: the soft-label students (biodiversity only), under *_soft keys; "
                          "linkbert: the BioLinkBERT-base arms, under *_linkbert keys")
    ap_.add_argument("--llm-only", action="store_true",
                     help="recompute only the LLM rows, from the saved S_*.npy of an earlier full run")
    a = ap_.parse_args()
    if a.llm_only:
        res = json.loads((OUT / f"tables_{a.bench}.json").read_text())
        S = {f.stem.split("_", 2)[2]: np.load(f) for f in OUT.glob(f"S_{a.bench}_*.npy")}
        if a.bench == "biodiv":
            d = pd.read_csv(REPO / "data/evaluation/unified_test_set.csv")
            scan = pd.read_csv(REPO / "results/test_contamination_scan.csv")
            d = d[~(d.in_train.to_numpy() | (scan.maxj.to_numpy() > 0.5))].reset_index(drop=True)
            nt = pd.read_csv(REPO / "results/paperA_rebuild_2026-09-28/n_taxa_per_row.csv")
            res["llm"] = llm_rows("biodiv", d.label.to_numpy(), S, nt.n_taxa.to_numpy() >= 3, d.source.to_numpy())
        else:
            d = pd.read_csv(REPO / "data/benchmarks/biored_bc8/test.csv")
            res["llm"] = llm_rows("biored", d.label.to_numpy(), S, d.n_concepts.to_numpy() >= 3)
        (OUT / f"tables_{a.bench}.json").write_text(json.dumps(res, indent=2, default=float))
        print(json.dumps(res["llm"], indent=1, default=float))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    if a.arms == "large":
        new = biodiv(dev, ARMS_BIODIV_LARGE, "_large") if a.bench == "biodiv" else biored(dev, ARMS_BIORED_LARGE, "_large")
        res = {**json.loads((OUT / f"tables_{a.bench}.json").read_text()), **new}
    elif a.arms == "linkbert":
        new = (biodiv(dev, ARMS_BIODIV_LINKBERT, "_linkbert") if a.bench == "biodiv"
               else biored(dev, ARMS_BIORED_LINKBERT, "_linkbert"))
        res = {**json.loads((OUT / f"tables_{a.bench}.json").read_text()), **new}
    elif a.arms == "soft":
        if a.bench != "biodiv":
            ap_.error("--arms soft: the soft-label students exist for the biodiversity corpus only")
        res = {**json.loads((OUT / "tables_biodiv.json").read_text()), **biodiv(dev, ARMS_BIODIV_SOFT, "_soft")}
    else:
        res = biodiv(dev) if a.bench == "biodiv" else biored(dev)
        prev = OUT / f"tables_{a.bench}.json"   # keep the suffixed keys of earlier --arms large/soft runs
        if prev.exists():
            res = {**{k: v for k, v in json.loads(prev.read_text()).items() if k.endswith(SUFFIXES)}, **res}
    (OUT / f"tables_{a.bench}.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
