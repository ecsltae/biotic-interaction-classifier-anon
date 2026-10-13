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
  * Biodiversity labels come from src/eval/core.py's clean_benchmark() (hash-checked), for the
    label version given by --labels. Every score, the LLM rows' included, is computed against those
    labels; the label column an LLM run stored in its CSV is never used.

Order of runs for the biodiversity benchmark: the base run first (it writes the thresholds and
score vectors the suffixed runs compare against, and it drops suffixed keys scored against other
labels), then each --arms suffix, then --llm-only (trained arms on the LLMs' rows, all S_*.npy).

Usage
  python3 scripts/paperA_tables.py --bench biodiv
  python3 scripts/paperA_tables.py --bench biored
  python3 scripts/paperA_tables.py --bench biodiv --arms large   # adds *_large keys, keeps the rest
  python3 scripts/paperA_tables.py --bench biodiv --arms soft    # adds *_soft keys, keeps the rest
  python3 scripts/paperA_tables.py --bench biored --arms linkbert  # adds *_linkbert keys
  python3 scripts/paperA_tables.py --bench biodiv --labels pre_review_2026-10-06 --out-dir <dir>
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
sys.path.insert(0, str(REPO / "src"))
from eval.core import LABEL_VERSIONS, benchmark_provenance, clean_benchmark  # noqa: E402
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
# BioLinkBERT-large (lr 2e-5, 3 epochs, micro-batch 16), same screen. --arms linkbertL: "_linkbertL" keys.
ARMS_BIODIV_LINKBERTL = {a: p.replace("models/linkbert/", "models/linkbertL/") for a, p in ARMS_BIODIV_LINKBERT.items()}
ARMS_BIORED_LINKBERTL = {a: p.replace("models/linkbert/", "models/linkbertL/") for a, p in ARMS_BIORED_LINKBERT.items()}
# the three arms distilled from the newer teacher (qwen3.8:27b, same prompt and corpus): --arms qwen38
ARMS_BIODIV_QWEN38 = {"sentence": "models/teacher_qwen38/sentence_s{}", "pair": "models/teacher_qwen38/pair_s{}",
                      "triple": "models/teacher_qwen38/triple_s{}"}
SUFFIXES = ("_large", "_soft", "_linkbert", "_linkbertL", "_qwen38")
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


def n_taxa_multi(d):
    """Rows whose passage names >= 3 distinct taxa (TaxoNERD counts, label-independent)."""
    nt = pd.read_csv(REPO / "results/paperA_rebuild_2026-09-28/n_taxa_per_row.csv")
    assert (nt.species1.values == d.species1.values).all(), "n_taxa table misaligned"
    return nt.n_taxa.to_numpy() >= 3


def benchmark_summary(y, blocks, prov):
    """Counts and the accept-everything row: the label-dependent numbers that are not a model's."""
    ones = np.ones(len(y), int)
    return {"n": int(len(y)), "positives": int(y.sum()), "base_rate": float(y.mean()),
            "accept_everything": {"auprc": float(ap(y, ones)), **prf(y, ones)},
            "positives_by_block": {b: {"n": int((blocks == b).sum()), "positives": int(y[blocks == b].sum()),
                                       "rate": float(y[blocks == b].mean())} for b in sorted(set(blocks))},
            "labels": prov["labels"], "benchmark_sha256": prov["sha256"]}


def biodiv(dev, arms=ARMS_BIODIV, sfx="", labels="current", out_dir=OUT):
    d = clean_benchmark(labels)           # hash-checked; 437 rows, positives as the label version says
    prov = benchmark_provenance(labels)
    y, blocks = d.label.to_numpy(), d.source.to_numpy()
    assert len(d) == prov["n"] and int(y.sum()) == prov["positives"], "clean benchmark changed"
    multi = n_taxa_multi(d)
    out, S, PRED = {}, {}, {}
    if sfx:                # which labels this suffix was scored against; a base run drops it otherwise
        out["benchmark_sha256" + sfx] = prov["sha256"]
    else:
        out.update(benchmark_summary(y, blocks, prov))
    arms = complete(arms)
    (out_dir / "per_seed").mkdir(parents=True, exist_ok=True)

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
        np.save(out_dir / f"S_biodiv_{arm}{sfx}.npy", S[arm])
        # the three checkpoints' own scores (label-independent), for per-seed figures computed
        # elsewhere (scripts/paperA_derived_numbers.py); the ensemble above is their mean
        np.save(out_dir / "per_seed" / f"S_biodiv_{arm}{sfx}.npy", np.array(per))
    if not sfx:
        out["n_le2_taxa"], out["n_ge3_taxa"] = int((~multi).sum()), int(multi.sum())
        out["n_ge3_taxa_positives"] = int(y[multi].sum())
    out["mcnemar" + sfx] = {f"{b}_vs_{a}": dict(zip(("p", "fixes", "breaks"), mcnemar(PRED[a], PRED[b], y)))
                            for a, b in (("sentence", "pair"), ("sentence", "triple"), ("pair", "triple"))
                            if a in PRED and b in PRED}
    if not sfx:            # the LLM rows of a suffixed run are refreshed with --llm-only (all saved S_*.npy)
        out["llm"] = llm_rows("biodiv", y, S, d, multi, blocks)
    else:                  # each arm against the base arm of the same format, at its saved thresholds
        base = json.loads((out_dir / "tables_biodiv.json").read_text())
        out["mcnemar_vs_base" + sfx] = {}
        for arm in PRED:
            bp = per_block_pred(np.load(out_dir / f"S_biodiv_{arm}.npy"), blocks, base[arm]["thresholds"])
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


def biored_frame():
    d = pd.read_csv(REPO / "data/benchmarks/biored_bc8/test.csv")
    d = d.rename(columns={"source_species": "species1", "target_species": "species2",
                          "interaction_type": "relation", "text": "sentence"})
    d["relation"] = d.relation.fillna("")
    return d


def biored(dev, arms=ARMS_BIORED, sfx="", out_dir=OUT):
    d = biored_frame()
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
        np.save(out_dir / f"S_biored_{arm}{sfx}.npy", S[arm])
    if "sentence" in PRED and "pair" in PRED:
        out["mcnemar_pair_vs_sentence" + sfx] = dict(zip(("p", "fixes", "breaks"), mcnemar(PRED["sentence"], PRED["pair"], y)))
    for arm in ("pair_mark", "mark_canon"):
        if arm in PRED and "pair" in PRED:
            out[f"mcnemar_{arm}_vs_pair{sfx}"] = dict(zip(("p", "fixes", "breaks"), mcnemar(PRED["pair"], PRED[arm], y)))
    if sfx:                # each arm against the base arm of the same format, at its pre-specified threshold
        base = json.loads((out_dir / "tables_biored.json").read_text())
        out["mcnemar_vs_base" + sfx] = {}
        for arm in PRED:
            bp = (np.load(out_dir / f"S_biored_{arm}.npy") >= base[arm]["threshold_prespecified"]).astype(int)
            assert abs(prf(y, bp)["F1"] - base[arm]["F1"]) < 1e-9, f"base {arm} predictions not reproduced"
            out["mcnemar_vs_base" + sfx][arm] = dict(zip(("p", "fixes", "breaks"), mcnemar(bp, PRED[arm], y)))
    out["base_rate"] = float(y.mean())
    if not sfx:
        out["llm"] = llm_rows("biored", y, S, d, multi)
    return out


def llm_rows(bench, y, S, d, multi=None, blocks=None):
    """Zero-shot local-LLM rows (scripts/llm_baseline.py), with the trained arms re-scored on
    exactly the LLM's rows so every number in a comparison shares its items.

    Every row is scored against the benchmark labels y[idx]. The CSVs also store the label at the
    time of the LLM run; that column goes stale whenever the gold is revised, so it is never used.
    Alignment is checked on the candidate and its passage instead."""
    out, stale = {}, {}
    for f in sorted(LLM_DIR.glob(f"{bench}_*.csv")):
        o = pd.read_csv(f)
        idx = o.row.to_numpy() if "row" in o else np.arange(len(o))
        for c in ("species1", "species2", "sentence"):
            assert (o[c].astype(str).to_numpy() == d[c].astype(str).to_numpy()[idx]).all(), \
                f"{f.name}: column {c} is not row-aligned with the benchmark"
        yl = y[idx]
        if "label" in o and (o.label.to_numpy() != yl).any():
            stale[f.name] = int((o.label.to_numpy() != yl).sum())
        p_yes = o.p_yes.to_numpy()
        r = {"n": int(len(o)), "auprc": float(ap(yl, p_yes)),
             "greedy": prf(yl, o.verdict.to_numpy())}
        if blocks is not None:                 # the trained arms' protocol: block-held-out thresholds
            pred, thr = block_held_out(yl, p_yes, blocks[idx])
            r["block_held_out"] = {**prf(yl, pred), "thresholds": thr}
        if multi is not None:
            m = multi[idx]
            r["auprc_ge3"] = float(ap(yl[m], p_yes[m]))
            r["auprc_le2"] = float(ap(yl[~m], p_yes[~m])) if (~m).sum() and len(np.unique(yl[~m])) > 1 else None
        r["trained_arms_same_rows"] = {arm: float(ap(yl, S[arm][idx])) for arm in S}
        out[f.stem] = r
    if stale:
        print(f"note: {len(stale)} {bench} LLM files store a label column that differs from the benchmark "
              f"({sorted(set(stale.values()))} rows each); scored against the benchmark labels", file=sys.stderr)
    return out


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--bench", choices=("biodiv", "biored"), required=True)
    ap_.add_argument("--arms", choices=("base", "large", "soft", "linkbert", "linkbertL", "qwen38"), default="base",
                     help="base: the BiomedBERT-base arms (writes the unsuffixed keys); large: the "
                          "BiomedBERT-large arms, merged into the existing JSON under *_large keys; "
                          "soft: the soft-label students (biodiversity only), under *_soft keys; "
                          "linkbert / linkbertL: the BioLinkBERT-base / -large arms, under "
                          "*_linkbert / *_linkbertL keys")
    ap_.add_argument("--llm-only", action="store_true",
                     help="recompute only the LLM rows, from the saved S_*.npy of an earlier full run")
    ap_.add_argument("--labels", choices=sorted(LABEL_VERSIONS), default="current",
                     help="label version of the biodiversity benchmark (src/eval/core.py LABEL_VERSIONS)")
    ap_.add_argument("--out-dir", default=str(OUT),
                     help="where tables_<bench>.json and the S_*.npy score vectors are read and written "
                          f"(default {OUT.relative_to(REPO)}); a regression run points it elsewhere")
    a = ap_.parse_args()
    if a.bench == "biored" and a.labels != "current":
        ap_.error("--labels applies to the biodiversity benchmark only")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = out_dir / f"tables_{a.bench}.json"
    sha = benchmark_provenance(a.labels)["sha256"]

    def same_labels(res, sfx=""):
        """A suffixed run or an LLM refresh is merged into the base run's file: same labels only."""
        if a.bench == "biodiv" and res.get("benchmark_sha256" + sfx) != sha:
            sys.exit(f"{tables} was not scored against the {a.labels} labels (sha {sha[:12]}); "
                     f"run the base arms first: --bench biodiv --labels {a.labels} --out-dir {out_dir}")

    if a.llm_only:
        res = json.loads(tables.read_text())
        S = {f.stem.split("_", 2)[2]: np.load(f) for f in sorted(out_dir.glob(f"S_{a.bench}_*.npy"))}
        if a.bench == "biodiv":
            same_labels(res)
            d = clean_benchmark(a.labels)
            res["llm"] = llm_rows("biodiv", d.label.to_numpy(), S, d, n_taxa_multi(d), d.source.to_numpy())
        else:
            d = biored_frame()
            res["llm"] = llm_rows("biored", d.label.to_numpy(), S, d, d.n_concepts.to_numpy() >= 3)
        tables.write_text(json.dumps(res, indent=2, default=float))
        print(json.dumps(res["llm"], indent=1, default=float))
        return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    if a.arms != "base":
        res = json.loads(tables.read_text())
        if a.bench == "biodiv":
            same_labels(res)
    if a.arms == "large":
        new = (biodiv(dev, ARMS_BIODIV_LARGE, "_large", a.labels, out_dir) if a.bench == "biodiv"
               else biored(dev, ARMS_BIORED_LARGE, "_large", out_dir))
        res = {**res, **new}
    elif a.arms in ("linkbert", "linkbertL"):
        sfx = "_" + a.arms
        new = (biodiv(dev, ARMS_BIODIV_LINKBERTL if a.arms == "linkbertL" else ARMS_BIODIV_LINKBERT, sfx, a.labels, out_dir)
               if a.bench == "biodiv" else
               biored(dev, ARMS_BIORED_LINKBERTL if a.arms == "linkbertL" else ARMS_BIORED_LINKBERT, sfx, out_dir))
        res = {**res, **new}
    elif a.arms in ("soft", "qwen38"):
        if a.bench != "biodiv":
            ap_.error(f"--arms {a.arms}: these arms exist for the biodiversity corpus only")
        arms = ARMS_BIODIV_SOFT if a.arms == "soft" else ARMS_BIODIV_QWEN38
        res = {**res, **biodiv(dev, arms, "_" + a.arms, a.labels, out_dir)}
    else:
        res = biodiv(dev, labels=a.labels, out_dir=out_dir) if a.bench == "biodiv" else biored(dev, out_dir=out_dir)
        if tables.exists():   # keep the suffixed keys of earlier --arms runs ...
            prev = json.loads(tables.read_text())
            keep, dropped = {}, set()
            for k, v in prev.items():
                s = next((x for x in SUFFIXES if k.endswith(x)), None)
                if s is None:
                    continue
                # ... but on the biodiversity benchmark only those scored against these labels
                if a.bench == "biored" or prev.get("benchmark_sha256" + s) == res["benchmark_sha256"]:
                    keep[k] = v
                else:
                    dropped.add(s)
            if dropped:
                print(f"dropped the {', '.join(sorted(dropped))} keys of the previous {tables.name}: not scored "
                      f"against these labels; rerun --arms {' / '.join(x.lstrip('_') for x in sorted(dropped))}",
                      file=sys.stderr)
            res = {**keep, **res}
    tables.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
