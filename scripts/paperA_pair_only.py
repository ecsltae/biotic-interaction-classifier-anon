#!/usr/bin/env python3
"""Pair-only controls for Paper A: how much of the pair-conditioned gain could come from the two
argument names alone (which taxa tend to interact), rather than from reading the passage?

The opposite partial input to the sentence-level arm is the candidate pair with no passage:
  pair_only    BiomedBERT-base on "s1 [SEP] s2", one segment        (biodiversity and BioRED)
  triple_only  BiomedBERT-base on "s1 [SEP] relation [SEP] s2"      (biodiversity)
  nopassage    zero-shot Qwen3 asked the pair prompt minus every reference to the passage
               (scripts/llm_baseline.py --forms nopassage)
Encoder, corpus, seeds and recipe are those of the paper's pair arms (models/pair_baseline/xenc_s*,
models/biored_bc8/pair_s*); only --input-format differs.

Subcommands
  score   order-free scores of one checkpoint, computed once and saved (label-independent):
          biodiversity, the 437 clean rows; BioRED, the BC8 test set and the development split
          (BioRED's own test split), on which the entity-pair (EP) threshold is fitted.
  report  every number from saved scores -> results/paperA_v2/pair_only.json. Arms, seeds and
          LLM files that do not exist yet are reported as missing and skipped.

Protocol, as in scripts/paperA_tables.py (nothing is fitted on the rows it is reported on):
  * labels: src/eval/core.py clean_benchmark() (hash-checked); an LLM CSV's label column is never read
  * every query-reading model is scored order-free (max over the two argument orders)
  * biodiversity F1 at block-held-out thresholds (fitted on the other two source blocks); BioRED F1
    at the mean of the checkpoints' development thresholds; BioRED EP F1 at each checkpoint's
    F1-optimal development threshold (scripts/biored_ep_eval.py), seed mean
  * subsets (seen / unseen, Biotx100 halves) keep each model's operating point on the full set;
    no threshold is refitted on a subset
  * paired bootstrap of AUPRC differences: candidates resampled with replacement within the subset,
    2,000 resamples, numpy default_rng seed 0, a single-class resample redrawn, percentile 95%
    interval (the convention of the paper's Figure 3 intervals); every arm sees the same resamples

Seen vs unseen. A benchmark pair is seen when its order-free normalised pair (lowercase, whitespace
collapsed) is among the training corpus's (source_species, target_species) pairs, and a taxon is
seen when it is a source or target species of some training row. Benchmark arguments may list
alternative surface forms joined by "|" ("Gobio gobio|gudgeon"); any alternative counts (the
training corpus has none). BioRED: the training candidates (train.csv), once by normalised surface
strings and once by concept identifiers (the cid1/cid2 keys, compared whole: a variant identifier
itself contains "|").

Usage
  python3 scripts/paperA_pair_only.py score --arm biodiv_pair_only --seed 1 --threads 8
  python3 scripts/paperA_pair_only.py report
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
from eval.core import benchmark_provenance, clean_benchmark  # noqa: E402
from paperA_tables import biored_frame, block_held_out, mcnemar, n_taxa_multi, order_free, prf  # noqa: E402

RES = REPO / "results/paperA_v2"
SCORES = RES / "pair_only_scores"
MODELS = REPO / "models/pair_only"
LLM_DIR = RES / "llm"
OUT_JSON = RES / "pair_only.json"
BIORED = REPO / "data/benchmarks/biored_bc8"
CORPUS = REPO / "data/training/distill/v3_combined_train.csv"
# BioREDirect's PubTator files (dev = BioRED test split, test = BioCreative VIII test set):
# $BIORED_PUBTATOR_DIR, else the first of these that exists
PUBTATOR = Path(os.environ.get("BIORED_PUBTATOR_DIR") or next(
    (p for p in (REPO / "data/raw/bioredirect", Path.home() / "biored_baseline/bioredirect") if p.exists()),
    REPO / "data/raw/bioredirect"))

# trained arms: name -> (benchmark, input format); checkpoints at models/pair_only/<name>_s<seed>
ARMS = {"biodiv_pair_only": ("biodiv", "pair_only"), "biodiv_triple_only": ("biodiv", "triple_only"),
        "biored_pair_only": ("biored", "pair_only")}
REF_BIODIV = ("sentence", "pair", "triple")      # the paper's arms (results/paperA_v2/S_biodiv_*.npy)
REF_BIORED = ("sentence", "pair")
LLM_BIODIV = ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b-q4_K_M", "qwen3-8b", "qwen3-14b",
              "qwen3-30b-a3b-q4_K_M", "qwen3-32b")
LLM_BIORED = ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b-q4_K_M", "qwen3-8b", "qwen3-30b-a3b-q4_K_M")
BOOT = (("pair", "pair_only"), ("pair", "sentence"))   # AUPRC differences with a paired bootstrap
B, SEED = 2000, 0
LOOKUP = "train_pair_lookup"


def rel(p) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else p.name


def save_npy(path: Path, a: np.ndarray) -> None:
    """Write atomically, so a killed run never leaves a truncated file that looks finished."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, a)
    os.replace(tmp, path)


# ── score ─────────────────────────────────────────────────────────────────────────────────────

def biored_split(split: str) -> pd.DataFrame:
    """A BioRED split as scripts/biored_ep_eval.py loads it (string ids, empty relation)."""
    from biored_ep_eval import load
    return load("biored_bc8", split)


def cmd_score(a) -> None:
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    scores_dir = Path(a.scores_dir)
    if a.model:                                   # an explicit checkpoint (smoke tests)
        if not a.bench:
            sys.exit("--model needs --bench")
        md, bench, name = Path(a.model), a.bench, a.name or Path(a.model).name
    else:
        if not (a.arm and a.seed):
            sys.exit("give --arm and --seed (or --model, --bench and --name)")
        md, bench, name = MODELS / f"{a.arm}_s{a.seed}", ARMS[a.arm][0], f"{a.arm}_s{a.seed}"
    if a.limit and scores_dir.resolve() == SCORES.resolve():
        sys.exit("--limit scores part of a test set; give a scratch --scores-dir")
    if not (md / "student_config.json").exists():
        sys.exit(f"no finished checkpoint at {md}")
    scores_dir.mkdir(parents=True, exist_ok=True)
    frames = ({"": clean_benchmark()} if bench == "biodiv"
              else {"_test": biored_split("test"), "_dev": biored_split("dev")})
    side = scores_dir / f"{name}.json"
    meta = json.loads(side.read_text()) if side.exists() else {}
    meta.update({"model": rel(md), "input_format": json.loads((md / "student_config.json").read_text())
                 .get("input_format"), "bench": bench, "order_free": True})
    for sfx, f in frames.items():
        out = scores_dir / f"{name}{sfx}.npy"
        if out.exists():                          # each test set is scored once
            print(f"scored already: {out.name}", flush=True)
            continue
        if a.limit:
            f = f.head(a.limit)
        t0 = time.time()
        s = order_free(md, f, dev)
        save_npy(out, s)
        meta[sfx.lstrip("_") or "biodiv"] = {"file": out.name, "rows": int(len(f)), "seconds": round(time.time() - t0, 1),
                                             "device": dev.type, "threads": torch.get_num_threads(),
                                             "scored_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        print(f"scored {name}{sfx}: {len(f)} rows in {time.time() - t0:.0f}s -> {out}", flush=True)
    side.write_text(json.dumps(meta, indent=2))


# ── shared helpers ────────────────────────────────────────────────────────────────────────────

def norm(s) -> str:
    """Lowercase, whitespace collapsed."""
    return " ".join(str(s).lower().split())


def alternatives(s) -> set:
    return {norm(x) for x in str(s).split("|") if norm(x)}


class FastAP:
    """sklearn's average_precision_score for one score vector under resampling multiplicities.

    The scores are sorted once; a bootstrap resample is a vector of multiplicities w (np.bincount
    of the resampled indices), and AUPRC is summed over the distinct-score groups. Groups that the
    resample leaves out add nothing (their recall step is zero), so the value equals sklearn's on
    the resampled rows (checked in check_fast_ap)."""

    def __init__(self, y: np.ndarray, s: np.ndarray):
        self.o = np.argsort(-np.asarray(s, float), kind="mergesort")
        self.y = np.asarray(y, float)[self.o]
        ss = np.asarray(s, float)[self.o]
        self.last = np.r_[np.nonzero(np.diff(ss))[0], len(ss) - 1]

    def __call__(self, w: np.ndarray) -> float:
        w = np.asarray(w, float)[self.o]
        tp = np.cumsum(w * self.y)[self.last]
        fp = np.cumsum(w * (1 - self.y))[self.last]
        prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        return float(np.sum(np.diff(np.r_[0.0, tp / tp[-1]]) * prec))


def check_fast_ap(y, s, k=5) -> None:
    rng = np.random.default_rng(123)
    f = FastAP(y, s)
    assert abs(f(np.ones(len(y))) - ap(y, s)) < 1e-12
    for _ in range(k):
        i = rng.integers(0, len(y), len(y))
        assert abs(f(np.bincount(i, minlength=len(y))) - ap(y[i], s[i])) < 1e-12, "FastAP != sklearn"


def boot_deltas(y: np.ndarray, S: dict, comps) -> dict:
    """Percentile 95% intervals of AUPRC differences on shared resamples (paper convention)."""
    comps = [(a, b) for a, b in comps if a in S and b in S]
    if not comps or len(np.unique(y)) < 2:
        return {}
    arms = sorted({x for c in comps for x in c})
    fast = {k: FastAP(y, S[k]) for k in arms}
    rng, n, vals = np.random.default_rng(SEED), len(y), {k: [] for k in arms}
    while len(vals[arms[0]]) < B:
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        w = np.bincount(i, minlength=n)
        for k in arms:
            vals[k].append(fast[k](w))
    out = {}
    for a, b in comps:
        d = np.array(vals[a]) - np.array(vals[b])
        out[f"{a}_minus_{b}"] = {"delta": float(ap(y, S[a]) - ap(y, S[b])),
                                 "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                 "share_resamples_gt_0": float((d > 0).mean())}
    return out


def two_class(y) -> bool:
    return len(y) > 0 and 0 < int(np.sum(y)) < len(y)


def subset_rows(y, masks: dict, S: dict, PRED: dict, llm: dict | None = None, boot: bool = True) -> dict:
    """Counts, base rate, AUPRC and F1 (each model's full-set operating point) per subset."""
    out = {}
    for name, m in masks.items():
        m = np.asarray(m, bool)
        ym = y[m]
        r = {"n": int(m.sum()), "positives": int(ym.sum()), "pos_rate": float(ym.mean()) if m.any() else None}
        if not m.any():
            out[name] = r
            continue
        r["arms"] = {k: {"auprc": float(ap(ym, S[k][m])) if two_class(ym) else None, **prf(ym, PRED[k][m])}
                     for k in S}
        if llm:
            r["llm_nopassage"] = {}
            for k, v in llm.items():
                mm = m[v["idx"]]
                yl = v["y"][mm]
                if not mm.any():
                    r["llm_nopassage"][k] = {"n": 0}
                    continue
                r["llm_nopassage"][k] = {"n": int(mm.sum()), "auprc": float(ap(yl, v["p"][mm])) if two_class(yl) else None,
                                         **prf(yl, v["pred"][mm])}
                if "p_pair" in v:
                    r["llm_nopassage"][k]["auprc_pair_prompt"] = float(ap(yl, v["p_pair"][mm])) if two_class(yl) else None
        if boot:
            r["bootstrap"] = boot_deltas(ym, {k: S[k][m] for k in S}, BOOT)
        out[name] = r
    return out


def seen_index(src, tgt, labels):
    """Training taxa and order-free training pairs (with their labels), normalised."""
    taxa, pairs = set(), {}
    for a, b, l in zip(src, tgt, labels):
        na, nb = norm(a), norm(b)
        taxa.update((na, nb))
        pairs.setdefault(tuple(sorted((na, nb))), []).append(int(l))
    return taxa, pairs


def seen_flags(s1, s2, taxa, pairs, split_alts=True) -> pd.DataFrame:
    rows = []
    for a, b in zip(s1, s2):
        A = alternatives(a) if split_alts else {norm(a)}
        Bs = alternatives(b) if split_alts else {norm(b)}
        hit = {tuple(sorted((x, z))) for x in A for z in Bs} & pairs.keys()
        lab = [l for k in hit for l in pairs[k]]
        rows.append({"pair_seen": bool(hit), "t1": bool(A & taxa), "t2": bool(Bs & taxa),
                     "train_rows": len(lab), "train_pos": int(sum(lab))})
    return pd.DataFrame(rows)


def seen_masks(f: pd.DataFrame) -> dict:
    ps, t1, t2 = f.pair_seen.to_numpy(), f.t1.to_numpy(), f.t2.to_numpy()
    anyt = t1 | t2
    m = {"all": np.ones(len(f), bool), "pair_seen": ps, "pair_unseen": ~ps,
         "any_taxon_seen": anyt, "neither_taxon_seen": ~anyt,
         "pair_unseen_both_taxa_seen": ~ps & t1 & t2, "pair_unseen_one_taxon_seen": ~ps & (t1 ^ t2),
         "pair_unseen_any_taxon_seen": ~ps & anyt,
         "pair_seen_positive_in_train": ps & (f.train_pos.to_numpy() > 0),
         "pair_seen_negative_only_in_train": ps & (f.train_pos.to_numpy() == 0)}
    assert (m["pair_seen"].astype(int) + m["pair_unseen_both_taxa_seen"] + m["pair_unseen_one_taxon_seen"]
            + m["neither_taxon_seen"] == 1).all(), "seen partition is not a partition"
    return m


def lookup_scores(f: pd.DataFrame, base_rate: float) -> np.ndarray:
    """No-model diagnostic: the share of a pair's training rows labelled positive (the base rate
    for an unseen pair). What memorising the training labels of the pair alone would give."""
    r, p = f.train_rows.to_numpy(), f.train_pos.to_numpy()
    return np.where(r > 0, p / np.maximum(r, 1), base_rate)


def per_seed(scores_dir: Path, stem: str, n: int, sfx: str = "") -> dict:
    out = {}
    for s in (1, 2, 3):
        f = scores_dir / f"{stem}_s{s}{sfx}.npy"
        if f.exists():
            a = np.load(f)
            if len(a) != n:
                raise ValueError(f"{f}: {len(a)} scores, expected {n} (a smoke-test file in the real directory?)")
            out[s] = a
    return out


def arm_stats(y, per: dict, blocks=None, thr=None):
    seeds = sorted(per)
    P = np.array([per[s] for s in seeds])
    S = np.mean(P, axis=0)
    aps = [float(ap(y, p)) for p in P]
    r = {"seeds": seeds, "complete": seeds == [1, 2, 3], "auprc_per_seed": aps, "auprc_mean": float(np.mean(aps)),
         "auprc_sd": float(np.std(aps, ddof=1)) if len(aps) > 1 else None, "auprc_ensemble": float(ap(y, S))}
    if blocks is not None:
        pred, t = block_held_out(y, S, blocks)
        r["thresholds"] = t
    else:
        pred = (S >= thr).astype(int)
        r["threshold_prespecified"] = float(thr)
    r.update(prf(y, pred))
    return S, pred, r


def mc(PRED, y, pairs) -> dict:
    """b_vs_a: fixes = rows b gets right and a wrong; breaks = the reverse (paperA_tables.mcnemar)."""
    return {f"{b}_vs_{a}": dict(zip(("p", "fixes", "breaks"), mcnemar(PRED[a], PRED[b], y)))
            for a, b in pairs if a in PRED and b in PRED}


def llm_file_status(f: Path, n: int):
    if f.exists():
        k = len(pd.read_csv(f))
        return None if k == n else {"status": "incomplete", "rows": k, "of": n}
    part = f.with_name(f.name + ".part")
    if part.exists():
        return {"status": "running or interrupted", "rows_checkpointed": len(pd.read_csv(part)), "of": n}
    return {"status": "missing"}


# ── biodiversity ──────────────────────────────────────────────────────────────────────────────

def biotx100_types(d: pd.DataFrame) -> np.ndarray:
    """'globi' / 'random' per Biotx100 row ('' elsewhere), joined to the source file exactly as
    scripts/paperA_derived_numbers.py biotx100_halves does."""
    from paperA_derived_numbers import norm as dnorm
    src = pd.read_csv(REPO / "data/evaluation/biotx_retrieval_eval_100.csv", sep=";", encoding="utf-8-sig")
    kind = {}
    for _, r in src.iterrows():
        kind.setdefault((dnorm(r.sentence), dnorm(r.species1_form), dnorm(r.species2_form)), set()).add(r.type)
    out = np.array([""] * len(d), dtype=object)
    for i, (b, s, x, z) in enumerate(zip(d.source, d.sentence, d.species1, d.species2)):
        if b == "biotx100":
            k = kind.get((dnorm(s), dnorm(x), dnorm(z)))
            assert k is not None and len(k) == 1, f"Biotx100 row {i} not matched to one source type"
            out[i] = next(iter(k))
    return out


def biodiv(a) -> dict:
    d = clean_benchmark()
    prov = benchmark_provenance()
    y, blocks = d.label.to_numpy(), d.source.to_numpy()
    multi = n_taxa_multi(d)
    tables = json.loads((RES / "tables_biodiv.json").read_text())
    if tables.get("benchmark_sha256") != prov["sha256"]:
        sys.exit("tables_biodiv.json was not scored against the current labels; run paperA_tables.py first")
    out = {"benchmark": prov, "status": {}}
    S, PRED, PER = {}, {}, {}

    # the paper's three arms, from their saved scores, checked against tables_biodiv.json
    for arm in REF_BIODIV:
        per = np.load(RES / "per_seed" / f"S_biodiv_{arm}.npy")
        assert np.array_equal(np.mean(per, axis=0), np.load(RES / f"S_biodiv_{arm}.npy")), arm
        S[arm], PRED[arm], r = arm_stats(y, dict(zip((1, 2, 3), per)), blocks)
        t = tables[arm]
        assert r["thresholds"] == t["thresholds"] and abs(r["F1"] - t["F1"]) < 1e-12 \
            and np.allclose(r["auprc_per_seed"], t["auprc_per_seed"], rtol=0, atol=1e-12), f"{arm} differs from tables_biodiv.json"
        out[arm] = r
        PER[arm] = per

    # the new arms
    for arm, stem in (("pair_only", "biodiv_pair_only"), ("triple_only", "biodiv_triple_only")):
        per = per_seed(Path(a.scores_dir), stem, len(d))
        if not per:
            out["status"][arm] = "missing: no scored checkpoint yet"
            continue
        S[arm], PRED[arm], r = arm_stats(y, per, blocks)
        r["auprc_le2_taxa"] = float(ap(y[~multi], S[arm][~multi]))
        r["auprc_ge3_taxa"] = float(ap(y[multi], S[arm][multi]))
        r["auprc_by_block"] = {b: float(ap(y[blocks == b], S[arm][blocks == b])) for b in sorted(set(blocks))}
        r["checkpoints"] = [rel(Path(a.models_root) / f"{stem}_s{s}") for s in r["seeds"]]
        out[arm] = r
        out["status"][arm] = "complete" if r["complete"] else f"partial: seeds {r['seeds']}"
    out["mcnemar"] = mc(PRED, y, (("pair", "pair_only"), ("sentence", "pair_only"), ("pair", "triple_only"),
                                  ("sentence", "triple_only"), ("triple", "triple_only"), ("pair_only", "triple_only")))
    out["mcnemar_convention"] = "b_vs_a: fixes = rows b gets right and a gets wrong; breaks = the reverse"

    # seen vs unseen in the training corpus
    tr = pd.read_csv(CORPUS)
    taxa, pairs = seen_index(tr.source_species, tr.target_species, tr.label)
    f = seen_flags(d.species1, d.species2, taxa, pairs)
    strict = seen_flags(d.species1, d.species2, taxa, pairs, split_alts=False)
    S[LOOKUP] = lookup_scores(f, float(tr.label.mean()))
    PRED[LOOKUP], thr_lookup = block_held_out(y, S[LOOKUP], blocks)
    # the pair-grouped development split of each seed (train_student.py) is not trained on:
    # how many seen benchmark pairs occur only there
    pk = pd.Series([tuple(sorted([str(x).lower(), str(z).lower()]))
                    for x, z in zip(tr.source_species, tr.target_species)], dtype=object)
    upk = pk.unique()                               # as train_student.py: df._pk.unique()
    dev_only = {}
    for s in (1, 2, 3):
        rng = np.random.RandomState(s)
        pp = upk.copy()
        rng.shuffle(pp)
        devp = set(pp[:int(len(pp) * 0.1)])
        _, pairs_tr = seen_index(*zip(*[(x, z, l) for x, z, l, k in
                                        zip(tr.source_species, tr.target_species, tr.label, pk) if k not in devp]))
        ft = seen_flags(d.species1, d.species2, set(), pairs_tr)
        dev_only[f"seed{s}"] = int((f.pair_seen & ~ft.pair_seen).sum())
    masks = seen_masks(f)
    llm = llm_biodiv(a, d, y, blocks)
    out["seen_unseen"] = {
        "definition": "pair seen: the order-free normalised pair (lowercase, whitespace collapsed; any '|' "
                      "alternative of either argument) is a (source_species, target_species) pair of the "
                      "48,338-row training corpus; taxon seen: a source or target species of some training row",
        "n_training_pairs": len(pairs), "n_training_taxa": len(taxa),
        "pair_seen_without_splitting_alternatives": int(strict.pair_seen.sum()),
        "seen_pairs_only_in_the_dev_split_of": dev_only,
        "lookup": {"name": LOOKUP, "what": "no model: share of the pair's training rows labelled positive, "
                   "corpus base rate for an unseen pair", "thresholds": thr_lookup},
        "subsets": subset_rows(y, masks, S, PRED, {k: v for k, v in llm.items() if "p" in v}),
    }

    # Biotx100: GloBI-matched against random halves
    from paperA_derived_numbers import biotx100_halves
    types = biotx100_types(d)
    ref = biotx100_halves(d, y)
    hm = {t: types == t for t in ("globi", "random")}
    for t, m in hm.items():
        assert ref[t] == {"n": int(m.sum()), "positives": int(y[m].sum())}, f"biotx100 {t} half differs"
    halves = subset_rows(y, hm, S, PRED, {k: v for k, v in llm.items() if "p" in v})
    for t, m in hm.items():
        halves[t]["pair_seen"] = int((m & masks["pair_seen"]).sum())
        halves[t]["any_taxon_seen"] = int((m & masks["any_taxon_seen"]).sum())
    out["biotx100_halves"] = halves

    out["llm_nopassage"] = {k: {x: y_ for x, y_ in v.items() if x not in ("idx", "y", "p", "pred", "p_pair")}
                            for k, v in llm.items()}
    out["llm_note"] = ("zero-shot, the pair prompt minus the passage; one argument order like the pair "
                       "prompt; scored against clean_benchmark labels; F1 at block-held-out thresholds")
    return out


def llm_biodiv(a, d, y, blocks) -> dict:
    out = {}
    names = list(LLM_BIODIV) + sorted({f.name[len("biodiv_"):-len("_nopassage.csv")]
                                       for f in Path(a.llm_dir).glob("biodiv_*_nopassage.csv")} - set(LLM_BIODIV))
    for m in names:
        f = Path(a.llm_dir) / f"biodiv_{m}_nopassage.csv"
        st = llm_file_status(f, len(d))
        if st:
            out[m] = st
            continue
        o = pd.read_csv(f)
        for c in ("species1", "species2", "sentence"):
            assert (o[c].astype(str).to_numpy() == d[c].astype(str).to_numpy()).all(), f"{f.name}: {c} misaligned"
        p, v = o.p_yes.to_numpy(), o.verdict.to_numpy().astype(int)
        pred, thr = block_held_out(y, p, blocks)
        r = {"status": "complete", "n": int(len(o)), "auprc": float(ap(y, p)),
             "block_held_out": {**prf(y, pred), "thresholds": thr}, "greedy": prf(y, v),
             "idx": np.arange(len(d)), "y": y, "p": p, "pred": pred}
        for form in ("sentence", "pair", "triple"):
            g = Path(a.llm_dir) / f"biodiv_{m}_{form}.csv"
            if g.exists():
                go = pd.read_csv(g)
                if len(go) == len(d) and (go.species1.astype(str).to_numpy() == d.species1.astype(str).to_numpy()).all():
                    r[f"auprc_{form}_prompt"] = float(ap(y, go.p_yes.to_numpy()))
                    if form == "pair":
                        r["p_pair"] = go.p_yes.to_numpy()
        if "auprc_pair_prompt" in r:
            r["pair_prompt_minus_nopassage"] = r["auprc_pair_prompt"] - r["auprc"]
        out[m] = r
    return out


# ── BioRED ────────────────────────────────────────────────────────────────────────────────────

def biored(a) -> dict:
    d = biored_frame()
    y, multi = d.label.to_numpy(), d.n_concepts.to_numpy() >= 3
    tables = json.loads((RES / "tables_biored.json").read_text())
    out = {"n": int(len(d)), "positives": int(y.sum()), "base_rate": float(y.mean()), "status": {}}
    S, PRED = {}, {}
    for arm in REF_BIORED:
        S[arm] = np.load(RES / f"S_biored_{arm}.npy")
        thr = tables[arm]["threshold_prespecified"]
        PRED[arm] = (S[arm] >= thr).astype(int)
        r = {"auprc_mean": tables[arm]["auprc_mean"], "auprc_sd": tables[arm]["auprc_sd"],
             "auprc_ensemble": float(ap(y, S[arm])), "threshold_prespecified": thr, **prf(y, PRED[arm])}
        assert abs(r["F1"] - tables[arm]["F1"]) < 1e-12 and abs(r["auprc_ensemble"] - tables[arm]["auprc_ensemble"]) < 1e-12, \
            f"{arm} differs from tables_biored.json"
        out[arm] = r

    stem = "biored_pair_only"
    per = per_seed(Path(a.scores_dir), stem, len(d), "_test")
    if per:
        cfg = {s: json.loads((Path(a.models_root) / f"{stem}_s{s}" / "student_config.json").read_text()) for s in per}
        thr = float(np.mean([cfg[s]["threshold_dev"] for s in sorted(per)]))
        S["pair_only"], PRED["pair_only"], r = arm_stats(y, per, thr=thr)
        r["threshold_dev_per_seed"] = [cfg[s]["threshold_dev"] for s in sorted(per)]
        r["auprc_2_concepts"] = float(ap(y[~multi], S["pair_only"][~multi]))
        r["auprc_ge3_concepts"] = float(ap(y[multi], S["pair_only"][multi]))
        r["checkpoints"] = [rel(Path(a.models_root) / f"{stem}_s{s}") for s in r["seeds"]]
        out["pair_only"] = r
        out["status"]["pair_only"] = "complete" if r["complete"] else f"partial: seeds {r['seeds']}"
        out["ep"] = ep_f1(a, len(d))
    else:
        out["status"]["pair_only"] = "missing: no scored checkpoint yet"
    out["mcnemar"] = mc(PRED, y, (("pair", "pair_only"), ("sentence", "pair_only")))
    out["mcnemar_convention"] = "b_vs_a: fixes = rows b gets right and a gets wrong; breaks = the reverse"

    # seen vs unseen among the training candidates
    tr = pd.read_csv(BIORED / "train.csv", dtype={"cid1": str, "cid2": str})
    te = pd.read_csv(BIORED / "test.csv", dtype={"cid1": str, "cid2": str})
    assert (te.source_species.astype(str).to_numpy() == d.species1.astype(str).to_numpy()).all()
    out["seen_unseen"] = {"definition": "training candidates = BC8 train.csv (BioRED train+dev); surface: "
                          "normalised mention strings; concept: cid1/cid2 keys compared whole; each "
                          "model at its pre-specified threshold"}
    llm = llm_biored(a, d, y)
    for how, (s1, s2, t1, t2) in {"surface": (te.source_species, te.target_species, tr.source_species, tr.target_species),
                                  "concept": (te.cid1, te.cid2, tr.cid1, tr.cid2)}.items():
        taxa, pairs = seen_index(t1, t2, tr.label)
        f = seen_flags(s1, s2, taxa, pairs, split_alts=False)     # BioRED strings are single forms
        Sx, Px = dict(S), dict(PRED)
        Sx[LOOKUP] = lookup_scores(f, float(tr.label.mean()))
        Px[LOOKUP] = (Sx[LOOKUP] >= 0.5).astype(int)
        out["seen_unseen"][how] = {"n_training_pairs": len(pairs), "n_training_entities": len(taxa),
                                   "subsets": subset_rows(y, seen_masks(f), Sx, Px,
                                                          {k: v for k, v in llm.items() if "p" in v})}
    out["seen_unseen"]["lookup"] = {"name": LOOKUP, "what": "no model: share of the pair's training candidates "
                                    "labelled positive, train base rate when unseen; F1 at 0.5"}
    out["llm_nopassage"] = {k: {x: y_ for x, y_ in v.items() if x not in ("idx", "y", "p", "pred", "p_pair")}
                            for k, v in llm.items()}
    out["llm_note"] = ("zero-shot, the pair prompt minus the passage, on the seed-0 sample of the test set "
                       "(a partial run is a prefix of it); F1, also within subsets, is the greedy verdict")
    return out


def llm_biored(a, d, y) -> dict:
    out = {}
    names = list(LLM_BIORED) + sorted({f.name[len("biored_"):-len("_nopassage.csv")]
                                       for f in Path(a.llm_dir).glob("biored_*_nopassage.csv")} - set(LLM_BIORED))
    for m in names:
        f = Path(a.llm_dir) / f"biored_{m}_nopassage.csv"
        part = f.with_name(f.name + ".part")
        # a prefix of the seed-0 sample is itself a random sample, so an unfinished run's
        # checkpoint is usable here (never on the biodiversity benchmark, whose rows are blocked)
        src = f if f.exists() else part if part.exists() else None
        if src is None:
            out[m] = {"status": "missing"}
            continue
        o = pd.read_csv(src)
        idx = o.row.to_numpy()
        for c in ("species1", "species2", "sentence"):
            assert (o[c].astype(str).to_numpy() == d[c].astype(str).to_numpy()[idx]).all(), f"{src.name}: {c} misaligned"
        yl, p, v = y[idx], o.p_yes.to_numpy(), o.verdict.to_numpy().astype(int)
        if not two_class(yl):
            out[m] = {"status": f"too few rows so far ({len(o)})"}
            continue
        r = {"status": "complete" if src == f and len(o) >= 3000 else
             f"partial: first {len(o)} rows of the seed-0 sample" + (" (checkpoint)" if src == part else ""),
             "n": int(len(o)), "auprc": float(ap(yl, p)), "greedy": prf(yl, v), "base_rate": float(yl.mean()),
             "idx": idx, "y": yl, "p": p, "pred": v}
        for form in ("sentence", "pair"):
            g = Path(a.llm_dir) / f"biored_{m}_{form}.csv"
            if g.exists():
                go = pd.read_csv(g).set_index("row")
                if set(idx) <= set(go.index):
                    pp = go.loc[idx, "p_yes"].to_numpy()
                    r[f"auprc_{form}_prompt_same_rows"] = float(ap(yl, pp))
                    if form == "pair":
                        r["p_pair"] = pp
        if "auprc_pair_prompt_same_rows" in r:
            r["pair_prompt_minus_nopassage"] = r["auprc_pair_prompt_same_rows"] - r["auprc"]
        out[m] = r
    return out


def ep_f1(a, n_test: int) -> dict:
    """BioRED entity-pair F1 of pair_only, scripts/biored_ep_eval.py's procedure on saved scores."""
    if not (PUBTATOR / "bioredirect_test.pubtator").exists():
        return {"status": f"skipped: PubTator files not found (set BIORED_PUBTATOR_DIR); looked in {rel(PUBTATOR)}"}
    from biored_ep_eval import GRID as EGRID, ceiling, doc_scores, ep, gold_relations, load
    from convert_biored_verify import read_pubtator
    gold = {"dev": gold_relations(read_pubtator(PUBTATOR / "bioredirect_test.pubtator")),
            "test": gold_relations(read_pubtator(PUBTATOR / "bioredirect_bc8_test.pubtator"))}
    D = {sp: load("biored_bc8", sp) for sp in ("dev", "test")}
    ref = json.loads((RES / "biored_ep.json").read_text())
    assert ref["gold_relations"] == {k: len(v) for k, v in gold.items()}, "gold relations differ from biored_ep.json"
    dev_s = per_seed(Path(a.scores_dir), "biored_pair_only", len(D["dev"]), "_dev")
    test_s = per_seed(Path(a.scores_dir), "biored_pair_only", n_test, "_test")
    per = []
    for s in sorted(set(dev_s) & set(test_s)):
        pairs = {"dev": doc_scores(D["dev"], dev_s[s]), "test": doc_scores(D["test"], test_s[s])}
        thr = float(max(EGRID, key=lambda t: ep(pairs["dev"], gold["dev"], t)["F1"]))
        per.append({"seed": s, "threshold_dev": thr, "dev": ep(pairs["dev"], gold["dev"], thr),
                    "test": ep(pairs["test"], gold["test"], thr)})
    if not per:
        return {"status": "missing: no checkpoint with both dev and test scores"}
    f1 = [p["test"]["F1"] for p in per]
    return {"per_seed": per, "test_F1_mean": float(np.mean(f1)),
            "test_F1_sd": float(np.std(f1, ddof=1)) if len(f1) > 1 else None,
            "test_P_mean": float(np.mean([p["test"]["P"] for p in per])),
            "test_R_mean": float(np.mean([p["test"]["R"] for p in per])),
            "recall_ceiling_test": ceiling(pairs["test"], gold["test"]),
            "reference": {k: {x: ref[k][x] for x in ("test_F1_mean", "test_F1_sd")} for k in REF_BIORED}}


# ── report ────────────────────────────────────────────────────────────────────────────────────

def summary(bd: dict, br: dict) -> dict:
    def arm(r):
        return {k: r.get(k) for k in ("auprc_mean", "auprc_sd", "auprc_ensemble", "F1", "P", "R")} if r else None
    s = {"biodiv": {k: arm(bd.get(k)) for k in ("sentence", "pair", "triple", "pair_only", "triple_only")},
         "biored": {k: arm(br.get(k)) for k in ("sentence", "pair", "pair_only")}}
    if br.get("ep", {}).get("test_F1_mean") is not None:
        s["biored"]["pair_only_ep_f1"] = br["ep"]["test_F1_mean"]
    sub = bd["seen_unseen"]["subsets"]
    s["biodiv_seen_unseen_auprc"] = {k: {"n": sub[k]["n"], "pos_rate": sub[k]["pos_rate"],
                                         **{a: v["auprc"] for a, v in sub[k].get("arms", {}).items()}}
                                     for k in ("pair_seen", "pair_unseen", "any_taxon_seen", "neither_taxon_seen")}
    for how in ("surface", "concept"):
        sub = br["seen_unseen"][how]["subsets"]
        s[f"biored_seen_unseen_auprc_{how}"] = {k: {"n": sub[k]["n"], "pos_rate": sub[k]["pos_rate"],
                                                    **{a: v["auprc"] for a, v in sub[k].get("arms", {}).items()}}
                                                for k in ("pair_seen", "pair_unseen", "any_taxon_seen", "neither_taxon_seen")}
    s["llm_nopassage_auprc"] = {
        "biodiv": {m: [v.get("auprc"), v.get("auprc_pair_prompt")] for m, v in bd["llm_nopassage"].items() if "auprc" in v},
        "biored": {m: [v.get("auprc"), v.get("auprc_pair_prompt_same_rows")] for m, v in br["llm_nopassage"].items() if "auprc" in v},
        "columns": ["nopassage", "pair prompt"]}
    return s


def cmd_report(a) -> None:
    t0 = time.time()
    d = clean_benchmark()
    check_fast_ap(d.label.to_numpy(), np.load(RES / "S_biodiv_pair.npy").astype(float))
    bd = biodiv(a)
    br = biored(a)
    res = {"what": "pair-only controls (the pair or triple with no passage) and the no-passage zero-shot rows",
           "protocol": "see the docstring of scripts/paperA_pair_only.py",
           "bootstrap": {"resamples": B, "seed": SEED, "rng": "numpy default_rng", "unit": "candidate, within the subset",
                         "interval": "percentile 2.5-97.5", "comparisons": [f"{x}_minus_{z}" for x, z in BOOT]},
           "inputs": {"scores": rel(a.scores_dir), "models": rel(a.models_root), "llm": rel(a.llm_dir),
                      "reference_scores": "results/paperA_v2/S_biodiv_*.npy, per_seed/, S_biored_*.npy",
                      "corpus": rel(CORPUS)},
           "summary": summary(bd, br), "biodiv": bd, "biored": br,
           "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 1)}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():                              # keep the previous version
        hist = out.parent / "pair_only_history"
        hist.mkdir(exist_ok=True)
        shutil.copy2(out, hist / f"{out.stem}.{time.strftime('%Y%m%d-%H%M%S', time.localtime(out.stat().st_mtime))}.json")
    out.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res["summary"], indent=1, default=float))
    print(f"status biodiv {bd['status']}  biored {br['status']}")
    print(f"wrote {out} in {time.time() - t0:.0f}s")


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap_.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score", help="score one checkpoint (once) and save the order-free scores")
    sc.add_argument("--arm", choices=sorted(ARMS))
    sc.add_argument("--seed", type=int)
    sc.add_argument("--model", help="explicit checkpoint directory (smoke tests); needs --bench")
    sc.add_argument("--bench", choices=("biodiv", "biored"))
    sc.add_argument("--name", help="output stem with --model")
    sc.add_argument("--limit", type=int, help="first N rows only (smoke tests; needs a scratch --scores-dir)")
    sc.add_argument("--scores-dir", default=str(SCORES))
    sc.add_argument("--threads", type=int, default=int(os.environ.get("TRAIN_THREADS", 0)) or None)
    rp = sub.add_parser("report", help="every number from saved scores -> pair_only.json")
    rp.add_argument("--scores-dir", default=str(SCORES))
    rp.add_argument("--models-root", default=str(MODELS), help="checkpoints (for the BioRED development thresholds)")
    rp.add_argument("--llm-dir", default=str(LLM_DIR))
    rp.add_argument("--out", default=str(OUT_JSON))
    a = ap_.parse_args()
    cmd_score(a) if a.cmd == "score" else cmd_report(a)


if __name__ == "__main__":
    main()
