#!/usr/bin/env python3
"""Per-checkpoint BioRED candidate scores for scripts/biored_ep_eval.py, computed on CPU and saved.

The entity-pair evaluation chooses a threshold per checkpoint on development data, so it needs each
checkpoint's own scores on the development and test candidates. biored_ep_eval.py used to compute
them on the GPU without saving them; this script recomputes them on CPU and writes one vector per
(arm, seed, split) to results/paperA_v2/biored_ep_scores/<arm>_s<seed>_<split>.npy, which
biored_ep_eval.py reads instead of rescoring.

The score is the quantity of scripts/paperA_tables.py's order_free(): the positive-class
probability, the maximum over the two argument orders for a model that reads the query, a single
pass for the sentence-level model and for the document-scope model (whose markers fix the order).
Inputs are batched by length and identical inputs are scored once (all candidates of one sentence
give the sentence-level model the same input); neither changes a score beyond float rounding.

Usage
  python3 scripts/biored_ep_scores_cpu.py --arms sentence pair --workers 4 --threads 4
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "results/paperA_v2/biored_ep_scores"

_MODEL = {"dir": None}


def _init(threads: int) -> None:
    import torch
    torch.set_num_threads(threads)


def _score_chunk(job):
    """Score one chunk of unique inputs with one checkpoint (the model stays loaded across jobs)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    md, ml, two_seg, A, B, bs = job["md"], job["ml"], job["two_seg"], job["A"], job["B"], job["bs"]
    if _MODEL["dir"] != md:
        _MODEL.update(dir=md, tok=AutoTokenizer.from_pretrained(md, local_files_only=True),
                      m=AutoModelForSequenceClassification.from_pretrained(md, local_files_only=True).eval())
    tok, m = _MODEL["tok"], _MODEL["m"]
    out = np.zeros(len(A), np.float32)
    with torch.inference_mode():
        for i in range(0, len(A), bs):
            if two_seg:
                e = tok(A[i:i + bs], B[i:i + bs], truncation="only_second", max_length=ml,
                        padding=True, return_tensors="pt")
            else:
                e = tok(A[i:i + bs], truncation=True, max_length=ml, padding=True, return_tensors="pt")
            out[i:i + bs] = torch.softmax(m(**e).logits.float(), -1)[:, 1].numpy()
    return job["key"], job["lo"], out


def plan(arm: str, seed: int, split: str, chunk: int, bs: int):
    """The unique inputs of one (arm, seed, split), cut into length-sorted chunks."""
    from biored_ep_eval import ARMS, load
    from eval_unified import model_format, model_max_len
    import xenc_format
    bench, pat = ARMS[arm]
    md = str(REPO / pat.format(seed))
    fmt, ml = model_format(md), model_max_len(md)
    d = load(bench, split)
    orders = [(d.species1, d.species2)]
    if not bench.endswith("_doc") and fmt != "sentence":       # order_free(): both argument orders
        orders.append((d.species2, d.species1))
    rows = []                                                  # (order, row) -> input
    for o, (s1, s2) in enumerate(orders):
        A, B = xenc_format.build_many(fmt, s1, d.relation, s2, d.sentence.astype(str))
        rows += [(o, i, A[i], None if B is None else B[i]) for i in range(len(A))]
    uniq = {}
    for _, _, a, b in rows:
        uniq.setdefault((a, b), len(uniq))
    inputs = sorted(uniq, key=lambda ab: len(ab[0]) + len(ab[1] or ""))
    pos = {ab: k for k, ab in enumerate(inputs)}
    index = np.array([[pos[(a, b)] for (o2, _, a, b) in rows if o2 == o] for o in range(len(orders))])
    two_seg = inputs[0][1] is not None
    key = f"{arm}_s{seed}_{split}"
    jobs = [{"key": key, "lo": lo, "md": md, "ml": ml, "two_seg": two_seg, "bs": bs,
             "A": [a for a, _ in inputs[lo:lo + chunk]], "B": [b for _, b in inputs[lo:lo + chunk]] if two_seg else None}
            for lo in range(0, len(inputs), chunk)]
    meta = {"arm": arm, "seed": seed, "split": split, "model": pat.format(seed), "format": fmt, "max_len": ml,
            "rows": len(d), "orders": len(orders), "unique_inputs": len(inputs)}
    return key, jobs, index, len(inputs), meta


def main() -> None:
    from biored_ep_eval import ARMS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", required=True, choices=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4, help="torch threads per worker")
    ap.add_argument("--chunk", type=int, default=1024, help="unique inputs per job")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    todo, jobs = {}, []
    for arm in a.arms:
        for seed in a.seeds:
            for split in a.splits:
                if (out / f"{arm}_s{seed}_{split}.npy").exists():
                    print(f"have {arm}_s{seed}_{split}", flush=True)
                    continue
                key, js, index, n, meta = plan(arm, seed, split, a.chunk, a.bs)
                todo[key] = {"index": index, "scores": np.full(n, np.nan, np.float32), "left": len(js),
                             "meta": meta, "t0": None}
                jobs += js
    print(f"{len(jobs)} jobs over {len(todo)} score vectors, {a.workers} workers x {a.threads} threads", flush=True)
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(a.workers, initializer=_init, initargs=(a.threads,), maxtasksperchild=None) as pool:
        for key, lo, s in pool.imap(_score_chunk, jobs):
            t = todo[key]
            t["scores"][lo:lo + len(s)] = s
            t["left"] -= 1
            if t["left"] == 0:
                assert not np.isnan(t["scores"]).any(), key
                vec = t["scores"][t["index"]].max(axis=0)          # max over argument orders
                np.save(out / f"{key}.npy", vec)
                (out / f"{key}.json").write_text(json.dumps({**t["meta"], "device": "cpu",
                                                             "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
                print(f"done {key}: {t['meta']['unique_inputs']} unique inputs, {time.time() - t0:.0f}s since start",
                      flush=True)


if __name__ == "__main__":
    main()
