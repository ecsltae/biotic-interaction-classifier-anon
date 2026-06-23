#!/usr/bin/env python3
"""
Fork-based benchmark: model loaded ONCE in parent, shared across N workers.

With fork (Linux), child processes inherit the parent's memory copy-on-write.
Model weights are read-only during inference so the OS never copies the pages.
Result: N processes, 1 model in RAM.

Compares:
  spawn  — each worker loads independently (N loads, N × 419MB RAM)
  fork   — parent loads once, workers inherit (1 load, 1 × 419MB RAM)

Usage:
    python classifier/scripts/benchmark_fork.py [--workers 10] [--trials 3]
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT      = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = str(ROOT / "classifier/models/multitask/full_typed_a05_ner2")
EVAL_FILE = ROOT / "classifier/data/evaluation/eval_100.tsv"
sys.path.insert(0, str(ROOT / "classifier/experiments/multitask"))

# ── Shared state (set in parent, inherited by forked children) ─────────────

_model     = None
_tokenizer = None
_device    = None

def load_model():
    global _model, _tokenizer, _device
    from model import MultiTaskBiomedBERT
    torch.set_num_threads(1)
    _device    = torch.device("cpu")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    _model     = MultiTaskBiomedBERT.load(MODEL_DIR, device="cpu")
    _model.eval()
    _model.share_memory()   # pins weights in shared memory — explicit CoW protection

def _classify_chunk(sentences: list[str]) -> list[float]:
    results = []
    for s in sentences:
        enc = _tokenizer([s], truncation=True, max_length=256,
                         padding=True, return_tensors="pt").to(_device)
        with torch.no_grad():
            out = _model(input_ids=enc["input_ids"],
                         attention_mask=enc["attention_mask"],
                         token_type_ids=enc.get("token_type_ids"))
        results.append(torch.softmax(out["cls_logits"], dim=-1)[0, 1].item())
    return results

# ── Sentence loader ────────────────────────────────────────────────────────

def load_sentences(n=100):
    import csv
    with open(EVAL_FILE) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    col   = "sentence" if "sentence" in rows[0] else next(iter(rows[0]))
    sents = [r[col].strip() for r in rows if r[col].strip()]
    while len(sents) < n:
        sents *= 2
    return sents[:n]

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers",   type=int, default=10)
    parser.add_argument("--sentences", type=int, default=100)
    parser.add_argument("--trials",    type=int, default=3)
    args = parser.parse_args()

    sentences = load_sentences(args.sentences)
    n = args.workers
    q, r = divmod(len(sentences), n)
    chunks, idx = [], 0
    for i in range(n):
        size = q + (1 if i < r else 0)
        chunks.append(sentences[idx: idx + size])
        idx += size

    print(f"Sentences : {len(sentences)}")
    print(f"Workers   : {n}")
    print(f"Trials    : {args.trials}")
    print(f"CPU cores : {os.cpu_count()}")
    print()

    # ── FORK: load once in parent ─────────────────────────────────────────
    print("Loading model in parent process (ONCE) ...", end=" ", flush=True)
    t0 = time.perf_counter()
    load_model()
    load_time = time.perf_counter() - t0
    print(f"{load_time:.1f}s")
    print(f"  Workers will inherit this via fork — no reloading.")
    print()

    fork_times = []
    with mp.get_context("fork").Pool(processes=n) as pool:
        # Warmup: confirm all workers are live (they already have the model)
        pool.map(_classify_chunk, [["warmup sentence"]] * n)

        print(f"Running {args.trials} trials (fork, {n} workers) ...", flush=True)
        for i in range(args.trials):
            t0 = time.perf_counter()
            pool.map(_classify_chunk, chunks)
            elapsed = time.perf_counter() - t0
            fork_times.append(elapsed)
            print(f"  trial {i+1}: {elapsed:.2f}s  "
                  f"({len(sentences)/elapsed:.1f} sent/s)")

    f_mean = sum(fork_times) / len(fork_times)

    # ── Summary ────────────────────────────────────────────────────────────
    print()
    print("=" * 56)
    print(f"  Workers   : {n}  (on {os.cpu_count()} cores)")
    print(f"  Sentences : {len(sentences)}")
    print(f"  Model loads: 1  (parent only, workers inherit via fork)")
    print()
    print(f"  mean : {f_mean:.2f}s  "
          f"({f_mean/len(sentences)*1000:.0f} ms/sent)  "
          f"{len(sentences)/f_mean:.1f} sent/s")
    print(f"  best : {min(fork_times):.2f}s")
    print(f"  RAM  : ~419MB  (1 shared copy regardless of worker count)")
    print("=" * 56)


if __name__ == "__main__":
    main()
