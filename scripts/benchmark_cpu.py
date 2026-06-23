#!/usr/bin/env python3
"""
CPU inference benchmark: full_typed_a05_ner2 (MultiTaskBiomedBERT)

Measures ONLY inference time. Model loading and process startup are
excluded from all measurements.

Modes compared:
  1 process  — 100 sentences classified one at a time
  10 processes — 10 sentences per worker, 10 workers in parallel

Workers are pre-warmed (model confirmed loaded) before timing starts.
Each worker uses torch.set_num_threads(1) so N processes = N CPU cores.

Usage (from MetaP root, venv active):
    python classifier/scripts/benchmark_cpu.py
    python classifier/scripts/benchmark_cpu.py --trials 5 --workers 10
"""

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

# ── Paths ──────────────────────────────────────────────────────────────────

ROOT      = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = str(ROOT / "classifier/models/multitask/full_typed_a05_ner2")
EVAL_FILE = ROOT / "classifier/data/evaluation/eval_100.tsv"

sys.path.insert(0, str(ROOT / "classifier/experiments/multitask"))

# ── Sentences ─────────────────────────────────────────────────────────────

def _load_sentences(n: int = 100) -> list[str]:
    if EVAL_FILE.exists():
        import csv
        with open(EVAL_FILE) as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        col = "sentence" if "sentence" in rows[0] else next(iter(rows[0]))
        sents = [r[col].strip() for r in rows if r[col].strip()]
        while len(sents) < n:
            sents = sents * 2
        return sents[:n]
    # Fallback synthetic set
    base = [
        "Wolbachia infects Drosophila melanogaster and manipulates its reproductive system.",
        "Haemonchus contortus was recovered from the abomasum of naturally infected sheep.",
        "Plasmodium falciparum is transmitted to humans by Anopheles gambiae mosquitoes.",
        "Borrelia burgdorferi is maintained through a tick-vertebrate transmission cycle.",
        "Toxoplasma gondii can infect virtually all warm-blooded animals including humans.",
        "No significant differences in body weight were observed between the two groups.",
        "The study was conducted using standard laboratory protocols described previously.",
        "Statistical analysis was performed using SPSS version 25.0 software package.",
    ]
    result = []
    while len(result) < n:
        result.extend(base)
    return result[:n]

# ── Worker globals (set once per process in _init) ─────────────────────────

_model     = None
_tokenizer = None
_device    = None

def _init():
    """Load model once per worker. Not timed."""
    global _model, _tokenizer, _device
    from model import MultiTaskBiomedBERT
    torch.set_num_threads(1)   # 1 OMP thread per process → N processes = N cores
    _device    = torch.device("cpu")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    _model     = MultiTaskBiomedBERT.load(MODEL_DIR, device="cpu")
    _model.eval()

def _classify_one(sentence: str) -> float:
    enc = _tokenizer(
        [sentence], truncation=True, max_length=256,
        padding=True, return_tensors="pt",
    ).to(_device)
    with torch.no_grad():
        out = _model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            token_type_ids=enc.get("token_type_ids"),
        )
    return torch.softmax(out["cls_logits"], dim=-1)[0, 1].item()

def _classify_chunk(sentences: list[str]) -> list[float]:
    """Classify a list of sentences one at a time. Used as pool task."""
    return [_classify_one(s) for s in sentences]

# ── Benchmark helpers ─────────────────────────────────────────────────────

def _stats(times: list[float], n_sent: int) -> dict:
    mean = sum(times) / len(times)
    best = min(times)
    return {
        "mean_s":      mean,
        "best_s":      best,
        "ms_per_sent": mean / n_sent * 1000,
        "sent_per_s":  n_sent / mean,
    }

def _fmt(label: str, s: dict) -> str:
    return (
        f"  {label:<18} "
        f"mean {s['mean_s']:6.2f}s  "
        f"best {s['best_s']:6.2f}s  "
        f"{s['ms_per_sent']:6.1f} ms/sent  "
        f"{s['sent_per_s']:5.1f} sent/s"
    )

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",    type=int, default=5,
                        help="Timed trials per mode (default 5)")
    parser.add_argument("--sentences", type=int, default=100,
                        help="Sentences to classify (default 100)")
    parser.add_argument("--workers",   type=int, default=10,
                        help="Parallel worker processes (default 10)")
    args = parser.parse_args()

    sentences = _load_sentences(args.sentences)
    src = "eval_100.tsv" if EVAL_FILE.exists() else "synthetic"
    print(f"Sentences : {len(sentences)} (from {src})")
    print(f"Model     : {MODEL_DIR}")
    print(f"Trials    : {args.trials} per mode")
    print()

    # ── Mode 1: single process ─────────────────────────────────────────────
    print(f"[1/2] Loading model for single-process benchmark ...", end=" ", flush=True)
    t0 = time.perf_counter()
    _init()
    print(f"{time.perf_counter()-t0:.1f}s  (excluded from timing)")
    print(f"      Running {args.trials} trials × {args.sentences} sentences ...", flush=True)

    single_times = []
    for i in range(args.trials):
        t0 = time.perf_counter()
        for s in sentences:
            _classify_one(s)
        elapsed = time.perf_counter() - t0
        single_times.append(elapsed)
        print(f"      trial {i+1}: {elapsed:.2f}s")

    s = _stats(single_times, args.sentences)
    print(_fmt("1 process:", s))
    print()

    # ── Mode 2: N workers ──────────────────────────────────────────────────
    # Split sentences into equal chunks
    n = args.workers
    q, r = divmod(args.sentences, n)
    chunks = []
    idx = 0
    for i in range(n):
        size = q + (1 if i < r else 0)
        chunks.append(sentences[idx: idx + size])
        idx += size

    ctx = mp.get_context("spawn")   # spawn: safe with PyTorch, no shared state

    print(f"[2/2] Spawning {n} workers, loading model in each ...", end=" ", flush=True)
    t0 = time.perf_counter()
    with ctx.Pool(processes=n, initializer=_init) as pool:
        # Warmup: send one dummy sentence to every worker, confirming all
        # initializers have completed before we start the clock.
        pool.map(_classify_chunk, [["Wolbachia infects Drosophila."]] * n)
        print(f"{time.perf_counter()-t0:.1f}s  (excluded from timing)")
        print(f"      Running {args.trials} trials × {args.sentences} sentences "
              f"({n} workers × {[len(c) for c in chunks]} sents) ...", flush=True)

        multi_times = []
        for i in range(args.trials):
            t0 = time.perf_counter()
            pool.map(_classify_chunk, chunks)
            elapsed = time.perf_counter() - t0
            multi_times.append(elapsed)
            print(f"      trial {i+1}: {elapsed:.2f}s")

    m = _stats(multi_times, args.sentences)
    print(_fmt(f"{n} processes:", m))
    print()

    # ── Summary ────────────────────────────────────────────────────────────
    speedup = s["mean_s"] / m["mean_s"]
    efficiency = speedup / n * 100

    print("=" * 62)
    print(f"  Sentences    : {args.sentences}")
    print(f"  1 process    : {s['mean_s']:.2f}s  ({s['ms_per_sent']:.0f} ms/sent)")
    print(f"  {n} processes  : {m['mean_s']:.2f}s  ({m['ms_per_sent']:.0f} ms/sent)")
    print(f"  Speedup      : {speedup:.1f}×  (parallel efficiency {efficiency:.0f}%)")
    print("=" * 62)


if __name__ == "__main__":
    main()
