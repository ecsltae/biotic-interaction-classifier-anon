#!/usr/bin/env python3
"""Label the pair-diverse pool with the query-conditioned local LLM teacher.

The prompt (PROMPT below) is the triple question of the paper's Appendix A. This script
records the teacher's YES/NO verdict per row; scripts/soft_relabel.py re-queries the same
prompt for P(yes).
Resumable: re-running skips rows already in the output.
"""
import json, re, sys, time
from pathlib import Path
import pandas as pd, requests

REPO = Path(__file__).resolve().parents[1]
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3:32b"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO/"data/training/distill/teacher_labels.csv"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 0
URL = "http://localhost:11434/api/generate"

PROMPT = """You verify candidate biotic-interaction triples extracted from scientific literature.

Sentence: {sent}

Candidate triple:
  Entity 1: "{s1}"
  Relation: "{rel}"
  Entity 2: "{s2}"

Answer YES only if ALL of the following hold:
 1. Both entities are genuinely the organisms referred to by those surface strings in this sentence (not a gene, protein, chemical, author name, or a different species).
 2. The sentence asserts a direct biological interaction between these two organisms (not merely a co-mention, and not an interaction each has with some third organism).
 3. The stated relation is the correct type of interaction between them, in the stated direction, and is not negated.

Reply with exactly one word: YES or NO."""

import os
POOL=os.environ.get("POOL","data/training/distill/distill_pool.csv")
df = pd.read_csv(REPO/POOL)
if N: df = df.head(N)
done = set()
if OUT.exists():
    prev = pd.read_csv(OUT); done = set(prev["row"].tolist())
    print(f"resuming: {len(done)} already labelled", flush=True)

rows, t0, since = [], time.time(), 0
for i, r in df.iterrows():
    if i in done: continue
    p = PROMPT.format(sent=r.get("passage", r.get("text")), s1=r.get("species1_form", r.get("source_species")), s2=r.get("species2_form", r.get("target_species")), rel=r.get("interaction_form", r.get("interaction_type")))
    try:
        resp = requests.post(URL, json={"model": MODEL, "prompt": p, "stream": False,
            "options": {"temperature": 0, "num_predict": 4, "seed": 0}, "think": False},
            timeout=300).json().get("response", "")
    except Exception as e:
        resp = "ERR:" + type(e).__name__
    rows.append({"row": i, "label": 1 if re.search(r"\byes\b", resp, re.I) else 0,
                 "raw": resp.strip()[:24]})
    since += 1
    if since % 200 == 0:
        pd.DataFrame(rows).to_csv(OUT, mode="a" if OUT.exists() and done else "w",
                                  header=not (OUT.exists() and done), index=False)
        done |= {x["row"] for x in rows}; rows = []
        el = time.time()-t0
        print(f"  {len(done)}/{len(df)}  {el:.0f}s  {since/el:.2f} it/s  "
              f"eta {(len(df)-len(done))/(since/el)/3600:.1f}h", flush=True)
if rows:
    pd.DataFrame(rows).to_csv(OUT, mode="a" if done else "w", header=not done, index=False)
print("done", round(time.time()-t0), "s", flush=True)
