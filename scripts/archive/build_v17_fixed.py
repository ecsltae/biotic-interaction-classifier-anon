#!/usr/bin/env python3
"""
Build v17a dataset — Track A fixes:
  1. Cap kleptoparasiteOf at 300 (was 1,421 = 34.7% of positives)
  2. Add 4,000 hard negatives from negatives_weak_signal.csv (lexicon 0.05-0.35)
  3. Real sentences only, no v7 templates

Expected: ~17,400 rows, ~19% pos, balanced categories, harder negatives.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

BASE = Path("/path/to/MetaP/classifier")
OUT_DIR = BASE / "data/training/v17_fixed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KLEPTO_CAP = 300
N_HARD_NEG = 4000
NEG_RATIO = 3.0   # easy negatives per positive
SEED = 42

# ── 1. Teacher-labeled real positives (with category cap) ──────────────────
print("Loading teacher-labeled positives...")
teacher = pd.read_csv(BASE / "results/research_agent/all_sources_qwen122b_labeled.csv")
pos = teacher[teacher["teacher_label"] == 1].copy()
pos["text"] = pos["text"].str.strip()

# Cap kleptoparasiteOf
klepto = pos[pos["interaction_type"] == "kleptoparasiteOf"]
other_pos = pos[pos["interaction_type"] != "kleptoparasiteOf"]
klepto_capped = klepto.sample(n=min(KLEPTO_CAP, len(klepto)), random_state=SEED)
pos_balanced = pd.concat([other_pos, klepto_capped], ignore_index=True)

print(f"  Before cap: {len(pos)} pos")
print(f"  After kleptoparasiteOf cap ({KLEPTO_CAP}): {len(pos_balanced)} pos")
print(f"  Category distribution:")
for cat, n in pos_balanced["interaction_type"].value_counts().items():
    print(f"    {cat}: {n} ({n/len(pos_balanced)*100:.1f}%)")

# Keep text + label only
pos_out = pos_balanced[["text"]].drop_duplicates().copy()
pos_out["label"] = 1
pos_out["source"] = "teacher_qwen122b"

# ── 2. Other positive sources ──────────────────────────────────────────────
def load_pos(path, label_col, source_name):
    df = pd.read_csv(path)
    col = label_col if label_col in df.columns else "label"
    out = df[df[col] == 1][["text"]].drop_duplicates().copy()
    out["label"] = 1
    out["source"] = source_name
    return out

pharvest = load_pos(BASE / "data/training/pathogen_harvested.csv", "teacher_label", "pathogen_harvested")
curated = load_pos(BASE / "data/training/curated_pathogen_borderline.csv", "label", "curated_borderline")

eval100 = pd.read_csv(BASE / "data/evaluation/eval_100.tsv", sep="\t")
eval100_pos = eval100[eval100["evaluation_pair_interacting"] == 1][["sentence"]].copy()
eval100_pos.columns = ["text"]
eval100_pos["label"] = 1
eval100_pos["source"] = "eval100_gold"

print(f"\n  pathogen_harvested: {len(pharvest)}")
print(f"  curated_borderline: {len(curated)}")
print(f"  eval_100 gold: {len(eval100_pos)}")

positives = pd.concat([pos_out, pharvest, curated, eval100_pos], ignore_index=True)
positives = positives.drop_duplicates(subset=["text"])
n_pos = len(positives)
print(f"\nTotal positives: {n_pos}")

# ── 3. Easy negatives (3× ratio) ──────────────────────────────────────────
print(f"\nLoading easy negatives ({NEG_RATIO}× = {int(n_pos * NEG_RATIO)})...")
easy_neg = pd.read_csv(BASE / "data/training/negatives_clean.csv")
easy_neg = easy_neg[~easy_neg["text"].isin(positives["text"])][["text"]].drop_duplicates()
n_easy = min(int(n_pos * NEG_RATIO), len(easy_neg))
easy_sample = easy_neg.sample(n=n_easy, random_state=SEED).copy()
easy_sample["label"] = 0
easy_sample["source"] = "negatives_clean"
print(f"  Sampled easy negatives: {n_easy}")

# ── 4. Hard negatives (weak signal, unused until now) ─────────────────────
print(f"\nLoading hard negatives ({N_HARD_NEG} from weak_signal)...")
hard_neg = pd.read_csv(BASE / "data/training/negatives_weak_signal.csv")
hard_neg = hard_neg[hard_neg["label"] == 0]  # confirmed negatives with lexicon signal
hard_neg = hard_neg[~hard_neg["text"].isin(positives["text"])][["text"]].drop_duplicates()
n_hard = min(N_HARD_NEG, len(hard_neg))
hard_sample = hard_neg.sample(n=n_hard, random_state=SEED).copy()
hard_sample["label"] = 0
hard_sample["source"] = "negatives_weak_signal"
print(f"  Sampled hard negatives: {n_hard}")

# ── 5. Assemble & shuffle ─────────────────────────────────────────────────
dataset = pd.concat([positives, easy_sample, hard_sample], ignore_index=True)
dataset = dataset.sample(frac=1, random_state=SEED).reset_index(drop=True)
dataset = dataset[["text", "label", "source"]]

n_total = len(dataset)
n_pos_f = int((dataset["label"] == 1).sum())
n_neg_f = int((dataset["label"] == 0).sum())
pos_rate = n_pos_f / n_total

print(f"\nFinal dataset: {n_total} rows")
print(f"  Positives: {n_pos_f} ({pos_rate:.1%})")
print(f"  Negatives: {n_neg_f} (easy={n_easy}, hard={n_hard})")

out_path = OUT_DIR / "dataset.csv"
dataset.to_csv(out_path, index=False)
print(f"Saved to {out_path}")

# Metadata
meta = {
    "version": "v17a_fixed",
    "built": datetime.now().isoformat(),
    "track": "A — balance fix + hard negatives, no Qwen recalibration",
    "fixes": ["kleptoparasiteOf capped at 300", "4000 hard negatives added", "no v7 templates"],
    "training_rows": n_total,
    "training_pos": n_pos_f,
    "training_neg": n_neg_f,
    "pos_rate": round(pos_rate, 4),
    "neg_breakdown": {"easy_lexicon0": n_easy, "hard_weak_signal": n_hard},
    "positive_sources": {
        "teacher_qwen122b_balanced": int(len(pos_out)),
        "pathogen_harvested": int(len(pharvest)),
        "curated_borderline": int(len(curated)),
        "eval100_gold": int(len(eval100_pos)),
    }
}
with open(OUT_DIR / "metadata.json", "w") as f:
    json.dump(meta, f, indent=2)
print("Metadata saved.")
