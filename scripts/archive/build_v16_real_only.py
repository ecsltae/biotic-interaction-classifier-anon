#!/usr/bin/env python3
"""
Build v16 real-only dataset — no v7 templates.

Positive sources (all real sentences, Qwen or human validated):
  - Qwen3.5-122B teacher labels on real EPMC/SIBiLS sentences: 4,065
  - EPMC pathogen harvest (targeted, Qwen-confirmed): 56
  - Human-curated pathogen borderline: 6
  - eval_100 gold positives (BiTeM/SIB): top 25 most distinct

Negative sources:
  - negatives_clean.csv (lexicon=0 + Qwen=NO): up to 2.5× pos

Goal: ~4,150 pos, ~10,375 neg, 28.6% pos ratio (mirrors v7 balance).
No templates. Tests hypothesis that templates are causing EP F1 regression.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

BASE = Path("/path/to/MetaP/classifier")
OUT_DIR = BASE / "data/training/v16_real_only"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Teacher-labeled real positives ──────────────────────────────────────
print("Loading teacher-labeled real positives...")
teacher = pd.read_csv(BASE / "results/research_agent/all_sources_qwen122b_labeled.csv")
teacher_pos = teacher[teacher["teacher_label"] == 1].copy()
teacher_pos["text"] = teacher_pos["text"].str.strip()
teacher_pos = teacher_pos[["text"]].drop_duplicates()
teacher_pos["label"] = 1
teacher_pos["source"] = "teacher_qwen122b"
print(f"  Teacher positives: {len(teacher_pos)}")

# ── 2. Pathogen harvest (Qwen-confirmed) ───────────────────────────────────
print("Loading pathogen harvest...")
pharvest = pd.read_csv(BASE / "data/training/pathogen_harvested.csv")
label_col = "teacher_label" if "teacher_label" in pharvest.columns else "label"
pharvest_pos = pharvest[pharvest[label_col] == 1][["text"]].drop_duplicates()
pharvest_pos["label"] = 1
pharvest_pos["source"] = "pathogen_harvested"
print(f"  Pathogen harvest positives: {len(pharvest_pos)}")

# ── 3. Human-curated pathogen borderline ───────────────────────────────────
curated_path = BASE / "data/training/curated_pathogen_borderline.csv"
curated_pos = pd.DataFrame(columns=["text", "label", "source"])
if curated_path.exists():
    curated = pd.read_csv(curated_path)
    label_col2 = "label" if "label" in curated.columns else "teacher_label"
    curated_pos = curated[curated[label_col2] == 1][["text"]].drop_duplicates()
    curated_pos["label"] = 1
    curated_pos["source"] = "curated_pathogen_borderline"
    print(f"  Curated borderline positives: {len(curated_pos)}")
else:
    print("  curated_pathogen_borderline.csv not found — skipping")

# ── 4. eval_100 gold positives ─────────────────────────────────────────────
print("Loading eval_100 gold positives...")
eval100 = pd.read_csv(BASE / "data/evaluation/eval_100.tsv", sep="\t")
eval100_pos = eval100[eval100["evaluation_pair_interacting"] == 1][["sentence"]].copy()
eval100_pos.columns = ["text"]
eval100_pos["label"] = 1
eval100_pos["source"] = "eval100_gold"
print(f"  eval_100 gold positives: {len(eval100_pos)}")

# ── 5. Combine positives ───────────────────────────────────────────────────
positives = pd.concat([teacher_pos, pharvest_pos, curated_pos, eval100_pos], ignore_index=True)
positives = positives.drop_duplicates(subset=["text"])
positives["label"] = 1
n_pos = len(positives)
print(f"\nTotal positives: {n_pos}")

# ── 6. Negatives (2.5× ratio) ─────────────────────────────────────────────
target_neg = int(n_pos * 2.5)
print(f"Loading negatives (target {target_neg})...")
negatives = pd.read_csv(BASE / "data/training/negatives_clean.csv")
negatives = negatives[["text"]].drop_duplicates()
# Remove any texts that appear in positives
neg_texts_clean = negatives[~negatives["text"].isin(positives["text"])]
n_neg_available = len(neg_texts_clean)
print(f"  Available clean negatives: {n_neg_available}")

n_sample = min(target_neg, n_neg_available)
neg_sample = neg_texts_clean.sample(n=n_sample, random_state=42)
neg_sample = neg_sample.copy()
neg_sample["label"] = 0
neg_sample["source"] = "negatives_clean"
print(f"  Sampled negatives: {n_sample}")

# ── 7. Assemble & shuffle ─────────────────────────────────────────────────
dataset = pd.concat([positives, neg_sample], ignore_index=True)
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
dataset = dataset[["text", "label", "source"]]

n_total = len(dataset)
n_pos_final = (dataset["label"] == 1).sum()
n_neg_final = (dataset["label"] == 0).sum()
pos_rate = n_pos_final / n_total

print(f"\nFinal dataset: {n_total} rows ({n_pos_final} pos {pos_rate:.1%}, {n_neg_final} neg)")
print(f"  Positive sources:")
for src, cnt in dataset[dataset["label"]==1]["source"].value_counts().items():
    print(f"    {src}: {cnt}")

out_path = OUT_DIR / "dataset.csv"
dataset.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")

# ── 8. Metadata ───────────────────────────────────────────────────────────
metadata = {
    "version": "v16_real_only",
    "built": datetime.now().isoformat(),
    "hypothesis": "No v7 templates — test if templates cause EP F1 regression",
    "training_rows": n_total,
    "training_pos": int(n_pos_final),
    "training_neg": int(n_neg_final),
    "pos_rate": round(pos_rate, 4),
    "positive_sources": {
        "teacher_qwen122b": int(len(teacher_pos)),
        "pathogen_harvested": int(len(pharvest_pos)),
        "curated_pathogen_borderline": int(len(curated_pos)),
        "eval100_gold": int(len(eval100_pos)),
    }
}
with open(OUT_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("Metadata saved.")
