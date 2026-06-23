#!/usr/bin/env python3
"""
Build v17b dataset — Track B: recalibrated Qwen labels + Track A fixes.
Must run after recalibrate_qwen_labels.py produces qwen_positives_recalibrated.csv.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE = Path("/path/to/MetaP/classifier")
OUT_DIR = BASE / "data/training/v17b_recalibrated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KLEPTO_CAP = 300
N_HARD_NEG = 4000
NEG_RATIO = 3.0
SEED = 42

# ── 1. Recalibrated positives ──────────────────────────────────────────────
print("Loading recalibrated Qwen positives...")
recal = pd.read_csv(BASE / "data/training/qwen_positives_recalibrated.csv")
recal_pos = recal[recal["recalibrated_label"] == 1].copy()
print(f"  Survived recalibration: {len(recal_pos)}/{len(recal)} ({len(recal_pos)/len(recal)*100:.1f}%)")

# Cap kleptoparasiteOf
klepto = recal_pos[recal_pos["interaction_type"] == "kleptoparasiteOf"]
other = recal_pos[recal_pos["interaction_type"] != "kleptoparasiteOf"]
klepto_capped = klepto.sample(n=min(KLEPTO_CAP, len(klepto)), random_state=SEED)
pos_balanced = pd.concat([other, klepto_capped], ignore_index=True)
print(f"  After kleptoparasiteOf cap: {len(pos_balanced)}")
print("  Category distribution:")
for cat, n in pos_balanced["interaction_type"].value_counts().items():
    print(f"    {cat}: {n} ({n/len(pos_balanced)*100:.1f}%)")

pos_out = pos_balanced[["text"]].drop_duplicates().copy()
pos_out["label"] = 1
pos_out["source"] = "qwen_recalibrated"

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
eval100_pos = eval100[eval100["evaluation_pair_interacting"] == 1][["sentence"]].rename(columns={"sentence": "text"})
eval100_pos["label"] = 1
eval100_pos["source"] = "eval100_gold"

positives = pd.concat([pos_out, pharvest, curated, eval100_pos], ignore_index=True)
positives = positives.drop_duplicates(subset=["text"])
n_pos = len(positives)
print(f"\nTotal positives: {n_pos}")

# ── 3. Negatives ───────────────────────────────────────────────────────────
easy_neg = pd.read_csv(BASE / "data/training/negatives_clean.csv")
easy_neg = easy_neg[~easy_neg["text"].isin(positives["text"])][["text"]].drop_duplicates()
n_easy = min(int(n_pos * NEG_RATIO), len(easy_neg))
easy_sample = easy_neg.sample(n=n_easy, random_state=SEED).copy()
easy_sample["label"] = 0
easy_sample["source"] = "negatives_clean"

hard_neg = pd.read_csv(BASE / "data/training/negatives_weak_signal.csv")
hard_neg = hard_neg[hard_neg["label"] == 0][~hard_neg["text"].isin(positives["text"])][["text"]].drop_duplicates()
n_hard = min(N_HARD_NEG, len(hard_neg))
hard_sample = hard_neg.sample(n=n_hard, random_state=SEED).copy()
hard_sample["label"] = 0
hard_sample["source"] = "negatives_weak_signal"

print(f"Negatives: {n_easy} easy + {n_hard} hard = {n_easy + n_hard}")

# ── 4. Assemble ────────────────────────────────────────────────────────────
dataset = pd.concat([positives, easy_sample, hard_sample], ignore_index=True)
dataset = dataset.sample(frac=1, random_state=SEED).reset_index(drop=True)[["text", "label", "source"]]

n_total = len(dataset)
n_pos_f = int((dataset["label"] == 1).sum())
pos_rate = n_pos_f / n_total

print(f"\nFinal: {n_total} rows, {n_pos_f} pos ({pos_rate:.1%})")
dataset.to_csv(OUT_DIR / "dataset.csv", index=False)

meta = {
    "version": "v17b_recalibrated",
    "built": datetime.now().isoformat(),
    "track": "B — Qwen recalibrated with few-shot EP-relax examples + balance + hard negs",
    "training_rows": n_total,
    "training_pos": n_pos_f,
    "pos_rate": round(pos_rate, 4),
}
with open(OUT_DIR / "metadata.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"Saved to {OUT_DIR}/dataset.csv")
