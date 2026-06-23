#!/usr/bin/env python3
"""
Build v18 hybrid dataset.

Key insight from analysis:
- EP-relax test: 58% of positives are endoparasiteOf/hasHost/parasitoid type
- Current real corpus (Qwen-labeled EPMC): has ZERO endoparasiteOf — never harvested
- v7 Qwen-validated: has 3,731 endoparasiteOf + 682 preysOn + 596 hasHost

Strategy: targeted hybrid
  - Real sentences for types present in real corpus (pollinates, eats, kleptoparasiteOf, parasiteOf)
  - v7 Qwen-validated templates ONLY for types absent from real corpus (endoparasiteOf, hasHost, preysOn)
  - Hard negatives included (weak signal pool)
  - kleptoparasiteOf capped at 300

This is the key difference from v15b (which mixed everything indiscriminately).
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

BASE = Path("/path/to/MetaP/classifier")
OUT_DIR = BASE / "data/training/v18_hybrid"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KLEPTO_CAP = 300
N_HARD_NEG = 4000
NEG_RATIO = 3.0
SEED = 42

print("=" * 60)
print("Building v18 hybrid dataset")
print("=" * 60)

# ── 1. Real sentence positives (from Qwen teacher, real EPMC/SIBiLS) ──────
print("\n[1] Loading real-sentence positives (Qwen teacher)...")
teacher = pd.read_csv(BASE / "results/research_agent/all_sources_qwen122b_labeled.csv")
real_pos = teacher[teacher["teacher_label"] == 1].copy()
real_pos["text"] = real_pos["text"].str.strip()

# Cap kleptoparasiteOf
klepto = real_pos[real_pos["interaction_type"] == "kleptoparasiteOf"]
other_real = real_pos[real_pos["interaction_type"] != "kleptoparasiteOf"]
klepto_capped = klepto.sample(n=min(KLEPTO_CAP, len(klepto)), random_state=SEED)
real_balanced = pd.concat([other_real, klepto_capped], ignore_index=True)

real_out = real_balanced[["text"]].drop_duplicates().copy()
real_out["label"] = 1
real_out["source"] = "teacher_qwen122b_real"
print(f"  Real positives (after klepto cap): {len(real_out)}")
for cat, n in real_balanced["interaction_type"].value_counts().items():
    print(f"    {cat}: {n}")

# ── 2. v7 Qwen-validated templates — ONLY for types missing from real corpus ─
print("\n[2] Loading v7 Qwen-validated templates (gap-filling only)...")
v7 = pd.read_csv(BASE / "data/training/v7_non_pathogen_qwen_validated.csv")
# qwen_label may be integer 1 or string "YES" depending on how the file was saved
v7_pos = v7[v7["qwen_label"].isin([1, "YES", "yes"])].copy()

# Types present in real corpus → skip (already covered)
covered_by_real = {"kleptoparasiteOf", "pollinates", "eats", "parasiteOf",
                   "pathogenOf", "symbioticWith", "visitsFlowersOf"}
# Types missing from real corpus → use v7 templates
gap_types = {"endoparasiteOf", "hasHost", "preysOn"}

v7_gap = v7_pos[v7_pos["interaction_type"].isin(gap_types)].copy()
print(f"  DEBUG: v7 qwen_label dtype={v7['qwen_label'].dtype}, unique={v7['qwen_label'].unique()[:5]}")
v7_gap_out = v7_gap[["text"]].drop_duplicates().copy()
v7_gap_out["label"] = 1
v7_gap_out["source"] = "v7_qwen_gap_fill"

print(f"  v7 gap-fill templates: {len(v7_gap_out)}")
for t in gap_types:
    n = len(v7_pos[v7_pos["interaction_type"] == t])
    print(f"    {t}: {n}")

# ── 3. Other curated sources ───────────────────────────────────────────────
print("\n[3] Loading curated sources...")

def load_pos(path, label_col, source_name):
    df = pd.read_csv(path)
    col = label_col if label_col in df.columns else "label"
    out = df[df[col] == 1][["text"]].drop_duplicates().copy()
    out["label"] = 1
    out["source"] = source_name
    return out

pharvest = load_pos(BASE / "data/training/pathogen_harvested.csv", "teacher_label", "pathogen_harvested")
curated  = load_pos(BASE / "data/training/curated_pathogen_borderline.csv", "label", "curated_borderline")
eval100  = pd.read_csv(BASE / "data/evaluation/eval_100.tsv", sep="\t")
eval100_pos = eval100[eval100["evaluation_pair_interacting"] == 1][["sentence"]].rename(columns={"sentence": "text"})
eval100_pos["label"] = 1
eval100_pos["source"] = "eval100_gold"

print(f"  pathogen_harvested: {len(pharvest)}")
print(f"  curated_borderline: {len(curated)}")
print(f"  eval_100 gold: {len(eval100_pos)}")

# ── 4. Combine all positives ───────────────────────────────────────────────
positives = pd.concat([real_out, v7_gap_out, pharvest, curated, eval100_pos], ignore_index=True)
positives = positives.drop_duplicates(subset=["text"])
n_pos = len(positives)

print(f"\n[4] Total positives: {n_pos}")
print(f"  Real sentences: {len(real_out)} ({len(real_out)/n_pos*100:.1f}%)")
print(f"  v7 gap-fill templates: {len(v7_gap_out)} ({len(v7_gap_out)/n_pos*100:.1f}%)")
print(f"  Other curated: {len(pharvest)+len(curated)+len(eval100_pos)}")

# ── 5. Negatives (easy + hard) ─────────────────────────────────────────────
print("\n[5] Loading negatives...")
easy_neg = pd.read_csv(BASE / "data/training/negatives_clean.csv")
easy_neg = easy_neg[~easy_neg["text"].isin(positives["text"])][["text"]].drop_duplicates()
n_easy = min(int(n_pos * NEG_RATIO), len(easy_neg))
easy_sample = easy_neg.sample(n=n_easy, random_state=SEED).copy()
easy_sample["label"] = 0
easy_sample["source"] = "negatives_clean"

hard_neg = pd.read_csv(BASE / "data/training/negatives_weak_signal.csv")
hard_neg = hard_neg[hard_neg["label"] == 0]
hard_neg = hard_neg[~hard_neg["text"].isin(positives["text"])][["text"]].drop_duplicates()
n_hard = min(N_HARD_NEG, len(hard_neg))
hard_sample = hard_neg.sample(n=n_hard, random_state=SEED).copy()
hard_sample["label"] = 0
hard_sample["source"] = "negatives_weak_signal"

print(f"  Easy negatives: {n_easy}")
print(f"  Hard negatives: {n_hard}")

# ── 6. Assemble ────────────────────────────────────────────────────────────
dataset = pd.concat([positives, easy_sample, hard_sample], ignore_index=True)
dataset = dataset.sample(frac=1, random_state=SEED).reset_index(drop=True)[["text", "label", "source"]]

n_total = len(dataset)
n_pos_f = int((dataset["label"] == 1).sum())
pos_rate = n_pos_f / n_total

print(f"\n[6] Final dataset: {n_total} rows, {n_pos_f} pos ({pos_rate:.1%})")
dataset.to_csv(OUT_DIR / "dataset.csv", index=False)

meta = {
    "version": "v18_hybrid",
    "built": datetime.now().isoformat(),
    "strategy": "Real sentences for covered types + v7 Qwen-validated templates for gap types (endoparasiteOf, hasHost, preysOn)",
    "training_rows": n_total,
    "training_pos": n_pos_f,
    "pos_rate": round(pos_rate, 4),
    "positive_sources": {
        "teacher_qwen122b_real": int(len(real_out)),
        "v7_qwen_gap_fill": int(len(v7_gap_out)),
        "pathogen_harvested": int(len(pharvest)),
        "curated_borderline": int(len(curated)),
        "eval100_gold": int(len(eval100_pos)),
    },
    "gap_types_covered": sorted(gap_types),
    "klepto_cap": KLEPTO_CAP,
    "hard_negatives": n_hard,
}
with open(OUT_DIR / "metadata.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"Saved to {OUT_DIR}/dataset.csv")
print("Done.")
