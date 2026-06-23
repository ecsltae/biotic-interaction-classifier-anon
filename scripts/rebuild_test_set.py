#!/usr/bin/env python3
"""
Rebuild the 500-sentence test set from raw sources, fixing the duplication bug:
- EP-A has 1 internal duplicate (same sentence/pair/label twice) -> 99 unique.
- eval-100 and BioTx-random are 96% the same underlying content exported in
  two formats -> merged into one 104-unique-sentence group (96 shared + 4 + 4).
Result: 99 (EP-A) + 100 (EP-passage) + 100 (gen-set-100) + 104 (eval/BioTx merged) = 403 rows.

Usage:
    python classifier/scripts/rebuild_test_set.py
"""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "classifier/data/evaluation"
OUT = EVAL_DIR / "biotic_interaction_test_set.csv"
BACKUP = EVAL_DIR / "biotic_interaction_test_set.csv.bak_pre_dedup"


def normalize(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def main():
    # Backup the old (corrupted) file first.
    if OUT.exists() and not BACKUP.exists():
        import shutil
        shutil.copy(OUT, BACKUP)
        print(f"Backed up old test set -> {BACKUP.name}")

    rows = []

    # EP-A: dedupe internal duplicate, keep first occurrence.
    ep_a = pd.read_csv(EVAL_DIR / "globi-relax_passages-triplets_2024-02-28_curation_EP.tsv", sep="\t")
    ep_a["norm"] = ep_a["sentence"].astype(str).map(normalize)
    ep_a = ep_a.drop_duplicates("norm", keep="first")
    assert len(ep_a) == 99, f"EP-A expected 99, got {len(ep_a)}"
    for _, r in ep_a.iterrows():
        rows.append({"sentence": r["sentence"], "label": int(r["evaluation_pair_interacting"]), "source": "EP-A"})

    # EP-passage: no internal dups.
    ep_p = pd.read_csv(EVAL_DIR / "globi-passage_passages-triplets_2024-02-28_curation_EP.tsv", sep="\t")
    assert len(ep_p) == 100
    for _, r in ep_p.iterrows():
        rows.append({"sentence": r["sentence"], "label": int(r["evaluation_pair_interacting"]), "source": "EP-passage"})

    # gen-set-100: no internal dups.
    gen100 = pd.read_csv(EVAL_DIR / "gen_set_100.csv")
    assert len(gen100) == 100
    for _, r in gen100.iterrows():
        rows.append({"sentence": r["sentence"], "label": int(r["label"]), "source": "gen-set-100"})

    # eval-100 + BioTx-random: merge into one group, dedupe by normalized text,
    # prefer BioTx-random's row (richer metadata, properly cased/punctuated)
    # when the same content exists in both.
    eval100 = pd.read_csv(EVAL_DIR / "eval_100.tsv", sep="\t")
    eval100["norm"] = eval100["sentence"].astype(str).map(normalize)
    biotx = pd.read_csv(EVAL_DIR / "biotx-random_passages-triplets_2024-02-28_curation_EP_100original.tsv", sep="\t")
    biotx["norm"] = biotx["sentence"].astype(str).map(normalize)

    merged = {}
    for _, r in eval100.iterrows():
        merged[r["norm"]] = {"sentence": r["sentence"], "label": int(r["evaluation_pair_interacting"])}
    for _, r in biotx.iterrows():
        # BioTx-random overwrites eval-100 on overlap (richer metadata, preferred casing)
        merged[r["norm"]] = {"sentence": r["sentence"], "label": int(r["evaluation_pair_interacting"])}

    assert len(merged) == 104, f"merged eval-100/BioTx-random expected 104, got {len(merged)}"
    for v in merged.values():
        rows.append({"sentence": v["sentence"], "label": v["label"], "source": "eval-100/BioTx-random"})

    df = pd.DataFrame(rows)
    assert len(df) == 403, f"expected 403 total rows, got {len(df)}"

    # Final cross-source duplicate check.
    df["norm"] = df["sentence"].astype(str).map(normalize)
    n_cross_dup = df["norm"].duplicated().sum()
    if n_cross_dup:
        print(f"WARNING: {n_cross_dup} cross-source duplicates found after merge!")
        print(df[df.duplicated("norm", keep=False)][["sentence", "source"]])
    df = df.drop(columns=["norm"])

    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} rows -> {OUT}")
    print(df["source"].value_counts())
    print(f"Positives: {df['label'].sum()} ({100*df['label'].mean():.1f}%)")


if __name__ == "__main__":
    main()
