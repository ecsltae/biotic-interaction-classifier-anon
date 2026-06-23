#!/usr/bin/env python3
"""
Validate v7 non-pathogenOf positives through Qwen3.5-122B.

Reads training_data_globi_v7_llm_cleaned.csv (label=1, excl. pathogenOf),
asks Qwen YES/NO for each sentence, saves accepted ones.

Output: classifier/data/training/v7_non_pathogen_qwen_validated.csv

Usage:
    python scripts/validate_v7_with_qwen.py
    python scripts/validate_v7_with_qwen.py --resume   # skip already-done rows
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data/training"
INPUT_FILE = DATA_DIR / "training_data_globi_v7_llm_cleaned.csv"
OUTPUT_FILE = DATA_DIR / "v7_non_pathogen_qwen_validated.csv"
CHECKPOINT  = DATA_DIR / "v7_qwen_checkpoint.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "qwen3.5:122b"
TIMEOUT    = 300

PROMPT_TEMPLATE = """You are an expert biologist. Does the following sentence describe a direct biotic interaction between two species (e.g. predation, parasitism, pollination, symbiosis, herbivory)?

Answer with only YES or NO.

Sentence: {sentence}"""


def ask_qwen(text: str) -> str | None:
    payload = {
        "model": MODEL,
        "prompt": PROMPT_TEMPLATE.format(sentence=text),
        "stream": False,
        "think": False,
        "keep_alive": -1,
        "options": {"temperature": 0, "num_predict": 10},
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        resp = r.json().get("response", "").strip().upper()
        if resp.startswith("YES"):
            return "YES"
        if resp.startswith("NO"):
            return "NO"
        return None
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Skip already-validated rows")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    args = parser.parse_args()

    df = pd.read_csv(INPUT_FILE)
    df = df[(df["label"] == 1) & (df["interaction_type"] != "pathogenOf")].copy()
    df = df.reset_index(drop=True)
    print(f"Loaded {len(df)} v7 non-pathogenOf positives")

    # Resume from checkpoint
    done_texts: set[str] = set()
    results: list[dict] = []
    if args.resume and CHECKPOINT.exists():
        ck = pd.read_csv(CHECKPOINT)
        done_texts = set(ck["text"].astype(str).str.strip())
        results = ck.to_dict("records")
        print(f"Resuming — {len(done_texts)} already done")

    pending = df[~df["text"].astype(str).str.strip().isin(done_texts)]
    print(f"Pending: {len(pending)}")

    yes_count = sum(1 for r in results if r.get("qwen_label") == 1)

    for i, (_, row) in enumerate(pending.iterrows()):
        text = str(row["text"]).strip()
        resp = ask_qwen(text)
        label = 1 if resp == "YES" else 0

        if label == 1:
            yes_count += 1

        results.append({
            "text":             text,
            "label":            label,
            "qwen_label":       label,
            "qwen_response":    resp or "",
            "interaction_type": row.get("interaction_type", ""),
            "source_species":   row.get("source_species", ""),
            "target_species":   row.get("target_species", ""),
        })

        if (i + 1) % 10 == 0:
            done = len(done_texts) + i + 1
            total = len(df)
            pct = yes_count / done * 100
            print(f"  [{done}/{total}] YES so far: {yes_count} ({pct:.1f}%)")

        if (i + 1) % args.checkpoint_every == 0:
            pd.DataFrame(results).to_csv(CHECKPOINT, index=False)

    # Final save
    result_df = pd.DataFrame(results)
    result_df.to_csv(CHECKPOINT, index=False)

    accepted = result_df[result_df["qwen_label"] == 1].copy()
    accepted.to_csv(OUTPUT_FILE, index=False)

    total = len(result_df)
    n_yes = len(accepted)
    print(f"\n=== Done ===")
    print(f"  Total validated: {total}")
    print(f"  Accepted (YES):  {n_yes} ({100*n_yes/total:.1f}%)")
    print(f"  Rejected (NO):   {total - n_yes}")
    print(f"  Saved to: {OUTPUT_FILE}")
    print(f"\nNext: re-run assemble_v15_dataset.py to include these")


if __name__ == "__main__":
    main()
