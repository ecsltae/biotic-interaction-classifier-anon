#!/usr/bin/env python3
"""
Track B: Recalibrate Qwen labels using EP-relax gold positives as few-shot examples.

Problem: Qwen agrees with EP-relax gold labels only 52% of the time — it's too
strict or uses different criteria than BiTeM curators.

Fix: Show Qwen 5 confirmed EP-relax positives before each query so it understands
what kind of sentence counts as a valid biotic interaction.

Input:  all_sources_qwen122b_labeled.csv (4,065 positives to re-validate)
Output: data/training/qwen_positives_recalibrated.csv
Runtime: ~3.4h at 3s/sentence
"""

import json
import time
import requests
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE = Path("/path/to/MetaP/classifier")
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:122b"
CHECKPOINT = BASE / "data/training/qwen_recalibrate_checkpoint.csv"
OUT = BASE / "data/training/qwen_positives_recalibrated.csv"

# ── 5 clear EP-relax gold positives as few-shot examples ──────────────────
# Selected manually: explicit, diverse interaction types, unambiguous
FEW_SHOT_EXAMPLES = [
    "Phytopythium litorale: A Novel Killer Pathogen of Plane (Platanus orientalis) Causing Canker Stain and Root and Collar Rot.",
    "Biological traits of Quadrastichus mendeli (Hymenoptera, Eulophidae), parasitoid of the eucalyptus gall wasp Leptocybe invasa (Hymenoptera, Eulophidae) in Thailand.",
    "Ultrastructural study of the spermatozoon of Pronoprymna ventricosa (Digenea, Baccigerinae), parasite of the twaite shad Alosa fallax Lacepede (Pisces, Teleostei).",
    "Experimental infection of ponies with Sarcocystis fayeri and differentiation from Sarcocystis neurona infections in horses.",
    "Histopathologic studies on cerebrospinal nematodiasis of moose in Minnesota naturally infected with Pneumostrongylus tenuis.",
]

EXAMPLES_TEXT = "\n".join(f"{i+1}. {s}" for i, s in enumerate(FEW_SHOT_EXAMPLES))

PROMPT_TEMPLATE = f"""Here are examples of sentences that describe a valid biotic interaction between two organisms:

{EXAMPLES_TEXT}

These sentences count as valid because they describe an actual, specific interaction occurring between named species (infection, parasitism, parasitoidism, predation, pollination, etc.).

Does the following sentence describe a biotic interaction of the same kind?
Answer YES or NO only. No explanation.

Sentence: {{sentence}}"""


def ask_qwen(sentence: str, retries: int = 3) -> str:
    prompt = PROMPT_TEMPLATE.format(sentence=sentence)
    for attempt in range(retries):
        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": MODEL,
                "think": False,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "keep_alive": -1,
                "options": {"temperature": 0, "num_predict": 10},
            }, timeout=120)
            resp.raise_for_status()
            answer = resp.json()["message"]["content"].strip().upper()
            if "YES" in answer:
                return "YES"
            elif "NO" in answer:
                return "NO"
            else:
                return "UNCLEAR"
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(30)
            else:
                print(f"  ERROR after {retries} attempts: {e}")
                return "ERROR"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--dry-run", type=int, default=0, help="Process only N rows (test mode)")
    args = parser.parse_args()

    # Load all positives
    teacher = pd.read_csv(BASE / "results/research_agent/all_sources_qwen122b_labeled.csv")
    positives = teacher[teacher["teacher_label"] == 1].copy()
    positives["text"] = positives["text"].str.strip()
    print(f"Total positives to recalibrate: {len(positives)}")

    # Resume from checkpoint — only skip rows with confirmed YES/NO (not ERROR/UNCLEAR)
    done_texts = set()
    results = []
    if args.resume and CHECKPOINT.exists():
        checkpoint = pd.read_csv(CHECKPOINT)
        confirmed = checkpoint[checkpoint["recalibrated_response"].isin(["YES", "NO"])]
        done_texts = set(confirmed["text"].tolist())
        results = confirmed.to_dict("records")
        n_errors = len(checkpoint) - len(confirmed)
        print(f"Resuming: {len(done_texts)} confirmed, {n_errors} errors/unclear will be retried")

    todo = positives[~positives["text"].isin(done_texts)]
    if args.dry_run:
        todo = todo.head(args.dry_run)
        print(f"Dry run: processing {len(todo)} rows")

    print(f"Remaining: {len(todo)}")

    # Warm up Qwen
    print("Warming up Qwen...")
    ask_qwen("test sentence")

    start = time.time()
    for i, (_, row) in enumerate(todo.iterrows()):
        answer = ask_qwen(row["text"])
        results.append({
            "text": row["text"],
            "interaction_type": row.get("interaction_type", ""),
            "_source_file": row.get("_source_file", ""),
            "original_teacher_label": 1,
            "recalibrated_label": 1 if answer == "YES" else 0,
            "recalibrated_response": answer,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (len(todo) - i - 1) / rate / 3600
            print(f"  [{i+1}/{len(todo)}] {rate:.2f} rows/s | ~{remaining:.1f}h remaining | "
                  f"YES so far: {sum(1 for r in results if r['recalibrated_label']==1)}")
            pd.DataFrame(results).to_csv(CHECKPOINT, index=False)

    # Final save
    df = pd.DataFrame(results)
    df.to_csv(OUT, index=False)
    pd.DataFrame(results).to_csv(CHECKPOINT, index=False)

    n_yes = (df["recalibrated_label"] == 1).sum()
    n_no = (df["recalibrated_label"] == 0).sum()
    n_unclear = (df["recalibrated_response"] == "UNCLEAR").sum()
    print(f"\nDone. {len(df)} sentences processed:")
    print(f"  YES (kept):    {n_yes} ({n_yes/len(df)*100:.1f}%)")
    print(f"  NO (rejected): {n_no} ({n_no/len(df)*100:.1f}%)")
    print(f"  UNCLEAR:       {n_unclear}")
    print(f"  Saved to: {OUT}")

    if "interaction_type" in df.columns:
        print("\nRecalibration rate by category:")
        for cat, grp in df.groupby("interaction_type"):
            yes = (grp["recalibrated_label"] == 1).sum()
            print(f"  {cat}: {yes}/{len(grp)} kept ({yes/len(grp)*100:.0f}%)")


if __name__ == "__main__":
    main()
