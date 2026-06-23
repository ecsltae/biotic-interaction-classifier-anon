#!/usr/bin/env python3
"""
Step 1: Run all existing evaluation sets through Qwen3.5-122B for re-validation.

- Gold labels remain authoritative (human-curated by BiTeM/SIB team)
- Qwen labels are informational — disagreements are flagged, not auto-fixed
- Output: classifier/data/evaluation/eval_sets_qwen_validated.csv
- Resumes from checkpoint — safe to kill and restart

Usage:
    python scripts/validate_eval_sets.py
    python scripts/validate_eval_sets.py --model qwen3.5:122b --dry-run
"""

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = BASE_DIR / "data/evaluation"
OUT_FILE = EVAL_DIR / "eval_sets_qwen_validated.csv"
CHECKPOINT_EVERY = 20

MODEL = "qwen3.5:122b"
OLLAMA_URL = "http://localhost:11434/api/chat"

# (path, sentence_col, label_col, species1_col, species2_col, interaction_col)
EVAL_FILES = [
    (EVAL_DIR / "eval_100.tsv",
     "sentence", "evaluation_pair_interacting", None, None, None),
    (EVAL_DIR / "globi-relax_passages-triplets_2024-02-28_curation_EP.tsv",
     "sentence", "evaluation_pair_interacting", "species1_term", "species2_term", "interaction_term"),
    (EVAL_DIR / "globi-passage_passages-triplets_2024-02-28_curation_EP.tsv",
     "sentence", "evaluation_pair_interacting", "species1_term", "species2_term", "interaction_term"),
    (EVAL_DIR / "biotx-random_passages-triplets_2024-02-28_curation_EP_100original.tsv",
     "sentence", "evaluation_pair_interacting", "species1_term", "species2_term", "interaction_term"),
    (EVAL_DIR / "biotx-random_passages-triplets_2024-04-22b_curation_EP_50best-multiples.tsv",
     "sentence", "evaluation_pair_interacting", "species1_term", "species2_term", "interaction_term"),
    (EVAL_DIR / "biotx-random_passages-triplets_2024-05-15_curation_EP_50nomultiple.tsv",
     "sentence", "evaluation_pair_interacting", "species1_term", "species2_term", "interaction_term"),
    (EVAL_DIR / "gen_set_100.csv",
     "sentence", "label", None, None, "category"),
]

PROMPT_TEMPLATE = (
    "Does this sentence describe a direct biotic interaction between two named organisms? "
    "Biotic interactions include: predation, parasitism, pollination, herbivory, mutualism, "
    "symbiosis, seed dispersal, competition, pathogen infection, or disease transmission. "
    "The sentence must describe an actual interaction occurring, not just mention organisms. "
    "Answer YES or NO only.\n\n"
    "Sentence: {sentence}"
)


def ask_qwen(sentence: str, model: str = MODEL, timeout: int = 300) -> tuple[int, str]:
    """Returns (label, raw_response): 1=YES, 0=NO, -1=unclear."""
    prompt = PROMPT_TEMPLATE.format(sentence=sentence)
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "keep_alive": -1,
                "options": {"temperature": 0, "num_predict": 10, "num_ctx": 2048},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json().get("message", {}).get("content", "").strip()
        clean = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', raw).strip().upper()
        if re.match(r'^YES\b', clean):
            return 1, raw
        elif re.match(r'^NO\b', clean):
            return 0, raw
        if 'YES' in clean[:20] and 'NO' not in clean[:20]:
            return 1, raw
        elif 'NO' in clean[:20]:
            return 0, raw
        return -1, raw
    except requests.Timeout:
        return -1, "TIMEOUT"
    except Exception as e:
        return -1, f"ERROR:{e}"


def load_eval_file(path: Path, sent_col: str, label_col: str,
                   sp1_col, sp2_col, int_col) -> pd.DataFrame:
    """Load an eval file and normalise to a common schema."""
    sep = "\t" if path.suffix == ".tsv" else ","
    df = pd.read_csv(path, sep=sep)

    # Normalise label column — some use evaluation_pair_interacting (int), some use label (0/1)
    df["gold_label"] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    df["text"] = df[sent_col].astype(str).str.strip()
    df["source_file"] = path.name
    df["species1"] = df[sp1_col].astype(str) if sp1_col and sp1_col in df.columns else ""
    df["species2"] = df[sp2_col].astype(str) if sp2_col and sp2_col in df.columns else ""
    df["interaction_term"] = df[int_col].astype(str) if int_col and int_col in df.columns else ""

    return df[["text", "gold_label", "source_file", "species1", "species2", "interaction_term"]].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dry-run", action="store_true",
                        help="Load files and show counts without calling Qwen")
    args = parser.parse_args()

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Load all eval files
    frames = []
    for path, sc, lc, s1, s2, ic in EVAL_FILES:
        if not path.exists():
            print(f"  SKIP (not found): {path.name}")
            continue
        df = load_eval_file(path, sc, lc, s1, s2, ic)
        frames.append(df)
        print(f"  Loaded {path.name}: {len(df)} rows, {df.gold_label.sum()} pos")

    all_evals = pd.concat(frames, ignore_index=True)
    # Deduplicate by text (same sentence may appear in multiple eval files)
    all_evals = all_evals.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"\nTotal unique sentences: {len(all_evals)} "
          f"({all_evals.gold_label.sum()} pos, {(all_evals.gold_label == 0).sum()} neg)")

    if args.dry_run:
        print("\n[dry-run] Stopping before Qwen calls.")
        return

    # Resume from checkpoint
    already_done: set[str] = set()
    if OUT_FILE.exists():
        done_df = pd.read_csv(OUT_FILE)
        already_done = set(done_df["text"].astype(str))
        print(f"Resuming: {len(already_done)} rows already done, "
              f"{len(all_evals) - len(already_done)} remaining")

    results: list[dict] = []
    if OUT_FILE.exists():
        results = pd.read_csv(OUT_FILE).to_dict("records")

    todo = all_evals[~all_evals["text"].isin(already_done)].reset_index(drop=True)
    n_total = len(todo)

    t0 = time.time()
    for i, row in todo.iterrows():
        qwen_label, qwen_response = ask_qwen(row["text"], model=args.model)

        results.append({
            "text": row["text"],
            "gold_label": row["gold_label"],
            "qwen_label": qwen_label,
            "source_file": row["source_file"],
            "species1": row["species1"],
            "species2": row["species2"],
            "interaction_term": row["interaction_term"],
            "qwen_response": qwen_response[:200],
        })

        if (i + 1) % CHECKPOINT_EVERY == 0 or i == n_total - 1:
            pd.DataFrame(results).to_csv(OUT_FILE, index=False)
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta_min = (n_total - i - 1) / rate / 60 if rate > 0 else 0
            print(f"  [{i+1}/{n_total}] {rate:.1f} rows/s | ETA {eta_min:.0f} min", flush=True)

    # Final report
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUT_FILE, index=False)

    print(f"\n=== DONE — {len(df_out)} sentences validated ===")
    print(f"Output: {OUT_FILE}")

    # Agreement analysis
    valid = df_out[df_out.qwen_label.isin([0, 1])]
    agree = (valid.gold_label == valid.qwen_label).sum()
    print(f"\nAgreement: {agree}/{len(valid)} ({100*agree/len(valid):.1f}%)")

    # Disagreements worth reviewing
    fn = valid[(valid.gold_label == 1) & (valid.qwen_label == 0)]  # gold=pos, Qwen=neg
    fp = valid[(valid.gold_label == 0) & (valid.qwen_label == 1)]  # gold=neg, Qwen=pos
    print(f"\nDisagreements requiring review:")
    print(f"  Gold=POS but Qwen=NEG (potential true positives Qwen missed): {len(fn)}")
    print(f"  Gold=NEG but Qwen=POS (potential gold label errors): {len(fp)}")

    if len(fn) > 0:
        print("\n--- Gold=POS / Qwen=NEG (Qwen may be too strict on these) ---")
        for _, r in fn.iterrows():
            print(f"  [{r.source_file}] {r.text[:120]}")

    if len(fp) > 0:
        print("\n--- Gold=NEG / Qwen=POS (check if gold label is wrong) ---")
        for _, r in fp.iterrows():
            print(f"  [{r.source_file}] {r.text[:120]}")


if __name__ == "__main__":
    main()
