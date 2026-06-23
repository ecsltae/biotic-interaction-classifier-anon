#!/usr/bin/env python3
"""
Full-corpus teacher labeling with Qwen3.5-122B via ollama HTTP API.

- Uses ollama REST API (not subprocess) → ~3-8s/row instead of ~60s
- /no_think prefix disables Qwen3 chain-of-thought for fast YES/NO
- Resumes from checkpoint — safe to kill and restart
- Labels ALL source files into a single merged output

Usage:
    python scripts/teacher_label_full.py
    python scripts/teacher_label_full.py --model qwen3.5:122b --batch-report 100
"""

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path('/path/to/MetaP/classifier')
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3.5:122b"

# All source files to label — ordered by priority
SOURCE_FILES = [
    BASE_DIR / 'data/training/globi_sibils_real.csv',
    BASE_DIR / 'data/training/epmc_direct_sentences.csv',
    BASE_DIR / 'data/training/epmc_direct_sentences_v2.csv',
    BASE_DIR / 'data/training/globi_pmc_real_sentences.csv',
    BASE_DIR / 'data/training/globi_pmc_sentences_v2.csv',
    BASE_DIR / 'data/training/external_db_sentences.csv',
]

OUT_FILE = BASE_DIR / 'results/research_agent/all_sources_qwen122b_labeled.csv'
CHECKPOINT_EVERY = 50  # save progress every N rows


def get_text_col(df: pd.DataFrame) -> str:
    for col in ('text', 'sentence'):
        if col in df.columns:
            return col
    raise ValueError(f"No text column found. Columns: {list(df.columns)}")


def ask_teacher(sentence: str, model: str = MODEL, timeout: int = 60) -> tuple[int, str]:
    """Query model via ollama chat API. Returns (label, raw_response).

    Uses chat API (not generate) for correct Qwen3 template handling.
    think=False disables chain-of-thought for fast YES/NO (~1-2s/sentence).
    """
    prompt = (
        "Does this sentence describe a direct biotic interaction between two named organisms? "
        "Biotic interactions include: predation, parasitism, pollination, herbivory, mutualism, "
        "symbiosis, seed dispersal, competition, pathogen infection, or disease transmission. "
        "The sentence must describe an actual interaction occurring, not just mention organisms. "
        "Answer YES or NO only.\n\n"
        f"Sentence: {sentence}"
    )
    try:
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_predict": 10, "num_ctx": 2048},
            },
            timeout=timeout
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--batch-report', type=int, default=100,
                        help='Print progress every N rows')
    args = parser.parse_args()

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Load all source files
    dfs = []
    for f in SOURCE_FILES:
        if not f.exists():
            print(f"  SKIP (not found): {f.name}")
            continue
        df = pd.read_csv(f)
        tcol = get_text_col(df)
        df = df.rename(columns={tcol: 'text'})
        df['_source_file'] = f.name
        dfs.append(df[['text', '_source_file'] + [c for c in df.columns
                        if c not in ('text', '_source_file')]])
        print(f"  Loaded {len(df):>6} rows from {f.name}")

    all_data = pd.concat(dfs, ignore_index=True)
    all_data = all_data.drop_duplicates(subset='text')
    print(f"\nTotal unique sentences: {len(all_data)}")

    # Resume from checkpoint
    already_done = set()
    results = []
    if OUT_FILE.exists():
        done_df = pd.read_csv(OUT_FILE)
        already_done = set(done_df['text'].tolist())
        results = done_df.to_dict('records')
        print(f"Resuming: {len(already_done)} already labeled")

    todo = all_data[~all_data['text'].isin(already_done)]
    print(f"Remaining: {len(todo)} rows to label")
    if len(todo) == 0:
        print("All done!")
        return

    # Estimate time
    print(f"\nEstimated time at 5s/row: {len(todo)*5/3600:.1f}h  |  at 10s/row: {len(todo)*10/3600:.1f}h")
    print(f"Output: {OUT_FILE}\n")

    t_start = time.time()
    n_done = 0

    for _, row in todo.iterrows():
        text = str(row['text'])
        label, raw = ask_teacher(text, model=args.model)

        record = row.to_dict()
        record['teacher_label'] = label
        record['teacher_response'] = raw[:200]
        results.append(record)
        n_done += 1

        if n_done % CHECKPOINT_EVERY == 0:
            pd.DataFrame(results).to_csv(OUT_FILE, index=False)
            elapsed = time.time() - t_start
            rate = elapsed / n_done
            remaining_h = (len(todo) - n_done) * rate / 3600
            pos_rate = sum(1 for r in results[-CHECKPOINT_EVERY:] if r.get('teacher_label') == 1)
            print(f"  [{n_done}/{len(todo)}] {rate:.1f}s/row | "
                  f"ETA {remaining_h:.1f}h | "
                  f"last {CHECKPOINT_EVERY}: {pos_rate} pos")

        if n_done % args.batch_report == 0:
            total_pos = sum(1 for r in results if r.get('teacher_label') == 1)
            total_neg = sum(1 for r in results if r.get('teacher_label') == 0)
            total_unk = sum(1 for r in results if r.get('teacher_label') == -1)
            print(f"  Progress: {len(results)} labeled | "
                  f"{total_pos} YES / {total_neg} NO / {total_unk} unclear")

    # Final save
    pd.DataFrame(results).to_csv(OUT_FILE, index=False)

    total_pos = sum(1 for r in results if r.get('teacher_label') == 1)
    total_neg = sum(1 for r in results if r.get('teacher_label') == 0)
    total_unk = sum(1 for r in results if r.get('teacher_label') == -1)
    elapsed_h = (time.time() - t_start) / 3600

    print(f"\n=== DONE ===")
    print(f"Total labeled: {len(results)}")
    print(f"  YES (positive): {total_pos}")
    print(f"  NO (negative):  {total_neg}")
    print(f"  Unclear:        {total_unk}")
    print(f"Time: {elapsed_h:.1f}h  ({elapsed_h*60/len(todo):.1f} min/row avg)")
    print(f"Saved → {OUT_FILE}")


if __name__ == '__main__':
    main()
