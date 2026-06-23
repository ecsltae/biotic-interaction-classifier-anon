#!/usr/bin/env python3
"""
Fast teacher labeling using ollama Python API.

Uses Qwen3.5:122b (or qwen3:32b) loaded in ollama for binary biotic interaction classification.
The Python API is faster than subprocess calls due to connection reuse.

Usage:
    python classifier/scripts/teacher_label_ollama.py \
        --input classifier/results/research_agent/sibils_weak_cats_500.csv \
        --output classifier/results/research_agent/sibils_weak_qwen_labeled.csv \
        --model qwen3:32b

Author: Research Agent
Date: 2026-03-26
"""

import argparse
import time
from pathlib import Path

import ollama
import pandas as pd
from tqdm import tqdm


PROMPT_TEMPLATE = """Does this sentence describe a direct biotic interaction between two named organisms?

Biotic interactions include: predation, parasitism, pollination, herbivory, mutualism, seed dispersal, competition, pathogen infection, vector transmission, or host-parasite relationships.

Answer with YES or NO only. Do not explain.

Sentence: {text}"""


def ask_teacher(text: str, model: str = "qwen3:32b") -> tuple[int, str, float]:
    """
    Query the teacher model for a binary label.

    Returns:
        Tuple of (label, raw_response, inference_time)
        label: 1 (YES), 0 (NO), -1 (unclear)
    """
    prompt = PROMPT_TEMPLATE.format(text=text)

    start = time.time()
    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": 0.0,  # Deterministic
                "num_predict": 20,   # Short response
            }
        )
        elapsed = time.time() - start
        raw = response["response"].strip()
    except Exception as e:
        return -1, f"ERROR: {e}", time.time() - start

    # Parse YES/NO
    upper = raw.upper()
    if upper.startswith("YES") or (upper.startswith("/NO THINK") is False and "YES" in upper[:20] and "NO" not in upper[:20]):
        # Qwen3 has thinking mode - check for /no_think prefix
        label = 1
    elif upper.startswith("NO") or "NO" in upper[:20]:
        label = 0
    elif "YES" in upper and "NO" not in upper:
        label = 1
    else:
        label = -1

    return label, raw[:100], elapsed


def main():
    parser = argparse.ArgumentParser(description="Teacher labeling with ollama")
    parser.add_argument("--input", required=True, help="Input CSV with 'text' column")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--model", default="qwen3:32b", help="Ollama model name")
    parser.add_argument("--text-col", default="text", help="Column name for text")
    parser.add_argument("--limit", type=int, default=0, help="Limit rows (0=all)")
    parser.add_argument("--checkpoint", type=int, default=50, help="Checkpoint every N rows")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)
    text_col = args.text_col
    if text_col not in df.columns and "sentence" in df.columns:
        text_col = "sentence"

    if args.limit > 0:
        df = df.head(args.limit)

    print(f"Loaded {len(df)} sentences from {args.input}")
    print(f"Using model: {args.model}")

    # Check for resume
    output_path = Path(args.output)
    if output_path.exists():
        existing = pd.read_csv(output_path)
        start_idx = len(existing)
        print(f"Resuming from row {start_idx}")
        results = existing.to_dict("records")
    else:
        start_idx = 0
        results = []

    if start_idx >= len(df):
        print("Already complete!")
        return

    # Warmup query
    print("Warming up model...")
    _, _, warmup_time = ask_teacher("The lion hunted the zebra.", args.model)
    print(f"Warmup done ({warmup_time:.1f}s)")

    # Process sentences
    times = []
    for i in tqdm(range(start_idx, len(df)), desc="Teacher labeling"):
        row = df.iloc[i]
        text = str(row[text_col]) if pd.notna(row[text_col]) else ""

        if len(text) < 20:
            label, raw, elapsed = -1, "SKIPPED_SHORT", 0.0
        else:
            label, raw, elapsed = ask_teacher(text, args.model)

        times.append(elapsed)

        result = row.to_dict()
        result["teacher_label"] = label
        result["teacher_response"] = raw
        results.append(result)

        # Checkpoint
        if (i + 1) % args.checkpoint == 0:
            pd.DataFrame(results).to_csv(args.output, index=False)
            avg_time = sum(times[-50:]) / min(50, len(times[-50:]))
            remaining = len(df) - i - 1
            eta = remaining * avg_time / 60
            print(f"\n  Checkpoint at {i+1}, avg {avg_time:.1f}s/sent, ETA {eta:.0f}min")

    # Final save
    result_df = pd.DataFrame(results)
    result_df.to_csv(args.output, index=False)

    # Stats
    avg_time = sum(times) / len(times) if times else 0
    print(f"\n=== Teacher Labeling Complete ===")
    print(f"Total sentences: {len(result_df)}")
    print(f"Avg time: {avg_time:.1f}s/sentence")
    print(f"YES (label=1): {(result_df['teacher_label'] == 1).sum()}")
    print(f"NO (label=0): {(result_df['teacher_label'] == 0).sum()}")
    print(f"Unclear (label=-1): {(result_df['teacher_label'] == -1).sum()}")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
