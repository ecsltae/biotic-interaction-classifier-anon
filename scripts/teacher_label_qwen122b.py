#!/usr/bin/env python3
"""
Teacher labeling script using Qwen3.5-122B via ollama.
Labels sentences for biotic interaction classification.
"""

import subprocess
import sys
import re
import requests
import pandas as pd
from pathlib import Path
import time
import argparse

OLLAMA_API = "http://localhost:11434/api/generate"


def ask_teacher(sentence: str, model: str = "qwen3:32b", timeout: int = 60) -> tuple[int, str]:
    """Query a Qwen model via Ollama HTTP API for a binary biotic interaction label.

    Uses think=false to suppress extended thinking (much faster).
    Returns:
        (label, raw_response) where label is 1/0/-1 (yes/no/error)
    """
    prompt = (
        "Does this sentence describe a direct biotic interaction between two named organisms? "
        "Biotic interactions include: predation, parasitism, pollination, herbivory, mutualism, "
        "symbiosis, seed dispersal, competition, pathogen infection, or disease transmission. "
        "Answer with YES or NO only, then briefly explain.\n\n"
        f"Sentence: {sentence}"
    )

    try:
        resp = requests.post(
            OLLAMA_API,
            json={"model": model, "prompt": prompt, "stream": False, "think": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        response = resp.json().get("response", "").strip()

        # Strip any residual <think>...</think> block
        cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        if not cleaned:
            cleaned = response

        response_upper = cleaned.upper()
        if response_upper.startswith("YES") or "YES\n" in response_upper[:50]:
            return 1, cleaned
        elif response_upper.startswith("NO") or "NO\n" in response_upper[:50]:
            return 0, cleaned
        else:
            if "YES" in response_upper[:100]:
                return 1, cleaned
            elif "NO" in response_upper[:100]:
                return 0, cleaned
            return -1, cleaned

    except requests.Timeout:
        return -1, "TIMEOUT"
    except Exception as e:
        return -1, f"ERROR: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Label sentences with Qwen3.5-122B teacher")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--text-col", default="text", help="Column name for text")
    parser.add_argument("--limit", type=int, default=0, help="Max sentences to label (0=all)")
    parser.add_argument("--model", default="qwen3.5:122b", help="Ollama model name")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if output exists and append")
    args = parser.parse_args()

    # Load input
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} sentences from {args.input}")

    # Determine text column
    text_col = args.text_col
    if text_col not in df.columns:
        if "sentence" in df.columns:
            text_col = "sentence"
        else:
            print(f"Error: no '{args.text_col}' or 'sentence' column found")
            sys.exit(1)

    # Handle existing output
    start_idx = 0
    if args.skip_existing and Path(args.output).exists():
        existing = pd.read_csv(args.output)
        start_idx = len(existing)
        print(f"Resuming from index {start_idx} (found {start_idx} existing)")

    # Limit
    if args.limit > 0:
        df = df.iloc[:args.limit]

    # Skip already processed
    df = df.iloc[start_idx:]

    if len(df) == 0:
        print("No sentences to process")
        return

    print(f"Labeling {len(df)} sentences with {args.model}...")

    results = []
    for i, row in df.iterrows():
        text = str(row[text_col])

        start_time = time.time()
        label, response = ask_teacher(text, model=args.model)
        elapsed = time.time() - start_time

        results.append({
            **row.to_dict(),
            "teacher_label": label,
            "teacher_response": response[:500],  # Truncate long responses
        })

        label_str = {1: "YES", 0: "NO", -1: "ERR"}[label]
        idx = start_idx + len(results)
        print(f"  [{idx}/{start_idx + len(df)}] {label_str} ({elapsed:.1f}s): {text[:60]}...")

        # Save every 10 sentences
        if len(results) % 10 == 0:
            result_df = pd.DataFrame(results)
            if start_idx > 0 and Path(args.output).exists():
                existing = pd.read_csv(args.output)
                result_df = pd.concat([existing, result_df], ignore_index=True)
            result_df.to_csv(args.output, index=False)
            print(f"    Checkpoint saved: {len(result_df)} total")

    # Final save
    result_df = pd.DataFrame(results)
    if start_idx > 0 and Path(args.output).exists():
        existing = pd.read_csv(args.output)
        result_df = pd.concat([existing, result_df], ignore_index=True)
    result_df.to_csv(args.output, index=False)

    # Summary
    labels = result_df["teacher_label"]
    print(f"\nDone! {len(result_df)} sentences labeled:")
    print(f"  YES: {(labels == 1).sum()}")
    print(f"  NO:  {(labels == 0).sum()}")
    print(f"  ERR: {(labels == -1).sum()}")


if __name__ == "__main__":
    main()
