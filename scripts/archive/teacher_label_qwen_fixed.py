#!/usr/bin/env python3
"""
Teacher labeling script using Qwen3.5-122B via ollama.
Fixed version: properly handles Qwen3's "Thinking..." chain-of-thought output.
"""

import subprocess
import sys
import pandas as pd
from pathlib import Path
import time
import argparse
import re


def parse_thinking_response(response: str) -> tuple[int, str]:
    """Parse Qwen3 response that may include thinking mode output.

    Qwen3 outputs format:
    Thinking...
    [chain of thought reasoning]
    ...done thinking.

    YES/NO
    [optional explanation]

    Returns:
        (label, clean_answer) where label is 1/0/-1 (yes/no/error)
    """
    if not response:
        return -1, "EMPTY"

    # Remove ANSI escape codes if present
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    response = ansi_escape.sub('', response)

    # Check for "done thinking" marker and get text after it
    done_marker = "...done thinking."
    if done_marker in response:
        # Get everything after the thinking section
        after_thinking = response.split(done_marker)[-1].strip()
        response_upper = after_thinking.upper()
    else:
        # No thinking mode, use full response
        response_upper = response.upper().strip()

    # Look for YES/NO at the start of the final answer
    # Common patterns: "YES", "YES.", "YES\n", "**YES**"
    yes_patterns = [r'^YES\b', r'^\*\*YES\*\*', r'^YES[.\n]']
    no_patterns = [r'^NO\b', r'^\*\*NO\*\*', r'^NO[.\n]']

    for pattern in yes_patterns:
        if re.match(pattern, response_upper):
            return 1, response_upper[:100]

    for pattern in no_patterns:
        if re.match(pattern, response_upper):
            return 0, response_upper[:100]

    # Fallback: look for YES/NO anywhere in first 50 chars of final answer
    first_50 = response_upper[:50]
    if "YES" in first_50 and "NO" not in first_50:
        return 1, response_upper[:100]
    elif "NO" in first_50 and "YES" not in first_50:
        return 0, response_upper[:100]

    # Couldn't parse
    return -1, response[:200]


def ask_teacher(sentence: str, model: str = "qwen3.5:122b", timeout: int = 300) -> tuple[int, str]:
    """Query Qwen3.5-122B via ollama for a binary biotic interaction label.

    Returns:
        (label, raw_response) where label is 1/0/-1 (yes/no/error)
    """
    # Use /no_think flag to disable thinking mode if supported
    # But also handle parsing if thinking is enabled
    prompt = (
        "Does this sentence describe a direct biotic interaction between two named organisms? "
        "Biotic interactions include: predation, parasitism, pollination, herbivory, mutualism, "
        "symbiosis, seed dispersal, competition, pathogen infection, or disease transmission. "
        "The sentence must describe an actual interaction happening, not just mention organism names. "
        "Answer with YES or NO only, then briefly explain.\n\n"
        f"Sentence: {sentence}"
    )

    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        response = result.stdout.strip()

        label, parsed = parse_thinking_response(response)
        return label, response

    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"
    except Exception as e:
        return -1, f"ERROR: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Label sentences with Qwen3.5-122B teacher (fixed)")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--text-col", default="text", help="Column name for text")
    parser.add_argument("--limit", type=int, default=0, help="Max sentences to label (0=all)")
    parser.add_argument("--model", default="qwen3.5:122b", help="Ollama model name")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    args = parser.parse_args()

    # Load input
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} sentences from {args.input}")

    # Determine text column
    text_col = args.text_col
    if text_col not in df.columns:
        if "sentence" in df.columns:
            text_col = "sentence"
        elif "text" in df.columns:
            text_col = "text"
        else:
            print(f"Error: no '{args.text_col}', 'sentence', or 'text' column found")
            sys.exit(1)

    # Handle resume
    start_idx = 0
    existing_df = None
    if args.resume and Path(args.output).exists():
        existing_df = pd.read_csv(args.output)
        start_idx = len(existing_df)
        print(f"Resuming from index {start_idx} (found {start_idx} existing)")

    # Limit
    end_idx = len(df) if args.limit <= 0 else min(args.limit, len(df))

    if start_idx >= end_idx:
        print("All sentences already processed")
        return

    print(f"Labeling sentences {start_idx} to {end_idx} with {args.model}...")

    results = []
    for idx in range(start_idx, end_idx):
        row = df.iloc[idx]
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
        print(f"  [{idx+1}/{end_idx}] {label_str} ({elapsed:.1f}s): {text[:60]}...")

        # Save checkpoint every 10 sentences
        if len(results) % 10 == 0:
            save_results(results, existing_df, args.output)
            print(f"    Checkpoint saved: {start_idx + len(results)} total")

    # Final save
    save_results(results, existing_df, args.output)

    # Summary
    all_labels = pd.read_csv(args.output)["teacher_label"]
    print(f"\nDone! {len(all_labels)} total sentences labeled:")
    print(f"  YES: {(all_labels == 1).sum()}")
    print(f"  NO:  {(all_labels == 0).sum()}")
    print(f"  ERR: {(all_labels == -1).sum()}")


def save_results(results: list, existing_df: pd.DataFrame | None, output_path: str):
    """Save results, optionally merging with existing data."""
    result_df = pd.DataFrame(results)
    if existing_df is not None:
        result_df = pd.concat([existing_df, result_df], ignore_index=True)
    result_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
