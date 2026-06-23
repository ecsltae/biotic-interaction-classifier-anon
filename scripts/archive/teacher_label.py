#!/usr/bin/env python3
"""
Teacher labeling script using a large local LLM (Qwen3 via ollama) to label
biotic interaction sentences for training smaller student models.

The teacher model provides high-quality binary labels that can be used to train
BiomedBERT and FLAN-T5-base (the student models).

Usage:
    python classifier/scripts/teacher_label.py \
        --input classifier/data/training/globi_sibils_real.csv \
        --output classifier/results/research_agent/sibils_qwen_labeled.csv \
        --model qwen3:32b \
        --batch-size 1

Author: Research Agent
Date: 2026-03-26
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def ask_teacher(sentence: str, model: str = "qwen3:32b", timeout: int = 120) -> tuple[str, str]:
    """
    Query the teacher LLM via ollama for a binary biotic interaction label.

    Args:
        sentence: The sentence to classify
        model: Ollama model name
        timeout: Max seconds to wait for response

    Returns:
        Tuple of (label, raw_response) where label is "YES", "NO", or "ERROR"
    """
    prompt = (
        "Does this sentence describe a direct biotic interaction between two named organisms? "
        "Biotic interactions include: predation, parasitism, pollination, herbivory, mutualism, "
        "seed dispersal, competition, pathogen infection, vector transmission. "
        "Answer with YES or NO only, nothing else.\n\n"
        f"Sentence: {sentence}"
    )

    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        raw = result.stdout.strip()

        # Parse YES/NO from response
        upper = raw.upper()
        if "YES" in upper and "NO" not in upper:
            return "YES", raw
        elif "NO" in upper:
            return "NO", raw
        else:
            # Ambiguous - try to infer from response
            if any(word in upper for word in ["DESCRIBES", "DIRECT", "INTERACTION", "BETWEEN"]):
                return "YES", raw
            else:
                return "UNCLEAR", raw

    except subprocess.TimeoutExpired:
        return "ERROR", "TIMEOUT"
    except Exception as e:
        return "ERROR", str(e)


def label_dataset(
    input_path: Path,
    output_path: Path,
    model: str = "qwen3:32b",
    text_col: str = "text",
    resume: bool = True,
    checkpoint_every: int = 100
) -> pd.DataFrame:
    """
    Label an entire dataset using the teacher model.

    Args:
        input_path: Path to input CSV
        output_path: Path to save labeled output
        model: Ollama model name
        text_col: Name of the text column
        resume: If True, resume from existing output file
        checkpoint_every: Save checkpoint every N sentences

    Returns:
        DataFrame with teacher labels
    """
    # Load input data
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} sentences from {input_path}")

    # Initialize or resume
    if resume and output_path.exists():
        existing = pd.read_csv(output_path)
        start_idx = len(existing)
        print(f"Resuming from row {start_idx} (found existing output)")
        results = existing.to_dict('records')
    else:
        start_idx = 0
        results = []

    # Process remaining sentences
    for i in tqdm(range(start_idx, len(df)), desc="Teacher labeling", initial=start_idx, total=len(df)):
        row = df.iloc[i]
        text = str(row.get(text_col, row.get("sentence", "")))

        if not text or len(text) < 10:
            results.append({
                **row.to_dict(),
                "teacher_label": -1,
                "teacher_response": "SKIPPED_SHORT",
                "teacher_raw": ""
            })
            continue

        label_str, raw = ask_teacher(text, model=model)

        # Convert to numeric label
        if label_str == "YES":
            teacher_label = 1
        elif label_str == "NO":
            teacher_label = 0
        else:
            teacher_label = -1  # Unclear/Error - needs review

        results.append({
            **row.to_dict(),
            "teacher_label": teacher_label,
            "teacher_response": label_str,
            "teacher_raw": raw[:200] if raw else ""  # Truncate raw response
        })

        # Checkpoint
        if (i + 1) % checkpoint_every == 0:
            pd.DataFrame(results).to_csv(output_path, index=False)
            print(f"  Checkpoint saved at row {i + 1}")

    # Final save
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)
    print(f"Saved {len(result_df)} labeled sentences to {output_path}")

    # Print statistics
    print("\n=== Teacher Labeling Statistics ===")
    print(f"Total sentences: {len(result_df)}")
    print(f"Teacher YES (label=1): {(result_df['teacher_label'] == 1).sum()}")
    print(f"Teacher NO (label=0): {(result_df['teacher_label'] == 0).sum()}")
    print(f"Unclear/Error (label=-1): {(result_df['teacher_label'] == -1).sum()}")

    if 'label' in result_df.columns:
        # Compare with original labels
        mask = result_df['teacher_label'] >= 0
        original = result_df.loc[mask, 'label']
        teacher = result_df.loc[mask, 'teacher_label']
        agreement = (original == teacher).mean()
        print(f"\nAgreement with original labels: {agreement:.1%}")

        # Confusion breakdown
        orig_pos_teach_pos = ((original == 1) & (teacher == 1)).sum()
        orig_pos_teach_neg = ((original == 1) & (teacher == 0)).sum()
        orig_neg_teach_pos = ((original == 0) & (teacher == 1)).sum()
        orig_neg_teach_neg = ((original == 0) & (teacher == 0)).sum()

        print(f"Original=1, Teacher=1: {orig_pos_teach_pos}")
        print(f"Original=1, Teacher=0: {orig_pos_teach_neg} (teacher downgraded)")
        print(f"Original=0, Teacher=1: {orig_neg_teach_pos} (teacher upgraded)")
        print(f"Original=0, Teacher=0: {orig_neg_teach_neg}")

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Label sentences using a large LLM teacher model"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input CSV file with sentences to label"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output CSV file for labeled data"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="qwen3:32b",
        help="Ollama model name (default: qwen3:32b)"
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="text",
        help="Name of the text column (default: text)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh instead of resuming from existing output"
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=100,
        help="Save checkpoint every N sentences (default: 100)"
    )

    args = parser.parse_args()

    # Validate input exists
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Check ollama is available
    try:
        subprocess.run(["ollama", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ollama is not installed or not in PATH", file=sys.stderr)
        sys.exit(1)

    # Check model is available
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if args.model not in result.stdout:
        print(f"Warning: Model '{args.model}' may not be available. Available models:")
        print(result.stdout)
        print(f"\nAttempting to use {args.model} anyway...")

    # Run labeling
    start_time = time.time()
    label_dataset(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        text_col=args.text_col,
        resume=not args.no_resume,
        checkpoint_every=args.checkpoint
    )
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
