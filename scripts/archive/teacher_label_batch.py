#!/usr/bin/env python3
"""
Batched teacher labeling using HuggingFace transformers with GPU acceleration.

This is MUCH faster than ollama subprocess calls because it:
1. Uses batch processing on GPU
2. Avoids subprocess overhead
3. Can process multiple sentences in parallel

Usage:
    python classifier/scripts/teacher_label_batch.py \
        --input classifier/data/training/globi_sibils_real.csv \
        --output classifier/results/research_agent/sibils_qwen122b_labeled.csv \
        --model Qwen/Qwen2.5-72B-Instruct \
        --batch-size 4

Author: Research Agent
Date: 2026-03-26
"""

import argparse
import gc
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def create_prompt(sentence: str) -> str:
    """Create the classification prompt for the teacher model."""
    return f"""<|im_start|>system
You are a biotic interaction classifier. Answer only YES or NO.
<|im_end|>
<|im_start|>user
Does this sentence describe a direct biotic interaction between two named organisms?
Biotic interactions include: predation, parasitism, pollination, herbivory, mutualism, seed dispersal, competition, pathogen infection, vector transmission, or host-parasite relationships.

Sentence: {sentence}

Answer with YES or NO only.
<|im_end|>
<|im_start|>assistant
"""


def parse_response(response: str) -> tuple[int, str]:
    """
    Parse model response to extract YES/NO label.

    Returns:
        Tuple of (label, raw_response) where label is 1 (YES), 0 (NO), or -1 (unclear)
    """
    # Clean up response
    response = response.strip().upper()

    # Handle common patterns
    if response.startswith("YES"):
        return 1, response
    elif response.startswith("NO"):
        return 0, response
    elif "YES" in response and "NO" not in response:
        return 1, response
    elif "NO" in response:
        return 0, response
    else:
        return -1, response


def label_batch(
    model,
    tokenizer,
    sentences: list[str],
    max_new_tokens: int = 10
) -> list[tuple[int, str]]:
    """
    Label a batch of sentences using the teacher model.

    Returns:
        List of (label, raw_response) tuples
    """
    # Create prompts
    prompts = [create_prompt(s) for s in sentences]

    # Tokenize with padding
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for consistency
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode responses (only new tokens)
    results = []
    for i, output in enumerate(outputs):
        # Get only the generated tokens (after input)
        input_len = inputs.input_ids[i].shape[0]
        generated = output[input_len:]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        label, raw = parse_response(response)
        results.append((label, raw))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch teacher labeling using HuggingFace transformers"
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
        default="Qwen/Qwen2.5-72B-Instruct",
        help="HuggingFace model name (default: Qwen/Qwen2.5-72B-Instruct)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=4,
        help="Batch size for inference (default: 4)"
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
        help="Start fresh instead of resuming"
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=100,
        help="Save checkpoint every N sentences"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N sentences (for testing)"
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading input from {args.input}")
    df = pd.read_csv(args.input)

    # Handle text column name
    text_col = args.text_col
    if text_col not in df.columns:
        if "sentence" in df.columns:
            text_col = "sentence"
        else:
            print(f"Error: Could not find text column. Columns: {df.columns.tolist()}")
            return 1

    if args.limit:
        df = df.head(args.limit)

    print(f"Loaded {len(df)} sentences")

    # Check for resume
    start_idx = 0
    results = []
    if not args.no_resume and args.output.exists():
        existing = pd.read_csv(args.output)
        start_idx = len(existing)
        results = existing.to_dict('records')
        print(f"Resuming from row {start_idx}")

    if start_idx >= len(df):
        print("Already complete!")
        return 0

    # Load model with 4-bit quantization
    print(f"Loading model {args.model} with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    print(f"Model loaded. Starting labeling from row {start_idx}")
    start_time = time.time()

    # Process in batches
    batch_size = args.batch_size
    for batch_start in tqdm(range(start_idx, len(df), batch_size), desc="Labeling"):
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]

        sentences = batch_df[text_col].tolist()

        # Filter out empty/short sentences
        valid_indices = []
        valid_sentences = []
        for i, s in enumerate(sentences):
            s_str = str(s) if pd.notna(s) else ""
            if len(s_str) >= 20:
                valid_indices.append(i)
                valid_sentences.append(s_str)

        # Label valid sentences
        if valid_sentences:
            batch_results = label_batch(model, tokenizer, valid_sentences)
        else:
            batch_results = []

        # Build results for this batch
        result_idx = 0
        for i, row in enumerate(batch_df.itertuples()):
            row_dict = batch_df.iloc[i].to_dict()

            if i in valid_indices:
                label, raw = batch_results[result_idx]
                result_idx += 1
                row_dict["teacher_label"] = label
                row_dict["teacher_response"] = raw[:100] if raw else ""
            else:
                row_dict["teacher_label"] = -1
                row_dict["teacher_response"] = "SKIPPED_SHORT"

            results.append(row_dict)

        # Checkpoint
        if (batch_end % args.checkpoint == 0) or (batch_end == len(df)):
            pd.DataFrame(results).to_csv(args.output, index=False)

    # Final save
    result_df = pd.DataFrame(results)
    result_df.to_csv(args.output, index=False)

    elapsed = time.time() - start_time
    rate = (len(df) - start_idx) / elapsed if elapsed > 0 else 0

    print(f"\n=== Teacher Labeling Complete ===")
    print(f"Total sentences: {len(result_df)}")
    print(f"Time: {elapsed/60:.1f} minutes ({rate:.1f} sent/sec)")
    print(f"YES (label=1): {(result_df['teacher_label'] == 1).sum()}")
    print(f"NO (label=0): {(result_df['teacher_label'] == 0).sum()}")
    print(f"Unclear (label=-1): {(result_df['teacher_label'] == -1).sum()}")
    print(f"Output saved to: {args.output}")

    # Cleanup
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    exit(main())
