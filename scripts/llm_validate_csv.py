#!/usr/bin/env python3
"""
LLM validation of harvest CSV files.

Runs the existing LLMValidator (claude-haiku, batch_size=10) on all positive
rows of a harvest CSV and adds a `llm_validated` boolean column.

Only positives (label==1) are sent to the LLM — negatives are already quality-
filtered by _is_false_negative() in the build scripts and pass through as True.

Usage:
    python llm_validate_csv.py \\
        --input  data/training/epmc_direct_sentences.csv \\
        --output data/training/epmc_direct_llm_validated.csv

    # Validate a small test batch first (dry-run on 20 positives):
    python llm_validate_csv.py \\
        --input  data/training/external_db_sentences.csv \\
        --output /tmp/extdb_test.csv \\
        --max-positives 20

Expected pass rates (positives that survive LLM validation):
    epmc_direct_sentences.csv  : ~67-80%  (1,124 positives)
    external_db_sentences.csv  : ~52-67%  (134 positives, Mangal/SiBILS is noisy)
    globi_pmc_sentences_v2.csv : ~79-89%  (140 positives)
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# Allow running from project root or from classifier/ subdir
for _candidate in [
    Path(__file__).parent.parent / "src",    # classifier/src/
    Path(__file__).parent / "src",           # if run from classifier/
]:
    if (_candidate / "data" / "llm_validator.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from data.llm_validator import LLMValidator  # noqa: E402


def validate_csv(
    input_path: str,
    output_path: str,
    text_col: str = "text",
    label_col: str = "label",
    batch_size: int = 10,
    max_positives: int = None,
    delay: float = 0.5,
    api_key: str = None,
) -> pd.DataFrame:
    """
    Validate positives in a harvest CSV using LLM and save results.

    Adds column `llm_validated` (bool):
      - label==1 rows: True if LLM says YES (is an interaction)
      - label==0 rows: True (negatives not sent to LLM)

    Args:
        input_path:     Input harvest CSV.
        output_path:    Output path (new file, never overwrites input).
        text_col:       Column containing sentence text.
        label_col:      Column containing 0/1 labels.
        batch_size:     Sentences per LLM API call (default 10).
        max_positives:  Cap on positives to validate (None = all). Useful for dry-runs.
        delay:          Seconds to sleep between API batches (rate limiting).

    Returns:
        DataFrame with added `llm_validated` column.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if output_path.resolve() == input_path.resolve():
        raise ValueError("Output path must differ from input path — never overwrite source data.")

    # Load
    df = pd.read_csv(input_path)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not in {list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not in {list(df.columns)}")

    # All rows start as validated=True; only positives get LLM check
    df = df.copy()
    df["llm_validated"] = True

    positives = df[df[label_col] == 1].copy()
    total_pos = len(positives)

    if max_positives and max_positives < total_pos:
        positives = positives.sample(n=max_positives, random_state=42)
        print(f"Dry-run mode: validating {max_positives}/{total_pos} positives")
    else:
        print(f"Validating all {total_pos} positives...")

    print(f"Input : {input_path.name}  ({len(df)} rows, {total_pos} positives)")
    print(f"Output: {output_path}")
    print(f"Batch : {batch_size} sentences/call, {delay}s delay\n")

    validator = LLMValidator(batch_size=batch_size, api_key=api_key)

    if not validator.client:
        print("WARNING: No Anthropic API key found. Set ANTHROPIC_API_KEY env var.")
        print("All positives will be marked llm_validated=True (no filtering).")
        df.to_csv(output_path, index=False)
        return df

    # Validate in batches
    n_batches = (len(positives) + batch_size - 1) // batch_size
    passed = 0
    failed = 0
    errors = 0

    for batch_num, start in enumerate(range(0, len(positives), batch_size)):
        batch = positives.iloc[start:start + batch_size]
        sentences = [
            (int(idx), str(row[text_col]), int(row[label_col]))
            for idx, row in batch.iterrows()
        ]

        results = validator.validate_batch(sentences)

        for idx, text, label in sentences:
            is_valid = results.get(idx, True)  # default True on parse failure
            if is_valid:
                passed += 1
            else:
                failed += 1
                df.at[idx, "llm_validated"] = False

        # Progress
        done = min(start + batch_size, len(positives))
        pct = 100 * done / len(positives)
        print(f"  Batch {batch_num+1}/{n_batches}  [{done}/{len(positives)} = {pct:.0f}%]  "
              f"passed={passed}  failed={failed}", end="\r")

        if delay > 0 and batch_num < n_batches - 1:
            time.sleep(delay)

    print()  # newline after \r

    # Summary
    print(f"\n{'='*60}")
    print(f"LLM Validation Summary: {input_path.name}")
    print(f"{'='*60}")
    print(f"  Total positives validated: {len(positives)}")
    print(f"  Passed (llm_validated=True) : {passed}  ({100*passed/max(len(positives),1):.1f}%)")
    print(f"  Failed (llm_validated=False): {failed}  ({100*failed/max(len(positives),1):.1f}%)")
    print(f"  Errors (defaulted True)     : {errors}")
    print(f"\n  Negatives (unchanged)       : {(df[label_col]==0).sum()}")
    print(f"\n  Rows available for training:")
    validated_pos = ((df[label_col]==1) & (df["llm_validated"]==True)).sum()
    total_neg = (df[label_col]==0).sum()
    print(f"    Positives: {validated_pos}")
    print(f"    Negatives: {total_neg}")
    print(f"    Neg:pos ratio: {total_neg/max(validated_pos,1):.2f}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    return df


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LLM-validate positive rows in a harvest CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate all positives in epmc_direct
  python llm_validate_csv.py \\
    --input  classifier/data/training/epmc_direct_sentences.csv \\
    --output classifier/data/training/epmc_direct_llm_validated.csv

  # Dry-run: validate only 20 positives
  python llm_validate_csv.py \\
    --input  classifier/data/training/external_db_sentences.csv \\
    --output /tmp/extdb_test.csv \\
    --max-positives 20
        """,
    )
    parser.add_argument("--input",  "-i", required=True, help="Input harvest CSV")
    parser.add_argument("--output", "-o", required=True, help="Output CSV (new file, not overwritten)")
    parser.add_argument("--text-col",  default="text",  help="Column containing sentence text")
    parser.add_argument("--label-col", default="label", help="Column containing 0/1 labels")
    parser.add_argument("--batch-size", type=int, default=10, help="Sentences per API call")
    parser.add_argument("--max-positives", type=int, default=None,
                        help="Validate at most N positives (dry-run mode)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between API batches (rate limiting)")
    parser.add_argument("--api-key", default=None,
                        help="Anthropic API key (overrides ANTHROPIC_API_KEY env var)")
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    validate_csv(
        input_path=args.input,
        output_path=args.output,
        text_col=args.text_col,
        label_col=args.label_col,
        batch_size=args.batch_size,
        max_positives=args.max_positives,
        delay=args.delay,
        api_key=args.api_key,
    )
