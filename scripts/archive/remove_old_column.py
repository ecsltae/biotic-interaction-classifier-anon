#!/usr/bin/env python3
"""
Remove the evaluation_interaction_identified column from all evaluation TSV files.
Keep only evaluation_pair_interacting as the ground truth.
"""

import pandas as pd
import os
from pathlib import Path

# Files to process
data_dir = Path("/path/to/MetaP/classifier/data/evaluation")
files_to_process = [
    "biotx-random_passages-triplets_2024-02-28_curation_EP_100original.tsv",
    "biotx-random_passages-triplets_2024-04-22b_curation_EP_50best-multiples.tsv",
    "biotx-random_passages-triplets_2024-05-15_curation_EP_50nomultiple.tsv",
    "globi-relax_passages-triplets_2024-02-28_curation_EP.tsv",
    "globi-passage_passages-triplets_2024-02-28_curation_EP.tsv",
]

print("Removing 'evaluation_interaction_identified' column from evaluation files")
print("=" * 70)

for filename in files_to_process:
    filepath = data_dir / filename
    if filepath.exists():
        print(f"\nProcessing: {filename}")

        # Read the file
        df = pd.read_csv(filepath, sep='\t', encoding='latin1')

        # Check if column exists
        if 'evaluation_interaction_identified' in df.columns:
            # Remove the column
            df = df.drop(columns=['evaluation_interaction_identified'])

            # Save back
            df.to_csv(filepath, sep='\t', index=False, encoding='utf-8')
            print(f"  ✓ Removed column, saved file")
            print(f"  Remaining columns: {list(df.columns)}")
        else:
            print(f"  - Column not found (already removed?)")
    else:
        print(f"\nSkipping: {filename} (not found)")

# Also process CSV files
csv_files = [
    "eval_100_ensemble_predictions.csv",
]

for filename in csv_files:
    filepath = data_dir / filename
    if filepath.exists():
        print(f"\nProcessing: {filename}")

        df = pd.read_csv(filepath)

        if 'evaluation_interaction_identified' in df.columns:
            df = df.drop(columns=['evaluation_interaction_identified'])
            df.to_csv(filepath, index=False)
            print(f"  ✓ Removed column, saved file")
        else:
            print(f"  - Column not found")

# Also update processed data file
processed_file = Path("/path/to/MetaP/classifier/data/processed/predictions_with_BiomedBERT.csv")
if processed_file.exists():
    print(f"\nProcessing: {processed_file.name}")
    df = pd.read_csv(processed_file)
    if 'evaluation_interaction_identified' in df.columns:
        df = df.drop(columns=['evaluation_interaction_identified'])
        df.to_csv(processed_file, index=False)
        print(f"  ✓ Removed column")

print("\n" + "=" * 70)
print("COMPLETE")
