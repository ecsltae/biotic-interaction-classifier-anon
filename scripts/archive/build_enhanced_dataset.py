#!/usr/bin/env python3
"""
Build Enhanced Training Dataset
================================
Creates a larger, balanced training dataset with ~20k samples:
- 10k positive samples from true_positives.csv
- 10k negative samples from three categories:
  1. Random sentences (no species mentions)
  2. 3+ species sentences (too many species, likely false positives)
  3. No interaction sentences (species mentioned but no interaction)

Strategy for better negatives:
- Mix all three categories for diversity
- Random sampling to avoid bias
- Maintain balance
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "predictions"

# Input files
TRUE_POSITIVES = RESULTS_DIR / "true_positives.csv"
THREE_SPECIES = DATA_DIR / "processed" / "passages_with_3species_nointeractions.csv"
UNIQUE_SENTENCES = DATA_DIR / "processed" / "unique_sentences.csv"

# Output files
TRAINING_DIR = DATA_DIR / "training"
OUTPUT_FILE = TRAINING_DIR / "training_data_enhanced_20k.csv"
BACKUP_FILE = TRAINING_DIR / "training_data_cleaned.csv.backup"

def load_positives(n_samples=10000):
    """Load positive samples"""
    print(f"\n[1/4] Loading positive samples from {TRUE_POSITIVES.name}...")
    df = pd.read_csv(TRUE_POSITIVES)

    # Clean and deduplicate
    df['passage'] = df['passage'].astype(str).str.strip().str.lower()
    df = df[df['passage'].str.len() > 20]  # Remove very short passages
    df = df.drop_duplicates(subset=['passage'])

    print(f"  Available: {len(df)} samples")

    # Sample if we have more than needed
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=42)
        print(f"  Sampled: {n_samples} samples")
    else:
        print(f"  Using all: {len(df)} samples")

    df['label'] = 1
    return df[['passage', 'label']]

def load_negatives_category(file_path, category_name, n_samples):
    """Load negative samples from a category"""
    print(f"  Loading {category_name} from {file_path.name}...")
    df = pd.read_csv(file_path)

    # Clean and deduplicate
    df['passage'] = df['passage'].astype(str).str.strip().str.lower()
    df = df[df['passage'].str.len() > 20]
    df = df.drop_duplicates(subset=['passage'])

    available = len(df)
    print(f"    Available: {available} samples")

    # Sample
    if available > n_samples:
        df = df.sample(n=n_samples, random_state=42)

    actual = len(df)
    print(f"    Sampled: {actual} samples")

    return df[['passage']]

def load_negatives(n_samples=10000):
    """Load balanced negative samples from three categories"""
    print(f"\n[2/4] Loading negative samples (target: {n_samples})...")

    # Strategy:
    # - 3+ species: ~3000 (all available)
    # - Random sentences: ~4000 (diverse, no species)
    # - No interaction: ~3000 (species present but no interaction)

    # Category 1: 3+ species (use most/all available)
    three_species_df = load_negatives_category(
        THREE_SPECIES, "3+ species passages", 5000
    )

    # Category 2: Random sentences (sample from large pool)
    random_df = load_negatives_category(
        UNIQUE_SENTENCES, "random sentences", 5000
    )

    # Combine
    negatives = pd.concat([three_species_df, random_df], ignore_index=True)

    # Deduplicate across categories
    print(f"  Deduplicating across categories...")
    negatives = negatives.drop_duplicates(subset=['passage'])
    print(f"  After deduplication: {len(negatives)} samples")

    # Adjust to target
    if len(negatives) > n_samples:
        negatives = negatives.sample(n=n_samples, random_state=42)
        print(f"  Final sample: {n_samples} samples")

    negatives['label'] = 0
    return negatives

def build_dataset():
    """Build the enhanced dataset"""
    print("\n" + "="*70)
    print("BUILDING ENHANCED TRAINING DATASET")
    print("="*70)

    # Backup original training data
    print(f"\n[Backup] Creating backup of original training data...")
    original_training = TRAINING_DIR / "training_data_cleaned.csv"
    if original_training.exists():
        import shutil
        shutil.copy(original_training, BACKUP_FILE)
        print(f"  Backed up to: {BACKUP_FILE.name}")

    # Load positives
    positives = load_positives(n_samples=10000)

    # Load negatives
    negatives = load_negatives(n_samples=len(positives))  # Match positive count

    # Combine and shuffle
    print(f"\n[3/4] Combining datasets...")
    combined = pd.concat([positives, negatives], ignore_index=True)

    # Final deduplication
    print(f"  Total before dedup: {len(combined)}")
    combined = combined.drop_duplicates(subset=['passage'])
    print(f"  Total after dedup: {len(combined)}")

    # Shuffle
    print(f"  Shuffling...")
    combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Statistics
    print(f"\n[4/4] Dataset Statistics:")
    print(f"  Total samples: {len(combined)}")
    print(f"  Positive (label=1): {(combined['label'] == 1).sum()}")
    print(f"  Negative (label=0): {(combined['label'] == 0).sum()}")
    print(f"  Balance ratio: {(combined['label'] == 1).sum() / len(combined):.2%}")

    # Save
    print(f"\n[Save] Saving to {OUTPUT_FILE}...")
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"  ✓ Saved {len(combined)} samples")

    # Sample preview
    print(f"\n[Preview] Sample rows:")
    print(combined.head(10).to_string())

    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\nNew dataset: {OUTPUT_FILE}")
    print(f"Backup of original: {BACKUP_FILE}")
    print(f"\nNext step: Train with enhanced dataset:")
    print(f"  python src/models/transformer_classifier.py \\")
    print(f"    --train_data data/training/training_data_enhanced_20k.csv \\")
    print(f"    --model biobert --epochs 3")

    return combined

if __name__ == "__main__":
    df = build_dataset()
