#!/usr/bin/env python3
"""
Build Enhanced Training Dataset (Pure Python version)
No external dependencies required
"""

import csv
import random
import shutil
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

random.seed(42)

def read_csv(filepath, max_rows=None):
    """Read CSV file"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            passage = row.get('passage', '').strip().lower()
            if len(passage) > 20:  # Filter short passages
                data.append(passage)
    return data

def load_positives(n_samples=10000):
    """Load positive samples"""
    print(f"\n[1/4] Loading positive samples from {TRUE_POSITIVES.name}...")
    passages = read_csv(TRUE_POSITIVES)

    # Deduplicate
    passages = list(set(passages))
    print(f"  Available: {len(passages)} unique samples")

    # Sample
    if len(passages) > n_samples:
        passages = random.sample(passages, n_samples)
        print(f"  Sampled: {n_samples} samples")
    else:
        print(f"  Using all: {len(passages)} samples")

    return passages

def load_negatives(n_samples=10000):
    """Load negative samples"""
    print(f"\n[2/4] Loading negative samples (target: {n_samples})...")

    # Load 3+ species passages
    print(f"  Loading 3+ species from {THREE_SPECIES.name}...")
    three_species = read_csv(THREE_SPECIES, max_rows=6000)
    print(f"    Loaded: {len(three_species)} samples")

    # Load random sentences
    print(f"  Loading random sentences from {UNIQUE_SENTENCES.name}...")
    random_sentences = read_csv(UNIQUE_SENTENCES, max_rows=8000)
    print(f"    Loaded: {len(random_sentences)} samples")

    # Combine
    all_negatives = three_species + random_sentences

    # Deduplicate
    print(f"  Deduplicating...")
    all_negatives = list(set(all_negatives))
    print(f"  After dedup: {len(all_negatives)} samples")

    # Sample to target
    if len(all_negatives) > n_samples:
        all_negatives = random.sample(all_negatives, n_samples)
        print(f"  Sampled: {n_samples} samples")

    return all_negatives

def build_dataset():
    """Build the enhanced dataset"""
    print("\n" + "="*70)
    print("BUILDING ENHANCED TRAINING DATASET")
    print("="*70)

    # Backup
    print(f"\n[Backup] Creating backup...")
    original = TRAINING_DIR / "training_data_cleaned.csv"
    if original.exists():
        shutil.copy(original, BACKUP_FILE)
        print(f"  ✓ Backed up to: {BACKUP_FILE.name}")

    # Load data
    positives = load_positives(10000)
    negatives = load_negatives(len(positives))

    # Combine
    print(f"\n[3/4] Combining datasets...")
    combined = []

    for passage in positives:
        combined.append({'passage': passage, 'label': 1})

    for passage in negatives:
        combined.append({'passage': passage, 'label': 0})

    # Final deduplication by passage
    seen = set()
    unique_combined = []
    for item in combined:
        if item['passage'] not in seen:
            seen.add(item['passage'])
            unique_combined.append(item)

    combined = unique_combined
    print(f"  Total after dedup: {len(combined)}")

    # Shuffle
    print(f"  Shuffling...")
    random.shuffle(combined)

    # Statistics
    print(f"\n[4/4] Dataset Statistics:")
    n_positive = sum(1 for x in combined if x['label'] == 1)
    n_negative = sum(1 for x in combined if x['label'] == 0)
    print(f"  Total samples: {len(combined)}")
    print(f"  Positive (label=1): {n_positive}")
    print(f"  Negative (label=0): {n_negative}")
    print(f"  Balance ratio: {n_positive / len(combined):.2%}")

    # Save
    print(f"\n[Save] Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['passage', 'label'])
        writer.writeheader()
        writer.writerows(combined)

    print(f"  ✓ Saved {len(combined)} samples")

    # Preview
    print(f"\n[Preview] First 5 positive samples:")
    for i, item in enumerate([x for x in combined if x['label'] == 1][:5], 1):
        print(f"  {i}. {item['passage'][:80]}...")

    print(f"\n[Preview] First 5 negative samples:")
    for i, item in enumerate([x for x in combined if x['label'] == 0][:5], 1):
        print(f"  {i}. {item['passage'][:80]}...")

    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\nNew dataset: {OUTPUT_FILE}")
    print(f"Backup of original: {BACKUP_FILE}")
    print(f"\nNext: Train with:")
    print(f"  python src/models/transformer_classifier.py \\")
    print(f"    --train_data data/training/training_data_enhanced_20k.csv \\")
    print(f"    --model biobert --epochs 3")

if __name__ == "__main__":
    build_dataset()
