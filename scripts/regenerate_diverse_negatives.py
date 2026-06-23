#!/usr/bin/env python3
"""
Regenerate diverse negative examples for v5 clean dataset.

Keeps positives unchanged, replaces negatives with diverse patterns:
- Single-species with interaction words (but no interaction)
- Active voice patterns
- Three-species negatives
- Negation patterns
- Traditional two-species co-occurrence

Target: 24,000 negatives total, keeping volume same but increasing diversity.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import random
from pathlib import Path

# Import the updated template generator
from data.template_generator import (
    generate_negatives_from_species,
    generate_hard_negatives,
    generate_three_species_negatives,
)

# Paths
BASE_DIR = Path('/path/to/MetaP/classifier')
INPUT_FILE = BASE_DIR / 'data/training/training_data_globi_v5_clean.csv'
OUTPUT_FILE = BASE_DIR / 'data/training/training_data_globi_v6_diverse.csv'

# Seed for reproducibility
random.seed(42)


def main():
    print("=" * 70)
    print("REGENERATING DIVERSE NEGATIVES FOR V6")
    print("=" * 70)

    # Load existing clean dataset
    print(f"\nLoading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    print(f"\nCurrent dataset:")
    print(f"  Total: {len(df)}")
    print(f"  Positives: {(df['label'] == 1).sum()}")
    print(f"  Negatives: {(df['label'] == 0).sum()}")

    # Separate positives and negatives
    positives_df = df[df['label'] == 1].copy()
    print(f"\nKeeping {len(positives_df)} positives unchanged")

    # Get all species for generating new negatives
    all_species = set()
    all_species.update(df['source_species'].dropna().unique())
    target_species = df['target_species'].dropna()
    # Handle cases where target_species might contain multiple species
    for ts in target_species:
        if isinstance(ts, str):
            for sp in ts.split(','):
                all_species.add(sp.strip())

    species_list = [s for s in all_species if len(str(s)) > 5]
    print(f"\nUsing {len(species_list)} unique species for negatives")

    # Target negative counts with diverse distribution
    n_negatives_total = 24000

    # Distribution:
    # - 65% hard two-species (15,600) - core precision training
    # - 20% easy single-species (4,800) - includes interaction words, active voice, negation
    # - 10% three-species (2,400) - new pattern
    # - 5% buffer for deduplication losses

    n_hard_two = int(n_negatives_total * 0.65)  # 15,600
    n_easy_single = int(n_negatives_total * 0.25)  # 6,000 (includes new patterns)
    n_three = int(n_negatives_total * 0.10)  # 2,400

    print(f"\nGenerating diverse negatives:")
    print(f"  Hard two-species: {n_hard_two}")
    print(f"  Easy single-species (w/ interaction words, active, negation): {n_easy_single}")
    print(f"  Three-species: {n_three}")

    # Generate negatives
    print("\n[1/3] Generating hard two-species negatives...")
    hard_negatives = generate_hard_negatives(species_list, n_hard_two)
    print(f"       Generated: {len(hard_negatives)}")

    print("\n[2/3] Generating easy single-species negatives...")
    easy_negatives = generate_negatives_from_species(species_list, n_easy_single)
    print(f"       Generated: {len(easy_negatives)}")

    print("\n[3/3] Generating three-species negatives...")
    three_negatives = generate_three_species_negatives(species_list, n_three)
    print(f"       Generated: {len(three_negatives)}")

    # Combine all negatives
    all_negatives = hard_negatives + easy_negatives + three_negatives

    # Convert to DataFrame
    neg_rows = []
    for neg in all_negatives:
        neg_rows.append({
            'text': neg.sentence,
            'label': 0,
            'source_species': neg.source_species,
            'target_species': neg.target_species,
            'interaction_type': neg.interaction_type,
            'quality_score': neg.quality_score
        })

    negatives_df = pd.DataFrame(neg_rows)

    # Deduplicate negatives
    print(f"\nDeduplicating negatives...")
    before_dedup = len(negatives_df)
    negatives_df = negatives_df.drop_duplicates(subset=['text'])
    after_dedup = len(negatives_df)
    print(f"  Before: {before_dedup}, After: {after_dedup}, Removed: {before_dedup - after_dedup}")

    # Trim to exact target if needed
    if len(negatives_df) > n_negatives_total:
        negatives_df = negatives_df.sample(n=n_negatives_total, random_state=42)

    # Combine positives and negatives
    combined_df = pd.concat([positives_df, negatives_df], ignore_index=True)

    # Shuffle
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Summary statistics
    print(f"\n{'=' * 70}")
    print("FINAL DATASET SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total samples: {len(combined_df)}")
    print(f"  Positives: {(combined_df['label'] == 1).sum()}")
    print(f"  Negatives: {(combined_df['label'] == 0).sum()}")

    # Negative breakdown
    n_hard = (combined_df['interaction_type'] == 'none_two_species').sum()
    n_easy = (combined_df['interaction_type'] == 'none').sum()
    n_three = (combined_df['interaction_type'] == 'none_three_species').sum()

    total_neg = n_hard + n_easy + n_three
    print(f"\nNegative diversity breakdown:")
    print(f"  Hard two-species:   {n_hard:6d} ({100*n_hard/total_neg:.1f}%)")
    print(f"  Easy single-species: {n_easy:6d} ({100*n_easy/total_neg:.1f}%)")
    print(f"  Three-species:      {n_three:6d} ({100*n_three/total_neg:.1f}%)")

    # Save
    print(f"\nSaving to: {OUTPUT_FILE}")
    combined_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Done!")

    # Sample preview of new patterns
    print(f"\n{'=' * 70}")
    print("SAMPLE NEW PATTERNS")
    print(f"{'=' * 70}")

    print("\n[Three-species examples]:")
    three_samples = combined_df[combined_df['interaction_type'] == 'none_three_species'].sample(3)
    for _, row in three_samples.iterrows():
        print(f"  - {row['text'][:100]}...")

    print("\n[Single-species with interaction words]:")
    single_samples = combined_df[combined_df['interaction_type'] == 'none'].sample(5)
    for _, row in single_samples.iterrows():
        if any(word in row['text'].lower() for word in ['pathogen', 'parasite', 'predator', 'vector', 'host']):
            print(f"  - {row['text']}")

    print("\n[Negation patterns]:")
    neg_patterns = combined_df[combined_df['text'].str.contains('not |no |neither', case=False, na=False)]
    if len(neg_patterns) > 0:
        for _, row in neg_patterns.head(5).iterrows():
            print(f"  - {row['text'][:100]}...")


if __name__ == "__main__":
    main()
