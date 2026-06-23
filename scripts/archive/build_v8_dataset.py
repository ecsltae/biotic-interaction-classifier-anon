#!/usr/bin/env python3
"""
Build v8 Dataset: Merge v7 (LLM-validated) + SIBiLS (real diverse sentences)
Goal: Diverse, high-quality training data with real literature sentences.
"""

import pandas as pd
import numpy as np
from collections import Counter
import re

BASE_DIR = '/path/to/MetaP/classifier'

# Input files
V7_FILE = f'{BASE_DIR}/data/training/training_data_globi_v7_llm_cleaned.csv'
SIBILS_FILE = f'{BASE_DIR}/data/training/sibils_diverse_real.csv'

# Output file
OUTPUT_FILE = f'{BASE_DIR}/data/training/training_data_v8_diverse.csv'

# Config
MAX_PER_INTERACTION_TYPE = 500  # Limit infection dominance
MIN_SIBILS_SAMPLES = 6000  # Minimum SIBiLS samples to include


def balance_sibils_data(df, max_per_type=MAX_PER_INTERACTION_TYPE):
    """Balance SIBiLS positives to avoid single interaction type dominance."""
    positives = df[df['label'] == 1].copy()
    negatives = df[df['label'] == 0].copy()

    print(f"  Original positives: {len(positives)}")
    print(f"  Interaction type distribution before balancing:")

    type_counts = positives['interaction_type'].value_counts()
    for t, c in type_counts.head(10).items():
        print(f"    {t}: {c}")

    # Group by interaction type base (normalize variants)
    def normalize_type(t):
        if pd.isna(t):
            return "unknown"
        t = str(t).lower().strip()
        # Group infection variants
        if 'infect' in t:
            return 'infection'
        if 'parasit' in t:
            return 'parasitism'
        if 'feed' in t or 'fed' in t:
            return 'feeding'
        if 'pollinat' in t:
            return 'pollination'
        if 'prey' in t or 'predat' in t:
            return 'predation'
        if 'host' in t:
            return 'host'
        if 'symbio' in t:
            return 'symbiosis'
        if 'compet' in t:
            return 'competition'
        if 'coloni' in t:
            return 'colonization'
        if 'graz' in t:
            return 'grazing'
        if 'consum' in t:
            return 'consumption'
        return t

    positives['type_group'] = positives['interaction_type'].apply(normalize_type)

    # Sample with cap per type
    balanced = []
    for type_group, group_df in positives.groupby('type_group'):
        n_samples = min(len(group_df), max_per_type)
        balanced.append(group_df.sample(n=n_samples, random_state=42))

    balanced_positives = pd.concat(balanced, ignore_index=True)

    print(f"\n  Balanced positives: {len(balanced_positives)}")
    print(f"  Type distribution after balancing:")
    for t, c in balanced_positives['type_group'].value_counts().head(10).items():
        print(f"    {t}: {c}")

    # Combine with negatives (sample proportionally)
    n_negatives = int(len(balanced_positives) * 1.0)  # 1:1 ratio
    sampled_negatives = negatives.sample(n=min(n_negatives, len(negatives)), random_state=42)

    result = pd.concat([balanced_positives, sampled_negatives], ignore_index=True)
    result = result.drop(columns=['type_group'], errors='ignore')

    return result


def main():
    print("="*70)
    print("BUILDING V8 DATASET")
    print("="*70)

    # Load v7
    print("\n1. Loading v7 (LLM-validated)...")
    v7 = pd.read_csv(V7_FILE)
    print(f"   v7 samples: {len(v7)}")
    print(f"   v7 positives: {sum(v7['label']==1)}")
    print(f"   v7 negatives: {sum(v7['label']==0)}")

    # Standardize columns
    v7 = v7[['text', 'label']].copy()
    v7['source'] = 'v7_llm_cleaned'

    # Load SIBiLS
    print("\n2. Loading SIBiLS (real sentences)...")
    sibils = pd.read_csv(SIBILS_FILE)
    print(f"   SIBiLS samples: {len(sibils)}")
    print(f"   SIBiLS positives: {sum(sibils['label']==1)}")
    print(f"   SIBiLS negatives: {sum(sibils['label']==0)}")

    # Balance SIBiLS
    print("\n3. Balancing SIBiLS interaction types...")
    sibils_balanced = balance_sibils_data(sibils)
    print(f"   Balanced SIBiLS: {len(sibils_balanced)}")

    # Standardize columns
    sibils_balanced = sibils_balanced[['text', 'label', 'source']].copy()

    # Check for duplicates between v7 and SIBiLS
    print("\n4. Removing duplicates...")
    v7_texts = set(v7['text'].str.lower().str.strip())
    sibils_clean = sibils_balanced[~sibils_balanced['text'].str.lower().str.strip().isin(v7_texts)]
    print(f"   SIBiLS after dedup: {len(sibils_clean)} (removed {len(sibils_balanced) - len(sibils_clean)} duplicates)")

    # Merge
    print("\n5. Merging datasets...")
    combined = pd.concat([v7, sibils_clean], ignore_index=True)

    # Shuffle
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    print("\n6. Saving v8 dataset...")
    combined.to_csv(OUTPUT_FILE, index=False)

    # Summary
    print("\n" + "="*70)
    print("V8 DATASET SUMMARY")
    print("="*70)
    print(f"\nTotal samples: {len(combined)}")
    print(f"  Positives: {sum(combined['label']==1)} ({100*sum(combined['label']==1)/len(combined):.1f}%)")
    print(f"  Negatives: {sum(combined['label']==0)} ({100*sum(combined['label']==0)/len(combined):.1f}%)")

    print(f"\nBy source:")
    for source, count in combined['source'].value_counts().items():
        print(f"  {source}: {count}")

    print(f"\nSaved to: {OUTPUT_FILE}")

    # Show samples
    print("\n" + "="*70)
    print("SAMPLE SENTENCES")
    print("="*70)

    print("\nFrom v7 (positive):")
    v7_pos = combined[(combined['source']=='v7_llm_cleaned') & (combined['label']==1)]
    if len(v7_pos) > 0:
        print(f"  - {v7_pos.iloc[0]['text'][:120]}...")

    print("\nFrom SIBiLS (positive):")
    sibils_pos = combined[(combined['source'].str.contains('sibils')) & (combined['label']==1)]
    if len(sibils_pos) > 0:
        print(f"  - {sibils_pos.iloc[0]['text'][:120]}...")

    print("\nFrom v7 (negative):")
    v7_neg = combined[(combined['source']=='v7_llm_cleaned') & (combined['label']==0)]
    if len(v7_neg) > 0:
        print(f"  - {v7_neg.iloc[0]['text'][:120]}...")

    print("\nFrom SIBiLS (negative):")
    sibils_neg = combined[(combined['source'].str.contains('sibils')) & (combined['label']==0)]
    if len(sibils_neg) > 0:
        print(f"  - {sibils_neg.iloc[0]['text'][:120]}...")


if __name__ == "__main__":
    main()
