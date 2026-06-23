#!/usr/bin/env python3
"""
Build v9 Dataset: Merge v7 (LLM-validated) + GloBI/SIBiLS (real diverse sentences)

Data already diverse across 9 interaction categories from SIBiLS biodiversity API.
No need for heavy balancing - just merge, dedup, and shuffle.
"""

import pandas as pd

BASE_DIR = '/path/to/MetaP/classifier'

V7_FILE = f'{BASE_DIR}/data/training/training_data_globi_v7_llm_cleaned.csv'
SIBILS_FILE = f'{BASE_DIR}/data/training/globi_sibils_real.csv'
OUTPUT_FILE = f'{BASE_DIR}/data/training/training_data_v9.csv'


def main():
    print("="*70)
    print("BUILDING V9 DATASET")
    print("="*70)

    # Load v7
    print("\n1. Loading v7 (LLM-validated)...")
    v7 = pd.read_csv(V7_FILE)
    print(f"   v7 samples: {len(v7)}")
    print(f"   v7 positives: {sum(v7['label']==1)}")
    print(f"   v7 negatives: {sum(v7['label']==0)}")

    v7 = v7[['text', 'label']].copy()
    v7['source'] = 'v7_llm_cleaned'

    # Load SIBiLS real sentences
    print("\n2. Loading GloBI/SIBiLS real sentences...")
    sibils = pd.read_csv(SIBILS_FILE)
    print(f"   SIBiLS samples: {len(sibils)}")
    print(f"   SIBiLS positives: {sum(sibils['label']==1)}")
    print(f"   SIBiLS negatives: {sum(sibils['label']==0)}")
    print(f"   Unique PMIDs: {sibils['pmid'].nunique()}")
    print(f"\n   By source:")
    for src, cnt in sibils['source'].value_counts().items():
        print(f"     {src}: {cnt}")

    sibils = sibils[['text', 'label', 'source']].copy()

    # Dedup: remove SIBiLS sentences that already exist in v7
    print("\n3. Removing duplicates...")
    v7_texts = set(v7['text'].str.lower().str.strip())
    before = len(sibils)
    sibils = sibils[~sibils['text'].str.lower().str.strip().isin(v7_texts)]
    print(f"   Removed {before - len(sibils)} duplicates")
    print(f"   SIBiLS after dedup: {len(sibils)}")

    # Merge
    print("\n4. Merging...")
    combined = pd.concat([v7, sibils], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    combined.to_csv(OUTPUT_FILE, index=False)

    # Summary
    print("\n" + "="*70)
    print("V9 DATASET SUMMARY")
    print("="*70)
    total_pos = sum(combined['label']==1)
    total_neg = sum(combined['label']==0)
    print(f"\nTotal: {len(combined)}")
    print(f"  Positives: {total_pos} ({100*total_pos/len(combined):.1f}%)")
    print(f"  Negatives: {total_neg} ({100*total_neg/len(combined):.1f}%)")
    print(f"\nBy source:")
    for src, cnt in combined['source'].value_counts().items():
        print(f"  {src}: {cnt}")
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
