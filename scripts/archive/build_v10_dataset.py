#!/usr/bin/env python3
"""
Build v10 Dataset: Merge v7 (LLM-validated) + GloBI-PMC (real co-occurrence sentences)

Key difference from v8/v9: PMC sentences are labeled via GloBI-curated co-occurrence,
NOT regex-based interaction verb matching. This avoids the noise that degraded v8/v9.

v8 (SIBiLS infection-biased) → F1=0.695 (REGRESSED)
v9 (SIBiLS regex-labeled)    → F1=0.644 (REGRESSED)
v10 (GloBI-PMC co-occurrence) → target: F1>0.75
"""

import argparse
import pandas as pd
from pathlib import Path

BASE_DIR = Path('/path/to/MetaP/classifier')

V7_FILE = BASE_DIR / 'data/training/training_data_globi_v7_llm_cleaned.csv'
PMC_FILE = BASE_DIR / 'data/training/globi_pmc_real_sentences.csv'
OUTPUT_FILE = BASE_DIR / 'data/training/training_data_v10.csv'


def main():
    parser = argparse.ArgumentParser(description="Build v10 training dataset")
    parser.add_argument("--v7", default=str(V7_FILE), help="Path to v7 LLM-cleaned data")
    parser.add_argument("--pmc", default=str(PMC_FILE), help="Path to GloBI-PMC real sentences")
    parser.add_argument("--output", default=str(OUTPUT_FILE), help="Output path")
    parser.add_argument("--max-pmc-per-type", type=int, default=800,
                        help="Max PMC sentences per interaction type")
    args = parser.parse_args()

    print("=" * 70)
    print("BUILDING V10 DATASET")
    print("=" * 70)

    # ---- Step 1: Load v7 (LLM-validated baseline) ----
    print("\n1. Loading v7 (LLM-validated)...")
    v7 = pd.read_csv(args.v7)
    print(f"   v7 samples: {len(v7)}")
    print(f"   v7 positives: {sum(v7['label'] == 1)}")
    print(f"   v7 negatives: {sum(v7['label'] == 0)}")

    # Preserve interaction_type for quality gates
    keep_cols = ['text', 'label', 'interaction_type']
    if 'source_species' in v7.columns:
        keep_cols.append('source_species')
    if 'target_species' in v7.columns:
        keep_cols.append('target_species')
    v7 = v7[[c for c in keep_cols if c in v7.columns]].copy()
    v7['source'] = 'v7_llm_cleaned'

    # ---- Step 2: Load GloBI-PMC real sentences ----
    print("\n2. Loading GloBI-PMC real sentences...")
    pmc = pd.read_csv(args.pmc)
    print(f"   PMC samples: {len(pmc)}")
    print(f"   PMC positives: {sum(pmc['label'] == 1)}")
    print(f"   PMC negatives: {sum(pmc['label'] == 0)}")

    if 'category' in pmc.columns:
        print(f"\n   By interaction category:")
        for cat, cnt in pmc[pmc['label'] == 1]['category'].value_counts().items():
            print(f"     {cat}: {cnt}")

    if 'match_type' in pmc.columns:
        print(f"\n   By match type:")
        for mt, cnt in pmc['match_type'].value_counts().items():
            print(f"     {mt}: {cnt}")

    # ---- Step 3: Cap per interaction type ----
    if args.max_pmc_per_type and 'category' in pmc.columns:
        print(f"\n3. Capping PMC positives to {args.max_pmc_per_type} per interaction type...")
        pos = pmc[pmc['label'] == 1]
        neg = pmc[pmc['label'] == 0]

        capped_pos = pos.groupby('category').apply(
            lambda g: g.sample(n=min(len(g), args.max_pmc_per_type), random_state=42)
        ).reset_index(drop=True)

        # Cap negatives to match positive count
        neg_target = len(capped_pos)
        if len(neg) > neg_target:
            neg = neg.sample(n=neg_target, random_state=42)

        pmc = pd.concat([capped_pos, neg], ignore_index=True)
        print(f"   After capping: {len(pmc)} ({sum(pmc['label'] == 1)} pos, {sum(pmc['label'] == 0)} neg)")
    else:
        print("\n3. No capping applied (no category column or flag disabled)")

    # Map PMC columns to match v7 schema
    pmc_out = pmc[['text', 'label']].copy()
    if 'category' in pmc.columns:
        pmc_out['interaction_type'] = pmc['category']
    if 'source_species' in pmc.columns:
        pmc_out['source_species'] = pmc['source_species']
    if 'target_species' in pmc.columns:
        pmc_out['target_species'] = pmc['target_species']
    # Map negative match_types to interaction_type labels expected by quality gates
    neg_mask = pmc_out['label'] == 0
    if 'match_type' in pmc.columns:
        pmc_out.loc[neg_mask & (pmc['match_type'] == 'other_species'), 'interaction_type'] = 'none_two_species'
        pmc_out.loc[neg_mask & (pmc['match_type'] == 'single_species'), 'interaction_type'] = 'none'
        pmc_out.loc[neg_mask & (pmc['match_type'] == 'no_species'), 'interaction_type'] = 'none'
    pmc_out['source'] = 'globi_pmc_real'

    # Filter incomplete sentences (must end with punctuation)
    before_filter = len(pmc_out)
    pmc_out = pmc_out[pmc_out['text'].str.strip().str.endswith(('.', '!', '?', ')'))].copy()
    print(f"   Filtered {before_filter - len(pmc_out)} incomplete sentences")

    # ---- Step 4: Deduplicate ----
    print("\n4. Removing duplicates...")
    v7_texts = set(v7['text'].str.lower().str.strip())
    before = len(pmc_out)
    pmc_out = pmc_out[~pmc_out['text'].str.lower().str.strip().isin(v7_texts)]
    print(f"   Removed {before - len(pmc_out)} duplicates already in v7")
    print(f"   PMC after dedup: {len(pmc_out)}")

    # ---- Step 5: Merge and shuffle ----
    print("\n5. Merging and shuffling...")
    combined = pd.concat([v7, pmc_out], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # ---- Step 6: Save ----
    combined.to_csv(args.output, index=False)

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("V10 DATASET SUMMARY")
    print("=" * 70)
    total_pos = sum(combined['label'] == 1)
    total_neg = sum(combined['label'] == 0)
    print(f"\nTotal: {len(combined)}")
    print(f"  Positives: {total_pos} ({100 * total_pos / len(combined):.1f}%)")
    print(f"  Negatives: {total_neg} ({100 * total_neg / len(combined):.1f}%)")
    print(f"\nBy source:")
    for src, cnt in combined['source'].value_counts().items():
        print(f"  {src}: {cnt}")

    ratio = total_neg / total_pos if total_pos > 0 else 0
    print(f"\nNeg:Pos ratio: {ratio:.2f}")
    if ratio < 1.5:
        print("  ⚠ Low negative ratio — model may lack discriminative power")
    elif ratio > 3.0:
        print("  ⚠ High negative ratio — may bias model toward negative class")
    else:
        print("  ✓ Healthy balance")

    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
