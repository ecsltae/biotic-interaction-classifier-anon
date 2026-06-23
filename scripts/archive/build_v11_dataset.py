#!/usr/bin/env python3
"""
Build v11 Dataset: v7 (LLM-validated) + Europe PMC direct harvest + GloBI-PMC v2

Key improvements over v10:
  - epmc_direct: 1,124 DIVERSE positives (predation, pollination, parasitism, eats, symbiosis)
    vs v10 which was 92% pathogen/infection-biased from globi_pmc_real_sentences.csv
  - globi_pmc_v2: 140 additional positives with fixed interaction types
  - Healthy neg:pos ratio (~2.4:1) preserved

Why v10 regressed (F1=0.722 regularized vs v7 F1=0.788):
  - pathogen bias (800/872 positives = pathogenOf) caused over-prediction on ecological test set
  - Precision dropped to 0.574, recall inflated to 0.975

v11 strategy:
  - Cap epmc positives at --max-pos-per-type (default 300) per interaction type
  - Include quality hard negatives: 'other_species' (2 species, no signal) and
    'single_species_with_signal' (has verb, 1 species) — exactly what Gate A/B catch
  - Cap total epmc negatives at --max-epmc-neg (default 1500)
"""

import argparse
import re
import pandas as pd
from pathlib import Path


# Patterns that definitively mark a sentence as a real interaction (Gate 4a patterns)
# These should NEVER appear in the negative pool
_FALSE_NEG_PATTERNS = [
    re.compile(r'\b(?:is|are)\s+(?:a\s+)?(?:parasite|parasites)\s+of\b'),
    re.compile(r'\bparasitizes?\b'),
    re.compile(r'\bpreys?\s+on\b'),
    re.compile(r'\bis\s+(?:a\s+)?(?:predator|prey)\s+(?:of|for)\b'),
    re.compile(r'\bpollinates?\b'),
    re.compile(r'\binfects?\b.*\bwas\s+(?:confirmed|detected|documented)\b'),
    re.compile(r'\bhost\s+(?:of|for)\b'),
    re.compile(r'\bis\s+(?:a\s+)?vector\s+(?:of|for)\b'),
]
_NEGATION_PATTERNS = [
    re.compile(r'\bnot?\b'), re.compile(r'\bno\b'), re.compile(r'\bneither\b'),
    re.compile(r'\bwithout\b'), re.compile(r'\babsence\b'), re.compile(r'\bfailed\b'),
    re.compile(r'\bdid\s+not\b'), re.compile(r'\bwas\s+not\b'), re.compile(r'\bcould\s+not\b'),
]


def _is_false_negative(text: str) -> bool:
    """Return True if a label=0 sentence looks like a real interaction (should be dropped)."""
    t = text.lower()
    has_negation = any(p.search(t) for p in _NEGATION_PATTERNS)
    if has_negation:
        return False
    return any(p.search(t) for p in _FALSE_NEG_PATTERNS)

BASE_DIR = Path('/path/to/MetaP/classifier')

V7_FILE    = BASE_DIR / 'data/training/training_data_globi_v7_llm_cleaned.csv'
EPMC_FILE  = BASE_DIR / 'data/training/epmc_direct_sentences.csv'
V2_FILE    = BASE_DIR / 'data/training/globi_pmc_sentences_v2.csv'
OUTPUT_FILE = BASE_DIR / 'data/training/training_data_v11.csv'


def main():
    parser = argparse.ArgumentParser(description="Build v11 training dataset")
    parser.add_argument("--v7",     default=str(V7_FILE),    help="Path to v7 LLM-cleaned data")
    parser.add_argument("--epmc",   default=str(EPMC_FILE),  help="Path to epmc_direct_sentences.csv")
    parser.add_argument("--v2",     default=str(V2_FILE),    help="Path to globi_pmc_sentences_v2.csv")
    parser.add_argument("--output", default=str(OUTPUT_FILE), help="Output path")
    parser.add_argument("--max-pos-per-type", type=int, default=300,
                        help="Max epmc positives per interaction_type (default: 300)")
    parser.add_argument("--max-epmc-neg", type=int, default=1500,
                        help="Max total epmc negatives to include (default: 1500)")
    args = parser.parse_args()

    print("=" * 70)
    print("BUILDING V11 DATASET")
    print("=" * 70)

    # ---- Step 1: Load v7 (LLM-validated baseline) ----
    print("\n1. Loading v7 (LLM-validated)...")
    v7 = pd.read_csv(args.v7)
    print(f"   v7 samples: {len(v7)}  ({sum(v7['label']==1)} pos, {sum(v7['label']==0)} neg)")

    keep_cols = ['text', 'label', 'interaction_type']
    for c in ['source_species', 'target_species']:
        if c in v7.columns:
            keep_cols.append(c)
    v7 = v7[[c for c in keep_cols if c in v7.columns]].copy()
    v7['source'] = 'v7_llm_cleaned'

    v7_texts = set(v7['text'].str.lower().str.strip())

    # ---- Step 2: Load and process Europe PMC direct harvest ----
    print("\n2. Loading Europe PMC direct sentences...")
    epmc = pd.read_csv(args.epmc)
    print(f"   Total: {len(epmc)}  ({sum(epmc['label']==1)} pos, {sum(epmc['label']==0)} neg)")
    print(f"\n   Positives by interaction_type:")
    for t, c in epmc[epmc['label']==1]['interaction_type'].value_counts().items():
        print(f"     {t}: {c}")

    # 2a. Cap positives per interaction type
    pos_epmc = epmc[epmc['label'] == 1].copy()
    neg_epmc = epmc[epmc['label'] == 0].copy()

    capped_pos = pos_epmc.groupby('interaction_type').apply(
        lambda g: g.sample(n=min(len(g), args.max_pos_per_type), random_state=42)
    ).reset_index(drop=True)
    print(f"\n   Positives after capping ({args.max_pos_per_type}/type): {len(capped_pos)}")
    for t, c in capped_pos['interaction_type'].value_counts().items():
        print(f"     {t}: {c}")

    # 2b. Sample negatives — prioritize 'other_species' (hard negatives with 2 species)
    # then 'single_species_with_signal' (hard negatives with interaction verb, 1 species)
    neg_other = neg_epmc[neg_epmc['source'].str.contains('other_species')].copy()
    neg_signal = neg_epmc[neg_epmc['source'].str.contains('single_species_with_signal')].copy()
    neg_rest   = neg_epmc[
        ~neg_epmc['source'].str.contains('other_species') &
        ~neg_epmc['source'].str.contains('single_species_with_signal')
    ].copy()

    # Allocate negatives: 2/3 other_species, 1/3 signal, remaining from rest
    budget_other  = min(len(neg_other),  int(args.max_epmc_neg * 0.60))
    budget_signal = min(len(neg_signal), int(args.max_epmc_neg * 0.30))
    budget_rest   = min(len(neg_rest),   args.max_epmc_neg - budget_other - budget_signal)

    sampled_neg = pd.concat([
        neg_other.sample(n=budget_other, random_state=42),
        neg_signal.sample(n=budget_signal, random_state=42),
        neg_rest.sample(n=budget_rest, random_state=42),
    ], ignore_index=True)
    print(f"\n   Negatives sampled: {len(sampled_neg)}")
    print(f"     other_species:             {budget_other}")
    print(f"     single_species_with_signal: {budget_signal}")
    print(f"     other types:               {budget_rest}")

    # 2c. Build epmc output df with consistent schema
    epmc_out_rows = []
    for part, is_pos in [(capped_pos, True), (sampled_neg, False)]:
        for _, row in part.iterrows():
            if is_pos:
                itype = row.get('interaction_type', 'unknown')
            else:
                src = str(row.get('source', ''))
                if 'other_species' in src or 'both_species' in src:
                    itype = 'none_two_species'
                else:
                    itype = 'none'
            epmc_out_rows.append({
                'text': row['text'],
                'label': row['label'],
                'interaction_type': itype,
                'source_species': row.get('source_species', ''),
                'target_species': row.get('target_species', ''),
                'source': 'epmc_direct',
            })
    epmc_out = pd.DataFrame(epmc_out_rows)

    # 2d. Filter incomplete sentences + false negatives from negative pool
    before_filter = len(epmc_out)
    epmc_out = epmc_out[epmc_out['text'].str.strip().str.endswith(('.', '!', '?', ')'))].copy()
    print(f"\n   Filtered {before_filter - len(epmc_out)} incomplete sentences")

    # Remove negatives that are actually real interactions (Gate 4a would flag them)
    neg_mask = epmc_out['label'] == 0
    false_neg_mask = neg_mask & epmc_out['text'].apply(_is_false_negative)
    n_false_neg = false_neg_mask.sum()
    if n_false_neg > 0:
        print(f"   Removed {n_false_neg} false negatives from negative pool")
        epmc_out = epmc_out[~false_neg_mask].copy()

    # 2e. Deduplicate against v7
    before_dedup = len(epmc_out)
    epmc_out = epmc_out[~epmc_out['text'].str.lower().str.strip().isin(v7_texts)].copy()
    print(f"   Removed {before_dedup - len(epmc_out)} duplicates already in v7")
    print(f"   epmc_direct after dedup: {len(epmc_out)} ({sum(epmc_out['label']==1)} pos, {sum(epmc_out['label']==0)} neg)")

    # ---- Step 3: Load and process GloBI-PMC v2 ----
    print("\n3. Loading GloBI-PMC v2 sentences...")
    v2 = pd.read_csv(args.v2)
    print(f"   Total: {len(v2)}  ({sum(v2['label']==1)} pos, {sum(v2['label']==0)} neg)")
    print(f"\n   Positives by interaction_type:")
    for t, c in v2[v2['label']==1]['interaction_type'].value_counts().head(8).items():
        print(f"     {t}: {c}")

    v2_out_rows = []
    for _, row in v2.iterrows():
        is_pos = row['label'] == 1
        if is_pos:
            itype = row.get('interaction_type', 'unknown')
        else:
            src = str(row.get('source', ''))
            if 'other_species' in src or 'both_species' in src:
                itype = 'none_two_species'
            else:
                itype = 'none'
        v2_out_rows.append({
            'text': row['text'],
            'label': row['label'],
            'interaction_type': itype,
            'source_species': row.get('source_species', ''),
            'target_species': row.get('target_species', ''),
            'source': 'globi_pmc_v2',
        })
    v2_out = pd.DataFrame(v2_out_rows)

    # Filter incomplete + false negatives + deduplicate
    before = len(v2_out)
    v2_out = v2_out[v2_out['text'].str.strip().str.endswith(('.', '!', '?', ')'))].copy()
    neg_mask_v2 = v2_out['label'] == 0
    false_neg_mask_v2 = neg_mask_v2 & v2_out['text'].apply(_is_false_negative)
    if false_neg_mask_v2.sum() > 0:
        print(f"   Removed {false_neg_mask_v2.sum()} false negatives from v2 negative pool")
        v2_out = v2_out[~false_neg_mask_v2].copy()
    v2_out = v2_out[~v2_out['text'].str.lower().str.strip().isin(v7_texts)].copy()
    print(f"   After filter+dedup: {len(v2_out)} ({sum(v2_out['label']==1)} pos, {sum(v2_out['label']==0)} neg)")
    print(f"   Removed: {before - len(v2_out)}")

    # ---- Step 4: Merge and shuffle ----
    print("\n4. Merging and shuffling...")
    combined = pd.concat([v7, epmc_out, v2_out], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # ---- Step 5: Save ----
    combined.to_csv(args.output, index=False)

    # ---- Summary ----
    total_pos = sum(combined['label'] == 1)
    total_neg = sum(combined['label'] == 0)
    print("\n" + "=" * 70)
    print("V11 DATASET SUMMARY")
    print("=" * 70)
    print(f"\nTotal: {len(combined)}")
    print(f"  Positives: {total_pos} ({100 * total_pos / len(combined):.1f}%)")
    print(f"  Negatives: {total_neg} ({100 * total_neg / len(combined):.1f}%)")
    print(f"\nBy source:")
    for src, cnt in combined['source'].value_counts().items():
        pos_cnt = sum((combined['source'] == src) & (combined['label'] == 1))
        neg_cnt = sum((combined['source'] == src) & (combined['label'] == 0))
        print(f"  {src:<30s}: {cnt:5d}  ({pos_cnt} pos, {neg_cnt} neg)")
    print(f"\nPositive interaction_type distribution:")
    for t, c in combined[combined['label']==1]['interaction_type'].value_counts().head(12).items():
        print(f"  {t:<30s}: {c}")

    ratio = total_neg / total_pos if total_pos > 0 else 0
    print(f"\nNeg:Pos ratio: {ratio:.2f}")
    if ratio < 1.5:
        print("  WARNING: Low negative ratio — model may lack discriminative power")
    elif ratio > 3.0:
        print("  WARNING: High negative ratio — may bias model toward negative class")
    else:
        print("  OK: Healthy balance")

    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
