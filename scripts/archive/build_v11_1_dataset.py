#!/usr/bin/env python3
"""
Build v11.1 Dataset: v11 + epmc_direct_v2 + external_db

v11.1 adds two new harvested sources on top of the validated v11 baseline:
  - epmc_direct_sentences_v2.csv : 885 positives focused on herbivory / predation
    (categories under-represented in v11: herbivory=73, predation=286)
  - external_db_sentences.csv    : 134 positives from Mangal / Web of Life / OpenAlex
    (new sources: food webs, mutualistic networks)

Strategy:
  - v11 is kept intact as the base
  - New positives: cap 300/category, apply interaction signal filter
  - New negatives: keep hard negatives (both_species / no signal), filter false negatives
  - Target neg:pos ratio ~2.4:1 (same as v11)
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path('/path/to/MetaP/classifier')
sys.path.insert(0, str(BASE_DIR / 'src'))

# ── interaction lexicon for signal filter ──────────────────────────────────────
try:
    from data.interaction_lexicon import score_sentence
    HAS_LEXICON = True
except ImportError:
    HAS_LEXICON = False

V11_FILE       = BASE_DIR / 'data/training/training_data_v11.csv'
EPMC_V2_FILE   = BASE_DIR / 'data/training/epmc_direct_sentences_v2.csv'
EXTERNAL_FILE  = BASE_DIR / 'data/training/external_db_sentences.csv'
OUTPUT_FILE    = BASE_DIR / 'data/training/training_data_v11_1.csv'

# Per-category cap for NEW positives only (v11 already balanced)
MAX_NEW_POS_PER_CATEGORY = 300

# False-negative filter (same as v11)
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
    t = text.lower()
    if any(p.search(t) for p in _NEGATION_PATTERNS):
        return False
    return any(p.search(t) for p in _FALSE_NEG_PATTERNS)


def _has_interaction_signal(text: str) -> bool:
    """Return True if the sentence has at least one interaction signal term."""
    if HAS_LEXICON:
        has_signal, _, _ = score_sentence(text.lower())
        return has_signal
    # Fallback: minimal keyword check
    keywords = {
        "prey", "preys", "predator", "predation", "hunt", "kill",
        "feed", "feeds", "forage", "graze", "browse", "consume",
        "parasite", "parasit", "infest", "host",
        "pollinate", "pollinator", "visit",
        "symbiont", "symbiosis", "mutuali", "commensal",
        "disperse", "dispersal", "infect", "pathogen", "vector",
        "transmit", "interact",
    }
    t = text.lower()
    return any(kw in t for kw in keywords)


def _process_new_source(
    df: pd.DataFrame,
    source_tag: str,
    existing_texts: set,
    max_pos_per_cat: int,
) -> pd.DataFrame:
    """
    Clean, filter and cap a new harvested source DataFrame.

    Steps:
      1. Filter incomplete sentences (must end with punctuation)
      2. Positives: require interaction signal
      3. Negatives: remove false negatives
      4. Deduplicate against existing dataset
      5. Cap positives at max_pos_per_cat per category
      6. Balance negatives to match new positive count
    """
    print(f"\n   Source: {source_tag} ({len(df)} rows, "
          f"{sum(df['label']==1)} pos, {sum(df['label']==0)} neg)")

    # 1. Punctuation filter
    before = len(df)
    df = df[df['text'].str.strip().str.endswith(('.', '!', '?', ')'))].copy()
    print(f"   After punctuation filter: {len(df)} (-{before - len(df)})")

    # 2. Positives: interaction signal required
    pos = df[df['label'] == 1].copy()
    before_sig = len(pos)
    pos = pos[pos['text'].apply(_has_interaction_signal)].copy()
    print(f"   Positives with signal: {len(pos)} / {before_sig} "
          f"(dropped {before_sig - len(pos)} no-signal)")

    # 3. Negatives: remove false negatives
    neg = df[df['label'] == 0].copy()
    false_neg = neg['text'].apply(_is_false_negative)
    if false_neg.sum() > 0:
        print(f"   Removed {false_neg.sum()} false negatives")
        neg = neg[~false_neg].copy()

    # 4. Deduplicate against existing
    before_dedup = len(pos) + len(neg)
    pos = pos[~pos['text'].str.lower().str.strip().isin(existing_texts)].copy()
    neg = neg[~neg['text'].str.lower().str.strip().isin(existing_texts)].copy()
    print(f"   After dedup: {len(pos)} pos, {len(neg)} neg "
          f"(-{before_dedup - len(pos) - len(neg)} duplicates)")

    # 5. Cap positives per category
    cat_col = 'category' if 'category' in pos.columns else 'interaction_type'
    if cat_col in pos.columns:
        capped = pos.groupby(cat_col).apply(
            lambda g: g.sample(n=min(len(g), max_pos_per_cat), random_state=42)
        ).reset_index(drop=True)
    else:
        capped = pos.sample(n=min(len(pos), max_pos_per_cat * 5), random_state=42)

    print(f"   After cap ({max_pos_per_cat}/category): {len(capped)} positives")
    if cat_col in capped.columns:
        for cat, cnt in capped[cat_col].value_counts().items():
            print(f"     {cat}: {cnt}")

    # 6. Balance negatives (1:1 with new positives)
    n_pos = len(capped)
    if len(neg) > n_pos:
        # Prioritise hard negatives (both species present, no signal)
        hard = neg[neg['source'].str.contains('neg|hard|two_species|other_species',
                                               case=False, na=False)]
        soft = neg[~neg['source'].str.contains('neg|hard|two_species|other_species',
                                                case=False, na=False)]
        n_hard = min(len(hard), int(n_pos * 0.7))
        n_soft = min(len(soft), n_pos - n_hard)
        neg = pd.concat([
            hard.sample(n=n_hard, random_state=42) if n_hard > 0 else hard.iloc[:0],
            soft.sample(n=n_soft, random_state=42) if n_soft > 0 else soft.iloc[:0],
        ], ignore_index=True)

    # Standardise columns
    keep = ['text', 'label', 'interaction_type', 'source_species', 'target_species']
    for col in keep:
        if col not in capped.columns:
            capped[col] = ''
        if col not in neg.columns:
            neg[col] = ''

    capped['source'] = source_tag
    neg['source']    = source_tag + '_neg'

    out = pd.concat([capped[keep + ['source']], neg[keep + ['source']]], ignore_index=True)
    print(f"   Final from {source_tag}: {len(out)} rows "
          f"({sum(out['label']==1)} pos, {sum(out['label']==0)} neg)")
    return out


def main():
    parser = argparse.ArgumentParser(description="Build v11.1 training dataset")
    parser.add_argument("--v11",      default=str(V11_FILE),      help="Base v11 dataset")
    parser.add_argument("--epmc-v2",  default=str(EPMC_V2_FILE),  help="epmc_direct_sentences_v2.csv")
    parser.add_argument("--external", default=str(EXTERNAL_FILE), help="external_db_sentences.csv")
    parser.add_argument("--output",   default=str(OUTPUT_FILE),   help="Output path")
    parser.add_argument("--max-pos-per-category", type=int, default=MAX_NEW_POS_PER_CATEGORY)
    args = parser.parse_args()

    print("=" * 70)
    print("BUILDING V11.1 DATASET")
    print("=" * 70)

    # ── Step 1: Load v11 base ────────────────────────────────────────────────
    print("\n1. Loading v11 base...")
    v11 = pd.read_csv(args.v11)
    print(f"   v11: {len(v11)} rows ({sum(v11['label']==1)} pos, {sum(v11['label']==0)} neg)")
    v11_texts = set(v11['text'].str.lower().str.strip())

    # ── Step 2: Process new sources ──────────────────────────────────────────
    print("\n2. Processing new sources...")

    new_parts = []

    if Path(args.epmc_v2).exists():
        epmc_v2 = pd.read_csv(args.epmc_v2)
        part = _process_new_source(epmc_v2, 'epmc_direct_v2', v11_texts, args.max_pos_per_category)
        new_parts.append(part)
        # Update seen texts to avoid cross-source duplicates
        v11_texts.update(part['text'].str.lower().str.strip())
    else:
        print(f"   WARNING: {args.epmc_v2} not found, skipping")

    if Path(args.external).exists():
        ext = pd.read_csv(args.external)
        part = _process_new_source(ext, 'external_db', v11_texts, args.max_pos_per_category)
        new_parts.append(part)
    else:
        print(f"   WARNING: {args.external} not found, skipping")

    if not new_parts:
        print("ERROR: No new sources available.")
        return

    # ── Step 3: Merge ────────────────────────────────────────────────────────
    print("\n3. Merging and shuffling...")

    # Ensure v11 has same columns
    for col in ['source_species', 'target_species', 'source']:
        if col not in v11.columns:
            v11[col] = ''

    combined = pd.concat([v11] + new_parts, ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # ── Step 4: Save ─────────────────────────────────────────────────────────
    combined.to_csv(args.output, index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    total_pos = sum(combined['label'] == 1)
    total_neg = sum(combined['label'] == 0)
    ratio = total_neg / total_pos if total_pos > 0 else 0

    print("\n" + "=" * 70)
    print("V11.1 DATASET SUMMARY")
    print("=" * 70)
    print(f"\nTotal : {len(combined)}")
    print(f"  Pos : {total_pos} ({100 * total_pos / len(combined):.1f}%)")
    print(f"  Neg : {total_neg} ({100 * total_neg / len(combined):.1f}%)")
    print(f"  Ratio neg:pos = {ratio:.2f}", end="  ")
    if ratio < 1.5:
        print("⚠ low")
    elif ratio > 3.5:
        print("⚠ high")
    else:
        print("✓")

    print("\nBy source:")
    for src, cnt in combined['source'].value_counts().items():
        pos_cnt = sum((combined['source'] == src) & (combined['label'] == 1))
        neg_cnt = cnt - pos_cnt
        print(f"  {src:<35s}: {cnt:5d}  ({pos_cnt} pos, {neg_cnt} neg)")

    print("\nPositives by interaction_type (top 15):")
    pos_df = combined[combined['label'] == 1]
    for t, c in pos_df['interaction_type'].value_counts().head(15).items():
        print(f"  {t:<35s}: {c}")

    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
