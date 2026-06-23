#!/usr/bin/env python3
"""
Build v14_quality Dataset: Same as v14 but with HIGHER score threshold (>=0.2)

Analysis showed:
- 51.8% of SIBiLS positives have score < 0.2 (low quality)
- These low-quality samples correlate with the v14 regression (F1: 0.776 → 0.706)

This version uses score >= 0.2 threshold to keep only high+medium quality samples:
  - score >= 0.5: High quality (23.4%)
  - score 0.2-0.5: Medium quality (24.8%)
  - score 0-0.2: LOW quality (51.8%) → EXCLUDED
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

for _candidate in [
    Path(__file__).parent.parent / "src",
    Path(__file__).parent / "src",
]:
    if (_candidate / "data" / "interaction_lexicon.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from data.interaction_lexicon import score_sentence  # noqa: E402

BASE_DIR = Path('/path/to/MetaP/classifier')


MIN_SCORE_THRESHOLD = 0.2  # Higher threshold for quality filtering

def has_signal(text: str) -> bool:
    _, strength, _ = score_sentence(str(text).lower())
    return strength >= MIN_SCORE_THRESHOLD  # Changed from > 0.0


def add_source(df_new: pd.DataFrame, existing_texts: set, source_name: str,
               filter_positives: bool = True, max_pos_per_type: int = 500,
               neg_per_pos: float = 1.5) -> pd.DataFrame:
    """Filter, dedup, and balance a new source before merging."""
    if 'sentence' in df_new.columns and 'text' not in df_new.columns:
        df_new = df_new.rename(columns={'sentence': 'text'})
    for col in ['interaction_type', 'source_species', 'target_species']:
        if col not in df_new.columns:
            df_new[col] = ''

    # Dedup against existing
    before = len(df_new)
    df_new = df_new[~df_new['text'].str.strip().str.lower().isin(existing_texts)].copy()
    print(f"   Dedup: {before} → {len(df_new)} rows")

    pos = df_new[df_new['label'] == 1].copy()
    neg = df_new[df_new['label'] == 0].copy()

    # Quality filter on positives
    if filter_positives:
        before_pos = len(pos)
        pos = pos[pos['text'].apply(has_signal)].copy()
        print(f"   score>={MIN_SCORE_THRESHOLD} filter: {before_pos} → {len(pos)} positives kept "
              f"({100*len(pos)/max(before_pos,1):.1f}%)")

    # Cap per interaction_type
    if 'interaction_type' in pos.columns and len(pos):
        pos = pos.groupby('interaction_type', group_keys=False).apply(
            lambda g: g.sample(n=min(len(g), max_pos_per_type), random_state=42)
        ).reset_index(drop=True)
        print(f"   After cap ({max_pos_per_type}/type): {len(pos)} positives")

    # Sample negatives proportionally
    n_neg = min(len(neg), int(len(pos) * neg_per_pos))
    neg = neg.sample(n=n_neg, random_state=42) if n_neg < len(neg) else neg
    print(f"   Negatives sampled: {len(neg)}")

    out = pd.concat([pos, neg], ignore_index=True)
    out['source'] = source_name
    return out


def main():
    parser = argparse.ArgumentParser(description="Build v14 training dataset")
    parser.add_argument("--base",    default=str(BASE_DIR / 'data/training/training_data_v13.csv'))
    parser.add_argument("--sibils", default=str(BASE_DIR / 'data/training/sibils_diverse_real.csv'))
    parser.add_argument("--globi",  default=str(BASE_DIR / 'data/training/globi_sibils_real.csv'))
    parser.add_argument("--student",default=str(BASE_DIR / 'data/training/student_set_high_quality.csv'))
    parser.add_argument("--output", default=str(BASE_DIR / 'data/training/training_data_v14_quality.csv'))
    parser.add_argument("--max-pos-per-type", type=int, default=500)
    args = parser.parse_args()

    print("=" * 70)
    print(f"BUILDING V14_QUALITY DATASET (score>={MIN_SCORE_THRESHOLD})")
    print("=" * 70)

    # ── 1. Load v13 base ──────────────────────────────────────────────────────
    print("\n1. Loading v13 base...")
    base = pd.read_csv(args.base, encoding='latin-1')
    for col in ['interaction_type', 'source_species', 'target_species', 'source']:
        if col not in base.columns:
            base[col] = ''
    print(f"   v13: {len(base)} rows ({(base['label']==1).sum()} pos, {(base['label']==0).sum()} neg)")
    existing_texts = set(base['text'].str.strip().str.lower())

    parts = [base]

    # ── 2. sibils_diverse_real ────────────────────────────────────────────────
    print("\n2. Adding sibils_diverse_real...")
    sibils = pd.read_csv(args.sibils, encoding='latin-1')
    sibils_out = add_source(sibils, existing_texts, 'sibils_diverse',
                            filter_positives=True, max_pos_per_type=args.max_pos_per_type,
                            neg_per_pos=1.5)
    existing_texts.update(sibils_out['text'].str.strip().str.lower())
    parts.append(sibils_out)
    print(f"   → Adding {len(sibils_out)} rows")

    # ── 3. globi_sibils_real ──────────────────────────────────────────────────
    print("\n3. Adding globi_sibils_real (stricter filter)...")
    globi = pd.read_csv(args.globi, encoding='latin-1')
    globi_out = add_source(globi, existing_texts, 'globi_sibils',
                           filter_positives=True, max_pos_per_type=args.max_pos_per_type,
                           neg_per_pos=1.5)
    existing_texts.update(globi_out['text'].str.strip().str.lower())
    parts.append(globi_out)
    print(f"   → Adding {len(globi_out)} rows")

    # ── 4. student_set_high_quality (pre-validated, no filter) ───────────────
    print("\n4. Adding student_set_high_quality (pre-validated)...")
    student = pd.read_csv(args.student, encoding='latin-1')
    if 'sentence' in student.columns and 'text' not in student.columns:
        student = student.rename(columns={'sentence': 'text'})
    for col in ['interaction_type', 'source_species', 'target_species']:
        if col not in student.columns:
            student[col] = ''
    student = student[~student['text'].str.strip().str.lower().isin(existing_texts)].copy()
    student['source'] = 'student_curated'
    parts.append(student)
    print(f"   → Adding {len(student)} rows")

    # ── 5. Merge ──────────────────────────────────────────────────────────────
    print("\n5. Merging and shuffling...")
    combined = pd.concat(parts, ignore_index=True)
    combined = combined[['text', 'label', 'interaction_type',
                         'source_species', 'target_species', 'source']].copy()
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    combined.to_csv(args.output, index=False)

    total_pos = (combined['label'] == 1).sum()
    total_neg = (combined['label'] == 0).sum()
    ratio = total_neg / total_pos if total_pos else 0

    print("\n" + "=" * 70)
    print("V14_QUALITY DATASET SUMMARY")
    print("=" * 70)
    print(f"Total:     {len(combined)}")
    print(f"Positives: {total_pos} ({100*total_pos/len(combined):.1f}%)")
    print(f"Negatives: {total_neg} ({100*total_neg/len(combined):.1f}%)")
    print(f"Neg:Pos:   {ratio:.2f}  {'OK' if 1.5 <= ratio <= 3.0 else 'WARNING'}")
    print(f"\nBy source:")
    for src, cnt in combined['source'].value_counts().items():
        p = (combined['source'] == src) & (combined['label'] == 1)
        n = (combined['source'] == src) & (combined['label'] == 0)
        print(f"  {src:<30s}: {cnt:6d}  ({p.sum()} pos, {n.sum()} neg)")
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
