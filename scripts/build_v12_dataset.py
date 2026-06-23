#!/usr/bin/env python3
"""
Build v12 Dataset: v7 (LLM-validated) + rule-filtered harvest positives

Key improvements over v11:
  - Rule-based quality filter: only positives with interaction_lexicon score > 0.0
    are included.  Removes sentences with zero interaction signal (heavy metals
    papers, pure methodology, species co-occurrence without any verb).
  - external_db_sentences.csv added as 4th source (was absent from v11).
    It adds ~71 diverse positives (preysOn / parasiteOf / pollinates / eats /
    mutualistOf from Mangal / Web-of-Life / OpenAlex).
  - Neg:pos ratio kept ~2.4–2.5 via the same allocation strategy as v11.

Why v11 regressed (F1=0.745 regularized vs v7 F1=0.788):
  - 34.6 % of epmc_direct positives had score=0.0 (no interaction verb at all).
  - 47.0 % of external_db positives were heavy-metals / feed-additive papers.
  - Including these noisy positives confuses the boundary between pos/neg classes.
  - Precision stayed low (0.657), recall high (0.863) → over-predicts positives.

v12 strategy:
  - Keep only positives where interaction_lexicon.score_sentence() returns
    strength > 0.0 (at least one interaction term matched, even if penalised).
  - Cap epmc positives at --max-pos-per-type (default 300) per interaction_type.
  - Same negative strategy: 60 % other_species, 30 % single_species_with_signal,
    10 % other; cap at --max-epmc-neg (default 1500).
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Allow running from project root or from classifier/ subdir
for _candidate in [
    Path(__file__).parent.parent / "src",
    Path(__file__).parent / "src",
]:
    if (_candidate / "data" / "interaction_lexicon.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from data.interaction_lexicon import score_sentence  # noqa: E402

# ---------------------------------------------------------------------------
# False-negative filter (same as v11 — catches mislabelled negatives that
# are actually real interactions).
# ---------------------------------------------------------------------------
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
    """Return True if a label=0 sentence looks like a real interaction (drop it)."""
    t = text.lower()
    if any(p.search(t) for p in _NEGATION_PATTERNS):
        return False
    return any(p.search(t) for p in _FALSE_NEG_PATTERNS)


def _has_interaction_signal(text: str) -> bool:
    """Return True if at least one interaction term matched (score > 0)."""
    _, strength, _ = score_sentence(str(text).lower())
    return strength > 0.0


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path('/path/to/MetaP/classifier')

V7_FILE     = BASE_DIR / 'data/training/training_data_globi_v7_llm_cleaned.csv'
EPMC_FILE   = BASE_DIR / 'data/training/epmc_direct_sentences.csv'
V2_FILE     = BASE_DIR / 'data/training/globi_pmc_sentences_v2.csv'
EXTDB_FILE  = BASE_DIR / 'data/training/external_db_sentences.csv'
OUTPUT_FILE = BASE_DIR / 'data/training/training_data_v12.csv'


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v12 training dataset")
    parser.add_argument("--v7",     default=str(V7_FILE),     help="v7 LLM-cleaned data")
    parser.add_argument("--epmc",   default=str(EPMC_FILE),   help="epmc_direct_sentences.csv")
    parser.add_argument("--v2",     default=str(V2_FILE),     help="globi_pmc_sentences_v2.csv")
    parser.add_argument("--extdb",  default=str(EXTDB_FILE),  help="external_db_sentences.csv")
    parser.add_argument("--output", default=str(OUTPUT_FILE), help="Output path")
    parser.add_argument("--max-pos-per-type", type=int, default=300,
                        help="Max epmc positives per interaction_type (default: 300)")
    parser.add_argument("--max-epmc-neg", type=int, default=1500,
                        help="Max epmc negatives to include (default: 1500)")
    parser.add_argument("--extra-sources", nargs="+", default=[],
                        help="Additional pre-validated CSV files to include "
                             "(e.g. ep_curation_2024.csv sibils_triplets_mined.csv). "
                             "Must have columns: text, label, interaction_type. "
                             "No score>0 filter applied — assumed pre-validated.")
    args = parser.parse_args()

    print("=" * 70)
    print("BUILDING V12 DATASET  (rule-filtered positives + external_db)")
    print("=" * 70)

    # ---- Step 1: Load v7 (LLM-validated baseline) ----
    print("\n1. Loading v7 (LLM-validated baseline)...")
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

    pos_epmc = epmc[epmc['label'] == 1].copy()
    neg_epmc = epmc[epmc['label'] == 0].copy()

    # 2a. Rule-based quality filter on positives (score > 0.0)
    before_filter = len(pos_epmc)
    signal_mask = pos_epmc['text'].apply(_has_interaction_signal)
    pos_epmc = pos_epmc[signal_mask].copy()
    print(f"   Positives after score>0 filter: {len(pos_epmc)}/{before_filter} "
          f"({100*len(pos_epmc)/before_filter:.1f}% kept, {before_filter-len(pos_epmc)} removed)")

    print(f"   Positives by interaction_type:")
    for t, c in pos_epmc['interaction_type'].value_counts().items():
        print(f"     {t}: {c}")

    # 2b. Cap positives per interaction_type
    capped_pos = pos_epmc.groupby('interaction_type').apply(
        lambda g: g.sample(n=min(len(g), args.max_pos_per_type), random_state=42)
    ).reset_index(drop=True)
    print(f"\n   Positives after cap ({args.max_pos_per_type}/type): {len(capped_pos)}")
    for t, c in capped_pos['interaction_type'].value_counts().items():
        print(f"     {t}: {c}")

    # 2c. Sample negatives — prioritise hard negatives
    neg_other  = neg_epmc[neg_epmc['source'].str.contains('other_species',  na=False)].copy()
    neg_signal = neg_epmc[neg_epmc['source'].str.contains('single_species_with_signal', na=False)].copy()
    neg_rest   = neg_epmc[
        ~neg_epmc['source'].str.contains('other_species',  na=False) &
        ~neg_epmc['source'].str.contains('single_species_with_signal', na=False)
    ].copy()

    budget_other  = min(len(neg_other),  int(args.max_epmc_neg * 0.60))
    budget_signal = min(len(neg_signal), int(args.max_epmc_neg * 0.30))
    budget_rest   = min(len(neg_rest),   args.max_epmc_neg - budget_other - budget_signal)

    sampled_neg = pd.concat([
        neg_other.sample( n=budget_other,  random_state=42),
        neg_signal.sample(n=budget_signal, random_state=42),
        neg_rest.sample(  n=budget_rest,   random_state=42),
    ], ignore_index=True)
    print(f"\n   Negatives sampled: {len(sampled_neg)}")
    print(f"     other_species:             {budget_other}")
    print(f"     single_species_with_signal: {budget_signal}")
    print(f"     other types:               {budget_rest}")

    # 2d. Build epmc output df
    epmc_rows = []
    for part, is_pos in [(capped_pos, True), (sampled_neg, False)]:
        for _, row in part.iterrows():
            if is_pos:
                itype = row.get('interaction_type', 'unknown')
            else:
                src = str(row.get('source', ''))
                itype = 'none_two_species' if ('other_species' in src or 'both_species' in src) else 'none'
            epmc_rows.append({
                'text': row['text'],
                'label': row['label'],
                'interaction_type': itype,
                'source_species': row.get('source_species', ''),
                'target_species': row.get('target_species', ''),
                'source': 'epmc_direct',
            })
    epmc_out = pd.DataFrame(epmc_rows)

    # 2e. Filter incomplete + false negatives + dedup
    before = len(epmc_out)
    epmc_out = epmc_out[epmc_out['text'].str.strip().str.endswith(('.', '!', '?', ')'))].copy()
    neg_mask  = epmc_out['label'] == 0
    fn_mask   = neg_mask & epmc_out['text'].apply(_is_false_negative)
    if fn_mask.sum():
        print(f"   Removed {fn_mask.sum()} false negatives from epmc pool")
        epmc_out = epmc_out[~fn_mask].copy()
    epmc_out = epmc_out[~epmc_out['text'].str.lower().str.strip().isin(v7_texts)].copy()
    print(f"   Filtered {before - len(epmc_out)} incomplete/dup/fn  →  "
          f"{len(epmc_out)} ({sum(epmc_out['label']==1)} pos, {sum(epmc_out['label']==0)} neg)")

    # ---- Step 3: Load and process GloBI-PMC v2 ----
    print("\n3. Loading GloBI-PMC v2 sentences...")
    v2 = pd.read_csv(args.v2)
    print(f"   Total: {len(v2)}  ({sum(v2['label']==1)} pos, {sum(v2['label']==0)} neg)")

    # Rule-based quality filter on positives
    v2_pos = v2[v2['label'] == 1].copy()
    v2_neg = v2[v2['label'] == 0].copy()
    before_v2 = len(v2_pos)
    v2_pos = v2_pos[v2_pos['text'].apply(_has_interaction_signal)].copy()
    print(f"   Positives after score>0 filter: {len(v2_pos)}/{before_v2} "
          f"({100*len(v2_pos)/before_v2:.1f}% kept)")

    v2_rows = []
    for part, is_pos in [(v2_pos, True), (v2_neg, False)]:
        for _, row in part.iterrows():
            if is_pos:
                itype = row.get('interaction_type', 'unknown')
            else:
                src = str(row.get('source', ''))
                itype = 'none_two_species' if ('other_species' in src or 'both_species' in src) else 'none'
            v2_rows.append({
                'text': row['text'],
                'label': row['label'],
                'interaction_type': itype,
                'source_species': row.get('source_species', ''),
                'target_species': row.get('target_species', ''),
                'source': 'globi_pmc_v2',
            })
    v2_out = pd.DataFrame(v2_rows)
    before = len(v2_out)
    v2_out = v2_out[v2_out['text'].str.strip().str.endswith(('.', '!', '?', ')'))].copy()
    fn_mask_v2 = (v2_out['label'] == 0) & v2_out['text'].apply(_is_false_negative)
    if fn_mask_v2.sum():
        print(f"   Removed {fn_mask_v2.sum()} false negatives from v2 pool")
        v2_out = v2_out[~fn_mask_v2].copy()
    v2_out = v2_out[~v2_out['text'].str.lower().str.strip().isin(v7_texts)].copy()
    print(f"   After filter+dedup: {len(v2_out)} ({sum(v2_out['label']==1)} pos, {sum(v2_out['label']==0)} neg)"
          f"  [removed {before - len(v2_out)}]")

    # ---- Step 4: Load and process external_db (NEW in v12) ----
    print("\n4. Loading external_db sentences (new in v12)...")
    extdb = pd.read_csv(args.extdb)
    print(f"   Total: {len(extdb)}  ({sum(extdb['label']==1)} pos, {sum(extdb['label']==0)} neg)")
    if 'interaction_type' in extdb.columns:
        print(f"   Pos interaction_type: {extdb[extdb['label']==1]['interaction_type'].value_counts().to_dict()}")

    extdb_pos = extdb[extdb['label'] == 1].copy()
    extdb_neg = extdb[extdb['label'] == 0].copy()

    # Rule-based quality filter on positives
    before_extdb = len(extdb_pos)
    extdb_pos = extdb_pos[extdb_pos['text'].apply(_has_interaction_signal)].copy()
    print(f"   Positives after score>0 filter: {len(extdb_pos)}/{before_extdb} "
          f"({100*len(extdb_pos)/before_extdb:.1f}% kept, "
          f"{before_extdb-len(extdb_pos)} noise removed)")

    extdb_rows = []
    for part, is_pos in [(extdb_pos, True), (extdb_neg, False)]:
        for _, row in part.iterrows():
            if is_pos:
                itype = row.get('interaction_type', row.get('category', 'unknown'))
            else:
                itype = 'none'
            extdb_rows.append({
                'text': row['text'],
                'label': row['label'],
                'interaction_type': itype,
                'source_species': row.get('source_species', ''),
                'target_species': row.get('target_species', ''),
                'source': 'external_db',
            })
    extdb_out = pd.DataFrame(extdb_rows)
    before = len(extdb_out)
    extdb_out = extdb_out[extdb_out['text'].str.strip().str.endswith(('.', '!', '?', ')'))].copy()
    fn_mask_extdb = (extdb_out['label'] == 0) & extdb_out['text'].apply(_is_false_negative)
    if fn_mask_extdb.sum():
        print(f"   Removed {fn_mask_extdb.sum()} false negatives from extdb pool")
        extdb_out = extdb_out[~fn_mask_extdb].copy()
    extdb_out = extdb_out[~extdb_out['text'].str.lower().str.strip().isin(v7_texts)].copy()
    print(f"   After filter+dedup: {len(extdb_out)} ({sum(extdb_out['label']==1)} pos, "
          f"{sum(extdb_out['label']==0)} neg)  [removed {before - len(extdb_out)}]")

    # ---- Step 5: Load extra pre-validated sources (e.g. ep_curation_2024, sibils mined) ----
    extra_parts = []
    if args.extra_sources:
        all_existing_texts = set()
        for part in [v7, epmc_out, v2_out, extdb_out]:
            all_existing_texts.update(part['text'].str.lower().str.strip())

        for src_path in args.extra_sources:
            p = Path(src_path)
            if not p.exists():
                print(f"\nWARNING: --extra-sources file not found, skipping: {p}")
                continue
            print(f"\n5+. Loading extra source: {p.name}")
            extra = pd.read_csv(str(p))

            # Normalise column names (some extra sources use 'sentence' not 'text')
            if 'sentence' in extra.columns and 'text' not in extra.columns:
                extra = extra.rename(columns={'sentence': 'text'})

            required = {'text', 'label'}
            missing = required - set(extra.columns)
            if missing:
                print(f"   SKIP: missing required columns {missing}")
                continue

            for col in ['interaction_type', 'source_species', 'target_species']:
                if col not in extra.columns:
                    extra[col] = ''
            if 'source' not in extra.columns:
                extra['source'] = p.stem

            extra = extra[['text', 'label', 'interaction_type',
                           'source_species', 'target_species', 'source']].copy()

            # Dedup against all existing texts
            before_extra = len(extra)
            extra = extra[~extra['text'].str.lower().str.strip().isin(all_existing_texts)].copy()
            all_existing_texts.update(extra['text'].str.lower().str.strip())
            n_pos = int((extra['label'] == 1).sum())
            n_neg = int((extra['label'] == 0).sum())
            print(f"   {len(extra)}/{before_extra} rows after dedup "
                  f"({n_pos} pos, {n_neg} neg)")
            extra_parts.append(extra)

    # ---- Step 6: Merge and shuffle ----
    print("\n6. Merging and shuffling...")
    parts_to_merge = [v7, epmc_out, v2_out, extdb_out] + extra_parts
    combined = pd.concat(parts_to_merge, ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # ---- Step 7: Save ----
    combined.to_csv(args.output, index=False)

    # ---- Summary ----
    total_pos = sum(combined['label'] == 1)
    total_neg = sum(combined['label'] == 0)
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
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
    for t, c in combined[combined['label'] == 1]['interaction_type'].value_counts().head(15).items():
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
