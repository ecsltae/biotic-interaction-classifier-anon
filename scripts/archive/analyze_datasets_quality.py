#!/usr/bin/env python3
"""
Analyze the quality difference between 6k original dataset and 20k dataset.
Understand why 6k produces better results on eval100.
"""

import csv
import re
from collections import Counter
from pathlib import Path

BASE_DIR = Path("/path/to/MetaP/classifier")

# Load datasets
def load_dataset(path):
    positives = []
    negatives = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text_col = 'passage' if 'passage' in row else 'text'
            if int(row['label']) == 1:
                positives.append(row[text_col])
            else:
                negatives.append(row[text_col])
    return positives, negatives

print("="*80)
print("DATASET QUALITY ANALYSIS")
print("="*80)

# Load both datasets
print("\nLoading datasets...")
pos_6k, neg_6k = load_dataset(BASE_DIR / "data/training/training_data_cleaned.csv")
pos_20k, neg_20k = load_dataset(BASE_DIR / "data/training/training_data_improved_20k.csv")

print(f"\n6k Dataset:")
print(f"  Positives: {len(pos_6k)}")
print(f"  Negatives: {len(neg_6k)}")

print(f"\n20k Dataset:")
print(f"  Positives: {len(pos_20k)}")
print(f"  Negatives: {len(neg_20k)}")

# Strong interaction keywords
STRONG_KEYWORDS = [
    'infect', 'infection', 'infected', 'infecting',
    'parasite', 'parasites', 'parasitic', 'parasitize', 'parasitized',
    'pathogen', 'pathogens', 'pathogenic',
    'vector', 'vectors',
    'host', 'hosts',
    'prey', 'predator',
    'transmitted', 'transmission',
    'symbiont', 'symbiosis',
]

# Noisy terms that might indicate non-interactions
NOISY_KEYWORDS = [
    'phylogenet', 'taxonom', 'classific',
    'sequenc', 'genome', 'pcr', 'amplif',
    'morpholog', 'anatom',
    'distribut', 'habitat', 'range',
    'evolution', 'divergen',
]

def analyze_positives(positives, name):
    """Analyze quality of positive examples"""
    strong_count = 0
    noisy_count = 0

    strong_examples = []
    noisy_examples = []

    for sent in positives:
        s = sent.lower()
        has_strong = any(kw in s for kw in STRONG_KEYWORDS)
        has_noisy = any(kw in s for kw in NOISY_KEYWORDS)

        if has_strong:
            strong_count += 1
            if len(strong_examples) < 3:
                strong_examples.append(sent[:100])
        if has_noisy and not has_strong:
            noisy_count += 1
            if len(noisy_examples) < 3:
                noisy_examples.append(sent[:100])

    print(f"\n{name} POSITIVES:")
    print(f"  With strong interaction keywords: {strong_count} ({100*strong_count/len(positives):.1f}%)")
    print(f"  With noisy keywords (no strong): {noisy_count} ({100*noisy_count/len(positives):.1f}%)")

    return strong_count / len(positives), noisy_count / len(positives)

def analyze_negatives(negatives, name):
    """Analyze negative examples - look for potential false negatives"""
    false_neg_count = 0
    false_neg_examples = []

    for sent in negatives:
        s = sent.lower()
        # Strong interaction term in a negative = potential false negative
        if any(kw in s for kw in ['infect', 'parasite', 'pathogen', 'prey on', 'feeds on']):
            false_neg_count += 1
            if len(false_neg_examples) < 5:
                false_neg_examples.append(sent[:100])

    print(f"\n{name} NEGATIVES:")
    print(f"  Potential false negatives: {false_neg_count} ({100*false_neg_count/len(negatives):.1f}%)")
    if false_neg_examples:
        print("  Examples:")
        for i, ex in enumerate(false_neg_examples, 1):
            print(f"    {i}. {ex}...")

    return false_neg_count / len(negatives)

# Analyze 6k dataset
print("\n" + "="*80)
print("6K DATASET ANALYSIS")
print("="*80)
strong_6k, noisy_6k = analyze_positives(pos_6k, "6k")
fn_rate_6k = analyze_negatives(neg_6k, "6k")

# Analyze 20k dataset
print("\n" + "="*80)
print("20K DATASET ANALYSIS")
print("="*80)
strong_20k, noisy_20k = analyze_positives(pos_20k, "20k")
fn_rate_20k = analyze_negatives(neg_20k, "20k")

# Compare
print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"""
                            6k Dataset    20k Dataset
Positives with strong kw:   {100*strong_6k:.1f}%          {100*strong_20k:.1f}%
Positives with noisy kw:    {100*noisy_6k:.1f}%          {100*noisy_20k:.1f}%
Negatives as false neg:     {100*fn_rate_6k:.1f}%          {100*fn_rate_20k:.1f}%

CONCLUSION:
- 6k dataset has {100*(strong_6k-strong_20k):.1f}% more positives with strong keywords
- 6k dataset has {100*(noisy_6k-noisy_20k):.1f}% more noisy positives
- 6k negatives have {100*(fn_rate_6k-fn_rate_20k):.1f}% more potential false negatives
""")

# Check overlap between datasets
print("\n" + "="*80)
print("OVERLAP ANALYSIS")
print("="*80)

# How many 6k positives are in 20k?
pos_6k_set = set(p.lower().strip() for p in pos_6k)
pos_20k_set = set(p.lower().strip() for p in pos_20k)

overlap = pos_6k_set & pos_20k_set
only_6k = pos_6k_set - pos_20k_set
only_20k = pos_20k_set - pos_6k_set

print(f"Positives overlap: {len(overlap)}")
print(f"Only in 6k: {len(only_6k)}")
print(f"Only in 20k: {len(only_20k)}")

# Sample positives only in 6k
print("\nSample positives ONLY in 6k (may be higher quality):")
for i, sent in enumerate(list(only_6k)[:5], 1):
    print(f"  {i}. {sent[:100]}...")

# Sample positives only in 20k
print("\nSample positives ONLY in 20k (may be lower quality):")
for i, sent in enumerate(list(only_20k)[:5], 1):
    print(f"  {i}. {sent[:100]}...")

print("\n" + "="*80)
