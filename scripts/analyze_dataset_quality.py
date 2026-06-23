#!/usr/bin/env python3
"""
Analyze what makes the 6k dataset better than the 20k dataset.
Look at:
1. Positive quality - do they clearly describe biotic interactions?
2. Negative quality - are they truly non-interactions?
3. Length distribution
4. Interaction keyword patterns
5. Species mention patterns
"""

import pandas as pd
import numpy as np
from collections import Counter
import re

BASE_DIR = '/path/to/MetaP/classifier'

print("="*80)
print("DATASET QUALITY ANALYSIS: 6k vs 20k")
print("="*80)

# Load datasets
df_6k = pd.read_csv(f'{BASE_DIR}/data/training/training_data_cleaned.csv')
df_20k = pd.read_csv(f'{BASE_DIR}/data/training/training_data_improved_20k.csv')

print(f"\n6k dataset: {len(df_6k)} samples")
print(f"  Positives: {sum(df_6k['label']==1)}")
print(f"  Negatives: {sum(df_6k['label']==0)}")

print(f"\n20k dataset: {len(df_20k)} samples")
print(f"  Positives: {sum(df_20k['label']==1)}")
print(f"  Negatives: {sum(df_20k['label']==0)}")

# Strong interaction keywords
STRONG_INTERACTION = [
    'infect', 'infection', 'infected', 'infecting', 'infectious',
    'parasite', 'parasites', 'parasitic', 'parasitize', 'parasitized', 'parasitism',
    'pathogen', 'pathogens', 'pathogenic',
    'host', 'hosts', 'hosted',
    'vector', 'vectors', 'transmitted',
    'prey', 'preys', 'preyed', 'predator', 'predators', 'predation',
    'symbiont', 'symbionts', 'symbiosis', 'symbiotic',
    'mutualist', 'mutualism', 'mutualistic',
    'feeds on', 'fed on', 'feeding on',
    'colonize', 'colonized', 'colonization',
    'infestation', 'infest', 'infested',
]

# Weak/noisy interaction terms (context-dependent)
WEAK_INTERACTION = [
    'ate', 'eat', 'eats', 'eaten', 'eating',
    'found in', 'found on', 'associated with',
    'isolated from', 'collected from',
    'detected in', 'present in',
]

# Non-interaction context keywords
NON_INTERACTION = [
    'phylogenet', 'taxonom', 'systemat', 'classif',
    'sequenc', 'genom', 'pcr', 'amplif', 'primer',
    'morpholog', 'anatomic', 'structur',
    'distribut', 'range', 'habitat', 'ecosystem',
    'conserv', 'endanger', 'extinct',
    'fossil', 'evolution', 'diverge',
]

def count_keywords(text, keywords):
    text = text.lower()
    return sum(1 for kw in keywords if kw in text)

def has_species_pattern(text):
    """Check if text has species-like patterns (Genus species)"""
    # Pattern: Capital letter followed by lowercase, space, lowercase word
    pattern = r'\b[A-Z][a-z]+\s+[a-z]+\b'
    matches = re.findall(pattern, text)
    return len(matches)

# Analyze 6k positives vs 20k positives
print("\n" + "="*80)
print("POSITIVE SAMPLE ANALYSIS")
print("="*80)

pos_6k = df_6k[df_6k['label']==1]['passage'].tolist()
pos_20k = df_20k[df_20k['label']==1]['passage'].tolist()

# Count strong interaction keywords
strong_6k = [count_keywords(p, STRONG_INTERACTION) for p in pos_6k]
strong_20k = [count_keywords(p, STRONG_INTERACTION) for p in pos_20k]

print(f"\nStrong interaction keywords per positive:")
print(f"  6k:  mean={np.mean(strong_6k):.2f}, median={np.median(strong_6k):.0f}, >0: {100*sum(1 for s in strong_6k if s>0)/len(strong_6k):.1f}%")
print(f"  20k: mean={np.mean(strong_20k):.2f}, median={np.median(strong_20k):.0f}, >0: {100*sum(1 for s in strong_20k if s>0)/len(strong_20k):.1f}%")

# Count weak interaction keywords
weak_6k = [count_keywords(p, WEAK_INTERACTION) for p in pos_6k]
weak_20k = [count_keywords(p, WEAK_INTERACTION) for p in pos_20k]

print(f"\nWeak/noisy interaction keywords per positive:")
print(f"  6k:  mean={np.mean(weak_6k):.2f}, >0: {100*sum(1 for s in weak_6k if s>0)/len(weak_6k):.1f}%")
print(f"  20k: mean={np.mean(weak_20k):.2f}, >0: {100*sum(1 for s in weak_20k if s>0)/len(weak_20k):.1f}%")

# Species patterns
species_6k = [has_species_pattern(p) for p in pos_6k]
species_20k = [has_species_pattern(p) for p in pos_20k]

print(f"\nSpecies mentions per positive:")
print(f"  6k:  mean={np.mean(species_6k):.2f}, >=2: {100*sum(1 for s in species_6k if s>=2)/len(species_6k):.1f}%")
print(f"  20k: mean={np.mean(species_20k):.2f}, >=2: {100*sum(1 for s in species_20k if s>=2)/len(species_20k):.1f}%")

# Length analysis
len_6k = [len(p.split()) for p in pos_6k]
len_20k = [len(p.split()) for p in pos_20k]

print(f"\nSentence length (words) for positives:")
print(f"  6k:  mean={np.mean(len_6k):.1f}, median={np.median(len_6k):.0f}")
print(f"  20k: mean={np.mean(len_20k):.1f}, median={np.median(len_20k):.0f}")

# Analyze negatives
print("\n" + "="*80)
print("NEGATIVE SAMPLE ANALYSIS")
print("="*80)

neg_6k = df_6k[df_6k['label']==0]['passage'].tolist()
neg_20k = df_20k[df_20k['label']==0]['passage'].tolist()

# Strong interaction keywords in negatives (potential false negatives!)
strong_neg_6k = [count_keywords(p, STRONG_INTERACTION) for p in neg_6k]
strong_neg_20k = [count_keywords(p, STRONG_INTERACTION) for p in neg_20k]

print(f"\nStrong interaction keywords in negatives (potential FN):")
print(f"  6k:  >0: {100*sum(1 for s in strong_neg_6k if s>0)/len(strong_neg_6k):.1f}%, mean={np.mean(strong_neg_6k):.2f}")
print(f"  20k: >0: {100*sum(1 for s in strong_neg_20k if s>0)/len(strong_neg_20k):.1f}%, mean={np.mean(strong_neg_20k):.2f}")

# Non-interaction context in negatives (good!)
non_int_6k = [count_keywords(p, NON_INTERACTION) for p in neg_6k]
non_int_20k = [count_keywords(p, NON_INTERACTION) for p in neg_20k]

print(f"\nNon-interaction context keywords in negatives (good):")
print(f"  6k:  >0: {100*sum(1 for s in non_int_6k if s>0)/len(non_int_6k):.1f}%, mean={np.mean(non_int_6k):.2f}")
print(f"  20k: >0: {100*sum(1 for s in non_int_20k if s>0)/len(non_int_20k):.1f}%, mean={np.mean(non_int_20k):.2f}")

# Sample problematic cases
print("\n" + "="*80)
print("SAMPLE PROBLEMATIC CASES")
print("="*80)

print("\n--- 20k Positives with NO strong interaction keywords ---")
weak_positives = [(i, p) for i, p in enumerate(pos_20k) if count_keywords(p, STRONG_INTERACTION) == 0]
print(f"Count: {len(weak_positives)} ({100*len(weak_positives)/len(pos_20k):.1f}%)")
for i, (idx, p) in enumerate(weak_positives[:5]):
    print(f"\n{i+1}. {p[:200]}...")

print("\n--- 20k Negatives with strong interaction keywords ---")
false_neg_candidates = [(i, p, count_keywords(p, STRONG_INTERACTION)) for i, p in enumerate(neg_20k)
                        if count_keywords(p, STRONG_INTERACTION) >= 2]
print(f"Count: {len(false_neg_candidates)} ({100*len(false_neg_candidates)/len(neg_20k):.1f}%)")
for i, (idx, p, score) in enumerate(sorted(false_neg_candidates, key=lambda x: -x[2])[:5]):
    print(f"\n{i+1}. [score={score}] {p[:200]}...")

# Quality score
print("\n" + "="*80)
print("QUALITY SCORING")
print("="*80)

def quality_score_positive(text):
    """Score a positive sample - higher is better"""
    score = 0
    # Strong interaction keywords (very good)
    score += count_keywords(text, STRONG_INTERACTION) * 2
    # Multiple species mentions (good)
    score += min(has_species_pattern(text), 3)
    # Penalize weak-only interactions
    if count_keywords(text, STRONG_INTERACTION) == 0 and count_keywords(text, WEAK_INTERACTION) > 0:
        score -= 1
    # Penalize if mostly non-interaction context
    if count_keywords(text, NON_INTERACTION) > count_keywords(text, STRONG_INTERACTION):
        score -= 2
    return score

def quality_score_negative(text):
    """Score a negative sample - higher is better (more confident it's truly negative)"""
    score = 0
    # Non-interaction context (good)
    score += count_keywords(text, NON_INTERACTION)
    # Penalize strong interaction keywords (potential false negative)
    score -= count_keywords(text, STRONG_INTERACTION) * 3
    # Single or no species is OK for negative
    if has_species_pattern(text) <= 1:
        score += 1
    return score

# Score all samples
pos_scores_6k = [quality_score_positive(p) for p in pos_6k]
pos_scores_20k = [quality_score_positive(p) for p in pos_20k]
neg_scores_6k = [quality_score_negative(p) for p in neg_6k]
neg_scores_20k = [quality_score_negative(p) for p in neg_20k]

print(f"\nPositive quality scores:")
print(f"  6k:  mean={np.mean(pos_scores_6k):.2f}, median={np.median(pos_scores_6k):.0f}, >=2: {100*sum(1 for s in pos_scores_6k if s>=2)/len(pos_scores_6k):.1f}%")
print(f"  20k: mean={np.mean(pos_scores_20k):.2f}, median={np.median(pos_scores_20k):.0f}, >=2: {100*sum(1 for s in pos_scores_20k if s>=2)/len(pos_scores_20k):.1f}%")

print(f"\nNegative quality scores:")
print(f"  6k:  mean={np.mean(neg_scores_6k):.2f}, median={np.median(neg_scores_6k):.0f}, >=0: {100*sum(1 for s in neg_scores_6k if s>=0)/len(neg_scores_6k):.1f}%")
print(f"  20k: mean={np.mean(neg_scores_20k):.2f}, median={np.median(neg_scores_20k):.0f}, >=0: {100*sum(1 for s in neg_scores_20k if s>=0)/len(neg_scores_20k):.1f}%")

# Recommendations
print("\n" + "="*80)
print("RECOMMENDATIONS FOR IMPROVED 20k DATASET")
print("="*80)

low_quality_pos = sum(1 for s in pos_scores_20k if s < 2)
potential_fn = sum(1 for s in neg_scores_20k if s < -2)

print(f"""
1. LOW QUALITY POSITIVES TO REMOVE: {low_quality_pos} ({100*low_quality_pos/len(pos_20k):.1f}%)
   - Positives without strong interaction keywords
   - Positives dominated by non-interaction context

2. POTENTIAL FALSE NEGATIVES TO PROMOTE: {potential_fn} ({100*potential_fn/len(neg_20k):.1f}%)
   - Negatives with multiple strong interaction keywords

3. STRATEGY:
   - Keep positives with quality_score >= 2
   - Remove negatives with quality_score < -2 (promote to positive or discard)
   - Add hard negatives: sentences with interaction terms but no actual interaction
""")

# Save quality scores for further analysis
quality_df = pd.DataFrame({
    'passage': pos_20k + neg_20k,
    'label': [1]*len(pos_20k) + [0]*len(neg_20k),
    'quality_score': pos_scores_20k + neg_scores_20k,
    'strong_kw': [count_keywords(p, STRONG_INTERACTION) for p in pos_20k + neg_20k],
    'species_count': [has_species_pattern(p) for p in pos_20k + neg_20k],
})
quality_df.to_csv(f'{BASE_DIR}/analysis/dataset_quality_scores.csv', index=False)
print(f"\nQuality scores saved to: analysis/dataset_quality_scores.csv")
