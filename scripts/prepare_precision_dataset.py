#!/usr/bin/env python3
"""
Prepare High-Quality Dataset for Precision-Focused Training
============================================================

Strategy:
1. Keep only HIGH-CONFIDENCE positives (clear interaction keywords)
2. Include HARD negatives (sentences with some interaction words but no actual interaction)
3. Clean negatives (clearly non-interaction sentences)
4. Balance carefully to avoid bias

Goal: Train a model that says "interaction" only when it's really confident
"""

import csv
import random
import re
from pathlib import Path
from collections import Counter

random.seed(42)

BASE_DIR = Path("/path/to/MetaP/classifier")
DATA_DIR = BASE_DIR / "data/training"

print("="*70)
print("PREPARING PRECISION-FOCUSED DATASET")
print("="*70)

# =============================================================================
# INTERACTION KEYWORDS (hierarchical)
# =============================================================================

# Level 1: VERY STRONG - almost always indicates biotic interaction
KEYWORDS_L1 = [
    'infect', 'infection', 'infected', 'infecting', 'infectious',
    'parasite', 'parasites', 'parasitic', 'parasitize', 'parasitized', 'parasitism',
    'pathogen', 'pathogens', 'pathogenic', 'pathogenicity',
    'vector', 'vectors', 'vectored', 'vectoring',
    'virulent', 'virulence', 'avirulent',
]

# Level 2: STRONG - usually indicates interaction
KEYWORDS_L2 = [
    'host', 'hosts', 'hosted', 'hosting', 'host-parasite', 'host-pathogen',
    'prey', 'preys', 'preyed', 'preying', 'predator', 'predators', 'predation', 'predatory',
    'symbiont', 'symbionts', 'symbiosis', 'symbiotic', 'endosymbiont',
    'mutualist', 'mutualism', 'mutualistic',
    'transmitted', 'transmit', 'transmission', 'transmitting',
    'colonize', 'colonized', 'colonization', 'colonizing',
    'infestation', 'infest', 'infested', 'infesting',
]

# Level 3: MODERATE - needs context
KEYWORDS_L3 = [
    'feeds on', 'fed on', 'feeding on', 'feed on',
    'preys on', 'preys upon', 'preyed on', 'preyed upon',
    'attacks', 'attacked', 'attacking',
    'disease', 'diseases', 'diseased',
    'epidemic', 'endemic', 'outbreak',
]

# Interaction phrases (very strong signal)
INTERACTION_PHRASES = [
    'is a parasite of', 'parasitizes', 'is parasitized by',
    'is a host of', 'is hosted by', 'serves as host', 'acts as host',
    'is a vector of', 'is vectored by', 'transmits', 'transmitted by',
    'preys on', 'preys upon', 'is prey of', 'is preyed upon',
    'infects', 'is infected by', 'causes infection in',
    'feeds on', 'is fed upon by',
    'attacks', 'is attacked by',
    'colonizes', 'is colonized by',
    'causes disease in', 'pathogen of',
]

# NON-INTERACTION contexts (negative signals)
NON_INTERACTION = [
    'phylogenet', 'taxonom', 'systemat', 'classif', 'clade', 'monophyl',
    'sequenc', 'genom', 'pcr', 'amplif', 'primer', 'dna', 'rna', 'gene', 'locus',
    'morpholog', 'anatomic', 'structur', 'ultrastructur',
    'distribut', 'geograph', 'habitat', 'ecosystem', 'biome', 'range',
    'conserv', 'endanger', 'extinct', 'iucn', 'threatened',
    'fossil', 'evolution', 'diverge', 'speciat', 'ancestr',
    'cultivation', 'laboratory', 'in vitro', 'cell line', 'culture',
    'museum', 'specimen', 'holotype', 'voucher',
]


def count_keywords(text, keywords):
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def has_interaction_phrase(text):
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in INTERACTION_PHRASES)


def positive_confidence_score(text):
    """
    Score how confident we are that this is a TRUE positive.
    Higher score = more confident.
    """
    score = 0
    text_lower = text.lower()

    # L1 keywords (very strong)
    l1 = count_keywords(text, KEYWORDS_L1)
    score += l1 * 5

    # L2 keywords (strong)
    l2 = count_keywords(text, KEYWORDS_L2)
    score += l2 * 3

    # L3 keywords (moderate)
    l3 = count_keywords(text, KEYWORDS_L3)
    score += l3 * 1

    # Interaction phrases (very strong)
    if has_interaction_phrase(text):
        score += 6

    # Multiple interaction keywords is a strong signal
    total_interaction = l1 + l2 + l3
    if total_interaction >= 3:
        score += 3

    # Penalize if dominated by non-interaction context
    non_int = count_keywords(text, NON_INTERACTION)
    if non_int > total_interaction:
        score -= (non_int - total_interaction) * 2

    return score


def negative_confidence_score(text):
    """
    Score how confident we are that this is a TRUE negative.
    Higher score = more confident.
    """
    score = 0

    # Non-interaction context is good
    non_int = count_keywords(text, NON_INTERACTION)
    score += non_int * 2

    # Interaction keywords are bad for negatives
    l1 = count_keywords(text, KEYWORDS_L1)
    score -= l1 * 6

    l2 = count_keywords(text, KEYWORDS_L2)
    score -= l2 * 4

    l3 = count_keywords(text, KEYWORDS_L3)
    score -= l3 * 2

    # Interaction phrases are very bad
    if has_interaction_phrase(text):
        score -= 8

    return score


# =============================================================================
# LOAD DATA
# =============================================================================

print("\n[1] Loading source datasets...")

# Original 6k (best performing)
positives_6k, negatives_6k = [], []
with open(DATA_DIR / "training_data_cleaned.csv", 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['label']) == 1:
            positives_6k.append(row['passage'])
        else:
            negatives_6k.append(row['passage'])

print(f"  6k original: {len(positives_6k)} pos, {len(negatives_6k)} neg")

# 20k extended
positives_20k, negatives_20k = [], []
with open(DATA_DIR / "training_data_improved_20k.csv", 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['label']) == 1:
            positives_20k.append(row['passage'])
        else:
            negatives_20k.append(row['passage'])

print(f"  20k extended: {len(positives_20k)} pos, {len(negatives_20k)} neg")

# Quality v2
positives_qv2, negatives_qv2 = [], []
with open(DATA_DIR / "training_data_quality_v2.csv", 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['label']) == 1:
            positives_qv2.append(row['passage'])
        else:
            negatives_qv2.append(row['passage'])

print(f"  Quality v2: {len(positives_qv2)} pos, {len(negatives_qv2)} neg")


# =============================================================================
# BUILD HIGH-CONFIDENCE POSITIVE SET
# =============================================================================

print("\n[2] Building high-confidence positive set...")

# Combine all positives and deduplicate
all_positives = list(set(positives_6k + positives_20k + positives_qv2))
print(f"  Total unique positives: {len(all_positives)}")

# Score each positive
scored_positives = [(p, positive_confidence_score(p)) for p in all_positives]
scored_positives.sort(key=lambda x: -x[1])

# Statistics
scores = [s for p, s in scored_positives]
print(f"  Score distribution: min={min(scores)}, max={max(scores)}, mean={sum(scores)/len(scores):.1f}")

# Keep only HIGH-CONFIDENCE positives (score >= 5)
HIGH_CONF_THRESHOLD = 5
high_conf_positives = [(p, s) for p, s in scored_positives if s >= HIGH_CONF_THRESHOLD]
print(f"  High-confidence positives (score >= {HIGH_CONF_THRESHOLD}): {len(high_conf_positives)}")

# Also keep some medium confidence for diversity
MEDIUM_CONF_THRESHOLD = 3
medium_conf_positives = [(p, s) for p, s in scored_positives if MEDIUM_CONF_THRESHOLD <= s < HIGH_CONF_THRESHOLD]
print(f"  Medium-confidence positives ({MEDIUM_CONF_THRESHOLD} <= score < {HIGH_CONF_THRESHOLD}): {len(medium_conf_positives)}")

# Final positives: all high-conf + sample of medium-conf
final_positives = [p for p, s in high_conf_positives]
random.shuffle(medium_conf_positives)
final_positives.extend([p for p, s in medium_conf_positives[:1000]])  # Add up to 1000 medium

print(f"  Final positives: {len(final_positives)}")


# =============================================================================
# BUILD NEGATIVE SET (mix of hard and clean)
# =============================================================================

print("\n[3] Building negative set...")

# Combine all negatives
all_negatives = list(set(negatives_6k + negatives_20k + negatives_qv2))
print(f"  Total unique negatives: {len(all_negatives)}")

# Score each negative
scored_negatives = [(n, negative_confidence_score(n)) for n in all_negatives]

# Find FALSE NEGATIVES to remove (negatives that are probably actually positive)
false_negatives = [(n, s) for n, s in scored_negatives if s < -5]
print(f"  Potential false negatives (score < -5): {len(false_negatives)} - will be promoted")

# Move false negatives to positives
for n, s in false_negatives:
    if n not in final_positives:
        final_positives.append(n)

print(f"  Positives after promotion: {len(final_positives)}")

# Clean negatives: confident true negatives
clean_negatives = [(n, s) for n, s in scored_negatives if s >= 2]
print(f"  Clean negatives (score >= 2): {len(clean_negatives)}")

# Hard negatives: have some interaction words but are true negatives
# These are valuable for teaching discrimination
hard_negatives = [(n, s) for n, s in scored_negatives if -3 <= s < 2]
print(f"  Hard negatives (-3 <= score < 2): {len(hard_negatives)}")

# Build final negative set
# Mix: 70% clean + 30% hard negatives
target_negatives = len(final_positives)  # Balance with positives

n_clean = int(target_negatives * 0.7)
n_hard = target_negatives - n_clean

random.shuffle(clean_negatives)
random.shuffle(hard_negatives)

final_negatives = [n for n, s in clean_negatives[:n_clean]]
final_negatives.extend([n for n, s in hard_negatives[:n_hard]])

# If not enough, fill with more clean
if len(final_negatives) < target_negatives:
    remaining = target_negatives - len(final_negatives)
    more_clean = [n for n, s in clean_negatives[n_clean:n_clean+remaining]]
    final_negatives.extend(more_clean)

print(f"  Final negatives: {len(final_negatives)}")


# =============================================================================
# BALANCE AND SAVE
# =============================================================================

print("\n[4] Balancing and saving...")

# Balance
min_count = min(len(final_positives), len(final_negatives))
random.shuffle(final_positives)
random.shuffle(final_negatives)

final_positives = final_positives[:min_count]
final_negatives = final_negatives[:min_count]

print(f"  Balanced: {len(final_positives)} positives, {len(final_negatives)} negatives")
print(f"  Total: {len(final_positives) + len(final_negatives)}")

# Compute final quality stats
final_pos_scores = [positive_confidence_score(p) for p in final_positives]
final_neg_scores = [negative_confidence_score(n) for n in final_negatives]

print(f"\n  Positive scores: mean={sum(final_pos_scores)/len(final_pos_scores):.2f}, "
      f"min={min(final_pos_scores)}, max={max(final_pos_scores)}")
print(f"  Negative scores: mean={sum(final_neg_scores)/len(final_neg_scores):.2f}, "
      f"min={min(final_neg_scores)}, max={max(final_neg_scores)}")

# Save
output_file = DATA_DIR / "training_data_precision.csv"
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['passage', 'label'])
    for p in final_positives:
        writer.writerow([p, 1])
    for n in final_negatives:
        writer.writerow([n, 0])

print(f"\n  Saved to: {output_file}")


# =============================================================================
# SHOW SAMPLES
# =============================================================================

print("\n" + "="*70)
print("SAMPLE HIGH-CONFIDENCE POSITIVES")
print("="*70)

for i, (p, s) in enumerate(high_conf_positives[:5]):
    print(f"\n[score={s}] {p[:150]}...")

print("\n" + "="*70)
print("SAMPLE HARD NEGATIVES")
print("="*70)

for i, (n, s) in enumerate(hard_negatives[:5]):
    print(f"\n[score={s}] {n[:150]}...")

print("\n" + "="*70)
print("DATASET READY FOR PRECISION TRAINING")
print("="*70)
