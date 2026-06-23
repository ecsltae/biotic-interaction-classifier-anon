#!/usr/bin/env python3
"""
Create an optimal training dataset based on our analysis:

Key findings:
1. BiomedBERT on 6k (3k+3k) performs BEST on eval100 (F1=0.488)
2. The 6k positives are high quality - keep them
3. Need good quality negatives that don't overlap with positive patterns

Strategy:
1. Use all 3k positives from 6k dataset
2. Create balanced negatives from:
   - Taxonomic/phylogenetic descriptions (no interaction)
   - Multi-species lists without interaction context
   - Lab/molecular biology contexts
   - General ecological descriptions without explicit interactions
"""

import csv
import random
from pathlib import Path
from collections import Counter

random.seed(42)

BASE_DIR = Path("/path/to/MetaP/classifier")

print("="*80)
print("CREATING OPTIMAL TRAINING DATASET")
print("="*80)

# Load original 6k positives (these work well!)
print("\n[1] Loading 6k positives...")
positives_6k = []
with open(BASE_DIR / "data/training/training_data_cleaned.csv", 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['label']) == 1:
            positives_6k.append(row['passage'])

print(f"  Loaded {len(positives_6k)} positives from 6k dataset")

# Load 20k negatives pool
print("\n[2] Loading negative pool...")
negatives_20k = []
with open(BASE_DIR / "data/training/training_data_improved_20k.csv", 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['label']) == 0:
            negatives_20k.append(row['passage'])

print(f"  Loaded {len(negatives_20k)} negatives from 20k dataset")

# Filter negatives - remove potential false negatives
print("\n[3] Filtering negatives...")

# Keywords that indicate REAL biotic interactions (filter these OUT)
INTERACTION_KEYWORDS = [
    'infect', 'infection', 'infected',
    'parasite', 'parasites', 'parasitic', 'parasitize',
    'pathogen', 'pathogens', 'pathogenic',
    'prey on', 'preys on', 'prey of',
    'predator of', 'predators of',
    'feeds on', 'fed on', 'feeding on',
    'host of', 'hosts of', 'host for',
    'vector of', 'vector for', 'transmitted by',
    'symbiont', 'symbiosis', 'mutualism',
    'colonize', 'colonized', 'infestation',
]

def is_clean_negative(sentence):
    """Check if sentence is a clean negative (no interaction keywords)"""
    s = sentence.lower()
    for kw in INTERACTION_KEYWORDS:
        if kw in s:
            return False
    return True

clean_negatives = [s for s in negatives_20k if is_clean_negative(s)]
print(f"  Clean negatives (no interaction keywords): {len(clean_negatives)}")

# Also add some from 6k negatives for diversity
negatives_6k = []
with open(BASE_DIR / "data/training/training_data_cleaned.csv", 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['label']) == 0:
            negatives_6k.append(row['passage'])

# Some 6k negatives might have interaction terms but were labeled negative for a reason
# (e.g., context shows no interaction) - keep some of these as "hard negatives"
hard_negatives = [s for s in negatives_6k if not is_clean_negative(s)]
print(f"  Hard negatives from 6k: {len(hard_negatives)}")

# Combine
print("\n[4] Building final dataset...")

# Target: 6k total (3k pos + 3k neg) to match original successful size
target_positives = len(positives_6k)  # All 3k
target_negatives = len(positives_6k)  # Match with 3k negatives

# Negative composition:
# - Use all hard negatives from 6k (these teach the model nuance)
# - Fill rest with clean negatives
n_hard = min(len(hard_negatives), int(target_negatives * 0.3))
n_clean = target_negatives - n_hard

random.shuffle(clean_negatives)
random.shuffle(hard_negatives)

final_negatives = hard_negatives[:n_hard] + clean_negatives[:n_clean]
random.shuffle(final_negatives)

print(f"  Final positives: {len(positives_6k)}")
print(f"  Final negatives: {len(final_negatives)}")
print(f"    - Clean: {n_clean}")
print(f"    - Hard: {n_hard}")

# Save dataset
print("\n[5] Saving dataset...")
output_file = BASE_DIR / "data/training/training_data_optimal.csv"
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['passage', 'label'])
    for sent in positives_6k:
        writer.writerow([sent, 1])
    for sent in final_negatives:
        writer.writerow([sent, 0])

print(f"  Saved: {output_file}")
print(f"  Total samples: {len(positives_6k) + len(final_negatives)}")

# Analyze final composition
print("\n[6] Final dataset analysis...")

def count_keywords(sentences, keywords):
    count = 0
    for s in sentences:
        if any(kw in s.lower() for kw in keywords):
            count += 1
    return count

strong_pos = count_keywords(positives_6k, ['infect', 'parasite', 'pathogen', 'prey', 'predator', 'host'])
print(f"  Positives with strong interaction: {strong_pos} ({100*strong_pos/len(positives_6k):.1f}%)")

strong_neg = count_keywords(final_negatives, ['infect', 'parasite', 'pathogen', 'prey', 'predator', 'host'])
print(f"  Negatives with interaction terms: {strong_neg} ({100*strong_neg/len(final_negatives):.1f}%)")

print("\n" + "="*80)
print("DATASET CREATED SUCCESSFULLY")
print("="*80)
