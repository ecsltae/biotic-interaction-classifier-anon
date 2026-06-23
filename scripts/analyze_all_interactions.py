#!/usr/bin/env python3
"""
Count ALL interactions from interaction_dict.csv in the 20k sentences.
Output: interaction : count format for all 591 interactions
"""

import csv
import json
from collections import Counter
from pathlib import Path

# Paths
base_dir = Path("/path/to/MetaP/classifier")
training_file = base_dir / "data/training/training_data_improved_20k.csv"
interaction_dict_file = base_dir / "data/processed/interaction_dict.csv"
virus_file = base_dir / "ncbi_taxon_viruses_full_v3.json"
output_dir = base_dir / "analysis"
output_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("INTERACTION FREQUENCY ANALYSIS")
print("="*80)

# =============================================================================
# Load interaction dictionary (all 591 terms)
# =============================================================================
print("\n[1] Loading interaction dictionary...")
interactions = []
with open(interaction_dict_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        if row:
            interactions.append(row[0].strip())

print(f"  Loaded {len(interactions)} interaction terms")

# =============================================================================
# Load 20k training data
# =============================================================================
print("\n[2] Loading 20k training data...")
all_sentences = []
positives = []
negatives = []

with open(training_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        all_sentences.append(row['passage'])
        if int(row['label']) == 1:
            positives.append(row['passage'])
        else:
            negatives.append(row['passage'])

print(f"  Total sentences: {len(all_sentences)}")
print(f"  Positives: {len(positives)}")
print(f"  Negatives: {len(negatives)}")

# =============================================================================
# Count interactions in ALL 20k sentences
# =============================================================================
print("\n[3] Counting interactions in ALL 20k sentences...")

interaction_counts_all = Counter()
interaction_counts_pos = Counter()
interaction_counts_neg = Counter()

for sentence in all_sentences:
    sentence_lower = sentence.lower()
    for interaction in interactions:
        if interaction.lower() in sentence_lower:
            interaction_counts_all[interaction] += 1

for sentence in positives:
    sentence_lower = sentence.lower()
    for interaction in interactions:
        if interaction.lower() in sentence_lower:
            interaction_counts_pos[interaction] += 1

for sentence in negatives:
    sentence_lower = sentence.lower()
    for interaction in interactions:
        if interaction.lower() in sentence_lower:
            interaction_counts_neg[interaction] += 1

# =============================================================================
# Print ALL interactions with counts (sorted by frequency)
# =============================================================================
print("\n" + "="*80)
print("INTERACTION FREQUENCIES IN 20k SENTENCES")
print("="*80)
print(f"\n{'Interaction':<50} {'All':>8} {'Pos':>8} {'Neg':>8}")
print("-"*80)

# Sort by count in all sentences
sorted_interactions = sorted(interaction_counts_all.items(), key=lambda x: -x[1])

# Print all that have at least 1 occurrence
found_count = 0
for interaction, count in sorted_interactions:
    if count > 0:
        found_count += 1
        pos_count = interaction_counts_pos.get(interaction, 0)
        neg_count = interaction_counts_neg.get(interaction, 0)
        print(f"{interaction:<50} {count:>8} {pos_count:>8} {neg_count:>8}")

print("-"*80)
print(f"Total interactions with matches: {found_count} / {len(interactions)}")

# =============================================================================
# Also print interactions with ZERO matches
# =============================================================================
print(f"\n\nInteractions with NO matches ({len(interactions) - found_count}):")
zero_count = 0
for interaction in interactions:
    if interaction not in interaction_counts_all or interaction_counts_all[interaction] == 0:
        zero_count += 1
        if zero_count <= 50:  # Show first 50
            print(f"  - {interaction}")
if zero_count > 50:
    print(f"  ... and {zero_count - 50} more")

# =============================================================================
# Save to CSV
# =============================================================================
output_file = output_dir / "interaction_frequency_complete.csv"
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['interaction', 'count_all', 'count_positive', 'count_negative'])
    for interaction in interactions:
        count_all = interaction_counts_all.get(interaction, 0)
        count_pos = interaction_counts_pos.get(interaction, 0)
        count_neg = interaction_counts_neg.get(interaction, 0)
        writer.writerow([interaction, count_all, count_pos, count_neg])

print(f"\n\nSaved complete frequency table to: {output_file}")

# =============================================================================
# Summary statistics
# =============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

total_matches = sum(interaction_counts_all.values())
print(f"""
Total interaction matches in 20k sentences: {total_matches}
Unique interactions found: {found_count}
Interactions with no matches: {len(interactions) - found_count}

Top 20 interactions:
""")

for i, (interaction, count) in enumerate(sorted_interactions[:20], 1):
    pos_count = interaction_counts_pos.get(interaction, 0)
    neg_count = interaction_counts_neg.get(interaction, 0)
    print(f"  {i:2d}. {interaction:<40}: {count:5d} (pos:{pos_count:4d}, neg:{neg_count:4d})")

print("="*80)
