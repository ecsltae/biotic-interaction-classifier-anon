#!/usr/bin/env python3
"""
Analyze frequency of interactions from the interaction_dict.csv in positive examples
"""

import csv
import re
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Paths
base_dir = Path("/path/to/MetaP/classifier")
training_file = base_dir / "data/training/training_data_improved_20k.csv"
interaction_dict_file = base_dir / "data/processed/interaction_dict.csv"
output_dir = base_dir / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("ANALYZING INTERACTION DICTIONARY FREQUENCY IN POSITIVES")
print("="*70)

# Load interaction dictionary
interactions = []
with open(interaction_dict_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        if row:
            interactions.append(row[0].lower().strip())

print(f"\nLoaded {len(interactions)} interaction terms from dictionary")

# Load positive examples
positives = []
with open(training_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['label']) == 1:
            positives.append(row['passage'].lower())

print(f"Loaded {len(positives)} positive examples")

# Count interaction frequencies
interaction_counts = Counter()
sentences_with_interaction = {}

for sentence in positives:
    for interaction in interactions:
        if interaction in sentence:
            interaction_counts[interaction] += 1
            if interaction not in sentences_with_interaction:
                sentences_with_interaction[interaction] = []
            sentences_with_interaction[interaction].append(sentence)

# Sort by frequency
sorted_interactions = interaction_counts.most_common()

print("\n" + "="*70)
print("TOP 50 MOST FREQUENT INTERACTIONS IN POSITIVES")
print("="*70)
for interaction, count in sorted_interactions[:50]:
    pct = (count / len(positives)) * 100
    print(f"  {interaction:40s}: {count:5d} ({pct:5.2f}%)")

# Analyze very rare interactions (might be noise or very specific)
print("\n" + "="*70)
print("INTERACTIONS WITH HIGH FREQUENCY (>1%)")
print("="*70)
high_freq = [(i, c) for i, c in sorted_interactions if c/len(positives) > 0.01]
for interaction, count in high_freq:
    pct = (count / len(positives)) * 100
    print(f"  {interaction:40s}: {count:5d} ({pct:5.2f}%)")

# Check for potentially problematic interactions
print("\n" + "="*70)
print("POTENTIALLY PROBLEMATIC INTERACTIONS (Ambiguous or too generic)")
print("="*70)

problematic_keywords = [
    'ate', 'eat', 'eating', 'eats', 'eated',  # Already removed
    'affected', 'affecting', 'affects',  # Too generic
    'associated', 'association',  # Can be statistical
    'interacts', 'interaction',  # Generic
    'related', 'regulate', 'regulates',  # Can be molecular
    'produces', 'provides',  # Can be non-biological
    'involves', 'involved',  # Generic
]

problematic_counts = {}
for kw in problematic_keywords:
    for interaction, count in sorted_interactions:
        if kw in interaction:
            problematic_counts[interaction] = count

print("\nInteractions containing potentially ambiguous terms:")
for interaction, count in sorted(problematic_counts.items(), key=lambda x: -x[1])[:30]:
    pct = (count / len(positives)) * 100
    print(f"  {interaction:40s}: {count:5d} ({pct:5.2f}%)")

# Create figure: Top 40 interactions
print("\n" + "="*70)
print("CREATING FIGURES")
print("="*70)

fig, ax = plt.subplots(figsize=(14, 12))
top40 = sorted_interactions[:40]
interactions_names = [i for i, c in top40]
counts = [c for i, c in top40]

colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(interactions_names)))
bars = ax.barh(interactions_names[::-1], counts[::-1], color=colors[::-1], edgecolor='black', linewidth=0.5)

ax.set_xlabel('Frequency in Positive Examples', fontsize=12, fontweight='bold')
ax.set_title(f'Top 40 Interaction Terms from Dictionary\n(n={len(positives)} positive sentences)',
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add count labels
for bar, count in zip(bars, counts[::-1]):
    pct = (count / len(positives)) * 100
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            f'{count} ({pct:.1f}%)', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'interaction_dict_frequency.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'interaction_dict_frequency.pdf', bbox_inches='tight')
plt.close()
print("  ✓ Saved: interaction_dict_frequency.png/pdf")

# Create category-based analysis
print("\n" + "="*70)
print("CATEGORIZING INTERACTIONS")
print("="*70)

categories = {
    'Parasitism': ['parasite', 'parasit', 'endoparasite', 'ectoparasite', 'mesoparasite'],
    'Predation': ['prey', 'hunt', 'kill', 'devour', 'predator', 'predation'],
    'Infection/Pathogen': ['infect', 'pathogen', 'disease', 'transmit', 'vector', 'contamin'],
    'Symbiosis': ['symbiont', 'mutualis', 'commensal', 'co-'],
    'Feeding': ['feed', 'graze', 'forage', 'nutrient', 'consum', 'ingest'],
    'Competition': ['compet', 'displace', 'exclude', 'amensali'],
    'Pollination': ['pollinat', 'flower'],
    'Habitat/Colonization': ['habitat', 'coloniz', 'nest'],
    'Regulation': ['regulat', 'inhibit', 'activat'],
    'Generic': ['interact', 'associat', 'affect', 'relat'],
}

category_totals = {}
for cat_name, keywords in categories.items():
    total = 0
    for interaction, count in sorted_interactions:
        if any(kw in interaction for kw in keywords):
            total += count
    category_totals[cat_name] = total

# Sort and plot
sorted_cats = sorted(category_totals.items(), key=lambda x: -x[1])

fig, ax = plt.subplots(figsize=(12, 6))
cat_names = [c for c, v in sorted_cats]
cat_counts = [v for c, v in sorted_cats]

colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12',
          '#1abc9c', '#e91e63', '#00bcd4', '#ff9800', '#95a5a6']

bars = ax.bar(cat_names, cat_counts, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
ax.set_ylabel('Total Occurrences', fontsize=12, fontweight='bold')
ax.set_title('Interaction Categories in Positive Training Examples', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=35, ha='right')

for bar, count in zip(bars, cat_counts):
    pct = (count / len(positives)) * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'interaction_categories_from_dict.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'interaction_categories_from_dict.pdf', bbox_inches='tight')
plt.close()
print("  ✓ Saved: interaction_categories_from_dict.png/pdf")

# Show example sentences for problematic interactions
print("\n" + "="*70)
print("EXAMPLE SENTENCES FOR REVIEW")
print("="*70)

review_interactions = ['affected', 'affecting', 'associated', 'regulate', 'regulates']
for interaction in review_interactions:
    if interaction in sentences_with_interaction:
        print(f"\n'{interaction}' examples:")
        for sent in sentences_with_interaction[interaction][:3]:
            print(f"  - {sent[:120]}...")

# Save detailed analysis
analysis_file = base_dir / "analysis" / "interaction_dict_analysis.csv"
analysis_file.parent.mkdir(exist_ok=True)

with open(analysis_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['interaction', 'count', 'percentage', 'example_sentence'])
    for interaction, count in sorted_interactions:
        pct = (count / len(positives)) * 100
        example = sentences_with_interaction.get(interaction, [''])[0][:200]
        writer.writerow([interaction, count, f'{pct:.3f}', example])

print(f"\n✓ Detailed analysis saved to: {analysis_file}")

# Final recommendations
print("\n" + "="*70)
print("RECOMMENDATIONS FOR NOISE REMOVAL")
print("="*70)
print("""
ALREADY REMOVED:
  - 'ate', 'eat', 'eated', 'eating'

CONSIDER REMOVING (too generic, may cause false positives):
  1. 'affected' / 'affecting' / 'affects' - Very generic, not specific to biotic interactions
  2. 'associated' / 'association' - Often statistical, not biological
  3. 'regulate' / 'regulates' / 'regulated' - Often molecular/cellular, not species-level
  4. 'involved in' - Too generic
  5. 'related' - Often taxonomic, not interaction

KEEP BUT VERIFY (legitimate but check for false matches):
  - 'infect*' - Core interaction, but check for metaphorical use
  - 'host' - Good, but can refer to conference host etc.
  - 'parasite' - Good, clear biotic interaction

NEXT STEPS:
  1. Review example sentences for each problematic term
  2. Create a filtered version removing generic terms
  3. Compare model performance on filtered vs original
""")
