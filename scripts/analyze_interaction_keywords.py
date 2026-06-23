#!/usr/bin/env python3
"""
Analyze interaction keywords in positive training examples
to identify potential noise and improve dataset quality
"""

import csv
import re
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
base_dir = Path("/path/to/MetaP/classifier")
training_file = base_dir / "data/training/training_data_improved_20k.csv"
output_dir = base_dir / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

# Keywords that typically indicate biotic interactions
INTERACTION_KEYWORDS = [
    # Predation/Consumption
    'prey', 'predator', 'predation', 'preys', 'predators', 'preyed', 'preying',
    'eat', 'ate', 'eaten', 'eats', 'eating', 'eater',
    'consume', 'consumed', 'consuming', 'consumption', 'consumer',
    'feed', 'fed', 'feeding', 'feeds', 'feeder',
    'hunt', 'hunted', 'hunting', 'hunts', 'hunter',
    'capture', 'captured', 'capturing', 'captures',
    'kill', 'killed', 'killing', 'kills',
    'devour', 'devoured', 'devouring',

    # Parasitism
    'parasite', 'parasites', 'parasitic', 'parasitism', 'parasitize', 'parasitized',
    'host', 'hosts', 'hosted', 'hosting',
    'infect', 'infected', 'infecting', 'infection', 'infections', 'infectious',
    'infest', 'infested', 'infestation', 'infestations',
    'pathogen', 'pathogens', 'pathogenic', 'pathogenicity',
    'disease', 'diseases', 'diseased',
    'vector', 'vectors', 'vectored',
    'transmit', 'transmitted', 'transmission', 'transmitting',

    # Symbiosis/Mutualism
    'symbiosis', 'symbiotic', 'symbiont', 'symbionts',
    'mutualism', 'mutualistic', 'mutualist',
    'commensalism', 'commensal', 'commensals',
    'endosymbiont', 'endosymbionts', 'endosymbiotic',

    # Competition
    'compete', 'competed', 'competing', 'competition', 'competitor', 'competitors', 'competitive',
    'outcompete', 'outcompeted', 'outcompeting',
    'displace', 'displaced', 'displacement',
    'exclude', 'excluded', 'exclusion',

    # Herbivory
    'herbivore', 'herbivores', 'herbivory', 'herbivorous',
    'graze', 'grazed', 'grazing', 'grazer', 'grazers',
    'browse', 'browsed', 'browsing', 'browser',
    'forage', 'foraged', 'foraging', 'forager',
    'defoliate', 'defoliated', 'defoliation',

    # Pollination
    'pollinate', 'pollinated', 'pollinating', 'pollination', 'pollinator', 'pollinators',

    # Other interactions
    'interact', 'interacted', 'interacting', 'interaction', 'interactions', 'interactive',
    'associate', 'associated', 'association', 'associations',
    'coexist', 'coexistence', 'coexisting',
    'colonize', 'colonized', 'colonization', 'colonizing',
    'attack', 'attacked', 'attacking', 'attacks',
    'defend', 'defended', 'defense', 'defensive',
    'prey on', 'preyed on', 'preys on',
]

# Keywords that might indicate NON-interactions (potential noise)
NOISE_KEYWORDS = [
    # Lab/experimental context (not ecological interactions)
    'cultured', 'incubated', 'inoculated', 'transfected', 'expressed',
    'cloned', 'sequenced', 'amplified', 'extracted',
    'stained', 'labeled', 'measured', 'assayed',

    # Biochemical/molecular (not species interactions)
    'binds', 'binding', 'bound', 'receptor', 'ligand',
    'enzyme', 'substrate', 'inhibitor', 'activator',
    'protein', 'gene', 'dna', 'rna', 'mrna',
    'phosphorylation', 'methylation', 'acetylation',

    # Generic verbs that aren't interactions
    'studied', 'investigated', 'examined', 'analyzed', 'observed',
    'reported', 'described', 'identified', 'detected', 'found',
    'compared', 'evaluated', 'assessed', 'determined',
]

print("="*70)
print("ANALYZING INTERACTION KEYWORDS IN POSITIVE EXAMPLES")
print("="*70)

# Load positive examples
positives = []
with open(training_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['label']) == 1:
            positives.append(row['passage'].lower())

print(f"\nLoaded {len(positives)} positive examples")

# Count interaction keywords
interaction_counts = Counter()
for sentence in positives:
    words = set(re.findall(r'\b\w+\b', sentence))
    for keyword in INTERACTION_KEYWORDS:
        if keyword in words or keyword in sentence:
            interaction_counts[keyword] += 1

# Count noise keywords
noise_counts = Counter()
for sentence in positives:
    words = set(re.findall(r'\b\w+\b', sentence))
    for keyword in NOISE_KEYWORDS:
        if keyword in words:
            noise_counts[keyword] += 1

# Print top interaction keywords
print("\n" + "="*70)
print("TOP 40 INTERACTION KEYWORDS IN POSITIVES")
print("="*70)
for keyword, count in interaction_counts.most_common(40):
    pct = (count / len(positives)) * 100
    print(f"  {keyword:20s}: {count:5d} ({pct:5.1f}%)")

# Print top noise keywords
print("\n" + "="*70)
print("TOP 20 POTENTIAL NOISE KEYWORDS IN POSITIVES")
print("="*70)
for keyword, count in noise_counts.most_common(20):
    pct = (count / len(positives)) * 100
    print(f"  {keyword:20s}: {count:5d} ({pct:5.1f}%)")

# Find sentences with noise keywords but no clear interaction keywords
print("\n" + "="*70)
print("ANALYZING POTENTIALLY NOISY POSITIVES")
print("="*70)

noisy_positives = []
strong_interaction_kw = ['prey', 'predator', 'parasite', 'host', 'infect', 'pollinator',
                         'herbivore', 'symbiont', 'pathogen', 'vector', 'feed', 'attack']

for sentence in positives:
    words = set(re.findall(r'\b\w+\b', sentence))
    has_strong_interaction = any(kw in sentence for kw in strong_interaction_kw)
    has_noise = any(kw in words for kw in ['protein', 'gene', 'receptor', 'binding', 'enzyme'])

    if has_noise and not has_strong_interaction:
        noisy_positives.append(sentence)

print(f"\nFound {len(noisy_positives)} potentially noisy positives (molecular/biochemical without clear interaction)")
print("\nExamples of potentially noisy positives:")
for i, sent in enumerate(noisy_positives[:10]):
    print(f"  {i+1}. {sent[:100]}...")

# Create visualization
print("\n" + "="*70)
print("CREATING FIGURES")
print("="*70)

# Figure 1: Top 30 interaction keywords
fig, ax = plt.subplots(figsize=(14, 10))
top_keywords = interaction_counts.most_common(30)
keywords = [k for k, v in top_keywords]
counts = [v for k, v in top_keywords]

colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(keywords)))
import numpy as np

bars = ax.barh(keywords[::-1], counts[::-1], color=colors[::-1], edgecolor='black', linewidth=0.5)
ax.set_xlabel('Frequency in Positive Examples', fontsize=12, fontweight='bold')
ax.set_title('Top 30 Interaction Keywords in Training Positives\n(n=10,000 sentences)',
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add count labels
for bar, count in zip(bars, counts[::-1]):
    ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
            f'{count}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'interaction_keywords_frequency.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'interaction_keywords_frequency.pdf', bbox_inches='tight')
plt.close()
print("  ✓ Saved: interaction_keywords_frequency.png/pdf")

# Figure 2: Interaction categories
categories = {
    'Parasitism/Disease': ['parasite', 'host', 'infect', 'pathogen', 'disease', 'vector', 'transmit'],
    'Predation': ['prey', 'predator', 'hunt', 'kill', 'capture', 'attack'],
    'Feeding': ['feed', 'consume', 'eat', 'forage', 'graze'],
    'Symbiosis': ['symbiont', 'mutualism', 'commensal', 'associate'],
    'Competition': ['compete', 'competition', 'displace', 'exclude'],
    'Pollination': ['pollinate', 'pollinator', 'pollination'],
    'Generic': ['interact', 'interaction', 'colonize']
}

category_counts = {}
for cat, keywords in categories.items():
    total = sum(interaction_counts.get(kw, 0) for kw in keywords)
    category_counts[cat] = total

fig, ax = plt.subplots(figsize=(10, 6))
cats = list(category_counts.keys())
counts = list(category_counts.values())
colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#95a5a6']

bars = ax.bar(cats, counts, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
ax.set_ylabel('Total Mentions', fontsize=12, fontweight='bold')
ax.set_title('Interaction Types in Positive Training Examples', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=30, ha='right')

for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'interaction_categories.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'interaction_categories.pdf', bbox_inches='tight')
plt.close()
print("  ✓ Saved: interaction_categories.png/pdf")

# Figure 3: Potential noise analysis
fig, ax = plt.subplots(figsize=(12, 6))
top_noise = noise_counts.most_common(15)
noise_kw = [k for k, v in top_noise]
noise_ct = [v for k, v in top_noise]

bars = ax.barh(noise_kw[::-1], noise_ct[::-1], color='#e74c3c', alpha=0.7, edgecolor='black')
ax.set_xlabel('Frequency in Positive Examples', fontsize=12, fontweight='bold')
ax.set_title('Potential Noise Keywords in Positive Examples\n(Molecular/Lab terms that may not indicate biotic interactions)',
             fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

for bar, count in zip(bars, noise_ct[::-1]):
    pct = (count / len(positives)) * 100
    ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
            f'{count} ({pct:.1f}%)', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'potential_noise_keywords.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'potential_noise_keywords.pdf', bbox_inches='tight')
plt.close()
print("  ✓ Saved: potential_noise_keywords.png/pdf")

# Summary
print("\n" + "="*70)
print("SUMMARY & RECOMMENDATIONS")
print("="*70)
print(f"""
CURRENT FILTERING:
  - Removed: 'ate', 'eat', 'eated', 'eating'

RECOMMENDATIONS FOR ADDITIONAL FILTERING:

1. MOLECULAR/BIOCHEMICAL NOISE (consider removing sentences with these WITHOUT clear interaction keywords):
   - 'binding', 'binds', 'bound' (protein-protein, not species interactions)
   - 'receptor', 'ligand' (molecular level)
   - 'inhibit', 'inhibitor' (biochemical)
   - 'express', 'expressed', 'expression' (gene expression)

2. LAB/EXPERIMENTAL NOISE:
   - 'cultured', 'incubated' (in vitro, not ecological)
   - 'transfected', 'cloned' (molecular biology)

3. AMBIGUOUS TERMS (keep but verify):
   - 'interact', 'interaction' - sometimes molecular, not species level
   - 'associate', 'associated' - can be statistical, not biological

4. POTENTIAL FALSE POSITIVES COUNT: ~{len(noisy_positives)} sentences ({len(noisy_positives)/len(positives)*100:.1f}%)
   These have molecular terms but no clear ecological interaction keywords.

SUGGESTED APPROACH:
  1. Create a filtered dataset removing sentences with molecular terms
  2. OR add a second-pass filter: only keep if contains strong interaction keyword
  3. Compare model performance on filtered vs unfiltered data
""")

# Save detailed analysis
analysis_file = base_dir / "analysis" / "keyword_analysis.csv"
analysis_file.parent.mkdir(exist_ok=True)

with open(analysis_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['keyword', 'count', 'percentage', 'category'])
    for keyword, count in interaction_counts.most_common():
        pct = (count / len(positives)) * 100
        # Determine category
        cat = 'other'
        for c, kws in categories.items():
            if keyword in kws:
                cat = c
                break
        writer.writerow([keyword, count, f'{pct:.2f}', cat])

print(f"\n✓ Detailed analysis saved to: {analysis_file}")
