#!/usr/bin/env python3
"""
Improve training data by:
1. Analyzing interaction frequencies in positives
2. Removing noisy/ambiguous interactions
3. Filtering weak positive sentences
4. Improving negative examples
5. Including viruses from NCBI taxonomy
"""

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Paths
base_dir = Path("/path/to/MetaP/classifier")
training_file = base_dir / "data/training/training_data_improved_20k.csv"
interaction_dict_file = base_dir / "data/processed/interaction_dict.csv"
virus_file = base_dir / "ncbi_taxon_viruses_full_v3.json"
output_dir = base_dir / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("IMPROVING TRAINING DATA")
print("="*80)

# =============================================================================
# STEP 1: Load all data
# =============================================================================
print("\n[1] LOADING DATA...")

# Load interaction dictionary
interactions = []
with open(interaction_dict_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        if row:
            interactions.append(row[0].lower().strip())
print(f"  Loaded {len(interactions)} interaction terms")

# Load training data
positives = []
negatives = []
with open(training_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['label']) == 1:
            positives.append(row['passage'])
        else:
            negatives.append(row['passage'])

print(f"  Loaded {len(positives)} positives, {len(negatives)} negatives")

# Load viruses
with open(virus_file, 'r', encoding='utf-8') as f:
    virus_data = json.load(f)
virus_names = set()
for concept in virus_data.get('concepts', []):
    if 'preferred_term' in concept:
        virus_names.add(concept['preferred_term']['term'].lower())
    for syn in concept.get('synonyms', []):
        virus_names.add(syn['term'].lower())
print(f"  Loaded {len(virus_names)} virus names")

# =============================================================================
# STEP 2: Count interaction frequencies in positives (INCLUDING eat-like)
# =============================================================================
print("\n[2] ANALYZING INTERACTION FREQUENCIES...")

interaction_counts = Counter()
interaction_examples = defaultdict(list)

for sentence in positives:
    sentence_lower = sentence.lower()
    for interaction in interactions:
        if interaction in sentence_lower:
            interaction_counts[interaction] += 1
            if len(interaction_examples[interaction]) < 5:
                interaction_examples[interaction].append(sentence[:150])

# Sort by frequency
sorted_interactions = interaction_counts.most_common()

print(f"\n  Total interactions found in positives: {len(sorted_interactions)}")
print(f"\n  TOP 50 INTERACTIONS:")
for i, (interaction, count) in enumerate(sorted_interactions[:50], 1):
    pct = (count / len(positives)) * 100
    print(f"    {i:2d}. {interaction:40s}: {count:5d} ({pct:5.2f}%)")

# =============================================================================
# STEP 3: Identify noisy/problematic interactions to REMOVE
# =============================================================================
print("\n[3] IDENTIFYING NOISY INTERACTIONS TO REMOVE...")

# Terms that are too generic, ambiguous, or don't indicate true biotic interactions
NOISY_INTERACTIONS = [
    # Already removed eat-like terms
    'ate', 'eat', 'eated', 'eating', 'eats', 'eats by', 'eat by', 'eated by',
    'has ate', 'have ate', 'was ate by',

    # Too generic - can be molecular/statistical, not species interactions
    'interacts with', 'interact with', 'interacting with', 'interacted with',
    'biotically interacts with', 'biotically interact with', 'biotically interacted with',
    'affecting', 'affected by',
    'regulate', 'regulates', 'regulated by', 'regulating',
    'negatively regulate', 'positively regulate', 'negatively regulates', 'positively regulates',
    'indirectly regulate', 'indirectly regulates',
    'directly regulate', 'directly regulates', 'directly regulating',
    'capable of regulating', 'capable of positively regulating', 'capable of negatively regulating',

    # Molecular level, not species level
    'regulates activity of', 'regulate activity of', 'regulates level of', 'regulate level of',
    'regulates quantity of', 'regulate quantity of', 'regulates levels of', 'regulate levels of',
    'increases expression of', 'represses expression of', 'represse expression of', 'repressing expression of',
    'increase expression of', 'increasing expression of',
    'directly activates', 'directly activate', 'indirectly activate',
    'directly inhibit', 'direclty inhibits', 'indirectly inhibit',

    # Too vague - could be anything
    'depends on', 'depend on',
    'contributes to', 'contribute to', 'contributing to', 'contributes to morphology of',
    'involved in or involved in regulation of', 'involved in regulation of',
    'involved in positive regulation of', 'involved in negative regulation of',
    'involved in positive regulations of', 'involved in negative regulations of',
    'involved in regulations of',

    # Temporal/spatial, not interaction
    'adjacent to', 'temporally related to', 'developmentally related to',
    'co-occur with', 'co-occurs with', 'co-occured with', 'co-occuring with',
    'ecologically co-occurs with', 'ecologically co-occur with', 'ecologically co-occured with',

    # Input/output (too generic)
    'has input', 'has output',
    'target participant in', 'subject participant in', 'partner in',

    # Colocalization (molecular)
    'colocalize with', 'colocalizes with', 'colocalized with',

    # Non-specific participation
    'participates in a biotic-biotic interaction with', 'participate in a biotic-biotic interaction with',
    'participated in a biotic-biotic interaction with', 'participating in a biotic-biotic interaction with',
    'participates in a abiotic-biotic interaction with',
]

# Convert to set for faster lookup
noisy_set = set(term.lower() for term in NOISY_INTERACTIONS)

# Count how many positives will be affected
affected_positives = set()
for i, sentence in enumerate(positives):
    sentence_lower = sentence.lower()
    for noisy in noisy_set:
        if noisy in sentence_lower:
            affected_positives.add(i)
            break

print(f"\n  Noisy interactions to remove: {len(noisy_set)}")
print(f"  Positives containing noisy terms: {len(affected_positives)} ({100*len(affected_positives)/len(positives):.1f}%)")

# Show which noisy terms appear most
noisy_counts = {}
for noisy in noisy_set:
    if noisy in interaction_counts:
        noisy_counts[noisy] = interaction_counts[noisy]

print(f"\n  Most frequent noisy terms in positives:")
for term, count in sorted(noisy_counts.items(), key=lambda x: -x[1])[:20]:
    pct = (count / len(positives)) * 100
    print(f"    {term:45s}: {count:5d} ({pct:5.2f}%)")

# =============================================================================
# STEP 4: Define STRONG biotic interaction terms (keep these)
# =============================================================================
print("\n[4] DEFINING STRONG INTERACTION TERMS...")

STRONG_INTERACTIONS = [
    # Parasitism - very clear
    'parasite of', 'parasites of', 'parasitized by', 'parasitize', 'parasitizes', 'parasitizing',
    'ectoparasite of', 'endoparasite of', 'mesoparasite of', 'hemiparasite of',
    'ectoparasites', 'endoparasites', 'mesoparasites', 'hemiparasites',
    'has parasite', 'have parasite', 'has endoparasite', 'has ectoparasite',
    'facultative parasite of', 'obligate parasite of', 'root parasite of', 'stem parasite of',
    'hyperparasite of', 'hyperparasitized by', 'kleptoparasite of', 'kleptoparasitized by',

    # Parasitoids
    'parasitoid of', 'had parasitoid', 'has parasitoid', 'have parasitoid',
    'idiobiont parasitoid of', 'koinobiont parasitoid of',
    'endoparasitoid of', 'ectoparasitoid of',

    # Predation
    'prey on', 'preys on', 'preyed on', 'preying on', 'prey of',
    'predator of', 'predators of', 'predatory',
    'hunts', 'hunted by', 'hunting', 'hunt',
    'devours', 'devoured by', 'devouring',
    'kills', 'killed by', 'is killed by',

    # Infection/disease
    'infect', 'infects', 'infected', 'infecting', 'infection',
    'infest', 'infests', 'infested', 'infesting', 'infestation',
    'pathogen of', 'pathogens of', 'has pathogen', 'have pathogen',
    'transmitted by', 'transmit', 'transmitting', 'transmission',
    'is vector for', 'vector for', 'vectors for', 'had vector', 'have vector',
    'reservoir host of', 'reservoir host', 'competent host',

    # Symbiosis
    'symbiont of', 'symbionts of', 'symbiotic', 'symbiosis',
    'mutualist of', 'mutualists of', 'mutualism', 'mutualistic',
    'commensalist of', 'commensalists of', 'commensal', 'commensalism',
    'endosymbiont', 'symbiotically interacts with',

    # Feeding
    'feeds on', 'feed on', 'fed on', 'feeding on',
    'ingests', 'ingested by', 'ingestion',
    'grazes', 'grazed', 'grazing', 'graze',
    'acquires nutrients from', 'acquire nutrients',

    # Pollination
    'pollinator of', 'pollinators of', 'pollinate', 'pollinates', 'pollinated by',
    'visits flowers of', 'visited flowers of', 'has flowers visited by',

    # Host relationships
    'host of', 'hosts of', 'host for', 'hosts for', 'host to',
    'hosted', 'hosting', 'has host', 'have host',

    # Dispersal
    'disperses seed of', 'dispersed seed of', 'seeds dispersed by',
    'transported by', 'transporting by',

    # Phoresis
    'phoresis', 'phoretic', 'phoresy interaction',

    # Epiphyte
    'epiphyte of', 'epiphytes of', 'has epiphyte', 'has epiphytes',

    # Colonization
    'colonizes', 'colonized by', 'colonization',

    # Invasion
    'invades', 'invaded', 'invasion',

    # Coevolution (implies interaction)
    'coevolves', 'coevolved', 'coevolution',
]

strong_set = set(term.lower() for term in STRONG_INTERACTIONS)
print(f"  Strong interaction terms defined: {len(strong_set)}")

# =============================================================================
# STEP 5: Filter positives - keep only those with STRONG interactions
# =============================================================================
print("\n[5] FILTERING POSITIVES...")

strong_positives = []
weak_positives = []

for sentence in positives:
    sentence_lower = sentence.lower()

    # Check if contains any strong interaction
    has_strong = False
    for strong in strong_set:
        if strong in sentence_lower:
            has_strong = True
            break

    # Check if contains only noisy interaction (no strong)
    has_noisy = False
    for noisy in noisy_set:
        if noisy in sentence_lower:
            has_noisy = True
            break

    if has_strong:
        strong_positives.append(sentence)
    elif has_noisy:
        weak_positives.append(sentence)
    else:
        # Has some other interaction term or none at all - need to check
        # Keep for now but mark for review
        strong_positives.append(sentence)

print(f"  Strong positives (with clear interaction): {len(strong_positives)}")
print(f"  Weak positives (only noisy terms): {len(weak_positives)}")

# Show examples of weak positives
print(f"\n  Examples of WEAK positives to review:")
for i, sent in enumerate(weak_positives[:10], 1):
    print(f"    {i}. {sent[:120]}...")

# =============================================================================
# STEP 6: Analyze and improve negatives
# =============================================================================
print("\n[6] ANALYZING NEGATIVES...")

# Check for negatives that might actually be positives (have strong interaction terms)
false_negatives = []
for sentence in negatives:
    sentence_lower = sentence.lower()
    for strong in strong_set:
        if strong in sentence_lower:
            false_negatives.append((sentence, strong))
            break

print(f"  Potential false negatives (have strong interaction terms): {len(false_negatives)}")
if false_negatives:
    print(f"\n  Examples of potential false negatives:")
    for i, (sent, term) in enumerate(false_negatives[:5], 1):
        print(f"    {i}. Term: '{term}'")
        print(f"       {sent[:120]}...")

# =============================================================================
# STEP 7: Create frequency plot (ALL interactions including eat-like)
# =============================================================================
print("\n[7] CREATING FREQUENCY PLOTS...")

# Plot top 60 interactions
fig, ax = plt.subplots(figsize=(16, 14))
top60 = sorted_interactions[:60]
interactions_names = [i for i, c in top60]
counts = [c for i, c in top60]

# Color by type: green=strong, red=noisy, gray=other
colors = []
for name in interactions_names:
    if name in noisy_set:
        colors.append('#e74c3c')  # red for noisy
    elif name in strong_set:
        colors.append('#2ecc71')  # green for strong
    else:
        colors.append('#95a5a6')  # gray for other

bars = ax.barh(interactions_names[::-1], counts[::-1], color=colors[::-1], edgecolor='black', linewidth=0.5)

ax.set_xlabel('Frequency in Positive Examples', fontsize=12, fontweight='bold')
ax.set_title(f'Top 60 Interaction Terms in Positives (n={len(positives)})\nGreen=Strong, Red=Noisy, Gray=Other',
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add count labels
for bar, count in zip(bars, counts[::-1]):
    pct = (count / len(positives)) * 100
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            f'{count} ({pct:.1f}%)', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'interaction_frequency_all.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'interaction_frequency_all.pdf', bbox_inches='tight')
plt.close()
print("  Saved: interaction_frequency_all.png/pdf")

# =============================================================================
# STEP 8: Summary statistics
# =============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

# Count by category
strong_in_positives = sum(1 for s in positives if any(st in s.lower() for st in strong_set))
noisy_only = sum(1 for s in positives if any(n in s.lower() for n in noisy_set) and not any(st in s.lower() for st in strong_set))

print(f"""
CURRENT DATASET:
  Positives: {len(positives)}
  Negatives: {len(negatives)}

INTERACTION ANALYSIS:
  Unique interactions found: {len(sorted_interactions)}
  Strong interactions defined: {len(strong_set)}
  Noisy interactions to remove: {len(noisy_set)}

POSITIVE QUALITY:
  With strong interaction terms: {strong_in_positives} ({100*strong_in_positives/len(positives):.1f}%)
  With only noisy terms: {noisy_only} ({100*noisy_only/len(positives):.1f}%)

POTENTIAL FALSE NEGATIVES:
  Negatives with strong terms: {len(false_negatives)} ({100*len(false_negatives)/len(negatives):.1f}%)

RECOMMENDATIONS:
  1. Remove positives with only noisy terms: ~{noisy_only} sentences
  2. Review potential false negatives: {len(false_negatives)} sentences
  3. Final clean positives: ~{len(positives) - noisy_only}
""")

# =============================================================================
# STEP 9: Save analysis for next step
# =============================================================================
analysis_dir = base_dir / "analysis"
analysis_dir.mkdir(exist_ok=True)

# Save interaction frequency
with open(analysis_dir / 'interaction_frequency_full.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['interaction', 'count', 'percentage', 'category'])
    for interaction, count in sorted_interactions:
        pct = (count / len(positives)) * 100
        if interaction in noisy_set:
            cat = 'noisy'
        elif interaction in strong_set:
            cat = 'strong'
        else:
            cat = 'other'
        writer.writerow([interaction, count, f'{pct:.3f}', cat])

print(f"\nSaved analysis to: {analysis_dir}/interaction_frequency_full.csv")

# Save weak positives for review
with open(analysis_dir / 'weak_positives.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['sentence'])
    for sent in weak_positives:
        writer.writerow([sent])

print(f"Saved weak positives to: {analysis_dir}/weak_positives.csv")

# Save false negatives for review
with open(analysis_dir / 'potential_false_negatives.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['sentence', 'matching_term'])
    for sent, term in false_negatives:
        writer.writerow([sent, term])

print(f"Saved false negatives to: {analysis_dir}/potential_false_negatives.csv")

print("\n" + "="*80)
print("NEXT STEP: Run create_improved_dataset.py to generate the cleaned training data")
print("="*80)
