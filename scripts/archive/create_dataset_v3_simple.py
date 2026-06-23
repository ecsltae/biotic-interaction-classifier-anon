#!/usr/bin/env python3
"""
Create improved dataset v3 - simpler approach:
1. Trust the original positives more (they were curated)
2. Remove only clearly noisy positives (eat-like already removed)
3. Find and fix obvious false negatives
4. Save pathogen list
"""

import csv
import json
import random
from collections import Counter
from pathlib import Path

random.seed(42)

# Paths
base_dir = Path("/path/to/MetaP/classifier")
training_file = base_dir / "data/training/training_data_improved_20k.csv"
interaction_dict_file = base_dir / "data/processed/interaction_dict.csv"
virus_file = base_dir / "ncbi_taxon_viruses_full_v3.json"
output_dir = base_dir / "data/training"
analysis_dir = base_dir / "analysis"

print("="*80)
print("CREATING DATASET V3 - SIMPLE APPROACH")
print("="*80)

# =============================================================================
# STEP 1: Create pathogen list
# =============================================================================
print("\n[1] CREATING PATHOGEN LIST...")

# Load viruses
with open(virus_file, 'r', encoding='utf-8') as f:
    virus_data = json.load(f)

# Extract unique virus genus/family names (shorter, more general)
virus_genera = set()
for concept in virus_data.get('concepts', []):
    if 'preferred_term' in concept:
        term = concept['preferred_term']['term']
        # Get first word (usually genus)
        parts = term.split()
        if len(parts) > 0:
            first = parts[0].lower()
            if len(first) > 4 and not first[0].isdigit():
                virus_genera.add(first)
    for syn in concept.get('synonyms', []):
        parts = syn['term'].split()
        if len(parts) > 0:
            first = parts[0].lower()
            if len(first) > 4 and not first[0].isdigit():
                virus_genera.add(first)

# Common pathogen names
PATHOGENS = {
    # Bacteria
    'escherichia', 'staphylococcus', 'streptococcus', 'salmonella', 'pseudomonas',
    'mycobacterium', 'bacillus', 'clostridium', 'listeria', 'vibrio', 'helicobacter',
    'campylobacter', 'legionella', 'bordetella', 'borrelia', 'treponema', 'neisseria',
    'haemophilus', 'klebsiella', 'enterococcus', 'rickettsia', 'chlamydia', 'mycoplasma',
    'brucella', 'yersinia', 'shigella', 'acinetobacter', 'enterobacter', 'proteus',

    # Parasites
    'plasmodium', 'trypanosoma', 'leishmania', 'toxoplasma', 'giardia', 'entamoeba',
    'cryptosporidium', 'schistosoma', 'ascaris', 'trichinella', 'strongyloides',
    'ancylostoma', 'echinococcus', 'taenia', 'fasciola', 'opisthorchis', 'eimeria',
    'babesia', 'theileria', 'anaplasma', 'ehrlichia', 'nematode', 'cestode', 'trematode',

    # Fungi
    'candida', 'aspergillus', 'cryptococcus', 'histoplasma', 'blastomyces',
    'coccidioides', 'pneumocystis', 'fusarium', 'trichophyton', 'microsporum',

    # Viruses (common short names)
    'influenza', 'hepatitis', 'herpes', 'coronavirus', 'rhinovirus', 'adenovirus',
    'rotavirus', 'norovirus', 'papillomavirus', 'cytomegalovirus', 'poliovirus',
    'rabies', 'measles', 'mumps', 'rubella', 'dengue', 'zika', 'ebola', 'hiv',
}

all_pathogens = PATHOGENS | virus_genera
print(f"  Pathogen genera from viruses: {len(virus_genera)}")
print(f"  Total pathogens: {len(all_pathogens)}")

# Save pathogen list
pathogen_file = analysis_dir / "pathogen_list_v3.txt"
with open(pathogen_file, 'w', encoding='utf-8') as f:
    for p in sorted(all_pathogens):
        f.write(p + '\n')
print(f"  Saved: {pathogen_file}")

# =============================================================================
# STEP 2: Load interaction dictionary
# =============================================================================
print("\n[2] LOADING INTERACTIONS...")

interactions = []
with open(interaction_dict_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if row:
            interactions.append(row[0].strip().lower())

print(f"  Total interactions: {len(interactions)}")

# Define noisy terms to filter
NOISY_TERMS = {
    'ate', 'eat', 'eated', 'eating', 'eats', 'eat by', 'eated by',
    'affecting', 'affected by',
}

# Define strong interaction terms (clearly biotic)
STRONG_TERMS = {
    'infect', 'infection', 'infected', 'infecting', 'infections',
    'infest', 'infested', 'infestation', 'infesting',
    'parasite', 'parasites', 'parasitize', 'parasitized', 'parasitizing',
    'pathogen', 'pathogens', 'pathogenic',
    'transmit', 'transmission', 'transmitted', 'transmitting',
    'vector', 'vectors',
    'host', 'hosts', 'hosted', 'hosting',
    'prey', 'preys', 'preyed', 'preying', 'predator', 'predators',
    'feed', 'feeds', 'fed', 'feeding',
    'hunt', 'hunts', 'hunted', 'hunting',
    'colonize', 'colonized', 'colonization',
    'invade', 'invaded', 'invasion', 'invading',
    'symbiont', 'symbiosis', 'symbiotic',
    'phoretic', 'phoresis',
    'trophic',
    'exposed to',
}

# =============================================================================
# STEP 3: Load and process sentences
# =============================================================================
print("\n[3] LOADING AND PROCESSING SENTENCES...")

positives = []
negatives = []

with open(training_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['label']) == 1:
            positives.append(row['passage'])
        else:
            negatives.append(row['passage'])

print(f"  Original positives: {len(positives)}")
print(f"  Original negatives: {len(negatives)}")

# =============================================================================
# STEP 4: Filter positives - remove only clearly bad ones
# =============================================================================
print("\n[4] FILTERING POSITIVES...")

def has_strong_interaction(s):
    """Check if sentence has a strong interaction term."""
    s_lower = s.lower()
    for term in STRONG_TERMS:
        if term in s_lower:
            # Special case: phoresis but not electrophoresis
            if 'phoresis' in term and 'electrophoresis' in s_lower:
                continue
            return term
    return None

def has_noisy_term(s):
    """Check if sentence only has noisy terms."""
    s_lower = s.lower()
    for term in NOISY_TERMS:
        if term in s_lower:
            return term
    return None

# Keep most positives - the original curation was good
# Only remove those with ONLY noisy terms
clean_positives = []
removed_positives = []

for sentence in positives:
    noisy = has_noisy_term(sentence)
    strong = has_strong_interaction(sentence)

    # Remove only if it has noisy term and NO strong term
    if noisy and not strong:
        removed_positives.append((sentence, noisy))
    else:
        clean_positives.append(sentence)

print(f"  Clean positives: {len(clean_positives)}")
print(f"  Removed positives (only noisy terms): {len(removed_positives)}")

# =============================================================================
# STEP 5: Find false negatives
# =============================================================================
print("\n[5] FINDING FALSE NEGATIVES...")

false_negatives = []
clean_negatives = []

for sentence in negatives:
    strong = has_strong_interaction(sentence)

    # If it has a strong interaction term, it might be a false negative
    if strong:
        false_negatives.append((sentence, strong))
    else:
        clean_negatives.append(sentence)

print(f"  Potential false negatives: {len(false_negatives)}")
print(f"  Clean negatives: {len(clean_negatives)}")

# Show examples
print("\n  Examples of potential FALSE NEGATIVES:")
for i, (sent, term) in enumerate(false_negatives[:10], 1):
    print(f"    {i}. Term: '{term}' -> {sent[:80]}...")

# =============================================================================
# STEP 6: Build final dataset
# =============================================================================
print("\n[6] BUILDING FINAL DATASET...")

# Final positives = clean positives + false negatives
final_positives = clean_positives + [s for s, _ in false_negatives]
final_negatives = clean_negatives

print(f"  Final positives: {len(final_positives)}")
print(f"  Final negatives: {len(final_negatives)}")

# Balance
min_size = min(len(final_positives), len(final_negatives))
target_size = min(min_size, 10000)

random.shuffle(final_positives)
random.shuffle(final_negatives)

balanced_positives = final_positives[:target_size]
balanced_negatives = final_negatives[:target_size]

print(f"\n  Balanced dataset:")
print(f"    Positives: {len(balanced_positives)}")
print(f"    Negatives: {len(balanced_negatives)}")
print(f"    Total: {len(balanced_positives) + len(balanced_negatives)}")

# =============================================================================
# STEP 7: Save dataset
# =============================================================================
print("\n[7] SAVING DATASET...")

output_file = output_dir / "training_data_v3.csv"
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['passage', 'label'])
    for sent in balanced_positives:
        writer.writerow([sent, 1])
    for sent in balanced_negatives:
        writer.writerow([sent, 0])

print(f"  Saved: {output_file}")

# =============================================================================
# STEP 8: Count interaction types in final positives
# =============================================================================
print("\n[8] INTERACTION BREAKDOWN IN FINAL POSITIVES...")

interaction_counts = Counter()
for sentence in balanced_positives:
    s_lower = sentence.lower()
    for interaction in interactions:
        if interaction in s_lower:
            # Skip phoresis if electrophoresis
            if 'phoresis' in interaction and 'electrophoresis' in s_lower:
                continue
            interaction_counts[interaction] += 1

print("\n  Top 30 interactions:")
for term, count in interaction_counts.most_common(30):
    pct = 100 * count / len(balanced_positives)
    print(f"    {term:<40}: {count:5d} ({pct:5.1f}%)")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
CHANGES FROM ORIGINAL 20k:
  - Removed {len(removed_positives)} positives with only noisy terms (e.g., 'ate', 'eating')
  - Promoted {len(false_negatives)} false negatives with strong interaction terms

FINAL DATASET:
  File: {output_file}
  Positives: {len(balanced_positives)}
  Negatives: {len(balanced_negatives)}
  Total: {len(balanced_positives) + len(balanced_negatives)}

PATHOGEN LIST:
  File: {pathogen_file}
  Count: {len(all_pathogens)}
""")

print("="*80)
