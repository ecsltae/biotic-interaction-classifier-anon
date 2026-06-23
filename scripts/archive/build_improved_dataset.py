#!/usr/bin/env python3
"""
Build improved 20k balanced dataset with diverse negative examples:
- 10,000 positives from true_positives.csv
- 10,000 negatives from multiple sources:
  * Co-occurrence sentences (species mentioned together, no interaction)
  * Scientific descriptions (taxonomy, morphology, distribution)
  * Multi-species lists (3+ species)
  * Random sentences from corpus
"""

import csv
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Paths
base_dir = Path("/path/to/MetaP/classifier")
true_pos_file = base_dir / "data/processed/true_positives.csv"
all_sentences_file = base_dir / "data/processed/unique_sentences.csv"
output_file = base_dir / "data/training/training_data_improved_20k.csv"

print("Building improved 20k dataset...")
print(f"Reading from: {true_pos_file}")
print(f"Reading from: {all_sentences_file}")
print(f"Output to: {output_file}")

# Read true positives (interactions)
print("\n1. Loading true positive interactions...")
positives = []
with open(true_pos_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        positives.append(row['passage'])

print(f"   Loaded {len(positives)} positive examples")

# Sample 10,000 positives
if len(positives) >= 10000:
    sampled_positives = random.sample(positives, 10000)
else:
    # If less than 10k, use all and duplicate some
    sampled_positives = positives * (10000 // len(positives) + 1)
    sampled_positives = random.sample(sampled_positives, 10000)

print(f"   Sampled {len(sampled_positives)} positives")

# Read all sentences for negative sampling
print("\n2. Loading all sentences for negative mining...")
all_sentences = []
with open(all_sentences_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        sentence = row['passage']
        # Skip if it's a known positive
        if sentence not in positives:
            all_sentences.append(sentence)

print(f"   Loaded {len(all_sentences)} candidate sentences")

# Define templates for synthetic co-occurrence negatives (no interaction)
co_occurrence_templates = [
    "Both {sp1} and {sp2} were observed in the study area.",
    "{sp1} and {sp2} are found in similar habitats.",
    "The distribution of {sp1} overlaps with that of {sp2}.",
    "{sp1} occurs sympatrically with {sp2} in many regions.",
    "We documented the presence of {sp1} and {sp2} in the same location.",
    "{sp1} and {sp2} were collected during the survey.",
    "The study area supports populations of both {sp1} and {sp2}.",
    "{sp1} and {sp2} share similar geographic ranges.",
    "Both {sp1} and {sp2} are common in this ecosystem.",
    "{sp1} was found alongside {sp2} in several sites.",
]

# Scientific description templates (no interaction)
scientific_templates = [
    "{sp1} is characterized by its distinctive morphology.",
    "The taxonomy of {sp1} has been revised recently.",
    "{sp1} belongs to the family {family} and is widely distributed.",
    "{sp1} exhibits significant genetic diversity across its range.",
    "The phylogenetic position of {sp1} remains debated.",
    "{sp1} shows remarkable adaptations to its environment.",
    "Conservation status of {sp1} is of concern due to habitat loss.",
    "{sp1} has a complex life cycle with multiple developmental stages.",
    "The coloration of {sp1} varies considerably among populations.",
    "{sp1} is endemic to specific geographic regions.",
]

# Common species names and families for templates
species_names = [
    "foxes", "rabbits", "deer", "wolves", "bears", "squirrels", "birds", "fish",
    "frogs", "lizards", "snakes", "turtles", "insects", "spiders", "bees", "ants",
    "mice", "rats", "bats", "owls", "eagles", "hawks", "sparrows", "finches",
    "salmon", "trout", "bass", "sharks", "whales", "dolphins", "seals", "otters",
]

families = [
    "Canidae", "Felidae", "Cervidae", "Ursidae", "Sciuridae", "Ranidae",
    "Colubridae", "Viperidae", "Salmonidae", "Delphinidae", "Accipitridae",
]

print("\n3. Generating diverse negative examples...")

# Generate co-occurrence negatives (2000)
print("   - Generating co-occurrence sentences...")
co_occurrence_negatives = []
for _ in range(2000):
    template = random.choice(co_occurrence_templates)
    sp1, sp2 = random.sample(species_names, 2)
    sentence = template.format(sp1=sp1, sp2=sp2)
    co_occurrence_negatives.append(sentence)

# Generate scientific description negatives (2000)
print("   - Generating scientific description sentences...")
scientific_negatives = []
for _ in range(2000):
    template = random.choice(scientific_templates)
    sp1 = random.choice(species_names)
    family = random.choice(families)
    sentence = template.format(sp1=sp1, family=family)
    scientific_negatives.append(sentence)

# Find multi-species sentences (3+ species mentioned, likely no single interaction)
print("   - Finding multi-species sentences (3+)...")
multi_species_keywords = [
    "species", "taxa", "organisms", "animals", "insects", "birds", "fish", "mammals",
    "reptiles", "amphibians", "invertebrates", "vertebrates", "community", "assemblage",
]

def count_potential_species(sentence):
    """Heuristic to count potential species mentions"""
    sentence_lower = sentence.lower()
    # Count commas and "and" as indicators of lists
    comma_count = sentence.lower().count(',')
    and_count = sentence.lower().count(' and ')
    # Check for list keywords
    has_list_keyword = any(kw in sentence_lower for kw in ['including', 'such as', 'e.g.', 'i.e.'])
    # Estimate species count
    estimated_count = comma_count + and_count
    if has_list_keyword:
        estimated_count += 2
    return estimated_count

multi_species = [s for s in all_sentences if count_potential_species(s) >= 3]
random.shuffle(multi_species)
sampled_multi = multi_species[:2000] if len(multi_species) >= 2000 else multi_species

# Sample random sentences from corpus (4000 + whatever is needed to reach 10k)
print("   - Sampling random sentences from corpus...")
remaining_needed = 10000 - len(co_occurrence_negatives) - len(scientific_negatives) - len(sampled_multi)
random.shuffle(all_sentences)
random_samples = all_sentences[:remaining_needed]

# Combine all negatives
all_negatives = co_occurrence_negatives + scientific_negatives + sampled_multi + random_samples
all_negatives = all_negatives[:10000]  # Ensure exactly 10k

print(f"\n   Negative breakdown:")
print(f"   - Co-occurrence: {len(co_occurrence_negatives)}")
print(f"   - Scientific descriptions: {len(scientific_negatives)}")
print(f"   - Multi-species (3+): {len(sampled_multi)}")
print(f"   - Random corpus: {len(random_samples)}")
print(f"   - Total negatives: {len(all_negatives)}")

# Create final dataset
print("\n4. Creating balanced dataset...")
dataset = []
for sent in sampled_positives:
    dataset.append({'sentence': sent, 'label': 1})
for sent in all_negatives:
    dataset.append({'sentence': sent, 'label': 0})

# Shuffle the dataset
random.shuffle(dataset)

print(f"   Total samples: {len(dataset)}")
print(f"   Positives: {sum(1 for d in dataset if d['label'] == 1)}")
print(f"   Negatives: {sum(1 for d in dataset if d['label'] == 0)}")

# Write to CSV
print(f"\n5. Writing to {output_file}...")
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['sentence', 'label'])
    writer.writeheader()
    writer.writerows(dataset)

print(f"\n✓ Successfully created improved dataset with {len(dataset)} samples")
print(f"  Output: {output_file}")
