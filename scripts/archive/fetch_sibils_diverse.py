#!/usr/bin/env python3
"""
Fetch DIVERSE real sentences from SIBiLS MongoDB for training data.
Goal: Get authentic scientific literature sentences (not synthetic).

Strategy:
- Positives: Passages containing interaction VERBS (prey, feed, pollinate, etc.)
- Negatives: Passages with species mentions but NO interaction verbs
- Ensure diversity by sampling across different interaction types
"""

import pandas as pd
import numpy as np
import re
import time
from pymongo import MongoClient
from collections import Counter
import random

# Configuration
BASE_DIR = '/path/to/MetaP/classifier'
OUTPUT_FILE = f'{BASE_DIR}/data/training/sibils_diverse_real.csv'

# Target counts
TARGET_POSITIVES = 8000
TARGET_NEGATIVES = 8000

# MongoDB connection
MONGO_URI = "mongodb://sibils-mongodb.lan.text-analytics.ch:27017/"
DB_NAME = "sibils_v4_2"
COLLECTION_NAME = "med25_r1_v5.5_passages"

# Interaction verb patterns (regex)
INTERACTION_PATTERNS = [
    # Predation
    r'\b(prey(?:s|ed|ing)?\s+(?:on|upon))',
    r'\b(predat(?:e|es|ed|ing|ion|or|ors))',
    r'\b(hunt(?:s|ed|ing)?)\b',
    r'\b(eat(?:s|en|ing)?|ate)\s+(?:\w+\s+){0,3}(?:by|from|\w+(?:s|es))',
    r'\b(consum(?:e|es|ed|ing|ption))',
    r'\b(feed(?:s|ing)?)\s+(?:on|upon)',
    r'\b(fed\s+(?:on|upon))',
    # Parasitism
    r'\b(parasiti[zs](?:e|es|ed|ing))',
    r'\b(infect(?:s|ed|ing|ion)?)',
    r'\b(host(?:s|ed|ing)?)\s+(?:for|by|to)',
    r'\b(vector(?:s|ed)?)\s+(?:of|for|by)',
    # Pollination
    r'\b(pollinat(?:e|es|ed|ing|ion|or|ors))',
    # Symbiosis
    r'\b(symbio(?:sis|tic|nt|nts))',
    r'\b(mutualis(?:m|tic))',
    r'\b(commensalis(?:m|tic))',
    # Competition
    r'\b(compet(?:e|es|ed|ing|ition))\s+(?:with|for|against)',
    r'\b(displac(?:e|es|ed|ing))',
    r'\b(outcompet(?:e|es|ed|ing))',
    # Herbivory
    r'\b(herbivor(?:e|es|y|ous))',
    r'\b(graz(?:e|es|ed|ing))\s+(?:on|upon)?',
    r'\b(brows(?:e|es|ed|ing))\s+(?:on)?',
    # Dispersal
    r'\b(dispers(?:e|es|ed|ing|al))\s+(?:of|by)?(?:\s+seeds?)?',
    # Colonization
    r'\b(coloni[zs](?:e|es|ed|ing|ation))',
    # General interactions
    r'\b(interact(?:s|ed|ing|ion)?)\s+(?:with|between)',
    r'\b(association)\s+(?:with|between)',
    r'\b(relationship)\s+(?:with|between)',
]

# Compile patterns
COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INTERACTION_PATTERNS]


def normalize_passage(passage):
    """Light normalization - preserve original text quality."""
    if not passage:
        return ""
    passage = passage.strip()
    passage = re.sub(r'\s+', ' ', passage)
    passage = passage.replace('\u00A0', ' ')
    return passage


def is_valid_sentence(text):
    """Check if text is a valid sentence for training."""
    if not text or len(text) < 50:
        return False
    if len(text) > 800:
        return False
    if sum(c.isalpha() for c in text) < 30:
        return False
    if len(text.split()) < 8:
        return False
    return True


def find_interaction_type(text):
    """Find which interaction type is present in text."""
    for i, pattern in enumerate(COMPILED_PATTERNS):
        match = pattern.search(text)
        if match:
            return match.group(0).lower().strip()
    return None


def has_species_mentions(doc):
    """Check if document has species forms."""
    sp1 = doc.get('species1_form', [])
    sp2 = doc.get('species2_form', [])
    return bool(sp1) and bool(sp2)


def fetch_positives(collection, target=TARGET_POSITIVES):
    """Fetch positive passages containing interaction verbs."""
    print(f"\nFetching {target} positive passages with interaction verbs...")

    unique_passages = {}
    interaction_types = Counter()

    # Query documents that have both species mentioned
    query = {
        "species1_form": {"$ne": []},
        "species2_form": {"$ne": []}
    }

    print("  Scanning passages for interaction verbs...")
    scanned = 0

    for doc in collection.find(query):
        scanned += 1
        if scanned % 50000 == 0:
            print(f"    Scanned {scanned}, found {len(unique_passages)} positives...")

        passage = normalize_passage(doc.get("passage", ""))

        if not is_valid_sentence(passage):
            continue

        # Check for interaction verbs
        interaction_type = find_interaction_type(passage)
        if not interaction_type:
            continue

        passage_key = passage.lower().strip()
        if passage_key in unique_passages:
            continue

        interaction_types[interaction_type] += 1
        unique_passages[passage_key] = {
            'text': passage,
            'interaction_type': interaction_type,
            'species1': doc.get("species1_form", []),
            'species2': doc.get("species2_form", [])
        }

        if len(unique_passages) >= target:
            break

    print(f"  Scanned {scanned} documents total")
    print(f"  Retrieved {len(unique_passages)} unique positive passages")
    print(f"  Interaction types found: {len(interaction_types)}")
    print(f"  Top 10: {interaction_types.most_common(10)}")

    return list(unique_passages.values())


def fetch_negatives(collection, target=TARGET_NEGATIVES, exclude_keys=None):
    """Fetch negative passages - scientific text WITHOUT interaction verbs."""
    print(f"\nFetching {target} negative passages (no interaction verbs)...")

    unique_passages = {}
    exclude_set = exclude_keys or set()

    # Query documents with species but we'll filter out those with interactions
    query = {
        "species1_form": {"$ne": []},
        "species2_form": {"$ne": []}
    }

    scanned = 0
    for doc in collection.find(query):
        # Random sampling for diversity
        if random.random() > 0.2:
            continue

        scanned += 1
        if scanned % 50000 == 0:
            print(f"    Scanned {scanned}, found {len(unique_passages)} negatives...")

        passage = normalize_passage(doc.get("passage", ""))

        if not is_valid_sentence(passage):
            continue

        # EXCLUDE passages with interaction verbs
        if find_interaction_type(passage):
            continue

        passage_key = passage.lower().strip()
        if passage_key in unique_passages or passage_key in exclude_set:
            continue

        unique_passages[passage_key] = {
            'text': passage,
            'interaction_type': None,
            'species1': doc.get("species1_form", []),
            'species2': doc.get("species2_form", [])
        }

        if len(unique_passages) >= target:
            break

    print(f"  Scanned {scanned} documents total")
    print(f"  Retrieved {len(unique_passages)} unique negative passages")

    return list(unique_passages.values())


def main():
    start_time = time.time()

    print("="*70)
    print("FETCHING DIVERSE REAL SENTENCES FROM SIBiLS")
    print("Using KEYWORD-BASED filtering for interaction verbs")
    print("="*70)

    # Connect to MongoDB
    print(f"\nConnecting to MongoDB at {MONGO_URI}...")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)

    try:
        client.server_info()
        print("  Connected successfully!")
    except Exception as e:
        print(f"  ERROR: Could not connect to MongoDB: {e}")
        return

    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    total_docs = collection.count_documents({})
    print(f"  Total documents in collection: {total_docs:,}")

    # Fetch positives
    positives = fetch_positives(collection, TARGET_POSITIVES)
    positive_keys = set(p['text'].lower().strip() for p in positives)

    # Fetch negatives
    negatives = fetch_negatives(collection, TARGET_NEGATIVES, positive_keys)

    client.close()

    # Create DataFrame
    print("\n" + "="*70)
    print("CREATING DATASET")
    print("="*70)

    data = []
    for p in positives:
        data.append({
            'text': p['text'],
            'label': 1,
            'source': 'sibils_positive',
            'interaction_type': p['interaction_type']
        })

    for n in negatives:
        data.append({
            'text': n['text'],
            'label': 0,
            'source': 'sibils_negative',
            'interaction_type': None
        })

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    df.to_csv(OUTPUT_FILE, index=False)

    # Summary
    elapsed = time.time() - start_time
    print(f"\nDataset saved to: {OUTPUT_FILE}")
    print(f"Total samples: {len(df)}")
    print(f"  Positives: {sum(df['label']==1)}")
    print(f"  Negatives: {sum(df['label']==0)}")
    print(f"\nElapsed time: {elapsed:.1f} seconds")

    # Show samples
    print("\n" + "="*70)
    print("SAMPLE SENTENCES")
    print("="*70)
    print("\nPositive examples:")
    for _, row in df[df['label']==1].head(5).iterrows():
        print(f"  [{row['interaction_type']}] {row['text'][:120]}...")

    print("\nNegative examples:")
    for _, row in df[df['label']==0].head(5).iterrows():
        print(f"  - {row['text'][:120]}...")


if __name__ == "__main__":
    main()
