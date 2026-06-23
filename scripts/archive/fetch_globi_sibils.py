#!/usr/bin/env python3
"""
Fetch REAL sentences from GloBI-referenced articles via SIBiLS.

Two complementary strategies:
1. SIBiLS Search API: Search for biodiversity articles by interaction keywords
   -> Get full text -> Extract sentences
2. SIBiLS MongoDB: Query passages by GloBI PMIDs directly

Result: Real ecological literature sentences for training.
"""

import pandas as pd
import numpy as np
import re
import time
import random
import json
import requests
from pymongo import MongoClient
from collections import Counter

BASE_DIR = '/path/to/MetaP/classifier'
GLOBI_FILE = f'{BASE_DIR}/data/globi/interactions.tsv.gz'
OUTPUT_FILE = f'{BASE_DIR}/data/training/globi_sibils_real.csv'

# SIBiLS API
SIBILS_API = "https://biodiversitypmc.sibils.org/api/search"

# MongoDB
MONGO_URI = "mongodb://sibils-mongodb.lan.text-analytics.ch:27017/"
DB_NAME = "sibils_v4_2"
COLLECTION_NAME = "med25_r1_v5.5_passages"

# Targets
TARGET_POSITIVES = 5000
TARGET_NEGATIVES = 5000
MAX_SENTS_PER_ARTICLE = 30  # Cap per article to avoid one article dominating
POS_PER_CATEGORY = 600  # Spread positives across categories

# Interaction search queries for the SIBiLS API
INTERACTION_QUERIES = [
    # Predation
    ("predation prey", "predation"),
    ("predator prey relationship", "predation"),
    ("feeding ecology diet prey", "predation"),
    # Pollination
    ("pollination flower visitor", "pollination"),
    ("pollinator plant interaction", "pollination"),
    ("bee pollination flower", "pollination"),
    # Parasitism
    ("parasite host ecology", "parasitism"),
    ("parasitoid host", "parasitism"),
    ("ectoparasite host", "parasitism"),
    # Herbivory
    ("herbivore plant grazing", "herbivory"),
    ("herbivory leaf damage", "herbivory"),
    # Symbiosis
    ("mutualism symbiosis", "symbiosis"),
    ("symbiont host", "symbiosis"),
    # Competition
    ("interspecific competition species", "competition"),
    # Dispersal
    ("seed dispersal animal", "dispersal"),
    # General
    ("biotic interaction species", "general"),
    ("species interaction ecology", "general"),
    ("trophic interaction food web", "general"),
    ("host pathogen interaction", "pathogen"),
]

# Sentence splitting regex
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

# Interaction verb patterns for labeling
INTERACTION_VERBS = re.compile(
    r'\b('
    r'prey(?:s|ed|ing)?\s+(?:on|upon)|'
    r'predat(?:e|es|ed|ing|ion|or|ors)|'
    r'hunt(?:s|ed|ing)?\b|'
    r'eat(?:s|en|ing)?|ate\b|'
    r'consum(?:e|es|ed|ing|ption)|'
    r'feed(?:s|ing)?\s+(?:on|upon)|'
    r'fed\s+(?:on|upon)|'
    r'forag(?:e|es|ed|ing)\s+(?:on|for)|'
    r'parasiti[zs](?:e|es|ed|ing)|'
    r'infect(?:s|ed|ing|ion)|'
    r'host(?:s|ed|ing)?\s+(?:for|by|to|of)|'
    r'vector(?:s|ed)?\s+(?:of|for|by)|'
    r'pollinat(?:e|es|ed|ing|ion|or|ors)|'
    r'visit(?:s|ed|ing)?\s+(?:the\s+)?flower|'
    r'symbio(?:sis|tic|nt|nts)|'
    r'mutualis(?:m|tic)|'
    r'commensalis(?:m|tic)|'
    r'compet(?:e|es|ed|ing|ition)\s+(?:with|for)|'
    r'outcompet(?:e|es|ed|ing)|'
    r'dispers(?:e|es|ed|ing|al)\s+(?:of|by)?(?:\s+seeds?)?|'
    r'coloni[zs](?:e|es|ed|ing|ation)|'
    r'herbivor(?:e|es|y|ous)|'
    r'graz(?:e|es|ed|ing)\s+(?:on|upon)?|'
    r'brows(?:e|es|ed|ing)\s+(?:on)?|'
    r'pathogen(?:ic|s)?\s+(?:of|to|for)|'
    r'transmit(?:s|ted|ting)?\s+(?:by|to|from)|'
    r'interact(?:s|ed|ing|ion)?\s+(?:with|between)|'
    r'(?:definitive|intermediate|paratenic)\s+host|'
    r'(?:prey|food)\s+(?:item|species|of)|'
    r'(?:larval|adult)\s+(?:parasitoid|parasite)\s+of'
    r')\b',
    re.IGNORECASE
)


def is_valid_sentence(text):
    """Check if text is a valid sentence for training."""
    if not text or len(text) < 50 or len(text) > 600:
        return False
    if sum(c.isalpha() for c in text) < 30:
        return False
    if len(text.split()) < 8:
        return False
    # Skip references, figure captions
    if re.match(r'^\s*(Fig|Figure|Table|Supplementary|S\d)', text):
        return False
    if re.search(r'\d{4};\s*\d+\s*:', text):  # Citation patterns
        return False
    return True


def extract_sentences(full_text):
    """Split full text into individual sentences."""
    if not full_text:
        return []
    # Remove section headers (lines that are too short)
    lines = full_text.split('\n')
    text_lines = [l.strip() for l in lines if len(l.strip()) > 40]
    text = ' '.join(text_lines)
    # Split into sentences
    sentences = SENT_SPLIT.split(text)
    return [s.strip() for s in sentences if is_valid_sentence(s)]


def fetch_from_sibils_api():
    """Use SIBiLS Search API to find biodiversity articles and extract sentences.

    Enforces diversity:
    - Per-category quota for positives (POS_PER_CATEGORY)
    - Per-article cap (MAX_SENTS_PER_ARTICLE)
    - ALL categories queried regardless of total count
    """
    print("\n--- Strategy 1: SIBiLS Search API ---")

    positives = {}
    negatives = {}
    articles_processed = 0
    seen_pmids = set()

    # Get unique categories and compute per-category quota
    categories = list(set(cat for _, cat in INTERACTION_QUERIES))
    cat_pos_count = {cat: 0 for cat in categories}

    for query, category in INTERACTION_QUERIES:
        if cat_pos_count[category] >= POS_PER_CATEGORY:
            print(f"\n  Skipping '{query}' ({category}) - category quota reached ({POS_PER_CATEGORY})")
            continue

        print(f"\n  Searching: '{query}' ({category})...")

        try:
            resp = requests.get(SIBILS_API, params={
                'q': query,
                'col': 'pmc',
                'n': 100
            }, timeout=30)
            data = resp.json()
        except Exception as e:
            print(f"    Error: {e}")
            continue

        hits = data.get('elastic_output', {}).get('hits', {}).get('hits', [])
        print(f"    Found {len(hits)} articles")

        query_pos = 0
        query_neg = 0

        for hit in hits:
            if cat_pos_count[category] >= POS_PER_CATEGORY:
                break

            source = hit.get('_source', {})
            full_text = source.get('full_text', '') or ''
            abstract = source.get('abstract', '') or ''
            pmid = source.get('pmid', '')
            seen_pmids.add(pmid)

            all_sentences = extract_sentences(full_text) + extract_sentences(abstract)
            random.shuffle(all_sentences)  # Randomize to avoid always picking first sentences

            article_count = 0
            for sent in all_sentences:
                if article_count >= MAX_SENTS_PER_ARTICLE:
                    break

                sent_key = sent.lower().strip()
                has_interaction = bool(INTERACTION_VERBS.search(sent))

                if has_interaction and sent_key not in positives:
                    if cat_pos_count[category] < POS_PER_CATEGORY:
                        positives[sent_key] = {
                            'text': sent,
                            'pmid': pmid,
                            'source': f'sibils_api_{category}',
                        }
                        cat_pos_count[category] += 1
                        query_pos += 1
                        article_count += 1
                elif not has_interaction and sent_key not in negatives and sent_key not in positives:
                    negatives[sent_key] = {
                        'text': sent,
                        'pmid': pmid,
                        'source': f'sibils_api_{category}',
                    }
                    query_neg += 1
                    article_count += 1

            articles_processed += 1

        print(f"    -> {query_pos} pos, {query_neg} neg (category total: {cat_pos_count[category]} pos)")
        time.sleep(0.5)

    # Trim negatives to match positives
    total_pos = len(positives)
    if len(negatives) > total_pos:
        neg_keys = list(negatives.keys())
        random.shuffle(neg_keys)
        negatives = {k: negatives[k] for k in neg_keys[:total_pos]}

    print(f"\n  API Results: {len(positives)} positives, {len(negatives)} negatives")
    print(f"  From {articles_processed} articles, {len(seen_pmids)} unique PMIDs")
    print(f"\n  Per-category positives:")
    for cat in sorted(cat_pos_count, key=cat_pos_count.get, reverse=True):
        if cat_pos_count[cat] > 0:
            print(f"    {cat}: {cat_pos_count[cat]}")

    return positives, negatives


def extract_pmids_from_globi():
    """Extract PMIDs from GloBI interactions data."""
    print("\n--- Extracting PMIDs from GloBI ---")
    cols = ['interactionTypeName', 'referenceUrl', 'referenceCitation', 'sourceCitation', 'sourceArchiveURI']
    df = pd.read_csv(GLOBI_FILE, sep='\t', usecols=cols, nrows=3000000, low_memory=False)
    print(f"  Loaded {len(df)} rows")

    pmid_pattern = re.compile(r'(?:pubmed|pmid|PMID)[/:]?\s*(\d{6,9})')
    pmid_interactions = {}
    for _, row in df.iterrows():
        interaction = row.get('interactionTypeName', '')
        if pd.isna(interaction):
            continue
        for col in ['referenceUrl', 'referenceCitation', 'sourceCitation', 'sourceArchiveURI']:
            val = row.get(col)
            if pd.isna(val):
                continue
            for pmid in pmid_pattern.findall(str(val)):
                if pmid not in pmid_interactions:
                    pmid_interactions[pmid] = set()
                pmid_interactions[pmid].add(interaction)

    print(f"  Found {len(pmid_interactions)} unique PMIDs")
    return pmid_interactions


def fetch_from_mongodb(pmid_interactions, existing_pos, existing_neg, target_pos, target_neg):
    """Query SIBiLS MongoDB for passages from GloBI PMIDs."""
    print("\n--- Strategy 2: SIBiLS MongoDB (GloBI PMIDs) ---")

    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    positives = dict(existing_pos)
    negatives = dict(existing_neg)
    pmids_checked = 0

    pmid_list = list(pmid_interactions.keys())
    random.shuffle(pmid_list)

    for pmid in pmid_list:
        if len(positives) >= target_pos and len(negatives) >= target_neg:
            break

        pmids_checked += 1
        if pmids_checked % 2000 == 0:
            print(f"  Checked {pmids_checked} PMIDs, {len(positives)} pos / {len(negatives)} neg")

        for doc in collection.find({"doc_id": pmid}).limit(10):
            passage = doc.get("passage", "").strip()
            passage = re.sub(r'\s+', ' ', passage)

            if not is_valid_sentence(passage):
                continue

            pkey = passage.lower().strip()
            has_interaction = bool(INTERACTION_VERBS.search(passage))

            if has_interaction and pkey not in positives and len(positives) < target_pos:
                positives[pkey] = {
                    'text': passage,
                    'pmid': pmid,
                    'source': 'globi_mongodb',
                }
            elif not has_interaction and pkey not in negatives and pkey not in positives:
                if len(negatives) < target_neg:
                    negatives[pkey] = {
                        'text': passage,
                        'pmid': pmid,
                        'source': 'globi_mongodb',
                    }

    client.close()
    print(f"  MongoDB Results: {len(positives)} positives, {len(negatives)} negatives ({pmids_checked} PMIDs checked)")
    return positives, negatives


def main():
    start_time = time.time()

    print("="*70)
    print("FETCH REAL SENTENCES: GloBI + SIBiLS")
    print("="*70)

    # Strategy 1: SIBiLS Search API (diverse, ecological)
    api_pos, api_neg = fetch_from_sibils_api()

    all_pos, all_neg = api_pos, api_neg

    # Strategy 2: GloBI PMIDs -> MongoDB (supplement if needed)
    if len(all_pos) < TARGET_POSITIVES * 0.8 or len(all_neg) < TARGET_NEGATIVES * 0.8:
        print(f"\n  Need more data ({len(all_pos)} pos, {len(all_neg)} neg). Trying MongoDB...")
        pmid_interactions = extract_pmids_from_globi()
        all_pos, all_neg = fetch_from_mongodb(
            pmid_interactions, api_pos, api_neg,
            TARGET_POSITIVES, TARGET_NEGATIVES
        )
    else:
        print("\n  Sufficient data from API, skipping MongoDB.")

    # Create dataset
    print("\n" + "="*70)
    print("CREATING DATASET")
    print("="*70)

    data = []
    for p in all_pos.values():
        data.append({'text': p['text'], 'label': 1, 'source': p['source'], 'pmid': p['pmid']})
    for n in all_neg.values():
        data.append({'text': n['text'], 'label': 0, 'source': n['source'], 'pmid': n['pmid']})

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False)

    elapsed = time.time() - start_time
    print(f"\nSaved to: {OUTPUT_FILE}")
    print(f"Total: {len(df)} ({sum(df['label']==1)} pos / {sum(df['label']==0)} neg)")
    print(f"Unique PMIDs: {df['pmid'].nunique()}")
    print(f"\nBy source:")
    for src, cnt in df['source'].value_counts().items():
        print(f"  {src}: {cnt}")
    print(f"\nTime: {elapsed:.0f}s")

    print("\nPositive samples:")
    for _, row in df[df['label']==1].head(3).iterrows():
        print(f"  [{row['source']}] {row['text'][:120]}...")
    print("\nNegative samples:")
    for _, row in df[df['label']==0].head(3).iterrows():
        print(f"  [{row['source']}] {row['text'][:120]}...")


if __name__ == "__main__":
    main()
