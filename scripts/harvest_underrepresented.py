#!/usr/bin/env python3
"""
harvest_underrepresented.py — Harvest training sentences for underrepresented interaction types.

Targets: herbivory, mutualism, seed dispersal (currently 0-15 examples each in v12 data).
Uses Europe PMC API to search for relevant sentences.

Usage:
    python classifier/scripts/harvest_underrepresented.py --category herbivory --count 100
"""

import argparse
import json
import re
import time
from pathlib import Path
import requests
import pandas as pd

BASE_DIR = Path('/path/to/MetaP/classifier')
EUROPE_PMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

# Target interaction categories with example search queries
CATEGORY_QUERIES = {
    "herbivory": [
        "herbivore feeding plant",
        "insect herbivory damage",
        "bark beetle host tree",
        "caterpillar feeding leaf",
        "deer browsing vegetation",
        "beetle larvae phloem",
        "aphid feeding plant sap",
        "grasshopper herbivory",
        "leaf miner host plant",
        "wood borer tree damage",
    ],
    "mutualism": [
        "mycorrhizal fungi plant root",
        "cleaner fish mutualism",
        "pollinator plant mutualism",
        "nitrogen fixing bacteria legume",
        "coral zooxanthellae symbiosis",
        "ant plant mutualism",
        "honeybee pollination",
        "endophyte plant mutualism",
        "oxpecker mammal symbiosis",
        "lichen mutualism fungus alga",
    ],
    "seed_dispersal": [
        "seed dispersal bird",
        "frugivore seed dispersal",
        "bat fruit dispersal",
        "elephant seed dispersal",
        "scatter-hoarding rodent",
        "seed dispersal primate",
        "ant seed dispersal myrmecochory",
        "fish seed dispersal",
        "endozoochory seed dispersal",
        "seed dispersal feces dung",
    ],
    "indirect": [
        "vector transmission pathogen",
        "intermediate host parasite",
        "reservoir host disease",
        "spillover infection wildlife",
        "zoonotic transmission",
        "migratory bird pathogen spread",
        "environmental contamination pathogen",
        "mechanical vector transmission",
        "waterborne pathogen transmission",
        "fomite disease transmission",
    ],
}


def search_europe_pmc(query: str, cursor: str = "*", page_size: int = 100) -> dict:
    """Search Europe PMC for articles matching query."""
    params = {
        "query": query,
        "format": "json",
        "pageSize": page_size,
        "cursorMark": cursor,
        "resultType": "core",
    }
    response = requests.get(EUROPE_PMC_API, params=params)
    response.raise_for_status()
    return response.json()


def extract_sentences(text: str) -> list:
    """Extract sentences from text."""
    if not text:
        return []
    # Simple sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Filter to reasonable length sentences (10-50 words)
    valid = []
    for s in sentences:
        words = len(s.split())
        if 10 <= words <= 50:
            valid.append(s.strip())
    return valid


def harvest_category(category: str, max_sentences: int = 100) -> pd.DataFrame:
    """Harvest sentences for a specific interaction category."""
    if category not in CATEGORY_QUERIES:
        raise ValueError(f"Unknown category: {category}")

    queries = CATEGORY_QUERIES[category]
    all_sentences = []
    seen_texts = set()

    print(f"Harvesting {category} sentences...")

    for query in queries:
        if len(all_sentences) >= max_sentences:
            break

        print(f"  Query: {query}")
        try:
            result = search_europe_pmc(query)
            articles = result.get("resultList", {}).get("result", [])

            for article in articles:
                if len(all_sentences) >= max_sentences:
                    break

                # Get abstract
                abstract = article.get("abstractText", "")
                sentences = extract_sentences(abstract)

                for sent in sentences:
                    # Skip if already seen
                    if sent.lower() in seen_texts:
                        continue

                    # Basic quality filter: must contain species-like terms
                    if not re.search(r'[A-Z][a-z]+ [a-z]+', sent):
                        continue

                    seen_texts.add(sent.lower())
                    all_sentences.append({
                        "text": sent,
                        "category": category,
                        "query": query,
                        "pmid": article.get("pmid", ""),
                        "title": article.get("title", ""),
                    })

                    if len(all_sentences) >= max_sentences:
                        break

            time.sleep(0.3)  # Rate limiting

        except Exception as e:
            print(f"    Error: {e}")
            continue

    print(f"  Found {len(all_sentences)} sentences")
    return pd.DataFrame(all_sentences)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True,
                       choices=list(CATEGORY_QUERIES.keys()),
                       help="Interaction category to harvest")
    parser.add_argument("--count", type=int, default=100,
                       help="Number of sentences to harvest")
    parser.add_argument("--output", default=None,
                       help="Output CSV path")
    args = parser.parse_args()

    df = harvest_category(args.category, args.count)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = BASE_DIR / f"results/research_agent/harvest_{args.category}.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} sentences to {output_path}")


if __name__ == "__main__":
    main()
