#!/usr/bin/env python3
"""
Build v9 Dataset: SIBiLS PMC Full-Text Extraction

Uses SIBiLS biodiversity PMC API to fetch full-text articles for GloBI
species pairs, then extracts real interaction sentences from the full text.

Pipeline:
  1. Load GloBI interactions (species pairs + interaction types + DOIs/PMIDs)
  2. Query SIBiLS PMC collection for full-text articles per interaction type
  3. Extract real sentences with relaxed matching (2+ taxa + interaction verbs)
  4. Also extract targeted sentences when GloBI species pair is found
  5. Generate hard negatives from same articles (2+ taxa, no interaction verb)
  6. Merge with v8, deduplicate, save as v9

Key improvement over v8: SIBiLS PMC returns full-text articles (not just
abstracts), yielding more diverse and authentic interaction sentences.
"""

import sys
import json
import time
import hashlib
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

import requests
import pandas as pd
import numpy as np

# Setup paths
CLASSIFIER_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(CLASSIFIER_ROOT))

from src.data.sentence_extractor import (
    extract_best_interaction_sentences,
    extract_interaction_sentences_relaxed,
    extract_matching_sentences,
    split_sentences,
    find_all_taxa,
    find_all_interactions,
    RelaxedSentenceMatch,
)
from src.data.globi_loader import (
    extract_pmid_from_url,
    filter_by_interaction_types,
    get_interaction_stats,
    GLOBI_ZENODO_URL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

SIBILS_API_URL = "https://biodiversitypmc.sibils.org/api/search"
SIBILS_COLLECTIONS = ["pmc", "medline"]  # PMC first for full text

# Interaction queries for SIBiLS (species pair + interaction term)
INTERACTION_QUERIES = {
    "parasitism": [
        "{source} parasitizes {target}",
        "{source} parasite {target}",
        "parasitic infection {source} {target}",
    ],
    "predation": [
        "{source} preys on {target}",
        "{source} predator {target}",
        "predation {source} {target}",
    ],
    "infection": [
        "{source} infects {target}",
        "{source} pathogen {target}",
        "infection {source} {target} host",
    ],
    "pollination": [
        "{source} pollinates {target}",
        "{source} pollinator {target}",
        "pollination {source} {target}",
    ],
    "herbivory": [
        "{source} feeds on {target}",
        "{source} herbivore {target}",
        "herbivory {source} {target}",
    ],
    "vector": [
        "{source} vector {target}",
        "{source} transmits {target}",
        "transmission {source} {target}",
    ],
    "symbiosis": [
        "{source} symbiont {target}",
        "symbiosis {source} {target}",
        "mutualism {source} {target}",
    ],
    "competition": [
        "competition {source} {target}",
        "{source} competes {target}",
    ],
}

# GloBI interaction type -> query category mapping
GLOBI_TYPE_MAP = {
    "parasiteOf": "parasitism",
    "hasHost": "parasitism",
    "endoparasiteOf": "parasitism",
    "ectoParasiteOf": "parasitism",
    "parasitoidOf": "parasitism",
    "preysOn": "predation",
    "preyedUponBy": "predation",
    "eats": "herbivory",
    "flowersVisitedBy": "pollination",
    "pollinates": "pollination",
    "visitsFlowersOf": "pollination",
    "pathogenOf": "infection",
    "vectorOf": "vector",
    "symbioticWith": "symbiosis",
    "mutualistOf": "symbiosis",
    "interactsWith": "parasitism",  # fallback
    "kills": "predation",
    "hasDispersalVector": "symbiosis",
}

# Dataset targets
TARGET_POSITIVES = 6000  # New sentences from SIBiLS
TARGET_NEGATIVES_RATIO = 1.0  # 1:1 with new positives
MAX_PER_INTERACTION_TYPE = 1000
MAX_ARTICLES_PER_QUERY = 50
MAX_SENTENCES_PER_ARTICLE = 5
RATE_LIMIT_DELAY = 0.5  # seconds between SIBiLS requests

# File paths
V8_FILE = CLASSIFIER_ROOT / "data/training/training_data_globi_v8.csv"
OUTPUT_FILE = CLASSIFIER_ROOT / "data/training/training_data_globi_v9.csv"
CACHE_DIR = CLASSIFIER_ROOT / "data/sibils_cache"
METADATA_FILE = CLASSIFIER_ROOT / "data/training/v9_metadata.json"


# ============================================================================
# SIBiLS Full-Text Fetcher
# ============================================================================

@dataclass
class SibilsArticle:
    """Article retrieved from SIBiLS API."""
    doc_id: str
    title: str
    abstract: str
    full_text: Optional[str]
    pmid: Optional[str]
    pmcid: Optional[str]
    doi: Optional[str]
    score: float
    collection: str

    def get_text(self) -> str:
        """Get best available text (full text > abstract)."""
        if self.full_text:
            return self.full_text
        return f"{self.title}. {self.abstract}" if self.abstract else self.title or ""


class SibilsCache:
    """File-based cache for SIBiLS API responses."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, query: str, collection: str) -> Path:
        h = hashlib.md5(f"{query}:{collection}".encode()).hexdigest()
        return self.cache_dir / f"{h}.json"

    def get(self, query: str, collection: str) -> Optional[List[dict]]:
        path = self._key_path(query, collection)
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def set(self, query: str, collection: str, hits: List[dict]) -> None:
        path = self._key_path(query, collection)
        with open(path, "w") as f:
            json.dump(hits, f)


def search_sibils(
    query: str,
    collection: str = "pmc",
    n: int = 50,
    cache: Optional[SibilsCache] = None,
    timeout: int = 30,
) -> List[SibilsArticle]:
    """
    Search SIBiLS API for articles.

    Args:
        query: Search query (species names + interaction terms)
        collection: SIBiLS collection ("pmc", "medline", "plazi", "suppdata")
        n: Max number of results
        cache: Optional cache for responses
        timeout: HTTP timeout in seconds

    Returns:
        List of SibilsArticle objects
    """
    # Check cache
    if cache:
        cached = cache.get(query, collection)
        if cached is not None:
            return [_parse_hit(h, collection) for h in cached]

    params = {"q": query, "col": collection, "n": n}

    try:
        response = requests.get(SIBILS_API_URL, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()

        if not data.get("success", False):
            logger.warning(f"SIBiLS error for query '{query[:50]}': {data.get('error')}")
            return []

        hits = data.get("elastic_output", {}).get("hits", {}).get("hits", [])

        # Cache raw hits
        if cache:
            cache.set(query, collection, hits)

        return [_parse_hit(h, collection) for h in hits]

    except requests.RequestException as e:
        logger.warning(f"SIBiLS request failed: {e}")
        return []


def _parse_hit(hit: dict, collection: str) -> SibilsArticle:
    """Parse a SIBiLS API hit into a SibilsArticle."""
    source = hit.get("_source", {})
    return SibilsArticle(
        doc_id=hit.get("_id", ""),
        title=source.get("title", ""),
        abstract=source.get("abstract", ""),
        full_text=source.get("full_text"),
        pmid=source.get("pmid"),
        pmcid=source.get("pmcid"),
        doi=source.get("doi"),
        score=hit.get("_score", 0.0),
        collection=collection,
    )


# ============================================================================
# Sentence Extraction from SIBiLS Articles
# ============================================================================

MAX_TEXT_LENGTH = 500_000  # Truncate texts longer than this to avoid spaCy OOM


def extract_positives_from_article(
    article: SibilsArticle,
    source_species: Optional[str] = None,
    target_species: Optional[str] = None,
    interaction_type: Optional[str] = None,
) -> List[dict]:
    """
    Extract positive interaction sentences from a SIBiLS article.

    Uses two strategies:
    1. TARGETED: If species pair provided, look for specific matches
    2. RELAXED: Find any sentences with 2+ taxa + interaction verbs

    Returns list of dicts with: text, label, source_species, target_species,
    interaction_type, quality_score, pmid, pmcid, source
    """
    text = article.get_text()
    if not text or len(text) < 100:
        return []

    # Truncate very long texts to avoid spaCy max_length errors
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]

    results = []
    seen_sentences = set()

    # Strategy 1: Targeted extraction (if species pair available)
    if source_species and target_species and interaction_type:
        matches = extract_matching_sentences(
            text, source_species, target_species, interaction_type
        )
        for m in matches[:3]:
            sent_hash = hashlib.md5(m.sentence.encode()).hexdigest()
            if sent_hash not in seen_sentences:
                seen_sentences.add(sent_hash)
                results.append({
                    "text": m.sentence,
                    "label": 1,
                    "source_species": source_species,
                    "target_species": target_species,
                    "interaction_type": interaction_type,
                    "quality_score": 90.0,  # High quality: specific match
                    "pmid": article.pmid,
                    "pmcid": article.pmcid,
                    "source": "sibils_pmc_targeted",
                })

    # Strategy 2: Relaxed extraction (any interaction sentences)
    relaxed = extract_best_interaction_sentences(
        text,
        max_sentences=MAX_SENTENCES_PER_ARTICLE,
        doi=article.doi,
        pmid=article.pmid,
    )
    for m in relaxed:
        sent_hash = hashlib.md5(m.sentence.encode()).hexdigest()
        if sent_hash not in seen_sentences:
            seen_sentences.add(sent_hash)
            # Infer interaction type from found terms
            inferred_type = _infer_interaction_type(m.interactions_found)
            # Infer species from taxa found
            sp1 = m.taxa_found[0] if len(m.taxa_found) > 0 else ""
            sp2 = m.taxa_found[1] if len(m.taxa_found) > 1 else ""
            results.append({
                "text": m.sentence,
                "label": 1,
                "source_species": sp1,
                "target_species": sp2,
                "interaction_type": inferred_type,
                "quality_score": 70.0 + min(m.n_taxa, 4) * 2.5 + min(m.n_interactions, 3) * 2.5,
                "pmid": article.pmid,
                "pmcid": article.pmcid,
                "source": "sibils_pmc_relaxed",
            })

    return results


def extract_negatives_from_article(
    article: SibilsArticle,
    max_negatives: int = 3,
) -> List[dict]:
    """
    Extract hard negative sentences from article text.

    Hard negatives: sentences with 2+ taxa but NO interaction verb.
    These are the most challenging for the classifier.
    """
    text = article.get_text()
    if not text or len(text) < 100:
        return []

    # Truncate very long texts
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]

    sentences = split_sentences(text)
    negatives = []

    for sentence in sentences:
        if len(sentence) < 40 or len(sentence) > 400:
            continue

        taxa = find_all_taxa(sentence)
        interactions = find_all_interactions(sentence)

        # Hard negative: 2+ taxa, 0 interaction terms
        if len(taxa) >= 2 and len(interactions) == 0:
            negatives.append({
                "text": sentence,
                "label": 0,
                "source_species": taxa[0] if taxa else "",
                "target_species": taxa[1] if len(taxa) > 1 else "",
                "interaction_type": "none_two_species",
                "quality_score": 75.0,
                "pmid": article.pmid,
                "pmcid": article.pmcid,
                "source": "sibils_pmc_hard_negative",
            })

        if len(negatives) >= max_negatives:
            break

    return negatives


def _infer_interaction_type(interaction_terms: List[str]) -> str:
    """Infer interaction category from found interaction terms."""
    terms_lower = [t.lower() for t in interaction_terms]
    joined = " ".join(terms_lower)

    if any(t in joined for t in ["parasit", "infest"]):
        return "parasitism"
    if any(t in joined for t in ["infect", "pathogen", "disease"]):
        return "infection"
    if any(t in joined for t in ["prey", "predat", "hunt"]):
        return "predation"
    if any(t in joined for t in ["pollinat", "flower"]):
        return "pollination"
    if any(t in joined for t in ["feed", "eat", "consum", "herbivor", "graz"]):
        return "herbivory"
    if any(t in joined for t in ["vector", "transmit", "carrier"]):
        return "vector"
    if any(t in joined for t in ["symbio", "mutuali"]):
        return "symbiosis"
    if any(t in joined for t in ["compet"]):
        return "competition"
    if any(t in joined for t in ["dispers"]):
        return "dispersal"
    if any(t in joined for t in ["host"]):
        return "host"
    if any(t in joined for t in ["kill", "attack"]):
        return "predation"
    return "interaction"


# ============================================================================
# Main Pipeline
# ============================================================================

def load_globi_chunked(
    filepath: str,
    interaction_types: List[str],
    max_rows: int = 500_000,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """
    Load GloBI interactions in chunks to avoid OOM on the 2.3GB file.

    Only keeps rows that:
    - Match the requested interaction types
    - Have a DOI or PMID reference
    """
    # Minimal columns to reduce memory
    use_cols = [
        "sourceTaxonName", "targetTaxonName", "interactionTypeName",
        "referenceDoi", "referenceUrl",
    ]

    logger.info(f"Loading GloBI from {filepath} in chunks of {chunksize}...")
    chunks = []
    total_read = 0

    for chunk in pd.read_csv(
        filepath, sep="\t", usecols=use_cols,
        chunksize=chunksize, compression="infer", low_memory=False,
    ):
        total_read += len(chunk)

        # Filter to target interaction types
        chunk = chunk[chunk["interactionTypeName"].isin(interaction_types)]

        # Filter to rows with DOI or PMID in URL
        has_doi = chunk["referenceDoi"].notna() & (chunk["referenceDoi"] != "")
        has_pmid_url = chunk["referenceUrl"].apply(
            lambda u: bool(extract_pmid_from_url(u)) if pd.notna(u) else False
        )
        chunk = chunk[has_doi | has_pmid_url].copy()

        if len(chunk) > 0:
            # Extract PMID from URL
            chunk["pmid"] = chunk["referenceUrl"].apply(extract_pmid_from_url)
            chunks.append(chunk)

        if sum(len(c) for c in chunks) >= max_rows:
            logger.info(f"  Reached {max_rows} rows, stopping.")
            break

        if total_read % 500_000 == 0:
            logger.info(f"  Read {total_read:,} rows, kept {sum(len(c) for c in chunks):,}...")

    if not chunks:
        raise ValueError("No matching interactions found in GloBI data")

    df = pd.concat(chunks, ignore_index=True)

    # Deduplicate by species pair + interaction type
    df = df.drop_duplicates(
        subset=["sourceTaxonName", "targetTaxonName", "interactionTypeName"],
        keep="first",
    )

    logger.info(f"  Final: {len(df):,} unique interactions from {total_read:,} total rows")
    return df


def build_sibils_queries(globi_df: pd.DataFrame, max_per_type: int = 200) -> List[dict]:
    """
    Build SIBiLS search queries from GloBI interaction data.

    Samples species pairs per interaction type and generates search queries.
    """
    queries = []

    # Group by interaction type
    for int_type, group in globi_df.groupby("interactionTypeName"):
        category = GLOBI_TYPE_MAP.get(int_type, "parasitism")
        templates = INTERACTION_QUERIES.get(category, INTERACTION_QUERIES["parasitism"])

        # Sample species pairs
        sample_size = min(len(group), max_per_type)
        sampled = group.sample(n=sample_size, random_state=42)

        for _, row in sampled.iterrows():
            source = str(row.get("sourceTaxonName", "")).strip()
            target = str(row.get("targetTaxonName", "")).strip()

            if not source or not target or source == "nan" or target == "nan":
                continue

            # Use first template (most natural query)
            template = templates[0]
            query = template.format(source=source, target=target)

            queries.append({
                "query": query,
                "source_species": source,
                "target_species": target,
                "interaction_type": int_type,
                "category": category,
                "doi": row.get("referenceDoi"),
                "pmid": row.get("pmid"),
            })

    logger.info(f"Built {len(queries)} SIBiLS queries from {globi_df['interactionTypeName'].nunique()} interaction types")
    return queries


def fetch_and_extract(
    queries: List[dict],
    cache: SibilsCache,
    max_positives: int = TARGET_POSITIVES,
) -> Tuple[List[dict], List[dict]]:
    """
    Fetch articles from SIBiLS and extract sentences.

    Returns (positives, negatives) as lists of dicts.
    """
    all_positives = []
    all_negatives = []
    seen_sentences = set()
    articles_processed = 0
    type_counts = {}

    total_queries = len(queries)
    logger.info(f"Processing {total_queries} queries...")

    for i, q in enumerate(queries):
        if len(all_positives) >= max_positives:
            logger.info(f"Reached target of {max_positives} positives, stopping.")
            break

        category = q["category"]

        # Skip if this type already has enough
        if type_counts.get(category, 0) >= MAX_PER_INTERACTION_TYPE:
            continue

        # Search SIBiLS (PMC first for full text, then medline)
        articles = []
        for collection in SIBILS_COLLECTIONS:
            hits = search_sibils(
                q["query"],
                collection=collection,
                n=MAX_ARTICLES_PER_QUERY,
                cache=cache,
            )
            articles.extend(hits)
            time.sleep(RATE_LIMIT_DELAY)

        # Process articles
        for article in articles:
            if len(all_positives) >= max_positives:
                break

            try:
                # Extract positives
                positives = extract_positives_from_article(
                    article,
                    source_species=q["source_species"],
                    target_species=q["target_species"],
                    interaction_type=q["interaction_type"],
                )

                for p in positives:
                    sent_hash = hashlib.md5(p["text"].encode()).hexdigest()
                    if sent_hash not in seen_sentences:
                        seen_sentences.add(sent_hash)
                        all_positives.append(p)
                        type_counts[category] = type_counts.get(category, 0) + 1

                # Extract negatives (from same articles for realistic distribution)
                negatives = extract_negatives_from_article(article, max_negatives=2)
                for n in negatives:
                    sent_hash = hashlib.md5(n["text"].encode()).hexdigest()
                    if sent_hash not in seen_sentences:
                        seen_sentences.add(sent_hash)
                        all_negatives.append(n)

                articles_processed += 1

            except Exception as e:
                logger.warning(f"Error processing article {article.doc_id}: {e}")
                continue

        # Progress
        if (i + 1) % 50 == 0:
            logger.info(
                f"  [{i+1}/{total_queries}] positives={len(all_positives)}, "
                f"negatives={len(all_negatives)}, articles={articles_processed}"
            )

    logger.info(f"\nExtraction complete:")
    logger.info(f"  Positives: {len(all_positives)}")
    logger.info(f"  Negatives: {len(all_negatives)}")
    logger.info(f"  Articles processed: {articles_processed}")
    logger.info(f"  By category: {json.dumps(type_counts, indent=2)}")

    return all_positives, all_negatives


def merge_with_v8(
    new_positives: List[dict],
    new_negatives: List[dict],
    v8_path: Path,
) -> pd.DataFrame:
    """Merge new SIBiLS sentences with v8 dataset, deduplicating."""

    # Load v8
    logger.info(f"Loading v8 from {v8_path}...")
    v8 = pd.read_csv(v8_path)
    logger.info(f"  v8: {len(v8)} samples ({sum(v8['label']==1)} pos, {sum(v8['label']==0)} neg)")

    # Build new data DataFrame
    new_df = pd.DataFrame(new_positives + new_negatives)

    # Standardize columns to match v8
    v8_cols = ["text", "label", "source_species", "target_species", "interaction_type", "quality_score"]

    # Ensure v8 has all columns
    for col in v8_cols:
        if col not in v8.columns:
            v8[col] = ""

    # Add source tracking
    if "source" not in v8.columns:
        v8["source"] = "v8"

    # Keep only shared columns + source
    keep_cols = v8_cols + ["source"]
    for col in keep_cols:
        if col not in new_df.columns:
            new_df[col] = ""

    v8_keep = v8[keep_cols].copy()
    new_keep = new_df[keep_cols].copy()

    # Deduplicate: remove new sentences that already exist in v8
    v8_texts = set(v8_keep["text"].str.lower().str.strip())
    before = len(new_keep)
    new_keep = new_keep[~new_keep["text"].str.lower().str.strip().isin(v8_texts)]
    logger.info(f"  Removed {before - len(new_keep)} duplicates from new data")
    logger.info(f"  New unique samples: {len(new_keep)} ({sum(new_keep['label']==1)} pos, {sum(new_keep['label']==0)} neg)")

    # Merge
    combined = pd.concat([v8_keep, new_keep], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Build v9 dataset with SIBiLS PMC full-text extraction"
    )
    parser.add_argument(
        "--max-positives", type=int, default=TARGET_POSITIVES,
        help=f"Target number of new positive sentences (default: {TARGET_POSITIVES})"
    )
    parser.add_argument(
        "--max-queries", type=int, default=2000,
        help="Maximum number of SIBiLS queries to run"
    )
    parser.add_argument(
        "--v8-path", type=str, default=str(V8_FILE),
        help="Path to v8 training data"
    )
    parser.add_argument(
        "--output", type=str, default=str(OUTPUT_FILE),
        help="Output path for v9 dataset"
    )
    parser.add_argument(
        "--globi-dir", type=str,
        default=str(CLASSIFIER_ROOT / "data/globi"),
        help="Directory with GloBI interactions.tsv.gz"
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Don't download GloBI data if missing"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show plan without executing"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("BUILDING V9 DATASET: SIBiLS PMC Full-Text Extraction")
    print("=" * 70)

    # Step 1: Load GloBI interactions (chunked to avoid OOM)
    print("\n1. Loading GloBI interactions (memory-efficient chunked loading)...")
    globi_path = Path(args.globi_dir) / "interactions.tsv.gz"
    if not globi_path.exists():
        print(f"   ERROR: GloBI data not found at {globi_path}")
        print(f"   Download from: {GLOBI_ZENODO_URL}")
        sys.exit(1)

    target_types = [
        "parasiteOf", "hasHost", "preysOn", "preyedUponBy",
        "eats", "pollinates", "visitsFlowersOf", "pathogenOf",
        "vectorOf", "symbioticWith", "interactsWith",
        "endoparasiteOf", "ectoParasiteOf", "kills",
    ]
    globi_df = load_globi_chunked(
        str(globi_path),
        interaction_types=target_types,
        max_rows=300_000,
    )
    print(f"   Loaded {len(globi_df):,} unique interactions")
    print(f"   Types: {globi_df['interactionTypeName'].value_counts().head(8).to_dict()}")
    print(f"   {globi_df['referenceDoi'].notna().sum():,} with DOI, {globi_df['pmid'].notna().sum():,} with PMID")

    # Step 2: Build search queries
    print("\n2. Building SIBiLS search queries...")
    queries = build_sibils_queries(globi_df, max_per_type=200)
    queries = queries[:args.max_queries]
    print(f"   {len(queries)} queries prepared")

    if args.dry_run:
        print("\n[DRY RUN] Would execute these queries:")
        for cat, count in pd.DataFrame(queries)["category"].value_counts().items():
            print(f"   {cat}: {count} queries")
        print(f"\n   Target: {args.max_positives} new positives")
        print(f"   Output: {args.output}")
        return

    # Step 3: Fetch and extract
    print("\n3. Fetching articles from SIBiLS PMC and extracting sentences...")
    cache = SibilsCache(CACHE_DIR)
    positives, negatives = fetch_and_extract(
        queries, cache, max_positives=args.max_positives
    )

    if not positives:
        print("\nERROR: No positive sentences extracted. Check SIBiLS API connectivity.")
        sys.exit(1)

    # Step 4: Balance negatives
    print("\n4. Balancing negatives...")
    target_neg = int(len(positives) * TARGET_NEGATIVES_RATIO)
    if len(negatives) > target_neg:
        np.random.seed(42)
        indices = np.random.choice(len(negatives), target_neg, replace=False)
        negatives = [negatives[i] for i in indices]
    print(f"   Positives: {len(positives)}, Negatives: {len(negatives)}")

    # Step 5: Merge with v8
    print(f"\n5. Merging with v8 ({args.v8_path})...")
    combined = merge_with_v8(positives, negatives, Path(args.v8_path))

    # Step 6: Save
    print(f"\n6. Saving v9 dataset to {args.output}...")
    combined.to_csv(args.output, index=False)

    # Save metadata
    metadata = {
        "version": "v9",
        "build_date": pd.Timestamp.now().isoformat(),
        "total_samples": len(combined),
        "positives": int(sum(combined["label"] == 1)),
        "negatives": int(sum(combined["label"] == 0)),
        "new_sibils_positives": len(positives),
        "new_sibils_negatives": len(negatives),
        "v8_base": str(args.v8_path),
        "sibils_collections": SIBILS_COLLECTIONS,
        "globi_queries": len(queries),
        "source_distribution": combined["source"].value_counts().to_dict(),
    }
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Summary
    print("\n" + "=" * 70)
    print("V9 DATASET SUMMARY")
    print("=" * 70)
    total_pos = int(sum(combined["label"] == 1))
    total_neg = int(sum(combined["label"] == 0))
    print(f"\nTotal: {len(combined)}")
    print(f"  Positives: {total_pos} ({100*total_pos/len(combined):.1f}%)")
    print(f"  Negatives: {total_neg} ({100*total_neg/len(combined):.1f}%)")
    print(f"\nBy source:")
    for src, cnt in combined["source"].value_counts().items():
        print(f"  {src}: {cnt}")
    print(f"\nBy interaction type (positives):")
    pos_df = combined[combined["label"] == 1]
    if "interaction_type" in pos_df.columns:
        for t, c in pos_df["interaction_type"].value_counts().head(15).items():
            print(f"  {t}: {c}")
    print(f"\nSaved to: {args.output}")
    print(f"Metadata: {METADATA_FILE}")


if __name__ == "__main__":
    main()
