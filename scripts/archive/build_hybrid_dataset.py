#!/usr/bin/env python3
"""
Build Hybrid Training Dataset v3

Combines:
1. REAL positive sentences extracted from scientific articles (via GloBI references)
2. Template-generated positives (as backup/fill)
3. Hard negatives (two-species, no interaction) - critical for precision
4. Easy negatives (single-species)

Target: F1 > 0.75, Precision > F1
"""

import os
import sys
import random
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Set
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.globi_loader import load_and_prepare_globi, get_interaction_stats
from data.article_fetcher import ArticleFetcher, get_article_text
from data.sentence_extractor import (
    extract_best_sentence, SentenceMatch,
    extract_best_interaction_sentences, RelaxedSentenceMatch
)
from data.kingdom_mapper import get_kingdom
from data.quality_filter import load_domain_rules, validate_domain_rules
from data.template_generator import (
    generate_from_globi,
    generate_hard_negatives,
    generate_negatives_from_species,
    GeneratedSentence
)

# Try to import MCP validation (optional)
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from validator import validate_interaction_sentence, is_llm_available
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Configuration
DEFAULT_DATA_DIR = "./data"
DEFAULT_OUTPUT_DIR = "./data/training"
DEFAULT_CACHE_DIR = "./data/article_cache"

# Dataset composition targets
TARGET_POSITIVES = 15000
TARGET_NEGATIVES = 30000
HARD_NEGATIVE_RATIO = 0.8  # 80% hard negatives (two-species, no interaction)
MIN_REAL_POSITIVES = 2000  # Minimum real sentences to be useful

# Interaction types to focus on (most common in GloBI)
PRIORITY_INTERACTIONS = [
    "parasiteOf", "hasHost", "preyedUponBy", "preysOn",
    "pollinates", "eats", "visitsFlowersOf", "pathogenOf",
    "vectorOf", "symbioticWith", "interactsWith"
]


def extract_real_positives(
    globi_df: pd.DataFrame,
    fetcher: ArticleFetcher,
    max_articles: int = 5000,
    max_sentences: int = 10000,
    use_llm_validation: bool = False,
    domain_rules: dict = None
) -> List[Tuple[str, str]]:
    """
    Extract real positive sentences from articles referenced in GloBI.

    Uses RELAXED matching: finds any sentences with 2+ taxa + interaction terms,
    then validates with domain rules and optionally LLM.

    Args:
        globi_df: GloBI interactions with DOI/PMID references
        fetcher: ArticleFetcher instance
        max_articles: Maximum articles to fetch
        max_sentences: Maximum sentences to extract
        use_llm_validation: Whether to use LLM for validation
        domain_rules: Domain rules for validation

    Returns:
        List of (sentence, interaction_type) tuples
    """
    real_positives = []
    articles_fetched = 0
    articles_with_sentences = 0
    fetch_failures = 0
    validation_rejected = 0
    seen_sentences = set()  # Deduplicate

    # Get unique articles from GloBI (many interactions per article)
    unique_articles = globi_df.drop_duplicates(subset=['referenceDoi'])[['referenceDoi']].dropna()
    article_dois = unique_articles['referenceDoi'].tolist()
    random.shuffle(article_dois)

    print(f"\nExtracting real positives (RELAXED matching) from up to {max_articles} articles...")
    print(f"Using LLM validation: {use_llm_validation and MCP_AVAILABLE and is_llm_available()}")

    for doi in tqdm(article_dois[:max_articles], desc="Processing articles"):
        if len(real_positives) >= max_sentences:
            break

        # Fetch the article
        article = fetcher.fetch_article(doi=doi)
        articles_fetched += 1

        if not article:
            fetch_failures += 1
            continue

        # Get article text (full text or abstract)
        text = get_article_text(article)
        if not text or len(text) < 100:
            continue

        # RELAXED extraction: find sentences with 2+ taxa + interaction terms
        matches = extract_best_interaction_sentences(
            text,
            max_sentences=5,
            doi=doi
        )

        if not matches:
            continue

        article_sentences = 0
        for match in matches:
            # Deduplicate
            sent_hash = hash(match.sentence[:100])
            if sent_hash in seen_sentences:
                continue
            seen_sentences.add(sent_hash)

            # Domain rule validation (if rules provided)
            if domain_rules and len(match.taxa_found) >= 2:
                sp1 = match.taxa_found[0]
                sp2 = match.taxa_found[1]
                sp1_kingdom = get_kingdom(sp1)
                sp2_kingdom = get_kingdom(sp2)
                interaction_term = match.interactions_found[0] if match.interactions_found else ''

                is_valid, violations = validate_domain_rules(
                    interaction=interaction_term,
                    species1=sp1,
                    species1_kingdom=sp1_kingdom or '',
                    species2=sp2,
                    species2_kingdom=sp2_kingdom or '',
                    match_length=len(sp1) + len(sp2) + len(interaction_term),
                    rules=domain_rules
                )

                if not is_valid:
                    validation_rejected += 1
                    continue

            # Optional LLM validation
            if use_llm_validation and MCP_AVAILABLE and is_llm_available():
                result = validate_interaction_sentence(match.sentence, use_llm=True)
                if not result.is_interaction or result.confidence < 0.6:
                    validation_rejected += 1
                    continue

            # Determine interaction type from the matched terms
            interaction_type = match.interactions_found[0] if match.interactions_found else 'interacts'

            real_positives.append((match.sentence, interaction_type))
            article_sentences += 1

            if len(real_positives) >= max_sentences:
                break

        if article_sentences > 0:
            articles_with_sentences += 1

    print(f"\nReal positive extraction complete:")
    print(f"  Articles fetched: {articles_fetched}")
    print(f"  Fetch failures: {fetch_failures}")
    print(f"  Articles with interaction sentences: {articles_with_sentences}")
    print(f"  Validation rejected: {validation_rejected}")
    print(f"  Total real positives: {len(real_positives)}")

    return real_positives


def get_species_list(globi_df: pd.DataFrame) -> List[str]:
    """Extract unique species names from GloBI data."""
    species = set()
    species.update(globi_df['sourceTaxonName'].dropna().unique())
    species.update(globi_df['targetTaxonName'].dropna().unique())

    # Filter to reasonable species names (binomial-like)
    valid_species = []
    for name in species:
        parts = str(name).split()
        if len(parts) >= 2 and len(parts[0]) > 2:
            valid_species.append(name)

    return valid_species


def build_hybrid_dataset(
    data_dir: str = DEFAULT_DATA_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    cache_dir: str = DEFAULT_CACHE_DIR,
    max_articles: int = 5000,
    target_positives: int = TARGET_POSITIVES,
    target_negatives: int = TARGET_NEGATIVES,
    download_globi: bool = True,
    globi_nrows: int = None,
    use_llm_validation: bool = False
) -> pd.DataFrame:
    """
    Build the hybrid training dataset.

    Args:
        data_dir: Directory for GloBI data
        output_dir: Directory to save output dataset
        cache_dir: Directory for article cache
        max_articles: Maximum articles to fetch
        target_positives: Target number of positive examples
        target_negatives: Target number of negative examples
        download_globi: Whether to download GloBI data if missing
        globi_nrows: Limit GloBI rows (for testing)

    Returns:
        DataFrame with hybrid training data
    """
    os.makedirs(output_dir, exist_ok=True)

    # =================================================================
    # Step 1: Load GloBI interactions with references
    # =================================================================
    print("=" * 60)
    print("STEP 1: Loading GloBI interactions with references")
    print("=" * 60)

    globi_dir = os.path.join(data_dir, "globi")

    try:
        globi_df = load_and_prepare_globi(
            data_dir=globi_dir,
            download=download_globi,
            nrows=globi_nrows,
            interaction_types=PRIORITY_INTERACTIONS
        )
    except FileNotFoundError as e:
        print(f"Warning: GloBI data not available: {e}")
        print("Will use template-only dataset.")
        globi_df = pd.DataFrame()

    if len(globi_df) > 0:
        stats = get_interaction_stats(globi_df)
        print(f"\nLoaded {stats['total_interactions']:,} interactions with references")
        print(f"Unique source taxa: {stats['unique_source_taxa']:,}")
        print(f"Unique target taxa: {stats['unique_target_taxa']:,}")
        print(f"Interactions with DOI: {stats['has_doi']:,}")
        print(f"Interactions with PMID: {stats['has_pmid']:,}")

    # =================================================================
    # Step 2: Extract real positive sentences from articles
    # =================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Extracting real positive sentences from articles")
    print("=" * 60)

    # Load domain rules for validation
    domain_rules_path = os.path.join(Path(__file__).parent.parent, "data", "domain_rules.json")
    if os.path.exists(domain_rules_path):
        domain_rules = load_domain_rules(domain_rules_path)
        print(f"Loaded {len(domain_rules)} domain rules")
    else:
        domain_rules = None
        print(f"No domain rules found at {domain_rules_path} - skipping rule validation")

    real_positives = []

    if len(globi_df) > 0:
        fetcher = ArticleFetcher(cache_dir=cache_dir)

        real_positives = extract_real_positives(
            globi_df=globi_df,
            fetcher=fetcher,
            max_articles=max_articles,
            max_sentences=target_positives,
            use_llm_validation=use_llm_validation,
            domain_rules=domain_rules
        )

        print(f"\nFetcher stats: {fetcher.get_stats()}")

    # =================================================================
    # Step 3: Generate template positives to fill target
    # =================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Generating template positives (backup/fill)")
    print("=" * 60)

    # Get species list for template generation
    if len(globi_df) > 0:
        species_list = get_species_list(globi_df)
    else:
        # Fallback: load from v1 dataset if available
        v1_path = os.path.join(output_dir, "training_data_globi_v1.csv")
        if os.path.exists(v1_path):
            v1_df = pd.read_csv(v1_path)
            species_list = list(set(
                v1_df[v1_df['label'] == 1]['text'].str.extract(
                    r'([A-Z][a-z]+ [a-z]+)'
                )[0].dropna().unique()
            ))
            print(f"Loaded {len(species_list)} species from v1 dataset")
        else:
            raise ValueError("No species list available - need GloBI data or v1 dataset")

    print(f"Using {len(species_list)} species for template/negative generation")

    # Calculate how many template positives we need
    real_count = len(real_positives)
    template_needed = max(0, target_positives - real_count)

    print(f"Real positives: {real_count}")
    print(f"Template positives needed: {template_needed}")

    template_positives = []
    if template_needed > 0:
        if len(globi_df) > 0:
            # Use GloBI DataFrame directly for template generation
            # Sample a subset if we have many interactions
            sample_size = min(len(globi_df), template_needed * 2)
            sample_df = globi_df.sample(n=sample_size, random_state=42)
            generated = generate_from_globi(sample_df, max_per_interaction=3)
            # Trim to needed amount
            generated = generated[:template_needed]
        else:
            # Fallback: load v1 positives directly
            v1_path = os.path.join(output_dir, "training_data_globi_v1.csv")
            if os.path.exists(v1_path):
                v1_df = pd.read_csv(v1_path)
                v1_positives = v1_df[v1_df['label'] == 1].sample(n=min(template_needed, len(v1_df[v1_df['label']==1])), random_state=42)
                generated = [
                    GeneratedSentence(
                        sentence=row['text'],
                        source_species=row.get('source_species', ''),
                        target_species=row.get('target_species', ''),
                        interaction_type=row.get('interaction_type', 'unknown')
                    )
                    for _, row in v1_positives.iterrows()
                ]
            else:
                generated = []
                print("WARNING: No source for template positives!")

        template_positives = [(g.sentence, g.interaction_type) for g in generated]
        print(f"Generated {len(template_positives)} template positives")

    # Combine positives
    all_positives = real_positives + template_positives
    print(f"\nTotal positives: {len(all_positives)}")
    if len(all_positives) > 0:
        print(f"  - Real: {len(real_positives)} ({100*len(real_positives)/len(all_positives):.1f}%)")
        print(f"  - Template: {len(template_positives)} ({100*len(template_positives)/len(all_positives):.1f}%)")
    else:
        print("WARNING: No positives generated!")

    # =================================================================
    # Step 4: Generate negatives (hard + easy)
    # =================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Generating negatives (hard + easy)")
    print("=" * 60)

    hard_negative_count = int(target_negatives * HARD_NEGATIVE_RATIO)
    easy_negative_count = target_negatives - hard_negative_count

    print(f"Hard negatives (two-species, no interaction): {hard_negative_count}")
    print(f"Easy negatives (single-species): {easy_negative_count}")

    # Generate hard negatives (critical for precision)
    hard_negatives = generate_hard_negatives(
        species_names=species_list,
        count=hard_negative_count
    )
    print(f"Generated {len(hard_negatives)} hard negatives")

    # Generate easy negatives
    easy_negatives = generate_negatives_from_species(
        species_names=species_list,
        count=easy_negative_count
    )
    print(f"Generated {len(easy_negatives)} easy negatives")

    all_negatives = [(g.sentence, None) for g in hard_negatives + easy_negatives]

    # =================================================================
    # Step 5: Build final DataFrame
    # =================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Building final dataset")
    print("=" * 60)

    # Create records
    records = []

    # Add positives
    for sentence, interaction_type in all_positives:
        records.append({
            'text': sentence,
            'label': 1,
            'interaction_type': interaction_type,
            'source': 'real' if (sentence, interaction_type) in real_positives else 'template'
        })

    # Add negatives
    for sentence, _ in all_negatives:
        records.append({
            'text': sentence,
            'label': 0,
            'interaction_type': None,
            'source': 'generated'
        })

    df = pd.DataFrame(records)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Print statistics
    print(f"\nFinal dataset statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Positives: {(df['label'] == 1).sum()}")
    print(f"  Negatives: {(df['label'] == 0).sum()}")
    print(f"  Ratio (neg:pos): {(df['label'] == 0).sum() / (df['label'] == 1).sum():.2f}:1")

    print(f"\nSource distribution:")
    print(df['source'].value_counts().to_string())

    # Save dataset
    output_path = os.path.join(output_dir, "training_data_globi_v3.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved dataset to: {output_path}")

    # Save metadata
    metadata = {
        'total_samples': len(df),
        'positives': int((df['label'] == 1).sum()),
        'negatives': int((df['label'] == 0).sum()),
        'real_positives': len(real_positives),
        'template_positives': len(template_positives),
        'hard_negatives': len(hard_negatives),
        'easy_negatives': len(easy_negatives),
        'species_count': len(species_list),
        'globi_interactions': len(globi_df) if len(globi_df) > 0 else 0
    }

    import json
    metadata_path = os.path.join(output_dir, "training_data_globi_v3_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Build hybrid training dataset (real + template positives)"
    )
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR,
        help="Directory for input data (GloBI)"
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output dataset"
    )
    parser.add_argument(
        "--cache-dir", default=DEFAULT_CACHE_DIR,
        help="Directory for article cache"
    )
    parser.add_argument(
        "--max-articles", type=int, default=5000,
        help="Maximum articles to fetch"
    )
    parser.add_argument(
        "--target-positives", type=int, default=TARGET_POSITIVES,
        help="Target number of positive examples"
    )
    parser.add_argument(
        "--target-negatives", type=int, default=TARGET_NEGATIVES,
        help="Target number of negative examples"
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Don't download GloBI data if missing"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode: use small subset"
    )
    parser.add_argument(
        "--globi-nrows", type=int, default=None,
        help="Limit GloBI rows to load (reduces memory usage)"
    )
    parser.add_argument(
        "--use-llm", action="store_true",
        help="Use LLM validation for extracted sentences (requires ANTHROPIC_API_KEY)"
    )

    args = parser.parse_args()

    # Test mode settings
    globi_nrows = 10000 if args.test else args.globi_nrows
    max_articles = 100 if args.test else args.max_articles
    target_pos = 500 if args.test else args.target_positives
    target_neg = 1000 if args.test else args.target_negatives

    df = build_hybrid_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        max_articles=max_articles,
        target_positives=target_pos,
        target_negatives=target_neg,
        download_globi=not args.no_download,
        globi_nrows=globi_nrows,
        use_llm_validation=args.use_llm
    )

    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"\nNext step: Run train_globi_v3_hybrid.py to train on this dataset")


if __name__ == "__main__":
    main()
