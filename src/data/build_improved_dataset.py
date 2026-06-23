#!/usr/bin/env python3
"""
Build Improved Training Dataset

Main orchestration script that:
1. Loads GloBI interactions with article references
2. Fetches article text from PMC/PubMed
3. Extracts sentences with species + interaction
4. Applies quality filtering and domain rules
5. Optionally collects false positives as negatives
6. Outputs the final training dataset
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict
import logging
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.globi_loader import load_and_prepare_globi, get_interaction_stats
from src.data.article_fetcher import ArticleFetcher, get_article_text
from src.data.sentence_extractor import batch_extract_sentences, extract_best_sentence
from src.data.quality_filter import (
    load_domain_rules, filter_sentences, calculate_quality_score,
    DEFAULT_DOMAIN_RULES
)
from src.data.fp_collector import FalsePositiveCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Orchestrates the improved training dataset creation pipeline."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        cache_dir: str = None,
        domain_rules_path: str = None
    ):
        """
        Initialize the dataset builder.

        Args:
            data_dir: Directory for GloBI and intermediate data
            output_dir: Directory for output files
            cache_dir: Directory for article cache (default: data_dir/article_cache)
            domain_rules_path: Path to domain_rules.json
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / "article_cache"

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load domain rules
        if domain_rules_path and Path(domain_rules_path).exists():
            self.domain_rules = load_domain_rules(domain_rules_path)
            logger.info(f"Loaded domain rules from {domain_rules_path}")
        else:
            self.domain_rules = DEFAULT_DOMAIN_RULES
            logger.info("Using default domain rules")

        # Initialize components
        self.fetcher = ArticleFetcher(cache_dir=str(self.cache_dir))
        self.fp_collector = None

        # Statistics
        self.stats = {
            "globi_interactions": 0,
            "articles_fetched": 0,
            "sentences_extracted": 0,
            "sentences_accepted": 0,
            "sentences_rejected": 0,
            "false_positives_collected": 0,
        }

    def load_globi_data(
        self,
        download: bool = True,
        nrows: Optional[int] = None,
        interaction_types: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Load and prepare GloBI interactions."""
        logger.info("Loading GloBI data...")

        df = load_and_prepare_globi(
            data_dir=str(self.data_dir),
            download=download,
            nrows=nrows,
            interaction_types=interaction_types
        )

        self.stats["globi_interactions"] = len(df)

        # Log statistics
        stats = get_interaction_stats(df)
        logger.info(f"Loaded {stats['total_interactions']:,} interactions")
        logger.info(f"  - {stats['has_doi']:,} with DOI")
        logger.info(f"  - {stats.get('has_pmid', 0):,} with PMID")

        return df

    def fetch_articles(
        self,
        interactions: pd.DataFrame,
        max_articles: int = 5000,
        progress_interval: int = 100
    ) -> Dict:
        """Fetch article texts for interactions."""
        logger.info(f"Fetching articles (max {max_articles})...")

        # Prepare identifiers
        identifiers = interactions[['referenceDoi', 'pmid']].dropna(how='all')
        identifiers = identifiers.head(max_articles).to_dict('records')

        def progress_callback(current, total):
            logger.info(f"Fetched {current}/{total} articles")

        articles = self.fetcher.fetch_batch(
            identifiers=identifiers,
            max_articles=max_articles,
            progress_callback=progress_callback
        )

        self.stats["articles_fetched"] = len(articles)
        logger.info(f"Fetched {len(articles)} articles")
        logger.info(f"Fetcher stats: {self.fetcher.get_stats()}")

        return articles

    def extract_sentences(
        self,
        interactions: pd.DataFrame,
        articles: Dict,
        max_per_interaction: int = 1
    ) -> List[dict]:
        """Extract matching sentences from articles."""
        logger.info("Extracting sentences...")

        # Convert DataFrame to list of dicts
        interaction_list = interactions.to_dict('records')

        # Add kingdom info if available
        for interaction in interaction_list:
            # Ensure pmid is set
            if 'pmid' not in interaction:
                interaction['pmid'] = None

        sentences = batch_extract_sentences(
            articles=articles,
            interactions=interaction_list,
            max_per_interaction=max_per_interaction
        )

        self.stats["sentences_extracted"] = len(sentences)
        logger.info(f"Extracted {len(sentences)} sentences")

        return sentences

    def filter_sentences(
        self,
        sentences: List[dict],
        min_quality_score: float = 50.0
    ) -> tuple:
        """Apply quality filtering and domain rules."""
        logger.info(f"Filtering sentences (min score: {min_quality_score})...")

        accepted, rejected = filter_sentences(
            sentences=sentences,
            min_quality_score=min_quality_score,
            rules=self.domain_rules
        )

        self.stats["sentences_accepted"] = len(accepted)
        self.stats["sentences_rejected"] = len(rejected)

        logger.info(f"Accepted: {len(accepted)}, Rejected: {len(rejected)}")

        # Log rejection reasons
        if rejected:
            reasons = {}
            for r in rejected:
                reason = r.get('rejection_reason', 'unknown')[:50]
                reasons[reason] = reasons.get(reason, 0) + 1
            logger.info(f"Top rejection reasons: {dict(list(reasons.items())[:5])}")

        return accepted, rejected

    def collect_false_positives(
        self,
        classifier_func,
        corpus: List[str],
        confidence_threshold: float = 0.7
    ) -> List[str]:
        """Collect false positives from classifier as negatives."""
        logger.info("Collecting false positives...")

        self.fp_collector = FalsePositiveCollector(
            output_dir=str(self.output_dir / "fp_review"),
            confidence_threshold=confidence_threshold
        )

        # Run classifier on corpus
        # classifier_func should return list of dicts with 'text', 'prediction', 'confidence'
        predictions = classifier_func(corpus)

        # Process predictions
        self.fp_collector.process_predictions(predictions, source="corpus")

        # Auto-filter high-probability FPs
        auto_fps, needs_review = self.fp_collector.auto_filter_candidates()

        # Export for manual review
        review_file = self.fp_collector.export_for_review()
        logger.info(f"Exported {len(needs_review)} candidates for review: {review_file}")

        # Get confirmed FPs (auto-accepted)
        fps = self.fp_collector.get_confirmed_false_positives()
        self.stats["false_positives_collected"] = len(fps)

        logger.info(f"Collected {len(fps)} false positives")
        return fps

    def build_dataset(
        self,
        positives: List[dict],
        negatives: List[str] = None,
        output_name: str = "training_data_v4.csv"
    ) -> str:
        """Build final training dataset."""
        logger.info("Building final dataset...")

        rows = []

        # Add positives
        for p in positives:
            sentence = p.get('sentence', '')
            if sentence:
                rows.append({'passage': sentence, 'label': 1})

        # Add negatives
        if negatives:
            for neg in negatives:
                rows.append({'passage': neg, 'label': 0})

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Deduplicate
        original_len = len(df)
        df = df.drop_duplicates(subset=['passage'])
        if len(df) < original_len:
            logger.info(f"Removed {original_len - len(df)} duplicates")

        # Save
        output_path = self.output_dir / output_name
        df.to_csv(output_path, index=False)

        logger.info(f"Saved dataset with {len(df)} samples to {output_path}")
        logger.info(f"  - Positives: {(df['label'] == 1).sum()}")
        logger.info(f"  - Negatives: {(df['label'] == 0).sum()}")

        return str(output_path)

    def run_full_pipeline(
        self,
        max_interactions: int = 10000,
        max_articles: int = 5000,
        min_quality_score: float = 50.0,
        interaction_types: Optional[List[str]] = None,
        include_existing_negatives: str = None,
        output_name: str = "training_data_v4.csv"
    ) -> str:
        """
        Run the complete pipeline.

        Args:
            max_interactions: Max GloBI interactions to process
            max_articles: Max articles to fetch
            min_quality_score: Minimum quality score for sentences
            interaction_types: Filter to specific interaction types
            include_existing_negatives: Path to existing negatives CSV to include
            output_name: Output filename

        Returns:
            Path to the generated dataset
        """
        start_time = time.time()
        logger.info("Starting improved dataset pipeline...")

        # Step 1: Load GloBI data
        interactions = self.load_globi_data(
            nrows=max_interactions,
            interaction_types=interaction_types
        )

        # Step 2: Fetch articles
        articles = self.fetch_articles(
            interactions=interactions,
            max_articles=max_articles
        )

        # Step 3: Extract sentences
        sentences = self.extract_sentences(
            interactions=interactions,
            articles=articles
        )

        # Step 4: Filter by quality
        accepted, rejected = self.filter_sentences(
            sentences=sentences,
            min_quality_score=min_quality_score
        )

        # Step 5: Collect negatives
        negatives = []

        # Add rejected sentences as potential negatives (low quality positives)
        # but only those rejected for non-rule reasons
        for r in rejected:
            if 'rule_violations' not in r.get('rejection_reason', ''):
                negatives.append(r['sentence'])

        # Include existing negatives if provided
        if include_existing_negatives and Path(include_existing_negatives).exists():
            existing = pd.read_csv(include_existing_negatives)
            if 'passage' in existing.columns:
                negatives.extend(existing['passage'].tolist())
                logger.info(f"Added {len(existing)} existing negatives")

        # Step 6: Build final dataset
        output_path = self.build_dataset(
            positives=accepted,
            negatives=negatives,
            output_name=output_name
        )

        elapsed = time.time() - start_time
        logger.info(f"\n=== Pipeline Complete ===")
        logger.info(f"Time elapsed: {elapsed:.1f} seconds")
        logger.info(f"Statistics: {self.stats}")

        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Build improved training dataset from GloBI"
    )
    parser.add_argument(
        "--data-dir",
        default="./data/globi",
        help="Directory for GloBI and intermediate data"
    )
    parser.add_argument(
        "--output-dir",
        default="./data/training",
        help="Directory for output files"
    )
    parser.add_argument(
        "--domain-rules",
        default="./data/domain_rules.json",
        help="Path to domain rules JSON"
    )
    parser.add_argument(
        "--max-interactions",
        type=int,
        default=10000,
        help="Maximum GloBI interactions to process"
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=5000,
        help="Maximum articles to fetch"
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=50.0,
        help="Minimum quality score (0-100)"
    )
    parser.add_argument(
        "--interaction-types",
        nargs="+",
        help="Filter to specific interaction types"
    )
    parser.add_argument(
        "--include-negatives",
        help="Path to existing negatives CSV to include"
    )
    parser.add_argument(
        "--output-name",
        default="training_data_v4.csv",
        help="Output filename"
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Don't download GloBI data if missing"
    )

    args = parser.parse_args()

    builder = DatasetBuilder(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        domain_rules_path=args.domain_rules
    )

    output_path = builder.run_full_pipeline(
        max_interactions=args.max_interactions,
        max_articles=args.max_articles,
        min_quality_score=args.min_quality,
        interaction_types=args.interaction_types,
        include_existing_negatives=args.include_negatives,
        output_name=args.output_name
    )

    print(f"\nDataset created: {output_path}")


if __name__ == "__main__":
    main()
