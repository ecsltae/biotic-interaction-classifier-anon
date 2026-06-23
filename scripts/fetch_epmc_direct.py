#!/usr/bin/env python3
"""
Europe PMC Direct Search Harvester
===================================

Searches Europe PMC directly for species pairs known to interact (from GloBI),
downloads open-access full texts, extracts co-occurrence sentences.

Unlike fetch_globi_pmc.py (which only follows GloBI citation links), this script:
1. Extracts unique species pairs per category from GloBI (not just cited papers)
2. Searches Europe PMC API directly: e.g. '"Canis lupus" "Rangifer tarandus" predation'
3. Filters for open-access PMC articles (full text available)
4. Fetches full text via PMC XML API
5. Extracts sentences where both species co-occur (positive) or appear with
   interaction signals (hard negative)

This bypasses the citation bottleneck for ecological categories where GloBI-cited
papers are often paywalled. All positives remain GloBI-grounded since we only
search for verified interaction pairs.

Usage:
    python scripts/fetch_epmc_direct.py --max-positives 2000
    python scripts/fetch_epmc_direct.py --max-positives 5000 --categories predation herbivory
    python scripts/fetch_epmc_direct.py --dry-run --max-pairs 20
"""

import os
import sys
import re
import json
import time
import hashlib
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, field

import pandas as pd
import requests

# Add project root and scripts dir to path
CLASSIFIER_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(CLASSIFIER_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

from data.sentence_extractor import split_sentences, generate_name_variants, find_match_in_sentence

# Import shared components from fetch_globi_pmc
from fetch_globi_pmc import (
    PMCFetcher,
    SpeciesNameResolver,
    GlobiRecord,
    ExtractedSentence,
    is_good_sentence,
    extract_sentences_from_article,
    COMMON_NAMES_TABLE,
    AMBIGUOUS_COMMON_NAMES,
    BIOTIC_INTERACTION_SIGNALS,
    MAX_PER_CATEGORY,
    CATEGORY_PRIORITY,
    MAX_SENTS_PER_ARTICLE,
    MAX_NEG_PER_ARTICLE,
    MIN_SENT_LEN,
    MAX_SENT_LEN,
    INTERACTION_CATEGORY_MAP,
    TARGET_INTERACTION_TYPES,
    RATE_LIMIT_DELAY,
    GLOBI_PATH,
    CACHE_DIR_DEFAULT,
    NAME_CACHE_DEFAULT,
    _filter_ambiguous,
    _pluralize,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DEFAULT = CLASSIFIER_ROOT / "data" / "training" / "epmc_direct_sentences.csv"
EUROPE_PMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

# Search keywords per category (added to species name query)
CATEGORY_SEARCH_TERMS: Dict[str, List[str]] = {
    "predation":  ["predation", "predator prey", "prey"],
    "herbivory":  ["herbivory", "grazing", "herbivore plant"],
    "parasitism": ["parasite host", "parasitism", "ectoparasite", "endoparasite", "host of", "internal parasite"],
    "pathogen":   ["pathogen", "infects", "infection", "disease agent", "causative agent"],
    "pollination": ["pollination", "pollinator"],
    "symbiosis":  ["symbiosis", "symbiont"],
    "dispersal":  ["seed dispersal", "dispersal"],
    "vector":     ["vector host", "disease vector"],
    "general":    ["interaction", "biotic interaction"],
}

# Max species pairs to query per category
MAX_PAIRS_PER_CATEGORY_DEFAULT = 200

# Max PMC results per query (paginated; increase for longer runs)
RESULTS_PER_QUERY = 200

# Max unique PMCIDs to process per category
MAX_PMCIDS_PER_CATEGORY = 2000


# =============================================================================
# GLOBI SPECIES PAIR LOADING
# =============================================================================

def load_top_species_pairs(
    globi_path: Path,
    categories: List[str],
    max_pairs: int = MAX_PAIRS_PER_CATEGORY_DEFAULT,
    max_rows: Optional[int] = 5_000_000,
) -> Dict[str, List[Tuple[str, str]]]:
    """Load top species pairs from GloBI for given categories.

    Scores pairs by frequency (more GloBI records = better-documented interaction).

    Args:
        globi_path: Path to interactions.tsv.gz
        categories: Target categories (e.g. ["predation", "herbivory"])
        max_pairs: Max pairs to return per category
        max_rows: Max rows to scan from GloBI (None = full file, takes ~10 min)

    Returns:
        Dict mapping category → list of (source, target) species pairs
    """
    logger.info(f"Loading GloBI species pairs for categories: {categories}")
    if max_rows:
        logger.info(f"  (scanning first {max_rows:,} rows — use --full-globi-scan for complete)")

    target_types = {t for t, c in INTERACTION_CATEGORY_MAP.items() if c in categories}
    pair_counts: Dict[str, Counter] = {cat: Counter() for cat in categories}

    use_cols = ["sourceTaxonName", "targetTaxonName", "interactionTypeName"]

    rows_processed = 0
    for chunk in pd.read_csv(
        globi_path,
        sep="\t",
        usecols=use_cols,
        chunksize=500_000,
        dtype=str,
        on_bad_lines="skip",
        low_memory=False,
    ):
        # Vectorized filtering
        chunk = chunk.dropna(subset=["sourceTaxonName", "targetTaxonName", "interactionTypeName"])
        mask = chunk["interactionTypeName"].isin(target_types)
        filtered = chunk[mask].copy()

        if len(filtered) == 0:
            rows_processed += len(chunk)
            if max_rows and rows_processed >= max_rows:
                break
            continue

        # Filter to valid binomial names (must have space = at least 2 words)
        filtered = filtered[
            filtered["sourceTaxonName"].str.contains(" ", na=False) &
            filtered["targetTaxonName"].str.contains(" ", na=False)
        ]

        # Remove self-interactions
        filtered = filtered[
            filtered["sourceTaxonName"].str.lower() != filtered["targetTaxonName"].str.lower()
        ]

        # Map to categories and count pairs
        for itype, grp in filtered.groupby("interactionTypeName"):
            cat = INTERACTION_CATEGORY_MAP.get(str(itype), "general")
            if cat not in pair_counts:
                continue
            for (src, tgt), cnt in grp.groupby(["sourceTaxonName", "targetTaxonName"]).size().items():
                pair_counts[cat][(src, tgt)] += cnt

        rows_processed += len(chunk)
        if rows_processed % 1_000_000 == 0:
            logger.info(f"  Scanned {rows_processed:,} rows...")

        if max_rows and rows_processed >= max_rows:
            logger.info(f"  Reached max_rows limit ({max_rows:,}), stopping scan.")
            break

    result: Dict[str, List[Tuple[str, str]]] = {}
    for cat in categories:
        top = pair_counts[cat].most_common(max_pairs)
        result[cat] = [(s, t) for (s, t), _ in top]
        logger.info(f"  {cat}: {len(result[cat])} unique pairs (from {sum(pair_counts[cat].values()):,} records)")

    return result


# =============================================================================
# EUROPE PMC SEARCH
# =============================================================================

class EuPMCSearcher:
    """Search Europe PMC for open-access articles about species interactions."""

    def __init__(self):
        self.last_request_time = 0.0
        self.stats = {"queries": 0, "pmcids_found": 0, "api_errors": 0}

    def _rate_limit(self) -> None:
        elapsed = time.time() - self.last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self.last_request_time = time.time()

    def search_pair(
        self,
        source: str,
        target: str,
        keyword: str,
        max_results: int = RESULTS_PER_QUERY,
    ) -> List[str]:
        """Search for open-access PMC articles about a species interaction pair.

        Args:
            source: Source species scientific name (e.g. "Canis lupus")
            target: Target species scientific name (e.g. "Rangifer tarandus")
            keyword: Interaction keyword (e.g. "predation")
            max_results: Max PMCIDs to return (paginates if > 100)

        Returns:
            List of PMCID strings (e.g. ["PMC1234567", ...])
        """
        query = f'"{source}" "{target}" {keyword}'
        pmcids = []
        cursor = "*"  # Europe PMC cursor-based pagination

        remaining = max_results
        while remaining > 0:
            params = {
                "query": query,
                "format": "json",
                "pageSize": min(remaining, 100),
                "isOpenAccess": "Y",
                "resultType": "lite",
                "cursorMark": cursor,
            }

            self._rate_limit()
            try:
                resp = requests.get(EUROPE_PMC_SEARCH, params=params, timeout=30)
                self.stats["queries"] += 1
                if resp.status_code != 200:
                    self.stats["api_errors"] += 1
                    break

                data = resp.json()
                page = data.get("resultList", {}).get("result", [])
                for r in page:
                    pmcid = r.get("pmcid", "")
                    if pmcid and pmcid.startswith("PMC"):
                        pmcids.append(pmcid)

                remaining -= len(page)

                # Get next cursor for pagination
                next_cursor = data.get("nextCursorMark", "")
                if not next_cursor or next_cursor == cursor or len(page) == 0:
                    break  # No more results
                cursor = next_cursor

            except Exception as e:
                logger.debug(f"Search failed for '{query}': {e}")
                self.stats["api_errors"] += 1
                break

        self.stats["pmcids_found"] += len(pmcids)
        return pmcids

    def search_pair_with_common_names(
        self,
        source: str,
        target: str,
        source_names: List[str],
        target_names: List[str],
        keyword: str,
        max_results: int = RESULTS_PER_QUERY,
    ) -> List[str]:
        """Also search using common names if available.

        Runs scientific name search + common name search if common names exist.
        """
        pmcids = set(self.search_pair(source, target, keyword, max_results))

        # Try with common names if available and not ambiguous
        s_common = [n for n in source_names
                    if n.lower() not in AMBIGUOUS_COMMON_NAMES and len(n) > 3]
        t_common = [n for n in target_names
                    if n.lower() not in AMBIGUOUS_COMMON_NAMES and len(n) > 3]

        if s_common and t_common:
            # Use most specific common name (longest)
            s_best = max(s_common, key=len)
            t_best = max(t_common, key=len)
            more = self.search_pair(s_best, t_best, keyword, max_results)
            pmcids.update(more)

        return list(pmcids)


# =============================================================================
# HELPERS
# =============================================================================

def _save_results(
    all_positives: list,
    all_negatives: list,
    output_path: "Path",
    args: argparse.Namespace,
    checkpoint: bool = False,
) -> None:
    """Save extracted sentences to CSV, appending to existing file."""
    import pandas as pd
    rows = []
    for sent in all_positives + all_negatives:
        rows.append({
            "text": sent.text,
            "label": sent.label,
            "source_species": sent.source_species,
            "target_species": sent.target_species,
            "interaction_type": sent.interaction_type,
            "category": sent.category,
            "pmcid": sent.pmcid,
            "source": f"epmc_direct_{sent.match_type}",
            "match_type": sent.match_type,
        })
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        existing = pd.read_csv(output_path)
        combined = pd.concat([existing, df], ignore_index=True).drop_duplicates(subset=["text"])
        combined.to_csv(output_path, index=False)
        label = "Checkpoint" if checkpoint else "Appended to"
        logger.info(f"{label} {output_path}: now {len(combined)} total rows")
    else:
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} sentences to {output_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run(args: argparse.Namespace) -> None:
    """Run the Europe PMC direct search harvest pipeline."""
    logger.info("=" * 70)
    logger.info("Europe PMC Direct Search Harvester")
    logger.info("=" * 70)

    categories = args.categories or list(CATEGORY_SEARCH_TERMS.keys())
    logger.info(f"Target categories: {categories}")

    output_path = Path(args.output) if args.output else OUTPUT_DEFAULT
    cache_dir = Path(args.cache_dir) if args.cache_dir else CACHE_DIR_DEFAULT
    name_cache = Path(args.name_cache) if args.name_cache else NAME_CACHE_DEFAULT

    # Step 1: Load species pairs from GloBI
    max_pairs = 10 if args.dry_run else args.max_pairs
    globi_max_rows = None if getattr(args, "full_globi_scan", False) else 2_000_000
    species_pairs = load_top_species_pairs(
        GLOBI_PATH, categories, max_pairs=max_pairs, max_rows=globi_max_rows
    )

    # Step 2: Set up name resolver and fetcher
    name_resolver = SpeciesNameResolver(
        cache_path=name_cache,
        ncbi_api_key=args.ncbi_api_key,
    )

    all_species = set()
    for pairs in species_pairs.values():
        for src, tgt in pairs:
            all_species.add(src)
            all_species.add(tgt)

    if not args.no_common_names:
        name_resolver.batch_resolve(list(all_species))

    fetcher = PMCFetcher(cache_dir=cache_dir, api_key=args.ncbi_api_key)
    searcher = EuPMCSearcher()

    # Step 3: Search Europe PMC for each pair and collect PMCIDs
    logger.info("Searching Europe PMC for species pairs...")

    # Map PMCID → list of (source, target, category, interaction_type) records
    # Use a synthetic "interactsWith" interaction_type since we found via search
    pmcid_to_records: Dict[str, List[GlobiRecord]] = defaultdict(list)

    for cat in categories:
        pairs = species_pairs.get(cat, [])
        keywords = CATEGORY_SEARCH_TERMS.get(cat, ["interaction"])
        logger.info(f"  Category '{cat}': {len(pairs)} pairs × {len(keywords)} keywords")

        for source, target in pairs:
            if args.dry_run and len(pmcid_to_records) >= 20:
                break

            # Get common names for this pair
            src_common = name_resolver.get_common_names(source)
            tgt_common = name_resolver.get_common_names(target)

            # Build a synthetic GlobiRecord for this pair
            # Use the first interaction type that maps to this category
            itype_for_cat = next(
                (t for t, c in INTERACTION_CATEGORY_MAP.items() if c == cat),
                "interactsWith"
            )

            for keyword in keywords:
                pmcids = searcher.search_pair_with_common_names(
                    source, target, src_common, tgt_common, keyword
                )

                for pmcid in pmcids:
                    # Check if we haven't already queued too many for this category
                    cat_pmcids = {r.pmcid for records in pmcid_to_records.values()
                                  for r in records if r.category == cat}
                    if len(cat_pmcids) >= MAX_PMCIDS_PER_CATEGORY:
                        break

                    # Don't re-add the same pair to the same article
                    existing_pairs = {(r.source_taxon, r.target_taxon)
                                      for r in pmcid_to_records[pmcid]}
                    if (source, target) in existing_pairs:
                        continue

                    pmcid_to_records[pmcid].append(GlobiRecord(
                        source_taxon=source,
                        target_taxon=target,
                        interaction_type=itype_for_cat,
                        category=cat,
                        doi=None,
                        pmcid=pmcid,
                    ))

        logger.info(f"    Total unique PMCIDs so far: {len(pmcid_to_records)}")

    logger.info(f"Total unique PMCIDs to process: {len(pmcid_to_records)}")
    logger.info(f"Searcher stats: {searcher.stats}")

    if not pmcid_to_records:
        logger.error("No PMCIDs found! Check network access and species pairs.")
        return

    # Step 4 & 5: Fetch articles and extract sentences
    all_positives: List[ExtractedSentence] = []
    all_negatives: List[ExtractedSentence] = []
    category_pos_counts: Dict[str, int] = defaultdict(int)
    seen_texts: Set[str] = set()
    articles_processed = 0
    articles_with_hits = 0
    total_articles = len(pmcid_to_records)

    # Sort by category priority: ecological first, pathogen last
    def _pmcid_priority(item: Tuple[str, list]) -> int:
        cats = {r.category for r in item[1]}
        best = min(
            (CATEGORY_PRIORITY.index(c) if c in CATEGORY_PRIORITY else len(CATEGORY_PRIORITY)
             for c in cats),
            default=len(CATEGORY_PRIORITY),
        )
        return best

    sorted_pmcids = sorted(pmcid_to_records.items(), key=_pmcid_priority)
    logger.info("Article queue sorted by ecological priority.")

    for pmcid, article_records in sorted_pmcids:
        # Check if we have enough positives overall
        total_pos = sum(category_pos_counts.values())
        if total_pos >= args.max_positives:
            logger.info(f"Reached target of {args.max_positives} positives, stopping.")
            break

        # Check per-category caps
        article_records = [
            r for r in article_records
            if category_pos_counts[r.category] < MAX_PER_CATEGORY.get(r.category, 150)
        ]
        if not article_records:
            continue

        articles_processed += 1
        if articles_processed % 50 == 0:
            cat_summary = ", ".join(f"{c}={n}" for c, n in sorted(category_pos_counts.items()) if n > 0)
            logger.info(
                f"  Article {articles_processed}/{total_articles}: "
                f"{sum(category_pos_counts.values())} positives [{cat_summary}]"
            )

        # Fetch full text
        article_text = fetcher.fetch_fulltext(pmcid)
        if not article_text:
            continue

        # Extract sentences for all interaction pairs in this article
        article_positives = []
        article_negatives = []
        pairs_seen = set()  # avoid extracting same pair twice per article

        for record in article_records:
            pair_key = (record.source_taxon.lower(), record.target_taxon.lower())
            if pair_key in pairs_seen:
                continue
            pairs_seen.add(pair_key)

            # Skip if category is already capped
            if category_pos_counts[record.category] >= MAX_PER_CATEGORY.get(record.category, 150):
                continue

            # Resolve name variants
            src_variants = name_resolver.get_all_name_forms(record.source_taxon)
            tgt_variants = name_resolver.get_all_name_forms(record.target_taxon)

            pos, neg = extract_sentences_from_article(
                article_text, record, src_variants, tgt_variants
            )

            # Deduplicate by text hash
            for sent in pos:
                h = hashlib.md5(sent.text.encode()).hexdigest()
                if h not in seen_texts:
                    seen_texts.add(h)
                    article_positives.append(sent)

            for sent in neg:
                h = hashlib.md5(sent.text.encode()).hexdigest()
                if h not in seen_texts:
                    seen_texts.add(h)
                    article_negatives.append(sent)

        # Apply per-article caps
        article_positives = article_positives[:MAX_SENTS_PER_ARTICLE]
        article_negatives = article_negatives[:MAX_NEG_PER_ARTICLE]

        if article_positives:
            articles_with_hits += 1

        for sent in article_positives:
            category_pos_counts[sent.category] += 1

        all_positives.extend(article_positives)
        all_negatives.extend(article_negatives)

        # Incremental checkpoint every 200 articles (survive crashes)
        if not args.dry_run and articles_processed % 200 == 0 and all_positives:
            _save_results(all_positives, all_negatives, output_path, args, checkpoint=True)

    # Step 6: Balance negatives to match positives
    total_pos = len(all_positives)
    all_negatives = all_negatives[:total_pos]  # balanced

    # Step 7: Save results
    logger.info("")
    logger.info("=" * 70)
    logger.info("HARVEST SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Articles processed: {articles_processed}")
    logger.info(f"Articles with hits: {articles_with_hits}")
    logger.info(f"Positives: {len(all_positives)}")
    logger.info(f"Negatives: {len(all_negatives)}")
    logger.info(f"Total sentences: {len(all_positives) + len(all_negatives)}")

    logger.info("\nCategory distribution (positives):")
    for cat, count in sorted(category_pos_counts.items(), key=lambda x: -x[1]):
        cap = MAX_PER_CATEGORY.get(cat, 150)
        logger.info(f"  {cat}: {count} / {cap}")

    if not all_positives and not all_negatives:
        logger.warning("No sentences extracted!")
        return

    if args.dry_run:
        logger.info(f"[DRY RUN] Would save {len(all_positives) + len(all_negatives)} sentences to {output_path}")
        for sent in all_positives[:5]:
            logger.info(f"  [{sent.category}] {sent.text[:120]}")
    else:
        _save_results(all_positives, all_negatives, output_path, args)

    logger.info(f"Fetcher stats: {fetcher.stats}")
    logger.info(f"Searcher stats: {searcher.stats}")
    logger.info(f"Name resolver stats: {name_resolver.stats}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--max-positives", type=int, default=2000,
        help="Target number of positive sentences to harvest (default: 2000)"
    )
    p.add_argument(
        "--max-pairs", type=int, default=MAX_PAIRS_PER_CATEGORY_DEFAULT,
        help=f"Max species pairs to query per category (default: {MAX_PAIRS_PER_CATEGORY_DEFAULT})"
    )
    p.add_argument(
        "--categories", nargs="+", choices=list(CATEGORY_SEARCH_TERMS.keys()),
        default=None,
        help="Categories to harvest (default: all)"
    )
    p.add_argument(
        "--output", type=str, default=None,
        help=f"Output CSV path (default: {OUTPUT_DEFAULT})"
    )
    p.add_argument(
        "--cache-dir", type=str, default=None,
        help=f"PMC article cache directory (default: {CACHE_DIR_DEFAULT})"
    )
    p.add_argument(
        "--name-cache", type=str, default=None,
        help=f"Species name cache file (default: {NAME_CACHE_DEFAULT})"
    )
    p.add_argument(
        "--ncbi-api-key", type=str, default=None,
        help="NCBI API key for higher rate limits"
    )
    p.add_argument(
        "--no-common-names", action="store_true",
        help="Skip common name resolution (faster, lower recall)"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Process only 20 articles, do not write output"
    )
    p.add_argument(
        "--full-globi-scan", action="store_true",
        help="Scan entire GloBI file for species pairs (slow, ~10 min, but finds more pairs)"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
