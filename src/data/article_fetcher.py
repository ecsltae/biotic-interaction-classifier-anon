"""
Article Text Fetcher

Fetch full text or abstracts from PMC and PubMed APIs.
Implements caching and rate limiting for responsible API usage.
"""

import os
import json
import time
import hashlib
import requests
import xml.etree.ElementTree as ET
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API endpoints
EUROPE_PMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest"
PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

# Rate limiting: NCBI allows 3 requests/second without API key, 10 with key
RATE_LIMIT_DELAY = 0.35  # seconds between requests


@dataclass
class ArticleText:
    """Container for fetched article text."""
    pmid: Optional[str]
    doi: Optional[str]
    title: Optional[str]
    abstract: Optional[str]
    full_text: Optional[str]
    source: str  # 'pmc', 'pubmed', 'europe_pmc'
    fetch_time: float


class ArticleCache:
    """Simple file-based cache for fetched articles."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Generate cache file path from key."""
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}.json"

    def get(self, key: str) -> Optional[ArticleText]:
        """Retrieve article from cache."""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                return ArticleText(**data)
            except (json.JSONDecodeError, TypeError):
                return None
        return None

    def set(self, key: str, article: ArticleText) -> None:
        """Store article in cache."""
        cache_path = self._get_cache_path(key)
        with open(cache_path, 'w') as f:
            json.dump(asdict(article), f)

    def has(self, key: str) -> bool:
        """Check if key exists in cache."""
        return self._get_cache_path(key).exists()


class ArticleFetcher:
    """Fetches article text from PMC and PubMed APIs."""

    def __init__(
        self,
        cache_dir: str = "./data/article_cache",
        api_key: Optional[str] = None,
        email: Optional[str] = None
    ):
        """
        Initialize the fetcher.

        Args:
            cache_dir: Directory for caching fetched articles
            api_key: NCBI API key (optional, increases rate limit)
            email: Email for NCBI API (recommended)
        """
        self.cache = ArticleCache(cache_dir)
        self.api_key = api_key or os.environ.get("NCBI_API_KEY")
        self.email = email or os.environ.get("NCBI_EMAIL", "user@example.com")
        self.last_request_time = 0
        self.stats = {"cache_hits": 0, "api_calls": 0, "failures": 0}

    def _rate_limit(self) -> None:
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self.last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, url: str, params: dict = None) -> Optional[requests.Response]:
        """Make rate-limited HTTP request."""
        self._rate_limit()
        try:
            response = requests.get(url, params=params, timeout=30)
            self.stats["api_calls"] += 1
            if response.status_code == 200:
                return response
            logger.warning(f"API returned status {response.status_code}: {url}")
        except requests.RequestException as e:
            logger.warning(f"Request failed: {e}")
        self.stats["failures"] += 1
        return None

    def fetch_from_europe_pmc(self, pmid: str = None, doi: str = None) -> Optional[ArticleText]:
        """
        Fetch article from Europe PMC API.

        Args:
            pmid: PubMed ID
            doi: Digital Object Identifier

        Returns:
            ArticleText if found, None otherwise
        """
        if pmid:
            query = f"EXT_ID:{pmid} AND SRC:MED"
        elif doi:
            query = f"DOI:{doi}"
        else:
            return None

        # Search for the article
        search_url = f"{EUROPE_PMC_API}/search"
        params = {
            "query": query,
            "format": "json",
            "resultType": "core"
        }

        response = self._make_request(search_url, params)
        if not response:
            return None

        try:
            data = response.json()
            results = data.get("resultList", {}).get("result", [])
            if not results:
                return None

            article = results[0]
            pmid = article.get("pmid")
            pmcid = article.get("pmcid")

            # Try to get full text if PMC article
            full_text = None
            if pmcid:
                full_text = self._fetch_pmc_fulltext(pmcid)

            return ArticleText(
                pmid=pmid,
                doi=article.get("doi"),
                title=article.get("title"),
                abstract=article.get("abstractText"),
                full_text=full_text,
                source="europe_pmc",
                fetch_time=time.time()
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse Europe PMC response: {e}")
            return None

    def _fetch_pmc_fulltext(self, pmcid: str) -> Optional[str]:
        """Fetch full text from PMC."""
        url = f"{EUROPE_PMC_API}/{pmcid}/fullTextXML"
        response = self._make_request(url)

        if not response:
            return None

        try:
            # Parse XML and extract text content
            root = ET.fromstring(response.content)
            # Extract all text from body paragraphs
            texts = []
            for elem in root.iter():
                if elem.tag in ('p', 'title', 'sec', 'abstract'):
                    if elem.text:
                        texts.append(elem.text.strip())
                    for child in elem:
                        if child.tail:
                            texts.append(child.tail.strip())

            return ' '.join(texts) if texts else None
        except ET.ParseError:
            return None

    def fetch_from_pubmed(self, pmid: str) -> Optional[ArticleText]:
        """
        Fetch article abstract from PubMed E-utilities.

        Args:
            pmid: PubMed ID

        Returns:
            ArticleText with abstract (no full text)
        """
        params = {
            "db": "pubmed",
            "id": pmid,
            "rettype": "abstract",
            "retmode": "xml"
        }

        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email

        response = self._make_request(PUBMED_EFETCH, params)
        if not response:
            return None

        try:
            root = ET.fromstring(response.content)
            article = root.find(".//PubmedArticle")
            if article is None:
                return None

            # Extract title
            title_elem = article.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else None

            # Extract abstract
            abstract_elem = article.find(".//Abstract/AbstractText")
            abstract = abstract_elem.text if abstract_elem is not None else None

            # Handle structured abstracts
            if abstract is None:
                abstract_parts = article.findall(".//Abstract/AbstractText")
                if abstract_parts:
                    abstract = ' '.join(
                        (elem.get('Label', '') + ': ' if elem.get('Label') else '') + (elem.text or '')
                        for elem in abstract_parts
                    )

            # Extract DOI
            doi = None
            for id_elem in article.findall(".//ArticleId"):
                if id_elem.get("IdType") == "doi":
                    doi = id_elem.text
                    break

            return ArticleText(
                pmid=pmid,
                doi=doi,
                title=title,
                abstract=abstract,
                full_text=None,  # PubMed doesn't provide full text
                source="pubmed",
                fetch_time=time.time()
            )
        except ET.ParseError as e:
            logger.warning(f"Failed to parse PubMed XML: {e}")
            return None

    def pmid_from_doi(self, doi: str) -> Optional[str]:
        """Look up PMID from DOI using PubMed search."""
        params = {
            "db": "pubmed",
            "term": f"{doi}[doi]",
            "retmode": "json"
        }

        if self.api_key:
            params["api_key"] = self.api_key

        response = self._make_request(PUBMED_ESEARCH, params)
        if not response:
            return None

        try:
            data = response.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])
            return id_list[0] if id_list else None
        except (json.JSONDecodeError, KeyError, IndexError):
            return None

    def fetch_article(
        self,
        doi: str = None,
        pmid: str = None,
        prefer_fulltext: bool = True
    ) -> Optional[ArticleText]:
        """
        Fetch article text, trying multiple sources.

        Args:
            doi: Digital Object Identifier
            pmid: PubMed ID
            prefer_fulltext: If True, try PMC first for full text

        Returns:
            ArticleText if found, None otherwise
        """
        # Normalize DOI to lowercase for consistent cache keys
        if doi:
            doi = doi.lower()

        # Generate cache key
        cache_key = f"doi:{doi}|pmid:{pmid}"

        # Check cache first
        cached = self.cache.get(cache_key)
        if cached:
            self.stats["cache_hits"] += 1
            return cached

        article = None

        # Try Europe PMC first (often has full text)
        if prefer_fulltext:
            if pmid:
                article = self.fetch_from_europe_pmc(pmid=pmid)
            elif doi:
                article = self.fetch_from_europe_pmc(doi=doi)

            # If we got full text, cache and return
            if article and article.full_text:
                self.cache.set(cache_key, article)
                return article

        # Try PubMed for abstract
        if pmid:
            article = self.fetch_from_pubmed(pmid)
        elif doi and not pmid:
            # Look up PMID from DOI
            pmid = self.pmid_from_doi(doi)
            if pmid:
                article = self.fetch_from_pubmed(pmid)

        # Cache result (even if None to avoid repeated lookups)
        if article:
            self.cache.set(cache_key, article)

        return article

    def fetch_batch(
        self,
        identifiers: list,
        max_articles: int = None,
        progress_callback=None
    ) -> Dict[str, ArticleText]:
        """
        Fetch multiple articles.

        Args:
            identifiers: List of dicts with 'doi' and/or 'pmid' keys
            max_articles: Maximum number to fetch
            progress_callback: Optional callback(current, total) for progress

        Returns:
            Dict mapping identifier string to ArticleText
        """
        results = {}
        total = min(len(identifiers), max_articles) if max_articles else len(identifiers)

        for i, ident in enumerate(identifiers[:total]):
            doi = ident.get('doi') or ident.get('referenceDoi')
            pmid = ident.get('pmid')

            # Normalize DOI for consistent key generation
            doi_norm = doi.lower() if doi else None
            key = f"doi:{doi_norm}|pmid:{pmid}"

            article = self.fetch_article(doi=doi, pmid=pmid)
            if article:
                results[key] = article

            if progress_callback and (i + 1) % 100 == 0:
                progress_callback(i + 1, total)

        return results

    def get_stats(self) -> dict:
        """Get fetcher statistics."""
        return self.stats.copy()


def get_article_text(article: ArticleText) -> str:
    """Get the best available text from an article (full text or abstract)."""
    if article.full_text:
        return article.full_text
    if article.abstract:
        return article.abstract
    return ""


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Fetch article text from PMC/PubMed")
    parser.add_argument("--pmid", help="PubMed ID to fetch")
    parser.add_argument("--doi", help="DOI to fetch")
    parser.add_argument("--cache-dir", default="./data/article_cache", help="Cache directory")

    args = parser.parse_args()

    if not args.pmid and not args.doi:
        print("Please provide --pmid or --doi")
        exit(1)

    fetcher = ArticleFetcher(cache_dir=args.cache_dir)
    article = fetcher.fetch_article(doi=args.doi, pmid=args.pmid)

    if article:
        print(f"\n=== Article ===")
        print(f"PMID: {article.pmid}")
        print(f"DOI: {article.doi}")
        print(f"Title: {article.title}")
        print(f"Source: {article.source}")
        print(f"\nAbstract ({len(article.abstract or '')} chars):")
        print(article.abstract[:500] if article.abstract else "N/A")
        if article.full_text:
            print(f"\nFull text ({len(article.full_text)} chars):")
            print(article.full_text[:500] + "...")
    else:
        print("Article not found")

    print(f"\nStats: {fetcher.get_stats()}")
