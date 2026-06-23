#!/usr/bin/env python3
"""
External Database Harvester
============================

Harvests diverse, high-quality biotic interaction sentences from sources
that complement GloBI — targeting categories under-represented in v10:
predation, herbivory, pollination, mutualism.

Sources:
  1. Mangal (mangal.io)       → food web networks → predation / herbivory
  2. Web of Life (web-of-life.es) → mutualistic networks → pollination / symbiosis
  3. OpenAlex (openalex.org)  → open-access paper search → all categories

For each source we obtain:
  - verified species pairs (ecologically grounded, not regex-labeled)
  - the original paper DOI / PMID

Then we search SiBILS (full-text PMC index) for articles mentioning each
species pair, and extract sentences that satisfy ALL quality criteria:
  - contains BOTH species (any name form)
  - contains ≥1 biotic interaction signal word
  - length 40–600 chars, no figure captions, not reference-heavy

Output: data/training/external_db_sentences.csv
  columns: text, label, source_species, target_species,
           interaction_type, category, source

Usage:
    python scripts/fetch_external_databases.py
    python scripts/fetch_external_databases.py --max-positives 2000
    python scripts/fetch_external_databases.py --sources mangal weboflife
    python scripts/fetch_external_databases.py --dry-run
"""

import sys
import re
import json
import time
import hashlib
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import requests
import pandas as pd

# ── project paths ──────────────────────────────────────────────────────────────
CLASSIFIER_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(CLASSIFIER_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

from data.sentence_extractor import split_sentences, generate_name_variants, find_match_in_sentence

# Re-use quality filters & helpers from the main harvester
from fetch_globi_pmc import (
    is_good_sentence,
    BIOTIC_INTERACTION_SIGNALS,
    COMMON_NAMES_TABLE,
    AMBIGUOUS_COMMON_NAMES,
    MIN_SENT_LEN,
    MAX_SENT_LEN,
    _filter_ambiguous,
    _pluralize,
)

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── output ─────────────────────────────────────────────────────────────────────
OUTPUT_DEFAULT = CLASSIFIER_ROOT / "data/training/external_db_sentences.csv"
CACHE_DIR      = CLASSIFIER_ROOT / "data/external_db_cache"

# ── API endpoints ──────────────────────────────────────────────────────────────
SIBILS_URL    = "https://biodiversitypmc.sibils.org/api/search"
MANGAL_URL    = "https://mangal.io/api/v2"
WOL_BASE_URL  = "https://www.web-of-life.es"
OPENALEX_URL  = "https://api.openalex.org/works"

# ── diversity caps (positives per category) ────────────────────────────────────
# Keep pathogen intentionally low — it's already dominant in v10.
CATEGORY_CAPS: Dict[str, int] = {
    "predation":  400,
    "herbivory":  400,
    "pollination": 350,
    "symbiosis":  300,
    "parasitism": 200,
    "dispersal":  150,
    "vector":     100,
    "pathogen":    80,
    "general":     80,
}
DEFAULT_CAP = 80

MAX_SENTS_PER_ARTICLE = 5   # positives extracted per article
MAX_NEG_PER_ARTICLE   = 3   # hard negatives per article
MAX_PAIRS_PER_SOURCE  = 500  # species pairs pulled per external DB
RATE_LIMIT_DELAY      = 0.4  # seconds between HTTP requests

# ── OpenAlex search queries per category ───────────────────────────────────────
OPENALEX_QUERIES: Dict[str, List[str]] = {
    "predation":  [
        "wolf prey elk predation",
        "predator prey interaction field study",
        "raptor prey hunting behavior",
        "shark prey feeding ecology",
    ],
    "herbivory":  [
        "insect herbivory plant damage",
        "caterpillar feeding host plant",
        "aphid plant feeding damage",
        "deer browsing shrub",
    ],
    "pollination": [
        "bee pollination flower plant",
        "butterfly pollinator plant visit",
        "hummingbird floral visitor",
        "moth pollination night flower",
    ],
    "symbiosis":  [
        "mycorrhizal fungus plant symbiosis",
        "ant plant mutualism",
        "cleaner fish mutualism reef",
        "nitrogen fixation Rhizobium legume",
    ],
    "parasitism": [
        "ectoparasite host mammal",
        "parasitoid wasp host insect",
        "nematode host plant parasitism",
    ],
    "dispersal":  [
        "seed dispersal frugivore bird",
        "fruit bat seed dispersal",
        "scatter-hoarding squirrel acorn",
    ],
}


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class SpeciesPair:
    """A verified interacting species pair from an external database."""
    source: str          # scientific name
    target: str          # scientific name
    category: str        # predation / herbivory / pollination / symbiosis / …
    interaction_type: str  # fine-grained label
    doi: Optional[str] = None
    pmid: Optional[str] = None
    db_source: str = ""   # mangal / weboflife / openalex


@dataclass
class HarvestedSentence:
    text: str
    label: int           # 1 = positive, 0 = hard-negative
    source_species: str
    target_species: str
    interaction_type: str
    category: str
    source: str          # e.g. "mangal_sibils" / "weboflife_sibils"


# ==============================================================================
# SIBILS FULL-TEXT SEARCHER
# ==============================================================================

class SibilsSearcher:
    """Searches SiBILS for articles and returns full-text sentences."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir / "sibils"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, query: str, collection: str) -> Path:
        h = hashlib.md5(f"{query}:{collection}".encode()).hexdigest()
        return self.cache_dir / f"{h}.json"

    def search(
        self,
        query: str,
        collection: str = "pmc",
        n: int = 20,
        timeout: int = 30,
    ) -> List[dict]:
        """Return list of article dicts {title, abstract, full_text, pmid, doi}."""
        cache_path = self._cache_path(query, collection)
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)

        params = {"q": query, "col": collection, "n": n}
        try:
            r = requests.get(SIBILS_URL, params=params, timeout=timeout)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.warning(f"SiBILS error for '{query[:60]}': {e}")
            return []

        if not data.get("success", False):
            logger.debug(f"SiBILS no success: {data.get('error', '')}")
            return []

        hits = data.get("elastic_output", {}).get("hits", {}).get("hits", [])
        articles = []
        for h in hits:
            src = h.get("_source", {})
            ft = src.get("full_text") or ""
            abstract = src.get("abstract") or ""
            title = src.get("title") or ""
            text = ft if ft else f"{title}. {abstract}"
            articles.append({
                "text":   text,
                "pmid":   src.get("pmid"),
                "pmcid":  src.get("pmcid"),
                "doi":    src.get("doi"),
                "score":  h.get("_score", 0),
                "collection": collection,
            })

        with open(cache_path, "w") as f:
            json.dump(articles, f)

        return articles

    def search_by_doi(self, doi: str, collection: str = "pmc") -> List[dict]:
        """Search SiBILS for a specific paper by DOI."""
        return self.search(f'doi:"{doi}"', collection=collection, n=5)

    def search_by_pair(
        self, species1: str, species2: str, collection: str = "pmc", n: int = 15
    ) -> List[dict]:
        """Search SiBILS for articles mentioning both species."""
        # Use genus+epithet only for cleaner matching
        query = f'"{species1}" "{species2}"'
        return self.search(query, collection=collection, n=n)


# ==============================================================================
# SENTENCE EXTRACTION
# ==============================================================================

def _name_variants(scientific_name: str) -> Set[str]:
    """Generate name variants (binomial, genus only, common names)."""
    variants: Set[str] = set()
    name = scientific_name.strip()
    if not name or name.lower() == "nan":
        return variants

    variants.add(name)
    parts = name.split()
    if len(parts) >= 2:
        variants.add(parts[0])                   # genus
        variants.add(f"{parts[0]} spp.")
        variants.add(f"{parts[0][0]}. {parts[1]}")  # abbreviated
    if len(parts) >= 1:
        variants.add(_pluralize(name))

    # Common names from lookup table
    for common in COMMON_NAMES_TABLE.get(name, []):
        if common.lower() not in AMBIGUOUS_COMMON_NAMES:
            variants.add(common)
            variants.add(_pluralize(common))

    # Also try generate_name_variants from sentence_extractor
    try:
        variants.update(generate_name_variants(name))
    except Exception:
        pass

    return variants


def extract_sentences(
    article: dict,
    pair: SpeciesPair,
    seen: Set[str],
) -> Tuple[List[HarvestedSentence], List[HarvestedSentence]]:
    """
    Extract positive and hard-negative sentences from an article for a pair.

    Positive  = both species present + biotic interaction signal word.
    Hard-neg  = both species present, NO interaction signal word.
    """
    text = article.get("text", "")
    if not text or len(text) < 100:
        return [], []

    src_variants = _name_variants(pair.source)
    tgt_variants = _name_variants(pair.target)

    if not src_variants or not tgt_variants:
        return [], []

    src_safe = _filter_ambiguous(src_variants, tgt_variants)
    tgt_safe = _filter_ambiguous(tgt_variants, src_variants)

    sentences = split_sentences(text[:600_000])  # guard against huge texts
    positives, hard_negs = [], []

    for sent in sentences:
        if not is_good_sentence(sent):
            continue

        h = hashlib.md5(sent.encode()).hexdigest()
        if h in seen:
            continue

        src_m = find_match_in_sentence(sent, src_safe)
        tgt_m = find_match_in_sentence(sent, tgt_safe)

        if src_m is None or tgt_m is None:
            continue

        # Reject overlapping matches
        if src_m[1] < tgt_m[2] and tgt_m[1] < src_m[2]:
            continue

        sent_lower = sent.lower()
        has_signal = any(sig in sent_lower for sig in BIOTIC_INTERACTION_SIGNALS)

        if has_signal:
            seen.add(h)
            positives.append(HarvestedSentence(
                text=sent.strip(),
                label=1,
                source_species=pair.source,
                target_species=pair.target,
                interaction_type=pair.interaction_type,
                category=pair.category,
                source=f"{pair.db_source}_sibils",
            ))
        else:
            seen.add(h)
            hard_negs.append(HarvestedSentence(
                text=sent.strip(),
                label=0,
                source_species=pair.source,
                target_species=pair.target,
                interaction_type="none_two_species",
                category=pair.category,
                source=f"{pair.db_source}_sibils_neg",
            ))

    return positives[:MAX_SENTS_PER_ARTICLE], hard_negs[:MAX_NEG_PER_ARTICLE]


# ==============================================================================
# SOURCE 1: MANGAL  (food-web networks → predation / herbivory)
# ==============================================================================

class MangalClient:
    """
    Fetches species pairs from the Mangal ecological network database.

    API structure: network → dataset (dataset_id) → reference (ref_id) → doi
    """

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache = cache_dir / "mangal"
        self.cache.mkdir(parents=True, exist_ok=True)

    def _get(self, endpoint: str, params: dict = None) -> list:
        url = f"{MANGAL_URL}/{endpoint}"
        cache_key = hashlib.md5(f"{url}:{json.dumps(params or {}, sort_keys=True)}".encode()).hexdigest()
        cache_path = self.cache / f"{cache_key}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            with open(cache_path, "w") as f:
                json.dump(data, f)
            time.sleep(RATE_LIMIT_DELAY)
            return data
        except Exception as e:
            logger.warning(f"Mangal error {endpoint}: {e}")
            return []

    def _build_ref_index(self) -> Dict[int, str]:
        """Build a mapping ref_id → doi by paginating /reference."""
        ref_doi: Dict[int, str] = {}
        page = 1
        while True:
            data = self._get("reference", {"page": page, "page_size": 100})
            if not data or not isinstance(data, list):
                break
            for ref in data:
                doi = ref.get("doi", "")
                if doi and "10." in str(doi):
                    ref_doi[ref["id"]] = str(doi).strip()
            if len(data) < 100:
                break
            page += 1
        logger.info(f"Mangal: {len(ref_doi)} references with DOIs")
        return ref_doi

    def _build_dataset_doi_index(self, ref_doi: Dict[int, str]) -> Dict[int, str]:
        """Build dataset_id → doi by joining datasets with references."""
        ds_doi: Dict[int, str] = {}
        page = 1
        while True:
            data = self._get("dataset", {"page": page, "page_size": 100})
            if not data or not isinstance(data, list):
                break
            for ds in data:
                ref_id = ds.get("ref_id")
                if ref_id and ref_id in ref_doi:
                    ds_doi[ds["id"]] = ref_doi[ref_id]
            if len(data) < 100:
                break
            page += 1
        logger.info(f"Mangal: {len(ds_doi)} datasets with DOIs")
        return ds_doi

    def get_networks(self, max_networks: int = 200) -> List[dict]:
        """Return networks enriched with a paper DOI."""
        ref_doi   = self._build_ref_index()
        ds_doi    = self._build_dataset_doi_index(ref_doi)

        networks = []
        page = 1
        while len(networks) < max_networks:
            data = self._get("network", {"page": page, "page_size": 100})
            if not data or not isinstance(data, list):
                break
            for net in data:
                ds_id = net.get("dataset_id")
                doi = ds_doi.get(ds_id, "")
                if doi:
                    net["_doi"] = doi
                    networks.append(net)
            if len(data) < 100:
                break
            page += 1
        logger.info(f"Mangal: {len(networks)} networks with DOIs")
        return networks[:max_networks]

    def get_pairs(self, network_id: int, network_doi: str) -> List[SpeciesPair]:
        """Extract species pairs from a Mangal network."""
        nodes_data = self._get("node", {"network_id": network_id, "page_size": 200})
        if not nodes_data or not isinstance(nodes_data, list):
            return []

        node_map: Dict[int, str] = {}
        for node in nodes_data:
            taxonomy = node.get("taxonomy") or {}
            name = (
                taxonomy.get("name")
                or node.get("original_name", "")
                or ""
            ).strip()
            if name and len(name.split()) >= 2:
                node_map[node["id"]] = name

        if not node_map:
            return []

        ints_data = self._get("interaction", {"network_id": network_id, "page_size": 500})
        if not ints_data or not isinstance(ints_data, list):
            return []

        pairs = []
        for inter in ints_data:
            src_id = inter.get("node_from")
            tgt_id = inter.get("node_to")
            if src_id not in node_map or tgt_id not in node_map:
                continue
            src = node_map[src_id]
            tgt = node_map[tgt_id]
            if src == tgt:
                continue

            itype = str(inter.get("type", "predation")).lower()
            if any(t in itype for t in ["predation", "prey", "kill"]):
                category, interaction_type = "predation", "preysOn"
            elif any(t in itype for t in ["herbi", "folivor"]):
                category, interaction_type = "herbivory", "eats"
            elif "parasit" in itype:
                category, interaction_type = "parasitism", "parasiteOf"
            elif any(t in itype for t in ["mutuali", "symbio"]):
                category, interaction_type = "symbiosis", "mutualistOf"
            else:
                category, interaction_type = "predation", "preysOn"  # food web default

            pairs.append(SpeciesPair(
                source=src, target=tgt,
                category=category, interaction_type=interaction_type,
                doi=network_doi, db_source="mangal",
            ))
        return pairs


# ==============================================================================
# SOURCE 2: WEB OF LIFE  (mutualistic networks → pollination / symbiosis)
# ==============================================================================

class WebOfLifeClient:
    """
    Fetches species pairs from the Web of Life database.

    Uses the flat /get_networks.php endpoint which returns all interactions
    directly as {network_name, species1, species2, connection_strength}.

    Network name prefixes encode interaction type:
      M_PL_* = Mutualistic Plant → pollination
      M_SD_* = Mutualistic Seed Dispersal
      M_AF_* = Mutualistic Animal-Fungi
      M_PA_* = Mutualistic Plant-Animal (general)
      A_HP_* = Animal Host-Parasite
      A_PH_* = Animal Plant-Herbivore
      FW_*   = Food Web → predation
    """

    # Map network prefix → (category, interaction_type)
    PREFIX_MAP: Dict[str, Tuple[str, str]] = {
        "M_PL": ("pollination", "pollinates"),
        "M_SD": ("dispersal",   "dispersalVectorOf"),
        "M_AF": ("symbiosis",   "mutualistOf"),
        "M_PA": ("symbiosis",   "mutualistOf"),
        "A_HP": ("parasitism",  "parasiteOf"),
        "A_PH": ("herbivory",   "eats"),
        "FW":   ("predation",   "preysOn"),
    }

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache = cache_dir / "weboflife"
        self.cache.mkdir(parents=True, exist_ok=True)

    def _network_category(self, network_name: str) -> Tuple[str, str]:
        """Infer category and interaction_type from network name prefix."""
        for prefix, cat_itype in self.PREFIX_MAP.items():
            if network_name.startswith(prefix):
                return cat_itype
        return ("symbiosis", "mutualistOf")

    def get_all_pairs(self) -> List[SpeciesPair]:
        """
        Download all interactions from Web of Life in one request.
        Returns species pairs for all networks.
        """
        cache_path = self.cache / "all_interactions.json"
        if cache_path.exists():
            with open(cache_path) as f:
                raw = json.load(f)
        else:
            try:
                r = requests.get(f"{WOL_BASE_URL}/get_networks.php", timeout=30)
                r.raise_for_status()
                raw = r.json()
                with open(cache_path, "w") as f:
                    json.dump(raw, f)
            except Exception as e:
                logger.warning(f"Web of Life error: {e}")
                return []

        pairs: List[SpeciesPair] = []
        for row in raw:
            sp1 = row.get("species1", "").strip().replace("_", " ")
            sp2 = row.get("species2", "").strip().replace("_", " ")
            net_name = row.get("network_name", "")
            # Require binomial names (at least 2 words)
            if len(sp1.split()) < 2 or len(sp2.split()) < 2:
                continue
            try:
                strength = float(row.get("connection_strength", 0))
            except (ValueError, TypeError):
                strength = 1.0
            if strength <= 0:
                continue
            category, interaction_type = self._network_category(net_name)
            pairs.append(SpeciesPair(
                source=sp1, target=sp2,
                category=category, interaction_type=interaction_type,
                doi="", db_source="weboflife",
            ))

        logger.info(f"Web of Life: {len(pairs)} species pairs loaded")
        return pairs


# ==============================================================================
# SOURCE 3: OPENALEX  (open-access paper search → all categories)
# ==============================================================================

class OpenAlexClient:
    """Searches OpenAlex for open-access papers on biotic interactions."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache = cache_dir / "openalex"
        self.cache.mkdir(parents=True, exist_ok=True)

    def search_papers(
        self,
        query: str,
        max_results: int = 50,
    ) -> List[dict]:
        """Return list of {doi, pmid, title} for open-access papers."""
        cache_key = hashlib.md5(query.encode()).hexdigest()
        cache_path = self.cache / f"{cache_key}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)

        params = {
            "search": query,
            "filter": "open_access.is_oa:true,primary_topic.field.id:fields/17",  # biology
            "per-page": max_results,
            "select": "doi,ids,title,primary_topic",
            "mailto": "metap@biodiversity.research",  # polite pool
        }
        try:
            r = requests.get(OPENALEX_URL, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.warning(f"OpenAlex error for '{query}': {e}")
            return []

        results = []
        for work in data.get("results", []):
            doi = work.get("doi", "")
            if doi:
                doi = doi.replace("https://doi.org/", "").strip()
            pmid = work.get("ids", {}).get("pmid", "")
            if pmid:
                pmid = str(pmid).replace("https://pubmed.ncbi.nlm.nih.gov/", "").strip()
            if doi or pmid:
                results.append({"doi": doi, "pmid": pmid, "title": work.get("title", "")})

        time.sleep(RATE_LIMIT_DELAY)
        with open(cache_path, "w") as f:
            json.dump(results, f)
        return results


def _extract_species_from_title(title: str) -> Tuple[Optional[str], Optional[str]]:
    """Heuristic: extract two binomial names from a paper title."""
    binomials = re.findall(r"\b[A-Z][a-z]+\s[a-z]{3,}\b", title)
    # Filter out common non-species words
    stop = {"This", "The", "Our", "Their", "Such", "When", "With", "Here", "These"}
    binomials = [b for b in binomials if b.split()[0] not in stop]
    if len(binomials) >= 2:
        return binomials[0], binomials[1]
    return None, None


# ==============================================================================
# MAIN HARVESTING PIPELINE
# ==============================================================================

def harvest_pairs_from_mangal(max_pairs: int = MAX_PAIRS_PER_SOURCE) -> List[SpeciesPair]:
    """Pull species pairs from Mangal food-web networks."""
    client = MangalClient()
    networks = client.get_networks(max_networks=100)
    all_pairs: List[SpeciesPair] = []
    seen_pairs: Set[Tuple[str, str]] = set()

    for net in networks:
        if len(all_pairs) >= max_pairs:
            break
        net_id = net.get("id")
        doi = net.get("_doi", "")
        if not net_id or not doi:
            continue
        try:
            pairs = client.get_pairs(net_id, doi)
        except Exception as e:
            logger.warning(f"Mangal network {net_id}: {e}")
            continue
        for p in pairs:
            key = (p.source.lower(), p.target.lower())
            if key not in seen_pairs:
                seen_pairs.add(key)
                all_pairs.append(p)

    logger.info(f"Mangal: {len(all_pairs)} unique species pairs")
    return all_pairs[:max_pairs]


def harvest_pairs_from_weboflife(max_pairs: int = MAX_PAIRS_PER_SOURCE) -> List[SpeciesPair]:
    """Pull species pairs from Web of Life (single bulk download)."""
    client = WebOfLifeClient()
    all_pairs = client.get_all_pairs()

    # Deduplicate by (source, target) — keep first occurrence
    seen: Set[Tuple[str, str]] = set()
    unique: List[SpeciesPair] = []
    for p in all_pairs:
        key = (p.source.lower(), p.target.lower())
        if key not in seen:
            seen.add(key)
            unique.append(p)

    # Prioritise pollination (most under-represented)
    pol  = [p for p in unique if p.category == "pollination"]
    rest = [p for p in unique if p.category != "pollination"]
    ordered = pol + rest

    logger.info(f"Web of Life: {len(ordered)} unique species pairs")
    return ordered[:max_pairs]


def harvest_papers_from_openalex(categories: List[str] = None) -> List[SpeciesPair]:
    """
    Search OpenAlex for papers, extract species pairs from titles,
    create pseudo-pairs for SiBILS lookup.
    """
    if categories is None:
        categories = list(OPENALEX_QUERIES.keys())

    client = OpenAlexClient()
    all_pairs: List[SpeciesPair] = []

    # Map category to interaction_type
    category_itype = {
        "predation": "preysOn", "herbivory": "eats",
        "pollination": "pollinates", "symbiosis": "mutualistOf",
        "parasitism": "parasiteOf", "dispersal": "dispersalVectorOf",
    }

    for cat in categories:
        queries = OPENALEX_QUERIES.get(cat, [])
        for q in queries:
            papers = client.search_papers(q, max_results=30)
            for paper in papers:
                doi = paper.get("doi", "")
                pmid = paper.get("pmid", "")
                title = paper.get("title", "")
                sp1, sp2 = _extract_species_from_title(title)

                if sp1 and sp2:
                    all_pairs.append(SpeciesPair(
                        source=sp1, target=sp2,
                        category=cat,
                        interaction_type=category_itype.get(cat, "interactsWith"),
                        doi=doi, pmid=pmid,
                        db_source="openalex",
                    ))
                elif doi or pmid:
                    # Use paper DOI even without pair — SiBILS may surface good sentences
                    all_pairs.append(SpeciesPair(
                        source="", target="",
                        category=cat,
                        interaction_type=category_itype.get(cat, "interactsWith"),
                        doi=doi, pmid=pmid,
                        db_source="openalex",
                    ))

    logger.info(f"OpenAlex: {len(all_pairs)} paper entries")
    return all_pairs


def fetch_sentences(
    pairs: List[SpeciesPair],
    searcher: SibilsSearcher,
    category_caps: Dict[str, int],
) -> Tuple[List[HarvestedSentence], List[HarvestedSentence]]:
    """
    For each species pair, search SiBILS and extract quality sentences.

    Returns (positives, hard_negatives).
    """
    all_pos:  List[HarvestedSentence] = []
    all_neg:  List[HarvestedSentence] = []
    seen:     Set[str] = set()
    cat_counts: Dict[str, int] = defaultdict(int)

    total = len(pairs)
    for i, pair in enumerate(pairs):
        cat = pair.category
        cap = category_caps.get(cat, DEFAULT_CAP)
        if cat_counts[cat] >= cap:
            continue

        # Strategy A: search by DOI (if available) — most precise
        articles = []
        if pair.doi:
            articles = searcher.search_by_doi(pair.doi)
            time.sleep(RATE_LIMIT_DELAY)

        # Strategy B: search by species pair (always, supplements strategy A)
        if pair.source and pair.target:
            pair_articles = searcher.search_by_pair(pair.source, pair.target)
            seen_pmids = {a.get("pmid") or a.get("doi") for a in articles}
            for a in pair_articles:
                uid = a.get("pmid") or a.get("doi")
                if uid not in seen_pmids:
                    articles.append(a)
            time.sleep(RATE_LIMIT_DELAY)

        if not articles:
            continue

        for article in articles:
            if cat_counts[cat] >= cap:
                break
            if not pair.source or not pair.target:
                # Can't do species-pair extraction without known pair
                continue
            try:
                pos, neg = extract_sentences(article, pair, seen)
                for p in pos:
                    all_pos.append(p)
                    cat_counts[cat] += 1
                all_neg.extend(neg)
            except Exception as e:
                logger.debug(f"Extraction error for {pair.source}/{pair.target}: {e}")

        if (i + 1) % 50 == 0:
            total_pos = len(all_pos)
            logger.info(
                f"  [{i+1}/{total}] pos={total_pos} "
                f"by category: { {k: v for k, v in cat_counts.items()} }"
            )

    return all_pos, all_neg


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Harvest diverse biotic interaction sentences from Mangal, Web of Life, OpenAlex"
    )
    parser.add_argument("--output", default=str(OUTPUT_DEFAULT), help="Output CSV path")
    parser.add_argument("--max-positives", type=int, default=2000,
                        help="Stop after this many positives total")
    parser.add_argument("--max-pairs", type=int, default=MAX_PAIRS_PER_SOURCE,
                        help="Max species pairs pulled per source")
    parser.add_argument("--sources", nargs="+",
                        choices=["mangal", "weboflife", "openalex"],
                        default=["mangal", "weboflife", "openalex"],
                        help="Which external databases to use")
    parser.add_argument("--categories", nargs="+",
                        help="Restrict OpenAlex to these categories (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without API calls")
    args = parser.parse_args()

    print("=" * 70)
    print("EXTERNAL DATABASE HARVESTER")
    print("=" * 70)
    print(f"Sources : {args.sources}")
    print(f"Max pos : {args.max_positives}")
    print(f"Output  : {args.output}")

    if args.dry_run:
        print("\n[DRY RUN] Category caps:")
        for cat, cap in sorted(CATEGORY_CAPS.items()):
            print(f"  {cat:15s}: {cap}")
        return

    # ── Step 1: Collect species pairs ─────────────────────────────────────────
    all_pairs: List[SpeciesPair] = []

    if "mangal" in args.sources:
        print("\n1a. Fetching Mangal food-web pairs...")
        mangal_pairs = harvest_pairs_from_mangal(max_pairs=args.max_pairs)
        all_pairs.extend(mangal_pairs)
        by_cat = defaultdict(int)
        for p in mangal_pairs:
            by_cat[p.category] += 1
        print(f"    {len(mangal_pairs)} pairs: {dict(by_cat)}")

    if "weboflife" in args.sources:
        print("\n1b. Fetching Web of Life mutualistic pairs...")
        wol_pairs = harvest_pairs_from_weboflife(max_pairs=args.max_pairs)
        all_pairs.extend(wol_pairs)
        by_cat = defaultdict(int)
        for p in wol_pairs:
            by_cat[p.category] += 1
        print(f"    {len(wol_pairs)} pairs: {dict(by_cat)}")

    if "openalex" in args.sources:
        print("\n1c. Searching OpenAlex for interaction papers...")
        oa_pairs = harvest_papers_from_openalex(categories=args.categories)
        all_pairs.extend(oa_pairs)
        by_cat = defaultdict(int)
        for p in oa_pairs:
            by_cat[p.category] += 1
        print(f"    {len(oa_pairs)} entries: {dict(by_cat)}")

    if not all_pairs:
        print("ERROR: No pairs collected. Check API connectivity.")
        sys.exit(1)

    print(f"\nTotal pairs/entries: {len(all_pairs)}")

    # Shuffle to interleave sources and avoid per-category exhaustion
    import random
    random.seed(42)
    random.shuffle(all_pairs)

    # ── Step 2: Fetch full text + extract sentences ────────────────────────────
    print("\n2. Searching SiBILS and extracting sentences...")
    searcher = SibilsSearcher()

    # Enforce global max-positives by adjusting per-category caps proportionally
    total_cap = sum(CATEGORY_CAPS.values())
    if args.max_positives < total_cap:
        scale = args.max_positives / total_cap
        caps = {k: max(10, int(v * scale)) for k, v in CATEGORY_CAPS.items()}
    else:
        caps = CATEGORY_CAPS.copy()

    positives, hard_negs = fetch_sentences(all_pairs, searcher, caps)

    print(f"\nExtracted: {len(positives)} positives, {len(hard_negs)} hard-negatives")

    if not positives:
        print("ERROR: No positive sentences extracted. Check SiBILS connectivity.")
        sys.exit(1)

    # ── Step 3: Balance negatives ──────────────────────────────────────────────
    target_neg = len(positives)
    if len(hard_negs) > target_neg:
        import random
        hard_negs = random.sample(hard_negs, target_neg)

    # ── Step 4: Save ───────────────────────────────────────────────────────────
    output_path = Path(args.output)

    rows = [
        {
            "text": s.text,
            "label": s.label,
            "source_species": s.source_species,
            "target_species": s.target_species,
            "interaction_type": s.interaction_type,
            "category": s.category,
            "source": s.source,
        }
        for s in positives + hard_negs
    ]
    df_new = pd.DataFrame(rows)

    # Append to existing output if present
    if output_path.exists():
        df_existing = pd.read_csv(output_path)
        existing_texts = set(df_existing["text"].str.lower().str.strip())
        df_new = df_new[~df_new["text"].str.lower().str.strip().isin(existing_texts)]
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
        logger.info(f"Appended {len(df_new)} new rows to existing {len(df_existing)}")
    else:
        df_out = df_new

    df_out.to_csv(output_path, index=False)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal saved: {len(df_out)} rows ({output_path})")
    pos_df = df_out[df_out["label"] == 1]
    neg_df = df_out[df_out["label"] == 0]
    print(f"  Positives : {len(pos_df)}")
    print(f"  Negatives : {len(neg_df)}")
    if len(pos_df):
        print("\nPositives by category:")
        for cat, cnt in pos_df["category"].value_counts().items():
            print(f"  {cat:15s}: {cnt}")
        print("\nPositives by source:")
        for src, cnt in pos_df["source"].value_counts().items():
            print(f"  {src:30s}: {cnt}")


if __name__ == "__main__":
    main()
