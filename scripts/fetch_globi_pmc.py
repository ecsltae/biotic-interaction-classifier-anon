#!/usr/bin/env python3
"""
Harvest Real Training Sentences from GloBI-Cited PMC Articles
=============================================================

Pipeline:
1. Load GloBI interactions with DOI/PMC references
2. Batch DOI → PMCID conversion via NCBI ID Converter
3. Resolve common names via NCBI Taxonomy API
4. Fetch PMC full text via Europe PMC / NCBI E-utilities
5. Extract co-occurrence sentences (both species in same sentence)
6. Generate hard negatives from same papers
7. Output balanced training CSV

Key innovation: No regex-based labeling. GloBI already curated the interaction,
so any sentence mentioning BOTH species from a known pair is a positive example.

Usage:
    python scripts/fetch_globi_pmc.py --max-positives 5000 --max-articles 10000
    python scripts/fetch_globi_pmc.py --dry-run --max-articles 20
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
from dataclasses import dataclass, field, asdict

import pandas as pd
import requests
import xml.etree.ElementTree as ET

# Add project root to path
CLASSIFIER_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(CLASSIFIER_ROOT / "src"))

from data.sentence_extractor import (
    split_sentences,
    generate_name_variants,
    find_match_in_sentence,
)
from data.article_fetcher import ArticleFetcher, ArticleCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBI_PATH = CLASSIFIER_ROOT / "data" / "globi" / "interactions.tsv.gz"
OUTPUT_DEFAULT = CLASSIFIER_ROOT / "data" / "training" / "globi_pmc_real_sentences.csv"
CACHE_DIR_DEFAULT = CLASSIFIER_ROOT / "data" / "pmc_cache"
NAME_CACHE_DEFAULT = CLASSIFIER_ROOT / "data" / "species_names_cache.json"

# NCBI API endpoints
NCBI_IDCONV = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
EUROPE_PMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest"

# Interaction types to target (GloBI interactionTypeName values)
TARGET_INTERACTION_TYPES = {
    # Parasitism (note: GloBI uses lowercase 'p' in endoparasiteOf / ectoparasiteOf)
    "parasiteOf", "hasHost", "endoparasiteOf", "ectoparasiteOf", "parasitoidOf",
    "hemiparasiteOf",  # 24,428 DOIs — hemiparasitic plants (mistletoe, etc.)
    "hyperparasiteOf",  # parasite of a parasite
    "kleptoparasiteOf",  # food theft
    # Predation
    "preysOn", "preyedUponBy", "kills",
    # Herbivory
    "eats", "ateBy",
    # Pathogen / vector
    "pathogenOf",
    "vectorOf",
    # Pollination
    "pollinates", "flowersVisitedBy", "visitedBy", "visitsFlowersOf",  # visitsFlowersOf: 34,014 DOIs
    # Symbiosis / commensalism
    "symbioticWith", "mutualistOf", "symbiontOf",  # symbiontOf: 2,949 DOIs
    "commensalistOf",  # 325 DOIs
    # Dispersal
    "dispersalVectorOf",
    # General / ecological
    "interactsWith", "adjacentTo", "coOccursWith",
    "ecologicallyRelatedTo",  # 6,892 DOIs
    "hostOf",
}

# Map GloBI types to broader categories for diversity control
INTERACTION_CATEGORY_MAP = {
    # Parasitism
    "parasiteOf": "parasitism", "hasHost": "parasitism",
    "endoparasiteOf": "parasitism", "ectoparasiteOf": "parasitism",
    "parasitoidOf": "parasitism", "hostOf": "parasitism",
    "hemiparasiteOf": "parasitism", "hyperparasiteOf": "parasitism",
    "kleptoparasiteOf": "predation",
    # Predation
    "preysOn": "predation", "preyedUponBy": "predation", "kills": "predation",
    # Herbivory
    "eats": "herbivory", "ateBy": "herbivory",
    # Pathogen / vector
    "pathogenOf": "pathogen",
    "vectorOf": "vector",
    # Pollination
    "pollinates": "pollination", "flowersVisitedBy": "pollination",
    "visitedBy": "pollination", "visitsFlowersOf": "pollination",
    # Symbiosis
    "symbioticWith": "symbiosis", "mutualistOf": "symbiosis",
    "symbiontOf": "symbiosis", "commensalistOf": "symbiosis",
    # Dispersal
    "dispersalVectorOf": "dispersal",
    # General
    "interactsWith": "general", "adjacentTo": "general", "coOccursWith": "general",
    "ecologicallyRelatedTo": "general",
}

# Limits
MAX_SENTS_PER_ARTICLE = 8    # Max positive sentences per article (increased for diversity)
MAX_NEG_PER_ARTICLE = 8      # Max negative sentences per article
# Per-category caps — keep pathogen low, boost biodiversity categories
MAX_PER_CATEGORY: Dict[str, int] = {
    "parasitism":  400,
    "herbivory":   400,
    "predation":   300,
    "pollination": 200,
    "symbiosis":   200,
    "dispersal":   150,
    "vector":      150,
    "pathogen":    150,  # deliberately capped low
    "general":     100,
}
DEFAULT_CATEGORY_CAP = 150   # fallback for unmapped categories

# Priority order for processing: ecological first, pathogen last
CATEGORY_PRIORITY = [
    "predation", "herbivory", "parasitism", "pollination",
    "symbiosis", "dispersal", "vector", "general", "pathogen",
]

MIN_SENT_LEN = 40
MAX_SENT_LEN = 600            # increased from 500 — ecological descriptions can be long
RATE_LIMIT_DELAY = 0.35       # seconds between API calls

# Words that signal a genuine biotic interaction in a sentence.
# Based on GloBI interaction vocabulary (sentence_extractor.py globi_mappings).
# Only "eat/eats/ate/eaten/eating" excluded — too generic ("bacteria eat substrate",
# "patients ate a meal"). All other interaction terms kept.
BIOTIC_INTERACTION_SIGNALS: Set[str] = {
    # Predation (preysOn, preyedUponBy, kills)
    "prey", "preys", "preyed", "preying",
    "predator", "predation", "predatory",
    "hunt", "hunts", "hunted", "hunting", "hunter",
    "kill", "kills", "killed", "killing",
    # Feeding / herbivory (eats excluded, but feed/forage/graze/ingest kept)
    "feed", "feeds", "feeding", "fed", "feeds on", "fed on",
    "forage", "forages", "foraging",
    "graze", "grazes", "grazing",
    "browse", "browses", "browsing",
    "ingest", "ingests", "ingested", "ingestion",
    "diet", "consume", "consumes", "consumed", "consumption",
    # Parasitism (parasiteOf, ectoParasiteOf, endoParasiteOf, parasitoidOf)
    "parasite", "parasitic", "parasitism",
    "parasitize", "parasitizes", "parasitized", "parasitizing",
    "parasitoid",
    "ectoparasite", "ectoparasitic",
    "endoparasite", "endoparasitic",
    "infest", "infests", "infested", "infestation",
    # Host (hasHost, hostOf)
    "host", "hosts", "hosted", "hosting",
    # Pathogen / infection (pathogenOf, vectorOf)
    "infect", "infects", "infected", "infecting", "infection", "infectious",
    "pathogen", "pathogenic",
    "transmit", "transmits", "transmitted", "transmitting", "transmission",
    "vector", "reservoir", "virulence",
    # Pollination / flower visiting (pollinates, flowersVisitedBy, visitedBy)
    "pollinate", "pollinates", "pollinated", "pollinating", "pollinator", "pollination",
    "visit", "visits", "visited", "visitor",
    # Symbiosis / mutualism / commensalism (symbioticWith, mutualistOf)
    "symbiont", "symbiosis", "symbiotic",
    "mutualist", "mutualism", "mutualistic",
    "commensal", "commensalism",
    # Dispersal (dispersalVectorOf)
    "disperse", "disperses", "dispersed", "dispersal",
    # General (interactsWith, colonizes, attacks)
    "interact", "interacts", "interaction",
    "colonize", "colonizes", "colonized", "colonization",
    "attack", "attacks", "attacked", "attacking",
    "associat",
}

# Ambiguous common names that are too generic to match alone
AMBIGUOUS_COMMON_NAMES = {
    "fly", "flies", "mouse", "mice", "bug", "bugs", "mite", "mites",
    "tick", "ticks", "ant", "ants", "bee", "bees", "wasp", "wasps",
    "worm", "worms", "frog", "frogs", "fish", "rat", "rats",
    "bird", "birds", "snake", "snakes", "crab", "crabs",
    "louse", "lice", "flea", "fleas", "moth", "moths",
}

# =============================================================================
# HARDCODED COMMON NAMES (top ecological species)
# =============================================================================

COMMON_NAMES_TABLE: Dict[str, List[str]] = {
    # ── Primates ──────────────────────────────────────────────────────────────
    "Homo sapiens": ["human", "humans", "person", "people"],
    "Pan troglodytes": ["chimpanzee", "chimpanzees", "chimp", "chimps"],
    "Pan paniscus": ["bonobo", "bonobos"],
    "Gorilla gorilla": ["gorilla", "western gorilla"],
    "Macaca mulatta": ["rhesus macaque", "rhesus monkey", "rhesus"],
    "Macaca fascicularis": ["crab-eating macaque", "long-tailed macaque", "cynomolgus monkey"],
    "Papio ursinus": ["chacma baboon", "baboon"],
    "Papio hamadryas": ["hamadryas baboon", "baboon"],
    # ── Carnivores ────────────────────────────────────────────────────────────
    "Canis lupus": ["gray wolf", "grey wolf", "wolf", "wolves"],
    "Canis lupus familiaris": ["dog", "domestic dog", "dogs"],
    "Canis latrans": ["coyote", "coyotes"],
    "Canis aureus": ["golden jackal", "jackal"],
    "Canis mesomelas": ["black-backed jackal", "jackal"],
    "Vulpes vulpes": ["red fox", "fox"],
    "Vulpes lagopus": ["Arctic fox", "arctic fox"],
    "Felis catus": ["cat", "domestic cat", "cats"],
    "Felis silvestris": ["wildcat", "European wildcat"],
    "Panthera leo": ["lion", "lions"],
    "Panthera pardus": ["leopard", "leopards"],
    "Panthera tigris": ["tiger", "tigers"],
    "Panthera onca": ["jaguar", "jaguars"],
    "Puma concolor": ["mountain lion", "cougar", "puma"],
    "Lynx lynx": ["Eurasian lynx", "lynx"],
    "Lynx canadensis": ["Canada lynx", "lynx"],
    "Lynx rufus": ["bobcat"],
    "Ursus arctos": ["brown bear", "grizzly bear", "bear"],
    "Ursus americanus": ["black bear", "American black bear"],
    "Ursus maritimus": ["polar bear"],
    "Enhydra lutris": ["sea otter"],
    "Lutra lutra": ["European otter", "otter"],
    "Mustela erminea": ["stoat", "ermine"],
    "Mustela nivalis": ["weasel", "least weasel"],
    "Mustela putorius": ["European polecat", "polecat"],
    "Martes martes": ["pine marten", "marten"],
    "Martes foina": ["stone marten", "beech marten", "marten"],
    "Meles meles": ["European badger", "badger"],
    "Taxidea taxus": ["American badger", "badger"],
    "Gulo gulo": ["wolverine"],
    "Hyena hyena": ["striped hyena", "hyena"],
    "Crocuta crocuta": ["spotted hyena", "hyena"],
    # ── Ungulates ─────────────────────────────────────────────────────────────
    "Bos taurus": ["cattle", "cow", "cows", "bovine"],
    "Sus scrofa": ["wild boar", "boar", "pig", "feral pig"],
    "Ovis aries": ["sheep", "domestic sheep"],
    "Capra hircus": ["goat", "domestic goat"],
    "Equus caballus": ["horse", "horses"],
    "Equus asinus": ["donkey", "ass"],
    "Equus zebra": ["zebra"],
    "Cervus elaphus": ["red deer", "elk", "wapiti"],
    "Odocoileus virginianus": ["white-tailed deer", "white-tailed deer", "deer"],
    "Odocoileus hemionus": ["mule deer", "deer"],
    "Rangifer tarandus": ["reindeer", "caribou"],
    "Alces alces": ["moose", "elk"],
    "Capreolus capreolus": ["roe deer", "deer"],
    "Dama dama": ["fallow deer", "deer"],
    "Bison bison": ["American bison", "bison", "buffalo"],
    # ── Small mammals ─────────────────────────────────────────────────────────
    "Mus musculus": ["house mouse", "mouse"],
    "Rattus norvegicus": ["brown rat", "Norway rat", "rat"],
    "Rattus rattus": ["black rat", "ship rat", "rat"],
    "Myodes glareolus": ["bank vole", "vole"],
    "Clethrionomys rutilus": ["northern red-backed vole", "vole"],
    "Microtus oeconomus": ["tundra vole", "vole"],
    "Microtus pennsylvanicus": ["meadow vole", "vole"],
    "Microtus agrestis": ["field vole", "short-tailed vole", "vole"],
    "Arvicola amphibius": ["European water vole", "water vole", "vole"],
    "Apodemus sylvaticus": ["wood mouse", "long-tailed field mouse"],
    "Apodemus flavicollis": ["yellow-necked mouse"],
    "Peromyscus maniculatus": ["deer mouse"],
    "Peromyscus keeni": ["keen's mouse", "deer mouse"],
    "Dipodomys merriami": ["Merriam's kangaroo rat", "kangaroo rat"],
    "Dipodomys ordii": ["Ord's kangaroo rat", "kangaroo rat"],
    "Lepus europaeus": ["European hare", "brown hare", "hare"],
    "Lepus americanus": ["snowshoe hare", "hare"],
    "Lepus timidus": ["mountain hare", "blue hare", "hare"],
    "Oryctolagus cuniculus": ["European rabbit", "rabbit"],
    "Sciurus vulgaris": ["Eurasian red squirrel", "squirrel"],
    "Sciurus carolinensis": ["gray squirrel", "grey squirrel"],
    "Castor fiber": ["Eurasian beaver", "beaver"],
    "Castor canadensis": ["North American beaver", "beaver"],
    "Sorex araneus": ["common shrew", "shrew"],
    "Talpa europaea": ["European mole", "mole"],
    "Erinaceus europaeus": ["western European hedgehog", "hedgehog"],
    # ── Bats ──────────────────────────────────────────────────────────────────
    "Myotis lucifugus": ["little brown bat", "little brown myotis"],
    "Myotis velifer": ["cave myotis bat", "bat"],
    "Eptesicus fuscus": ["big brown bat", "bat"],
    "Tadarida brasiliensis": ["Brazilian free-tailed bat", "bat"],
    # ── Marine mammals ────────────────────────────────────────────────────────
    "Phoca vitulina": ["harbor seal", "common seal", "seal"],
    "Halichoerus grypus": ["grey seal", "gray seal", "seal"],
    "Orcinus orca": ["killer whale", "orca"],
    "Tursiops truncatus": ["bottlenose dolphin", "dolphin"],
    "Megaptera novaeangliae": ["humpback whale"],
    # ── Birds ──────────────────────────────────────────────────────────────────
    "Gallus gallus": ["chicken", "red junglefowl", "domestic chicken"],
    "Passer domesticus": ["house sparrow", "sparrow"],
    "Sturnus vulgaris": ["common starling", "starling"],
    "Corvus corax": ["common raven", "raven"],
    "Corvus corone": ["carrion crow", "crow"],
    "Corvus brachyrhynchos": ["American crow", "crow"],
    "Buteo buteo": ["common buzzard", "buzzard"],
    "Buteo jamaicensis": ["red-tailed hawk", "hawk"],
    "Accipiter nisus": ["Eurasian sparrowhawk", "sparrowhawk"],
    "Accipiter gentilis": ["northern goshawk", "goshawk"],
    "Accipiter cooperii": ["Cooper's hawk", "hawk"],
    "Falco peregrinus": ["peregrine falcon", "falcon"],
    "Falco tinnunculus": ["common kestrel", "kestrel"],
    "Aquila chrysaetos": ["golden eagle", "eagle"],
    "Haliaeetus leucocephalus": ["bald eagle", "eagle"],
    "Tyto alba": ["barn owl", "owl"],
    "Strix aluco": ["tawny owl", "owl"],
    "Asio otus": ["long-eared owl", "owl"],
    "Bubo bubo": ["Eurasian eagle-owl", "eagle owl"],
    "Cuculus canorus": ["common cuckoo", "cuckoo"],
    "Turdus migratorius": ["American robin", "robin"],
    "Turdus merula": ["common blackbird", "blackbird"],
    "Hirundo rustica": ["barn swallow", "swallow"],
    "Acrocephalus arundinaceus": ["great reed warbler", "warbler"],
    "Spinus tristis": ["American goldfinch", "goldfinch"],
    "Columba livia": ["rock pigeon", "pigeon", "dove"],
    "Phasianus colchicus": ["ring-necked pheasant", "pheasant"],
    "Coturnix japonica": ["Japanese quail", "quail"],
    "Anser anser": ["greylag goose", "goose"],
    "Anas platyrhynchos": ["mallard duck", "mallard", "duck"],
    # ── Reptiles ──────────────────────────────────────────────────────────────
    "Thamnophis sirtalis": ["common garter snake", "garter snake"],
    "Python molurus": ["Indian python", "python"],
    "Crocodylus niloticus": ["Nile crocodile", "crocodile"],
    "Iguana iguana": ["green iguana", "iguana"],
    "Chelonia mydas": ["green sea turtle", "sea turtle"],
    # ── Amphibians ────────────────────────────────────────────────────────────
    "Rana temporaria": ["common frog", "European common frog"],
    "Bufo bufo": ["common toad", "toad"],
    "Xenopus laevis": ["African clawed frog", "clawed frog"],
    # ── Fish ──────────────────────────────────────────────────────────────────
    "Salmo salar": ["Atlantic salmon", "salmon"],
    "Salmo trutta": ["brown trout", "trout"],
    "Oncorhynchus mykiss": ["rainbow trout", "trout"],
    "Oncorhynchus tshawytscha": ["Chinook salmon", "king salmon", "salmon"],
    "Gadus morhua": ["Atlantic cod", "cod"],
    "Thunnus thynnus": ["Atlantic bluefin tuna", "tuna"],
    "Clupea harengus": ["Atlantic herring", "herring"],
    "Anguilla anguilla": ["European eel", "eel"],
    "Esox lucius": ["northern pike", "pike"],
    "Perca fluviatilis": ["European perch", "perch"],
    "Brevoortia patronus": ["Gulf menhaden", "menhaden"],
    "Anchoa mitchilli": ["bay anchovy", "anchovy"],
    "Micropogonias undulatus": ["Atlantic croaker"],
    "Leiostomus xanthurus": ["spot croaker", "spot"],
    "Mugil cephalus": ["flathead grey mullet", "mullet"],
    # ── Insects ───────────────────────────────────────────────────────────────
    "Apis mellifera": ["honey bee", "honeybee", "western honey bee"],
    "Bombus terrestris": ["buff-tailed bumblebee", "bumblebee"],
    "Bombus impatiens": ["common eastern bumblebee", "bumblebee"],
    "Vespa velutina": ["Asian hornet", "yellow-legged hornet", "hornet"],
    "Vespa crabro": ["European hornet", "hornet"],
    "Drosophila melanogaster": ["fruit fly"],
    "Aedes aegypti": ["yellow fever mosquito", "mosquito"],
    "Anopheles gambiae": ["African malaria mosquito", "mosquito"],
    "Culex pipiens": ["common house mosquito", "mosquito"],
    "Ixodes ricinus": ["castor bean tick", "sheep tick", "tick"],
    "Ixodes scapularis": ["black-legged tick", "deer tick", "tick"],
    "Dermacentor variabilis": ["American dog tick", "tick"],
    "Pieris brassicae": ["large white butterfly", "cabbage white butterfly", "butterfly"],
    "Pieris rapae": ["small white butterfly", "cabbage butterfly"],
    "Manduca sexta": ["tobacco hornworm", "hornworm"],
    "Spodoptera frugiperda": ["fall armyworm", "armyworm"],
    "Helicoverpa armigera": ["cotton bollworm", "bollworm"],
    "Locusta migratoria": ["migratory locust", "locust"],
    "Schistocerca gregaria": ["desert locust", "locust"],
    "Acyrthosiphon pisum": ["pea aphid", "aphid"],
    "Myzus persicae": ["green peach aphid", "aphid"],
    "Tribolium castaneum": ["red flour beetle", "flour beetle"],
    "Tenebrio molitor": ["mealworm beetle", "mealworm"],
    "Chrysobothris mali": ["flatheaded apple tree borer", "borer"],
    "Chrysophana placida": ["golden buprestid beetle"],
    "Cynthia cardui": ["painted lady butterfly", "butterfly"],
    "Halictus tripartitus": ["furrow bee", "sweat bee"],
    # ── Spiders & Arachnids ───────────────────────────────────────────────────
    "Tibellus oblongus": ["slender crab spider", "spider"],
    "Argiope bruennichi": ["wasp spider", "garden spider"],
    # ── Marine invertebrates ──────────────────────────────────────────────────
    "Crassostrea gigas": ["Pacific oyster", "oyster"],
    "Mytilus edulis": ["blue mussel", "mussel"],
    "Strongylocentrotus droebachiensis": ["green sea urchin", "sea urchin"],
    "Pisaster ochraceus": ["ochre sea star", "purple sea star", "starfish"],
    "Chrysaora quinquecirrha": ["sea nettle jellyfish", "sea nettle"],
    "Stomolophus meleagris": ["cannonball jellyfish"],
    "Katharina tunicata": ["black chiton", "chiton"],
    "Chthamalus dalli": ["small acorn barnacle", "barnacle"],
    "Pilumnus lacteus": ["hairy crab", "crab"],
    "Libinia dubia": ["spider crab", "longnose spider crab"],
    "Squatina dumeril": ["Atlantic angelshark", "angelshark"],
    # ── Plants ────────────────────────────────────────────────────────────────
    "Arabidopsis thaliana": ["thale cress", "mouse-ear cress"],
    "Zea mays": ["maize", "corn"],
    "Oryza sativa": ["rice"],
    "Triticum aestivum": ["wheat", "bread wheat", "common wheat"],
    "Solanum lycopersicum": ["tomato"],
    "Quercus robur": ["English oak", "pedunculate oak", "oak"],
    "Quercus alba": ["white oak", "oak"],
    "Quercus ilex": ["holm oak", "evergreen oak", "oak"],
    "Pseudotsuga menziesii": ["Douglas fir", "Douglas-fir"],
    "Pinus ponderosa": ["ponderosa pine", "pine"],
    "Pinus sylvestris": ["Scots pine", "pine"],
    "Malus pumila": ["apple tree", "apple"],
    "Malus domestica": ["apple", "apple tree"],
    "Taraxacum officinale": ["common dandelion", "dandelion"],
    "Ericameria nauseosa": ["rubber rabbitbrush", "rabbitbrush"],
    "Chrysopsis villosa": ["hairy golden aster"],
    "Heliomeris multiflora": ["showy goldeneye"],
    # ── Pathogens (kept for reference) ────────────────────────────────────────
    "Plasmodium falciparum": ["malaria parasite"],
    "Trypanosoma cruzi": ["Chagas disease parasite"],
    "Toxoplasma gondii": ["toxoplasma"],
    "Borrelia burgdorferi": ["Lyme disease spirochete"],
    "Echinococcus multilocularis": ["alveolar echinococcosis tapeworm", "tapeworm"],
}


# =============================================================================
# SPECIES NAME RESOLVER
# =============================================================================

class SpeciesNameResolver:
    """Resolves scientific names to common names with multi-source lookup and caching."""

    def __init__(self, cache_path: Path, ncbi_api_key: Optional[str] = None):
        self.cache_path = cache_path
        self.ncbi_api_key = ncbi_api_key
        self.cache: Dict[str, List[str]] = {}
        self.last_request_time = 0.0
        self.stats = {"cache_hits": 0, "api_lookups": 0, "api_failures": 0}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cached common names from disk."""
        # Start with hardcoded table
        for sci_name, common_names in COMMON_NAMES_TABLE.items():
            self.cache[sci_name.lower()] = common_names

        # Load persisted cache
        if self.cache_path.exists():
            try:
                with open(self.cache_path) as f:
                    saved = json.load(f)
                self.cache.update(saved)
                logger.info(f"Loaded {len(saved)} cached common names")
            except (json.JSONDecodeError, IOError):
                pass

    def _save_cache(self) -> None:
        """Persist cache to disk."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f, indent=2)

    def _rate_limit(self) -> None:
        elapsed = time.time() - self.last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self.last_request_time = time.time()

    def _lookup_ncbi(self, scientific_name: str) -> List[str]:
        """Look up common names via NCBI Taxonomy API."""
        self._rate_limit()
        self.stats["api_lookups"] += 1

        try:
            # Step 1: esearch to get taxon ID
            params = {"db": "taxonomy", "term": scientific_name, "retmode": "json"}
            if self.ncbi_api_key:
                params["api_key"] = self.ncbi_api_key

            resp = requests.get(NCBI_ESEARCH, params=params, timeout=15)
            if resp.status_code != 200:
                self.stats["api_failures"] += 1
                return []

            data = resp.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return []

            taxon_id = id_list[0]

            # Step 2: efetch to get common names
            self._rate_limit()
            params = {
                "db": "taxonomy", "id": taxon_id, "rettype": "xml",
            }
            if self.ncbi_api_key:
                params["api_key"] = self.ncbi_api_key

            resp = requests.get(NCBI_EFETCH, params=params, timeout=15)
            if resp.status_code != 200:
                self.stats["api_failures"] += 1
                return []

            root = ET.fromstring(resp.content)
            common_names = []

            # Extract common name fields
            for tag in ["GenbankCommonName", "CommonName"]:
                elem = root.find(f".//{tag}")
                if elem is not None and elem.text:
                    name = elem.text.strip().lower()
                    if name and name not in common_names:
                        common_names.append(name)

            # Also check OtherNames/CommonName
            for elem in root.findall(".//OtherNames/CommonName"):
                if elem.text:
                    name = elem.text.strip().lower()
                    if name and name not in common_names:
                        common_names.append(name)

            return common_names

        except (requests.RequestException, ET.ParseError, KeyError) as e:
            logger.debug(f"NCBI lookup failed for {scientific_name}: {e}")
            self.stats["api_failures"] += 1
            return []

    def get_common_names(self, scientific_name: str) -> List[str]:
        """Get common names for a scientific name (cached + API)."""
        key = scientific_name.lower().strip()

        if key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[key]

        # API lookup
        names = self._lookup_ncbi(scientific_name)
        self.cache[key] = names  # Cache even empty results
        return names

    def batch_resolve(self, species_list: List[str]) -> None:
        """Pre-resolve a batch of species names and save cache."""
        uncached = [s for s in species_list if s.lower().strip() not in self.cache]
        logger.info(f"Resolving common names: {len(uncached)} uncached of {len(species_list)} total")

        for i, species in enumerate(uncached):
            if i > 0 and i % 100 == 0:
                logger.info(f"  Resolved {i}/{len(uncached)} species...")
                self._save_cache()  # Periodic save
            self.get_common_names(species)

        self._save_cache()
        logger.info(f"Name resolution done. Stats: {self.stats}")

    def get_all_name_forms(self, scientific_name: str) -> Set[str]:
        """Get all searchable name forms including common names.

        Returns set of name variants for matching in article text.
        """
        # Scientific name variants from existing utility
        variants = generate_name_variants(scientific_name)

        # Common name variants
        common_names = self.get_common_names(scientific_name)
        for name in common_names:
            variants.add(name.lower())
            # Title case
            variants.add(name.title())
            # Simple pluralization
            plural = _pluralize(name)
            if plural != name:
                variants.add(plural.lower())
                variants.add(plural.title())

        return variants


def _pluralize(name: str) -> str:
    """Simple English pluralization for common names."""
    if not name:
        return name
    # Irregular plurals
    irregulars = {
        "mouse": "mice", "louse": "lice", "goose": "geese",
        "foot": "feet", "tooth": "teeth", "ox": "oxen",
        "child": "children", "man": "men", "woman": "women",
        "deer": "deer", "sheep": "sheep", "fish": "fish",
        "moose": "moose", "salmon": "salmon", "trout": "trout",
    }
    last_word = name.split()[-1].lower()
    if last_word in irregulars:
        return name.rsplit(last_word, 1)[0] + irregulars[last_word]

    # Regular rules
    if last_word.endswith(("s", "x", "z", "ch", "sh")):
        return name + "es"
    if last_word.endswith("y") and len(last_word) >= 2 and last_word[-2] not in "aeiou":
        return name[:-1] + "ies"
    return name + "s"


# =============================================================================
# DOI → PMCID CONVERSION
# =============================================================================

def batch_doi_to_pmcid(
    dois: List[str],
    batch_size: int = 200,
    api_key: Optional[str] = None,
) -> Dict[str, str]:
    """Convert DOIs to PMCIDs using NCBI ID Converter API.

    Args:
        dois: List of DOI strings
        batch_size: Max DOIs per API call (NCBI limit: 200)
        api_key: Optional NCBI API key

    Returns:
        Dict mapping DOI → PMCID (only for DOIs with PMC articles)
    """
    doi_to_pmcid: Dict[str, str] = {}
    unique_dois = list(set(d for d in dois if d and isinstance(d, str)))
    logger.info(f"Converting {len(unique_dois)} unique DOIs to PMCIDs...")

    for i in range(0, len(unique_dois), batch_size):
        batch = unique_dois[i:i + batch_size]
        ids_str = ",".join(batch)

        params = {
            "ids": ids_str, "format": "json",
            "tool": "metap_classifier", "email": "metap@research.edu",
        }
        if api_key:
            params["api_key"] = api_key

        try:
            time.sleep(RATE_LIMIT_DELAY)
            resp = requests.get(NCBI_IDCONV, params=params, timeout=30, allow_redirects=True)
            if resp.status_code != 200:
                logger.warning(f"ID converter returned {resp.status_code} for batch {i}")
                continue

            data = resp.json()
            for record in data.get("records", []):
                pmcid = record.get("pmcid")
                doi = record.get("doi")
                if pmcid and doi:
                    doi_to_pmcid[doi] = pmcid

        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.warning(f"Batch {i} failed: {e}")

        if (i // batch_size + 1) % 10 == 0:
            logger.info(f"  Processed {i + len(batch)}/{len(unique_dois)} DOIs, found {len(doi_to_pmcid)} PMCIDs")

    logger.info(f"DOI→PMCID conversion done: {len(doi_to_pmcid)}/{len(unique_dois)} have PMC full text")
    return doi_to_pmcid


def extract_pmcid_from_url(url: str) -> Optional[str]:
    """Extract PMCID from a URL if present."""
    if not url or not isinstance(url, str):
        return None
    match = re.search(r"PMC(\d{5,9})", url)
    if match:
        return f"PMC{match.group(1)}"
    return None


# =============================================================================
# GLOBI DATA LOADING
# =============================================================================

@dataclass
class GlobiRecord:
    """A GloBI interaction record with article reference."""
    source_taxon: str
    target_taxon: str
    interaction_type: str
    category: str
    doi: Optional[str]
    pmcid: Optional[str]


def load_globi_interactions(
    globi_path: Path,
    max_rows: Optional[int] = None,
) -> List[GlobiRecord]:
    """Load and filter GloBI interactions with DOI/PMC references.

    Args:
        globi_path: Path to interactions.tsv.gz
        max_rows: Limit rows for dry-run mode

    Returns:
        List of GlobiRecord with deduplicated interactions
    """
    logger.info(f"Loading GloBI interactions from {globi_path}...")

    use_cols = ["sourceTaxonName", "targetTaxonName", "interactionTypeName",
                "referenceDoi", "referenceUrl"]

    records = []
    seen = set()
    chunk_size = 100_000

    try:
        reader = pd.read_csv(
            globi_path,
            sep="\t",
            usecols=use_cols,
            chunksize=chunk_size,
            dtype=str,
            on_bad_lines="skip",
            low_memory=False,
        )
    except Exception as e:
        logger.error(f"Failed to open GloBI file: {e}")
        return []

    rows_processed = 0
    for chunk in reader:
        for _, row in chunk.iterrows():
            interaction_type = str(row.get("interactionTypeName", ""))
            if interaction_type not in TARGET_INTERACTION_TYPES:
                continue

            source = str(row.get("sourceTaxonName", "")).strip()
            target = str(row.get("targetTaxonName", "")).strip()
            doi = row.get("referenceDoi")
            ref_url = str(row.get("referenceUrl", ""))

            # Need at least a DOI or PMC URL
            doi = str(doi).strip() if pd.notna(doi) and doi else None
            pmcid = extract_pmcid_from_url(ref_url)

            if not doi and not pmcid:
                continue

            # Need valid species names (at least 2 parts for binomial)
            if not source or not target or len(source.split()) < 2 or len(target.split()) < 2:
                continue

            # Skip self-interactions
            if source.lower() == target.lower():
                continue

            # Deduplicate
            dedup_key = (source.lower(), target.lower(), doi or pmcid)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            category = INTERACTION_CATEGORY_MAP.get(interaction_type, "general")
            records.append(GlobiRecord(
                source_taxon=source,
                target_taxon=target,
                interaction_type=interaction_type,
                category=category,
                doi=doi,
                pmcid=pmcid,
            ))

        rows_processed += len(chunk)
        if max_rows and rows_processed >= max_rows:
            break

        if rows_processed % 1_000_000 == 0:
            logger.info(f"  Processed {rows_processed:,} rows, found {len(records):,} unique records")

    logger.info(f"GloBI loading done: {len(records):,} unique interaction-paper pairs from {rows_processed:,} rows")

    # Log category distribution
    cat_counts = Counter(r.category for r in records)
    for cat, count in cat_counts.most_common():
        logger.info(f"  {cat}: {count:,}")

    return records


# =============================================================================
# ARTICLE FETCHING
# =============================================================================

class PMCFetcher:
    """Fetches full text from PMC with caching."""

    def __init__(self, cache_dir: Path, api_key: Optional[str] = None):
        self.cache = ArticleCache(str(cache_dir))
        self.api_key = api_key
        self.last_request_time = 0.0
        self.stats = {"cache_hits": 0, "fetched": 0, "failures": 0}

    def _rate_limit(self) -> None:
        elapsed = time.time() - self.last_request_time
        delay = 0.15 if self.api_key else RATE_LIMIT_DELAY
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self.last_request_time = time.time()

    def fetch_fulltext(self, pmcid: str) -> Optional[str]:
        """Fetch full text for a PMCID, using cache when available.

        Tries Europe PMC first, falls back to NCBI E-utilities.
        Returns clean text or None.
        """
        cache_key = f"pmc_fulltext:{pmcid}"

        # Check cache
        cached = self.cache.get(cache_key)
        if cached and cached.full_text:
            self.stats["cache_hits"] += 1
            return cached.full_text

        # Try Europe PMC
        text = self._fetch_europe_pmc(pmcid)
        if not text:
            text = self._fetch_ncbi_efetch(pmcid)

        if text:
            self.stats["fetched"] += 1
            # Cache the result
            from data.article_fetcher import ArticleText
            article = ArticleText(
                pmid=None, doi=None, title=None, abstract=None,
                full_text=text, source="pmc", fetch_time=time.time(),
            )
            self.cache.set(cache_key, article)
        else:
            self.stats["failures"] += 1

        return text

    def _fetch_europe_pmc(self, pmcid: str) -> Optional[str]:
        """Fetch full text from Europe PMC XML API."""
        self._rate_limit()
        url = f"{EUROPE_PMC_API}/{pmcid}/fullTextXML"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                return None
            return self._parse_pmc_xml(resp.content)
        except (requests.RequestException, ET.ParseError):
            return None

    def _fetch_ncbi_efetch(self, pmcid: str) -> Optional[str]:
        """Fetch full text from NCBI E-utilities (fallback)."""
        self._rate_limit()
        # Strip "PMC" prefix for numeric ID
        numeric_id = pmcid.replace("PMC", "")
        params = {"db": "pmc", "id": numeric_id, "rettype": "xml"}
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            resp = requests.get(NCBI_EFETCH, params=params, timeout=15)
            if resp.status_code != 200:
                return None
            return self._parse_pmc_xml(resp.content)
        except (requests.RequestException, ET.ParseError):
            return None

    def _parse_pmc_xml(self, xml_content: bytes) -> Optional[str]:
        """Extract clean body text from PMC JATS XML."""
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError:
            return None

        texts = []

        # Extract abstract
        for abstract in root.iter("abstract"):
            texts.append(self._extract_text(abstract))

        # Extract body paragraphs
        for body in root.iter("body"):
            for p in body.iter("p"):
                texts.append(self._extract_text(p))

        # Also get section titles for context
        for sec in root.iter("sec"):
            title_elem = sec.find("title")
            if title_elem is not None:
                title_text = self._extract_text(title_elem).strip()
                if title_text:
                    texts.append(title_text + ".")

        clean_text = " ".join(t.strip() for t in texts if t and t.strip())

        # Remove inline citations like [1,2] or (Smith et al., 2020)
        clean_text = re.sub(r"\[\d+(?:[,\-]\d+)*\]", "", clean_text)
        clean_text = re.sub(r"\s+", " ", clean_text)

        return clean_text if len(clean_text) > 100 else None

    @staticmethod
    def _extract_text(elem: ET.Element) -> str:
        """Recursively extract all text from an XML element."""
        parts = []
        if elem.text:
            parts.append(elem.text)
        for child in elem:
            # Skip reference elements
            if child.tag in ("xref", "ext-link", "uri"):
                if child.tail:
                    parts.append(child.tail)
                continue
            parts.append(PMCFetcher._extract_text(child))
            if child.tail:
                parts.append(child.tail)
        return " ".join(parts)


# =============================================================================
# SENTENCE EXTRACTION
# =============================================================================

@dataclass
class ExtractedSentence:
    """A sentence extracted from a PMC article."""
    text: str
    label: int
    source_species: str
    target_species: str
    interaction_type: str
    category: str
    pmcid: str
    source_match: str = ""  # What name form matched for source
    target_match: str = ""  # What name form matched for target
    match_type: str = ""    # e.g., "binomial+binomial", "common+binomial"


def is_good_sentence(text: str) -> bool:
    """Check if a sentence passes quality filters."""
    if not text or len(text) < MIN_SENT_LEN or len(text) > MAX_SENT_LEN:
        return False

    # Must have enough words
    words = text.split()
    if len(words) < 8:
        return False

    # Must have enough alphabetic characters
    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars < 30:
        return False

    # Reject figure/table captions
    if re.match(r"^(Fig(ure)?|Table|Supplementary)\s", text, re.IGNORECASE):
        return False

    # Reject reference-heavy text (>30% digits/brackets)
    special = sum(1 for c in text if c in "[]()0123456789")
    if special / len(text) > 0.3:
        return False

    # Reject species lists (>3 italicized binomials and few verbs)
    binomial_count = len(re.findall(r"[A-Z][a-z]+\s[a-z]+", text))
    if binomial_count > 3 and len(words) < 20:
        return False

    return True


def extract_sentences_from_article(
    article_text: str,
    record: GlobiRecord,
    source_variants: Set[str],
    target_variants: Set[str],
) -> Tuple[List[ExtractedSentence], List[ExtractedSentence]]:
    """Extract positive and negative sentences from an article.

    Positive: sentence contains BOTH source and target species (any name form).
    Negative: from same article, tiered quality.

    Args:
        article_text: Full article text
        record: GloBI interaction record
        source_variants: All name forms for source species
        target_variants: All name forms for target species

    Returns:
        (positives, negatives)
    """
    sentences = split_sentences(article_text)
    positives = []
    negatives_tier0 = []  # Both target species present BUT no interaction signal → hard negative
    negatives_tier1 = []  # Two species, not the pair
    negatives_tier2 = []  # One species only
    negatives_tier3 = []  # No species, scientific prose

    # Filter ambiguous common names
    source_safe = _filter_ambiguous(source_variants, target_variants)
    target_safe = _filter_ambiguous(target_variants, source_variants)

    for sent in sentences:
        if not is_good_sentence(sent):
            continue

        # Check for source and target species matches
        source_match = find_match_in_sentence(sent, source_safe)
        target_match = find_match_in_sentence(sent, target_safe)

        has_source = source_match is not None
        has_target = target_match is not None

        if has_source and has_target:
            # Both species present — check for overlapping matches first
            if source_match and target_match:
                s_start, s_end = source_match[1], source_match[2]
                t_start, t_end = target_match[1], target_match[2]
                if s_start < t_end and t_start < s_end:
                    continue  # Overlapping matches, skip

            # Check whether the sentence contains any biotic interaction signal word.
            sent_lower = sent.lower()
            has_signal = any(sig in sent_lower for sig in BIOTIC_INTERACTION_SIGNALS)

            match_type = _classify_match(source_match[0], target_match[0],
                                          record.source_taxon, record.target_taxon)

            if has_signal:
                # Confirmed positive: both species + interaction language
                positives.append(ExtractedSentence(
                    text=sent.strip(),
                    label=1,
                    source_species=record.source_taxon,
                    target_species=record.target_taxon,
                    interaction_type=record.interaction_type,
                    category=record.category,
                    pmcid=record.pmcid or "",
                    source_match=source_match[0] if source_match else "",
                    target_match=target_match[0] if target_match else "",
                    match_type=match_type,
                ))
            else:
                # Close false positive → hard negative tier 0.
                # Both species mentioned but no interaction signal: co-occurrence
                # without interaction context (lists, habitat descriptions, etc.)
                negatives_tier0.append(ExtractedSentence(
                    text=sent.strip(),
                    label=0,
                    source_species=record.source_taxon,
                    target_species=record.target_taxon,
                    interaction_type=record.interaction_type,
                    category=record.category,
                    pmcid=record.pmcid or "",
                    match_type="both_species_no_signal",
                ))
        elif has_source or has_target:
            # One species present — check for interaction signal.
            # "single_species_with_signal": has interaction language but partner unnamed
            # → harder negative than plain single-species mention.
            sent_lower = sent.lower()
            has_signal = any(sig in sent_lower for sig in BIOTIC_INTERACTION_SIGNALS)
            if has_signal:
                negatives_tier0.append(ExtractedSentence(
                    text=sent.strip(),
                    label=0,
                    source_species=record.source_taxon,
                    target_species=record.target_taxon,
                    interaction_type=record.interaction_type,
                    category=record.category,
                    pmcid=record.pmcid or "",
                    match_type="single_species_with_signal",
                ))
            else:
                negatives_tier2.append(ExtractedSentence(
                    text=sent.strip(),
                    label=0,
                    source_species=record.source_taxon,
                    target_species=record.target_taxon,
                    interaction_type=record.interaction_type,
                    category=record.category,
                    pmcid=record.pmcid or "",
                    match_type="single_species",
                ))
        else:
            # Check if it mentions any species at all (tier 1 or 3)
            has_any_species = bool(re.search(r"[A-Z][a-z]+\s[a-z]{3,}", sent))
            if has_any_species:
                negatives_tier1.append(ExtractedSentence(
                    text=sent.strip(),
                    label=0,
                    source_species=record.source_taxon,
                    target_species=record.target_taxon,
                    interaction_type=record.interaction_type,
                    category=record.category,
                    pmcid=record.pmcid or "",
                    match_type="other_species",
                ))
            elif len(sent.split()) >= 12:  # Only keep substantial sentences
                negatives_tier3.append(ExtractedSentence(
                    text=sent.strip(),
                    label=0,
                    source_species=record.source_taxon,
                    target_species=record.target_taxon,
                    interaction_type=record.interaction_type,
                    category=record.category,
                    pmcid=record.pmcid or "",
                    match_type="no_species",
                ))

    # Combine negatives: tier 0 (both species, no signal) first — hardest negatives
    remaining = MAX_NEG_PER_ARTICLE
    negatives = []
    for tier in [negatives_tier0, negatives_tier1, negatives_tier2, negatives_tier3]:
        take = tier[:remaining]
        negatives.extend(take)
        remaining -= len(take)
        if remaining <= 0:
            break

    return positives[:MAX_SENTS_PER_ARTICLE], negatives


def _filter_ambiguous(
    variants: Set[str],
    other_variants: Set[str],
) -> Set[str]:
    """Remove ambiguous common names unless the other species has a specific match."""
    filtered = set()
    for v in variants:
        if v.lower() in AMBIGUOUS_COMMON_NAMES:
            continue  # Skip ambiguous names
        filtered.add(v)

    # If nothing left after filtering, add back all variants
    if not filtered:
        return variants

    return filtered


def _classify_match(
    source_matched: str,
    target_matched: str,
    source_sci: str,
    target_sci: str,
) -> str:
    """Classify the type of name match (e.g., binomial+common)."""
    def _match_type(matched: str, scientific: str) -> str:
        if not matched:
            return "none"
        parts = scientific.split()
        if len(parts) >= 2 and matched.lower() == f"{parts[0].lower()} {parts[1].lower()}":
            return "binomial"
        if len(parts) >= 2 and matched.lower() == f"{parts[0][0].lower()}. {parts[1].lower()}":
            return "abbreviated"
        if len(parts) >= 1 and matched.lower() == parts[0].lower():
            return "genus"
        return "common"

    st = _match_type(source_matched, source_sci)
    tt = _match_type(target_matched, target_sci)
    return f"{st}+{tt}"


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(args: argparse.Namespace) -> None:
    """Run the full harvesting pipeline."""
    logger.info("=" * 70)
    logger.info("GloBI-PMC Sentence Harvester")
    logger.info("=" * 70)

    # Step 1: Load GloBI interactions
    max_rows = 500_000 if args.dry_run else None
    records = load_globi_interactions(GLOBI_PATH, max_rows=max_rows)
    if not records:
        logger.error("No GloBI records found!")
        return

    # Step 2: Batch DOI → PMCID conversion
    # First, collect all DOIs that don't already have PMCIDs
    dois_needing_pmcid = [r.doi for r in records if r.doi and not r.pmcid]
    doi_to_pmcid = batch_doi_to_pmcid(dois_needing_pmcid, api_key=args.ncbi_api_key)

    # Update records with resolved PMCIDs
    records_with_pmc = []
    for r in records:
        if not r.pmcid and r.doi:
            r.pmcid = doi_to_pmcid.get(r.doi)
        if r.pmcid:
            records_with_pmc.append(r)

    logger.info(f"Records with PMC full text: {len(records_with_pmc):,}")

    if not records_with_pmc:
        logger.error("No records with PMC IDs found!")
        return

    # Limit articles for processing
    if len(records_with_pmc) > args.max_articles:
        # Sample with category diversity
        sampled = _stratified_sample(records_with_pmc, args.max_articles)
        records_with_pmc = sampled

    # Step 3: Resolve common names
    unique_species = set()
    for r in records_with_pmc:
        unique_species.add(r.source_taxon)
        unique_species.add(r.target_taxon)

    name_resolver = SpeciesNameResolver(
        cache_path=Path(args.name_cache) if args.name_cache else NAME_CACHE_DEFAULT,
        ncbi_api_key=args.ncbi_api_key,
    )

    if not args.no_common_names:
        name_resolver.batch_resolve(list(unique_species))

    # Step 4 & 5: Fetch articles and extract sentences
    fetcher = PMCFetcher(
        cache_dir=Path(args.cache_dir) if args.cache_dir else CACHE_DIR_DEFAULT,
        api_key=args.ncbi_api_key,
    )

    all_positives: List[ExtractedSentence] = []
    all_negatives: List[ExtractedSentence] = []
    category_pos_counts: Dict[str, int] = defaultdict(int)
    seen_texts = set()
    articles_processed = 0
    articles_with_hits = 0

    # Group records by PMCID to avoid fetching same article multiple times
    pmcid_to_records: Dict[str, List[GlobiRecord]] = defaultdict(list)
    for r in records_with_pmc:
        pmcid_to_records[r.pmcid].append(r)

    logger.info(f"Processing {len(pmcid_to_records)} unique PMC articles...")

    # Sort article queue by category priority: ecological first, pathogen last.
    # Each PMCID gets its best (highest-priority) category as sort key.
    def _pmcid_priority(item: Tuple[str, list]) -> int:
        cats = {r.category for r in item[1]}
        best = min(
            (CATEGORY_PRIORITY.index(c) if c in CATEGORY_PRIORITY else len(CATEGORY_PRIORITY)
             for c in cats),
            default=len(CATEGORY_PRIORITY),
        )
        return best

    sorted_pmcids = sorted(pmcid_to_records.items(), key=_pmcid_priority)
    logger.info("Article queue sorted by ecological priority (predation first, pathogen last).")

    fetch_attempts = 0
    fetch_failures = 0
    total_articles = len(pmcid_to_records)

    for pmcid, article_records in sorted_pmcids:
        # Check if we have enough
        total_pos = sum(category_pos_counts.values())
        if total_pos >= args.max_positives:
            logger.info(f"Reached target of {args.max_positives} positives, stopping.")
            break

        # Fetch article
        fetch_attempts += 1
        if fetch_attempts % 10 == 0:
            logger.info(
                f"  Article {fetch_attempts}/{total_articles}: "
                f"{articles_processed} fetched, {fetch_failures} failed, "
                f"{sum(category_pos_counts.values())} positives so far"
            )

        article_text = fetcher.fetch_fulltext(pmcid)
        if not article_text:
            fetch_failures += 1
            continue

        articles_processed += 1
        article_had_hits = False

        for record in article_records:
            # Check category cap (per-category dict)
            cat_cap = MAX_PER_CATEGORY.get(record.category, DEFAULT_CATEGORY_CAP)
            if category_pos_counts[record.category] >= cat_cap:
                continue

            # Get name variants
            if args.no_common_names:
                source_variants = generate_name_variants(record.source_taxon)
                target_variants = generate_name_variants(record.target_taxon)
            else:
                source_variants = name_resolver.get_all_name_forms(record.source_taxon)
                target_variants = name_resolver.get_all_name_forms(record.target_taxon)

            # Extract sentences
            positives, negatives = extract_sentences_from_article(
                article_text, record, source_variants, target_variants,
            )

            # Deduplicate
            for sent in positives:
                text_hash = hashlib.md5(sent.text.lower().strip().encode()).hexdigest()
                if text_hash not in seen_texts:
                    seen_texts.add(text_hash)
                    all_positives.append(sent)
                    category_pos_counts[record.category] += 1
                    article_had_hits = True

            for sent in negatives:
                text_hash = hashlib.md5(sent.text.lower().strip().encode()).hexdigest()
                if text_hash not in seen_texts:
                    seen_texts.add(text_hash)
                    all_negatives.append(sent)

        if article_had_hits:
            articles_with_hits += 1

        if articles_processed % 100 == 0:
            total_pos = sum(category_pos_counts.values())
            logger.info(
                f"  Articles: {articles_processed}/{len(pmcid_to_records)} | "
                f"Positives: {total_pos} | Negatives: {len(all_negatives)} | "
                f"Hit rate: {articles_with_hits}/{articles_processed}"
            )

    # Balance negatives to match positives
    if len(all_negatives) > len(all_positives):
        import random
        random.seed(42)
        all_negatives = random.sample(all_negatives, len(all_positives))

    # Step 7: Save output
    output_path = Path(args.output) if args.output else OUTPUT_DEFAULT
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
            "source": f"globi_pmc_{sent.match_type}" if sent.label == 1 else f"globi_pmc_{sent.match_type}",
            "match_type": sent.match_type,
        })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(output_path, index=False)

    # Print summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("HARVEST SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Articles processed: {articles_processed}")
    logger.info(f"Articles with hits: {articles_with_hits}")
    logger.info(f"Positives: {len(all_positives)}")
    logger.info(f"Negatives: {len(all_negatives)}")
    logger.info(f"Total sentences: {len(df)}")
    logger.info(f"Unique PMCIDs: {df['pmcid'].nunique()}")
    logger.info(f"Output: {output_path}")
    logger.info("")
    logger.info("Category distribution (positives):")
    for cat, count in sorted(category_pos_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {cat}: {count}")
    logger.info("")
    logger.info("Match type distribution:")
    if len(all_positives) > 0:
        match_types = Counter(s.match_type for s in all_positives)
        for mt, count in match_types.most_common():
            logger.info(f"  {mt}: {count}")
    logger.info("")
    logger.info(f"Fetcher stats: {fetcher.stats}")
    logger.info(f"Name resolver stats: {name_resolver.stats}")


def _stratified_sample(records: List[GlobiRecord], max_records: int) -> List[GlobiRecord]:
    """Sample records with category diversity."""
    import random
    random.seed(42)

    by_category: Dict[str, List[GlobiRecord]] = defaultdict(list)
    for r in records:
        by_category[r.category].append(r)

    per_cat = max(max_records // len(by_category), 100)
    sampled = []
    for cat, recs in by_category.items():
        random.shuffle(recs)
        sampled.extend(recs[:per_cat])

    random.shuffle(sampled)
    return sampled[:max_records]


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Harvest real training sentences from GloBI-cited PMC articles"
    )
    parser.add_argument("--max-positives", type=int, default=5000,
                        help="Target number of positive sentences")
    parser.add_argument("--max-articles", type=int, default=10000,
                        help="Maximum articles to process")
    parser.add_argument("--output", default=None,
                        help="Output CSV path")
    parser.add_argument("--cache-dir", default=None,
                        help="Article cache directory")
    parser.add_argument("--name-cache", default=None,
                        help="Species name cache file")
    parser.add_argument("--ncbi-api-key", default=None,
                        help="NCBI API key (env: NCBI_API_KEY)")
    parser.add_argument("--no-common-names", action="store_true",
                        help="Skip common name resolution (faster, less recall)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process only a small subset for testing")
    parser.add_argument("--interaction-types", nargs="+", default=None,
                        help="Limit to specific interaction categories")

    args = parser.parse_args()

    # Get API key from env if not provided
    if not args.ncbi_api_key:
        args.ncbi_api_key = os.environ.get("NCBI_API_KEY")

    # Dry run overrides
    if args.dry_run:
        args.max_positives = min(args.max_positives, 50)
        args.max_articles = min(args.max_articles, 30)
        logger.info("DRY RUN MODE: limited to 50 positives, 30 articles")

    run_pipeline(args)


if __name__ == "__main__":
    main()
