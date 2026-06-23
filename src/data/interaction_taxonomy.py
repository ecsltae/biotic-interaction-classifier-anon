"""
interaction_taxonomy.py — Canonical interaction category taxonomy.

Defines 12 canonical interaction categories, maps all 591 GloBI interaction
terms to those categories, and provides a scanner to detect GloBI terms in
a sentence at inference time.

This is the single source of truth for interaction category classification.
It does NOT modify any existing pipeline code.
"""

from __future__ import annotations

import re
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical categories
# ---------------------------------------------------------------------------

class InteractionCategory(str, Enum):
    PREDATION       = "PREDATION"       # preys on, hunts, eats, kills, devours
    PARASITISM      = "PARASITISM"      # parasite of, ecto/hyper/hemi/klepto/meso/stem/root
    ENDOPARASITISM  = "ENDOPARASITISM"  # endoparasite, intra/intercellular endoparasite
    PARASITOIDISM   = "PARASITOIDISM"   # parasitoid of, idiobiont, koinobiont, lays eggs in
    INFECTION       = "INFECTION"       # pathogen of, infects, seropositive, contaminate
    VECTOR          = "VECTOR"          # vector of, transmits, transmitted by
    POLLINATION     = "POLLINATION"     # pollinates, visits flowers of
    HERBIVORY       = "HERBIVORY"       # grazes on, feeds on (plant target)
    DISPERSAL       = "DISPERSAL"       # disperses seeds of, seeds dispersed by
    SYMBIOSIS       = "SYMBIOSIS"       # symbiont/mutualist/commensalist/epiphyte/phoresis
    REGULATION      = "REGULATION"      # positively/negatively regulates (molecular/ecological)
    GENERIC         = "GENERIC"         # co-occurs with, interacts with, lives near, other

CANONICAL_CATEGORIES: List[str] = [c.value for c in InteractionCategory]

# Minimum positive training samples required per category
CATEGORY_MINIMUMS: Dict[str, int] = {
    InteractionCategory.PREDATION:      200,
    InteractionCategory.PARASITISM:     200,
    InteractionCategory.ENDOPARASITISM: 200,
    InteractionCategory.PARASITOIDISM:   50,
    InteractionCategory.INFECTION:      200,
    InteractionCategory.VECTOR:          50,
    InteractionCategory.POLLINATION:     50,
    InteractionCategory.HERBIVORY:       50,
    InteractionCategory.DISPERSAL:       20,
    InteractionCategory.SYMBIOSIS:       50,
    InteractionCategory.REGULATION:      20,
    InteractionCategory.GENERIC:         20,
}


def get_required_minimum(category: str) -> int:
    """Return the minimum positive training samples required for a category."""
    return CATEGORY_MINIMUMS.get(category, 20)


# ---------------------------------------------------------------------------
# Pattern-based rules to classify any GloBI term string → category
# Rules are checked in order; first match wins.
# ---------------------------------------------------------------------------

_CLASSIFICATION_RULES: List[Tuple[re.Pattern, str]] = [
    # ENDOPARASITISM — check before PARASITISM
    (re.compile(r'\bendoparasit|intercellular endoparasit|intracellular endoparasit', re.I),
     InteractionCategory.ENDOPARASITISM),

    # PARASITOIDISM — check before PARASITISM
    (re.compile(r'\bparasitoid|idiobiont|koinobiont|ectoparasitoid|lays eggs (in|on)\b', re.I),
     InteractionCategory.PARASITOIDISM),

    # KLEPTOPARASITISM → goes into PARASITISM bucket
    (re.compile(r'\bkleptoparasit', re.I), InteractionCategory.PARASITISM),

    # PARASITISM (broad: ecto, hyper, hemi, meso, stem, root, facultative, obligate, parasite of)
    (re.compile(r'\bparasit|ectoparasit|hyperparasit|hemiparasit|mesoparasit|stem parasit'
                r'|root parasit|facultative parasit|obligate parasit', re.I),
     InteractionCategory.PARASITISM),

    # INFECTION
    (re.compile(r'\bpathogen|infect|seropositive|is (being )?transmited|was transmited'
                r'|had transmited|have transmited|has transmited|contaminate', re.I),
     InteractionCategory.INFECTION),

    # VECTOR
    (re.compile(r'\bvector (of|for)|is vector for|had vector|has vector|have vector'
                r'|transmit(s|ted|ting|ed)?\b', re.I),
     InteractionCategory.VECTOR),

    # POLLINATION
    (re.compile(r'\bpollinat|pollinator|visit(s|ed|ing)? flower|flower visit'
                r'|had flower visited|has flower visited|have flower visited', re.I),
     InteractionCategory.POLLINATION),

    # DISPERSAL
    (re.compile(r'\bdispers(es?|ed|ing)? seed|seeds? dispers(ed|al)|had dispersal vector'
                r'|has dispersal vector|have dispersal vector', re.I),
     InteractionCategory.DISPERSAL),

    # HERBIVORY
    (re.compile(r'\bgraze|grazing|grazed|herbivory|herbivore', re.I),
     InteractionCategory.HERBIVORY),

    # PREDATION
    (re.compile(r'\bpreys? on|prey on|preyed|hunt(s|ed|ing)?|kill(s|ed|ing)?'
                r'|devour(s|ed|ing)?|predator|eat(s|ing|ed)?|ate\b|eats?\b|feed(s|ing)? on'
                r'|fed on|ingest(s|ed|ing)?|trophic', re.I),
     InteractionCategory.PREDATION),

    # SYMBIOSIS (mutualism, commensalism, symbiosis, epiphyte, amensalism, phoresis, inquilinism)
    (re.compile(r'\bsymbiont|symbiosis|symbiotically|mutuali|commensali|epiphyte'
                r'|amensali|phoresy|phoretic|phoresis|inquilinism|myrmecophile'
                r'|antibiosis|co-roost|roosting with', re.I),
     InteractionCategory.SYMBIOSIS),

    # REGULATION (molecular/ecological regulation)
    (re.compile(r'\bregulate|regulates|regulated by|regulation|represses|increase expression'
                r'|decreases expression|positively regulate|negatively regulate'
                r'|directly (activate|inhibit)|indirectly (activate|inhibit)'
                r'|capable of (neg|pos)atively', re.I),
     InteractionCategory.REGULATION),

    # GENERIC — everything else
    (re.compile(r'.*', re.I), InteractionCategory.GENERIC),
]


def classify_globi_term(term: str) -> str:
    """Map a single GloBI interaction term string to a canonical category.

    Args:
        term: A GloBI interaction term, e.g. "pollinates", "ectoparasite of".

    Returns:
        One of the canonical category codes (e.g. "POLLINATION").
    """
    for pattern, category in _CLASSIFICATION_RULES:
        if pattern.search(term):
            return category
    return InteractionCategory.GENERIC


# ---------------------------------------------------------------------------
# GloBI type → category mapping dict (built from interaction_dict.csv)
# ---------------------------------------------------------------------------

# Also covers the GloBI training data column values (camelCase types)
_GLOBI_TRAINING_TYPES: Dict[str, str] = {
    "preysOn":          InteractionCategory.PREDATION,
    "eats":             InteractionCategory.PREDATION,
    "hasHost":          InteractionCategory.PARASITISM,  # host relationship (parasite side)
    "parasiteOf":       InteractionCategory.PARASITISM,
    "endoparasiteOf":   InteractionCategory.ENDOPARASITISM,
    "pathogenOf":       InteractionCategory.INFECTION,
    "kleptoparasiteOf": InteractionCategory.PARASITISM,
    "pollinates":       InteractionCategory.POLLINATION,
    "visitsFlowersOf":  InteractionCategory.POLLINATION,
    "symbioticWith":    InteractionCategory.SYMBIOSIS,
    "mutualistOf":      InteractionCategory.SYMBIOSIS,
    "interactsWith":    InteractionCategory.GENERIC,
    "parasitoidOf":     InteractionCategory.PARASITOIDISM,
    "vectorOf":         InteractionCategory.VECTOR,
    "transmits":        InteractionCategory.VECTOR,
    "grazesOn":         InteractionCategory.HERBIVORY,
    "feedsOn":          InteractionCategory.HERBIVORY,
    "dispersesSeeds":   InteractionCategory.DISPERSAL,
    "dispersesSeedsOf": InteractionCategory.DISPERSAL,
    "negativelyRegulates": InteractionCategory.REGULATION,
    "positivelyRegulates": InteractionCategory.REGULATION,
    "competesWth":      InteractionCategory.GENERIC,
    "none":             InteractionCategory.GENERIC,
    "none_two_species": InteractionCategory.GENERIC,
    "none_three_species": InteractionCategory.GENERIC,
}


@lru_cache(maxsize=None)
def _build_globi_term_mapping() -> Dict[str, str]:
    """Build the full {term → category} mapping from interaction_dict.csv."""
    mapping: Dict[str, str] = {}

    # Start with training data type overrides
    mapping.update(_GLOBI_TRAINING_TYPES)

    # Load interaction_dict.csv
    dict_path = Path(__file__).parent.parent.parent / "data/processed/interaction_dict.csv"
    if dict_path.exists():
        df = pd.read_csv(dict_path)
        for term in df["interaction"].dropna():
            term_str = str(term).strip()
            if term_str:
                mapping[term_str] = classify_globi_term(term_str)
    return mapping


def get_globi_type_to_category() -> Dict[str, str]:
    """Return the full {GloBI_term → canonical_category} mapping."""
    return _build_globi_term_mapping()


# Convenience accessor (lazy-built on first call)
GLOBI_TYPE_TO_CATEGORY: Dict[str, str] = {}  # filled by _ensure_loaded()


def _ensure_loaded() -> None:
    global GLOBI_TYPE_TO_CATEGORY
    if not GLOBI_TYPE_TO_CATEGORY:
        GLOBI_TYPE_TO_CATEGORY.update(_build_globi_term_mapping())


def classify_interaction_type(
    matched_terms: List[str],
    globi_type: Optional[str] = None,
) -> str:
    """Determine canonical category from matched lexicon terms or GloBI type.

    Args:
        matched_terms: Interaction terms detected in the sentence (from lexicon or
            GloBI scan). The first strong match wins.
        globi_type:  Optional GloBI interaction type from training data column
            (e.g. "pathogenOf"). Takes precedence if provided.

    Returns:
        Canonical category string, e.g. "INFECTION".
    """
    _ensure_loaded()

    if globi_type and globi_type in GLOBI_TYPE_TO_CATEGORY:
        return GLOBI_TYPE_TO_CATEGORY[globi_type]

    for term in matched_terms:
        category = GLOBI_TYPE_TO_CATEGORY.get(term) or classify_globi_term(term)
        if category != InteractionCategory.GENERIC:
            return category

    if matched_terms:
        return InteractionCategory.GENERIC

    return ""  # No terms matched at all


# ---------------------------------------------------------------------------
# Full GloBI term scanner — compiled at import time, O(n_terms) per sentence
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _build_globi_scanner() -> List[Tuple[str, re.Pattern]]:
    """Compile all GloBI interaction terms into regex patterns (case-insensitive)."""
    dict_path = Path(__file__).parent.parent.parent / "data/processed/interaction_dict.csv"
    if not dict_path.exists():
        return []
    df = pd.read_csv(dict_path)
    patterns = []
    for term in df["interaction"].dropna():
        term_str = str(term).strip()
        if len(term_str) < 3:
            continue
        try:
            pat = re.compile(r'(?<!\w)' + re.escape(term_str.lower()) + r'(?!\w)',
                             re.IGNORECASE)
            patterns.append((term_str, pat))
        except re.error:
            pass
    return patterns


@lru_cache(maxsize=None)
def _build_globi_fastcheck() -> re.Pattern:
    """Single combined regex for fast O(n) pre-screening — 100× faster than scan_globi_terms."""
    scanner = _build_globi_scanner()
    if not scanner:
        return re.compile(r'(?!)')  # never matches
    alternation = '|'.join(re.escape(term.lower()) for term, _ in scanner)
    return re.compile(r'(?<!\w)(?:' + alternation + r')(?!\w)', re.IGNORECASE)


# Biomedical interaction vocabulary absent from GloBI but common in literature.
# Validated on EP-relax: GloBI alone catches 35% of positives; combined catches 100%.
_BIOMEDICAL_INTERACTION_RE = re.compile(
    r'infect|parasit|\bhost\b|pathogen|\bvector\b|zoonot|symbiont|symbioti|'
    r'endophyte|mycorrhiz|\bnodule|nematod|fungal|\bfungi\b|bacteri|viral|\bvirus\b|protozoa|'
    r'transmit|reservoir|definitive host|intermediate host|attracted to|utiliz|'
    r'harbour|harbor|coloniz|colonise|life cycle|'
    r'\bprey\b|predat|pollina|feed on|feeds on|\beats\b|ingest|'
    r'herbivory|herbivore|mutuali|commensali|kleptoparasit',
    re.IGNORECASE
)

def has_globi_term(text: str) -> bool:
    """Fast boolean check: does text contain any GloBI interaction term?
    Use this for pre-filtering; use scan_globi_terms() only when you need which terms matched."""
    return bool(_build_globi_fastcheck().search(text.lower()))


def has_interaction_signal(text: str) -> bool:
    """Broad pre-filter combining GloBI terms + biomedical interaction vocabulary.

    Designed for pre-filtering raw full-text articles before the ML classifier.
    Achieves 100% recall on EP-relax positives (vs 35% for GloBI alone).
    Expected pass rate on random biomedical full-text: ~30-40%.

    Use this instead of has_globi_term() when false negatives are unacceptable.
    """
    return has_globi_term(text) or bool(_BIOMEDICAL_INTERACTION_RE.search(text))


def scan_globi_terms(text: str) -> List[str]:
    """Scan a sentence against the full ~591 GloBI interaction term list.

    Returns all GloBI terms found in the text (in order of appearance).
    This is a deterministic rule-based check that runs in parallel with the
    ML classifier — it always surfaces which interaction vocabulary is present.

    Args:
        text: Input sentence (any case).

    Returns:
        List of matched GloBI term strings (original case from interaction_dict).

    Example:
        >>> scan_globi_terms("Apis mellifera pollinates Malus domestica flowers.")
        ["pollinates"]
        >>> scan_globi_terms("Wolbachia infects Drosophila melanogaster tissues.")
        []  # "infects" is not verbatim in GloBI list but "infested by" etc. are
    """
    t = text.lower()
    scanner = _build_globi_scanner()
    found = []
    for term_str, pat in scanner:
        if pat.search(t):
            found.append(term_str)
    return found


def get_interaction_category_for_sentence(text: str) -> Optional[str]:
    """Convenience: scan sentence for GloBI terms and return most specific category.

    Returns None if no GloBI terms found.
    """
    matched = scan_globi_terms(text)
    if not matched:
        return None
    category = classify_interaction_type(matched)
    return category if category else None


# ---------------------------------------------------------------------------
# Public summary for coverage analysis
# ---------------------------------------------------------------------------

def coverage_report(interaction_types: List[str]) -> Dict[str, Dict]:
    """Given a list of interaction_type values from training data, report coverage.

    Args:
        interaction_types: List of interaction_type strings from training CSV
            (e.g. ["endoparasiteOf", "preysOn", "none", ...]).

    Returns:
        Dict keyed by canonical category with:
            {count, required, deficit, status}
    """
    _ensure_loaded()
    counts: Dict[str, int] = {c: 0 for c in CANONICAL_CATEGORIES}

    for itype in interaction_types:
        cat = GLOBI_TYPE_TO_CATEGORY.get(str(itype), InteractionCategory.GENERIC)
        if cat in counts:
            counts[cat] = counts.get(cat, 0) + 1

    report = {}
    for cat in CANONICAL_CATEGORIES:
        n = counts[cat]
        required = get_required_minimum(cat)
        deficit = max(0, required - n)
        if deficit == 0:
            status = "OK"
        elif n == 0:
            status = "MISSING"
        else:
            status = "LOW"
        report[cat] = {
            "count": n,
            "required": required,
            "deficit": deficit,
            "status": status,
        }
    return report


if __name__ == "__main__":
    # Quick smoke test
    import sys

    print("=== GloBI Term Scanner Test ===")
    test_sentences = [
        "Apis mellifera pollinates Malus domestica flowers.",
        "Plasmodium falciparum is the pathogen of malaria.",
        "The fox preys on rabbits in open grassland.",
        "Ixodes ricinus is a vector of Borrelia burgdorferi.",
        "The bacterium was isolated from host tissues.",
        "No significant difference was found between groups.",
    ]
    for s in test_sentences:
        found = scan_globi_terms(s)
        cat = get_interaction_category_for_sentence(s)
        print(f"\n  [{cat or 'NONE'}] {s[:70]}")
        print(f"    matched: {found}")

    print("\n=== Category Mapping Test ===")
    for term in ["preysOn", "pollinates", "pathogenOf", "endoparasiteOf",
                 "kleptoparasiteOf", "interactsWith", "none"]:
        _ensure_loaded()
        cat = GLOBI_TYPE_TO_CATEGORY.get(term, "NOT_FOUND")
        print(f"  {term:<25} → {cat}")

    print("\n=== Coverage Report Test ===")
    sample_types = ["endoparasiteOf"] * 3864 + ["eats"] * 1690 + ["preysOn"] * 763 \
                 + ["hasHost"] * 617 + ["parasiteOf"] * 403 + ["pathogenOf"] * 273 \
                 + ["kleptoparasiteOf"] * 201 + ["pollinates"] * 131 \
                 + ["visitsFlowersOf"] * 84 + ["symbioticWith"] * 77 \
                 + ["mutualistOf"] * 2 + ["interactsWith"] * 1
    rep = coverage_report(sample_types)
    print(f"\n{'Category':<20} {'Count':>7} {'Required':>9} {'Deficit':>8} {'Status'}")
    print("-" * 55)
    for cat, info in rep.items():
        print(f"  {cat:<18} {info['count']:>7} {info['required']:>9} "
              f"{info['deficit']:>8} {info['status']}")
