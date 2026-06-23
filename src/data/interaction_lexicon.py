"""
interaction_lexicon.py — Canonical biotic interaction term lexicon.

Single source of truth for interaction signal detection, consolidating:
  - domain_rules.json regex patterns
  - quality_filter.py INTERACTION_SPECIFICITY (high-specificity terms)
  - validator/interaction_validator.py STRONG/WEAK/NEGATIVE patterns
  - fetch_globi_pmc.py BIOTIC_INTERACTION_SIGNALS (vocabulary source)
  - tests/test_training_data.py Gate 4a inline patterns

Usage:
    from data.interaction_lexicon import score_sentence, count_species_mentions

    has_signal, strength, matched = score_sentence("Apis mellifera pollinates Malus domestica.")
    n_species = count_species_mentions("Apis mellifera pollinates Malus domestica.")
"""

import re
from typing import List, Tuple


# =============================================================================
# STRONG TERMS  (specificity >= 0.85)
# High-specificity interaction verbs and phrases. Each match is a strong signal
# that a biotic interaction is being described.
# Patterns are written for lowercased text.
# =============================================================================
STRONG_TERMS: List[str] = [
    # --- Parasitism (all inflected forms) ---
    r"\bparasiti[zs]e[sd]?\b",            # parasitize, parasitized, parasitises
    r"\bparasiti[zs]ing\b",               # parasitizing, parasitising
    r"\bparasitic\s+on\b",                # parasitic on
    r"\bparasite\s+of\b",                 # parasite of
    r"\bectoparasite(?:\s+of)?\b",        # ectoparasite, ectoparasite of
    r"\bendoparasite(?:\s+of)?\b",        # endoparasite, endoparasite of
    r"\bhyperparasite(?:\s+of)?\b",       # hyperparasite, hyperparasite of
    r"\bparasitoid(?:\s+of)?\b",          # parasitoid, parasitoid of
    r"\bnematode\s*[- ]?parasit\w+\b",   # nematode-parasitic fungus
    # --- Predation ---
    r"\bpreys?\s+(?:up)?on\b",            # preys on, prey on, preys upon
    r"\bpreyed\s+(?:up)?on\b",            # preyed on, preyed upon
    r"\bpredation\b",                     # predation (the act)
    r"\bpredator\s+of\b",                 # predator of
    # --- Pathogen / Infection (all forms) ---
    r"\binfect(?:s|ed|ing)\b",            # infects, infected, infecting
    r"\binfected\s+(?:by|with)\b",        # infected by / infected with
    r"\bpathogen\s+of\b",                 # pathogen of
    r"\bpathogenic\s+(?:to|for|in)\b",   # pathogenic to/for/in
    r"\bcausative\s+agent\s+of\b",        # causative agent of
    r"\bcause[sd]?\s+infection\b",        # caused infection
    # --- Vector / Transmission ---
    r"\bvector\s+of\b",                   # vector of
    r"\btransmit(?:s|ted|ting)\b",        # transmits, transmitted, transmitting
    r"\btransmitted\s+by\b",              # transmitted by
    r"\btransmission\s+of\b",             # transmission of [disease]
    r"\bwellestablished\s+vector\b",      # well-established vector (eval_100 style)
    # --- Pollination ---
    r"\bpollinat(?:es?|ed|ing)\b",        # pollinates, pollinated, pollinating
    r"\bpollinator\s+of\b",               # pollinator of
    r"\bvisits?\s+(?:the\s+)?flowers?\s+(?:of|from)\b",  # visits flowers of / visits the flowers of
    r"\bfloral\s+visitor\b",              # floral visitor
    # --- Seed dispersal ---
    r"\bdisperses?\s+seeds?\s+of\b",      # disperses seeds of
    r"\bseed\s+dispers(?:al|es?)\b",      # seed dispersal, seed disperses
    # --- Feeding (specific) ---
    r"\bfeeds?\s+on\b",                   # feeds on, feed on
    r"\bfed\s+(?:on|foliage)\b",          # fed on, fed foliage
    r"\bconsumes?\s+\w+",                 # consumes [organism]
    r"\bconsumed\s+by\b",                 # consumed by
    # --- Host relationship ---
    r"\bhost\s+of\b",                     # host of
    r"\bhost\s+to\b",                     # host to
    r"\bhost\s+for\b",                    # host for  (template: "serves as host for")
    r"\bhashost\b",                       # hasHost (GloBI term)
    r"\bhost\s+range\b",                  # host range (implies host–parasite relationship)
    r"\bserves?\s+as\s+(?:a\s+)?host\b",  # serves as host / serve as a host
    r"\buses?\s+\w+\s+as\s+(?:its\s+)?host\b",  # uses X as its host
    r"\bfound\s+on\s+host\b",             # found on host [species]
    # --- Symbiosis / Mutualism ---
    r"\bsymbiont\s+of\b",                 # symbiont of
    r"\bmutualist\s+of\b",               # mutualist of
    # --- Herbivory (specific) ---
    r"\bherbivore\s+of\b",               # herbivore of
    r"\bgrazes?\s+on\b",                  # grazes on, graze on
    r"\bfed\s+foliage\s+from\b",          # fed foliage from [plant]
    # --- Infesting / parasitizing (gerund/compound, extra forms) ---
    r"\binfest(?:ing|ed|ation)\b",        # infesting, infested, infestation
    # --- Obligate relationships ---
    r"\bobligate\s+(?:parasite|pathogen|symbiont|endoparasite)\b",  # obligate parasite
    # --- Endoparasite detection patterns (template-generated) ---
    r"\bin\s+(?:the\s+)?tissues?\s+of\b", # "in tissues of" / "in the tissues of"
    r"\bnecropsy\s+revealed\b",           # "necropsy revealed X in Y"
    r"\bdetected\s+in\s+(?:the\s+)?tissues?\s+of\b",  # "detected in tissues of"
    r"\bspecimens?\s+harboring\b",        # "specimens harboring [parasite]"
    r"\bharbo(?:r|uring)\w*\b",           # harbors, harboring (parasite inside host)
    r"\bgall\s+bladder\s+of\b",           # "gall bladder of [host]"
    r"\bisolated\s+from\s+(?:tissue|gall|cyst)\b",  # "isolated from tissue/gall/cyst of"
    # --- Food source / diet patterns (template-generated "eats") ---
    r"\bfood\s+source\s+for\b",           # "food source for"
    r"\bdiet\s+(?:of|includes?|contains?)\b",  # "diet of", "diet includes", "diet contains"
    r"\bgut\s+contents?\b",               # "gut contents" (predation indicator)
    r"\bcapturing\s+and\s+consuming\b",   # "capturing and consuming" (predation)
    r"\bpredatory\s+relationship\b",      # "predatory relationship between X and Y"
    r"\bfeeding\s+(?:behavior|on)\b",     # "feeding on X" / "feeding behavior of X on Y"
    # --- Causal agent / disease caused by (template-generated pathogen) ---
    r"\bcausal\s+agent\s+(?:of|for)?\b",  # "causal agent of/for" or just "causal agent"
    r"\b(?:disease|infection)\s+caused\s+by\b",  # "disease caused by", "infection caused by"
    r"\bcausative\s+agent\b",             # causative agent
]

# =============================================================================
# WEAK TERMS  (specificity 0.3-0.7)
# Moderate-specificity terms. May indicate interaction but could also appear
# in non-interaction contexts. Alone they reduce certainty; in combination
# or alongside strong terms they reinforce the signal.
# =============================================================================
WEAK_TERMS: List[str] = [
    # Feeding (generic)
    r"\beats?\b",                         # eats, eat
    r"\bate\b",                           # ate
    r"\beaten\b",                         # eaten
    r"\bhunts?\b",                        # hunts, hunt
    r"\bhunting\b",                       # hunting
    # Killing / aggression
    r"\bkills?\b",                        # kills, kill
    r"\bkilled\b",                        # killed
    r"\battacks?\b",                      # attacks, attack
    r"\battacked\b",                      # attacked
    # Herbivory (generic)
    r"\bherbivory\b",                     # herbivory
    r"\bherbivor\w+\b",                   # herbivore, herbivores, herbivorous
    r"\bgrazing\b",                       # grazing
    # Colonization / infestation
    r"\bcolonize[sd]?\b",                 # colonize, colonized, colonizes
    r"\bcolonization\b",                  # colonization
    r"\binfests?\b",                      # infests, infest
    r"\binfestation\b",                   # infestation
    r"\binfested\b",                      # infested
    # Foraging / consumption
    r"\bforages?\b",                      # forages, forage
    r"\bforaging\b",                      # foraging
    r"\bconsumes?\b",                     # consumes (without a following noun — weaker)
    r"\bconsumption\b",                   # consumption
    # Generic predation terms (without "of")
    r"\bpredator\b",                      # predator (without "of")
    r"\bprey\b",                          # prey (noun form)
    # Generic parasite/pathogen/infection (without "of" — weaker signal)
    # NOTE: "\bhosts?\b" intentionally excluded — "host" is too ambiguous
    #       ("host country", "host cell"); use STRONG "host of" / "host to" instead.
    r"\bparasit\w+\b",                    # parasitic, parasitism, parasites (generic)
    r"\bpathogen\b",                      # pathogen (without "of")
    r"\binfections?\b",                   # infection or infections (broad — WEAK only)
    r"\bvirulence\b",                     # virulence — WEAK (often methodology context)
    # Pollination (generic)
    r"\bvisits?\s+flowers?\b",            # visits flowers
    r"\bflower\s+visits?\b",              # flower visit/visits
    r"\bpollination\b",                   # pollination (the process)
    # Symbiosis (generic)
    r"\bsymbiosis\b",                     # symbiosis
    r"\bmutualism\b",                     # mutualism
    r"\bsymbiotic\b",                     # symbiotic
    # Epiphytism / myrmecophily
    r"\bepiphyt\w+\b",                    # epiphyte, epiphytic, epiphytism
    r"\bmyrmecophil\w+\b",               # myrmecophile, myrmecophilous
    # Generic interaction language
    r"\binteracts?\s+with\b",             # interacts with
    r"\binteraction\b",                   # interaction (noun, broad)
    r"\bassociated\s+with\b",             # associated with (weak signal)
    # Competition
    r"\bcompetes?\s+with\b",              # competes with
    r"\bcompetition\b",                   # competition
    r"\bcompetitor\b",                    # competitor
]

# =============================================================================
# NEGATION PATTERNS
# Sentence-level negation indicators. Presence suggests an interaction term
# may be negated, reducing confidence that the sentence is a positive example.
# =============================================================================
NEGATION_PATTERNS: List[str] = [
    r"\b(?:not?|no|never|without|neither)\b",
    r"\b(?:failed\s+to|did\s+not|does\s+not|do\s+not)\b",
    r"\b(?:was\s+not|were\s+not|has\s+not|have\s+not)\b",
    r"\b(?:could\s+not|would\s+not|cannot|can\'t)\b",
    r"\babsence\s+of\b",
    r"\bno\s+evidence\s+(?:of|for)\b",
    r"\bnegative\s+for\b",
    r"\bnot\s+observed\b",
    r"\bnot\s+detected\b",
    r"\bfailed\s+to\s+(?:infect|parasiti|transmit|pollinate)\b",
]

# =============================================================================
# METHODOLOGY PATTERNS
# Indicators that the sentence is describing methods, statistics, or results
# tables rather than a biological relationship.
# =============================================================================
METHODOLOGY_PATTERNS: List[str] = [
    r"\b(?:method|protocol|experiment|study\s+design)\b",
    r"\b(?:table|figure|appendix|supplementary)\b",
    r"\b(?:we\s+used|we\s+measured|we\s+analyzed|we\s+collected)\b",
    r"\b(?:statistical|significance|p\s*[<=]\s*0?\.\d+)\b",
    r"\b(?:sample\s+size|n\s*=\s*\d+)\b",
    r"\b(?:confidence\s+interval|odds\s+ratio|hazard\s+ratio)\b",
]

# =============================================================================
# COMPILED PATTERN OBJECTS  (compile once at import — thread-safe, fast)
# =============================================================================
_STRONG_COMPILED:    List[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in STRONG_TERMS]
_WEAK_COMPILED:      List[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in WEAK_TERMS]
_NEGATION_COMPILED:  List[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in NEGATION_PATTERNS]
_METHOD_COMPILED:    List[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in METHODOLOGY_PATTERNS]

# =============================================================================
# SPECIES DETECTION
# =============================================================================

# Common English words that happen to start with a capital letter and could
# falsely match a binomial "Genus species" pattern at the start of sentences
# or after punctuation.
_GENUS_STOPWORDS: frozenset = frozenset({
    "The", "This", "These", "That", "Those", "Such", "Each", "Both", "Other",
    "Many", "Some", "All", "Any", "Few", "Most", "More", "New", "High", "Low",
    "Based", "Using", "From", "With", "Into", "Over", "Under", "Between",
    "Among", "During", "After", "Before", "Since", "While", "Without",
    "Table", "Figure", "Study", "Field", "Sample", "Data", "However",
    "Therefore", "Although", "Because", "Thus", "Here", "There", "Where",
    "When", "Which", "These", "For", "But", "And", "Our", "Its",
    "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
    "First", "Second", "Third", "Last", "Next", "Previous",
})

# Capitalized binomial: "Homo sapiens", "Apis mellifera"
_BINOMIAL_CAP_RE: re.Pattern = re.compile(
    r"\b([A-Z][a-z]{2,})(?:\s+(?:cf\.|aff\.|sp\.?))?\s+([a-z]{4,})\b"
)

# Lowercase binomial fallback (for pre-processed / all-lowercase text).
# Matches two consecutive lowercase words (5+ chars each) that are
# not common English words — likely a Latin species name.
# E.g. "leishmania infantum", "plasmodium falciparum"
_BINOMIAL_LOWER_RE: re.Pattern = re.compile(r"\b([a-z]{4,})\s+([a-z]{5,})\b")

# Common English words that could appear as lowercase "genus" in pre-processed text.
# Lowercase version of _GENUS_STOPWORDS plus extra high-frequency English words.
_LOWER_STOPWORDS: frozenset = frozenset({
    "the", "this", "these", "that", "those", "such", "each", "both", "other",
    "many", "some", "all", "any", "few", "most", "more", "some", "new", "high",
    "low", "based", "using", "from", "with", "into", "over", "under", "between",
    "among", "during", "after", "before", "since", "while", "without", "table",
    "figure", "study", "field", "sample", "data", "however", "therefore",
    "although", "because", "thus", "here", "there", "where", "when", "which",
    "their", "have", "been", "were", "they", "than", "also", "only", "well",
    "upon", "also", "were", "from", "that", "this", "with", "were", "our",
    "two", "three", "four", "five", "first", "second", "third", "last", "next",
    "results", "analysis", "method", "methods", "approach", "model", "models",
    "suggest", "indicate", "show", "found", "known", "used", "observed",
    "present", "described", "collected", "identified", "included", "measured",
    "were", "conducted", "performed", "tested", "compared", "analyzed",
    "significant", "different", "similar", "common", "large", "small",
    "number", "total", "average", "level", "levels", "effect", "effects",
    "species", "genus", "family", "order", "class", "phylum", "kingdom",
    "infection", "parasite", "pathogen", "vector", "host", "prey", "predator",
})


def count_species_mentions(text: str) -> int:
    """
    Estimate the number of binomial species names in text.

    Works on both original (mixed-case) text and pre-processed (all-lowercase)
    text such as the eval_100 benchmark sentences.

    For mixed-case text: matches "Genus species" patterns (capital genus).
    For all-lowercase text: falls back to detecting Latin-looking word pairs
    (both words 5+ chars, neither in a common English vocabulary).

    Args:
        text: Sentence text (any casing).

    Returns:
        Integer count of likely binomial species name occurrences.
    """
    # Try capitalized binomials first
    cap_matches = _BINOMIAL_CAP_RE.findall(text)
    cap_count = sum(1 for genus, _ in cap_matches if genus not in _GENUS_STOPWORDS)

    if cap_count > 0:
        return cap_count

    # Fallback: text is likely all-lowercase — scan for Latin-looking bigrams
    lower_matches = _BINOMIAL_LOWER_RE.findall(text)
    lower_count = sum(
        1 for w1, w2 in lower_matches
        if w1 not in _LOWER_STOPWORDS and w2 not in _LOWER_STOPWORDS
    )
    return lower_count


# =============================================================================
# CORE SCORING FUNCTION
# =============================================================================

def score_sentence(text: str) -> Tuple[bool, float, List[str]]:
    """
    Score a sentence for biotic interaction signal strength.

    Designed to be called on lowercased/preprocessed text for term matching
    (species detection should use the original text via count_species_mentions).

    Scoring logic:
      - Each STRONG match contributes +0.40 to raw strength (capped at 1.0).
      - Each WEAK match contributes +0.15 to raw strength (capped at 1.0).
      - Negation present: multiply effective strength by 0.35 (strong penalty).
      - 2+ methodology indicators: subtract 0.25 from strength.
      - has_signal = True if effective strength >= 0.15 OR >=2 strong matches
        (the two-strong-term threshold handles "X does not infect Y but parasitizes Z").

    Args:
        text: Sentence text, ideally lowercased.

    Returns:
        Tuple of:
          has_signal  (bool):  True if credible interaction signal detected.
          strength    (float): Continuous score 0.0–1.0.
          matched     (List[str]): Regex patterns that matched.
    """
    has_negation = any(p.search(text) for p in _NEGATION_COMPILED)
    method_count = sum(1 for p in _METHOD_COMPILED if p.search(text))

    strong_matches = [p.pattern for p in _STRONG_COMPILED if p.search(text)]
    weak_matches   = [p.pattern for p in _WEAK_COMPILED   if p.search(text)]
    all_matches    = strong_matches + weak_matches

    if not all_matches:
        return False, 0.0, []

    # Base strength
    raw_strength = min(1.0, len(strong_matches) * 0.40 + len(weak_matches) * 0.15)

    # Negation penalty: reduces confidence but doesn't zero out (sentence may
    # contain both a negated and a positive interaction clause).
    if has_negation:
        raw_strength *= 0.35

    # Methodology language penalty
    if method_count >= 2:
        raw_strength = max(0.05, raw_strength - 0.25)

    # has_signal: True if post-adjustment strength is meaningful,
    # OR if 2+ strong terms were found (catches complex sentences).
    has_signal = (raw_strength >= 0.15) or (len(strong_matches) >= 2)

    return has_signal, round(raw_strength, 3), all_matches
