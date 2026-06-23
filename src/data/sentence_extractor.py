"""
Sentence Extractor

Extract sentences from article text that contain both species and interaction terms.
Calculates match quality based on length and completeness.
"""

import re
from typing import List, Optional, Tuple, Set
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to use spaCy for better sentence splitting, fall back to regex
try:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])
    nlp.add_pipe("sentencizer")
    nlp.max_length = 50_000_000  # sentencizer only — no NER/parser memory overhead
    USE_SPACY = True
except (ImportError, OSError):
    USE_SPACY = False
    logger.warning("spaCy not available, using regex sentence splitting")


@dataclass
class SentenceMatch:
    """A sentence that matches species and interaction criteria."""
    sentence: str
    species1_match: str  # The actual matched text for species1
    species2_match: str  # The actual matched text for species2
    interaction_match: str  # The actual matched interaction term
    match_length: int  # Total characters of matched terms
    species1_is_binomial: bool  # True if full binomial name matched
    species2_is_binomial: bool
    sentence_length: int  # Length of the sentence
    source_pmid: Optional[str] = None
    source_doi: Optional[str] = None


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences.

    Args:
        text: Full article text

    Returns:
        List of sentence strings
    """
    if not text:
        return []

    if USE_SPACY:
        # Chunk large texts to avoid spaCy's max_length limit (some PMC articles exceed 2M chars)
        CHUNK_SIZE = 800_000
        if len(text) <= CHUNK_SIZE:
            doc = nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        # Split on paragraph boundaries to avoid cutting mid-sentence
        paragraphs = text.split('\n')
        chunks, current, current_len = [], [], 0
        for para in paragraphs:
            if current_len + len(para) > CHUNK_SIZE and current:
                chunks.append('\n'.join(current))
                current, current_len = [], 0
            current.append(para)
            current_len += len(para)
        if current:
            chunks.append('\n'.join(current))
        sentences = []
        for chunk in chunks:
            doc = nlp(chunk)
            sentences.extend(sent.text.strip() for sent in doc.sents if sent.text.strip())
        return sentences

    # Fallback: regex-based sentence splitting
    # Split on period, question mark, exclamation followed by space and capital
    # But not on common abbreviations
    text = re.sub(r'\s+', ' ', text)

    # Protect common abbreviations
    abbrevs = ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'et al.', 'etc.', 'vs.',
               'Fig.', 'Figs.', 'sp.', 'spp.', 'cf.', 'i.e.', 'e.g.', 'ca.']
    for abbr in abbrevs:
        text = text.replace(abbr, abbr.replace('.', '<DOT>'))

    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    # Restore abbreviations
    sentences = [s.replace('<DOT>', '.').strip() for s in sentences]

    return [s for s in sentences if s and len(s) > 10]


def generate_name_variants(species_name: str) -> Set[str]:
    """
    Generate variants of a species name for matching.

    Args:
        species_name: Full species name (e.g., "Canis lupus familiaris")

    Returns:
        Set of name variants to search for
    """
    variants = set()
    name = species_name.strip()

    if not name:
        return variants

    # Add the full name (case-insensitive matching later)
    variants.add(name)

    parts = name.split()

    if len(parts) >= 2:
        genus = parts[0]
        species = parts[1]

        # Full binomial
        variants.add(f"{genus} {species}")

        # Abbreviated genus (G. species)
        if len(genus) > 1:
            variants.add(f"{genus[0]}. {species}")

        # Just the genus (less reliable)
        if len(genus) > 3:  # Avoid very short genera
            variants.add(genus)

        # Handle subspecies/variety
        if len(parts) >= 3:
            subspecies = parts[2]
            variants.add(f"{genus} {species} {subspecies}")
            variants.add(f"{genus[0]}. {species} {subspecies}")

    elif len(parts) == 1:
        # Single word - might be common name or genus
        if len(name) > 3:
            variants.add(name)

    return variants


def _camel_to_spaces(term: str) -> str:
    """Convert camelCase to space-separated lowercase (preysOn -> preys on)."""
    import re
    # Insert space before uppercase letters and lowercase the result
    spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', term)
    return spaced.lower()


def generate_interaction_variants(interaction_term: str) -> Set[str]:
    """
    Generate variants of an interaction term for matching.

    Args:
        interaction_term: Interaction type (e.g., "parasitizes", "preys on", "preysOn")

    Returns:
        Set of term variants
    """
    variants = set()
    term = interaction_term.strip()

    if not term:
        return variants

    # Handle GloBI camelCase terms (preysOn -> preys on)
    term_normalized = _camel_to_spaces(term)
    variants.add(term_normalized)
    variants.add(term.lower())  # Also add original lowercase

    # GloBI interaction type mappings to natural language variants
    globi_mappings = {
        'preys on': ['preys', 'prey', 'preyed', 'preying', 'predator', 'predation', 'hunts', 'eats'],
        'parasite of': ['parasite', 'parasitizes', 'parasitizing', 'parasitic', 'parasitism', 'parasitoid', 'infested'],
        'has host': ['host', 'hosts', 'hosting', 'hosted', 'parasitizing', 'parasitizes', 'parasitism', 'parasitic', 'infest', 'infested'],
        'host of': ['host', 'hosts', 'hosting', 'hosted', 'parasitized by'],
        'eats': ['eat', 'ate', 'eaten', 'eating', 'feeds on', 'fed on', 'consumes'],
        'pollinates': ['pollinate', 'pollinated', 'pollinating', 'pollinator', 'pollination'],
        'visits flowers of': ['visits', 'flower visitor', 'visits flowers'],
        'pathogen of': ['pathogen', 'pathogenic', 'infects', 'infection'],
        'vector of': ['vector', 'transmits', 'transmission', 'carrier'],
        'symbiont of': ['symbiont', 'symbiosis', 'symbiotic', 'mutualist'],
        'interacts with': ['interacts', 'interaction', 'associated with'],
        'kills': ['kill', 'killed', 'killing'],
        'dispersal vector of': ['disperses', 'dispersal', 'seed disperser'],
        'ectoparasite of': ['ectoparasite', 'ectoparasitic', 'ectoparasitism', 'external parasite'],
    }

    # Common verb form variations
    verb_forms = {
        'parasitizes': ['parasitize', 'parasitized', 'parasitizing', 'parasite of', 'parasitic on', 'parasite'],
        'parasitize': ['parasitizes', 'parasitized', 'parasitizing', 'parasite of', 'parasite'],
        'preys on': ['prey on', 'preyed on', 'preying on', 'predator of', 'predation on', 'preys', 'prey'],
        'preys': ['prey', 'preyed', 'preying', 'predator', 'predation'],
        'prey on': ['preys on', 'preyed on', 'preying on'],
        'infects': ['infect', 'infected', 'infecting', 'infection of', 'infectious to', 'infection'],
        'infect': ['infects', 'infected', 'infecting', 'infection'],
        'pollinates': ['pollinate', 'pollinated', 'pollinating', 'pollinator of', 'pollination of', 'pollinator'],
        'pollinate': ['pollinates', 'pollinated', 'pollinating', 'pollinator'],
        'feeds on': ['fed on', 'feeding on', 'feed on', 'feeds', 'eats'],
        'eats': ['eat', 'ate', 'eating', 'eaten by', 'fed on', 'feeds on'],
        'hosts': ['host', 'hosted', 'hosting', 'host of', 'host to'],
        'host of': ['hosts', 'host', 'hosting'],
        'colonizes': ['colonize', 'colonized', 'colonizing', 'colonization of'],
        'attacks': ['attack', 'attacked', 'attacking'],
        'kills': ['kill', 'killed', 'killing'],
        'hunts': ['hunt', 'hunted', 'hunting', 'hunter'],
        'transmits': ['transmit', 'transmitted', 'transmitting', 'transmission of', 'vector of'],
        'disperses': ['disperse', 'dispersed', 'dispersing', 'dispersal'],
        'visits': ['visit', 'visited', 'visiting', 'visitor'],
    }

    # Apply GloBI mappings
    for pattern, forms in globi_mappings.items():
        if pattern in term_normalized or term_normalized in pattern:
            variants.update(forms)
            variants.add(pattern)

    # Add verb forms
    for base, forms in verb_forms.items():
        if base in term_normalized or term_normalized in base:
            variants.update(forms)
            variants.add(base)

    # Handle multi-word terms
    if ' ' in term_normalized:
        variants.add(term_normalized.replace(' ', '-'))  # "preys-on" variant

    return variants


def is_binomial_name(name: str) -> bool:
    """Check if a species name appears to be a full binomial."""
    parts = name.strip().split()
    if len(parts) < 2:
        return False

    # First part should be capitalized (genus)
    # Second part should be lowercase (species epithet)
    genus = parts[0]
    species = parts[1]

    # Check pattern: Genus species or G. species
    if re.match(r'^[A-Z][a-z]+$', genus) and re.match(r'^[a-z]+$', species):
        return True
    if re.match(r'^[A-Z]\.$', genus) and re.match(r'^[a-z]+$', species):
        return True

    return False


def find_match_in_sentence(
    sentence: str,
    search_terms: Set[str],
    case_sensitive: bool = False
) -> Optional[Tuple[str, int, int]]:
    """
    Find the best match for any of the search terms in a sentence.

    Args:
        sentence: The sentence to search
        search_terms: Set of terms to look for
        case_sensitive: Whether to match case

    Returns:
        Tuple of (matched_text, start_pos, end_pos) or None
    """
    best_match = None
    best_length = 0

    flags = 0 if case_sensitive else re.IGNORECASE

    for term in search_terms:
        # Escape special regex characters but preserve word boundaries
        pattern = r'\b' + re.escape(term) + r'\b'
        match = re.search(pattern, sentence, flags)

        if match:
            matched_text = match.group(0)
            if len(matched_text) > best_length:
                best_match = (matched_text, match.start(), match.end())
                best_length = len(matched_text)

    return best_match


def extract_matching_sentences(
    article_text: str,
    species1: str,
    species2: str,
    interaction: str,
    min_sentence_length: int = 20,
    max_sentence_length: int = 500
) -> List[SentenceMatch]:
    """
    Extract sentences containing both species and the interaction term.

    Args:
        article_text: Full text of the article
        species1: Source species name
        species2: Target species name
        interaction: Interaction type
        min_sentence_length: Minimum sentence length to consider
        max_sentence_length: Maximum sentence length to consider

    Returns:
        List of SentenceMatch objects for matching sentences
    """
    if not article_text:
        return []

    # Generate search variants
    species1_variants = generate_name_variants(species1)
    species2_variants = generate_name_variants(species2)
    interaction_variants = generate_interaction_variants(interaction)

    # Split into sentences
    sentences = split_sentences(article_text)
    matches = []

    for sentence in sentences:
        # Check length constraints
        sent_len = len(sentence)
        if sent_len < min_sentence_length or sent_len > max_sentence_length:
            continue

        # Look for species1
        sp1_match = find_match_in_sentence(sentence, species1_variants)
        if not sp1_match:
            continue

        # Look for species2
        sp2_match = find_match_in_sentence(sentence, species2_variants)
        if not sp2_match:
            continue

        # Make sure we're not matching the same text for both species
        # (could happen if searching for genus only)
        if sp1_match[1] == sp2_match[1]:  # Same start position
            continue

        # Look for interaction term
        int_match = find_match_in_sentence(sentence, interaction_variants)
        if not int_match:
            continue

        # Calculate total match length
        match_length = len(sp1_match[0]) + len(sp2_match[0]) + len(int_match[0])

        # Check if matches are binomial names
        sp1_is_binomial = is_binomial_name(sp1_match[0])
        sp2_is_binomial = is_binomial_name(sp2_match[0])

        matches.append(SentenceMatch(
            sentence=sentence,
            species1_match=sp1_match[0],
            species2_match=sp2_match[0],
            interaction_match=int_match[0],
            match_length=match_length,
            species1_is_binomial=sp1_is_binomial,
            species2_is_binomial=sp2_is_binomial,
            sentence_length=sent_len
        ))

    return matches


def extract_best_sentence(
    article_text: str,
    species1: str,
    species2: str,
    interaction: str
) -> Optional[SentenceMatch]:
    """
    Extract the single best matching sentence from an article.

    The "best" sentence is the one with:
    1. Longest total match length (species + interaction terms)
    2. Preference for binomial species names
    3. Moderate sentence length (not too short or long)

    Args:
        article_text: Full text of the article
        species1: Source species name
        species2: Target species name
        interaction: Interaction type

    Returns:
        Best SentenceMatch or None if no match found
    """
    matches = extract_matching_sentences(
        article_text, species1, species2, interaction
    )

    if not matches:
        return None

    # Score each match
    def score_match(m: SentenceMatch) -> float:
        score = 0.0

        # Match length bonus (normalized)
        score += min(m.match_length / 60, 1.0) * 40

        # Binomial name bonus
        if m.species1_is_binomial:
            score += 15
        if m.species2_is_binomial:
            score += 15

        # Sentence length: prefer medium length (100-300 chars)
        if 100 <= m.sentence_length <= 300:
            score += 20
        elif 50 <= m.sentence_length <= 400:
            score += 10

        # Penalty for very short sentences (might be fragments)
        if m.sentence_length < 40:
            score -= 20

        return score

    # Sort by score and return best
    matches.sort(key=score_match, reverse=True)
    return matches[0]


def batch_extract_sentences(
    articles: dict,
    interactions: list,
    max_per_interaction: int = 3
) -> List[dict]:
    """
    Extract sentences from multiple articles for multiple interactions.

    Args:
        articles: Dict mapping article identifier to text
        interactions: List of dicts with species1, species2, interaction, doi/pmid
        max_per_interaction: Max sentences to extract per interaction

    Returns:
        List of dicts with extracted sentence info
    """
    results = []

    for interaction in interactions:
        species1 = interaction.get('sourceTaxonName', interaction.get('species1', ''))
        species2 = interaction.get('targetTaxonName', interaction.get('species2', ''))
        int_type = interaction.get('interactionTypeName', interaction.get('interaction', ''))
        doi = interaction.get('referenceDoi', interaction.get('doi'))
        pmid = interaction.get('pmid')

        # Find article text (normalize DOI to lowercase for consistent key matching)
        doi_norm = doi.lower() if doi else None
        article_key = f"doi:{doi_norm}|pmid:{pmid}"
        article = articles.get(article_key)

        if not article:
            continue

        text = article.full_text or article.abstract or ""
        if not text:
            continue

        # Extract matches
        matches = extract_matching_sentences(text, species1, species2, int_type)

        # Sort by quality and take top N
        matches.sort(key=lambda m: m.match_length, reverse=True)

        for match in matches[:max_per_interaction]:
            match.source_pmid = pmid
            match.source_doi = doi
            results.append({
                'sentence': match.sentence,
                'species1': species1,
                'species2': species2,
                'interaction': int_type,
                'species1_match': match.species1_match,
                'species2_match': match.species2_match,
                'interaction_match': match.interaction_match,
                'match_length': match.match_length,
                'species1_is_binomial': match.species1_is_binomial,
                'species2_is_binomial': match.species2_is_binomial,
                'pmid': pmid,
                'doi': doi
            })

    logger.info(f"Extracted {len(results)} sentences from {len(interactions)} interactions")
    return results


# ============================================================================
# RELAXED EXTRACTION: Find ANY interaction sentences (not requiring GloBI match)
# ============================================================================

# Common taxonomic names and patterns for biotic interaction detection
TAXONOMIC_PATTERNS = [
    # Order/Class level (common in ecology papers)
    r'\b(Diptera|Lepidoptera|Coleoptera|Hymenoptera|Arachnida|Araneae)\b',
    r'\b(Nematoda|Trematoda|Cestoda|Acari|Hemiptera|Orthoptera)\b',
    r'\b(Mammalia|Aves|Reptilia|Amphibia|Actinopterygii)\b',
    r'\b(Gastropoda|Bivalvia|Annelida|Crustacea)\b',
    # Common names for taxa groups
    r'\b(spiders?|insects?|flies|fly|moths?|butterflies|butterfly)\b',
    r'\b(beetles?|wasps?|bees?|ants?|mosquitoes?|midges?)\b',
    r'\b(birds?|rodents?|mammals?|fish|fishes|amphibians?)\b',
    r'\b(parasites?|parasitoids?|pathogens?|predators?|herbivores?)\b',
    r'\b(nematodes?|mites?|ticks?|worms?|larvae|larva)\b',
    r'\b(plants?|trees?|flowers?|grasses?|shrubs?)\b',
    r'\b(fungi|fungus|mushrooms?|bacteria|virus|viruses)\b',
    # Abbreviated binomials: G. species (high confidence)
    r'\b[A-Z]\.\s*[a-z]{3,15}\b',
]

# Separate pattern for binomial names - needs more validation
BINOMIAL_PATTERN = re.compile(r'\b([A-Z][a-z]{2,15})\s+([a-z]{3,15})\b')

# Words that look like binomials but aren't (common false positives)
BINOMIAL_BLACKLIST = {
    # Common sentence starters / phrases
    'the', 'this', 'that', 'these', 'those', 'which', 'where', 'when',
    'what', 'while', 'within', 'without', 'with', 'were', 'will',
    'would', 'could', 'should', 'have', 'has', 'had', 'been', 'being',
    'from', 'into', 'onto', 'upon', 'under', 'over', 'after', 'before',
    'between', 'among', 'through', 'during', 'since', 'until',
    'each', 'every', 'either', 'neither', 'both', 'some', 'many',
    'most', 'more', 'less', 'other', 'another', 'such', 'same',
    # Academic writing patterns
    'figure', 'table', 'method', 'result', 'study', 'data', 'analysis',
    'effect', 'evidence', 'example', 'however', 'therefore', 'moreover',
    'although', 'because', 'various', 'several', 'different', 'similar',
    'previous', 'current', 'present', 'recent', 'future', 'total',
    'using', 'based', 'found', 'showed', 'observed', 'reported',
    'compared', 'including', 'following', 'according', 'suggesting',
    # Common nouns that might match
    'research', 'species', 'sample', 'number', 'level', 'group',
    'system', 'model', 'pattern', 'structure', 'function', 'process',
    'offered', 'text', 'appendix', 'supplementary', 'material',
}

INTERACTION_PATTERNS = [
    # Predation
    r'\b(preys?\s+(?:up)?on|preyed\s+(?:up)?on|preying\s+(?:up)?on|predation|predator\s+of)\b',
    r'\b(hunts?|hunted|hunting|eats?|ate|eaten|eating|consumes?|consumed|consuming)\b',
    r'\b(feeds?\s+on|fed\s+on|feeding\s+on|forages?\s+on|foraging\s+on)\b',
    # Parasitism
    r'\b(parasiti[zs]es?|parasiti[zs]ed|parasiti[zs]ing|parasitic\s+on|parasite\s+of)\b',
    r'\b(infects?|infected|infecting|infection\s+of|infests?|infested|infesting)\b',
    # Host relationships
    r'\b(hosts?|hosted|hosting|host\s+of|host\s+to|host\s+for)\b',
    # Pollination/mutualism
    r'\b(pollinates?|pollinated|pollinating|pollinator\s+of|visits?\s+flowers?)\b',
    r'\b(disperses?|dispersed|dispersing|seed\s+dispers)\b',
    # Pathogenicity
    r'\b(pathogen\s+of|pathogenic\s+to|causes?\s+disease|disease\s+(?:of|in))\b',
    r'\b(transmits?|transmitted|transmitting|vector\s+of|carries|carrier\s+of)\b',
    # General
    r'\b(attacks?|attacked|attacking|kills?|killed|killing)\b',
    r'\b(symbiont|symbiosis|symbiotic|mutualist|mutualistic)\b',
]

# Compile patterns for efficiency
TAXONOMIC_REGEX = [re.compile(p, re.IGNORECASE) for p in TAXONOMIC_PATTERNS]
INTERACTION_REGEX = [re.compile(p, re.IGNORECASE) for p in INTERACTION_PATTERNS]


@dataclass
class RelaxedSentenceMatch:
    """A sentence containing taxonomic terms and interaction verbs."""
    sentence: str
    taxa_found: List[str]  # All taxonomic terms found
    interactions_found: List[str]  # All interaction terms found
    n_taxa: int
    n_interactions: int
    sentence_length: int
    source_doi: Optional[str] = None
    source_pmid: Optional[str] = None


def find_all_taxa(sentence: str) -> List[str]:
    """Find all taxonomic terms in a sentence."""
    taxa = []

    # 1. Match known taxonomic patterns (orders, common names)
    for pattern in TAXONOMIC_REGEX:
        matches = pattern.findall(sentence)
        taxa.extend(matches)

    # 2. Match binomial names with validation
    for match in BINOMIAL_PATTERN.finditer(sentence):
        genus, species = match.groups()
        # Filter out common false positives
        if genus.lower() not in BINOMIAL_BLACKLIST:
            # Additional validation: species epithet should be >3 chars
            if len(species) >= 4:
                taxa.append(f"{genus} {species}")

    # Deduplicate while preserving order
    seen = set()
    unique_taxa = []
    for t in taxa:
        t_lower = t.lower()
        if t_lower not in seen:
            seen.add(t_lower)
            unique_taxa.append(t)
    return unique_taxa


def find_all_interactions(sentence: str) -> List[str]:
    """Find all interaction terms in a sentence."""
    interactions = []
    for pattern in INTERACTION_REGEX:
        matches = pattern.findall(sentence)
        interactions.extend(matches)
    # Deduplicate
    seen = set()
    unique = []
    for i in interactions:
        i_lower = i.lower()
        if i_lower not in seen:
            seen.add(i_lower)
            unique.append(i)
    return unique


def extract_interaction_sentences_relaxed(
    article_text: str,
    min_taxa: int = 2,
    min_sentence_length: int = 40,
    max_sentence_length: int = 400,
    doi: str = None,
    pmid: str = None
) -> List[RelaxedSentenceMatch]:
    """
    Extract sentences that describe biotic interactions using relaxed matching.

    This doesn't require specific GloBI species pairs - instead it finds sentences
    containing multiple taxonomic terms AND interaction verbs.

    Args:
        article_text: Full text of the article
        min_taxa: Minimum number of taxonomic terms required (default 2)
        min_sentence_length: Minimum sentence length
        max_sentence_length: Maximum sentence length
        doi: Source DOI
        pmid: Source PMID

    Returns:
        List of RelaxedSentenceMatch objects
    """
    if not article_text:
        return []

    sentences = split_sentences(article_text)
    matches = []

    for sentence in sentences:
        sent_len = len(sentence)
        if sent_len < min_sentence_length or sent_len > max_sentence_length:
            continue

        # Find taxa and interactions
        taxa = find_all_taxa(sentence)
        interactions = find_all_interactions(sentence)

        # Require at least min_taxa and at least 1 interaction term
        if len(taxa) >= min_taxa and len(interactions) >= 1:
            matches.append(RelaxedSentenceMatch(
                sentence=sentence,
                taxa_found=taxa,
                interactions_found=interactions,
                n_taxa=len(taxa),
                n_interactions=len(interactions),
                sentence_length=sent_len,
                source_doi=doi,
                source_pmid=pmid
            ))

    return matches


def extract_best_interaction_sentences(
    article_text: str,
    max_sentences: int = 5,
    doi: str = None,
    pmid: str = None
) -> List[RelaxedSentenceMatch]:
    """
    Extract the best interaction sentences from an article.

    Scores sentences by: number of taxa, number of interaction terms, sentence length.

    Args:
        article_text: Full text of the article
        max_sentences: Maximum sentences to return
        doi: Source DOI
        pmid: Source PMID

    Returns:
        List of best RelaxedSentenceMatch objects
    """
    matches = extract_interaction_sentences_relaxed(
        article_text, min_taxa=2, doi=doi, pmid=pmid
    )

    if not matches:
        return []

    # Score function: prioritize more taxa, more interaction terms, medium length
    def score(m: RelaxedSentenceMatch) -> float:
        s = 0.0
        # Taxa count (diminishing returns)
        s += min(m.n_taxa, 4) * 15
        # Interaction count
        s += min(m.n_interactions, 3) * 10
        # Prefer medium-length sentences (100-250 chars)
        if 100 <= m.sentence_length <= 250:
            s += 20
        elif 80 <= m.sentence_length <= 350:
            s += 10
        # Penalty for very short
        if m.sentence_length < 50:
            s -= 15
        return s

    # Sort by score and return top N
    matches.sort(key=score, reverse=True)
    return matches[:max_sentences]


if __name__ == "__main__":
    # Example usage
    sample_text = """
    The red fox (Vulpes vulpes) is a common predator in temperate regions.
    In a study of forest ecosystems, we observed that V. vulpes frequently
    preys on small rodents including the house mouse (Mus musculus).
    The fox population was estimated at 2.3 individuals per square kilometer.
    Parasitic infections by Toxoplasma gondii were found in 23% of examined mice.
    T. gondii infects various mammalian hosts and can cause toxoplasmosis.
    Pollination of apple trees by honeybees (Apis mellifera) was studied.
    """

    print("=== Sentence Extraction Demo ===\n")

    # Test 1: Predation
    print("Test 1: Fox preys on mouse")
    matches = extract_matching_sentences(
        sample_text, "Vulpes vulpes", "Mus musculus", "preys on"
    )
    for m in matches:
        print(f"  Match length: {m.match_length}")
        print(f"  Species1: '{m.species1_match}' (binomial: {m.species1_is_binomial})")
        print(f"  Species2: '{m.species2_match}' (binomial: {m.species2_is_binomial})")
        print(f"  Interaction: '{m.interaction_match}'")
        print(f"  Sentence: {m.sentence}\n")

    # Test 2: Parasitism
    print("Test 2: Toxoplasma infects mice")
    matches = extract_matching_sentences(
        sample_text, "Toxoplasma gondii", "Mus musculus", "infects"
    )
    for m in matches:
        print(f"  Match length: {m.match_length}")
        print(f"  Sentence: {m.sentence}\n")

    # Test 3: Best sentence
    print("Test 3: Best sentence for fox-mouse predation")
    best = extract_best_sentence(
        sample_text, "Vulpes vulpes", "Mus musculus", "preys on"
    )
    if best:
        print(f"  Best match: {best.sentence}")
        print(f"  Total match length: {best.match_length}")
