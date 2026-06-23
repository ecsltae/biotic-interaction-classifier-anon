"""
Quality Filter

Apply quality scoring and domain rules to extracted sentences.
Filters out low-quality matches and validates biological constraints.
"""

import json
import re
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Interaction term specificity scores (0-1 scale)
# Higher = more specific and reliable
INTERACTION_SPECIFICITY = {
    # Very specific (0.9-1.0)
    'parasitizes': 1.0,
    'parasitize': 1.0,
    'parasitized by': 1.0,
    'ectoparasite of': 1.0,
    'endoparasite of': 1.0,
    'hyperparasite of': 1.0,
    'parasitoid of': 1.0,
    'pathogen of': 0.95,
    'infects': 0.95,
    'vector of': 0.95,

    # Specific (0.7-0.9)
    'preys on': 0.9,
    'predator of': 0.9,
    'hunts': 0.85,
    'kills': 0.85,
    'pollinates': 0.9,
    'pollinator of': 0.9,
    'disperses seeds of': 0.9,
    'symbiont of': 0.85,
    'mutualist of': 0.85,

    # Moderate (0.5-0.7)
    'feeds on': 0.7,
    'eats': 0.65,
    'consumes': 0.65,
    'herbivore of': 0.7,
    'grazes on': 0.7,
    'host of': 0.6,
    'colonizes': 0.6,

    # Less specific (0.3-0.5)
    'interacts with': 0.4,
    'associated with': 0.35,
    'found on': 0.3,
    'found in': 0.3,
    'occurs with': 0.3,

    # Very non-specific (0.0-0.3)
    'with': 0.1,
    'and': 0.05,
    'near': 0.1,
}

# Default specificity for unknown terms
DEFAULT_SPECIFICITY = 0.5


@dataclass
class QualityScore:
    """Quality assessment of a sentence match."""
    total_score: float
    length_score: float
    specificity_score: float
    binomial_score: float
    sentence_quality_score: float
    passes_rules: bool
    rule_violations: List[str]
    details: Dict[str, Any]


@dataclass
class DomainRule:
    """A domain validation rule for interaction types."""
    interaction_pattern: str  # Regex pattern for interaction types
    requires_taxon: List[str]  # Required taxa (at least one species must be in these)
    requires_role: Optional[str]  # 'source', 'target', or None
    forbidden_taxa: List[str]  # Taxa that cannot be involved
    forbidden_same_species: bool  # Whether species1 == species2 is forbidden
    min_match_length: int  # Minimum match length for this interaction type
    notes: str


def load_domain_rules(rules_path: str) -> Dict[str, DomainRule]:
    """
    Load domain rules from JSON file.

    Expected format:
    {
      "pollination": {
        "requires_taxon": ["Plantae", "Viridiplantae"],
        "requires_role": "target",
        "forbidden_taxa": [],
        "forbidden_same_species": true,
        "min_match_length": 20,
        "notes": "Pollinator visits plant"
      },
      ...
    }
    """
    rules = {}

    if not Path(rules_path).exists():
        logger.warning(f"Domain rules file not found: {rules_path}")
        return rules

    with open(rules_path, 'r') as f:
        data = json.load(f)

    for interaction, rule_data in data.items():
        # Skip metadata fields (starting with _)
        if interaction.startswith('_'):
            continue
        # Skip if not a dict (invalid rule format)
        if not isinstance(rule_data, dict):
            continue
        rules[interaction.lower()] = DomainRule(
            interaction_pattern=rule_data.get('pattern', interaction),
            requires_taxon=rule_data.get('requires_taxon', []),
            requires_role=rule_data.get('requires_role'),
            forbidden_taxa=rule_data.get('forbidden_taxa', []),
            forbidden_same_species=rule_data.get('forbidden_same_species', True),
            min_match_length=rule_data.get('min_match_length', 15),
            notes=rule_data.get('notes', '')
        )

    logger.info(f"Loaded {len(rules)} domain rules from {rules_path}")
    return rules


def get_interaction_specificity(interaction_term: str) -> float:
    """
    Get the specificity score for an interaction term.

    Args:
        interaction_term: The interaction term to score

    Returns:
        Specificity score between 0 and 1
    """
    term = interaction_term.lower().strip()

    # Check exact match
    if term in INTERACTION_SPECIFICITY:
        return INTERACTION_SPECIFICITY[term]

    # Check partial matches
    for known_term, score in INTERACTION_SPECIFICITY.items():
        if known_term in term or term in known_term:
            return score

    return DEFAULT_SPECIFICITY


def is_binomial_name(name: str) -> bool:
    """Check if a name appears to be a binomial species name."""
    parts = name.strip().split()
    if len(parts) < 2:
        return False

    genus = parts[0]
    species = parts[1]

    # Pattern: Capitalized genus + lowercase species
    if re.match(r'^[A-Z][a-z]+$', genus) and re.match(r'^[a-z]+$', species):
        return True
    # Pattern: Abbreviated genus (G. species)
    if re.match(r'^[A-Z]\.$', genus) and re.match(r'^[a-z]+$', species):
        return True

    return False


def calculate_length_score(match_length: int, max_expected: int = 60) -> float:
    """
    Calculate quality score based on match length.

    Short matches like "cat eat dog" (11 chars) score low.
    Longer matches like "Felis catus preys upon Mus musculus" (35 chars) score high.

    Args:
        match_length: Total characters matched (species1 + species2 + interaction)
        max_expected: Length considered "perfect" (score = 1.0)

    Returns:
        Score between 0 and 1
    """
    if match_length <= 0:
        return 0.0

    # Very short matches are suspicious
    if match_length < 15:
        return match_length / 30  # Max 0.5 for very short

    # Linear scale up to max_expected
    return min(match_length / max_expected, 1.0)


def calculate_quality_score(
    sentence: str,
    species1_match: str,
    species2_match: str,
    interaction_match: str,
    match_length: int,
    species1_is_binomial: bool = None,
    species2_is_binomial: bool = None
) -> QualityScore:
    """
    Calculate overall quality score for a sentence match.

    Scoring breakdown (0-100 scale):
    - Match length: 0-40 points
    - Binomial names: 0-30 points (15 each)
    - Interaction specificity: 0-20 points
    - Sentence quality: 0-10 points

    Args:
        sentence: The matched sentence
        species1_match: Matched text for species1
        species2_match: Matched text for species2
        interaction_match: Matched interaction term
        match_length: Total match length
        species1_is_binomial: Whether species1 is binomial (auto-detect if None)
        species2_is_binomial: Whether species2 is binomial (auto-detect if None)

    Returns:
        QualityScore with detailed breakdown
    """
    details = {}

    # 1. Length score (0-40 points)
    length_score = calculate_length_score(match_length) * 40
    details['match_length'] = match_length

    # 2. Binomial name scores (0-30 points)
    if species1_is_binomial is None:
        species1_is_binomial = is_binomial_name(species1_match)
    if species2_is_binomial is None:
        species2_is_binomial = is_binomial_name(species2_match)

    binomial_score = 0
    if species1_is_binomial:
        binomial_score += 15
    if species2_is_binomial:
        binomial_score += 15
    details['species1_is_binomial'] = species1_is_binomial
    details['species2_is_binomial'] = species2_is_binomial

    # 3. Interaction specificity (0-20 points)
    specificity = get_interaction_specificity(interaction_match)
    specificity_score = specificity * 20
    details['interaction_specificity'] = specificity

    # 4. Sentence quality (0-10 points)
    sentence_quality_score = 0
    sent_len = len(sentence)

    # Prefer medium-length sentences
    if 80 <= sent_len <= 250:
        sentence_quality_score += 5
    elif 50 <= sent_len <= 350:
        sentence_quality_score += 3

    # Bonus for proper capitalization
    if sentence[0].isupper():
        sentence_quality_score += 2

    # Bonus for ending with period
    if sentence.rstrip().endswith('.'):
        sentence_quality_score += 2

    # Penalty for too many special characters (might be noise)
    special_ratio = len(re.findall(r'[^a-zA-Z0-9\s.,;:\'"()-]', sentence)) / max(sent_len, 1)
    if special_ratio > 0.1:
        sentence_quality_score -= 3

    sentence_quality_score = max(0, min(10, sentence_quality_score))
    details['sentence_length'] = sent_len

    # Total score
    total_score = length_score + binomial_score + specificity_score + sentence_quality_score

    return QualityScore(
        total_score=total_score,
        length_score=length_score,
        specificity_score=specificity_score,
        binomial_score=binomial_score,
        sentence_quality_score=sentence_quality_score,
        passes_rules=True,  # Will be updated by validate_domain_rules
        rule_violations=[],
        details=details
    )


def validate_domain_rules(
    interaction: str,
    species1: str,
    species1_kingdom: str,
    species2: str,
    species2_kingdom: str,
    match_length: int,
    rules: Dict[str, DomainRule]
) -> Tuple[bool, List[str]]:
    """
    Validate an interaction against domain rules.

    Args:
        interaction: Interaction type name
        species1: Source species name
        species1_kingdom: Kingdom of species1 (e.g., "Animalia", "Plantae")
        species2: Target species name
        species2_kingdom: Kingdom of species2
        match_length: Total match length
        rules: Domain rules dictionary

    Returns:
        Tuple of (is_valid, list of violation messages)
    """
    violations = []
    interaction_lower = interaction.lower().strip()

    # Find applicable rule
    rule = None
    for rule_name, rule_obj in rules.items():
        pattern = rule_obj.interaction_pattern
        if re.search(pattern, interaction_lower, re.IGNORECASE):
            rule = rule_obj
            break

    if not rule:
        # No rule for this interaction type - pass by default
        return True, []

    # Check minimum match length
    if match_length < rule.min_match_length:
        violations.append(
            f"Match length {match_length} below minimum {rule.min_match_length} "
            f"for {interaction}"
        )

    # Check required taxa
    if rule.requires_taxon:
        source_ok = any(
            taxon.lower() in (species1_kingdom or '').lower()
            for taxon in rule.requires_taxon
        )
        target_ok = any(
            taxon.lower() in (species2_kingdom or '').lower()
            for taxon in rule.requires_taxon
        )

        if rule.requires_role == 'source' and not source_ok:
            violations.append(
                f"{interaction} requires source to be in {rule.requires_taxon}, "
                f"but got {species1_kingdom}"
            )
        elif rule.requires_role == 'target' and not target_ok:
            violations.append(
                f"{interaction} requires target to be in {rule.requires_taxon}, "
                f"but got {species2_kingdom}"
            )
        elif rule.requires_role is None and not (source_ok or target_ok):
            violations.append(
                f"{interaction} requires at least one species in {rule.requires_taxon}"
            )

    # Check forbidden taxa
    for taxon in rule.forbidden_taxa:
        if taxon.lower() in (species1_kingdom or '').lower():
            violations.append(f"{interaction}: source cannot be {taxon}")
        if taxon.lower() in (species2_kingdom or '').lower():
            violations.append(f"{interaction}: target cannot be {taxon}")

    # Check same species
    if rule.forbidden_same_species:
        sp1_norm = species1.lower().strip()
        sp2_norm = species2.lower().strip()
        if sp1_norm == sp2_norm:
            violations.append(f"{interaction}: source and target cannot be the same species")

    is_valid = len(violations) == 0
    return is_valid, violations


def filter_sentences(
    sentences: List[dict],
    min_quality_score: float = 50.0,
    rules: Dict[str, DomainRule] = None
) -> Tuple[List[dict], List[dict]]:
    """
    Filter sentences by quality score and domain rules.

    Args:
        sentences: List of sentence dicts from sentence_extractor
        min_quality_score: Minimum quality score to accept (0-100)
        rules: Domain rules for validation

    Returns:
        Tuple of (accepted sentences, rejected sentences)
    """
    accepted = []
    rejected = []

    for sent in sentences:
        # Calculate quality score
        quality = calculate_quality_score(
            sentence=sent['sentence'],
            species1_match=sent.get('species1_match', ''),
            species2_match=sent.get('species2_match', ''),
            interaction_match=sent.get('interaction_match', ''),
            match_length=sent.get('match_length', 0),
            species1_is_binomial=sent.get('species1_is_binomial'),
            species2_is_binomial=sent.get('species2_is_binomial')
        )

        sent['quality_score'] = quality.total_score
        sent['quality_details'] = quality.details

        # Check domain rules if provided
        if rules:
            is_valid, violations = validate_domain_rules(
                interaction=sent.get('interaction', ''),
                species1=sent.get('species1', ''),
                species1_kingdom=sent.get('species1_kingdom', ''),
                species2=sent.get('species2', ''),
                species2_kingdom=sent.get('species2_kingdom', ''),
                match_length=sent.get('match_length', 0),
                rules=rules
            )
            quality.passes_rules = is_valid
            quality.rule_violations = violations
            sent['rule_violations'] = violations

        # Accept or reject
        if quality.total_score >= min_quality_score and quality.passes_rules:
            accepted.append(sent)
        else:
            sent['rejection_reason'] = (
                f"Score {quality.total_score:.1f} < {min_quality_score}"
                if quality.total_score < min_quality_score
                else f"Rule violations: {quality.rule_violations}"
            )
            rejected.append(sent)

    logger.info(f"Filtered: {len(accepted)} accepted, {len(rejected)} rejected")
    return accepted, rejected


# Default domain rules (can be overridden by loading from file)
DEFAULT_DOMAIN_RULES = {
    "pollination": DomainRule(
        interaction_pattern=r"pollinat",
        requires_taxon=["Plantae", "Viridiplantae"],
        requires_role="target",
        forbidden_taxa=[],
        forbidden_same_species=True,
        min_match_length=20,
        notes="Pollinator (animal) visits plant"
    ),
    "herbivory": DomainRule(
        interaction_pattern=r"herbiv|grazes?|browses?",
        requires_taxon=["Plantae", "Viridiplantae"],
        requires_role="target",
        forbidden_taxa=[],
        forbidden_same_species=True,
        min_match_length=15,
        notes="Herbivore (animal) eats plant"
    ),
    "parasitism": DomainRule(
        interaction_pattern=r"parasit",
        requires_taxon=[],
        requires_role=None,
        forbidden_taxa=[],
        forbidden_same_species=True,
        min_match_length=20,
        notes="Parasite and host must be different species"
    ),
    "predation": DomainRule(
        interaction_pattern=r"preys?|predat|hunts?|kills?",
        requires_taxon=[],
        requires_role=None,
        forbidden_taxa=["Plantae"],  # Plants don't prey
        forbidden_same_species=True,
        min_match_length=15,
        notes="Predator kills prey, plants cannot be predators"
    ),
}


if __name__ == "__main__":
    # Example usage
    print("=== Quality Filter Demo ===\n")

    # Test quality scoring
    test_cases = [
        {
            'sentence': "The fox eats mice.",
            'species1_match': "fox",
            'species2_match': "mice",
            'interaction_match': "eats",
            'match_length': 11,
        },
        {
            'sentence': "Vulpes vulpes frequently preys upon Mus musculus in forest ecosystems.",
            'species1_match': "Vulpes vulpes",
            'species2_match': "Mus musculus",
            'interaction_match': "preys upon",
            'match_length': 35,
        },
        {
            'sentence': "The parasitic wasp Cotesia glomerata parasitizes larvae of Pieris brassicae.",
            'species1_match': "Cotesia glomerata",
            'species2_match': "Pieris brassicae",
            'interaction_match': "parasitizes",
            'match_length': 44,
        },
    ]

    for case in test_cases:
        score = calculate_quality_score(**case)
        print(f"Sentence: {case['sentence']}")
        print(f"  Total score: {score.total_score:.1f}/100")
        print(f"  - Length: {score.length_score:.1f}/40")
        print(f"  - Binomial: {score.binomial_score:.1f}/30")
        print(f"  - Specificity: {score.specificity_score:.1f}/20")
        print(f"  - Sentence quality: {score.sentence_quality_score:.1f}/10")
        print()

    # Test domain rules
    print("=== Domain Rule Validation ===\n")

    # Pollination: should require plant as target
    is_valid, violations = validate_domain_rules(
        interaction="pollinates",
        species1="Apis mellifera",
        species1_kingdom="Animalia",
        species2="Malus domestica",
        species2_kingdom="Plantae",
        match_length=30,
        rules=DEFAULT_DOMAIN_RULES
    )
    print(f"Bee pollinates apple: valid={is_valid}, violations={violations}")

    # Invalid: animal pollinates animal
    is_valid, violations = validate_domain_rules(
        interaction="pollinates",
        species1="Apis mellifera",
        species1_kingdom="Animalia",
        species2="Bombus terrestris",
        species2_kingdom="Animalia",
        match_length=30,
        rules=DEFAULT_DOMAIN_RULES
    )
    print(f"Bee pollinates bee: valid={is_valid}, violations={violations}")
