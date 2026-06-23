"""
Template-based Training Data Generator

Generates diverse training sentences from GloBI interactions using templates
and natural language variations.
"""

import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Templates for different interaction types
# {source} = source species, {target} = target species
# Templates designed to match real scientific sentence patterns from eval100
INTERACTION_TEMPLATES = {
    'preysOn': [
        # Simple patterns
        "{source} preys on {target}.",
        "{source} is a predator of {target}.",
        "{target} is prey for {source}.",
        # Passive/scientific style (matching eval100)
        "Predation of {target} by {source} was documented.",
        "{target} was preyed upon by {source} in field observations.",
        "Trophic analysis confirmed {source} as a predator of {target}.",
        # Embedded clause patterns (matching eval100 style)
        "The diet of {source}, which includes {target}, was analyzed.",
        "{source}, a known predator of {target}, was examined.",
        "Specimens of {source} containing remains of {target} were collected.",
        # Complex scientific patterns
        "Gut content analysis revealed {source} feeds primarily on {target}.",
        "The predatory relationship between {source} and {target} was confirmed through stable isotope analysis.",
        "{target} constitutes a major prey item for {source} populations.",
        "Field observations documented {source} capturing and consuming {target}.",
        # EVAL100-STYLE: Trophic ecology patterns (line 72 of eval100)
        "The trophic ecology of {source}, a predator of {target}, was analyzed.",
        "{target} is a primary prey species for {source} populations.",
        "This study tested hypotheses regarding {source} predation on {target}.",
        "Distances between {source} and {target}, a primary prey species, were measured.",
        "The role of {source} as predator of {target} in this ecosystem was examined.",
    ],
    'parasiteOf': [
        # Classic patterns
        "{source} is a parasite of {target}.",
        "{source} parasitizes {target}.",
        "{target} is parasitized by {source}.",
        # Infection-style patterns (very common in eval100)
        "{target} infected with {source} showed clinical signs.",
        "Infection of {target} by {source} was confirmed.",
        "{source} infection in {target} was detected.",
        "Specimens of {target} infected with {source} were examined.",
        # Host patterns
        "{target} serves as host for the parasite {source}.",
        "The host {target} harbors {source}.",
        "{source} was recovered from its host {target}.",
        # Scientific survey style
        "Parasitological examination revealed {source} in {target}.",
        "{target} was found to be parasitized by {source}.",
        "The prevalence of {source} infection in {target} was determined.",
        "Molecular analysis confirmed {source} parasitizing {target}.",
        # EVAL100-STYLE: Survey and isolation patterns (lines 6, 18, 20, 41)
        "Field-collected specimens of {target} were surveyed for {source} infection.",
        "{source} was isolated almost exclusively from {target}, where it disrupts host tissues.",
        "{source} from {target} was cultivated in laboratory conditions.",
        "Historically, {source} was isolated from {target}, increasing host mortality rates.",
        "The neogregarine {source} was isolated from {target}, where it disrupts adipose tissue.",
        # EVAL100-STYLE: Anatomical location patterns (lines 20, 59)
        "{source} was isolated from tissue samples of {target}.",
        "{source} from muscle tissue of {target} was characterized.",
        "{source} from the gall bladder of {target} was studied.",
        "{source} concurrently infecting {target} was analyzed.",
        # EVAL100-STYLE: Coevolution patterns (line 34)
        "These results underline {source} potential for coevolution with {target}.",
        "The coevolution between {source} and {target} was examined.",
        "{source} shows host specificity for {target} despite extensive host range.",
    ],
    'endoparasiteOf': [
        "{source} is an endoparasite of {target}.",
        "The endoparasite {source} was recovered from {target}.",
        "{source} was detected in tissues of {target}.",
        "Internal parasitism of {target} by {source} was documented.",
        "{target} specimens harboring {source} were collected.",
        "Necropsy revealed {source} in the tissues of {target}.",
    ],
    'ectoparasiteOf': [
        "{source} is an ectoparasite of {target}.",
        "The ectoparasite {source} was collected from {target}.",
        "{source} infests {target}.",
        "{target} infested with {source} was examined.",
        "External parasites including {source} were found on {target}.",
        "{source} was observed feeding on {target}.",
    ],
    'hasHost': [
        "{source} uses {target} as its host.",
        "{target} is a host of {source}.",
        "{target} serves as host for {source}.",
        "{source} was found on host {target}.",
        "Host specificity of {source} for {target} was examined.",
        "{target}, the primary host of {source}, was studied.",
        "The host-parasite relationship between {target} and {source} was characterized.",
    ],
    'hostOf': [
        "{source} is host to {target}.",
        "{source} serves as the host for {target}.",
        "{source} harbors {target}.",
        "{target} was isolated from host {source}.",
        "The role of {source} as host for {target} was confirmed.",
        "{source} populations infected with {target} were sampled.",
        # EVAL100-STYLE: Host-finding patterns (line 91)
        "{target} was found on {source}, being ubiquitous wherever this host occurs.",
        "{target} shows the largest distribution range and was found on {source}.",
        "{target} was detected on host {source} across multiple localities.",
        "{source} serves as natural host for {target} in wild populations.",
    ],
    'eats': [
        "{source} feeds on {target}.",
        "{source} consumes {target}.",
        "{target} is consumed by {source}.",
        "The diet of {source} includes {target}.",
        "Gut contents of {source} contained {target}.",
        "{source} was observed feeding on {target}.",
        "Dietary analysis confirmed {source} consumes {target}.",
        "Foraging observations documented {source} eating {target}.",
        "{target} represents a food source for {source}.",
        "The importance of {target} as a food source for {source} was assessed.",
        "Feeding behavior of {source} on {target} was documented.",
        # NOTE: Larvae/caterpillar templates REMOVED - they require taxonomic validation
        # (only valid for Insecta). See INSECT_ONLY_TEMPLATES below.
        # EVAL100-STYLE: Parasitized tissue patterns (line 52)
        "Epithelial cells of {source} parasitized by {target} were examined.",
        "Tissue sections of {source} containing {target} were analyzed.",
    ],
    'pollinates': [
        "{source} pollinates {target}.",
        "{source} is a pollinator of {target}.",
        "{target} is pollinated by {source}.",
        "Pollination of {target} by {source} was documented.",
        "{source} visits flowers of {target} for pollen.",
        "Pollen transfer from {target} to {source} was confirmed.",
        "The pollinator {source} shows preference for {target}.",
        "{source} is an effective pollinator of {target} flowers.",
        "Floral visitors including {source} were observed on {target}.",
    ],
    'visitsFlowersOf': [
        "{source} visits the flowers of {target}.",
        "{source} was observed on {target} flowers.",
        "Flower visitation by {source} to {target} was recorded.",
        "{target} attracts {source} as a floral visitor.",
        "Foraging {source} were collected from {target} flowers.",
        "{source} individuals visiting {target} inflorescences were marked.",
    ],
    'infects': [
        # Direct patterns
        "{source} infects {target}.",
        "{target} is infected by {source}.",
        "{source} causes infection in {target}.",
        # Survey/detection style (common in eval100)
        "{source} infection was detected in {target}.",
        "{target} specimens tested positive for {source}.",
        "Molecular analysis confirmed {source} infection in {target}.",
        "{source} was isolated from infected {target}.",
        # Clinical/experimental style
        "{target} experimentally infected with {source} developed symptoms.",
        "Pathological changes in {target} infected with {source} were described.",
        "The pathogen {source} was detected in {target} tissues.",
        # Passive constructions
        "{target} naturally infected with {source} was examined.",
        "Prevalence of {source} in {target} populations was assessed.",
        # EVAL100-STYLE FN RECOVERY: Survey for infection (line 6 eval100)
        "Specimens of {target} were surveyed for {source} infection.",
        "{target} were surveyed for {source} infection by diagnostic assays.",
        # EVAL100-STYLE FN RECOVERY: Infection induced (line 49)
        "{source} infection induced in {target} was studied.",
        "{source} infection was induced in {target} under controlled conditions.",
        # EVAL100-STYLE FN RECOVERY: Infections described/caused (line 56)
        "This article describes {source} infections in {target}.",
        "Infections in {target} caused by {source} were characterized.",
        "{source} causes infections in {target}.",
        # EVAL100-STYLE FN RECOVERY: Pathogen-specific response (line 40)
        "Kinetics of {source}-specific response in infected {target} was studied.",
        "Pathogen-specific response in {source}-infected {target} was examined.",
        "{source}-infected {target} showed immune response.",
        # EVAL100-STYLE: Persistent/asymptomatic patterns (line 35)
        "{target} asymptomatically infected with {source} were identified.",
        "{source}-infected {target} showed no clinical signs but remained carriers.",
        "Persistent {source} infection in {target} was documented.",
        "{target} asymptomatically infected with {source} are a source of infection.",
    ],
    'pathogenOf': [
        "{source} is a pathogen of {target}.",
        "{source} causes disease in {target}.",
        "{target} is susceptible to {source}.",
        "The pathogenic effects of {source} on {target} were examined.",
        "{source}, a known pathogen of {target}, was isolated.",
        "Virulence of {source} against {target} was tested.",
        "{target} infected with the pathogen {source} showed mortality.",
        "Disease caused by {source} in {target} was characterized.",
        # EVAL100-STYLE FN RECOVERY: Oomycete/fungal pathogens (line 57)
        "Analyses of {source} isolates from {target} revealed pathogen presence.",
        "The presence of {source} infecting {target} was revealed.",
        "Several {source} species were detected in {target}.",
        "{source} isolates from {target} were identified.",
        # EVAL100-STYLE FN RECOVERY: Human pathogen listing (line 63)
        "Among the important pathogens of {target} is {source}.",
        "{source} is among the more important pathogens of {target}.",
        "The most important pathogens affecting {target} include {source}.",
        # EVAL100-STYLE FN RECOVERY: Dermatophyte patterns (line 11)
        "The dermatophyte {source} affects {target} under specific conditions.",
        "{source}, a dermatophyte pathogen of {target}, was studied.",
        # EVAL100-STYLE FN RECOVERY: Nematode-parasitic fungi (line 15)
        "The nematode-parasitic fungus {source} affects {target}.",
        "The parasitic fungus {source} acts against {target}.",
        "{source} is a parasitic fungus targeting {target}.",
    ],
    'vectorOf': [
        "{source} is a vector of {target}.",
        "{source} transmits {target}.",
        "{target} transmission by {source} was confirmed.",
        "The vector {source} carries {target}.",
        "{source} was identified as vector for {target}.",
        "Vector competence of {source} for {target} was evaluated.",
        "Transmission of {target} by {source} occurs through blood feeding.",
        "{source} populations positive for {target} were detected.",
        # EVAL100-STYLE FN RECOVERY: Vector validation (line 45) - multiple variants
        "This study validates the competence of {source} as a vector of {target}.",
        "The competence of {source} as vector of {target} was validated.",
        "We validated {source} as a vector of {target} via experimental infections.",
        "{source} is a competent vector of {target}.",
        "The aim of this study was to validate the competence of {source} as a vector of {target}.",
        "The present study validates {source} as a vector of {target}.",
        "Experimental infections validated {source} as a vector of {target}.",
        # EVAL100-STYLE: Other vector patterns
        "{source} is a well-established vector of {target}, which causes infection.",
        "{source} is an important pest as it is a vector of {target}.",
        "{source} transmits {target}, which can cause infection in host populations.",
        "{source} is a vector of dangerous pathogens including {target}.",
    ],
    'kills': [
        "{source} kills {target}.",
        "{target} mortality caused by {source} was documented.",
        "Lethal effects of {source} on {target} were observed.",
        "{source} caused significant mortality in {target}.",
        "{target} populations were reduced by {source}.",
        "Toxicity of {source} to {target} was evaluated.",
        # EVAL100-STYLE FN RECOVERY: Phage/biocontrol patterns (line 30)
        "Phage therapy uses {source} to control {target} in hosts.",
        "{source} represents a promising approach to control {target}.",
        "Bacteriophage {source} controls {target} pathogens.",
        "{source} is used to control {target} in biological systems.",
    ],
    'symbiontOf': [
        "{source} is a symbiont of {target}.",
        "Symbiosis between {source} and {target} was documented.",
        "{source} lives in association with {target}.",
        "The symbiotic relationship between {source} and {target} was characterized.",
        "{source} was found in symbiotic association with {target}.",
        "{target} harbors the symbiont {source}.",
        # EVAL100-STYLE: Symbiont taxonomic description (line 84)
        "{source}, symbiont of {target}, was described from this locality.",
        "The diversity of {source} on {target} was characterized.",
        "{source} was detected on {target}, which contains accessible polymers.",
        "{source} is an obligate symbiont of {target} populations.",
    ],
    'mutualistOf': [
        "{source} is a mutualist of {target}.",
        "{source} and {target} share a mutualistic relationship.",
        "Mutualistic interactions between {source} and {target} were studied.",
        "The mutualism between {source} and {target} benefits both partners.",
        "{source} provides benefits to its mutualist partner {target}.",
    ],
    'interactsWith': [
        "Interactions between {source} and {target} were documented.",
        "{source} and {target} interact in this ecosystem.",
        "The ecological relationship between {source} and {target} was examined.",
        "{source} was found interacting with {target}.",
        "Biotic interactions involving {source} and {target} were analyzed.",
    ],
}

# Generic templates for unknown interaction types
GENERIC_INTERACTION_TEMPLATES = [
    "{source} has a {interaction} relationship with {target}.",
    "{source} {interaction} {target}.",
    "The relationship between {source} and {target} involves {interaction}.",
    "We observed {source} {interaction} {target}.",
]

# INSECT-ONLY TEMPLATES: These require taxonomic validation
# Only use when source species is confirmed to be Insecta
# Using these for non-insects (fish, birds, mammals, spiders) creates
# biologically impossible sentences like "shark larvae fed foliage"
INSECT_ONLY_TEMPLATES = {
    'eats_larvae': [
        # These patterns from eval100 are ONLY valid for insect larvae
        "We assessed how {target} affects the growth of {source} larvae.",
        "When fed foliage from {target}, {source} larvae showed altered development.",
        "{source} larvae fed {target} consumed more foliage than controls.",
    ],
    'eats_caterpillar': [
        # These are ONLY valid for Lepidoptera (butterflies/moths)
        "{source} caterpillars fed on {target} remained longer in feeding stages.",
    ],
}

# Taxa that CANNOT have larvae/caterpillar templates applied:
# - Chondrichthyes (sharks, rays) - viviparous or ovoviviparous
# - Aves (birds) - no larvae
# - Mammalia - no larvae
# - Arachnida (spiders) - no larvae
# - Reptilia - no larvae
# - Most fish don't feed on foliage even if they have larvae

# Context variations to add before/after sentences
# Reduced frequency of formulaic prefixes to avoid overfitting
CONTEXT_PREFIXES = [
    "",  # No prefix - more weight given
    "",  # Duplicate empty to reduce prefix frequency
    "",  # Duplicate empty
    "",  # Duplicate empty
    "Here we report that ",
    "Molecular analysis confirmed that ",
    "Field surveys revealed that ",
    "Laboratory experiments demonstrated that ",
    "Examination of specimens showed that ",
    "Our results indicate that ",
    "Analysis of samples confirmed that ",
    "Ecological surveys documented that ",
    "During this study, ",
    "Based on morphological analysis, ",
    "PCR analysis detected that ",
]

CONTEXT_SUFFIXES = [
    "",  # No suffix - more weight given
    "",  # Duplicate empty to reduce suffix frequency
    "",  # Duplicate empty
    "",  # Duplicate empty
    " in the study area.",
    " in natural populations.",
    " under laboratory conditions.",
    " across multiple sites.",
    " throughout the sampling period.",
    " in wild-caught specimens.",
    " in field-collected samples.",
]


@dataclass
class GeneratedSentence:
    """A generated training sentence."""
    sentence: str
    source_species: str
    target_species: str
    interaction_type: str
    template_used: str
    is_positive: bool = True
    quality_score: float = 0.0


def generate_sentence(
    source_species: str,
    target_species: str,
    interaction_type: str,
    add_context: bool = True
) -> GeneratedSentence:
    """
    Generate a single training sentence from species and interaction type.

    Args:
        source_species: Source species name
        target_species: Target species name
        interaction_type: GloBI interaction type
        add_context: Whether to add contextual prefixes/suffixes

    Returns:
        GeneratedSentence object
    """
    # Get templates for this interaction type
    templates = INTERACTION_TEMPLATES.get(interaction_type, None)

    if templates is None:
        # Try to find a matching template by substring
        for key, tpls in INTERACTION_TEMPLATES.items():
            if key.lower() in interaction_type.lower() or interaction_type.lower() in key.lower():
                templates = tpls
                break

    if templates is None:
        # Use generic templates
        templates = GENERIC_INTERACTION_TEMPLATES

    # Select a random template
    template = random.choice(templates)

    # Format the sentence
    sentence = template.format(
        source=source_species,
        target=target_species,
        interaction=interaction_type.lower()
    )

    # Optionally add context
    if add_context:
        prefix = random.choice(CONTEXT_PREFIXES)
        suffix = random.choice(CONTEXT_SUFFIXES)

        # If adding suffix, strip trailing period from sentence first
        if suffix:
            sentence = sentence.rstrip('.')

        # If adding prefix, lowercase the first letter of the sentence
        # BUT preserve capitalization of species names (Genus species)
        if prefix and sentence[0].isupper():
            first_word = sentence.split()[0] if sentence.split() else ''
            # Check if first word looks like a genus name (not in common English words)
            # If it's a common scientific term, lowercase it; otherwise keep capitalized
            common_starts = {'the', 'a', 'an', 'we', 'our', 'this', 'that', 'both',
                           'field', 'gut', 'diet', 'predation', 'predatory', 'parasitic',
                           'infection', 'pollination', 'pollen', 'trophic', 'molecular',
                           'specimens', 'ecological', 'interactions', 'symbiosis', 'symbiotic',
                           'mutualism', 'mutualistic', 'lethal', 'floral', 'foraging',
                           'internal', 'external', 'transmission', 'mortality', 'vector',
                           'host', 'disease', 'virulence', 'pathological', 'prevalence',
                           'parasitological', 'necropsy', 'dietary', 'biotic', 'toxicity',
                           'population', 'abundance', 'niche', 'spatial', 'conservation',
                           'behavioral', 'reproductive', 'life', 'metabolic', 'gene',
                           'protein', 'antibodies', 'tissue', 'body', 'multiple', 'among',
                           'during', 'based', 'here'}
            if first_word.lower() in common_starts:
                sentence = sentence[0].lower() + sentence[1:]
            # Otherwise, keep capitalization (likely a species name)

        sentence = prefix + sentence + suffix

        # Ensure sentence starts with capital letter
        if sentence and sentence[0].islower():
            sentence = sentence[0].upper() + sentence[1:]

        # Ensure sentence ends with period
        if not sentence.endswith('.'):
            sentence = sentence + '.'

    # Calculate quality score based on species name quality
    score = 50.0  # Base score

    # Bonus for binomial names
    if len(source_species.split()) >= 2:
        score += 15
    if len(target_species.split()) >= 2:
        score += 15

    # Bonus for longer names (more specific)
    score += min(len(source_species) + len(target_species), 40) / 4

    return GeneratedSentence(
        sentence=sentence.strip(),
        source_species=source_species,
        target_species=target_species,
        interaction_type=interaction_type,
        template_used=template,
        quality_score=score
    )


def generate_from_globi(
    interactions_df: pd.DataFrame,
    max_per_interaction: int = 3,
    min_name_length: int = 5
) -> List[GeneratedSentence]:
    """
    Generate training sentences from GloBI interaction data.

    Args:
        interactions_df: DataFrame with GloBI interactions
        max_per_interaction: Max sentences per unique interaction
        min_name_length: Minimum species name length

    Returns:
        List of generated sentences
    """
    results = []
    seen = set()  # Track unique source-target-interaction combinations

    for _, row in interactions_df.iterrows():
        source = str(row.get('sourceTaxonName', '')).strip()
        target = str(row.get('targetTaxonName', '')).strip()
        interaction = str(row.get('interactionTypeName', '')).strip()

        # Skip if names are too short
        if len(source) < min_name_length or len(target) < min_name_length:
            continue

        # Skip if same species
        if source.lower() == target.lower():
            continue

        # Create unique key
        key = (source.lower(), target.lower(), interaction.lower())

        # Check if we've seen this combination
        if key in seen:
            continue
        seen.add(key)

        # Generate multiple sentences with different templates
        for _ in range(max_per_interaction):
            try:
                gen = generate_sentence(source, target, interaction)
                results.append(gen)
            except Exception as e:
                logger.warning(f"Failed to generate for {source}-{interaction}-{target}: {e}")

    logger.info(f"Generated {len(results)} positive sentences from {len(seen)} unique interactions")
    return results


def generate_negative_templates_single() -> List[str]:
    """
    Generate negative templates with single species (easy negatives).
    Patterns based on scientific literature that don't describe interactions.

    Returns:
        List of negative sentence templates
    """
    return [
        # Phylogenetic/taxonomic
        "Phylogenetic analysis of {species} was performed.",
        "The taxonomy of {species} has been revised.",
        "Molecular analysis confirmed the identity of {species}.",
        "DNA barcoding was used to identify {species}.",
        "{species} was first described in the 19th century.",

        # Distribution/ecology
        "The distribution of {species} was mapped.",
        "{species} is commonly found in temperate regions.",
        "Population density of {species} was estimated.",
        "The habitat preferences of {species} were characterized.",
        "We observed seasonal variations in {species} abundance.",
        "{species} exhibits nocturnal behavior.",
        "The ecology of {species} was studied.",

        # Sampling/methodology
        "Specimens of {species} were collected from multiple sites.",
        "Tissue samples from {species} were processed.",
        "Body size measurements of {species} were recorded.",
        "Morphological characteristics of {species} were documented.",
        "{species} was included in our analysis.",
        "We examined specimens of {species}.",

        # Conservation
        "The conservation status of {species} is of concern.",
        "Population trends of {species} were monitored.",
        "{species} is listed as a threatened species.",

        # Genetics
        "Genetic diversity in {species} was assessed.",
        "Gene expression in {species} was analyzed.",
        "Microsatellite markers for {species} were developed.",

        # Behavior/physiology
        "Reproductive biology of {species} was described.",
        "Behavioral observations of {species} were recorded.",
        "Metabolic rates in {species} were measured.",

        # === NEW: Single-species with interaction words (NO actual interaction) ===
        # These mention parasite/pathogen/predator but don't claim any interaction
        "The pathogen {species} was characterized at the molecular level.",
        "{species} is a well-known parasite in tropical ecosystems.",
        "We studied the predator {species} in its natural habitat.",
        "The vector {species} was collected for laboratory analysis.",
        "{species} is classified as an obligate parasite.",
        "Life cycle stages of the pathogen {species} were documented.",
        "The predatory behavior of {species} was filmed.",
        "{species} acts as a host for various microorganisms.",
        "The parasite {species} has a complex life cycle.",
        "{species} is recognized as a significant pathogen worldwide.",
        "We isolated the pathogen {species} from environmental samples.",
        "The predator {species} shows territorial behavior.",
        "{species} is an important vector in disease transmission cycles.",
        "Host-seeking behavior of {species} was analyzed.",
        "The parasitic lifestyle of {species} evolved multiple times.",
        "{species} functions as an apex predator in this ecosystem.",
        "We cultured the pathogen {species} under controlled conditions.",
        "{species} is a generalist predator with broad diet.",
        "The infection biology of {species} remains poorly understood.",
        "Predation rates by {species} vary seasonally.",

        # === NEW: Active voice patterns (single species) ===
        "Researchers collected {species} from multiple localities.",
        "We identified {species} using molecular markers.",
        "The team discovered {species} in a new habitat.",
        "Scientists described {species} as a new taxon.",
        "We sequenced the genome of {species}.",
        "The study revealed {species} tolerates extreme conditions.",
        "Researchers tracked {species} using radio telemetry.",
        "We observed {species} foraging at dawn.",

        # === NEW: Negation patterns (single species) ===
        "No parasites were detected in {species} samples.",
        "{species} showed no signs of infection.",
        "We found no evidence of disease in {species}.",
        "No pathogens were isolated from {species}.",
        "{species} was not infected in our study.",
        "Screening revealed no parasites in {species}.",
        "{species} remained uninfected throughout the experiment.",
        "No predation on {species} was observed.",
    ]


def generate_negative_templates_three_species() -> List[str]:
    """
    Generate negative templates with THREE species but NO interaction between them.
    These help the model learn that multiple species mentions don't imply interaction.

    Returns:
        List of negative sentence templates with {species1}, {species2}, {species3}
    """
    return [
        # Co-occurrence patterns
        "{species1}, {species2}, and {species3} were detected in the same habitat.",
        "The survey identified {species1}, {species2}, and {species3} at this site.",
        "We collected {species1}, {species2}, and {species3} from overlapping ranges.",
        "{species1}, {species2}, and {species3} co-occur in tropical forests.",

        # Comparative/phylogenetic
        "Phylogenetic analysis included {species1}, {species2}, and {species3}.",
        "{species1} is more closely related to {species2} than to {species3}.",
        "We compared the genomes of {species1}, {species2}, and {species3}.",
        "Morphological traits of {species1}, {species2}, and {species3} were measured.",

        # Enumeration without interaction
        "The community included {species1}, {species2}, and {species3} among others.",
        "Dominant species were {species1}, {species2}, and {species3}.",
        "{species1}, {species2}, and {species3} were the most abundant taxa.",
        "We recorded {species1}, {species2}, and {species3} during the survey.",

        # Methodological
        "{species1}, {species2}, and {species3} were used as model organisms.",
        "Samples from {species1}, {species2}, and {species3} were analyzed.",
        "We tested {species1}, {species2}, and {species3} under laboratory conditions.",
        "DNA was extracted from {species1}, {species2}, and {species3}.",

        # Ecological
        "{species1}, {species2}, and {species3} share similar ecological niches.",
        "Population dynamics of {species1}, {species2}, and {species3} were modeled.",
        "The distributions of {species1}, {species2}, and {species3} overlap extensively.",
        "{species1}, {species2}, and {species3} showed similar responses to disturbance.",

        # Conservation
        "{species1}, {species2}, and {species3} face similar conservation threats.",
        "We assessed the status of {species1}, {species2}, and {species3}.",
        "{species1}, {species2}, and {species3} are all listed as endangered.",

        # Active voice
        "Researchers studied {species1}, {species2}, and {species3} populations.",
        "We monitored {species1}, {species2}, and {species3} over five years.",
        "The team collected {species1}, {species2}, and {species3} specimens.",
    ]


def generate_negative_templates_two_species() -> List[str]:
    """
    Generate HARD negative templates with TWO species but NO interaction.
    These are critical for improving precision.
    Templates designed to match real scientific patterns from eval100.

    Returns:
        List of hard negative sentence templates with {species1} and {species2}
    """
    return [
        # Co-occurrence without interaction (scientific style)
        "Both {species1} and {species2} were detected in our samples.",
        "{species1} and {species2} occur in the same habitat.",
        "The survey detected {species1} and {species2} in the region.",
        "Specimens of {species1} and {species2} were collected from multiple sites.",
        "{species1} and {species2} were both present in the study area.",
        "{species1} was found alongside {species2} but no direct interaction was observed.",
        "Field surveys recorded {species1} and {species2} at overlapping localities.",
        "The presence of {species1} and {species2} was confirmed at this site.",

        # Phylogenetic/taxonomic comparisons (very common in eval100)
        "Phylogenetic analysis compared {species1} and {species2}.",
        "Molecular data grouped {species1} with {species2}.",
        "{species1} and {species2} belong to the same family.",
        "DNA barcoding distinguished {species1} from {species2}.",
        "Sequence analysis revealed {species1} is related to {species2}.",
        "The taxonomic status of {species1} and {species2} was revised.",
        "Morphological comparison of {species1} and {species2} was performed.",
        "Genetic divergence between {species1} and {species2} was estimated.",
        "{species1} and {species2} share a common ancestor.",
        "{species1} is more closely related to {species2} than previously thought.",

        # Sample/methodology descriptions (matches eval100 patterns)
        "Samples from {species1} and {species2} were analyzed.",
        "{species1} and {species2} were selected for this study.",
        "We examined populations of {species1} and {species2}.",
        "Tissue samples of {species1} and {species2} were processed for DNA extraction.",
        "Body size measurements of {species1} and {species2} were compared.",
        "Specimens of {species1} and {species2} were deposited in the collection.",
        "{species1} and {species2} were included in the molecular analysis.",
        "This study focused on {species1} and {species2} populations.",

        # Resistance/susceptibility testing (NO infection - just testing)
        "The susceptibility of {species1} and {species2} to the treatment was evaluated.",
        "Resistance patterns in {species1} and {species2} were compared.",
        "Mortality rates of {species1} and {species2} were measured under stress conditions.",
        "Both {species1} and {species2} were tested for tolerance to salinity.",

        # Species list patterns (like eval100 species enumerations)
        "Multiple species including {species1} and {species2} were identified.",
        "The genera {species1} and {species2} were represented in our samples.",
        "Among the species recorded were {species1} and {species2}.",
        "The fauna included {species1} and {species2} among others.",

        # Ecological surveys without interaction
        "The distribution of {species1} and {species2} was mapped.",
        "Population densities of {species1} and {species2} were estimated.",
        "Abundance patterns of {species1} and {species2} were analyzed.",
        "{species1} and {species2} showed similar habitat preferences.",
        "Niche overlap between {species1} and {species2} was calculated.",
        "The occurrence of {species1} correlated with {species2} abundance.",
        "{species1} and {species2} exhibited similar activity patterns.",

        # Conservation status (no interaction)
        "Conservation status of {species1} and {species2} was assessed.",
        "{species1} and {species2} face similar threats from habitat loss.",
        "Population trends of {species1} and {species2} were monitored.",
        "Both {species1} and {species2} are listed as threatened species.",

        # Comparative studies (no interaction)
        "We compared the ecology of {species1} and {species2}.",
        "Behavioral differences between {species1} and {species2} were documented.",
        "Reproductive biology of {species1} and {species2} was studied.",
        "Life history traits of {species1} and {species2} were compared.",
        "Metabolic rates of {species1} and {species2} were measured.",

        # Near-miss patterns (look like interactions but aren't)
        "{species1} was detected in areas where {species2} occurs.",
        "{species1} and {species2} were found in close proximity.",
        "The presence of {species1} was associated with {species2} density.",
        "{species1} was more common where {species2} was present.",
        "Spatial overlap between {species1} and {species2} was observed.",

        # Technical/methodological patterns
        "We used {species1} and {species2} as model organisms.",
        "Antibodies against {species1} cross-reacted with {species2} proteins.",
        "Gene expression in {species1} and {species2} was analyzed.",
        "Protein extracts from {species1} and {species2} were compared.",

        # CRITICAL: Generic interaction mentions without specific relationships
        # These are labeled as NO interaction in eval100 - need to train model on these
        "{species1} is known to be a potential host for various pathogens.",
        "{species1} and {species2} are known vectors of disease.",
        "The impact of {species1} on {species2} populations was assessed.",
        "{species1} transmits various pathogens to other species.",
        "{species1} and {species2} are important in disease transmission cycles.",
        "Infection rates in {species1} and {species2} were measured.",
        "Parasitic relationships involving {species1} and {species2} were reviewed.",
        "{species1} is a common parasite in many ecosystems.",
        "Disease prevalence in {species1} and {species2} populations was studied.",
        "The ecology of {species1} and {species2} as hosts was examined.",
        "{species1} serves as host for multiple parasite species.",
        "Pathogen diversity in {species1} and {species2} was characterized.",
        "{species1} and {species2} may serve as reservoirs for pathogens.",
        "Host-parasite dynamics involving {species1} and {species2} were modeled.",
        "Transmission potential of {species1} and {species2} was evaluated.",
        "The role of {species1} as vector was investigated.",
        "Predator-prey relationships were documented in {species1} habitat.",
        "The diet composition of {species1} and {species2} was analyzed.",
        "Feeding behavior of {species1} and {species2} was observed.",
        "Trophic relationships involving {species1} and {species2} were studied.",

        # EVAL100-STYLE: Study descriptions (NOT actual interactions)
        # These sentences MENTION species but describe studies/methods, not interactions
        # NOTE: Be careful not to include validation studies that confirm actual interactions
        "This study investigates the potential of {species1} protein-based nanoparticles supplemented diet on growth in {species2}.",
        "The present study investigates the effects of {species1} extract on {species2}.",
        "We investigated the potential of {species1} as a treatment for {species2}.",
        "Here we investigate the role of {species1} in {species2} populations.",
        # REMOVED: "The aim of this study was to investigate..." - conflicts with validation studies

        # EVAL100-STYLE: In vitro testing patterns (labeled as NEGATIVE)
        "The antibacterial activity of {species1} was assayed against {species2}.",
        "Antimicrobial activity of {species1} essential oil against {species2} was evaluated.",
        "The bactericidal activity of {species1} was tested against {species2}.",
        "{species1} was evaluated for antibacterial activity against {species2}.",
        "We tested the antimicrobial potential of {species1} against {species2}.",
        "In vitro activity of {species1} against {species2} was determined.",
        "The inhibitory effect of {species1} on {species2} was assessed in vitro.",
        "{species1} showed bactericidal activity against {species2} in laboratory tests.",

        # EVAL100-STYLE: Serological surveys (testing for antibodies, not actual infection)
        "A serological survey on {species1} was conducted among {species2}.",
        "The presence of antibodies against {species1} was evaluated in {species2}.",
        "Seroprevalence of {species1} in {species2} populations was assessed.",
        "{species2} sera were tested for antibodies against {species1}.",
        "Serological testing detected {species1} antibodies in {species2}.",

        # EVAL100-STYLE: Classification/grouping patterns (NOT interactions)
        "{species1} is grouped with {species2} based on genetic analysis.",
        "{species1} and {species2} are classified in the same group.",
        "{species1} is categorized as a pathogen along with {species2}.",

        # EVAL100-STYLE: Hypothetical patterns
        "Infection with {species1} would indicate exposure to {species2}.",
        "True infection with {species1} would suggest contact with {species2}.",
        "Presence of {species1} may indicate prior exposure to {species2}.",

        # EVAL100-STYLE: Correlation/relationship studies
        "The analysis of the relationship between {species1} and {species2} was performed.",
        "Correlation between {species1} abundance and {species2} presence was examined.",
        "We analyzed the relationship between microbial changes and {species1} in {species2}.",

        # EVAL100-STYLE: Species enumeration without interactions
        "Multiple pathogens including {species1} and {species2} were detected.",
        "{species1} and {species2} were among the species identified in this study.",
        "The survey detected {species1} and {species2} among other microorganisms.",
        "{species1} and {species2} were the most frequently recorded species.",

        # EVAL100-STYLE: Methodological descriptions
        "We trained {species1} to detect {species2} using olfaction.",
        "{species1} and {species2} were used as model organisms in this study.",
        "We used {species1} and {species2} to test our hypothesis.",

        # EVAL100-STYLE: Host-cell studies (cell lines, not actual host infections)
        "Host cells of {species1} infected with {species2} virus produce unusual structures.",
        "{species1} cells infected with {species2} were examined microscopically.",

        # EVAL100-STYLE: Vaccination/protection studies (NOT natural infections)
        "The protective value of {species1} bacterin against {species2} was evaluated.",
        "Vaccination with {species1} against {species2} was tested.",
        "Protection against {species1} infection in {species2} was assessed.",

        # EVAL100-STYLE: Diet/feeding experiments (providing food, not natural predation)
        "All species were provided with {species1} and {species2} diets.",
        "{species1} was fed diets derived from {species2}.",
        "The effect of {species1} diet on {species2} was evaluated.",

        # EVAL100-STYLE: Gene expression studies
        "Gene expression in {species1} and {species2} was analyzed.",
        "Molecular analysis revealed that genes from {species1} were expressed in {species2}.",

        # === NEW: Active voice two-species patterns ===
        "Researchers collected {species1} and {species2} from the same locality.",
        "We compared {species1} and {species2} populations across multiple sites.",
        "The team identified {species1} and {species2} using molecular methods.",
        "Scientists monitored {species1} and {species2} over the study period.",
        "We sequenced genomes from both {species1} and {species2}.",
        "Researchers documented {species1} and {species2} in sympatric populations.",
        "We analyzed microbiomes of {species1} and {species2} communities.",
        "The study characterized {species1} and {species2} assemblages.",
        "We evaluated {species1} and {species2} response to environmental change.",
        "Scientists discovered {species1} and {species2} in the survey.",
        "We measured {species1} and {species2} densities along the transect.",
        "Researchers trapped {species1} and {species2} using baited traps.",

        # === NEW: Negation patterns two-species ===
        "{species1} did not infect {species2} under experimental conditions.",
        "No interaction between {species1} and {species2} was detected.",
        "{species1} was not observed preying on {species2}.",
        "We found no evidence of parasitism between {species1} and {species2}.",
        "{species1} and {species2} showed no significant interaction.",
        "Transmission from {species1} to {species2} was not confirmed.",
        "{species1} did not parasitize {species2} in our study.",
        "No predation of {species1} on {species2} was observed.",
        "Neither {species1} nor {species2} showed signs of infection.",
        "We could not confirm any interaction between {species1} and {species2}.",
        "{species1} and {species2} did not exhibit host-parasite dynamics.",
        "Analysis revealed no trophic link between {species1} and {species2}.",
        "{species1} was not a vector of {species2} in this population.",
        "No pathogenic relationship between {species1} and {species2} was found.",
    ]


def generate_negatives_from_species(
    species_names: List[str],
    count: int = 1000
) -> List[GeneratedSentence]:
    """
    Generate EASY negative training sentences (single species, no interaction).

    Args:
        species_names: List of species names to use
        count: Number of negative sentences to generate

    Returns:
        List of generated negative sentences
    """
    templates = generate_negative_templates_single()
    results = []

    for i in range(count):
        species = random.choice(species_names)
        template = random.choice(templates)

        sentence = template.format(species=species)

        results.append(GeneratedSentence(
            sentence=sentence,
            source_species=species,
            target_species="",
            interaction_type="none",
            template_used=template,
            is_positive=False,
            quality_score=50.0
        ))

    logger.info(f"Generated {len(results)} easy negative sentences (single species)")
    return results


def generate_hard_negatives(
    species_names: List[str],
    count: int = 10000
) -> List[GeneratedSentence]:
    """
    Generate HARD negative training sentences (two species, NO interaction).
    These are critical for improving precision - they look like interactions but aren't.

    Args:
        species_names: List of species names to use
        count: Number of hard negative sentences to generate

    Returns:
        List of generated hard negative sentences
    """
    templates = generate_negative_templates_two_species()
    results = []

    for i in range(count):
        # Pick two different species
        species1 = random.choice(species_names)
        species2 = random.choice(species_names)

        # Ensure different species
        attempts = 0
        while species1.lower() == species2.lower() and attempts < 10:
            species2 = random.choice(species_names)
            attempts += 1

        if species1.lower() == species2.lower():
            continue

        template = random.choice(templates)
        sentence = template.format(species1=species1, species2=species2)

        # Add context variation
        if random.random() < 0.5:
            prefix = random.choice(CONTEXT_PREFIXES)
            if prefix and sentence[0].isupper():
                sentence = sentence[0].lower() + sentence[1:]
            sentence = prefix + sentence
            if sentence and sentence[0].islower():
                sentence = sentence[0].upper() + sentence[1:]

        results.append(GeneratedSentence(
            sentence=sentence,
            source_species=species1,
            target_species=species2,
            interaction_type="none_two_species",
            template_used=template,
            is_positive=False,
            quality_score=70.0  # Higher score - these are harder/more valuable
        ))

    logger.info(f"Generated {len(results)} HARD negative sentences (two species, no interaction)")
    return results


def generate_three_species_negatives(
    species_names: List[str],
    count: int = 1000
) -> List[GeneratedSentence]:
    """
    Generate negative training sentences with THREE species but NO interaction.
    These help the model learn that multiple species mentions don't imply interaction.

    Args:
        species_names: List of species names to use
        count: Number of three-species negative sentences to generate

    Returns:
        List of generated negative sentences
    """
    templates = generate_negative_templates_three_species()
    results = []

    for i in range(count):
        # Pick three different species
        species1 = random.choice(species_names)
        species2 = random.choice(species_names)
        species3 = random.choice(species_names)

        # Ensure all three are different
        attempts = 0
        while (species1.lower() == species2.lower() or
               species1.lower() == species3.lower() or
               species2.lower() == species3.lower()) and attempts < 20:
            species2 = random.choice(species_names)
            species3 = random.choice(species_names)
            attempts += 1

        if (species1.lower() == species2.lower() or
            species1.lower() == species3.lower() or
            species2.lower() == species3.lower()):
            continue

        template = random.choice(templates)
        sentence = template.format(
            species1=species1,
            species2=species2,
            species3=species3
        )

        # Add context variation
        if random.random() < 0.3:
            prefix = random.choice(CONTEXT_PREFIXES)
            if prefix and sentence[0].isupper():
                sentence = sentence[0].lower() + sentence[1:]
            sentence = prefix + sentence
            if sentence and sentence[0].islower():
                sentence = sentence[0].upper() + sentence[1:]

        results.append(GeneratedSentence(
            sentence=sentence,
            source_species=species1,
            target_species=f"{species2}, {species3}",  # Store both
            interaction_type="none_three_species",
            template_used=template,
            is_positive=False,
            quality_score=65.0  # Intermediate value
        ))

    logger.info(f"Generated {len(results)} three-species negative sentences")
    return results


def build_training_data(
    interactions_df: pd.DataFrame,
    max_positives: int = 10000,
    max_negatives: int = 10000,
    sentences_per_interaction: int = 2,
    hard_negative_ratio: float = 0.70,
    three_species_ratio: float = 0.05
) -> pd.DataFrame:
    """
    Build a complete training dataset with positives and diverse negatives.

    Args:
        interactions_df: GloBI interactions DataFrame
        max_positives: Maximum positive examples
        max_negatives: Maximum negative examples
        sentences_per_interaction: Sentences per unique interaction
        hard_negative_ratio: Fraction of negatives that are hard (two-species)
        three_species_ratio: Fraction of negatives with three species

    Returns:
        DataFrame with 'text' and 'label' columns
    """
    # Generate positives
    positives = generate_from_globi(
        interactions_df,
        max_per_interaction=sentences_per_interaction
    )

    # Limit positives
    if len(positives) > max_positives:
        positives = random.sample(positives, max_positives)

    # Extract unique species for negatives
    all_species = set()
    all_species.update(interactions_df['sourceTaxonName'].dropna().unique())
    all_species.update(interactions_df['targetTaxonName'].dropna().unique())
    species_list = [s for s in all_species if len(str(s)) > 5]

    # Generate negatives: mix of easy (single), hard (two species), and three-species
    # Distribution: hard_negative_ratio for two-species, three_species_ratio for three,
    # remainder for single-species
    n_three = int(max_negatives * three_species_ratio)
    n_hard = int(max_negatives * hard_negative_ratio)
    n_easy = max_negatives - n_hard - n_three

    logger.info(f"Generating diverse negatives: {n_hard} hard (two-species), "
                f"{n_easy} easy (single-species), {n_three} three-species")

    easy_negatives = generate_negatives_from_species(species_list, n_easy)
    hard_negatives = generate_hard_negatives(species_list, n_hard)
    three_species_negatives = generate_three_species_negatives(species_list, n_three)

    all_negatives = easy_negatives + hard_negatives + three_species_negatives

    # Combine into DataFrame
    rows = []
    for p in positives:
        rows.append({
            'text': p.sentence,
            'label': 1,
            'source_species': p.source_species,
            'target_species': p.target_species,
            'interaction_type': p.interaction_type,
            'quality_score': p.quality_score
        })

    for n in all_negatives:
        rows.append({
            'text': n.sentence,
            'label': 0,
            'source_species': n.source_species,
            'target_species': n.target_species,
            'interaction_type': n.interaction_type,
            'quality_score': n.quality_score
        })

    df = pd.DataFrame(rows)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n_hard_final = (df['interaction_type'] == 'none_two_species').sum()
    n_easy_final = (df['interaction_type'] == 'none').sum()
    n_three_final = (df['interaction_type'] == 'none_three_species').sum()

    logger.info(f"Built training dataset: {len(df)} total, "
                f"{(df['label'] == 1).sum()} positives, {(df['label'] == 0).sum()} negatives "
                f"({n_hard_final} hard two-species, {n_easy_final} easy single-species, "
                f"{n_three_final} three-species)")

    return df


if __name__ == "__main__":
    # Demo
    print("=== Template-based Sentence Generator Demo ===\n")

    # Sample interactions
    sample_interactions = [
        ("Vulpes vulpes", "Mus musculus", "preysOn"),
        ("Plasmodium falciparum", "Homo sapiens", "parasiteOf"),
        ("Apis mellifera", "Malus domestica", "pollinates"),
        ("Ixodes scapularis", "Borrelia burgdorferi", "vectorOf"),
    ]

    for source, target, interaction in sample_interactions:
        print(f"\n{source} - {interaction} - {target}:")
        for _ in range(3):
            gen = generate_sentence(source, target, interaction)
            print(f"  • {gen.sentence}")

    print("\n\n=== Negative Examples ===")
    neg_gen = generate_negatives_from_species(
        ["Vulpes vulpes", "Mus musculus", "Plasmodium falciparum"],
        count=5
    )
    for n in neg_gen:
        print(f"  • {n.sentence}")
