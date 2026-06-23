"""
Kingdom Mapper

Maps species/taxon names to their biological kingdom using:
1. Hardcoded lookup for common taxa (orders, families, common names)
2. Heuristic rules based on naming patterns
3. Default fallback for unknown taxa

This avoids loading the massive OTT taxonomy files (2.5GB) while still
providing accurate kingdom classification for domain rule validation.
"""

import re
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# KNOWN TAXA MAPPINGS
# ============================================================================

# Major taxonomic groups mapped to kingdoms
KNOWN_TAXA: Dict[str, str] = {
    # === ANIMALIA ===
    # Arthropod orders
    'diptera': 'Animalia',
    'lepidoptera': 'Animalia',
    'coleoptera': 'Animalia',
    'hymenoptera': 'Animalia',
    'hemiptera': 'Animalia',
    'orthoptera': 'Animalia',
    'odonata': 'Animalia',
    'ephemeroptera': 'Animalia',
    'trichoptera': 'Animalia',
    'neuroptera': 'Animalia',
    'mecoptera': 'Animalia',
    'siphonaptera': 'Animalia',
    'phthiraptera': 'Animalia',
    'thysanoptera': 'Animalia',
    'dermaptera': 'Animalia',
    'blattodea': 'Animalia',
    'mantodea': 'Animalia',
    'isoptera': 'Animalia',
    'plecoptera': 'Animalia',

    # Arachnids
    'araneae': 'Animalia',
    'arachnida': 'Animalia',
    'acari': 'Animalia',
    'scorpiones': 'Animalia',
    'opiliones': 'Animalia',

    # Other invertebrates
    'nematoda': 'Animalia',
    'trematoda': 'Animalia',
    'cestoda': 'Animalia',
    'annelida': 'Animalia',
    'mollusca': 'Animalia',
    'gastropoda': 'Animalia',
    'bivalvia': 'Animalia',
    'crustacea': 'Animalia',
    'decapoda': 'Animalia',
    'isopoda': 'Animalia',
    'amphipoda': 'Animalia',
    'copepoda': 'Animalia',

    # Vertebrates
    'mammalia': 'Animalia',
    'aves': 'Animalia',
    'reptilia': 'Animalia',
    'amphibia': 'Animalia',
    'actinopterygii': 'Animalia',
    'chondrichthyes': 'Animalia',

    # Common names - animals
    'spider': 'Animalia',
    'spiders': 'Animalia',
    'insect': 'Animalia',
    'insects': 'Animalia',
    'fly': 'Animalia',
    'flies': 'Animalia',
    'moth': 'Animalia',
    'moths': 'Animalia',
    'butterfly': 'Animalia',
    'butterflies': 'Animalia',
    'beetle': 'Animalia',
    'beetles': 'Animalia',
    'wasp': 'Animalia',
    'wasps': 'Animalia',
    'bee': 'Animalia',
    'bees': 'Animalia',
    'ant': 'Animalia',
    'ants': 'Animalia',
    'mosquito': 'Animalia',
    'mosquitoes': 'Animalia',
    'midge': 'Animalia',
    'midges': 'Animalia',
    'tick': 'Animalia',
    'ticks': 'Animalia',
    'mite': 'Animalia',
    'mites': 'Animalia',
    'bird': 'Animalia',
    'birds': 'Animalia',
    'mammal': 'Animalia',
    'mammals': 'Animalia',
    'rodent': 'Animalia',
    'rodents': 'Animalia',
    'fish': 'Animalia',
    'fishes': 'Animalia',
    'amphibian': 'Animalia',
    'amphibians': 'Animalia',
    'reptile': 'Animalia',
    'reptiles': 'Animalia',
    'worm': 'Animalia',
    'worms': 'Animalia',
    'nematode': 'Animalia',
    'nematodes': 'Animalia',
    'snail': 'Animalia',
    'snails': 'Animalia',
    'slug': 'Animalia',
    'slugs': 'Animalia',
    'caterpillar': 'Animalia',
    'caterpillars': 'Animalia',
    'larva': 'Animalia',
    'larvae': 'Animalia',
    'predator': 'Animalia',
    'predators': 'Animalia',
    'prey': 'Animalia',
    'herbivore': 'Animalia',
    'herbivores': 'Animalia',
    'carnivore': 'Animalia',
    'carnivores': 'Animalia',
    'parasitoid': 'Animalia',
    'parasitoids': 'Animalia',

    # === PLANTAE ===
    # Plant divisions/phyla
    'magnoliophyta': 'Plantae',
    'angiosperm': 'Plantae',
    'angiosperms': 'Plantae',
    'gymnosperm': 'Plantae',
    'gymnosperms': 'Plantae',
    'pteridophyta': 'Plantae',
    'bryophyta': 'Plantae',

    # Common names - plants
    'plant': 'Plantae',
    'plants': 'Plantae',
    'tree': 'Plantae',
    'trees': 'Plantae',
    'flower': 'Plantae',
    'flowers': 'Plantae',
    'grass': 'Plantae',
    'grasses': 'Plantae',
    'shrub': 'Plantae',
    'shrubs': 'Plantae',
    'herb': 'Plantae',
    'herbs': 'Plantae',
    'fern': 'Plantae',
    'ferns': 'Plantae',
    'moss': 'Plantae',
    'mosses': 'Plantae',
    'algae': 'Plantae',  # Simplified - some are Protista
    'seaweed': 'Plantae',

    # === FUNGI ===
    'fungi': 'Fungi',
    'fungus': 'Fungi',
    'ascomycota': 'Fungi',
    'basidiomycota': 'Fungi',
    'zygomycota': 'Fungi',
    'mushroom': 'Fungi',
    'mushrooms': 'Fungi',
    'yeast': 'Fungi',
    'yeasts': 'Fungi',
    'mold': 'Fungi',
    'molds': 'Fungi',
    'lichen': 'Fungi',  # Simplified - actually symbiotic
    'lichens': 'Fungi',

    # === BACTERIA ===
    'bacteria': 'Bacteria',
    'bacterium': 'Bacteria',
    'proteobacteria': 'Bacteria',
    'firmicutes': 'Bacteria',
    'actinobacteria': 'Bacteria',
    'cyanobacteria': 'Bacteria',

    # === VIRUSES (not technically a kingdom but useful) ===
    'virus': 'Virus',
    'viruses': 'Virus',
    'viridae': 'Virus',
    'phage': 'Virus',
    'phages': 'Virus',
    'bacteriophage': 'Virus',

    # === PROTISTA ===
    'protist': 'Protista',
    'protists': 'Protista',
    'protozoa': 'Protista',
    'amoeba': 'Protista',
    'plasmodium': 'Protista',

    # === COMMON GENERA ===
    # Plant genera (trees, crops, common species)
    'quercus': 'Plantae',      # Oak
    'pinus': 'Plantae',        # Pine
    'acer': 'Plantae',         # Maple
    'betula': 'Plantae',       # Birch
    'fagus': 'Plantae',        # Beech
    'fraxinus': 'Plantae',     # Ash
    'salix': 'Plantae',        # Willow
    'populus': 'Plantae',      # Poplar
    'ulmus': 'Plantae',        # Elm
    'tilia': 'Plantae',        # Linden
    'malus': 'Plantae',        # Apple
    'prunus': 'Plantae',       # Cherry/Plum
    'rosa': 'Plantae',         # Rose
    'solanum': 'Plantae',      # Tomato/Potato
    'triticum': 'Plantae',     # Wheat
    'zea': 'Plantae',          # Corn
    'oryza': 'Plantae',        # Rice
    'arabidopsis': 'Plantae',  # Model plant
    'nicotiana': 'Plantae',    # Tobacco
    'helianthus': 'Plantae',   # Sunflower
    'taraxacum': 'Plantae',    # Dandelion
    'trifolium': 'Plantae',    # Clover
    'medicago': 'Plantae',     # Alfalfa
    'eucalyptus': 'Plantae',
    'acacia': 'Plantae',

    # Bacteria genera
    'escherichia': 'Bacteria',
    'salmonella': 'Bacteria',
    'staphylococcus': 'Bacteria',
    'streptococcus': 'Bacteria',
    'bacillus': 'Bacteria',
    'clostridium': 'Bacteria',
    'pseudomonas': 'Bacteria',
    'vibrio': 'Bacteria',
    'helicobacter': 'Bacteria',
    'mycobacterium': 'Bacteria',
    'listeria': 'Bacteria',
    'campylobacter': 'Bacteria',
    'legionella': 'Bacteria',
    'borrelia': 'Bacteria',
    'rickettsia': 'Bacteria',
    'chlamydia': 'Bacteria',
    'wolbachia': 'Bacteria',

    # Fungal genera
    'candida': 'Fungi',
    'aspergillus': 'Fungi',
    'penicillium': 'Fungi',
    'fusarium': 'Fungi',
    'trichoderma': 'Fungi',
    'saccharomyces': 'Fungi',
    'agaricus': 'Fungi',
    'amanita': 'Fungi',
    'boletus': 'Fungi',

    # Protist genera
    'trypanosoma': 'Protista',
    'leishmania': 'Protista',
    'giardia': 'Protista',
    'toxoplasma': 'Protista',
    'cryptosporidium': 'Protista',
}

# ============================================================================
# HEURISTIC PATTERNS
# ============================================================================

# Suffixes that indicate plant families/orders
PLANT_SUFFIXES = [
    'aceae',   # Plant families (Rosaceae, Fabaceae)
    'ales',    # Plant orders (Rosales)
    'phyta',   # Plant divisions (Magnoliophyta)
    'phyte',   # Plant types (epiphyte is organism, but -phyte often plant)
    'opsida',  # Plant classes (Magnoliopsida)
]

# Suffixes that indicate fungal groups
FUNGI_SUFFIXES = [
    'mycota',  # Fungal phyla (Ascomycota)
    'mycetes', # Fungal classes (Ascomycetes)
    'mycetidae', # Fungal subclasses
]

# Suffixes that indicate animal groups
ANIMAL_SUFFIXES = [
    'idae',    # Animal families (Canidae, Felidae)
    'inae',    # Animal subfamilies
    'oidea',   # Animal superfamilies
    'iformes', # Bird orders (Passeriformes)
    'morpha',  # Some animal groups
]

# Suffixes for bacteria
BACTERIA_SUFFIXES = [
    'bacteria',
    'bacillus',
    'coccus',
    'monas',
]


def get_kingdom(taxon_name: str) -> Optional[str]:
    """
    Map a taxon name to its biological kingdom.

    Uses a combination of:
    1. Direct lookup in known taxa dictionary
    2. Heuristic rules based on naming conventions
    3. Default to None for truly unknown taxa

    Args:
        taxon_name: Species name, genus, order, common name, etc.

    Returns:
        Kingdom name ('Animalia', 'Plantae', 'Fungi', 'Bacteria', 'Virus', 'Protista')
        or None if unknown
    """
    if not taxon_name:
        return None

    # Normalize: lowercase, strip whitespace
    name_lower = taxon_name.lower().strip()

    # 1. Direct lookup
    if name_lower in KNOWN_TAXA:
        return KNOWN_TAXA[name_lower]

    # For multi-word names, try the first word (genus)
    words = name_lower.split()
    if len(words) > 1:
        genus = words[0]
        if genus in KNOWN_TAXA:
            return KNOWN_TAXA[genus]

    # 2. Check suffixes (heuristics)
    for suffix in PLANT_SUFFIXES:
        if name_lower.endswith(suffix):
            return 'Plantae'

    for suffix in FUNGI_SUFFIXES:
        if name_lower.endswith(suffix):
            return 'Fungi'

    for suffix in ANIMAL_SUFFIXES:
        if name_lower.endswith(suffix):
            return 'Animalia'

    for suffix in BACTERIA_SUFFIXES:
        if name_lower.endswith(suffix):
            return 'Bacteria'

    # 3. Check if it looks like a binomial name (Genus species)
    # Most binomials in ecology papers are animals
    if re.match(r'^[A-Z][a-z]+\s+[a-z]+$', taxon_name.strip()):
        # Could be animal, plant, or fungi - default to Animalia
        # (most common in biotic interaction contexts)
        return 'Animalia'

    # 4. Unknown
    return None


def get_kingdom_with_confidence(taxon_name: str) -> tuple:
    """
    Get kingdom with confidence level.

    Args:
        taxon_name: Species/taxon name

    Returns:
        Tuple of (kingdom, confidence) where confidence is:
        - 'high': Direct lookup match
        - 'medium': Suffix-based heuristic
        - 'low': Binomial guess
        - None: Unknown
    """
    if not taxon_name:
        return None, None

    name_lower = taxon_name.lower().strip()

    # Direct lookup = high confidence
    if name_lower in KNOWN_TAXA:
        return KNOWN_TAXA[name_lower], 'high'

    # Check genus for multi-word
    words = name_lower.split()
    if len(words) > 1 and words[0] in KNOWN_TAXA:
        return KNOWN_TAXA[words[0]], 'high'

    # Suffix heuristics = medium confidence
    for suffix in PLANT_SUFFIXES:
        if name_lower.endswith(suffix):
            return 'Plantae', 'medium'

    for suffix in FUNGI_SUFFIXES:
        if name_lower.endswith(suffix):
            return 'Fungi', 'medium'

    for suffix in ANIMAL_SUFFIXES:
        if name_lower.endswith(suffix):
            return 'Animalia', 'medium'

    for suffix in BACTERIA_SUFFIXES:
        if name_lower.endswith(suffix):
            return 'Bacteria', 'medium'

    # Binomial = low confidence
    if re.match(r'^[A-Z][a-z]+\s+[a-z]+$', taxon_name.strip()):
        return 'Animalia', 'low'

    return None, None


def batch_get_kingdoms(taxon_names: list) -> Dict[str, Optional[str]]:
    """
    Get kingdoms for multiple taxa at once.

    Args:
        taxon_names: List of taxon names

    Returns:
        Dictionary mapping taxon name to kingdom
    """
    return {name: get_kingdom(name) for name in taxon_names}


if __name__ == "__main__":
    # Test examples
    test_cases = [
        # Animals
        ("Apis mellifera", "Animalia"),
        ("Canis lupus", "Animalia"),
        ("spiders", "Animalia"),
        ("Diptera", "Animalia"),
        ("Canidae", "Animalia"),
        ("parasitoid", "Animalia"),

        # Plants
        ("Quercus robur", "Plantae"),
        ("Rosaceae", "Plantae"),
        ("flowers", "Plantae"),
        ("Magnoliophyta", "Plantae"),

        # Fungi
        ("Ascomycota", "Fungi"),
        ("mushroom", "Fungi"),

        # Bacteria
        ("Escherichia coli", "Bacteria"),  # Will default to Animalia (binomial)
        ("Proteobacteria", "Bacteria"),

        # Unknown
        ("xyz123", None),
    ]

    print("=== Kingdom Mapper Tests ===\n")
    for taxon, expected in test_cases:
        result, confidence = get_kingdom_with_confidence(taxon)
        status = "✓" if result == expected else "✗"
        result_str = result or "None"
        expected_str = expected or "None"
        conf_str = confidence or "None"
        print(f"{status} {taxon:25} → {result_str:15} (expected: {expected_str}, conf: {conf_str})")
