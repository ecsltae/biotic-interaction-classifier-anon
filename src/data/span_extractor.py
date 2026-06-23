#!/usr/bin/env python3
"""
Species Span Extraction and Entity Tagging for LUKE Model.

Uses TaxoNERD for robust taxonomic named entity recognition,
with regex fallback for simple cases.

The main output is text with entity markers:
    "The wolf <e1>Canis lupus</e1> preys on <e2>Ovis aries</e2> in mountains."
"""

import re
from typing import List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import warnings

# Try to import TaxoNERD
try:
    from taxonerd import TaxoNERD
    TAXONERD_AVAILABLE = True
except ImportError:
    TAXONERD_AVAILABLE = False
    warnings.warn("TaxoNERD not installed. Install with: pip install taxonerd")


@dataclass
class SpeciesSpan:
    """A detected species mention with character offsets."""
    text: str
    start: int
    end: int
    score: float = 1.0


class SpanExtractor:
    """
    Extract species mentions from text using TaxoNERD or regex fallback.
    """

    # Regex pattern for binomial nomenclature: Genus species
    # Matches: Canis lupus, C. lupus, Escherichia coli
    # Does NOT match: "The wolf", common names, or verbs after genus
    BINOMIAL_PATTERN = re.compile(
        r'\b([A-Z][a-z]{2,})\s+([a-z]{3,})\b'  # Full: Genus species (min 3 chars for genus, 3 for species)
        r'|\b([A-Z])\.\s*([a-z]{3,})\b'  # Abbreviated: G. species
    )

    # Pattern for species with qualifiers (Genus sp., Genus spp.)
    # Requires genus to be at least 3 chars to avoid matching common words
    SPECIES_QUALIFIER_PATTERN = re.compile(
        r'\b([A-Z][a-z]{2,})\s+(sp\.|spp\.|cf\.|aff\.)\b'
    )

    # Common words to exclude (not species names)
    EXCLUDED_WORDS = {
        'the', 'this', 'that', 'these', 'those', 'their',
        'from', 'with', 'into', 'onto', 'upon',
        'wolf', 'bear', 'fish', 'bird', 'snake', 'frog',  # common names
    }

    def __init__(self, use_taxonerd: bool = True, taxonerd_model: str = "en_ner_eco_biobert"):
        """
        Initialize the span extractor.

        Args:
            use_taxonerd: Whether to use TaxoNERD (recommended for accuracy)
            taxonerd_model: TaxoNERD model to use. Options:
                - "en_ner_eco_biobert" (default, best accuracy)
                - "en_ner_eco_md" (faster, good accuracy)
        """
        self.use_taxonerd = use_taxonerd and TAXONERD_AVAILABLE
        self.taxonerd = None
        self.taxonerd_model = taxonerd_model

        if self.use_taxonerd:
            self._init_taxonerd()

    def _init_taxonerd(self):
        """Initialize TaxoNERD model (lazy loading)."""
        if self.taxonerd is None and TAXONERD_AVAILABLE:
            try:
                self.taxonerd = TaxoNERD(prefer_gpu=False)
                self.taxonerd.load(self.taxonerd_model)
            except Exception as e:
                warnings.warn(f"Failed to initialize TaxoNERD: {e}. Using regex fallback.")
                self.use_taxonerd = False

    def extract_species(self, text: str, max_species: int = 10) -> List[SpeciesSpan]:
        """
        Extract species mentions from text.

        Args:
            text: Input text
            max_species: Maximum number of species to return

        Returns:
            List of SpeciesSpan objects sorted by position
        """
        if self.use_taxonerd and self.taxonerd is not None:
            return self._extract_with_taxonerd(text, max_species)
        else:
            return self._extract_with_regex(text, max_species)

    def _extract_with_taxonerd(self, text: str, max_species: int) -> List[SpeciesSpan]:
        """Extract species using TaxoNERD."""
        try:
            # TaxoNERD returns a DataFrame with columns: offsets, text, entity, sent
            result = self.taxonerd.find_in_text(text)

            if result is None or len(result) == 0:
                # Fallback to regex if TaxoNERD finds nothing
                return self._extract_with_regex(text, max_species)

            spans = []
            for _, row in result.iterrows():
                # Parse offsets (format: "start end")
                if isinstance(row['offsets'], str):
                    parts = row['offsets'].split()
                    if len(parts) >= 2:
                        start, end = int(parts[0]), int(parts[1])
                    else:
                        continue
                else:
                    # Handle tuple/list format
                    start, end = int(row['offsets'][0]), int(row['offsets'][1])

                spans.append(SpeciesSpan(
                    text=row['text'],
                    start=start,
                    end=end,
                    score=0.9  # TaxoNERD doesn't provide confidence scores
                ))

            # Sort by position and limit
            spans = sorted(spans, key=lambda x: x.start)[:max_species]
            return spans

        except Exception as e:
            warnings.warn(f"TaxoNERD extraction failed: {e}. Using regex fallback.")
            return self._extract_with_regex(text, max_species)

    def _extract_with_regex(self, text: str, max_species: int) -> List[SpeciesSpan]:
        """Extract species using regex patterns (fallback)."""
        spans = []
        seen_positions = set()

        # Find binomial names
        for match in self.BINOMIAL_PATTERN.finditer(text):
            start, end = match.start(), match.end()

            # Avoid overlaps
            if any(start < seen_end and end > seen_start
                   for seen_start, seen_end in seen_positions):
                continue

            species_text = match.group(0)

            # Skip if first word is in excluded list (case-insensitive)
            first_word = species_text.split()[0].lower()
            if first_word in self.EXCLUDED_WORDS:
                continue

            spans.append(SpeciesSpan(
                text=species_text,
                start=start,
                end=end,
                score=0.8
            ))
            seen_positions.add((start, end))

        # Find species with qualifiers (sp., spp.)
        for match in self.SPECIES_QUALIFIER_PATTERN.finditer(text):
            start, end = match.start(), match.end()

            if any(start < seen_end and end > seen_start
                   for seen_start, seen_end in seen_positions):
                continue

            species_text = match.group(0)
            spans.append(SpeciesSpan(
                text=species_text,
                start=start,
                end=end,
                score=0.7
            ))
            seen_positions.add((start, end))

        # Sort by position and limit
        spans = sorted(spans, key=lambda x: x.start)[:max_species]
        return spans


def tag_entities(
    text: str,
    species1_span: Optional[SpeciesSpan] = None,
    species2_span: Optional[SpeciesSpan] = None,
    e1_start: str = "<e1>",
    e1_end: str = "</e1>",
    e2_start: str = "<e2>",
    e2_end: str = "</e2>"
) -> str:
    """
    Add entity markers around species mentions.

    Args:
        text: Original text
        species1_span: First species span (or None to auto-detect)
        species2_span: Second species span (or None to auto-detect)
        e1_start, e1_end: Markers for first entity
        e2_start, e2_end: Markers for second entity

    Returns:
        Text with entity markers added
    """
    if species1_span is None or species2_span is None:
        return text  # Can't tag without both species

    # Ensure species1 comes before species2 in text
    if species1_span.start > species2_span.start:
        species1_span, species2_span = species2_span, species1_span
        e1_start, e1_end, e2_start, e2_end = e2_start, e2_end, e1_start, e1_end

    # Build tagged text (work from end to start to preserve offsets)
    result = text

    # Insert second entity markers first (preserves first entity offsets)
    result = (
        result[:species2_span.start] +
        e2_start +
        result[species2_span.start:species2_span.end] +
        e2_end +
        result[species2_span.end:]
    )

    # Insert first entity markers
    result = (
        result[:species1_span.start] +
        e1_start +
        result[species1_span.start:species1_span.end] +
        e1_end +
        result[species1_span.end:]
    )

    return result


def extract_and_tag(
    text: str,
    extractor: Optional[SpanExtractor] = None,
    return_species: bool = False
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Extract species from text and add entity markers.

    Args:
        text: Input text
        extractor: SpanExtractor instance (created if None)
        return_species: Whether to return species names

    Returns:
        Tuple of (tagged_text, species1_name, species2_name)
        Returns (original_text, None, None) if fewer than 2 species found
    """
    if extractor is None:
        extractor = SpanExtractor(use_taxonerd=TAXONERD_AVAILABLE)

    # Extract species
    species = extractor.extract_species(text, max_species=2)

    if len(species) < 2:
        # Not enough species found
        if return_species:
            sp1 = species[0].text if len(species) > 0 else None
            return text, sp1, None
        return text, None, None

    # Tag the text
    tagged = tag_entities(text, species[0], species[1])

    if return_species:
        return tagged, species[0].text, species[1].text
    return tagged, species[0].text, species[1].text


def batch_extract_and_tag(
    texts: List[str],
    extractor: Optional[SpanExtractor] = None,
    show_progress: bool = True
) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """
    Process multiple texts, extracting species and adding entity markers.

    Args:
        texts: List of input texts
        extractor: SpanExtractor instance (created if None)
        show_progress: Whether to show progress bar

    Returns:
        List of (tagged_text, species1, species2) tuples
    """
    if extractor is None:
        extractor = SpanExtractor(use_taxonerd=TAXONERD_AVAILABLE)

    results = []

    iterator = texts
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(texts, desc="Extracting species")
        except ImportError:
            pass

    for text in iterator:
        result = extract_and_tag(text, extractor, return_species=True)
        results.append(result)

    return results


# Convenience function for simple usage
def tag_species_in_text(text: str) -> str:
    """
    Simple function to tag the first two species in text.

    Args:
        text: Input text containing species mentions

    Returns:
        Text with <e1>...</e1> and <e2>...</e2> markers around species
    """
    tagged, _, _ = extract_and_tag(text)
    return tagged


if __name__ == "__main__":
    # Test the extractor
    test_sentences = [
        "The gray wolf Canis lupus is a predator of Ovis aries in mountainous regions.",
        "Plasmodium falciparum infects Anopheles gambiae mosquitoes.",
        "In this study, we examined how Drosophila melanogaster responds to infection by Pseudomonas aeruginosa.",
        "The parasitic wasp Cotesia glomerata lays its eggs in Pieris rapae caterpillars.",
        "No species mentioned here at all.",
        "Only one species: Homo sapiens.",
    ]

    print("Testing SpanExtractor...")
    print(f"TaxoNERD available: {TAXONERD_AVAILABLE}")
    print()

    extractor = SpanExtractor(use_taxonerd=TAXONERD_AVAILABLE)

    for sent in test_sentences:
        print(f"Original: {sent}")
        tagged, sp1, sp2 = extract_and_tag(sent, extractor, return_species=True)
        print(f"Species 1: {sp1}")
        print(f"Species 2: {sp2}")
        print(f"Tagged:    {tagged}")
        print()
