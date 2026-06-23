"""
False Positive Collector

Collect false positives from classifier predictions to use as high-quality negatives.
Implements automatic heuristics and exports for manual review.
"""

import re
import csv
import json
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Interaction verbs that should be present for a true positive
INTERACTION_VERBS = {
    'parasit', 'infect', 'prey', 'predat', 'eat', 'consum', 'feed',
    'pollinat', 'host', 'coloniz', 'attack', 'kill', 'hunt', 'vector',
    'transmit', 'symbio', 'mutual', 'herbivor', 'graz', 'pathogen',
    'diseas', 'associat', 'interact'
}

# Words that often appear in false positives (non-interaction contexts)
FALSE_POSITIVE_INDICATORS = {
    'phylogen', 'taxonom', 'classif', 'nomenclat', 'systemat',
    'sequenc', 'genom', 'dna', 'rna', 'gene ', 'transcript',
    'morpholog', 'anatomic', 'fossil', 'extinct',
    'distribut', 'range', 'habitat', 'biogeograph',
    'conserv', 'endanger', 'iucn', 'protect',
    'evolut', 'diverge', 'clade', 'monophyl',
    'specimen', 'museum', 'collect', 'voucher',
    'describ', 'species nov', 'sp. nov', 'n. sp'
}


@dataclass
class FalsePositiveCandidate:
    """A candidate false positive for review."""
    sentence: str
    confidence: float
    heuristic_flags: List[str]
    fp_probability: float  # Estimated probability this is a false positive
    source: str  # Where the sentence came from
    reviewed: bool = False
    is_false_positive: Optional[bool] = None  # Set during manual review


def has_interaction_verb(text: str) -> bool:
    """Check if text contains an interaction verb."""
    text_lower = text.lower()
    return any(verb in text_lower for verb in INTERACTION_VERBS)


def has_fp_indicator(text: str) -> List[str]:
    """Check for false positive indicators, return list of found indicators."""
    text_lower = text.lower()
    found = []
    for indicator in FALSE_POSITIVE_INDICATORS:
        if indicator in text_lower:
            found.append(indicator)
    return found


def count_species_mentions(text: str) -> int:
    """
    Estimate number of species-like mentions in text.

    Looks for binomial names (Genus species) patterns.
    """
    # Pattern for binomial names: Capitalized word followed by lowercase word
    pattern = r'\b[A-Z][a-z]+\s+[a-z]{3,}\b'
    matches = re.findall(pattern, text)
    return len(matches)


def sentence_has_interaction_structure(text: str) -> bool:
    """
    Check if sentence has the structure of describing an interaction.

    Looks for patterns like "X [verb] Y" where X and Y could be species.
    """
    # Simple pattern: species1 ... verb ... species2
    # Check for at least 2 potential species names with something between them
    binomial_pattern = r'([A-Z][a-z]+(?:\s+[a-z]+)?)'

    matches = list(re.finditer(binomial_pattern, text))
    if len(matches) < 2:
        return False

    # Check if there's an interaction verb between species mentions
    if len(matches) >= 2:
        between_text = text[matches[0].end():matches[-1].start()]
        if has_interaction_verb(between_text):
            return True

    return False


def calculate_fp_probability(
    sentence: str,
    confidence: float,
    flags: List[str]
) -> float:
    """
    Estimate probability that a positive prediction is actually a false positive.

    Args:
        sentence: The classified sentence
        confidence: Classifier confidence (0-1)
        flags: Heuristic flags already computed

    Returns:
        Estimated FP probability (0-1)
    """
    fp_prob = 0.0

    # High confidence but no interaction verb = likely FP
    if not has_interaction_verb(sentence):
        fp_prob += 0.4

    # Presence of FP indicators
    fp_prob += min(len(flags) * 0.15, 0.4)

    # Very short sentences are suspicious
    if len(sentence) < 50:
        fp_prob += 0.1

    # Very long sentences often contain co-mentions without interaction
    if len(sentence) > 300:
        fp_prob += 0.1

    # No clear interaction structure
    if not sentence_has_interaction_structure(sentence):
        fp_prob += 0.2

    # Too many species (might be a list, not an interaction)
    species_count = count_species_mentions(sentence)
    if species_count > 4:
        fp_prob += 0.15

    return min(fp_prob, 1.0)


def apply_heuristics(
    sentence: str,
    confidence: float
) -> Tuple[List[str], float]:
    """
    Apply all heuristics to a sentence.

    Args:
        sentence: The classified sentence
        confidence: Classifier confidence

    Returns:
        Tuple of (list of flags, FP probability)
    """
    flags = []

    # Check for FP indicators
    fp_indicators = has_fp_indicator(sentence)
    if fp_indicators:
        flags.append(f"fp_indicators: {fp_indicators[:3]}")

    # Check for missing interaction verb
    if not has_interaction_verb(sentence):
        flags.append("no_interaction_verb")

    # Check sentence structure
    if not sentence_has_interaction_structure(sentence):
        flags.append("no_interaction_structure")

    # Check species count
    species_count = count_species_mentions(sentence)
    if species_count > 4:
        flags.append(f"many_species: {species_count}")
    elif species_count < 2:
        flags.append("few_species")

    # Calculate overall FP probability
    fp_prob = calculate_fp_probability(sentence, confidence, flags)

    return flags, fp_prob


class FalsePositiveCollector:
    """Collects and manages false positive candidates."""

    def __init__(
        self,
        output_dir: str,
        confidence_threshold: float = 0.7,
        fp_probability_threshold: float = 0.5
    ):
        """
        Initialize the collector.

        Args:
            output_dir: Directory for output files
            confidence_threshold: Min classifier confidence to consider
            fp_probability_threshold: Min FP probability to flag for review
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.confidence_threshold = confidence_threshold
        self.fp_probability_threshold = fp_probability_threshold
        self.candidates: List[FalsePositiveCandidate] = []
        self.stats = {
            "total_processed": 0,
            "above_confidence": 0,
            "flagged_as_fp": 0,
            "auto_rejected": 0
        }

    def process_predictions(
        self,
        predictions: List[Dict],
        source: str = "unknown"
    ) -> List[FalsePositiveCandidate]:
        """
        Process classifier predictions to find FP candidates.

        Args:
            predictions: List of dicts with 'text', 'prediction', 'confidence'
            source: Source identifier for tracking

        Returns:
            List of FalsePositiveCandidate objects
        """
        candidates = []

        for pred in predictions:
            self.stats["total_processed"] += 1

            # Only look at positive predictions
            if pred.get('prediction', 0) != 1:
                continue

            confidence = pred.get('confidence', pred.get('probability', 0))
            if confidence < self.confidence_threshold:
                continue

            self.stats["above_confidence"] += 1

            sentence = pred.get('text', pred.get('sentence', ''))
            if not sentence:
                continue

            # Apply heuristics
            flags, fp_prob = apply_heuristics(sentence, confidence)

            # Create candidate if FP probability is high enough
            if fp_prob >= self.fp_probability_threshold:
                candidate = FalsePositiveCandidate(
                    sentence=sentence,
                    confidence=confidence,
                    heuristic_flags=flags,
                    fp_probability=fp_prob,
                    source=source
                )
                candidates.append(candidate)
                self.stats["flagged_as_fp"] += 1

        self.candidates.extend(candidates)
        logger.info(
            f"Processed {len(predictions)} predictions, "
            f"found {len(candidates)} FP candidates"
        )
        return candidates

    def auto_filter_candidates(
        self,
        min_fp_probability: float = 0.7
    ) -> Tuple[List[FalsePositiveCandidate], List[FalsePositiveCandidate]]:
        """
        Automatically classify candidates based on FP probability.

        Args:
            min_fp_probability: Threshold for auto-accepting as FP

        Returns:
            Tuple of (auto-accepted FPs, needs manual review)
        """
        auto_accepted = []
        needs_review = []

        for candidate in self.candidates:
            if candidate.fp_probability >= min_fp_probability:
                candidate.is_false_positive = True
                candidate.reviewed = True
                auto_accepted.append(candidate)
                self.stats["auto_rejected"] += 1
            else:
                needs_review.append(candidate)

        logger.info(
            f"Auto-filtered: {len(auto_accepted)} auto-accepted as FP, "
            f"{len(needs_review)} need manual review"
        )
        return auto_accepted, needs_review

    def export_for_review(
        self,
        filepath: str = None,
        format: str = "csv"
    ) -> str:
        """
        Export candidates for manual review.

        Args:
            filepath: Output file path
            format: 'csv' or 'json'

        Returns:
            Path to exported file
        """
        if filepath is None:
            filepath = self.output_dir / f"fp_candidates_for_review.{format}"
        else:
            filepath = Path(filepath)

        # Get candidates needing review
        to_review = [c for c in self.candidates if not c.reviewed]

        if format == "csv":
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'sentence', 'confidence', 'fp_probability',
                    'heuristic_flags', 'source', 'is_false_positive'
                ])
                for c in to_review:
                    writer.writerow([
                        c.sentence,
                        f"{c.confidence:.3f}",
                        f"{c.fp_probability:.3f}",
                        '; '.join(c.heuristic_flags),
                        c.source,
                        ''  # To be filled during review
                    ])
        else:  # json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump([asdict(c) for c in to_review], f, indent=2)

        logger.info(f"Exported {len(to_review)} candidates to {filepath}")
        return str(filepath)

    def import_reviewed(self, filepath: str) -> int:
        """
        Import manually reviewed candidates.

        Expects CSV with 'sentence' and 'is_false_positive' columns,
        where is_false_positive is 'true', 'false', '1', '0', or 'yes'/'no'.

        Args:
            filepath: Path to reviewed CSV

        Returns:
            Number of candidates updated
        """
        updated = 0

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            reviews = {row['sentence']: row.get('is_false_positive', '') for row in reader}

        for candidate in self.candidates:
            if candidate.sentence in reviews:
                value = reviews[candidate.sentence].lower().strip()
                if value in ('true', '1', 'yes', 'y'):
                    candidate.is_false_positive = True
                    candidate.reviewed = True
                    updated += 1
                elif value in ('false', '0', 'no', 'n'):
                    candidate.is_false_positive = False
                    candidate.reviewed = True
                    updated += 1

        logger.info(f"Updated {updated} candidates from review file")
        return updated

    def get_confirmed_false_positives(self) -> List[str]:
        """Get sentences confirmed as false positives."""
        return [
            c.sentence for c in self.candidates
            if c.reviewed and c.is_false_positive
        ]

    def export_negatives(self, filepath: str = None) -> str:
        """
        Export confirmed false positives as negative training examples.

        Args:
            filepath: Output file path

        Returns:
            Path to exported file
        """
        if filepath is None:
            filepath = self.output_dir / "collected_negatives.csv"
        else:
            filepath = Path(filepath)

        fps = self.get_confirmed_false_positives()

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['passage', 'label'])
            for sentence in fps:
                writer.writerow([sentence, 0])

        logger.info(f"Exported {len(fps)} negatives to {filepath}")
        return str(filepath)

    def get_stats(self) -> dict:
        """Get collection statistics."""
        stats = self.stats.copy()
        stats["total_candidates"] = len(self.candidates)
        stats["reviewed"] = sum(1 for c in self.candidates if c.reviewed)
        stats["confirmed_fp"] = sum(
            1 for c in self.candidates if c.reviewed and c.is_false_positive
        )
        return stats


def collect_false_positives_from_classifier(
    classifier_func: Callable[[List[str]], List[Dict]],
    corpus: List[str],
    output_dir: str,
    batch_size: int = 100,
    confidence_threshold: float = 0.7,
    auto_accept_threshold: float = 0.8
) -> FalsePositiveCollector:
    """
    Run classifier on corpus and collect false positive candidates.

    Args:
        classifier_func: Function that takes list of texts and returns predictions
        corpus: List of sentences to classify
        output_dir: Directory for output files
        batch_size: Batch size for classification
        confidence_threshold: Min confidence to consider
        auto_accept_threshold: FP probability threshold for auto-accept

    Returns:
        FalsePositiveCollector with collected candidates
    """
    collector = FalsePositiveCollector(
        output_dir=output_dir,
        confidence_threshold=confidence_threshold
    )

    # Process in batches
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i + batch_size]
        predictions = classifier_func(batch)

        # Ensure predictions have the right format
        for j, pred in enumerate(predictions):
            if 'text' not in pred:
                pred['text'] = batch[j]

        collector.process_predictions(predictions, source=f"batch_{i//batch_size}")

        if (i + batch_size) % 1000 == 0:
            logger.info(f"Processed {i + batch_size}/{len(corpus)} sentences")

    # Auto-filter high-probability FPs
    collector.auto_filter_candidates(min_fp_probability=auto_accept_threshold)

    return collector


if __name__ == "__main__":
    # Example usage with synthetic data
    print("=== False Positive Collector Demo ===\n")

    # Simulate some classifier predictions
    test_predictions = [
        {
            "text": "Phylogenetic analysis of Canis lupus and Vulpes vulpes revealed divergent clades.",
            "prediction": 1,
            "confidence": 0.85
        },
        {
            "text": "The wolf (Canis lupus) frequently preys on white-tailed deer (Odocoileus virginianus).",
            "prediction": 1,
            "confidence": 0.92
        },
        {
            "text": "DNA sequences of Apis mellifera were deposited in GenBank.",
            "prediction": 1,
            "confidence": 0.78
        },
        {
            "text": "Specimens of Mus musculus and Rattus norvegicus were collected from museum holdings.",
            "prediction": 1,
            "confidence": 0.81
        },
        {
            "text": "The parasitic wasp Cotesia glomerata parasitizes larvae of Pieris brassicae.",
            "prediction": 1,
            "confidence": 0.95
        },
    ]

    collector = FalsePositiveCollector(
        output_dir="./test_fp_output",
        confidence_threshold=0.7
    )

    candidates = collector.process_predictions(test_predictions, source="test")

    print("Candidates for review:")
    for c in candidates:
        print(f"\n  Sentence: {c.sentence[:80]}...")
        print(f"  Confidence: {c.confidence:.2f}")
        print(f"  FP Probability: {c.fp_probability:.2f}")
        print(f"  Flags: {c.heuristic_flags}")

    auto_fps, needs_review = collector.auto_filter_candidates(min_fp_probability=0.6)

    print(f"\n\nAuto-accepted as FP: {len(auto_fps)}")
    print(f"Needs manual review: {len(needs_review)}")
    print(f"\nStats: {collector.get_stats()}")
