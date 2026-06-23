"""
Interaction Validator

Uses LLM (Claude) to validate whether sentences describe biotic interactions.
Falls back to rule-based validation if API is not available.

Usage:
    from validator import validate_interaction_sentence

    result = validate_interaction_sentence("Bees pollinate flowers.")
    # Returns: {"is_interaction": True, "confidence": 0.95, "reasoning": "..."}

Environment:
    Set ANTHROPIC_API_KEY to enable LLM validation.
    Without API key, falls back to heuristic-based validation.
"""

import os
import re
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Import canonical lexicon from src/data/
# Falls back gracefully if the project layout differs.
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from data.interaction_lexicon import (  # noqa: E402
        _STRONG_COMPILED,
        _WEAK_COMPILED,
        _NEGATION_COMPILED,
        _METHOD_COMPILED,
        score_sentence as _lexicon_score,
    )
    _LEXICON_AVAILABLE = True
except ImportError:
    _LEXICON_AVAILABLE = False
    logger.warning("interaction_lexicon not found — falling back to local patterns.")

# Try to import Anthropic SDK
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.info("Anthropic SDK not installed. Using rule-based validation only.")


@dataclass
class ValidationResult:
    """Result of sentence validation."""
    is_interaction: bool
    confidence: float  # 0.0 to 1.0
    reasoning: str
    method: str  # 'llm' or 'heuristic'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_interaction': self.is_interaction,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'method': self.method,
        }


def is_llm_available() -> bool:
    """Check if LLM validation is available (API key set)."""
    return ANTHROPIC_AVAILABLE and bool(os.environ.get('ANTHROPIC_API_KEY'))


# ============================================================================
# LLM-BASED VALIDATION
# ============================================================================

VALIDATION_PROMPT = """You are an expert in ecology and biotic interactions. Your task is to determine if a sentence describes a biotic interaction between two or more organisms.

A biotic interaction is a relationship between organisms that affects one or both. Types include:
- Predation (one organism eats another)
- Parasitism (one organism lives on/in and harms another)
- Pollination (animal transfers pollen to plant)
- Herbivory (animal eats plant)
- Symbiosis/Mutualism (mutually beneficial relationship)
- Competition (organisms compete for resources)
- Infection/Pathogenicity (pathogen infects host)
- Vector transmission (organism transmits pathogen)

Analyze this sentence and determine if it describes a biotic interaction:

Sentence: "{sentence}"

Respond with a JSON object:
{{
    "is_interaction": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

Only output the JSON, nothing else."""


def _validate_with_llm(sentence: str) -> ValidationResult:
    """Validate sentence using Claude API."""
    if not is_llm_available():
        raise RuntimeError("LLM validation not available")

    client = anthropic.Anthropic()

    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",  # Use fast/cheap model for validation
            max_tokens=256,
            messages=[
                {"role": "user", "content": VALIDATION_PROMPT.format(sentence=sentence)}
            ]
        )

        # Parse JSON response
        content = response.content[0].text.strip()
        # Handle potential markdown code blocks
        if content.startswith('```'):
            content = content.split('```')[1]
            if content.startswith('json'):
                content = content[4:]
        content = content.strip()

        result = json.loads(content)

        return ValidationResult(
            is_interaction=bool(result.get('is_interaction', False)),
            confidence=float(result.get('confidence', 0.5)),
            reasoning=str(result.get('reasoning', '')),
            method='llm'
        )

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM response: {e}")
        # Fall back to heuristic
        return _validate_with_heuristics(sentence)
    except Exception as e:
        logger.error(f"LLM validation error: {e}")
        return _validate_with_heuristics(sentence)


# ============================================================================
# HEURISTIC-BASED VALIDATION (FALLBACK)
# ============================================================================

# Interaction pattern lists — sourced from the canonical lexicon when available,
# otherwise defined locally as a fallback (kept for backward compatibility).
if _LEXICON_AVAILABLE:
    # Use compiled objects from interaction_lexicon.py directly
    STRONG_INTERACTION_PATTERNS = [p.pattern for p in _STRONG_COMPILED]
    WEAK_INTERACTION_PATTERNS   = [p.pattern for p in _WEAK_COMPILED]
    NEGATIVE_PATTERNS           = [p.pattern for p in _NEGATION_COMPILED]
else:
    # Local fallback (subset of the canonical lexicon)
    STRONG_INTERACTION_PATTERNS = [
        r'\b(preys?\s+(?:up)?on|preyed\s+on|predation)\b',
        r'\b(parasiti[zs]es?|parasiti[zs]ed|parasitic\s+on|parasite\s+of)\b',
        r'\b(infects?|infected\s+by|infection\s+of)\b',
        r'\b(pollinates?|pollinated\s+by|pollinator\s+of)\b',
        r'\b(feeds?\s+on|fed\s+on|consumes?|consumed\s+by)\b',
        r'\b(hosts?|host\s+of|host\s+to)\b',
        r'\b(vector\s+of|transmits?|transmitted\s+by)\b',
        r'\b(symbiont|symbiosis|mutualist)\b',
    ]
    WEAK_INTERACTION_PATTERNS = [
        r'\b(eats?|ate|eaten)\b',
        r'\b(kills?|killed)\b',
        r'\b(attacks?|attacked)\b',
        r'\b(associated\s+with)\b',
    ]
    NEGATIVE_PATTERNS = [
        r'\b(method|protocol|experiment|study\s+design)\b',
        r'\b(table|figure|appendix|supplementary)\b',
        r'\b(we\s+used|we\s+measured|we\s+analyzed)\b',
        r'\b(statistical|analysis|significance|p\s*[<=])\b',
        r'\b(sample\s+size|n\s*=\s*\d+)\b',
    ]


def _validate_with_heuristics(sentence: str) -> ValidationResult:
    """Validate sentence using rule-based heuristics.

    Delegates to the canonical lexicon's score_sentence() when available,
    preserving the same ValidationResult interface for callers.
    """
    sentence_lower = sentence.lower()

    if _LEXICON_AVAILABLE:
        has_signal, strength, matched = _lexicon_score(sentence_lower)

        if not has_signal:
            reasoning = "No interaction patterns detected"
            confidence = 0.15
            is_interaction = False
        else:
            # Map continuous strength to confidence/is_interaction
            n_strong = sum(1 for p in _STRONG_COMPILED if p.search(sentence_lower))
            n_weak   = sum(1 for p in _WEAK_COMPILED   if p.search(sentence_lower))

            if n_strong >= 2:
                confidence = 0.90
                reasoning  = f"Multiple strong interaction terms ({n_strong}): {matched[:3]}"
            elif n_strong == 1 and n_weak >= 1:
                confidence = 0.80
                reasoning  = f"Strong + weak interaction terms: {matched[:3]}"
            elif n_strong == 1:
                confidence = max(0.50, strength + 0.10)
                reasoning  = f"Single strong interaction term: {matched[:2]}"
            else:
                confidence = min(0.65, strength + 0.05)
                reasoning  = f"Weak interaction indicators: {matched[:3]}"

            is_interaction = confidence >= 0.50

        return ValidationResult(
            is_interaction=is_interaction,
            confidence=max(0.0, min(1.0, confidence)),
            reasoning=reasoning,
            method="heuristic",
        )

    # ── Fallback: local pattern matching (lexicon not available) ──────────────
    negative_score = sum(
        1 for p in NEGATIVE_PATTERNS
        if re.search(p, sentence_lower, re.IGNORECASE)
    )
    strong_matches = sum(
        1 for p in STRONG_INTERACTION_PATTERNS
        if re.search(p, sentence_lower, re.IGNORECASE)
    )
    weak_matches = sum(
        1 for p in WEAK_INTERACTION_PATTERNS
        if re.search(p, sentence_lower, re.IGNORECASE)
    )

    if strong_matches >= 2:
        confidence = 0.9; is_interaction = True
        reasoning = f"Multiple strong interaction terms found ({strong_matches})"
    elif strong_matches == 1 and weak_matches >= 1:
        confidence = 0.8; is_interaction = True
        reasoning = "Strong interaction term with supporting context"
    elif strong_matches == 1:
        confidence = 0.7 - (negative_score * 0.1)
        is_interaction = confidence > 0.5
        reasoning = "Single strong interaction term"
    elif weak_matches >= 2:
        confidence = 0.6 - (negative_score * 0.15)
        is_interaction = confidence > 0.5
        reasoning = "Multiple weak interaction indicators"
    elif weak_matches == 1:
        confidence = 0.4 - (negative_score * 0.1)
        is_interaction = False
        reasoning = "Only weak indicator found"
    else:
        confidence = 0.2; is_interaction = False
        reasoning = "No interaction patterns detected"

    if negative_score >= 2:
        confidence = max(0.1, confidence - 0.3)
        is_interaction = False
        reasoning += f" (methodology language detected: -{negative_score * 0.15:.1f})"

    return ValidationResult(
        is_interaction=is_interaction,
        confidence=max(0.0, min(1.0, confidence)),
        reasoning=reasoning,
        method="heuristic",
    )


# ============================================================================
# PUBLIC API
# ============================================================================

def validate_interaction_sentence(
    sentence: str,
    use_llm: bool = True,
    min_confidence: float = 0.0
) -> ValidationResult:
    """
    Validate whether a sentence describes a biotic interaction.

    Args:
        sentence: The sentence to validate
        use_llm: Whether to use LLM validation (if available)
        min_confidence: Minimum confidence threshold

    Returns:
        ValidationResult with is_interaction, confidence, reasoning, method
    """
    if use_llm and is_llm_available():
        result = _validate_with_llm(sentence)
    else:
        result = _validate_with_heuristics(sentence)

    return result


def batch_validate_sentences(
    sentences: List[str],
    use_llm: bool = True,
    min_confidence: float = 0.5,
    max_llm_calls: int = 100
) -> List[Dict[str, Any]]:
    """
    Validate multiple sentences, optionally filtering by confidence.

    Args:
        sentences: List of sentences to validate
        use_llm: Whether to use LLM validation
        min_confidence: Minimum confidence to include in results
        max_llm_calls: Maximum number of LLM API calls (to control cost)

    Returns:
        List of dicts with sentence and validation results
    """
    results = []
    llm_calls = 0

    for sentence in sentences:
        # Limit LLM calls
        effective_use_llm = use_llm and llm_calls < max_llm_calls

        result = validate_interaction_sentence(
            sentence,
            use_llm=effective_use_llm,
            min_confidence=min_confidence
        )

        if effective_use_llm and result.method == 'llm':
            llm_calls += 1

        if result.confidence >= min_confidence:
            results.append({
                'sentence': sentence,
                **result.to_dict()
            })

    logger.info(
        f"Validated {len(sentences)} sentences, "
        f"{len(results)} passed threshold (min_conf={min_confidence}), "
        f"{llm_calls} LLM calls"
    )

    return results


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    test_sentences = [
        # Clear interactions
        "Vulpes vulpes preys on Mus musculus in forest ecosystems.",
        "The parasitic wasp Cotesia glomerata parasitizes larvae of Pieris brassicae.",
        "Apis mellifera pollinates apple flowers during spring.",
        "Spiders consume flies and other small insects.",

        # Unclear / methodology
        "We collected 50 samples from the study area.",
        "Statistical analysis was performed using R software.",
        "Table 1 shows the distribution of species across sites.",

        # Borderline
        "The fox population was found near rabbit burrows.",
        "Both species were observed in the same habitat.",
    ]

    print("=== Interaction Validator Tests ===\n")
    print(f"LLM available: {is_llm_available()}\n")

    for sentence in test_sentences:
        result = validate_interaction_sentence(sentence, use_llm=False)
        status = "✓" if result.is_interaction else "✗"
        print(f"{status} [{result.confidence:.2f}] {sentence[:60]}...")
        print(f"   {result.reasoning}")
        print()
