"""
outcome_codes.py — Structured outcome classification for interaction predictions.

Defines the OutcomeCode enum and functions to synthesize an outcome from the
outputs of the NER, lexicon scoring, GloBI term scan, and ML classifier layers.

This module does NOT modify any existing prediction code.  It is called by the
new fastapi_pipeline.py service (port 8002) and can also be used standalone.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional


# ---------------------------------------------------------------------------
# Outcome codes
# ---------------------------------------------------------------------------

class OutcomeCode(str, Enum):
    """Structured diagnostic codes explaining a prediction result.

    Each code represents a distinct structural or semantic pattern detected in
    the sentence.  The codes are designed to be:
      - Mutually exclusive (one primary code per prediction)
      - Auditable (user can verify the reasoning independently)
      - Actionable (code hints at what would change the outcome)
    """
    # ── Positive outcomes ──────────────────────────────────────────────────
    INTERACTION       = "INTERACTION"
    """Two or more species detected; an interaction term is present (GloBI or
    lexicon); ML probability > 0.5.  All three signals agree."""

    INTERACTION_WEAK  = "INTERACTION_WEAK"
    """ML probability > 0.5 but the lexicon/GloBI signal is weak or absent.
    The model may be picking up on implicit context."""

    # ── Negative — structural reasons ─────────────────────────────────────
    NO_SPECIES        = "NO_SPECIES"
    """No species names were identified in the sentence.  Without at least two
    organisms an interaction cannot be described."""

    ONE_SPECIES       = "ONE_SPECIES"
    """Only one organism was detected.  An interaction requires ≥ 2 parties."""

    TWO_SPECIES_NO_VERB = "TWO_SPECIES_NO_VERB"
    """Two or more species were detected but no interaction vocabulary was found
    (neither GloBI terms nor lexicon patterns).  The sentence likely describes
    co-occurrence, methodology, or background context."""

    NEGATED           = "NEGATED"
    """An interaction term was detected but the surrounding context contains
    explicit negation (e.g. 'does not infect', 'no evidence of parasitism').
    The interaction is asserted to be absent."""

    METHODOLOGY       = "METHODOLOGY"
    """The sentence is dominated by methodology or statistical language.  Any
    apparent interaction terms are likely in a methods or results-of-sampling
    context rather than describing a real interaction."""

    # ── Ambiguous ─────────────────────────────────────────────────────────
    BORDERLINE        = "BORDERLINE"
    """The ML classifier and the rule-based layer disagree:
      - ML > 0.5 but no lexicon/GloBI signal, OR
      - ML < 0.5 but a strong lexicon signal is present.
    Manual review is recommended."""

    IMPLICIT          = "IMPLICIT"
    """Two or more species detected and ML > 0.5, but the interaction is
    expressed indirectly (diet shift context, co-occurrence description, etc.)
    without explicit interaction vocabulary.  The sentence may describe a real
    interaction in an implicit way."""


# ---------------------------------------------------------------------------
# Reasoning templates
# ---------------------------------------------------------------------------

_TEMPLATES: dict[OutcomeCode, str] = {
    OutcomeCode.INTERACTION: (
        "{subjects} are linked by the interaction term '{term}' "
        "(category: {category}, confidence: {prob:.0%})."
    ),
    OutcomeCode.INTERACTION_WEAK: (
        "The ML model predicts an interaction ({prob:.0%} confidence) but no "
        "explicit interaction vocabulary was found.  This may reflect implicit "
        "interaction context."
    ),
    OutcomeCode.NO_SPECIES: (
        "No species names were identified in this sentence.  An interaction "
        "requires at least two organisms."
    ),
    OutcomeCode.ONE_SPECIES: (
        "Only one organism was detected: {subjects}.  An interaction requires "
        "at least two parties."
    ),
    OutcomeCode.TWO_SPECIES_NO_VERB: (
        "Two organisms detected ({subjects}) but no interaction vocabulary was "
        "found.  The sentence likely describes co-occurrence, methodology, or "
        "background context."
    ),
    OutcomeCode.NEGATED: (
        "An interaction term ('{term}') was detected but is explicitly negated.  "
        "The sentence asserts the interaction is absent."
    ),
    OutcomeCode.METHODOLOGY: (
        "The sentence contains methodology or statistical language that suggests "
        "any apparent interaction terms are in a methods/sampling context."
    ),
    OutcomeCode.BORDERLINE: (
        "The ML model and rule-based signals disagree (ML: {prob:.0%}, lexicon "
        "signal: {signal_strength:.2f}).  Manual review recommended."
    ),
    OutcomeCode.IMPLICIT: (
        "{subjects} co-occur in a context that may describe an interaction, "
        "but no explicit interaction vocabulary was found."
    ),
}


def format_reasoning(
    code: OutcomeCode,
    subjects: str = "",
    term: str = "",
    category: str = "",
    prob: float = 0.0,
    signal_strength: float = 0.0,
) -> str:
    """Format the human-readable reasoning string for an outcome code.

    Args:
        code:           The OutcomeCode to format.
        subjects:       Comma-separated species names (e.g. "Apis mellifera, Malus domestica").
        term:           The primary matched interaction term.
        category:       The canonical interaction category (e.g. "POLLINATION").
        prob:           ML classifier probability (0.0–1.0).
        signal_strength: Lexicon signal strength (0.0–1.0).

    Returns:
        A human-readable explanation string.
    """
    template = _TEMPLATES.get(code, "No reasoning template available.")
    try:
        return template.format(
            subjects=subjects or "the detected organisms",
            term=term or "(unknown term)",
            category=category or "UNKNOWN",
            prob=prob,
            signal_strength=signal_strength,
        )
    except KeyError:
        return template  # fallback: return template as-is if missing keys


# ---------------------------------------------------------------------------
# Outcome synthesis
# ---------------------------------------------------------------------------

def synthesize_outcome(
    n_species: int,
    species_names: List[str],
    matched_globi_terms: List[str],
    interaction_terms: List[str],         # from interaction_lexicon STRONG/WEAK
    signal_strength: float,
    has_negation: bool,
    has_methodology: bool,
    ml_probability: float,
    ml_threshold: float = 0.5,
    interaction_category: Optional[str] = None,
) -> tuple[OutcomeCode, str]:
    """Combine all layer outputs into a single outcome code + reasoning string.

    Decision logic (checked in order):

    1. Methodology detected → METHODOLOGY (even if ML > threshold)
    2. No species found     → NO_SPECIES
    3. One species          → ONE_SPECIES
    4. Negation present     → NEGATED (even if ML > threshold)
    5. ML > threshold AND (GloBI term OR lexicon signal > 0.15)
                            → INTERACTION
    6. ML > threshold AND no signal  (implicit context only)
                            → INTERACTION_WEAK or IMPLICIT (2 species: IMPLICIT)
    7. ML < threshold AND strong signal (> 0.40)
                            → BORDERLINE
    8. ML < threshold AND GloBI term found
                            → BORDERLINE
    9. Two species, no signal, ML < threshold
                            → TWO_SPECIES_NO_VERB
    10. Default             → NO_SPECIES (shouldn't reach here)

    Args:
        n_species:          Number of species detected by NER.
        species_names:      List of species name strings.
        matched_globi_terms: GloBI terms found by scan_globi_terms().
        interaction_terms:  STRONG/WEAK lexicon matches from score_sentence().
        signal_strength:    Continuous score 0–1 from score_sentence().
        has_negation:       True if negation patterns were detected.
        has_methodology:    True if ≥2 methodology indicators were detected.
        ml_probability:     Classifier probability for label=1.
        ml_threshold:       Decision boundary (default 0.5).
        interaction_category: Canonical category string or None.

    Returns:
        Tuple of (OutcomeCode, reasoning_string).
    """
    subjects = ", ".join(species_names) if species_names else ""
    primary_term = (matched_globi_terms or interaction_terms or [""])[0]
    ml_pos = ml_probability >= ml_threshold
    has_signal = signal_strength > 0.0
    strong_signal = signal_strength >= 0.40
    has_globi = bool(matched_globi_terms)

    # 1. Methodology
    if has_methodology and not (has_globi and ml_pos):
        code = OutcomeCode.METHODOLOGY
        return code, format_reasoning(code, subjects=subjects, prob=ml_probability)

    # 2. No species
    if n_species == 0:
        code = OutcomeCode.NO_SPECIES
        return code, format_reasoning(code)

    # 3. One species
    if n_species == 1:
        code = OutcomeCode.ONE_SPECIES
        return code, format_reasoning(code, subjects=subjects)

    # 4. Negation overrides (only if signal was found but negated)
    if has_negation and (has_signal or has_globi):
        code = OutcomeCode.NEGATED
        return code, format_reasoning(code, subjects=subjects, term=primary_term)

    # 5. Clear positive: ML agrees + explicit signal
    if ml_pos and (has_globi or signal_strength >= 0.15):
        code = OutcomeCode.INTERACTION
        return code, format_reasoning(
            code,
            subjects=subjects,
            term=primary_term,
            category=interaction_category or "",
            prob=ml_probability,
        )

    # 6. ML positive but no explicit signal
    if ml_pos and n_species >= 2:
        code = OutcomeCode.IMPLICIT
        return code, format_reasoning(code, subjects=subjects, prob=ml_probability)

    if ml_pos:
        code = OutcomeCode.INTERACTION_WEAK
        return code, format_reasoning(code, prob=ml_probability)

    # 7. ML negative but strong lexicon signal
    if not ml_pos and strong_signal:
        code = OutcomeCode.BORDERLINE
        return code, format_reasoning(
            code, prob=ml_probability, signal_strength=signal_strength
        )

    # 8. ML negative but GloBI term found
    if not ml_pos and has_globi:
        code = OutcomeCode.BORDERLINE
        return code, format_reasoning(
            code, prob=ml_probability, signal_strength=signal_strength
        )

    # 9. Two+ species, no signal, ML negative
    if n_species >= 2:
        code = OutcomeCode.TWO_SPECIES_NO_VERB
        return code, format_reasoning(code, subjects=subjects)

    # 10. Fallback
    code = OutcomeCode.NO_SPECIES
    return code, format_reasoning(code)


if __name__ == "__main__":
    # Smoke test
    cases = [
        # (n_species, names, globi, lex_terms, strength, neg, meth, prob, expected)
        (2, ["Apis mellifera", "Malus domestica"], ["pollinates"], ["pollinates"], 0.40,
         False, False, 0.95, OutcomeCode.INTERACTION),
        (0, [], [], [], 0.0, False, False, 0.3, OutcomeCode.NO_SPECIES),
        (1, ["Homo sapiens"], [], [], 0.0, False, False, 0.6, OutcomeCode.ONE_SPECIES),
        (2, ["A", "B"], [], [], 0.0, False, False, 0.2, OutcomeCode.TWO_SPECIES_NO_VERB),
        (2, ["A", "B"], ["infects"], ["infects"], 0.4, True, False, 0.7, OutcomeCode.NEGATED),
        (2, ["A", "B"], [], [], 0.0, False, True, 0.3, OutcomeCode.METHODOLOGY),
        (2, ["A", "B"], [], [], 0.0, False, False, 0.8, OutcomeCode.IMPLICIT),
        (2, ["A", "B"], ["preys on"], ["preys on"], 0.6, False, False, 0.3, OutcomeCode.BORDERLINE),
    ]
    print("=== Outcome Synthesis Test ===\n")
    all_pass = True
    for (n_sp, names, globi, lex, strength, neg, meth, prob, expected) in cases:
        code, reasoning = synthesize_outcome(
            n_species=n_sp, species_names=names,
            matched_globi_terms=globi, interaction_terms=lex,
            signal_strength=strength, has_negation=neg,
            has_methodology=meth, ml_probability=prob,
        )
        status = "✓" if code == expected else "✗"
        if code != expected:
            all_pass = False
        print(f"  {status} {code:<25} (expected {expected})")
        print(f"    {reasoning[:100]}")
    print(f"\n{'All tests passed!' if all_pass else 'SOME TESTS FAILED'}")
