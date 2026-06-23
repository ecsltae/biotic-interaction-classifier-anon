"""
Biotic interaction validation tools.

This module provides LLM-based and heuristic validation for interaction sentences.
Renamed from mcp/ to validator/ to avoid collision with the mcp Python package.
"""

from .interaction_validator import (
    validate_interaction_sentence,
    batch_validate_sentences,
    is_llm_available,
)

__all__ = [
    'validate_interaction_sentence',
    'batch_validate_sentences',
    'is_llm_available',
]
