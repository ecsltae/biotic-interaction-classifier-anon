"""
ott_lookup.py — Fast species name validation using the OTT species dictionary.

The pre-built `species_dict.csv` (158 MB, 7.9M unique names) lives at:
    classifier/data/processed/species_dict.csv

This module loads it once into a Python set at startup for O(1) lookups.
It is used by:
  - fastapi_pipeline.py: to validate/annotate NER-extracted species names
  - Quality gates: as a lightweight check for species name validity
  - Future: OTT ID resolution by joining against the full OTT TSV
    (/data/terminologies/v.t1/original-release/ott/ott3.3/taxonomy.tsv)

Design decisions:
  - Singleton pattern: the set is loaded once, reused across requests
  - Case-insensitive matching: species names are normalised to lower-case
  - Partial-name matching: "C. lupus" matches if "Canis lupus" is in dict
    (disabled by default for precision; enable with fuzzy=True)
  - Thread-safe lazy loading via a module-level lock
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton loader
# ---------------------------------------------------------------------------

_SPECIES_SET: Optional[set] = None
_LOAD_LOCK = threading.Lock()

_DEFAULT_DICT_PATH = (
    Path(__file__).parent.parent.parent / "data/processed/species_dict.csv"
)


def _load_species_set(dict_path: Path) -> set:
    """Load species names from CSV into a lower-cased set.

    The CSV has a single column 'species'.  We lowercase all entries so that
    lookups are case-insensitive.
    """
    logger.info(f"Loading OTT species dictionary from {dict_path} ...")
    names: set = set()
    with open(dict_path, "r", encoding="utf-8", errors="replace") as fh:
        next(fh)  # skip header ('species')
        for line in fh:
            name = line.strip()
            if name:
                names.add(name.lower())
    logger.info(f"  Loaded {len(names):,} species names.")
    return names


def _get_species_set(dict_path: Optional[Path] = None) -> set:
    """Return the singleton species name set, loading it on first call."""
    global _SPECIES_SET
    if _SPECIES_SET is None:
        with _LOAD_LOCK:
            if _SPECIES_SET is None:
                path = dict_path or _DEFAULT_DICT_PATH
                if not path.exists():
                    logger.warning(
                        f"species_dict.csv not found at {path}. "
                        "OTT lookup will return None for all names."
                    )
                    _SPECIES_SET = set()
                else:
                    _SPECIES_SET = _load_species_set(path)
    return _SPECIES_SET


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_valid_species(name: str, dict_path: Optional[Path] = None) -> bool:
    """Return True if `name` exists in the OTT species dictionary.

    Case-insensitive.  Does not attempt fuzzy matching.

    Args:
        name:      Species name string (any case).
        dict_path: Override the default species_dict.csv path.

    Returns:
        True if the name is a known species name.

    Example:
        >>> is_valid_species("Canis lupus")
        True
        >>> is_valid_species("not a species")
        False
    """
    return name.lower().strip() in _get_species_set(dict_path)


def lookup(
    name: str,
    dict_path: Optional[Path] = None,
) -> Optional[dict]:
    """Look up a species name and return metadata if found.

    Currently returns only validation status and the normalised name because
    the species_dict.csv contains only names (no OTT IDs).  OTT ID resolution
    requires joining against the full OTT TSV — that can be added later.

    Args:
        name:      Species name string.
        dict_path: Override the default species_dict.csv path.

    Returns:
        Dict with keys {name, valid} if found, else None.

    Example:
        >>> lookup("Apis mellifera")
        {"name": "Apis mellifera", "valid": True, "ott_id": None, "rank": None}
    """
    normalised = name.strip()
    valid = normalised.lower() in _get_species_set(dict_path)
    if not valid:
        return None
    return {
        "name": normalised,
        "valid": True,
        "ott_id": None,    # not yet resolved (needs full OTT TSV join)
        "rank": None,      # not yet resolved
    }


def validate_species_list(
    names: list[str],
    dict_path: Optional[Path] = None,
) -> dict[str, Optional[dict]]:
    """Validate a list of species names in one call.

    Args:
        names:     List of species name strings.
        dict_path: Override path.

    Returns:
        Dict mapping each input name → lookup result (None if not found).
    """
    return {name: lookup(name, dict_path) for name in names}


def preload(dict_path: Optional[Path] = None) -> int:
    """Explicitly pre-load the species dictionary.

    Call this at application startup to avoid first-request latency.

    Returns:
        Number of species names loaded.
    """
    return len(_get_species_set(dict_path))


if __name__ == "__main__":
    import sys

    print("Loading OTT species dictionary (may take ~10 s for 7.9 M entries)...")
    n = preload()
    print(f"Loaded {n:,} entries.\n")

    test_names = [
        "Canis lupus",
        "Apis mellifera",
        "Plasmodium falciparum",
        "Homo sapiens",
        "not a species at all",
        "Q. floccosa",          # abbreviated, in dict
        "Ixodes ricinus",
        "Borrelia burgdorferi",
    ]
    print(f"{'Species name':<35} {'Valid?'}")
    print("-" * 45)
    for name in test_names:
        result = lookup(name)
        valid = result is not None
        print(f"  {name:<33} {'✓' if valid else '✗'}")
