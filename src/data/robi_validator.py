"""
ROBI Rule Validator for Training Data Quality Control

Validates biotic interactions against domain rules defined in domain_rules.json.
Based on ROBI (Relations in OBservations of bIotic interactions) ontology.
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobiValidator:
    """
    Validates biotic interactions against ROBI-based domain rules.

    Rules define taxonomic constraints for each interaction type:
    - requires_taxon: at least one species must be in these taxa
    - requires_role: which species (source/target) must be in required taxon
    - forbidden_taxa: taxa that cannot be involved
    - forbidden_same_species: source and target must be different
    """

    def __init__(self, rules_path: str = None):
        """
        Initialize ROBI validator.

        Args:
            rules_path: Path to domain_rules.json
        """
        base_dir = Path(__file__).parent.parent.parent
        self.rules_path = rules_path or base_dir / "data/domain_rules.json"

        self.rules: Dict = {}
        self._load_rules()

    def _load_rules(self):
        """Load domain rules from JSON file."""
        logger.info(f"Loading ROBI rules from {self.rules_path}")
        with open(self.rules_path, 'r') as f:
            data = json.load(f)

        # Filter out metadata keys
        self.rules = {
            k: v for k, v in data.items()
            if not k.startswith('_')
        }
        logger.info(f"Loaded {len(self.rules)} ROBI rules")

    def _match_interaction_type(self, interaction_type: str) -> Optional[str]:
        """
        Find which rule applies to an interaction type.

        Args:
            interaction_type: Interaction type name (e.g., "preysOn", "parasiteOf")

        Returns:
            Rule name if matched, None otherwise
        """
        interaction_lower = interaction_type.lower()

        for rule_name, rule in self.rules.items():
            pattern = rule.get('pattern', '')
            if pattern:
                if re.search(pattern, interaction_lower, re.IGNORECASE):
                    return rule_name

        return None

    def _check_taxon_requirement(
        self,
        source_kingdom: str,
        target_kingdom: str,
        requires_taxon: List[str],
        requires_role: Optional[str]
    ) -> Tuple[bool, str]:
        """
        Check if taxonomic requirements are met.

        Args:
            source_kingdom: Kingdom of source species
            target_kingdom: Kingdom of target species
            requires_taxon: List of required taxa
            requires_role: Which species must match ("source", "target", or None for either)

        Returns:
            (is_valid, error_message)
        """
        if not requires_taxon:
            return True, ""

        # Normalize for comparison
        source_k = (source_kingdom or "").strip()
        target_k = (target_kingdom or "").strip()
        required_set = {t.lower() for t in requires_taxon}

        # Check based on role requirement
        if requires_role == "source":
            if source_k.lower() in required_set:
                return True, ""
            return False, f"Source must be in {requires_taxon}, got '{source_k}'"

        elif requires_role == "target":
            if target_k.lower() in required_set:
                return True, ""
            return False, f"Target must be in {requires_taxon}, got '{target_k}'"

        else:
            # Either source or target can match
            if source_k.lower() in required_set or target_k.lower() in required_set:
                return True, ""
            return False, f"At least one species must be in {requires_taxon}"

    def _check_forbidden_taxa(
        self,
        source_kingdom: str,
        target_kingdom: str,
        forbidden_taxa: List[str]
    ) -> Tuple[bool, str]:
        """
        Check if any species is in a forbidden taxon.

        Args:
            source_kingdom: Kingdom of source species
            target_kingdom: Kingdom of target species
            forbidden_taxa: List of forbidden taxa

        Returns:
            (is_valid, error_message)
        """
        if not forbidden_taxa:
            return True, ""

        forbidden_set = {t.lower() for t in forbidden_taxa}
        source_k = (source_kingdom or "").lower()
        target_k = (target_kingdom or "").lower()

        if source_k in forbidden_set:
            return False, f"Source cannot be in {forbidden_taxa}, got '{source_kingdom}'"

        if target_k in forbidden_set:
            return False, f"Target cannot be in {forbidden_taxa}, got '{target_kingdom}'"

        return True, ""

    def validate_interaction(
        self,
        source: str,
        target: str,
        interaction_type: str,
        source_kingdom: str = None,
        target_kingdom: str = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate an interaction against ROBI domain rules.

        Args:
            source: Source species name
            target: Target species name
            interaction_type: Type of interaction (e.g., "preysOn", "parasiteOf")
            source_kingdom: Kingdom of source species (e.g., "Animalia")
            target_kingdom: Kingdom of target species

        Returns:
            (is_valid, list_of_violations)
        """
        violations = []

        # Find matching rule
        rule_name = self._match_interaction_type(interaction_type)

        if not rule_name:
            # No specific rule - just check same species
            if source and target and source.lower() == target.lower():
                violations.append("Source and target are the same species")
            return len(violations) == 0, violations

        rule = self.rules[rule_name]

        # Check forbidden_same_species
        if rule.get('forbidden_same_species', True):
            if source and target and source.lower() == target.lower():
                violations.append(f"[{rule_name}] Source and target must be different species")

        # Check requires_taxon
        requires_taxon = rule.get('requires_taxon', [])
        requires_role = rule.get('requires_role')

        if requires_taxon and (source_kingdom or target_kingdom):
            is_valid, error = self._check_taxon_requirement(
                source_kingdom, target_kingdom,
                requires_taxon, requires_role
            )
            if not is_valid:
                violations.append(f"[{rule_name}] {error}")

        # Check forbidden_taxa
        forbidden_taxa = rule.get('forbidden_taxa', [])
        if forbidden_taxa and (source_kingdom or target_kingdom):
            is_valid, error = self._check_forbidden_taxa(
                source_kingdom, target_kingdom, forbidden_taxa
            )
            if not is_valid:
                violations.append(f"[{rule_name}] {error}")

        return len(violations) == 0, violations

    def get_rule_for_interaction(self, interaction_type: str) -> Optional[Dict]:
        """
        Get the rule that applies to an interaction type.

        Args:
            interaction_type: Interaction type name

        Returns:
            Rule dict or None
        """
        rule_name = self._match_interaction_type(interaction_type)
        if rule_name:
            return self.rules.get(rule_name)
        return None

    def list_rules(self) -> Dict[str, Dict]:
        """Get all rules."""
        return self.rules


if __name__ == "__main__":
    # Demo / test
    print("Initializing RobiValidator...")
    validator = RobiValidator()

    print("\n=== Available Rules ===")
    for name in validator.rules.keys():
        print(f"  - {name}")

    # Test interactions
    test_cases = [
        # (source, target, interaction, src_kingdom, tgt_kingdom, expected_valid)
        ("Vulpes vulpes", "Mus musculus", "preysOn", "Animalia", "Animalia", True),
        ("Apis mellifera", "Malus domestica", "pollinates", "Animalia", "Plantae", True),
        ("Apis mellifera", "Bombus terrestris", "pollinates", "Animalia", "Animalia", False),  # Bee can't pollinate bee
        ("Arabidopsis thaliana", "Mus musculus", "preysOn", "Plantae", "Animalia", False),  # Plant can't prey
        ("Vulpes vulpes", "Vulpes vulpes", "parasiteOf", "Animalia", "Animalia", False),  # Same species
        ("Plasmodium falciparum", "Homo sapiens", "parasiteOf", None, "Animalia", True),  # Parasite
        ("Ixodes scapularis", "Borrelia burgdorferi", "vectorOf", "Animalia", None, True),  # Vector
    ]

    print("\n=== Validation Tests ===")
    for source, target, interaction, src_k, tgt_k, expected in test_cases:
        is_valid, violations = validator.validate_interaction(
            source, target, interaction, src_k, tgt_k
        )
        status = "PASS" if is_valid == expected else "FAIL"
        print(f"\n[{status}] {source} --{interaction}--> {target}")
        print(f"  Kingdoms: {src_k} -> {tgt_k}")
        print(f"  Valid: {is_valid} (expected: {expected})")
        if violations:
            for v in violations:
                print(f"  Violation: {v}")
