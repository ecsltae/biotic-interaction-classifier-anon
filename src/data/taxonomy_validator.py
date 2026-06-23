"""
Taxonomy Validator for Training Data Quality Control

Validates species names and provides taxonomic information (kingdom, class, order)
using GloBI interaction data and optionally OTT taxonomy.
"""

import gzip
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Set, Tuple
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaxonomyValidator:
    """
    Validates species names and provides taxonomic lookups.

    Uses GloBI interactions.tsv.gz as primary source (pre-computed taxonomy columns).
    Optionally uses OTT taxonomy for additional coverage.
    """

    def __init__(
        self,
        globi_path: str = None,
        ott_path: str = None,
        cache_path: str = None
    ):
        """
        Initialize taxonomy validator.

        Args:
            globi_path: Path to GloBI interactions.tsv.gz
            ott_path: Path to OTT taxonomy JSON (optional)
            cache_path: Path to save/load extracted taxonomy cache
        """
        # Maps: species_name (lowercase) -> {kingdom, class, order, phylum}
        self.species_taxonomy: Dict[str, Dict[str, str]] = {}

        # Sets for quick membership tests
        self.valid_species: Set[str] = set()

        # Default paths
        base_dir = Path(__file__).parent.parent.parent
        self.globi_path = globi_path or base_dir / "data/globi/interactions.tsv.gz"
        self.ott_path = ott_path or Path("/data/terminologies/v.t8/current-release/ott_v3.7.2.json")
        self.cache_path = cache_path or base_dir / "data/taxonomies/globi_taxonomy_cache.json"

        # Try to load from cache first
        if self.cache_path and Path(self.cache_path).exists():
            self._load_cache()
        else:
            self._extract_from_globi()
            if self.cache_path:
                self._save_cache()

    def _extract_from_globi(self):
        """Extract species taxonomy from GloBI interactions file."""
        logger.info(f"Extracting taxonomy from GloBI: {self.globi_path}")

        # Column indices (0-based) from GloBI TSV
        # Source columns
        SRC_NAME = 2       # sourceTaxonName
        SRC_ORDER = 15     # sourceTaxonOrderName
        SRC_CLASS = 17     # sourceTaxonClassName
        SRC_PHYLUM = 19    # sourceTaxonPhylumName
        SRC_KINGDOM = 21   # sourceTaxonKingdomName

        # Target columns
        TGT_NAME = 42      # targetTaxonName
        TGT_ORDER = 55     # targetTaxonOrderName
        TGT_CLASS = 57     # targetTaxonClassName
        TGT_PHYLUM = 59    # targetTaxonPhylumName
        TGT_KINGDOM = 61   # targetTaxonKingdomName

        count = 0
        with gzip.open(self.globi_path, 'rt', encoding='utf-8') as f:
            # Skip header
            header = next(f)

            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 63:
                    continue

                # Process source taxon
                src_name = parts[SRC_NAME].strip() if len(parts) > SRC_NAME else ""
                if src_name and src_name.lower() not in self.species_taxonomy:
                    self.species_taxonomy[src_name.lower()] = {
                        'original_name': src_name,
                        'kingdom': parts[SRC_KINGDOM].strip() if len(parts) > SRC_KINGDOM else "",
                        'phylum': parts[SRC_PHYLUM].strip() if len(parts) > SRC_PHYLUM else "",
                        'class': parts[SRC_CLASS].strip() if len(parts) > SRC_CLASS else "",
                        'order': parts[SRC_ORDER].strip() if len(parts) > SRC_ORDER else "",
                    }
                    self.valid_species.add(src_name.lower())

                # Process target taxon
                tgt_name = parts[TGT_NAME].strip() if len(parts) > TGT_NAME else ""
                if tgt_name and tgt_name.lower() not in self.species_taxonomy:
                    self.species_taxonomy[tgt_name.lower()] = {
                        'original_name': tgt_name,
                        'kingdom': parts[TGT_KINGDOM].strip() if len(parts) > TGT_KINGDOM else "",
                        'phylum': parts[TGT_PHYLUM].strip() if len(parts) > TGT_PHYLUM else "",
                        'class': parts[TGT_CLASS].strip() if len(parts) > TGT_CLASS else "",
                        'order': parts[TGT_ORDER].strip() if len(parts) > TGT_ORDER else "",
                    }
                    self.valid_species.add(tgt_name.lower())

                count += 1
                if count % 500000 == 0:
                    logger.info(f"  Processed {count:,} rows, {len(self.valid_species):,} unique species")

        logger.info(f"Extracted {len(self.valid_species):,} unique species from GloBI")

    def _load_cache(self):
        """Load taxonomy from cache file."""
        logger.info(f"Loading taxonomy cache from {self.cache_path}")
        with open(self.cache_path, 'r') as f:
            data = json.load(f)
            self.species_taxonomy = data.get('species_taxonomy', {})
            self.valid_species = set(data.get('valid_species', []))
        logger.info(f"Loaded {len(self.valid_species):,} species from cache")

    def _save_cache(self):
        """Save taxonomy to cache file."""
        logger.info(f"Saving taxonomy cache to {self.cache_path}")
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, 'w') as f:
            json.dump({
                'species_taxonomy': self.species_taxonomy,
                'valid_species': list(self.valid_species)
            }, f)

    def is_valid_species(self, name: str) -> bool:
        """
        Check if species name exists in our taxonomy database.

        Args:
            name: Species name to check

        Returns:
            True if species is known, False otherwise
        """
        if not name or not name.strip():
            return False
        return name.lower().strip() in self.valid_species

    def get_kingdom(self, name: str) -> Optional[str]:
        """
        Get kingdom for a species.

        Args:
            name: Species name

        Returns:
            Kingdom name (e.g., "Animalia", "Plantae") or None
        """
        if not name:
            return None
        info = self.species_taxonomy.get(name.lower().strip())
        if info:
            return info.get('kingdom') or None
        return None

    def get_class(self, name: str) -> Optional[str]:
        """
        Get taxonomic class for a species.

        Args:
            name: Species name

        Returns:
            Class name (e.g., "Insecta", "Mammalia") or None
        """
        if not name:
            return None
        info = self.species_taxonomy.get(name.lower().strip())
        if info:
            return info.get('class') or None
        return None

    def get_order(self, name: str) -> Optional[str]:
        """
        Get taxonomic order for a species.

        Args:
            name: Species name

        Returns:
            Order name (e.g., "Lepidoptera", "Diptera") or None
        """
        if not name:
            return None
        info = self.species_taxonomy.get(name.lower().strip())
        if info:
            return info.get('order') or None
        return None

    def get_phylum(self, name: str) -> Optional[str]:
        """
        Get phylum for a species.

        Args:
            name: Species name

        Returns:
            Phylum name or None
        """
        if not name:
            return None
        info = self.species_taxonomy.get(name.lower().strip())
        if info:
            return info.get('phylum') or None
        return None

    def is_insecta(self, name: str) -> bool:
        """
        Check if species is an insect (class Insecta).

        Important for validating "larvae" templates.

        Args:
            name: Species name

        Returns:
            True if species is in class Insecta
        """
        taxon_class = self.get_class(name)
        return taxon_class and taxon_class.lower() == 'insecta'

    def is_lepidoptera(self, name: str) -> bool:
        """
        Check if species is a butterfly/moth (order Lepidoptera).

        Important for validating "caterpillar" templates.

        Args:
            name: Species name

        Returns:
            True if species is in order Lepidoptera
        """
        order = self.get_order(name)
        return order and order.lower() == 'lepidoptera'

    def is_oomycete(self, name: str) -> bool:
        """
        Check if species is an oomycete.

        Important for validating "oomycete" templates.

        Args:
            name: Species name

        Returns:
            True if species is an oomycete
        """
        # Known oomycete genera
        oomycete_genera = {
            'phytophthora', 'pythium', 'aphanomyces', 'saprolegnia',
            'achlya', 'peronospora', 'plasmopara', 'bremia', 'albugo'
        }
        if not name:
            return False
        genus = name.lower().split()[0] if name.split() else ""
        return genus in oomycete_genera

    def get_taxonomy_stats(self) -> Dict:
        """Get statistics about the taxonomy database."""
        kingdoms = defaultdict(int)
        classes = defaultdict(int)

        for info in self.species_taxonomy.values():
            k = info.get('kingdom', 'Unknown')
            c = info.get('class', 'Unknown')
            if k:
                kingdoms[k] += 1
            if c:
                classes[c] += 1

        return {
            'total_species': len(self.valid_species),
            'kingdoms': dict(kingdoms),
            'top_classes': dict(sorted(classes.items(), key=lambda x: -x[1])[:20])
        }


if __name__ == "__main__":
    # Demo / test
    print("Initializing TaxonomyValidator...")
    validator = TaxonomyValidator()

    # Test some species
    test_species = [
        "Vulpes vulpes",           # Red fox - Animalia, Mammalia
        "Apis mellifera",          # Honey bee - Animalia, Insecta
        "Danaus plexippus",        # Monarch butterfly - Insecta, Lepidoptera
        "Arabidopsis thaliana",    # Plant
        "Phytophthora infestans",  # Oomycete
        "Escherichia coli",        # Bacteria
        "NotARealSpecies xyz",     # Invalid
    ]

    print("\n=== Species Validation Tests ===")
    for sp in test_species:
        valid = validator.is_valid_species(sp)
        kingdom = validator.get_kingdom(sp)
        taxon_class = validator.get_class(sp)
        order = validator.get_order(sp)
        is_insect = validator.is_insecta(sp)
        is_lepid = validator.is_lepidoptera(sp)

        print(f"\n{sp}:")
        print(f"  Valid: {valid}")
        print(f"  Kingdom: {kingdom}")
        print(f"  Class: {taxon_class}")
        print(f"  Order: {order}")
        print(f"  Is Insecta: {is_insect}")
        print(f"  Is Lepidoptera: {is_lepid}")

    print("\n=== Taxonomy Stats ===")
    stats = validator.get_taxonomy_stats()
    print(f"Total species: {stats['total_species']:,}")
    print(f"Kingdoms: {stats['kingdoms']}")
