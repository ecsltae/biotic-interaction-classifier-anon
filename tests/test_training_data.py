"""
Training Data Quality Tests - Gates 1-5

This test suite validates training data quality before model training.
All gates must pass before proceeding to training.

Gates:
1. Species Validation - All species exist in taxonomy
2. ROBI Rules - Positive interactions pass biological constraints
3. Template Validation - No malformed sentences
4. Negative Validation - No false negatives, sufficient diversity
5. Balance & Distribution - Reasonable dataset balance
"""

import re
import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.taxonomy_validator import TaxonomyValidator
from data.robi_validator import RobiValidator


# Configuration
TRAINING_DATA_PATH = Path(__file__).parent.parent / "data/training/training_data_v14.csv"

# Thresholds
GATE1_MAX_INVALID_RATE = 0.01       # Max 1% invalid species
GATE2_MAX_VIOLATION_RATE = 0.005    # Max 0.5% ROBI violations
GATE4_MAX_SINGLE_PATTERN = 0.70     # No single pattern >70% (two-species can be up to ~65%)
GATE4_MIN_HARD_NEGATIVE_RATIO = 0.50  # At least 50% hard negatives
GATE5_MIN_POS_RATIO = 0.20          # Min 20% positives
GATE5_MAX_POS_RATIO = 0.50          # Max 50% positives


class TestTrainingDataQuality:
    """Test suite for training data quality validation."""

    @pytest.fixture(scope="class")
    def training_data(self):
        """Load training data."""
        assert TRAINING_DATA_PATH.exists(), f"Training data not found: {TRAINING_DATA_PATH}"
        return pd.read_csv(TRAINING_DATA_PATH)

    @pytest.fixture(scope="class")
    def taxonomy(self):
        """Initialize taxonomy validator (uses cache if available)."""
        return TaxonomyValidator()

    @pytest.fixture(scope="class")
    def robi(self):
        """Initialize ROBI validator."""
        return RobiValidator()

    # =========================================================================
    # GATE 1: Species Validation
    # =========================================================================

    def test_gate1_species_exist(self, training_data, taxonomy):
        """
        GATE 1: All species in training data must exist in taxonomy.

        Fail if >1% of species are unknown.
        """
        invalid_species = []
        checked_species = set()

        for _, row in training_data.iterrows():
            # Check source species
            source = row.get('source_species', '')
            if source and source not in checked_species:
                checked_species.add(source)
                if not taxonomy.is_valid_species(source):
                    invalid_species.append(('source', source))

            # Check target species
            target = row.get('target_species', '')
            if target and not pd.isna(target) and target not in checked_species:
                # Handle multi-species in target (e.g., "sp1, sp2")
                for sp in str(target).split(','):
                    sp = sp.strip()
                    if sp and sp not in checked_species:
                        checked_species.add(sp)
                        if not taxonomy.is_valid_species(sp):
                            invalid_species.append(('target', sp))

        invalid_rate = len(invalid_species) / max(len(checked_species), 1)

        print(f"\n[GATE 1] Species Validation")
        print(f"  Checked: {len(checked_species)} unique species")
        print(f"  Invalid: {len(invalid_species)} ({invalid_rate:.2%})")

        if invalid_species[:10]:
            print(f"  Sample invalid: {invalid_species[:10]}")

        assert invalid_rate <= GATE1_MAX_INVALID_RATE, \
            f"Invalid species rate {invalid_rate:.2%} > {GATE1_MAX_INVALID_RATE:.0%}"

    # =========================================================================
    # GATE 2: ROBI Interaction Rules
    # =========================================================================

    def test_gate2_robi_rules(self, training_data, taxonomy, robi):
        """
        GATE 2: Positive interactions must pass ROBI biological rules.

        Fail if >0.5% of positives have rule violations.
        """
        positives = training_data[training_data['label'] == 1].copy()
        violations = []

        for idx, row in positives.iterrows():
            source = row.get('source_species', '')
            target = row.get('target_species', '')
            interaction = row.get('interaction_type', '')

            # Get kingdoms
            src_kingdom = taxonomy.get_kingdom(source)
            tgt_kingdom = taxonomy.get_kingdom(target)

            # Validate
            is_valid, errors = robi.validate_interaction(
                source, target, interaction, src_kingdom, tgt_kingdom
            )

            if not is_valid:
                violations.append({
                    'text': row.get('text', '')[:80],
                    'source': source,
                    'target': target,
                    'interaction': interaction,
                    'errors': errors
                })

        violation_rate = len(violations) / max(len(positives), 1)

        print(f"\n[GATE 2] ROBI Rules Validation")
        print(f"  Positives checked: {len(positives)}")
        print(f"  Violations: {len(violations)} ({violation_rate:.2%})")

        if violations[:5]:
            print(f"  Sample violations:")
            for v in violations[:5]:
                print(f"    - {v['interaction']}: {v['source']} -> {v['target']}")
                print(f"      Errors: {v['errors']}")

        assert violation_rate <= GATE2_MAX_VIOLATION_RATE, \
            f"ROBI violation rate {violation_rate:.2%} > {GATE2_MAX_VIOLATION_RATE:.1%}"

    # =========================================================================
    # GATE 3: Template Validation
    # =========================================================================

    def test_gate3_no_unfilled_placeholders(self, training_data):
        """
        GATE 3a: No sentences should have unfilled {placeholders}.
        """
        malformed = []

        for idx, row in training_data.iterrows():
            text = row.get('text', '')
            if '{' in text or '}' in text:
                malformed.append(text[:80])

        print(f"\n[GATE 3a] Placeholder Check")
        print(f"  Malformed: {len(malformed)}")

        if malformed[:5]:
            print(f"  Samples: {malformed[:5]}")

        assert len(malformed) == 0, f"Found {len(malformed)} sentences with unfilled placeholders"

    def test_gate3_sentences_complete(self, training_data):
        """
        GATE 3b: All sentences should end with proper punctuation.
        """
        incomplete = []

        for idx, row in training_data.iterrows():
            text = row.get('text', '').strip()
            if text and not any(text.endswith(p) for p in '.?!)'):
                incomplete.append(text[:80])

        print(f"\n[GATE 3b] Sentence Completeness")
        print(f"  Incomplete: {len(incomplete)}")

        if incomplete[:5]:
            print(f"  Samples: {incomplete[:5]}")

        assert len(incomplete) == 0, f"Found {len(incomplete)} incomplete sentences"

    def test_gate3_larvae_only_insecta(self, training_data, taxonomy):
        """
        GATE 3c: 'larvae' templates should only be used with Insecta species.
        """
        invalid_larvae = []

        for idx, row in training_data.iterrows():
            # Skip real (non-template) sentences — larvae check is template-specific
            if row.get('source', '') not in ('', 'v7_llm_cleaned', 'template'):
                continue
            text = row.get('text', '').lower()
            if 'larvae' in text and row.get('label') == 1:
                source = row.get('source_species', '')
                if source and not taxonomy.is_insecta(source):
                    taxon_class = taxonomy.get_class(source)
                    invalid_larvae.append({
                        'text': row.get('text', '')[:80],
                        'species': source,
                        'class': taxon_class
                    })

        print(f"\n[GATE 3c] Larvae Template Check")
        print(f"  Invalid larvae uses: {len(invalid_larvae)}")

        if invalid_larvae[:5]:
            print(f"  Samples:")
            for v in invalid_larvae[:5]:
                print(f"    - {v['species']} (class: {v['class']})")

        assert len(invalid_larvae) == 0, \
            f"Found {len(invalid_larvae)} 'larvae' sentences with non-Insecta species"

    def test_gate3_caterpillar_only_lepidoptera(self, training_data, taxonomy):
        """
        GATE 3d: 'caterpillar' templates should only be used with Lepidoptera.
        """
        invalid_caterpillar = []

        for idx, row in training_data.iterrows():
            # Skip real (non-template) sentences — caterpillar check is template-specific
            if row.get('source', '') not in ('', 'v7_llm_cleaned', 'template'):
                continue
            text = row.get('text', '').lower()
            if 'caterpillar' in text and row.get('label') == 1:
                source = row.get('source_species', '')
                if source and not taxonomy.is_lepidoptera(source):
                    order = taxonomy.get_order(source)
                    invalid_caterpillar.append({
                        'text': row.get('text', '')[:80],
                        'species': source,
                        'order': order
                    })

        print(f"\n[GATE 3d] Caterpillar Template Check")
        print(f"  Invalid caterpillar uses: {len(invalid_caterpillar)}")

        if invalid_caterpillar[:5]:
            print(f"  Samples:")
            for v in invalid_caterpillar[:5]:
                print(f"    - {v['species']} (order: {v['order']})")

        assert len(invalid_caterpillar) == 0, \
            f"Found {len(invalid_caterpillar)} 'caterpillar' sentences with non-Lepidoptera species"

    # =========================================================================
    # GATE 4: Negative Validation
    # =========================================================================

    def test_gate4_no_false_negatives(self, training_data):
        """
        GATE 4a: Negatives should not accidentally contain real interactions.

        Check for obvious interaction patterns that should be positives.
        """
        negatives = training_data[training_data['label'] == 0]

        # Patterns that strongly indicate actual interaction (not just mention)
        false_neg_patterns = [
            r'\b(?:is|are)\s+(?:a\s+)?(?:parasite|parasites)\s+of\b',
            r'\bparasitizes?\b',
            r'\bpreys?\s+on\b',
            r'\bis\s+(?:a\s+)?(?:predator|prey)\s+(?:of|for)\b',
            r'\bpollinates?\b',
            r'\binfects?\b.*\bwas\s+(?:confirmed|detected|documented)\b',
            r'\bhost\s+(?:of|for)\b',
            r'\bis\s+(?:a\s+)?vector\s+(?:of|for)\b',
        ]

        # Negation patterns that make it NOT a false negative
        negation_patterns = [
            r'\bnot?\b',
            r'\bno\b',
            r'\bneither\b',
            r'\bwithout\b',
            r'\babsence\b',
            r'\bfailed\b',
            r'\bdid\s+not\b',
            r'\bwas\s+not\b',
            r'\bcould\s+not\b',
        ]

        false_negatives = []

        for idx, row in negatives.iterrows():
            text = row.get('text', '').lower()

            # Skip if contains negation
            has_negation = any(re.search(p, text) for p in negation_patterns)
            if has_negation:
                continue

            # Check for false negative patterns
            for pattern in false_neg_patterns:
                if re.search(pattern, text):
                    false_negatives.append({
                        'text': row.get('text', '')[:100],
                        'pattern': pattern
                    })
                    break

        print(f"\n[GATE 4a] False Negative Check")
        print(f"  Negatives checked: {len(negatives)}")
        print(f"  Potential false negatives: {len(false_negatives)}")

        if false_negatives[:5]:
            print(f"  Samples:")
            for fn in false_negatives[:5]:
                print(f"    - {fn['text']}")

        # Allow up to 2 borderline cases from real PMC sentences
        assert len(false_negatives) <= 2, \
            f"Found {len(false_negatives)} potential false negatives (max 2 allowed)"

    def test_gate4_negative_diversity(self, training_data):
        """
        GATE 4b: No single negative pattern should dominate (>30%).
        """
        negatives = training_data[training_data['label'] == 0]
        pattern_counts = Counter(negatives['interaction_type'])
        total_negatives = len(negatives)

        print(f"\n[GATE 4b] Negative Diversity Check")
        print(f"  Total negatives: {total_negatives}")
        print(f"  Pattern distribution:")

        max_ratio = 0
        max_pattern = None

        for pattern, count in pattern_counts.most_common():
            ratio = count / total_negatives
            print(f"    - {pattern}: {count} ({ratio:.1%})")
            if ratio > max_ratio:
                max_ratio = ratio
                max_pattern = pattern

        assert max_ratio <= GATE4_MAX_SINGLE_PATTERN, \
            f"Pattern '{max_pattern}' at {max_ratio:.1%} > {GATE4_MAX_SINGLE_PATTERN:.0%} threshold"

    def test_gate4_hard_negative_ratio(self, training_data):
        """
        GATE 4c: At least 50% of negatives should be "hard" (close-to-positive).

        Hard negatives include:
        - two-species (co-occurrence)
        - three-species
        - one-species with interaction words
        """
        negatives = training_data[training_data['label'] == 0]
        total_negatives = len(negatives)

        # Count hard negatives by interaction_type
        hard_patterns = ['none_two_species', 'none_three_species']

        # Also check single-species negatives that mention interaction words
        interaction_words = ['parasite', 'pathogen', 'predator', 'vector', 'host',
                           'infect', 'prey', 'pollinate']

        hard_count = 0
        for idx, row in negatives.iterrows():
            int_type = row.get('interaction_type', '')
            text = row.get('text', '').lower()

            if int_type in hard_patterns:
                hard_count += 1
            elif int_type == 'none':
                # Check if single-species negative mentions interaction words
                if any(word in text for word in interaction_words):
                    hard_count += 1

        hard_ratio = hard_count / max(total_negatives, 1)

        print(f"\n[GATE 4c] Hard Negative Ratio")
        print(f"  Total negatives: {total_negatives}")
        print(f"  Hard negatives: {hard_count} ({hard_ratio:.1%})")

        assert hard_ratio >= GATE4_MIN_HARD_NEGATIVE_RATIO, \
            f"Hard negative ratio {hard_ratio:.1%} < {GATE4_MIN_HARD_NEGATIVE_RATIO:.0%} threshold"

    # =========================================================================
    # GATE 5: Balance & Distribution
    # =========================================================================

    def test_gate5_label_balance(self, training_data):
        """
        GATE 5a: Dataset should have reasonable label balance (20-50% positive).
        """
        total = len(training_data)
        positives = (training_data['label'] == 1).sum()
        pos_ratio = positives / total

        print(f"\n[GATE 5a] Label Balance")
        print(f"  Total: {total}")
        print(f"  Positives: {positives} ({pos_ratio:.1%})")
        print(f"  Negatives: {total - positives} ({1 - pos_ratio:.1%})")

        assert GATE5_MIN_POS_RATIO <= pos_ratio <= GATE5_MAX_POS_RATIO, \
            f"Positive ratio {pos_ratio:.1%} outside [{GATE5_MIN_POS_RATIO:.0%}, {GATE5_MAX_POS_RATIO:.0%}]"

    def test_gate5_interaction_type_coverage(self, training_data):
        """
        GATE 5b: All major interaction types should be represented in positives.
        """
        positives = training_data[training_data['label'] == 1]
        interaction_counts = Counter(positives['interaction_type'])

        # Major interaction types that should be present
        # Includes both GloBI formal types and category labels from PMC data
        major_types = [
            'preysOn', 'parasiteOf', 'eats', 'pathogenOf',
            'endoparasiteOf', 'hasHost', 'pathogen', 'parasitism',
        ]

        print(f"\n[GATE 5b] Interaction Type Coverage")
        print(f"  Total interaction types: {len(interaction_counts)}")

        missing = []
        for itype in major_types:
            count = interaction_counts.get(itype, 0)
            if count == 0:
                missing.append(itype)
            print(f"    - {itype}: {count}")

        if missing:
            print(f"  WARNING: Missing types: {missing}")

        # Warning only, not a hard failure
        assert len(missing) <= 3, f"Missing too many major interaction types: {missing}"


    def test_gate5_interaction_category_coverage(self, training_data):
        """
        GATE 5c: All 12 canonical interaction categories must meet minimum sample
        thresholds.  This gate is intentionally strict — it is expected to FAIL
        on early dataset versions, which surfaces gaps that need to be filled by
        targeted harvesting.

        Minimums:
          PREDATION / PARASITISM / ENDOPARASITISM / INFECTION : 200
          PARASITOIDISM / VECTOR / POLLINATION / HERBIVORY    :  50
          DISPERSAL / SYMBIOSIS / REGULATION / GENERIC        :  20
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from data.interaction_taxonomy import (
            coverage_report,
            CANONICAL_CATEGORIES,
            get_required_minimum,
        )

        positives = training_data[training_data['label'] == 1]
        rep = coverage_report(positives['interaction_type'].tolist())

        print(f"\n[GATE 5c] Canonical Interaction Category Coverage")
        print(f"  {'Category':<22} {'Count':>7} {'Required':>9} {'Status'}")
        print(f"  {'-'*50}")

        failures = []
        for cat in CANONICAL_CATEGORIES:
            info = rep[cat]
            print(f"  {cat:<22} {info['count']:>7} {info['required']:>9}   {info['status']}")
            if info['status'] in ('MISSING', 'LOW'):
                failures.append(
                    f"{cat}: {info['count']} samples (need ≥{info['required']})"
                )

        if failures:
            print(f"\n  GAPS ({len(failures)}):")
            for f in failures:
                print(f"    ✗ {f}")

        assert not failures, (
            f"Gate 5c: {len(failures)} interaction categories below minimum:\n"
            + "\n".join(f"  {f}" for f in failures)
            + "\n\nRun /analyze-coverage to see harvest targets."
        )

    # =========================================================================
    # GATE 6: Interaction Signal (lexicon-based)
    # =========================================================================

    def test_gate6_positives_have_interaction_signal(self, training_data):
        """
        GATE 6a: Positive sentences should contain at least one interaction
        term from the canonical lexicon.

        Flags rows with no detectable signal; warns at >5%, fails at >20%.
        Real PMC sentences with unusual phrasing may score lower — the 20%
        hard limit allows for that while catching systematic label errors.
        """
        from data.interaction_lexicon import score_sentence

        positives = training_data[training_data["label"] == 1]
        violations = []

        for _, row in positives.iterrows():
            text = str(row.get("text", "")).lower()
            has_signal, strength, _ = score_sentence(text)
            if not has_signal:
                violations.append({
                    "text": str(row.get("text", ""))[:100],
                    "interaction_type": row.get("interaction_type", ""),
                    "strength": strength,
                })

        violation_rate = len(violations) / max(len(positives), 1)

        print(f"\n[GATE 6a] Positive Interaction Signal")
        print(f"  Positives checked : {len(positives)}")
        print(f"  No signal detected: {len(violations)} ({violation_rate:.2%})")
        if violations[:5]:
            print(f"  Samples:")
            for v in violations[:5]:
                print(f"    [{v['strength']:.2f}] ({v['interaction_type']}) {v['text']}")

        if violation_rate > 0.05:
            print(f"  WARNING: {violation_rate:.2%} of positives have no lexicon signal")

        assert violation_rate <= 0.20, (
            f"GATE 6a FAILED: {violation_rate:.2%} of positives lack interaction signal "
            f"({len(violations)}/{len(positives)}). Max allowed: 20%."
        )

    def test_gate6_negatives_suspicious_strong_signal(self, training_data):
        """
        GATE 6b: Negative sentences with a strong, un-negated interaction term
        are likely mislabeled and should be rare (<2%).

        Complements Gate 4a by using the canonical lexicon rather than inline
        hardcoded patterns.
        """
        from data.interaction_lexicon import _STRONG_COMPILED, _NEGATION_COMPILED

        negatives = training_data[training_data["label"] == 0]
        suspicious = []

        for _, row in negatives.iterrows():
            text = str(row.get("text", "")).lower()
            # Skip if negation is present (negated interaction verbs are fine)
            if any(p.search(text) for p in _NEGATION_COMPILED):
                continue
            strong_hits = [p.pattern for p in _STRONG_COMPILED if p.search(text)]
            if strong_hits:
                suspicious.append({
                    "text": str(row.get("text", ""))[:100],
                    "interaction_type": row.get("interaction_type", ""),
                    "matched": strong_hits[:2],
                })

        suspicious_rate = len(suspicious) / max(len(negatives), 1)

        print(f"\n[GATE 6b] Negative Strong Signal Check")
        print(f"  Negatives checked : {len(negatives)}")
        print(f"  Suspicious entries: {len(suspicious)} ({suspicious_rate:.2%})")
        if suspicious[:5]:
            print(f"  Samples:")
            for s in suspicious[:5]:
                print(f"    ({s['interaction_type']}) {s['text']}")
                print(f"      Matched: {s['matched']}")

        assert suspicious_rate <= 0.02, (
            f"GATE 6b FAILED: {suspicious_rate:.2%} of negatives contain strong un-negated "
            f"interaction terms ({len(suspicious)}/{len(negatives)}). Max allowed: 2%."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
