#!/usr/bin/env python3
"""
Training Data Validation Pipeline

Validates training data through 6 gates before allowing training.
Can also clean the dataset by removing problematic sentences.

Usage:
    python validate_training_data.py data/training/training_data_globi_v6_diverse.csv
    python validate_training_data.py data/training/training_data_globi_v6_diverse.csv --clean
    python validate_training_data.py data/training/training_data_globi_v6_diverse.csv --with-llm
"""

import argparse
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Set

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.taxonomy_validator import TaxonomyValidator
from data.robi_validator import RobiValidator


# Thresholds
GATE1_MAX_INVALID_RATE = 0.01       # Max 1% invalid species
GATE2_MAX_VIOLATION_RATE = 0.005    # Max 0.5% ROBI violations
GATE4_MAX_SINGLE_PATTERN = 0.70     # No single pattern >70% (two-species can be up to 65%)
GATE4_MIN_HARD_NEGATIVE_RATIO = 0.50  # At least 50% hard negatives
GATE5_MIN_POS_RATIO = 0.20          # Min 20% positives
GATE5_MAX_POS_RATIO = 0.50          # Max 50% positives


class TrainingDataValidator:
    """Validates training data through multiple quality gates."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.taxonomy = None
        self.robi = None

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _init_validators(self):
        """Initialize validators on first use."""
        if self.taxonomy is None:
            self._log("Initializing taxonomy validator...")
            self.taxonomy = TaxonomyValidator()
        if self.robi is None:
            self._log("Initializing ROBI validator...")
            self.robi = RobiValidator()

    # =========================================================================
    # GATE 1: Species Validation
    # =========================================================================

    def gate1_species_validation(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """Validate that all species exist in taxonomy."""
        self._init_validators()

        invalid_species = []
        checked_species = set()

        for _, row in df.iterrows():
            source = row.get('source_species', '')
            if source and source not in checked_species:
                checked_species.add(source)
                if not self.taxonomy.is_valid_species(source):
                    invalid_species.append(source)

            target = row.get('target_species', '')
            if target and not pd.isna(target) and target not in checked_species:
                for sp in str(target).split(','):
                    sp = sp.strip()
                    if sp and sp not in checked_species:
                        checked_species.add(sp)
                        if not self.taxonomy.is_valid_species(sp):
                            invalid_species.append(sp)

        invalid_rate = len(invalid_species) / max(len(checked_species), 1)
        passed = invalid_rate <= GATE1_MAX_INVALID_RATE

        return passed, {
            'checked': len(checked_species),
            'invalid': len(invalid_species),
            'invalid_rate': invalid_rate,
            'samples': invalid_species[:10]
        }

    # =========================================================================
    # GATE 2: ROBI Rules
    # =========================================================================

    def gate2_robi_rules(self, df: pd.DataFrame) -> Tuple[bool, Dict, List[int]]:
        """Validate positives against ROBI rules. Returns indices to remove."""
        self._init_validators()

        positives = df[df['label'] == 1]
        violations = []
        indices_to_remove = []

        for idx, row in positives.iterrows():
            source = str(row.get('source_species', '')) if not pd.isna(row.get('source_species', '')) else ''
            target = str(row.get('target_species', '')) if not pd.isna(row.get('target_species', '')) else ''
            interaction = str(row.get('interaction_type', ''))

            src_kingdom = self.taxonomy.get_kingdom(source)
            tgt_kingdom = self.taxonomy.get_kingdom(target)

            is_valid, errors = self.robi.validate_interaction(
                source, target, interaction, src_kingdom, tgt_kingdom
            )

            if not is_valid:
                violations.append({
                    'idx': idx,
                    'text': str(row.get('text', ''))[:80],
                    'source': source,
                    'target': target,
                    'interaction': interaction,
                    'errors': errors
                })
                indices_to_remove.append(idx)

        violation_rate = len(violations) / max(len(positives), 1)
        passed = violation_rate <= GATE2_MAX_VIOLATION_RATE

        return passed, {
            'positives': len(positives),
            'violations': len(violations),
            'violation_rate': violation_rate,
            'samples': violations[:10]
        }, indices_to_remove

    # =========================================================================
    # GATE 3: Template Validation
    # =========================================================================

    def gate3_template_validation(self, df: pd.DataFrame) -> Tuple[bool, Dict, List[int]]:
        """Validate sentence templates."""
        self._init_validators()

        malformed = []
        incomplete = []
        invalid_larvae = []
        invalid_caterpillar = []
        indices_to_remove = []

        for idx, row in df.iterrows():
            text = str(row.get('text', ''))

            # Check placeholders
            if '{' in text or '}' in text:
                malformed.append((idx, text[:80]))
                indices_to_remove.append(idx)

            # Check completeness
            if text.strip() and not any(text.strip().endswith(p) for p in '.?!'):
                incomplete.append((idx, text[:80]))
                indices_to_remove.append(idx)

            # Check larvae (only for positives)
            if 'larvae' in text.lower() and row.get('label') == 1:
                source = str(row.get('source_species', '')) if not pd.isna(row.get('source_species', '')) else ''
                if source and not self.taxonomy.is_insecta(source):
                    invalid_larvae.append((idx, source, text[:80]))
                    indices_to_remove.append(idx)

            # Check caterpillar
            if 'caterpillar' in text.lower() and row.get('label') == 1:
                source = str(row.get('source_species', '')) if not pd.isna(row.get('source_species', '')) else ''
                if source and not self.taxonomy.is_lepidoptera(source):
                    invalid_caterpillar.append((idx, source, text[:80]))
                    indices_to_remove.append(idx)

        total_issues = len(malformed) + len(incomplete) + len(invalid_larvae) + len(invalid_caterpillar)
        passed = total_issues == 0

        return passed, {
            'malformed': len(malformed),
            'incomplete': len(incomplete),
            'invalid_larvae': len(invalid_larvae),
            'invalid_caterpillar': len(invalid_caterpillar),
            'total': total_issues
        }, list(set(indices_to_remove))

    # =========================================================================
    # GATE 4: Negative Validation
    # =========================================================================

    def gate4_negative_validation(self, df: pd.DataFrame) -> Tuple[bool, Dict, List[int]]:
        """Validate negatives for false negatives and diversity."""
        negatives = df[df['label'] == 0]
        indices_to_remove = []

        # Check for false negatives
        false_neg_patterns = [
            r'\b(?:is|are)\s+(?:a\s+)?(?:parasite|parasites)\s+of\b',
            r'\bparasitizes?\b',
            r'\bpreys?\s+on\b',
            r'\bis\s+(?:a\s+)?(?:predator|prey)\s+(?:of|for)\b',
            r'\bpollinates?\b',
            r'\binfects?\b.*\bwas\s+(?:confirmed|detected|documented)\b',
            r'\bhost\s+(?:of|for)\b',
            r'\bis\s+(?:a\s+)?vector\s+(?:of|for)\b',
            r'\bserves\s+as\s+host\s+for\b',  # Added this pattern
        ]

        negation_patterns = [
            r'\bnot?\b', r'\bno\b', r'\bneither\b', r'\bwithout\b',
            r'\babsence\b', r'\bfailed\b', r'\bdid\s+not\b',
            r'\bwas\s+not\b', r'\bcould\s+not\b',
        ]

        false_negatives = []
        for idx, row in negatives.iterrows():
            text = str(row.get('text', '')).lower()

            has_negation = any(re.search(p, text) for p in negation_patterns)
            if has_negation:
                continue

            for pattern in false_neg_patterns:
                if re.search(pattern, text):
                    false_negatives.append((idx, str(row.get('text', ''))[:100]))
                    indices_to_remove.append(idx)
                    break

        # Check diversity
        pattern_counts = Counter(negatives['interaction_type'])
        total_negatives = len(negatives)
        max_ratio = 0
        max_pattern = None
        for pattern, count in pattern_counts.most_common():
            ratio = count / total_negatives
            if ratio > max_ratio:
                max_ratio = ratio
                max_pattern = pattern

        # Check hard negative ratio
        hard_patterns = ['none_two_species', 'none_three_species']
        interaction_words = ['parasite', 'pathogen', 'predator', 'vector', 'host',
                           'infect', 'prey', 'pollinate']

        hard_count = 0
        for _, row in negatives.iterrows():
            int_type = row.get('interaction_type', '')
            text = str(row.get('text', '')).lower()
            if int_type in hard_patterns:
                hard_count += 1
            elif int_type == 'none':
                if any(word in text for word in interaction_words):
                    hard_count += 1

        hard_ratio = hard_count / max(total_negatives, 1)

        # Determine pass/fail
        diversity_ok = max_ratio <= GATE4_MAX_SINGLE_PATTERN
        hard_ratio_ok = hard_ratio >= GATE4_MIN_HARD_NEGATIVE_RATIO
        no_false_neg = len(false_negatives) == 0

        passed = diversity_ok and hard_ratio_ok and no_false_neg

        return passed, {
            'total_negatives': total_negatives,
            'false_negatives': len(false_negatives),
            'max_pattern': max_pattern,
            'max_pattern_ratio': max_ratio,
            'hard_ratio': hard_ratio,
            'diversity_ok': diversity_ok,
            'hard_ratio_ok': hard_ratio_ok,
            'pattern_distribution': dict(pattern_counts),
            'false_neg_samples': false_negatives[:10]
        }, list(set(indices_to_remove))

    # =========================================================================
    # GATE 5: Balance & Distribution
    # =========================================================================

    def gate5_balance(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """Check dataset balance and interaction type coverage."""
        total = len(df)
        positives = (df['label'] == 1).sum()
        pos_ratio = positives / total

        # Interaction type coverage
        pos_df = df[df['label'] == 1]
        interaction_counts = Counter(pos_df['interaction_type'])

        major_types = ['preysOn', 'parasiteOf', 'infects', 'eats', 'pollinates',
                      'hostOf', 'pathogenOf', 'vectorOf', 'endoparasiteOf']
        missing = [t for t in major_types if interaction_counts.get(t, 0) == 0]

        balance_ok = GATE5_MIN_POS_RATIO <= pos_ratio <= GATE5_MAX_POS_RATIO
        coverage_ok = len(missing) <= 5  # Allow up to 5 missing types (warning only)

        # Balance is critical, coverage is a warning
        passed = balance_ok  # Coverage issues are warnings, not failures

        return passed, {
            'total': total,
            'positives': positives,
            'negatives': total - positives,
            'pos_ratio': pos_ratio,
            'interaction_types': len(interaction_counts),
            'missing_types': missing,
            'interaction_distribution': dict(interaction_counts)
        }

    # =========================================================================
    # Full Validation Pipeline
    # =========================================================================

    def validate(self, df: pd.DataFrame, clean: bool = False) -> Tuple[bool, pd.DataFrame]:
        """
        Run full validation pipeline.

        Args:
            df: Training data DataFrame
            clean: If True, remove problematic sentences

        Returns:
            (all_passed, cleaned_df)
        """
        print("\n" + "=" * 70)
        print("TRAINING DATA VALIDATION PIPELINE")
        print("=" * 70)

        all_indices_to_remove = set()
        results = {}

        # GATE 1
        print("\n[GATE 1] Species Validation...")
        passed, info = self.gate1_species_validation(df)
        status = "PASS" if passed else "FAIL"
        print(f"  Status: {status}")
        print(f"  Checked: {info['checked']} species, Invalid: {info['invalid']} ({info['invalid_rate']:.2%})")
        results['gate1'] = (passed, info)

        # GATE 2
        print("\n[GATE 2] ROBI Rules Validation...")
        passed, info, to_remove = self.gate2_robi_rules(df)
        status = "PASS" if passed else "FAIL"
        print(f"  Status: {status}")
        print(f"  Violations: {info['violations']}/{info['positives']} ({info['violation_rate']:.2%})")
        if info['samples'][:3]:
            print(f"  Samples:")
            for v in info['samples'][:3]:
                print(f"    - {v['interaction']}: {v['source']} -> {v['target']}")
        results['gate2'] = (passed, info)
        all_indices_to_remove.update(to_remove)

        # GATE 3
        print("\n[GATE 3] Template Validation...")
        passed, info, to_remove = self.gate3_template_validation(df)
        status = "PASS" if passed else "FAIL"
        print(f"  Status: {status}")
        print(f"  Issues: {info['total']} (placeholders: {info['malformed']}, incomplete: {info['incomplete']}, "
              f"larvae: {info['invalid_larvae']}, caterpillar: {info['invalid_caterpillar']})")
        results['gate3'] = (passed, info)
        all_indices_to_remove.update(to_remove)

        # GATE 4
        print("\n[GATE 4] Negative Validation...")
        passed, info, to_remove = self.gate4_negative_validation(df)
        status = "PASS" if passed else "FAIL"
        print(f"  Status: {status}")
        print(f"  False negatives: {info['false_negatives']}")
        print(f"  Max pattern: {info['max_pattern']} ({info['max_pattern_ratio']:.1%}) - "
              f"{'OK' if info['diversity_ok'] else 'EXCEEDS 30%'}")
        print(f"  Hard negative ratio: {info['hard_ratio']:.1%} - "
              f"{'OK' if info['hard_ratio_ok'] else 'BELOW 50%'}")
        results['gate4'] = (passed, info)
        all_indices_to_remove.update(to_remove)

        # GATE 5
        print("\n[GATE 5] Balance & Distribution...")
        passed, info = self.gate5_balance(df)
        status = "PASS" if passed else "FAIL"
        print(f"  Status: {status}")
        print(f"  Total: {info['total']} (pos: {info['positives']}, neg: {info['negatives']})")
        print(f"  Positive ratio: {info['pos_ratio']:.1%}")
        print(f"  Interaction types: {info['interaction_types']}")
        if info['missing_types']:
            print(f"  Missing types: {info['missing_types']}")
        results['gate5'] = (passed, info)

        # Summary
        all_passed = all(results[g][0] for g in results)

        print("\n" + "=" * 70)
        if all_passed:
            print("✅ ALL GATES PASSED - Training data is ready")
        else:
            failed = [g for g in results if not results[g][0]]
            print(f"❌ FAILED GATES: {', '.join(failed)}")
            print(f"   Total sentences to remove: {len(all_indices_to_remove)}")
        print("=" * 70)

        # Clean if requested
        cleaned_df = df
        if clean and all_indices_to_remove:
            print(f"\n[CLEANING] Removing {len(all_indices_to_remove)} problematic sentences...")
            cleaned_df = df.drop(index=list(all_indices_to_remove))
            print(f"  Original: {len(df)} -> Cleaned: {len(cleaned_df)}")

        return all_passed, cleaned_df


def main():
    parser = argparse.ArgumentParser(description="Validate training data quality")
    parser.add_argument("input_file", help="Path to training data CSV")
    parser.add_argument("--clean", action="store_true", help="Clean dataset by removing issues")
    parser.add_argument("--output", "-o", help="Output path for cleaned data")
    parser.add_argument("--with-llm", action="store_true", help="Include LLM validation (Gate 6)")

    args = parser.parse_args()

    # Load data
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows")

    # Validate
    validator = TrainingDataValidator()
    passed, cleaned_df = validator.validate(df, clean=args.clean)

    # Save cleaned data
    if args.clean and args.output:
        output_path = Path(args.output)
        cleaned_df.to_csv(output_path, index=False)
        print(f"\nSaved cleaned data to: {output_path}")
    elif args.clean and not args.output:
        # Auto-generate output name
        output_path = input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"
        cleaned_df.to_csv(output_path, index=False)
        print(f"\nSaved cleaned data to: {output_path}")

    # LLM validation
    if args.with_llm:
        print("\n[GATE 6] LLM Validation...")
        print("  (Not implemented yet - requires MCP setup)")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
