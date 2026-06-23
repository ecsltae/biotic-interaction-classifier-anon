#!/usr/bin/env python3
"""
Extract host-pathogen interaction sentences from training data.

This script filters the training dataset to extract sentences that are likely
related to host-pathogen interactions based on keyword matching.

Output: training_data_host_pathogen_subset.csv
"""

import pandas as pd
import re
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
TRAINING_FILE = DATA_DIR / "training" / "training_data_enhanced_20k.csv"
OUTPUT_FILE = DATA_DIR / "training" / "training_data_host_pathogen_subset.csv"

# Keywords for host-pathogen interactions
HP_KEYWORDS = [
    # Infection-related
    r'\binfect\w*\b',      # infect, infected, infection, infectious
    r'\bvirus\w*\b',       # virus, viral, viruses
    r'\bbacteri\w*\b',     # bacteria, bacterial, bacterium
    r'\bpathogen\w*\b',    # pathogen, pathogenic, pathogens
    r'\bdisease\w*\b',     # disease, diseases
    r'\bparasit\w*\b',     # parasite, parasitic, parasitism

    # Transmission
    r'\btransmit\w*\b',    # transmit, transmission, transmitted
    r'\bvector\w*\b',      # vector, vectors
    r'\bspillover\b',
    r'\bzoonoti\w*\b',     # zoonotic, zoonosis

    # Host-related
    r'\bhost[s]?\b',       # host, hosts
    r'\breservoir\b',
    r'\bcarrier\b',

    # Interaction types
    r'\bcoloniz\w*\b',     # colonize, colonization
    r'\binvad\w*\b',       # invade, invasion
    r'\bsusceptib\w*\b',   # susceptible, susceptibility
    r'\bresistan\w*\b',    # resistant, resistance

    # Specific pathogens (common terms)
    r'\bfungal\b',
    r'\bfungi\b',
    r'\bprotozoa\w*\b',
    r'\bprion\b',
]

def extract_host_pathogen_sentences(input_file: Path, output_file: Path) -> dict:
    """
    Extract sentences containing host-pathogen related keywords.

    Returns statistics about the extraction.
    """
    print(f"Reading data from: {input_file}")
    df = pd.read_csv(input_file)

    original_size = len(df)
    print(f"Total sentences in dataset: {original_size}")

    # Combine all keywords into one regex pattern
    pattern = '|'.join(HP_KEYWORDS)

    # Filter sentences
    mask = df['passage'].str.lower().str.contains(pattern, regex=True, na=False)
    hp_subset = df[mask].copy()

    # Add a column showing which keywords matched
    def find_matches(text):
        text_lower = str(text).lower()
        matches = []
        for kw in HP_KEYWORDS:
            if re.search(kw, text_lower):
                # Clean up regex for display
                clean_kw = kw.replace(r'\b', '').replace(r'\w*', '*')
                matches.append(clean_kw)
        return '; '.join(matches)

    hp_subset['matched_keywords'] = hp_subset['passage'].apply(find_matches)

    # Save
    hp_subset.to_csv(output_file, index=False)
    print(f"\nHost-pathogen subset saved to: {output_file}")

    # Statistics
    stats = {
        'original_size': original_size,
        'hp_subset_size': len(hp_subset),
        'percentage': len(hp_subset) / original_size * 100,
        'positive_samples': hp_subset['label'].sum(),
        'negative_samples': len(hp_subset) - hp_subset['label'].sum(),
    }

    # Keyword frequency
    keyword_counts = {}
    for kw in HP_KEYWORDS:
        clean_kw = kw.replace(r'\b', '').replace(r'\w*', '*')
        count = df['passage'].str.lower().str.contains(kw, regex=True, na=False).sum()
        keyword_counts[clean_kw] = count
    stats['keyword_counts'] = keyword_counts

    return stats


def main():
    stats = extract_host_pathogen_sentences(TRAINING_FILE, OUTPUT_FILE)

    print("\n" + "="*60)
    print("EXTRACTION STATISTICS")
    print("="*60)
    print(f"Original dataset size:  {stats['original_size']:,}")
    print(f"Host-pathogen subset:   {stats['hp_subset_size']:,} ({stats['percentage']:.1f}%)")
    print(f"  - Positive samples:   {stats['positive_samples']:,}")
    print(f"  - Negative samples:   {stats['negative_samples']:,}")

    print("\nKeyword frequencies:")
    for kw, count in sorted(stats['keyword_counts'].items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {kw:20s}: {count:,}")


if __name__ == "__main__":
    main()
