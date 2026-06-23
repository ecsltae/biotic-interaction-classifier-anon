#!/usr/bin/env python3
"""
Explore PHI-base data structure and extract useful statistics.

This script analyzes the downloaded PHI-base dataset to understand:
- Entity types and distributions
- Relation patterns
- Overlap with our training data
"""

import pandas as pd
from pathlib import Path
from collections import Counter

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
PHI_BASE_FILE = DATA_DIR / "external" / "phi_base" / "phi-base_4.13_data.csv"
OUTPUT_DIR = DATA_DIR / "external" / "phi_base"


def analyze_phi_base():
    """Analyze PHI-base dataset structure."""
    print("Loading PHI-base data...")
    df = pd.read_csv(PHI_BASE_FILE)

    print(f"\n{'='*60}")
    print("PHI-BASE DATASET ANALYSIS")
    print(f"{'='*60}")

    print(f"\nTotal records: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    # Key columns for knowledge graph
    key_columns = [
        'pathogen_species', 'host_species', 'disease',
        'mutant_phenotype', 'interaction_phenotype', 'gene'
    ]

    print("\n--- KEY COLUMNS FOR KNOWLEDGE GRAPH ---")
    for col in key_columns:
        if col in df.columns:
            non_null = df[col].notna().sum()
            unique = df[col].nunique()
            print(f"\n{col}:")
            print(f"  Non-null values: {non_null:,} ({non_null/len(df)*100:.1f}%)")
            print(f"  Unique values: {unique:,}")
            if unique <= 20:
                print(f"  Values: {df[col].dropna().unique().tolist()}")
            else:
                print(f"  Top 10: {df[col].value_counts().head(10).to_dict()}")

    # Extract host-pathogen pairs
    print("\n--- HOST-PATHOGEN PAIRS ---")
    pairs = df[['host_species', 'pathogen_species']].dropna()
    print(f"Total pairs with both host and pathogen: {len(pairs):,}")

    # Unique hosts and pathogens
    unique_hosts = pairs['host_species'].nunique()
    unique_pathogens = pairs['pathogen_species'].nunique()
    print(f"Unique host species: {unique_hosts}")
    print(f"Unique pathogen species: {unique_pathogens}")

    # Top hosts
    print("\nTop 10 host species:")
    for host, count in pairs['host_species'].value_counts().head(10).items():
        print(f"  {host}: {count:,}")

    # Top pathogens
    print("\nTop 10 pathogen species:")
    for pathogen, count in pairs['pathogen_species'].value_counts().head(10).items():
        print(f"  {pathogen}: {count:,}")

    # Interaction phenotypes (relation types)
    print("\n--- INTERACTION PHENOTYPES (RELATION TYPES) ---")
    if 'mutant_phenotype' in df.columns:
        phenotypes = df['mutant_phenotype'].dropna()
        print(f"Total with phenotype: {len(phenotypes):,}")
        print("\nPhenotype distribution:")
        for phenotype, count in phenotypes.value_counts().head(15).items():
            print(f"  {phenotype}: {count:,}")

    # Diseases
    print("\n--- DISEASES ---")
    if 'disease' in df.columns:
        diseases = df['disease'].dropna()
        print(f"Unique diseases: {diseases.nunique()}")
        print("\nTop 10 diseases:")
        for disease, count in diseases.value_counts().head(10).items():
            print(f"  {disease}: {count:,}")

    # Create summary for knowledge graph construction
    print("\n--- KNOWLEDGE GRAPH POTENTIAL ---")

    # Count potential triples
    hp_triples = len(pairs)
    disease_triples = df[['pathogen_species', 'disease']].dropna().drop_duplicates()

    print(f"Potential (Host, INFECTED_BY, Pathogen) triples: {hp_triples:,}")
    print(f"Potential (Pathogen, CAUSES, Disease) triples: {len(disease_triples):,}")

    # Save extracted entities for later use
    entities = {
        'hosts': pairs['host_species'].unique().tolist(),
        'pathogens': pairs['pathogen_species'].unique().tolist(),
        'diseases': df['disease'].dropna().unique().tolist()
    }

    import json
    entities_file = OUTPUT_DIR / "phi_base_entities.json"
    with open(entities_file, 'w') as f:
        json.dump(entities, f, indent=2)
    print(f"\nEntities saved to: {entities_file}")

    return df


def extract_kg_triples(df):
    """Extract knowledge graph triples from PHI-base."""
    triples = []

    for _, row in df.iterrows():
        host = row.get('host_species')
        pathogen = row.get('pathogen_species')
        disease = row.get('disease')
        phenotype = row.get('mutant_phenotype')

        # Host-Pathogen interaction
        if pd.notna(host) and pd.notna(pathogen):
            relation = 'INFECTED_BY'
            if pd.notna(phenotype):
                if 'resistance' in str(phenotype).lower():
                    relation = 'RESISTANT_TO'
                elif 'suscept' in str(phenotype).lower():
                    relation = 'SUSCEPTIBLE_TO'
            triples.append({
                'head': host,
                'head_type': 'HOST',
                'relation': relation,
                'tail': pathogen,
                'tail_type': 'PATHOGEN',
                'source': 'PHI-base'
            })

        # Pathogen-Disease relation
        if pd.notna(pathogen) and pd.notna(disease):
            triples.append({
                'head': pathogen,
                'head_type': 'PATHOGEN',
                'relation': 'CAUSES_DISEASE',
                'tail': disease,
                'tail_type': 'DISEASE',
                'source': 'PHI-base'
            })

    triples_df = pd.DataFrame(triples).drop_duplicates()

    output_file = OUTPUT_DIR / "phi_base_kg_triples.csv"
    triples_df.to_csv(output_file, index=False)
    print(f"\nKG triples saved to: {output_file}")
    print(f"Total unique triples: {len(triples_df):,}")

    return triples_df


if __name__ == "__main__":
    df = analyze_phi_base()
    print("\n" + "="*60)
    print("EXTRACTING KNOWLEDGE GRAPH TRIPLES")
    print("="*60)
    extract_kg_triples(df)
