"""
GloBI Data Loader

Load biotic interactions from GloBI (Global Biotic Interactions) with article references.
Downloads interactions.tsv from Zenodo and filters for entries with DOI/PMID references.
"""

import os
import pandas as pd
import requests
from typing import Optional, List, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GloBI Zenodo URL - update to latest release as needed
# Find latest at: https://zenodo.org/search?q=globi%20interactions
# Version 0.7 from June 2024: https://zenodo.org/records/11552565
GLOBI_ZENODO_URL = "https://zenodo.org/api/records/11552565/files/interactions.tsv.gz/content"

# Key columns from GloBI interactions.tsv
GLOBI_COLUMNS = [
    'sourceTaxonName',
    'sourceTaxonId',
    'sourceTaxonKingdomName',
    'targetTaxonName',
    'targetTaxonId',
    'targetTaxonKingdomName',
    'interactionTypeName',
    'referenceDoi',
    'referenceCitation',
    'referenceUrl',
]


@dataclass
class GlobiInteraction:
    """Represents a single GloBI interaction with article reference."""
    source_taxon: str
    source_taxon_id: str
    source_kingdom: str
    target_taxon: str
    target_taxon_id: str
    target_kingdom: str
    interaction_type: str
    doi: Optional[str]
    pmid: Optional[str]
    citation: Optional[str]
    reference_url: Optional[str]


def download_globi_data(
    output_path: str,
    url: str = GLOBI_ZENODO_URL,
    force: bool = False
) -> str:
    """
    Download GloBI interactions.tsv.gz from Zenodo.

    Args:
        output_path: Directory to save the file
        url: Zenodo URL for the interactions file
        force: Re-download even if file exists

    Returns:
        Path to the downloaded file
    """
    os.makedirs(output_path, exist_ok=True)
    filename = os.path.basename(url)
    filepath = os.path.join(output_path, filename)

    if os.path.exists(filepath) and not force:
        logger.info(f"File already exists: {filepath}")
        return filepath

    logger.info(f"Downloading GloBI data from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0

    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = (downloaded / total_size) * 100
                if downloaded % (10 * 1024 * 1024) < 8192:  # Log every ~10MB
                    logger.info(f"Downloaded {downloaded / (1024*1024):.1f}MB ({pct:.1f}%)")

    logger.info(f"Download complete: {filepath}")
    return filepath


def extract_pmid_from_url(url: Optional[str]) -> Optional[str]:
    """Extract PMID from PubMed URL if present."""
    # Handle NaN, None, and empty strings
    if url is None or (isinstance(url, float) and pd.isna(url)) or url == '':
        return None
    if not isinstance(url, str):
        return None
    if 'pubmed' in url.lower() or 'ncbi.nlm.nih.gov' in url.lower():
        # Try to extract PMID from URL patterns like:
        # https://pubmed.ncbi.nlm.nih.gov/12345678/
        # https://www.ncbi.nlm.nih.gov/pubmed/12345678
        import re
        match = re.search(r'/(\d{6,9})/?$', url)
        if match:
            return match.group(1)
        match = re.search(r'pubmed[/=](\d{6,9})', url, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def load_globi_interactions(
    filepath: str,
    nrows: Optional[int] = None,
    chunksize: int = 100000
) -> pd.DataFrame:
    """
    Load GloBI interactions from TSV file.

    Args:
        filepath: Path to interactions.tsv or interactions.tsv.gz
        nrows: Limit number of rows to load (for testing)
        chunksize: Process in chunks to manage memory

    Returns:
        DataFrame with interaction data
    """
    logger.info(f"Loading GloBI interactions from {filepath}...")

    # Determine which columns exist in the file
    # First, read just the header
    sample = pd.read_csv(filepath, sep='\t', nrows=0, compression='infer')
    available_cols = [c for c in GLOBI_COLUMNS if c in sample.columns]

    if not available_cols:
        raise ValueError(f"None of expected columns found. Available: {list(sample.columns)[:20]}")

    logger.info(f"Loading columns: {available_cols}")

    # Load the data
    df = pd.read_csv(
        filepath,
        sep='\t',
        usecols=available_cols,
        nrows=nrows,
        low_memory=False,
        compression='infer'
    )

    logger.info(f"Loaded {len(df):,} interactions")
    return df


def filter_interactions_with_refs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter interactions to only those with article references (DOI or PMID).

    Args:
        df: DataFrame from load_globi_interactions

    Returns:
        Filtered DataFrame with reference info
    """
    logger.info("Filtering interactions with references...")

    # Extract PMID from referenceUrl where available
    if 'referenceUrl' in df.columns:
        df['pmid'] = df['referenceUrl'].apply(extract_pmid_from_url)
    else:
        df['pmid'] = None

    # Check for DOI
    has_doi = df['referenceDoi'].notna() & (df['referenceDoi'] != '')
    has_pmid = df['pmid'].notna()

    # Filter to rows with at least one reference
    df_with_refs = df[has_doi | has_pmid].copy()

    logger.info(f"Found {len(df_with_refs):,} interactions with references "
                f"({has_doi.sum():,} with DOI, {has_pmid.sum():,} with PMID)")

    return df_with_refs


def filter_by_interaction_types(
    df: pd.DataFrame,
    interaction_types: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Filter to specific interaction types.

    Args:
        df: DataFrame with interactions
        interaction_types: List of interaction type names to include.
                          If None, include all.

    Returns:
        Filtered DataFrame
    """
    if interaction_types is None:
        return df

    # Normalize interaction type names for matching
    df['_interaction_norm'] = df['interactionTypeName'].str.lower().str.strip()
    types_norm = [t.lower().strip() for t in interaction_types]

    filtered = df[df['_interaction_norm'].isin(types_norm)].copy()
    filtered.drop(columns=['_interaction_norm'], inplace=True)

    logger.info(f"Filtered to {len(filtered):,} interactions of types: {interaction_types}")
    return filtered


def get_unique_interactions(
    df: pd.DataFrame,
    dedupe_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Deduplicate interactions based on species pair and interaction type.

    Args:
        df: DataFrame with interactions
        dedupe_cols: Columns to use for deduplication

    Returns:
        Deduplicated DataFrame
    """
    if dedupe_cols is None:
        dedupe_cols = ['sourceTaxonName', 'targetTaxonName', 'interactionTypeName']

    # Keep first occurrence (which has a reference)
    df_unique = df.drop_duplicates(subset=dedupe_cols, keep='first')

    logger.info(f"Deduplicated to {len(df_unique):,} unique interactions")
    return df_unique


def load_and_prepare_globi(
    data_dir: str,
    download: bool = True,
    nrows: Optional[int] = None,
    interaction_types: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Main function: download, load, and prepare GloBI data for sentence extraction.

    Args:
        data_dir: Directory for data storage
        download: Whether to download if file doesn't exist
        nrows: Limit rows (for testing)
        interaction_types: Filter to specific interaction types

    Returns:
        Prepared DataFrame ready for article fetching
    """
    filepath = os.path.join(data_dir, "interactions.tsv.gz")

    # Download if needed
    if download and not os.path.exists(filepath):
        download_globi_data(data_dir)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"GloBI data not found at {filepath}. "
            f"Set download=True or manually download from {GLOBI_ZENODO_URL}"
        )

    # Load and process
    df = load_globi_interactions(filepath, nrows=nrows)
    df = filter_interactions_with_refs(df)

    if interaction_types:
        df = filter_by_interaction_types(df, interaction_types)

    df = get_unique_interactions(df)

    return df


def get_interaction_stats(df: pd.DataFrame) -> dict:
    """Get statistics about the loaded interactions."""
    stats = {
        'total_interactions': len(df),
        'unique_source_taxa': df['sourceTaxonName'].nunique(),
        'unique_target_taxa': df['targetTaxonName'].nunique(),
        'interaction_types': df['interactionTypeName'].value_counts().to_dict(),
        'has_doi': df['referenceDoi'].notna().sum(),
        'has_pmid': df['pmid'].notna().sum() if 'pmid' in df.columns else 0,
    }

    if 'sourceTaxonKingdomName' in df.columns:
        stats['source_kingdoms'] = df['sourceTaxonKingdomName'].value_counts().to_dict()
    if 'targetTaxonKingdomName' in df.columns:
        stats['target_kingdoms'] = df['targetTaxonKingdomName'].value_counts().to_dict()

    return stats


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Load GloBI interaction data")
    parser.add_argument("--data-dir", default="./data/globi", help="Data directory")
    parser.add_argument("--nrows", type=int, default=None, help="Limit rows for testing")
    parser.add_argument("--no-download", action="store_true", help="Don't download if missing")

    args = parser.parse_args()

    df = load_and_prepare_globi(
        data_dir=args.data_dir,
        download=not args.no_download,
        nrows=args.nrows
    )

    print("\n=== GloBI Data Statistics ===")
    stats = get_interaction_stats(df)
    for key, value in stats.items():
        if isinstance(value, dict) and len(value) > 10:
            print(f"{key}: {len(value)} unique values")
        else:
            print(f"{key}: {value}")

    print(f"\nSample interactions:")
    print(df.head(10).to_string())
