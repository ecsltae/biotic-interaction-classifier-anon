#!/usr/bin/env python3
"""
Enrich existing training sentences with SibiLS gold NER surface forms.

The SibiLS MongoDB stores, per passage:
  species1_form    list — exact surface strings for species 1 as they appear in text
  species2_form    list — exact surface strings for species 2
  interaction_form list — exact interaction verb/phrase surface strings

These are outputs of the SibiLS annotation pipeline (Aho-Corasick over NCBI Taxonomy,
ICTV, and GloBI interaction terms) — much better than gazetteer matching for NER because
they handle abbreviations ("A. mellifera"), common names, and exact interaction verbs.

IMPORTANT: The MongoDB sentences are NOT reliable for interaction classification labels
(int_present=1 does not reliably mean the sentence describes a biotic interaction).
This script only uses surface forms for NER enrichment on sentences that ALREADY have
trusted classification labels from our pipeline (training_data_v14.csv etc.).

How it works:
  1. Load input CSV (must have text + label columns, trusted labels)
  2. Query MongoDB: for each sentence, find a matching passage by text prefix
  3. Add columns: source_sp_forms, target_sp_forms, interaction_forms (JSON lists)
  4. Save enriched CSV — rows without a MongoDB match keep the columns empty

Usage:
    python fetch_sibils_ner_data.py \
        --input  classifier/data/training/training_data_v14.csv \
        --output classifier/data/training/training_data_v14_ner_enriched.csv

    # Only enrich rows that came from SibiLS (faster — they're guaranteed to exist)
    python fetch_sibils_ner_data.py \
        --input  classifier/data/training/training_data_v14.csv \
        --output ... --sibils-only
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

MONGO_URI  = "mongodb://sibils-mongodb.lan.text-analytics.ch:27017/"
DB_NAME    = "sibils_v4_2"
COLLECTION = "med25_r1_v5.5_passages"


def enrich_with_surface_forms(
    df: pd.DataFrame,
    col,
    sibils_only: bool = False,
) -> pd.DataFrame:
    """
    For each row in df, look up the passage in MongoDB and add surface form columns.
    Matches on first 120 characters of text (fast index lookup).
    """
    df = df.copy()
    df["source_sp_forms"]   = None
    df["target_sp_forms"]   = None
    df["interaction_forms"] = None

    if sibils_only:
        mask = df["source"].str.contains("sibils", case=False, na=False)
        candidates = df[mask]
        print(f"Enriching {len(candidates)} SibiLS-source rows (--sibils-only)", flush=True)
    else:
        candidates = df
        print(f"Enriching all {len(candidates)} rows", flush=True)

    # Build lookup: text_prefix → index in df
    text_index: dict[str, list[int]] = {}
    for idx in candidates.index:
        key = str(df.at[idx, "text"])[:120].lower().strip()
        text_index.setdefault(key, []).append(idx)

    # Batch query MongoDB
    n_matched = 0
    cursor = col.find(
        {"passage": {"$exists": True}},
        {"passage": 1, "species1_form": 1, "species2_form": 1, "interaction_form": 1, "_id": 0},
    )
    for doc in cursor:
        passage = doc.get("passage", "").strip()
        key = passage[:120].lower().strip()
        if key not in text_index:
            continue
        for idx in text_index[key]:
            df.at[idx, "source_sp_forms"]   = json.dumps(doc.get("species1_form", []))
            df.at[idx, "target_sp_forms"]   = json.dumps(doc.get("species2_form", []))
            df.at[idx, "interaction_forms"] = json.dumps(doc.get("interaction_form", []))
            n_matched += 1
        del text_index[key]  # avoid re-processing
        if not text_index:
            break  # all rows matched

    coverage = n_matched / len(candidates) * 100 if len(candidates) else 0
    print(f"Matched {n_matched}/{len(candidates)} rows ({coverage:.1f}%)", flush=True)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Enrich training CSV with SibiLS NER surface forms (NER only, no new labels)"
    )
    parser.add_argument("--input",       required=True, help="Input CSV (text + label, trusted labels)")
    parser.add_argument("--output",      required=True, help="Output enriched CSV path")
    parser.add_argument("--sibils-only", action="store_true",
                        help="Only enrich rows where source contains 'sibils' (faster)")
    parser.add_argument("--mongo-uri",   default=MONGO_URI)
    args = parser.parse_args()

    try:
        from pymongo import MongoClient
    except ImportError:
        print("ERROR: pip install pymongo", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}", flush=True)

    print(f"Connecting to {args.mongo_uri} ...", flush=True)
    client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    col = client[DB_NAME][COLLECTION]
    print("Connected.", flush=True)

    df_enriched = enrich_with_surface_forms(df, col, args.sibils_only)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_enriched.to_csv(out, index=False)

    enriched_count = df_enriched["source_sp_forms"].notna().sum()
    print(f"\nSaved {len(df_enriched)} rows to {out}")
    print(f"  Rows with NER surface forms: {enriched_count}/{len(df_enriched)}")
    print(f"  Classification labels unchanged — only NER columns added")


if __name__ == "__main__":
    main()
