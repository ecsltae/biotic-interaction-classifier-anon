#!/usr/bin/env python3
"""
Batch article processor — biotic interaction sentence detection.

Two-stage pipeline:
  1. GloBI term pre-filter  (local, pure Python, ~1ms/sentence)
  2. Distilled BiomedBERT   (remote API, ~14ms/sentence on CPU)

Usage:
    # Process a folder of .txt files
    python process_articles.py --input articles/ --output results.csv --api http://172.30.120.7:8003

    # Process a CSV with an 'abstract' or 'full_text' column
    python process_articles.py --input articles.csv --text-col full_text --output results.csv

    # Adjust threshold (default 0.25 = balanced F1=0.808; higher = more precise)
    python process_articles.py --input articles/ --output results.csv --threshold 0.4
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterator

import requests

# ── GloBI pre-filter (pure Python, no model needed) ───────────────────────

_GLOBI_PATTERN = None
_SPECIES_AUTOMATON = None

def _load_globi_pattern(terms_file: Path | None = None) -> re.Pattern:
    """Build combined regex from GloBI interaction terms file, or use fallback list."""
    global _GLOBI_PATTERN
    if _GLOBI_PATTERN is not None:
        return _GLOBI_PATTERN

    # Try to load from the interaction_dict.csv (if running alongside the classifier repo)
    if terms_file is None:
        candidate = Path(__file__).parent.parent / "data/processed/interaction_dict.csv"
        if candidate.exists():
            terms_file = candidate

    globi_terms = []
    if terms_file and terms_file.exists():
        import csv as _csv
        with open(terms_file) as f:
            for row in _csv.DictReader(f):
                t = row.get("interaction", "").strip()
                if len(t) >= 3:
                    globi_terms.append(re.escape(t.lower()))
        print(f"  Filter: loaded {len(globi_terms)} GloBI terms from {terms_file}", flush=True)
    else:
        print(f"  Filter: no interaction_dict.csv — using biomedical vocabulary only", flush=True)

    # Biomedical interaction vocabulary (GloBI misses these; validated 100% recall on EP-relax)
    # Use stems (not full words) so variants are caught: endophyt → endophyte/endophytic/endophytes
    biomedical = (
        r'infect|parasit|\bhost\b|pathogen|\bvector\b|zoonot|symbiont|symbioti|'
        r'endophyt|mycorrhiz|\bnodule|nematod|fungal|\bfungi\b|bacteri|viral|\bvirus\b|protozoa|'
        r'transmit|reservoir|definitive host|intermediate host|'
        r'harbour|harbor|coloniz|colonise|life cycle|'
        r'\bprey\b|predat|pollina|feed on|feeds on|\beats\b|ingest|'
        r'herbivory|herbivore|mutuali|commensali|kleptoparasit|'
        # Common disease/pathogen names not covered by GloBI or binomial gazetteer
        r'\bHIV\b|\bAIDS\b|\bSARS\b|\bMERS\b|\bCOVID\b|'
        r'chickenpox|smallpox|monkeypox|\bmeasles\b|\bmumps\b|\brubella\b|'
        r'\bmalaria\b|\bdengue\b|\bebola\b|\brabies\b|\btuberculosis\b|\bTB\b|'
        r'influenza|\bflu\b|\bplague\b|\bcholera\b|\btyphus\b|\btyphoid\b|'
        r'\bsyphilis\b|\bleprosy\b|\banthrax\b|\bbotulism\b|\btetanus\b|'
        r'leishmanian|\btrypanosomia|schistosom|\btoxoplasm|\bcryptospor'
    )

    if globi_terms:
        alternation = "|".join(globi_terms)
        _GLOBI_PATTERN = re.compile(
            r"(?:(?<!\w)(?:" + alternation + r")(?!\w))|(?:" + biomedical + r")",
            re.IGNORECASE
        )
    else:
        _GLOBI_PATTERN = re.compile(biomedical, re.IGNORECASE)
    return _GLOBI_PATTERN


def _load_species_automaton() -> "ahocorasick.Automaton":
    """Build Aho-Corasick automaton over clean binomial species names (Genus species)."""
    global _SPECIES_AUTOMATON
    if _SPECIES_AUTOMATON is not None:
        return _SPECIES_AUTOMATON
    try:
        import ahocorasick
    except ImportError:
        print("  Species fallback: pyahocorasick not installed — species check disabled", flush=True)
        return None

    species_csv = Path(__file__).parent.parent / "data/processed/species_dict.csv"
    if not species_csv.exists():
        print(f"  Species fallback: {species_csv} not found — species check disabled", flush=True)
        return None

    binomial_re = re.compile(r'^[A-Z][a-z]+ [a-z]+$')
    A = ahocorasick.Automaton()
    count = 0
    with open(species_csv) as f:
        next(f)  # skip header
        for line in f:
            name = line.strip()
            if binomial_re.match(name):
                key = name.lower()
                if key not in A:
                    A.add_word(key, name)
                    count += 1
    A.make_automaton()
    _SPECIES_AUTOMATON = A
    print(f"  Species fallback: loaded {count:,} binomial names", flush=True)
    return _SPECIES_AUTOMATON


def _has_species_mention(sentence: str) -> bool:
    """Check for any binomial species name (Genus species) via Aho-Corasick. ~0.5ms/sentence."""
    automaton = _load_species_automaton()
    if automaton is None or len(automaton) == 0:
        return False
    text_lower = sentence.lower()
    for end_idx, value in automaton.iter(text_lower):
        key = value.lower()
        start = end_idx - len(key) + 1
        end = end_idx + 1
        before_ok = start == 0 or not text_lower[start - 1].isalpha()
        after_ok  = end >= len(text_lower) or not text_lower[end].isalpha()
        if before_ok and after_ok:
            return True
    return False


def has_interaction_signal(sentence: str) -> bool:
    """
    Pre-filter: passes sentence to classifier if either condition holds:
      1. Known interaction term (GloBI vocab + biomedical terms) — catches most cases
      2. Any binomial species name present — safety net for interactions phrased
         without standard interaction verbs (e.g. "X was found in Y")

    Goal is zero false negatives, not precision. False positives are handled by
    the classifier downstream.
    """
    if _load_globi_pattern().search(sentence):
        return True
    return _has_species_mention(sentence)


# ── Sentence splitter ─────────────────────────────────────────────────────

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
_XML_TAG    = re.compile(r'<[^>]+>')

def split_sentences(text: str) -> list[str]:
    """Simple sentence splitter — good enough for biomedical text."""
    # Strip XML/HTML tags so species names inside <italic> are visible to the filter
    text = _XML_TAG.sub(' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = _SENT_SPLIT.split(text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


# ── Article loaders ───────────────────────────────────────────────────────

def iter_txt_folder(folder: Path) -> Iterator[tuple[str, str]]:
    """Yield (article_id, full_text) from a folder of .txt files."""
    files = sorted(folder.glob("*.txt"))
    if not files:
        print(f"WARNING: no .txt files found in {folder}", file=sys.stderr)
    for f in files:
        yield f.stem, f.read_text(encoding="utf-8", errors="replace")


def iter_csv_file(path: Path, text_col: str, id_col: str | None) -> Iterator[tuple[str, str]]:
    """Yield (article_id, full_text) from a CSV file."""
    import pandas as pd
    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found. Available: {df.columns.tolist()}")
    id_col = id_col or (df.columns[0] if id_col is None else id_col)
    for _, row in df.iterrows():
        aid = str(row[id_col]) if id_col in df.columns else str(_)
        yield aid, str(row[text_col])


# ── API caller ────────────────────────────────────────────────────────────

def classify_batch(sentences: list[str], api_url: str, threshold: float,
                   retries: int = 3) -> list[dict]:
    """Send a batch of sentences to the API, return list of result dicts."""
    payload = {"sentences": sentences, "threshold": threshold}
    for attempt in range(retries):
        try:
            r = requests.post(f"{api_url}/batch", json=payload, timeout=60)
            r.raise_for_status()
            return r.json()["results"]
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return []


# ── Main pipeline ─────────────────────────────────────────────────────────

def process(articles: Iterator[tuple[str, str]], api_url: str, output: Path,
            threshold: float, batch_size: int, terms_file: Path | None) -> None:

    _load_globi_pattern(terms_file)  # build interaction pattern once
    _load_species_automaton()        # build species automaton once (4.2M names)

    total_articles = 0
    total_sentences = 0
    total_filtered = 0
    total_filtered_by_species = 0   # sentences passed only by species fallback
    total_positive = 0
    t_start = time.time()

    with open(output, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=[
            "article_id", "sentence", "label", "probability", "threshold_used"
        ])
        writer.writeheader()

        pending_sentences: list[str] = []
        pending_meta: list[tuple[str, str]] = []  # (article_id, sentence)

        def flush_batch():
            nonlocal total_positive
            if not pending_sentences:
                return
            results = classify_batch(pending_sentences, api_url, threshold)
            for (aid, sent), res in zip(pending_meta, results):
                writer.writerow({
                    "article_id": aid,
                    "sentence": sent,
                    "label": res["label"],
                    "probability": res["probability"],
                    "threshold_used": res["threshold_used"],
                })
                if res["label"] == 1:
                    total_positive += 1
            fout.flush()
            pending_sentences.clear()
            pending_meta.clear()

        for article_id, text in articles:
            total_articles += 1
            sentences = split_sentences(text)
            total_sentences += len(sentences)

            for sent in sentences:
                has_int = bool(_load_globi_pattern().search(sent))
                if not has_int:
                    if not _has_species_mention(sent):
                        continue
                    total_filtered_by_species += 1
                total_filtered += 1
                pending_sentences.append(sent)
                pending_meta.append((article_id, sent))
                if len(pending_sentences) >= batch_size:
                    flush_batch()

            if total_articles % 100 == 0:
                elapsed = time.time() - t_start
                rate = total_articles / elapsed
                print(f"  {total_articles} articles | {total_sentences} sentences | "
                      f"{total_filtered} passed filter | {total_positive} positive | "
                      f"{rate:.1f} art/s | eta {(4000-total_articles)/rate/60:.0f} min",
                      flush=True)

        flush_batch()

    elapsed = time.time() - t_start
    filter_rate = total_filtered / total_sentences * 100 if total_sentences else 0
    species_rate = total_filtered_by_species / total_sentences * 100 if total_sentences else 0
    print(f"\n=== Done ===")
    print(f"  Articles:           {total_articles}")
    print(f"  Total sentences:    {total_sentences}")
    print(f"  Passed filter:      {total_filtered} ({filter_rate:.1f}%)")
    print(f"    via interaction:  {total_filtered - total_filtered_by_species}")
    print(f"    via species only: {total_filtered_by_species} ({species_rate:.1f}% of all sentences)")
    print(f"  Positive interactions: {total_positive} ({total_positive/total_filtered*100:.1f}% of filtered)" if total_filtered else "")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Output: {output}")


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch biotic interaction detector")
    parser.add_argument("--input",      required=True, help="Folder of .txt files OR a .csv file")
    parser.add_argument("--output",     required=True, help="Output CSV path")
    parser.add_argument("--api",        default="http://172.30.120.7:8003",
                        help="Classifier API base URL (default: %(default)s)")
    parser.add_argument("--threshold",  type=float, default=0.25,
                        help="Classification threshold (default 0.25 = EP F1=0.808)")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Sentences per API call (max 500, default 500)")
    parser.add_argument("--text-col",   default="full_text",
                        help="Column name for text in CSV input (default: full_text)")
    parser.add_argument("--id-col",     default=None,
                        help="Column name for article ID in CSV input (default: first column)")
    parser.add_argument("--terms-file", default=None,
                        help="Path to interaction_dict.csv for GloBI filter")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    terms = Path(args.terms_file) if args.terms_file else None

    # Health check
    print(f"Checking API at {args.api} ...", flush=True)
    try:
        r = requests.get(f"{args.api}/health", timeout=5)
        info = r.json()
        print(f"  Model: {info['model']}  EP F1={info['ep_relax_f1']}  device={info['device']}", flush=True)
    except Exception as e:
        print(f"ERROR: cannot reach API at {args.api}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Input: {inp}", flush=True)
    print(f"Threshold: {args.threshold}", flush=True)

    if inp.is_dir():
        articles = iter_txt_folder(inp)
    elif inp.suffix == ".csv":
        articles = iter_csv_file(inp, args.text_col, args.id_col)
    else:
        print(f"ERROR: --input must be a folder of .txt files or a .csv file", file=sys.stderr)
        sys.exit(1)

    process(articles, args.api, out, args.threshold, args.batch_size, terms)


if __name__ == "__main__":
    main()
