#!/usr/bin/env python3
"""
Targeted harvest of pathogen-host interaction sentences from Europe PMC.

Uses explicit "X is a pathogen of Y" / "X infects Y" query patterns with
well-known host-pathogen pairs from GloBI. Sentences are then validated
by Qwen3.5-122B. Target: 150-300 confirmed positives.

Usage:
    python scripts/fetch_pathogen_sentences.py
    python scripts/fetch_pathogen_sentences.py --max-positives 200 --dry-run
"""

import argparse
import re
import sys
import time
import json
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_FILE = BASE_DIR / "data/training/pathogen_harvested.csv"
CACHE_DIR = BASE_DIR / "data/pmc_cache"
CHECKPOINT_EVERY = 25

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:122b"

EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EPMC_FULL   = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

# Well-known pathogen-host pairs (pathogen, host) — diverse taxonomic coverage
PATHOGEN_PAIRS = [
    # Fungal pathogens
    ("Botrytis cinerea", "tomato"), ("Botrytis cinerea", "grape"),
    ("Fusarium oxysporum", "tomato"), ("Fusarium oxysporum", "wheat"),
    ("Magnaporthe oryzae", "rice"), ("Puccinia striiformis", "wheat"),
    ("Blumeria graminis", "barley"), ("Sclerotinia sclerotiorum", "soybean"),
    ("Phytophthora infestans", "potato"), ("Alternaria alternata", "citrus"),
    # Bacterial pathogens
    ("Pseudomonas syringae", "Arabidopsis"), ("Xanthomonas oryzae", "rice"),
    ("Ralstonia solanacearum", "tomato"), ("Erwinia amylovora", "apple"),
    ("Agrobacterium tumefaciens", "plant"), ("Salmonella", "chicken"),
    ("Salmonella enterica", "cattle"), ("Campylobacter jejuni", "poultry"),
    ("Staphylococcus aureus", "human"), ("Mycobacterium tuberculosis", "human"),
    ("Yersinia pestis", "rodent"), ("Borrelia burgdorferi", "deer"),
    # Viral pathogens
    ("Influenza", "bird"), ("Influenza virus", "human"),
    ("Tobacco mosaic virus", "tobacco"), ("Cucumber mosaic virus", "cucumber"),
    ("Tomato spotted wilt virus", "tomato"),
    # Oomycetes
    ("Phytophthora sojae", "soybean"), ("Plasmopara viticola", "grape"),
    ("Peronospora tabacina", "tobacco"),
    # Animal pathogens
    ("Trypanosoma brucei", "cattle"), ("Plasmodium falciparum", "human"),
    ("Leishmania", "dog"), ("Toxoplasma gondii", "cat"),
]

# Query templates that explicitly name the relationship
QUERY_TEMPLATES = [
    '"{pathogen}" "pathogen of" "{host}"',
    '"{pathogen}" "infects" "{host}"',
    '"{pathogen}" "causes disease" "{host}"',
    '"{pathogen}" "pathogenic" "{host}"',
]

PROMPT = (
    "Does this sentence describe a direct biotic interaction between two named organisms? "
    "Biotic interactions include: predation, parasitism, pollination, herbivory, mutualism, "
    "symbiosis, seed dispersal, competition, pathogen infection, or disease transmission. "
    "The sentence must describe an actual interaction occurring, not just mention organisms. "
    "Answer YES or NO only.\n\n"
    "Sentence: {sentence}"
)


def search_epmc(query: str, max_results: int = 50) -> list[str]:
    """Return list of PMCIDs."""
    try:
        r = requests.get(EPMC_SEARCH, params={
            "query": query + " HAS_FULLTEXT:Y SRC:PMC",
            "resultType": "lite",
            "pageSize": max_results,
            "format": "json",
        }, timeout=30)
        r.raise_for_status()
        results = r.json().get("resultList", {}).get("result", [])
        return [x["pmcid"] for x in results if x.get("pmcid")]
    except Exception:
        return []


def fetch_sentences(pmcid: str, pathogen: str, host: str) -> list[str]:
    """Extract candidate sentences from PMC full text."""
    cache = CACHE_DIR / f"{pmcid}.txt"
    if cache.exists():
        text = cache.read_text(errors="replace")
    else:
        try:
            r = requests.get(EPMC_FULL.format(pmcid=pmcid), timeout=30)
            r.raise_for_status()
            text = r.text
            cache.write_text(text, errors="replace")
        except Exception:
            return []

    # Strip XML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # Split into sentences (rough)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Keep sentences mentioning both organisms and an interaction keyword
    p_lower = pathogen.split()[0].lower()
    h_lower = host.split()[0].lower()
    interaction_kw = re.compile(
        r'infect|pathogen|caus.{0,15}disease|parasit|coloniz|virulence|host|susceptib', re.I
    )
    candidates = []
    for s in sentences:
        s = s.strip()
        if 20 < len(s) < 400 and p_lower in s.lower() and h_lower in s.lower():
            if interaction_kw.search(s):
                candidates.append(s)
    return candidates[:5]  # max 5 per article


def ask_qwen(sentence: str, timeout: int = 300) -> int:
    """Returns 1=YES, 0=NO, -1=unclear."""
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT.format(sentence=sentence)}],
            "stream": False, "think": False, "keep_alive": -1,
            "options": {"temperature": 0, "num_predict": 10, "num_ctx": 2048},
        }, timeout=timeout)
        r.raise_for_status()
        raw = r.json().get("message", {}).get("content", "").strip().upper()
        clean = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', raw)
        if re.match(r'^YES\b', clean): return 1
        if re.match(r'^NO\b', clean): return 0
        if 'YES' in clean[:20] and 'NO' not in clean[:20]: return 1
        if 'NO' in clean[:20]: return 0
        return -1
    except Exception:
        return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-positives", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
    done_texts: set[str] = set()
    results: list[dict] = []
    if OUT_FILE.exists():
        existing = pd.read_csv(OUT_FILE)
        done_texts = set(existing["text"].astype(str))
        results = existing.to_dict("records")
        n_pos = (existing.teacher_label == 1).sum()
        print(f"Resuming: {len(done_texts)} done, {n_pos} positives so far")
    else:
        n_pos = 0

    t0 = time.time()
    n_queries = 0

    for pathogen, host in PATHOGEN_PAIRS:
        if n_pos >= args.max_positives:
            break

        for tmpl in QUERY_TEMPLATES:
            if n_pos >= args.max_positives:
                break

            query = tmpl.format(pathogen=pathogen, host=host)
            pmcids = search_epmc(query, max_results=30)
            n_queries += 1

            for pmcid in pmcids[:10]:
                sents = fetch_sentences(pmcid, pathogen, host)
                for s in sents:
                    if s in done_texts:
                        continue
                    done_texts.add(s)

                    if args.dry_run:
                        print(f"  [dry-run] {s[:100]}")
                        continue

                    label = ask_qwen(s)
                    if label == 1:
                        n_pos += 1
                    results.append({
                        "text": s, "label": label, "teacher_label": label,
                        "interaction_type": "pathogenOf",
                        "source_species": pathogen, "target_species": host,
                        "source": "epmc_pathogen_targeted", "pmcid": pmcid,
                        "_source_file": "pathogen_harvested.csv",
                    })

                    if len(results) % CHECKPOINT_EVERY == 0:
                        pd.DataFrame(results).to_csv(OUT_FILE, index=False)
                        elapsed = time.time() - t0
                        print(f"  [{len(results)} sentences | {n_pos} YES | "
                              f"{elapsed/60:.0f}min elapsed]", flush=True)

            time.sleep(0.3)  # EPMC rate limit

    pd.DataFrame(results).to_csv(OUT_FILE, index=False)
    print(f"\n=== Done: {len(results)} sentences, {n_pos} Qwen-confirmed pathogenOf positives ===")
    print(f"Output: {OUT_FILE}")


if __name__ == "__main__":
    main()
