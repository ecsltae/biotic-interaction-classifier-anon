#!/usr/bin/env python3
"""
fetch_sibils_triplets.py — Mine SIBiLS MongoDB for gap-category positives and hard negatives.

The SIBiLS MongoDB (sibils_v4_2.med25_r1_v5.5_passages) indexes biodiversity literature passages
with pre-extracted species pairs and interaction terms.  This script uses the `score` subdocument
to identify:

  - HIGH-QUALITY POSITIVES: int_present=1, int_middle=1, specific interaction verb forms
    → sentence describes a biotic interaction between two named species
    → targeted at the 6 gap categories missing from v12 (gate 5c fails)

  - HARD NEGATIVES: two species present + an interaction-word exists in passage but the passage
    does not describe a real biotic interaction (e.g., methodological context, co-occurrence study)
    → FLAN-T5 confuses these with positives → most informative negatives for precision training

  Outputs a CSV in the same format as other training source files so it can be passed to
  build_v12_dataset.py via --extra-sources.

Usage:
    # Harvest positives for gap categories + hard negatives (default)
    python classifier/scripts/fetch_sibils_triplets.py \
        --model classifier/models/flan_t5_v12 \
        --output classifier/data/training/sibils_triplets_mined.csv

    # Positives only, no FLAN-T5 validation (faster, less precise)
    python classifier/scripts/fetch_sibils_triplets.py \
        --no-validate \
        --output classifier/data/training/sibils_triplets_mined_raw.csv

    # Hard negatives only
    python classifier/scripts/fetch_sibils_triplets.py \
        --hard-negatives-only \
        --model classifier/models/flan_t5_v12 \
        --output classifier/data/training/sibils_hard_negatives.csv

MongoDB details:
    host: sibils-mongodb.lan.text-analytics.ch:27017
    db:   sibils_v4_2
    col:  med25_r1_v5.5_passages

Document schema (relevant fields):
    passage          str   — full sentence / passage text
    species1_form    list  — surface forms of species 1 (e.g. ["Apis mellifera", "A. mellifera"])
    species2_form    list  — surface forms of species 2
    interaction_form list  — surface forms of detected interaction term (can be empty [])
    score            dict  — {int_present, int_middle, sp1_pt, sp2_pt, sp1_len, sp2_len, sp_dist, size}
    triplet_key      str   — "species1_id;species2_id;ro_interaction_id"
"""

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

BASE_DIR = Path("/path/to/MetaP/classifier")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MONGO_URI = "mongodb://sibils-mongodb.lan.text-analytics.ch:27017/"
DB_NAME = "sibils_v4_2"
COLLECTION_NAME = "med25_r1_v5.5_passages"

# ── interaction patterns per gap category ────────────────────────────────────────
# These match interaction_form values OR passage text.

CATEGORY_PATTERNS: Dict[str, List[str]] = {
    "PARASITOIDISM": [
        r"\bparasitoid(?:ism|s|ize[sd]?)?\b",
        r"\bparasitoid\s+of\b",
        r"\bichneumon\b",
        r"\bbraconid\b",
        r"\bparasiti[zs]e(?:d|s)?\s+(?:by|in)\b",
        r"\bparasitic\s+wasp\b",
        r"\bectoparasitoid\b",
        r"\bendoparasitoid\b",
        r"\bparasitoid\s+wasp\b",
    ],
    "VECTOR": [
        r"\bvector(?:s|ed)?\s+(?:of|for)\b",
        r"\btransmit(?:s|ted|ting|ter|ters)\b",
        r"\bvector(?:ial)?\s+(?:capacity|competence|transmission)\b",
        r"\bbitten\s+by\b",
        r"\bblood\s+feeding\b",
        r"\btick(?:s|ed)?\s+(?:transmit|carry|vector)\b",
        r"\bmosquito(?:es)?\s+transmit\b",
        r"\bvectored\s+by\b",
        r"\bspreads?\s+(?:the\s+)?(?:virus|pathogen|disease)\b",
    ],
    "HERBIVORY": [
        r"\bherbivor(?:e|es|y|ous|ism)\b",
        r"\bgraz(?:e|es|ed|ing)\s+(?:on|upon)?\b",
        r"\bbrows(?:e|es|ed|ing)\s+(?:on|upon)?\b",
        r"\bfeed(?:s|ing)?\s+on\b",
        r"\bfed\s+on\b",
        r"\bconsum(?:e|es|ed|ing|ption)\s+(?:of\s+)?(?:plant|leaf|leaf|foliage|vegetation)\b",
        r"\bphytophag(?:ous|y|e|es)\b",
        r"\bfoliivor(?:e|y|ous)\b",
        r"\bdefoliat(?:e|es|ed|ing|ion)\b",
        r"\bleaf\s+miner\b",
        r"\broot\s+feeder\b",
        r"\bsap.sucker\b",
    ],
    "DISPERSAL": [
        r"\bdispers(?:e|es|ed|ing|al)\s+(?:of|by)?\s*seeds?\b",
        r"\bseed\s+dispers(?:al|er|ers)\b",
        r"\bzoochory\b",
        r"\bendozoochory\b",
        r"\bexozoochory\b",
        r"\bfruit\s+(?:eating|consumption|eating)\b",
        r"\bmyrmecochory\b",
        r"\bcarr(?:y|ies|ied|ying)\s+seeds?\b",
        r"\bseed\s+predation\b",
        r"\bfrugiv(?:ore|orous|ory)\b",
    ],
    "REGULATION": [
        r"\bregulat(?:e|es|ed|ing|ion)\s+(?:population|growth|abundance)\b",
        r"\bbiological\s+control\b",
        r"\bbiocontrol\b",
        r"\bsuppress(?:es|ed|ing|ion)\s+(?:of\s+)?(?:population|growth)\b",
        r"\binhibit(?:s|ed|ing|ion)\s+(?:of\s+)?(?:growth|reproduction)\b",
        r"\bpopulation\s+(?:control|regulation|suppression)\b",
        r"\bantagonist(?:ic)?\b",
        r"\bcompetitive\s+(?:exclusion|inhibition|suppression)\b",
    ],
    "GENERIC": [
        r"\bassociat(?:e[sd]?|ion)\s+(?:with|between)\b",
        r"\binteract(?:s|ed|ing|ion)\s+(?:with|between)\b",
        r"\bco-occur(?:s|red|ring|rence)\b",
        r"\bco.exist(?:s|ed|ing|ence)\b",
        r"\brelationship\s+(?:with|between)\b",
        r"\btrophic\s+(?:interaction|relationship|level)\b",
        r"\bfood\s+web\b",
        r"\bfood\s+chain\b",
    ],
}

# Interaction forms that indicate NON-biotic-interaction context (useful for hard negatives)
METHODOLOGICAL_INTERACTION_FORMS = {
    "present in", "presence of", "carried", "compared with", "contribute to",
    "contamination", "associated with", "found in", "isolated from",
    "detected in", "observed in", "used in", "tested in", "applied to",
    "treated with", "combined with", "measured in", "extracted from",
}


def compile_patterns(category: str) -> List[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in CATEGORY_PATTERNS[category]]


COMPILED = {cat: compile_patterns(cat) for cat in CATEGORY_PATTERNS}


def matches_category(text: str, category: str) -> bool:
    return any(pat.search(text) for pat in COMPILED[category])


def any_biotic_interaction(text: str) -> bool:
    return any(matches_category(text, cat) for cat in CATEGORY_PATTERNS)


# ── sentence quality filters ────────────────────────────────────────────────────

def is_valid_passage(passage: str) -> bool:
    if not passage or len(passage) < 50 or len(passage) > 800:
        return False
    words = passage.split()
    if len(words) < 8:
        return False
    if sum(c.isalpha() for c in passage) < 30:
        return False
    # Reject clearly methodological sentences
    method_patterns = [
        r"\bwe\s+(?:used|employed|treated|incubated|inoculated)\b",
        r"\bwere\s+(?:inoculated|treated|incubated|exposed)\s+(?:with|to)\b",
        r"\bin\s+vitro\b",
        r"\bpetri\s+dish\b",
        r"\bcell\s+culture\b",
        r"\bexperimental\s+(?:infection|setup|design)\b",
    ]
    for p in method_patterns:
        if re.search(p, passage, re.IGNORECASE):
            return False
    return True


# ── MongoDB connection ──────────────────────────────────────────────────────────

def get_collection():
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
        client.server_info()
        return client[DB_NAME][COLLECTION_NAME]
    except Exception as e:
        print(f"ERROR: MongoDB connection failed: {e}", file=sys.stderr)
        print(f"  Tried: {MONGO_URI}", file=sys.stderr)
        sys.exit(1)


# ── FLAN-T5 validation ──────────────────────────────────────────────────────────

def load_flan_t5(model_path: Path):
    """Load FLAN-T5 for sentence scoring."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    print(f"Loading FLAN-T5 from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))
    model.to(DEVICE)
    model.eval()
    yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id  = tokenizer.encode("no",  add_special_tokens=False)[0]
    return model, tokenizer, yes_id, no_id


PROMPT = (
    "Does this sentence describe a biotic interaction between two organisms?\n"
    "Sentence: {sentence}\n"
    "Answer:"
)


def score_with_flan_t5(model, tokenizer, yes_id: int, no_id: int,
                        sentences: List[str], batch_size: int = 32) -> np.ndarray:
    prompts = [PROMPT.format(sentence=s) for s in sentences]
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = tokenizer(batch, max_length=256, padding=True,
                            truncation=True, return_tensors="pt").to(DEVICE)
            bos = torch.full((len(batch), 1), model.config.decoder_start_token_id,
                             dtype=torch.long).to(DEVICE)
            out = model(input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                        decoder_input_ids=bos)
            logits = out.logits[:, 0, :]
            lp = torch.log_softmax(logits.float(), dim=-1)
            yes_lp = lp[:, yes_id].cpu().numpy()
            no_lp  = lp[:, no_id].cpu().numpy()
            prob_yes = np.exp(yes_lp) / (np.exp(yes_lp) + np.exp(no_lp))
            all_probs.extend(prob_yes.tolist())
    return np.array(all_probs)


# ── harvesting ──────────────────────────────────────────────────────────────────

def harvest_positives(
    collection,
    categories: List[str],
    target_per_category: int = 200,
    max_scan: int = 500_000,
) -> List[dict]:
    """
    Harvest passages where an interaction verb for the target category appears
    AND the interaction term is positioned between the two species (int_middle=1).
    """
    results = []
    by_category = {cat: [] for cat in categories}

    # Query: two species present, interaction term detected between them
    query = {
        "species1_form": {"$ne": []},
        "species2_form": {"$ne": []},
        "score.int_present": 1,
        "score.int_middle": 1,
    }

    scanned = 0
    print(f"\nHarvesting positives for: {categories}")
    print(f"  Scanning up to {max_scan:,} MongoDB documents ...")

    for doc in collection.find(query):
        scanned += 1
        if scanned % 100_000 == 0:
            total = sum(len(v) for v in by_category.values())
            print(f"    Scanned {scanned:,}  |  found {total} candidates ...")

        if scanned > max_scan:
            break

        passage = doc.get("passage", "").strip()
        if not is_valid_passage(passage):
            continue

        # Check which gap category this passage belongs to
        matched_cat = None
        for cat in categories:
            if matches_category(passage, cat):
                matched_cat = cat
                break

        if matched_cat is None:
            continue
        if len(by_category[matched_cat]) >= target_per_category:
            continue

        sp1 = doc.get("species1_form", [])
        sp2 = doc.get("species2_form", [])

        by_category[matched_cat].append({
            "text": passage,
            "source_species": sp1[0] if sp1 else "",
            "target_species": sp2[0] if sp2 else "",
            "candidate_category": matched_cat,
            "int_middle": doc.get("score", {}).get("int_middle", 0),
            "sp1_pt": doc.get("score", {}).get("sp1_pt", 0),
            "sp2_pt": doc.get("score", {}).get("sp2_pt", 0),
        })

        # Stop early if all categories are filled
        if all(len(v) >= target_per_category for v in by_category.values()):
            break

    for cat, items in by_category.items():
        print(f"  {cat}: {len(items)} candidates")
        results.extend(items)

    print(f"  Total positive candidates: {len(results)} (scanned {scanned:,})")
    return results


def harvest_hard_negatives(
    collection,
    target: int = 500,
    max_scan: int = 200_000,
) -> List[dict]:
    """
    Harvest hard negatives: passages with two species + an interaction term is present
    BUT the term is NOT positioned between them (int_middle=0) OR the interaction_form
    is clearly methodological.

    These look like positives to the model (two species, interaction word in sentence)
    but don't describe a real biotic interaction.
    """
    results = []

    # int_present=1 but int_middle=0 → interaction word is in the sentence but not between species
    query = {
        "species1_form": {"$ne": []},
        "species2_form": {"$ne": []},
        "score.int_present": 1,
        "score.int_middle": 0,
    }

    scanned = 0
    seen_passages = set()
    print(f"\nHarvesting {target} hard negatives ...")

    import random
    for doc in collection.find(query):
        scanned += 1
        if scanned > max_scan:
            break
        if len(results) >= target:
            break

        # Sample 30% to get diverse passages
        if random.random() > 0.3:
            continue

        passage = doc.get("passage", "").strip()
        if not is_valid_passage(passage):
            continue

        # Skip passages that clearly describe biotic interactions
        if any_biotic_interaction(passage):
            continue

        key = passage[:80].lower()
        if key in seen_passages:
            continue
        seen_passages.add(key)

        sp1 = doc.get("species1_form", [])
        sp2 = doc.get("species2_form", [])

        # Prefer passages with methodological interaction forms
        iforms = [f.lower().strip() for f in doc.get("interaction_form", [])]
        is_methodological = any(f in METHODOLOGICAL_INTERACTION_FORMS for f in iforms)

        results.append({
            "text": passage,
            "source_species": sp1[0] if sp1 else "",
            "target_species": sp2[0] if sp2 else "",
            "candidate_category": "NEGATIVE",
            "int_middle": 0,
            "sp1_pt": doc.get("score", {}).get("sp1_pt", 0),
            "sp2_pt": doc.get("score", {}).get("sp2_pt", 0),
            "is_methodological": is_methodological,
        })

    print(f"  Hard negative candidates: {len(results)} (scanned {scanned:,})")
    return results


# ── FLAN-T5 filtering ───────────────────────────────────────────────────────────

def filter_with_flan_t5(
    candidates: List[dict],
    model, tokenizer, yes_id: int, no_id: int,
    pos_threshold: float = 0.65,
    neg_lo: float = 0.25,
    neg_hi: float = 0.60,
    is_negative: bool = False,
) -> List[dict]:
    """
    For positives: keep candidates where FLAN-T5 score >= pos_threshold.
    For negatives: keep candidates where FLAN-T5 score is in [neg_lo, neg_hi]
                   (model is confused → most informative hard negatives).
    """
    if not candidates:
        return []

    sentences = [c["text"] for c in candidates]
    probs = score_with_flan_t5(model, tokenizer, yes_id, no_id, sentences)

    filtered = []
    for cand, prob in zip(candidates, probs):
        cand["flan_t5_score"] = round(float(prob), 4)
        if is_negative:
            # Hard negatives: model is confused (prob in ambiguous range)
            if neg_lo <= prob <= neg_hi:
                filtered.append(cand)
        else:
            # Positives: model agrees
            if prob >= pos_threshold:
                filtered.append(cand)

    label = "negative" if is_negative else "positive"
    print(f"  After FLAN-T5 filtering ({label}): {len(filtered)}/{len(candidates)} kept")
    return filtered


# ── output formatting ───────────────────────────────────────────────────────────

CATEGORY_TO_INTERACTION_TYPE = {
    "PARASITOIDISM": "parasitoidOf",
    "VECTOR": "vectorOf",
    "HERBIVORY": "grazesOn",
    "DISPERSAL": "dispersesSeedsOf",
    "REGULATION": "negativelyRegulates",
    "GENERIC": "interactsWith",
    "NEGATIVE": "interactsWith",
}


def to_training_rows(candidates: List[dict], label: int) -> List[dict]:
    rows = []
    for c in candidates:
        cat = c.get("candidate_category", "GENERIC")
        rows.append({
            "text": c["text"],
            "label": label,
            "source_species": c.get("source_species", ""),
            "target_species": c.get("target_species", ""),
            "interaction_type": CATEGORY_TO_INTERACTION_TYPE.get(cat, "interactsWith"),
            "source": "sibils_mongodb",
            "llm_validated": 1 if label == 1 else 0,
            "flan_t5_score": c.get("flan_t5_score", -1),
            "candidate_category": cat,
        })
    return rows


# ── main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine SIBiLS MongoDB for gap-category positives and hard negatives"
    )
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--model", default=None,
                        help="Path to FLAN-T5 model for validation (skip if --no-validate)")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip FLAN-T5 validation (faster, less precise)")
    parser.add_argument("--hard-negatives-only", action="store_true",
                        help="Only harvest hard negatives, no positives")
    parser.add_argument("--positives-only", action="store_true",
                        help="Only harvest positives, no hard negatives")
    parser.add_argument("--categories", nargs="+",
                        default=list(CATEGORY_PATTERNS.keys()),
                        choices=list(CATEGORY_PATTERNS.keys()),
                        help="Gap categories to target (default: all 6)")
    parser.add_argument("--target-per-category", type=int, default=200,
                        help="Target positive candidates per category before FLAN-T5 filter (default 200)")
    parser.add_argument("--target-negatives", type=int, default=500,
                        help="Target hard negative candidates before FLAN-T5 filter (default 500)")
    parser.add_argument("--pos-threshold", type=float, default=0.65,
                        help="FLAN-T5 score threshold for accepting positives (default 0.65)")
    parser.add_argument("--max-scan", type=int, default=500_000,
                        help="Max MongoDB documents to scan for positives (default 500000)")
    args = parser.parse_args()

    if not args.no_validate and args.model is None:
        print("ERROR: --model required unless --no-validate is set", file=sys.stderr)
        sys.exit(1)

    print(f"Using device: {DEVICE}")

    # Connect to MongoDB
    collection = get_collection()
    print(f"Connected to MongoDB: {DB_NAME}.{COLLECTION_NAME}")

    # Load FLAN-T5 if needed
    flan_model = flan_tok = flan_yes = flan_no = None
    if not args.no_validate:
        flan_model, flan_tok, flan_yes, flan_no = load_flan_t5(Path(args.model))

    all_rows = []

    # ── Positives ──
    if not args.hard_negatives_only:
        pos_candidates = harvest_positives(
            collection,
            categories=args.categories,
            target_per_category=args.target_per_category,
            max_scan=args.max_scan,
        )

        if not args.no_validate and flan_model is not None:
            print("\nFiltering positive candidates with FLAN-T5 ...")
            pos_candidates = filter_with_flan_t5(
                pos_candidates, flan_model, flan_tok, flan_yes, flan_no,
                pos_threshold=args.pos_threshold,
                is_negative=False,
            )

        pos_rows = to_training_rows(pos_candidates, label=1)
        all_rows.extend(pos_rows)
        print(f"\nPositive rows: {len(pos_rows)}")

    # ── Hard Negatives ──
    if not args.positives_only:
        neg_candidates = harvest_hard_negatives(
            collection,
            target=args.target_negatives,
            max_scan=200_000,
        )

        if not args.no_validate and flan_model is not None:
            print("\nFiltering hard negative candidates with FLAN-T5 ...")
            neg_candidates = filter_with_flan_t5(
                neg_candidates, flan_model, flan_tok, flan_yes, flan_no,
                is_negative=True,  # keep confused-model cases
            )

        neg_rows = to_training_rows(neg_candidates, label=0)
        all_rows.extend(neg_rows)
        print(f"Hard negative rows: {len(neg_rows)}")

    # ── Save ──
    if not all_rows:
        print("\nWARNING: no rows to save!", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out), index=False)

    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    print(f"\n{'='*60}")
    print(f"Saved {len(df)} rows to {out}")
    print(f"  Positives: {n_pos}")
    print(f"  Negatives: {n_neg}")
    if n_pos > 0:
        print(f"  neg:pos ratio: {n_neg/n_pos:.2f}")
    print(f"\nCategory breakdown:")
    for cat, grp in df.groupby("candidate_category"):
        n1 = int((grp["label"] == 1).sum())
        n0 = int((grp["label"] == 0).sum())
        print(f"  {cat:<22} pos={n1}  neg={n0}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
