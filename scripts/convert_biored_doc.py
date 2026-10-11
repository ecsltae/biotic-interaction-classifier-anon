#!/usr/bin/env python3
"""BioRED at document level: the same pair-conditioned verifier, given the whole abstract.

The sentence-level benchmark (convert_biored_verify.py) mirrors the biodiversity pipeline, which
only ever sees a passage. BioRED's relations are document-level, though, and published BioRED
systems read the whole abstract, so this builds the input those systems see: one row per pair of
distinct annotated concepts in an abstract (relation-bearing entity-type pairs only, as in the
sentence-level benchmark), the title + abstract as the passage with every mention of the first
concept wrapped in @ ... @ and every mention of the second in # ... #, and the label "BioRED
annotates a relation between them". Concepts are ordered by first mention, so one pass scores a
pair.

Splits follow the BioREDirect protocol (see convert_biored_verify.py --protocol bc8).

Output columns match experiments/multitask/train_student.py (train with --input-format pair; the
markers are already in `text`):
  text, source_species, interaction_type, target_species, label, pmid, cid1, cid2, type1, type2,
  n_concepts, co_mentioned (the pair shares a sentence somewhere in the abstract)

Usage
  python3 scripts/convert_biored_doc.py --src data/raw/bioredirect --out data/benchmarks/biored_bc8_doc
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_biored_verify import concept_key, read_pubtator, related, relation_type_pairs  # noqa: E402

M1O, M1C, M2O, M2C = "@", "@", "#", "#"


def mark_all(text: str, spans_a, spans_b) -> str:
    """Wrap every span of the first concept in @ @ and of the second in # #; overlaps are skipped."""
    spans = sorted([(s, e, M1O, M1C) for s, e in spans_a] + [(s, e, M2O, M2C) for s, e in spans_b])
    out, prev = [], 0
    for st, en, o, c in spans:
        if st < prev:
            continue
        out.append(text[prev:st]); out.append(f"{o} {text[st:en]} {c}"); prev = en
    out.append(text[prev:])
    return "".join(out)


def rows_for(doc, type_pairs, nlp, typed=False) -> list[dict]:
    by_key = {}
    for e in sorted(doc["ents"], key=lambda e: e["start"]):
        by_key.setdefault(concept_key(e), []).append(e)
    sents = [(s.start_char, s.end_char) for s in nlp(doc["text"]).sents]
    sent_of = lambda e: next((i for i, (a, b) in enumerate(sents) if a <= e["start"] < b), -1)  # noqa: E731
    sent_sets = {k: {sent_of(e) for e in v} for k, v in by_key.items()}
    keys = list(by_key)                                   # already in first-mention order
    out = []
    for ka, kb in itertools.combinations(keys, 2):
        a, b = by_key[ka][0], by_key[kb][0]
        if frozenset((a["type"], b["type"])) not in type_pairs:
            continue
        if a["mention"].lower() == b["mention"].lower():
            continue
        out.append({"pmid": doc["pmid"],
                    "text": mark_all(doc["text"], [(e["start"], e["end"]) for e in by_key[ka]],
                                     [(e["start"], e["end"]) for e in by_key[kb]]),
                    # --typed puts each concept's BioRED type in the query ("ChemicalEntity: aspirin")
                    "source_species": f"{a['type']}: {a['mention']}" if typed else a["mention"],
                    "interaction_type": "",
                    "target_species": f"{b['type']}: {b['mention']}" if typed else b["mention"],
                    "label": int(related(doc, a, b)), "cid1": ka, "cid2": kb,
                    "type1": a["type"], "type2": b["type"], "n_concepts": len(keys),
                    "co_mentioned": int(bool(sent_sets[ka] & sent_sets[kb]))})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/raw/bioredirect")
    ap.add_argument("--out", default="data/benchmarks/biored_bc8_doc")
    ap.add_argument("--typed", action="store_true", help="prefix each query entity with its BioRED type")
    a = ap.parse_args()
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {"train": "train_dev", "dev": "test", "test": "bc8_test"}
    splits = {s: read_pubtator(src / f"bioredirect_{f}.pubtator") for s, f in files.items()}
    type_pairs = relation_type_pairs(splits["train"])
    for s, docs in splits.items():
        df = pd.DataFrame([r for d in docs for r in rows_for(d, type_pairs, nlp, a.typed)])
        df.to_csv(out / f"{s}.csv", index=False)
        print(f"{s:5}: {len(docs)} abstracts -> {len(df):6d} pairs, positive rate {df.label.mean():.3f}, "
              f"co-mentioned {df.co_mentioned.mean():.0%}, positives co-mentioned "
              f"{df[df.label == 1].co_mentioned.mean():.0%}")


if __name__ == "__main__":
    main()
