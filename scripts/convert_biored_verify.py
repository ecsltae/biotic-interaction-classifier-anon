#!/usr/bin/env python3
"""BioRED as a candidate-verification benchmark, built the way the biodiversity pipeline builds
its candidates.

The biodiversity pipeline proposes a candidate whenever a passage co-mentions two taxa. Here the
same construction is applied to BioRED (Luo et al., 2022): every pair of distinct annotated
concepts co-mentioned in one sentence of an abstract is a candidate, the sentence is its passage,
and the candidate is positive when BioRED annotates a relation (of any type) between the two
concepts in that abstract. BioRED's relations are document-level, so a positive pair co-mentioned
in a sentence that does not itself state the relation is labelled positive here: the labels carry
distant-supervision noise, identically for every model compared on them.

Only entity-type pairs that carry at least one relation in the training split become candidates,
so the benchmark does not fill up with type combinations BioRED never annotates.

Output columns match experiments/multitask/train_student.py:
  text, source_species, interaction_type, target_species, label
plus pmid, type1, type2, n_concepts (distinct concepts in the sentence) and sent_id.

Usage
  python3 scripts/convert_biored_verify.py \
      --src data/raw/bioredirect --out data/benchmarks/biored_verify
"""
from __future__ import annotations

import argparse
import itertools
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


def read_pubtator(path: Path) -> list[dict]:
    docs = []
    for block in path.read_text(encoding="utf-8").strip().split("\n\n"):
        lines = [l for l in block.split("\n") if l.strip()]
        if not lines:
            continue
        doc = {"title": "", "abstract": "", "ents": [], "rels": set()}
        for l in lines:
            if "|t|" in l[:20]:
                doc["pmid"], doc["title"] = l.split("|t|", 1)
            elif "|a|" in l[:20]:
                doc["abstract"] = l.split("|a|", 1)[1]
            else:
                f = l.split("\t")
                if len(f) >= 6 and f[1].isdigit() and f[2].isdigit():
                    ids = {x.strip() for x in re.split(r"[;,]", f[5]) if x.strip() and x.strip() != "-"}
                    doc["ents"].append({"start": int(f[1]), "end": int(f[2]), "mention": f[3],
                                        "type": f[4], "ids": ids})
                elif len(f) >= 5:
                    doc["rels"].add(frozenset((f[2], f[3])))
        doc["text"] = doc["title"] + " " + doc["abstract"]
        docs.append(doc)
    return docs


def concept_key(ent) -> str:
    """One stable key per entity: the sorted set of its concept IDs."""
    return "|".join(sorted(ent["ids"])) if ent["ids"] else f"_{ent['mention'].lower()}"


def related(doc, a, b) -> bool:
    return any(frozenset((x, y)) in doc["rels"] for x in a["ids"] for y in b["ids"])


def relation_type_pairs(docs) -> set:
    """Entity-type pairs that carry at least one relation (unordered)."""
    out = set()
    for d in docs:
        by_id = defaultdict(set)
        for e in d["ents"]:
            for i in e["ids"]:
                by_id[i].add(e["type"])
        for r in d["rels"]:
            r = tuple(r) if len(r) == 2 else (next(iter(r)),) * 2
            for t1 in by_id.get(r[0], ()):
                for t2 in by_id.get(r[1], ()):
                    out.add(frozenset((t1, t2)))
    return out


def candidates(docs, type_pairs, nlp) -> pd.DataFrame:
    rows = []
    for d in docs:
        sdoc = nlp(d["text"])
        for si, sent in enumerate(sdoc.sents):
            s0, s1 = sent.start_char, sent.end_char
            inside = [e for e in d["ents"] if e["start"] >= s0 and e["end"] <= s1]
            first = {}
            for e in sorted(inside, key=lambda e: e["start"]):
                first.setdefault(concept_key(e), e)            # first mention of each concept
            concepts = list(first.values())
            n = len(concepts)
            for a, b in itertools.combinations(concepts, 2):
                if frozenset((a["type"], b["type"])) not in type_pairs:
                    continue
                if a["mention"].lower() == b["mention"].lower():
                    continue                                    # the query could not tell them apart
                rows.append({"pmid": d["pmid"], "sent_id": f"{d['pmid']}_{si}",
                             "text": sent.text.strip(), "source_species": a["mention"],
                             "interaction_type": "", "target_species": b["mention"],
                             "type1": a["type"], "type2": b["type"],
                             "label": int(related(d, a, b)), "n_concepts": n})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/raw/bioredirect")
    ap.add_argument("--out", default="data/benchmarks/biored_verify")
    a = ap.parse_args()
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = {s: read_pubtator(src / f"bioredirect_{s}.pubtator") for s in ("train", "dev", "test")}
    type_pairs = relation_type_pairs(splits["train"])
    print(f"relation-bearing type pairs (from train): {len(type_pairs)}")
    for s, docs in splits.items():
        df = candidates(docs, type_pairs, nlp)
        df.to_csv(out / f"{s}.csv", index=False)
        multi = (df.n_concepts >= 3).mean()
        print(f"{s:5}: {len(docs)} abstracts -> {len(df):6d} candidates, positive rate {df.label.mean():.3f}, "
              f"{multi:.0%} from sentences naming >= 3 concepts")


if __name__ == "__main__":
    main()
