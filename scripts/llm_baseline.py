#!/usr/bin/env python3
"""Zero-shot local LLM baselines for Paper A, in the same three formulations as the trained arms.

  sentence  the passage alone: "does it assert an interaction?"           (no pair named)
  pair      the passage and the two arguments: "do THESE two interact?"   (no relation)
  triple    the teacher's own prompt (pair + retrieved relation), scored in both argument orders
            and maxed, like the trained triple arm; biodiversity only, BioRED has no relation string
  nopassage the pair prompt with every reference to the passage removed: "are these two strings
            entities of the right kind, and are they known to interact?" (the pair-only control;
            run only when asked for with --forms, never by default)

The score is P(YES) from the first answer token's log-probabilities (temperature 0, no thinking),
so AUPRC is defined; the verdict is the greedy answer. Models run locally through Ollama; nothing
leaves the machine. Resumable: rows already in the output are skipped, and an interrupted run
continues from its checkpoint (<output>.part, written every --save-every rows).

Usage
  python3 scripts/llm_baseline.py --bench biodiv --model qwen3:32b
  python3 scripts/llm_baseline.py --bench biored --model qwen3:32b --n 3000
  python3 scripts/llm_baseline.py --bench biodiv --model qwen3.5:122b --num-gpu 44 --num-ctx 4096
  python3 scripts/llm_baseline.py --bench biodiv --model qwen3:8b --forms nopassage --workers 1 --num-thread 8

qwen3.5:122b (81 GB) does not fit on the 80 GB card; with Ollama's own layer split it aborts in
cuBLAS while loading, so --num-gpu caps the GPU layers (44 of 49 load; the rest run on the CPU).
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import os
URL = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/generate"   # OLLAMA_URL picks another server
OUT = REPO / "results/paperA_v2/llm"

BIODIV = {
    "sentence": """You check sentences retrieved from the scientific literature for a biotic-interaction database.

Sentence: {sent}

Answer YES only if the sentence asserts a direct biological interaction between organisms (for example predation, parasitism, infection, herbivory, pollination, mutualism, symbiosis or disease transmission), and the interaction is not negated.

Reply with exactly one word: YES or NO.""",
    "pair": """You verify candidate biotic-interaction pairs extracted from scientific literature.

Sentence: {sent}

Candidate pair:
  Entity 1: "{s1}"
  Entity 2: "{s2}"

Answer YES only if ALL of the following hold:
 1. Both entities are genuinely the organisms referred to by those surface strings in this sentence (not a gene, protein, chemical, author name, or a different species).
 2. The sentence asserts a direct biological interaction between these two organisms (not merely a co-mention, and not an interaction each has with some third organism), and the interaction is not negated.

Reply with exactly one word: YES or NO.""",
    # the pair prompt minus every reference to the sentence; the interaction examples are the
    # sentence prompt's
    "nopassage": """You verify candidate biotic-interaction pairs extracted from scientific literature.

Candidate pair:
  Entity 1: "{s1}"
  Entity 2: "{s2}"

Answer YES only if ALL of the following hold:
 1. Both surface strings genuinely refer to organisms (not a gene, protein, chemical, author name, or place).
 2. These two organisms are known to interact directly (for example predation, parasitism, infection, herbivory, pollination, mutualism, symbiosis or disease transmission), not merely through an interaction each has with some third organism.

Reply with exactly one word: YES or NO.""",
}

BIORED = {
    "sentence": """You check sentences from biomedical abstracts for a relation database.

Sentence: {sent}

Answer YES only if the sentence states a relation between two of the biomedical entities it mentions (genes or proteins, chemicals or drugs, diseases or phenotypes, sequence variants, organisms, cell lines): for example an association, a positive or negative correlation, binding, a co-treatment, a comparison, a conversion or a drug interaction.

Reply with exactly one word: YES or NO.""",
    "pair": """You verify candidate relations between biomedical entities extracted from abstracts.

Sentence: {sent}

Candidate pair:
  Entity 1: "{s1}"
  Entity 2: "{s2}"

Answer YES only if the sentence states a relation between these two entities (for example an association, a positive or negative correlation, binding, a co-treatment, a comparison, a conversion or a drug interaction), not merely that both are mentioned.

Reply with exactly one word: YES or NO.""",
    # the biodiversity no-passage prompt with BioRED's entity types (the sentence prompt's list)
    # and BioRED's relation examples (the pair prompt's)
    "nopassage": """You verify candidate relations between biomedical entities extracted from abstracts.

Candidate pair:
  Entity 1: "{s1}"
  Entity 2: "{s2}"

Answer YES only if ALL of the following hold:
 1. Both surface strings genuinely refer to biomedical entities (a gene or protein, a chemical or drug, a disease or phenotype, a sequence variant, an organism or a cell line; not an author name or place).
 2. These two entities are known to be related (for example an association, a positive or negative correlation, binding, a co-treatment, a comparison, a conversion or a drug interaction).

Reply with exactly one word: YES or NO.""",
}
OPT_IN = ("nopassage",)        # forms that run only when named in --forms


def teacher_prompt() -> str:
    src = (REPO / "scripts/teacher_label_triples.py").read_text()
    return re.search(r'PROMPT = """(.*?)"""', src, re.S).group(1)


OPTIONS = {"temperature": 0, "num_predict": 1, "seed": 0}


def ask(model: str, prompt: str) -> tuple[float, str]:
    r = requests.post(URL, timeout=600, json={
        "model": model, "prompt": prompt, "stream": False, "think": False, "logprobs": True,
        "top_logprobs": 10, "options": OPTIONS}).json()
    text = r.get("response", "").strip()
    yes = no = 0.0
    for t in (r.get("logprobs") or [{}])[0].get("top_logprobs", []):
        w = t["token"].strip().upper()
        if w == "YES":
            yes += math.exp(t["logprob"])
        elif w == "NO":
            no += math.exp(t["logprob"])
    p = yes / (yes + no) if yes + no > 0 else float(text.upper().startswith("YES"))
    return p, text


def load(bench: str, n: int) -> pd.DataFrame:
    if bench == "biodiv":
        d = pd.read_csv(REPO / "data/evaluation/unified_test_set.csv")
        scan = pd.read_csv(REPO / "results/test_contamination_scan.csv")
        d = d[~(d.in_train.to_numpy() | (scan.maxj.to_numpy() > 0.5))].reset_index(drop=True)
        assert len(d) == 437
        return d[["sentence", "species1", "relation", "species2", "label", "source"]]
    # biored = the BC8 test set; bioreddev = BioRED's own test split, the BC8 protocol's development
    # set -- disjoint from both reporting sets, so it is where a teacher may be chosen
    d = pd.read_csv(REPO / f"data/benchmarks/biored_bc8/{'dev' if bench == 'bioreddev' else 'test'}.csv")
    d = d.rename(columns={"text": "sentence", "source_species": "species1", "target_species": "species2"})
    d = d.sample(n, random_state=0) if n and n < len(d) else d
    return d[["sentence", "species1", "species2", "label", "pmid", "n_concepts"]].assign(row=d.index)


def is_prefix(p: pd.DataFrame, d: pd.DataFrame) -> bool:
    """True when the rows of an earlier (partial) output are the first len(p) rows of d."""
    if not 0 < len(p) < len(d):
        return False
    head = d.iloc[:len(p)]
    if "row" in p and "row" in head:
        return bool((p.row.to_numpy() == head.row.to_numpy()).all())
    cols = ("species1", "species2", "sentence")
    return all(c in p for c in cols) and all(
        (p[c].astype(str).to_numpy() == head[c].astype(str).to_numpy()).all() for c in cols)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", choices=("biodiv", "biored", "bioreddev"), required=True)
    ap.add_argument("--model", default="qwen3:32b")
    ap.add_argument("--forms", nargs="*", default=None,
                    help="prompt forms to run; default every form except the opt-in ones "
                         f"({', '.join(OPT_IN)}), i.e. sentence and pair, plus triple on biodiv")
    ap.add_argument("--n", type=int, default=3000, help="BioRED test candidates to sample (seed 0)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="", help="suffix for output names, e.g. -v035 for runs on another Ollama version")
    ap.add_argument("--num-gpu", type=int, default=None, help="GPU layers (Ollama num_gpu); default: Ollama decides")
    ap.add_argument("--num-ctx", type=int, default=None, help="context window (Ollama num_ctx)")
    ap.add_argument("--num-thread", type=int, default=None,
                    help="CPU threads per request (Ollama num_thread); default: Ollama decides")
    ap.add_argument("--save-every", type=int, default=25,
                    help="checkpoint to <output>.part every N rows (0 = only at the end)")
    ap.add_argument("--out-dir", default=str(OUT), help="where the CSVs go (default results/paperA_v2/llm)")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N rows only (smoke tests); needs an --out-dir other than the default")
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    if a.limit is not None and out_dir.resolve() == OUT.resolve():
        ap.error("--limit writes a partial benchmark; give a scratch --out-dir")
    OPTIONS.update({k: v for k, v in (("num_gpu", a.num_gpu), ("num_ctx", a.num_ctx),
                                      ("num_thread", a.num_thread)) if v is not None})
    out_dir.mkdir(parents=True, exist_ok=True)
    d = load(a.bench, a.n)
    if a.limit is not None:
        d = d.head(a.limit)
    prompts = dict(BIODIV if a.bench == "biodiv" else BIORED)
    if a.bench == "biodiv":
        prompts["triple"] = teacher_prompt()
    forms = a.forms or [f for f in prompts if f not in OPT_IN]
    if set(forms) - set(prompts):
        ap.error(f"unknown form(s) {sorted(set(forms) - set(prompts))} for --bench {a.bench}; one of {sorted(prompts)}")
    for form in forms:
        out = out_dir / f"{a.bench}_{a.model.replace(':', '-')}{a.tag}_{form}.csv"
        part = out.with_name(out.name + ".part")    # checkpoint; the *_<form>.csv globs never match it
        if out.exists() and len(pd.read_csv(out)) == len(d):
            print(f"done already: {out.name}"); continue
        tmpl = prompts[form]

        def job(r):
            if form == "triple":                      # both argument orders, max: as the trained arm
                f = tmpl.format(sent=r.sentence, s1=r.species1, rel=r.relation, s2=r.species2)
                b = tmpl.format(sent=r.sentence, s1=r.species2, rel=r.relation, s2=r.species1)
                (pf, tf), (pb, tb) = ask(a.model, f), ask(a.model, b)
                return max(pf, pb), f"{tf}/{tb}", int(tf.upper().startswith("YES") or tb.upper().startswith("YES"))
            p, t = ask(a.model, tmpl.format(sent=r.sentence, s1=r.species1, s2=r.species2))
            return p, t, int(t.upper().startswith("YES"))

        # fail fast on a model that does not answer YES/NO with thinking off (the qwen3:4b and
        # qwen3:30b 2507 re-releases start with prose, so P(YES) would be 0 on every row)
        probe = [job(r)[1] for r in d.head(8).itertuples(index=False)]
        if not any(t.split("/")[0].strip().upper().startswith(("YES", "NO")) for t in probe):
            raise SystemExit(f"{a.model} answers {probe[:3]} instead of YES/NO; not running {form}")
        # resume: a shorter earlier run (e.g. --n 1000) is a prefix of the seed-0 sample, so its rows
        # are reused; so are the rows an interrupted run checkpointed in <output>.part
        prev = None
        for src in (out, part):
            if src.exists():
                try:
                    p = pd.read_csv(src)
                except Exception as e:                 # a checkpoint cut off mid-write: start over
                    print(f"ignoring unreadable {src.name}: {e}", flush=True); continue
                if is_prefix(p, d) and (prev is None or len(p) > len(prev)):
                    prev = p
        if prev is not None:
            print(f"resuming {out.name}: reusing {len(prev)} rows", flush=True)
        todo = d.iloc[len(prev):] if prev is not None else d
        res = []

        def frame(k):
            o = todo.iloc[:k].assign(p_yes=[x[0] for x in res[:k]], raw=[x[1] for x in res[:k]],
                                     verdict=[x[2] for x in res[:k]])
            return pd.concat([prev[o.columns], o], ignore_index=True) if prev is not None else o

        with ThreadPoolExecutor(a.workers) as ex:
            for x in ex.map(job, todo.itertuples(index=False)):
                res.append(x)
                if a.save_every and len(res) % a.save_every == 0 and len(res) < len(todo):
                    tmp = part.with_name(part.name + ".tmp")
                    frame(len(res)).to_csv(tmp, index=False)
                    os.replace(tmp, part)             # atomic: a kill never leaves half a checkpoint
        o = frame(len(res))
        o.to_csv(out, index=False)
        part.unlink(missing_ok=True)
        from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
        y = o.label.to_numpy()
        print(f"{a.bench} {a.model} {form:8}: AUPRC {average_precision_score(y, o.p_yes):.3f} | greedy "
              f"P {precision_score(y, o.verdict, zero_division=0):.3f} R {recall_score(y, o.verdict):.3f} "
              f"F1 {f1_score(y, o.verdict):.3f}", flush=True)


if __name__ == "__main__":
    main()
