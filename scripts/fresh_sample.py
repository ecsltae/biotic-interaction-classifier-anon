#!/usr/bin/env python3
"""The 'fresh candidates' check of Paper A's Limitations, as a script.

The rows of data/evaluation/eval_sets_qwen_validated.csv whose passage is not in the benchmark
(normalised text) and that carry both taxa and an interaction term: 100 retrieved candidates drawn
after the benchmark. Each arm is scored as the mean of its three checkpoints' order-free
probabilities, at the mean of the checkpoints' development thresholds.

The query names the taxa and the relation as the passage words them, like the benchmark's
candidates: the surface forms (*_form) the retrieval system recorded in the curation file the row
comes from, '|'-joined when it recorded several, as the benchmark keeps them. The CSV holds the
canonical names (*_term) instead; the scores under those are kept under "canonical_names" for
comparison.

Usage
  python3 scripts/fresh_sample.py      # -> results/paperA_v2/fresh_sample.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score as ap
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from paperA_tables import ARMS_BIODIV, order_free  # noqa: E402
sys.path.insert(0, str(REPO / "src"))
from eval.core import clean_benchmark  # noqa: E402


def main() -> None:
    norm = lambda s: re.sub(r"\W+", " ", str(s)).lower().strip()  # noqa: E731
    bench = set(norm(s) for s in pd.read_csv(REPO / "data/evaluation/unified_test_set.csv").sentence)
    q = pd.read_csv(REPO / "data/evaluation/eval_sets_qwen_validated.csv")
    dv = q[[norm(s) not in bench for s in q.text]]
    dv = dv[dv.species1.notna() & dv.species2.notna() & dv.interaction_term.notna()].reset_index(drop=True)
    # the passage's wording of each name, from the curation file the row was drawn from
    src = pd.concat([pd.read_csv(REPO / "data/evaluation" / f, sep="\t") for f in sorted(dv.source_file.unique())])
    form = {(norm(t), a, b): (str(f1).strip(), str(f2).strip(), str(fr).strip()) for t, a, b, f1, f2, fr in
            zip(src.sentence, src.species1_term, src.species2_term, src.species1_form, src.species2_form, src.interaction_form)}
    keys = [(norm(t), a, b) for t, a, b in zip(dv.text, dv.species1, dv.species2)]
    assert all(k in form for k in keys), "a fresh candidate is missing from its curation file"
    canon = dv.rename(columns={"text": "sentence", "interaction_term": "relation", "gold_label": "label"})
    worded = canon.assign(species1=[form[k][0] for k in keys], species2=[form[k][1] for k in keys],
                          relation=[form[k][2] for k in keys])
    y = canon.label.to_numpy().astype(int)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def arms(d, scores=None) -> dict:
        res = {}
        for arm, pat in ARMS_BIODIV.items():
            mds = [REPO / pat.format(s) for s in (1, 2, 3)]
            sc = np.mean([order_free(md, d, dev) for md in mds], axis=0)
            if scores is not None:
                scores[arm] = sc
            thr = float(np.mean([json.loads((md / "student_config.json").read_text())["threshold_dev"] for md in mds]))
            p = (sc >= thr).astype(int)
            res[arm] = {"auprc": float(ap(y, sc)), "auroc": float(roc_auc_score(y, sc)), "threshold": thr,
                        "P": float(precision_score(y, p, zero_division=0)),
                        "R": float(recall_score(y, p)), "F1": float(f1_score(y, p))}
            print(f"{arm:8} AUPRC {res[arm]['auprc']:.3f}  AUROC {res[arm]['auroc']:.3f}")
        return res

    def bootstrap(sa, sb, n=2000, seed=0) -> dict:
        """Paired bootstrap of AUPRC(b) - AUPRC(a) over candidates (2,000 resamples, a single-class
        resample redrawn, percentile 95% interval), the convention of scripts/paperA_pair_only.py."""
        rng, diffs = np.random.default_rng(seed), []
        while len(diffs) < n:
            i = rng.integers(0, len(y), len(y))
            if y[i].min() == y[i].max():
                continue
            diffs.append(ap(y[i], sb[i]) - ap(y[i], sa[i]))
        return {"diff": float(ap(y, sb) - ap(y, sa)), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                "resamples": n, "seed": seed}

    S = {}
    out = {"n": int(len(y)), "positives": int(y.sum()), "names": "passage wording (*_form as recorded)",
           "n_rows_names_differ": int(sum((a, b) != (c, e) for a, b, c, e in
                                          zip(canon.species1, canon.species2, worded.species1, worded.species2))),
           **arms(worded, S), "canonical_names": arms(canon)}
    out["pair_minus_sentence_auprc"] = bootstrap(S["sentence"], S["pair"])
    out["triple_minus_sentence_auprc"] = bootstrap(S["sentence"], S["triple"])
    # the benchmark's AUROC for comparison: three-checkpoint ensembles on the 437 clean rows
    yb = clean_benchmark().label.to_numpy().astype(int)
    out["benchmark_auroc"] = {arm: float(roc_auc_score(yb, np.load(REPO / f"results/paperA_v2/S_biodiv_{arm}.npy")))
                              for arm in ARMS_BIODIV}
    (REPO / "results/paperA_v2/fresh_sample.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
