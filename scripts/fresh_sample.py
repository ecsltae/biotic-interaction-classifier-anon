#!/usr/bin/env python3
"""The 'fresh candidates' check of Paper A's Limitations, as a script.

The rows of data/evaluation/eval_sets_qwen_validated.csv whose passage is not in the benchmark
(normalised text) and that carry both taxa and an interaction term: 100 retrieved candidates drawn
after the benchmark. Each arm is scored as the mean of its three checkpoints' order-free
probabilities, at the mean of the checkpoints' development thresholds.

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
from sklearn.metrics import f1_score, precision_score, recall_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from paperA_tables import ARMS_BIODIV, order_free  # noqa: E402


def main() -> None:
    norm = lambda s: re.sub(r"\W+", " ", str(s)).lower().strip()  # noqa: E731
    bench = set(norm(s) for s in pd.read_csv(REPO / "data/evaluation/unified_test_set.csv").sentence)
    q = pd.read_csv(REPO / "data/evaluation/eval_sets_qwen_validated.csv")
    dv = q[[norm(s) not in bench for s in q.text]]
    dv = dv[dv.species1.notna() & dv.species2.notna() & dv.interaction_term.notna()].reset_index(drop=True)
    dv = dv.rename(columns={"text": "sentence", "interaction_term": "relation", "gold_label": "label"})
    y = dv.label.to_numpy().astype(int)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {"n": int(len(y)), "positives": int(y.sum())}
    for arm, pat in ARMS_BIODIV.items():
        mds = [REPO / pat.format(s) for s in (1, 2, 3)]
        sc = np.mean([order_free(md, dv, dev) for md in mds], axis=0)
        thr = float(np.mean([json.loads((md / "student_config.json").read_text())["threshold_dev"] for md in mds]))
        p = (sc >= thr).astype(int)
        out[arm] = {"auprc": float(ap(y, sc)), "threshold": thr, "P": float(precision_score(y, p, zero_division=0)),
                    "R": float(recall_score(y, p)), "F1": float(f1_score(y, p))}
        print(f"{arm:8} AUPRC {out[arm]['auprc']:.3f}")
    (REPO / "results/paperA_v2/fresh_sample.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
