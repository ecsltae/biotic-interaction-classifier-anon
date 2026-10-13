#!/usr/bin/env python3
"""Escalation: the recommended verifier decides, and only its least confident candidates go to an LLM.

The recommended configuration (inference-wrapper `predict()` scores of joint_a05_s1 at its fixed threshold 0.5,
with the candidate rules applied first; results/shipping_2026-10-02/bench_a05.csv) keeps its own
decision except on the band of candidates whose score lies nearest 0.5; those take the zero-shot
LLM's greedy answer to the pair question (results/paperA_v2/llm/biodiv_<model>_pair.csv). A rule
rejection is final. The band is reported as a dial -- several fractions, none chosen on these rows.

Besides the bands, the output records what the paper says about the model alone: its metrics
without the rules, the McNemar test of the rules against no rules, what the co_listed rule would
add on the accept set, and, for every band, the first threshold tau (0.50 to 0.99 in steps of 0.01)
at which the model plus rules alone reaches the band's precision, with its recall. Labels come from
src/eval/core.py's clean_benchmark() for the label version given by --labels.

Usage
  python3 scripts/cascade_tables.py
  python3 scripts/cascade_tables.py --deploy
  python3 scripts/cascade_tables.py --labels pre_review_2026-10-06 --out-dir <dir> --bench-scores <bench_a05.csv>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "handoff/biotic_verifier"))
from eval.core import LABEL_VERSIONS, benchmark_provenance, clean_benchmark  # noqa: E402
import candidate_rules as CR  # noqa: E402

BANDS = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)
TAU_GRID = np.round(np.arange(0.50, 1.00, 0.01), 2)   # the model's own threshold is 0.5
LLMS = ("qwen3-32b", "qwen3.5-122b")              # the paper's appendix
# --deploy: an alternative pair of escalation models (not in the paper)
DEPLOY_LLMS = ("qwen3.8-27b-v035", "qwen3-32b-v035")


def prf(y, p):
    tp, fp, fn = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum()), int(((p == 0) & (y == 1)).sum())
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn)
    return {"P": P, "R": R, "F1": 2 * P * R / (P + R) if P + R else 0.0, "FP": fp, "FN": fn}


def mcnemar(a, b, y):
    ca, cb = a == y, b == y
    k, m = int((~ca & cb).sum()), int((ca & ~cb).sum())
    return (1.0 if k + m == 0 else float(chi2.sf((abs(k - m) - 1) ** 2 / (k + m), 1))), k, m


def first_tau(y, s, rej, target_p):
    """First tau on TAU_GRID at which the model plus rules alone reaches precision target_p."""
    for t in TAU_GRID:
        m = prf(y, (s >= t).astype(int) * ~rej)
        if m["P"] >= target_p:
            return {"tau": float(t), **m}
    return None


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", action="store_true", help="an alternative pair of escalation models -> cascade_deploy.json")
    ap.add_argument("--labels", choices=sorted(LABEL_VERSIONS), default="current",
                    help="label version of the benchmark (src/eval/core.py LABEL_VERSIONS)")
    ap.add_argument("--out-dir", default=str(REPO / "results/paperA_v2"))
    ap.add_argument("--bench-scores", default=str(REPO / "results/shipping_2026-10-02/bench_a05.csv"),
                    help="the recommended model's per-row scores (scripts/eval_shipping.py); only p_interact is read")
    a = ap.parse_args()
    llms, out_name = (DEPLOY_LLMS, "cascade_deploy.json") if a.deploy else (LLMS, "cascade.json")
    d = clean_benchmark(a.labels)
    prov = benchmark_provenance(a.labels)
    y = d.label.to_numpy()
    st = pd.read_csv(a.bench_scores)
    assert (st.species1.values == d.species1.values).all(), "recommended scores misaligned"
    s = st.p_interact.to_numpy()
    rej = np.array([CR.reject_reason(str(p), a, r, b) is not None
                    for p, a, r, b in zip(d.sentence, d.species1, d.relation, d.species2)])
    raw = (s >= 0.5).astype(int)
    base = raw * ~rej
    order = np.argsort(np.abs(s - 0.5), kind="stable")          # least confident first
    out = {"n": int(len(y)), "positives": int(y.sum()), "student": prf(y, base), "llm": {},
           "labels": prov["labels"], "benchmark_sha256": prov["sha256"]}
    # the model alone: without the rules, the rules against no rules, and the eighth rule
    p, fx, br = mcnemar(raw, base, y)
    out["student_no_rules"] = prf(y, raw)
    out["rules_vs_no_rules"] = {"mcnemar_p": p, "fixed": fx, "broken": br}
    colist = np.array([CR.co_listed(str(p), a, b) for p, a, b in zip(d.sentence, d.species1, d.species2)])
    hit = (base == 1) & colist
    p, fx, br = mcnemar(base, base * ~colist, y)
    out["co_listed_on_accept_set"] = {"fires": int(hit.sum()), "removes_fp": int((hit & (y == 0)).sum()),
                                      "removes_tp": int((hit & (y == 1)).sum()),
                                      "mcnemar_p": p, "fixed": fx, "broken": br}
    for tag in llms:
        L = pd.read_csv(REPO / f"results/paperA_v2/llm/biodiv_{tag}_pair.csv")
        assert (L.species1.values == d.species1.values).all(), f"{tag} answers misaligned"
        v = L.verdict.to_numpy()
        rows = []
        for f in BANDS:
            k = int(round(f * len(y)))
            band = np.zeros(len(y), bool); band[order[:k]] = True
            pred = np.where(band, v, (s >= 0.5).astype(int)) * ~rej
            p, fx, br = mcnemar(base, pred, y)
            m = prf(y, pred)
            rows.append({"band": f, "llm_calls": k, **m, "mcnemar_vs_student": p, "fixed": fx, "broken": br,
                         "model_alone_reaches_P": first_tau(y, s, rej, m["P"])})
        out["llm"][tag] = rows
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(a.out_dir) / out_name).write_text(json.dumps(out, indent=2))
    print(f"student alone: {out['student']}")
    print(f"without rules: {out['student_no_rules']}; rules vs none: {out['rules_vs_no_rules']}; "
          f"co_listed on the accept set: {out['co_listed_on_accept_set']}")
    for tag, rows in out["llm"].items():
        print(f"\n{tag}:  band  calls     P      R     F1   FP   p(vs student)")
        for r in rows:
            t = r["model_alone_reaches_P"]
            print(f"       {r['band']:4.0%} {r['llm_calls']:5d}  {r['P']:.3f}  {r['R']:.3f}  {r['F1']:.3f}  {r['FP']:3d}  "
                  f"{r['mcnemar_vs_student']:.3g} ({r['fixed']}/{r['broken']})  "
                  + (f"model alone: tau {t['tau']:.2f}, R {t['R']:.3f}" if t else "model alone: never"))


if __name__ == "__main__":
    main()
