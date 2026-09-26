#!/usr/bin/env python3
"""Regenerate Paper A's core tables from current artifacts, under the CORRECT label column.

Why this exists
  The draft's empirical core is computed against `triples_ok_full` (is the stated relation
  right, in the stated direction?). The project's label semantics -- and the benchmark's --
  are SPECIES-LEVEL: `triples_ok_species`, do these two taxa interact. Under the correct
  column the draft's headline McNemar goes from p=0.018 to p=0.584 and its three-mechanism
  error taxonomy collapses to two, because a wrong relation term is not a species-level error.

  Emitting the tables from a script means they cannot silently drift from the artifacts again.

  python scripts/paper_tables.py            # human-readable
  python scripts/paper_tables.py --latex    # LaTeX bodies
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_unified import mcnemar                                       # noqa: E402
from sklearn.metrics import (precision_score, recall_score, f1_score,  # noqa: E402
                             average_precision_score)

REPO = Path(__file__).resolve().parents[1]
LABEL = "triples_ok_species"        # NOT triples_ok_full


def audit_sheet():
    return pd.read_csv(REPO / "results/audit_v1_v3/biotx100_audit.csv")


def clean_benchmark():
    d = pd.read_csv(REPO / "data/evaluation/unified_test_set.csv")
    d["maxj"] = pd.read_csv(REPO / "results/test_contamination_scan.csv").maxj.values
    leak = (d.maxj >= 0.5) | d.in_train
    return d[~leak].reset_index(drop=True), (~leak).to_numpy()


def t1_error_decomposition():
    """V1's errors and all unsupported candidates, by mechanism, under both columns."""
    b = audit_sheet(); rows = []
    for col in ("triples_ok_full", LABEL):
        y = b[col].astype(int); v1 = b.classifier.astype(int)
        acc_wrong = b.mech[(v1 == 1) & (y == 0)].value_counts().to_dict()
        unsup = b.mech[y == 0].value_counts().to_dict()
        pb = unsup.get("pair-binding", 0); pos = int((y == 1).sum())
        rows.append(dict(label_column=col, positives=pos, base_rate=round(float(y.mean()), 4),
                         v1_accepts=int(v1.sum()),
                         v1_precision=round(float((v1 & y).sum() / v1.sum()), 4),
                         v1_false_positives=int(((v1 == 1) & (y == 0)).sum()),
                         v1_fp_by_mech=acc_wrong, all_unsupported=int((y == 0).sum()),
                         unsupported_by_mech=unsup,
                         component_bound_at_full_recall=round(pos / (pos + pb), 4)))
    return pd.DataFrame(rows)


def t2_operating_points():
    """The shipped model across thresholds on the clean benchmark, vs V1's real decisions."""
    d, sel = clean_benchmark()
    y = d.label.to_numpy()
    V = np.load(REPO / "results/v1_decisions_449.npy")[sel]
    S = np.load(REPO / "results/dirhead/joint_a05_s1_binary_scores.npy")[
        sel[~pd.read_csv(REPO / "data/evaluation/unified_test_set.csv").in_train.to_numpy()]]
    rows = [dict(system="V1 (deployed)", threshold=None,
                 precision=round(float(precision_score(y, V)), 4),
                 recall=round(float(recall_score(y, V)), 4),
                 f1=round(float(f1_score(y, V)), 4), fp=int(((V == 1) & (y == 0)).sum()),
                 fn=int(((V == 0) & (y == 1)).sum()), mcnemar_p=None)]
    for t in (0.50, 0.70, 0.90, 0.95, 0.99):
        pr = (S >= t).astype(int); p, _, _ = mcnemar(V, pr, y)
        rows.append(dict(system="joint_a05_s1", threshold=t,
                         precision=round(float(precision_score(y, pr)), 4),
                         recall=round(float(recall_score(y, pr)), 4),
                         f1=round(float(f1_score(y, pr)), 4),
                         fp=int(((pr == 1) & (y == 0)).sum()),
                         fn=int(((pr == 0) & (y == 1)).sum()), mcnemar_p=float(p)))
    return pd.DataFrame(rows)


def t3_blocks():
    """Per-block decomposition -- the table a reviewer will ask for."""
    d, sel = clean_benchmark()
    y = d.label.to_numpy()
    V = np.load(REPO / "results/v1_decisions_449.npy")[sel]
    S = np.load(REPO / "results/dirhead/joint_a05_s1_binary_scores.npy")[
        sel[~pd.read_csv(REPO / "data/evaluation/unified_test_set.csv").in_train.to_numpy()]]
    pr = (S >= 0.5).astype(int); rows = []
    for src in list(d.source.unique()) + ["ALL"]:
        m = (d.source == src).to_numpy() if src != "ALL" else np.ones(len(y), bool)
        p, k, mm = mcnemar(V[m], pr[m], y[m])
        rows.append(dict(block=src, n=int(m.sum()), prevalence=round(float(y[m].mean()), 3),
                         v1_f1=round(float(f1_score(y[m], V[m], zero_division=0)), 4),
                         model_f1=round(float(f1_score(y[m], pr[m], zero_division=0)), 4),
                         k_fixes=k, m_breaks=mm, mcnemar_p=float(p)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    tabs = [("Table 1 -- error decomposition, both label columns", t1_error_decomposition()),
            ("Table 2 -- operating points vs V1's recorded decisions", t2_operating_points()),
            ("Table 3 -- per-block decomposition", t3_blocks())]
    for name, t in tabs:
        print("\n" + "=" * 96); print(name); print("=" * 96)
        print(t.to_latex(index=False, float_format="%.4f") if a.latex else t.to_string(index=False))
    print(f"\nlabel column used throughout: {LABEL}")


if __name__ == "__main__":
    main()
