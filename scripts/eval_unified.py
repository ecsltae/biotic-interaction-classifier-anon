#!/usr/bin/env python3
"""Evaluate any cross-encoder on the unified 449-item test set.

Label semantics throughout: SPECIES-LEVEL ("do these two taxa interact?"), which is
what the deployed filter is actually for. biotx100 uses triples_ok_species,
reject50 uses gold_species_pair, test299 uses its pair-level label.

V1's recorded decisions exist on 150 of the 449 rows; the V1 comparison is
restricted to those and reported as such.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import xenc_format
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from scipy.stats import chi2, norm as snorm
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eval.core import clean_benchmark, to_clean  # noqa: E402  (the one 437-row loader)

def model_format(md):
    """Input format a checkpoint was trained with; pre-format checkpoints are triple-query."""
    cfg = Path(md)/"student_config.json"
    if cfg.exists():
        try: return json.loads(cfg.read_text()).get("input_format", "triple")
        except Exception: pass
    return "triple"

def model_max_len(md):
    """Token budget a checkpoint was trained with (256 for every checkpoint before the
    document-level BioRED arm); scoring at a shorter budget would truncate what it was trained on."""
    cfg = Path(md)/"student_config.json"
    if cfg.exists():
        try: return int(json.loads(cfg.read_text()).get("max_len", 256))
        except Exception: pass
    return 256

def score(md, d, dev, bs=64):
    fmt = model_format(md)
    ml = model_max_len(md)
    q, p = xenc_format.build_many(fmt, d.species1, d.relation, d.species2, d.sentence.astype(str))
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True)
    m = AutoModelForSequenceClassification.from_pretrained(md, local_files_only=True).to(dev).eval()
    P = []
    with torch.no_grad():
        for i in range(0, len(q), bs):
            if p is None:
                e = tok(q[i:i+bs], truncation=True, max_length=ml,
                        padding=True, return_tensors="pt").to(dev)
            else:
                e = tok(q[i:i+bs], p[i:i+bs], truncation="only_second", max_length=ml,
                        padding=True, return_tensors="pt").to(dev)
            P.extend(torch.softmax(m(**e).logits.float(), -1)[:, 1].cpu().numpy())
    del m; torch.cuda.empty_cache()
    return np.array(P)

def _degenerate(pred):
    """All-positive / all-negative predictions make McNemar meaningless; name them."""
    a = np.asarray(pred)
    if (a == 1).all(): return "all-positive"
    if (a == 0).all(): return "all-negative"
    return None


def mcnemar(a_pred, b_pred, y):
    a = (a_pred == y); b = (b_pred == y)
    n01 = int((~a & b).sum()); n10 = int((a & ~b).sum())
    if n01 + n10 == 0: return 1.0, n01, n10
    return float(chi2.sf((abs(n01-n10)-1)**2/(n01+n10), 1)), n01, n10

def wilson(k, n):
    if n == 0: return (0, 0)
    z = snorm.ppf(0.975); p = k/n; d = 1+z*z/n
    c = (p+z*z/(2*n))/d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (c-h, c+h)

def evaluate(model_dirs, name, exclude_leaked=True, thresholds=None, df=None, scores=None):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    if df is None:   # the shared 437-row clean benchmark (in_train and near-duplicates dropped)
        d = clean_benchmark() if exclude_leaked else pd.read_csv(REPO/"data/evaluation/unified_test_set.csv")
    else:
        d = df.copy()
        if exclude_leaked: d = d[~d.in_train].reset_index(drop=True)
    y = d.label.to_numpy()
    if scores is not None:
        S = to_clean(scores) if df is None and exclude_leaked else np.asarray(scores)
    else:
        S = np.mean([score(md, d, dev) for md in model_dirs], axis=0)
    out = {"model": name, "n": int(len(d)), "prevalence": float(y.mean()),
           "input_format": sorted({model_format(md) for md in model_dirs}),
           "auprc": float(average_precision_score(y, S)), "by_threshold": []}
    grid = thresholds if thresholds is not None else np.arange(0.01, 1.0, 0.01)
    best = (-1, None)
    for t in grid:
        pred = (S >= t).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best[0]: best = (f1, float(t))
    out["best_f1"], out["best_thr"] = best
    # per source
    out["per_source"] = {}
    for s, g in d.groupby("source"):
        m = (d.source == s).to_numpy()
        out["per_source"][s] = {"n": int(m.sum()), "pos": int(y[m].sum()),
                                "auprc": float(average_precision_score(y[m], S[m])) if len(set(y[m]))>1 else None}
    # V1 comparison on the rows where its decision is recorded
    hv = d.v1.notna().to_numpy()
    if hv.sum():
        v1 = d.v1.fillna(0).to_numpy().astype(int)
        yv = y[hv]; v1v = v1[hv]
        pr, rc = precision_score(yv, v1v, zero_division=0), recall_score(yv, v1v, zero_division=0)
        out["v1_subset"] = {"n": int(hv.sum()), "precision": float(pr), "recall": float(rc),
                            "f1": float(f1_score(yv, v1v, zero_division=0))}
        cmp = []
        for t in (np.arange(0.01,1.0,0.01) if thresholds is None else grid):
            pred = (S[hv] >= t).astype(int)
            pv, n01, n10 = mcnemar(v1v, pred, yv)
            cmp.append({"thr": float(t), "degenerate": _degenerate(pred),
                        "f1": float(f1_score(yv, pred, zero_division=0)),
                        "precision": float(precision_score(yv, pred, zero_division=0)),
                        "recall": float(recall_score(yv, pred, zero_division=0)),
                        "mcnemar_p": pv, "v1_only_right": n10, "model_only_right": n01})
        nd = [c for c in cmp if c["degenerate"] is None]
        out["vs_v1"] = sorted(nd or cmp, key=lambda c: -c["f1"])[:5]
        out["n_degenerate_thresholds"] = sum(1 for c in cmp if c["degenerate"])
    return out, S, d

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    o, S, d = evaluate([REPO/m for m in a.models], a.name)
    print(json.dumps({k: v for k, v in o.items() if k != "by_threshold"}, indent=2, default=float))
    if a.out:
        np.save(REPO/a.out, S)
