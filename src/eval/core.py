"""Single source of truth for classifier evaluation.

Design rules, each of which fixes a specific defect found in the 2026-08-19 audit:

1. ``metrics_at`` takes the threshold as a REQUIRED argument. There is no
   function in this module that picks a threshold by maximising a metric on
   the data it then reports. V1 had 51 evaluation implementations and 8
   different threshold grids; every reported number was a best-of-N on the
   reporting set.
2. Thresholds come from ``threshold_from_prior`` (closed form) or from a
   checkpoint's recorded value. Never from the test set.
3. Every benchmark load asserts a SHA-256, so a silently mutated file fails
   loudly instead of shifting a published number.
4. ``report`` always emits the majority-class baseline, AUPRC, TPR/FPR and a
   per-source breakdown. Two of V1's five reported sets were near-worthless
   (EP-passage: trivial all-positive F1 = 0.919 vs champion 0.854) and
   nothing in the pipeline said so.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE = Path(__file__).resolve().parents[2]


# ── provenance ────────────────────────────────────────────────────────────

def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── the clean biodiversity benchmark ─────────────────────────────────────
#
# One loader for every script that scores the 437-row benchmark. unified_test_set.csv has 449
# rows; 9 are flagged `in_train` (exact match with a training passage) and the 5-gram scan of
# scripts/scan_contamination.py finds 3 more (Jaccard 1.000, all biotx100). Scripts that dropped
# only `in_train` scored 440 rows, so their numbers did not line up with the papers' 437.
#
# Gold revisions (each leaves a frozen copy of the file before it, listed in LABEL_VERSIONS):
#   2026-09-25  row 400 (Acanthamoeba / Pseudomonas) 1 -> 0                       -> 246 clean positives
#   2026-10-06  blind re-annotation of 22 items (11 contested, 11 random controls;
#               labels and scores hidden): 7 labels changed, rows 66, 121, 126,
#               177, 189 and 232 0 -> 1, row 416 1 -> 0 (gold_rev 2, gold_note)  -> 251 clean positives

UNIFIED = BASE / "data/evaluation/unified_test_set.csv"
UNIFIED_SHA = "c17cc4bec6ae6c04d0b3eb923167df4f6e3e874ea81833116200a10a0f9dc856"  # after the 2026-10-06 re-annotation
CONTAM = BASE / "results/test_contamination_scan.csv"
JACCARD_MAX = 0.5          # no row sits at exactly 0.5, so > and >= agree
CLEAN_N, CLEAN_POS = 437, 251   # CLEAN_POS: positives under the "current" labels

# Label versions of the benchmark. Every version is a full 449-row copy with the same rows,
# sentences, sources and `in_train` flags, so the clean mask is the same in all of them; only labels
# differ. "current" is what the paper reports; the frozen copies keep earlier numbers reproducible.
LABEL_VERSIONS = {
    "current": {"path": UNIFIED, "sha256": UNIFIED_SHA, "positives": CLEAN_POS},
    "pre_review_2026-10-06": {
        "path": BASE / "data/evaluation/unified_test_set.pre_review_2026-10-06.csv",
        "sha256": "057416074764bdb1c6697d95fa58b68b9755176f6b2d2ad2e78b4ae3656b547b", "positives": 246,
        "note": "after the 2026-09-25 row-400 fix, before the 2026-10-06 blind re-annotation"},
    "pre_goldfix": {
        "path": BASE / "data/evaluation/unified_test_set.pre_goldfix.csv",
        "sha256": "cac7dc60dd8385f217c1bf2a4ec1a6b2ea821acab467ca441b982a715d27ded2", "positives": 247,
        "note": "before the 2026-09-25 revision of row 400"},
}


def label_version(labels: str = "current") -> dict:
    """The file, SHA-256 and expected clean positives of one label version."""
    if labels not in LABEL_VERSIONS:
        raise ValueError(f"unknown label version {labels!r}; one of {sorted(LABEL_VERSIONS)}")
    return LABEL_VERSIONS[labels]


def clean_mask(labels: str = "current") -> np.ndarray:
    """Boolean mask over the 449 benchmark rows: True for the 437 clean ones.

    The mask is computed as it always was (``in_train`` or 5-gram Jaccard > 0.5); ``labels`` picks
    which copy of the file is hash-checked and whose positives are counted.
    """
    v = label_version(labels)
    path = v["path"]
    got = sha256(path)
    if got != v["sha256"]:
        raise ValueError(f"{path} changed (sha {got}); refusing to score a mutated benchmark")
    d = pd.read_csv(path)
    scan = pd.read_csv(CONTAM)
    if len(scan) != len(d) or not (scan.sentence.values == d.sentence.values).all():
        raise ValueError(f"{CONTAM} is not row-aligned with {path}")
    keep = ~(d.in_train.to_numpy().astype(bool) | (scan.maxj.to_numpy() > JACCARD_MAX))
    if keep.sum() != CLEAN_N or int(d.label.to_numpy()[keep].sum()) != v["positives"]:
        raise ValueError(f"clean benchmark ({labels}) is {keep.sum()} rows / {int(d.label.to_numpy()[keep].sum())} "
                         f"positive, expected {CLEAN_N} / {v['positives']}")
    return keep


def clean_benchmark(labels: str = "current") -> pd.DataFrame:
    """The 437 clean rows of the benchmark in file order, labelled by ``labels``
    (current labels: 251 positive)."""
    return pd.read_csv(label_version(labels)["path"])[clean_mask(labels)].reset_index(drop=True)


def benchmark_provenance(labels: str = "current") -> dict:
    """What a result file should record about the labels it was scored against."""
    v = label_version(labels)
    return {"labels": labels, "file": str(Path(v["path"]).relative_to(BASE)), "sha256": v["sha256"],
            "n": CLEAN_N, "positives": v["positives"]}


def to_clean(a) -> np.ndarray:
    """Restrict a per-row array to the 437 clean rows.

    Accepts arrays over all 449 rows, over the 440 rows left after dropping `in_train` (what
    the older scripts saved), or already over the 437.
    """
    a = np.asarray(a)
    keep = clean_mask()
    if len(a) == CLEAN_N:
        return a
    if len(a) == len(keep):
        return a[keep]
    in_train = pd.read_csv(UNIFIED).in_train.to_numpy().astype(bool)
    if len(a) == int((~in_train).sum()):
        return a[keep[~in_train]]
    raise ValueError(f"cannot align an array of length {len(a)} with the benchmark "
                     f"({len(keep)} rows, {int((~in_train).sum())} after in_train, {CLEAN_N} clean)")


@dataclass
class Benchmark:
    name: str
    path: Path
    sha: str
    texts: list[str]
    labels: np.ndarray
    sources: list[str] | None = None

    @property
    def prevalence(self) -> float:
        return float(self.labels.mean())

    def __len__(self) -> int:
        return len(self.labels)


def load_benchmark(
    name: str,
    path: str | Path,
    *,
    expect_sha: str | None = None,
    text_col: str = "sentence",
    label_col: str = "label",
    source_col: str | None = "source",
    sep: str = ",",
    encoding: str = "utf-8",
) -> Benchmark:
    """Load a benchmark, refusing to proceed if its content hash has moved."""
    path = Path(path)
    got = sha256(path)
    if expect_sha is not None and got != expect_sha:
        raise ValueError(
            f"{name}: content hash changed.\n  expected {expect_sha}\n  got      {got}\n"
            f"  file     {path}\nRefusing to evaluate against a mutated benchmark."
        )
    df = pd.read_csv(path, sep=sep, encoding=encoding)
    if text_col not in df.columns:
        raise KeyError(f"{name}: no column {text_col!r}. Columns: {list(df.columns)}")
    if label_col not in df.columns:
        raise KeyError(f"{name}: no column {label_col!r}. Columns: {list(df.columns)}")
    sources = df[source_col].astype(str).tolist() if source_col and source_col in df.columns else None
    return Benchmark(
        name=name,
        path=path,
        sha=got,
        texts=df[text_col].astype(str).tolist(),
        labels=df[label_col].astype(int).to_numpy(),
        sources=sources,
    )


def degenerate(pred) -> str | None:
    """Name the degenerate operating point, or None if the prediction is non-trivial.

    An all-positive or all-negative prediction can produce a spectacular McNemar p-value
    that says nothing about the model -- results/loss_shaping/summary.csv carries
    p=3.18e-10 from exactly this. Any table that reports a p-value should call this and
    refuse to print the p, or print it flagged.
    """
    import numpy as _np
    a = _np.asarray(pred)
    if a.size == 0:
        return "empty"
    if (a == 1).all():
        return "all-positive"
    if (a == 0).all():
        return "all-negative"
    return None


# ── thresholds (never fitted on the reporting set) ────────────────────────

def threshold_from_prior(train_pos_rate: float, target_pos_rate: float) -> float:
    """Closed-form prior-shift threshold.

    A model trained at prior ``p_tr`` and applied to a population with prior
    ``p_te`` needs its decision boundary moved by the log-odds difference.
    No fitting, no held-out data, and identical for every model, which is what
    makes cross-model comparison meaningful.
    """
    def logit(p: float) -> float:
        p = min(max(p, 1e-9), 1 - 1e-9)
        return float(np.log(p / (1 - p)))

    shift = logit(target_pos_rate) - logit(train_pos_rate)
    return float(1.0 / (1.0 + np.exp(shift)))


def threshold_from_dev(dev_probs: np.ndarray, dev_labels: np.ndarray,
                       grid: np.ndarray | None = None) -> float:
    """Pick a threshold on a DEV split. Legal only if dev is disjoint from test."""
    grid = np.arange(0.01, 1.00, 0.01) if grid is None else grid
    scores = [f1_score(dev_labels, (dev_probs >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


# ── metrics ───────────────────────────────────────────────────────────────

def metrics_at(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """Point metrics at a GIVEN threshold. The threshold is never chosen here."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "tpr": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "auc": float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else float("nan"),
        "auprc": float(average_precision_score(labels, probs)) if len(set(labels)) > 1 else float("nan"),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n": int(len(labels)), "n_pos": int(labels.sum()),
    }


def majority_baseline(labels: np.ndarray) -> dict:
    """All-positive and all-negative baselines. If the model cannot beat these,
    the benchmark is not evidence of anything."""
    labels = np.asarray(labels, dtype=int)
    allpos = f1_score(labels, np.ones_like(labels), zero_division=0)
    allneg = f1_score(labels, np.zeros_like(labels), zero_division=0)
    return {
        "all_positive_f1": float(allpos),
        "all_negative_f1": float(allneg),
        "best_trivial_f1": float(max(allpos, allneg)),
        "prevalence": float(labels.mean()),
    }


def precision_at_prevalence(tpr: float, fpr: float, prevalence: float) -> float:
    """Precision the model would achieve at a deployment prevalence.

    TPR/FPR are prevalence-invariant; precision is not. V1 trained at an 8.5%
    prior and reported precision at 48-56%, which overstates deployment
    precision by roughly a factor of three.
    """
    num = tpr * prevalence
    den = num + fpr * (1 - prevalence)
    return float(num / den) if den > 0 else 0.0


# ── uncertainty ───────────────────────────────────────────────────────────

def _one_metric(probs, labels, threshold, metric):
    """Compute a single metric cheaply, the bootstrap must not recompute
    AUC and AUPRC ten thousand times to report an F1 interval."""
    if metric == "auc":
        return roc_auc_score(labels, probs)
    if metric == "auprc":
        return average_precision_score(labels, probs)
    preds = (probs >= threshold).astype(int)
    if metric == "f1":
        return f1_score(labels, preds, zero_division=0)
    if metric == "precision":
        return precision_score(labels, preds, zero_division=0)
    if metric == "recall":
        return recall_score(labels, preds, zero_division=0)
    return metrics_at(probs, labels, threshold)[metric]


def bootstrap_ci(probs: np.ndarray, labels: np.ndarray, threshold: float,
                 metric: str = "f1", n_boot: int = 10_000, seed: int = 20260820,
                 alpha: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    probs, labels = np.asarray(probs, float), np.asarray(labels, int)
    n = len(labels)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yl = labels[idx]
        if yl.min() == yl.max():
            continue
        vals.append(_one_metric(probs[idx], yl, threshold, metric))
    vals = np.array(vals)
    return {
        "mean": float(vals.mean()),
        "ci_low": float(np.percentile(vals, 100 * alpha / 2)),
        "ci_high": float(np.percentile(vals, 100 * (1 - alpha / 2))),
        "n_boot": int(len(vals)),
    }


def paired_bootstrap_delta(probs_a: np.ndarray, probs_b: np.ndarray, labels: np.ndarray,
                           t_a: float, t_b: float, metric: str = "f1",
                           n_boot: int = 10_000, seed: int = 20260820) -> dict:
    """CI on (A - B) using the SAME resample for both, which is the only way to
    get a usable interval on a small paired comparison."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, int)
    n = len(labels)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yl = labels[idx]
        if yl.min() == yl.max():
            continue
        deltas.append(_one_metric(probs_a[idx], yl, t_a, metric)
                      - _one_metric(probs_b[idx], yl, t_b, metric))
    deltas = np.array(deltas)
    return {
        "delta_mean": float(deltas.mean()),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "p_delta_gt_0": float((deltas > 0).mean()),
        "n_boot": int(len(deltas)),
    }


def mcnemar(probs_a: np.ndarray, probs_b: np.ndarray, labels: np.ndarray,
            t_a: float, t_b: float) -> dict:
    """Continuity-corrected two-tailed McNemar on the paired decisions."""
    from scipy.stats import chi2
    labels = np.asarray(labels, int)
    a = (np.asarray(probs_a) >= t_a).astype(int) == labels
    b = (np.asarray(probs_b) >= t_b).astype(int) == labels
    n01 = int((~a & b).sum())   # B right, A wrong
    n10 = int((a & ~b).sum())   # A right, B wrong
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "n_disagree": 0, "chi2": 0.0, "p_value": 1.0}
    stat = (abs(n01 - n10) - 1) ** 2 / n
    return {"n01": n01, "n10": n10, "n_disagree": n,
            "chi2": round(float(stat), 4),
            "p_value": round(float(chi2.sf(stat, 1)), 4)}


# ── reporting ─────────────────────────────────────────────────────────────

def report(bench: Benchmark, probs: np.ndarray, threshold: float, *,
           model_name: str = "model", deploy_prevalence: float = 0.0848,
           with_ci: bool = True) -> dict:
    """Full honest report for one model on one benchmark."""
    out = {
        "model": model_name,
        "benchmark": bench.name,
        "benchmark_path": str(bench.path),
        "benchmark_sha256": bench.sha,
        "n": len(bench),
        "prevalence": bench.prevalence,
        "overall": metrics_at(probs, bench.labels, threshold),
        "trivial_baseline": majority_baseline(bench.labels),
    }
    o = out["overall"]
    out["beats_trivial"] = bool(o["f1"] > out["trivial_baseline"]["best_trivial_f1"])
    out["precision_at_deploy_prevalence"] = {
        "prevalence": deploy_prevalence,
        "precision": precision_at_prevalence(o["tpr"], o["fpr"], deploy_prevalence),
    }
    if with_ci:
        out["f1_ci"] = bootstrap_ci(probs, bench.labels, threshold, "f1")
        out["auprc_ci"] = bootstrap_ci(probs, bench.labels, threshold, "auprc")
    if bench.sources:
        per = {}
        src = np.array(bench.sources)
        for s in sorted(set(bench.sources)):
            m = src == s
            if m.sum() < 5 or len(set(bench.labels[m])) < 2:
                per[s] = {"n": int(m.sum()), "note": "too small or single-class"}
                continue
            per[s] = metrics_at(probs[m], bench.labels[m], threshold)
            per[s]["trivial_baseline"] = majority_baseline(bench.labels[m])
            per[s]["beats_trivial"] = bool(
                per[s]["f1"] > per[s]["trivial_baseline"]["best_trivial_f1"])
        out["per_source"] = per
    return out


def format_report(rep: dict) -> str:
    o = rep["overall"]
    L = [
        f"{rep['model']}  on  {rep['benchmark']}  (n={rep['n']}, prevalence={rep['prevalence']:.3f})",
        f"  threshold {o['threshold']:.4f}",
        f"  F1 {o['f1']:.4f}   P {o['precision']:.4f}   R {o['recall']:.4f}",
        f"  AUC {o['auc']:.4f}   AUPRC {o['auprc']:.4f}",
        f"  TPR {o['tpr']:.4f}   FPR {o['fpr']:.4f}",
    ]
    if "f1_ci" in rep:
        c = rep["f1_ci"]
        L.append(f"  F1 95% CI [{c['ci_low']:.4f}, {c['ci_high']:.4f}]")
    tb = rep["trivial_baseline"]
    flag = "" if rep["beats_trivial"] else "   <-- DOES NOT BEAT TRIVIAL BASELINE"
    L.append(f"  trivial best F1 {tb['best_trivial_f1']:.4f}{flag}")
    p = rep["precision_at_deploy_prevalence"]
    L.append(f"  precision at deploy prevalence {p['prevalence']:.3f}: {p['precision']:.4f}")
    if "per_source" in rep:
        L.append("  per source:")
        for s, m in rep["per_source"].items():
            if "note" in m:
                L.append(f"    {s:24} n={m['n']:4}  ({m['note']})")
            else:
                bt = m["trivial_baseline"]["best_trivial_f1"]
                fl = "" if m["beats_trivial"] else "  <-- below trivial"
                L.append(f"    {s:24} n={m['n']:4}  F1 {m['f1']:.4f}  "
                         f"AUPRC {m['auprc']:.4f}  trivial {bt:.4f}{fl}")
    return "\n".join(L)
