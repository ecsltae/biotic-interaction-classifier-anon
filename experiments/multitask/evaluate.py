#!/usr/bin/env python3
"""
Evaluate a saved MultiTaskBiomedBERT checkpoint on the EP-relax test set
and compare against the distilled_BiomedBERT_v2 baseline.

Usage:
    python evaluate.py \
        --model ../../models/multitask_basic_a05 \
        --ep-relax ../../data/evaluation/globi-relax_passages-triplets_2024-02-28_curation_EP.tsv \
        [--distilled-baseline ../../models/distilled_BiomedBERT_v2] \
        [--threshold 0.25] \
        --results-dir ../../results/multitask/basic_a05
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, str(Path(__file__).parent))
from model import MultiTaskBiomedBERT


# ── EP-relax loader ───────────────────────────────────────────────────────

def load_ep_relax(path: str) -> tuple[list[str], list[int]]:
    df = pd.read_csv(path, sep="\t")
    # EP-relax columns: sentence + evaluation_pair_interacting (1/0)
    # Fallback for other eval files: any col with label/gold
    text_col = next(
        (c for c in df.columns if c == "sentence"),
        next(c for c in df.columns if "sentence" in c.lower() or "text" in c.lower())
    )
    label_col = next(
        (c for c in df.columns if c == "evaluation_pair_interacting"
         or "label" in c.lower() or "gold" in c.lower()),
        None
    )
    if label_col is None:
        raise ValueError(f"No label column found. Columns: {df.columns.tolist()}")
    texts  = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(int).tolist()
    return texts, labels


# ── Inference helpers ─────────────────────────────────────────────────────

def predict_multitask(model, tokenizer, texts, device, batch_size=32) -> np.ndarray:
    model.eval()
    probs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(
            batch_texts, truncation=True, max_length=256,
            padding=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out = model(
                input_ids      = enc["input_ids"],
                attention_mask = enc["attention_mask"],
                token_type_ids = enc.get("token_type_ids"),
            )
        p = torch.softmax(out["cls_logits"], dim=-1)[:, 1].cpu().numpy()
        probs.extend(p.tolist())
    return np.array(probs)


def predict_distilled(model, tokenizer, texts, device, batch_size=32) -> np.ndarray:
    model.eval()
    probs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = [t.lower() for t in texts[i:i+batch_size]]
        enc = tokenizer(
            batch_texts, truncation=True, max_length=256,
            padding=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            p = torch.softmax(model(**enc).logits, dim=-1)[:, 1].cpu().numpy()
        probs.extend(p.tolist())
    return np.array(probs)


def metrics_at(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "f1":    float(f1_score(labels, preds, zero_division=0)),
        "prec":  float(precision_score(labels, preds, zero_division=0)),
        "rec":   float(recall_score(labels, preds, zero_division=0)),
        "auc":   float(roc_auc_score(labels, probs)),
        "n_pos": int(sum(preds)),
        "n_total": len(preds),
    }


def best_f1_threshold(probs, labels):
    best_t, best_f = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        f = f1_score(labels, (probs >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return float(best_t), float(best_f)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       required=True, help="Path to multitask checkpoint dir")
    parser.add_argument("--ep-relax",    required=True, help="Path to EP-relax .tsv")
    parser.add_argument("--distilled-baseline", default=None,
                        help="Path to distilled_BiomedBERT_v2 dir for comparison")
    parser.add_argument("--threshold",   type=float, default=0.25,
                        help="Fixed threshold for evaluation (default 0.25)")
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    texts, labels = load_ep_relax(args.ep_relax)
    print(f"EP-relax: {len(texts)} sentences, {sum(labels)} positive", flush=True)

    report = {}

    # ── Multi-task model ──────────────────────────────────────────────────
    print(f"\nLoading multi-task model from {args.model} ...", flush=True)
    mt_model = MultiTaskBiomedBERT.load(args.model, device=str(device))
    from transformers import AutoTokenizer as _AT
    # Load tokenizer from the encoder name stored in the checkpoint config,
    # not from the local dir (which has no vocab files).
    _mt_cfg = json.load(open(Path(args.model) / "multitask_config.json"))
    mt_tok = _AT.from_pretrained(_mt_cfg["encoder_name"])

    mt_probs = predict_multitask(mt_model, mt_tok, texts, device)
    best_t, best_f = best_f1_threshold(mt_probs, labels)
    fixed = metrics_at(mt_probs, labels, args.threshold)
    best  = metrics_at(mt_probs, labels, best_t)

    report["multitask"] = {
        "fixed_threshold_0.25": fixed,
        "best_threshold":       best,
    }
    print(f"  [fixed t=0.25] F1={fixed['f1']:.4f}  P={fixed['prec']:.3f}  R={fixed['rec']:.3f}  AUC={fixed['auc']:.4f}")
    print(f"  [best  t={best_t:.2f}] F1={best['f1']:.4f}  P={best['prec']:.3f}  R={best['rec']:.3f}")

    # ── Distilled baseline ────────────────────────────────────────────────
    if args.distilled_baseline:
        print(f"\nLoading distilled baseline from {args.distilled_baseline} ...", flush=True)
        dist_tok   = _AT.from_pretrained(args.distilled_baseline, local_files_only=True)
        dist_model = AutoModelForSequenceClassification.from_pretrained(
            args.distilled_baseline, local_files_only=True
        ).to(device).eval()

        d_probs = predict_distilled(dist_model, dist_tok, texts, device)
        d_best_t, _ = best_f1_threshold(d_probs, labels)
        d_fixed = metrics_at(d_probs, labels, 0.25)
        d_best  = metrics_at(d_probs, labels, d_best_t)

        report["distilled_v2_baseline"] = {
            "fixed_threshold_0.25": d_fixed,
            "best_threshold":       d_best,
        }
        print(f"  [fixed t=0.25] F1={d_fixed['f1']:.4f}  P={d_fixed['prec']:.3f}  R={d_fixed['rec']:.3f}  AUC={d_fixed['auc']:.4f}")
        print(f"  [best  t={d_best_t:.2f}] F1={d_best['f1']:.4f}  P={d_best['prec']:.3f}  R={d_best['rec']:.3f}")

        delta_f1 = best["f1"] - d_best["f1"]
        print(f"\n  Δ F1 (multitask - baseline) @ best threshold: {delta_f1:+.4f}")
        report["delta_f1_vs_baseline"] = float(delta_f1)

    out_path = results_dir / "ep_relax_eval.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
