#!/usr/bin/env python3
"""
Comprehensive evaluation of all available models on both test sets.

Evaluates:
  - Multi-task configs (experiments/multitask/)
  - Distilled BiomedBERT variants
  - BiomedBERT cv_regularized (discriminative baseline)

Test sets:
  - EP-relax (99 sentences, 48 positive) — real literature, hard
  - Synthetic gold (100 sentences, 50/50) — curated types

Usage:
    python scripts/eval_all_models.py
    python scripts/eval_all_models.py --output results/full_eval.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments/multitask"))
from model import MultiTaskBiomedBERT

ROOT = Path(__file__).parent.parent

# ── Test set loaders ──────────────────────────────────────────────────────

def load_ep_relax() -> tuple[list[str], list[int]]:
    path = ROOT / "data/evaluation/globi-relax_passages-triplets_2024-02-28_curation_EP.tsv"
    df = pd.read_csv(path, sep="\t")
    return df["sentence"].astype(str).tolist(), df["evaluation_pair_interacting"].astype(int).tolist()


def load_synthetic_gold() -> tuple[list[str], list[int], list[str]]:
    path = ROOT / "data/evaluation/synthetic_gold_100.tsv"
    df = pd.read_csv(path, sep="\t")
    return df["text"].astype(str).tolist(), df["label"].astype(int).tolist(), df["interaction_type"].astype(str).tolist()


# ── Inference ─────────────────────────────────────────────────────────────

def predict_hf(model, tokenizer, texts, device, batch_size=64, lowercase=False) -> np.ndarray:
    """Run HuggingFace AutoModelForSequenceClassification."""
    model.eval()
    probs = []
    for i in range(0, len(texts), batch_size):
        batch = [t.lower() for t in texts[i:i+batch_size]] if lowercase else texts[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, max_length=256,
                        padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            p = torch.softmax(model(**enc).logits, dim=-1)[:, 1].cpu().numpy()
        probs.extend(p.tolist())
    return np.array(probs)


def predict_multitask(model, tokenizer, texts, device, batch_size=64) -> np.ndarray:
    """Run MultiTaskBiomedBERT (classification head only)."""
    model.eval()
    probs = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i+batch_size], truncation=True, max_length=256,
                        padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                        token_type_ids=enc.get("token_type_ids"))
            p = torch.softmax(out["cls_logits"], dim=-1)[:, 1].cpu().numpy()
        probs.extend(p.tolist())
    return np.array(probs)


# ── Metrics ───────────────────────────────────────────────────────────────

def metrics_at(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "f1":    float(f1_score(labels, preds, zero_division=0)),
        "prec":  float(precision_score(labels, preds, zero_division=0)),
        "rec":   float(recall_score(labels, preds, zero_division=0)),
        "auc":   float(roc_auc_score(labels, probs)),
        "n_pos": int(sum(preds)),
    }


def best_threshold(probs, labels):
    best_t, best_f = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        f = f1_score(labels, (probs >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return float(best_t), float(best_f)


def per_type_f1(probs, labels, types, threshold):
    """F1 per interaction_type on the synthetic gold set."""
    preds = (probs >= threshold).astype(int)
    result = {}
    for t in sorted(set(types)):
        idx = [i for i, x in enumerate(types) if x == t]
        l = [labels[i] for i in idx]
        p = [preds[i]  for i in idx]
        if len(set(l)) < 2:
            continue  # skip pure-negative types (they're negatives)
        result[t] = round(float(f1_score(l, p, zero_division=0)), 3)
    return result


def eval_model(probs, labels, types=None, fixed_t=0.25):
    bt, bf = best_threshold(probs, labels)
    row = {
        "f1_fixed": metrics_at(probs, labels, fixed_t)["f1"],
        "prec_fixed": metrics_at(probs, labels, fixed_t)["prec"],
        "rec_fixed": metrics_at(probs, labels, fixed_t)["rec"],
        "f1_best":  bf,
        "best_t":   bt,
        "auc":      metrics_at(probs, labels, fixed_t)["auc"],
    }
    if types is not None:
        row["per_type"] = per_type_f1(probs, labels, types, fixed_t)
    return row


# ── Model registry ────────────────────────────────────────────────────────

def collect_models():
    """Return list of (name, kind, path, lowercase) tuples to evaluate."""
    models = []
    models_dir = ROOT / "models"

    # Multi-task configs
    mt_dir = models_dir / "multitask"
    if mt_dir.exists():
        for d in sorted(mt_dir.iterdir()):
            if d.is_dir() and (d / "config.json").exists():
                models.append((f"mt_{d.name}", "multitask", d, False))

    # Distilled BiomedBERT variants
    for name in sorted(models_dir.glob("distilled_*/config.json")):
        d = name.parent
        models.append((d.name, "hf", d, True))  # distilled uses lowercase

    # BiomedBERT cv_regularized (discriminative baseline)
    biomedbert_cv = models_dir / "transformer_BiomedBERT_cv_regularized"
    if biomedbert_cv.exists() and (biomedbert_cv / "config.json").exists():
        models.append(("BiomedBERT_cv_reg", "hf", biomedbert_cv, False))

    return models


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/full_eval.json")
    parser.add_argument("--fixed-threshold", type=float, default=0.25)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load test sets
    ep_texts, ep_labels = load_ep_relax()
    sg_texts, sg_labels, sg_types = load_synthetic_gold()
    print(f"EP-relax:     {len(ep_texts)} sentences, {sum(ep_labels)} positive")
    print(f"Synth gold:   {len(sg_texts)} sentences, {sum(sg_labels)} positive\n")

    model_list = collect_models()
    print(f"Models to evaluate: {len(model_list)}\n")

    results = {}

    for name, kind, path, lowercase in model_list:
        print(f"── {name} ──")
        try:
            tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)

            if kind == "multitask":
                model = MultiTaskBiomedBERT.load(str(path), device=str(device))
                ep_probs  = predict_multitask(model, tok, ep_texts,  device)
                sg_probs  = predict_multitask(model, tok, sg_texts,  device)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(
                    str(path), local_files_only=True).to(device).eval()
                ep_probs  = predict_hf(model, tok, ep_texts,  device, lowercase=lowercase)
                sg_probs  = predict_hf(model, tok, sg_texts,  device, lowercase=lowercase)

            ep_row = eval_model(ep_probs, ep_labels, fixed_t=args.fixed_threshold)
            sg_row = eval_model(sg_probs, sg_labels, types=sg_types, fixed_t=args.fixed_threshold)

            results[name] = {"ep_relax": ep_row, "synth_gold": sg_row}

            print(f"  EP-relax:   F1={ep_row['f1_fixed']:.4f} (t=0.25)  "
                  f"F1={ep_row['f1_best']:.4f} (best t={ep_row['best_t']:.2f})  "
                  f"AUC={ep_row['auc']:.4f}")
            print(f"  Synth gold: F1={sg_row['f1_fixed']:.4f} (t=0.25)  "
                  f"F1={sg_row['f1_best']:.4f} (best t={sg_row['best_t']:.2f})  "
                  f"AUC={sg_row['auc']:.4f}")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {"error": str(e)}

        print()

    # Save
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    # Summary table
    print("\n" + "="*100)
    print(f"{'Model':<35} {'EP F1@0.25':>10} {'EP F1best':>10} {'EP AUC':>8} {'SG F1@0.25':>10} {'SG F1best':>10} {'SG AUC':>8}")
    print("-"*100)

    rows = [(n, v) for n, v in results.items() if "error" not in v]
    rows.sort(key=lambda x: x[1]["ep_relax"]["f1_best"], reverse=True)

    for name, v in rows:
        ep = v["ep_relax"]
        sg = v["synth_gold"]
        print(f"{name:<35} {ep['f1_fixed']:>10.4f} {ep['f1_best']:>10.4f} {ep['auc']:>8.4f} "
              f"{sg['f1_fixed']:>10.4f} {sg['f1_best']:>10.4f} {sg['auc']:>8.4f}")

    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
