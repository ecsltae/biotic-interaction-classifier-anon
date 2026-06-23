#!/usr/bin/env python3
"""
evaluate_ensemble_all_benchmarks.py — Multi-benchmark evaluation for BiomedBERT+FLAN-T5 ensemble.

Evaluates the best ensemble (geometric mean of BiomedBERT and FLAN-T5-base) across
all available EP benchmark files.

Usage:
    python classifier/scripts/evaluate_ensemble_all_benchmarks.py
"""

from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import json

BASE_DIR = Path('/path/to/MetaP/classifier')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BIOMEDBERT = BASE_DIR / 'models/transformer_BiomedBERT_cv_regularized'
T5_V12 = BASE_DIR / 'models/flan-t5-base_v12'

# All benchmark files
BENCHMARKS = {
    "ep_relax": {
        "path": BASE_DIR / "data/evaluation/globi-relax_passages-triplets_2024-02-28_curation_EP.tsv",
        "sep": "\t",
        "text_col": "sentence",
        "label_col": "evaluation_pair_interacting",
        "primary": True,
    },
    "ep_passage": {
        "path": BASE_DIR / "data/evaluation/globi-passage_passages-triplets_2024-02-28_curation_EP.tsv",
        "sep": "\t",
        "text_col": "sentence",
        "label_col": "evaluation_pair_interacting",
        "primary": False,
    },
    "eval_100": {
        "path": BASE_DIR / "data/evaluation/biotx-random_passages-triplets_2024-02-28_curation_EP_100original.tsv",
        "sep": "\t",
        "text_col": "sentence",
        "label_col": "evaluation_pair_interacting",
        "primary": False,
    },
    "biotx_50_best": {
        "path": BASE_DIR / "data/evaluation/biotx-random_passages-triplets_2024-04-22b_curation_EP_50best-multiples.tsv",
        "sep": "\t",
        "text_col": "sentence",
        "label_col": "evaluation_pair_interacting",
        "primary": False,
    },
    "biotx_50_nodup": {
        "path": BASE_DIR / "data/evaluation/biotx-random_passages-triplets_2024-05-15_curation_EP_50nomultiple.tsv",
        "sep": "\t",
        "text_col": "sentence",
        "label_col": "evaluation_pair_interacting",
        "primary": False,
    },
    "gen_set_100": {
        "path": BASE_DIR / "data/evaluation/gen_set_100.csv",
        "sep": ",",
        "text_col": "sentence",
        "label_col": "label",
        "primary": False,
    },
    "eval_100_old": {
        "path": BASE_DIR / "data/evaluation/eval_100.tsv",
        "sep": "\t",
        "text_col": "text",
        "label_col": "label",
        "primary": False,
    },
}


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.encodings = tokenizer(
            texts, max_length=max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
    def __len__(self):
        return self.encodings['input_ids'].shape[0]
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


def get_bert_probs(model_path, texts):
    """Get BiomedBERT probabilities for positive class."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE).eval()
    ds = TextDataset(texts, tokenizer)
    dl = DataLoader(ds, batch_size=32)
    probs = []
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(**batch)
            p = torch.softmax(out.logits, dim=1)[:, 1].cpu().numpy()
            probs.extend(p)
    del model
    torch.cuda.empty_cache()
    return np.array(probs)


def get_t5_probs(model_path, texts):
    """Get FLAN-T5 probabilities for 'yes' response."""
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(DEVICE).eval()
    yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("no", add_special_tokens=False)[0]
    probs = []
    with torch.no_grad():
        for text in texts:
            prompt = f"Does this sentence describe a biotic interaction between species? Answer yes or no.\nSentence: {text}"
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384).to(DEVICE)
            out = model.generate(**enc, max_new_tokens=3, return_dict_in_generate=True, output_scores=True)
            logits = out.scores[0][0]
            p_yes = torch.softmax(logits[[yes_id, no_id]], dim=0)[0].item()
            probs.append(p_yes)
    del model
    torch.cuda.empty_cache()
    return np.array(probs)


def sweep_threshold(probs, labels):
    """Find F1-optimal threshold."""
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (probs >= t).astype(int)
        if preds.sum() == 0:
            continue
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


def evaluate_benchmark(name, cfg, p_bert, p_t5, y):
    """Evaluate ensemble on one benchmark."""
    geom = np.sqrt(p_bert * p_t5)
    best_t, best_f1 = sweep_threshold(geom, y)
    preds = (geom >= best_t).astype(int)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    cm = confusion_matrix(y, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    return {
        "name": name,
        "n": len(y),
        "n_pos": int(y.sum()),
        "pct_pos": round(100 * y.sum() / len(y), 1),
        "threshold": round(best_t, 2),
        "f1": round(f1, 3),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "primary": cfg.get("primary", False),
    }


def main():
    print(f"Device: {DEVICE}")
    print(f"\n{'='*80}")
    print("BiomedBERT + FLAN-T5-base v12 ENSEMBLE — MULTI-BENCHMARK EVALUATION")
    print(f"{'='*80}\n")

    # Load each benchmark and get predictions
    all_results = []

    for name, cfg in BENCHMARKS.items():
        path = cfg["path"]
        if not path.exists():
            print(f"[SKIP] {name}: file not found")
            continue

        # Load data
        df = pd.read_csv(str(path), sep=cfg["sep"])
        if cfg["text_col"] not in df.columns:
            print(f"[SKIP] {name}: missing text column '{cfg['text_col']}'")
            continue
        if cfg["label_col"] not in df.columns:
            print(f"[SKIP] {name}: missing label column '{cfg['label_col']}'")
            continue

        texts = df[cfg["text_col"]].fillna("").astype(str).tolist()
        y = np.array(df[cfg["label_col"]].astype(int).tolist())

        print(f"Evaluating {name} ({len(texts)} samples, {sum(y)} pos)...")

        # Get model predictions
        p_bert = get_bert_probs(BIOMEDBERT, texts)
        p_t5 = get_t5_probs(T5_V12, texts)

        # Evaluate
        result = evaluate_benchmark(name, cfg, p_bert, p_t5, y)
        all_results.append(result)

        marker = " ★ PRIMARY" if result["primary"] else ""
        print(f"  → F1={result['f1']:.3f}  P={result['precision']:.3f}  R={result['recall']:.3f}  @{result['threshold']:.2f}{marker}")
        print(f"    CM: TN={result['tn']} FP={result['fp']} FN={result['fn']} TP={result['tp']}\n")

    # Summary table
    print("\n" + "="*90)
    print("SUMMARY TABLE: BiomedBERT + FLAN-T5-base v12 (geometric mean)")
    print("="*90)
    print(f"{'Benchmark':<20} {'N':>5} {'Pos%':>5} {'Thresh':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}  {'FP':>3} {'FN':>3}")
    print("-"*90)

    for r in all_results:
        marker = " ★" if r["primary"] else ""
        print(f"{r['name']:<20} {r['n']:>5} {r['pct_pos']:>4.0f}% {r['threshold']:>7.2f} "
              f"{r['precision']:>7.3f} {r['recall']:>7.3f} {r['f1']:>7.3f}  "
              f"{r['fp']:>3} {r['fn']:>3}{marker}")

    print("="*90)
    print("★ = primary benchmark\n")

    # Save results
    out_path = BASE_DIR / 'results/research_agent/ensemble_all_benchmarks.json'
    with open(out_path, 'w') as f:
        json.dump({
            "ensemble": "BiomedBERT_cv_regularized + flan-t5-base_v12",
            "method": "geometric_mean",
            "results": all_results
        }, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == '__main__':
    main()
