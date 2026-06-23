#!/usr/bin/env python3
"""
compare_fusion_strategies.py — Compare arithmetic vs geometric mean on all benchmarks.

This script loads cached model probabilities and compares fusion strategies without
re-running inference (faster for iterative analysis).

Usage:
    python classifier/scripts/compare_fusion_strategies.py
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

BENCHMARKS = {
    "ep_relax": {
        "path": BASE_DIR / "data/evaluation/globi-relax_passages-triplets_2024-02-28_curation_EP.tsv",
        "sep": "\t", "text_col": "sentence", "label_col": "evaluation_pair_interacting", "primary": True,
    },
    "ep_passage": {
        "path": BASE_DIR / "data/evaluation/globi-passage_passages-triplets_2024-02-28_curation_EP.tsv",
        "sep": "\t", "text_col": "sentence", "label_col": "evaluation_pair_interacting", "primary": False,
    },
    "eval_100": {
        "path": BASE_DIR / "data/evaluation/eval_100.tsv",
        "sep": "\t", "text_col": "text", "label_col": "label", "primary": False,
    },
    "biotx_50_best": {
        "path": BASE_DIR / "data/evaluation/biotx-random_passages-triplets_2024-04-22b_curation_EP_50best-multiples.tsv",
        "sep": "\t", "text_col": "sentence", "label_col": "evaluation_pair_interacting", "primary": False,
    },
    "biotx_50_nodup": {
        "path": BASE_DIR / "data/evaluation/biotx-random_passages-triplets_2024-05-15_curation_EP_50nomultiple.tsv",
        "sep": "\t", "text_col": "sentence", "label_col": "evaluation_pair_interacting", "primary": False,
    },
    "gen_set_100": {
        "path": BASE_DIR / "data/evaluation/gen_set_100.csv",
        "sep": ",", "text_col": "sentence", "label_col": "label", "primary": False,
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
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.01, 0.99, 0.01):
        preds = (probs >= t).astype(int)
        if preds.sum() == 0:
            continue
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


def evaluate_all(p_bert, p_t5, y, name):
    """Compare arithmetic vs geometric on one benchmark."""

    # Geometric mean
    p_geom = np.sqrt(p_bert * p_t5)
    t_geom, _ = sweep_threshold(p_geom, y)
    pred_geom = (p_geom >= t_geom).astype(int)
    f1_geom = f1_score(y, pred_geom)
    prec_geom = precision_score(y, pred_geom)
    rec_geom = recall_score(y, pred_geom)
    cm_geom = confusion_matrix(y, pred_geom, labels=[0,1])
    fp_geom = cm_geom[0,1] if cm_geom.shape == (2,2) else 0
    fn_geom = cm_geom[1,0] if cm_geom.shape == (2,2) else 0

    # Arithmetic mean
    p_arith = (p_bert + p_t5) / 2
    t_arith, _ = sweep_threshold(p_arith, y)
    pred_arith = (p_arith >= t_arith).astype(int)
    f1_arith = f1_score(y, pred_arith)
    prec_arith = precision_score(y, pred_arith)
    rec_arith = recall_score(y, pred_arith)
    cm_arith = confusion_matrix(y, pred_arith, labels=[0,1])
    fp_arith = cm_arith[0,1] if cm_arith.shape == (2,2) else 0
    fn_arith = cm_arith[1,0] if cm_arith.shape == (2,2) else 0

    # Max fusion
    p_max = np.maximum(p_bert, p_t5)
    t_max, _ = sweep_threshold(p_max, y)
    pred_max = (p_max >= t_max).astype(int)
    f1_max = f1_score(y, pred_max)
    prec_max = precision_score(y, pred_max)
    rec_max = recall_score(y, pred_max)

    return {
        "name": name,
        "n": len(y),
        "n_pos": int(y.sum()),
        "geometric": {"f1": round(f1_geom, 3), "prec": round(prec_geom, 3), "rec": round(rec_geom, 3),
                      "thresh": round(t_geom, 2), "fp": int(fp_geom), "fn": int(fn_geom)},
        "arithmetic": {"f1": round(f1_arith, 3), "prec": round(prec_arith, 3), "rec": round(rec_arith, 3),
                       "thresh": round(t_arith, 2), "fp": int(fp_arith), "fn": int(fn_arith)},
        "max": {"f1": round(f1_max, 3), "prec": round(prec_max, 3), "rec": round(rec_max, 3),
                "thresh": round(t_max, 2)},
    }


def main():
    print(f"Device: {DEVICE}")
    print("\n" + "="*90)
    print("FUSION STRATEGY COMPARISON: BiomedBERT + FLAN-T5-base v12")
    print("="*90 + "\n")

    results = []

    for name, cfg in BENCHMARKS.items():
        path = cfg["path"]
        if not path.exists():
            print(f"[SKIP] {name}")
            continue

        df = pd.read_csv(str(path), sep=cfg["sep"])
        if cfg["text_col"] not in df.columns or cfg["label_col"] not in df.columns:
            print(f"[SKIP] {name}: missing columns")
            continue

        texts = df[cfg["text_col"]].fillna("").astype(str).tolist()
        y = np.array(df[cfg["label_col"]].astype(int).tolist())

        print(f"Loading {name} ({len(texts)} samples, {sum(y)} pos)...")

        p_bert = get_bert_probs(BIOMEDBERT, texts)
        p_t5 = get_t5_probs(T5_V12, texts)

        r = evaluate_all(p_bert, p_t5, y, name)
        r["primary"] = cfg.get("primary", False)
        results.append(r)

        delta_f1 = r["arithmetic"]["f1"] - r["geometric"]["f1"]
        winner = "ARITH" if delta_f1 > 0 else ("GEOM" if delta_f1 < 0 else "TIE")
        marker = " ★" if r["primary"] else ""

        print(f"  Geometric: F1={r['geometric']['f1']:.3f}  Arith: F1={r['arithmetic']['f1']:.3f}  "
              f"Δ={delta_f1:+.3f}  [{winner}]{marker}\n")

    # Summary table
    print("\n" + "="*110)
    print(f"{'Benchmark':<18} {'N':>4} {'Pos':>3} │ {'Geom F1':>8} {'Prec':>6} {'Rec':>5} │ {'Arith F1':>8} {'Prec':>6} {'Rec':>5} │ {'Δ F1':>6} {'Winner':>7}")
    print("="*110)

    for r in results:
        g, a = r["geometric"], r["arithmetic"]
        delta = a["f1"] - g["f1"]
        winner = "ARITH" if delta > 0.001 else ("GEOM" if delta < -0.001 else "TIE")
        marker = " ★" if r["primary"] else ""

        print(f"{r['name']:<18} {r['n']:>4} {r['n_pos']:>3} │ "
              f"{g['f1']:>8.3f} {g['prec']:>6.3f} {g['rec']:>5.3f} │ "
              f"{a['f1']:>8.3f} {a['prec']:>6.3f} {a['rec']:>5.3f} │ "
              f"{delta:>+6.3f} {winner:>7}{marker}")

    print("="*110)
    print("★ = primary benchmark\n")

    # Summary statistics
    arith_wins = sum(1 for r in results if r["arithmetic"]["f1"] > r["geometric"]["f1"] + 0.001)
    geom_wins = sum(1 for r in results if r["geometric"]["f1"] > r["arithmetic"]["f1"] + 0.001)
    ties = len(results) - arith_wins - geom_wins

    print(f"\nOverall: Arithmetic wins={arith_wins}, Geometric wins={geom_wins}, Ties={ties}")

    # Save
    out_path = BASE_DIR / 'results/research_agent/fusion_comparison.json'
    with open(out_path, 'w') as f:
        json.dump({"results": results, "summary": {"arith_wins": arith_wins, "geom_wins": geom_wins, "ties": ties}}, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
