#!/usr/bin/env python3
"""
Fine-tune a distilled model for a few more epochs on a new dataset.

Usage:
    python classifier/scripts/finetune_distilled.py \
        --model classifier/models/distilled_BiomedBERT_v2 \
        --data classifier/data/training/v18_hybrid/dataset.csv \
        --epochs 3 --lr 5e-6 \
        --output-dir classifier/models/distilled_BiomedBERT_v2_finetuned \
        --results-dir classifier/results/distillation_v2_finetuned
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, precision_score, recall_score

BASE   = Path("/path/to/MetaP/classifier")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_LEN = 256

EP_TEST    = BASE / "data/evaluation/globi-relax_passages-triplets_2024-02-28_curation_EP.tsv"
EVAL_100   = BASE / "data/evaluation/eval_100.tsv"
SYNTH_GOLD = BASE / "data/evaluation/synthetic_gold_100.tsv"


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(
            texts, max_length=MAX_LEN, padding="max_length",
            truncation=True, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {**{k: v[idx] for k, v in self.enc.items()}, "labels": self.labels[idx]}


def evaluate_model(model, tok, test_file, text_col, label_col, sep="\t", name=""):
    df = pd.read_csv(test_file, sep=sep)
    texts  = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(int).tolist()
    probs = []
    model.eval()
    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
            p = torch.softmax(model(**enc).logits, dim=-1)[0, 1].item()
            probs.append(p)
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.05):
        f = f1_score(labels, [1 if p >= t else 0 for p in probs], zero_division=0)
        if f > best_f1: best_f1, best_t = f, t
    preds = [1 if p >= best_t else 0 for p in probs]
    pr = precision_score(labels, preds, zero_division=0)
    rc = recall_score(labels, preds, zero_division=0)
    print(f"  {name}: F1={best_f1:.3f}  Prec={pr:.3f}  Rec={rc:.3f}  thr={best_t:.2f}", flush=True)
    return {"name": name, "f1": best_f1, "prec": pr, "rec": rc, "threshold": best_t}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       required=True, help="Path to distilled model to fine-tune")
    parser.add_argument("--data",        required=True, help="CSV with 'text' and 'label' columns")
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--lr",          type=float, default=5e-6)
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--output-dir",  required=True)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    res_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Data: {args.data}", flush=True)
    print(f"Epochs={args.epochs}  LR={args.lr}", flush=True)

    # Load data
    df = pd.read_csv(args.data)
    texts  = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    print(f"  Loaded {len(df)} rows, {sum(labels)} pos ({sum(labels)/len(labels):.1%})", flush=True)

    # Load model from checkpoint
    tok   = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, local_files_only=True).to(DEVICE)

    ds = TextDataset(texts, labels, tok)
    n_val   = max(int(0.1 * len(ds)), 200)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

    best_val_f1 = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            labs   = batch.pop("labels").to(DEVICE)
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            out    = model(**inputs, labels=labs)
            loss   = out.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                labs   = batch.pop("labels").cpu().numpy()
                inputs = {k: v.to(DEVICE) for k, v in batch.items()}
                preds  = torch.argmax(model(**inputs).logits, dim=-1).cpu().numpy()
                val_preds.extend(preds.tolist())
                val_true.extend(labs.tolist())

        val_f1 = f1_score(val_true, val_preds, zero_division=0)
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch}/{args.epochs} | loss={avg_loss:.4f} | val_F1={val_f1:.3f}", flush=True)
        history.append({"epoch": epoch, "loss": avg_loss, "val_f1": val_f1})

        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            model.save_pretrained(str(out_dir))
            tok.save_pretrained(str(out_dir))

    print(f"  Best val F1={best_val_f1:.3f}. Saved → {out_dir}", flush=True)
    with open(res_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Evaluate
    print("\n=== Evaluating fine-tuned model ===", flush=True)
    # Reload best checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(
        str(out_dir), local_files_only=True).to(DEVICE)
    results = [
        evaluate_model(model, tok, EP_TEST,    "sentence", "evaluation_pair_interacting", name="Finetuned — EP-relax"),
        evaluate_model(model, tok, EVAL_100,   "sentence", "evaluation_pair_interacting", name="Finetuned — eval_100"),
        evaluate_model(model, tok, SYNTH_GOLD, "text",     "label",                       name="Finetuned — synthetic_gold"),
    ]
    with open(res_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {res_dir}/eval_results.json")


if __name__ == "__main__":
    main()
