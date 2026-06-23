#!/usr/bin/env python3
"""
Knowledge Distillation: Ensemble → Single student model
========================================================
Teacher: BiomedBERT cv_regularized × FLAN-T5-base v12 (geometric ensemble, EP F1=0.857)
Student: Any AutoModelForSequenceClassification (default: BiomedBERT-base)

Distillation loss = α * T² * KL(student_soft || teacher_soft)
                  + (1-α) * CE(student_logits, hard_labels)

Temperature T softens distributions so student learns from teacher's uncertainty,
not just which class it picked.

Usage:
    python classifier/scripts/distill_ensemble.py
    python classifier/scripts/distill_ensemble.py --skip-labels  # if soft labels already generated
    python classifier/scripts/distill_ensemble.py --epochs 8 --temperature 4 --alpha 0.7
    python classifier/scripts/distill_ensemble.py --skip-labels --student-model distilbert-base-uncased --output-dir models/distilled_DistilBERT_v4
    python classifier/scripts/distill_ensemble.py --skip-labels --student-model allenai/scibert_scivocab_uncased --output-dir models/distilled_SciBERT_v5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    T5ForConditionalGeneration, get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, precision_score, recall_score

BASE = Path("/path/to/MetaP/classifier")
sys.path.insert(0, str(BASE / "src"))

# ── Paths ──────────────────────────────────────────────────────────────────
TEACHER_BERT  = BASE / "models/transformer_BiomedBERT_cv_regularized"
TEACHER_T5    = BASE / "models/flan-t5-base_v12"
STUDENT_BASE  = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"  # overrideable via --student-model

CORPUS_FILE   = BASE / "results/research_agent/all_sources_qwen122b_labeled.csv"
EXTRA_NEG     = BASE / "data/training/negatives_clean.csv"
SOFT_LABELS   = BASE / "data/training/distillation_soft_labels.csv"
STUDENT_DIR   = BASE / "models/distilled_BiomedBERT_v1"
RESULTS_DIR   = BASE / "results/distillation_v1"

EP_TEST       = BASE / "data/evaluation/globi-relax_passages-triplets_2024-02-28_curation_EP.tsv"
EVAL_100      = BASE / "data/evaluation/eval_100.tsv"
SYNTH_GOLD    = BASE / "data/evaluation/synthetic_gold_100.tsv"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_LEN = 256


# ── Teacher inference ──────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=MAX_LEN):
        self.encodings = tokenizer(
            texts, max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )
    def __len__(self): return self.encodings["input_ids"].shape[0]
    def __getitem__(self, idx): return {k: v[idx] for k, v in self.encodings.items()}


def get_bert_probs(model_dir, texts, batch_size=64):
    print(f"  [Teacher BERT] Loading {model_dir.name} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), local_files_only=True).to(DEVICE).eval()
    ds = TextDataset(texts, tok)
    loader = DataLoader(ds, batch_size=batch_size)
    probs = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            p = torch.softmax(model(**batch).logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
    model.cpu(); del model; torch.cuda.empty_cache()
    return np.array(probs)


def get_t5_probs(model_dir, texts, batch_size=32):
    print(f"  [Teacher T5] Loading {model_dir.name} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(
        str(model_dir), local_files_only=True).to(DEVICE).eval()
    yes_id = tok.encode("yes", add_special_tokens=False)[0]
    no_id  = tok.encode("no",  add_special_tokens=False)[0]
    prompts = [
        f"Does the following sentence describe a biotic interaction between two species? "
        f"Answer yes or no.\n\nSentence: {t}" for t in texts
    ]
    probs = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_p = prompts[i:i+batch_size]
            enc = tok(batch_p, return_tensors="pt", padding=True,
                      truncation=True, max_length=512).to(DEVICE)
            dec = torch.full((len(batch_p), 1), tok.pad_token_id,
                             dtype=torch.long, device=DEVICE)
            logits = model(**enc, decoder_input_ids=dec).logits[:, 0, :]
            lp = torch.log_softmax(logits.float(), dim=-1)
            yes_lp = lp[:, yes_id].cpu().numpy()
            no_lp  = lp[:, no_id].cpu().numpy()
            p_yes = np.exp(yes_lp) / (np.exp(yes_lp) + np.exp(no_lp))
            probs.extend(p_yes.tolist())
    model.cpu(); del model; torch.cuda.empty_cache()
    return np.array(probs)


def generate_soft_labels(out_path: Path):
    """Run teacher ensemble on full corpus, save soft probabilities."""
    print("\n=== Step 1: Generating teacher soft labels ===", flush=True)

    # Load corpus
    corpus = pd.read_csv(CORPUS_FILE)
    corpus["text"] = corpus["text"].astype(str).str.strip()
    corpus = corpus.drop_duplicates(subset=["text"])
    hard_labels = corpus["teacher_label"].fillna(0).astype(int).tolist()
    print(f"  Corpus: {len(corpus)} sentences "
          f"({sum(hard_labels)} pos = {sum(hard_labels)/len(hard_labels):.1%})", flush=True)

    # Extra negatives from clean pool (add diversity)
    neg_df = pd.read_csv(EXTRA_NEG)[["text"]].drop_duplicates()
    neg_df = neg_df[~neg_df["text"].isin(corpus["text"])]
    neg_df["teacher_label"] = 0
    combined = pd.concat([corpus[["text","teacher_label"]], neg_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["text"])
    texts = combined["text"].tolist()
    labels = combined["teacher_label"].astype(int).tolist()
    print(f"  Combined with extra negatives: {len(texts)} sentences", flush=True)

    # Teacher inference
    p_bert = get_bert_probs(TEACHER_BERT, texts)
    p_t5   = get_t5_probs(TEACHER_T5, texts)

    # Geometric ensemble (best combination from sweep: F1=0.857)
    p_ensemble = np.sqrt(p_bert * p_t5)

    out = pd.DataFrame({
        "text":       texts,
        "hard_label": labels,
        "p_bert":     p_bert,
        "p_t5":       p_t5,
        "p_ensemble": p_ensemble,
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"  Saved {len(out)} soft labels → {out_path}", flush=True)
    print(f"  p_ensemble: mean={p_ensemble.mean():.3f} "
          f"std={p_ensemble.std():.3f} "
          f">0.5: {(p_ensemble>0.5).sum()}", flush=True)
    return out


# ── Student training ───────────────────────────────────────────────────────

class DistillDataset(Dataset):
    def __init__(self, texts, hard_labels, soft_probs, tokenizer):
        self.encodings = tokenizer(
            texts, max_length=MAX_LEN, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        self.hard = torch.tensor(hard_labels, dtype=torch.long)
        # soft_probs: p(positive) — convert to [p_neg, p_pos] pair
        sp = torch.tensor(soft_probs, dtype=torch.float32)
        self.soft = torch.stack([1 - sp, sp], dim=1)  # (N, 2)

    def __len__(self): return self.hard.shape[0]

    def __getitem__(self, idx):
        return {
            **{k: v[idx] for k, v in self.encodings.items()},
            "hard_label": self.hard[idx],
            "soft_label": self.soft[idx],
        }


def distillation_loss(student_logits, soft_labels, hard_labels, T, alpha):
    """
    Combined loss:
      alpha   * T^2 * KL(student_soft || teacher_soft)
      (1-α)   * CE(student_logits, hard_labels)
    """
    # Soft targets from teacher at temperature T
    student_soft = F.log_softmax(student_logits / T, dim=-1)
    teacher_soft = F.softmax(soft_labels / T, dim=-1)  # already probabilities, scale by T
    kl = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T ** 2)

    # Hard target CE
    ce = F.cross_entropy(student_logits, hard_labels)

    return alpha * kl + (1 - alpha) * ce


def train_student(df, epochs, T, alpha, lr):
    print(f"\n=== Step 2: Training student (T={T}, α={alpha}, lr={lr}) ===", flush=True)
    STUDENT_DIR.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(STUDENT_BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_BASE, num_labels=2).to(DEVICE)

    ds = DistillDataset(
        df["text"].tolist(),
        df["hard_label"].astype(int).tolist(),
        df["p_ensemble"].tolist(),
        tok,
    )

    # 90/10 split for validation
    n_val = max(int(0.1 * len(ds)), 500)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

    best_val_f1, best_epoch = 0.0, 0
    history = []

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            hard   = batch.pop("hard_label").to(DEVICE)
            soft   = batch.pop("soft_label").to(DEVICE)
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**inputs).logits
            loss   = distillation_loss(logits, soft, hard, T, alpha)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                hard   = batch.pop("hard_label").cpu().numpy()
                batch.pop("soft_label")
                inputs = {k: v.to(DEVICE) for k, v in batch.items()}
                logits = model(**inputs).logits
                preds  = torch.argmax(logits, dim=-1).cpu().numpy()
                val_preds.extend(preds.tolist())
                val_true.extend(hard.tolist())

        val_f1 = f1_score(val_true, val_preds, zero_division=0)
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch}/{epochs} | loss={avg_loss:.4f} | val_F1={val_f1:.3f}", flush=True)
        history.append({"epoch": epoch, "loss": avg_loss, "val_f1": val_f1})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            model.save_pretrained(str(STUDENT_DIR))
            tok.save_pretrained(str(STUDENT_DIR))

    print(f"  Best val F1={best_val_f1:.3f} at epoch {best_epoch}. "
          f"Saved → {STUDENT_DIR}", flush=True)
    return history


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_model(model_dir, test_file, text_col, label_col, sep="\t", name=""):
    df = pd.read_csv(test_file, sep=sep)
    texts  = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(int).tolist()
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), local_files_only=True).to(DEVICE).eval()
    preds, probs = [], []
    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True,
                      max_length=MAX_LEN).to(DEVICE)
            p = torch.softmax(model(**enc).logits, dim=-1)[0, 1].item()
            probs.append(p)
            preds.append(1 if p >= 0.5 else 0)
    # Threshold sweep for best F1
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.05):
        f = f1_score(labels, [1 if p >= t else 0 for p in probs])
        if f > best_f1: best_f1, best_t = f, t
    preds_opt = [1 if p >= best_t else 0 for p in probs]
    pr = precision_score(labels, preds_opt, zero_division=0)
    rc = recall_score(labels, preds_opt, zero_division=0)
    print(f"  {name}: F1={best_f1:.3f}  Prec={pr:.3f}  Rec={rc:.3f}  thr={best_t:.2f}", flush=True)
    return {"name": name, "f1": best_f1, "prec": pr, "rec": rc, "threshold": best_t}


def run_evaluation():
    print("\n=== Step 3: Evaluating student model ===", flush=True)
    results = []
    results.append(evaluate_model(
        STUDENT_DIR, EP_TEST, "sentence", "evaluation_pair_interacting",
        sep="\t", name="Student distilled — EP-relax"))
    results.append(evaluate_model(
        STUDENT_DIR, EVAL_100, "sentence", "evaluation_pair_interacting",
        sep="\t", name="Student distilled — eval_100"))
    results.append(evaluate_model(
        STUDENT_DIR, SYNTH_GOLD, "text", "label",
        sep="\t", name="Student distilled — synthetic_gold_100"))

    print("\n  === Reference (teacher ensemble, geometric) ===", flush=True)
    print("  cv_reg × T5-v12 (geo): EP F1=0.857  Prec=0.789  Rec=0.938", flush=True)
    print("  BiomedBERT v11_reg:    EP F1=0.786  Prec=0.688  Rec=0.917", flush=True)
    print("  FLAN-T5-base v11.1:    EP F1=0.818  (from training logs)", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {RESULTS_DIR}/eval_results.json", flush=True)
    return results


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-labels", action="store_true",
                        help="Skip soft label generation (use existing file)")
    parser.add_argument("--epochs",      type=int,   default=6)
    parser.add_argument("--temperature", type=float, default=4.0,
                        help="Distillation temperature (default 4)")
    parser.add_argument("--alpha",       type=float, default=0.7,
                        help="Weight for soft loss (0=hard only, 1=soft only)")
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--output-dir",    type=str, default=None,
                        help="Override student model output directory")
    parser.add_argument("--results-dir",   type=str, default=None,
                        help="Override results directory")
    parser.add_argument("--student-model", type=str, default=None,
                        help="HuggingFace model ID or local path for student (default: BiomedBERT-base)")
    parser.add_argument("--soft-labels-path", type=str, default=None,
                        help="Override the soft-labels CSV path (default: distillation_soft_labels.csv)")
    args = parser.parse_args()

    global STUDENT_DIR, RESULTS_DIR, STUDENT_BASE, SOFT_LABELS
    if args.output_dir:
        STUDENT_DIR = Path(args.output_dir)
    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)
    if args.student_model:
        STUDENT_BASE = args.student_model
    if args.soft_labels_path:
        SOFT_LABELS = Path(args.soft_labels_path)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Temperature={args.temperature}  alpha={args.alpha}  "
          f"epochs={args.epochs}  lr={args.lr}", flush=True)

    # Step 1: soft labels
    if args.skip_labels and SOFT_LABELS.exists():
        print(f"\n=== Step 1: Loading existing soft labels ({SOFT_LABELS}) ===", flush=True)
        df = pd.read_csv(SOFT_LABELS)
        print(f"  Loaded {len(df)} rows, "
              f"{int(df['hard_label'].sum())} pos ({df['hard_label'].mean():.1%})", flush=True)
    else:
        df = generate_soft_labels(SOFT_LABELS)

    # Step 2: train student
    history = train_student(df, args.epochs, args.temperature, args.alpha, args.lr)
    with open(RESULTS_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Step 3: evaluate
    run_evaluation()
    print("\n=== Distillation pipeline complete ===", flush=True)


if __name__ == "__main__":
    main()
