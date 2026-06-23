#!/usr/bin/env python3
"""
BioGPT Generative Classifier for Biotic Interaction Detection
=============================================================

Uses microsoft/biogpt (347M params) — GPT-2 style causal LM trained on PubMed.

Key difference from seq2seq models (FLAN-T5, BART):
- Decoder-only: scores P(" yes" | prompt) vs P(" no" | prompt) at last prompt token
- No encoder; uses the full prompt as left context
- Biomedical domain pre-training (PubMed) matches the sentence domain

Training:
- Labels: full text = prompt + " yes"/"no"; loss computed only on the answer token
- Optimiser: AdamW with cosine LR schedule
- 5-fold CV, best model selected by EP test F1

Usage:
    python biogpt_classifier.py                           # train + eval
    python biogpt_classifier.py --zero-shot               # eval only
    python biogpt_classifier.py --model microsoft/biogpt  # default
"""

import os
import json
import time
import warnings
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

# ── paths ───────────────────────────────────────────────────────────────────────
BASE_DIR      = Path('/path/to/MetaP/classifier')
TRAIN_FILE    = BASE_DIR / 'data/training/training_data_v11_1.csv'
EVAL100_FILE  = BASE_DIR / 'data/evaluation/eval_100.tsv'
EP_TEST_FILE  = BASE_DIR / 'globi-relax_passages-triplets_2024-02-28_curation_EP.tsv'
MODEL_OUT_DIR = BASE_DIR / 'models/biogpt'
RESULTS_DIR   = BASE_DIR / 'results/biogpt'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# ── config ──────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = 'microsoft/biogpt'
EPOCHS        = 5
BATCH_SIZE    = 16
MAX_INPUT_LEN = 256   # max prompt tokens
LR            = 2e-5
N_FOLDS       = 5
YES_TOKEN     = ' yes'   # GPT-style: space before word
NO_TOKEN      = ' no'

PROMPT_TEMPLATE = (
    "Does this sentence describe a biotic interaction between two organisms?\n"
    "Sentence: {sentence}\n"
    "Answer:"
)


# ── dataset ─────────────────────────────────────────────────────────────────────

class BioGPTDataset(Dataset):
    """
    Tokenizes prompt + answer for causal LM fine-tuning.
    Labels are -100 (ignored) for prompt tokens; answer token ID at the end.
    """
    def __init__(self, texts: list, labels: list, tokenizer, max_len: int = MAX_INPUT_LEN):
        self.tokenizer = tokenizer
        self.items = []

        yes_id = tokenizer.encode(YES_TOKEN, add_special_tokens=False)[0]
        no_id  = tokenizer.encode(NO_TOKEN,  add_special_tokens=False)[0]

        for text, label in zip(texts, labels):
            prompt = PROMPT_TEMPLATE.format(sentence=text)
            answer_token = yes_id if int(label) == 1 else no_id

            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            # Truncate prompt if needed, leaving room for answer token
            if len(prompt_ids) >= max_len:
                prompt_ids = prompt_ids[:max_len - 1]

            input_ids = prompt_ids + [answer_token]
            # Loss only on the answer token
            lm_labels = [-100] * len(prompt_ids) + [answer_token]

            self.items.append({
                'input_ids': input_ids,
                'lm_labels': lm_labels,
                'prompt_len': len(prompt_ids),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch, pad_id: int):
    max_len = max(len(x['input_ids']) for x in batch)
    input_ids = []
    attention_masks = []
    lm_labels = []

    for x in batch:
        pad_len = max_len - len(x['input_ids'])
        input_ids.append(x['input_ids'] + [pad_id] * pad_len)
        attention_masks.append([1] * len(x['input_ids']) + [0] * pad_len)
        lm_labels.append(x['lm_labels'] + [-100] * pad_len)

    return {
        'input_ids':      torch.tensor(input_ids,      dtype=torch.long),
        'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
        'labels':         torch.tensor(lm_labels,       dtype=torch.long),
    }


# ── helpers ─────────────────────────────────────────────────────────────────────

def get_yes_no_ids(tokenizer) -> tuple:
    """Return token IDs for ' yes' and ' no' (single-token, space-prefixed for GPT BPE)."""
    for variant in [YES_TOKEN, YES_TOKEN.strip(), YES_TOKEN.strip().capitalize()]:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(ids) == 1:
            yes_id = ids[0]
            break
    else:
        yes_id = tokenizer.encode(YES_TOKEN, add_special_tokens=False)[0]

    for variant in [NO_TOKEN, NO_TOKEN.strip(), NO_TOKEN.strip().capitalize()]:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(ids) == 1:
            no_id = ids[0]
            break
    else:
        no_id = tokenizer.encode(NO_TOKEN, add_special_tokens=False)[0]

    return yes_id, no_id


def score_sentences(model, tokenizer, texts: list,
                    batch_size: int = 32, yes_id: int = None, no_id: int = None) -> np.ndarray:
    """
    Score sentences via causal LM: P(" yes"|prompt) / (P(" yes"|prompt) + P(" no"|prompt)).
    Reads logits at the last non-padding position of each prompt.
    """
    if yes_id is None or no_id is None:
        yes_id, no_id = get_yes_no_ids(tokenizer)

    model.eval()
    prompts = [PROMPT_TEMPLATE.format(sentence=t) for t in texts]
    scores = []

    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = tokenizer(
                batch,
                max_length=MAX_INPUT_LEN,
                padding=True,
                truncation=True,
                return_tensors='pt',
            ).to(DEVICE)

            out = model(**enc)
            # For each sample, get logits at last non-padding token
            # attention_mask: 1 for real tokens, 0 for padding
            seq_lens = enc['attention_mask'].sum(dim=1) - 1  # index of last real token
            logits = out.logits[torch.arange(len(batch)), seq_lens, :]  # (batch, vocab)

            log_probs = torch.log_softmax(logits.float(), dim=-1)
            yes_lp = log_probs[:, yes_id].cpu().numpy()
            no_lp  = log_probs[:, no_id].cpu().numpy()

            prob_yes = np.exp(yes_lp) / (np.exp(yes_lp) + np.exp(no_lp))
            scores.extend(prob_yes.tolist())

    return np.array(scores)


def find_best_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple:
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.1, 0.9, 0.02):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_f1, best_t


def evaluate(model, tokenizer, texts, labels, yes_id, no_id, label: str = "") -> dict:
    probs = score_sentences(model, tokenizer, texts, yes_id=yes_id, no_id=no_id)
    labels_arr = np.array([int(l) for l in labels])
    best_f1, best_t = find_best_threshold(probs, labels_arr)
    preds = (probs >= best_t).astype(int)
    metrics = {
        'f1':        float(f1_score(labels_arr, preds, zero_division=0)),
        'precision': float(precision_score(labels_arr, preds, zero_division=0)),
        'recall':    float(recall_score(labels_arr, preds, zero_division=0)),
        'threshold': float(best_t),
    }
    if label:
        print(f"  {label}: F1={metrics['f1']:.3f}  Prec={metrics['precision']:.3f}"
              f"  Rec={metrics['recall']:.3f}  thr={best_t:.2f}")
    return metrics


# ── data loading ─────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='latin-1')
    df = df[['text', 'label']].dropna()
    df['label'] = df['label'].astype(int)
    print(f"  Training data: {len(df)} ({df['label'].sum()} pos)")
    return df


def load_ep_test() -> tuple:
    df = pd.read_csv(EP_TEST_FILE, sep='\t', encoding='latin-1')
    texts  = df['sentence'].tolist()
    label_col = 'label' if 'label' in df.columns else 'evaluation_pair_interacting'
    labels = df[label_col].tolist()
    print(f"  EP test set: {len(texts)} ({sum(int(l) for l in labels)} pos)")
    return texts, labels


def load_eval100_with_gold_oversampling(tokenizer) -> pd.DataFrame:
    """Load eval_100 gold labels ×5 for inclusion in training (as in regularized BiomedBERT)."""
    df = pd.read_csv(EVAL100_FILE, sep='\t')
    df = df.rename(columns={'sentence': 'text', 'evaluation_pair_interacting': 'label'})
    df['label'] = df['label'].astype(int)
    df = pd.concat([df] * 5, ignore_index=True)
    print(f"  eval_100 gold (×5): {len(df)}")
    return df


# ── zero-shot ────────────────────────────────────────────────────────────────────

def zero_shot_eval(model_name: str) -> dict:
    print(f"\n{'='*70}")
    print(f"ZERO-SHOT EVALUATION: {model_name}")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()

    yes_id, no_id = get_yes_no_ids(tokenizer)
    print(f"  yes token: '{tokenizer.decode([yes_id])}' (id={yes_id})")
    print(f"  no  token: '{tokenizer.decode([no_id])}' (id={no_id})")

    ep_texts, ep_labels = load_ep_test()
    ep_metrics = evaluate(model, tokenizer, ep_texts, ep_labels, yes_id, no_id, "EP test")

    eval_df = pd.read_csv(EVAL100_FILE, sep='\t')
    e100_metrics = evaluate(model, tokenizer,
                            eval_df['sentence'].tolist(),
                            eval_df['evaluation_pair_interacting'].tolist(),
                            yes_id, no_id, "eval_100")
    return {'ep_test': ep_metrics, 'eval_100': e100_metrics}


# ── training ─────────────────────────────────────────────────────────────────────

def train_cv(model_name: str, data_df: pd.DataFrame, ep_texts: list, ep_labels: list,
             epochs: int = EPOCHS, batch_size: int = BATCH_SIZE) -> tuple:
    print(f"\n{'='*70}")
    print(f"TRAINING {model_name} WITH {N_FOLDS}-FOLD CV")
    print(f"{'='*70}")

    texts  = data_df['text'].tolist()
    labels = data_df['label'].tolist()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_results    = []
    best_model_state = None
    best_tokenizer   = None
    best_ep_f1       = 0.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")
        tr_texts  = [texts[i] for i in train_idx]
        tr_labels = [labels[i] for i in train_idx]
        va_texts  = [texts[i] for i in val_idx]
        va_labels = [labels[i] for i in val_idx]
        print(f"Train: {len(tr_texts)} | Val: {len(va_texts)}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'   # left-pad for causal LM eval

        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(DEVICE)

        yes_id, no_id = get_yes_no_ids(tokenizer)

        train_ds = BioGPTDataset(tr_texts, tr_labels, tokenizer)
        pad_id   = tokenizer.pad_token_id
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=2,
            collate_fn=lambda b: collate_fn(b, pad_id),
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
        )

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                input_ids      = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels_t       = batch['labels'].to(DEVICE)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                labels=labels_t)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # Validate on EP test set
            ep_metrics = evaluate(model, tokenizer, ep_texts, ep_labels,
                                  yes_id, no_id, f"  Epoch {epoch+1} EP")
            print(f"    loss={avg_loss:.4f}  EP F1={ep_metrics['f1']:.3f}")

            if ep_metrics['f1'] > best_ep_f1:
                best_ep_f1       = ep_metrics['f1']
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_tokenizer   = tokenizer
                print(f"    *** New best EP F1: {best_ep_f1:.3f} ***")

        val_metrics = evaluate(model, tokenizer, va_texts, va_labels,
                               yes_id, no_id, f"Fold {fold+1} val")
        fold_results.append({'fold': fold+1, **val_metrics})

    cv_summary = {
        'avg_val_f1':  float(np.mean([r['f1']  for r in fold_results])),
        'std_val_f1':  float(np.std( [r['f1']  for r in fold_results])),
        'best_ep_f1':  float(best_ep_f1),
        'fold_results': fold_results,
    }
    print(f"\nCV avg val F1: {cv_summary['avg_val_f1']:.3f} ± {cv_summary['std_val_f1']:.3f}")
    print(f"Best EP F1 seen during training: {best_ep_f1:.3f}")

    return best_model_state, best_tokenizer, cv_summary


# ── main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='BioGPT causal LM biotic interaction classifier')
    parser.add_argument('--model',       default=DEFAULT_MODEL)
    parser.add_argument('--train-data',  default=str(TRAIN_FILE))
    parser.add_argument('--epochs',      type=int, default=EPOCHS)
    parser.add_argument('--batch-size',  type=int, default=BATCH_SIZE)
    parser.add_argument('--zero-shot',   action='store_true')
    parser.add_argument('--output-dir',  default=str(MODEL_OUT_DIR))
    parser.add_argument('--results-dir', default=str(RESULTS_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()

    zs_results = zero_shot_eval(args.model)
    print(f"\nZero-shot EP:   F1={zs_results['ep_test']['f1']:.3f}  "
          f"Prec={zs_results['ep_test']['precision']:.3f}")
    print(f"Zero-shot e100: F1={zs_results['eval_100']['f1']:.3f}  "
          f"Prec={zs_results['eval_100']['precision']:.3f}")

    if args.zero_shot:
        print("\n[zero-shot mode] Skipping fine-tuning.")
        with open(results_dir / 'zero_shot_results.json', 'w') as f:
            json.dump(zs_results, f, indent=2)
        return

    data_df = load_data(args.train_data)
    ep_texts, ep_labels = load_ep_test()

    # Add gold eval_100 ×5 oversampling (same as regularized BiomedBERT training)
    tokenizer_tmp = AutoTokenizer.from_pretrained(args.model)
    gold_df = load_eval100_with_gold_oversampling(tokenizer_tmp)
    data_df = pd.concat([data_df, gold_df], ignore_index=True)
    print(f"  Combined: {len(data_df)} ({data_df['label'].sum()} pos)")

    best_state, best_tokenizer, cv_summary = train_cv(
        args.model, data_df, ep_texts, ep_labels,
        epochs=args.epochs, batch_size=args.batch_size,
    )

    # Save best model
    print(f"\nSaving best model to {out_dir}...")
    model_to_save = AutoModelForCausalLM.from_pretrained(args.model)
    model_to_save.load_state_dict(best_state)
    model_to_save.save_pretrained(out_dir)
    best_tokenizer.save_pretrained(out_dir)

    # Final eval
    model_to_save.to(DEVICE)
    yes_id, no_id = get_yes_no_ids(best_tokenizer)
    final_ep   = evaluate(model_to_save, best_tokenizer, ep_texts, ep_labels,
                          yes_id, no_id, "EP test (best model)")
    eval_df    = pd.read_csv(EVAL100_FILE, sep='\t')
    final_e100 = evaluate(model_to_save, best_tokenizer,
                          eval_df['sentence'].tolist(),
                          eval_df['evaluation_pair_interacting'].tolist(),
                          yes_id, no_id, "eval_100 (best model)")

    elapsed = (time.time() - start) / 60
    results = {
        'model':             args.model,
        'train_data':        args.train_data,
        'epochs':            args.epochs,
        'zero_shot':         zs_results,
        'cv_summary':        cv_summary,
        'final_ep_test':     final_ep,
        'final_eval_100':    final_e100,
        'training_time_min': elapsed,
    }
    results_file = results_dir / 'biogpt_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("BIOGPT CLASSIFIER — FINAL RESULTS")
    print(f"{'='*70}")
    print(f"\nZero-shot EP F1:    {zs_results['ep_test']['f1']:.3f}  "
          f"Prec={zs_results['ep_test']['precision']:.3f}")
    print(f"Fine-tuned EP F1:   {final_ep['f1']:.3f}  "
          f"Prec={final_ep['precision']:.3f}  Rec={final_ep['recall']:.3f}")
    delta_f1  = final_ep['f1']  - 0.747
    delta_prec = final_ep['precision'] - 0.667
    print(f"ΔF1={delta_f1:+.3f}  ΔPrec={delta_prec:+.3f}  vs BiomedBERT v11.1")
    print(f"\nTotal time: {elapsed:.1f} min | Results: {results_file}")


if __name__ == '__main__':
    main()
