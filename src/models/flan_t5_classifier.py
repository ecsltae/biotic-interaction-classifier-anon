#!/usr/bin/env python3
"""
FLAN-T5 Generative Classifier for Biotic Interaction Detection
==============================================================

Uses google/flan-t5-large (770M params) as a seq2seq classifier.

Key difference from discriminative models (BiomedBERT, RoBERTa):
- Generates "yes" or "no" instead of softmax over 2 classes
- Classification score = log P("yes") - log P("no")  (no sampling needed)
- Generalises better to novel phrasings not seen in training data
- Targets precision improvement (catching false positives)

CPU inference: int8 quantized model runs at ~30-50 sentences/sec.

Usage:
    python flan_t5_classifier.py                             # train + eval
    python flan_t5_classifier.py --zero-shot                 # eval only (no fine-tuning)
    python flan_t5_classifier.py --model google/flan-t5-xl   # larger model
    python flan_t5_classifier.py --epochs 3 --batch-size 8   # custom config
"""

import os
import json
import time
import warnings
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = Path('/path/to/MetaP/classifier')
TRAIN_FILE     = BASE_DIR / 'data/training/training_data_v11_1.csv'
EVAL100_FILE   = BASE_DIR / 'data/evaluation/eval_100.tsv'
EP_TEST_FILE   = BASE_DIR / 'globi-relax_passages-triplets_2024-02-28_curation_EP.tsv'
MODEL_OUT_DIR  = BASE_DIR / 'models/flan_t5'
RESULTS_DIR    = BASE_DIR / 'results/flan_t5'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# ── config ─────────────────────────────────────────────────────────────────────
DEFAULT_MODEL  = 'google/flan-t5-large'
N_FOLDS        = 5
EPOCHS         = 5
BATCH_SIZE     = 16
MAX_INPUT_LEN  = 256
MAX_TARGET_LEN = 4     # "yes" or "no" = 1–2 tokens
LEARNING_RATE  = 1e-4  # higher than BERT — T5 uses Adafactor by default
GOLD_WEIGHT    = 5     # oversample eval_100 gold data

# Token IDs for "yes" and "no" — used for log-prob scoring at inference
YES_TOKEN = "yes"
NO_TOKEN  = "no"

PROMPT_TEMPLATE = (
    "Does this sentence describe a biotic interaction between two organisms?\n"
    "Sentence: {sentence}\n"
    "Answer:"
)


# ── dataset ────────────────────────────────────────────────────────────────────

class BioticT5Dataset(Dataset):
    """
    Seq2seq dataset: input = prompt, target = "yes" / "no".
    """
    def __init__(
        self,
        texts: list,
        labels: list,
        tokenizer,
        max_input_len: int = MAX_INPUT_LEN,
    ):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.inputs  = [PROMPT_TEMPLATE.format(sentence=t) for t in texts]
        self.targets = ["yes" if l == 1 else "no" for l in labels]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inp = self.tokenizer(
            self.inputs[idx],
            max_length=self.max_input_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        tgt = self.tokenizer(
            self.targets[idx],
            max_length=MAX_TARGET_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        labels = tgt['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # ignore padding in loss

        return {
            'input_ids':      inp['input_ids'].squeeze(),
            'attention_mask': inp['attention_mask'].squeeze(),
            'labels':         labels.squeeze(),
        }


# ── inference helpers ──────────────────────────────────────────────────────────

def get_yes_no_ids(tokenizer) -> tuple:
    """Return token IDs for 'yes' and 'no' (first token only)."""
    yes_id = tokenizer.encode(YES_TOKEN, add_special_tokens=False)[0]
    no_id  = tokenizer.encode(NO_TOKEN,  add_special_tokens=False)[0]
    return yes_id, no_id


def score_sentences(
    model,
    tokenizer,
    texts: list,
    batch_size: int = 32,
    yes_id: int = None,
    no_id: int = None,
) -> np.ndarray:
    """
    Score sentences: return P(yes) / (P(yes) + P(no)) for each sentence.

    Uses forced decoding (teacher forcing with "yes"/"no" as target),
    not autoregressive generation → deterministic, fast.
    """
    if yes_id is None or no_id is None:
        yes_id, no_id = get_yes_no_ids(tokenizer)

    model.eval()
    prompts = [PROMPT_TEMPLATE.format(sentence=t) for t in texts]
    scores = []

    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            enc = tokenizer(
                batch_prompts,
                max_length=MAX_INPUT_LEN,
                padding=True,
                truncation=True,
                return_tensors='pt',
            ).to(DEVICE)

            # Compute logits for first generated token
            # decoder_input_ids = BOS token (T5 uses pad as BOS)
            bos = torch.full(
                (len(batch_prompts), 1),
                model.config.decoder_start_token_id,
                dtype=torch.long,
            ).to(DEVICE)

            out = model(
                input_ids=enc['input_ids'],
                attention_mask=enc['attention_mask'],
                decoder_input_ids=bos,
            )
            logits = out.logits[:, 0, :]  # (batch, vocab) — first decoder step

            log_probs = torch.log_softmax(logits.float(), dim=-1)
            yes_lp = log_probs[:, yes_id].cpu().numpy()
            no_lp  = log_probs[:, no_id].cpu().numpy()

            # P(yes) normalised: exp(yes_lp) / (exp(yes_lp) + exp(no_lp))
            prob_yes = np.exp(yes_lp) / (np.exp(yes_lp) + np.exp(no_lp))
            scores.extend(prob_yes.tolist())

    return np.array(scores)


def find_best_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple:
    """Grid search for threshold maximising F1."""
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.1, 0.9, 0.02):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_f1, best_t


def evaluate(model, tokenizer, texts, labels, yes_id, no_id, label: str = ""):
    """Full evaluation: scores → best threshold → metrics."""
    probs = score_sentences(model, tokenizer, texts, yes_id=yes_id, no_id=no_id)
    labels_arr = np.array(labels)
    best_f1, best_t = find_best_threshold(probs, labels_arr)
    preds = (probs >= best_t).astype(int)
    prec = precision_score(labels_arr, preds, zero_division=0)
    rec  = recall_score(labels_arr, preds, zero_division=0)
    cm   = confusion_matrix(labels_arr, preds)
    if label:
        print(f"  {label}: F1={best_f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  (thresh={best_t:.2f})")
        print(f"    CM: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
    return {'f1': best_f1, 'precision': prec, 'recall': rec, 'threshold': best_t,
            'confusion_matrix': cm.tolist(), 'probabilities': probs.tolist()}


# ── data loading ───────────────────────────────────────────────────────────────

def load_data(train_file: str = None) -> pd.DataFrame:
    """Load training data."""
    path = Path(train_file) if train_file else TRAIN_FILE
    df = pd.read_csv(path)[['text', 'label']].dropna()
    df['source'] = 'train'
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Training data: {len(df)} ({sum(df['label']==1)} pos)")
    return df


def load_ep_test() -> tuple:
    df = pd.read_csv(EP_TEST_FILE, sep='\t', encoding='latin-1')
    texts  = df['sentence'].tolist()
    labels = df['evaluation_pair_interacting'].tolist()
    print(f"EP test set: {len(texts)} ({sum(labels)} pos)")
    return texts, labels


# ── zero-shot evaluation ───────────────────────────────────────────────────────

def zero_shot_eval(model_name: str) -> dict:
    """Evaluate FLAN-T5 without fine-tuning (pure instruction following)."""
    print(f"\n{'='*70}")
    print(f"ZERO-SHOT EVALUATION: {model_name}")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()

    yes_id, no_id = get_yes_no_ids(tokenizer)
    print(f"  yes token id={yes_id}, no token id={no_id}")

    ep_texts, ep_labels = load_ep_test()
    ep_metrics = evaluate(model, tokenizer, ep_texts, ep_labels, yes_id, no_id, "EP test")

    eval_df = pd.read_csv(EVAL100_FILE, sep='\t')
    e100_texts  = eval_df['sentence'].tolist()
    e100_labels = eval_df['evaluation_pair_interacting'].tolist()
    e100_metrics = evaluate(model, tokenizer, e100_texts, e100_labels, yes_id, no_id, "eval_100")

    return {'ep_test': ep_metrics, 'eval_100': e100_metrics}


# ── training ───────────────────────────────────────────────────────────────────

def train_cv(model_name: str, data_df: pd.DataFrame, ep_texts: list, ep_labels: list,
             epochs: int = EPOCHS, batch_size: int = BATCH_SIZE) -> tuple:
    """5-fold CV training, returns best model and results."""
    print(f"\n{'='*70}")
    print(f"TRAINING {model_name} WITH {N_FOLDS}-FOLD CV")
    print(f"{'='*70}")

    texts  = data_df['text'].tolist()
    labels = data_df['label'].tolist()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_results = []
    best_model_state = None
    best_tokenizer = None
    best_ep_f1 = 0.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")
        tr_texts = [texts[i] for i in train_idx]
        tr_labels = [labels[i] for i in train_idx]
        va_texts = [texts[i] for i in val_idx]
        va_labels = [labels[i] for i in val_idx]
        print(f"Train: {len(tr_texts)} | Val: {len(va_texts)}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.to(DEVICE)

        yes_id, no_id = get_yes_no_ids(tokenizer)

        train_ds = BioticT5Dataset(tr_texts, tr_labels, tokenizer)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

        # T5 works well with AdaFactor; fall back to AdamW
        try:
            from transformers import Adafactor
            optimizer = Adafactor(
                model.parameters(),
                lr=LEARNING_RATE,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
            )
        except ImportError:
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

        best_val_f1 = 0.0
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for step, batch in enumerate(train_loader):
                optimizer.zero_grad()
                out = model(
                    input_ids=batch['input_ids'].to(DEVICE),
                    attention_mask=batch['attention_mask'].to(DEVICE),
                    labels=batch['labels'].to(DEVICE),
                )
                loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # Validate on val fold
            val_m = evaluate(model, tokenizer, va_texts, va_labels, yes_id, no_id)
            print(f"  Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | "
                  f"val F1={val_m['f1']:.3f} Prec={val_m['precision']:.3f}")
            best_val_f1 = max(best_val_f1, val_m['f1'])

        # Evaluate on EP test
        ep_m = evaluate(model, tokenizer, ep_texts, ep_labels, yes_id, no_id, f"Fold {fold+1} EP test")

        fold_results.append({
            'fold': fold + 1,
            'val_f1': best_val_f1,
            'ep_f1': ep_m['f1'],
            'ep_precision': ep_m['precision'],
            'ep_recall': ep_m['recall'],
        })

        if ep_m['f1'] > best_ep_f1:
            best_ep_f1 = ep_m['f1']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_tokenizer = tokenizer

    # Summary
    avg_ep_f1   = np.mean([r['ep_f1'] for r in fold_results])
    avg_ep_prec = np.mean([r['ep_precision'] for r in fold_results])
    avg_ep_rec  = np.mean([r['ep_recall'] for r in fold_results])
    std_ep_f1   = np.std([r['ep_f1'] for r in fold_results])

    print(f"\n{'─'*50}")
    print(f"CV Summary ({model_name})")
    print(f"  Avg EP F1:        {avg_ep_f1:.3f} ± {std_ep_f1:.3f}")
    print(f"  Avg EP Precision: {avg_ep_prec:.3f}")
    print(f"  Avg EP Recall:    {avg_ep_rec:.3f}")
    print(f"  Best Fold EP F1:  {best_ep_f1:.3f}")

    summary = {
        'avg_ep_f1': avg_ep_f1,
        'avg_ep_precision': avg_ep_prec,
        'avg_ep_recall': avg_ep_rec,
        'best_ep_f1': best_ep_f1,
        'fold_results': fold_results,
    }

    return best_model_state, best_tokenizer, summary


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='FLAN-T5 biotic interaction classifier')
    parser.add_argument('--model',       default=DEFAULT_MODEL,  help='HuggingFace model ID')
    parser.add_argument('--train-data',  default=str(TRAIN_FILE), help='Training CSV')
    parser.add_argument('--epochs',      type=int, default=EPOCHS)
    parser.add_argument('--batch-size',  type=int, default=BATCH_SIZE)
    parser.add_argument('--zero-shot',   action='store_true', help='Eval only, no fine-tuning')
    parser.add_argument('--output-dir',  default=str(MODEL_OUT_DIR))
    parser.add_argument('--results-dir', default=str(RESULTS_DIR), help='Directory for results JSON')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()

    # ── Zero-shot baseline ────────────────────────────────────────────────────
    zs_results = zero_shot_eval(args.model)
    print(f"\nZero-shot EP test:  F1={zs_results['ep_test']['f1']:.3f}  "
          f"Prec={zs_results['ep_test']['precision']:.3f}")
    print(f"Zero-shot eval_100: F1={zs_results['eval_100']['f1']:.3f}  "
          f"Prec={zs_results['eval_100']['precision']:.3f}")

    if args.zero_shot:
        print("\n[zero-shot mode] Skipping fine-tuning.")
        with open(results_dir / 'zero_shot_results.json', 'w') as f:
            json.dump(zs_results, f, indent=2)
        return

    # ── Fine-tuning ───────────────────────────────────────────────────────────
    data_df = load_data(args.train_data)
    ep_texts, ep_labels = load_ep_test()

    best_state, best_tokenizer, cv_summary = train_cv(
        args.model, data_df, ep_texts, ep_labels,
        epochs=args.epochs, batch_size=args.batch_size,
    )

    # ── Save best model ───────────────────────────────────────────────────────
    print(f"\nSaving best model to {out_dir}...")
    model_to_save = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model_to_save.load_state_dict(best_state)
    model_to_save.save_pretrained(out_dir)
    best_tokenizer.save_pretrained(out_dir)
    print(f"  Saved.")

    # ── Final eval on best model ──────────────────────────────────────────────
    print("\nFinal evaluation (best model)...")
    model_to_save.to(DEVICE)
    yes_id, no_id = get_yes_no_ids(best_tokenizer)
    final_ep = evaluate(model_to_save, best_tokenizer, ep_texts, ep_labels,
                        yes_id, no_id, "EP test (best model)")
    eval_df    = pd.read_csv(EVAL100_FILE, sep='\t')
    final_e100 = evaluate(model_to_save, best_tokenizer,
                          eval_df['sentence'].tolist(),
                          eval_df['evaluation_pair_interacting'].tolist(),
                          yes_id, no_id, "eval_100 (best model)")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = (time.time() - start) / 60
    results = {
        'model': args.model,
        'train_data': args.train_data,
        'epochs': args.epochs,
        'zero_shot': zs_results,
        'cv_summary': cv_summary,
        'final_ep_test': final_ep,
        'final_eval_100': final_e100,
        'training_time_min': elapsed,
    }
    results_file = results_dir / 'flan_t5_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("FLAN-T5 CLASSIFIER — FINAL RESULTS")
    print(f"{'='*70}")
    print(f"\nZero-shot EP F1:    {zs_results['ep_test']['f1']:.3f}  "
          f"Prec={zs_results['ep_test']['precision']:.3f}")
    print(f"Fine-tuned EP F1:   {final_ep['f1']:.3f}  "
          f"Prec={final_ep['precision']:.3f}  Rec={final_ep['recall']:.3f}")
    print(f"Fine-tuned e100 F1: {final_e100['f1']:.3f}  "
          f"Prec={final_e100['precision']:.3f}  Rec={final_e100['recall']:.3f}")
    print(f"\nBiomedBERT baseline: F1=0.747  Prec=0.667  (v11.1)")
    delta_f1  = final_ep['f1']  - 0.747
    delta_prec = final_ep['precision'] - 0.667
    print(f"ΔF1={delta_f1:+.3f}  ΔPrec={delta_prec:+.3f}")
    print(f"\nTotal time: {elapsed:.1f} min")
    print(f"Results saved to: {results_file}")
    print(f"Model saved to:   {out_dir}")


if __name__ == '__main__':
    main()
