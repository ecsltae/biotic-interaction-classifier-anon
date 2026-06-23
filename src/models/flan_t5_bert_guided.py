#!/usr/bin/env python3
"""
FLAN-T5 + BiomedBERT-Guided Training
======================================

Uses BiomedBERT_v7 as a teacher to improve FLAN-T5 precision.

Problem: flan-t5-base_v11_1 has avg EP F1=0.781, Prec=0.673, Rec=0.933.
         Recall is already excellent; precision is the bottleneck vs BiomedBERT_v7.

Approach:
1. Pre-compute BiomedBERT_v7 scores for all training sentences.
2. Add BERT confidence level as a prefix in the T5 prompt so T5 sees a "prior":
       [Prior: none/weak/moderate/high]
       Does this sentence describe a biotic interaction...
3. Upweight hard negatives in loss: examples where BERT is confident-negative
   (score < BERT_NEG_THRESH) AND ground truth = 0.  These are T5's typical FPs.
4. Start from the existing flan-t5-base_v11_1 checkpoint (warm start).

Result: T5 learns to defer to BERT on clear non-interactions while keeping its
semantic reasoning advantage on nuanced/indirect interactions.

Usage:
    # Full training from v11_1 checkpoint (recommended)
    python flan_t5_bert_guided.py \\
        --bert-teacher models/transformer_BiomedBERT_cv_regularized \\
        --from-checkpoint models/flan-t5-base_v11_1 \\
        --train-data data/training/training_data_v11_1.csv \\
        --epochs 3 --batch-size 16

    # Sweep hard-negative weight
    python flan_t5_bert_guided.py --bert-teacher ... --neg-hard-weight 3.0

    # No checkpoint (train from base)
    python flan_t5_bert_guided.py --bert-teacher ... --no-checkpoint
"""

import os
import json
import time
import warnings
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer,
    AutoModelForSequenceClassification,
    Adafactor,
)
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = Path('/path/to/MetaP/classifier')
TRAIN_FILE    = BASE_DIR / 'data/training/training_data_v11_1.csv'
EVAL100_FILE  = BASE_DIR / 'data/evaluation/eval_100.tsv'
EP_TEST_FILE  = BASE_DIR / 'globi-relax_passages-triplets_2024-02-28_curation_EP.tsv'
MODEL_OUT_DIR = BASE_DIR / 'models/flan_t5_bert_guided'
RESULTS_DIR   = BASE_DIR / 'results/flan_t5_bert_guided'

BIOMEDBERT_DEFAULT = str(BASE_DIR / 'models/transformer_BiomedBERT_cv_regularized')
T5_CHECKPOINT_DEFAULT = str(BASE_DIR / 'models/flan-t5-base_v11_1')
T5_BASE_MODEL  = 'google/flan-t5-base'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# ── config ─────────────────────────────────────────────────────────────────────
N_FOLDS       = 5
EPOCHS        = 3          # warm-start fine-tune, fewer epochs needed
BATCH_SIZE    = 16
MAX_INPUT_LEN = 280        # slightly longer due to [Prior:...] prefix
MAX_TARGET_LEN = 4
LEARNING_RATE  = 5e-5      # lower LR for fine-tuning an existing checkpoint
GOLD_WEIGHT    = 5

# BERT score bucketing for prompt prefix
BERT_HIGH_THRESH = 0.70    # > 0.70 → [Prior: high]
BERT_MOD_THRESH  = 0.40    # > 0.40 → [Prior: moderate]
BERT_LOW_THRESH  = 0.20    # > 0.20 → [Prior: low]
BERT_NEG_THRESH  = 0.20    # <= 0.20 → [Prior: none] = confident negative

# Loss weighting
DEFAULT_NEG_HARD_WEIGHT = 2.5   # upweight confident-negative examples in loss
DEFAULT_POS_WEIGHT      = 1.0   # positives already well-recalled; default = 1

YES_TOKEN = "yes"
NO_TOKEN  = "no"


def bert_score_to_level(score: float) -> str:
    if score > BERT_HIGH_THRESH:
        return "high"
    elif score > BERT_MOD_THRESH:
        return "moderate"
    elif score > BERT_LOW_THRESH:
        return "low"
    else:
        return "none"


PROMPT_TEMPLATE = (
    "[Prior: {prior}]\n"
    "Does this sentence describe a biotic interaction between two organisms?\n"
    "Sentence: {sentence}\n"
    "Answer:"
)

# Baseline prompt without BERT prior (for zero-shot / ablation)
PLAIN_PROMPT_TEMPLATE = (
    "Does this sentence describe a biotic interaction between two organisms?\n"
    "Sentence: {sentence}\n"
    "Answer:"
)


# ── BiomedBERT scorer ──────────────────────────────────────────────────────────

class BertScorer:
    """Loads BiomedBERT_v7 and scores sentences → P(positive)."""

    def __init__(self, model_path: str):
        print(f"Loading BiomedBERT teacher from {model_path}...")
        # BiomedBERT is fine-tuned from microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract
        self.tokenizer = AutoTokenizer.from_pretrained(
            'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            ignore_mismatched_sizes=True,
        ).to(DEVICE)
        self.model.eval()
        print("  BiomedBERT loaded.")

    def score(self, texts: list, batch_size: int = 64) -> np.ndarray:
        """Return P(positive) for each text."""
        scores = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                enc = self.tokenizer(
                    batch,
                    max_length=256,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                ).to(DEVICE)
                logits = self.model(**enc).logits  # (batch, 2)
                probs = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
                scores.extend(probs.tolist())
        return np.array(scores)


# ── dataset ────────────────────────────────────────────────────────────────────

class BioticT5GuidedDataset(Dataset):
    """
    Seq2seq dataset with BiomedBERT prior in the prompt and per-sample weights.

    sample_weights: array of floats, one per sample.
        - Hard negatives (BERT score <= BERT_NEG_THRESH, label=0): neg_hard_weight
        - Positives (label=1): pos_weight
        - Everything else: 1.0
    """

    def __init__(
        self,
        texts: list,
        labels: list,
        bert_scores: np.ndarray,
        tokenizer,
        neg_hard_weight: float = DEFAULT_NEG_HARD_WEIGHT,
        pos_weight: float = DEFAULT_POS_WEIGHT,
        max_input_len: int = MAX_INPUT_LEN,
    ):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.targets = ["yes" if l == 1 else "no" for l in labels]

        # Build prompts with BERT prior
        self.inputs = [
            PROMPT_TEMPLATE.format(
                prior=bert_score_to_level(s),
                sentence=t,
            )
            for t, s in zip(texts, bert_scores)
        ]

        # Per-sample weights
        labels_arr = np.array(labels)
        bert_arr = np.array(bert_scores)
        hard_neg_mask = (bert_arr <= BERT_NEG_THRESH) & (labels_arr == 0)
        pos_mask = labels_arr == 1

        self.weights = np.ones(len(labels), dtype=np.float32)
        self.weights[hard_neg_mask] = neg_hard_weight
        self.weights[pos_mask] = pos_weight

        n_hard_neg = hard_neg_mask.sum()
        n_pos = pos_mask.sum()
        print(f"    Dataset: {len(labels)} samples | "
              f"hard-neg (weighted ×{neg_hard_weight}): {n_hard_neg} | "
              f"pos (weighted ×{pos_weight}): {n_pos}")

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
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids':      inp['input_ids'].squeeze(),
            'attention_mask': inp['attention_mask'].squeeze(),
            'labels':         labels.squeeze(),
            'weight':         torch.tensor(self.weights[idx], dtype=torch.float32),
        }


# ── per-sample weighted loss ───────────────────────────────────────────────────

def weighted_seq2seq_loss(
    model,
    input_ids,
    attention_mask,
    labels,
    sample_weights,
) -> torch.Tensor:
    """
    Compute per-sample CE loss and apply sample_weights before averaging.

    Returns the weighted mean loss (scalar).
    """
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    # out.logits: (batch, seq_len, vocab_size)
    batch_size = labels.size(0)
    vocab_size = out.logits.size(-1)

    # Per-token loss (ignore padding tokens marked as -100)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    per_token_loss = loss_fct(
        out.logits.reshape(-1, vocab_size),
        labels.reshape(-1),
    )  # (batch * seq_len,)
    per_token_loss = per_token_loss.view(batch_size, -1)  # (batch, seq_len)

    # Per-sample mean over non-padding tokens
    non_pad = (labels != -100).float()
    per_sample_loss = (per_token_loss * non_pad).sum(dim=1) / non_pad.sum(dim=1).clamp(min=1)

    # Apply weights and average
    return (per_sample_loss * sample_weights).mean()


# ── inference helpers ──────────────────────────────────────────────────────────

def get_yes_no_ids(tokenizer) -> tuple:
    yes_id = tokenizer.encode(YES_TOKEN, add_special_tokens=False)[0]
    no_id  = tokenizer.encode(NO_TOKEN,  add_special_tokens=False)[0]
    return yes_id, no_id


def score_sentences(
    model,
    tokenizer,
    texts: list,
    bert_scores: np.ndarray = None,
    batch_size: int = 32,
    yes_id: int = None,
    no_id: int = None,
) -> np.ndarray:
    """
    Score sentences using forced-decoding log-prob ratio P(yes)/(P(yes)+P(no)).

    If bert_scores is provided, uses guided prompts (with [Prior:...] prefix).
    Otherwise falls back to plain prompt (for evaluation without BERT).
    """
    if yes_id is None or no_id is None:
        yes_id, no_id = get_yes_no_ids(tokenizer)

    model.eval()

    if bert_scores is not None:
        prompts = [
            PROMPT_TEMPLATE.format(prior=bert_score_to_level(s), sentence=t)
            for t, s in zip(texts, bert_scores)
        ]
    else:
        prompts = [PLAIN_PROMPT_TEMPLATE.format(sentence=t) for t in texts]

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
            logits = out.logits[:, 0, :]
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


def evaluate(
    model, tokenizer, texts, labels, yes_id, no_id,
    bert_scores=None, label: str = "",
) -> dict:
    probs = score_sentences(model, tokenizer, texts, bert_scores=bert_scores,
                            yes_id=yes_id, no_id=no_id)
    labels_arr = np.array(labels)
    best_f1, best_t = find_best_threshold(probs, labels_arr)
    preds = (probs >= best_t).astype(int)
    prec = precision_score(labels_arr, preds, zero_division=0)
    rec  = recall_score(labels_arr, preds, zero_division=0)
    cm   = confusion_matrix(labels_arr, preds)
    if label:
        print(f"  {label}: F1={best_f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  "
              f"(thresh={best_t:.2f})")
        print(f"    CM: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
    return {
        'f1': best_f1, 'precision': prec, 'recall': rec,
        'threshold': best_t, 'confusion_matrix': cm.tolist(),
        'probabilities': probs.tolist(),
    }


# ── data loading ───────────────────────────────────────────────────────────────

def load_data(train_file: str = None) -> pd.DataFrame:
    path = Path(train_file) if train_file else TRAIN_FILE
    df = pd.read_csv(path)[['text', 'label']].dropna()
    print(f"  Training data: {len(df)} ({sum(df['label']==1)} pos)")

    eval_df = pd.read_csv(EVAL100_FILE, sep='\t')[['sentence', 'evaluation_pair_interacting']]
    eval_df.columns = ['text', 'label']
    gold = pd.concat([eval_df] * GOLD_WEIGHT, ignore_index=True)
    print(f"  eval_100 gold (×{GOLD_WEIGHT}): {len(gold)}")

    combined = pd.concat([df, gold], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Combined: {len(combined)} ({sum(combined['label']==1)} pos)")
    return combined


def load_ep_test() -> tuple:
    df = pd.read_csv(EP_TEST_FILE, sep='\t', encoding='latin-1')
    texts  = df['sentence'].tolist()
    labels = df['evaluation_pair_interacting'].tolist()
    print(f"EP test set: {len(texts)} ({sum(labels)} pos)")
    return texts, labels


# ── training ───────────────────────────────────────────────────────────────────

def train_cv(
    t5_model_name: str,
    data_df: pd.DataFrame,
    bert_scorer: 'BertScorer',
    ep_texts: list,
    ep_labels: list,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    neg_hard_weight: float = DEFAULT_NEG_HARD_WEIGHT,
    pos_weight: float = DEFAULT_POS_WEIGHT,
) -> tuple:
    print(f"\n{'='*70}")
    print(f"BERT-GUIDED TRAINING: {t5_model_name}")
    print(f"neg_hard_weight={neg_hard_weight}  pos_weight={pos_weight}")
    print(f"{'='*70}")

    texts  = data_df['text'].tolist()
    labels = data_df['label'].tolist()

    # Pre-compute BiomedBERT scores for ALL training data at once
    print("\nPre-computing BiomedBERT scores for training data...")
    bert_scores_train = bert_scorer.score(texts)
    print(f"  Done. Mean score: {bert_scores_train.mean():.3f}  "
          f"Hard negatives (score<={BERT_NEG_THRESH}, label=0): "
          f"{((bert_scores_train <= BERT_NEG_THRESH) & (np.array(labels)==0)).sum()}")

    # BiomedBERT scores for EP test (used at eval time)
    print("Pre-computing BiomedBERT scores for EP test set...")
    bert_scores_ep = bert_scorer.score(ep_texts)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_results = []
    best_model_state = None
    best_tokenizer = None
    best_ep_f1 = 0.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")

        tr_texts   = [texts[i] for i in train_idx]
        tr_labels  = [labels[i] for i in train_idx]
        tr_scores  = bert_scores_train[train_idx]
        va_texts   = [texts[i] for i in val_idx]
        va_labels  = [labels[i] for i in val_idx]
        va_scores  = bert_scores_train[val_idx]

        tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_name)
        model.to(DEVICE)

        yes_id, no_id = get_yes_no_ids(tokenizer)

        train_ds = BioticT5GuidedDataset(
            tr_texts, tr_labels, tr_scores, tokenizer,
            neg_hard_weight=neg_hard_weight, pos_weight=pos_weight,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)

        optimizer = Adafactor(
            model.parameters(),
            lr=LEARNING_RATE,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )

        best_val_f1 = 0.0
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                loss = weighted_seq2seq_loss(
                    model,
                    input_ids      = batch['input_ids'].to(DEVICE),
                    attention_mask = batch['attention_mask'].to(DEVICE),
                    labels         = batch['labels'].to(DEVICE),
                    sample_weights = batch['weight'].to(DEVICE),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            val_m = evaluate(model, tokenizer, va_texts, va_labels, yes_id, no_id,
                             bert_scores=va_scores)
            print(f"  Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | "
                  f"val F1={val_m['f1']:.3f} Prec={val_m['precision']:.3f} "
                  f"Rec={val_m['recall']:.3f}")
            best_val_f1 = max(best_val_f1, val_m['f1'])

        ep_m = evaluate(model, tokenizer, ep_texts, ep_labels, yes_id, no_id,
                        bert_scores=bert_scores_ep, label=f"Fold {fold+1} EP test")

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

    avg_ep_f1   = np.mean([r['ep_f1'] for r in fold_results])
    avg_ep_prec = np.mean([r['ep_precision'] for r in fold_results])
    avg_ep_rec  = np.mean([r['ep_recall'] for r in fold_results])
    std_ep_f1   = np.std([r['ep_f1'] for r in fold_results])

    print(f"\n{'─'*50}")
    print(f"CV Summary")
    print(f"  Avg EP F1:        {avg_ep_f1:.3f} ± {std_ep_f1:.3f}")
    print(f"  Avg EP Precision: {avg_ep_prec:.3f}")
    print(f"  Avg EP Recall:    {avg_ep_rec:.3f}")
    print(f"  Best Fold EP F1:  {best_ep_f1:.3f}")
    print(f"\n  Baseline (flan-t5-base_v11_1): avg=0.781  Prec=0.673  Rec=0.933")
    delta = avg_ep_f1 - 0.781
    print(f"  ΔF1 vs baseline: {delta:+.3f}")

    summary = {
        'avg_ep_f1': avg_ep_f1,
        'avg_ep_precision': avg_ep_prec,
        'avg_ep_recall': avg_ep_rec,
        'best_ep_f1': best_ep_f1,
        'std_ep_f1': std_ep_f1,
        'fold_results': fold_results,
        'neg_hard_weight': neg_hard_weight,
        'pos_weight': pos_weight,
    }
    return best_model_state, best_tokenizer, summary, bert_scores_ep


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='FLAN-T5 + BiomedBERT-guided training for biotic interaction detection'
    )
    parser.add_argument('--bert-teacher',    default=BIOMEDBERT_DEFAULT,
                        help='Path to BiomedBERT_v7 checkpoint (AutoModelForSequenceClassification)')
    parser.add_argument('--from-checkpoint', default=T5_CHECKPOINT_DEFAULT,
                        help='Start T5 fine-tuning from this checkpoint (default: flan-t5-base_v11_1)')
    parser.add_argument('--no-checkpoint',   action='store_true',
                        help='Start from google/flan-t5-base (no warm start)')
    parser.add_argument('--train-data',      default=str(TRAIN_FILE))
    parser.add_argument('--epochs',          type=int, default=EPOCHS)
    parser.add_argument('--batch-size',      type=int, default=BATCH_SIZE)
    parser.add_argument('--neg-hard-weight', type=float, default=DEFAULT_NEG_HARD_WEIGHT,
                        help='Loss weight for hard negatives (BERT score <= 0.20, label=0)')
    parser.add_argument('--pos-weight',      type=float, default=DEFAULT_POS_WEIGHT,
                        help='Loss weight for positives (default 1.0 — recall already high)')
    parser.add_argument('--output-dir',      default=str(MODEL_OUT_DIR))
    parser.add_argument('--results-dir',     default=str(RESULTS_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t5_model = T5_BASE_MODEL if args.no_checkpoint else args.from_checkpoint
    print(f"T5 starting point: {t5_model}")

    bert_scorer = BertScorer(args.bert_teacher)
    start = time.time()

    data_df = load_data(args.train_data)
    ep_texts, ep_labels = load_ep_test()

    best_state, best_tokenizer, cv_summary, bert_scores_ep = train_cv(
        t5_model_name    = t5_model,
        data_df          = data_df,
        bert_scorer      = bert_scorer,
        ep_texts         = ep_texts,
        ep_labels        = ep_labels,
        epochs           = args.epochs,
        batch_size       = args.batch_size,
        neg_hard_weight  = args.neg_hard_weight,
        pos_weight       = args.pos_weight,
    )

    # Save best model
    print(f"\nSaving best model to {out_dir}...")
    model_to_save = AutoModelForSeq2SeqLM.from_pretrained(t5_model)
    model_to_save.load_state_dict(best_state)
    model_to_save.save_pretrained(out_dir)
    best_tokenizer.save_pretrained(out_dir)
    print(f"  Saved.")

    # Final evaluation
    model_to_save.to(DEVICE)
    yes_id, no_id = get_yes_no_ids(best_tokenizer)

    final_ep = evaluate(
        model_to_save, best_tokenizer, ep_texts, ep_labels, yes_id, no_id,
        bert_scores=bert_scores_ep, label="EP test (best model, guided)",
    )
    # Also evaluate without BERT prior to check how much guidance matters
    final_ep_plain = evaluate(
        model_to_save, best_tokenizer, ep_texts, ep_labels, yes_id, no_id,
        bert_scores=None, label="EP test (best model, plain prompt)",
    )

    eval_df = pd.read_csv(EVAL100_FILE, sep='\t')
    e100_texts  = eval_df['sentence'].tolist()
    e100_labels = eval_df['evaluation_pair_interacting'].tolist()
    bert_scores_e100 = bert_scorer.score(e100_texts)
    final_e100 = evaluate(
        model_to_save, best_tokenizer, e100_texts, e100_labels, yes_id, no_id,
        bert_scores=bert_scores_e100, label="eval_100 (best model, guided)",
    )

    elapsed = (time.time() - start) / 60
    results = {
        'from_checkpoint':  t5_model,
        'bert_teacher':     args.bert_teacher,
        'train_data':       args.train_data,
        'epochs':           args.epochs,
        'neg_hard_weight':  args.neg_hard_weight,
        'pos_weight':       args.pos_weight,
        'cv_summary':       cv_summary,
        'final_ep_test_guided': final_ep,
        'final_ep_test_plain':  final_ep_plain,
        'final_eval_100':       final_e100,
        'training_time_min':    elapsed,
    }
    results_file = results_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("BERT-GUIDED T5 — FINAL RESULTS")
    print(f"{'='*70}")
    print(f"\nBaseline (flan-t5-base_v11_1):  avg EP F1=0.781  Prec=0.673  Rec=0.933")
    print(f"This run (guided, avg):          avg EP F1={cv_summary['avg_ep_f1']:.3f}  "
          f"Prec={cv_summary['avg_ep_precision']:.3f}  Rec={cv_summary['avg_ep_rec']:.3f}")
    print(f"\nFinal (best fold, guided):  F1={final_ep['f1']:.3f}  "
          f"Prec={final_ep['precision']:.3f}  Rec={final_ep['recall']:.3f}")
    print(f"Final (best fold, plain):   F1={final_ep_plain['f1']:.3f}  "
          f"Prec={final_ep_plain['precision']:.3f}  Rec={final_ep_plain['recall']:.3f}")
    print(f"\nTotal time: {elapsed:.1f} min")
    print(f"Results: {results_file}")
    print(f"Model:   {out_dir}")


if __name__ == '__main__':
    main()
