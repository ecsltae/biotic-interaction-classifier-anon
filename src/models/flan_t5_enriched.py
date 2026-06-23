#!/usr/bin/env python3
"""
FLAN-T5 Enriched Classifier for Biotic Interaction Detection
=============================================================

Extends flan_t5_classifier.py with:

1. **Enriched input prompt** — injects NER-validated species names and GloBI-matched
   interaction terms alongside the raw sentence.  At training time these come from the
   CSV columns (source_species, target_species, interaction_type / scan_globi_terms).
   At inference time (port 8002) they come from Layers 1+2 of the enriched pipeline.

2. **Structured output (--structured flag)** — instead of generating "yes"/"no" the model
   outputs "label: yes | category: INFECTION | host: Homo sapiens | other: SARS-CoV-2".
   Training targets are generated automatically from the existing CSV columns
   (label, interaction_type, source_species, target_species) — no new annotation needed.

   Default (no flag) keeps binary yes/no output for backward compatibility.

Usage:
    # Enriched prompt, binary output (fair comparison with flan_t5_classifier.py)
    python flan_t5_enriched.py --train classifier/data/training/training_data_v12.csv

    # Enriched prompt + structured output
    python flan_t5_enriched.py --train classifier/data/training/training_data_v12.csv --structured

    # Zero-shot (no fine-tuning, enriched prompt)
    python flan_t5_enriched.py --zero-shot
"""

import os
import sys
import json
import time
import warnings
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

# ── path setup ─────────────────────────────────────────────────────────────────
BASE_DIR = Path('/path/to/MetaP/classifier')
sys.path.insert(0, str(BASE_DIR / 'src'))

from data.interaction_taxonomy import scan_globi_terms, GLOBI_TYPE_TO_CATEGORY

# ── paths ──────────────────────────────────────────────────────────────────────
TRAIN_FILE    = BASE_DIR / 'data/training/training_data_v12.csv'
EVAL100_FILE  = BASE_DIR / 'data/evaluation/eval_100.tsv'
EP_TEST_FILE  = BASE_DIR / 'globi-relax_passages-triplets_2024-02-28_curation_EP.tsv'
MODEL_OUT_DIR = BASE_DIR / 'models/flan_t5_enriched'
RESULTS_DIR   = BASE_DIR / 'results/flan_t5_enriched'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# ── config ─────────────────────────────────────────────────────────────────────
DEFAULT_MODEL   = 'google/flan-t5-large'
N_FOLDS         = 5
EPOCHS          = 5
BATCH_SIZE      = 16
MAX_INPUT_LEN   = 320   # slightly longer than original (256) to fit species/terms context
MAX_TARGET_LEN  = 64    # 4 for binary, 64 for structured
LEARNING_RATE   = 1e-4
GOLD_WEIGHT     = 5

YES_TOKEN = "yes"
NO_TOKEN  = "no"

# ── prompt templates ────────────────────────────────────────────────────────────

ENRICHED_PROMPT = (
    "Does this sentence describe a biotic interaction between two organisms?\n"
    "Sentence: {sentence}\n"
    "Species detected: {species}\n"
    "Interaction terms found: {terms}\n"
    "Answer:"
)

def build_prompt(sentence: str, species_str: str, terms_str: str) -> str:
    return ENRICHED_PROMPT.format(
        sentence=sentence,
        species=species_str or "none",
        terms=terms_str or "none",
    )


# ── structured target helpers ───────────────────────────────────────────────────

def make_structured_target(label: int, interaction_type: str,
                           source_species: str, target_species: str) -> str:
    """
    Build the structured generation target from existing CSV columns.

    Positive: "label: yes | category: INFECTION | host: Homo sapiens | other: SARS-CoV-2"
    Negative: "label: no"
    """
    if label == 0:
        return "label: no"
    cat = GLOBI_TYPE_TO_CATEGORY.get(str(interaction_type), 'GENERIC')
    src = str(source_species) if pd.notna(source_species) else "unknown"
    tgt = str(target_species) if pd.notna(target_species) else "unknown"
    return f"label: yes | category: {cat} | host: {src} | other: {tgt}"


def parse_structured_output(generated: str) -> dict:
    """
    Parse a structured target string back into a dict.

    Returns: {label, category, host, other, raw}
    Handles partial/malformed output gracefully.
    """
    result = {"label": "no", "category": None, "host": None, "other": None, "raw": generated}
    if not generated:
        return result
    parts = [p.strip() for p in generated.split("|")]
    for part in parts:
        if ":" not in part:
            continue
        key, _, val = part.partition(":")
        key = key.strip().lower()
        val = val.strip()
        if key == "label":
            result["label"] = val
        elif key == "category":
            result["category"] = val
        elif key == "host":
            result["host"] = val
        elif key == "other":
            result["other"] = val
    return result


# ── species / terms context from CSV ───────────────────────────────────────────

def build_species_str(source: str, target: str) -> str:
    """Combine source + target species columns into a comma-separated string."""
    parts = []
    if pd.notna(source) and str(source).strip() not in ("", "nan", "unknown"):
        parts.append(str(source).strip())
    if pd.notna(target) and str(target).strip() not in ("", "nan", "unknown"):
        parts.append(str(target).strip())
    return ", ".join(parts) if parts else "none"


def build_terms_str(text: str) -> str:
    """Run the GloBI 591-term scanner and return matched terms as a string."""
    matched = scan_globi_terms(text)
    return ", ".join(matched) if matched else "none"


# ── dataset ─────────────────────────────────────────────────────────────────────

class BioticT5EnrichedDataset(Dataset):
    """
    Seq2seq dataset with enriched prompts.

    targets: "yes"/"no" for binary mode, structured string for --structured mode.
    prompts and targets are pre-built at construction time.
    """

    def __init__(
        self,
        prompts: List[str],
        targets: List[str],
        tokenizer,
        max_input_len: int = MAX_INPUT_LEN,
        max_target_len: int = MAX_TARGET_LEN,
    ):
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.targets = targets
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        inp = self.tokenizer(
            self.prompts[idx],
            max_length=self.max_input_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        tgt = self.tokenizer(
            self.targets[idx],
            max_length=self.max_target_len,
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
        }


# ── inference helpers ───────────────────────────────────────────────────────────

def get_yes_no_ids(tokenizer) -> Tuple[int, int]:
    yes_id = tokenizer.encode(YES_TOKEN, add_special_tokens=False)[0]
    no_id  = tokenizer.encode(NO_TOKEN,  add_special_tokens=False)[0]
    return yes_id, no_id


def score_prompts(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int = 32,
    yes_id: int = None,
    no_id: int = None,
) -> np.ndarray:
    """
    Score pre-built prompts using forced decoding (log P("yes") vs log P("no")).
    Works for both binary and structured modes — always reads yes/no first token.
    """
    if yes_id is None or no_id is None:
        yes_id, no_id = get_yes_no_ids(tokenizer)

    model.eval()
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

            bos = torch.full(
                (len(batch), 1),
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


def score_sentences_enriched(
    model,
    tokenizer,
    texts: List[str],
    species_strs: List[str],
    terms_strs: List[str],
    batch_size: int = 32,
    yes_id: int = None,
    no_id: int = None,
) -> np.ndarray:
    """Convenience wrapper: build enriched prompts then score them."""
    prompts = [build_prompt(t, s, tr) for t, s, tr in zip(texts, species_strs, terms_strs)]
    return score_prompts(model, tokenizer, prompts, batch_size, yes_id, no_id)


def find_best_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    precision_mode: bool = False,
    min_recall: float = 0.65,
) -> Tuple[float, float]:
    """
    Grid-search threshold.

    precision_mode=False (default): maximise F1.
    precision_mode=True: maximise Precision subject to Recall >= min_recall.
    Returns (best_score, best_threshold).
    """
    best_score, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (probs >= t).astype(int)
        if preds.sum() == 0:
            continue
        if precision_mode:
            rec  = recall_score(labels, preds, zero_division=0)
            if rec < min_recall:
                continue
            score = precision_score(labels, preds, zero_division=0)
        else:
            score = f1_score(labels, preds, zero_division=0)
        if score > best_score:
            best_score, best_t = score, t
    return best_score, best_t


def evaluate(
    model, tokenizer,
    texts: List[str],
    labels: List[int],
    species_strs: List[str],
    terms_strs: List[str],
    yes_id: int,
    no_id: int,
    label_str: str = "",
    precision_mode: bool = False,
    min_recall: float = 0.65,
) -> dict:
    probs = score_sentences_enriched(
        model, tokenizer, texts, species_strs, terms_strs, yes_id=yes_id, no_id=no_id
    )
    labels_arr = np.array(labels)
    best_f1, best_t = find_best_threshold(probs, labels_arr,
                                          precision_mode=precision_mode,
                                          min_recall=min_recall)
    preds = (probs >= best_t).astype(int)
    prec = precision_score(labels_arr, preds, zero_division=0)
    rec  = recall_score(labels_arr, preds, zero_division=0)
    cm   = confusion_matrix(labels_arr, preds)
    if label_str:
        print(f"  {label_str}: F1={best_f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  (thresh={best_t:.2f})")
        print(f"    CM: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
    return {'f1': best_f1, 'precision': prec, 'recall': rec, 'threshold': best_t,
            'confusion_matrix': cm.tolist(), 'probabilities': probs.tolist()}


# ── data loading ────────────────────────────────────────────────────────────────

def _enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add enriched_prompt and binary_target columns; build GloBI terms at load time."""
    print("  Building GloBI term context for training data...")
    if 'source_species' not in df.columns:
        df['source_species'] = ''
    if 'target_species' not in df.columns:
        df['target_species'] = ''
    if 'interaction_type' not in df.columns:
        df['interaction_type'] = ''

    df['species_str'] = df.apply(
        lambda r: build_species_str(r['source_species'], r['target_species']), axis=1
    )
    df['terms_str'] = df['text'].apply(build_terms_str)
    df['prompt'] = df.apply(
        lambda r: build_prompt(r['text'], r['species_str'], r['terms_str']), axis=1
    )
    return df


def load_data(train_file: str = None, structured: bool = False) -> pd.DataFrame:
    """Load training data + eval_100 gold (oversampled), add enrichment columns."""
    path = Path(train_file) if train_file else TRAIN_FILE
    df = pd.read_csv(path).dropna(subset=['text', 'label'])
    df = df[['text', 'label'] + [c for c in ['interaction_type', 'source_species', 'target_species', 'source'] if c in df.columns]]
    df['source_tag'] = 'train'
    print(f"  Training data: {len(df)} ({sum(df['label']==1)} pos)")

    # Eval-100 gold (no structured targets — it has no species columns)
    eval_df = pd.read_csv(EVAL100_FILE, sep='\t')[['sentence', 'evaluation_pair_interacting']]
    eval_df.columns = ['text', 'label']
    eval_df['source_tag'] = 'gold'
    gold = pd.concat([eval_df] * GOLD_WEIGHT, ignore_index=True)
    print(f"  eval_100 gold (×{GOLD_WEIGHT}): {len(gold)}")

    combined = pd.concat([df, gold], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Combined: {len(combined)} ({sum(combined['label']==1)} pos)")

    combined = _enrich_df(combined)

    if structured:
        combined['target'] = combined.apply(
            lambda r: make_structured_target(
                int(r['label']),
                r.get('interaction_type', ''),
                r.get('source_species', ''),
                r.get('target_species', ''),
            ), axis=1
        )
    else:
        combined['target'] = combined['label'].apply(lambda l: 'yes' if l == 1 else 'no')

    return combined


def load_ep_test() -> Tuple[List[str], List[int], List[str], List[str]]:
    """Load EP test set and pre-compute enriched context (GloBI scan, no species available)."""
    df = pd.read_csv(EP_TEST_FILE, sep='\t', encoding='latin-1')
    texts  = df['sentence'].tolist()
    labels = df['evaluation_pair_interacting'].tolist()
    # EP test has no species columns — use GloBI scan only
    terms_strs   = [build_terms_str(t) for t in texts]
    species_strs = ["none"] * len(texts)
    print(f"EP test set: {len(texts)} ({sum(labels)} pos)")
    return texts, labels, species_strs, terms_strs


# ── zero-shot evaluation ────────────────────────────────────────────────────────

def zero_shot_eval(model_name: str) -> dict:
    print(f"\n{'='*70}")
    print(f"ZERO-SHOT ENRICHED EVALUATION: {model_name}")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()

    yes_id, no_id = get_yes_no_ids(tokenizer)

    ep_texts, ep_labels, ep_species, ep_terms = load_ep_test()
    ep_metrics = evaluate(model, tokenizer, ep_texts, ep_labels,
                          ep_species, ep_terms, yes_id, no_id, "EP test")

    eval_df = pd.read_csv(EVAL100_FILE, sep='\t')
    e100_texts  = eval_df['sentence'].tolist()
    e100_labels = eval_df['evaluation_pair_interacting'].tolist()
    e100_terms  = [build_terms_str(t) for t in e100_texts]
    e100_species = ["none"] * len(e100_texts)
    e100_metrics = evaluate(model, tokenizer, e100_texts, e100_labels,
                            e100_species, e100_terms, yes_id, no_id, "eval_100")

    return {'ep_test': ep_metrics, 'eval_100': e100_metrics}


# ── training ────────────────────────────────────────────────────────────────────

def train_cv(
    model_name: str,
    data_df: pd.DataFrame,
    ep_texts: List[str],
    ep_labels: List[int],
    ep_species: List[str],
    ep_terms: List[str],
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    structured: bool = False,
) -> Tuple[dict, object, dict]:

    max_tgt = 64 if structured else 4
    print(f"\n{'='*70}")
    print(f"TRAINING {model_name} WITH {N_FOLDS}-FOLD CV")
    print(f"  Mode: {'structured' if structured else 'binary (yes/no)'}")
    print(f"  max_target_len={max_tgt}")
    print(f"{'='*70}")

    prompts = data_df['prompt'].tolist()
    targets = data_df['target'].tolist()
    labels  = data_df['label'].tolist()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_results = []
    best_model_state = None
    best_tokenizer = None
    best_ep_f1 = 0.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(prompts, labels)):
        print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")
        tr_prompts  = [prompts[i] for i in train_idx]
        tr_targets  = [targets[i] for i in train_idx]
        va_prompts  = [prompts[i] for i in val_idx]
        va_labels_f = [labels[i]  for i in val_idx]
        # For val evaluation, extract texts/species/terms back from val rows
        va_texts   = data_df['text'].iloc[val_idx].tolist()
        va_species = data_df['species_str'].iloc[val_idx].tolist()
        va_terms   = data_df['terms_str'].iloc[val_idx].tolist()
        print(f"Train: {len(tr_prompts)} | Val: {len(va_prompts)}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.to(DEVICE)

        yes_id, no_id = get_yes_no_ids(tokenizer)

        train_ds = BioticT5EnrichedDataset(tr_prompts, tr_targets, tokenizer,
                                            max_target_len=max_tgt)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)

        try:
            from transformers import Adafactor
            optimizer = Adafactor(
                model.parameters(), lr=LEARNING_RATE,
                scale_parameter=False, relative_step=False, warmup_init=False,
            )
        except ImportError:
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

        best_val_f1 = 0.0
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                out = model(
                    input_ids=batch['input_ids'].to(DEVICE),
                    attention_mask=batch['attention_mask'].to(DEVICE),
                    labels=batch['labels'].to(DEVICE),
                )
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += out.loss.item()

            avg_loss = total_loss / len(train_loader)
            val_m = evaluate(model, tokenizer, va_texts, va_labels_f,
                             va_species, va_terms, yes_id, no_id)
            print(f"  Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | "
                  f"val F1={val_m['f1']:.3f} Prec={val_m['precision']:.3f}")
            best_val_f1 = max(best_val_f1, val_m['f1'])

        ep_m = evaluate(model, tokenizer, ep_texts, ep_labels, ep_species, ep_terms,
                        yes_id, no_id, f"Fold {fold+1} EP test",
                        precision_mode=args.precision_mode,
                        min_recall=args.min_recall)

        fold_results.append({
            'fold': fold + 1, 'val_f1': best_val_f1,
            'ep_f1': ep_m['f1'], 'ep_precision': ep_m['precision'], 'ep_recall': ep_m['recall'],
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
    print(f"CV Summary ({model_name})")
    print(f"  Avg EP F1:        {avg_ep_f1:.3f} ± {std_ep_f1:.3f}")
    print(f"  Avg EP Precision: {avg_ep_prec:.3f}")
    print(f"  Avg EP Recall:    {avg_ep_rec:.3f}")
    print(f"  Best Fold EP F1:  {best_ep_f1:.3f}")
    print(f"  Baseline (FLAN-T5 simple, v12): avg=0.780, best=0.800")

    summary = {
        'avg_ep_f1': avg_ep_f1, 'avg_ep_precision': avg_ep_prec,
        'avg_ep_recall': avg_ep_rec, 'best_ep_f1': best_ep_f1,
        'fold_results': fold_results,
    }
    return best_model_state, best_tokenizer, summary


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='FLAN-T5 enriched biotic interaction classifier')
    parser.add_argument('--model',       default=DEFAULT_MODEL, help='HuggingFace model ID')
    parser.add_argument('--train',       default=str(TRAIN_FILE), dest='train_data', help='Training CSV')
    parser.add_argument('--epochs',      type=int, default=EPOCHS)
    parser.add_argument('--batch-size',  type=int, default=BATCH_SIZE)
    parser.add_argument('--zero-shot',   action='store_true', help='Eval only, no fine-tuning')
    parser.add_argument('--structured',  action='store_true',
                        help='Use structured output (label | category | host | other)')
    parser.add_argument('--precision-mode', action='store_true',
                        help='Optimise threshold for Precision@Recall>=0.65 instead of max-F1. '
                             'Use when precision > recall is preferred.')
    parser.add_argument('--min-recall',  type=float, default=0.65,
                        help='Minimum recall floor when --precision-mode is set (default 0.65)')
    parser.add_argument('--output-dir',  default=str(MODEL_OUT_DIR))
    parser.add_argument('--results-dir', default=str(RESULTS_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()

    if args.precision_mode:
        print(f"[PRECISION MODE] Threshold optimised for Precision@Recall≥{args.min_recall:.2f} "
              f"(instead of max-F1)")

    zs_results = zero_shot_eval(args.model)
    print(f"\nZero-shot EP test:  F1={zs_results['ep_test']['f1']:.3f}  "
          f"Prec={zs_results['ep_test']['precision']:.3f}")

    if args.zero_shot:
        print("\n[zero-shot mode] Skipping fine-tuning.")
        with open(results_dir / 'zero_shot_results.json', 'w') as f:
            json.dump(zs_results, f, indent=2)
        return

    # ── Fine-tuning ────────────────────────────────────────────────────────────
    print(f"\nLoading training data: {args.train_data}")
    data_df = load_data(args.train_data, structured=args.structured)
    ep_texts, ep_labels, ep_species, ep_terms = load_ep_test()

    best_state, best_tokenizer, cv_summary = train_cv(
        args.model, data_df, ep_texts, ep_labels, ep_species, ep_terms,
        epochs=args.epochs, batch_size=args.batch_size, structured=args.structured,
    )

    # ── Save best model ────────────────────────────────────────────────────────
    print(f"\nSaving best model to {out_dir}...")
    model_to_save = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model_to_save.load_state_dict(best_state)
    model_to_save.save_pretrained(out_dir)
    best_tokenizer.save_pretrained(out_dir)
    # Save config so API knows which mode to use
    with open(out_dir / 'enriched_config.json', 'w') as f:
        json.dump({'structured': args.structured, 'model': args.model}, f)
    print("  Saved.")

    # ── Final eval on best model ───────────────────────────────────────────────
    print("\nFinal evaluation (best model)...")
    model_to_save.to(DEVICE)
    yes_id, no_id = get_yes_no_ids(best_tokenizer)

    final_ep = evaluate(model_to_save, best_tokenizer, ep_texts, ep_labels,
                        ep_species, ep_terms, yes_id, no_id, "EP test (best model)",
                        precision_mode=args.precision_mode, min_recall=args.min_recall)

    eval_df = pd.read_csv(EVAL100_FILE, sep='\t')
    e100_texts  = eval_df['sentence'].tolist()
    e100_labels = eval_df['evaluation_pair_interacting'].tolist()
    e100_terms   = [build_terms_str(t) for t in e100_texts]
    e100_species = ["none"] * len(e100_texts)
    final_e100 = evaluate(model_to_save, best_tokenizer, e100_texts, e100_labels,
                          e100_species, e100_terms, yes_id, no_id, "eval_100 (best model)",
                          precision_mode=args.precision_mode, min_recall=args.min_recall)

    # ── Results ────────────────────────────────────────────────────────────────
    elapsed = (time.time() - start) / 60
    results = {
        'model': args.model,
        'mode': 'structured' if args.structured else 'binary_enriched',
        'train_data': args.train_data,
        'epochs': args.epochs,
        'enriched_prompt': True,
        'zero_shot': zs_results,
        'cv_summary': cv_summary,
        'final_ep_test': final_ep,
        'final_eval_100': final_e100,
        'training_time_min': elapsed,
        'baseline_flan_t5_simple': {
            'avg_ep_f1': 0.780, 'best_ep_f1': 0.800, 'note': 'flan_t5_v12, unenriched prompt'
        },
        'baseline_biomedbert': {
            'ep_f1': 0.788, 'note': 'BiomedBERT v7 LLM-validated, regularized'
        },
    }
    results_file = results_dir / 'flan_t5_enriched_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("FLAN-T5 ENRICHED — FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Mode:              {'structured' if args.structured else 'binary (enriched prompt)'}")
    print(f"  Zero-shot EP F1:   {zs_results['ep_test']['f1']:.3f}  "
          f"Prec={zs_results['ep_test']['precision']:.3f}")
    print(f"  Fine-tuned EP F1:  {final_ep['f1']:.3f}  "
          f"Prec={final_ep['precision']:.3f}  Rec={final_ep['recall']:.3f}")
    print(f"  Fine-tuned e100:   {final_e100['f1']:.3f}  "
          f"Prec={final_e100['precision']:.3f}")
    print(f"\n  Baselines:")
    print(f"    FLAN-T5 simple (v12): avg=0.780, best=0.800")
    print(f"    BiomedBERT v7:        F1=0.788")
    delta = final_ep['f1'] - 0.780
    print(f"\n  ΔF1 vs FLAN-T5 simple: {delta:+.3f}")
    print(f"\n  Total time: {elapsed:.1f} min")
    print(f"  Results: {results_file}")
    print(f"  Model:   {out_dir}")


if __name__ == '__main__':
    main()
