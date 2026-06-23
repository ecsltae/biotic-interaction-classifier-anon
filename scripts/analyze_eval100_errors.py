#!/usr/bin/env python3
"""
Deep analysis of eval100 performance to understand why precision dropped
and find ways to improve.
"""

import sys
sys.path.insert(0, '/path/to/MetaP/classifier/src/models')

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Paths
BASE_DIR = '/path/to/MetaP/classifier'
EVAL_FILE = f'{BASE_DIR}/data/evaluation/eval_100.tsv'

print("="*80)
print("DEEP ANALYSIS OF EVAL100 ERRORS")
print("="*80)

# Load eval100
eval_df = pd.read_csv(EVAL_FILE, sep='\t')
sentences = eval_df['sentence'].tolist()
true_labels = eval_df['evaluation_pair_interacting'].tolist()

print(f"\nEval100 composition:")
print(f"  Total: {len(sentences)}")
print(f"  Positives (biotic): {sum(true_labels)}")
print(f"  Negatives: {len(true_labels) - sum(true_labels)}")

# Load both ensemble models and compare
print("\n" + "="*80)
print("LOADING MODELS...")
print("="*80)

from ensemble_classifier import OptimizedEnsembleClassifier, ENSEMBLE_CONFIG

# Baseline ensemble (original 20k models)
print("\n1. Loading BASELINE ensemble...")
baseline_ensemble = OptimizedEnsembleClassifier(optimize=True)
baseline_probs = baseline_ensemble.predict_proba(sentences)

# Get individual model predictions too
print("\n2. Getting individual model predictions...")

# BiomedBERT alone
biomedbert_probs = []
roberta_probs = []

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

class SimpleDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

device = 'cuda' if torch.cuda.is_available() else 'cpu'

for model_key in ['biomedbert', 'roberta']:
    model = baseline_ensemble.models[model_key]
    tokenizer = baseline_ensemble.tokenizers[model_key]

    dataset = SimpleDataset(sentences, tokenizer)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(batch_probs)

    if model_key == 'biomedbert':
        biomedbert_probs = np.array(probs)
    else:
        roberta_probs = np.array(probs)

print("\n" + "="*80)
print("INDIVIDUAL MODEL PERFORMANCE")
print("="*80)

def evaluate_at_threshold(probs, labels, threshold, name):
    preds = (probs >= threshold).astype(int)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)

    print(f"\n{name} (threshold={threshold:.3f}):")
    print(f"  Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")
    print(f"  Confusion: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

    return prec, rec, f1, preds

# Test different thresholds
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

print("\n--- BiomedBERT alone ---")
best_f1_bio = 0
best_thresh_bio = 0.5
for t in thresholds:
    p, r, f1, _ = evaluate_at_threshold(biomedbert_probs, true_labels, t, f"BiomedBERT")
    if f1 > best_f1_bio:
        best_f1_bio = f1
        best_thresh_bio = t

print("\n--- RoBERTa alone ---")
best_f1_rob = 0
best_thresh_rob = 0.5
for t in thresholds:
    p, r, f1, _ = evaluate_at_threshold(roberta_probs, true_labels, t, f"RoBERTa")
    if f1 > best_f1_rob:
        best_f1_rob = f1
        best_thresh_rob = t

print("\n--- Baseline Ensemble (65/35) ---")
best_f1_ens = 0
best_thresh_ens = 0.5
for t in thresholds:
    p, r, f1, _ = evaluate_at_threshold(baseline_probs[:, 1], true_labels, t, f"Ensemble")
    if f1 > best_f1_ens:
        best_f1_ens = f1
        best_thresh_ens = t

print("\n" + "="*80)
print("EXPLORING DIFFERENT ENSEMBLE WEIGHTS")
print("="*80)

# Try different weight combinations
weight_combinations = [
    (0.5, 0.5),   # Equal
    (0.6, 0.4),
    (0.65, 0.35), # Current
    (0.7, 0.3),
    (0.75, 0.25),
    (0.8, 0.2),
    (0.9, 0.1),   # Almost BiomedBERT only
    (1.0, 0.0),   # BiomedBERT only
    (0.0, 1.0),   # RoBERTa only
]

print("\nTesting weight combinations (BiomedBERT, RoBERTa):")
best_combo = None
best_combo_f1 = 0
best_combo_thresh = 0.5

for w_bio, w_rob in weight_combinations:
    # Weighted average
    if w_bio + w_rob > 0:
        combined_probs = (w_bio * biomedbert_probs + w_rob * roberta_probs) / (w_bio + w_rob)
    else:
        continue

    # Find best threshold for this combination
    for t in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        preds = (combined_probs >= t).astype(int)
        f1 = f1_score(true_labels, preds, zero_division=0)
        prec = precision_score(true_labels, preds, zero_division=0)
        rec = recall_score(true_labels, preds, zero_division=0)

        if f1 > best_combo_f1:
            best_combo_f1 = f1
            best_combo = (w_bio, w_rob)
            best_combo_thresh = t
            best_combo_prec = prec
            best_combo_rec = rec

print(f"\nBest combination found:")
print(f"  Weights: BiomedBERT={best_combo[0]}, RoBERTa={best_combo[1]}")
print(f"  Threshold: {best_combo_thresh}")
print(f"  F1: {best_combo_f1:.3f}")
print(f"  Precision: {best_combo_prec:.3f}")
print(f"  Recall: {best_combo_rec:.3f}")

# Apply best combination
w_bio, w_rob = best_combo
best_probs = (w_bio * biomedbert_probs + w_rob * roberta_probs) / (w_bio + w_rob)
best_preds = (best_probs >= best_combo_thresh).astype(int)

print("\n" + "="*80)
print("ERROR ANALYSIS")
print("="*80)

# Analyze errors with best configuration
cm = confusion_matrix(true_labels, best_preds)
print(f"\nConfusion Matrix (best config):")
print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

# Show false positives
print("\n--- FALSE POSITIVES (predicted positive but actually negative) ---")
fp_indices = [i for i in range(len(sentences)) if best_preds[i] == 1 and true_labels[i] == 0]
print(f"Count: {len(fp_indices)}")
for i, idx in enumerate(fp_indices[:10], 1):
    print(f"\n{i}. Prob={best_probs[idx]:.3f}")
    print(f"   {sentences[idx][:150]}...")

# Show false negatives
print("\n--- FALSE NEGATIVES (predicted negative but actually positive) ---")
fn_indices = [i for i in range(len(sentences)) if best_preds[i] == 0 and true_labels[i] == 1]
print(f"Count: {len(fn_indices)}")
for i, idx in enumerate(fn_indices[:10], 1):
    print(f"\n{i}. Prob={best_probs[idx]:.3f}")
    print(f"   {sentences[idx][:150]}...")

# Show true positives
print("\n--- TRUE POSITIVES (correctly identified biotic interactions) ---")
tp_indices = [i for i in range(len(sentences)) if best_preds[i] == 1 and true_labels[i] == 1]
print(f"Count: {len(tp_indices)}")
for i, idx in enumerate(tp_indices[:5], 1):
    print(f"\n{i}. Prob={best_probs[idx]:.3f}")
    print(f"   {sentences[idx][:150]}...")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
BEST CONFIGURATIONS ON EVAL100:

1. BiomedBERT alone:
   Best threshold: {best_thresh_bio}
   Best F1: {best_f1_bio:.3f}

2. RoBERTa alone:
   Best threshold: {best_thresh_rob}
   Best F1: {best_f1_rob:.3f}

3. Current Ensemble (65/35):
   Best threshold: {best_thresh_ens}
   Best F1: {best_f1_ens:.3f}

4. OPTIMIZED Ensemble:
   Weights: BiomedBERT={best_combo[0]}, RoBERTa={best_combo[1]}
   Threshold: {best_combo_thresh}
   F1: {best_combo_f1:.3f}
   Precision: {best_combo_prec:.3f}
   Recall: {best_combo_rec:.3f}
""")

print("="*80)
