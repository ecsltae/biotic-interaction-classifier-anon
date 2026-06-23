#!/usr/bin/env python3
"""
Check specific ensemble combinations to verify results
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_score, recall_score, f1_score

BASE_DIR = '/path/to/MetaP/classifier'
EVAL_FILE = f'{BASE_DIR}/data/evaluation/eval_100.tsv'
MODEL_DIR = f'{BASE_DIR}/models'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load eval100
eval_df = pd.read_csv(EVAL_FILE, sep='\t')
sentences = eval_df['sentence'].tolist()
labels = np.array(eval_df['evaluation_pair_interacting'].tolist())

class TextDataset(Dataset):
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

def get_probs(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    if device == 'cuda':
        model = model.half()

    dataset = TextDataset(sentences, tokenizer)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_probs = torch.softmax(outputs.logits.float(), dim=-1)[:, 1].cpu().numpy()
            probs.extend(batch_probs)

    return np.array(probs)

print("Loading models...")
probs_6k = get_probs(f'{MODEL_DIR}/transformer_BiomedBERT_model_6k_original')
probs_20k = get_probs(f'{MODEL_DIR}/transformer_BiomedBERT_model_enhanced_20k')
probs_biobert = get_probs(f'{MODEL_DIR}/transformer_biobert_model')
probs_roberta = get_probs(f'{MODEL_DIR}/transformer_roberta_model')
probs_quality_v2 = get_probs(f'{MODEL_DIR}/transformer_BiomedBERT_quality_v2')

print("\n" + "="*80)
print("DETAILED ENSEMBLE ANALYSIS")
print("="*80)

def test_ensemble(name, probs, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)

    print(f"\n{name}:")
    best_f1, best_t = 0, 0.5
    for t in thresholds:
        preds = (probs >= t).astype(int)
        p = precision_score(labels, preds, zero_division=0)
        r = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        print(f"  t={t:.2f}: F1={f1:.3f}, P={p:.3f}, R={r:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1

# Test individual models
print("\n--- SINGLE MODELS ---")
test_ensemble("BiomedBERT_6k_orig", probs_6k)
test_ensemble("RoBERTa", probs_roberta)

# Test the old best ensemble (3-model)
print("\n--- OLD BEST: 6k (40%) + 20k (30%) + BioBERT (30%) ---")
old_best = 0.40 * probs_6k + 0.30 * probs_20k + 0.30 * probs_biobert
test_ensemble("Old 3-model", old_best)

# Test 6k + RoBERTa combinations
print("\n--- 6k + RoBERTa COMBINATIONS ---")
for w1 in [0.5, 0.6, 0.7, 0.8, 0.9]:
    w2 = 1 - w1
    combined = w1 * probs_6k + w2 * probs_roberta
    test_ensemble(f"6k ({w1:.0%}) + RoBERTa ({w2:.0%})", combined)

# Test the NEW best ensemble (4-model)
print("\n--- NEW BEST: 6k + 20k + quality_v2 + BioBERT (25% each) ---")
new_best = 0.25 * probs_6k + 0.25 * probs_20k + 0.25 * probs_quality_v2 + 0.25 * probs_biobert
test_ensemble("New 4-model", new_best)

# Test 20k + quality_v2
print("\n--- 20k + quality_v2 COMBINATIONS ---")
for w1 in [0.3, 0.4, 0.5, 0.6, 0.7]:
    w2 = 1 - w1
    combined = w1 * probs_20k + w2 * probs_quality_v2
    test_ensemble(f"20k ({w1:.0%}) + quality_v2 ({w2:.0%})", combined)
