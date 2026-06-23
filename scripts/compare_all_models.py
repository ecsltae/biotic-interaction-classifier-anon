#!/usr/bin/env python3
"""
Compare ALL trained models on eval100 to find the best performer.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

BASE_DIR = '/path/to/MetaP/classifier'
EVAL_FILE = f'{BASE_DIR}/data/evaluation/eval_100.tsv'
MODEL_DIR = f'{BASE_DIR}/models'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load eval100
eval_df = pd.read_csv(EVAL_FILE, sep='\t')
sentences = eval_df['sentence'].tolist()
labels = np.array(eval_df['evaluation_pair_interacting'].tolist())

print(f"Eval100: {len(sentences)} samples, {sum(labels)} positives, {len(labels)-sum(labels)} negatives")


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


def get_model_probs(model_path, sentences):
    """Get probabilities from a transformer model"""
    try:
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
    except Exception as e:
        print(f"  Error: {e}")
        return None


def find_best_threshold(probs, labels):
    """Find best threshold and return metrics"""
    best_f1 = 0
    best_thresh = 0.5
    best_metrics = {}

    for t in np.arange(0.1, 0.9, 0.05):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_metrics = {
                'precision': precision_score(labels, preds, zero_division=0),
                'recall': recall_score(labels, preds, zero_division=0),
                'f1': f1,
                'accuracy': accuracy_score(labels, preds),
            }

    return best_thresh, best_metrics


# All models to test
models = {
    'BiomedBERT_6k_orig': f'{MODEL_DIR}/transformer_BiomedBERT_model_6k_original',
    'BiomedBERT_20k': f'{MODEL_DIR}/transformer_BiomedBERT_model_enhanced_20k',
    'BiomedBERT_optimal': f'{MODEL_DIR}/transformer_BiomedBERT_optimal',
    'BiomedBERT_quality_v2': f'{MODEL_DIR}/transformer_BiomedBERT_quality_v2',
    'BiomedBERT_quality_v3': f'{MODEL_DIR}/transformer_BiomedBERT_quality_v3',
    'BioBERT': f'{MODEL_DIR}/transformer_biobert_model',
    'RoBERTa': f'{MODEL_DIR}/transformer_roberta_model',
    'DistilBERT': f'{MODEL_DIR}/transformer_distilbert_model',
}

print("\n" + "="*80)
print("SINGLE MODEL RESULTS")
print("="*80)

all_probs = {}
results = []

for name, path in models.items():
    if os.path.exists(path):
        print(f"\nLoading {name}...")
        probs = get_model_probs(path, sentences)
        if probs is not None:
            all_probs[name] = probs
            thresh, metrics = find_best_threshold(probs, labels)
            results.append({
                'model': name,
                'threshold': thresh,
                **metrics
            })
            print(f"  Threshold: {thresh:.2f}")
            print(f"  F1: {metrics['f1']:.3f}, P: {metrics['precision']:.3f}, R: {metrics['recall']:.3f}")

# Sort by F1
results.sort(key=lambda x: -x['f1'])

print("\n" + "="*80)
print("SUMMARY (sorted by F1)")
print("="*80)

for r in results:
    print(f"  {r['model']:25s} F1={r['f1']:.3f} (P={r['precision']:.3f}, R={r['recall']:.3f}, t={r['threshold']:.2f})")

# Test ALL ensemble combinations systematically
print("\n" + "="*80)
print("EXHAUSTIVE ENSEMBLE SEARCH")
print("="*80)

from itertools import combinations

# Store all ensemble results
ensemble_results = []

# Test all 2-model combinations with various weights
print("\n--- 2-MODEL ENSEMBLES ---")
model_names = list(all_probs.keys())

weight_pairs = [(0.3, 0.7), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.65, 0.35), (0.7, 0.3), (0.8, 0.2)]

for m1, m2 in combinations(model_names, 2):
    for w1, w2 in weight_pairs:
        combined = w1 * all_probs[m1] + w2 * all_probs[m2]
        thresh, metrics = find_best_threshold(combined, labels)
        ensemble_results.append({
            'models': f"{m1} + {m2}",
            'weights': f"{w1:.0%}/{w2:.0%}",
            'threshold': thresh,
            **metrics
        })

# Test all 3-model combinations with various weights
print("\n--- 3-MODEL ENSEMBLES ---")
weight_triplets = [
    (0.33, 0.33, 0.34),
    (0.4, 0.3, 0.3),
    (0.5, 0.25, 0.25),
    (0.5, 0.3, 0.2),
    (0.6, 0.2, 0.2),
    (0.4, 0.4, 0.2),
]

for combo in combinations(model_names, 3):
    for weights in weight_triplets:
        combined = weights[0] * all_probs[combo[0]] + weights[1] * all_probs[combo[1]] + weights[2] * all_probs[combo[2]]
        thresh, metrics = find_best_threshold(combined, labels)
        ensemble_results.append({
            'models': f"{combo[0]} + {combo[1]} + {combo[2]}",
            'weights': f"{weights[0]:.0%}/{weights[1]:.0%}/{weights[2]:.0%}",
            'threshold': thresh,
            **metrics
        })

# Test 4-model ensembles
print("\n--- 4-MODEL ENSEMBLES ---")
weight_quads = [
    (0.25, 0.25, 0.25, 0.25),
    (0.4, 0.2, 0.2, 0.2),
    (0.3, 0.3, 0.2, 0.2),
]

for combo in combinations(model_names, 4):
    for weights in weight_quads:
        combined = sum(w * all_probs[m] for w, m in zip(weights, combo))
        thresh, metrics = find_best_threshold(combined, labels)
        ensemble_results.append({
            'models': f"{combo[0]} + {combo[1]} + {combo[2]} + {combo[3]}",
            'weights': f"{weights[0]:.0%}/{weights[1]:.0%}/{weights[2]:.0%}/{weights[3]:.0%}",
            'threshold': thresh,
            **metrics
        })

# Sort by F1
ensemble_results.sort(key=lambda x: -x['f1'])

print("\n" + "="*80)
print("TOP 20 ENSEMBLE CONFIGURATIONS (sorted by F1)")
print("="*80)

for i, r in enumerate(ensemble_results[:20], 1):
    print(f"\n{i:2d}. F1={r['f1']:.3f} (P={r['precision']:.3f}, R={r['recall']:.3f}, t={r['threshold']:.2f})")
    print(f"    Models: {r['models']}")
    print(f"    Weights: {r['weights']}")

# Also show top by precision (for those who want fewer false positives)
print("\n" + "="*80)
print("TOP 10 BY PRECISION (min F1 > 0.40)")
print("="*80)

precision_sorted = [r for r in ensemble_results if r['f1'] > 0.40]
precision_sorted.sort(key=lambda x: -x['precision'])

for i, r in enumerate(precision_sorted[:10], 1):
    print(f"\n{i:2d}. P={r['precision']:.3f} (F1={r['f1']:.3f}, R={r['recall']:.3f}, t={r['threshold']:.2f})")
    print(f"    Models: {r['models']}")
    print(f"    Weights: {r['weights']}")

# Show top by recall (for those who want to catch more interactions)
print("\n" + "="*80)
print("TOP 10 BY RECALL (min F1 > 0.40)")
print("="*80)

recall_sorted = [r for r in ensemble_results if r['f1'] > 0.40]
recall_sorted.sort(key=lambda x: -x['recall'])

for i, r in enumerate(recall_sorted[:10], 1):
    print(f"\n{i:2d}. R={r['recall']:.3f} (F1={r['f1']:.3f}, P={r['precision']:.3f}, t={r['threshold']:.2f})")
    print(f"    Models: {r['models']}")
    print(f"    Weights: {r['weights']}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

best = results[0]
print(f"\nBest single model: {best['model']}")
print(f"  F1={best['f1']:.3f}, Precision={best['precision']:.3f}, Recall={best['recall']:.3f}")
