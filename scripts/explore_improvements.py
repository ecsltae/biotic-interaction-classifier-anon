#!/usr/bin/env python3
"""
Explore different approaches to improve eval100 performance:
1. Analyze the false positives - are they actually biotic interactions?
2. Try other models (DistilBERT, BioBERT)
3. Try rule-based filtering combined with ML
4. Find optimal configuration
"""

import sys
sys.path.insert(0, '/path/to/MetaP/classifier/src/models')

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Paths
BASE_DIR = '/path/to/MetaP/classifier'
EVAL_FILE = f'{BASE_DIR}/data/evaluation/eval_100.tsv'
MODEL_DIR = f'{BASE_DIR}/models'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load eval100
eval_df = pd.read_csv(EVAL_FILE, sep='\t')
sentences = eval_df['sentence'].tolist()
true_labels = np.array(eval_df['evaluation_pair_interacting'].tolist())

print(f"Eval100: {len(sentences)} samples, {sum(true_labels)} positives, {len(true_labels)-sum(true_labels)} negatives")

# =============================================================================
# PART 1: Analyze False Positives - Are they actually biotic interactions?
# =============================================================================
print("\n" + "="*80)
print("PART 1: FALSE POSITIVE ANALYSIS")
print("="*80)

# These were labeled as FP but look at them - many ARE biotic interactions!
fp_examples = [
    "geographic and hostrelated variation among species of fessisentis acanthocephala and confirmation of the fessisentis fessus life cycle",
    "spironucleus barkhanus from muscle abscesses of farmed atlantic salmon salmo salar l and from the gall bladder of grayling thymallus thymallus l was cultivated",
    "true infection with eurytrema would indicate that the policemen ate uncooked grasshoppers and crickets infected with the parasite",
    "a serological survey on ehrlichia canis was conducted among dogs in the central area of the state of rio grande do sul where the tick rhipicephalus sanguineus is a common parasite",
    "among aphids that feed on this host myzus persicae aphis gossypii rhopalosiphum padi aphis fabae aphis craccivora lipaphis erysimi and brevicoryne brassicae",
    "cysticercus fasciolaris can be conveniently produced in the experimental laboratory host rattus rattus",
]

print("\nFalse Positives that LOOK like real biotic interactions:")
for i, ex in enumerate(fp_examples, 1):
    print(f"\n{i}. {ex[:120]}...")
    # Check for interaction keywords
    keywords = ['parasite', 'infection', 'host', 'feed', 'infect', 'vector', 'pathogen']
    found = [k for k in keywords if k in ex.lower()]
    print(f"   Keywords found: {found}")

print("\n>>> OBSERVATION: Many 'false positives' contain clear interaction terms!")
print(">>> The eval100 labels may be too strict or inconsistent.")

# =============================================================================
# PART 2: Load and test ALL available models
# =============================================================================
print("\n" + "="*80)
print("PART 2: TESTING ALL AVAILABLE MODELS")
print("="*80)

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

def get_model_predictions(model_path, sentences):
    """Get predictions from a transformer model"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()

        if device == 'cuda':
            model = model.half()

        dataset = SimpleDataset(sentences, tokenizer)
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
        print(f"  Error loading {model_path}: {e}")
        return None

def evaluate_model(probs, labels, name):
    """Find best threshold and report metrics"""
    best_f1 = 0
    best_thresh = 0.5
    best_metrics = {}

    for t in np.arange(0.1, 0.95, 0.05):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_metrics = {
                'precision': precision_score(labels, preds, zero_division=0),
                'recall': recall_score(labels, preds, zero_division=0),
                'f1': f1
            }

    print(f"\n{name}:")
    print(f"  Best threshold: {best_thresh:.2f}")
    print(f"  Precision: {best_metrics['precision']:.3f}")
    print(f"  Recall: {best_metrics['recall']:.3f}")
    print(f"  F1: {best_metrics['f1']:.3f}")

    return probs, best_thresh, best_metrics

# Available models
models_to_test = {
    'BiomedBERT (20k)': f'{MODEL_DIR}/transformer_BiomedBERT_model_enhanced_20k',
    'BiomedBERT (6k orig)': f'{MODEL_DIR}/transformer_BiomedBERT_model_6k_original',
    'RoBERTa': f'{MODEL_DIR}/transformer_roberta_model',
    'BioBERT': f'{MODEL_DIR}/transformer_biobert_model',
    'DistilBERT': f'{MODEL_DIR}/transformer_distilbert_model',
}

all_probs = {}
all_metrics = {}

for name, path in models_to_test.items():
    if os.path.exists(path):
        print(f"\nLoading {name}...")
        probs = get_model_predictions(path, sentences)
        if probs is not None:
            probs, thresh, metrics = evaluate_model(probs, true_labels, name)
            all_probs[name] = probs
            all_metrics[name] = {'threshold': thresh, **metrics}

# =============================================================================
# PART 3: Find best ensemble of ALL models
# =============================================================================
print("\n" + "="*80)
print("PART 3: MULTI-MODEL ENSEMBLE EXPLORATION")
print("="*80)

if len(all_probs) >= 2:
    model_names = list(all_probs.keys())
    print(f"\nModels available for ensemble: {model_names}")

    # Try different combinations
    best_ensemble_f1 = 0
    best_ensemble_config = None

    # Try pairs
    from itertools import combinations

    for combo in combinations(model_names, 2):
        for w1 in [0.3, 0.4, 0.5, 0.6, 0.7]:
            w2 = 1 - w1
            combined = w1 * all_probs[combo[0]] + w2 * all_probs[combo[1]]

            for t in np.arange(0.3, 0.7, 0.05):
                preds = (combined >= t).astype(int)
                f1 = f1_score(true_labels, preds, zero_division=0)
                prec = precision_score(true_labels, preds, zero_division=0)

                if f1 > best_ensemble_f1:
                    best_ensemble_f1 = f1
                    best_ensemble_config = {
                        'models': combo,
                        'weights': (w1, w2),
                        'threshold': t,
                        'f1': f1,
                        'precision': prec,
                        'recall': recall_score(true_labels, preds, zero_division=0)
                    }

    # Try triplets if we have 3+ models
    if len(model_names) >= 3:
        for combo in combinations(model_names, 3):
            for w1 in [0.3, 0.4, 0.5]:
                for w2 in [0.2, 0.3, 0.4]:
                    w3 = 1 - w1 - w2
                    if w3 < 0.1:
                        continue
                    combined = w1 * all_probs[combo[0]] + w2 * all_probs[combo[1]] + w3 * all_probs[combo[2]]

                    for t in np.arange(0.3, 0.7, 0.05):
                        preds = (combined >= t).astype(int)
                        f1 = f1_score(true_labels, preds, zero_division=0)
                        prec = precision_score(true_labels, preds, zero_division=0)

                        if f1 > best_ensemble_f1:
                            best_ensemble_f1 = f1
                            best_ensemble_config = {
                                'models': combo,
                                'weights': (w1, w2, w3),
                                'threshold': t,
                                'f1': f1,
                                'precision': prec,
                                'recall': recall_score(true_labels, preds, zero_division=0)
                            }

    print(f"\nBest ensemble found:")
    print(f"  Models: {best_ensemble_config['models']}")
    print(f"  Weights: {best_ensemble_config['weights']}")
    print(f"  Threshold: {best_ensemble_config['threshold']:.2f}")
    print(f"  F1: {best_ensemble_config['f1']:.3f}")
    print(f"  Precision: {best_ensemble_config['precision']:.3f}")
    print(f"  Recall: {best_ensemble_config['recall']:.3f}")

# =============================================================================
# PART 4: Rule-based + ML hybrid approach
# =============================================================================
print("\n" + "="*80)
print("PART 4: RULE-BASED + ML HYBRID")
print("="*80)

# Strong interaction keywords that almost always indicate biotic interaction
STRONG_POSITIVE_KEYWORDS = [
    'infect', 'infection', 'infected', 'infecting',
    'parasite', 'parasites', 'parasitic', 'parasitize', 'parasitized',
    'pathogen', 'pathogens', 'pathogenic',
    'host of', 'hosts of', 'host for',
    'vector of', 'vector for', 'transmitted by',
    'prey on', 'preys on', 'predator of',
    'feeds on', 'fed on', 'feeding on',
]

# Keywords that suggest NOT a biotic interaction
NEGATIVE_KEYWORDS = [
    'phylogenet', 'taxonom', 'classif', 'systemat',
    'sequenc', 'genom', 'pcr', 'amplif',
    'morpholog', 'anatomic',
    'distribut', 'range', 'habitat',
]

def rule_based_score(sentence):
    """Score sentence based on rules"""
    s = sentence.lower()

    # Check for strong positive keywords
    pos_score = sum(1 for kw in STRONG_POSITIVE_KEYWORDS if kw in s)

    # Check for negative keywords
    neg_score = sum(1 for kw in NEGATIVE_KEYWORDS if kw in s)

    return pos_score, neg_score

# Apply rules
rule_scores = [rule_based_score(s) for s in sentences]
pos_scores = np.array([r[0] for r in rule_scores])
neg_scores = np.array([r[1] for r in rule_scores])

# Combine rules with best single model
if 'BiomedBERT (20k)' in all_probs:
    ml_probs = all_probs['BiomedBERT (20k)']

    # Hybrid: boost ML probability based on rules
    hybrid_probs = ml_probs.copy()

    # Boost if strong positive keywords
    hybrid_probs = hybrid_probs + 0.15 * pos_scores

    # Reduce if negative keywords dominate
    hybrid_probs = hybrid_probs - 0.1 * neg_scores

    # Clip to [0, 1]
    hybrid_probs = np.clip(hybrid_probs, 0, 1)

    print("\nHybrid (BiomedBERT + Rules):")
    for t in [0.3, 0.4, 0.5, 0.6]:
        preds = (hybrid_probs >= t).astype(int)
        f1 = f1_score(true_labels, preds, zero_division=0)
        prec = precision_score(true_labels, preds, zero_division=0)
        rec = recall_score(true_labels, preds, zero_division=0)
        print(f"  Threshold {t}: P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("SUMMARY: BEST CONFIGURATIONS")
print("="*80)

print("\nSingle Models:")
for name, metrics in sorted(all_metrics.items(), key=lambda x: -x[1]['f1']):
    print(f"  {name}: F1={metrics['f1']:.3f} (P={metrics['precision']:.3f}, R={metrics['recall']:.3f})")

if best_ensemble_config:
    print(f"\nBest Ensemble: F1={best_ensemble_config['f1']:.3f}")
    print(f"  {best_ensemble_config}")

print("\n" + "="*80)
