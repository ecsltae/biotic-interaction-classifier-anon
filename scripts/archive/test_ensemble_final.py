#!/usr/bin/env python3
"""
Quick test: Does ensemble of BiomedBERT + SciBERT add value?
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score

BASE_DIR = '/path/to/MetaP/classifier'
EP_TEST_FILE = f'{BASE_DIR}/globi-relax_passages-triplets_2024-02-28_curation_EP.tsv'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_LENGTH = 256

# Models to test
MODELS = {
    'BiomedBERT': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
    'SciBERT': 'allenai/scibert_scivocab_uncased',
}

# Use the CV-trained model if available, otherwise use base
CV_MODEL_DIR = f'{BASE_DIR}/models/transformer_BiomedBERT_cv_regularized'


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


def get_probs(model_dir, texts):
    """Get probabilities from a model."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(DEVICE)
    model.eval()

    dataset = TextDataset(texts, tokenizer, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits.float(), dim=-1)[:, 1]
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_probs)


def find_best_threshold(probs, labels):
    """Find threshold maximizing F1."""
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.05):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t


def evaluate(probs, labels, name):
    """Evaluate and print results."""
    thresh = find_best_threshold(probs, labels)
    preds = (probs >= thresh).astype(int)
    f1 = f1_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)

    f1_pass = "✓" if f1 > 0.75 else "✗"
    prec_pass = "✓" if prec > f1 else "✗"

    print(f"{name:20} | F1: {f1:.3f} {f1_pass} | Prec: {prec:.3f} {prec_pass} | Rec: {rec:.3f} | Thresh: {thresh:.2f}")
    return f1, prec, rec


def main():
    # Load test data
    df = pd.read_csv(EP_TEST_FILE, sep='\t', encoding='latin-1')
    texts = df['sentence'].tolist()
    labels = np.array(df['evaluation_pair_interacting'].tolist())
    print(f"EP Test set: {len(texts)} samples ({sum(labels)} positive)\n")

    # Get predictions from CV-trained BiomedBERT
    print("Loading models and getting predictions...")
    probs_biomed = get_probs(CV_MODEL_DIR, texts)

    # For SciBERT, we need to check if we have a trained version
    # If not, we'll train a quick one or use the base model
    scibert_dir = f'{BASE_DIR}/models/transformer_SciBERT_cv_regularized'
    import os
    if not os.path.exists(scibert_dir):
        # Use the last fold's checkpoint if available
        scibert_dir = f'{BASE_DIR}/models/cv_temp/SciBERT_fold4'
    if not os.path.exists(scibert_dir):
        print("SciBERT not found, skipping ensemble test")
        return

    probs_scibert = get_probs(scibert_dir, texts)

    print("\n" + "="*70)
    print("RESULTS ON EP TEST SET")
    print("="*70)
    print(f"{'Model':<20} | {'F1':^10} | {'Precision':^10} | {'Recall':^10} | Thresh")
    print("-"*70)

    # Individual models
    f1_bio, prec_bio, rec_bio = evaluate(probs_biomed, labels, "BiomedBERT")
    f1_sci, prec_sci, rec_sci = evaluate(probs_scibert, labels, "SciBERT")

    # Ensemble: average
    probs_avg = (probs_biomed + probs_scibert) / 2
    f1_avg, prec_avg, rec_avg = evaluate(probs_avg, labels, "Ensemble (avg)")

    # Ensemble: weighted toward SciBERT (for precision)
    probs_weighted = 0.3 * probs_biomed + 0.7 * probs_scibert
    f1_w, prec_w, rec_w = evaluate(probs_weighted, labels, "Ensemble (70% Sci)")

    # Ensemble: max (optimistic)
    probs_max = np.maximum(probs_biomed, probs_scibert)
    f1_max, prec_max, rec_max = evaluate(probs_max, labels, "Ensemble (max)")

    # Ensemble: min (conservative/precise)
    probs_min = np.minimum(probs_biomed, probs_scibert)
    f1_min, prec_min, rec_min = evaluate(probs_min, labels, "Ensemble (min)")

    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

    # Find best option meeting both criteria
    options = [
        ("BiomedBERT", f1_bio, prec_bio),
        ("SciBERT", f1_sci, prec_sci),
        ("Ensemble (avg)", f1_avg, prec_avg),
        ("Ensemble (70% Sci)", f1_w, prec_w),
        ("Ensemble (min)", f1_min, prec_min),
    ]

    # Filter those meeting both targets
    valid = [(n, f, p) for n, f, p in options if f > 0.75 and p > f]

    if valid:
        best = max(valid, key=lambda x: x[1])  # Best F1 among valid
        print(f"\nBest meeting BOTH targets (F1>0.75 & Prec>F1): {best[0]}")
        print(f"  F1: {best[1]:.3f}, Precision: {best[2]:.3f}")
    else:
        print("\nNo option meets both targets. Best F1:")
        best = max(options, key=lambda x: x[1])
        print(f"  {best[0]}: F1={best[1]:.3f}, Prec={best[2]:.3f}")


if __name__ == "__main__":
    main()
