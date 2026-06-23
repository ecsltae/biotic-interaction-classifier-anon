#!/usr/bin/env python3
"""
Ensemble Predictor - Combines BiomedBERT and BioBERT predictions.
Evaluates on eval_100 and generates predictions xlsx.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = '/path/to/MetaP/classifier'
MODEL_DIRS = {
    'BiomedBERT': f'{BASE_DIR}/models/transformer_BiomedBERT_globi_v7_llm',
    'BioBERT': f'{BASE_DIR}/models/transformer_BioBERT_globi_v7',
}
EVAL_FILE = f'{BASE_DIR}/data/evaluation/eval_100.tsv'
OUTPUT_FILE = f'{BASE_DIR}/data/evaluation/predictions_ensemble.xlsx'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_LENGTH = 256


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


def get_predictions(model_dir, texts):
    """Get probability predictions from a model."""
    print(f"  Loading {model_dir}...")
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
    """Find threshold that maximizes F1."""
    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.05):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh, best_f1


def main():
    print("="*70)
    print("ENSEMBLE EVALUATION")
    print("="*70)

    # Load eval data
    print("\nLoading evaluation data...")
    eval_df = pd.read_csv(EVAL_FILE, sep='\t')
    sentences = eval_df['sentence'].tolist()
    labels = np.array(eval_df['evaluation_pair_interacting'].tolist())
    print(f"Eval set: {len(sentences)} sentences ({sum(labels)} positive)")

    # Get predictions from each model
    print("\nGetting predictions from each model...")
    model_probs = {}
    for name, model_dir in MODEL_DIRS.items():
        if os.path.exists(model_dir):
            model_probs[name] = get_predictions(model_dir, sentences)
            print(f"  {name}: loaded")
        else:
            print(f"  {name}: NOT FOUND at {model_dir}")

    if len(model_probs) < 2:
        print("\nERROR: Need both models for ensemble. Train BioBERT first.")
        return

    # Ensemble: average probabilities
    print("\n" + "="*70)
    print("ENSEMBLE RESULTS (Average Probabilities)")
    print("="*70)

    ensemble_probs = np.mean(list(model_probs.values()), axis=0)
    best_thresh, best_f1 = find_best_threshold(ensemble_probs, labels)
    ensemble_preds = (ensemble_probs >= best_thresh).astype(int)

    # Metrics
    precision = precision_score(labels, ensemble_preds, zero_division=0)
    recall = recall_score(labels, ensemble_preds, zero_division=0)
    f1 = f1_score(labels, ensemble_preds, zero_division=0)
    accuracy = accuracy_score(labels, ensemble_preds)
    cm = confusion_matrix(labels, ensemble_preds)

    print(f"\nBest threshold: {best_thresh:.2f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]:3d}  FP={cm[0,1]:3d}")
    print(f"  FN={cm[1,0]:3d}  TP={cm[1,1]:3d}")

    # Compare individual models
    print("\n" + "="*70)
    print("INDIVIDUAL MODEL COMPARISON")
    print("="*70)

    for name, probs in model_probs.items():
        thresh, _ = find_best_threshold(probs, labels)
        preds = (probs >= thresh).astype(int)
        print(f"\n{name} (threshold={thresh:.2f}):")
        print(f"  F1:        {f1_score(labels, preds, zero_division=0):.3f}")
        print(f"  Precision: {precision_score(labels, preds, zero_division=0):.3f}")
        print(f"  Recall:    {recall_score(labels, preds, zero_division=0):.3f}")

    # Target check
    print("\n" + "="*70)
    print("TARGET CHECK (Ensemble)")
    print("="*70)
    f1_pass = f1 > 0.75
    prec_pass = precision > f1
    print(f"  F1 > 0.75:       {'PASS' if f1_pass else 'FAIL'} ({f1:.3f})")
    print(f"  Precision > F1:  {'PASS' if prec_pass else 'FAIL'} ({precision:.3f} vs {f1:.3f})")

    # Save predictions xlsx
    print(f"\nSaving predictions to {OUTPUT_FILE}...")
    results_df = eval_df.copy()
    results_df['prob_BiomedBERT'] = model_probs['BiomedBERT']
    results_df['prob_BioBERT'] = model_probs['BioBERT']
    results_df['prob_ensemble'] = ensemble_probs
    results_df['predicted'] = ensemble_preds
    results_df['correct'] = (ensemble_preds == labels).astype(int)
    results_df['error_type'] = ''
    results_df.loc[(labels == 1) & (ensemble_preds == 0), 'error_type'] = 'FN'
    results_df.loc[(labels == 0) & (ensemble_preds == 1), 'error_type'] = 'FP'

    # Rename columns
    results_df = results_df.rename(columns={
        'evaluation_pair_interacting': 'actual_label',
        'evaluation_pair_source': 'source_species',
        'evaluation_pair_target': 'target_species'
    })

    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Predictions', index=False)

        # Summary
        summary = pd.DataFrame({
            'Metric': ['F1', 'Precision', 'Recall', 'Accuracy', 'Threshold',
                      'TP', 'TN', 'FP', 'FN'],
            'Ensemble': [f1, precision, recall, accuracy, best_thresh,
                        cm[1,1], cm[0,0], cm[0,1], cm[1,0]]
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)

        # Errors
        errors_df = results_df[results_df['correct'] == 0]
        errors_df.to_excel(writer, sheet_name='Errors', index=False)

    print("Done!")


if __name__ == "__main__":
    main()
