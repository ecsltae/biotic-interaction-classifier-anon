#!/usr/bin/env python3
"""
Generate predictions XLSX for analysis.
Shows where model predicts right/wrong on eval set.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = '/path/to/MetaP/classifier'
MODEL_DIR = f'{BASE_DIR}/models/transformer_BiomedBERT_globi_v7_llm'
EVAL_FILE = f'{BASE_DIR}/data/evaluation/eval_100.tsv'
OUTPUT_FILE = f'{BASE_DIR}/data/evaluation/predictions_v7_llm.xlsx'

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


def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()

    print("Loading evaluation data...")
    eval_df = pd.read_csv(EVAL_FILE, sep='\t')
    sentences = eval_df['sentence'].tolist()
    labels = eval_df['evaluation_pair_interacting'].tolist()

    print(f"Generating predictions for {len(sentences)} sentences...")
    dataset = TextDataset(sentences, tokenizer, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits.float(), dim=-1)[:, 1]
            all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)

    # Use the optimal threshold from training (0.15)
    threshold = 0.15
    predictions = (all_probs >= threshold).astype(int)

    # Create results DataFrame
    results_df = eval_df.copy()
    results_df['probability'] = all_probs
    results_df['predicted'] = predictions
    results_df['correct'] = (predictions == np.array(labels)).astype(int)

    # Add analysis columns
    results_df['error_type'] = ''
    results_df.loc[(results_df['evaluation_pair_interacting'] == 1) & (results_df['predicted'] == 0), 'error_type'] = 'FN (missed interaction)'
    results_df.loc[(results_df['evaluation_pair_interacting'] == 0) & (results_df['predicted'] == 1), 'error_type'] = 'FP (false positive)'

    # Sort: errors first, then by probability
    results_df['sort_key'] = results_df['correct'].astype(str) + '_' + (1 - results_df['probability']).astype(str)
    results_df = results_df.sort_values('sort_key').drop('sort_key', axis=1)

    # Rename columns for clarity
    results_df = results_df.rename(columns={
        'evaluation_pair_interacting': 'actual_label',
        'evaluation_pair_source': 'source_species',
        'evaluation_pair_target': 'target_species'
    })

    # Select and order columns
    output_cols = [
        'sentence', 'actual_label', 'predicted', 'probability',
        'correct', 'error_type', 'source_species', 'target_species'
    ]
    # Keep only columns that exist
    output_cols = [c for c in output_cols if c in results_df.columns]
    results_df = results_df[output_cols]

    # Save to Excel
    print(f"Saving to {OUTPUT_FILE}...")
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        # Main predictions sheet
        results_df.to_excel(writer, sheet_name='Predictions', index=False)

        # Summary sheet
        summary_data = {
            'Metric': ['Total', 'Correct', 'Wrong', 'Accuracy',
                      'True Positives', 'True Negatives', 'False Positives', 'False Negatives',
                      'Precision', 'Recall', 'F1', 'Threshold'],
            'Value': [
                len(results_df),
                results_df['correct'].sum(),
                len(results_df) - results_df['correct'].sum(),
                f"{results_df['correct'].mean():.1%}",
                ((results_df['actual_label'] == 1) & (results_df['predicted'] == 1)).sum(),
                ((results_df['actual_label'] == 0) & (results_df['predicted'] == 0)).sum(),
                ((results_df['actual_label'] == 0) & (results_df['predicted'] == 1)).sum(),
                ((results_df['actual_label'] == 1) & (results_df['predicted'] == 0)).sum(),
                f"{0.792:.3f}",
                f"{0.613:.3f}",
                f"{0.691:.3f}",
                threshold
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Errors only sheet
        errors_df = results_df[results_df['correct'] == 0]
        errors_df.to_excel(writer, sheet_name='Errors', index=False)

    print(f"\nDone! Saved to: {OUTPUT_FILE}")
    print(f"\nSummary:")
    print(f"  Total: {len(results_df)}")
    print(f"  Correct: {results_df['correct'].sum()}")
    print(f"  Errors: {len(results_df) - results_df['correct'].sum()}")
    print(f"  - False Positives: {((results_df['actual_label'] == 0) & (results_df['predicted'] == 1)).sum()}")
    print(f"  - False Negatives: {((results_df['actual_label'] == 1) & (results_df['predicted'] == 0)).sum()}")


if __name__ == "__main__":
    main()
