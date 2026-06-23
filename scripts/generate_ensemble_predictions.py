#!/usr/bin/env python3
"""
Generate predictions CSV using the best ensemble model (baseline).
Creates predictions_Ensemble_Best.csv in the same format as predictions_BiomedBERT.csv
"""

import sys
sys.path.insert(0, '/path/to/MetaP/classifier/src/models')

import pandas as pd
from ensemble_classifier import OptimizedEnsembleClassifier

# Paths
BASE_DIR = '/path/to/MetaP/classifier'
INPUT_FILE = f'{BASE_DIR}/results/predictions/predictions_ALL_MODELS_comparison.csv'
OUTPUT_FILE = f'{BASE_DIR}/results/predictions/predictions_Ensemble_Best.csv'

# Best threshold from baseline (F1-optimized)
BEST_THRESHOLD = 0.389886474609375

print("="*70)
print("GENERATING ENSEMBLE PREDICTIONS")
print("="*70)

# Load the comparison data to get sentences and true labels
print("\nLoading data...")
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} samples")

sentences = df['sentence'].tolist()
true_labels = df['true_label'].tolist()

# Load ensemble
print("\nLoading ensemble classifier...")
ensemble = OptimizedEnsembleClassifier(optimize=True)

# Get predictions
print(f"\nGenerating predictions with threshold={BEST_THRESHOLD}...")
probs = ensemble.predict_proba(sentences)
predictions = (probs[:, 1] >= BEST_THRESHOLD).astype(int)

# Create output dataframe
print("\nCreating output CSV...")
output_df = pd.DataFrame({
    'sentence': sentences,
    'True_label': true_labels,
    'Ensemble_prediction': predictions,
    'True_sentiment': ['positive' if l == 1 else 'negative' for l in true_labels],
    'Ensemble_sentiment': ['positive' if p == 1 else 'negative' for p in predictions],
    'Ensemble_probability': probs[:, 1],
})

# Save
output_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved to: {OUTPUT_FILE}")

# Print summary stats
correct = (predictions == true_labels).sum()
total = len(true_labels)
print(f"\nAccuracy: {correct}/{total} = {100*correct/total:.1f}%")

# Confusion matrix
tp = ((predictions == 1) & (true_labels == 1)).sum()
fp = ((predictions == 1) & (true_labels == 0)).sum()
tn = ((predictions == 0) & (true_labels == 0)).sum()
fn = ((predictions == 0) & (true_labels == 1)).sum()

print(f"\nConfusion Matrix:")
print(f"  TP: {tp}, FP: {fp}")
print(f"  FN: {fn}, TN: {tn}")

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nMetrics:")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1: {f1:.4f}")

print("\n" + "="*70)
