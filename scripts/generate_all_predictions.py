#!/usr/bin/env python3
"""
Generate detailed prediction CSV files for all models on eval_100.tsv
Similar to the screenshot format
"""
import sys
sys.path.insert(0, '/path/to/MetaP/classifier/src/models')

import pandas as pd
import numpy as np
from ensemble_classifier import OptimizedEnsembleClassifier

# Load the real test set
print("Loading eval_100.tsv...")
test_df = pd.read_csv('/path/to/MetaP/classifier/data/evaluation/eval_100.tsv', sep='\t')
test_texts = test_df['sentence'].tolist()
test_labels = test_df['evaluation_pair_interacting'].tolist()

print(f"Loaded {len(test_df)} samples\n")

# Create results dataframe
results_df = pd.DataFrame({
    'sentence': test_texts,
    'true_label': test_labels,
    'true_sentiment': ['positive' if l == 1 else 'negative' for l in test_labels]
})

print("="*70)
print("GENERATING PREDICTIONS FOR ALL MODELS")
print("="*70)

# 1. ENSEMBLE MODEL
print("\n1. Loading Ensemble (BiomedBERT + RoBERTa)...")
ensemble = OptimizedEnsembleClassifier(optimize=True)

# Get probabilities and predictions
probs_ensemble = ensemble.predict_proba(test_texts)
results_df['Ensemble_probability'] = probs_ensemble[:, 1]

# Multiple thresholds
results_df['Ensemble_pred_f1opt'] = (probs_ensemble[:, 1] >= 0.389886).astype(int)
results_df['Ensemble_pred_default'] = (probs_ensemble[:, 1] >= 0.5).astype(int)
results_df['Ensemble_pred_precision'] = (probs_ensemble[:, 1] >= 0.944263).astype(int)

results_df['Ensemble_sentiment_f1opt'] = ['positive' if p == 1 else 'negative' for p in results_df['Ensemble_pred_f1opt']]
results_df['Ensemble_sentiment_default'] = ['positive' if p == 1 else 'negative' for p in results_df['Ensemble_pred_default']]

print("✓ Ensemble predictions complete")

# 2. Individual model predictions (if we want to add them)
# Get individual model probabilities
print("\n2. Getting individual model contributions...")

# BiomedBERT contribution (from ensemble)
biomedbert_model = ensemble.models['biomedbert']
biomedbert_tokenizer = ensemble.tokenizers['biomedbert']

# Get BiomedBERT predictions
from torch.utils.data import DataLoader
from ensemble_classifier import BioticInteractionDataset
import torch

biomedbert_dataset = BioticInteractionDataset(
    test_texts, [0]*len(test_texts), biomedbert_tokenizer, 256
)
biomedbert_loader = DataLoader(biomedbert_dataset, batch_size=32, shuffle=False)

biomedbert_probs = []
with torch.no_grad():
    for batch in biomedbert_loader:
        input_ids = batch['input_ids'].to(ensemble.device)
        attention_mask = batch['attention_mask'].to(ensemble.device)
        outputs = biomedbert_model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1)
        biomedbert_probs.append(probs.cpu().numpy())

biomedbert_probs = np.vstack(biomedbert_probs)
results_df['BiomedBERT_probability'] = biomedbert_probs[:, 1]
results_df['BiomedBERT_pred'] = (biomedbert_probs[:, 1] >= 0.5).astype(int)
results_df['BiomedBERT_sentiment'] = ['positive' if p == 1 else 'negative' for p in results_df['BiomedBERT_pred']]

print("✓ BiomedBERT predictions complete")

# RoBERTa contribution
roberta_model = ensemble.models['roberta']
roberta_tokenizer = ensemble.tokenizers['roberta']

roberta_dataset = BioticInteractionDataset(
    test_texts, [0]*len(test_texts), roberta_tokenizer, 256
)
roberta_loader = DataLoader(roberta_dataset, batch_size=32, shuffle=False)

roberta_probs = []
with torch.no_grad():
    for batch in roberta_loader:
        input_ids = batch['input_ids'].to(ensemble.device)
        attention_mask = batch['attention_mask'].to(ensemble.device)
        outputs = roberta_model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1)
        roberta_probs.append(probs.cpu().numpy())

roberta_probs = np.vstack(roberta_probs)
results_df['RoBERTa_probability'] = roberta_probs[:, 1]
results_df['RoBERTa_pred'] = (roberta_probs[:, 1] >= 0.5).astype(int)
results_df['RoBERTa_sentiment'] = ['positive' if p == 1 else 'negative' for p in results_df['RoBERTa_pred']]

print("✓ RoBERTa predictions complete")

# 3. Add agreement columns
results_df['Ensemble_correct_f1opt'] = (results_df['Ensemble_pred_f1opt'] == results_df['true_label'])
results_df['BiomedBERT_correct'] = (results_df['BiomedBERT_pred'] == results_df['true_label'])
results_df['RoBERTa_correct'] = (results_df['RoBERTa_pred'] == results_df['true_label'])
results_df['All_models_agree'] = (
    (results_df['Ensemble_pred_f1opt'] == results_df['BiomedBERT_pred']) &
    (results_df['BiomedBERT_pred'] == results_df['RoBERTa_pred'])
)

# 4. Save multiple formats
output_dir = '/path/to/MetaP/classifier/results/predictions'

# Format 1: Like screenshot - Ensemble F1-optimized (RECOMMENDED)
print("\n" + "="*70)
print("SAVING PREDICTION FILES")
print("="*70)

ensemble_format = results_df[[
    'sentence', 'true_label', 'Ensemble_pred_f1opt',
    'true_sentiment', 'Ensemble_sentiment_f1opt', 'Ensemble_probability'
]].copy()
ensemble_format.columns = [
    'sentence', 'True_label', 'Ensemble_prediction',
    'True_sentiment', 'Ensemble_sentiment', 'Ensemble_probability'
]
ensemble_path = f'{output_dir}/predictions_Ensemble_F1optimized.csv'
ensemble_format.to_csv(ensemble_path, index=False)
print(f"✓ Saved: {ensemble_path}")

# Format 2: BiomedBERT only
biomedbert_format = results_df[[
    'sentence', 'true_label', 'BiomedBERT_pred',
    'true_sentiment', 'BiomedBERT_sentiment', 'BiomedBERT_probability'
]].copy()
biomedbert_format.columns = [
    'sentence', 'True_label', 'BiomedBERT_prediction',
    'True_sentiment', 'BiomedBERT_sentiment', 'BiomedBERT_probability'
]
biomedbert_path = f'{output_dir}/predictions_BiomedBERT.csv'
biomedbert_format.to_csv(biomedbert_path, index=False)
print(f"✓ Saved: {biomedbert_path}")

# Format 3: RoBERTa only
roberta_format = results_df[[
    'sentence', 'true_label', 'RoBERTa_pred',
    'true_sentiment', 'RoBERTa_sentiment', 'RoBERTa_probability'
]].copy()
roberta_format.columns = [
    'sentence', 'True_label', 'RoBERTa_prediction',
    'True_sentiment', 'RoBERTa_sentiment', 'RoBERTa_probability'
]
roberta_path = f'{output_dir}/predictions_RoBERTa.csv'
roberta_format.to_csv(roberta_path, index=False)
print(f"✓ Saved: {roberta_path}")

# Format 4: Complete comparison (all models)
comparison_cols = [
    'sentence', 'true_label', 'true_sentiment',
    'BiomedBERT_pred', 'BiomedBERT_sentiment', 'BiomedBERT_probability',
    'RoBERTa_pred', 'RoBERTa_sentiment', 'RoBERTa_probability',
    'Ensemble_pred_f1opt', 'Ensemble_sentiment_f1opt', 'Ensemble_probability',
    'BiomedBERT_correct', 'RoBERTa_correct', 'Ensemble_correct_f1opt',
    'All_models_agree'
]
comparison_format = results_df[comparison_cols].copy()
comparison_path = f'{output_dir}/predictions_ALL_MODELS_comparison.csv'
comparison_format.to_csv(comparison_path, index=False)
print(f"✓ Saved: {comparison_path}")

# 5. Generate error analysis
print("\n" + "="*70)
print("GENERATING ERROR ANALYSIS")
print("="*70)

# False positives and false negatives for each model
for model_name, pred_col, correct_col in [
    ('Ensemble_F1opt', 'Ensemble_pred_f1opt', 'Ensemble_correct_f1opt'),
    ('BiomedBERT', 'BiomedBERT_pred', 'BiomedBERT_correct'),
    ('RoBERTa', 'RoBERTa_pred', 'RoBERTa_correct')
]:
    # False positives (predicted 1, actually 0)
    fp = results_df[(results_df[pred_col] == 1) & (results_df['true_label'] == 0)]
    fp_path = f'{output_dir}/errors_FalsePositives_{model_name}.csv'
    fp.to_csv(fp_path, index=False)

    # False negatives (predicted 0, actually 1)
    fn = results_df[(results_df[pred_col] == 0) & (results_df['true_label'] == 1)]
    fn_path = f'{output_dir}/errors_FalseNegatives_{model_name}.csv'
    fn.to_csv(fn_path, index=False)

    print(f"✓ {model_name}: {len(fp)} false positives, {len(fn)} false negatives")

# 6. Print summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

for model_name, pred_col, prob_col in [
    ('Ensemble (F1-opt)', 'Ensemble_pred_f1opt', 'Ensemble_probability'),
    ('BiomedBERT', 'BiomedBERT_pred', 'BiomedBERT_probability'),
    ('RoBERTa', 'RoBERTa_pred', 'RoBERTa_probability')
]:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    acc = accuracy_score(results_df['true_label'], results_df[pred_col])
    prec = precision_score(results_df['true_label'], results_df[pred_col], zero_division=0)
    rec = recall_score(results_df['true_label'], results_df[pred_col], zero_division=0)
    f1 = f1_score(results_df['true_label'], results_df[pred_col], zero_division=0)

    print(f"\n{model_name}:")
    print(f"  Accuracy:  {acc:.1%}")
    print(f"  Precision: {prec:.1%}")
    print(f"  Recall:    {rec:.1%}")
    print(f"  F1:        {f1:.1%}")

print("\n" + "="*70)
print("✓ ALL PREDICTION FILES GENERATED")
print("="*70)
print(f"\nFiles saved to: {output_dir}/")
print("\nMain files:")
print("  - predictions_Ensemble_F1optimized.csv (RECOMMENDED)")
print("  - predictions_BiomedBERT.csv")
print("  - predictions_RoBERTa.csv")
print("  - predictions_ALL_MODELS_comparison.csv (complete comparison)")
print("\nError analysis:")
print("  - errors_FalsePositives_*.csv")
print("  - errors_FalseNegatives_*.csv")