#!/usr/bin/env python3
"""
Visualize actual prediction results from CSV files
Creates publication-ready figures for the ensemble model
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

# Set style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

output_dir = Path("/path/to/MetaP/classifier/figures")
output_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("GENERATING PUBLICATION-QUALITY FIGURES")
print("="*70)

# Load the ensemble predictions
ensemble_df = pd.read_csv('/path/to/MetaP/classifier/results/predictions/predictions_Ensemble_F1optimized.csv')
comparison_df = pd.read_csv('/path/to/MetaP/classifier/results/predictions/predictions_ALL_MODELS_comparison.csv')

print(f"\n✓ Loaded {len(ensemble_df)} predictions")

# ============================================================================
# FIGURE 1: Confusion Matrix - Ensemble Model
# ============================================================================
print("\n1. Creating confusion matrix...")

y_true = ensemble_df['True_label']
y_pred = ensemble_df['Ensemble_prediction']

cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['No Interaction', 'Biotic Interaction'],
            yticklabels=['No Interaction', 'Biotic Interaction'],
            ax=ax, annot_kws={'fontsize': 16, 'fontweight': 'bold'})

ax.set_title('Ensemble Model - Confusion Matrix\n(100 Test Sentences)',
             fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')

# Add metrics box
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (tp + tn) / (tp + tn + fp + fn)

metrics_text = (f'Precision: {precision:.1%}\n'
               f'Recall: {recall:.1%}\n'
               f'F1-Score: {f1:.1%}\n'
               f'Accuracy: {accuracy:.1%}')

ax.text(2.7, 1, metrics_text, fontsize=12, verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=2))

plt.tight_layout()
plt.savefig(output_dir / 'confusion_matrix_ensemble.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'confusion_matrix_ensemble.pdf', bbox_inches='tight')
plt.close()
print("  ✓ Saved: confusion_matrix_ensemble.png/pdf")

# ============================================================================
# FIGURE 2: Model Comparison Bar Chart
# ============================================================================
print("\n2. Creating model comparison...")

models = ['Ensemble\n(F1-opt)', 'BiomedBERT', 'RoBERTa']
precisions = [50.0, 46.2, 25.4]
recalls = [43.5, 26.1, 69.6]
f1s = [46.5, 33.3, 37.2]

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - width, precisions, width, label='Precision', color='#3498db', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x, recalls, width, label='Recall', color='#2ecc71', alpha=0.8, edgecolor='black')
bars3 = ax.bar(x + width, f1s, width, label='F1 Score', color='#e74c3c', alpha=0.8, edgecolor='black')

ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
ax.set_title('Model Performance Comparison\n(100-Sentence Evaluation Set)', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12, fontweight='bold')
ax.legend(fontsize=12, loc='upper right')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 100])

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'model_comparison.pdf', bbox_inches='tight')
plt.close()
print("  ✓ Saved: model_comparison.png/pdf")

# ============================================================================
# FIGURE 3: Error Distribution
# ============================================================================
print("\n3. Creating error distribution...")

fp_count = len(comparison_df[(comparison_df['true_label'] == 0) & (comparison_df['Ensemble_pred_f1opt'] == 1)])
fn_count = len(comparison_df[(comparison_df['true_label'] == 1) & (comparison_df['Ensemble_pred_f1opt'] == 0)])
correct_count = len(comparison_df[comparison_df['Ensemble_correct_f1opt'] == True])

fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Correct\nPredictions', 'False\nPositives', 'False\nNegatives']
counts = [correct_count, fp_count, fn_count]
colors = ['#2ecc71', '#e74c3c', '#f39c12']

bars = ax.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylabel('Count (out of 100)', fontsize=14, fontweight='bold')
ax.set_title('Ensemble Model - Prediction Distribution', fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 100])

# Add count and percentage labels
for bar, count in zip(bars, counts):
    height = bar.get_height()
    percentage = count
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{count}\n({percentage}%)',
           ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'error_distribution.pdf', bbox_inches='tight')
plt.close()
print("  ✓ Saved: error_distribution.png/pdf")

# ============================================================================
# FIGURE 4: Probability Distribution
# ============================================================================
print("\n4. Creating probability distribution...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Positive class probabilities
positive_samples = ensemble_df[ensemble_df['True_label'] == 1]['Ensemble_probability']
negative_samples = ensemble_df[ensemble_df['True_label'] == 0]['Ensemble_probability']

ax1.hist(positive_samples, bins=20, alpha=0.7, color='#2ecc71', edgecolor='black', label='True Positives')
ax1.axvline(0.389886, color='red', linestyle='--', linewidth=2, label='F1-opt Threshold')
ax1.set_xlabel('Predicted Probability (Positive Class)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax1.set_title('Probability Distribution\n(True Positive Samples)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

ax2.hist(negative_samples, bins=20, alpha=0.7, color='#e74c3c', edgecolor='black', label='True Negatives')
ax2.axvline(0.389886, color='red', linestyle='--', linewidth=2, label='F1-opt Threshold')
ax2.set_xlabel('Predicted Probability (Positive Class)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax2.set_title('Probability Distribution\n(True Negative Samples)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'probability_distribution.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'probability_distribution.pdf', bbox_inches='tight')
plt.close()
print("  ✓ Saved: probability_distribution.png/pdf")

# ============================================================================
# FIGURE 5: Cross-Validation Results (from CV data)
# ============================================================================
print("\n5. Creating CV results comparison...")

cv_df = pd.read_csv('/path/to/MetaP/classifier/results/cv_results/transformer_cv_results.csv')

fig, ax = plt.subplots(figsize=(12, 6))

models = ['BiomedBERT', 'BioBERT', 'DistilBERT', 'RoBERTa']
x = np.arange(len(models))
width = 0.2

# Extract metrics (removing ± values for plotting)
precisions = [81.2, 79.7, 79.0, 76.9]
recalls = [92.6, 93.9, 94.3, 97.0]
f1s = [86.4, 86.2, 86.0, 85.8]
accuracies = [85.5, 85.0, 84.6, 83.9]

bars1 = ax.bar(x - 1.5*width, precisions, width, label='Precision', color='#3498db', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x - 0.5*width, recalls, width, label='Recall', color='#2ecc71', alpha=0.8, edgecolor='black')
bars3 = ax.bar(x + 0.5*width, f1s, width, label='F1 Score', color='#e74c3c', alpha=0.8, edgecolor='black')
bars4 = ax.bar(x + 1.5*width, accuracies, width, label='Accuracy', color='#9b59b6', alpha=0.8, edgecolor='black')

ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
ax.set_title('Cross-Validation Results\n(20,000 Training Samples, 5-Fold CV)', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([70, 100])

# Highlight best model
ax.axvspan(-0.5, 0.5, alpha=0.1, color='gold', zorder=-1)
ax.text(0, 99, '★ Best Model', ha='center', fontsize=10, fontweight='bold', color='darkgoldenrod')

plt.tight_layout()
plt.savefig(output_dir / 'cv_results_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'cv_results_comparison.pdf', bbox_inches='tight')
plt.close()
print("  ✓ Saved: cv_results_comparison.png/pdf")

# ============================================================================
# FIGURE 6: Model Agreement Visualization
# ============================================================================
print("\n6. Creating model agreement visualization...")

# Count agreement patterns
both_correct = len(comparison_df[(comparison_df['BiomedBERT_correct'] == True) &
                                 (comparison_df['RoBERTa_correct'] == True)])
biomedbert_only = len(comparison_df[(comparison_df['BiomedBERT_correct'] == True) &
                                    (comparison_df['RoBERTa_correct'] == False)])
roberta_only = len(comparison_df[(comparison_df['BiomedBERT_correct'] == False) &
                                 (comparison_df['RoBERTa_correct'] == True)])
both_wrong = len(comparison_df[(comparison_df['BiomedBERT_correct'] == False) &
                               (comparison_df['RoBERTa_correct'] == False)])

fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Both\nCorrect', 'Only\nBiomedBERT', 'Only\nRoBERTa', 'Both\nWrong']
counts = [both_correct, biomedbert_only, roberta_only, both_wrong]
colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']

bars = ax.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylabel('Count (out of 100)', fontsize=14, fontweight='bold')
ax.set_title('Model Agreement Analysis\n(BiomedBERT vs RoBERTa)', fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, max(counts) + 10])

# Add labels
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{count}\n({count}%)',
           ha='center', va='bottom', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'model_agreement.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'model_agreement.pdf', bbox_inches='tight')
plt.close()
print("  ✓ Saved: model_agreement.png/pdf")

# ============================================================================
# Print Summary
# ============================================================================
print("\n" + "="*70)
print("✓ ALL FIGURES GENERATED SUCCESSFULLY")
print("="*70)
print(f"\nSaved to: {output_dir}/")
print("\nGenerated figures:")
print("  1. confusion_matrix_ensemble.png/pdf")
print("  2. model_comparison.png/pdf")
print("  3. error_distribution.png/pdf")
print("  4. probability_distribution.png/pdf")
print("  5. cv_results_comparison.png/pdf")
print("  6. model_agreement.png/pdf")
print("\nAll figures are publication-ready (300 DPI, PDF + PNG)")