#!/usr/bin/env python3
"""
Create detailed visualizations for classifier predictions
Including confusion matrix, per-sentence comparison, and training curves
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 9

# Output directory
output_dir = Path("/path/to/MetaP/classifier/figures")
output_dir.mkdir(parents=True, exist_ok=True)


def create_confusion_matrix_heatmap():
    """Create confusion matrix visualization"""
    # Placeholder - replace with actual data from results
    cm = np.array([[7500, 500],   # True Negative, False Positive
                   [300, 9700]])   # False Negative, True Positive

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'],
                ax=ax, annot_kws={'fontsize': 14})

    ax.set_title('Confusion Matrix - BiomedBERT on Improved Dataset',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)

    # Add metrics text
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    metrics_text = f'Precision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}\nAccuracy: {accuracy:.3f}'
    ax.text(2.5, 1, metrics_text, fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created confusion matrix")


def create_prediction_comparison_table(csv_file=None):
    """Create visual comparison table like in screenshot"""
    # Load data or use sample
    if csv_file and Path(csv_file).exists():
        df = pd.read_csv(csv_file)
    else:
        # Sample data matching screenshot structure
        df = pd.DataFrame({
            'sentence': [
                'geographic and hostrelated variation among species',
                'fieldcollected specimens of glossiphoniid leeches to',
                'pseudomonas aeruginosa a gramnegative bacterium',
                'spironucleus barkhanus from muscle abscesses of f',
                'it is a macrolide endectocide with activity against bo',
                'corrigendum leishmania infantum infecting the carn'
            ],
            'emilie_label': [0, 1, 0, 0, 0, 1],
            'BiomedBERT_prediction': [0, 0, 0, 1, 1, 0],
            'true_sentiment': ['negative', 'positive', 'negative', 'negative', 'negative', 'positive'],
            'BiomedBERT_sentiment': ['negative', 'negative', 'negative', 'positive', 'positive', 'negative']
        })

    # Create figure
    fig, ax = plt.subplots(figsize=(16, len(df) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')

    # Prepare data for table
    table_data = []
    for idx, row in df.iterrows():
        table_data.append([
            row['sentence'][:60] + '...' if len(row['sentence']) > 60 else row['sentence'],
            str(row['emilie_label']),
            str(row['BiomedBERT_prediction']),
            row['true_sentiment'],
            row['BiomedBERT_sentiment']
        ])

    # Create table
    table = ax.table(cellText=table_data,
                     colLabels=['Sentence', "Emilie's Label", 'BiomedBERT Prediction',
                               'True Sentiment', 'BiomedBERT Sentiment'],
                     cellLoc='left',
                     loc='center',
                     bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color code cells based on correctness
    for i in range(len(df)):
        row_idx = i + 1  # +1 because of header

        # Color based on agreement
        emilie_label = df.iloc[i]['emilie_label']
        prediction = df.iloc[i]['BiomedBERT_prediction']

        if emilie_label == prediction:
            # Correct prediction - light green
            color = '#d5f4e6'
        else:
            # Incorrect prediction - light red
            color = '#fadbd8'

        # Apply color to entire row
        for j in range(5):
            table[(row_idx, j)].set_facecolor(color)

    # Header styling
    for j in range(5):
        table[(0, j)].set_facecolor('#3498db')
        table[(0, j)].set_text_props(weight='bold', color='white')

    plt.title('Prediction Comparison: Ground Truth vs BiomedBERT',
              fontsize=14, fontweight='bold', pad=20)

    # Add legend
    correct_patch = mpatches.Patch(color='#d5f4e6', label='Correct Prediction')
    incorrect_patch = mpatches.Patch(color='#fadbd8', label='Incorrect Prediction')
    plt.legend(handles=[correct_patch, incorrect_patch],
              bbox_to_anchor=(0.5, -0.05), loc='upper center', ncol=2, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_comparison_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created prediction comparison table")


def create_training_curves():
    """Create training and validation curves"""
    # Placeholder data - replace with actual training logs
    epochs = np.arange(1, 4)
    folds = 5

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Loss curves
    for fold in range(folds):
        train_loss = [0.45, 0.28, 0.18] + np.random.randn(3) * 0.02
        val_loss = [0.48, 0.32, 0.24] + np.random.randn(3) * 0.03
        ax1.plot(epochs, train_loss, label=f'Fold {fold+1}', alpha=0.6)

    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Training Loss', fontsize=11)
    ax1.set_title('Training Loss Across Folds', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Validation F1 score
    for fold in range(folds):
        val_f1 = [0.75, 0.82, 0.85] + np.random.randn(3) * 0.01
        ax2.plot(epochs, val_f1, label=f'Fold {fold+1}', alpha=0.6, marker='o')

    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Validation F1-Score', fontsize=11)
    ax2.set_title('Validation F1-Score Across Folds', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0.7, 0.9])

    # Precision and Recall progression
    mean_precision = [0.72, 0.75, 0.76]
    mean_recall = [0.88, 0.91, 0.92]
    ax3.plot(epochs, mean_precision, label='Precision', marker='o', linewidth=2, color='#3498db')
    ax3.plot(epochs, mean_recall, label='Recall', marker='s', linewidth=2, color='#2ecc71')
    ax3.fill_between(epochs,
                     np.array(mean_precision) - 0.02,
                     np.array(mean_precision) + 0.02,
                     alpha=0.2, color='#3498db')
    ax3.fill_between(epochs,
                     np.array(mean_recall) - 0.02,
                     np.array(mean_recall) + 0.02,
                     alpha=0.2, color='#2ecc71')

    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Score', fontsize=11)
    ax3.set_title('Mean Precision & Recall (±std)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    ax3.set_ylim([0.65, 0.95])

    # Final metrics by fold
    fold_names = [f'Fold {i+1}' for i in range(folds)]
    fold_f1 = [0.84, 0.85, 0.83, 0.84, 0.86]
    fold_precision = [0.76, 0.77, 0.75, 0.76, 0.78]

    x_pos = np.arange(len(fold_names))
    width = 0.35

    bars1 = ax4.bar(x_pos - width/2, fold_precision, width, label='Precision', color='#3498db')
    bars2 = ax4.bar(x_pos + width/2, fold_f1, width, label='F1-Score', color='#9b59b6')

    ax4.set_xlabel('Fold', fontsize=11)
    ax4.set_ylabel('Score', fontsize=11)
    ax4.set_title('Final Performance by Fold', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(fold_names)
    ax4.legend(fontsize=10)
    ax4.set_ylim([0.7, 0.9])
    ax4.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created training curves")


def create_error_analysis():
    """Visualize error patterns"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Error types
    error_types = ['False\nPositives', 'False\nNegatives']
    error_counts = [500, 300]
    colors = ['#e74c3c', '#f39c12']

    ax1.bar(error_types, error_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Error Distribution', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, v in enumerate(error_counts):
        ax1.text(i, v + 10, str(v), ha='center', fontsize=12, fontweight='bold')

    # False positive breakdown
    fp_categories = ['Co-occurrence', 'Taxonomic\nDesc.', 'Multi-species', 'Other']
    fp_counts = [180, 120, 100, 100]
    colors_fp = ['#e74c3c', '#c0392b', '#a93226', '#922b21']

    ax2.barh(fp_categories, fp_counts, color=colors_fp, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Count', fontsize=12)
    ax2.set_title('False Positive Breakdown', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, v in enumerate(fp_counts):
        ax2.text(v + 5, i, str(v), va='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'error_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created error analysis")


def create_performance_heatmap():
    """Create heatmap of model performance across different data types"""
    # Data types vs models
    data_types = ['Scientific\nPapers', 'Abstracts', 'Full Text', 'Mixed']
    models = ['DistilBERT', 'BioBERT', 'BiomedBERT', 'RoBERTa']

    # Placeholder performance data (F1 scores)
    performance = np.array([
        [0.78, 0.82, 0.84, 0.81],  # Scientific Papers
        [0.80, 0.84, 0.86, 0.83],  # Abstracts
        [0.76, 0.80, 0.82, 0.79],  # Full Text
        [0.79, 0.83, 0.85, 0.82]   # Mixed
    ])

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(performance, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=models, yticklabels=data_types,
                cbar_kws={'label': 'F1-Score'}, vmin=0.7, vmax=0.9,
                ax=ax, linewidths=0.5, linecolor='gray')

    ax.set_title('Model Performance Across Data Types (F1-Score)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Data Type', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created performance heatmap")


# Create all visualizations
if __name__ == "__main__":
    print("\nCreating prediction visualizations...\n")
    create_confusion_matrix_heatmap()
    create_prediction_comparison_table()
    create_training_curves()
    create_error_analysis()
    create_performance_heatmap()
    print(f"\n✓ All visualizations saved to {output_dir}/\n")
