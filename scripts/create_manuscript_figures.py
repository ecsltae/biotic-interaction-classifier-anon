#!/usr/bin/env python3
"""
Create figures for the manuscript
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Set style
sns.set_style("whitegrid")
sns.set_palette("colorblind")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

# Output directory
output_dir = Path("/path/to/MetaP/classifier/manuscript/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# Figure 1: Dataset Composition
def create_dataset_composition():
    """Pie chart showing dataset composition"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Overall balance
    sizes = [10000, 10000]
    labels = ['Positive\n(Interactions)', 'Negative\n(Non-interactions)']
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0.05)

    ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.set_title('Dataset Balance\n(Total: 20,000 sentences)', fontsize=12, fontweight='bold')

    # Negative example breakdown
    neg_sizes = [2000, 2000, 2000, 4000]
    neg_labels = ['Co-occurrence\n(2,000)', 'Scientific\nDescriptions\n(2,000)',
                  'Multi-species\nLists\n(2,000)', 'Random\nCorpus\n(4,000)']
    neg_colors = ['#e74c3c', '#c0392b', '#a93226', '#922b21']

    ax2.pie(neg_sizes, labels=neg_labels, colors=neg_colors,
            autopct='%1.0f%%', shadow=True, startangle=90)
    ax2.set_title('Negative Examples Breakdown\n(Total: 10,000 sentences)',
                  fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'figure1_dataset_composition.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure1_dataset_composition.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Created Figure 1: Dataset Composition")


# Figure 2: Model Performance Comparison
def create_performance_comparison():
    """Bar chart comparing model performance"""
    # Placeholder data - update with actual results
    models = ['DistilBERT', 'BioBERT', 'BiomedBERT', 'RoBERTa']

    # CV metrics (from transformer_cv_results.csv when available)
    cv_precision = [0.75, 0.76, 0.78, 0.77]  # Placeholder
    cv_recall = [0.88, 0.92, 0.90, 0.91]     # Placeholder
    cv_f1 = [0.81, 0.84, 0.83, 0.83]         # Placeholder

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    rects1 = ax.bar(x - width, cv_precision, width, label='Precision', color='#3498db')
    rects2 = ax.bar(x, cv_recall, width, label='Recall', color='#2ecc71')
    rects3 = ax.bar(x + width, cv_f1, width, label='F1-Score', color='#9b59b6')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Cross-Validation Performance by Model (5-fold CV)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure2_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure2_performance_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Created Figure 2: Performance Comparison")


# Figure 3: False Positive Reduction
def create_false_positive_comparison():
    """Compare false positive rates with and without diverse negatives"""
    categories = ['Co-occurrence\nStatements', 'Taxonomic\nDescriptions',
                  'Multi-species\nLists', 'Overall']

    # Hypothetical comparison - baseline vs improved
    baseline_fp_rate = [0.45, 0.38, 0.42, 0.40]  # Placeholder
    improved_fp_rate = [0.12, 0.15, 0.18, 0.14]  # Placeholder

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    rects1 = ax.bar(x - width/2, baseline_fp_rate, width,
                    label='Baseline (random negatives)', color='#e74c3c', alpha=0.8)
    rects2 = ax.bar(x + width/2, improved_fp_rate, width,
                    label='Improved (diverse negatives)', color='#2ecc71', alpha=0.8)

    ax.set_ylabel('False Positive Rate', fontsize=12)
    ax.set_title('Impact of Strategic Negative Sampling on False Positives',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 0.5])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    # Add percentage reduction annotations
    for i, (baseline, improved) in enumerate(zip(baseline_fp_rate, improved_fp_rate)):
        reduction = ((baseline - improved) / baseline) * 100
        ax.annotate(f'↓{reduction:.0f}%',
                   xy=(i, max(baseline, improved) + 0.03),
                   ha='center', fontsize=10, fontweight='bold', color='#27ae60')

    plt.tight_layout()
    plt.savefig(output_dir / 'figure3_false_positive_reduction.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure3_false_positive_reduction.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Created Figure 3: False Positive Reduction")


# Figure 4: Precision-Recall Curve
def create_precision_recall_curves():
    """Precision-Recall curves for different models"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Placeholder data - replace with actual PR curves
    recall = np.linspace(0, 1, 100)

    # Simulated PR curves for different models
    models_data = {
        'BioBERT': 0.84,
        'BiomedBERT': 0.86,
        'DistilBERT': 0.81,
        'RoBERTa': 0.83
    }

    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

    for (model, auc), color in zip(models_data.items(), colors):
        # Simulate PR curve (replace with actual data)
        precision = (1 - recall) * 0.5 + auc * (1 - (1 - recall) * 0.5)
        ax.plot(recall, precision, label=f'{model} (AUC={auc:.2f})',
                linewidth=2, color=color)

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_dir / 'figure4_precision_recall.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure4_precision_recall.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Created Figure 4: Precision-Recall Curves")


# Create all figures
if __name__ == "__main__":
    print("\nCreating manuscript figures...\n")
    create_dataset_composition()
    create_performance_comparison()
    create_false_positive_comparison()
    create_precision_recall_curves()
    print(f"\n✓ All figures saved to {output_dir}/\n")
