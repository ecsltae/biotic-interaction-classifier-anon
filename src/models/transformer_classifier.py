#!/usr/bin/env python3
"""
Biotic Interaction Classifier - TRANSFORMER MODELS + ACTIVE LEARNING
=====================================================================
Uses pre-trained transformer models fine-tuned for biotic interaction detection.

Models:
- DistilBERT (distilbert-base-uncased)
- BioBERT (dmis-lab/biobert-base-cased-v1.2)
- RoBERTa (roberta-base)
- PubMedBERT (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract)

Features:
- 5-fold cross-validation
- Active learning loop
- High precision optimization via threshold tuning
- Evaluation on external test set

Requirements:
    pip install torch transformers datasets scikit-learn pandas numpy tqdm

Usage:
    python transformer_classifier.py --model distilbert --epochs 3
    python transformer_classifier.py --model biobert --active_learning
    python transformer_classifier.py --model all --epochs 5
"""

import os
import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve, classification_report
)

from tqdm import tqdm

warnings.filterwarnings('ignore')
# Set HF_TOKEN environment variable if needed for gated models
# os.environ["HF_TOKEN"] = "your_token_here"
# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_CONFIGS = {
    'distilbert': {
        'name': 'distilbert-base-uncased',
        'max_length': 256,
        'batch_size': 16,
        'learning_rate': 2e-5,
    },
    'biobert': {
        'name': 'dmis-lab/biobert-base-cased-v1.2',
        'max_length': 256,
        'batch_size': 16,
        'learning_rate': 2e-5,
    },
    'roberta': {
        'name': 'roberta-base',
        'max_length': 256,
        'batch_size': 16,
        'learning_rate': 2e-5,
    },
    'BiomedBERT': {
        'name': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        'max_length': 256,
        'batch_size': 16,
        'learning_rate': 2e-5,
    },
}

# =============================================================================
# DATASET CLASS
# =============================================================================

class BioticInteractionDataset(Dataset):
    """Dataset for biotic interaction classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_training_data(csv_path):
    """Load training data"""
    df = pd.read_csv(csv_path)
    text_col = 'passage' if 'passage' in df.columns else 'text'
    texts = df[text_col].tolist()
    labels = df['label'].values
    print(f"Training data: {len(texts)} samples")
    print(f"  Label 0: {np.sum(labels == 0)}, Label 1: {np.sum(labels == 1)}")
    return texts, labels


def load_evaluation_set(tsv_path, encoding='latin-1'):
    """Load external evaluation set"""
    df = pd.read_csv(tsv_path, sep='\t', encoding=encoding)
    texts = df['sentence'].apply(lambda x: str(x).lower().strip()).tolist()
    labels = df['evaluation_pair_interacting'].values
    print(f"External evaluation: {len(texts)} samples")
    print(f"  Label 0: {np.sum(labels == 0)}, Label 1: {np.sum(labels == 1)}")
    return texts, labels


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, return_predictions=False):
    """Evaluate model"""
    model.eval()
    predictions = []
    probabilities = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs[:, 1].cpu().numpy())  # Probability of class 1
            true_labels.extend(labels.cpu().numpy())
    
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    true_labels = np.array(true_labels)
    
    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions, zero_division=0),
        'recall': recall_score(true_labels, predictions, zero_division=0),
        'f1': f1_score(true_labels, predictions, zero_division=0),
    }
    
    if return_predictions:
        return metrics, predictions, probabilities, true_labels
    return metrics


def find_optimal_threshold(y_true, y_proba, target_precision=0.7):
    """Find threshold for target precision"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    valid_idx = np.where(precisions >= target_precision)[0]
    if len(valid_idx) == 0:
        best_idx = np.argmax(precisions[:-1])
    else:
        best_idx = valid_idx[np.argmax(recalls[valid_idx])]
    
    if best_idx >= len(thresholds):
        best_idx = len(thresholds) - 1
    
    return thresholds[best_idx], precisions[best_idx], recalls[best_idx]


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def cross_validate_transformer(
    model_key, texts, labels, n_folds=5, epochs=3, device='cuda'
):
    """5-fold cross-validation for transformer model"""
    
    config = MODEL_CONFIGS[model_key]
    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION: {model_key.upper()} ({config['name']})")
    print(f"{'='*70}")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        
        # Split data
        train_texts = [texts[i] for i in train_idx]
        train_labels = labels[train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = labels[val_idx]
        
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(config['name'])
        model = AutoModelForSequenceClassification.from_pretrained(
            config['name'], num_labels=2
        ).to(device)
        
        # Create datasets
        train_dataset = BioticInteractionDataset(
            train_texts, train_labels, tokenizer, config['max_length']
        )
        val_dataset = BioticInteractionDataset(
            val_texts, val_labels, tokenizer, config['max_length']
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=config['batch_size'], shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config['batch_size']
        )
        
        # Optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        
        # Training loop
        best_f1 = 0
        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
            metrics = evaluate(model, val_loader, device)
            
            print(f"  Epoch {epoch+1}: Loss={train_loss:.4f}, "
                  f"Prec={metrics['precision']:.4f}, "
                  f"Rec={metrics['recall']:.4f}, "
                  f"F1={metrics['f1']:.4f}")
            
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_metrics = metrics.copy()
        
        fold_results.append(best_metrics)
        
        # Clean up
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Aggregate results
    avg_results = {
        'Model': model_key,
        'CV_Precision': np.mean([r['precision'] for r in fold_results]),
        'CV_Precision_Std': np.std([r['precision'] for r in fold_results]),
        'CV_Recall': np.mean([r['recall'] for r in fold_results]),
        'CV_Recall_Std': np.std([r['recall'] for r in fold_results]),
        'CV_F1': np.mean([r['f1'] for r in fold_results]),
        'CV_F1_Std': np.std([r['f1'] for r in fold_results]),
        'CV_Accuracy': np.mean([r['accuracy'] for r in fold_results]),
    }
    
    print(f"\n  CV Results ({n_folds} folds):")
    print(f"  Precision: {avg_results['CV_Precision']:.4f} (+/- {avg_results['CV_Precision_Std']*2:.4f})")
    print(f"  Recall:    {avg_results['CV_Recall']:.4f} (+/- {avg_results['CV_Recall_Std']*2:.4f})")
    print(f"  F1:        {avg_results['CV_F1']:.4f} (+/- {avg_results['CV_F1_Std']*2:.4f})")
    
    return avg_results


# =============================================================================
# ACTIVE LEARNING
# =============================================================================

class ActiveLearner:
    """
    Active Learning for transformer models.
    
    Strategies:
    - uncertainty_sampling: Select samples with highest uncertainty
    - margin_sampling: Select samples with smallest margin between top 2 classes
    - entropy_sampling: Select samples with highest prediction entropy
    """
    
    def __init__(self, model_key, device='cuda'):
        self.model_key = model_key
        self.config = MODEL_CONFIGS[model_key]
        self.device = device
        self.model = None
        self.tokenizer = None
        
    def initialize_model(self):
        """Initialize fresh model"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['name'])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config['name'], num_labels=2
        ).to(self.device)
        
    def get_uncertainty_scores(self, texts, batch_size=32):
        """Get uncertainty scores for unlabeled samples"""
        self.model.eval()
        
        dataset = BioticInteractionDataset(
            texts, [0] * len(texts),  # Dummy labels
            self.tokenizer, self.config['max_length']
        )
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        uncertainties = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                probs = torch.softmax(outputs.logits, dim=1)
                
                # Uncertainty = 1 - max probability (closer to 0.5 = more uncertain)
                max_probs = torch.max(probs, dim=1)[0]
                batch_uncertainty = 1 - max_probs
                
                uncertainties.extend(batch_uncertainty.cpu().numpy())
        
        return np.array(uncertainties)
    
    def select_samples(self, unlabeled_indices, n_samples, texts):
        """Select most uncertain samples for labeling"""
        unlabeled_texts = [texts[i] for i in unlabeled_indices]
        uncertainties = self.get_uncertainty_scores(unlabeled_texts)
        
        # Select top n_samples most uncertain
        top_indices = np.argsort(uncertainties)[-n_samples:]
        selected_indices = [unlabeled_indices[i] for i in top_indices]
        
        return selected_indices, uncertainties[top_indices]
    
    def train(self, train_texts, train_labels, val_texts, val_labels, epochs=3):
        """Train the model"""
        train_dataset = BioticInteractionDataset(
            train_texts, train_labels, self.tokenizer, self.config['max_length']
        )
        val_dataset = BioticInteractionDataset(
            val_texts, val_labels, self.tokenizer, self.config['max_length']
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.config['batch_size'], shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config['batch_size']
        )
        
        optimizer = AdamW(self.model.parameters(), lr=self.config['learning_rate'])
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        
        for epoch in range(epochs):
            train_loss = train_epoch(
                self.model, train_loader, optimizer, scheduler, self.device
            )
            metrics = evaluate(self.model, val_loader, self.device)
            print(f"    Epoch {epoch+1}: Loss={train_loss:.4f}, "
                  f"Prec={metrics['precision']:.4f}, Rec={metrics['recall']:.4f}")
        
        return metrics
    
    def evaluate(self, test_texts, test_labels):
        """Evaluate on test set"""
        test_dataset = BioticInteractionDataset(
            test_texts, test_labels, self.tokenizer, self.config['max_length']
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.config['batch_size']
        )
        
        return evaluate(self.model, test_loader, self.device, return_predictions=True)


def run_active_learning(
    model_key, 
    train_texts, train_labels,
    eval_texts, eval_labels,
    initial_samples=500,
    samples_per_iteration=200,
    n_iterations=10,
    epochs_per_iteration=2,
    device='cuda'
):
    """
    Run active learning loop.
    
    1. Start with small labeled set
    2. Train model
    3. Use model to identify most uncertain samples
    4. Add those samples to training set
    5. Repeat
    """
    
    print(f"\n{'='*70}")
    print(f"ACTIVE LEARNING: {model_key.upper()}")
    print(f"{'='*70}")
    print(f"  Initial samples: {initial_samples}")
    print(f"  Samples per iteration: {samples_per_iteration}")
    print(f"  Iterations: {n_iterations}")
    
    # Initialize
    all_indices = list(range(len(train_texts)))
    np.random.seed(42)
    
    # Start with stratified sample
    labeled_indices, unlabeled_indices = train_test_split(
        all_indices, train_size=initial_samples, 
        stratify=train_labels, random_state=42
    )
    labeled_indices = list(labeled_indices)
    unlabeled_indices = list(unlabeled_indices)
    
    # Split some labeled data for validation
    train_idx, val_idx = train_test_split(
        labeled_indices, test_size=0.2, 
        stratify=[train_labels[i] for i in labeled_indices],
        random_state=42
    )
    
    learner = ActiveLearner(model_key, device)
    iteration_results = []
    
    for iteration in range(n_iterations):
        print(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")
        print(f"  Labeled samples: {len(labeled_indices)}")
        print(f"  Unlabeled samples: {len(unlabeled_indices)}")
        
        # Initialize fresh model
        learner.initialize_model()
        
        # Get training data
        iter_train_texts = [train_texts[i] for i in train_idx]
        iter_train_labels = np.array([train_labels[i] for i in train_idx])
        iter_val_texts = [train_texts[i] for i in val_idx]
        iter_val_labels = np.array([train_labels[i] for i in val_idx])
        
        # Train
        print(f"  Training on {len(iter_train_texts)} samples...")
        metrics = learner.train(
            iter_train_texts, iter_train_labels,
            iter_val_texts, iter_val_labels,
            epochs=epochs_per_iteration
        )
        
        # Evaluate on external test set
        test_metrics, preds, probs, true = learner.evaluate(eval_texts, eval_labels)
        
        print(f"  External Test: Prec={test_metrics['precision']:.4f}, "
              f"Rec={test_metrics['recall']:.4f}, F1={test_metrics['f1']:.4f}")
        
        iteration_results.append({
            'Iteration': iteration + 1,
            'Labeled_Samples': len(labeled_indices),
            'Precision': test_metrics['precision'],
            'Recall': test_metrics['recall'],
            'F1': test_metrics['f1'],
            'Accuracy': test_metrics['accuracy'],
        })
        
        # Select new samples via uncertainty sampling
        if len(unlabeled_indices) > 0 and iteration < n_iterations - 1:
            n_to_select = min(samples_per_iteration, len(unlabeled_indices))
            
            print(f"  Selecting {n_to_select} most uncertain samples...")
            selected, uncertainties = learner.select_samples(
                unlabeled_indices, n_to_select, train_texts
            )
            
            print(f"  Uncertainty range: {uncertainties.min():.4f} - {uncertainties.max():.4f}")
            
            # Move selected to labeled
            labeled_indices.extend(selected)
            unlabeled_indices = [i for i in unlabeled_indices if i not in selected]
            
            # Update train/val split
            train_idx, val_idx = train_test_split(
                labeled_indices, test_size=0.2,
                stratify=[train_labels[i] for i in labeled_indices],
                random_state=42
            )
        
        # Clean up
        del learner.model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return pd.DataFrame(iteration_results)


# =============================================================================
# FULL TRAINING AND EVALUATION
# =============================================================================

def train_and_evaluate_full(
    model_key, 
    train_texts, train_labels,
    eval_texts, eval_labels,
    epochs=3,
    target_precision=0.7,
    device='cuda'
):
    """Train on full data and evaluate with threshold optimization"""
    
    config = MODEL_CONFIGS[model_key]
    print(f"\n{'='*70}")
    print(f"FULL TRAINING: {model_key.upper()}")
    print(f"{'='*70}")
    
    # Split for threshold tuning
    train_texts_tr, train_texts_val, train_labels_tr, train_labels_val = train_test_split(
        train_texts, train_labels, test_size=0.1, stratify=train_labels, random_state=42
    )
    
    # Initialize
    tokenizer = AutoTokenizer.from_pretrained(config['name'])
    model = AutoModelForSequenceClassification.from_pretrained(
        config['name'], num_labels=2
    ).to(device)
    
    # Datasets
    train_dataset = BioticInteractionDataset(
        train_texts_tr, train_labels_tr, tokenizer, config['max_length']
    )
    val_dataset = BioticInteractionDataset(
        train_texts_val, train_labels_val, tokenizer, config['max_length']
    )
    eval_dataset = BioticInteractionDataset(
        eval_texts, eval_labels, tokenizer, config['max_length']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    eval_loader = DataLoader(eval_dataset, batch_size=config['batch_size'])
    
    # Training
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    
    print("\nTraining...")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_metrics = evaluate(model, val_loader, device)
        print(f"  Epoch {epoch+1}: Loss={train_loss:.4f}, "
              f"Val Prec={val_metrics['precision']:.4f}, "
              f"Val Rec={val_metrics['recall']:.4f}")
    
    # Find optimal threshold on validation set
    print("\nFinding optimal threshold...")
    val_metrics, val_preds, val_probs, val_true = evaluate(
        model, val_loader, device, return_predictions=True
    )
    
    opt_threshold, opt_prec, opt_rec = find_optimal_threshold(
        val_true, val_probs, target_precision
    )
    print(f"  Optimal threshold: {opt_threshold:.3f}")
    print(f"  Expected precision: {opt_prec:.4f}, recall: {opt_rec:.4f}")
    
    # Evaluate on external set with optimal threshold
    print("\nEvaluating on external test set...")
    eval_metrics, eval_preds, eval_probs, eval_true = evaluate(
        model, eval_loader, device, return_predictions=True
    )
    
    # Apply threshold
    eval_preds_thresh = (eval_probs >= opt_threshold).astype(int)
    
    results = {
        'Model': model_key,
        'Threshold': opt_threshold,
        'Precision_Default': precision_score(eval_true, eval_preds, zero_division=0),
        'Recall_Default': recall_score(eval_true, eval_preds, zero_division=0),
        'F1_Default': f1_score(eval_true, eval_preds, zero_division=0),
        'Precision_Optimized': precision_score(eval_true, eval_preds_thresh, zero_division=0),
        'Recall_Optimized': recall_score(eval_true, eval_preds_thresh, zero_division=0),
        'F1_Optimized': f1_score(eval_true, eval_preds_thresh, zero_division=0),
    }
    
    print(f"\n  Results (default threshold=0.5):")
    print(f"    Precision: {results['Precision_Default']:.4f}")
    print(f"    Recall:    {results['Recall_Default']:.4f}")
    print(f"    F1:        {results['F1_Default']:.4f}")
    
    print(f"\n  Results (optimized threshold={opt_threshold:.3f}):")
    print(f"    Precision: {results['Precision_Optimized']:.4f}")
    print(f"    Recall:    {results['Recall_Optimized']:.4f}")
    print(f"    F1:        {results['F1_Optimized']:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(eval_true, eval_preds_thresh)
    print(f"\n  Confusion Matrix (optimized):")
    print(f"    TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    
    # Save model
    save_path = f"transformer_{model_key}_model"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    with open(f"{save_path}/config.pkl", 'wb') as f:
        pickle.dump({
            'model_key': model_key,
            'threshold': opt_threshold,
            'results': results,
        }, f)
    
    print(f"\n  Model saved to: {save_path}/")
    
    return results, model, tokenizer, opt_threshold


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Transformer Classifier for Biotic Interactions')
    parser.add_argument('--model', type=str, default='all',
                        choices=['distilbert', 'biobert', 'roberta', 'BiomedBERT', 'all'],
                        help='Model to use')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--cv_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--active_learning', action='store_true', 
                        help='Run active learning')
    parser.add_argument('--al_iterations', type=int, default=10,
                        help='Active learning iterations')
    parser.add_argument('--target_precision', type=float, default=0.7,
                        help='Target precision for threshold optimization')
    parser.add_argument('--train_data', type=str, default='training_data_cleaned.csv')
    parser.add_argument('--eval_data', type=str, default='eval_100.tsv')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print("="*70)
    print("BIOTIC INTERACTION CLASSIFIER - TRANSFORMER MODELS")
    print("="*70)
    print(f"Device: {args.device}")
    if args.device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print("\n[1] Loading data...")
    train_texts, train_labels = load_training_data(args.train_data)
    eval_texts, eval_labels = load_evaluation_set(args.eval_data)
    
    # Determine which models to run
    if args.model == 'all':
        model_keys = list(MODEL_CONFIGS.keys())
    else:
        model_keys = [args.model]
    
    all_cv_results = []
    all_eval_results = []
    all_al_results = []
    
    for model_key in model_keys:
        print(f"\n{'#'*70}")
        print(f"# MODEL: {model_key.upper()}")
        print(f"{'#'*70}")
        
        # Cross-validation
        print("\n[2] Cross-validation...")
        cv_result = cross_validate_transformer(
            model_key, train_texts, train_labels,
            n_folds=args.cv_folds, epochs=args.epochs, device=args.device
        )
        all_cv_results.append(cv_result)
        
        # Full training and evaluation
        print("\n[3] Full training and evaluation...")
        eval_result, model, tokenizer, threshold = train_and_evaluate_full(
            model_key, train_texts, train_labels, eval_texts, eval_labels,
            epochs=args.epochs, target_precision=args.target_precision,
            device=args.device
        )
        all_eval_results.append(eval_result)
        
        # Active learning (optional)
        if args.active_learning:
            print("\n[4] Active learning...")
            al_results = run_active_learning(
                model_key, train_texts, train_labels, eval_texts, eval_labels,
                initial_samples=500,
                samples_per_iteration=200,
                n_iterations=args.al_iterations,
                epochs_per_iteration=2,
                device=args.device
            )
            al_results['Model'] = model_key
            all_al_results.append(al_results)
        
        # Clean up
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    cv_df = pd.DataFrame(all_cv_results)
    cv_df.to_csv('transformer_cv_results.csv', index=False)
    print("  Cross-validation results: transformer_cv_results.csv")
    
    eval_df = pd.DataFrame(all_eval_results)
    eval_df.to_csv('transformer_eval_results.csv', index=False)
    print("  Evaluation results: transformer_eval_results.csv")
    
    if all_al_results:
        al_df = pd.concat(all_al_results, ignore_index=True)
        al_df.to_csv('transformer_active_learning_results.csv', index=False)
        print("  Active learning results: transformer_active_learning_results.csv")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: CROSS-VALIDATION")
    print("="*70)
    print(cv_df.to_string(index=False))
    
    print("\n" + "="*70)
    print("SUMMARY: EXTERNAL EVALUATION")
    print("="*70)
    print(eval_df.to_string(index=False))
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()