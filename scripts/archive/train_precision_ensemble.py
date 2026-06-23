#!/usr/bin/env python3
"""
Train High-Precision Ensemble for Biotic Interaction Classification
====================================================================

Strategy for high precision:
1. Use BiomedBERT as base (best biomedical understanding)
2. Train with class weights to penalize false positives more
3. Use focal loss to focus on hard examples
4. Train multiple models on different data splits for diversity
5. Ensemble with precision-optimized threshold

GPU: 80GB - can use larger batch sizes and potentially multiple models in memory

Target: Precision > 0.70 while maintaining F1 > 0.45
"""

import os
import sys
import time
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.optim import AdamW

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = '/path/to/MetaP/classifier'
DATA_DIR = f'{BASE_DIR}/data/training'
MODEL_DIR = f'{BASE_DIR}/models'
OUTPUT_DIR = f'{MODEL_DIR}/precision_ensemble'

# Training config optimized for 80GB GPU
CONFIG = {
    'seed': 42,
    'max_length': 256,
    'batch_size': 64,          # Large batch for 80GB GPU
    'eval_batch_size': 128,
    'epochs': 6,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'fp16': True,

    # Precision-focused settings
    'pos_weight': 0.8,         # Weight for positive class (lower = fewer FP)
    'focal_gamma': 2.0,        # Focal loss gamma
    'label_smoothing': 0.1,    # Slight smoothing helps calibration

    # Models to train
    'models': [
        {
            'name': 'biomedbert_precision',
            'base': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
            'weight': 0.65,  # Higher weight for biomedical specialist
        },
        {
            'name': 'roberta_precision',
            'base': 'roberta-base',
            'weight': 0.35,  # General language understanding
        },
    ]
}

# Set seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG['seed'])

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ============================================================================
# DATASET
# ============================================================================

class BioticInteractionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
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
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ============================================================================
# FOCAL LOSS (focuses on hard examples)
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.
    Reduces weight of easy examples, focuses on hard ones.
    """
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Class weights
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits, targets,
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ============================================================================
# CUSTOM TRAINER WITH FOCAL LOSS
# ============================================================================

class PrecisionTrainer(Trainer):
    """Custom trainer with focal loss and precision-focused metrics"""

    def __init__(self, *args, focal_gamma=2.0, pos_weight=1.0, label_smoothing=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_gamma = focal_gamma
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing

        # Class weights: higher weight on negatives to reduce false positives
        weights = torch.tensor([1.0, pos_weight]).to(DEVICE)
        self.loss_fn = FocalLoss(
            gamma=focal_gamma,
            alpha=weights,
            label_smoothing=label_smoothing
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(pred):
    labels = pred.label_ids
    probs = torch.softmax(torch.tensor(pred.predictions), dim=-1)[:, 1].numpy()

    # Find threshold that maximizes precision while keeping F1 reasonable
    best_metrics = None
    best_score = 0

    for thresh in np.arange(0.3, 0.8, 0.05):
        preds = (probs >= thresh).astype(int)
        p = precision_score(labels, preds, zero_division=0)
        r = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        # Score: prioritize precision but require minimum recall
        if r >= 0.3:  # Minimum recall threshold
            score = p * 0.7 + f1 * 0.3  # Weighted toward precision
            if score > best_score:
                best_score = score
                best_metrics = {
                    'precision': p,
                    'recall': r,
                    'f1': f1,
                    'threshold': thresh,
                    'accuracy': accuracy_score(labels, preds),
                }

    # Fallback to default threshold
    if best_metrics is None:
        preds = (probs >= 0.5).astype(int)
        best_metrics = {
            'precision': precision_score(labels, preds, zero_division=0),
            'recall': recall_score(labels, preds, zero_division=0),
            'f1': f1_score(labels, preds, zero_division=0),
            'threshold': 0.5,
            'accuracy': accuracy_score(labels, preds),
        }

    return best_metrics


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_high_quality_data():
    """
    Prepare training data with focus on quality.
    Use the original 6k dataset (proven best) + high-confidence additions.
    """
    print("\n" + "="*70)
    print("PREPARING HIGH-QUALITY TRAINING DATA")
    print("="*70)

    # Load original 6k (our best performing dataset)
    df_6k = pd.read_csv(f'{DATA_DIR}/training_data_cleaned.csv')
    print(f"Original 6k: {len(df_6k)} samples")

    # Load quality v2 (improved dataset)
    df_quality = pd.read_csv(f'{DATA_DIR}/training_data_quality_v2.csv')
    print(f"Quality v2: {len(df_quality)} samples")

    # Combine, removing duplicates
    df_6k['source'] = '6k_original'
    df_quality['source'] = 'quality_v2'

    combined = pd.concat([df_6k, df_quality], ignore_index=True)
    combined = combined.drop_duplicates(subset=['passage'], keep='first')

    print(f"Combined (deduplicated): {len(combined)} samples")
    print(f"  Positives: {sum(combined['label']==1)}")
    print(f"  Negatives: {sum(combined['label']==0)}")

    return combined['passage'].tolist(), combined['label'].tolist()


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(model_config, train_texts, train_labels, val_texts, val_labels):
    """Train a single model with precision-focused settings"""

    model_name = model_config['name']
    base_model = model_config['base']
    output_path = f'{OUTPUT_DIR}/{model_name}'

    print(f"\n{'='*70}")
    print(f"TRAINING: {model_name}")
    print(f"Base: {base_model}")
    print(f"{'='*70}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
        problem_type="single_label_classification"
    )

    # Create datasets
    train_dataset = BioticInteractionDataset(
        train_texts, train_labels, tokenizer, CONFIG['max_length']
    )
    val_dataset = BioticInteractionDataset(
        val_texts, val_labels, tokenizer, CONFIG['max_length']
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=CONFIG['epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['eval_batch_size'],
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        warmup_ratio=CONFIG['warmup_ratio'],

        # Evaluation
        eval_strategy='steps',
        eval_steps=100,
        save_strategy='steps',
        save_steps=100,

        # Best model
        load_best_model_at_end=True,
        metric_for_best_model='precision',  # Optimize for precision!
        greater_is_better=True,

        # Performance
        fp16=CONFIG['fp16'] and DEVICE == 'cuda',
        dataloader_num_workers=4,

        # Logging
        logging_dir=f'{output_path}/logs',
        logging_steps=50,
        report_to='none',

        save_total_limit=2,
    )

    # Custom trainer with focal loss
    trainer = PrecisionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        focal_gamma=CONFIG['focal_gamma'],
        pos_weight=CONFIG['pos_weight'],
        label_smoothing=CONFIG['label_smoothing'],
    )

    # Train
    print(f"\nStarting training...")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Epochs: {CONFIG['epochs']}")

    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time

    print(f"\nTraining completed in {train_time/60:.1f} minutes")

    # Save
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Final evaluation
    print("\nFinal Validation Results:")
    eval_results = trainer.evaluate()
    for k, v in eval_results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    return output_path, eval_results


# ============================================================================
# ENSEMBLE CREATION
# ============================================================================

def create_ensemble_config(model_paths, val_texts, val_labels):
    """
    Create ensemble configuration by finding optimal weights.
    Focus on maximizing precision.
    """
    print("\n" + "="*70)
    print("CREATING ENSEMBLE CONFIGURATION")
    print("="*70)

    # Get predictions from each model
    all_probs = {}
    model_weights = {}

    for i, path in enumerate(model_paths):
        model_name = os.path.basename(path)
        model_weights[model_name] = CONFIG['models'][i]['weight']
        print(f"\nLoading {model_name} (weight: {model_weights[model_name]})...")

        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.to(DEVICE)
        model.eval()

        if DEVICE == 'cuda':
            model = model.half()

        dataset = BioticInteractionDataset(val_texts, [0]*len(val_texts), tokenizer)
        loader = DataLoader(dataset, batch_size=CONFIG['eval_batch_size'], shuffle=False)

        probs = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                batch_probs = torch.softmax(outputs.logits.float(), dim=-1)[:, 1]
                probs.extend(batch_probs.cpu().numpy())

        all_probs[model_name] = np.array(probs)

        # Clear GPU memory
        del model
        torch.cuda.empty_cache()

    # Use predefined weights and find best threshold
    labels = np.array(val_labels)
    model_names = list(all_probs.keys())

    # Combine with predefined weights (BiomedBERT 65%, RoBERTa 35%)
    combined = sum(model_weights[name] * all_probs[name] for name in model_names)

    best_config = None
    best_score = 0

    # Find best threshold prioritizing precision
    for thresh in np.arange(0.3, 0.85, 0.05):
        preds = (combined >= thresh).astype(int)
        p = precision_score(labels, preds, zero_division=0)
        r = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        # Score: prioritize precision but require minimum recall
        if r >= 0.30:
            score = p * 0.6 + f1 * 0.4
            if score > best_score:
                best_score = score
                best_config = {
                    'weights': model_weights,
                    'threshold': thresh,
                    'precision': p,
                    'recall': r,
                    'f1': f1,
                }

    # Also test grid search for weights
    print("\nGrid searching for optimal weights...")
    for w1 in np.arange(0.5, 0.85, 0.05):
        w2 = 1 - w1
        combined = w1 * all_probs[model_names[0]] + w2 * all_probs[model_names[1]]

        for thresh in np.arange(0.4, 0.8, 0.05):
            preds = (combined >= thresh).astype(int)
            p = precision_score(labels, preds, zero_division=0)
            r = recall_score(labels, preds, zero_division=0)
            f1 = f1_score(labels, preds, zero_division=0)

            if r >= 0.30:
                score = p * 0.6 + f1 * 0.4
                if score > best_score:
                    best_score = score
                    best_config = {
                        'weights': {model_names[0]: w1, model_names[1]: w2},
                        'threshold': thresh,
                        'precision': p,
                        'recall': r,
                        'f1': f1,
                    }

    print(f"\nBest Ensemble Configuration:")
    print(f"  Weights: {best_config['weights']}")
    print(f"  Threshold: {best_config['threshold']:.2f}")
    print(f"  Precision: {best_config['precision']:.3f}")
    print(f"  Recall: {best_config['recall']:.3f}")
    print(f"  F1: {best_config['f1']:.3f}")

    return best_config


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("HIGH-PRECISION ENSEMBLE TRAINING")
    print("="*70)
    print(f"\nDevice: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Prepare data
    texts, labels = prepare_high_quality_data()

    # Split: train/val/test
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=CONFIG['seed'], stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=CONFIG['seed'], stratify=temp_labels
    )

    print(f"\nData splits:")
    print(f"  Train: {len(train_texts)} ({sum(train_labels)} pos)")
    print(f"  Val: {len(val_texts)} ({sum(val_labels)} pos)")
    print(f"  Test: {len(test_texts)} ({sum(test_labels)} pos)")

    # Train models
    model_paths = []
    for model_config in CONFIG['models']:
        path, results = train_model(
            model_config, train_texts, train_labels, val_texts, val_labels
        )
        model_paths.append(path)

    # Create ensemble
    ensemble_config = create_ensemble_config(model_paths, val_texts, val_labels)

    # Save ensemble config
    import json
    config_path = f'{OUTPUT_DIR}/ensemble_config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'models': {name: path for name, path in zip(
                [os.path.basename(p) for p in model_paths], model_paths
            )},
            **ensemble_config
        }, f, indent=2)

    print(f"\nEnsemble config saved to: {config_path}")

    # Final test evaluation
    print("\n" + "="*70)
    print("FINAL TEST SET EVALUATION")
    print("="*70)

    # Load models and evaluate on test set
    all_probs = {}
    for path in model_paths:
        model_name = os.path.basename(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.to(DEVICE)
        model.eval()
        if DEVICE == 'cuda':
            model = model.half()

        dataset = BioticInteractionDataset(test_texts, [0]*len(test_texts), tokenizer)
        loader = DataLoader(dataset, batch_size=CONFIG['eval_batch_size'], shuffle=False)

        probs = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                batch_probs = torch.softmax(outputs.logits.float(), dim=-1)[:, 1]
                probs.extend(batch_probs.cpu().numpy())

        all_probs[model_name] = np.array(probs)
        del model
        torch.cuda.empty_cache()

    # Ensemble prediction
    weights = ensemble_config['weights']
    combined = sum(w * all_probs[name] for name, w in weights.items())
    preds = (combined >= ensemble_config['threshold']).astype(int)

    print(f"\nTest Set Results:")
    print(f"  Precision: {precision_score(test_labels, preds):.3f}")
    print(f"  Recall: {recall_score(test_labels, preds):.3f}")
    print(f"  F1: {f1_score(test_labels, preds):.3f}")
    print(f"  Accuracy: {accuracy_score(test_labels, preds):.3f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_labels, preds)
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nModels saved to: {OUTPUT_DIR}")
    print(f"Ensemble config: {config_path}")


if __name__ == '__main__':
    main()
