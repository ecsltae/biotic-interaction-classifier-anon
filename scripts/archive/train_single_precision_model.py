#!/usr/bin/env python3
"""
Train Single High-Precision Model
=================================

Lighter alternative to the ensemble - trains one BiomedBERT model
with precision-focused settings.

Faster training, less GPU memory, still good results.
"""

import os
import time
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = '/path/to/MetaP/classifier'
DATA_FILE = f'{BASE_DIR}/data/training/training_data_precision.csv'
OUTPUT_DIR = f'{BASE_DIR}/models/biomedbert_precision_single'

CONFIG = {
    'seed': 42,
    'model_name': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
    'max_length': 256,
    'batch_size': 32,          # Adjust based on GPU memory
    'epochs': 5,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'pos_weight': 0.7,         # Penalize false positives
    'focal_gamma': 2.0,
}

random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")


# =============================================================================
# DATASET & LOSS
# =============================================================================

class TextDataset(Dataset):
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


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class PrecisionTrainer(Trainer):
    def __init__(self, *args, pos_weight=1.0, focal_gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        weights = torch.tensor([1.0, pos_weight]).to(DEVICE)
        self.loss_fn = FocalLoss(gamma=focal_gamma, alpha=weights)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(pred):
    labels = pred.label_ids
    probs = torch.softmax(torch.tensor(pred.predictions), dim=-1)[:, 1].numpy()

    # Find best threshold for precision
    best_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'threshold': 0.5}
    best_score = 0

    for thresh in np.arange(0.3, 0.8, 0.05):
        preds = (probs >= thresh).astype(int)
        p = precision_score(labels, preds, zero_division=0)
        r = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        # Prioritize precision
        if r >= 0.3:
            score = p * 0.6 + f1 * 0.4
            if score > best_score:
                best_score = score
                best_metrics = {'precision': p, 'recall': r, 'f1': f1, 'threshold': thresh}

    return best_metrics


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("TRAINING SINGLE PRECISION MODEL")
    print("="*60)

    # Check if precision dataset exists, if not create it
    if not os.path.exists(DATA_FILE):
        print("\nPrecision dataset not found. Creating...")
        exec(open(f'{BASE_DIR}/scripts/prepare_precision_dataset.py').read())

    # Load data
    print("\nLoading data...")
    df = pd.read_csv(DATA_FILE)
    print(f"Dataset: {len(df)} samples ({sum(df['label']==1)} pos, {sum(df['label']==0)} neg)")

    # Split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['passage'].tolist(), df['label'].tolist(),
        test_size=0.1, random_state=CONFIG['seed'], stratify=df['label'].tolist()
    )
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")

    # Load model
    print(f"\nLoading {CONFIG['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = AutoModelForSequenceClassification.from_pretrained(CONFIG['model_name'], num_labels=2)

    # Datasets
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, CONFIG['max_length'])
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, CONFIG['max_length'])

    # Training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=CONFIG['epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'] * 2,
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        warmup_ratio=CONFIG['warmup_ratio'],
        eval_strategy='steps',
        eval_steps=100,
        save_strategy='steps',
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model='precision',
        greater_is_better=True,
        fp16=DEVICE == 'cuda',
        logging_steps=50,
        report_to='none',
        save_total_limit=2,
    )

    # Train
    trainer = PrecisionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        pos_weight=CONFIG['pos_weight'],
        focal_gamma=CONFIG['focal_gamma'],
    )

    print("\nTraining...")
    start = time.time()
    trainer.train()
    print(f"Done in {(time.time()-start)/60:.1f} minutes")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nSaved to: {OUTPUT_DIR}")

    # Final eval
    print("\nFinal Validation:")
    results = trainer.evaluate()
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()
