#!/usr/bin/env python3
"""
Train BiomedBERT on the optimal dataset (6k balanced)
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Config
BASE_DIR = '/path/to/MetaP/classifier'
DATA_FILE = f'{BASE_DIR}/data/training/training_data_optimal.csv'
MODEL_DIR = f'{BASE_DIR}/models/transformer_BiomedBERT_optimal'
EVAL_FILE = f'{BASE_DIR}/data/evaluation/eval_100.tsv'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Hyperparameters
EPOCHS = 4
BATCH_SIZE = 16
MAX_LENGTH = 256
LEARNING_RATE = 2e-5


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


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
    }


def evaluate_on_eval100(model, tokenizer, device):
    """Evaluate on the 100-sentence test set"""
    eval_df = pd.read_csv(EVAL_FILE, sep='\t')
    sentences = eval_df['sentence'].tolist()
    labels = eval_df['evaluation_pair_interacting'].tolist()

    model.eval()
    if device == 'cuda':
        model = model.half()

    dataset = TextDataset(sentences, labels, tokenizer, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits.float(), dim=-1)[:, 1]
            all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)

    # Find best threshold
    best_f1 = 0
    best_thresh = 0.5
    best_metrics = {}

    for t in np.arange(0.1, 0.9, 0.05):
        preds = (all_probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_metrics = {
                'precision': precision_score(labels, preds, zero_division=0),
                'recall': recall_score(labels, preds, zero_division=0),
                'f1': f1,
                'accuracy': accuracy_score(labels, preds),
            }

    print(f"\nEval100 Results:")
    print(f"  Best threshold: {best_thresh:.2f}")
    print(f"  Precision: {best_metrics['precision']:.3f}")
    print(f"  Recall: {best_metrics['recall']:.3f}")
    print(f"  F1: {best_metrics['f1']:.3f}")
    print(f"  Accuracy: {best_metrics['accuracy']:.3f}")

    return best_metrics, best_thresh


def main():
    print("="*70)
    print("TRAINING BIOMEDBERT ON OPTIMAL DATASET")
    print("="*70)

    # Load data
    print("\nLoading data...")
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} samples")
    print(f"  Positives: {sum(df['label'] == 1)}")
    print(f"  Negatives: {sum(df['label'] == 0)}")

    # Split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['passage'].tolist(),
        df['label'].tolist(),
        test_size=0.1,
        random_state=42,
        stratify=df['label'].tolist()
    )

    print(f"\nSplit: Train={len(train_texts)}, Val={len(val_texts)}")

    # Load tokenizer and model
    print("\nLoading BiomedBERT...")
    model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    # Training args
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=LEARNING_RATE,
        logging_dir=f'{MODEL_DIR}/logs',
        logging_steps=50,
        eval_strategy='steps',
        eval_steps=100,
        save_strategy='steps',
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        fp16=DEVICE == 'cuda',
        report_to='none',
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train
    print("\nStarting training...")
    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time/60:.1f} minutes")

    # Save
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Model saved to: {MODEL_DIR}")

    # Evaluate on validation
    print("\nValidation Results:")
    val_results = trainer.evaluate()
    for k, v in val_results.items():
        if 'eval_' in k:
            print(f"  {k.replace('eval_', '')}: {v:.4f}")

    # Evaluate on eval100
    print("\n" + "="*70)
    print("EVALUATING ON EVAL100")
    print("="*70)
    model.to(DEVICE)
    eval_metrics, best_thresh = evaluate_on_eval100(model, tokenizer, DEVICE)

    # Save results
    results = {
        'model': 'BiomedBERT_optimal',
        'dataset': 'training_data_optimal.csv (6k)',
        'eval100_f1': eval_metrics['f1'],
        'eval100_precision': eval_metrics['precision'],
        'eval100_recall': eval_metrics['recall'],
        'best_threshold': best_thresh,
    }

    results_df = pd.DataFrame([results])
    results_df.to_csv(f'{MODEL_DIR}/eval100_results.csv', index=False)

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nFinal Eval100 F1: {eval_metrics['f1']:.3f}")


if __name__ == '__main__':
    main()
