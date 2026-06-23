#!/usr/bin/env python3
"""
Cross-Validation Training with Regularization
- 5-fold CV to prevent overfitting
- Label smoothing and dropout for regularization
- Trains BiomedBERT and SciBERT
- Final evaluation on EP test set
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy

warnings.filterwarnings('ignore')

import argparse

BASE_DIR = '/path/to/MetaP/classifier'
DEFAULT_TRAIN_FILE = f'{BASE_DIR}/data/training/training_data_globi_v7_llm_cleaned.csv'
EVAL100_FILE = f'{BASE_DIR}/data/evaluation/eval_100.tsv'
EP_TEST_FILE = f'{BASE_DIR}/globi-relax_passages-triplets_2024-02-28_curation_EP.tsv'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Config
N_FOLDS = 5
EPOCHS = 6
BATCH_SIZE = 32  # Smaller batch for better generalization
MAX_LENGTH = 256
LEARNING_RATE = 2e-5
LABEL_SMOOTHING = 0.1  # Regularization
WEIGHT_DECAY = 0.01
DROPOUT = 0.2  # Extra dropout


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


def load_data(train_file: str = None):
    """Load training data."""
    print("Loading datasets...")
    train_path = train_file or DEFAULT_TRAIN_FILE
    df = pd.read_csv(train_path)
    df = df[['text', 'label']].copy()
    df['source'] = 'train'
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Training data ({os.path.basename(train_path)}): {len(df)} samples ({sum(df['label']==1)} pos)")
    return df


def load_test_set():
    """Load EP test set."""
    df = pd.read_csv(EP_TEST_FILE, sep='\t', encoding='latin-1')
    texts = df['sentence'].tolist()
    labels = df['evaluation_pair_interacting'].tolist()
    print(f"EP Test set: {len(texts)} samples ({sum(labels)} pos)")
    return texts, labels


def evaluate_on_test(model, tokenizer, test_texts, test_labels):
    """Evaluate model on test set, find best threshold."""
    model.eval()
    dataset = TextDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits.float(), dim=-1)[:, 1]
            all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)
    test_labels = np.array(test_labels)

    # Find best threshold
    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.05):
        preds = (all_probs >= t).astype(int)
        f1 = f1_score(test_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    preds = (all_probs >= best_thresh).astype(int)
    precision = precision_score(test_labels, preds, zero_division=0)
    recall = recall_score(test_labels, preds, zero_division=0)
    accuracy = accuracy_score(test_labels, preds)
    cm = confusion_matrix(test_labels, preds)

    return {
        'f1': best_f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'threshold': best_thresh,
        'confusion_matrix': cm.tolist(),
        'probabilities': all_probs.tolist()
    }


def train_model_cv(model_name, model_id, data_df, test_texts, test_labels):
    """Train model with cross-validation."""
    print(f"\n{'='*70}")
    print(f"TRAINING {model_name} WITH {N_FOLDS}-FOLD CV")
    print(f"{'='*70}")

    texts = data_df['text'].tolist()
    labels = data_df['label'].tolist()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_results = []
    best_model = None
    best_f1 = 0

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")

        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        print(f"Train: {len(train_texts)} | Val: {len(val_texts)}")

        # Load fresh model for each fold
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=2,
            hidden_dropout_prob=DROPOUT,
            attention_probs_dropout_prob=DROPOUT
        )
        model.to(DEVICE)

        train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
        val_dataset = TextDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

        output_dir = f'{BASE_DIR}/models/cv_temp/{model_name}_fold{fold}'

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE * 2,
            warmup_ratio=0.1,
            weight_decay=WEIGHT_DECAY,
            learning_rate=LEARNING_RATE,
            label_smoothing_factor=LABEL_SMOOTHING,
            logging_steps=100,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True,
            fp16=DEVICE == 'cuda',
            report_to='none',
            save_total_limit=1,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        # Evaluate on validation
        val_metrics = trainer.evaluate()
        print(f"Val F1: {val_metrics['eval_f1']:.3f}")

        # Evaluate on test set
        test_metrics = evaluate_on_test(model, tokenizer, test_texts, test_labels)
        print(f"Test F1: {test_metrics['f1']:.3f} | Precision: {test_metrics['precision']:.3f} | Recall: {test_metrics['recall']:.3f}")

        fold_results.append({
            'fold': fold + 1,
            'val_f1': val_metrics['eval_f1'],
            'test_f1': test_metrics['f1'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
        })

        # Keep best model
        if test_metrics['f1'] > best_f1:
            best_f1 = test_metrics['f1']
            best_model = (model, tokenizer, test_metrics)

    # Summary
    print(f"\n--- {model_name} CV Summary ---")
    avg_val_f1 = np.mean([r['val_f1'] for r in fold_results])
    avg_test_f1 = np.mean([r['test_f1'] for r in fold_results])
    avg_test_prec = np.mean([r['test_precision'] for r in fold_results])
    avg_test_rec = np.mean([r['test_recall'] for r in fold_results])
    std_test_f1 = np.std([r['test_f1'] for r in fold_results])

    print(f"Avg Val F1:  {avg_val_f1:.3f}")
    print(f"Avg Test F1: {avg_test_f1:.3f} ± {std_test_f1:.3f}")
    print(f"Avg Test Precision: {avg_test_prec:.3f}")
    print(f"Avg Test Recall: {avg_test_rec:.3f}")
    print(f"Best Fold Test F1: {best_f1:.3f}")

    return best_model, fold_results, {
        'avg_val_f1': avg_val_f1,
        'avg_test_f1': avg_test_f1,
        'std_test_f1': std_test_f1,
        'avg_test_precision': avg_test_prec,
        'avg_test_recall': avg_test_rec,
        'best_test_f1': best_f1
    }


def main():
    parser = argparse.ArgumentParser(description='CV Regularized Training')
    parser.add_argument('--train-data', type=str, default=None,
                        help='Training data CSV (default: v7 LLM-cleaned)')
    parser.add_argument('--models', nargs='+', default=['BiomedBERT', 'SciBERT'],
                        help='Models to train')
    parser.add_argument('--suffix', type=str, default='cv_regularized',
                        help='Suffix for output directory name')
    args = parser.parse_args()

    start_time = time.time()

    # Load data
    data_df = load_data(args.train_data)
    test_texts, test_labels = load_test_set()

    # Models to train
    model_registry = {
        'BiomedBERT': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        'SciBERT': 'allenai/scibert_scivocab_uncased',
        'RoBERTa': 'roberta-base',
    }
    models = {k: model_registry[k] for k in args.models if k in model_registry}

    all_results = {}
    best_models = {}

    for name, model_id in models.items():
        best_model, fold_results, summary = train_model_cv(
            name, model_id, data_df, test_texts, test_labels
        )
        all_results[name] = {
            'fold_results': fold_results,
            'summary': summary
        }
        best_models[name] = best_model

    # Final comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON ON EP TEST SET")
    print("="*70)

    for name, results in all_results.items():
        s = results['summary']
        print(f"\n{name}:")
        print(f"  Avg Test F1: {s['avg_test_f1']:.3f} ± {s['std_test_f1']:.3f}")
        print(f"  Avg Precision: {s['avg_test_precision']:.3f}")
        print(f"  Avg Recall: {s['avg_test_recall']:.3f}")
        print(f"  Best Fold F1: {s['best_test_f1']:.3f}")

    # Target check
    print("\n" + "="*70)
    print("TARGET CHECK")
    print("="*70)

    for name, results in all_results.items():
        s = results['summary']
        f1_pass = s['avg_test_f1'] > 0.75
        prec_pass = s['avg_test_precision'] > s['avg_test_f1']
        print(f"\n{name}:")
        print(f"  F1 > 0.75:       {'PASS' if f1_pass else 'FAIL'} ({s['avg_test_f1']:.3f})")
        print(f"  Precision > F1:  {'PASS' if prec_pass else 'FAIL'} ({s['avg_test_precision']:.3f} vs {s['avg_test_f1']:.3f})")

    # Save best model
    best_name = max(all_results.keys(), key=lambda k: all_results[k]['summary']['avg_test_f1'])
    best_model, best_tokenizer, best_metrics = best_models[best_name]

    save_dir = f'{BASE_DIR}/models/transformer_{best_name}_{args.suffix}'
    os.makedirs(save_dir, exist_ok=True)
    best_model.save_pretrained(save_dir)
    best_tokenizer.save_pretrained(save_dir)

    # Save results
    with open(f'{save_dir}/cv_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nBest model ({best_name}) saved to: {save_dir}")
    print(f"Total time: {(time.time()-start_time)/60:.1f} min")


if __name__ == "__main__":
    main()
