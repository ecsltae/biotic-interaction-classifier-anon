#!/usr/bin/env python3
"""
Optimized Ensemble Classifier for Biotic Interactions
======================================================
Best configuration found through exhaustive search (v2):
- BiomedBERT (6k original): 25% weight
- BiomedBERT (20k enhanced): 25% weight
- BiomedBERT (quality_v2): 25% weight
- BioBERT: 25% weight
- Threshold: 0.50

Performance on eval100: F1=0.524, P=0.579, R=0.478
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from typing import List, Dict, Tuple

# Best configuration (v2 - 4-model ensemble)
OPTIMIZED_CONFIG = {
    'biomedbert_6k': {
        'model_path': '/path/to/MetaP/classifier/models/transformer_BiomedBERT_model_6k_original',
        'weight': 0.25,
        'max_length': 256,
    },
    'biomedbert_20k': {
        'model_path': '/path/to/MetaP/classifier/models/transformer_BiomedBERT_model_enhanced_20k',
        'weight': 0.25,
        'max_length': 256,
    },
    'biomedbert_quality_v2': {
        'model_path': '/path/to/MetaP/classifier/models/transformer_BiomedBERT_quality_v2',
        'weight': 0.25,
        'max_length': 256,
    },
    'biobert': {
        'model_path': '/path/to/MetaP/classifier/models/transformer_biobert_model',
        'weight': 0.25,
        'max_length': 256,
    },
}

BEST_THRESHOLD = 0.50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
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
        }


class OptimizedEnsemble:
    """
    Optimized 4-model ensemble for biotic interaction classification.

    Best performance on eval100:
    - F1: 0.524
    - Precision: 0.579
    - Recall: 0.478
    """

    def __init__(self, config=OPTIMIZED_CONFIG, device=DEVICE, use_fp16=True):
        self.config = config
        self.device = device
        self.use_fp16 = use_fp16 and device == 'cuda'
        self.models = {}
        self.tokenizers = {}

        print(f"Initializing Optimized Ensemble on {device}")
        print(f"FP16: {self.use_fp16}")

        # Load models
        for model_key, model_config in config.items():
            print(f"\nLoading {model_key}...")

            tokenizer = AutoTokenizer.from_pretrained(model_config['model_path'])
            model = AutoModelForSequenceClassification.from_pretrained(model_config['model_path'])

            if self.use_fp16:
                model = model.half()

            model.to(device)
            model.eval()

            self.tokenizers[model_key] = tokenizer
            self.models[model_key] = model
            print(f"  ✓ {model_key} loaded (weight: {model_config['weight']})")

        # Store normalized weights
        total_weight = sum(c['weight'] for c in config.values())
        self.weights = {k: c['weight'] / total_weight for k, c in config.items()}
        print(f"\n✓ Ensemble ready with weights: {self.weights}")

    @torch.no_grad()
    def predict_proba(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Get probability scores from ensemble"""
        all_probs = {}

        for model_key, model in self.models.items():
            tokenizer = self.tokenizers[model_key]
            max_length = self.config[model_key]['max_length']

            dataset = TextDataset(texts, tokenizer, max_length)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            model_probs = []
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits.float(), dim=-1)[:, 1]
                model_probs.extend(probs.cpu().numpy())

            all_probs[model_key] = np.array(model_probs)

        # Weighted average
        ensemble_probs = np.zeros(len(texts))
        for model_key, probs in all_probs.items():
            ensemble_probs += self.weights[model_key] * probs

        return ensemble_probs

    def predict(self, texts: List[str], threshold: float = BEST_THRESHOLD,
                batch_size: int = 32) -> np.ndarray:
        """Predict class labels"""
        probs = self.predict_proba(texts, batch_size)
        return (probs >= threshold).astype(int)

    def predict_single(self, text: str, threshold: float = BEST_THRESHOLD) -> Tuple[int, float]:
        """Predict single sentence"""
        prob = self.predict_proba([text])[0]
        label = int(prob >= threshold)
        return label, float(prob)

    def evaluate(self, texts: List[str], labels: List[int],
                 threshold: float = BEST_THRESHOLD) -> Dict:
        """Evaluate on test data"""
        probs = self.predict_proba(texts)
        preds = (probs >= threshold).astype(int)

        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, zero_division=0),
            'recall': recall_score(labels, preds, zero_division=0),
            'f1': f1_score(labels, preds, zero_division=0),
            'threshold': threshold,
        }

        print(f"\nEvaluation Results (threshold={threshold}):")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")

        return metrics


if __name__ == '__main__':
    import pandas as pd

    # Test on eval100
    print("="*70)
    print("TESTING OPTIMIZED ENSEMBLE ON EVAL100")
    print("="*70)

    # Load eval100
    eval_df = pd.read_csv('/path/to/MetaP/classifier/data/evaluation/eval_100.tsv', sep='\t')
    sentences = eval_df['sentence'].tolist()
    labels = eval_df['evaluation_pair_interacting'].tolist()

    # Create ensemble
    ensemble = OptimizedEnsemble()

    # Evaluate
    metrics = ensemble.evaluate(sentences, labels, threshold=BEST_THRESHOLD)

    # Also test a few thresholds
    print("\nThreshold sweep:")
    for t in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        m = ensemble.evaluate(sentences, labels, threshold=t)
        print(f"  t={t}: F1={m['f1']:.3f}, P={m['precision']:.3f}, R={m['recall']:.3f}")

    print("\n" + "="*70)
