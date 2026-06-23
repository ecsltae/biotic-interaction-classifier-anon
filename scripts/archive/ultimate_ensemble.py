#!/usr/bin/env python3
"""
Ultimate Ensemble: BiomedBERT + Multiple FLAN-T5 versions
Tests if combining BiomedBERT with multi-T5 ensemble beats BiomedBERT + single T5.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset
import pandas as pd

BASE_DIR = Path('/path/to/MetaP/classifier')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BIOMEDBERT = BASE_DIR / 'models/transformer_BiomedBERT_cv_regularized'
T5_MODELS = [
    BASE_DIR / 'models/flan-t5-base_v10.1',
    BASE_DIR / 'models/flan-t5-base_v11_1', 
    BASE_DIR / 'models/flan-t5-base_v12',
    BASE_DIR / 'models/flan-t5-base_v13',
]

EP_TEST = BASE_DIR / 'data/evaluation/globi-relax_passages-triplets_2024-02-28_curation_EP.tsv'


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.encodings = tokenizer(
            texts, max_length=max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
    def __len__(self):
        return self.encodings['input_ids'].shape[0]
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


def get_bert_probs(model_path, texts):
    print(f"  Loading BiomedBERT...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE).eval()
    
    ds = TextDataset(texts, tokenizer)
    dl = DataLoader(ds, batch_size=32)
    
    probs = []
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(**batch)
            p = torch.softmax(out.logits, dim=1)[:, 1].cpu().numpy()
            probs.extend(p)
    
    del model
    torch.cuda.empty_cache()
    return np.array(probs)


def get_t5_probs(model_path, texts):
    print(f"  Loading {model_path.name}...")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(DEVICE).eval()
    
    yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("no", add_special_tokens=False)[0]
    
    probs = []
    with torch.no_grad():
        for text in texts:
            prompt = f"Does this sentence describe a biotic interaction between species? Answer yes or no.\nSentence: {text}"
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384).to(DEVICE)
            out = model.generate(**enc, max_new_tokens=3, return_dict_in_generate=True, output_scores=True)
            logits = out.scores[0][0]
            p_yes = torch.softmax(logits[[yes_id, no_id]], dim=0)[0].item()
            probs.append(p_yes)
    
    del model
    torch.cuda.empty_cache()
    return np.array(probs)


def eval_at_thresh(y_true, probs, thresh):
    preds = (probs >= thresh).astype(int)
    f1 = f1_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    return f1, prec, rec


def find_optimal_thresh(y_true, probs):
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        f1, _, _ = eval_at_thresh(y_true, probs, t)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def main():
    # Load EP test
    df = pd.read_csv(EP_TEST, sep='\t')
    texts = df['sentence'].astype(str).tolist()
    labels = df['evaluation_pair_interacting'].astype(int).tolist()
    y = np.array(labels)
    print(f"EP test: {len(texts)} samples, {sum(labels)} positives\n")
    
    # Get BiomedBERT probs
    p_bert = get_bert_probs(BIOMEDBERT, texts)
    
    # Get T5 probs
    t5_probs = {}
    for mp in T5_MODELS:
        if mp.exists():
            t5_probs[mp.name] = get_t5_probs(mp, texts)
    
    # Individual T5 with BiomedBERT
    print("\n" + "="*70)
    print("BiomedBERT + SINGLE T5 (geometric mean)")
    print("="*70)
    for name, p_t5 in t5_probs.items():
        geom = np.sqrt(p_bert * p_t5)
        t = find_optimal_thresh(y, geom)
        f1, prec, rec = eval_at_thresh(y, geom, t)
        print(f"  BERT + {name:25s}  F1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  @{t:.2f}")
    
    # Multi-T5 ensembles with BiomedBERT
    print("\n" + "="*70)
    print("BiomedBERT + MULTI-T5 ENSEMBLE")
    print("="*70)
    
    t5_list = list(t5_probs.values())
    
    # Arithmetic mean of T5s, then geom with BERT
    t5_arith = np.mean(t5_list, axis=0)
    geom = np.sqrt(p_bert * t5_arith)
    t = find_optimal_thresh(y, geom)
    f1, prec, rec = eval_at_thresh(y, geom, t)
    print(f"  BERT × mean(4×T5)                F1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  @{t:.2f}")
    
    # Geometric mean of T5s, then geom with BERT
    t5_geom = np.exp(np.mean(np.log(np.clip(t5_list, 1e-9, 1)), axis=0))
    geom = np.sqrt(p_bert * t5_geom)
    t = find_optimal_thresh(y, geom)
    f1, prec, rec = eval_at_thresh(y, geom, t)
    print(f"  BERT × geom(4×T5)                F1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  @{t:.2f}")
    
    # Max of T5s (any T5 confident), then geom with BERT
    t5_max = np.max(t5_list, axis=0)
    geom = np.sqrt(p_bert * t5_max)
    t = find_optimal_thresh(y, geom)
    f1, prec, rec = eval_at_thresh(y, geom, t)
    print(f"  BERT × max(4×T5)                 F1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  @{t:.2f}")
    
    # Best two T5s (v10.1, v11_1)
    if 'flan-t5-base_v10.1' in t5_probs and 'flan-t5-base_v11_1' in t5_probs:
        t5_pair = np.sqrt(t5_probs['flan-t5-base_v10.1'] * t5_probs['flan-t5-base_v11_1'])
        geom = np.sqrt(p_bert * t5_pair)
        t = find_optimal_thresh(y, geom)
        f1, prec, rec = eval_at_thresh(y, geom, t)
        print(f"  BERT × geom(v10.1, v11_1)        F1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  @{t:.2f}")
    
    # 5-way geometric (BERT + 4 T5s)
    all_probs = [p_bert] + t5_list
    geom_5 = np.exp(np.mean(np.log(np.clip(all_probs, 1e-9, 1)), axis=0))
    t = find_optimal_thresh(y, geom_5)
    f1, prec, rec = eval_at_thresh(y, geom_5, t)
    print(f"  geom(BERT, 4×T5)                 F1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  @{t:.2f}")
    
    print("\n" + "="*70)
    print("BASELINE COMPARISON")
    print("="*70)
    
    # BiomedBERT alone
    t = find_optimal_thresh(y, p_bert)
    f1, prec, rec = eval_at_thresh(y, p_bert, t)
    print(f"  BiomedBERT only                  F1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  @{t:.2f}")
    
    # Best single T5
    p_best_t5 = t5_probs['flan-t5-base_v11_1']
    t = find_optimal_thresh(y, p_best_t5)
    f1, prec, rec = eval_at_thresh(y, p_best_t5, t)
    print(f"  FLAN-T5-base v11_1 only          F1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  @{t:.2f}")


if __name__ == '__main__':
    main()
