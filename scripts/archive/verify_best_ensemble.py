#!/usr/bin/env python3
"""Verify the BERT + T5-v12 ensemble result with full confusion matrix."""

from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset
import pandas as pd

BASE_DIR = Path('/path/to/MetaP/classifier')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BIOMEDBERT = BASE_DIR / 'models/transformer_BiomedBERT_cv_regularized'
T5_V12 = BASE_DIR / 'models/flan-t5-base_v12'
T5_V11_1 = BASE_DIR / 'models/flan-t5-base_v11_1'
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


def main():
    df = pd.read_csv(EP_TEST, sep='\t')
    texts = df['sentence'].astype(str).tolist()
    y = np.array(df['evaluation_pair_interacting'].astype(int).tolist())
    print(f"EP test: {len(texts)} samples, {sum(y)} positives\n")
    
    print("Loading models...")
    p_bert = get_bert_probs(BIOMEDBERT, texts)
    p_t5_v12 = get_t5_probs(T5_V12, texts)
    p_t5_v11 = get_t5_probs(T5_V11_1, texts)
    
    # Test various thresholds for BERT + v12
    print("\n" + "="*70)
    print("BERT + v12 THRESHOLD SWEEP")
    print("="*70)
    geom_v12 = np.sqrt(p_bert * p_t5_v12)
    for t in [0.25, 0.30, 0.32, 0.35, 0.40]:
        preds = (geom_v12 >= t).astype(int)
        f1 = f1_score(y, preds)
        prec = precision_score(y, preds, zero_division=0)
        rec = recall_score(y, preds, zero_division=0)
        cm = confusion_matrix(y, preds)
        tn, fp, fn, tp = cm.ravel()
        print(f"  @{t:.2f}: F1={f1:.3f}  P={prec:.3f}  R={rec:.3f}  | TN={tn} FP={fp} FN={fn} TP={tp}")
    
    # Compare to BERT + v11_1 at same threshold
    print("\n" + "="*70)
    print("BERT + v11_1 (previous best)")
    print("="*70)
    geom_v11 = np.sqrt(p_bert * p_t5_v11)
    for t in [0.30, 0.32, 0.33, 0.35]:
        preds = (geom_v11 >= t).astype(int)
        f1 = f1_score(y, preds)
        prec = precision_score(y, preds, zero_division=0)
        rec = recall_score(y, preds, zero_division=0)
        cm = confusion_matrix(y, preds)
        tn, fp, fn, tp = cm.ravel()
        print(f"  @{t:.2f}: F1={f1:.3f}  P={prec:.3f}  R={rec:.3f}  | TN={tn} FP={fp} FN={fn} TP={tp}")
    
    # Save for analysis
    np.savez(
        BASE_DIR / 'results/research_agent/ensemble_probs_comparison.npz',
        p_bert=p_bert,
        p_t5_v12=p_t5_v12,
        p_t5_v11=p_t5_v11,
        y=y
    )
    print(f"\nProbs saved → results/research_agent/ensemble_probs_comparison.npz")


if __name__ == '__main__':
    main()
