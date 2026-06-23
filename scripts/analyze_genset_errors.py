#!/usr/bin/env python3
"""Analyze gen_set_100 errors by category and difficulty."""

from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration
from torch.utils.data import DataLoader, Dataset
import pandas as pd

BASE_DIR = Path('/path/to/MetaP/classifier')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BIOMEDBERT = BASE_DIR / 'models/transformer_BiomedBERT_cv_regularized'
T5_V12 = BASE_DIR / 'models/flan-t5-base_v12'
GEN_SET = BASE_DIR / 'data/evaluation/gen_set_100.csv'


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
    df = pd.read_csv(GEN_SET)
    texts = df['sentence'].astype(str).tolist()
    y = df['label'].values

    print(f"Loading models and predicting...")
    p_bert = get_bert_probs(BIOMEDBERT, texts)
    p_t5 = get_t5_probs(T5_V12, texts)

    # Geometric mean ensemble
    geom = np.sqrt(p_bert * p_t5)

    # Use threshold 0.14 (F1-optimal from earlier analysis)
    threshold = 0.14
    preds = (geom >= threshold).astype(int)

    df['p_bert'] = p_bert
    df['p_t5'] = p_t5
    df['p_ens'] = geom
    df['pred'] = preds
    df['error_type'] = 'correct'
    df.loc[(y == 1) & (preds == 0), 'error_type'] = 'FN'
    df.loc[(y == 0) & (preds == 1), 'error_type'] = 'FP'

    # Analysis
    print(f"\n{'='*80}")
    print(f"GEN_SET_100 ERROR ANALYSIS (threshold={threshold})")
    print(f"{'='*80}\n")

    print("ERROR BREAKDOWN BY CATEGORY:")
    print("-" * 60)
    for cat in sorted(df['category'].unique()):
        cat_df = df[df['category'] == cat]
        n = len(cat_df)
        n_pos = cat_df['label'].sum()
        fn = ((cat_df['label'] == 1) & (cat_df['pred'] == 0)).sum()
        fp = ((cat_df['label'] == 0) & (cat_df['pred'] == 1)).sum()
        if fn > 0 or fp > 0:
            print(f"  {cat:<18} n={n:>2}  pos={n_pos:>2}  FN={fn:>2}  FP={fp:>2}")

    print("\nERROR BREAKDOWN BY DIFFICULTY:")
    print("-" * 60)
    for diff in ['easy', 'medium', 'hard']:
        diff_df = df[df['difficulty'] == diff]
        n = len(diff_df)
        n_pos = diff_df['label'].sum()
        fn = ((diff_df['label'] == 1) & (diff_df['pred'] == 0)).sum()
        fp = ((diff_df['label'] == 0) & (diff_df['pred'] == 1)).sum()
        print(f"  {diff:<10} n={n:>2}  pos={n_pos:>2}  FN={fn:>2}  FP={fp:>2}")

    print("\n" + "="*80)
    print("FALSE NEGATIVES (20 missed positives):")
    print("="*80)
    fns = df[df['error_type'] == 'FN'].sort_values('p_ens')
    for _, row in fns.iterrows():
        print(f"\n[{row['category']}/{row['difficulty']}] p_ens={row['p_ens']:.3f} (BERT={row['p_bert']:.3f}, T5={row['p_t5']:.3f})")
        print(f"  {row['sentence'][:150]}...")

    print("\n" + "="*80)
    print("FALSE POSITIVES (2 incorrectly predicted):")
    print("="*80)
    fps = df[df['error_type'] == 'FP'].sort_values('p_ens', ascending=False)
    for _, row in fps.iterrows():
        print(f"\n[{row['category']}/{row['difficulty']}] p_ens={row['p_ens']:.3f} (BERT={row['p_bert']:.3f}, T5={row['p_t5']:.3f})")
        print(f"  {row['sentence'][:150]}...")

    # Save detailed results
    out_path = BASE_DIR / 'results/research_agent/gen_set_100_analysis.csv'
    df.to_csv(out_path, index=False)
    print(f"\nDetailed results saved → {out_path}")


if __name__ == '__main__':
    main()
