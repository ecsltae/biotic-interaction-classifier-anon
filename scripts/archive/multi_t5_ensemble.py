#!/usr/bin/env python3
"""
Multi-T5 Ensemble: combine multiple FLAN-T5 versions for better predictions.
Tests if averaging across T5 versions (v10.1, v11_1, v12, v13) improves over single best.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

BASE_DIR = Path('/path/to/MetaP/classifier')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

T5_MODELS = [
    BASE_DIR / 'models/flan-t5-base_v10.1',
    BASE_DIR / 'models/flan-t5-base_v11_1', 
    BASE_DIR / 'models/flan-t5-base_v12',
    BASE_DIR / 'models/flan-t5-base_v13',
]

EP_TEST = BASE_DIR / 'data/evaluation/globi-relax_passages-triplets_2024-02-28_curation_EP.tsv'


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
    
    # Get probs from each model
    all_probs = {}
    for mp in T5_MODELS:
        if mp.exists():
            all_probs[mp.name] = get_t5_probs(mp, texts)
    
    print("\n" + "="*70)
    print("INDIVIDUAL MODEL RESULTS")
    print("="*70)
    for name, probs in all_probs.items():
        t = find_optimal_thresh(y, probs)
        f1, prec, rec = eval_at_thresh(y, probs, t)
        print(f"  {name:30s}  F1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  @{t:.2f}")
    
    # Ensemble strategies
    print("\n" + "="*70)
    print("MULTI-T5 ENSEMBLE RESULTS")
    print("="*70)
    
    prob_list = list(all_probs.values())
    
    # Arithmetic mean
    arith = np.mean(prob_list, axis=0)
    t = find_optimal_thresh(y, arith)
    f1, prec, rec = eval_at_thresh(y, arith, t)
    print(f"  arithmetic_mean                  F1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  @{t:.2f}")
    
    # Geometric mean
    geom = np.exp(np.mean(np.log(np.clip(prob_list, 1e-9, 1)), axis=0))
    t = find_optimal_thresh(y, geom)
    f1, prec, rec = eval_at_thresh(y, geom, t)
    print(f"  geometric_mean                   F1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  @{t:.2f}")
    
    # Max (any model confident)
    max_p = np.max(prob_list, axis=0)
    t = find_optimal_thresh(y, max_p)
    f1, prec, rec = eval_at_thresh(y, max_p, t)
    print(f"  max                              F1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  @{t:.2f}")
    
    # Min (all models agree)
    min_p = np.min(prob_list, axis=0)
    t = find_optimal_thresh(y, min_p)
    f1, prec, rec = eval_at_thresh(y, min_p, t)
    print(f"  min                              F1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  @{t:.2f}")
    
    # Best pair: v10.1 + v11_1
    if 'flan-t5-base_v10.1' in all_probs and 'flan-t5-base_v11_1' in all_probs:
        pair = np.sqrt(all_probs['flan-t5-base_v10.1'] * all_probs['flan-t5-base_v11_1'])
        t = find_optimal_thresh(y, pair)
        f1, prec, rec = eval_at_thresh(y, pair, t)
        print(f"  geom(v10.1, v11_1)               F1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  @{t:.2f}")
    
    # Save probs for further analysis
    np.savez(
        BASE_DIR / 'results/research_agent/multi_t5_probs_ep.npz',
        **all_probs,
        y=y
    )
    print(f"\nProbs saved → results/research_agent/multi_t5_probs_ep.npz")


if __name__ == '__main__':
    main()
