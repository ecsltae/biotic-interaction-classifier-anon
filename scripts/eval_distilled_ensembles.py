#!/usr/bin/env python3
"""
Evaluate combinations of distilled models vs the original ensemble.

Tries: v2 only, v3 only, v2×v3 geo, v2×v3 arith,
       v2×T5 geo, v3×T5 geo, v2×v3×T5 geo,
       v2×orig_BERT geo, v2×v3×orig_BERT×T5 geo
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    T5ForConditionalGeneration,
)

BASE   = Path("/path/to/MetaP/classifier")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUT    = BASE / "results/autonomous_30h"
OUT.mkdir(parents=True, exist_ok=True)

EP_TEST    = BASE / "data/evaluation/globi-relax_passages-triplets_2024-02-28_curation_EP.tsv"
EVAL_100   = BASE / "data/evaluation/eval_100.tsv"
SYNTH_GOLD = BASE / "data/evaluation/synthetic_gold_100.tsv"


# ── Model loaders ──────────────────────────────────────────────────────────

def bert_probs(model_path, texts):
    tok = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    m = AutoModelForSequenceClassification.from_pretrained(
        str(model_path), local_files_only=True).to(DEVICE).eval()
    probs = []
    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
            probs.append(torch.softmax(m(**enc).logits, dim=-1)[0, 1].item())
    m.cpu(); del m; torch.cuda.empty_cache()
    return np.array(probs)


def t5_probs(model_path, texts):
    tok = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    m = T5ForConditionalGeneration.from_pretrained(
        str(model_path), local_files_only=True).to(DEVICE).eval()
    yes_id = tok.encode("yes", add_special_tokens=False)[0]
    no_id  = tok.encode("no",  add_special_tokens=False)[0]
    probs = []
    with torch.no_grad():
        for t in texts:
            prompt = (f"Does the following sentence describe a biotic interaction "
                      f"between two species? Answer yes or no.\n\nSentence: {t}")
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
            dec = torch.full((1, 1), tok.pad_token_id, dtype=torch.long, device=DEVICE)
            logits = m(**enc, decoder_input_ids=dec).logits[0, 0].float()
            lp = torch.log_softmax(logits, dim=-1)
            y, n = lp[yes_id].item(), lp[no_id].item()
            probs.append(np.exp(y) / (np.exp(y) + np.exp(n)))
    m.cpu(); del m; torch.cuda.empty_cache()
    return np.array(probs)


def best_f1(probs, labels):
    best, bt = 0.0, 0.5
    for t in np.arange(0.1, 0.9, 0.05):
        f = f1_score(labels, (probs >= t).astype(int), zero_division=0)
        if f > best:
            best, bt = f, t
    preds = (probs >= bt).astype(int)
    pr = precision_score(labels, preds, zero_division=0)
    rc = recall_score(labels, preds, zero_division=0)
    return {"f1": round(best, 3), "prec": round(pr, 3), "rec": round(rc, 3), "thr": round(bt, 2)}


def load_test(path, text_col, label_col, sep="\t"):
    df = pd.read_csv(path, sep=sep)
    return df[text_col].astype(str).tolist(), df[label_col].astype(int).tolist()


# ── Load all test sets ────────────────────────────────────────────────────

ep_texts,    ep_labels    = load_test(EP_TEST,    "sentence", "evaluation_pair_interacting")
e100_texts,  e100_labels  = load_test(EVAL_100,   "sentence", "evaluation_pair_interacting")
synth_texts, synth_labels = load_test(SYNTH_GOLD, "text",     "label")

print("=== Getting model probabilities ===", flush=True)

# Distilled v2 (T=2, α=0.5) — best EP single model
print("  distilled_v2...", flush=True)
p_v2_ep   = bert_probs(BASE / "models/distilled_BiomedBERT_v2", ep_texts)
p_v2_e100 = bert_probs(BASE / "models/distilled_BiomedBERT_v2", e100_texts)
p_v2_syn  = bert_probs(BASE / "models/distilled_BiomedBERT_v2", synth_texts)

# Distilled v3 (T=4, α=0.9) — best eval_100 single model
print("  distilled_v3...", flush=True)
p_v3_ep   = bert_probs(BASE / "models/distilled_BiomedBERT_v3", ep_texts)
p_v3_e100 = bert_probs(BASE / "models/distilled_BiomedBERT_v3", e100_texts)
p_v3_syn  = bert_probs(BASE / "models/distilled_BiomedBERT_v3", synth_texts)

# Original teacher BERT (cv_regularized, v7 knowledge)
print("  teacher_bert...", flush=True)
p_tb_ep   = bert_probs(BASE / "models/transformer_BiomedBERT_cv_regularized", ep_texts)
p_tb_e100 = bert_probs(BASE / "models/transformer_BiomedBERT_cv_regularized", e100_texts)
p_tb_syn  = bert_probs(BASE / "models/transformer_BiomedBERT_cv_regularized", synth_texts)

# Teacher T5
print("  teacher_t5...", flush=True)
p_t5_ep   = t5_probs(BASE / "models/flan-t5-base_v12", ep_texts)
p_t5_e100 = t5_probs(BASE / "models/flan-t5-base_v12", e100_texts)
p_t5_syn  = t5_probs(BASE / "models/flan-t5-base_v12", synth_texts)

print("\n=== Evaluating combinations ===", flush=True)

combos = {
    "distilled_v2":           (p_v2_ep,   p_v2_e100,   p_v2_syn),
    "distilled_v3":           (p_v3_ep,   p_v3_e100,   p_v3_syn),
    "v2×v3_geo":              (np.sqrt(p_v2_ep   * p_v3_ep),   np.sqrt(p_v2_e100 * p_v3_e100), np.sqrt(p_v2_syn * p_v3_syn)),
    "v2×v3_arith":            ((p_v2_ep   + p_v3_ep)   / 2,    (p_v2_e100 + p_v3_e100) / 2,   (p_v2_syn + p_v3_syn) / 2),
    "v2×T5_geo":              (np.sqrt(p_v2_ep   * p_t5_ep),   np.sqrt(p_v2_e100 * p_t5_e100), np.sqrt(p_v2_syn * p_t5_syn)),
    "v3×T5_geo":              (np.sqrt(p_v3_ep   * p_t5_ep),   np.sqrt(p_v3_e100 * p_t5_e100), np.sqrt(p_v3_syn * p_t5_syn)),
    "v2×v3×T5_geo":           ((p_v2_ep   * p_v3_ep   * p_t5_ep)   ** (1/3), (p_v2_e100 * p_v3_e100 * p_t5_e100) ** (1/3), (p_v2_syn * p_v3_syn * p_t5_syn) ** (1/3)),
    "orig_BERT×T5_geo":       (np.sqrt(p_tb_ep   * p_t5_ep),   np.sqrt(p_tb_e100 * p_t5_e100), np.sqrt(p_tb_syn * p_t5_syn)),
    "v2×orig_BERT_geo":       (np.sqrt(p_v2_ep   * p_tb_ep),   np.sqrt(p_v2_e100 * p_tb_e100), np.sqrt(p_v2_syn * p_tb_syn)),
    "v2×v3×orig_BERT×T5_geo": ((p_v2_ep   * p_v3_ep   * p_tb_ep   * p_t5_ep)   ** 0.25,
                                (p_v2_e100 * p_v3_e100 * p_tb_e100 * p_t5_e100) ** 0.25,
                                (p_v2_syn  * p_v3_syn  * p_tb_syn  * p_t5_syn)  ** 0.25),
}

results = {}
for name, (p_ep, p_e100, p_syn) in combos.items():
    ep_r   = best_f1(p_ep,   ep_labels)
    e100_r = best_f1(p_e100, e100_labels)
    syn_r  = best_f1(p_syn,  synth_labels)
    results[name] = {"ep_relax": ep_r, "eval_100": e100_r, "synthetic_gold": syn_r}
    print(f"  {name:<32} EP={ep_r['f1']:.3f}  eval100={e100_r['f1']:.3f}  synth={syn_r['f1']:.3f}", flush=True)

with open(OUT / "ensemble_comparison.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {OUT}/ensemble_comparison.json")
