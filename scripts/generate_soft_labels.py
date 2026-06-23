#!/usr/bin/env python3
"""
Generate soft labels for new sentences using BiomedBERT cv_regularized + FLAN-T5-base v12 teachers.
Appends to distillation_soft_labels.csv.
"""
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration

BASE = Path("/path/to/MetaP/classifier")
TEACHER_BERT = BASE / "models/transformer_BiomedBERT_cv_regularized"
TEACHER_T5   = BASE / "models/flan-t5-base_v12"
SOFT_LABELS  = BASE / "data/training/distillation_soft_labels.csv"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_LEN = 256


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=MAX_LEN):
        self.encodings = tokenizer(
            texts, max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )
    def __len__(self): return self.encodings["input_ids"].shape[0]
    def __getitem__(self, idx): return {k: v[idx] for k, v in self.encodings.items()}


def get_bert_probs(model_dir, texts, batch_size=64):
    print(f"  [BERT] Loading {model_dir.name} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), local_files_only=True).to(DEVICE).eval()
    ds = TextDataset(texts, tok)
    loader = DataLoader(ds, batch_size=batch_size)
    probs = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            p = torch.softmax(model(**batch).logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
            if i % 10 == 0:
                print(f"    {len(probs)}/{len(texts)}", flush=True)
    model.cpu(); del model; torch.cuda.empty_cache()
    return np.array(probs)


def get_t5_probs(model_dir, texts, batch_size=32):
    print(f"  [T5] Loading {model_dir.name} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(
        str(model_dir), local_files_only=True).to(DEVICE).eval()
    yes_id = tok.encode("yes", add_special_tokens=False)[0]
    no_id  = tok.encode("no",  add_special_tokens=False)[0]
    prompts = [
        f"Does the following sentence describe a biotic interaction between two species? "
        f"Answer yes or no.\n\nSentence: {t}" for t in texts
    ]
    probs = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        enc = tok(batch_prompts, max_length=MAX_LEN, padding="max_length",
                  truncation=True, return_tensors="pt").to(DEVICE)
        dec_ids = torch.zeros(len(batch_prompts), 1, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            out = model(**enc, decoder_input_ids=dec_ids)
        logits = out.logits[:, 0, :]
        yes_logit = logits[:, yes_id]
        no_logit  = logits[:, no_id]
        p = torch.softmax(torch.stack([no_logit, yes_logit], dim=1), dim=1)[:, 1].cpu().numpy()
        probs.extend(p.tolist())
        if (i // batch_size) % 10 == 0:
            print(f"    {len(probs)}/{len(texts)}", flush=True)
    model.cpu(); del model; torch.cuda.empty_cache()
    return np.array(probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(BASE / "data/training/new_sentences_for_soft_labels.csv"))
    parser.add_argument("--output", default=str(SOFT_LABELS))
    parser.add_argument("--append", action="store_true", default=True)
    parser.add_argument("--batch-size-bert", type=int, default=64)
    parser.add_argument("--batch-size-t5", type=int, default=32)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"New sentences: {len(df)} (pos={( df.hard_label==1).sum()}, neg={(df.hard_label==0).sum()})")

    texts = df.text.tolist()
    print(f"\nDevice: {DEVICE}")

    print("\nRunning BiomedBERT teacher...")
    p_bert = get_bert_probs(TEACHER_BERT, texts, batch_size=args.batch_size_bert)

    print("\nRunning FLAN-T5-base teacher...")
    p_t5 = get_t5_probs(TEACHER_T5, texts, batch_size=args.batch_size_t5)

    # Geometric mean ensemble
    p_ensemble = np.sqrt(p_bert * p_t5)

    result = pd.DataFrame({
        "text": texts,
        "hard_label": df.hard_label.values,
        "p_bert": p_bert,
        "p_t5": p_t5,
        "p_ensemble": p_ensemble,
    })

    print(f"\nSoft label stats:")
    print(f"  p_bert mean={p_bert.mean():.3f} std={p_bert.std():.3f}")
    print(f"  p_t5   mean={p_t5.mean():.3f} std={p_t5.std():.3f}")
    print(f"  p_ens  mean={p_ensemble.mean():.3f} std={p_ensemble.std():.3f}")

    if args.append and Path(args.output).exists():
        existing = pd.read_csv(args.output)
        combined = pd.concat([existing, result], ignore_index=True)
        combined = combined.drop_duplicates(subset="text")
        combined.to_csv(args.output, index=False)
        print(f"\nAppended {len(result)} rows → {len(combined)} total in {args.output}")
    else:
        result.to_csv(args.output, index=False)
        print(f"\nSaved {len(result)} rows to {args.output}")


if __name__ == "__main__":
    main()
