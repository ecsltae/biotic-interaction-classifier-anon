#!/usr/bin/env python3
"""
Multi-task training: interaction classification + species NER.

Usage:
    python train.py --data ../../data/training/training_data_v14.csv \
                    --encoder microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
                    --ner-scheme basic --alpha 0.5 --epochs 5 \
                    --output-dir ../../models/multitask_basic_a05 \
                    --results-dir ../../results/multitask/basic_a05

    # NER-pretrain then fine-tune (two-phase):
    python train.py --data ... --pretrain-ner-epochs 2 --epochs 5 --ner-scheme typed
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).parent))
from model import MultiTaskBiomedBERT
from data import load_multitask_splits

ENCODER       = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
SOFT_LABELS   = "classifier/data/training/distillation_soft_labels.csv"
TEMPERATURE   = 2.0   # same as distilled_v2


def cls_loss_fn(logits: torch.Tensor, hard_labels: torch.Tensor,
                soft_labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Classification loss: soft KL where soft_label >= 0, hard CE otherwise.
    Mirrors distilled_v2: loss = T² * KL(student_soft || teacher_soft).
    Rows with soft_label == -1 fall back to hard CE.
    """
    has_soft = soft_labels >= 0

    loss = torch.tensor(0.0, device=logits.device)
    n = 0

    if has_soft.any():
        sl = soft_labels[has_soft].clamp(1e-6, 1 - 1e-6)
        teacher = torch.stack([1 - sl, sl], dim=1)          # (B, 2)
        student = F.log_softmax(logits[has_soft] / temperature, dim=-1)
        kl = F.kl_div(student, teacher, reduction="batchmean")
        loss = loss + temperature ** 2 * kl
        n += 1

    if (~has_soft).any():
        ce = F.cross_entropy(logits[~has_soft], hard_labels[~has_soft])
        loss = loss + ce
        n += 1

    return loss / max(n, 1)


# ── Evaluation helpers ────────────────────────────────────────────────────

def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    cls_preds, cls_true = [], []
    ner_preds, ner_true = [], []
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(
                input_ids      = batch["input_ids"],
                attention_mask = batch["attention_mask"],
                token_type_ids = batch["token_type_ids"],
                cls_labels     = batch["cls_label"],
                ner_labels     = batch["ner_labels"],
            )

            if out["loss"] is not None:
                total_loss += out["loss"].item()
                n += 1

            # Classification
            probs = torch.softmax(out["cls_logits"], dim=-1)[:, 1].cpu().numpy()
            cls_preds.extend((probs >= threshold).astype(int).tolist())
            cls_true.extend(batch["cls_label"].cpu().numpy().tolist())

            # NER (flatten, ignore -100)
            ner_logits = out["ner_logits"]           # (B, L, n_ner)
            ner_pred   = ner_logits.argmax(-1)       # (B, L)
            nl = batch["ner_labels"]                  # (B, L)
            mask = nl != -100
            ner_preds.extend(ner_pred[mask].cpu().numpy().tolist())
            ner_true.extend(nl[mask].cpu().numpy().tolist())

    cls_f1  = f1_score(cls_true, cls_preds, zero_division=0)
    cls_prec = precision_score(cls_true, cls_preds, zero_division=0)
    cls_rec  = recall_score(cls_true, cls_preds, zero_division=0)

    # NER F1 (entity-level micro over non-O labels)
    ner_f1_micro = f1_score(ner_true, ner_preds, average="micro", labels=list(range(1, model.n_ner)), zero_division=0)

    avg_loss = total_loss / n if n else 0.0
    return {
        "loss": avg_loss,
        "cls_f1": cls_f1,
        "cls_prec": cls_prec,
        "cls_rec": cls_rec,
        "ner_f1": ner_f1_micro,
    }


# ── Threshold sweep on val ────────────────────────────────────────────────

def find_best_threshold(model, loader, device):
    model.eval()
    all_probs, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(
                input_ids      = batch["input_ids"],
                attention_mask = batch["attention_mask"],
                token_type_ids = batch["token_type_ids"],
            )
            probs = torch.softmax(out["cls_logits"], dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_true.extend(batch["cls_label"].cpu().numpy().tolist())

    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.02):
        preds = (np.array(all_probs) >= t).astype(int)
        f = f1_score(all_true, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t, best_f1


# ── Training loop ─────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Data
    print(f"Loading data from {args.data} ...", flush=True)
    soft_path = args.soft_labels if Path(args.soft_labels).exists() else None
    if soft_path:
        print(f"Soft labels: {soft_path}", flush=True)
    else:
        print("WARNING: no soft labels found — using hard CE for classification", flush=True)

    train_ds, val_ds = load_multitask_splits(
        args.data, args.encoder, args.ner_scheme,
        val_frac=0.1, max_length=args.max_length,
        soft_labels_path=soft_path,
    )
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model
    model = MultiTaskBiomedBERT(args.encoder, args.ner_scheme, args.alpha).to(device)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)

    history = []
    best_cls_f1 = 0.0
    best_state = None

    # ── Phase 1: NER pre-training (freeze cls_head, unfreeze ner_head + encoder) ──
    if args.pretrain_ner_epochs > 0:
        print(f"\n=== NER pre-train ({args.pretrain_ner_epochs} epochs) ===", flush=True)
        # Freeze classification head
        for p in model.cls_head.parameters():
            p.requires_grad = False

        ner_opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr
        )
        for ep in range(args.pretrain_ner_epochs):
            model.train()
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(
                    input_ids      = batch["input_ids"],
                    attention_mask = batch["attention_mask"],
                    token_type_ids = batch["token_type_ids"],
                    ner_labels     = batch["ner_labels"],
                )
                if out["loss"] is not None:
                    out["loss"].backward()
                    ner_opt.step()
                    ner_opt.zero_grad()
            metrics = evaluate(model, val_loader, device)
            print(f"  NER pretrain ep {ep+1}: NER F1={metrics['ner_f1']:.4f}", flush=True)

        # Unfreeze cls_head for joint training
        for p in model.cls_head.parameters():
            p.requires_grad = True

    # ── Phase 2: joint training ───────────────────────────────────────────
    print(f"\n=== Joint training ({args.epochs} epochs, α={args.alpha}) ===", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1,
        total_iters=args.epochs * len(train_loader)
    )

    for ep in range(args.epochs):
        model.train()
        ep_start = time.time()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(
                input_ids      = batch["input_ids"],
                attention_mask = batch["attention_mask"],
                token_type_ids = batch["token_type_ids"],
                ner_labels     = batch["ner_labels"],
                # Don't pass cls_labels — we compute cls loss manually below
            )

            # Classification loss: soft KL (with T=2) where available, hard CE fallback
            c_loss = cls_loss_fn(
                out["cls_logits"], batch["cls_label"],
                batch["soft_label"], temperature=args.temperature
            )
            # NER loss (already computed in forward if ner_labels passed)
            n_loss = out["ner_loss"] if out["ner_loss"] is not None else torch.tensor(0.0, device=device)

            loss = args.alpha * c_loss + (1 - args.alpha) * n_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - ep_start

        print(
            f"  Ep {ep+1}/{args.epochs}  "
            f"loss={metrics['loss']:.4f}  "
            f"cls_F1={metrics['cls_f1']:.4f}  "
            f"cls_P={metrics['cls_prec']:.3f}  cls_R={metrics['cls_rec']:.3f}  "
            f"ner_F1={metrics['ner_f1']:.4f}  "
            f"({elapsed:.0f}s)",
            flush=True
        )
        history.append({"epoch": ep + 1, **metrics})

        if metrics["cls_f1"] > best_cls_f1:
            best_cls_f1 = metrics["cls_f1"]
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Threshold optimisation
    best_t, best_f1_t = find_best_threshold(model, val_loader, device)
    print(f"\nBest threshold on val: {best_t:.2f} → F1={best_f1_t:.4f}", flush=True)

    # Save model
    model.save(str(output_dir))
    print(f"Model saved to {output_dir}", flush=True)

    # Save results
    summary = {
        "encoder": args.encoder,
        "ner_scheme": args.ner_scheme,
        "alpha": args.alpha,
        "temperature": args.temperature,
        "soft_labels": args.soft_labels,
        "epochs": args.epochs,
        "pretrain_ner_epochs": args.pretrain_ner_epochs,
        "best_val_cls_f1": best_cls_f1,
        "best_threshold": best_t,
        "history": history,
    }
    with open(results_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {results_dir}", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-task BiomedBERT trainer")
    parser.add_argument("--data",       required=True, help="Path to training CSV")
    parser.add_argument("--encoder",    default=ENCODER, help="HF encoder name or local path")
    parser.add_argument("--ner-scheme", default="full", choices=["basic", "typed", "full", "full_typed"])
    parser.add_argument("--alpha",      type=float, default=0.5, help="cls loss weight (1-alpha → NER)")
    parser.add_argument("--epochs",     type=int, default=5)
    parser.add_argument("--pretrain-ner-epochs", type=int, default=0,
                        help="Extra NER-only pre-train epochs before joint training")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--soft-labels",  default=SOFT_LABELS,
                        help="CSV with p_ensemble column for KL distillation loss")
    parser.add_argument("--temperature",  type=float, default=TEMPERATURE,
                        help="Distillation temperature (default 2.0)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
