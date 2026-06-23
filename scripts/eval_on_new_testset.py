#!/usr/bin/env python3
"""
Evaluate all comparison, ablation, and distillation models on the 499-sentence test set.

Usage:
    python classifier/scripts/eval_on_new_testset.py [--gpu]

Outputs (in classifier/results/new_testset/):
    main_models.json        — Table 2 models: champion, hardce, distilled_v2, single-task, ensemble
    distillation.json       — Appendix A: all distilled_* variants
    ablation.json           — Appendix B: all multitask ablation configs
    probs_main.npz          — Per-sentence probabilities for main models (for bootstrap CI)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    T5ForConditionalGeneration,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "classifier/experiments/multitask"))
from model import MultiTaskBiomedBERT  # noqa: E402

TEST_SET = ROOT / "classifier/data/evaluation/biotic_interaction_test_set.csv"
OUT_DIR  = ROOT / "classifier/results/new_testset"

PROMPT_TEMPLATE = (
    "Does this sentence describe a biotic interaction between two organisms?\n"
    "Sentence: {sentence}\n"
    "Answer:"
)
MAX_INPUT_LEN = 256

# ── Inference helpers ─────────────────────────────────────────────────────────

def predict_multitask(model_path: Path, texts: list, device) -> np.ndarray:
    cfg = json.load(open(model_path / "multitask_config.json"))
    model = MultiTaskBiomedBERT.load(str(model_path), device=str(device))
    model.eval()
    tok = AutoTokenizer.from_pretrained(cfg["encoder_name"])
    probs = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i+32]
        enc = tok(batch, truncation=True, max_length=256, padding=True,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(enc["input_ids"], enc["attention_mask"],
                        enc.get("token_type_ids"))
        probs.extend(torch.softmax(out["cls_logits"], -1)[:, 1].cpu().tolist())
    del model
    torch.cuda.empty_cache()
    return np.array(probs)


def predict_seq_cls(model_path: Path, texts: list, device) -> np.ndarray:
    tok   = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path), local_files_only=True).to(device).eval()
    probs = []
    for i in range(0, len(texts), 32):
        batch = [t.lower() for t in texts[i:i+32]]
        enc = tok(batch, truncation=True, max_length=256, padding=True,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            p = torch.softmax(
                model(input_ids=enc["input_ids"],
                      attention_mask=enc["attention_mask"]).logits,
                -1)[:, 1].cpu().tolist()
        probs.extend(p)
    del model
    torch.cuda.empty_cache()
    return np.array(probs)


def predict_flan_t5(model_path: Path, texts: list, device) -> np.ndarray:
    tok   = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(
        str(model_path), local_files_only=True).to(device).eval()
    yes_id = tok.encode("yes", add_special_tokens=False)[0]
    no_id  = tok.encode("no",  add_special_tokens=False)[0]
    prompts = [PROMPT_TEMPLATE.format(sentence=t) for t in texts]
    scores = []
    with torch.no_grad():
        for i in range(0, len(prompts), 32):
            batch = prompts[i:i+32]
            enc = tok(batch, max_length=MAX_INPUT_LEN, padding=True,
                      truncation=True, return_tensors="pt").to(device)
            bos = torch.full((len(batch), 1), model.config.decoder_start_token_id,
                             dtype=torch.long).to(device)
            out = model(input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                        decoder_input_ids=bos)
            logits = out.logits[:, 0, :]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            yes_lp = log_probs[:, yes_id].cpu().numpy()
            no_lp  = log_probs[:, no_id].cpu().numpy()
            prob_yes = np.exp(yes_lp) / (np.exp(yes_lp) + np.exp(no_lp))
            scores.extend(prob_yes.tolist())
    del model
    torch.cuda.empty_cache()
    return np.array(scores)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(labels: list, probs: np.ndarray, threshold: float) -> dict:
    labels_arr = np.array(labels)
    preds = (probs >= threshold).astype(int)
    f1  = f1_score(labels_arr, preds, zero_division=0)
    p   = precision_score(labels_arr, preds, zero_division=0)
    r   = recall_score(labels_arr, preds, zero_division=0)
    auc = roc_auc_score(labels_arr, probs)
    n_pred_pos = int(preds.sum())
    return {
        "threshold": threshold,
        "f1": round(float(f1), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "auc": round(float(auc), 4),
        "n_pred_pos": n_pred_pos,
        "n_pos": int(labels_arr.sum()),
        "n_total": len(labels_arr),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="Use GPU (CUDA:0)")
    args = parser.parse_args()

    device = torch.device("cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Load test set
    df = pd.read_csv(TEST_SET)
    texts  = df["sentence"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    # Champion scores already in CSV
    champion_scores = df["score"].astype(float).tolist()
    print(f"Test set: {len(texts)} sentences, {sum(labels)} positives", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Main Table 2 models ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("MAIN TABLE 2 MODELS")
    print("="*60)

    main_results = {}
    main_probs   = {"labels": np.array(labels)}

    # 1a. Champion — read from CSV directly
    champ_probs = np.array(champion_scores)
    main_results["multitask_champion"] = compute_metrics(labels, champ_probs, threshold=0.090)
    main_probs["multitask_champion"] = champ_probs
    m = main_results["multitask_champion"]
    print(f"\n  [champion] full_typed_a05_ner2  t=0.090"
          f"  F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  AUC={m['auc']:.4f}")

    # 1b. Hard CE ablation (multitask arch, hard cross-entropy, no soft labels)
    hardce_path = ROOT / "classifier/models/multitask/multitask_v12_hardce"
    print(f"\n  Loading multitask_v12_hardce ...", flush=True)
    hardce_probs = predict_multitask(hardce_path, texts, device)
    main_results["multitask_hardce"] = compute_metrics(labels, hardce_probs, threshold=0.51)
    main_probs["multitask_hardce"] = hardce_probs
    m = main_results["multitask_hardce"]
    print(f"  [hardce]   multitask_v12_hardce  t=0.51"
          f"  F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  AUC={m['auc']:.4f}")

    # 1c. Distilled BiomedBERT v2
    distv2_path = ROOT / "classifier/models/distilled_BiomedBERT_v2"
    print(f"\n  Loading distilled_BiomedBERT_v2 ...", flush=True)
    distv2_probs = predict_seq_cls(distv2_path, texts, device)
    main_results["distilled_BiomedBERT_v2"] = compute_metrics(labels, distv2_probs, threshold=0.25)
    main_probs["distilled_BiomedBERT_v2"] = distv2_probs
    m = main_results["distilled_BiomedBERT_v2"]
    print(f"  [distv2]   distilled_BiomedBERT_v2  t=0.25"
          f"  F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  AUC={m['auc']:.4f}")

    # 1d. Single-task BiomedBERT v7
    singletask_path = ROOT / "classifier/models/transformer_BiomedBERT_cv_regularized"
    print(f"\n  Loading transformer_BiomedBERT_cv_regularized ...", flush=True)
    singletask_probs = predict_seq_cls(singletask_path, texts, device)
    main_results["BiomedBERT_v7_singletask"] = compute_metrics(labels, singletask_probs, threshold=0.50)
    main_probs["BiomedBERT_v7_singletask"] = singletask_probs
    m = main_results["BiomedBERT_v7_singletask"]
    print(f"  [single]   BiomedBERT_v7  t=0.50"
          f"  F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  AUC={m['auc']:.4f}")

    # 1e. Ensemble: BiomedBERT v7 × FLAN-T5-base v12, geometric mean, t=0.32
    flant5_path = ROOT / "classifier/models/flan-t5-base_v12"
    print(f"\n  Loading flan-t5-base_v12 for ensemble ...", flush=True)
    flant5_probs = predict_flan_t5(flant5_path, texts, device)
    # Geometric mean of the two single-model probs
    ensemble_probs = np.sqrt(singletask_probs * flant5_probs)
    main_results["ensemble_BiomedBERT_FLANT5"] = compute_metrics(labels, ensemble_probs, threshold=0.32)
    main_probs["ensemble_BiomedBERT_FLANT5"] = ensemble_probs
    main_probs["flant5_v12"] = flant5_probs
    m = main_results["ensemble_BiomedBERT_FLANT5"]
    print(f"  [ensemble] BiomedBERT×FLAN-T5 geometric  t=0.32"
          f"  F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  AUC={m['auc']:.4f}")

    # Save main results + probs
    with open(OUT_DIR / "main_models.json", "w") as f:
        json.dump(main_results, f, indent=2)
    np.savez_compressed(OUT_DIR / "probs_main.npz", **main_probs)
    print(f"\nSaved main_models.json and probs_main.npz")

    # ── 2. Distillation variants (Appendix A) ──────────────────────────────────
    print("\n" + "="*60)
    print("APPENDIX A — DISTILLATION VARIANTS")
    print("="*60)

    distill_configs = {
        "distilled_BiomedBERT_v1": {"path": ROOT / "classifier/models/distilled_BiomedBERT_v1", "threshold": 0.25},
        "distilled_BiomedBERT_v2": {"path": ROOT / "classifier/models/distilled_BiomedBERT_v2", "threshold": 0.25},
        "distilled_BiomedBERT_v3": {"path": ROOT / "classifier/models/distilled_BiomedBERT_v3", "threshold": 0.25},
        "distilled_DistilBERT_v4": {"path": ROOT / "classifier/models/distilled_DistilBERT_v4", "threshold": 0.25},
        "distilled_SciBERT_v5":    {"path": ROOT / "classifier/models/distilled_SciBERT_v5",    "threshold": 0.25},
        "distilled_BiomedBERT_v6": {"path": ROOT / "classifier/models/distilled_BiomedBERT_v6", "threshold": 0.25},
    }

    distill_results = {}
    for name, cfg in distill_configs.items():
        if not cfg["path"].exists():
            print(f"  SKIP {name} — model not found at {cfg['path']}")
            continue
        print(f"\n  Loading {name} ...", flush=True)
        probs = predict_seq_cls(cfg["path"], texts, device)
        distill_results[name] = compute_metrics(labels, probs, cfg["threshold"])
        m = distill_results[name]
        print(f"  t={cfg['threshold']:.2f}  F1={m['f1']:.4f}  P={m['precision']:.4f}"
              f"  R={m['recall']:.4f}  AUC={m['auc']:.4f}")

    with open(OUT_DIR / "distillation.json", "w") as f:
        json.dump(distill_results, f, indent=2)
    print(f"\nSaved distillation.json")

    # ── 3. Multitask ablation configs (Appendix B) ─────────────────────────────
    print("\n" + "="*60)
    print("APPENDIX B — MULTITASK ABLATION CONFIGS")
    print("="*60)

    # Thresholds from each model's train_summary.json (val-derived)
    ablation_thresholds = {
        "basic_a05":                   0.36,
        "cls_only_a10":                0.25,   # fallback (no train_summary)
        "full_a03":                    0.10,
        "full_a05":                    0.16,
        "full_a05_ner2":               0.16,
        "full_a07":                    0.24,
        "full_typed_a03":              0.16,
        "full_typed_a03_ner2":         0.10,
        "full_typed_a03_ner2_warmstart": 0.16,
        "full_typed_a05":              0.38,
        "full_typed_a05_ner2":         0.090,  # champion — skip (already done)
        "full_typed_a05_ner2_5ep":     0.34,
        "full_typed_a05_ner2_aug":     0.74,
        "full_typed_a05_ner2_posonly":  0.26,
        "full_typed_a05_ner2_v15":     0.16,
        "full_typed_a05_ner2_warmstart": 0.52,
        "kg_enriched_a05":             0.25,   # fallback (no train_summary)
        "multitask_v12_hardce":        0.51,   # already done in main models
        "multitask_v14_hardce":        0.70,
        "multitask_v7_hardce":         0.10,
        "typed_a05":                   0.10,
    }
    # Skip configs already covered in main models table
    skip_configs = {"full_typed_a05_ner2", "multitask_v12_hardce"}

    ablation_results = {}
    multitask_base = ROOT / "classifier/models/multitask"

    for config_name, threshold in ablation_thresholds.items():
        if config_name in skip_configs:
            continue
        model_path = multitask_base / config_name
        if not model_path.exists():
            print(f"  SKIP {config_name} — model not found")
            continue
        print(f"\n  Loading {config_name} ...", flush=True)
        try:
            probs = predict_multitask(model_path, texts, device)
            ablation_results[config_name] = compute_metrics(labels, probs, threshold)
            m = ablation_results[config_name]
            print(f"  t={threshold:.2f}  F1={m['f1']:.4f}  P={m['precision']:.4f}"
                  f"  R={m['recall']:.4f}  AUC={m['auc']:.4f}")
        except Exception as e:
            print(f"  ERROR on {config_name}: {e}")
            ablation_results[config_name] = {"error": str(e)}

    with open(OUT_DIR / "ablation.json", "w") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"\nSaved ablation.json")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY — Main Table 2")
    print("="*60)
    order = ["multitask_champion", "multitask_hardce",
             "distilled_BiomedBERT_v2", "ensemble_BiomedBERT_FLANT5", "BiomedBERT_v7_singletask"]
    for name in order:
        m = main_results.get(name, {})
        print(f"  {name:40s}  F1={m.get('f1','?'):.4f}  P={m.get('precision','?'):.4f}"
              f"  R={m.get('recall','?'):.4f}  AUC={m.get('auc','?'):.4f}")

    print(f"\nAll results saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
