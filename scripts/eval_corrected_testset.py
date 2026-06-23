#!/usr/bin/env python3
"""
Re-evaluate all reported models on the corrected (deduplicated) 403-sentence
test set. Uses EXISTING validation-derived thresholds (no re-tuning on test).

Usage:
    python classifier/scripts/eval_corrected_testset.py --gpu
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, T5ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "classifier/experiments/multitask"))
from model import MultiTaskBiomedBERT  # noqa: E402

TEST_SET = ROOT / "classifier/data/evaluation/biotic_interaction_test_set.csv"
OUT_DIR = ROOT / "classifier/results/new_testset"
N_BOOT, SEED, ALPHA = 10_000, 42, 0.05

PROMPT_TEMPLATE = (
    "Does this sentence describe a biotic interaction between two organisms?\n"
    "Sentence: {sentence}\n"
    "Answer:"
)


def predict_multitask(model_path, texts, device):
    cfg = json.load(open(model_path / "multitask_config.json"))
    model = MultiTaskBiomedBERT.load(str(model_path), device=str(device))
    model.eval()
    tok = AutoTokenizer.from_pretrained(cfg["encoder_name"])
    probs = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i + 32]
        enc = tok(batch, truncation=True, max_length=256, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids"))
        probs.extend(torch.softmax(out["cls_logits"], -1)[:, 1].cpu().tolist())
    del model
    torch.cuda.empty_cache()
    return np.array(probs)


def predict_seq_cls(model_path, texts, device):
    tok = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True).to(device).eval()
    probs = []
    for i in range(0, len(texts), 32):
        batch = [t.lower() for t in texts[i:i + 32]]
        enc = tok(batch, truncation=True, max_length=256, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            p = torch.softmax(model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits, -1)[:, 1].cpu().tolist()
        probs.extend(p)
    del model
    torch.cuda.empty_cache()
    return np.array(probs)


def predict_flan_t5(model_path, texts, device):
    tok = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(str(model_path), local_files_only=True).to(device).eval()
    yes_id = tok.encode("yes", add_special_tokens=False)[0]
    no_id = tok.encode("no", add_special_tokens=False)[0]
    prompts = [PROMPT_TEMPLATE.format(sentence=t) for t in texts]
    scores = []
    with torch.no_grad():
        for i in range(0, len(prompts), 32):
            batch = prompts[i:i + 32]
            enc = tok(batch, max_length=256, padding=True, truncation=True, return_tensors="pt").to(device)
            bos = torch.full((len(batch), 1), model.config.decoder_start_token_id, dtype=torch.long).to(device)
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], decoder_input_ids=bos)
            logits = out.logits[:, 0, :]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            yes_lp = log_probs[:, yes_id].cpu().numpy()
            no_lp = log_probs[:, no_id].cpu().numpy()
            prob_yes = np.exp(yes_lp) / (np.exp(yes_lp) + np.exp(no_lp))
            scores.extend(prob_yes.tolist())
    del model
    torch.cuda.empty_cache()
    return np.array(scores)


def bootstrap_ci(labels, preds, n=N_BOOT, rng=None):
    if rng is None:
        rng = np.random.default_rng(SEED)
    labels, preds = np.array(labels), np.array(preds)
    idx = np.arange(len(labels))
    f1s, ps, rs = [], [], []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if labels[s].sum() == 0 or labels[s].sum() == len(s):
            continue
        f1s.append(f1_score(labels[s], preds[s], zero_division=0))
        ps.append(precision_score(labels[s], preds[s], zero_division=0))
        rs.append(recall_score(labels[s], preds[s], zero_division=0))
    lo, hi = ALPHA / 2 * 100, (1 - ALPHA / 2) * 100
    return {"f1_mean": float(np.mean(f1s)), "f1_ci": [float(np.percentile(f1s, lo)), float(np.percentile(f1s, hi))]}


def mcnemar(labels, a, b):
    y, a, b = np.array(labels), np.array(a), np.array(b)
    n01 = int(((a == y) & (b != y)).sum())
    n10 = int(((a != y) & (b == y)).sum())
    n_d = n01 + n10
    if n_d == 0:
        return {"n01": n01, "n10": n10, "chi2": 0.0, "p_value": 1.0}
    stat = (abs(n01 - n10) - 1) ** 2 / n_d
    return {"n01": n01, "n10": n10, "n_disagree": n_d, "chi2": round(float(stat), 3), "p_value": round(float(1 - chi2.cdf(stat, df=1)), 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    df = pd.read_csv(TEST_SET)
    texts = df["sentence"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    print(f"Test set: {len(texts)} sentences, {sum(labels)} positives ({100*sum(labels)/len(labels):.1f}%)\n")

    models = {
        "champion_warm_ner0":  {"path": ROOT / "classifier/models/multitask/mt_distill_warm_ner0", "kind": "mt", "t": 0.360},
        "warm_ner2":           {"path": ROOT / "classifier/models/multitask/mt_distill_warm_ner2", "kind": "mt", "t": 0.160},
        "cold_start_champion": {"path": ROOT / "classifier/models/multitask/mt_cold_start_champion", "kind": "mt", "t": 0.140},
        "hardce":              {"path": ROOT / "classifier/models/multitask/mt_hardce", "kind": "mt", "t": 0.360},
        "distilled_v2":        {"path": ROOT / "classifier/models/distilled_BiomedBERT_v2", "kind": "seqcls", "t": 0.250},
        "template_trained":    {"path": ROOT / "classifier/models/transformer_BiomedBERT_cv_regularized", "kind": "seqcls", "t": 0.500},
    }

    probs = {}
    for name, cfg in models.items():
        if not cfg["path"].exists():
            print(f"  SKIP {name} — not found at {cfg['path']}")
            continue
        print(f"  Inferring {name} ...", flush=True)
        fn = predict_multitask if cfg["kind"] == "mt" else predict_seq_cls
        probs[name] = fn(cfg["path"], texts, device)

    # Ensemble = geometric mean(template_trained, flan-t5-base_v12)
    flant5_path = ROOT / "classifier/models/flan-t5-base_v12"
    print("  Inferring flan-t5-base_v12 (for ensemble) ...", flush=True)
    flant5_probs = predict_flan_t5(flant5_path, texts, device)
    probs["ensemble"] = np.sqrt(probs["template_trained"] * flant5_probs)
    models["ensemble"] = {"t": 0.320}

    # Point metrics
    point = {}
    preds = {}
    print("\nPoint metrics:")
    for name, p in probs.items():
        t = models[name]["t"]
        pred = (p >= t).astype(int)
        preds[name] = pred
        f1 = f1_score(labels, pred, zero_division=0)
        prec = precision_score(labels, pred, zero_division=0)
        rec = recall_score(labels, pred, zero_division=0)
        auc = roc_auc_score(labels, p)
        point[name] = {"f1": round(float(f1), 4), "precision": round(float(prec), 4), "recall": round(float(rec), 4),
                       "auc": round(float(auc), 4), "threshold": t}
        print(f"  {name:<22} τ={t:.3f}  F1={f1:.4f}  P={prec:.4f}  R={rec:.4f}  AUC={auc:.4f}")

    # Bootstrap CIs
    print(f"\nBootstrap 95% CI (n={N_BOOT:,}):")
    rng = np.random.default_rng(SEED)
    ci = {}
    for name, pred in preds.items():
        c = bootstrap_ci(labels, pred, rng=rng)
        ci[name] = c
        print(f"  {name:<22} F1={c['f1_mean']:.4f}  [{c['f1_ci'][0]:.4f}, {c['f1_ci'][1]:.4f}]")

    # Key McNemar comparisons
    print("\nMcNemar:")
    comparisons = [
        ("champion_warm_ner0", "cold_start_champion"),
        ("champion_warm_ner0", "hardce"),
        ("champion_warm_ner0", "template_trained"),
        ("champion_warm_ner0", "ensemble"),
        ("warm_ner2", "champion_warm_ner0"),
        ("warm_ner2", "cold_start_champion"),
    ]
    mcn = {}
    for a, b in comparisons:
        if a not in preds or b not in preds:
            continue
        r = mcnemar(labels, preds[a], preds[b])
        mcn[f"{a}_vs_{b}"] = r
        sig = "SIGNIFICANT" if r["p_value"] < 0.05 else "n.s."
        print(f"  {a} vs {b}: chi2={r['chi2']:.3f} p={r['p_value']:.4f} ({sig})  n01={r['n01']} n10={r['n10']}")

    out = {"n_total": len(texts), "n_pos": int(sum(labels)), "point_metrics": point, "bootstrap_ci": ci, "mcnemar": mcn}
    out_file = OUT_DIR / "corrected_testset_results.json"
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_file}")


if __name__ == "__main__":
    main()
