#!/usr/bin/env python3
"""
evaluate_all_ep.py — Multi-benchmark evaluation for biotic interaction classifiers.

Runs any FLAN-T5 model against all available EP curation files and the
synthetic gen_set_100 test set, reporting precision, recall, and F1 at
F1-optimal, precision-optimal (Rec≥0.65), and fixed thresholds.

Usage:
    # Evaluate the simple FLAN-T5 (flan_t5_v12)
    python classifier/scripts/evaluate_all_ep.py \
        --model classifier/models/flan_t5_v12

    # Evaluate the enriched model
    python classifier/scripts/evaluate_all_ep.py \
        --model classifier/models/flan_t5_enriched

    # Fixed threshold (skip sweep)
    python classifier/scripts/evaluate_all_ep.py \
        --model classifier/models/flan_t5_v12 \
        --threshold 0.6

    # Save results to JSON
    python classifier/scripts/evaluate_all_ep.py \
        --model classifier/models/flan_t5_v12 \
        --output classifier/results/ep_expanded_baseline.json
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

warnings.filterwarnings("ignore")

BASE_DIR = Path("/path/to/MetaP/classifier")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── benchmark definitions ───────────────────────────────────────────────────────

BENCHMARKS = {
    "ep_relax": {
        "path": BASE_DIR / "data/evaluation/globi-relax_passages-triplets_2024-02-28_curation_EP.tsv",
        "sep": "\t",
        "text_col": "sentence",
        "label_col": "evaluation_pair_interacting",
        "primary": True,
        "description": "EP-relax (primary): GloBI-relax retrieval, mixed difficulty",
    },
    "ep_passage": {
        "path": BASE_DIR / "data/evaluation/globi-passage_passages-triplets_2024-02-28_curation_EP.tsv",
        "sep": "\t",
        "text_col": "sentence",
        "label_col": "evaluation_pair_interacting",
        "primary": False,
        "description": "EP-passage: GloBI-passage retrieval, high positive rate",
    },
    "eval_100": {
        "path": BASE_DIR / "data/evaluation/biotx-random_passages-triplets_2024-02-28_curation_EP_100original.tsv",
        "sep": "\t",
        "text_col": "sentence",
        "label_col": "evaluation_pair_interacting",
        "primary": False,
        "description": "eval_100: BiotXplorer random sample (= historical eval_100.tsv)",
    },
    "biotx_50_best": {
        "path": BASE_DIR / "data/evaluation/biotx-random_passages-triplets_2024-04-22b_curation_EP_50best-multiples.tsv",
        "sep": "\t",
        "text_col": "sentence",
        "label_col": "evaluation_pair_interacting",
        "primary": False,
        "description": "BiotXplorer 2024 best-multiples (harder cases)",
    },
    "biotx_50_nodup": {
        "path": BASE_DIR / "data/evaluation/biotx-random_passages-triplets_2024-05-15_curation_EP_50nomultiple.tsv",
        "sep": "\t",
        "text_col": "sentence",
        "label_col": "evaluation_pair_interacting",
        "primary": False,
        "description": "BiotXplorer 2024 no-duplicate (harder cases)",
    },
    "gen_set_100": {
        "path": BASE_DIR / "data/evaluation/gen_set_100.csv",
        "sep": ",",
        "text_col": "sentence",
        "label_col": "label",
        "primary": False,
        "description": "Synthetic gen_set_100: explicit sentences by category & difficulty",
    },
}

# ── prompt templates ────────────────────────────────────────────────────────────

SIMPLE_PROMPT = (
    "Does this sentence describe a biotic interaction between two organisms?\n"
    "Sentence: {sentence}\n"
    "Answer:"
)

ENRICHED_PROMPT = (
    "Does this sentence describe a biotic interaction between two organisms?\n"
    "Sentence: {sentence}\n"
    "Species detected: {species}\n"
    "Interaction terms found: {terms}\n"
    "Answer:"
)


def build_prompt(sentence: str, enriched: bool) -> str:
    if enriched:
        return ENRICHED_PROMPT.format(sentence=sentence, species="none", terms="none")
    return SIMPLE_PROMPT.format(sentence=sentence)


# ── model loading ───────────────────────────────────────────────────────────────

def load_model(model_path: Path) -> tuple:
    """Load FLAN-T5 model + tokenizer; detect if enriched."""
    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))
    model.to(DEVICE)
    model.eval()

    is_enriched = (model_path / "enriched_config.json").exists()
    if is_enriched:
        print("  → Enriched model detected (uses species/terms context at inference: 'none')")
    else:
        print("  → Simple model detected (plain prompt)")

    yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id  = tokenizer.encode("no",  add_special_tokens=False)[0]
    return model, tokenizer, yes_id, no_id, is_enriched


# ── scoring ─────────────────────────────────────────────────────────────────────

def score_sentences(
    model,
    tokenizer,
    texts: list,
    yes_id: int,
    no_id: int,
    enriched: bool,
    batch_size: int = 32,
    max_input_len: int = 320,
) -> np.ndarray:
    """Return P(yes) for each sentence via forced first-token decoding."""
    prompts = [build_prompt(t, enriched) for t in texts]
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            enc = tokenizer(
                batch,
                max_length=max_input_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(DEVICE)

            bos = torch.full(
                (len(batch), 1),
                model.config.decoder_start_token_id,
                dtype=torch.long,
            ).to(DEVICE)

            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                decoder_input_ids=bos,
            )
            logits = out.logits[:, 0, :]  # first decoder token
            lp = torch.log_softmax(logits.float(), dim=-1)
            yes_lp = lp[:, yes_id].cpu().numpy()
            no_lp  = lp[:, no_id].cpu().numpy()
            prob_yes = np.exp(yes_lp) / (np.exp(yes_lp) + np.exp(no_lp))
            all_probs.extend(prob_yes.tolist())

    return np.array(all_probs)


# ── threshold sweep ─────────────────────────────────────────────────────────────

def sweep_thresholds(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Find F1-optimal and Prec-optimal (Rec≥0.65) thresholds."""
    best_f1, best_f1_t = 0.0, 0.5
    best_prec, best_prec_t = 0.0, 0.5

    for t in np.arange(0.05, 0.96, 0.01):
        preds = (probs >= t).astype(int)
        if preds.sum() == 0:
            continue
        f1   = f1_score(labels, preds, zero_division=0)
        prec = precision_score(labels, preds, zero_division=0)
        rec  = recall_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_f1_t = f1, t
        if rec >= 0.65 and prec > best_prec:
            best_prec, best_prec_t = prec, t

    return {"f1_optimal": best_f1_t, "prec_optimal": best_prec_t}


def compute_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    cm   = confusion_matrix(labels, preds, labels=[0, 1])
    return {
        "threshold": round(threshold, 3),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "n_positive_predicted": int(preds.sum()),
    }


# ── per-group breakdown (gen_set_100) ───────────────────────────────────────────

def breakdown_gen_set(df: pd.DataFrame, probs: np.ndarray, threshold: float) -> dict:
    """Compute F1 per difficulty and per category for gen_set_100."""
    df = df.copy()
    df["prob"] = probs
    df["pred"] = (probs >= threshold).astype(int)

    results = {}
    for col in ["difficulty", "category"]:
        if col not in df.columns:
            continue
        groups = {}
        for val, grp in df.groupby(col):
            if len(grp) == 0:
                continue
            f1 = f1_score(grp["label"], grp["pred"], zero_division=0)
            prec = precision_score(grp["label"], grp["pred"], zero_division=0)
            rec  = recall_score(grp["label"], grp["pred"], zero_division=0)
            groups[str(val)] = {
                "n": len(grp),
                "pos": int(grp["label"].sum()),
                "f1": round(f1, 3),
                "precision": round(prec, 3),
                "recall": round(rec, 3),
            }
        results[col] = groups
    return results


# ── main evaluation ─────────────────────────────────────────────────────────────

def evaluate_benchmark(
    name: str,
    cfg: dict,
    model,
    tokenizer,
    yes_id: int,
    no_id: int,
    enriched: bool,
    fixed_threshold: float = None,
) -> dict:
    path = cfg["path"]
    if not path.exists():
        print(f"  [SKIP] {name}: file not found at {path}")
        return None

    df = pd.read_csv(str(path), sep=cfg["sep"])
    texts  = df[cfg["text_col"]].fillna("").tolist()
    labels = df[cfg["label_col"]].astype(int).values

    n_pos = int(labels.sum())
    n_total = len(labels)
    print(f"\n  {name} ({n_total} samples, {n_pos} pos [{100*n_pos/n_total:.0f}%])")

    probs = score_sentences(model, tokenizer, texts, yes_id, no_id, enriched)

    thresholds = sweep_thresholds(probs, labels)

    result = {
        "name": name,
        "description": cfg["description"],
        "n": n_total,
        "n_pos": n_pos,
        "pct_pos": round(100 * n_pos / n_total, 1),
    }

    if fixed_threshold is not None:
        result["fixed"] = compute_metrics(probs, labels, fixed_threshold)
    result["f1_optimal"] = compute_metrics(probs, labels, thresholds["f1_optimal"])
    result["prec_optimal"] = compute_metrics(probs, labels, thresholds["prec_optimal"])

    # Gen-set breakdown
    if name == "gen_set_100":
        result["breakdown"] = breakdown_gen_set(df, probs, thresholds["f1_optimal"])

    result["probabilities"] = [round(p, 4) for p in probs.tolist()]

    return result


def print_summary_table(results: list, fixed_threshold: float = None) -> None:
    """Print a compact comparison table."""
    print("\n" + "=" * 100)
    print("EVALUATION SUMMARY")
    print("=" * 100)
    header = f"{'Benchmark':<22} {'N':>5} {'Pos%':>5}  "
    if fixed_threshold is not None:
        fixed_col = f"Fixed({fixed_threshold:.2f})"
        header += f"{fixed_col:>14}  Prec   Rec    F1  |  "
    header += f"{'F1-opt thresh':>13}  Prec   Rec    F1  |  {'Prec-opt':>8}  Prec   Rec    F1"
    print(header)
    print("-" * 100)

    for r in results:
        if r is None:
            continue
        line = f"{r['name']:<22} {r['n']:>5} {r['pct_pos']:>4.0f}%  "
        if fixed_threshold is not None:
            fx = r.get("fixed", {})
            line += f"  {fx.get('threshold',0):.2f}  {fx.get('precision',0):.3f}  {fx.get('recall',0):.3f}  {fx.get('f1',0):.3f}  |  "
        fo = r["f1_optimal"]
        line += f"     {fo['threshold']:.2f}  {fo['precision']:.3f}  {fo['recall']:.3f}  {fo['f1']:.3f}  |  "
        po = r["prec_optimal"]
        line += f"     {po['threshold']:.2f}  {po['precision']:.3f}  {po['recall']:.3f}  {po['f1']:.3f}"

        primary_marker = "  ★" if any(BENCHMARKS[r["name"]]["primary"]
                                      for r in [r] if r["name"] in BENCHMARKS) else ""
        print(line + primary_marker)

    print("=" * 100)
    print("★ = primary benchmark (ep_relax)")


def print_gen_set_breakdown(result: dict) -> None:
    if "breakdown" not in result:
        return
    bd = result["breakdown"]
    print("\n  gen_set_100 breakdown by difficulty:")
    if "difficulty" in bd:
        for diff, m in sorted(bd["difficulty"].items()):
            print(f"    {diff:<10} n={m['n']:>3}  pos={m['pos']:>2}  "
                  f"Prec={m['precision']:.3f}  Rec={m['recall']:.3f}  F1={m['f1']:.3f}")
    print("\n  gen_set_100 breakdown by category:")
    if "category" in bd:
        for cat, m in sorted(bd["category"].items()):
            print(f"    {cat:<22} n={m['n']:>3}  pos={m['pos']:>2}  "
                  f"Prec={m['precision']:.3f}  Rec={m['recall']:.3f}  F1={m['f1']:.3f}")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-benchmark EP evaluation for FLAN-T5 models")
    parser.add_argument("--model", required=True, help="Path to trained FLAN-T5 model directory")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Fixed threshold to report (in addition to optimal sweep)")
    parser.add_argument("--output", default=None,
                        help="Save results to JSON file")
    parser.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS.keys()),
                        choices=list(BENCHMARKS.keys()),
                        help="Which benchmarks to run (default: all)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model path not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    model, tokenizer, yes_id, no_id, is_enriched = load_model(model_path)

    print(f"\nUsing device: {DEVICE}")
    print(f"Running {len(args.benchmarks)} benchmarks: {args.benchmarks}")

    all_results = []
    for name in args.benchmarks:
        if name not in BENCHMARKS:
            print(f"  [WARN] Unknown benchmark: {name}")
            continue
        r = evaluate_benchmark(
            name, BENCHMARKS[name], model, tokenizer,
            yes_id, no_id, is_enriched,
            fixed_threshold=args.threshold,
        )
        all_results.append(r)

    # Remove None entries (skipped benchmarks)
    all_results = [r for r in all_results if r is not None]

    print_summary_table(all_results, fixed_threshold=args.threshold)

    # Print gen_set breakdown
    for r in all_results:
        if r and r["name"] == "gen_set_100":
            print_gen_set_breakdown(r)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Remove raw probabilities from saved JSON to keep file small
        save_results = []
        for r in all_results:
            r_copy = {k: v for k, v in r.items() if k != "probabilities"}
            save_results.append(r_copy)
        with open(out_path, "w") as f:
            json.dump({
                "model": str(model_path),
                "is_enriched": is_enriched,
                "device": DEVICE,
                "benchmarks": save_results,
            }, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
