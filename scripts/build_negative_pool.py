#!/usr/bin/env python3
"""
Step 2: Build a clean + curated negative pool from teacher-labeled NO sentences.

Strategy:
  A) Score all NO-labeled sentences with the interaction lexicon.
  B) Definitely clean (teacher=0, lexicon=0): sample up to MAX_CLEAN_NEG rows directly.
  C) Weak signal (teacher=0, 0 < lexicon < STRONG_THRESHOLD): save to CSV for spot-check.
  D) Strong signal (teacher=0, lexicon >= STRONG_THRESHOLD): re-run through Qwen with a
     confidence prompt. Those confirmed as clearly NO (confidence >= CONF_CLEAN) go into the
     clean pool; borderline/uncertain cases are exported for user curation.

Output files:
  classifier/data/training/negatives_clean.csv           — confirmed clean negatives
  classifier/data/training/negatives_weak_signal.csv     — weak-signal, assumed clean (spot-check)
  classifier/data/training/uncertain_negatives_for_curation.csv — needs human review

Usage:
    python scripts/build_negative_pool.py
    python scripts/build_negative_pool.py --max-clean 12000 --strong-threshold 0.4
"""

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
LABELED_FILE = BASE_DIR / "results/research_agent/all_sources_qwen122b_labeled.csv"
OUT_DIR = BASE_DIR / "data/training"
CLEAN_OUT = OUT_DIR / "negatives_clean.csv"
WEAK_OUT = OUT_DIR / "negatives_weak_signal.csv"
UNCERTAIN_OUT = OUT_DIR / "uncertain_negatives_for_curation.csv"
CONF_CHECKPOINT = OUT_DIR / "negatives_confidence_checkpoint.csv"

MODEL = "qwen3.5:122b"
OLLAMA_URL = "http://localhost:11434/api/chat"
CHECKPOINT_EVERY = 50

# Thresholds
STRONG_THRESHOLD = 0.4   # lexicon strength ≥ this → binary re-check with Qwen

BINARY_PROMPT_TEMPLATE = (
    "Does this sentence describe a direct biotic interaction between two named organisms? "
    "Biotic interactions include: predation, parasitism, pollination, herbivory, mutualism, "
    "symbiosis, seed dispersal, competition, pathogen infection, or disease transmission. "
    "The sentence must describe an actual interaction occurring, not just mention organisms. "
    "Answer YES or NO only.\n\n"
    "Sentence: {sentence}"
)


def ask_recheck(sentence: str, model: str = MODEL, timeout: int = 300) -> tuple[int, str]:
    """Second-pass binary YES/NO re-check. Returns (label, raw_response): 1=YES, 0=NO, -1=error."""
    prompt = BINARY_PROMPT_TEMPLATE.format(sentence=sentence)
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "keep_alive": -1,
                "options": {"temperature": 0, "num_predict": 10, "num_ctx": 2048},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json().get("message", {}).get("content", "").strip()
        clean = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', raw).strip().upper()
        if re.match(r'^YES\b', clean):
            return 1, raw
        elif re.match(r'^NO\b', clean):
            return 0, raw
        if 'YES' in clean[:20] and 'NO' not in clean[:20]:
            return 1, raw
        elif 'NO' in clean[:20]:
            return 0, raw
        return -1, raw
    except requests.Timeout:
        return -1, "TIMEOUT"
    except Exception as e:
        return -1, f"ERROR:{e}"


def score_all_negatives(df_neg: pd.DataFrame) -> pd.DataFrame:
    """Add lexicon_score and lexicon_terms columns. Fast — no Qwen calls."""
    sys.path.insert(0, str(BASE_DIR / "src"))
    from data.interaction_lexicon import score_sentence  # type: ignore

    scores, terms = [], []
    for text in df_neg["text"]:
        has_signal, strength, matched = score_sentence(str(text).lower())
        scores.append(strength)
        terms.append("|".join(matched) if matched else "")

    df_neg = df_neg.copy()
    df_neg["lexicon_score"] = scores
    df_neg["lexicon_terms"] = terms
    return df_neg


def run_recheck(df_strong: pd.DataFrame, model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-run strong-signal negatives with the same binary YES/NO prompt.

    Sentences that flip from NO → YES are potential false negatives → curate.
    Sentences that confirm NO a second time → confirmed clean.
    Returns (confirmed_clean_df, curate_df).
    Resumes from CONF_CHECKPOINT if it exists.
    """
    already_done: set[str] = set()
    results: list[dict] = []

    if CONF_CHECKPOINT.exists():
        done_df = pd.read_csv(CONF_CHECKPOINT)
        already_done = set(done_df["text"].astype(str))
        results = done_df.to_dict("records")
        print(f"  Recheck checkpoint: {len(already_done)} already done")

    todo = df_strong[~df_strong["text"].isin(already_done)].reset_index(drop=True)
    n = len(todo)
    print(f"  Re-checking {n} strong-signal negatives (binary YES/NO pass 2)...")

    t0 = time.time()
    for i, row in todo.iterrows():
        label2, raw = ask_recheck(row["text"], model=model)
        results.append({**row.to_dict(), "qwen_recheck": label2, "qwen_recheck_response": raw[:200]})

        if (i + 1) % CHECKPOINT_EVERY == 0 or i == n - 1:
            pd.DataFrame(results).to_csv(CONF_CHECKPOINT, index=False)
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n - i - 1) / rate / 60 if rate > 0 else 0
            print(f"    [{i+1}/{n}] {rate:.1f} rows/s | ETA {eta:.0f} min", flush=True)

    df_result = pd.DataFrame(results)

    # Flipped to YES → false negative candidate → needs curation
    curate = df_result[df_result["qwen_recheck"] == 1].copy()
    # Confirmed NO again → clean negative
    confirmed_clean = df_result[df_result["qwen_recheck"] == 0].copy()
    # Error / unclear → add to curate (conservative)
    unclear = df_result[df_result["qwen_recheck"] == -1].copy()
    curate = pd.concat([curate, unclear], ignore_index=True)

    print(f"  Confirmed NO (clean): {len(confirmed_clean)} | Flipped YES (curate): {len(curate)}")
    return confirmed_clean, curate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-clean", type=int, default=12000,
                        help="Max definitely-clean negatives to keep (lexicon=0)")
    parser.add_argument("--strong-threshold", type=float, default=STRONG_THRESHOLD,
                        help="Lexicon strength ≥ this triggers confidence re-check")
    parser.add_argument("--skip-recheck", action="store_true",
                        help="Skip Qwen confidence re-check (export strong-signal as uncertain)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load teacher-labeled data
    print(f"Loading {LABELED_FILE.name}...")
    df = pd.read_csv(LABELED_FILE)
    df_neg = df[df["teacher_label"] == 0].copy()
    print(f"  Total NO-labeled: {len(df_neg)}")

    # Score with lexicon
    print("Scoring with interaction lexicon...")
    df_neg = score_all_negatives(df_neg)
    print(f"  lexicon=0 (definitely clean): {(df_neg.lexicon_score == 0).sum()}")
    print(f"  0 < lexicon < {args.strong_threshold} (weak signal): "
          f"{((df_neg.lexicon_score > 0) & (df_neg.lexicon_score < args.strong_threshold)).sum()}")
    print(f"  lexicon >= {args.strong_threshold} (strong signal, re-check): "
          f"{(df_neg.lexicon_score >= args.strong_threshold).sum()}")

    # --- A: Definitely clean (lexicon=0) ---
    df_clean_base = df_neg[df_neg["lexicon_score"] == 0].copy()
    if len(df_clean_base) > args.max_clean:
        # Stratified sample across source files
        df_clean_sample = (
            df_clean_base
            .groupby("_source_file", group_keys=False)
            .apply(lambda g: g.sample(
                min(len(g), max(1, int(args.max_clean * len(g) / len(df_clean_base)))),
                random_state=42
            ))
        )
        # Top up if under target due to rounding
        deficit = args.max_clean - len(df_clean_sample)
        if deficit > 0:
            remaining = df_clean_base[~df_clean_base.index.isin(df_clean_sample.index)]
            df_clean_sample = pd.concat([
                df_clean_sample,
                remaining.sample(min(deficit, len(remaining)), random_state=42)
            ])
        df_clean_final = df_clean_sample.reset_index(drop=True)
    else:
        df_clean_final = df_clean_base.reset_index(drop=True)
    print(f"\n[A] Definitely clean negatives: {len(df_clean_final)}")

    # --- B: Weak signal ---
    df_weak = df_neg[
        (df_neg.lexicon_score > 0) & (df_neg.lexicon_score < args.strong_threshold)
    ].copy()
    print(f"[B] Weak-signal negatives (assumed clean, exported for spot-check): {len(df_weak)}")

    # --- C: Strong signal — re-check or export directly ---
    df_strong = df_neg[df_neg.lexicon_score >= args.strong_threshold].copy()
    print(f"[C] Strong-signal negatives: {len(df_strong)}")

    if args.skip_recheck or len(df_strong) == 0:
        confirmed_from_recheck = pd.DataFrame()
        curate_df = df_strong.copy()
        curate_df["qwen_recheck"] = -1
        curate_df["priority"] = "high"
        print("  Skipping recheck — all strong-signal items exported for curation")
    else:
        confirmed_from_recheck, curate_df = run_recheck(df_strong, model=args.model)
        curate_df = curate_df.copy()
        # Flipped-to-YES = high priority; unclear = medium
        curate_df["priority"] = curate_df["qwen_recheck"].apply(
            lambda r: "high" if r == 1 else "medium"
        )

    # --- Save outputs ---
    # Clean negatives: base clean + confirmed from re-check
    all_clean = pd.concat(
        [df_clean_final, confirmed_from_recheck] if len(confirmed_from_recheck) > 0
        else [df_clean_final],
        ignore_index=True
    )
    keep_cols = ["text", "source", "_source_file", "interaction_type",
                 "source_species", "target_species", "pmid", "lexicon_score", "lexicon_terms"]
    all_clean = all_clean[[c for c in keep_cols if c in all_clean.columns]].copy()
    all_clean["label"] = 0
    all_clean.to_csv(CLEAN_OUT, index=False)
    print(f"\nSaved clean negatives: {len(all_clean)} → {CLEAN_OUT}")

    # Weak signal
    df_weak_out = df_weak[[c for c in keep_cols if c in df_weak.columns]].copy()
    df_weak_out["label"] = 0
    df_weak_out.to_csv(WEAK_OUT, index=False)
    print(f"Saved weak-signal negatives: {len(df_weak_out)} → {WEAK_OUT}")

    # Uncertain / curation queue
    if len(curate_df) > 0:
        curate_cols = ["text", "source", "_source_file", "interaction_type",
                       "source_species", "target_species", "lexicon_score", "lexicon_terms",
                       "qwen_recheck", "priority"]
        if "qwen_recheck_response" in curate_df.columns:
            curate_cols.append("qwen_recheck_response")
        curate_out = curate_df[[c for c in curate_cols if c in curate_df.columns]].copy()
        curate_out = curate_out.sort_values("qwen_confidence").reset_index(drop=True)
        curate_out.to_csv(UNCERTAIN_OUT, index=False)
        print(f"Saved uncertain (curation queue): {len(curate_out)} → {UNCERTAIN_OUT}")
        high = (curate_out.priority == "high").sum()
        med = (curate_out.priority == "medium").sum()
        print(f"  Priority breakdown: {high} high (Qwen flipped to YES = likely FN), {med} medium (unclear)")
    else:
        print("No uncertain negatives — no curation needed.")

    print("\n=== Summary ===")
    print(f"  Definitely clean (lexicon=0):    {len(df_clean_final)}")
    print(f"  Confirmed clean (confidence re-check): {len(confirmed_from_recheck)}")
    print(f"  Weak signal (spot-check CSV):    {len(df_weak_out)}")
    print(f"  Needs curation:                  {len(curate_df)}")
    print(f"\nTotal clean negatives available for training: {len(all_clean)}")
    print("Next step: run assemble_v15_dataset.py after completing curation.")


if __name__ == "__main__":
    main()
