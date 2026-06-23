#!/usr/bin/env python3
"""
Assemble the v15 teacher-labeled dataset for cross-validation training.

Sources combined:
  Positives:
    - Teacher positives: Qwen3.5-122B YES (4,065 rows)
    - pathogenOf Qwen-validated: v7 audit (64) + EPMC harvest (56) + human-curated (6)
    - pathogenOf: v7 Qwen-validated (64) + EPMC harvest (56) + human-curated (6)
    - Curated eval disagreements: approved items from eval_qwen_disagreements queue
  Negatives:
    - Clean negatives: lexicon=0 + Qwen-confirmed (12k)
    - Curated negatives: from MCP curation queue (if available)
    - Weak-signal negatives: optional (--include-weak)

Output: classifier/data/training/v15_teacher/dataset.csv  (single file for CV)

No train/test split — cross-validation handles that at training time.

Usage:
    python scripts/assemble_v15_dataset.py
    python scripts/assemble_v15_dataset.py --include-weak --neg-ratio 3.0
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data/training"
OUT_DIR  = DATA_DIR / "v15_teacher"

TEACHER_LABELED       = BASE_DIR / "results/research_agent/all_sources_qwen122b_labeled.csv"

V7_PATHOGEN_VALIDATED = DATA_DIR / "v7_pathogenOf_qwen_validated.csv"
PATHOGEN_HARVESTED    = DATA_DIR / "pathogen_harvested.csv"
CURATED_PATHOGEN      = DATA_DIR / "curated_pathogen_borderline.csv"
CLEAN_NEGS            = DATA_DIR / "negatives_clean.csv"
WEAK_NEGS             = DATA_DIR / "negatives_weak_signal.csv"
CURATED_NEGS          = DATA_DIR / "curated_negatives_from_mcp.csv"
CURATION_DB           = DATA_DIR / "curation.db"
EVAL100_GOLD_POS      = DATA_DIR / "eval100_gold_positives.csv"  # eval_100 gold=POS, label authoritative (25)
V7_QWEN_VALIDATED     = DATA_DIR / "v7_non_pathogen_qwen_validated.csv"  # v7 non-pathogenOf, Qwen-validated

TARGET_COLS = ["text", "label", "interaction_type", "source_species", "target_species", "source"]


def normalise_cols(df: pd.DataFrame, source_tag: str) -> pd.DataFrame:
    """Normalise to TARGET_COLS schema."""
    out = pd.DataFrame()
    out["text"]             = df["text"].astype(str).str.strip()
    out["label"]            = pd.to_numeric(df.get("label", pd.Series([0] * len(df))),
                                            errors="coerce").fillna(0).astype(int)
    out["interaction_type"] = df.get("interaction_type", pd.Series([""] * len(df))).fillna("").astype(str)
    out["source_species"]   = df.get("source_species",   pd.Series([""] * len(df))).fillna("").astype(str)
    out["target_species"]   = df.get("target_species",   pd.Series([""] * len(df))).fillna("").astype(str)
    out["source"]           = source_tag
    return out


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_teacher_positives() -> pd.DataFrame:
    df  = pd.read_csv(TEACHER_LABELED)
    pos = df[df["teacher_label"] == 1].copy()
    pos["label"] = 1
    return normalise_cols(pos, "qwen122b_teacher")



def load_pathogen_positives() -> pd.DataFrame:
    frames = []
    if V7_PATHOGEN_VALIDATED.exists():
        df = pd.read_csv(V7_PATHOGEN_VALIDATED)
        frames.append(normalise_cols(df, "v7_pathogenOf_qwen_validated"))
        print(f"    v7 pathogenOf (Qwen-validated): {len(df)}")
    if PATHOGEN_HARVESTED.exists():
        df  = pd.read_csv(PATHOGEN_HARVESTED)
        pos = df[df["teacher_label"] == 1].copy()
        pos["label"] = 1
        frames.append(normalise_cols(pos, "epmc_pathogen_targeted"))
        print(f"    EPMC harvested pathogenOf: {len(pos)}")
    if CURATED_PATHOGEN.exists():
        df  = pd.read_csv(CURATED_PATHOGEN)
        pos = df[df["label"] == 1].copy()
        frames.append(normalise_cols(pos, "curated_pathogen_borderline"))
        print(f"    Human-curated pathogenOf: {len(pos)}")
    if not frames:
        return pd.DataFrame(columns=TARGET_COLS)
    return pd.concat(frames, ignore_index=True)


def load_v7_qwen_validated() -> pd.DataFrame:
    """v7 non-pathogenOf positives accepted by Qwen3.5-122B (validate_v7_with_qwen.py)."""
    if not V7_QWEN_VALIDATED.exists():
        return pd.DataFrame(columns=TARGET_COLS)
    df = pd.read_csv(V7_QWEN_VALIDATED)
    df = df[df["qwen_label"] == 1].copy()
    return normalise_cols(df, "v7_qwen_validated")


def load_eval100_gold_positives() -> pd.DataFrame:
    """Load eval_100.tsv gold=POS sentences — label is authoritative, no curation needed."""
    if not EVAL100_GOLD_POS.exists():
        return pd.DataFrame(columns=TARGET_COLS)
    df = pd.read_csv(EVAL100_GOLD_POS)
    df = df[df["label"] == 1].copy()
    return normalise_cols(df, "eval100_gold_pos")


def load_curated_eval_disagreements() -> pd.DataFrame:
    """Load approved items from the eval_qwen_disagreements curation queue.

    These are sentences from the original eval sets where Qwen and the gold
    label disagreed. After human curation their label is authoritative.
    """
    if not CURATION_DB.exists():
        print("    WARNING: curation.db not found — skipping eval disagreements")
        return pd.DataFrame(columns=TARGET_COLS)

    con = sqlite3.connect(CURATION_DB)
    con.row_factory = sqlite3.Row
    rows = con.execute("""
        SELECT text, label, interaction_type, source_species, target_species, source_file
        FROM curation_queue
        WHERE source = 'eval_qwen_disagreements'
          AND status = 'approved'
          AND label IN (0, 1)
    """).fetchall()
    con.close()

    if not rows:
        print("    No approved eval_qwen_disagreements yet (still pending curation)")
        return pd.DataFrame(columns=TARGET_COLS)

    df = pd.DataFrame([dict(r) for r in rows])
    # Use source_file as part of source tag for traceability
    df["source_tag"] = "eval_curated_" + df["source_file"].str.replace(r"\.tsv|\.csv", "", regex=True)
    # Normalise each source_file group separately
    frames = []
    for tag, grp in df.groupby("source_tag"):
        frames.append(normalise_cols(grp, tag))
    return pd.concat(frames, ignore_index=True)


def load_clean_negatives(max_neg: int | None = None) -> pd.DataFrame:
    if not CLEAN_NEGS.exists():
        print(f"    WARNING: {CLEAN_NEGS.name} not found — run build_negative_pool.py first")
        return pd.DataFrame(columns=TARGET_COLS)
    df = pd.read_csv(CLEAN_NEGS)
    df["label"] = 0
    out = normalise_cols(df, "qwen122b_clean_neg")
    if max_neg and len(out) > max_neg:
        out = out.sample(max_neg, random_state=42).reset_index(drop=True)
    return out


def load_weak_negatives() -> pd.DataFrame:
    if not WEAK_NEGS.exists():
        return pd.DataFrame(columns=TARGET_COLS)
    df = pd.read_csv(WEAK_NEGS)
    df["label"] = 0
    return normalise_cols(df, "qwen122b_weak_neg")


def load_curated_negatives() -> pd.DataFrame:
    if not CURATED_NEGS.exists():
        return pd.DataFrame(columns=TARGET_COLS)
    df = pd.read_csv(CURATED_NEGS)
    return normalise_cols(df, "curated")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_category_coverage(df: pd.DataFrame) -> None:
    pos = df[df.label == 1]
    print(f"\nInteraction type coverage ({len(pos)} positives):")
    for itype, cnt in pos["interaction_type"].value_counts().head(20).items():
        print(f"  {str(itype):<40s}: {cnt}")


# ---------------------------------------------------------------------------
# Test candidate import
# ---------------------------------------------------------------------------

def _import_test_candidates_to_db(df: pd.DataFrame, seed: int) -> None:
    """Import 200 test candidates into the curation queue as two batches of 100.

    Stratified shuffle so each batch has a similar pos/neg ratio.
    Skips rows already present in the DB (unique text index).
    """
    if not CURATION_DB.parent.exists():
        CURATION_DB.parent.mkdir(parents=True, exist_ok=True)

    # Stratified split into batch_1 / batch_2
    from sklearn.model_selection import train_test_split
    b1, b2 = train_test_split(df, test_size=0.5, stratify=df["label"], random_state=seed)

    con = sqlite3.connect(CURATION_DB)
    con.execute("PRAGMA journal_mode=WAL")
    from datetime import timezone
    now = datetime.now(timezone.utc).isoformat()

    imported = {1: 0, 2: 0}
    skipped  = {1: 0, 2: 0}
    for batch_num, batch_df in [(1, b1), (2, b2)]:
        tag = f"v15_test_batch{batch_num}"
        for _, row in batch_df.iterrows():
            try:
                con.execute(
                    """INSERT INTO curation_queue
                       (text, source, orig_label, interaction_type,
                        source_species, target_species, heuristic_score, created_at)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        row["text"],
                        tag,
                        int(row["label"]),
                        row.get("interaction_type", ""),
                        row.get("source_species", ""),
                        row.get("target_species", ""),
                        0.0,
                        now,
                    ),
                )
                imported[batch_num] += 1
            except sqlite3.IntegrityError:
                skipped[batch_num] += 1
    con.commit()
    con.close()

    for b in (1, 2):
        tag = f"v15_test_batch{b}"
        print(f"  Imported {imported[b]} → curation queue '{tag}' "
              f"(skipped {skipped[b]} duplicates)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=int, default=200,
                        help="Number of sentences to hold out as test candidates (default 200)")
    parser.add_argument("--neg-ratio", type=float, default=2.5,
                        help="Target neg:pos ratio (default 2.5)")
    parser.add_argument("--include-weak", action="store_true",
                        help="Include weak-signal negatives")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Positives ---
    print("Loading positives...")
    teacher_pos = load_teacher_positives()
    print(f"  Teacher positives (Qwen3.5-122B YES): {len(teacher_pos)}")

    print("  pathogenOf sources:")
    pathogen_pos = load_pathogen_positives()
    print(f"  pathogenOf total: {len(pathogen_pos)}")

    v7_qwen = load_v7_qwen_validated()
    if v7_qwen.empty:
        print("  v7 Qwen-validated: (not ready yet — run validate_v7_with_qwen.py)")
    else:
        print(f"  v7 Qwen-validated (non-pathogenOf):   {len(v7_qwen)}")

    eval100_pos = load_eval100_gold_positives()
    print(f"  eval_100 gold=POS (authoritative):     {len(eval100_pos)}")

    print("  Curated eval disagreements (pending — 0 on first run):")
    eval_curated = load_curated_eval_disagreements()
    eval_pos = eval_curated[eval_curated["label"] == 1]
    eval_neg = eval_curated[eval_curated["label"] == 0]
    if eval_curated.empty:
        print("    (none yet — re-run after curating eval_qwen_disagreements)")
    else:
        print(f"    approved positives: {len(eval_pos)}  |  confirmed negatives: {len(eval_neg)}")

    all_pos = pd.concat([teacher_pos, v7_qwen, pathogen_pos, eval100_pos, eval_pos], ignore_index=True)
    all_pos = all_pos.drop_duplicates(subset=["text"]).reset_index(drop=True)
    n_pos = len(all_pos)
    print(f"\n  Total unique positives: {n_pos}")

    # --- Negatives ---
    print("\nLoading negatives...")
    max_neg = int(n_pos * args.neg_ratio)

    clean_negs = load_clean_negatives(max_neg=max_neg)
    print(f"  Clean negatives: {len(clean_negs)}")

    curated_negs = load_curated_negatives()
    print(f"  Curated negatives (MCP queue): {len(curated_negs)}")

    neg_frames = [clean_negs, curated_negs, eval_neg]
    if args.include_weak:
        weak_negs = load_weak_negatives()
        print(f"  Weak-signal negatives (included): {len(weak_negs)}")
        neg_frames.append(weak_negs)

    all_neg = pd.concat(neg_frames, ignore_index=True)
    all_neg = all_neg.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Trim to target ratio
    if len(all_neg) > max_neg:
        all_neg = (
            all_neg
            .groupby("source", group_keys=False)
            .apply(lambda g: g.sample(
                min(len(g), max(1, int(max_neg * len(g) / len(all_neg)))),
                random_state=args.seed,
            ))
            .reset_index(drop=True)
        )
    n_neg = len(all_neg)
    print(f"  Total negatives (after trim): {n_neg}")
    print(f"  Actual neg:pos ratio: {n_neg/n_pos:.2f}")

    # --- Combine and shuffle ---
    pool = pd.concat([all_pos, all_neg], ignore_index=True)
    pool = pool.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    before = len(pool)
    pool = pool.drop_duplicates(subset=["text"]).reset_index(drop=True)
    if len(pool) < before:
        print(f"  Removed {before - len(pool)} internal duplicates")

    # --- Split 200 stratified test candidates BEFORE saving training data ---
    # Stratify by label so pos/neg ratio is preserved in test candidates.
    # v7 synthetic templates are excluded from the test split — we want real sentences only.
    real_mask = ~pool["source"].isin(["v7_llm_validated", "v7_pathogenOf_qwen_validated"])
    pool_real = pool[real_mask].reset_index(drop=True)
    pool_synth = pool[~real_mask].reset_index(drop=True)

    n_test = args.test_size
    # Sample proportionally from pos / neg within real sentences
    test_frames = []
    train_frames = [pool_synth]  # synthetics go straight to training
    for lbl, grp in pool_real.groupby("label"):
        n_lbl = max(1, round(n_test * len(grp) / len(pool_real)))
        sampled = grp.sample(min(n_lbl, len(grp)), random_state=args.seed)
        test_frames.append(sampled)
        train_frames.append(grp.drop(sampled.index))

    df_test_candidates = pd.concat(test_frames, ignore_index=True).sample(
        frac=1, random_state=args.seed).reset_index(drop=True)
    df_train = pd.concat(train_frames, ignore_index=True).sample(
        frac=1, random_state=args.seed).reset_index(drop=True)

    tp, tn = int(df_test_candidates.label.sum()), int((df_test_candidates.label == 0).sum())
    print(f"\n  Test candidates held out: {len(df_test_candidates)} "
          f"({tp} pos / {tn} neg) — pending curation")
    print(f"  Training pool: {len(df_train)} rows")

    # Save test candidates (unverified — need curation before use as test set)
    test_cand_path = OUT_DIR / "test_candidates.csv"
    df_test_candidates.to_csv(test_cand_path, index=False)

    # Import test candidates into curation queue (split into two batches of 100)
    # so the user can curate batch_1 first and batch_2 later.
    _import_test_candidates_to_db(df_test_candidates, args.seed)

    # Save training dataset
    train_path = OUT_DIR / "dataset.csv"
    df_train.to_csv(train_path, index=False)

    n, p = len(df_train), int(df_train.label.sum())
    print(f"\n=== v15 training dataset: {train_path} ===")
    print(f"  {n:,} rows | {p:,} pos ({100*p/n:.1f}%) | {n-p:,} neg")

    check_category_coverage(df_train)

    # --- Metadata ---
    meta = {
        "version": "v15_teacher",
        "built": datetime.now().isoformat(),
        "teacher_model": "qwen3.5:122b",
        "training_strategy": "cross-validation on dataset.csv",
        "neg_ratio": round(n_neg / n_pos, 3),
        "training_rows": n,
        "training_pos": p,
        "training_neg": n - p,
        "test_candidates": {
            "total": len(df_test_candidates),
            "pos": tp,
            "neg": tn,
            "file": "test_candidates.csv",
            "status": "pending curation — import batch_1 (100) first",
        },
        "positive_sources": {
            "teacher_qwen122b": len(teacher_pos),
            "v7_qwen_validated": len(v7_qwen),

            "pathogenOf": len(pathogen_pos),
            "eval100_gold_pos": len(eval100_pos),
            "eval_curated_pos": len(eval_pos),
        },
        "negative_sources": {
            "clean_lexicon0": len(clean_negs),
            "curated_mcp": len(curated_negs),
            "eval_curated_neg": len(eval_neg),
            "weak_signal": len(load_weak_negatives()) if args.include_weak else 0,
        },
    }
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata: {OUT_DIR / 'metadata.json'}")

    print("\nNext steps:")
    print("  1. Curate test batch 1 (100 sentences):")
    print("     python classifier/tools/curate_web.py --port 7860")
    print("     → source: v15_test_batch1")
    print("  2. After batch 1 done, curate batch 2:")
    print("     → source: v15_test_batch2")
    print("  3. Finish eval_qwen_disagreements curation (162 pending)")
    print("  4. Re-run this script — training data will include curated eval disagreements")
    print("  5. Quality gates: python -m pytest classifier/tests/test_training_data.py -v")
    print("  6. Train: python classifier/scripts/train_cv_regularized.py \\")
    print("       --train-data classifier/data/training/v15_teacher/dataset.csv \\")
    print("       --models BiomedBERT --suffix v15")


if __name__ == "__main__":
    main()
