"""
rule_classifier.py — Standalone rule-based biotic interaction classifier.

Alternative path alongside the neural ensemble (fastapi_ensemble.py is unchanged).
Applies two hard gates derived from the canonical interaction lexicon:

  Gate A: sentence must contain at least one interaction verb/phrase
  Gate B: sentence must contain at least two binomial species names

If EITHER gate fails → label=0 (not an interaction).
If BOTH pass       → label=1 (interaction), confidence scaled by signal strength.

Usage (CLI):
    python classifier/src/utils/rule_classifier.py \\
        --input  classifier/data/evaluation/eval_100.tsv \\
        --text-col sentence \\
        --label-col evaluation_pair_interacting \\
        --output /tmp/rule_eval100.csv

Usage (Python):
    from src.utils.rule_classifier import classify, classify_dataframe
    result = classify("Apis mellifera pollinates Malus domestica.")
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support

# Allow running directly from project root or from classifier/ subdir
for _candidate in [
    Path(__file__).parent.parent,            # src/
    Path(__file__).parent.parent.parent / "src",  # classifier/src/
]:
    if (_candidate / "data" / "interaction_lexicon.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from data.interaction_lexicon import score_sentence, count_species_mentions  # noqa: E402


# =============================================================================
# CORE CLASSIFIER
# =============================================================================

def classify(text: str) -> dict:
    """
    Apply rule-based biotic interaction classification to a single sentence.

    Gate A: has_signal  — sentence contains an interaction verb/phrase.
    Gate B: n_species>=2 — sentence contains >=2 binomial species names.

    Both gates must pass for label=1. Either failure → label=0.

    Args:
        text: Raw sentence (original casing for species detection).

    Returns:
        Dict with keys:
          label          (int):   1 = interaction, 0 = not
          confidence     (float): 0.0–1.0 rule confidence
          reason         (str):   why the label was assigned
          signal_strength (float): raw lexicon score (0–1)
          species_count  (int):   number of binomial names found
          matched_terms  (list):  interaction patterns that matched
    """
    original_text  = text
    preprocessed   = text.lower().strip()

    has_signal, strength, matched = score_sentence(preprocessed)
    n_species = count_species_mentions(original_text)

    if not has_signal:
        return {
            "label": 0,
            "confidence": 0.90,
            "reason": "no_interaction_verb",
            "signal_strength": strength,
            "species_count": n_species,
            "matched_terms": matched,
        }

    if n_species < 2:
        return {
            "label": 0,
            "confidence": 0.85,
            "reason": "insufficient_species",
            "signal_strength": strength,
            "species_count": n_species,
            "matched_terms": matched,
        }

    # Both gates pass — positive
    confidence = min(0.95, 0.60 + strength * 0.40)
    return {
        "label": 1,
        "confidence": round(confidence, 3),
        "reason": "rule_positive",
        "signal_strength": strength,
        "species_count": n_species,
        "matched_terms": matched,
    }


# =============================================================================
# BATCH CLASSIFICATION
# =============================================================================

def classify_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
) -> pd.DataFrame:
    """
    Classify all rows in a DataFrame and add result columns.

    Adds columns:
      rule_label       (int)
      rule_confidence  (float)
      rule_reason      (str)
      rule_strength    (float)
      rule_n_species   (int)

    Args:
        df:       Input DataFrame.
        text_col: Name of the column containing sentence text.

    Returns:
        Original DataFrame with added rule_* columns.
    """
    results = df[text_col].apply(classify)
    df = df.copy()
    df["rule_label"]      = results.apply(lambda r: r["label"])
    df["rule_confidence"] = results.apply(lambda r: r["confidence"])
    df["rule_reason"]     = results.apply(lambda r: r["reason"])
    df["rule_strength"]   = results.apply(lambda r: r["signal_strength"])
    df["rule_n_species"]  = results.apply(lambda r: r["species_count"])
    return df


def classify_csv(
    input_path: str,
    text_col: str = "text",
    label_col: Optional[str] = None,
    output_path: Optional[str] = None,
    sep: str = "\t",
) -> pd.DataFrame:
    """
    Load a CSV/TSV, classify all rows, optionally compute F1, and save output.

    Args:
        input_path:  Path to input file (.csv or .tsv).
        text_col:    Column containing sentence text.
        label_col:   Optional ground-truth label column (1/0 or True/False).
                     If provided, prints precision/recall/F1 report.
        output_path: Optional path to save results CSV.
        sep:         Delimiter ('\t' for TSV, ',' for CSV).

    Returns:
        Classified DataFrame.
    """
    # Auto-detect separator from extension
    path = Path(input_path)
    if sep == "\t" and path.suffix.lower() == ".csv":
        sep = ","

    df = pd.read_csv(input_path, sep=sep)

    if text_col not in df.columns:
        raise ValueError(
            f"Column '{text_col}' not found. Available: {list(df.columns)}"
        )

    print(f"Classifying {len(df)} sentences from '{input_path}'...")
    df = classify_dataframe(df, text_col=text_col)

    # Summary
    n_pos   = (df["rule_label"] == 1).sum()
    n_neg   = (df["rule_label"] == 0).sum()
    reasons = df["rule_reason"].value_counts()
    print(f"\nResults:")
    print(f"  Positive (interaction):     {n_pos:5d}  ({n_pos/len(df):.1%})")
    print(f"  Negative (not interaction): {n_neg:5d}  ({n_neg/len(df):.1%})")
    print(f"\nNegative reasons:")
    for reason, count in reasons.items():
        if reason != "rule_positive":
            print(f"  {reason:<30s}: {count}")

    # Optional evaluation against ground truth
    if label_col and label_col in df.columns:
        y_true = df[label_col].astype(int)
        y_pred = df["rule_label"]

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        print(f"\n{'='*50}")
        print(f"Evaluation against '{label_col}':")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1        : {f1:.4f}")
        print(f"\nDetailed report:")
        print(classification_report(y_true, y_pred, target_names=["no_interaction", "interaction"]))

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")

    return df


# =============================================================================
# CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rule-based biotic interaction classifier using the canonical lexicon.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate against eval_100 gold standard
  python rule_classifier.py \\
    --input ../../data/evaluation/eval_100.tsv \\
    --text-col sentence \\
    --label-col evaluation_pair_interacting \\
    --output /tmp/rule_eval100.csv

  # Classify arbitrary CSV
  python rule_classifier.py --input sentences.csv --text-col text --sep ,

  # Quick test of individual sentences
  python rule_classifier.py --sentence "Apis mellifera pollinates Malus domestica."
        """,
    )
    parser.add_argument("--input",     "-i", help="Input CSV or TSV file path")
    parser.add_argument("--text-col",  "-t", default="text",
                        help="Column name containing sentence text (default: text)")
    parser.add_argument("--label-col", "-l", default=None,
                        help="Optional ground-truth label column for F1 evaluation")
    parser.add_argument("--output",    "-o", default=None,
                        help="Optional output CSV path for classified results")
    parser.add_argument("--sep",       default="\t",
                        help="Column separator: \\t for TSV (default), , for CSV")
    parser.add_argument("--sentence",  "-s", default=None,
                        help="Classify a single sentence (prints result and exits)")
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()

    if args.sentence:
        # Single-sentence mode
        result = classify(args.sentence)
        print(f"\nSentence : {args.sentence}")
        print(f"Label    : {'INTERACTION' if result['label'] == 1 else 'NOT interaction'}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Reason   : {result['reason']}")
        print(f"Strength : {result['signal_strength']:.3f}")
        print(f"Species  : {result['species_count']}")
        if result["matched_terms"]:
            print(f"Matched  : {result['matched_terms'][:5]}")
        sys.exit(0)

    if not args.input:
        parser.error("Either --input or --sentence is required.")

    classify_csv(
        input_path=args.input,
        text_col=args.text_col,
        label_col=args.label_col,
        output_path=args.output,
        sep=args.sep,
    )
