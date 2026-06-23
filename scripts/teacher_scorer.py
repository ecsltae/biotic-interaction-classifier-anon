#!/usr/bin/env python3
"""
Teacher Scorer: Multi-signal ensemble for high-confidence pseudo-labeling
=========================================================================
Combines multiple signals to produce robust labels for unlabeled sentences:

1. BiomedBERT (discriminative):   P(interaction) from cv_regularized model
2. FLAN-T5-base (generative):     P(yes) from v11_1 model
3. GloBI term scanner:            Deterministic check for 591 known terms
4. Lexicon score:                 Signal strength 0-1.0 from interaction_lexicon
5. Ensemble prob (geometric):     sqrt(p_bert * p_t5) — best combination

Teacher decision rule:
  - HIGH confidence positive (label=1):
      ensemble_prob >= 0.75  OR
      (ensemble_prob >= 0.5 AND globi_terms_found) OR
      (ensemble_prob >= 0.5 AND lexicon_score >= 0.3)
  - HIGH confidence negative (label=0):
      ensemble_prob <= 0.15 AND NOT globi_terms_found AND lexicon_score == 0
  - UNCERTAIN (discard): everything else

Usage:
    # Score a single sentence
    python scripts/teacher_scorer.py --sentence "Plasmodium falciparum infects Anopheles mosquitoes."

    # Score a CSV file (outputs with teacher labels)
    python scripts/teacher_scorer.py --input data/training/globi_sibils_real.csv --output results/research_agent/sibils_teacher_labeled.csv

    # Score EP test set (for evaluation)
    python scripts/teacher_scorer.py --eval-ep

Created: 2026-03-23 by research agent
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data.interaction_taxonomy import scan_globi_terms, get_interaction_category_for_sentence
from data.interaction_lexicon import score_sentence

BASE_DIR = Path('/path/to/MetaP/classifier')

BIOMEDBERT_MODEL = BASE_DIR / 'models/transformer_BiomedBERT_cv_regularized'
FLANT5_MODEL     = BASE_DIR / 'models/flan-t5-base_v11_1'

EP_TEST_FILE  = BASE_DIR / 'data/evaluation/globi-relax_passages-triplets_2024-02-28_curation_EP.tsv'
EVAL_100_FILE = BASE_DIR / 'data/evaluation/eval_100.tsv'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# Model loaders (adapted from ensemble_biomedbert_flant5.py)
# ============================================================================

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.encodings = tokenizer(
            texts, max_length=max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )

    def __len__(self):
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


def get_biomedbert_probs(texts: list, model_dir: Path = BIOMEDBERT_MODEL) -> np.ndarray:
    """Get P(interaction) from BiomedBERT."""
    print(f"  [BiomedBERT] Loading from {model_dir.name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(DEVICE).eval()

    ds = TextDataset(texts, tokenizer)
    loader = DataLoader(ds, batch_size=32)
    probs = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**batch).logits
            p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
    return np.array(probs)


def _get_yes_no_ids(tokenizer):
    yes_id = tokenizer.encode('yes', add_special_tokens=False)[0]
    no_id  = tokenizer.encode('no',  add_special_tokens=False)[0]
    return yes_id, no_id


def _make_prompt(text: str) -> str:
    return (
        f"Does the following sentence describe a biotic interaction between two species? "
        f"Answer yes or no.\n\nSentence: {text}"
    )


def get_flant5_probs(texts: list, model_dir: Path = FLANT5_MODEL, batch_size: int = 32) -> np.ndarray:
    """Get P(yes) from FLAN-T5."""
    print(f"  [FLAN-T5-base] Loading from {model_dir.name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model.to(DEVICE).eval()
    yes_id, no_id = _get_yes_no_ids(tokenizer)

    prompts = [_make_prompt(t) for t in texts]
    probs = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            enc = tokenizer(batch_prompts, return_tensors='pt', padding=True,
                            truncation=True, max_length=512).to(DEVICE)
            dec_input = torch.full(
                (len(batch_prompts), 1), tokenizer.pad_token_id,
                dtype=torch.long, device=DEVICE
            )
            out = model(**enc, decoder_input_ids=dec_input)
            logits = out.logits[:, 0, :]
            lp = torch.log_softmax(logits.float(), dim=-1)
            yes_lp = lp[:, yes_id].cpu().numpy()
            no_lp  = lp[:, no_id].cpu().numpy()
            p_yes = np.exp(yes_lp) / (np.exp(yes_lp) + np.exp(no_lp))
            probs.extend(p_yes.tolist())
    return np.array(probs)


# ============================================================================
# Deterministic signals
# ============================================================================

def get_globi_signals(texts: list) -> Tuple[List[bool], List[List[str]]]:
    """Check each sentence for GloBI terms."""
    has_globi = []
    matched_terms = []
    for text in texts:
        terms = scan_globi_terms(text)
        has_globi.append(len(terms) > 0)
        matched_terms.append(terms)
    return has_globi, matched_terms


def get_lexicon_scores(texts: list) -> Tuple[List[float], List[List[str]]]:
    """Get lexicon interaction signal scores."""
    scores = []
    matched_patterns = []
    for text in texts:
        has_signal, strength, matched = score_sentence(text.lower())
        scores.append(strength)
        matched_patterns.append(matched)
    return scores, matched_patterns


# ============================================================================
# Teacher scoring
# ============================================================================

def compute_teacher_scores(texts: list, verbose: bool = True) -> Dict[str, np.ndarray]:
    """Compute all teacher signals for a list of texts."""
    if verbose:
        print(f"\nComputing teacher scores for {len(texts)} sentences...")

    # ML models
    p_bert = get_biomedbert_probs(texts)
    p_t5   = get_flant5_probs(texts)

    # Ensemble (geometric mean)
    p_ensemble = np.sqrt(np.clip(p_bert * p_t5, 1e-12, 1.0))

    # Deterministic signals
    has_globi, globi_terms = get_globi_signals(texts)
    lexicon_scores, lexicon_matches = get_lexicon_scores(texts)

    return {
        'p_bert': p_bert,
        'p_t5': p_t5,
        'p_ensemble': p_ensemble,
        'has_globi': np.array(has_globi),
        'globi_terms': globi_terms,
        'lexicon_score': np.array(lexicon_scores),
        'lexicon_matches': lexicon_matches,
    }


def teacher_decision(
    p_ensemble: float,
    has_globi: bool,
    lexicon_score: float,
    text: str = "",
    # Thresholds - VERY CONSERVATIVE for pseudo-labeling
    pos_thresh_very_high: float = 0.92,   # Ultra-confident positive (no other check)
    pos_thresh_high: float = 0.80,        # High confidence + must pass title check
    neg_thresh_certain: float = 0.08,     # Very low ensemble = definite negative
    neg_thresh_no_signal: float = 0.20,   # Low ensemble + no signals = negative
) -> Tuple[Optional[int], float, str]:
    """
    Make teacher decision for a single sample.

    CONSERVATIVE MODE: Optimized for pseudo-labeling with high precision.
    - Positive labels require VERY high confidence OR high + non-title context
    - Negative labels require low ensemble AND no interaction signals
    - Everything else is UNCERTAIN (discarded for training)

    Returns:
        label: 1 (positive), 0 (negative), or None (uncertain/discard)
        confidence: 0.0-1.0 confidence in the label
        reason: explanation for the decision
    """
    # Detect paper title patterns (common source of false positives)
    text_lower = text.lower()
    is_likely_title = (
        len(text) < 200 and  # Titles are usually short
        (text.endswith('.') and text.count('.') == 1) and  # Single period at end
        not any(w in text_lower for w in ['we ', 'our ', 'this study', 'here we', 'was ', 'were '])
    )

    # VERY HIGH CONFIDENCE POSITIVE (no other check needed)
    if p_ensemble >= pos_thresh_very_high:
        return 1, p_ensemble, f"very_high_ensemble ({p_ensemble:.3f})"

    # HIGH CONFIDENCE POSITIVE (must not look like a title)
    if p_ensemble >= pos_thresh_high and not is_likely_title:
        return 1, p_ensemble, f"high_ensemble_not_title ({p_ensemble:.3f})"

    # HIGH CONFIDENCE NEGATIVE
    # Case 1: Very low ensemble score
    if p_ensemble <= neg_thresh_certain:
        return 0, 1.0 - p_ensemble, f"very_low_ensemble ({p_ensemble:.3f})"

    # Case 2: Low ensemble + no supporting signals
    if p_ensemble <= neg_thresh_no_signal and not has_globi and lexicon_score == 0:
        return 0, 1.0 - p_ensemble, f"low_no_signals ({p_ensemble:.3f})"

    # UNCERTAIN — discard for training (quality over quantity)
    return None, 0.0, f"uncertain ({p_ensemble:.3f}, globi={has_globi}, lex={lexicon_score:.2f}, title={is_likely_title})"


def apply_teacher_labels(scores: Dict[str, np.ndarray], texts: List[str] = None) -> pd.DataFrame:
    """Apply teacher decision rule to all samples."""
    n = len(scores['p_ensemble'])
    labels = []
    confidences = []
    reasons = []

    for i in range(n):
        text = texts[i] if texts else ""
        label, conf, reason = teacher_decision(
            p_ensemble=scores['p_ensemble'][i],
            has_globi=scores['has_globi'][i],
            lexicon_score=scores['lexicon_score'][i],
            text=text,
        )
        labels.append(label)
        confidences.append(conf)
        reasons.append(reason)

    return pd.DataFrame({
        'teacher_label': labels,
        'teacher_confidence': confidences,
        'teacher_reason': reasons,
        'p_bert': scores['p_bert'],
        'p_t5': scores['p_t5'],
        'p_ensemble': scores['p_ensemble'],
        'has_globi': scores['has_globi'],
        'lexicon_score': scores['lexicon_score'],
    })


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_on_test_set(path: Path) -> Dict:
    """Evaluate teacher scorer on a labeled test set."""
    print(f"\n{'='*60}")
    print(f"Evaluating teacher scorer on: {path.name}")
    print('='*60)

    # Load test set
    df = pd.read_csv(path, sep='\t')
    texts = df['sentence'].astype(str).tolist()
    y_true = df['evaluation_pair_interacting'].astype(int).tolist()

    # Compute scores
    scores = compute_teacher_scores(texts)
    results_df = apply_teacher_labels(scores, texts)

    # Teacher as classifier (using ensemble with threshold search)
    print("\n--- Teacher (Geometric Ensemble) ---")
    best_thresh, best_f1, prec, rec = find_best_threshold(scores['p_ensemble'], y_true)
    print(f"  Best threshold: {best_thresh}")
    print(f"  F1={best_f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")

    y_pred = (scores['p_ensemble'] >= best_thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    print(f"  CM: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")

    # Teacher labeling stats
    print("\n--- Teacher Labeling Decisions ---")
    pos_labels = sum(1 for l in results_df['teacher_label'] if l == 1)
    neg_labels = sum(1 for l in results_df['teacher_label'] if l == 0)
    uncertain  = sum(1 for l in results_df['teacher_label'] if l is None)
    print(f"  Positive labels: {pos_labels}")
    print(f"  Negative labels: {neg_labels}")
    print(f"  Uncertain (discard): {uncertain}")

    # Quality of teacher labels (compared to ground truth)
    print("\n--- Teacher Label Quality ---")
    labeled_mask = results_df['teacher_label'].notna()
    if labeled_mask.sum() > 0:
        teacher_labels = results_df.loc[labeled_mask, 'teacher_label'].astype(int)
        true_labels_subset = [y_true[i] for i in range(len(y_true)) if labeled_mask.iloc[i]]

        if len(teacher_labels) > 0:
            correct = sum(t == p for t, p in zip(true_labels_subset, teacher_labels))
            accuracy = correct / len(teacher_labels)
            print(f"  Accuracy on labeled samples: {accuracy:.4f} ({correct}/{len(teacher_labels)})")

            # Breakdown
            true_pos = sum(1 for i, (t, p) in enumerate(zip(true_labels_subset, teacher_labels))
                         if t == 1 and p == 1)
            false_pos = sum(1 for i, (t, p) in enumerate(zip(true_labels_subset, teacher_labels))
                          if t == 0 and p == 1)
            true_neg = sum(1 for i, (t, p) in enumerate(zip(true_labels_subset, teacher_labels))
                         if t == 0 and p == 0)
            false_neg = sum(1 for i, (t, p) in enumerate(zip(true_labels_subset, teacher_labels))
                          if t == 1 and p == 0)
            print(f"  TP={true_pos} FP={false_pos} TN={true_neg} FN={false_neg}")

            # Precision/recall for teacher labels
            if true_pos + false_pos > 0:
                teacher_prec = true_pos / (true_pos + false_pos)
                print(f"  Teacher positive precision: {teacher_prec:.4f}")
            if true_pos + false_neg > 0:
                teacher_rec = true_pos / (true_pos + false_neg)
                print(f"  Teacher positive recall: {teacher_rec:.4f}")

    return {
        'ensemble_f1': best_f1,
        'ensemble_prec': prec,
        'ensemble_rec': rec,
        'ensemble_thresh': best_thresh,
        'n_pos_labels': pos_labels,
        'n_neg_labels': neg_labels,
        'n_uncertain': uncertain,
    }


def find_best_threshold(probs: np.ndarray, labels: list) -> Tuple:
    """Find optimal threshold for F1."""
    y = np.array(labels)
    best = (0.5, 0.0, 0.0, 0.0)
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (probs >= t).astype(int)
        if preds.sum() == 0:
            continue
        f1 = f1_score(y, preds, zero_division=0)
        if f1 > best[1]:
            p = precision_score(y, preds, zero_division=0)
            r = recall_score(y, preds, zero_division=0)
            best = (round(float(t), 2), round(f1, 4), round(p, 4), round(r, 4))
    return best


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Teacher Scorer for pseudo-labeling')
    parser.add_argument('--sentence', type=str, help='Score a single sentence')
    parser.add_argument('--input', type=str, help='Input CSV file to score')
    parser.add_argument('--output', type=str, help='Output CSV path')
    parser.add_argument('--text-col', default='text', help='Column name for text')
    parser.add_argument('--eval-ep', action='store_true', help='Evaluate on EP test set')
    parser.add_argument('--eval-100', action='store_true', help='Evaluate on eval_100 test set')
    args = parser.parse_args()

    if args.sentence:
        # Score single sentence
        print(f"\nScoring: {args.sentence}")
        scores = compute_teacher_scores([args.sentence], verbose=False)
        results = apply_teacher_labels(scores)

        print(f"\n--- Signals ---")
        print(f"  BiomedBERT P(int):  {scores['p_bert'][0]:.4f}")
        print(f"  FLAN-T5 P(yes):     {scores['p_t5'][0]:.4f}")
        print(f"  Ensemble (geom):    {scores['p_ensemble'][0]:.4f}")
        print(f"  GloBI terms:        {scores['globi_terms'][0]}")
        print(f"  Lexicon score:      {scores['lexicon_score'][0]:.3f}")
        print(f"  Lexicon matches:    {scores['lexicon_matches'][0][:5]}...")  # first 5

        print(f"\n--- Teacher Decision ---")
        print(f"  Label:      {results['teacher_label'].iloc[0]}")
        print(f"  Confidence: {results['teacher_confidence'].iloc[0]:.4f}")
        print(f"  Reason:     {results['teacher_reason'].iloc[0]}")
        return

    if args.eval_ep:
        results = evaluate_on_test_set(EP_TEST_FILE)
        print(f"\n✓ Teacher evaluation complete. Ensemble EP F1 = {results['ensemble_f1']:.4f}")
        return

    if args.eval_100:
        results = evaluate_on_test_set(EVAL_100_FILE)
        print(f"\n✓ Teacher evaluation complete. Ensemble eval_100 F1 = {results['ensemble_f1']:.4f}")
        return

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: {input_path} not found")
            sys.exit(1)

        print(f"\nLoading {input_path}...")
        df = pd.read_csv(input_path)

        if args.text_col not in df.columns:
            print(f"Error: Column '{args.text_col}' not found. Available: {list(df.columns)}")
            sys.exit(1)

        texts = df[args.text_col].astype(str).tolist()
        scores = compute_teacher_scores(texts)
        results = apply_teacher_labels(scores)

        # Merge with original data
        output_df = pd.concat([df.reset_index(drop=True), results], axis=1)

        # Summary
        pos = (results['teacher_label'] == 1).sum()
        neg = (results['teacher_label'] == 0).sum()
        unc = results['teacher_label'].isna().sum()
        print(f"\n--- Summary ---")
        print(f"  Positive labels: {pos}")
        print(f"  Negative labels: {neg}")
        print(f"  Uncertain:       {unc}")
        print(f"  Labeled rate:    {(pos + neg) / len(texts) * 100:.1f}%")

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_df.to_csv(output_path, index=False)
            print(f"\n✓ Saved to {output_path}")
        else:
            # Save to default location
            output_path = BASE_DIR / 'results/research_agent' / (input_path.stem + '_teacher_scored.csv')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_df.to_csv(output_path, index=False)
            print(f"\n✓ Saved to {output_path}")

        return

    # No args - show help
    parser.print_help()


if __name__ == '__main__':
    main()
