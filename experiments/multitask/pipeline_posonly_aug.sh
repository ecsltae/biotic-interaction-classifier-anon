#!/bin/bash
# Retrain champion config with ONLY new positives added (no new negatives).
# Tests hypothesis: hard negatives from PMC host-pathogen papers hurt; positives alone should help.

set -euo pipefail
cd /path/to/MetaP
source MPvenv/bin/activate

SOFT_LABELS="classifier/data/training/distillation_soft_labels_posonly_aug.csv"
OUTPUT_DIR="classifier/models/multitask/full_typed_a05_ner2_posonly"
RESULTS_DIR="classifier/results/multitask/full_typed_a05_ner2_posonly"
LOG="$RESULTS_DIR/train.log"

mkdir -p "$RESULTS_DIR"

echo "=== Retraining full_typed_a05_ner2 — positives-only augmentation ===" | tee "$LOG"
echo "Soft labels: $SOFT_LABELS ($(wc -l < $SOFT_LABELS) rows)" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"

python classifier/experiments/multitask/train.py \
    --data "$SOFT_LABELS" \
    --ner-scheme full_typed \
    --pretrain-ner-epochs 2 \
    --alpha 0.5 \
    --epochs 3 \
    --output-dir "$OUTPUT_DIR" \
    --results-dir "$RESULTS_DIR" \
    2>&1 | tee -a "$LOG"

echo "Training done: $(date)" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Evaluating on EP-relax ===" | tee -a "$LOG"
python classifier/experiments/multitask/evaluate.py \
    --model "$OUTPUT_DIR" \
    --ep-relax classifier/data/evaluation/globi-relax_passages-triplets_2024-02-28_curation_EP.tsv \
    --results-dir "$RESULTS_DIR" \
    2>&1 | tee -a "$LOG"

echo "=== Done: $(date) ===" | tee -a "$LOG"

if [ -f "$RESULTS_DIR/ep_relax_eval.json" ]; then
    python -c "
import json
r = json.load(open('$RESULTS_DIR/ep_relax_eval.json'))['multitask']['best_threshold']
print(f'  F1={r[\"f1\"]:.4f}  Prec={r[\"precision\"]:.4f}  Rec={r[\"recall\"]:.4f}  AUC={r.get(\"auc\",0):.4f}  thresh={r.get(\"threshold\",0.5):.3f}')
print(f'  Baseline (full_typed_a05_ner2): F1=0.8680')
print(f'  Delta: {r[\"f1\"]-0.8680:+.4f}')
" 2>&1 | tee -a "$LOG"
fi
