#!/usr/bin/env bash
# pipeline_v15.sh — Full autonomous v15 training pipeline
#
# Steps:
#   1. Validate v7 non-pathogenOf positives through Qwen3.5-122B (~6h)
#   2. Reassemble v15 dataset (with v7 Qwen-validated positives)
#   3. Retrain BiomedBERT with fixed train_cv_regularized.py (~2h)
#   4. Email results at each step
#
# Usage: nohup bash classifier/scripts/pipeline_v15.sh > classifier/results/pipeline_v15.log 2>&1 &

set -euo pipefail

CLASSIFIER_DIR="/path/to/MetaP/classifier"
RESULTS_DIR="$CLASSIFIER_DIR/results"
VENV="source /path/to/MetaP/MPvenv/bin/activate"
NOTIFY="$CLASSIFIER_DIR/scripts/notify.sh"

notify() {
    bash "$NOTIFY" "$1" "$2" 2>/dev/null || echo "[notify failed] $1: $2"
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

cd /path/to/MetaP
eval "$VENV"

log "=== v15 Pipeline started ==="
notify "v15 pipeline started" "Steps: v7 Qwen validation → dataset assembly → BiomedBERT training"

# ── Step 1: Validate v7 through Qwen ───────────────────────────────────────
V7_OUT="$CLASSIFIER_DIR/data/training/v7_non_pathogen_qwen_validated.csv"
V7_LOG="$RESULTS_DIR/validate_v7_qwen.log"

if [ -f "$V7_OUT" ]; then
    N=$(python -c "import pandas as pd; print(len(pd.read_csv('$V7_OUT')))" 2>/dev/null || echo "?")
    log "Step 1: SKIPPED — v7_non_pathogen_qwen_validated.csv already exists ($N rows)"
else
    log "Step 1: Validating v7 non-pathogenOf through Qwen3.5-122B..."
    python -u classifier/scripts/validate_v7_with_qwen.py --resume >> "$V7_LOG" 2>&1
    EXIT=$?

    if [ $EXIT -ne 0 ]; then
        notify "v15 FAILED — v7 Qwen validation error" "$(tail -20 $V7_LOG)"
        log "ERROR: v7 validation failed. Check $V7_LOG"
        exit 1
    fi

    N_ACCEPTED=$(python -c "
import pandas as pd
df = pd.read_csv('classifier/data/training/v7_non_pathogen_qwen_validated.csv')
print(len(df))
" 2>/dev/null || echo "?")

    log "Step 1 done — $N_ACCEPTED v7 positives accepted by Qwen"
    notify "v15 Step 1 done — v7 Qwen validation" \
        "Accepted: $N_ACCEPTED / 7076 v7 non-pathogenOf positives. Starting dataset assembly."
fi

# ── Step 2: Reassemble dataset ─────────────────────────────────────────────
DATASET_OUT="$CLASSIFIER_DIR/data/training/v15_teacher/dataset.csv"
ASSEMBLE_LOG="$RESULTS_DIR/assemble_v15.log"

# Always reassemble — v7 data may have changed
log "Step 2: Reassembling v15 dataset..."
python -u classifier/scripts/assemble_v15_dataset.py > "$ASSEMBLE_LOG" 2>&1
EXIT=$?

if [ $EXIT -ne 0 ]; then
    notify "v15 FAILED — dataset assembly error" "$(tail -20 $ASSEMBLE_LOG)"
    log "ERROR: Assembly failed. Check $ASSEMBLE_LOG"
    exit 1
fi

N_ROWS=$(python -c "
import pandas as pd
df = pd.read_csv('classifier/data/training/v15_teacher/dataset.csv')
pos = int(df.label.sum())
print(f'{len(df)} rows, {pos} pos ({100*pos/len(df):.1f}%)')
" 2>/dev/null || echo "?")

log "Step 2 done — dataset: $N_ROWS"
notify "v15 Step 2 done — dataset assembled" \
    "Dataset: $N_ROWS. Starting BiomedBERT training (~2h)."

# ── Step 3: Retrain BiomedBERT ─────────────────────────────────────────────
TRAIN_LOG="$RESULTS_DIR/train_v15b_biomedbert.log"
MODEL_OUT="$CLASSIFIER_DIR/models/transformer_BiomedBERT_v15b"

if [ -f "$MODEL_OUT/config.json" ]; then
    log "Step 3: SKIPPED — BiomedBERT v15b already trained"
else
    log "Step 3: Training BiomedBERT v15b..."
    python -u classifier/scripts/train_cv_regularized.py \
    --train-data classifier/data/training/v15_teacher/dataset.csv \
    --models BiomedBERT \
    --suffix v15b \
        >> "$TRAIN_LOG" 2>&1
    EXIT=$?

    if [ $EXIT -ne 0 ]; then
        notify "v15 FAILED — BiomedBERT training error" "$(tail -20 $TRAIN_LOG)"
        log "ERROR: Training failed. Check $TRAIN_LOG"
        exit 1
    fi

    RESULTS=$(grep -E "Avg Test F1|Best Fold|Avg Val" "$TRAIN_LOG" | tail -5 || echo "see log")
    log "Step 3 done — training complete"
    log "$RESULTS"
    notify "v15 Step 3 done — BiomedBERT v15b results" \
"Training done.

$RESULTS

Full log: $TRAIN_LOG. Starting FLAN-T5-base training."
fi

# ── Step 4: Train FLAN-T5-base ─────────────────────────────────────────────
FLANT5_LOG="$RESULTS_DIR/train_v15b_flanT5base.log"
FLANT5_OUT="$CLASSIFIER_DIR/models/flan_t5_base_v15b"

if [ -d "$FLANT5_OUT" ] && [ "$(ls -A $FLANT5_OUT 2>/dev/null)" ]; then
    log "Step 4: SKIPPED — FLAN-T5-base v15b already trained"
else
    log "Step 4: Training FLAN-T5-base v15b..."
    CUDA_VISIBLE_DEVICES=0 python -u classifier/src/models/flan_t5_classifier.py \
        --train-data classifier/data/training/v15_teacher/dataset.csv \
        --model google/flan-t5-base \
        --epochs 6 \
        --output-dir "$FLANT5_OUT" \
        --results-dir "$RESULTS_DIR" \
        >> "$FLANT5_LOG" 2>&1
    EXIT=$?

    if [ $EXIT -ne 0 ]; then
        notify "v15 FAILED — FLAN-T5-base training error" "$(tail -20 $FLANT5_LOG)"
        log "ERROR: FLAN-T5-base training failed. Check $FLANT5_LOG"
        exit 1
    fi

    FLANT5_RESULTS=$(grep -E "Avg EP F1|Best Fold EP F1|Avg EP Precision|Avg EP Recall" "$FLANT5_LOG" | tail -6 || echo "see log")
    log "Step 4 done — FLAN-T5-base training complete"
    log "$FLANT5_RESULTS"
    notify "v15 Pipeline COMPLETE — FLAN-T5-base v15b results" \
"Training done.

$FLANT5_RESULTS

Full log: $FLANT5_LOG"
fi

log "=== Pipeline complete ==="
