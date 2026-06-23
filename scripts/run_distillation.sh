#!/usr/bin/env bash
# Distillation pipeline: ensemble teacher → single BiomedBERT student
# Step 1: generate soft labels from teacher ensemble (~30 min on GPU)
# Step 2: train student with distillation loss (~2h on GPU)
# Step 3: evaluate on EP-relax, eval_100, synthetic_gold

set -e
cd /path/to/MetaP
source MPvenv/bin/activate

NOTIFY="classifier/scripts/notify.sh"
LOG="classifier/results/distillation_v1/pipeline.log"
mkdir -p classifier/results/distillation_v1

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

log "=== Distillation pipeline started ==="
log "Teacher: BiomedBERT cv_regularized × FLAN-T5-base v12 (geo, EP F1=0.857)"
log "Student: BiomedBERT-base fine-tuned with T=4, alpha=0.7, 6 epochs"

# Check if soft labels already exist
SKIP_FLAG=""
if [ -f "classifier/data/training/distillation_soft_labels.csv" ]; then
    log "Soft labels already exist — skipping generation"
    SKIP_FLAG="--skip-labels"
fi

CUDA_VISIBLE_DEVICES=0 python -u classifier/scripts/distill_ensemble.py \
    $SKIP_FLAG \
    --epochs 6 \
    --temperature 4 \
    --alpha 0.7 \
    --lr 2e-5 \
    >> "$LOG" 2>&1

R=$(grep -E "Student distilled|Reference" "$LOG" | tail -6 | tr '\n' ' ')
log "=== Distillation complete === $R"
bash "$NOTIFY" "Distillation complete" "$R" 2>/dev/null || true
