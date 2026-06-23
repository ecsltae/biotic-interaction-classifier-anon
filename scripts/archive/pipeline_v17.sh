#!/usr/bin/env bash
# Pipeline v17: Track A then Track B
# Track A (tonight): balance fix + hard negatives → train BiomedBERT + FLAN-T5-base
# Track B (after A): Qwen recalibration → rebuild → retrain
# No API key. GPU 0 only for training, both GPUs for Qwen.

set -e
cd /path/to/MetaP
source MPvenv/bin/activate

NOTIFY="classifier/scripts/notify.sh"
RESULTS="classifier/results"
SESSION_START=$(date +%s)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$RESULTS/pipeline_v17.log"
}

check_5h() {
    local now=$(date +%s)
    local elapsed=$(( (now - SESSION_START) / 3600 ))
    if [ $elapsed -ge 5 ]; then
        log "5-HOUR LIMIT REACHED — relaunching pipeline to continue"
        bash "$NOTIFY" "v17 pipeline: 5h limit, relaunching" "Pipeline self-restarted. Check pipeline_v17.log." 2>/dev/null || true
        nohup bash /path/to/MetaP/classifier/scripts/pipeline_v17.sh \
            >> /path/to/MetaP/classifier/results/pipeline_v17.log 2>&1 &
        exit 0
    fi
}

log "=== Pipeline v17 started ==="

# ── Track A Step 1: Build v17a dataset ─────────────────────────────────────
DATA_V17A="classifier/data/training/v17_fixed/dataset.csv"

if [ -f "$DATA_V17A" ]; then
    log "Track A Step 1: SKIPPED — v17a dataset exists"
else
    log "Track A Step 1: Building v17a dataset (balance fix + hard negatives)..."
    python -u classifier/scripts/build_v17_fixed.py >> "$RESULTS/build_v17a.log" 2>&1
    log "Track A Step 1 done."
fi
check_5h

# ── Track A Step 2: Train BiomedBERT v17a ─────────────────────────────────
BERT_V17A="classifier/models/transformer_BiomedBERT_v17a"
BERT_V17A_LOG="$RESULTS/train_v17a_biomedbert.log"

if [ -f "$BERT_V17A/config.json" ]; then
    log "Track A Step 2: SKIPPED — BiomedBERT v17a already trained"
else
    log "Track A Step 2: Training BiomedBERT v17a..."
    CUDA_VISIBLE_DEVICES=1 python -u classifier/scripts/train_cv_regularized.py \
        --train-data "$DATA_V17A" \
        --models BiomedBERT \
        --suffix v17a \
        >> "$BERT_V17A_LOG" 2>&1
    R=$(grep -E "Avg Test F1|Best Fold Test F1" "$BERT_V17A_LOG" | tail -4 || echo "see log")
    log "Track A Step 2 done: $R"
    bash "$NOTIFY" "v17a BiomedBERT done" "$R" 2>/dev/null || true
fi
check_5h

# ── Track A Step 3: Train FLAN-T5-base v17a ───────────────────────────────
T5_V17A="classifier/models/flan_t5_base_v17a"
T5_V17A_LOG="$RESULTS/train_v17a_flanT5base.log"

if [ -d "$T5_V17A" ] && [ "$(ls -A $T5_V17A 2>/dev/null)" ]; then
    log "Track A Step 3: SKIPPED — FLAN-T5-base v17a already trained"
else
    log "Track A Step 3: Training FLAN-T5-base v17a..."
    CUDA_VISIBLE_DEVICES=1 python -u classifier/src/models/flan_t5_classifier.py \
        --train-data "$DATA_V17A" \
        --model google/flan-t5-base \
        --epochs 6 \
        --output-dir "$T5_V17A" \
        --results-dir "$RESULTS/flan_t5_base_v17a" \
        >> "$T5_V17A_LOG" 2>&1
    R=$(grep -E "Avg EP F1|Best Fold EP F1" "$T5_V17A_LOG" | tail -4 || echo "see log")
    log "Track A Step 3 done: $R"
    bash "$NOTIFY" "v17a FLAN-T5-base done" "$R" 2>/dev/null || true
fi
check_5h

log "=== Track A complete. Starting Track B (Qwen recalibration) ==="

# ── Track B Step 1: Recalibrate Qwen labels ───────────────────────────────
RECAL_OUT="classifier/data/training/qwen_positives_recalibrated.csv"

if [ -f "$RECAL_OUT" ]; then
    log "Track B Step 1: SKIPPED — recalibrated labels already exist"
else
    log "Track B Step 1: Recalibrating 4,065 Qwen positives with few-shot EP prompt (~3.4h)..."
    python -u classifier/scripts/recalibrate_qwen_labels.py --resume \
        >> "$RESULTS/recalibrate_qwen.log" 2>&1
    R=$(tail -10 "$RESULTS/recalibrate_qwen.log" | grep -E "YES|NO|kept" || echo "see log")
    log "Track B Step 1 done: $R"
    bash "$NOTIFY" "v17b Qwen recalibration done" "$R" 2>/dev/null || true
fi
check_5h

# ── Track B Step 2: Build v17b dataset (recalibrated + Track A fixes) ─────
DATA_V17B="classifier/data/training/v17b_recalibrated/dataset.csv"

if [ -f "$DATA_V17B" ]; then
    log "Track B Step 2: SKIPPED — v17b dataset exists"
else
    log "Track B Step 2: Building v17b dataset (recalibrated labels + balance + hard negs)..."
    python -u classifier/scripts/build_v17b_recalibrated.py >> "$RESULTS/build_v17b.log" 2>&1
    log "Track B Step 2 done."
fi
check_5h

# ── Track B Step 3: Train BiomedBERT v17b ─────────────────────────────────
BERT_V17B="classifier/models/transformer_BiomedBERT_v17b"
BERT_V17B_LOG="$RESULTS/train_v17b_biomedbert.log"

if [ -f "$BERT_V17B/config.json" ]; then
    log "Track B Step 3: SKIPPED — BiomedBERT v17b already trained"
else
    log "Track B Step 3: Training BiomedBERT v17b..."
    CUDA_VISIBLE_DEVICES=1 python -u classifier/scripts/train_cv_regularized.py \
        --train-data "$DATA_V17B" \
        --models BiomedBERT \
        --suffix v17b \
        >> "$BERT_V17B_LOG" 2>&1
    R=$(grep -E "Avg Test F1|Best Fold Test F1" "$BERT_V17B_LOG" | tail -4 || echo "see log")
    log "Track B Step 3 done: $R"
    bash "$NOTIFY" "v17b BiomedBERT done" "$R" 2>/dev/null || true
fi
check_5h

# ── Track B Step 4: Train FLAN-T5-base v17b ───────────────────────────────
T5_V17B="classifier/models/flan_t5_base_v17b"
T5_V17B_LOG="$RESULTS/train_v17b_flanT5base.log"

if [ -d "$T5_V17B" ] && [ "$(ls -A $T5_V17B 2>/dev/null)" ]; then
    log "Track B Step 4: SKIPPED — FLAN-T5-base v17b already trained"
else
    log "Track B Step 4: Training FLAN-T5-base v17b..."
    CUDA_VISIBLE_DEVICES=1 python -u classifier/src/models/flan_t5_classifier.py \
        --train-data "$DATA_V17B" \
        --model google/flan-t5-base \
        --epochs 6 \
        --output-dir "$T5_V17B" \
        --results-dir "$RESULTS/flan_t5_base_v17b" \
        >> "$T5_V17B_LOG" 2>&1
    R=$(grep -E "Avg EP F1|Best Fold EP F1" "$T5_V17B_LOG" | tail -4 || echo "see log")
    log "Track B Step 4 done: $R"
    bash "$NOTIFY" "v17b FLAN-T5-base done — pipeline complete" "$R" 2>/dev/null || true
fi

log "=== Track B complete. Starting v18 hybrid (real + gap-fill templates) ==="
check_5h

# ── v18: Hybrid dataset (real sentences + v7 Qwen templates for gap types) ─
DATA_V18="classifier/data/training/v18_hybrid/dataset.csv"
if [ ! -f "$DATA_V18" ]; then
    log "v18 Step 1: Building hybrid dataset..."
    python -u classifier/scripts/build_v18_hybrid.py >> "$RESULTS/build_v18.log" 2>&1
    log "v18 Step 1 done."
fi
check_5h

BERT_V18="classifier/models/transformer_BiomedBERT_v18"
if [ ! -f "$BERT_V18/config.json" ]; then
    log "v18 Step 2: Training BiomedBERT v18 hybrid..."
    CUDA_VISIBLE_DEVICES=1 python -u classifier/scripts/train_cv_regularized.py \
        --train-data "$DATA_V18" --models BiomedBERT --suffix v18 \
        >> "$RESULTS/train_v18_biomedbert.log" 2>&1
    R=$(grep -E "Avg Test F1|Best Fold Test F1" "$RESULTS/train_v18_biomedbert.log" | tail -4 || echo "see log")
    log "v18 Step 2 done: $R"
    bash "$NOTIFY" "v18 BiomedBERT hybrid done" "$R" 2>/dev/null || true
fi
check_5h

T5_V18="classifier/models/flan_t5_base_v18"
if [ ! -d "$T5_V18" ] || [ -z "$(ls -A $T5_V18 2>/dev/null)" ]; then
    log "v18 Step 3: Training FLAN-T5-base v18 hybrid..."
    CUDA_VISIBLE_DEVICES=1 python -u classifier/src/models/flan_t5_classifier.py \
        --train-data "$DATA_V18" --model google/flan-t5-base --epochs 6 \
        --output-dir "$T5_V18" --results-dir "$RESULTS/flan_t5_base_v18" \
        >> "$RESULTS/train_v18_flanT5base.log" 2>&1
    R=$(grep -E "Avg EP F1|Best Fold EP F1" "$RESULTS/train_v18_flanT5base.log" | tail -4 || echo "see log")
    log "v18 Step 3 done: $R"
    bash "$NOTIFY" "v18 FLAN-T5-base hybrid done — ALL DONE" "$R" 2>/dev/null || true
fi

log "=== Pipeline v17+v18 complete ==="
