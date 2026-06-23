#!/usr/bin/env bash
# Queue all generative models (FLAN-T5-base, BART-large, BioBART-large, BioGPT)
# on all 4 dataset versions. Waits for any running flan_t5 process first.
# Usage: nohup bash classifier/scripts/train_generative_queue.sh > /tmp/generative_queue.log 2>&1 &

set -euo pipefail

source /path/to/MetaP/MPvenv/bin/activate
cd /path/to/MetaP

SEQ2SEQ_SCRIPT="classifier/src/models/flan_t5_classifier.py"
CAUSAL_SCRIPT="classifier/src/models/biogpt_classifier.py"
DATA_DIR="classifier/data/training"
MODEL_BASE="classifier/models"
RESULTS_BASE="classifier/results"

DATASETS=(
    "v10:${DATA_DIR}/training_data_v10.csv"
    "v10.1:${DATA_DIR}/training_data_v10.1.csv"
    "v12:${DATA_DIR}/training_data_v12.csv"
    "v11_1:${DATA_DIR}/training_data_v11_1.csv"
)

# ── Wait for current FLAN-T5-large runs to finish ───────────────────────────────
echo "Waiting for running flan_t5 processes... ($(date))"
while pgrep -f "flan_t5_classifier.py" > /dev/null 2>&1; do
    sleep 60
done
echo "Previous runs done. Starting queue. ($(date))"

# ── Seq2seq models (FLAN-T5-base, BART-large, BioBART-large) ────────────────────
declare -A SEQ2SEQ_MODELS=(
    ["flan-t5-base"]="google/flan-t5-base"
    ["bart-large"]="facebook/bart-large"
    ["biobart-large"]="GanjinZero/biobart-v2-large"
)
SEQ2SEQ_ORDER=("flan-t5-base" "bart-large" "biobart-large")

for model_tag in "${SEQ2SEQ_ORDER[@]}"; do
    model_id="${SEQ2SEQ_MODELS[$model_tag]}"
    echo ""
    echo "############################################################"
    echo "  SEQ2SEQ MODEL: ${model_id}"
    echo "############################################################"

    for dataset_entry in "${DATASETS[@]}"; do
        version="${dataset_entry%%:*}"
        data="${dataset_entry##*:}"
        out_dir="${MODEL_BASE}/${model_tag}_${version}"
        res_dir="${RESULTS_BASE}/${model_tag}_${version}"

        echo "  --- ${model_tag} / ${version} | $(date) ---"
        python "${SEQ2SEQ_SCRIPT}" \
            --model    "${model_id}" \
            --train-data  "${data}" \
            --output-dir  "${out_dir}" \
            --results-dir "${res_dir}" \
            --epochs 5 \
            --batch-size 16
        echo "  Finished ${model_tag} / ${version}: $(date)"
    done
done

# ── Causal LM model (BioGPT) ────────────────────────────────────────────────────
echo ""
echo "############################################################"
echo "  CAUSAL LM MODEL: microsoft/biogpt"
echo "############################################################"

for dataset_entry in "${DATASETS[@]}"; do
    version="${dataset_entry%%:*}"
    data="${dataset_entry##*:}"
    out_dir="${MODEL_BASE}/biogpt_${version}"
    res_dir="${RESULTS_BASE}/biogpt_${version}"

    echo "  --- biogpt / ${version} | $(date) ---"
    python "${CAUSAL_SCRIPT}" \
        --train-data  "${data}" \
        --output-dir  "${out_dir}" \
        --results-dir "${res_dir}" \
        --epochs 5 \
        --batch-size 16
    echo "  Finished biogpt / ${version}: $(date)"
done

echo ""
echo "============================================================"
echo "ALL GENERATIVE MODELS DONE — $(date)"
echo "============================================================"
