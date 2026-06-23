#!/usr/bin/env bash
# Train FLAN-T5 on remaining dataset versions (v12, v11_1).
# nohup bash classifier/scripts/train_flan_t5_remaining.sh > /tmp/flan_t5_remaining.log 2>&1 &

set -euo pipefail

source /path/to/MetaP/MPvenv/bin/activate
cd /path/to/MetaP

SCRIPT="classifier/src/models/flan_t5_classifier.py"
DATA_DIR="classifier/data/training"
MODEL_BASE="classifier/models/flan_t5"
RESULTS_BASE="classifier/results/flan_t5"

VERSIONS=("v12" "v11_1")
declare -A DATASETS=(
    ["v12"]="${DATA_DIR}/training_data_v12.csv"
    ["v11_1"]="${DATA_DIR}/training_data_v11_1.csv"
)

for version in "${VERSIONS[@]}"; do
    data="${DATASETS[$version]}"
    out_dir="${MODEL_BASE}_${version}"
    res_dir="${RESULTS_BASE}_${version}"

    echo ""
    echo "======================================================================"
    echo "  VERSION: ${version}  |  DATA: ${data}"
    echo "  Started: $(date)"
    echo "======================================================================"

    python "${SCRIPT}" \
        --train-data "${data}" \
        --output-dir "${out_dir}" \
        --results-dir "${res_dir}" \
        --epochs 5 \
        --batch-size 16

    echo "  Finished ${version}: $(date)"
done

echo ""
echo "======================================================================"
echo "ALL REMAINING VERSIONS DONE — $(date)"
echo "======================================================================"
