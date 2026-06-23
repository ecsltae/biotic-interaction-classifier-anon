#!/usr/bin/env bash
# Train FLAN-T5 on each dataset version and save results separately.
# Intended to run overnight: nohup bash classifier/scripts/train_flan_t5_all_versions.sh > /tmp/flan_t5_all.log 2>&1 &

set -euo pipefail

source /path/to/MetaP/MPvenv/bin/activate
cd /path/to/MetaP

SCRIPT="classifier/src/models/flan_t5_classifier.py"
DATA_DIR="classifier/data/training"
MODEL_BASE="classifier/models/flan_t5"
RESULTS_BASE="classifier/results/flan_t5"

declare -A DATASETS=(
    ["v10"]="${DATA_DIR}/training_data_v10.csv"
    ["v10.1"]="${DATA_DIR}/training_data_v10.1.csv"
    ["v12"]="${DATA_DIR}/training_data_v12.csv"
    ["v11_1"]="${DATA_DIR}/training_data_v11_1.csv"
)

# Fixed order: ascending complexity so we can check early results
VERSIONS=("v10" "v10.1" "v12" "v11_1")

for version in "${VERSIONS[@]}"; do
    data="${DATASETS[$version]}"
    out_dir="${MODEL_BASE}_${version}"
    res_dir="${RESULTS_BASE}_${version}"

    echo ""
    echo "======================================================================"
    echo "  VERSION: ${version}  |  DATA: ${data}"
    echo "  Output:  ${out_dir}"
    echo "  Results: ${res_dir}"
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
echo "ALL VERSIONS DONE — $(date)"
echo "======================================================================"
