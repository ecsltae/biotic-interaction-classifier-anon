#!/usr/bin/env bash
# Train BioBART-large and BioGPT on all 4 dataset versions.
# Usage: nohup bash classifier/scripts/train_biobart_biogpt.sh > /tmp/biobart_biogpt.log 2>&1 &

source /path/to/MetaP/MPvenv/bin/activate
cd /path/to/MetaP

NOTIFY="bash classifier/scripts/notify.sh"
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

# ── helpers ──────────────────────────────────────────────────────────────────
get_f1() {
    # get_f1 <results_json>
    python3 -c "
import json, sys
try:
    d = json.load(open('$1'))
    ep = d.get('final_ep_test', {})
    print(f\"F1={ep.get('f1',0):.3f}  Prec={ep.get('precision',0):.3f}  Rec={ep.get('recall',0):.3f}\")
except: print('(no results)')
" 2>/dev/null
}

best_f1() {
    local tag="$1"
    python3 -c "
import json, glob
files = glob.glob('${RESULTS_BASE}/${tag}_v*/flan_t5_results.json') + \
        glob.glob('${RESULTS_BASE}/${tag}_v*/biogpt_results.json')
best = 0.0
for f in files:
    try:
        f1 = json.load(open(f)).get('final_ep_test', {}).get('f1', 0.0)
        if f1 > best: best = f1
    except: pass
print(f'{best:.3f}')
" 2>/dev/null
}

run_training() {
    # run_training <label> <results_json> <python_cmd...>
    local label="$1"; local results_json="$2"; shift 2
    echo "  --- ${label} | $(date) ---"
    if "$@"; then
        local score; score=$(get_f1 "${results_json}")
        $NOTIFY "Done: ${label}" "Training completed successfully.

${score}
Finished: $(date)"
        echo "  Finished ${label}: $(date) | ${score}"
    else
        local exit_code=$?
        $NOTIFY "CRASH: ${label}" "Training crashed with exit code ${exit_code}.

Check log for details.
Time: $(date)"
        echo "  ERROR: ${label} crashed (exit ${exit_code})"
    fi
}

notify_gpu_free() {
    local after="$1"
    $NOTIFY "GPU free after ${after}" "GPU is now idle.

$(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null || echo 'GPU info unavailable')
Time: $(date)"
}

THRESHOLD=0.800

# ── BioBART-large ─────────────────────────────────────────────────────────────
echo "############################################################"
echo "  BioBART-large: GanjinZero/biobart-v2-large"
echo "  Started: $(date)"
echo "############################################################"

for dataset_entry in "${DATASETS[@]}"; do
    version="${dataset_entry%%:*}"
    data="${dataset_entry##*:}"
    out_dir="${MODEL_BASE}/biobart-large_${version}"
    res_dir="${RESULTS_BASE}/biobart-large_${version}"

    run_training "biobart-large/${version}" "${res_dir}/flan_t5_results.json" \
        python "${SEQ2SEQ_SCRIPT}" \
            --model    "GanjinZero/biobart-v2-large" \
            --train-data  "${data}" \
            --output-dir  "${out_dir}" \
            --results-dir "${res_dir}" \
            --epochs 5 --batch-size 16
done

notify_gpu_free "BioBART-large"

BIOBART_BEST=$(best_f1 "biobart-large")
echo "BioBART-large best EP F1: ${BIOBART_BEST}  (threshold=${THRESHOLD})"
if python3 -c "exit(0 if float('${BIOBART_BEST}') >= ${THRESHOLD} else 1)" 2>/dev/null; then
    run_training "biobart-large/v7" "${RESULTS_BASE}/biobart-large_v7/flan_t5_results.json" \
        python "${SEQ2SEQ_SCRIPT}" \
            --model    "GanjinZero/biobart-v2-large" \
            --train-data  "${DATA_DIR}/training_data_globi_v8.csv" \
            --output-dir  "${MODEL_BASE}/biobart-large_v7" \
            --results-dir "${RESULTS_BASE}/biobart-large_v7" \
            --epochs 5 --batch-size 16
else
    echo "  → Below threshold, skipping v7."
fi

# ── BioGPT ────────────────────────────────────────────────────────────────────
echo ""
echo "############################################################"
echo "  BioGPT: microsoft/biogpt"
echo "  Started: $(date)"
echo "############################################################"

for dataset_entry in "${DATASETS[@]}"; do
    version="${dataset_entry%%:*}"
    data="${dataset_entry##*:}"
    out_dir="${MODEL_BASE}/biogpt_${version}"
    res_dir="${RESULTS_BASE}/biogpt_${version}"

    run_training "biogpt/${version}" "${res_dir}/biogpt_results.json" \
        python "${CAUSAL_SCRIPT}" \
            --train-data  "${data}" \
            --output-dir  "${out_dir}" \
            --results-dir "${res_dir}" \
            --epochs 5 --batch-size 16
done

notify_gpu_free "BioGPT"

BIOGPT_BEST=$(best_f1 "biogpt")
echo "BioGPT best EP F1: ${BIOGPT_BEST}  (threshold=${THRESHOLD})"
if python3 -c "exit(0 if float('${BIOGPT_BEST}') >= ${THRESHOLD} else 1)" 2>/dev/null; then
    run_training "biogpt/v7" "${RESULTS_BASE}/biogpt_v7/biogpt_results.json" \
        python "${CAUSAL_SCRIPT}" \
            --train-data  "${DATA_DIR}/training_data_globi_v8.csv" \
            --output-dir  "${MODEL_BASE}/biogpt_v7" \
            --results-dir "${RESULTS_BASE}/biogpt_v7" \
            --epochs 5 --batch-size 16
else
    echo "  → Below threshold, skipping v7."
fi

# ── FLAN-T5-base on v7 ────────────────────────────────────────────────────────
echo ""
echo "############################################################"
echo "  FLAN-T5-base on v7 (training_data_globi_v8.csv)"
echo "  Started: $(date)"
echo "############################################################"

run_training "flan-t5-base/v7" "${RESULTS_BASE}/flan-t5-base_v7/flan_t5_results.json" \
    python "${SEQ2SEQ_SCRIPT}" \
        --model    "google/flan-t5-base" \
        --train-data  "${DATA_DIR}/training_data_globi_v8.csv" \
        --output-dir  "${MODEL_BASE}/flan-t5-base_v7" \
        --results-dir "${RESULTS_BASE}/flan-t5-base_v7" \
        --epochs 5 --batch-size 16

notify_gpu_free "FLAN-T5-base v7"

# ── summary ───────────────────────────────────────────────────────────────────
SUMMARY=$(python3 << 'PYEOF'
import json, glob
rows = []
for f in sorted(glob.glob('classifier/results/*/flan_t5_results.json') +
                glob.glob('classifier/results/*/biogpt_results.json')):
    try:
        d = json.load(open(f))
        ep = d.get('final_ep_test', {})
        if isinstance(ep.get('f1'), float):
            import os
            rows.append((ep['f1'], os.path.basename(os.path.dirname(f)),
                         ep['precision'], ep['recall']))
    except: pass
rows.sort(reverse=True)
lines = [f"{'Model/Version':<32} {'F1':>6} {'Prec':>6} {'Rec':>6}",
         '-'*55]
for f1, name, prec, rec in rows:
    lines.append(f"{name:<32} {f1:>6.3f} {prec:>6.3f} {rec:>6.3f}")
print('\n'.join(lines))
PYEOF
)

$NOTIFY "Queue complete — full leaderboard" "All training runs finished.

${SUMMARY}

Time: $(date)"

echo ""
echo "============================================================"
echo "ALL DONE — $(date)"
echo "============================================================"
echo "${SUMMARY}"
