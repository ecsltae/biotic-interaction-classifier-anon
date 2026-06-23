#!/usr/bin/env bash
# Quality-focused multi-task experiments
#
# 1. full_typed_a05_ner2 — 5 joint epochs (was only 3; val F1 still climbing)
# 2. full_typed_a05_ner2_warmstart — start encoder from BiomedBERT_cv_reg
#    instead of raw pretrained; cv_reg already knows what interactions look like (EP F1=0.825)
#
# Usage:
#   bash pipeline_quality.sh [--dry-run]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROOT"
source MPvenv/bin/activate

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; fi

DATA="classifier/data/training/distillation_soft_labels.csv"
EP_RELAX="classifier/data/evaluation/globi-relax_passages-triplets_2024-02-28_curation_EP.tsv"
BASELINE="classifier/models/distilled_BiomedBERT_v2"
ENCODER_BASE="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
ENCODER_WARM="classifier/models/transformer_BiomedBERT_cv_regularized"
RESULTS_BASE="classifier/results/multitask"
MODELS_BASE="classifier/models/multitask"

START_TIME=$(date +%s)
MAX_RUNTIME=36000  # 10h

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$RESULTS_BASE/quality.log"; }
check_time() {
    local now; now=$(date +%s)
    if (( now - START_TIME > MAX_RUNTIME )); then
        log "10h limit reached — stopping."
        exit 0
    fi
}

run_config() {
    local name="$1"
    local encoder="$2"
    local ner_scheme="$3"
    local alpha="$4"
    local pretrain_epochs="${5:-2}"
    local joint_epochs="${6:-5}"

    local model_dir="$MODELS_BASE/$name"
    local res_dir="$RESULTS_BASE/$name"
    local done_flag="$res_dir/ep_relax_eval.json"

    if [[ -f "$done_flag" ]]; then
        log "SKIP $name — already done"
        return 0
    fi

    check_time
    log "START $name  encoder=$(basename $encoder)  ner=$ner_scheme  α=$alpha  ner_pretrain=$pretrain_epochs  epochs=$joint_epochs"

    if [[ $DRY_RUN -eq 1 ]]; then
        log "  [dry-run] $name"
        return 0
    fi

    python classifier/experiments/multitask/train.py \
        --data        "$DATA" \
        --encoder     "$encoder" \
        --ner-scheme  "$ner_scheme" \
        --alpha       "$alpha" \
        --epochs      "$joint_epochs" \
        --pretrain-ner-epochs "$pretrain_epochs" \
        --batch-size  16 \
        --output-dir  "$model_dir" \
        --results-dir "$res_dir" \
        2>&1 | tee -a "$RESULTS_BASE/quality.log"

    check_time

    python classifier/experiments/multitask/evaluate.py \
        --model              "$model_dir" \
        --ep-relax           "$EP_RELAX" \
        --distilled-baseline "$BASELINE" \
        --threshold          0.25 \
        --results-dir        "$res_dir" \
        2>&1 | tee -a "$RESULTS_BASE/quality.log"

    log "DONE $name"
}

mkdir -p "$RESULTS_BASE" "$MODELS_BASE"
log "=== Quality experiments started ==="

# 1. Same best config, but 5 joint epochs (was 3; val F1 was still climbing)
run_config "full_typed_a05_ner2_5ep"  "$ENCODER_BASE" "full_typed" 0.5  2  5

# 2. Warm-start: BiomedBERT_cv_reg encoder (EP F1=0.825 solo) + full_typed NER
run_config "full_typed_a05_ner2_warmstart"  "$ENCODER_WARM" "full_typed" 0.5  2  5

# 3. Warm-start with alpha=0.3 (more NER weight during fine-tuning)
run_config "full_typed_a03_ner2_warmstart"  "$ENCODER_WARM" "full_typed" 0.3  2  5

log "=== Quality experiments complete ==="

# Summary
python - <<'PYEOF'
import json, glob
from pathlib import Path

results_base = Path("classifier/results/multitask")
rows = []
for path in sorted(results_base.glob("*/ep_relax_eval.json")):
    name = path.parent.name
    with open(path) as f:
        data = json.load(f)
    mt = data.get("multitask", {})
    bl = data.get("distilled_v2_baseline", {})
    row = {
        "config":       name,
        "mt_f1_fixed":  mt.get("fixed_threshold_0.25", {}).get("f1"),
        "mt_f1_best":   mt.get("best_threshold",       {}).get("f1"),
        "mt_prec":      mt.get("best_threshold",       {}).get("prec"),
        "mt_rec":       mt.get("best_threshold",       {}).get("rec"),
        "mt_auc":       mt.get("fixed_threshold_0.25", {}).get("auc"),
        "bl_f1_best":   bl.get("best_threshold",       {}).get("f1"),
        "delta_f1":     data.get("delta_f1_vs_baseline"),
    }
    rows.append(row)

rows.sort(key=lambda r: r["mt_f1_best"] or 0, reverse=True)
print(f"\n{'Config':<30} {'F1@0.25':>8} {'F1best':>8} {'Prec':>6} {'Rec':>6} {'AUC':>6} {'Δbase':>8}")
print("-"*80)
for r in rows:
    marker = " ◄" if (r["mt_f1_best"] or 0) >= 0.86 else ""
    print(f"{r['config']:<30} {r['mt_f1_fixed'] or 0:>8.4f} {r['mt_f1_best'] or 0:>8.4f} "
          f"{r['mt_prec'] or 0:>6.3f} {r['mt_rec'] or 0:>6.3f} {r['mt_auc'] or 0:>6.4f} "
          f"{r['delta_f1'] or 0:>+8.4f}{marker}")
PYEOF
