#!/usr/bin/env bash
# Multi-task sandbox autonomous exploration
# Runs 6 configurations, skips if artifact already exists.
# Compares each against distilled_BiomedBERT_v2 baseline (EP F1=0.808).
#
# Configurations explored:
#   1. basic NER  α=0.5  (balanced)
#   2. basic NER  α=0.3  (NER-dominated)
#   3. basic NER  α=0.7  (cls-dominated)
#   4. typed NER  α=0.5
#   5. basic NER  α=0.5  + NER pretrain 2 epochs
#   6. typed NER  α=0.5  + NER pretrain 2 epochs
#
# Usage:
#   bash pipeline_explore.sh [--dry-run]
#
# Outputs:
#   results/multitask/*/ep_relax_eval.json
#   results/multitask/summary.json

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$ROOT"
source MPvenv/bin/activate

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; fi

# Use distillation soft labels as training data (same source as distilled_v2, 44K rows, 100% soft label coverage)
# NER labels generated via 4.2M-species Aho-Corasick gazetteer + 624 interaction terms
DATA="classifier/data/training/distillation_soft_labels.csv"
EP_RELAX="classifier/data/evaluation/globi-relax_passages-triplets_2024-02-28_curation_EP.tsv"
BASELINE="classifier/models/distilled_BiomedBERT_v2"
ENCODER="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
RESULTS_BASE="classifier/results/multitask"
MODELS_BASE="classifier/models/multitask"

START_TIME=$(date +%s)
MAX_RUNTIME=17000   # ~4.7h (under 5h session limit)

mkdir -p "$RESULTS_BASE" "$MODELS_BASE"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$RESULTS_BASE/explore.log"; }
check_time() {
    local now; now=$(date +%s)
    if (( now - START_TIME > MAX_RUNTIME )); then
        log "5h session limit approaching — stopping. Partial results saved."
        exit 0
    fi
}

run_config() {
    local name="$1"; local ner_scheme="$2"; local alpha="$3"
    local pretrain_epochs="${4:-0}"

    local model_dir="$MODELS_BASE/$name"
    local res_dir="$RESULTS_BASE/$name"
    local done_flag="$res_dir/ep_relax_eval.json"

    if [[ -f "$done_flag" ]]; then
        log "SKIP $name — artifact exists"
        return 0
    fi

    check_time
    log "START $name (ner=$ner_scheme α=$alpha pretrain_ner=$pretrain_epochs)"

    if [[ $DRY_RUN -eq 1 ]]; then
        log "  [dry-run] would train + evaluate $name"
        return 0
    fi

    python classifier/experiments/multitask/train.py \
        --data        "$DATA" \
        --encoder     "$ENCODER" \
        --ner-scheme  "$ner_scheme" \
        --alpha       "$alpha" \
        --epochs      3 \
        --pretrain-ner-epochs "$pretrain_epochs" \
        --batch-size  16 \
        --output-dir  "$model_dir" \
        --results-dir "$res_dir" \
        2>&1 | tee -a "$RESULTS_BASE/explore.log"

    check_time

    python classifier/experiments/multitask/evaluate.py \
        --model              "$model_dir" \
        --ep-relax           "$EP_RELAX" \
        --distilled-baseline "$BASELINE" \
        --threshold          0.25 \
        --results-dir        "$res_dir" \
        2>&1 | tee -a "$RESULTS_BASE/explore.log"

    log "DONE $name"
}

log "=== Multi-task exploration started ==="
log "Data: $DATA"
log "EP-relax: $EP_RELAX"
log "Baseline: $BASELINE"
log ""

# 8 configurations (roughly 45-60 min each on GPU)
# full/full_typed = species + interaction-term NER (recommended, new)
# basic/typed     = species only (baseline comparison)
run_config "full_a05"          full       0.5  0
run_config "full_a03"          full       0.3  0
run_config "full_a07"          full       0.7  0
run_config "full_typed_a05"    full_typed 0.5  0
run_config "full_a05_ner2"     full       0.5  2
run_config "full_typed_a05_ner2" full_typed 0.5  2
run_config "basic_a05"         basic      0.5  0   # ablation: no interaction tags
run_config "typed_a05"         typed      0.5  0   # ablation: no interaction tags
run_config "full_a05"          full       0.5  0   # main config (appended to run after others)

# ── Summarise all results ─────────────────────────────────────────────────

log ""
log "=== Generating summary ==="

python - <<'PYEOF'
import json, glob, os
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
        "config": name,
        "mt_f1_fixed":  mt.get("fixed_threshold_0.25", {}).get("f1",  None),
        "mt_f1_best":   mt.get("best_threshold",        {}).get("f1",  None),
        "mt_prec":      mt.get("best_threshold",        {}).get("prec", None),
        "mt_rec":       mt.get("best_threshold",        {}).get("rec",  None),
        "mt_auc":       mt.get("fixed_threshold_0.25",  {}).get("auc", None),
        "bl_f1_best":   bl.get("best_threshold",        {}).get("f1",  None),
        "delta_f1":     data.get("delta_f1_vs_baseline", None),
    }
    rows.append(row)

rows.sort(key=lambda r: r["mt_f1_best"] or 0, reverse=True)

summary = {"configs": rows}
out = results_base / "summary.json"
with open(out, "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Multi-task Exploration Summary ===")
print(f"{'Config':<22} {'F1(t=0.25)':>10} {'F1(best)':>10} {'Prec':>6} {'Rec':>6} {'AUC':>6} {'Δ vs base':>10}")
print("-" * 80)
for r in rows:
    print(
        f"{r['config']:<22} "
        f"{r['mt_f1_fixed'] or 0:>10.4f} "
        f"{r['mt_f1_best']  or 0:>10.4f} "
        f"{r['mt_prec']     or 0:>6.3f} "
        f"{r['mt_rec']      or 0:>6.3f} "
        f"{r['mt_auc']      or 0:>6.4f} "
        f"{r['delta_f1']    or 0:>+10.4f}"
    )
print(f"\nBaseline distilled_v2 F1 ≈ {rows[0]['bl_f1_best']:.4f}" if rows and rows[0]['bl_f1_best'] else "")
print(f"Results written to {out}")
PYEOF

log "=== Exploration complete ==="
