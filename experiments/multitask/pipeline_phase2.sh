#!/usr/bin/env bash
# Multi-task Phase 2: targeted follow-up on best config (full_typed_a05_ner2, F1=0.868)
#
# Phase 1 winner: full_typed_ner2 scheme (typed NER + 2ep pretrain, α=0.5) → F1=0.868
# Key unknowns:
#   1. Does α=0.3 (more NER weight) + typed NER + pretrain beat α=0.5?
#   2. Does 5 joint epochs (vs 3) improve the best config?
#   3. Does 4 NER pretrain epochs (vs 2) help?
#   4. Does α=0.3 + typed NER (no pretrain) fill the missing cell?
#
# Usage:
#   bash pipeline_phase2.sh [--dry-run]
#
# Outputs:
#   results/multitask/*/ep_relax_eval.json
#   results/multitask/phase2_summary.json

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
ENCODER="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
RESULTS_BASE="classifier/results/multitask"
MODELS_BASE="classifier/models/multitask"

START_TIME=$(date +%s)
MAX_RUNTIME=17000   # ~4.7h

mkdir -p "$RESULTS_BASE" "$MODELS_BASE"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$RESULTS_BASE/phase2.log"; }
check_time() {
    local now; now=$(date +%s)
    if (( now - START_TIME > MAX_RUNTIME )); then
        log "5h session limit approaching — stopping phase 2."
        exit 0
    fi
}

run_config() {
    local name="$1"; local ner_scheme="$2"; local alpha="$3"
    local epochs="$4"; local pretrain_epochs="${5:-0}"

    local model_dir="$MODELS_BASE/$name"
    local res_dir="$RESULTS_BASE/$name"
    local done_flag="$res_dir/ep_relax_eval.json"

    if [[ -f "$done_flag" ]]; then
        log "SKIP $name — artifact exists"
        return 0
    fi

    check_time
    log "START $name (ner=$ner_scheme α=$alpha epochs=$epochs pretrain=$pretrain_epochs)"

    if [[ $DRY_RUN -eq 1 ]]; then
        log "  [dry-run] $name"
        return 0
    fi

    python classifier/experiments/multitask/train.py \
        --data        "$DATA" \
        --encoder     "$ENCODER" \
        --ner-scheme  "$ner_scheme" \
        --alpha       "$alpha" \
        --epochs      "$epochs" \
        --pretrain-ner-epochs "$pretrain_epochs" \
        --batch-size  16 \
        --output-dir  "$model_dir" \
        --results-dir "$res_dir" \
        2>&1 | tee -a "$RESULTS_BASE/phase2.log"

    check_time

    python classifier/experiments/multitask/evaluate.py \
        --model              "$model_dir" \
        --ep-relax           "$EP_RELAX" \
        --distilled-baseline "$BASELINE" \
        --threshold          0.25 \
        --results-dir        "$res_dir" \
        2>&1 | tee -a "$RESULTS_BASE/phase2.log"

    log "DONE $name"
}

log "=== Multi-task Phase 2 started ==="
log "Phase 1 best: full_typed_a05_ner2 → EP F1=0.868"
log ""

# Priority order — most promising first
# ~90 min each (3ep), ~130 min for 5ep configs
run_config "full_typed_a03_ner2"   full_typed  0.3  3  2   # best α × best scheme × pretrain
run_config "full_typed_a03"        full_typed  0.3  3  0   # α=0.3 × typed NER, no pretrain
run_config "full_typed_a05_ner2_5ep" full_typed 0.5 5  2   # best config with more epochs
run_config "full_typed_a03_ner4"   full_typed  0.3  3  4   # more NER pretrain epochs

# ── Summary ──────────────────────────────────────────────────────────────

log ""
log "=== Generating phase 2 summary ==="

python - <<'PYEOF'
import json, glob
from pathlib import Path

results_base = Path("classifier/results/multitask")
rows = []

phase2_configs = ["full_typed_a03_ner2", "full_typed_a03", "full_typed_a05_ner2_5ep", "full_typed_a03_ner4"]
# Also include phase 1 results for comparison
phase1_configs = ["full_typed_a05_ner2", "full_a03", "full_typed_a05", "full_a07", "full_a05", "full_a05_ner2", "basic_a05", "typed_a05"]

for name in phase2_configs + phase1_configs:
    path = results_base / name / "ep_relax_eval.json"
    if not path.exists():
        continue
    with open(path) as f:
        data = json.load(f)
    mt = data.get("multitask", {})
    bl = data.get("distilled_v2_baseline", {})
    row = {
        "config":      name,
        "phase":       "2" if name in phase2_configs else "1",
        "f1_best":     mt.get("best_threshold", {}).get("f1", 0),
        "prec":        mt.get("best_threshold", {}).get("prec", 0),
        "rec":         mt.get("best_threshold", {}).get("rec", 0),
        "auc":         mt.get("fixed_threshold_0.25", {}).get("auc", 0),
        "f1_fixed":    mt.get("fixed_threshold_0.25", {}).get("f1", 0),
        "delta_f1":    data.get("delta_f1_vs_baseline", 0),
    }
    rows.append(row)

rows.sort(key=lambda r: r["f1_best"], reverse=True)

out = results_base / "phase2_summary.json"
with open(out, "w") as f:
    json.dump({"configs": rows}, f, indent=2)

print(f"\n{'Config':<28} {'Ph':>2} {'F1(best)':>9} {'Prec':>6} {'Rec':>6} {'AUC':>7} {'Δ base':>8}")
print("-" * 72)
for r in rows:
    flag = " ← NEW BEST" if r == rows[0] and r["phase"] == "2" else ""
    print(f"{r['config']:<28} {r['phase']:>2} {r['f1_best']:>9.4f} {r['prec']:>6.3f} {r['rec']:>6.3f} {r['auc']:>7.4f} {r['delta_f1']:>+8.4f}{flag}")

baseline_f1 = rows[0].get("delta_f1", 0) + 0.808 if rows else 0.808
print(f"\nBaseline distilled_v2: 0.808  |  Ensemble: 0.857")
print(f"Results: {out}")
PYEOF

log "=== Phase 2 complete ==="
