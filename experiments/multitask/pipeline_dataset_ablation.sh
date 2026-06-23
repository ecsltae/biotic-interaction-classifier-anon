#!/usr/bin/env bash
# Dataset ablation for multi-task BiomedBERT
#
# Research question: does training on the original labeled datasets (v7, v12)
# perform better or worse than training on the 44K soft-label distillation set?
#
# Configs trained (all: full_typed, α=0.5, 2ep NER pretrain, 5 joint epochs):
#   1. full_typed_a05_ner2_warmstart      encoder=transformer_BiomedBERT_cv_regularized   (missed from quality pipeline)
#   2. full_typed_a03_ner2_warmstart      encoder=transformer_BiomedBERT_cv_regularized   (missed from quality pipeline)
#   3. multitask_v7_hardce                data=v7 (25K, hard CE) — v7 is a complete subset of v12
#   4. multitask_v12_hardce               data=v12 (27.6K, hard CE) — v12 ⊃ v7, best ensemble training set
#   5. multitask_v14_hardce               data=v14 (34.9K, hard CE) — latest labeled dataset
#
# Baseline: full_typed_a05_ner2  EP F1=0.868  t=0.13
#
# Usage:
#   bash pipeline_dataset_ablation.sh [--dry-run]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$ROOT"
source MPvenv/bin/activate

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; fi

# ── Paths ─────────────────────────────────────────────────────────────────────

ENCODER="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
ENCODER_WS="classifier/models/transformer_BiomedBERT_cv_regularized"  # warmstart encoder

SOFT_DATA="classifier/data/training/distillation_soft_labels.csv"
V7_DATA="classifier/data/training/training_data_globi_v7_llm_cleaned.csv"
V12_DATA="classifier/data/training/training_data_v12.csv"
V14_DATA="classifier/data/training/training_data_v14.csv"

EP_RELAX="classifier/data/evaluation/globi-relax_passages-triplets_2024-02-28_curation_EP.tsv"
BASELINE="classifier/models/distilled_BiomedBERT_v2"
RESULTS_BASE="classifier/results/multitask"
MODELS_BASE="classifier/models/multitask"

START_TIME=$(date +%s)
MAX_RUNTIME=54000   # 15h ceiling

mkdir -p "$RESULTS_BASE" "$MODELS_BASE"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$RESULTS_BASE/ablation.log"; }

check_time() {
    local now; now=$(date +%s)
    if (( now - START_TIME > MAX_RUNTIME )); then
        log "15h ceiling reached — stopping. Partial results saved."
        exit 0
    fi
}


# ── run_config ─────────────────────────────────────────────────────────────────

run_config() {
    local name="$1"
    local data="$2"
    local encoder="$3"
    local alpha="${4:-0.5}"
    local pretrain_epochs="${5:-2}"
    local epochs="${6:-5}"
    local extra_args="${7:-}"   # e.g. "--soft-labels /nonexistent" for hard CE

    local model_dir="$MODELS_BASE/$name"
    local res_dir="$RESULTS_BASE/$name"
    local done_flag="$res_dir/ep_relax_eval.json"

    if [[ -f "$done_flag" ]]; then
        log "SKIP $name — artifact exists"
        return 0
    fi

    check_time
    log "START $name  data=$(basename $data)  encoder=$(basename $encoder)  α=$alpha  pretrain_ner=$pretrain_epochs  epochs=$epochs"

    if [[ $DRY_RUN -eq 1 ]]; then
        log "  [dry-run] would train + evaluate $name"
        return 0
    fi

    # shellcheck disable=SC2086
    python classifier/experiments/multitask/train.py \
        --data        "$data" \
        --encoder     "$encoder" \
        --ner-scheme  full_typed \
        --alpha       "$alpha" \
        --epochs      "$epochs" \
        --pretrain-ner-epochs "$pretrain_epochs" \
        --batch-size  16 \
        --output-dir  "$model_dir" \
        --results-dir "$res_dir" \
        $extra_args \
        2>&1 | tee -a "$RESULTS_BASE/ablation.log"

    check_time

    python classifier/experiments/multitask/evaluate.py \
        --model              "$model_dir" \
        --ep-relax           "$EP_RELAX" \
        --distilled-baseline "$BASELINE" \
        --threshold          0.25 \
        --results-dir        "$res_dir" \
        2>&1 | tee -a "$RESULTS_BASE/ablation.log"

    log "DONE $name"
}

# ── Phase 1: missed warmstart experiments ─────────────────────────────────────

log "=== Phase 1: Warmstart experiments (missed from quality pipeline) ==="

run_config "full_typed_a05_ner2_warmstart" \
    "$SOFT_DATA" "$ENCODER_WS" 0.5 2 5

run_config "full_typed_a03_ner2_warmstart" \
    "$SOFT_DATA" "$ENCODER_WS" 0.3 2 5

# ── Phase 2: Dataset ablations ────────────────────────────────────────────────

log "=== Phase 2: Dataset ablations (hard CE — no soft labels) ==="

# v7 hard CE  (25K, LLM-validated gold)
run_config "multitask_v7_hardce" \
    "$V7_DATA" "$ENCODER" 0.5 2 5 \
    "--soft-labels /nonexistent"

# v12 hard CE  (27K, signal-filtered, best ensemble training set)
run_config "multitask_v12_hardce" \
    "$V12_DATA" "$ENCODER" 0.5 2 5 \
    "--soft-labels /nonexistent"

# v14 hard CE  (34.9K, latest labeled dataset, more real sentences)
run_config "multitask_v14_hardce" \
    "$V14_DATA" "$ENCODER" 0.5 2 5 \
    "--soft-labels /nonexistent"

# ── Summary ───────────────────────────────────────────────────────────────────

log "=== Generating ablation summary ==="

python3 - <<'PYEOF'
import json
from pathlib import Path

results_base = Path("classifier/results/multitask")
baseline_name = "full_typed_a05_ner2"

baseline_path = results_base / baseline_name / "ep_relax_eval.json"
baseline_f1 = None
if baseline_path.exists():
    with open(baseline_path) as f:
        d = json.load(f)
    baseline_f1 = d.get("multitask", {}).get("best_threshold", {}).get("f1")

targets = [
    "full_typed_a05_ner2_warmstart",
    "full_typed_a03_ner2_warmstart",
    "multitask_v7_hardce",
    "multitask_v12_hardce",
    "multitask_v14_hardce",
]

print("\n=== Dataset Ablation Results ===")
print(f"Baseline: {baseline_name}  EP F1={baseline_f1:.4f}" if baseline_f1 else f"Baseline: {baseline_name} (no results file)")
print(f"\n{'Config':<34} {'F1(t=0.25)':>10} {'F1(best)':>10} {'Prec':>6} {'Rec':>6} {'t_best':>7} {'Δ vs base':>10}")
print("-" * 90)

rows = []
for name in targets:
    path = results_base / name / "ep_relax_eval.json"
    if not path.exists():
        rows.append({"config": name, "status": "NOT DONE"})
        continue
    with open(path) as f:
        d = json.load(f)
    mt = d.get("multitask", {})
    rows.append({
        "config": name,
        "f1_fixed": mt.get("fixed_threshold_0.25", {}).get("f1"),
        "f1_best":  mt.get("best_threshold", {}).get("f1"),
        "prec":     mt.get("best_threshold", {}).get("prec"),
        "rec":      mt.get("best_threshold", {}).get("rec"),
        "t_best":   mt.get("best_threshold", {}).get("threshold"),
        "delta":    (mt.get("best_threshold", {}).get("f1") or 0) - (baseline_f1 or 0),
    })

for r in rows:
    if r.get("status") == "NOT DONE":
        print(f"{r['config']:<34} NOT DONE")
    else:
        print(
            f"{r['config']:<34} "
            f"{r['f1_fixed'] or 0:>10.4f} "
            f"{r['f1_best']  or 0:>10.4f} "
            f"{r['prec']     or 0:>6.3f} "
            f"{r['rec']      or 0:>6.3f} "
            f"{r['t_best']   or 0:>7.3f} "
            f"{r['delta']    or 0:>+10.4f}"
        )

# Save summary
out = results_base / "ablation_summary.json"
with open(out, "w") as f:
    json.dump({"baseline": {"name": baseline_name, "f1": baseline_f1}, "configs": rows}, f, indent=2)
print(f"\nSummary written to {out}")
PYEOF

log "=== Ablation pipeline complete ==="
