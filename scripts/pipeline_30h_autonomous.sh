#!/usr/bin/env bash
# Autonomous 30-hour classifier improvement pipeline
# Runs through a fixed task list; skips tasks whose output artifacts exist.
# Self-restarts on 5h limit. Sends email at each milestone.
#
# Tasks:
#   1. eval_ensembles          — compare all distilled model combinations
#   2. distill_distilbert      — DistilBERT student (T=2, α=0.5)
#   3. distill_scibert         — SciBERT student (T=2, α=0.5)
#   4. distill_sharp           — BiomedBERT, T=1.5, α=0.5 (very sharp teacher)
#   5. finetune_v2_on_v18      — fine-tune distilled_v2 on v18_hybrid data
#   6. final_report            — update RESEARCH_LOG.md, email summary

set -e
cd /path/to/MetaP
source MPvenv/bin/activate

NOTIFY="classifier/scripts/notify.sh"
LOG="classifier/results/autonomous_30h/pipeline.log"
SESSION_START=$(date +%s)
mkdir -p classifier/results/autonomous_30h

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

check_5h() {
    local now=$(date +%s)
    local elapsed=$(( (now - SESSION_START) / 3600 ))
    if [ $elapsed -ge 5 ]; then
        log "5-HOUR LIMIT — relaunching pipeline_30h_autonomous.sh"
        bash "$NOTIFY" "autonomous pipeline: 5h limit, relaunching" "Check autonomous_30h/pipeline.log" 2>/dev/null || true
        nohup bash /path/to/MetaP/classifier/scripts/pipeline_30h_autonomous.sh \
            >> "$LOG" 2>&1 &
        exit 0
    fi
}

log "=== Autonomous 30h pipeline started ==="

# ── TASK 1: Evaluate all distilled model ensembles ──────────────────────────
ENSEMBLE_RESULT="classifier/results/autonomous_30h/ensemble_comparison.json"
if [ ! -f "$ENSEMBLE_RESULT" ]; then
    log "TASK 1: Evaluating all distilled model ensemble combinations..."
    CUDA_VISIBLE_DEVICES=0 python -u classifier/scripts/eval_distilled_ensembles.py \
        >> "$LOG" 2>&1
    R=$(python3 -c "
import json
r=json.load(open('$ENSEMBLE_RESULT'))
best = max(r.items(), key=lambda x: x[1]['ep_relax']['f1'])
print(f\"Best combo: {best[0]} EP={best[1]['ep_relax']['f1']:.3f} eval100={best[1]['eval_100']['f1']:.3f}\")
" 2>/dev/null || echo "see log")
    log "TASK 1 done: $R"
    bash "$NOTIFY" "Autonomous: ensemble comparison done" "$R" 2>/dev/null || true
else
    log "TASK 1: SKIPPED — ensemble_comparison.json exists"
fi
check_5h

# ── TASK 2: Distill DistilBERT student (T=2, α=0.5) ────────────────────────
DISTILBERT_DIR="classifier/models/distilled_DistilBERT_v4"
if [ ! -d "$DISTILBERT_DIR" ] || [ -z "$(ls -A $DISTILBERT_DIR 2>/dev/null)" ]; then
    log "TASK 2: Distilling DistilBERT student (T=2, α=0.5)..."
    mkdir -p classifier/results/distillation_v4_distilbert
    CUDA_VISIBLE_DEVICES=0 python -u classifier/scripts/distill_ensemble.py \
        --skip-labels \
        --student-model distilbert-base-uncased \
        --epochs 6 --temperature 2 --alpha 0.5 --lr 2e-5 \
        --output-dir classifier/models/distilled_DistilBERT_v4 \
        --results-dir classifier/results/distillation_v4_distilbert \
        >> classifier/results/distillation_v4_distilbert/pipeline.log 2>&1
    R=$(grep "Student distilled" classifier/results/distillation_v4_distilbert/pipeline.log | tail -3 || echo "see log")
    log "TASK 2 done: $R"
    bash "$NOTIFY" "Autonomous: DistilBERT distillation done" "$R" 2>/dev/null || true
else
    log "TASK 2: SKIPPED — distilled_DistilBERT_v4 exists"
fi
check_5h

# ── TASK 3: Distill SciBERT student (T=2, α=0.5) ───────────────────────────
SCIBERT_DIR="classifier/models/distilled_SciBERT_v5"
if [ ! -d "$SCIBERT_DIR" ] || [ -z "$(ls -A $SCIBERT_DIR 2>/dev/null)" ]; then
    log "TASK 3: Distilling SciBERT student (T=2, α=0.5)..."
    mkdir -p classifier/results/distillation_v5_scibert
    CUDA_VISIBLE_DEVICES=0 python -u classifier/scripts/distill_ensemble.py \
        --skip-labels \
        --student-model allenai/scibert_scivocab_uncased \
        --epochs 6 --temperature 2 --alpha 0.5 --lr 2e-5 \
        --output-dir classifier/models/distilled_SciBERT_v5 \
        --results-dir classifier/results/distillation_v5_scibert \
        >> classifier/results/distillation_v5_scibert/pipeline.log 2>&1
    R=$(grep "Student distilled" classifier/results/distillation_v5_scibert/pipeline.log | tail -3 || echo "see log")
    log "TASK 3 done: $R"
    bash "$NOTIFY" "Autonomous: SciBERT distillation done" "$R" 2>/dev/null || true
else
    log "TASK 3: SKIPPED — distilled_SciBERT_v5 exists"
fi
check_5h

# ── TASK 4: Distill BiomedBERT with very sharp teacher (T=1.5, α=0.5) ──────
SHARP_DIR="classifier/models/distilled_BiomedBERT_v6"
if [ ! -d "$SHARP_DIR" ] || [ -z "$(ls -A $SHARP_DIR 2>/dev/null)" ]; then
    log "TASK 4: Distilling BiomedBERT with T=1.5, α=0.5 (very sharp teacher)..."
    mkdir -p classifier/results/distillation_v6_sharp
    CUDA_VISIBLE_DEVICES=0 python -u classifier/scripts/distill_ensemble.py \
        --skip-labels \
        --epochs 6 --temperature 1.5 --alpha 0.5 --lr 2e-5 \
        --output-dir classifier/models/distilled_BiomedBERT_v6 \
        --results-dir classifier/results/distillation_v6_sharp \
        >> classifier/results/distillation_v6_sharp/pipeline.log 2>&1
    R=$(grep "Student distilled" classifier/results/distillation_v6_sharp/pipeline.log | tail -3 || echo "see log")
    log "TASK 4 done: $R"
    bash "$NOTIFY" "Autonomous: BiomedBERT T=1.5 distillation done" "$R" 2>/dev/null || true
else
    log "TASK 4: SKIPPED — distilled_BiomedBERT_v6 exists"
fi
check_5h

# ── TASK 5: Fine-tune distilled_v2 on v18_hybrid data ───────────────────────
FINETUNED_DIR="classifier/models/distilled_BiomedBERT_v2_finetuned"
if [ ! -d "$FINETUNED_DIR" ] || [ -z "$(ls -A $FINETUNED_DIR 2>/dev/null)" ]; then
    log "TASK 5: Fine-tuning distilled_v2 on v18_hybrid data (3 epochs, lr=5e-6)..."
    mkdir -p classifier/results/distillation_v2_finetuned
    CUDA_VISIBLE_DEVICES=0 python -u classifier/scripts/finetune_distilled.py \
        --model classifier/models/distilled_BiomedBERT_v2 \
        --data classifier/data/training/v18_hybrid/dataset.csv \
        --epochs 3 --lr 5e-6 \
        --output-dir classifier/models/distilled_BiomedBERT_v2_finetuned \
        --results-dir classifier/results/distillation_v2_finetuned \
        >> classifier/results/distillation_v2_finetuned/pipeline.log 2>&1
    R=$(grep "Finetuned" classifier/results/distillation_v2_finetuned/pipeline.log | tail -3 || echo "see log")
    log "TASK 5 done: $R"
    bash "$NOTIFY" "Autonomous: fine-tuned distilled_v2 on v18" "$R" 2>/dev/null || true
else
    log "TASK 5: SKIPPED — distilled_BiomedBERT_v2_finetuned exists"
fi
check_5h

# ── TASK 6: Final report ─────────────────────────────────────────────────────
FINAL_REPORT="classifier/results/autonomous_30h/final_report.txt"
if [ ! -f "$FINAL_REPORT" ]; then
    log "TASK 6: Generating final comparison report..."
    python3 -u - << 'PYEOF' > "$FINAL_REPORT" 2>&1
import json
from pathlib import Path

BASE = Path("classifier/results")

def load_eval(path):
    try:
        return json.load(open(path))
    except Exception:
        return None

lines = [
    "=" * 70,
    "AUTONOMOUS 30H PIPELINE — FINAL RESULTS",
    "=" * 70,
    "",
    "DISTILLATION VARIANTS (BiomedBERT student unless noted)",
    "-" * 70,
    f"{'Model':<40} {'EP F1':>7} {'e100 F1':>8} {'Synth F1':>9}",
    "-" * 70,
]

models = [
    ("distilled_v1 (T=4, α=0.7)",    "distillation_v1/eval_results.json"),
    ("distilled_v2 (T=2, α=0.5)",    "distillation_v2/eval_results.json"),
    ("distilled_v3 (T=4, α=0.9)",    "distillation_v3/eval_results.json"),
    ("distilled_v4 DistilBERT",       "distillation_v4_distilbert/eval_results.json"),
    ("distilled_v5 SciBERT",          "distillation_v5_scibert/eval_results.json"),
    ("distilled_v6 BiomedBERT T=1.5", "distillation_v6_sharp/eval_results.json"),
    ("distilled_v2 finetuned v18",    "distillation_v2_finetuned/eval_results.json"),
]

for label, path in models:
    r = load_eval(BASE / path)
    if r is None:
        lines.append(f"  {label:<38} {'N/A':>7}")
        continue
    ep   = next((x for x in r if "EP"    in x.get("name","") or "ep" in x.get("name","").lower()), None)
    e100 = next((x for x in r if "100"   in x.get("name","") or "eval_100" in x.get("name","")), None)
    syn  = next((x for x in r if "synth" in x.get("name","").lower()), None)
    ep_f1  = f"{ep['f1']:.3f}"   if ep   else " — "
    e100f1 = f"{e100['f1']:.3f}" if e100 else " — "
    syn_f1 = f"{syn['f1']:.3f}"  if syn  else " — "
    lines.append(f"  {label:<38} {ep_f1:>7} {e100f1:>8} {syn_f1:>9}")

lines += [
    "-" * 70,
    "Reference: ensemble (orig BERT×T5)  0.857            0.950",
    "Reference: BiomedBERT v7 cv_reg     0.788",
    "Reference: FLAN-T5-base v11.1       0.818",
    "",
    "ENSEMBLE COMBINATIONS",
    "-" * 70,
]

combo_r = load_eval(BASE / "autonomous_30h/ensemble_comparison.json")
if combo_r:
    for name, vals in combo_r.items():
        ep   = vals.get("ep_relax", {}).get("f1", 0)
        e100 = vals.get("eval_100", {}).get("f1", 0)
        syn  = vals.get("synthetic_gold", {}).get("f1", 0)
        lines.append(f"  {name:<40} {ep:.3f}   {e100:.3f}    {syn:.3f}")

lines += ["", "=" * 70]
print("\n".join(lines))
PYEOF
    cat "$FINAL_REPORT"
    bash "$NOTIFY" "Autonomous 30h pipeline COMPLETE" "$(cat $FINAL_REPORT | head -30)" 2>/dev/null || true
    log "TASK 6 done. Final report saved to $FINAL_REPORT"
else
    log "TASK 6: SKIPPED — final_report.txt exists"
fi

log "=== Autonomous 30h pipeline finished all tasks ==="
bash "$NOTIFY" "All autonomous tasks complete" "$(tail -5 $LOG)" 2>/dev/null || true
