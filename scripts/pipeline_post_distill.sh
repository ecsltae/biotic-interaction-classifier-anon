#!/usr/bin/env bash
# Post-distillation autonomous pipeline
# Chains: distillation completion → evaluation → hyperparameter variants → ensemble eval
# Rules: no API key, 5h limit, email on each step, self-restarts on limit

set -e
cd /path/to/MetaP
source MPvenv/bin/activate

NOTIFY="classifier/scripts/notify.sh"
LOG="classifier/results/pipeline_post_distill.log"
DISTILL_LOG="classifier/results/distillation_v1/pipeline.log"
SESSION_START=$(date +%s)
mkdir -p classifier/results/distillation_v1
mkdir -p classifier/results/distillation_v2
mkdir -p classifier/results/distillation_v3

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

check_5h() {
    local now=$(date +%s)
    local elapsed=$(( (now - SESSION_START) / 3600 ))
    if [ $elapsed -ge 5 ]; then
        log "5-HOUR LIMIT — relaunching pipeline_post_distill.sh"
        bash "$NOTIFY" "post-distill pipeline: 5h limit, relaunching" "Check pipeline_post_distill.log" 2>/dev/null || true
        nohup bash /path/to/MetaP/classifier/scripts/pipeline_post_distill.sh \
            >> "$LOG" 2>&1 &
        exit 0
    fi
}

log "=== Post-distillation pipeline started ==="

# ── Wait for distillation v1 to finish ─────────────────────────────────────
DISTILL_PID=$(pgrep -f "distill_ensemble.py" | head -1 || true)
if [ -n "$DISTILL_PID" ]; then
    log "Distillation v1 running (PID $DISTILL_PID) — waiting..."
    while kill -0 "$DISTILL_PID" 2>/dev/null; do
        sleep 60
        check_5h
    done
    log "Distillation v1 process finished."
fi

# ── Parse v1 results ────────────────────────────────────────────────────────
V1_EP_F1=""
if [ -f "classifier/results/distillation_v1/eval_results.json" ]; then
    V1_EP_F1=$(python3 -c "
import json
r = json.load(open('classifier/results/distillation_v1/eval_results.json'))
ep = [x for x in r if 'EP-relax' in x['name']]
print(f\"{ep[0]['f1']:.3f}\" if ep else '?')
" 2>/dev/null || echo "?")
    log "Distillation v1 results: EP F1=$V1_EP_F1"
    R=$(python3 -c "
import json
r = json.load(open('classifier/results/distillation_v1/eval_results.json'))
for x in r: print(f\"  {x['name']}: F1={x['f1']:.3f} Prec={x['prec']:.3f} Rec={x['rec']:.3f}\")
" 2>/dev/null || grep "Student distilled" "$DISTILL_LOG" | tail -4)
    log "Full results: $R"
    bash "$NOTIFY" "Distillation v1 done — EP F1=$V1_EP_F1" "$R" 2>/dev/null || true
else
    log "WARNING: distillation v1 eval_results.json not found — may have crashed"
    bash "$NOTIFY" "Distillation v1 may have crashed" "Check distillation_v1/pipeline.log" 2>/dev/null || true
fi
check_5h

# ── Evaluate ensemble on synthetic gold (not done yet) ──────────────────────
SYNTH_ENSEMBLE="classifier/results/ensemble_synthetic_eval.json"
if [ ! -f "$SYNTH_ENSEMBLE" ]; then
    log "Evaluating ensemble (cv_reg × T5-v12) on synthetic gold..."
    python3 -u - <<'PYEOF' >> "$LOG" 2>&1
import pandas as pd, torch, numpy as np, json
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration

BASE = Path("/path/to/MetaP/classifier")
synth = pd.read_csv(BASE/"data/evaluation/synthetic_gold_100.tsv", sep="\t")
texts = synth["text"].tolist(); y_true = synth["label"].tolist()
device = torch.device("cuda:0")

def get_bert(path, texts):
    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    m = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True).to(device).eval()
    probs = []
    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=256).to(device)
            probs.append(torch.softmax(m(**enc).logits, dim=-1)[0,1].item())
    m.cpu(); del m; torch.cuda.empty_cache()
    return np.array(probs)

def get_t5(path, texts):
    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    m = T5ForConditionalGeneration.from_pretrained(str(path), local_files_only=True).to(device).eval()
    yes_id = tok.encode("yes", add_special_tokens=False)[0]
    no_id  = tok.encode("no",  add_special_tokens=False)[0]
    probs = []
    with torch.no_grad():
        for t in texts:
            prompt = f"Does the following sentence describe a biotic interaction between two species? Answer yes or no.\n\nSentence: {t}"
            enc = tok(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            dec = torch.full((1,1), tok.pad_token_id, dtype=torch.long, device=device)
            logits = m(**enc, decoder_input_ids=dec).logits[0, 0]
            lp = torch.log_softmax(logits.float(), dim=-1)
            yes_lp = lp[yes_id].item(); no_lp = lp[no_id].item()
            probs.append(np.exp(yes_lp)/(np.exp(yes_lp)+np.exp(no_lp)))
    m.cpu(); del m; torch.cuda.empty_cache()
    return np.array(probs)

p_bert = get_bert(BASE/"models/transformer_BiomedBERT_cv_regularized", texts)
p_t5   = get_t5(BASE/"models/flan-t5-base_v12", texts)
p_ens  = np.sqrt(p_bert * p_t5)

results = {}
for name, p in [("bert_only", p_bert), ("t5_only", p_t5), ("ensemble_geo", p_ens)]:
    best_f1, bt = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.05):
        f = f1_score(y_true, (p>=t).astype(int))
        if f > best_f1: best_f1, bt = f, t
    preds = (p>=bt).astype(int)
    results[name] = {"f1": best_f1, "prec": float(precision_score(y_true,preds,zero_division=0)),
                     "rec": float(recall_score(y_true,preds,zero_division=0)), "thr": bt}
    print(f"  {name}: F1={best_f1:.3f}  Prec={results[name]['prec']:.3f}  Rec={results[name]['rec']:.3f}")

json.dump(results, open("/path/to/MetaP/classifier/results/ensemble_synthetic_eval.json","w"), indent=2)
print("Saved ensemble_synthetic_eval.json")
PYEOF
    log "Ensemble synthetic gold eval done."
fi
check_5h

# ── Distillation v2: lower temperature T=2 ─────────────────────────────────
STUDENT_V2="classifier/models/distilled_BiomedBERT_v2"
if [ ! -d "$STUDENT_V2" ] || [ -z "$(ls -A $STUDENT_V2 2>/dev/null)" ]; then
    log "Distillation v2: T=2, alpha=0.5 (sharper teacher, balanced loss)..."
    CUDA_VISIBLE_DEVICES=0 python -u classifier/scripts/distill_ensemble.py \
        --skip-labels \
        --epochs 6 --temperature 2 --alpha 0.5 --lr 2e-5 \
        --output-dir classifier/models/distilled_BiomedBERT_v2 \
        --results-dir classifier/results/distillation_v2 \
        >> classifier/results/distillation_v2/pipeline.log 2>&1
    R=$(grep "Student distilled" classifier/results/distillation_v2/pipeline.log | tail -3 || echo "see log")
    log "Distillation v2 done: $R"
    bash "$NOTIFY" "Distillation v2 done (T=2, α=0.5)" "$R" 2>/dev/null || true
else
    log "Distillation v2: SKIPPED — model exists"
fi
check_5h

# ── Distillation v3: alpha=0.9 (almost pure soft labels) ───────────────────
STUDENT_V3="classifier/models/distilled_BiomedBERT_v3"
if [ ! -d "$STUDENT_V3" ] || [ -z "$(ls -A $STUDENT_V3 2>/dev/null)" ]; then
    log "Distillation v3: T=4, alpha=0.9 (soft-label dominant)..."
    CUDA_VISIBLE_DEVICES=0 python -u classifier/scripts/distill_ensemble.py \
        --skip-labels \
        --epochs 6 --temperature 4 --alpha 0.9 --lr 2e-5 \
        --output-dir classifier/models/distilled_BiomedBERT_v3 \
        --results-dir classifier/results/distillation_v3 \
        >> classifier/results/distillation_v3/pipeline.log 2>&1
    R=$(grep "Student distilled" classifier/results/distillation_v3/pipeline.log | tail -3 || echo "see log")
    log "Distillation v3 done: $R"
    bash "$NOTIFY" "Distillation v3 done (T=4, α=0.9)" "$R" 2>/dev/null || true
else
    log "Distillation v3: SKIPPED — model exists"
fi
check_5h

log "=== Post-distillation pipeline complete ==="
bash "$NOTIFY" "All distillation variants complete" "Check pipeline_post_distill.log for full comparison" 2>/dev/null || true
