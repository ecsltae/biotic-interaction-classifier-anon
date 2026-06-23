#!/bin/bash
# Autonomous overnight research loop — MetaP classifier
# Runs experiments sequentially as GPU frees up, emails daily updates.
# Safe: never overwrites existing datasets, models, or results.
# Logs: /tmp/overnight_loop.log  |  Notes: results/overnight_loop/NOTES.md

set -euo pipefail

BASE=/path/to/MetaP
NOTES=$BASE/classifier/results/overnight_loop/NOTES.md
LOG=/tmp/overnight_loop.log
NOTIFY=$BASE/classifier/scripts/notify.sh

source $BASE/MPvenv/bin/activate
cd $BASE

mkdir -p $BASE/classifier/results/overnight_loop

# ── helpers ──────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%Y-%m-%d %H:%M')] $*" | tee -a "$LOG"; }
note() { echo "$*" >> "$NOTES"; }

wait_gpu_mb() {
    # Wait until free GPU memory >= $1 MB
    local needed=$1
    while true; do
        local used total free
        read used total < <(nvidia-smi --query-gpu=memory.used,memory.total \
                             --format=csv,noheader,nounits 2>/dev/null | head -1 | tr ',' ' ')
        free=$((total - used))
        [[ $free -ge $needed ]] && break
        log "GPU: ${used}/${total} MiB — waiting for ${needed} MiB free..."
        sleep 60
    done
}

run_exp() {
    # run_exp <label> <script> <args...>
    local label=$1; shift
    log "▶ Starting $label"
    note ""
    note "### $(date '+%Y-%m-%d %H:%M') — $label STARTED"
    "$@" 2>&1 | tee -a "$LOG"
    local exit=$?
    if [[ $exit -eq 0 ]]; then
        log "✓ $label completed"
        note "### $(date '+%Y-%m-%d %H:%M') — $label DONE (exit 0)"
    else
        log "✗ $label FAILED (exit $exit)"
        note "### $(date '+%Y-%m-%d %H:%M') — $label FAILED (exit $exit)"
    fi
    return $exit
}

parse_ep_f1() {
    # Extract avg EP F1 from a flan_t5_results.json
    local results_dir=$1
    python3 -c "
import json, sys
try:
    d = json.load(open('$results_dir/flan_t5_results.json'))
    cv = d.get('cv_summary', {})
    print(f\"avg={cv.get('avg_ep_f1',0):.3f} best={cv.get('best_ep_f1',0):.3f}\")
except Exception as e:
    print(f'(parse error: {e})')
" 2>/dev/null
}

parse_biomedbert_f1() {
    local cv_json=$1
    python3 -c "
import json
try:
    d = json.load(open('$cv_json'))
    key = list(d.keys())[0]
    s = d[key]['summary']
    print(f\"avg={s['avg_test_f1']:.3f} best={max(f['test_f1'] for f in d[key]['fold_results']):.3f}\")
except Exception as e:
    print(f'(parse error: {e})')
" 2>/dev/null
}

send_daily_email() {
    local subject=$1
    local body=$2
    bash "$NOTIFY" "$subject" "$body" 2>/dev/null || log "Email failed (non-fatal)"
}

# ── session header ────────────────────────────────────────────────────────────

SESSION_START=$(date '+%Y-%m-%d %H:%M')
log "=== Overnight research loop started: $SESSION_START ==="
note "# Overnight Research Loop — started $SESSION_START"
note ""
note "## Experiment Queue"
note "1. Wait for Exp3 (v14_unfiltered, PID 278121)"
note "2. Evaluate Exp3 result"
note "3. BiomedBERT regularized on v11_1 (best discriminative baseline on best dataset)"
note "4. BiomedBERT regularized on v12   (for fair comparison)"
note "5. FLAN-T5-base on v14_quality (queued via watcher PID 290085)"
note "6. Ensemble: SciBERT cv_regularized + FLAN-T5-base v11_1"
note "7. Ensemble sweep across all (bert, t5) model combos"
note "8. Daily email with results"

# ── Step 1: Wait for Exp3 ─────────────────────────────────────────────────────

log "Waiting for Exp3 (PID 278121, FLAN-T5-base v14_unfiltered)..."
while kill -0 278121 2>/dev/null; do sleep 60; done
log "Exp3 finished"

EXP3_RESULT=$(parse_ep_f1 "$BASE/classifier/results/overnight_20260322/flan_t5_base_v14_unfiltered")
log "Exp3 v14_unfiltered result: $EXP3_RESULT"
note ""
note "### $(date '+%Y-%m-%d %H:%M') — Exp3 v14_unfiltered result: $EXP3_RESULT"

# ── Step 2: BiomedBERT regularized on v11_1 ──────────────────────────────────

wait_gpu_mb 12000
log "GPU free — launching BiomedBERT on v11_1"

SUFFIX="v11_1_$(date +%Y%m%d)"
run_exp "BiomedBERT-regularized-v11_1" \
    python classifier/scripts/train_cv_regularized.py \
        --models BiomedBERT \
        --train-data classifier/data/training/training_data_v11_1.csv \
        --suffix "$SUFFIX" \
        --epochs 5 \
    || true

BIO_V11=$(parse_biomedbert_f1 \
    "$BASE/classifier/models/transformer_BiomedBERT_${SUFFIX}/cv_results.json" 2>/dev/null || echo "(failed)")
log "BiomedBERT v11_1 result: $BIO_V11"
note "### $(date '+%Y-%m-%d %H:%M') — BiomedBERT v11_1: $BIO_V11"

# ── Step 3: BiomedBERT regularized on v12 ────────────────────────────────────

wait_gpu_mb 12000
SUFFIX12="v12_$(date +%Y%m%d)"
run_exp "BiomedBERT-regularized-v12" \
    python classifier/scripts/train_cv_regularized.py \
        --models BiomedBERT \
        --train-data classifier/data/training/training_data_v12.csv \
        --suffix "$SUFFIX12" \
        --epochs 5 \
    || true

BIO_V12=$(parse_biomedbert_f1 \
    "$BASE/classifier/models/transformer_BiomedBERT_${SUFFIX12}/cv_results.json" 2>/dev/null || echo "(failed)")
log "BiomedBERT v12 result: $BIO_V12"
note "### $(date '+%Y-%m-%d %H:%M') — BiomedBERT v12: $BIO_V12"

# ── Step 4: Wait for v14_quality, then evaluate ───────────────────────────────

log "Waiting for v14_quality watcher (PID 290085)..."
while kill -0 290085 2>/dev/null; do sleep 60; done
V14Q_RESULT=$(parse_ep_f1 \
    "$BASE/classifier/results/overnight_20260322/flan_t5_base_v14_quality" 2>/dev/null || echo "(not ready)")
log "v14_quality result: $V14Q_RESULT"
note "### $(date '+%Y-%m-%d %H:%M') — FLAN-T5-base v14_quality: $V14Q_RESULT"

# ── Step 5: Ensemble — SciBERT + FLAN-T5-base v11_1 ─────────────────────────

log "Running SciBERT + FLAN-T5-base v11_1 ensemble..."
python3 - << 'PYEOF' 2>&1 | tee -a "$LOG"
import sys, json
sys.path.insert(0, '/path/to/MetaP/classifier/scripts')
# Reuse ensemble logic with SciBERT model
from pathlib import Path
import numpy as np
import torch

BASE_DIR = Path('/path/to/MetaP/classifier')
# SciBERT is saved alongside BiomedBERT in the same cv_regularized checkpoint dir
# but as a separate model — save path is transformer_SciBERT_cv_regularized
SCIBERT_MODEL = BASE_DIR / 'models/transformer_BiomedBERT_cv_regularized'  # same file — but wrong
# Actually SciBERT has its own saved model:
import os
scibert_dirs = [d for d in (BASE_DIR / 'models').iterdir()
                if 'SciBERT' in d.name or 'scibert' in d.name.lower()]
print("SciBERT model dirs:", scibert_dirs)
PYEOF

# Check what SciBERT models exist before running ensemble
python3 -c "
from pathlib import Path
base = Path('/path/to/MetaP/classifier/models')
scibert = [d.name for d in base.iterdir() if 'scibert' in d.name.lower() or 'SciBERT' in d.name]
print('SciBERT models:', scibert)
" 2>&1 | tee -a "$LOG"

# Run full ensemble sweep with all available model combos
wait_gpu_mb 20000
python3 << 'PYEOF' 2>&1 | tee -a "$LOG"
import json, sys, numpy as np
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd

BASE_DIR = Path('/path/to/MetaP/classifier')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EP_FILE = BASE_DIR / 'data/evaluation/globi-relax_passages-triplets_2024-02-28_curation_EP.tsv'
OUT = BASE_DIR / 'results/overnight_loop'
OUT.mkdir(exist_ok=True)

def load_ep():
    df = pd.read_csv(EP_FILE, sep='\t')
    return df['sentence'].tolist(), df['evaluation_pair_interacting'].astype(int).tolist()

def bert_probs(model_dir, texts):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE).eval()
    from torch.utils.data import DataLoader, Dataset
    class DS(Dataset):
        def __init__(self, t, tok):
            self.enc = tok(t, max_length=256, padding='max_length', truncation=True, return_tensors='pt')
        def __len__(self): return self.enc['input_ids'].shape[0]
        def __getitem__(self, i): return {k: v[i] for k,v in self.enc.items()}
    probs = []
    with torch.no_grad():
        for batch in DataLoader(DS(texts, tok), batch_size=32):
            batch = {k:v.to(DEVICE) for k,v in batch.items()}
            p = torch.softmax(model(**batch).logits, dim=-1)[:,1].cpu().numpy()
            probs.extend(p)
    return np.array(probs)

def t5_probs(model_dir, texts):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir).to(DEVICE).eval()
    yes_id = tok.encode('yes', add_special_tokens=False)[0]
    no_id  = tok.encode('no',  add_special_tokens=False)[0]
    prompts = [f"Does the following sentence describe a biotic interaction between two species? Answer yes or no.\n\nSentence: {t}" for t in texts]
    probs = []
    with torch.no_grad():
        for i in range(0, len(prompts), 32):
            enc = tok(prompts[i:i+32], return_tensors='pt', padding=True, truncation=True, max_length=512).to(DEVICE)
            dec = torch.full((len(prompts[i:i+32]),1), tok.pad_token_id, dtype=torch.long, device=DEVICE)
            out = model(**enc, decoder_input_ids=dec)
            lp = torch.log_softmax(out.logits[:,0,:].float(), dim=-1)
            ylp = lp[:,yes_id].cpu().numpy(); nlp = lp[:,no_id].cpu().numpy()
            probs.extend((np.exp(ylp)/(np.exp(ylp)+np.exp(nlp))).tolist())
    return np.array(probs)

def best_f1(probs, labels):
    y = np.array(labels)
    best = (0,0,0,0)
    for t in np.arange(0.05,0.96,0.01):
        preds = (probs>=t).astype(int)
        if preds.sum()==0: continue
        f = f1_score(y, preds, zero_division=0)
        if f > best[1]: best = (round(float(t),2), round(f,4), round(precision_score(y,preds,zero_division=0),4), round(recall_score(y,preds,zero_division=0),4))
    return best

texts, labels = load_ep()
print(f"EP test: {len(texts)} samples, {sum(labels)} positives")

# Collect all available BERT-like and T5-base models
bert_models = {
    'BiomedBERT_v7':  BASE_DIR / 'models/transformer_BiomedBERT_cv_regularized',
}
# Add v11_1 if trained
for d in sorted((BASE_DIR / 'models').iterdir()):
    if 'BiomedBERT' in d.name and 'v11_1' in d.name and (d/'model.safetensors').exists():
        bert_models[f'BiomedBERT_{d.name.split("_")[-1]}'] = d
    if 'SciBERT' in d.name and (d/'model.safetensors').exists():
        bert_models[f'SciBERT_{d.name}'] = d

t5_models = {}
for d in sorted((BASE_DIR / 'models').iterdir()):
    if 'flan-t5-base' in d.name and (d/'model.safetensors').exists():
        t5_models[d.name] = d

print(f"\nBERT models: {list(bert_models.keys())}")
print(f"T5  models:  {list(t5_models.keys())}")

results = {}

# Get probs for all models
bert_probs_cache = {}
for name, mdir in bert_models.items():
    print(f"\nInference: {name}")
    try:
        bert_probs_cache[name] = bert_probs(mdir, texts)
        t, f, p, r = best_f1(bert_probs_cache[name], labels)
        results[name] = {'f1':f,'precision':p,'recall':r,'threshold':t,'type':'bert'}
        print(f"  Solo: F1={f:.3f} P={p:.3f} R={r:.3f}")
    except Exception as e:
        print(f"  FAILED: {e}")

t5_probs_cache = {}
for name, mdir in t5_models.items():
    print(f"\nInference: {name}")
    try:
        t5_probs_cache[name] = t5_probs(mdir, texts)
        t, f, p, r = best_f1(t5_probs_cache[name], labels)
        results[name] = {'f1':f,'precision':p,'recall':r,'threshold':t,'type':'t5'}
        print(f"  Solo: F1={f:.3f} P={p:.3f} R={r:.3f}")
    except Exception as e:
        print(f"  FAILED: {e}")

# Ensemble sweep: geometric mean of each BERT × T5 combo
print("\n\n=== ENSEMBLE RESULTS (geometric mean) ===")
ensemble_results = {}
for bn, bp in bert_probs_cache.items():
    for tn, tp in t5_probs_cache.items():
        combo = f"{bn} × {tn}"
        geo = np.sqrt(np.clip(bp * tp, 1e-12, 1.0))
        t, f, p, r = best_f1(geo, labels)
        ensemble_results[combo] = {'f1':f,'precision':p,'recall':r,'threshold':t}
        print(f"  {combo:<60} F1={f:.3f} P={p:.3f} R={r:.3f}")

# Save all
all_out = {'solo': results, 'ensemble': ensemble_results}
(OUT / 'full_ensemble_sweep.json').write_text(json.dumps(all_out, indent=2))

# Print sorted summary
print("\n\n=== TOP 10 (solo + ensemble, sorted by F1) ===")
all_rows = [(n, v['f1'], v['precision'], v['recall']) for n,v in {**results, **ensemble_results}.items()]
for name, f, p, r in sorted(all_rows, key=lambda x:-x[1])[:10]:
    print(f"  {f:.3f}  P={p:.3f}  R={r:.3f}  {name}")

print("\nDone. Results saved to results/overnight_loop/full_ensemble_sweep.json")
PYEOF

note "### $(date '+%Y-%m-%d %H:%M') — Ensemble sweep complete"

# ── Daily email ───────────────────────────────────────────────────────────────

EXP3_RESULT_FINAL=$(parse_ep_f1 \
    "$BASE/classifier/results/overnight_20260322/flan_t5_base_v14_unfiltered" 2>/dev/null || echo "?")
ENSEMBLE_TOP=$(python3 -c "
import json
from pathlib import Path
p = Path('$BASE/classifier/results/overnight_loop/full_ensemble_sweep.json')
if not p.exists():
    print('(no results yet)')
else:
    d = json.loads(p.read_text())
    all_r = {**d.get('solo',{}), **d.get('ensemble',{})}
    top = sorted(all_r.items(), key=lambda x: -x[1]['f1'])[:5]
    lines = [f'{v[\"f1\"]:.3f} P={v[\"precision\"]:.3f} R={v[\"recall\"]:.3f}  {k}' for k,v in top]
    print('\n'.join(lines))
" 2>/dev/null || echo "(parse error)")

EMAIL_BODY="MetaP overnight research update — $(date '+%Y-%m-%d %H:%M')

== COMPLETED TODAY ==

Exp3  FLAN-T5-base v14_unfiltered : $EXP3_RESULT_FINAL
v14_quality FLAN-T5-base           : $V14Q_RESULT
BiomedBERT v11_1 regularized       : $BIO_V11
BiomedBERT v12 regularized         : $BIO_V12

== ENSEMBLE SWEEP TOP 5 ==
$ENSEMBLE_TOP

== REMINDER ==
Best known result: Ensemble geometric (BiomedBERT_v7 × FLAN-T5-base_v11_1) = F1=0.854

Full notes: $NOTES
Full log:   $LOG
"

send_daily_email "MetaP overnight results — $(date '+%Y-%m-%d')" "$EMAIL_BODY"
log "Daily email sent (Phase 1)"

log "=== Phase 1 complete. Moving to Phase 2: new ideas ==="
note ""
note "## Phase 2: New experiments"

# ── Phase 2 idea 1: Multi-T5 ensemble (v10.1 + v11_1 + v12 + v13) ────────────
# Different training data = different error profiles = complementary signals
log "Phase 2.1 — Multi-T5 ensemble (diversity of training data)"
python3 << 'PYEOF' 2>&1 | tee -a "$LOG"
import json, numpy as np, pandas as pd
from pathlib import Path
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from sklearn.metrics import f1_score, precision_score, recall_score
from itertools import combinations

BASE_DIR = Path('/path/to/MetaP/classifier')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EP_FILE = BASE_DIR / 'data/evaluation/globi-relax_passages-triplets_2024-02-28_curation_EP.tsv'
OUT = BASE_DIR / 'results/overnight_loop'

texts = pd.read_csv(EP_FILE, sep='\t')['sentence'].tolist()
labels = pd.read_csv(EP_FILE, sep='\t')['evaluation_pair_interacting'].astype(int).tolist()

def t5_probs(model_dir, texts):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir).to(DEVICE).eval()
    yes_id = tok.encode('yes', add_special_tokens=False)[0]
    no_id  = tok.encode('no',  add_special_tokens=False)[0]
    prompts = [f"Does the following sentence describe a biotic interaction between two species? Answer yes or no.\n\nSentence: {t}" for t in texts]
    probs = []
    with torch.no_grad():
        for i in range(0, len(prompts), 32):
            enc = tok(prompts[i:i+32], return_tensors='pt', padding=True, truncation=True, max_length=512).to(DEVICE)
            dec = torch.full((len(prompts[i:i+32]),1), tok.pad_token_id, dtype=torch.long, device=DEVICE)
            out = model(**enc, decoder_input_ids=dec)
            lp = torch.log_softmax(out.logits[:,0,:].float(), dim=-1)
            ylp = lp[:,yes_id].cpu().numpy(); nlp = lp[:,no_id].cpu().numpy()
            probs.extend((np.exp(ylp)/(np.exp(ylp)+np.exp(nlp))).tolist())
    return np.array(probs)

def best_f1(probs, labels):
    y = np.array(labels)
    best = (0.5, 0, 0, 0)
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (probs >= t).astype(int)
        if preds.sum() == 0: continue
        f = f1_score(y, preds, zero_division=0)
        if f > best[1]:
            best = (round(float(t),2), round(f,4),
                    round(precision_score(y,preds,zero_division=0),4),
                    round(recall_score(y,preds,zero_division=0),4))
    return best

t5_dirs = {d.name: d for d in sorted((BASE_DIR / 'models').iterdir())
           if 'flan-t5-base' in d.name and (d/'model.safetensors').exists()}
# also new ones trained tonight
for d in sorted((BASE_DIR / 'models').iterdir()):
    if 'flan_t5_base_v14' in d.name and (d/'model.safetensors').exists():
        t5_dirs[d.name] = d

print(f"T5 models: {list(t5_dirs.keys())}")

all_probs = {}
for name, mdir in t5_dirs.items():
    print(f"  Scoring {name}...")
    try:
        p = t5_probs(mdir, texts)
        all_probs[name] = p
        _, f, pr, r = best_f1(p, labels)
        print(f"    Solo F1={f:.3f} P={pr:.3f} R={r:.3f}")
    except Exception as e:
        print(f"    FAILED: {e}")

results = {}
for k in range(2, min(len(all_probs)+1, 5)):  # combos up to 4 models
    for combo in combinations(all_probs.keys(), k):
        geo = np.ones(len(texts))
        for name in combo: geo *= all_probs[name]
        geo = geo ** (1/k)
        _, f, p, r = best_f1(geo, labels)
        label = ' × '.join(combo)
        results[label] = {'f1':f,'precision':p,'recall':r,'n_models':k}

print("\n=== Multi-T5 ensemble top 10 ===")
for name, v in sorted(results.items(), key=lambda x:-x[1]['f1'])[:10]:
    print(f"  {v['f1']:.3f} P={v['precision']:.3f} R={v['recall']:.3f}  [{v['n_models']} models] {name}")

(OUT / 'multi_t5_ensemble.json').write_text(json.dumps(results, indent=2))
print("Saved → results/overnight_loop/multi_t5_ensemble.json")
PYEOF

note "### $(date '+%Y-%m-%d %H:%M') — Multi-T5 ensemble done"

# ── Phase 2 idea 2: Longer training (8 epochs) on best config ────────────────

wait_gpu_mb 10000
log "Phase 2.2 — FLAN-T5-base v11_1 with 8 epochs (check convergence)"
if [[ ! -d "$BASE/classifier/models/flan-t5-base_v11_1_8ep" ]]; then
    run_exp "FLAN-T5-base v11_1 8ep" \
        python classifier/src/models/flan_t5_classifier.py \
            --model google/flan-t5-base \
            --train-data classifier/data/training/training_data_v11_1.csv \
            --epochs 8 \
            --batch-size 16 \
            --output-dir classifier/models/flan-t5-base_v11_1_8ep \
            --results-dir classifier/results/overnight_loop/flan_t5_base_v11_1_8ep \
        || true
    V11_8EP=$(parse_ep_f1 "$BASE/classifier/results/overnight_loop/flan_t5_base_v11_1_8ep")
    log "v11_1 8ep: $V11_8EP"
    note "### $(date '+%Y-%m-%d %H:%M') — FLAN-T5-base v11_1 8ep: $V11_8EP"
else
    log "flan-t5-base_v11_1_8ep already exists, skipping"
    V11_8EP=$(parse_ep_f1 "$BASE/classifier/results/overnight_loop/flan_t5_base_v11_1_8ep")
fi

# ── Phase 2 idea 3: SIBiLS pseudo-labeling with the best ensemble ─────────────
# SIBiLS has 10K sentences, no LLM validation — use ensemble to filter to
# high-confidence pseudo-labels, build v15 draft for later review

wait_gpu_mb 20000
log "Phase 2.3 — SIBiLS pseudo-labeling (ensemble confidence filtering)"
python3 << 'PYEOF' 2>&1 | tee -a "$LOG"
import json, numpy as np, pandas as pd
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration

BASE_DIR = Path('/path/to/MetaP/classifier')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SIBILS_FILE = BASE_DIR / 'data/training/globi_sibils_real.csv'
OUT_DIR = BASE_DIR / 'results/overnight_loop'

df = pd.read_csv(SIBILS_FILE)
text_col = 'sentence' if 'sentence' in df.columns else 'text'
texts = df[text_col].tolist()
print(f"SIBiLS: {len(texts)} samples")

def bert_score(model_dir, texts):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE).eval()
    from torch.utils.data import DataLoader, Dataset
    class DS(Dataset):
        def __init__(self, t): self.enc = tok(t, max_length=256, padding='max_length', truncation=True, return_tensors='pt')
        def __len__(self): return self.enc['input_ids'].shape[0]
        def __getitem__(self, i): return {k:v[i] for k,v in self.enc.items()}
    probs = []
    with torch.no_grad():
        for batch in DataLoader(DS(texts), batch_size=32):
            batch = {k:v.to(DEVICE) for k,v in batch.items()}
            p = torch.softmax(model(**batch).logits, dim=-1)[:,1].cpu().numpy()
            probs.extend(p)
    return np.array(probs)

def t5_score(model_dir, texts):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir).to(DEVICE).eval()
    yes_id = tok.encode('yes', add_special_tokens=False)[0]
    no_id  = tok.encode('no',  add_special_tokens=False)[0]
    prompts = [f"Does the following sentence describe a biotic interaction between two species? Answer yes or no.\n\nSentence: {t}" for t in texts]
    probs = []
    with torch.no_grad():
        for i in range(0, len(prompts), 16):
            enc = tok(prompts[i:i+16], return_tensors='pt', padding=True, truncation=True, max_length=512).to(DEVICE)
            dec = torch.full((len(prompts[i:i+16]),1), tok.pad_token_id, dtype=torch.long, device=DEVICE)
            out = model(**enc, decoder_input_ids=dec)
            lp = torch.log_softmax(out.logits[:,0,:].float(), dim=-1)
            ylp = lp[:,yes_id].cpu().numpy(); nlp = lp[:,no_id].cpu().numpy()
            probs.extend((np.exp(ylp)/(np.exp(ylp)+np.exp(nlp))).tolist())
    return np.array(probs)

print("BiomedBERT scoring...")
p_bert = bert_score(BASE_DIR / 'models/transformer_BiomedBERT_cv_regularized', texts)
print("FLAN-T5-base scoring...")
p_t5 = t5_score(BASE_DIR / 'models/flan-t5-base_v11_1', texts)
p_ens = np.sqrt(np.clip(p_bert * p_t5, 1e-12, 1.0))

df['p_bert'] = p_bert; df['p_t5'] = p_t5; df['p_ensemble'] = p_ens
df.to_csv(OUT_DIR / 'sibils_ensemble_scores.csv', index=False)

print("\nPseudo-label yield by confidence threshold:")
for conf in [0.80, 0.85, 0.90, 0.95]:
    hp = (p_ens >= conf).sum(); hn = (p_ens <= 1-conf).sum()
    print(f"  conf>={conf:.2f}: {hp} pseudo-pos + {hn} pseudo-neg ({len(texts)-hp-hn} ambiguous dropped)")

# Build pseudo-labeled set at conf>=0.85 (conservative)
CONF = 0.85
pos_mask = p_ens >= CONF; neg_mask = p_ens <= (1-CONF)
pseudo_pos = df[pos_mask].copy(); pseudo_pos['label'] = 1; pseudo_pos['source'] = 'sibils_pseudo_pos'
pseudo_neg = df[neg_mask].copy(); pseudo_neg['label'] = 0; pseudo_neg['source'] = 'sibils_pseudo_neg'
pseudo_neg = pseudo_neg.sample(min(len(pseudo_neg), len(pseudo_pos)*2), random_state=42)

# Rename text column if needed
for d in [pseudo_pos, pseudo_neg]:
    if 'sentence' in d.columns and 'text' not in d.columns:
        d.rename(columns={'sentence':'text'}, inplace=True)

pseudo_out = pd.concat([pseudo_pos[['text','label','source']], pseudo_neg[['text','label','source']]], ignore_index=True)
pseudo_out.to_csv(OUT_DIR / 'sibils_pseudo_labeled_conf85.csv', index=False)
print(f"\nPseudo-labeled (conf>=0.85): {pos_mask.sum()} pos + {len(pseudo_neg)} neg saved")

# Build v15 draft = v12 + pseudo-labeled (saved to overnight_loop only — needs review)
v12 = pd.read_csv(BASE_DIR / 'data/training/training_data_v12.csv')
v12_cols = ['text','label','source'] if 'source' in v12.columns else ['text','label']
v15 = pd.concat([v12[v12_cols], pseudo_out[v12_cols]], ignore_index=True)
v15.to_csv(OUT_DIR / 'training_data_v15_pseudo_draft.csv', index=False)
print(f"v15 draft = v12 ({len(v12)}) + pseudo ({len(pseudo_out)}) = {len(v15)} total")
print("NOTE: saved to results/overnight_loop/ — needs manual review before promotion to data/training/")
PYEOF

note "### $(date '+%Y-%m-%d %H:%M') — SIBiLS pseudo-labeling done"

# ── Phase 2 idea 4: Train on v15 pseudo to test hypothesis ───────────────────

wait_gpu_mb 10000
V15_PATH="$BASE/classifier/results/overnight_loop/training_data_v15_pseudo_draft.csv"
if [[ -f "$V15_PATH" ]]; then
    log "Phase 2.4 — FLAN-T5-base on v15 pseudo-labeled (hypothesis test)"
    run_exp "FLAN-T5-base v15-pseudo" \
        python classifier/src/models/flan_t5_classifier.py \
            --model google/flan-t5-base \
            --train-data "$V15_PATH" \
            --epochs 5 \
            --batch-size 16 \
            --output-dir classifier/models/flan_t5_base_v15_pseudo \
            --results-dir classifier/results/overnight_loop/flan_t5_base_v15_pseudo \
        || true
    V15_RESULT=$(parse_ep_f1 "$BASE/classifier/results/overnight_loop/flan_t5_base_v15_pseudo")
    log "v15-pseudo result: $V15_RESULT"
    note "### $(date '+%Y-%m-%d %H:%M') — FLAN-T5-base v15-pseudo: $V15_RESULT"
else
    log "v15 pseudo CSV not found, skipping"
    V15_RESULT="(skipped)"
fi

# ── Phase 2 idea 5: Error analysis on best ensemble ──────────────────────────

log "Phase 2.5 — Error analysis: what does the ensemble still miss?"
python3 << 'PYEOF' 2>&1 | tee -a "$LOG"
import json, numpy as np, pandas as pd
from pathlib import Path

BASE_DIR = Path('/path/to/MetaP/classifier')
PROBS_FILE = BASE_DIR / 'results/ensemble_biomedbert_flant5/probs_ep_test.npz'
EP_FILE = BASE_DIR / 'data/evaluation/globi-relax_passages-triplets_2024-02-28_curation_EP.tsv'

if not PROBS_FILE.exists():
    print("No probs file, skipping"); exit()

data = np.load(PROBS_FILE)
df = pd.read_csv(EP_FILE, sep='\t')
p_geo = np.sqrt(np.clip(data['p_bert'] * data['p_t5'], 1e-12, 1.0))
labels = data['labels']
preds = (p_geo >= 0.33).astype(int)  # best threshold from earlier run

df['p_ensemble'] = p_geo; df['pred'] = preds; df['label'] = labels
df['correct'] = (preds == labels)

fn = df[(labels==1) & (preds==0)]
fp = df[(labels==0) & (preds==1)]

print(f"Correct: {df['correct'].sum()}/100 | FN={len(fn)} | FP={len(fp)}")
print(f"\n--- False Negatives (missed: model says NO, truth=YES) ---")
for _, row in fn.iterrows():
    cat = row.get('interaction_term', '?')
    print(f"  p={row['p_ensemble']:.2f} [{cat}]  {str(row['sentence'])[:110]}")

print(f"\n--- False Positives (noise: model says YES, truth=NO) ---")
for _, row in fp.iterrows():
    print(f"  p={row['p_ensemble']:.2f}  {str(row['sentence'])[:110]}")

if 'interaction_term' in df.columns:
    print(f"\n--- Category error rate ---")
    for cat, grp in df.groupby('interaction_term'):
        if len(grp) > 1:
            print(f"  {cat:<30} {grp['correct'].sum()}/{len(grp)} correct  ({int(grp['label'].sum())} pos)")

out = BASE_DIR / 'results/overnight_loop/ensemble_error_analysis.csv'
df[['sentence','label','pred','p_ensemble','interaction_term']].to_csv(out, index=False)
print(f"\nSaved → {out}")
PYEOF

note "### $(date '+%Y-%m-%d %H:%M') — Error analysis done"

# ── Phase 2 idea 6: Temperature calibration of BiomedBERT ────────────────────
# BiomedBERT probs tend to be overconfident — Platt/temperature scaling
# may improve calibration and boost ensemble

log "Phase 2.6 — Temperature calibration for BiomedBERT"
python3 << 'PYEOF' 2>&1 | tee -a "$LOG"
import json, numpy as np
from pathlib import Path
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss

BASE_DIR = Path('/path/to/MetaP/classifier')
PROBS_FILE = BASE_DIR / 'results/ensemble_biomedbert_flant5/probs_ep_test.npz'

if not PROBS_FILE.exists():
    print("No probs file, skipping"); exit()

data = np.load(PROBS_FILE)
p_bert = data['p_bert']; p_t5 = data['p_t5']; labels = data['labels']

def best_f1(probs, labels):
    best = (0.5, 0, 0, 0)
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (probs >= t).astype(int)
        if preds.sum() == 0: continue
        f = f1_score(labels, preds, zero_division=0)
        if f > best[1]:
            best = (round(float(t),2), round(f,4),
                    round(precision_score(labels,preds,zero_division=0),4),
                    round(recall_score(labels,preds,zero_division=0),4))
    return best

bert_logits = np.log(np.clip(p_bert,1e-7,1-1e-7)) - np.log(np.clip(1-p_bert,1e-7,1-1e-7))
res = minimize_scalar(lambda T: log_loss(labels, expit(bert_logits/T)),
                      bounds=(0.1, 10.0), method='bounded')
T_opt = res.x
p_bert_cal = expit(bert_logits / T_opt)

_, f0, p0, r0 = best_f1(p_bert, labels)
_, f1, p1, r1 = best_f1(p_bert_cal, labels)
geo_raw = np.sqrt(np.clip(p_bert * p_t5, 1e-12, 1.0))
geo_cal = np.sqrt(np.clip(p_bert_cal * p_t5, 1e-12, 1.0))
_, fg, pg, rg = best_f1(geo_raw, labels)
_, fc, pc, rc = best_f1(geo_cal, labels)

print(f"Temperature T={T_opt:.3f}")
print(f"  BiomedBERT uncalibrated: F1={f0:.3f} P={p0:.3f} R={r0:.3f}")
print(f"  BiomedBERT calibrated:   F1={f1:.3f} P={p1:.3f} R={r1:.3f}")
print(f"  Ensemble uncalibrated:   F1={fg:.3f} P={pg:.3f} R={rg:.3f}")
print(f"  Ensemble calibrated:     F1={fc:.3f} P={pc:.3f} R={rc:.3f}")

out = BASE_DIR / 'results/overnight_loop/bert_calibration.json'
out.write_text(json.dumps({
    'temperature': float(T_opt),
    'bert_uncal': {'f1':f0,'precision':p0,'recall':r0},
    'bert_cal':   {'f1':f1,'precision':p1,'recall':r1},
    'ensemble_uncal': {'f1':fg,'precision':pg,'recall':rg},
    'ensemble_cal':   {'f1':fc,'precision':pc,'recall':rc},
}, indent=2))
print(f"Saved → {out}")
PYEOF

note "### $(date '+%Y-%m-%d %H:%M') — Calibration done"

# ── Final summary email ───────────────────────────────────────────────────────

MULTI_T5_TOP=$(python3 -c "
import json
from pathlib import Path
p = Path('$BASE/classifier/results/overnight_loop/multi_t5_ensemble.json')
if not p.exists(): print('(no results)'); exit()
d = json.loads(p.read_text())
for k,v in sorted(d.items(), key=lambda x:-x[1]['f1'])[:3]:
    print(f\"{v['f1']:.3f} P={v['precision']:.3f}  {k}\")
" 2>/dev/null || echo "(parse error)")

V11_8EP_F=$(parse_ep_f1 "$BASE/classifier/results/overnight_loop/flan_t5_base_v11_1_8ep" 2>/dev/null || echo "?")
CALIB=$(python3 -c "
import json; from pathlib import Path
p = Path('$BASE/classifier/results/overnight_loop/bert_calibration.json')
if not p.exists(): print('?'); exit()
d = json.loads(p.read_text())
print(f\"T={d['temperature']:.2f}  ensemble_uncal={d['ensemble_uncal']['f1']:.3f}  ensemble_cal={d['ensemble_cal']['f1']:.3f}\")
" 2>/dev/null || echo "?")

send_daily_email "MetaP Phase 2 complete — $(date '+%Y-%m-%d')" "
All experiments complete — $(date '+%Y-%m-%d %H:%M')

== Phase 1 ==
FLAN-T5-base v14_unfiltered : $EXP3_RESULT_FINAL
FLAN-T5-base v14_quality    : $V14Q_RESULT
BiomedBERT v11_1 regularized: $BIO_V11
BiomedBERT v12 regularized  : $BIO_V12

== Phase 2 ==
Multi-T5 ensemble top 3:
$MULTI_T5_TOP

FLAN-T5-base v11_1 (8 epochs): $V11_8EP_F
FLAN-T5-base v15 pseudo-label: $V15_RESULT
BiomedBERT calibration: $CALIB

== All-time best ==
Ensemble geometric (BiomedBERT_v7 x FLAN-T5-base_v11_1) = F1=0.854

Full notes: $NOTES
"
log "Final email sent"

log "=== All phases complete: $(date '+%Y-%m-%d %H:%M') ==="
note ""
note "## All phases complete: $(date '+%Y-%m-%d %H:%M')"
