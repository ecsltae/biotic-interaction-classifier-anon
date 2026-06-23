#!/bin/bash
# Self-directing research agent — MetaP classifier
#
# Each iteration: Claude reads current state, picks next experiment, runs it,
# writes findings to NOTES.md, then hands back to this loop for the next step.
# Runs indefinitely. Safe: never overwrites existing data/models.
#
# Usage:  bash classifier/scripts/research_agent.sh
# Watch:  tail -f /tmp/research_agent.log
# Notes:  classifier/results/research_agent/NOTES.md

set -uo pipefail

BASE=/path/to/MetaP
NOTES=$BASE/classifier/results/research_agent/NOTES.md
LOG=/tmp/research_agent.log
NOTIFY=$BASE/classifier/scripts/notify.sh
CLAUDE=~/.local/bin/claude
MAX_ITERATIONS=999       # effectively unlimited — time-based control instead
STEP_TIMEOUT=3600        # 1h per claude step max
EMAIL_EVERY=5            # send email every N completed steps
RATE_LIMIT_WINDOW=18000  # 5h in seconds — Claude Code rate-limit window
RATE_LIMIT_BUFFER=120    # extra seconds to wait after window before retrying

SESSION_START=$(date +%s)

mkdir -p "$BASE/classifier/results/research_agent"
cd "$BASE"

log() { echo "[$(date '+%Y-%m-%d %H:%M')] $*" | tee -a "$LOG"; }

# ── Seed NOTES.md if empty ───────────────────────────────────────────────────
if [[ ! -s "$NOTES" ]]; then
cat > "$NOTES" << 'SEED'
# MetaP Classifier — Self-Directing Research Agent

## Context
PhD project: sentence-level biotic interaction detection (binary classification).
EP test set: 100 samples, 48 positives (globi-relax EP.tsv). This is the primary metric.
All F1 figures below are on the EP test set via threshold-optimized evaluation.

## Models available (saved checkpoints)
- transformer_BiomedBERT_cv_regularized   → avg EP F1=0.788  (v7 data, discriminative)
- transformer_BiomedBERT_v11_1            → avg EP F1=0.747  (v11_1 data)
- transformer_BiomedBERT_v11_regularized  → avg EP F1=0.745
- transformer_BiomedBERT_v12_regularized  → avg EP F1=0.729
- flan-t5-base_v10                        → avg EP F1=0.735
- flan-t5-base_v10.1                      → avg EP F1=0.781
- flan-t5-base_v11_1                      → avg EP F1=0.781  ← best generative solo
- flan-t5-base_v12                        → avg EP F1=0.757
- flan-t5-base_v13                        → avg EP F1=0.776
- flan-t5-base_v14                        → avg EP F1=0.706  (regression)
(overnight: flan-t5-base_v14_unfiltered and flan-t5-base_v14_quality — results pending)

## Ensemble results (geometric mean, BiomedBERT_v7 × FLAN-T5-base_v11_1)
- F1=0.854  Prec=0.800  Rec=0.917  ← CURRENT BEST (as of 2026-03-23)
- Saved probs: results/ensemble_biomedbert_flant5/probs_ep_test.npz

## Training datasets
- v7  (25K, templates, LLM-validated)  — v7 BiomedBERT F1=0.788, gold standard
- v11_1 (28.7K, +real PMC sentences, diverse)  — best for generative models
- v12 (27.7K, +signal-filtered PMC)
- v13 (newer, partial results)
- v14 (37K, larger — caused regression, likely too noisy)
- SIBiLS (10K, unvalidated — use ensemble to pseudo-label first)
Data files: classifier/data/training/
Eval files: classifier/data/evaluation/

## Key scripts
- Train generative:     python classifier/src/models/flan_t5_classifier.py --model google/flan-t5-base --train-data ... --epochs 5 --batch-size 16 --output-dir ... --results-dir ...
- Train discriminative: python classifier/scripts/train_cv_regularized.py --models BiomedBERT --train-data ... --suffix ... --epochs 5
- Ensemble eval:        python classifier/scripts/ensemble_biomedbert_flant5.py --eval-all
- Evaluate all EP sets: python classifier/scripts/evaluate_all_ep.py
- Notify:               bash classifier/scripts/notify.sh "Subject" "Body"

## Rules
- NEVER overwrite files in data/training/ or data/evaluation/
- NEVER overwrite existing model checkpoints in models/
- All new datasets go to results/research_agent/ first (draft), not data/training/
- Save new models to models/<name>_ra_<date>/ to avoid conflicts
- Max 8 epochs per training run
- Always append findings to THIS file (NOTES.md) before finishing a step
- GPU: ~40GB A100. Check with: nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
- venv: source /path/to/MetaP/MPvenv/bin/activate

## Completed steps
(none yet — agent starting now)

## Ideas to explore (prioritized)
1. Run FLAN-T5-base on v11_1 with 8 epochs — check if 5 was under-trained
2. Multi-T5 ensemble: geometric mean across multiple FLAN-T5 versions (v10.1+v11_1+v12)
3. SIBiLS pseudo-labeling: score 10K SIBiLS with ensemble, extract conf>=0.85 → v15 draft
4. Train on v15 pseudo → test if self-training improves solo model
5. BiomedBERT temperature calibration → test if calibrated ensemble > 0.854
6. Error analysis: categorize FN/FP by interaction type → identify harvest targets
7. BiomedBERT on v11_1 with regularized script (torch issue was with checkpoint resume, fresh train should work)
8. Cross-validation of ensemble threshold on held-out folds
9. Try different ensemble strategies: learned stacking vs geometric
10. Harvest more data for categories where ensemble still fails (from error analysis)
SEED
fi

# ── Main agent loop ───────────────────────────────────────────────────────────

log "=== Research agent started — runs indefinitely, auto-resumes after 5h rate-limit ==="
log "    Session start: $(date)"

for STEP in $(seq 1 $MAX_ITERATIONS); do
    log "──── Step $STEP ────"

    PROMPT="You are running autonomously as a research agent on the MetaP biotic interaction classifier project.
Working directory: /path/to/MetaP
Python venv: source /path/to/MetaP/MPvenv/bin/activate

Your job for this step:
1. Read classifier/results/research_agent/NOTES.md to understand current state and what has already been done
2. Check GPU memory: nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
3. Check any recently completed training logs or result files that exist but haven't been analysed yet
4. Based on results so far, decide the single most valuable next experiment to run
5. Run it (use Bash, prefer GPU training when memory allows, otherwise run analysis/ensemble work)
6. Parse and interpret the results
7. Append a concise findings block to classifier/results/research_agent/NOTES.md with:
   - What you ran and why you chose it
   - Key numbers (F1, Prec, Rec)
   - What this tells us
   - Updated priority list for next steps
8. If a result is notably good or bad, send an email via: bash classifier/scripts/notify.sh 'Subject' 'Body'

Constraints:
- NEVER overwrite existing files in data/training/, data/evaluation/, or models/
- New datasets → results/research_agent/ only
- New models → models/<name>_ra_$(date +%Y%m%d)/
- Max 8 training epochs
- If GPU is full (>35GB used), run analysis/ensemble work instead of training
- If nothing obvious to do, re-evaluate all existing models on the full EP set battery

Be decisive. Pick ONE thing and do it well. Do not ask questions — just act."

    # Run Claude non-interactively
    STEP_LOG=$(mktemp /tmp/research_agent_step_XXXX.log)
    timeout $STEP_TIMEOUT \
        "$CLAUDE" -p "$PROMPT" \
            --allowedTools "Bash,Read,Write,Edit,Glob,Grep" \
            --max-turns 20 \
        2>&1 | tee -a "$LOG" "$STEP_LOG"

    EXIT_CODE=${PIPESTATUS[0]}

    # ── Detect rate limit ─────────────────────────────────────────────────────
    if grep -qi "rate limit\|429\|quota\|too many requests\|usage limit" "$STEP_LOG" 2>/dev/null; then
        # Weekly limit → STOP completely (no point sleeping 5h, resets on Monday)
        if grep -qi "weekly\|week\|claude.ai/settings" "$STEP_LOG" 2>/dev/null; then
            log "=== WEEKLY LIMIT REACHED — stopping permanently. Resets Monday 11:00 AM. ==="
            bash "$NOTIFY" "MetaP agent — WEEKLY LIMIT REACHED — STOPPED" \
                "Weekly usage limit hit at step $STEP on $(date). Agent stopped. Restart manually after Monday 11:00 AM reset." 2>/dev/null || true
            rm -f "$STEP_LOG"
            exit 0
        fi
        # Session limit (5h window) → sleep until reset, then re-exec
        NOW=$(date +%s)
        ELAPSED=$(( NOW - SESSION_START ))
        SLEEP_SECS=$(( RATE_LIMIT_WINDOW - ELAPSED + RATE_LIMIT_BUFFER ))
        [[ $SLEEP_SECS -lt 60 ]] && SLEEP_SECS=60
        WAKE=$(date -d "+${SLEEP_SECS} seconds" '+%Y-%m-%d %H:%M')
        log "=== SESSION RATE LIMIT HIT — sleeping ${SLEEP_SECS}s, resuming at $WAKE ==="
        bash "$NOTIFY" "MetaP agent — session limit, resuming $WAKE" \
            "Hit session rate limit at step $STEP. Will auto-resume at $WAKE." 2>/dev/null || true
        rm -f "$STEP_LOG"
        sleep "$SLEEP_SECS"
        exec bash "$0"
    fi
    rm -f "$STEP_LOG"

    if [[ $EXIT_CODE -eq 124 ]]; then
        log "Step $STEP timed out after ${STEP_TIMEOUT}s — continuing"
    elif [[ $EXIT_CODE -ne 0 ]]; then
        log "Step $STEP exited with code $EXIT_CODE — pausing 60s then continuing"
        sleep 60
    else
        log "Step $STEP completed"
    fi

    # Email every N steps — only if NOTES.md has new content since last email
    if (( STEP % EMAIL_EVERY == 0 )); then
        NOTES_HASH=$(tail -80 "$NOTES" | md5sum | cut -d' ' -f1)
        LAST_HASH_FILE=/tmp/research_agent_last_email_hash
        LAST_HASH=$(cat "$LAST_HASH_FILE" 2>/dev/null || echo "")
        if [[ "$NOTES_HASH" != "$LAST_HASH" ]]; then
            SUMMARY=$(tail -80 "$NOTES" | head -60)
            bash "$NOTIFY" "MetaP research agent — step $STEP done" \
                "Step $STEP completed at $(date)

Latest notes:
$SUMMARY

Full log: $LOG" 2>/dev/null || true
            echo "$NOTES_HASH" > "$LAST_HASH_FILE"
            log "Status email sent (step $STEP) — content changed"
        else
            log "Skipped email (step $STEP) — no new content in NOTES"
        fi
    fi

    # Brief pause between steps (let GPU cool, avoid hammering API)
    sleep 10
done

log "=== Agent reached max iterations ($MAX_ITERATIONS). Stopping. ==="
bash "$NOTIFY" "MetaP research agent — finished" \
    "Reached $MAX_ITERATIONS steps at $(date). See $NOTES" 2>/dev/null || true
