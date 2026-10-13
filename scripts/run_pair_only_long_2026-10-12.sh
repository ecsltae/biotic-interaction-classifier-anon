#!/usr/bin/env bash
# Review response, 2026-10-12: is the pair-only model of Appendix Q under-trained? Its development
# AUPRC was still rising at epoch 3 (0.532, 0.563, 0.572 on seed 1). Same corpus, recipe, seeds and
# best-development-epoch selection as models/pair_only/biodiv_pair_only_s*, but up to 10 epochs.
# Also rescores the 100 fresh candidates (AUROC and a bootstrap interval, scripts/fresh_sample.py).
set -u
cd "$(dirname "$0")/.."
[ -n "${VENV:-}" ] && source "$VENV/bin/activate"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
LOG=results/paperA_v2/logs_pair_only_long_2026-10-12
python3 scripts/fresh_sample.py > $LOG/fresh_sample.log 2>&1
for s in 1 2 3; do
  out=models/pair_only_long/biodiv_pair_only_s$s
  [ -f $out/student_config.json ] || python3 experiments/multitask/train_student.py \
      --data data/training/distill/v3_combined_train.csv --input-format pair_only --seed $s \
      --epochs 10 --lr 2e-5 --batch-size 32 --max-len 256 --micro-batch 0 --out $out > $LOG/train_s$s.log 2>&1
  python3 scripts/paperA_pair_only.py score --model $out --bench biodiv --name biodiv_pair_only_long_s$s > $LOG/score_s$s.log 2>&1
done
touch $LOG/ALL.DONE
