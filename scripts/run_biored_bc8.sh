#!/usr/bin/env bash
# BioRED under the BioREDirect/BC8 protocol: train on BioRED train+dev, select on the BioRED
# test split, report on the BioCreative VIII test set. Same encoder and recipe as Paper A's arms.
set -u
cd "$(dirname "$0")/.."
source "${VENV:-../MPvenv}/bin/activate"     # the shared environment beside classifier/
D=data/benchmarks/biored_bc8; mkdir -p logs/biored_bc8
for fmt in ${FORMATS:-sentence pair pair_mark mark_canon}; do
  for s in 1 2 3; do
    out=models/biored_bc8/${fmt}_s${s}
    [ -f $out/student_config.json ] && { echo "skip $out"; continue; }
    python3 experiments/multitask/train_student.py --data $D/train.csv --dev-data $D/dev.csv \
        --input-format $fmt --seed $s --epochs ${EPOCHS:-3} --out $out > logs/biored_bc8/${fmt}_s${s}.log 2>&1
    tail -1 logs/biored_bc8/${fmt}_s${s}.log
  done
done
