#!/usr/bin/env bash
# Document-scope BioRED arm: the pair-conditioned verifier given the whole marked abstract.
set -u
cd "$(dirname "$0")/.."
source "${VENV:-../MPvenv}/bin/activate"     # the shared environment beside classifier/
D=data/benchmarks/biored_bc8_doc; mkdir -p logs/biored_bc8
for s in 1 2 3; do
  out=models/biored_bc8_doc/pair_s${s}
  [ -f $out/student_config.json ] && { echo "skip $out"; continue; }
  python3 experiments/multitask/train_student.py --data $D/train.csv --dev-data $D/dev.csv \
      --input-format pair --max-len 512 --seed $s --epochs 3 --out $out > logs/biored_bc8/doc_pair_s${s}.log 2>&1
  tail -1 logs/biored_bc8/doc_pair_s${s}.log
done
