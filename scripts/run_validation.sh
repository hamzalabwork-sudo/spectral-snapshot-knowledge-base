#!/bin/bash
# Validation-split runs (the test segment is never loaded): seeds 0-2, results in results/val/<benchmark>/.
# Usage: bash scripts/run_validation.sh <benchmark|all> <method> [method ...]
#   bash scripts/run_validation.sh all ifr ifr_uniform clfd clfd_ifr
#   bash scripts/run_validation.sh all sg_ft sg_er sg_ifr clfd_sg_ifr    # output-gate variants (Discussion)
source "$(dirname "$0")/common.sh"
target=$1; shift
benches=$([ "$target" = all ] && echo "$BENCHMARKS" || echo "$target")
for b in $benches; do
  read -r cfg data <<< "$(bench_config "$b")"
  for m in "$@"; do
    $PY -u run_continual.py --config "$cfg" --data "$data" --methods "$m" --seq_len 96 --pred_len 48 \
      --epochs 20 --seeds 0 1 2 --holdout val --out_dir "results/val/$b" 2>&1 | tee "logs/val_${b}_${m}.log"
  done
done
