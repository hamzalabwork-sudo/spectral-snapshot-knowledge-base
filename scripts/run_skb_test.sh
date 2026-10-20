#!/bin/bash
# SKB on the test split, with the rule fixed on validation (pooled per-bin stacking) and the same seeds
# as the baselines (five within a dataset, ten across domains). Results: results/test_route/<benchmark>/.
# Usage: bash scripts/run_skb_test.sh <benchmark|all> [learner ...]   (default: ifr_uniform ifr clfd clfd_ifr)
# The baselines themselves (official runs on all of the data) come from scripts/run_benchmark.sh.
source "$(dirname "$0")/common.sh"
target=$1; shift || true
learners=${*:-"ifr_uniform ifr clfd clfd_ifr"}
benches=$([ "$target" = all ] && echo "$BENCHMARKS" || echo "$target")
for b in $benches; do
  read -r cfg data <<< "$(bench_config "$b")"
  for m in $learners; do
    $PY -u run_continual.py --config "$cfg" --data "$data" --methods "$m" --seq_len 96 --pred_len 48 \
      --epochs 20 --seeds $(bench_seeds "$b") --set routing.case_ratio=0.1 \
      --out_dir "results/test_route/$b" 2>&1 | tee "logs/skb_test_${b}_${m}.log"
  done
done
