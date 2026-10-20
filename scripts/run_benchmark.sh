#!/bin/bash
# Test-split runs of the continual-forecasting benchmark (L=96, H=48, 20 epochs per regime).
# Usage: bash scripts/run_benchmark.sh <benchmark|all> [method ...]
#   bash scripts/run_benchmark.sh order_A ifr ifr_uniform     # two methods on cross-domain order A
#   bash scripts/run_benchmark.sh all                         # all 16 methods on all 8 benchmarks
# Results: results/main/<benchmark>/ (within-dataset) or results/crossdomain/<benchmark>/ (cross-domain).
source "$(dirname "$0")/common.sh"
target=$1; shift || true
methods=${*:-$METHODS}
benches=$([ "$target" = all ] && echo "$BENCHMARKS" || echo "$target")
for b in $benches; do
  read -r cfg data <<< "$(bench_config "$b")"
  for m in $methods; do
    $PY -u run_continual.py --config "$cfg" --data "$data" --methods "$m" --seq_len 96 --pred_len 48 \
      --epochs 20 --seeds $(bench_seeds "$b") --out_dir "results/$(bench_group "$b")/$b" \
      2>&1 | tee "logs/benchmark_${b}_${m}.log"
  done
done
