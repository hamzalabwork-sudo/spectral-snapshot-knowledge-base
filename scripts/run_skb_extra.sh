#!/bin/bash
# Additional SKB experiments.
# Usage: bash scripts/run_skb_extra.sh <casesize|h96|linear>
#   casesize : case segments of 5% and 20% (validation split, CLFD learner, seeds 0-2)
#              -> results/val_route_r05/, results/val_route_r20/
#   h96      : horizon 96 on the test split, official CLFD and CLFD + SKB (seeds 0-4)
#              -> results/test_h96/, results/test_h96_route/
#   linear   : linear backbone on the test split, official ER and ER + SKB (benchmark seeds)
#              -> results/test_linear/, results/test_linear_route/
source "$(dirname "$0")/common.sh"
run() {   # run <benchmark> <method> <holdout> <horizon> <seeds> <out_root> [k=v ...]
  local b=$1 m=$2 holdout=$3 H=$4 seeds=$5 out=$6; shift 6
  read -r cfg data <<< "$(bench_config "$b")"
  local sets=(); [ $# -gt 0 ] && sets=(--set "$@")
  $PY -u run_continual.py --config "$cfg" --data "$data" --methods "$m" --seq_len 96 --pred_len "$H" \
    --epochs 20 --seeds $seeds --holdout "$holdout" "${sets[@]}" --out_dir "$out/$b" \
    2>&1 | tee "logs/$(basename "$out")_${b}_${m}.log"
}
case ${1:-} in
  casesize)
    for b in $BENCHMARKS; do
      run "$b" clfd val 48 "0 1 2" results/val_route_r05 routing.case_ratio=0.05
      run "$b" clfd val 48 "0 1 2" results/val_route_r20 routing.case_ratio=0.2
    done ;;
  h96)
    for b in $BENCHMARKS; do
      run "$b" clfd test 96 "0 1 2 3 4" results/test_h96
      run "$b" clfd test 96 "0 1 2 3 4" results/test_h96_route routing.case_ratio=0.1
    done ;;
  linear)
    for b in $BENCHMARKS; do
      run "$b" ifr_uniform test 48 "$(bench_seeds "$b")" results/test_linear model.backbone=linear
      run "$b" ifr_uniform test 48 "$(bench_seeds "$b")" results/test_linear_route model.backbone=linear routing.case_ratio=0.1
    done ;;
  *) echo "usage: bash scripts/run_skb_extra.sh <casesize|h96|linear>" >&2; exit 1 ;;
esac
