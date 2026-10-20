#!/bin/bash
# Efficiency profiling: one method at a time on an otherwise idle GPU (seed 0, 3 epochs per regime),
# on Traffic and cross-domain order A. Results: results/profile/{traffic,order_A}/.
# Then: python experiments/count_params.py   (total parameter counts, incl. frozen ones)
source "$(dirname "$0")/common.sh"
for m in $METHODS; do
  $PY -u run_continual.py --config configs/within/traffic.yaml --data data/traffic.csv --methods "$m" \
    --seq_len 96 --pred_len 48 --epochs 3 --seeds 0 --out_dir results/profile/traffic
  $PY -u run_continual.py --config configs/crossdomain/crossdomain_A.yaml --data data --methods "$m" \
    --epochs 3 --seeds 0 --out_dir results/profile/order_A
done
$PY experiments/count_params.py
