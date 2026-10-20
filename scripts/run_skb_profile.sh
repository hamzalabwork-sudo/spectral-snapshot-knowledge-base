#!/bin/bash
# Efficiency profiling of SKB (CLFD learner, 10% case segments) with the protocol of scripts/run_profile.sh:
# one run at a time on an otherwise idle GPU, seed 0, 3 epochs per regime, Traffic and cross-domain order A.
# The run records the knowledge-base update time, the weight-fit time, the stored bytes and the inference
# latency of the deployed SKB module (profile["skb"]). Results: results/profile_skb/{traffic,order_A}/.
source "$(dirname "$0")/common.sh"
$PY -u run_continual.py --config configs/within/traffic.yaml --data data/traffic.csv --methods clfd \
  --seq_len 96 --pred_len 48 --epochs 3 --seeds 0 --set routing.case_ratio=0.1 --out_dir results/profile_skb/traffic
$PY -u run_continual.py --config configs/crossdomain/crossdomain_A.yaml --data data --methods clfd \
  --epochs 3 --seeds 0 --set routing.case_ratio=0.1 --out_dir results/profile_skb/order_A
