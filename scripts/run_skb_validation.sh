#!/bin/bash
# SKB on the validation split (design search): a learner trained with 10% case segments held out;
# every snapshot rule is evaluated on the same snapshots. Seeds 0-2. Results: results/val_route/<benchmark>/
# (result JSONs with every rule under "routing", and snapshot predictions under snapshots/*.npz).
# Usage: bash scripts/run_skb_validation.sh <benchmark|all> [learner ...]   (default learners: ifr clfd clfd_ifr)
# The official validation runs of the learners themselves, needed as references, come from
#   bash scripts/run_validation.sh all ifr_uniform ifr clfd clfd_ifr
source "$(dirname "$0")/common.sh"
target=$1; shift || true
learners=${*:-"ifr clfd clfd_ifr"}
benches=$([ "$target" = all ] && echo "$BENCHMARKS" || echo "$target")
for b in $benches; do
  read -r cfg data <<< "$(bench_config "$b")"
  for m in $learners; do
    $PY -u run_continual.py --config "$cfg" --data "$data" --methods "$m" --seq_len 96 --pred_len 48 \
      --epochs 20 --seeds 0 1 2 --holdout val --set routing.case_ratio=0.1 \
      --out_dir "results/val_route/$b" 2>&1 | tee "logs/skb_val_${b}_${m}.log"
  done
done
