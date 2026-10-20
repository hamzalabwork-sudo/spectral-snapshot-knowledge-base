#!/bin/bash
# Regenerate the paper's tables and figures from results/ (no training needed).
# Output: outputs/tables/*.tex and outputs/figs/*.pdf
# Rules evaluated offline and the two SKB figures use the snapshot predictions (results/*/snapshots/*.npz,
# written by the SKB runs). These files are too large for the repository; without them the scripts use
# the cached summaries in results/val_route/offline_*.json, results/test_route/offline_budget.json and
# results/derived/.
source "$(dirname "$0")/common.sh"
mkdir -p outputs/tables outputs/figs
have_dumps() { ls "$1"/*/snapshots/*.npz > /dev/null 2>&1; }

if have_dumps results/val_route; then   # four further rules of the design search (validation)
  $PY experiments/offline_rules.py --root results/val_route \
    --rules causal_buf_dlda causal_case_dlda stack_dlda sstack_dlda --out results/val_route/offline_rules.json
  $PY experiments/offline_rules.py --root results/val_route \
    --rules sstack_global skb_top2 skb_top3 skb_last2 skb_last3 --out results/val_route/offline_budget.json
fi
if have_dumps results/test_route; then  # memory budgets on the test split
  $PY experiments/offline_rules.py --root results/test_route \
    --rules sstack_global skb_top2 skb_top3 skb_last2 skb_last3 --out results/test_route/offline_budget.json
fi

# tables: all methods with SKB rows, overall ranks, rule ablation, plug-in, efficiency
$PY experiments/make_tables_skb.py --route_root results/test_route --out outputs/tables \
  --cite_companion suleiman2026intrusion
# tables: case-segment size, generalisation, design search
$PY experiments/make_tables_skb_extra.py --out outputs/tables
# memory budget (test split), printed
$PY experiments/compare_route.py --route_root results/test_route --refs clfd \
  --official_glob "results/pilot/*/continual_results_*.json" "results/main/*/continual_results_*.json" \
  "results/crossdomain/*/continual_results_*.json" --official clfd \
  --extra results/test_route/offline_budget.json \
  --only clfd+sstack_global clfd+skb_top2 clfd+skb_top3 clfd+skb_last2 clfd+skb_last3 clfd+final
# figures (and Table 2): from the snapshot predictions if present, otherwise from results/derived/
$PY experiments/skb_motivation.py --root results/test_route --out outputs/figs/fig_motivation.pdf \
  --table outputs/tables/tab_motivation.tex
$PY experiments/fig_skb_weights.py --root results/test_route --out outputs/figs/fig6_skb_weights.pdf
