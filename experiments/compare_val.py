from __future__ import annotations

"""
Model selection on the VALIDATION split only (results/val/<benchmark>/).

Prints per-benchmark validation MSE, average rank over benchmarks, and paired comparisons
(over shared seeds, pooled over benchmarks) of every method against the references.
The method to be evaluated on the test split is chosen from this output alone.

Usage:
    python experiments/compare_val.py --refs clfd ifr
"""

import argparse
import glob
import os
import sys

import numpy as np
from scipy.stats import rankdata, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_benchmark import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/val")
    ap.add_argument("--refs", nargs="+", default=["clfd", "ifr"])
    args = ap.parse_args()

    data = {}
    for ds, methods in load(glob.glob(os.path.join(args.root, "*", "continual_results_*.json"))).items():
        data[ds] = methods
    benches = sorted(data)
    methods = sorted({m for d in data.values() for m in d})
    complete = [m for m in methods if all(m in data[b] for b in benches)]

    print(f"{'method':14}" + "".join(f"{b[:9]:>10}" for b in benches) + f"{'avg rank':>10}")
    table = np.array([[np.mean([r["avg_mse"] for r in data[b][m].values()]) for b in benches] for m in complete])
    ranks = np.mean([rankdata(table[:, j]) for j in range(len(benches))], axis=0)
    for i in np.argsort(ranks):
        print(f"{complete[i]:14}" + "".join(f"{v:10.4f}" for v in table[i]) + f"{ranks[i]:10.2f}")

    for ref in args.refs:
        print(f"\nvs {ref} (negative = method lower validation MSE):")
        for m in complete:
            if m == ref:
                continue
            rel, won = [], 0
            for b in benches:
                seeds = sorted(set(data[b][m]) & set(data[b][ref]))
                a = np.array([data[b][m][s]["avg_mse"] for s in seeds])
                c = np.array([data[b][ref][s]["avg_mse"] for s in seeds])
                rel += list((a - c) / c)
                won += int(a.mean() < c.mean())
            p = wilcoxon(rel).pvalue if len(rel) >= 6 and np.any(rel) else float("nan")
            print(f"  {m:14} better on {won}/{len(benches)} benchmarks | mean {100 * np.mean(rel):+6.2f}% "
                  f"| median {100 * np.median(rel):+6.2f}% | n={len(rel)} | Wilcoxon p={p:.3g}")


if __name__ == "__main__":
    main()
