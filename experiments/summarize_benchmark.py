from __future__ import annotations

"""
Aggregate multi-seed run_continual.py outputs into paper-ready tables.

For every method: mean +- std over seeds of Avg MSE, Avg MAE, Forgetting and
BWT (all recomputed from the stored results matrices), plus paired tests
(over shared seeds) of a reference method against every other method.

Usage:
    python experiments/summarize_benchmark.py results/pilot/traffic/*.json --ref ifr
    python experiments/summarize_benchmark.py results/main/*/*.json --ref ifr --by-dataset
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

METRICS = ["avg_mse", "avg_mae", "avg_forgetting", "bwt"]


def cl_metrics(R: list[list[dict]]) -> dict:
    n = len(R)
    final = R[-1]
    forget = [final[j]["mse"] - min(R[i][j]["mse"] for i in range(j, n)) for j in range(n - 1)]
    bwt = [R[j][j]["mse"] - final[j]["mse"] for j in range(n - 1)]
    return {
        "avg_mse": float(np.mean([r["mse"] for r in final])),
        "avg_mae": float(np.mean([r["mae"] for r in final])),
        "avg_forgetting": float(np.mean(forget)),
        "bwt": float(np.mean(bwt)),
    }


def load(paths: list[str]) -> dict:
    """{dataset: {method: {seed: metrics}}}; dataset = parent directory name."""
    data = defaultdict(lambda: defaultdict(dict))
    for p in paths:
        ds = os.path.basename(os.path.dirname(os.path.abspath(p)))
        for e in json.load(open(p)):
            if e["method"] == "joint" or len(e["results_matrix"]) < 2:
                continue
            m = cl_metrics(e["results_matrix"])
            m["time"] = e.get("time", float("nan"))
            data[ds][e["method"]][e.get("seed", 0)] = m
    return data


def report(name: str, methods: dict, ref: str) -> None:
    print(f"\n=== {name} ===")
    print(f"{'method':14} {'n':>2}  " + "  ".join(f"{k:>18}" for k in METRICS) + f"  {'min/run':>8}")
    order = sorted(methods, key=lambda m: np.mean([v["avg_mse"] for v in methods[m].values()]))
    for meth in order:
        runs = methods[meth]
        cells = []
        for k in METRICS:
            v = [r[k] for r in runs.values()]
            cells.append(f"{np.mean(v):.4f}+-{np.std(v):.4f}")
        t = np.mean([r["time"] for r in runs.values()]) / 60
        print(f"{meth:14} {len(runs):2d}  " + "  ".join(f"{c:>18}" for c in cells) + f"  {t:8.1f}")

    if ref not in methods:
        return
    print(f"\nPaired tests: {ref} vs others (negative delta = {ref} lower)")
    for meth in order:
        if meth == ref:
            continue
        seeds = sorted(set(methods[ref]) & set(methods[meth]))
        if len(seeds) < 2:
            continue
        parts = []
        for k in ("avg_mse", "avg_forgetting"):
            a = np.array([methods[ref][s][k] for s in seeds])
            b = np.array([methods[meth][s][k] for s in seeds])
            d = a - b
            tp = ttest_rel(a, b).pvalue if np.any(d) else float("nan")
            wp = wilcoxon(d).pvalue if len(seeds) >= 5 and np.any(d) else float("nan")
            parts.append(f"{k}: delta={d.mean():+.4f} ({100 * d.mean() / b.mean():+.1f}%) "
                         f"wins {int((d < 0).sum())}/{len(d)} t-p={tp:.3g} W-p={wp:.3g}")
        print(f"  vs {meth:12} " + " | ".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--ref", default="ifr")
    args = ap.parse_args()
    paths = [p for pat in args.paths for p in glob.glob(pat)]
    for ds, methods in sorted(load(paths).items()):
        report(ds, methods, args.ref)


if __name__ == "__main__":
    main()
