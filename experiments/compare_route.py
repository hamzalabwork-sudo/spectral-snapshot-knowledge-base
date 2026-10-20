from __future__ import annotations

"""
Validation-split comparison of case-based snapshot routing (results/val_route/<bench>/)
against the official validation runs (results/val/<bench>/), paired by seed.

Every routed run stores all routing variants on the same snapshots; variant v of base method
b is reported as "b+v" ("b+final" is b trained without the held-out case segments).
"known" and "oracle" use evaluation labels and are diagnostics, not candidates.

Usage:
    python experiments/compare_route.py --refs clfd clfd_ifr
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_benchmark import cl_metrics, load

DIAGNOSTIC = ("known", "oracle")


def load_routed(root: str) -> tuple[dict, dict]:
    data = defaultdict(lambda: defaultdict(dict))
    diag = defaultdict(lambda: defaultdict(list))
    for p in glob.glob(os.path.join(root, "*", "continual_results_*.json")):
        ds = os.path.basename(os.path.dirname(p))
        for e in json.load(open(p)):
            if "routing" not in e:
                continue
            base = e["method"].split("@")[0]
            for v, R in e["routing"].items():
                data[ds][f"{base}+{v}"][e["seed"]] = cl_metrics(R)
            diag[ds][base].append(e["routing_diag"][-1])
    return data, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_root", default="results/val")
    ap.add_argument("--official_glob", nargs="+", default=None,
                    help="result-file patterns of the official runs, merged in order (default: val_root)")
    ap.add_argument("--route_root", default="results/val_route")
    ap.add_argument("--refs", nargs="+", default=["clfd", "clfd_ifr"])
    ap.add_argument("--official", nargs="+", default=["ifr_uniform", "ifr", "clfd", "clfd_ifr"])
    ap.add_argument("--metric", default="avg_mse")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="JSON from offline_rules.py: {bench: {method: {seed: metrics}}}")
    ap.add_argument("--only", nargs="*", default=None, help="restrict the comparison to these methods")
    args = ap.parse_args()

    official = {}
    for pat in args.official_glob or [os.path.join(args.val_root, "*", "continual_results_*.json")]:
        for ds, ms in load(glob.glob(pat)).items():
            for m, seeds in ms.items():
                official.setdefault(ds, {}).setdefault(m, {}).update(seeds)
    routed, diag = load_routed(args.route_root)
    for path in args.extra:
        for b, ms in json.load(open(path)).items():
            for m, seeds in ms.items():
                routed[b][m] = {int(k): v for k, v in seeds.items()}
    benches = sorted(b for b in routed if b in official)
    data = {b: {**{m: official[b][m] for m in args.official if m in official[b]}, **routed[b]}
            for b in benches}
    methods = sorted({m for b in benches for m in data[b]})
    complete = [m for m in methods if all(m in data[b] for b in benches)]
    if args.only:
        complete = [m for m in complete if m in args.only or m in args.refs]
    k = args.metric

    split = "Test" if args.official_glob else "Validation"
    absolute = k in ("avg_forgetting", "bwt")   # can be zero or change sign: compare differences
    print(f"{split} {k} (mean over seeds); benchmarks: {', '.join(benches)}")
    print(f"{'method':24}" + "".join(f"{b[:9]:>10}" for b in benches) + f"{'avg rank':>10}")
    table = np.array([[np.mean([r[k] for r in data[b][m].values()]) for b in benches] for m in complete])
    cand = [i for i, m in enumerate(complete) if not m.endswith(DIAGNOSTIC)]
    ranks = np.zeros(len(complete))
    ranks[cand] = np.mean([rankdata(table[cand, j]) for j in range(len(benches))], axis=0)
    for i in sorted(range(len(complete)), key=lambda i: (complete[i].endswith(DIAGNOSTIC), ranks[i])):
        tag = f"{ranks[i]:10.2f}" if i in cand else f"{'(diag)':>10}"
        print(f"{complete[i]:24}" + "".join(f"{v:10.4f}" for v in table[i]) + tag)

    for ref in args.refs:
        unit = "absolute difference" if absolute else "relative difference"
        print(f"\nvs official {ref} ({unit}; negative = lower {k}; paired by seed, pooled):")
        rows = []
        for m in complete:
            if m == ref:
                continue
            rel, won = [], 0
            for b in benches:
                seeds = sorted(set(data[b][m]) & set(data[b][ref]))
                a = np.array([data[b][m][s][k] for s in seeds])
                c = np.array([data[b][ref][s][k] for s in seeds])
                rel += list(a - c if absolute else (a - c) / np.abs(c))
                won += int(a.mean() < c.mean())
            p = wilcoxon(rel).pvalue if len(rel) >= 6 and np.any(rel) else float("nan")
            rows.append((np.mean(rel), m, won, np.median(rel), len(rel), p))
        for mean, m, won, med, n, p in sorted(rows):
            f = (lambda v: f"{v:+.4f}") if absolute else (lambda v: f"{100 * v:+6.2f}%")
            print(f"  {m:24} lower on {won}/{len(benches)} | mean {f(mean)} "
                  f"| median {f(med)} | n={n} | Wilcoxon p={p:.3g}")

    print("\nRouting diagnostics at the final stage (mean over seeds):")
    for b in benches:
        for base, ds in sorted(diag[b].items()):
            sel = [" ".join(map(str, d["selected"])) for d in ds]
            print(f"  {b:12} {base:9} regime acc knn={np.mean([d['regime_acc_knn'] for d in ds]):.2f} "
                  f"dlda={np.mean([d['regime_acc_dlda'] for d in ds]):.2f} "
                  f"to-final={np.mean([d['frac_final_knn'] for d in ds]):.2f} | selected: {' / '.join(sel)}")


if __name__ == "__main__":
    main()
