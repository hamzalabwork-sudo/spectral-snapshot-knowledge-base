from __future__ import annotations

"""
Tables for the additional SKB experiments (Paper B):

  tab_casesens.tex  size of the case segment (validation split, CLFD learner): SKB and the last
                    snapshot alone, relative to official CLFD, for case ratios 5%, 10% and 20%
  tab_general.tex   generalisation (test split): horizon H=96 (CLFD learner) and a linear backbone
                    (ER learner), SKB relative to the official run of the same learner

Usage:
    python experiments/make_tables_skb_extra.py --out outputs/tables
"""

import argparse
import glob
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_route import load_routed  # noqa: E402
from summarize_benchmark import load  # noqa: E402

SKB = "sstack_global"
WITHIN = ["electricity", "traffic", "ettm1", "ettm2", "weather"]
CROSS = ["order_A", "order_B", "order_C"]


def official(root: str) -> dict:
    return load(glob.glob(os.path.join(root, "*", "continual_results_*.json")))


def paired(a: dict, b: dict, benches, key="avg_mse", relative=True):
    """Pooled paired differences of runs a vs b ({bench: {seed: metrics}}), and benchmarks won."""
    d, won, n_b = [], 0, 0
    for bench in benches:
        x, y = a.get(bench, {}), b.get(bench, {})
        seeds = sorted(set(x) & set(y))
        if not seeds:
            continue
        u = np.array([x[s][key] for s in seeds])
        v = np.array([y[s][key] for s in seeds])
        d += list((u - v) / np.abs(v) if relative else u - v)
        won += int(u.mean() < v.mean())
        n_b += 1
    return np.array(d), won, n_b


def fmt_p(d):
    if len(d) < 6:
        return r"\tbd"
    p = wilcoxon(d).pvalue
    return "$<$0.001" if p < 1e-3 else f"{p:.3f}"


def pct(d):
    return f"{100 * d.mean():+.1f}\\%" if len(d) else r"\tbd"


def case_table(val_root, routes) -> str:
    ref = official(val_root)
    ref = {b: ref[b]["clfd"] for b in ref if "clfd" in ref[b]}
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Size of the case segment (validation split, CLFD learner, three seeds, eight benchmarks): "
             r"pooled relative change in MSE against CLFD trained on all of the data, for SKB and for the last "
             r"snapshot alone, which measures the cost of holding the case segment out of training. The test "
             r"results use 10\%, which was fixed before these runs.}",
             r"\label{tab:casesens}", r"\small", r"\begin{tabular}{lcccc}", r"\toprule",
             r"Case segment & SKB: $\Delta$MSE & improved & $p$ & last snapshot: $\Delta$MSE \\", r"\midrule"]
    for label, root in routes:
        routed, _ = load_routed(root)
        skb = {b: routed[b].get(f"clfd+{SKB}", {}) for b in routed}
        fin = {b: routed[b].get("clfd+final", {}) for b in routed}
        d, won, nb = paired(skb, ref, WITHIN + CROSS)
        df, _, _ = paired(fin, ref, WITHIN + CROSS)
        lines.append(rf"{label} & {pct(d)} & {won}/{nb} & {fmt_p(d)} & {pct(df)} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def general_table(settings) -> str:
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Generalisation (test split): SKB relative to the official run of the same learner, "
             r"trained on all of its data, for a longer horizon ($H=96$, CLFD learner, five seeds) and a linear "
             r"backbone ($H=48$, ER learner, five seeds within-dataset and ten cross-domain). $\Delta$: pooled "
             r"relative change in MSE; forgetting: absolute change; Wilcoxon test over all paired runs.}",
             r"\label{tab:general}", r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lccccc}", r"\toprule",
             r"Setting & $\Delta$ within & $\Delta$ cross & improved & $p$ & $\Delta$ forgetting \\",
             r"\midrule"]
    for label, ref_root, ref_method, route_root, base in settings:
        ref = official(ref_root)
        ref = {b: ref[b][ref_method] for b in ref if ref_method in ref[b]}
        routed, _ = load_routed(route_root)
        skb = {b: routed[b].get(f"{base}+{SKB}", {}) for b in routed}
        dw, ww, nw = paired(skb, ref, WITHIN)
        dc, wc, nc = paired(skb, ref, CROSS)
        d = np.concatenate([dw, dc])
        df, _, _ = paired(skb, ref, WITHIN + CROSS, key="avg_forgetting", relative=False)
        lines.append(rf"{label} & {pct(dw)} & {pct(dc)} & {ww + wc}/{nw + nc} & {fmt_p(d)} & "
                     + (f"{df.mean():+.4f}" if len(df) else r"\tbd") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


SEARCH = [  # (rule, description) in the order of Appendix B
    ("final", "final snapshot (reference)"),
    ("uniform", "uniform averaging"), ("soup", "parameter averaging"),
    ("causal_buf", "causal averaging, nearest-window recognition"),
    ("causal_buf_dlda", "causal averaging, discriminant on training windows"),
    ("causal_case_dlda", "causal averaging, discriminant on cases"),
    ("route_knn", "routing, nearest-case recognition"), ("route_knn_soft", "routing, soft vote"),
    ("route_dlda", "routing, discriminant recognition"), ("knn_err", "per-window selection"),
    ("fuse_knn", "per-bin selection, nearest-case recognition"),
    ("fuse_dlda", "per-bin selection, discriminant recognition"), ("fuse_global", "per-bin selection, pooled"),
    ("stack_knn", "time-domain stacking, nearest-case recognition"),
    ("stack_dlda", "time-domain stacking, discriminant recognition"), ("stack_global", "time-domain stacking, pooled"),
    ("sstack_knn", "per-bin stacking, nearest-case recognition"),
    ("sstack_dlda", "per-bin stacking, discriminant recognition"),
    (SKB, r"\textbf{per-bin stacking, pooled (SKB)}"),
]


def search_table(val_root="results/val", route_root="results/val_route",
                 extra=os.path.join("results", "val_route", "offline_rules.json")) -> str:
    """Validation MSE of every rule on every learner relative to CLFD: the full design search."""
    import json
    ref = official(val_root)
    ref = {b: ref[b]["clfd"] for b in ref if "clfd" in ref[b]}
    routed, _ = load_routed(route_root)
    if os.path.exists(extra):
        for b, ms in json.load(open(extra)).items():
            for m, seeds in ms.items():
                routed[b][m] = {int(k): v for k, v in seeds.items()}
    learners = [("ifr", "IFR"), ("clfd", "CLFD"), ("clfd_ifr", "CLFD + IFR")]
    lines = [r"\begin{table}[h]", r"\centering",
             r"\caption{The design search on the validation split: pooled relative change in validation MSE "
             r"against CLFD (trained on all of the data; three seeds, eight benchmarks) of every rule on every "
             r"learner, 54 configurations besides the final snapshots. The configuration with the lowest value "
             r"(bold) was fixed as SKB before the test split was used.}",
             r"\label{tab:search}", r"\footnotesize", r"\begin{tabular}{lccc}", r"\toprule",
             "Rule & " + " & ".join(n for _, n in learners) + r" \\", r"\midrule"]
    best = None
    vals = {}
    for r, _ in SEARCH:
        for b, _ in learners:
            d, _, _ = paired({k: routed[k].get(f"{b}+{r}", {}) for k in routed}, ref, WITHIN + CROSS)
            vals[(r, b)] = d.mean() if len(d) else np.nan
            if r != "final" and len(d) and (best is None or d.mean() < vals[best]):
                best = (r, b)
    for r, desc in SEARCH:
        cells = []
        for b, _ in learners:
            v = vals[(r, b)]
            s = r"\tbd" if np.isnan(v) else f"{100 * v:+.1f}\\%"
            cells.append(r"\textbf{" + s + "}" if (r, b) == best else s)
        lines.append(f"{desc} & " + " & ".join(cells) + r" \\")
        if r == "final":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/tables")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    files = {
        "tab_search.tex": search_table(),
        "tab_casesens.tex": case_table("results/val", [("5\\%", "results/val_route_r05"),
                                                       ("10\\% (used)", "results/val_route"),
                                                       ("20\\%", "results/val_route_r20")]),
        "tab_general.tex": general_table([
            ("$H=96$, CLFD", "results/test_h96", "clfd", "results/test_h96_route", "clfd"),
            ("linear backbone, ER", "results/test_linear", "ifr_uniform", "results/test_linear_route", "ifr_uniform"),
        ]),
    }
    for name, text in files.items():
        with open(os.path.join(args.out, name), "w") as f:
            f.write(text)
        print("wrote", name, "| unfinished cells:", text.count(r"\tbd"))
        print(text)


if __name__ == "__main__":
    main()
