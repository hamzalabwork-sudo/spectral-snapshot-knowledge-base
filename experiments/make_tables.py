from __future__ import annotations

"""
Write the manuscript's benchmark tables as LaTeX fragments, straight from result files.

Cells without enough seeds are rendered as \\tbd (a red marker defined in main.tex) so
nothing unfinished can be mistaken for a result. Re-run after every batch of experiments.

Tables written:
    tab_within.tex     within-dataset benchmark, all methods grouped by family
    tab_cross.tex      cross-domain benchmark, all methods grouped by family
    tab_paired.tex     IFR vs ER, paired per benchmark
    tab_ablation.tex   IFR variants (one design choice changed at a time), all benchmarks
    tab_efficiency.tex time per step, boundary overhead, peak memory, stored state, parameters
                       and inference latency, from dedicated profiling runs (results/profile/*)

Usage:
    python experiments/make_tables.py --out outputs/tables
"""

import argparse
import glob
import os
import sys

import numpy as np
from scipy.stats import rankdata, ttest_rel, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_benchmark import load

# (family label, [(method key, display name), ...]) in table order.
FAMILIES = [
    ("Lower bound", [("finetune", "Finetune")]),
    ("Regularisation", [("ewc", "EWC"), ("si", "SI"), ("mas", "MAS")]),
    ("Gradient projection", [("agem", "A-GEM")]),
    ("Replay", [("ifr_uniform", "ER"), ("mir", "MIR")]),
    ("Replay + distillation", [("derpp", "DER++"), ("clser", "CLS-ER"), ("adaptivecl", "AdaptiveCL")]),
    ("Fast/slow adaptation", [("fsnet", "FSNet")]),
    ("Frequency-domain", [("fredf", "ER + FreDF loss"), ("clfd", "ER + CLFD encoder"),
                          ("clfd_ifr", "CLFD encoder + IFR"),
                          ("spectracl", "SpectraCL-v1 (overlap)"), ("ifr", r"\textbf{IFR (ours)}")]),
]
METHODS = [m for _, group in FAMILIES for m in group]

ABLATIONS = [
    ("ifr", r"\textbf{IFR} (intrusion weights)", "full method"),
    ("ifr_uniform", r"uniform weights ($\kappa=0$, ER)", "is frequency weighting needed?"),
    ("ifr_shuffled", "shuffled intrusion weights", "do the selected bins matter?"),
    ("ifr_pooled", "pooled over stored regimes", "is per-regime knowledge needed?"),
    ("ifr_neglect", "neglect weights", "alternative mechanism"),
    ("ifr_overlap", r"overlap weights ($\Delta_S$)", "prior interference score"),
    ("ifr_old_energy", "old-energy weights", "design implied by overlap view"),
]

WITHIN = [("electricity", "Electricity"), ("traffic", "Traffic"), ("ettm1", "ETTm1"),
          ("ettm2", "ETTm2"), ("weather", "Weather")]
CROSS = [("order_A", "Order A"), ("order_B", "Order B"), ("order_C", "Order C")]


def merged(patterns: list[str]) -> dict:
    data = {}
    for pat in patterns:
        for ds, methods in load(glob.glob(pat)).items():
            for m, seeds in methods.items():
                data.setdefault(ds, {}).setdefault(m, {}).update(seeds)
    return data


def cell(runs: dict | None, key: str, best: bool, min_seeds: int, digits: int = 3) -> str:
    if not runs or len(runs) < min_seeds:
        return r"\tbd"
    v = [r[key] for r in runs.values()]
    s = f"{np.mean(v):.{digits}f}" + r"{\scriptsize$\pm$" + f"{np.std(v):.{digits}f}" + "}"
    return r"\textbf{" + s + "}" if best else s


def main_table(data: dict, groups: list, min_seeds: int, caption: str, label: str,
               metrics: tuple = (("avg_mse", "MSE"), ("avg_forgetting", "Forget."))) -> str:
    """Methods (grouped by family) x benchmarks, one column per benchmark and metric."""
    nm = len(metrics)
    cols = "ll" + "c" * (nm * len(groups))
    if nm > 1:
        head = " & ".join(rf"\multicolumn{{{nm}}}{{c}}{{{name}}}" for _, name in groups)
        rules = "".join(rf"\cmidrule(lr){{{3 + nm * i}-{2 + nm * (i + 1)}}}" for i in range(len(groups)))
        header = [rf" & & {head} \\", rules,
                  " & ".join(["Family", "Method"] + [lab for _, lab in metrics] * len(groups)) + r" \\"]
    else:
        header = [" & ".join(["Family", "Method"] + [name for _, name in groups]) + r" \\"]
    lines = [r"\begin{table*}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\resizebox{\textwidth}{!}{%",
             rf"\begin{{tabular}}{{{cols}}}", r"\toprule", *header]
    best = {}
    for ds, _ in groups:
        for key, _ in metrics:
            cand = {m: np.mean([r[key] for r in data.get(ds, {}).get(m, {}).values()])
                    for m, _ in METHODS if len(data.get(ds, {}).get(m, {})) >= min_seeds}
            best[(ds, key)] = min(cand, key=cand.get) if cand else None
    for family, members in FAMILIES:
        lines.append(r"\midrule")
        for k, (m, name) in enumerate(members):
            row = [family if k == 0 else "", name]
            for ds, _ in groups:
                runs = data.get(ds, {}).get(m)
                for key, _ in metrics:
                    row.append(cell(runs, key, best[(ds, key)] == m, min_seeds))
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def paired_table(data: dict, groups: list, min_seeds: int, caption: str, label: str) -> str:
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\small", r"\begin{tabular}{lccc}", r"\toprule",
             r"Benchmark & $\Delta$MSE vs.\ ER & seeds better & $p$ (paired $t$) \\", r"\midrule"]
    for ds, name in groups:
        a, b = data.get(ds, {}).get("ifr", {}), data.get(ds, {}).get("ifr_uniform", {})
        seeds = sorted(set(a) & set(b))
        if len(seeds) < min_seeds:
            lines.append(rf"{name} & \tbd & & \\")
            continue
        x = np.array([a[s]["avg_mse"] for s in seeds])
        y = np.array([b[s]["avg_mse"] for s in seeds])
        d = (x - y) / y
        lines.append(rf"{name} & {100 * d.mean():+.1f}\% & {int((x < y).sum())}/{len(seeds)} & "
                     rf"{ttest_rel(x, y).pvalue:.2f} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def ablation_table(data: dict, groups: list) -> str:
    """MSE of each variant per benchmark on the seeds shared by all variants, average rank,
    and a pooled paired test against full IFR."""
    shared = {}
    for ds, _ in groups:
        sets = [set(data.get(ds, {}).get(m, {})) for m, _, _ in ABLATIONS]
        shared[ds] = sorted(set.intersection(*sets)) if all(sets) else []
    n_within = sorted({len(shared[ds]) for ds, _ in groups if not ds.startswith("order_")})
    n_cross = sorted({len(shared[ds]) for ds, _ in groups if ds.startswith("order_")})
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Ablation of IFR. Each variant changes one design choice and keeps the buffer, "
             r"code path, $\kappa=4$ and compute fixed. Average test MSE per benchmark over the seeds "
             rf"shared by all variants ({'/'.join(map(str, n_within))} within-dataset, "
             rf"{'/'.join(map(str, n_cross))} cross-domain), average rank across benchmarks (1 = best), "
             r"and the pooled relative change in MSE versus full IFR over all paired runs, with a "
             r"two-sided Wilcoxon test. Best per column in bold.}", r"\label{tab:ablation}", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{l" + "c" * len(groups) + "ccc}", r"\toprule",
             "Variant & " + " & ".join(n for _, n in groups)
             + r" & Avg.\ rank & $\Delta$ vs.\ IFR & $p$ \\", r"\midrule"]

    def mean_mse(m, ds):
        runs = data.get(ds, {}).get(m, {})
        return np.mean([runs[s]["avg_mse"] for s in shared[ds]]) if len(shared[ds]) >= 5 else np.nan

    table = np.array([[mean_mse(m, ds) for ds, _ in groups] for m, _, _ in ABLATIONS])
    complete = ~np.isnan(table).any(axis=0)
    ranks = np.full(len(ABLATIONS), np.nan)
    if complete.any():
        ranks = np.mean([rankdata(table[:, j]) for j in np.where(complete)[0]], axis=0)
    best_col = np.nanargmin(np.where(np.isnan(table), np.inf, table), axis=0)

    for i, (m, name, _question) in enumerate(ABLATIONS):
        row = [name]
        for j, _ in enumerate(groups):
            v = table[i, j]
            if np.isnan(v):
                row.append(r"\tbd")
            else:
                s = f"{v:.3f}"
                row.append(r"\textbf{" + s + "}" if best_col[j] == i and complete[j] else s)
        row.append(f"{ranks[i]:.1f}" if not np.isnan(ranks[i]) else r"\tbd")
        if m == "ifr":
            row += ["--", "--"]
        else:
            rel = []
            for ds, _ in groups:
                a, b = data.get(ds, {}).get(m, {}), data.get(ds, {}).get("ifr", {})
                rel += [(a[s]["avg_mse"] - b[s]["avg_mse"]) / b[s]["avg_mse"] for s in shared[ds]]
            if len(rel) >= 10:
                row += [f"{100 * np.mean(rel):+.1f}\\%", f"{wilcoxon(rel).pvalue:.3f}"]
            else:
                row += [r"\tbd", ""]
        lines.append(" & ".join(row) + r" \\")
        if m == "ifr":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def overall_table(data: dict, groups: list, profile_dir: str = "results/profile/traffic") -> str:
    """Average rank of every method over all benchmarks (MSE and forgetting), with training
    cost relative to ER from the isolated profiling runs."""
    import json

    keys = [m for m, _ in METHODS]
    ranks = {}
    for metric in ("avg_mse", "avg_forgetting"):
        per_bench = []
        for ds, _ in groups:
            vals = [np.mean([r[metric] for r in data.get(ds, {}).get(m, {}).values()])
                    if data.get(ds, {}).get(m) else np.nan for m in keys]
            if not np.isnan(vals).any():
                per_bench.append(rankdata(vals))
        ranks[metric] = np.mean(per_bench, axis=0) if per_bench else np.full(len(keys), np.nan)

    prof = {}
    for p in glob.glob(os.path.join(profile_dir, "continual_results_*.json")):
        for e in json.load(open(p)):
            if "profile" in e:
                t = e["profile"]["tasks"][1:]
                prof[e["method"]] = sum(x["train_s"] for x in t) / sum(x["steps"] for x in t)

    order = np.argsort(ranks["avg_mse"])
    best_mse, best_fgt = np.nanmin(ranks["avg_mse"]), np.nanmin(ranks["avg_forgetting"])
    lines = [r"\begin{table}[t]", r"\centering",
             rf"\caption{{Overall comparison: average rank of each method over the {len(groups)} benchmarks "
             r"(1 = best of " + str(len(keys)) + r"), for test MSE and for forgetting, and training time per "
             r"step relative to ER (Traffic, isolated runs; Table~\ref{tab:efficiency}).}",
             r"\label{tab:overall}", r"\small", r"\begin{tabular}{lccc}", r"\toprule",
             r"Method & MSE rank & Forgetting rank & Time vs.\ ER \\", r"\midrule"]
    names = dict(METHODS)
    for i in order:
        m = keys[i]
        mse = f"{ranks['avg_mse'][i]:.2f}"
        fgt = f"{ranks['avg_forgetting'][i]:.2f}"
        mse = r"\textbf{" + mse + "}" if ranks["avg_mse"][i] == best_mse else mse
        fgt = r"\textbf{" + fgt + "}" if ranks["avg_forgetting"][i] == best_fgt else fgt
        t = rf"{prof[m] / prof['ifr_uniform']:.2f}$\times$" if m in prof and "ifr_uniform" in prof else r"\tbd"
        lines.append(rf"{names[m]} & {mse} & {fgt} & {t} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def efficiency_table(profile_dirs: list[tuple[str, str]]) -> str:
    """Computational efficiency from dedicated profiling runs (one method at a time, idle GPU).

    Per benchmark: training time per optimiser step while replay is active (regimes 2-5),
    relative to ER; end-of-regime overhead; peak GPU memory. Shared: stored state besides
    the model (largest across benchmarks), parameters, inference latency per batch.
    """
    import json

    prof = {}
    for bench, d in profile_dirs:
        for p in glob.glob(os.path.join(d, "continual_results_*.json")):
            for e in json.load(open(p)):
                if "profile" in e:
                    prof.setdefault(bench, {})[e["method"]] = e["profile"]
    # The profiling runs count trainable parameters only; use total counts (frozen ones included,
    # e.g. CLFD's encoder) from experiments/count_params.py when available.
    total_path = "results/profile/params_total.json"
    if os.path.exists(total_path):
        for bench, counts in json.load(open(total_path)).items():
            for m, n in counts.items():
                if m in prof.get(bench, {}):
                    prof[bench][m]["params"] = max(prof[bench][m]["params"], n)

    def step_ms(pr):
        t = pr["tasks"][1:]
        return 1000 * sum(x["train_s"] for x in t) / sum(x["steps"] for x in t)

    benches = [b for b, _ in profile_dirs]
    names = {"order_A": "cross-domain", "traffic": "Traffic (862 var.)"}
    lines = [r"\begin{sidewaystable}", r"\centering",
             r"\caption{Computational efficiency, measured in dedicated runs with one method at a time on an "
             r"idle NVIDIA RTX 3090 (seed 0, batch size 32, three epochs per regime). \emph{ms/step}: training "
             r"time per optimiser step while replay is active (regimes 2--5); \emph{rel.}: relative to ER; "
             r"\emph{boundary}: mean time per regime spent on end-of-regime updates (e.g.\ Fisher information, "
             r"buffer filling, model snapshots); \emph{peak}: peak GPU memory during training; \emph{stored}: "
             r"state kept besides the model, reported for Traffic, where a 500-window buffer of "
             r"$96\times862$ inputs dominates (IFR adds 100 bytes per regime to it); \emph{infer.}: "
             r"forward time per test batch. ER, IFR and the FreDF and CLFD variants keep their buffer on "
             r"the GPU for speed, which raises their peak memory relative to methods that sample it "
             r"from host memory.}",
             r"\label{tab:efficiency}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\resizebox{\textheight}{!}{%",   # rotated page: available width is the text height
             r"\begin{tabular}{l" + "rrrr" * len(benches) + "rrr}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{names.get(b, b)}}}" for b in benches)
             + r" & \multicolumn{3}{c}{model and memory} \\",
             "".join(rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(benches)))
             + rf"\cmidrule(lr){{{2 + 4 * len(benches)}-{4 + 4 * len(benches)}}}",
             "Method & " + " & ".join(["ms/step & rel. & boundary (s) & peak (MB)"] * len(benches))
             + r" & params & stored (MB) & infer.\ (ms) \\", r"\midrule"]
    for family, members in FAMILIES:
        for m, name in members:
            row = [name]
            for b in benches:
                pr, er = prof.get(b, {}).get(m), prof.get(b, {}).get("ifr_uniform")
                if pr is None or er is None:
                    row += [r"\tbd"] * 4
                    continue
                boundary = np.mean([x["boundary_s"] for x in pr["tasks"]])
                peak = max(x["peak_mem_mb"] for x in pr["tasks"])
                row += [f"{step_ms(pr):.1f}", rf"{step_ms(pr) / step_ms(er):.2f}$\times$",
                        f"{boundary:.1f}", f"{peak:.0f}"]
            prs = [prof[b][m] for b in benches if m in prof.get(b, {})]
            if prs:
                traffic = prof.get("traffic", {}).get(m)
                row += [f"{max(p['params'] for p in prs) / 1e3:.0f}K",
                        f"{traffic['stored_bytes'] / 2**20:.2f}" if traffic else r"\tbd",
                        f"{np.mean([p['infer_ms_per_batch'] for p in prs]):.2f}"]
            else:
                row += [r"\tbd"] * 3
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{sidewaystable}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/tables")
    ap.add_argument("--min_seeds", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    within = merged(["results/pilot/*/continual_results_*.json", "results/main/*/continual_results_*.json"])
    cross = merged(["results/crossdomain/*/continual_results_*.json"])
    both = {**within, **cross}
    files = {
        "tab_within.tex": main_table(
            within, WITHIN, args.min_seeds,
            r"Within-dataset benchmark: average test MSE over all regimes after the last one "
            r"(mean $\pm$ std over five seeds; $L=96$, $H=48$), grouped by method family. "
            r"Best per column in bold.", "tab:within", metrics=(("avg_mse", "MSE"),)),
        "tab_within_fgt.tex": main_table(
            within, WITHIN, args.min_seeds,
            r"Within-dataset benchmark: average forgetting after the last regime (mean $\pm$ std over "
            r"five seeds). Best per column in bold.", "tab:within_fgt",
            metrics=(("avg_forgetting", "Forgetting"),)),
        "tab_cross.tex": main_table(
            cross, CROSS, 10,
            r"Cross-domain benchmark: five univariate sources learned in sequence, three task orders, "
            r"10 seeds, grouped by method family. Best per column in bold.", "tab:cross"),
        "tab_paired.tex": paired_table(
            both, WITHIN + CROSS, args.min_seeds,
            r"IFR versus standard replay (ER, same code path with $\kappa=0$), paired over seeds.",
            "tab:paired"),
        "tab_ablation.tex": ablation_table(both, WITHIN + CROSS),
        "tab_overall.tex": overall_table(both, WITHIN + CROSS),
        "tab_efficiency.tex": efficiency_table([("traffic", "results/profile/traffic"),
                                                ("order_A", "results/profile/order_A")]),
    }
    for name, text in files.items():
        with open(os.path.join(args.out, name), "w") as f:
            f.write(text)
        print("wrote", name, "| unfinished cells:", text.count(r"\tbd"))


if __name__ == "__main__":
    main()
