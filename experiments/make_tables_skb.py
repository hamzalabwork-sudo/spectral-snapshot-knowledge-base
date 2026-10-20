from __future__ import annotations

"""
Tables for the manuscript with the spectral snapshot knowledge base (SKB), from result files.

Official test-split runs of all baselines (results/pilot, results/main, results/crossdomain) are
merged with the SKB test runs (results/test_route/<bench>/), in which every snapshot rule was
evaluated on the same snapshots; rule r on base learner b is the method "b+r". The SKB rule
(sstack_global on CLFD) was selected on the validation split before these runs.

Tables written:
    tab_within.tex, tab_within_fgt.tex, tab_cross.tex   all methods incl. SKB rows
    tab_overall.tex   average rank over the eight benchmarks
    tab_skbrules.tex  every snapshot rule on the CLFD learner (ablation of SKB)
    tab_skbbases.tex  SKB on four base learners (plug-in effect)

Usage:
    python experiments/make_tables_skb.py --out outputs/tables
"""

import argparse
import os
import sys

import numpy as np
from scipy.stats import rankdata, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_tables as mt  # noqa: E402
from compare_route import load_routed  # noqa: E402

SKB = "sstack_global"
SKB_FAMILY = ("Snapshot knowledge base", [
    (f"clfd+{SKB}", r"\textbf{CLFD + SKB (ours)}"),
    (f"clfd_ifr+{SKB}", "CLFD + IFR + SKB (ours)"),
    (f"ifr+{SKB}", "IFR + SKB (ours)"),
    (f"ifr_uniform+{SKB}", "ER + SKB (ours)"),
])
BASES = [("ifr_uniform", "ER"), ("ifr", "IFR"), ("clfd", "CLFD"), ("clfd_ifr", "CLFD + IFR")]

# (rule, description, snapshots evaluated per forecast); diagnostics use test labels.
RULES = [
    ("final", r"last snapshot only (90\% of the data)", "1"),
    ("uniform", "uniform average", "$t$"),
    ("soup", "parameter average (soup)", "1"),
    ("route_dlda", "routing, discriminant recognition", "1"),
    ("route_knn", "routing, nearest-case recognition", "1"),
    ("causal_buf", "average of snapshots that saw the regime", r"$\le t$"),
    ("fuse_global", "per-bin selection, pooled", r"$\le t$"),
    ("stack_knn", "time-domain stacking, per regime", r"$\le t$"),
    ("sstack_knn", "per-bin stacking, per regime", r"$\le t$"),
    ("stack_global", "time-domain stacking, pooled", r"$\le t$"),
    (SKB, r"\textbf{per-bin stacking, pooled (SKB)}", r"$\le t$"),
]
DIAGNOSTICS = [
    ("oracle", "oracle: best snapshot per regime"),
    ("fuse_oracle", "oracle: best snapshot per regime and bin"),
]


def pooled(data, groups, m, ref, key="avg_mse", relative=True):
    """Paired differences of m vs ref over all (benchmark, seed) pairs, and benchmarks won."""
    diffs, won = [], 0
    for ds, _ in groups:
        a, b = data.get(ds, {}).get(m, {}), data.get(ds, {}).get(ref, {})
        seeds = sorted(set(a) & set(b))
        if not seeds:
            continue
        x = np.array([a[s][key] for s in seeds])
        y = np.array([b[s][key] for s in seeds])
        diffs += list((x - y) / np.abs(y) if relative else x - y)
        won += int(x.mean() < y.mean())
    return np.array(diffs), won


def fmt_p(p):
    return "$<$0.001" if p < 1e-3 else f"{p:.3f}"


def skb_rules_table(data, groups, base="clfd"):
    def mean_mse(m, ds):
        runs = data.get(ds, {}).get(m, {})
        return np.mean([r["avg_mse"] for r in runs.values()]) if runs else np.nan

    cand = [f"{base}+{r}" for r, _, _ in RULES]
    table = np.array([[mean_mse(m, ds) for ds, _ in groups] for m in cand])
    ranks = np.mean([rankdata(table[:, j]) for j in range(len(groups))], axis=0)
    best = np.nanargmin(table, axis=0)
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Ablation of SKB: every rule for using the snapshots of the CLFD learner, evaluated on "
             r"the same snapshots (test split; mean over 5 seeds within-dataset and 10 cross-domain). "
             r"$\Delta$: pooled relative change in MSE versus the official CLFD runs (trained on all of the "
             r"data), paired by seed, with a two-sided Wilcoxon test; \emph{fwd}: snapshots evaluated per "
             r"forecast. The rule was selected on the validation split; the oracle rows use test labels and "
             r"are upper bounds, not methods. Best realisable value per column in bold.}",
             r"\label{tab:skbrules}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
             r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{l" + "c" * len(groups) + "cccc}", r"\toprule",
             "Rule & " + " & ".join(n for _, n in groups) + r" & rank & $\Delta$ vs.\ CLFD & $p$ & fwd \\",
             r"\midrule"]
    ref_row = [r"CLFD (official, all data)"]
    for ds, _ in groups:
        ref_row.append(f"{mean_mse(base, ds):.3f}")
    lines.append(" & ".join(ref_row + ["--", "--", "--", "1"]) + r" \\")
    lines.append(r"\midrule")
    for i, (r, desc, fwd) in enumerate(RULES):
        m = f"{base}+{r}"
        row = [desc]
        for j, _ in enumerate(groups):
            s = f"{table[i, j]:.3f}"
            row.append(r"\textbf{" + s + "}" if best[j] == i else s)
        d, _ = pooled(data, groups, m, base)
        row += [f"{ranks[i]:.1f}", f"{100 * d.mean():+.1f}\\%", fmt_p(wilcoxon(d).pvalue), fwd]
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")
    for r, desc in DIAGNOSTICS:
        m = f"{base}+{r}"
        row = [r"\emph{" + desc + "}"] + [f"{mean_mse(m, ds):.3f}" for ds, _ in groups]
        d, _ = pooled(data, groups, m, base)
        row += ["--", f"{100 * d.mean():+.1f}\\%", fmt_p(wilcoxon(d).pvalue), "1"]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def skb_bases_table(data, within, cross):
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{SKB as a plug-in: change in test MSE when SKB is added to four continual learners, "
             r"relative to the official runs of the same learner (paired by seed; within-dataset 5 benchmarks "
             r"$\times$ 5 seeds, cross-domain 3 orders $\times$ 10 seeds), number of benchmarks improved, "
             r"pooled Wilcoxon test, and the change in average forgetting (absolute).}",
             r"\label{tab:skbbases}", r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lccccc}", r"\toprule",
             r"Learner & $\Delta$ within & $\Delta$ cross & improved & $p$ & $\Delta$ forgetting \\",
             r"\midrule"]
    for b, name in BASES:
        m = f"{b}+{SKB}"
        dw, ww = pooled(data, within, m, b)
        dc, wc = pooled(data, cross, m, b)
        d = np.concatenate([dw, dc])
        df, _ = pooled(data, within + cross, m, b, key="avg_forgetting", relative=False)
        if len(d) < 10:
            lines.append(rf"{name} & \tbd & \tbd & & & \\")
            continue
        lines.append(rf"{name} & {100 * dw.mean():+.1f}\% & {100 * dc.mean():+.1f}\% & {ww + wc}/{len(within) + len(cross)} "
                     rf"& {fmt_p(wilcoxon(d).pvalue)} & {df.mean():+.4f} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def load_profiles(d: str) -> dict:
    import glob
    import json
    prof = {}
    for p in glob.glob(os.path.join(d, "continual_results_*.json")):
        for e in json.load(open(p)):
            if "profile" in e:
                prof[e["method"]] = e["profile"]
    return prof


def step_ms(pr) -> float:
    t = pr["tasks"][1:]
    return 1000 * sum(x["train_s"] for x in t) / sum(x["steps"] for x in t)


def overall_table_skb(data, groups, profile_dir="results/profile/traffic", skb_dir="results/profile_skb/traffic"):
    """mt.overall_table, with the training time of an SKB row taken from its learner (SKB does not
    change training; for CLFD + SKB it is measured in the SKB profiling run)."""
    text = mt.overall_table(data, groups, profile_dir)
    prof, skb = load_profiles(profile_dir), load_profiles(skb_dir)
    er = prof.get("ifr_uniform")
    for m, name in SKB_FAMILY[1]:
        base = m.split("+")[0]
        pr = skb.get(f"{base}@cases") or prof.get(base)
        if pr is None or er is None:
            continue
        old = [ln for ln in text.splitlines() if ln.startswith(name + " &")]
        if old:
            text = text.replace(old[0], old[0].replace(r"\tbd", rf"{step_ms(pr) / step_ms(er):.2f}$\times$"))
    return text


def efficiency_table_skb(benches=(("traffic", "results/profile/traffic", "results/profile_skb/traffic"),
                                  ("order_A", "results/profile/order_A", "results/profile_skb/order_A")),
                         params_json="results/profile/params_total.json"):
    """Computational efficiency of every method (dedicated profiling runs, one method at a time on an
    idle GPU) plus CLFD + SKB. Parameters are total counts, frozen ones included (the profiling runs
    record only trainable ones, which hides CLFD's frozen encoder). For CLFD + SKB: training step and
    peak memory of the learner (trained on 90% of the data); boundary time including the knowledge-base
    update and weight fit; stored state including the snapshots (frozen parameters shared by all
    snapshots stored once), case segments and their predictions; inference of the deployed SKB."""
    import json
    total = json.load(open(params_json)) if os.path.exists(params_json) else {}
    names = {"order_A": "cross-domain", "traffic": "Traffic (862 var.)"}
    prof = {b: load_profiles(d) for b, d, _ in benches}
    skb = {b: load_profiles(sd).get("clfd@cases") for b, _, sd in benches}
    lines = [r"\begin{sidewaystable}", r"\centering",
             r"\caption{Computational efficiency, measured in dedicated runs with one method at a time on an "
             r"idle NVIDIA RTX 3090 (seed 0, batch size 32, three epochs per regime). \emph{ms/step}: training "
             r"time per optimiser step while replay is active (regimes 2--5); \emph{rel.}: relative to ER; "
             r"\emph{boundary}: mean time per regime spent on end-of-regime updates (e.g.\ Fisher information, "
             r"buffer filling, model snapshots); \emph{peak}: peak GPU memory during training; \emph{params}: "
             r"all parameters of the deployed model, frozen ones included (largest across benchmarks); "
             r"\emph{stored}: state kept besides the model, reported for Traffic, where a 500-window buffer of "
             r"$96\times862$ inputs dominates (IFR adds 100 bytes per regime to it); \emph{infer.}: forward "
             r"time per test batch. For CLFD + SKB, \emph{boundary} includes the knowledge-base update and the "
             r"weight fit, \emph{params} counts every stored parameter once (the encoder that CLFD freezes "
             r"after the first regime is shared by all snapshots), \emph{stored} adds the snapshots' "
             r"remaining parameters, the case segments and their predictions to the replay buffer, and "
             r"\emph{infer.} evaluates every snapshot with non-zero weight. ER, IFR and the FreDF and CLFD "
             r"variants keep their buffer on the GPU for speed, which raises their peak memory relative to "
             r"methods that sample it from host memory.}",
             r"\label{tab:efficiency}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\resizebox{\textheight}{!}{%",
             r"\begin{tabular}{l" + "rrrr" * len(benches) + "rrr}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{names.get(b, b)}}}" for b, _, _ in benches)
             + r" & \multicolumn{3}{c}{model and memory} \\",
             "".join(rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(benches)))
             + rf"\cmidrule(lr){{{2 + 4 * len(benches)}-{4 + 4 * len(benches)}}}",
             "Method & " + " & ".join(["ms/step & rel. & boundary (s) & peak (MB)"] * len(benches))
             + r" & params & stored (MB) & infer.\ (ms) \\", r"\midrule"]

    def row_for(name, prs, extra_boundary=None, params=None, stored=None, infer=None):
        row = [name]
        for b, _, _ in benches:
            pr, er = prs.get(b), prof[b].get("ifr_uniform")
            if pr is None or er is None:
                row += [r"\tbd"] * 4
                continue
            boundary = np.mean([x["boundary_s"] for x in pr["tasks"]]) + (extra_boundary(b) if extra_boundary else 0)
            row += [f"{step_ms(pr):.1f}", rf"{step_ms(pr) / step_ms(er):.2f}$\times$", f"{boundary:.1f}",
                    f"{max(x['peak_mem_mb'] for x in pr['tasks']):.0f}"]
        row += [params, stored, infer]
        return " & ".join(row) + r" \\"

    for family, members in mt.FAMILIES:
        if family == SKB_FAMILY[0]:
            continue
        for m, name in members:
            prs = {b: prof[b].get(m) for b, _, _ in benches}
            have = [p for p in prs.values() if p]
            if not have:
                lines.append(" & ".join([name] + [r"\tbd"] * (4 * len(benches) + 3)) + r" \\")
                continue
            n = max(max(p["params"], total.get(b, {}).get(m, 0)) for b, p in prs.items() if p)
            tr = prs.get("traffic")
            lines.append(row_for(name, prs, params=f"{n / 1e3:.0f}K",
                                 stored=f"{tr['stored_bytes'] / 2**20:.2f}" if tr else r"\tbd",
                                 infer=f"{np.mean([p['infer_ms_per_batch'] for p in have]):.2f}"))
    if all(skb.values()):
        tr = skb["traffic"]
        rb = tr["routing_bytes"]
        own = tr.get("params_total", tr["params"]) * 4
        stored = tr["stored_bytes"] + rb["snapshots"] - own + rb["cases"] + rb["case_predictions"]
        n = max(s["skb"].get("params_unique", s["params"] * s["skb"]["active_snapshots"]) for s in skb.values())
        lines.append(r"\midrule")
        lines.append(row_for(r"\textbf{CLFD + SKB (ours)}", skb,
                             extra_boundary=lambda b: np.mean(skb[b]["skb"]["update_s"]) + skb[b]["skb"]["fit_s"],
                             params=f"{n / 1e3:.0f}K", stored=f"{stored / 2**20:.2f}",
                             infer=f"{np.mean([s['skb']['infer_ms_per_batch'] for s in skb.values()]):.2f}"))
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{sidewaystable}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/tables")
    ap.add_argument("--route_root", default="results/test_route")
    ap.add_argument("--min_seeds", type=int, default=5)
    ap.add_argument("--cite_companion", default=None,
                    help="BibTeX key of the companion paper: label IFR and SpectraCL-v1 with it and credit it "
                         "for the results of the methods without SKB in the captions")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    within = mt.merged(["results/pilot/*/continual_results_*.json", "results/main/*/continual_results_*.json"])
    cross = mt.merged(["results/crossdomain/*/continual_results_*.json"])
    routed, _ = load_routed(args.route_root)
    for ds, ms in routed.items():
        target = cross if ds.startswith("order_") else within
        for m, seeds in ms.items():
            target.setdefault(ds, {})[m] = dict(seeds)
    both = {**within, **cross}

    # IFR is no longer the headline method; SKB rows are added as their own family.
    key = args.cite_companion
    labels = {"ifr": rf"IFR~\cite{{{key}}}" if key else "IFR (ours)",
              "spectracl": rf"SpectraCL-v1~\cite{{{key}}}" if key else "SpectraCL-v1 (overlap)"}
    mt.FAMILIES[:] = [(f, [(m, labels.get(m, n)) for m, n in g]) for f, g in mt.FAMILIES]
    mt.FAMILIES.append(SKB_FAMILY)
    mt.METHODS[:] = [m for _, group in mt.FAMILIES for m in group]
    credit = (rf" Results of the methods without SKB are those of~\cite{{{key}}} (same runs, seeds and splits)."
              if key else "")

    files = {
        "tab_within.tex": mt.main_table(
            within, mt.WITHIN, args.min_seeds,
            r"Within-dataset benchmark: average test MSE over all regimes after the last one "
            r"(mean $\pm$ std over five seeds; $L=96$, $H=48$), grouped by method family. "
            r"Best per column in bold.", "tab:within", metrics=(("avg_mse", "MSE"),)),
        "tab_within_fgt.tex": mt.main_table(
            within, mt.WITHIN, args.min_seeds,
            r"Within-dataset benchmark: average forgetting after the last regime (mean $\pm$ std over "
            r"five seeds). Best per column in bold.", "tab:within_fgt",
            metrics=(("avg_forgetting", "Forgetting"),)),
        "tab_cross.tex": mt.main_table(
            cross, mt.CROSS, 10,
            r"Cross-domain benchmark: five univariate sources learned in sequence, three task orders, "
            r"10 seeds, grouped by method family. Best per column in bold.", "tab:cross"),
        "tab_overall.tex": overall_table_skb(both, mt.WITHIN + mt.CROSS),
        "tab_skbrules.tex": skb_rules_table(both, mt.WITHIN + mt.CROSS),
        "tab_skbbases.tex": skb_bases_table(both, mt.WITHIN, mt.CROSS),
        "tab_efficiency.tex": efficiency_table_skb(),
    }
    for name in ("tab_within.tex", "tab_within_fgt.tex", "tab_cross.tex", "tab_overall.tex"):
        if credit:   # append the credit sentence to the caption (first closing brace of \caption{...})
            text = files[name]
            start = text.index(r"\caption{") + len(r"\caption{")
            depth, i = 1, start
            while depth:
                depth += {"{": 1, "}": -1}.get(text[i], 0)
                i += 1
            files[name] = text[:i - 1] + credit + text[i - 1:]
    for name, text in files.items():
        with open(os.path.join(args.out, name), "w") as f:
            f.write(text)
        print("wrote", name, "| unfinished cells:", text.count(r"\tbd"))


if __name__ == "__main__":
    main()
