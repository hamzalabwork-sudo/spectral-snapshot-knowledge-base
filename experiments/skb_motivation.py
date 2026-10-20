from __future__ import annotations

"""
Is the knowledge a continual learner loses still present in its earlier snapshots, frequency by
frequency? For every old regime j (all but the last) and output frequency bin k of the test runs
(CLFD learner):

  * err_s(j, k): mean error power of snapshot s at bin k on regime j's test windows, s >= j
    (snapshots that have seen regime j);
  * earlier-better: some earlier snapshot (j <= s < final) has a lower error than the final one;
  * mixed: within regime j, the snapshot with the lowest error differs across bins.

Writes both shares per benchmark as a LaTeX table and draws the error spectra of the snapshot taken
right after regime j, the final snapshot and SKB for chosen (benchmark, regime) pairs. The statistics are
computed from the snapshot predictions (results/test_route/<bench>/snapshots/*.npz) when present and
cached in results/derived/skb_motivation.json; without the predictions the cache is used.

Usage:
    python experiments/skb_motivation.py --root results/test_route --base clfd \
        --show order_A:0 weather:0 --out outputs/figs/fig_motivation.pdf \
        --table outputs/tables/tab_motivation.tex
"""

import argparse
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from offline_rules import Dump, Stage, smix  # noqa: E402

NAMES = {"electricity": "Electricity", "traffic": "Traffic", "ettm1": "ETTm1", "ettm2": "ETTm2",
         "weather": "Weather", "order_A": "Cross-domain A", "order_B": "Cross-domain B",
         "order_C": "Cross-domain C"}
ORDER = ["electricity", "traffic", "ettm1", "ettm2", "weather", "order_A", "order_B", "order_C"]
BLUE, ORANGE, AQUA, SHADE = "#2a78d6", "#eb6834", "#1baf7a", "#ebf2fc"


def spectra(d: Dump):
    """Per regime j: (S, F) mean error power of every snapshot, and SKB's (F,), at the last stage."""
    S = d.S
    w = Stage(d, S - 1).weights("fglob")
    out = []
    for j in range(S):
        preds = torch.stack([d.eval_pred[s][j] for s in range(S)], dim=1)       # (n, S, H, C)
        y = d.eval_y[j]
        e = torch.fft.rfft(preds - y[:, None], dim=2).abs().pow(2).mean(dim=(0, 3))
        skb = smix(w.expand(preds.shape[0], -1, -1, -1), preds)
        es = torch.fft.rfft(skb - y, dim=1).abs().pow(2).mean(dim=(0, 2))
        out.append((e.numpy(), es.numpy()))
    return out


def summarise(runs, show_regimes):
    """Counts for the table and mean spectra for the requested regimes of one benchmark."""
    eb, mixed = [], []
    for run in runs:
        for j in range(len(run) - 1):
            e = run[j][0][j:]                                                   # snapshots s >= j
            eb += list(e[:-1].min(axis=0) < e[-1])
            mixed.append(len(set(e.argmin(axis=0).tolist())) > 1)
    out = {"earlier": [int(np.sum(eb)), len(eb)], "mixed": [int(np.sum(mixed)), len(mixed)],
           "seeds": len(runs), "spectra": {}}
    for j in show_regimes:
        out["spectra"][str(j)] = {"diag": np.mean([r[j][0][j] for r in runs], axis=0).tolist(),
                                  "final": np.mean([r[j][0][-1] for r in runs], axis=0).tolist(),
                                  "skb": np.mean([r[j][1] for r in runs], axis=0).tolist()}
    return out


def latex_table(rows, tot_e, tot_m) -> str:
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Knowledge that survives in earlier snapshots (test split, CLFD learner, all seeds). "
             r"\emph{Earlier better}: share of (old regime, forecast frequency bin) pairs in which a snapshot "
             r"taken after the regime was learned, but before the last regime, has a lower error power on "
             r"that regime than the final snapshot. \emph{Mixed}: share of old regimes for which the "
             r"snapshot with the lowest error differs between frequency bins.}",
             r"\label{tab:motivation}", r"\small", r"\begin{tabular}{lcc}", r"\toprule",
             r"Benchmark & Earlier better & Mixed \\", r"\midrule"]
    lines += [rf"{name} & {100 * e:.0f}\% & {100 * m:.0f}\% \\" for name, e, m in rows]
    lines += [r"\midrule", rf"All & {100 * tot_e:.0f}\% & {100 * tot_m:.0f}\% \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/test_route")
    ap.add_argument("--base", default="clfd")
    ap.add_argument("--show", nargs="*", default=["order_A:0", "weather:0"])
    ap.add_argument("--out", default="outputs/figs/fig_motivation.pdf")
    ap.add_argument("--table", default="outputs/tables/tab_motivation.tex")
    ap.add_argument("--cache", default="results/derived/skb_motivation.json")
    args = ap.parse_args()

    show = {}
    for spec in args.show:
        b, j = spec.split(":")
        show.setdefault(b, []).append(int(j))
    cache = json.load(open(args.cache)) if os.path.exists(args.cache) else {}
    stats, updated = {}, False
    for b in ORDER:
        runs = []
        for p in sorted(glob.glob(os.path.join(args.root, b, "snapshots", "*.npz"))):
            m = re.search(r"pid\d+_(?P<m>.+)_seed\d+\.npz$", os.path.basename(p))
            if m and m.group("m") == args.base:
                runs.append(spectra(Dump(p)))
        key = f"{args.base}:{b}"
        if runs:
            cache[key] = summarise(runs, show.get(b, []))
            updated = True
        if key in cache:
            stats[b] = cache[key]
    if updated:
        os.makedirs(os.path.dirname(os.path.abspath(args.cache)), exist_ok=True)
        json.dump(cache, open(args.cache, "w"), indent=1)
    if not stats:
        raise SystemExit(f"no snapshot predictions under {args.root} and no cache at {args.cache}")

    print(f"{'benchmark':12} {'earlier better':>15} {'mixed regimes':>14}")
    rows = []
    for b, st in stats.items():
        e, m = st["earlier"][0] / st["earlier"][1], st["mixed"][0] / st["mixed"][1]
        rows.append((NAMES[b], e, m))
        print(f"{b:12} {100 * e:14.0f}% {100 * m:13.0f}%")
    tot_e = sum(s["earlier"][0] for s in stats.values()) / sum(s["earlier"][1] for s in stats.values())
    tot_m = sum(s["mixed"][0] for s in stats.values()) / sum(s["mixed"][1] for s in stats.values())
    print(f"{'all':12} {100 * tot_e:14.0f}% {100 * tot_m:13.0f}%")
    os.makedirs(os.path.dirname(os.path.abspath(args.table)), exist_ok=True)
    with open(args.table, "w") as f:
        f.write(latex_table(rows, tot_e, tot_m))
    print("wrote", args.table)

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})
    fig, axes = plt.subplots(1, len(args.show), figsize=(3.3 * len(args.show), 2.4), constrained_layout=True)
    for ax, spec in zip(np.atleast_1d(axes), args.show):
        b, j = spec.split(":")
        sp = stats[b]["spectra"][j]
        diag, final, skb = (np.array(sp[k]) for k in ("diag", "final", "skb"))
        k = np.arange(len(diag))
        better = diag < final
        jj = int(j) + 1
        ax.bar(k[better], np.ones(better.sum()), width=1.0, color=SHADE, zorder=0,
               transform=ax.get_xaxis_transform(), label=f"regime-{jj} snapshot better")
        ax.plot(k, diag, color=BLUE, lw=2, label=f"snapshot after regime {jj}")
        ax.plot(k, final, color=ORANGE, lw=2, label="final snapshot")
        ax.plot(k, skb, color=AQUA, lw=2, ls="--", label="SKB")
        ax.set_yscale("log")
        ax.set_xlim(-0.5, len(k) - 0.5)
        ax.set_title(f"{NAMES[b]}: error on regime {jj} ({stats[b]['seeds']} seeds)")
        ax.set_xlabel("forecast frequency bin $k$")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("error power $E(k)$")
    np.atleast_1d(axes)[-1].legend(frameon=False, loc="upper right")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(os.path.splitext(args.out)[0] + ".png", dpi=200, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
