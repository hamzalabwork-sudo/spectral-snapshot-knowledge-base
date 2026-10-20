from __future__ import annotations

"""
Figure: per-frequency snapshot weights alpha(k) learned by SKB at the end of the regime sequence
(mean over seeds), for a few benchmarks.

The weights are computed from the snapshot predictions of the test runs
(results/test_route/<bench>/snapshots/*.npz) when they are present, and written to a small cache
(results/derived/skb_weights.json); without the predictions the figure is drawn from the cache.

Usage:
    python experiments/fig_skb_weights.py --root results/test_route --base clfd \
        --benches ettm1 weather order_A order_B --out outputs/figs/fig6_skb_weights.pdf
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from offline_rules import Dump, Stage  # noqa: E402

NAMES = {"electricity": "Electricity", "traffic": "Traffic", "ettm1": "ETTm1", "ettm2": "ETTm2",
         "weather": "Weather", "order_A": "Cross-domain A", "order_B": "Cross-domain B",
         "order_C": "Cross-domain C"}
# single-hue sequential ramp (light surface -> blue -> dark blue), as in the paper's palette
CMAP = LinearSegmentedColormap.from_list("seq", ["#f6f6f3", "#9cc3ef", "#2a78d6", "#123f75"])


def mean_weights(root: str, bench: str, base: str):
    """Mean SKB weights (F, S) over the seeds of `base` on `bench`, or (None, 0) without predictions."""
    ws = []
    for p in sorted(glob.glob(os.path.join(root, bench, "snapshots", "*.npz"))):
        m = re.search(r"pid\d+_(?P<m>.+)_seed\d+\.npz$", os.path.basename(p))
        if m is None or m.group("m") != base:
            continue
        d = Dump(p)
        ws.append(Stage(d, d.S - 1).weights("fglob")[:, 0, :].numpy())   # (F, S), channel 0
    return (np.mean(ws, axis=0), len(ws)) if ws else (None, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/test_route")
    ap.add_argument("--base", default="clfd")
    ap.add_argument("--benches", nargs="+", default=["ettm1", "weather", "order_A", "order_B"])
    ap.add_argument("--out", default="outputs/figs/fig6_skb_weights.pdf")
    ap.add_argument("--cache", default="results/derived/skb_weights.json")
    args = ap.parse_args()

    cache = json.load(open(args.cache)) if os.path.exists(args.cache) else {}
    data, updated = {}, False
    for b in args.benches:
        w, n = mean_weights(args.root, b, args.base)
        key = f"{args.base}:{b}"
        if n:
            cache[key] = {"weights": w.tolist(), "seeds": n}
            updated = True
        elif key not in cache:
            raise SystemExit(f"no snapshot predictions under {args.root}/{b} and no cached weights for {key}")
        data[b] = (np.array(cache[key]["weights"]), cache[key]["seeds"])
    if updated:
        os.makedirs(os.path.dirname(os.path.abspath(args.cache)), exist_ok=True)
        json.dump(cache, open(args.cache, "w"), indent=1)

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig, axes = plt.subplots(1, len(args.benches), figsize=(1.85 * len(args.benches) + 0.6, 2.05),
                             sharey=True, constrained_layout=True)
    for ax, b in zip(np.atleast_1d(axes), args.benches):
        w, n = data[b]
        im = ax.imshow(w.T, aspect="auto", origin="lower", cmap=CMAP, vmin=0, vmax=1,
                       extent=(-0.5, w.shape[0] - 0.5, 0.5, w.shape[1] + 0.5), interpolation="nearest")
        ax.set_title(f"{NAMES.get(b, b)} ({n} seeds)")
        ax.set_xlabel("forecast frequency bin $k$")
        ax.set_yticks(range(1, w.shape[1] + 1))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("snapshot after regime $s$")
    cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)
    cb.set_label(r"weight $\alpha_s(k)$")
    cb.outline.set_visible(False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(os.path.splitext(args.out)[0] + ".png", dpi=200, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
