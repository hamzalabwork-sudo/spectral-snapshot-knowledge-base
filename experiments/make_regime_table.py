from __future__ import annotations

"""
Appendix table of regime definitions: calendar range and train/test window counts per
regime, generated from the same data loader the experiments use.

Usage:
    python experiments/make_regime_table.py --out outputs/tables/tab_regimes.tex
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataloader import SEASONAL_REGIME_REGISTRY
from run_continual import build_task_data, load_config

DATASETS = [("Electricity", "configs/within/electricity.yaml", "data/electricity.csv"),
            ("Traffic", "configs/within/traffic.yaml", "data/traffic.csv"),
            ("ETTm1", "configs/within/ettm1.yaml", "data/ETTm1.csv"),
            ("ETTm2", "configs/within/ettm2.yaml", "data/ETTm2.csv"),
            ("Weather", "configs/within/weather.yaml", "data/weather.csv")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/tables/tab_regimes.tex")
    args = ap.parse_args()
    lines = [r"\begin{table}[h]", r"\centering",
             r"\caption{Regime definitions. Calendar range of each regime (the last regime runs to the end "
             r"of the series) and the number of training and test windows ($L=96$, $H=48$).}",
             r"\label{tab:regimes}", r"\small",
             r"\begin{tabular}{llrr}", r"\toprule", r"Dataset & Regime (calendar range) & Train & Test \\",
             r"\midrule"]
    for name, cfg_path, data in DATASETS:
        dc = load_config(cfg_path)["data"]
        split = dc.get("regime_split", "equal")
        tr, te, _ = build_task_data(data, 96, 48, 5, 32, feature_cols=dc.get("feature_cols"),
                                    target_cols=dc.get("target_cols"), date_col=dc.get("date_col", "date"),
                                    task_split=split)
        regimes = SEASONAL_REGIME_REGISTRY[split]
        for k, ((start, end), l_tr, l_te) in enumerate(zip(regimes, tr, te)):
            label = name if k == 0 else ""
            lines.append(rf"{label} & $T_{k + 1}$: {start} to {end} & {len(l_tr.dataset):,} & {len(l_te.dataset):,} \\")
        lines.append(r"\midrule" if name != DATASETS[-1][0] else r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table}"]
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
