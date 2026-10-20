from __future__ import annotations

"""
Offline evaluation of snapshot prediction rules from SnapshotRouter dumps
(results/val_route/<bench>/snapshots/*.npz), without retraining.

Every rule maps the knowledge available at stage i (snapshots 0..i, case segments and
recognition sets of regimes 0..i) to a forecast for every evaluation window; the results matrix
and continual-learning metrics are recomputed exactly as in the runs (mean of per-batch means).
Rules reproduced from the runs serve as a consistency check (--check).

Usage:
    python experiments/offline_rules.py --rules final uniform stack_knn --check
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from spectracl.routing import simplex_weights  # noqa: E402
from summarize_benchmark import cl_metrics  # noqa: E402

BS = 32


class Dump:
    def __init__(self, path: str):
        z = np.load(path)
        keys = z.files
        self.S = 1 + max(int(re.match(r"eval_pred_s(\d+)_r0$", k).group(1)) for k in keys
                         if re.match(r"eval_pred_s\d+_r0$", k))
        self.J = 1 + max(int(re.match(r"eval_y_r(\d+)$", k).group(1)) for k in keys if k.startswith("eval_y_r"))
        t = lambda k: torch.from_numpy(z[k].astype(np.float32))
        self.eval_pred = [[t(f"eval_pred_s{s}_r{j}") for j in range(self.J)] for s in range(self.S)]
        self.eval_y = [t(f"eval_y_r{j}") for j in range(self.J)]
        self.eval_feat = [t(f"eval_feat_r{j}") for j in range(self.J)]
        self.case_pred = {(s, r): t(f"case_pred_s{s}_r{r}") for s in range(self.S) for r in range(self.S)
                          if f"case_pred_s{s}_r{r}" in keys}
        self.case_y = [t(f"case_y_r{r}") for r in range(self.S)]
        self.case_feat = [t(f"case_feat_r{r}") for r in range(self.S)]
        self.rec_feat = [t(f"rec_feat_r{r}") for r in range(self.S)]
        self.groups = torch.from_numpy(z["groups"]).long()
        self.cache = {}


def batched(err: torch.Tensor) -> float:
    return float(torch.stack([c.mean() for c in err.split(BS)]).mean())


def metrics(pred, y) -> dict:
    return {"mse": batched(((pred - y) ** 2).mean(dim=(1, 2))), "mae": batched((pred - y).abs().mean(dim=(1, 2)))}


def knn_labels(f, ref, labels, n_cls, groups, k=10):
    mu, sd = ref.mean(dim=0), ref.std(dim=0) + 1e-6
    w = (1.0 / torch.bincount(groups).float()[groups]).sqrt()
    idx = torch.cdist((f - mu) / sd * w, (ref - mu) / sd * w).topk(min(k, ref.shape[0]), largest=False).indices
    votes = torch.zeros(f.shape[0], n_cls).scatter_add_(1, labels[idx], torch.ones_like(idx, dtype=torch.float))
    return votes.argmax(dim=1), votes


def dlda_labels(f, ref, labels, n_cls, groups):
    mu, sd = ref.mean(dim=0), ref.std(dim=0) + 1e-6
    wg = 1.0 / torch.bincount(groups).float()[groups]
    z, rz = (f - mu) / sd, (ref - mu) / sd
    cent = torch.stack([rz[labels == r].mean(dim=0) for r in range(n_cls)])
    wvar = torch.stack([((rz[labels == r] - cent[r]) ** 2).mean(dim=0) for r in range(n_cls)]).mean(dim=0) + 1e-6
    return (wg * (z[:, None, :] - cent[None]) ** 2 / wvar).sum(dim=-1).argmin(dim=1)


class Stage:
    """Knowledge available after regime i (snapshots 0..i), with derived quantities cached."""

    def __init__(self, d: Dump, i: int):
        self.d, self.S, self.R = d, i + 1, i + 1
        S, R = self.S, self.R
        self.cerr = [torch.stack([d.case_pred[(s, r)] for s in range(S)], dim=1) - d.case_y[r][:, None]
                     for r in range(R)]                                          # (m_r, S, H, C)
        self.cspec = [torch.fft.rfft(e, dim=2) for e in self.cerr]
        self.case_mse = torch.stack([(e ** 2).mean(dim=(0, 2, 3)) for e in self.cerr])   # (R, S)
        self.cf = torch.cat(d.case_feat[:R])
        self.cl = torch.cat([torch.full((f.shape[0],), r) for r, f in enumerate(d.case_feat[:R])])
        self.rf = torch.cat(d.rec_feat[:R])
        self.rl = torch.cat([torch.full((f.shape[0],), r) for r, f in enumerate(d.rec_feat[:R])])
        G = torch.stack([torch.einsum("nshc,nthc->st", e, e) / e[:, 0].numel() for e in self.cerr])
        Gf = torch.stack([torch.einsum("nsfc,ntfc->fcst", c, c.conj()).real / c.shape[0] for c in self.cspec])
        self.G, self.Gf = G, Gf
        self._w = {}

    def weights(self, key):
        if key not in self._w:
            self._w[key] = {
                "reg": lambda: simplex_weights(self.G),
                "glob": lambda: simplex_weights(self.G.mean(dim=0)),
                "freg": lambda: simplex_weights(self.Gf),
                "fglob": lambda: simplex_weights(self.Gf.mean(dim=0)),
            }[key]()
        return self._w[key]

    def recognise(self, j, how):
        f = self.d.eval_feat[j]
        g = self.d.groups
        if how == "case_knn":
            return knn_labels(f, self.cf, self.cl, self.R, g)[0]
        if how == "buf_knn":
            return knn_labels(f, self.rf, self.rl, self.R, g)[0]
        if how == "case_dlda":
            return dlda_labels(f, self.cf, self.cl, self.R, g)
        if how == "buf_dlda":
            return dlda_labels(f, self.rf, self.rl, self.R, g)
        raise KeyError(how)


def mix(a, preds):
    return torch.einsum("ns,nshc->nhc", a, preds)


def smix(a, preds):
    spec = torch.fft.rfft(preds, dim=2)
    return torch.fft.irfft(torch.einsum("nfcs,nsfc->nfc", a.to(spec.dtype), spec), n=preds.shape[2], dim=1)


def kept_sets(d: Dump, budget: int) -> list[list[int]]:
    """Snapshots kept under a budget, decided causally: after each regime, the new snapshot joins
    the kept set and, if the budget is exceeded, the snapshot with the least SKB weight is dropped
    for good."""
    key = ("kept", budget)
    if key not in d.cache:
        sets, prev = [], []
        for i in range(d.S):
            cand = prev + [i]
            if len(cand) > budget:
                idx = torch.tensor(cand)
                G = Stage(d, i).Gf.mean(dim=0)[..., idx[:, None], idx[None, :]]
                keep = simplex_weights(G).sum(dim=(0, 1)).topk(budget).indices.sort().values
                cand = [cand[k] for k in keep.tolist()]
            sets.append(cand)
            prev = cand
        d.cache[key] = sets
    return d.cache[key]


def budget_weights(st: Stage, budget: int, how: str) -> torch.Tensor:
    """SKB weights (F, C, S) using at most `budget` stored snapshots: the causally kept set
    (how="top") or the most recent snapshots (how="recent")."""
    S = st.S
    if S <= budget:
        return st.weights("fglob")
    keep = torch.tensor(kept_sets(st.d, budget)[S - 1]) if how == "top" else torch.arange(S - budget, S)
    G = st.Gf.mean(dim=0)[..., keep[:, None], keep[None, :]]
    w = torch.zeros(G.shape[:-2] + (S,))
    w[..., keep] = simplex_weights(G)
    return w


def causal_matrix(S, R):
    c = torch.stack([(torch.arange(S) >= r).float() for r in range(R)])
    return c / c.sum(dim=1, keepdim=True)


# Each rule: (stage, j, preds (n, S, H, C)) -> forecast (n, H, C)
RULES = {
    "final": lambda st, j, p: p[:, -1],
    "uniform": lambda st, j, p: p.mean(dim=1),
    "causal_buf": lambda st, j, p: mix(causal_matrix(st.S, st.R)[st.recognise(j, "buf_knn")], p),
    "causal_buf_dlda": lambda st, j, p: mix(causal_matrix(st.S, st.R)[st.recognise(j, "buf_dlda")], p),
    "causal_case_dlda": lambda st, j, p: mix(causal_matrix(st.S, st.R)[st.recognise(j, "case_dlda")], p),
    "stack_global": lambda st, j, p: mix(st.weights("glob").expand(p.shape[0], -1), p),
    "stack_knn": lambda st, j, p: mix(st.weights("reg")[st.recognise(j, "case_knn")], p),
    "stack_dlda": lambda st, j, p: mix(st.weights("reg")[st.recognise(j, "case_dlda")], p),
    "sstack_global": lambda st, j, p: smix(st.weights("fglob").expand(p.shape[0], -1, -1, -1), p),
    "sstack_knn": lambda st, j, p: smix(st.weights("freg")[st.recognise(j, "case_knn")], p),
    "sstack_dlda": lambda st, j, p: smix(st.weights("freg")[st.recognise(j, "case_dlda")], p),
    # memory budgets for SKB (analysis, not candidates): at most 2 or 3 stored snapshots
    "skb_top2": lambda st, j, p: smix(budget_weights(st, 2, "top").expand(p.shape[0], -1, -1, -1), p),
    "skb_top3": lambda st, j, p: smix(budget_weights(st, 3, "top").expand(p.shape[0], -1, -1, -1), p),
    "skb_last2": lambda st, j, p: smix(budget_weights(st, 2, "recent").expand(p.shape[0], -1, -1, -1), p),
    "skb_last3": lambda st, j, p: smix(budget_weights(st, 3, "recent").expand(p.shape[0], -1, -1, -1), p),
}


def evaluate(d: Dump, rules: list[str]) -> dict[str, list[list[dict]]]:
    out = {r: [] for r in rules}
    for i in range(d.S):
        st = Stage(d, i)
        rows = {r: [] for r in rules}
        for j in range(d.J):
            preds = torch.stack([d.eval_pred[s][j] for s in range(st.S)], dim=1)
            for r in rules:
                rows[r].append(metrics(RULES[r](st, j, preds), d.eval_y[j]))
        for r in rules:
            out[r].append(rows[r])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/val_route")
    ap.add_argument("--rules", nargs="+", default=list(RULES))
    ap.add_argument("--check", action="store_true", help="compare reproduced rules with the in-run values")
    ap.add_argument("--out", default=None, help="write {bench: {base+rule: {seed: metrics}}} as JSON")
    args = ap.parse_args()

    res = defaultdict(lambda: defaultdict(dict))
    worst = 0.0
    for path in sorted(glob.glob(os.path.join(args.root, "*", "snapshots", "*.npz"))):
        bench = os.path.basename(os.path.dirname(os.path.dirname(path)))
        m = re.search(r"pid\d+_(?P<m>.+)_seed(?P<s>\d+)\.npz$", os.path.basename(path))
        base, seed = m.group("m"), int(m.group("s"))
        mats = evaluate(Dump(path), args.rules)
        for r, R in mats.items():
            res[bench][f"{base}+{r}"][seed] = cl_metrics(R)
        if args.check:
            for jp in glob.glob(os.path.join(args.root, bench, "continual_results_*.json")):
                for e in json.load(open(jp)):
                    if e["method"].split("@")[0] == base and e["seed"] == seed and "routing" in e:
                        for r in args.rules:
                            if r in e["routing"]:
                                a = cl_metrics(e["routing"][r])["avg_mse"]
                                b = res[bench][f"{base}+{r}"][seed]["avg_mse"]
                                worst = max(worst, abs(a - b) / a)
        print(f"{bench:12} {base:9} seed {seed}: " + "  ".join(
            f"{r}={res[bench][f'{base}+{r}'][seed]['avg_mse']:.4f}" for r in args.rules), flush=True)
    if args.check:
        print(f"max relative difference to in-run values: {worst:.2e}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
