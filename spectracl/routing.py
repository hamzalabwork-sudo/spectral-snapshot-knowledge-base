from __future__ import annotations

"""
Snapshot knowledge base for continual forecasting: routing, fusion and ensembling.

After every regime the forecaster is snapshotted. Two kinds of stored knowledge are used:

  * a recognition set: features (level + relative band power) of random training windows of
    each regime, used only to recognise which regime a new window resembles (a classifier
    target, so training windows are unbiased for it);
  * a case segment: the last `case_ratio` of each regime's training segment, held out from
    training, on which every snapshot's errors are recorded (unbiased error estimates).

From these, SnapshotRouter evaluates a fixed set of prediction rules after every regime, all
on the same snapshots, so one training run yields the base method ("final") and every rule:

  final            the last snapshot (= the base continual learner)
  uniform          mean prediction of all snapshots
  soup             one model with the mean parameters of all snapshots
  causal_buf       mean of the snapshots trained after the recognised regime arrived
  route_*          per regime, the snapshot with the lowest case error; regime recognised by
                   k nearest cases (knn, knn_soft) or a diagonal discriminant (dlda)
  knn_err          per window, the snapshot with the lowest error on its nearest cases
  fuse_*           per output frequency bin, the snapshot with the lowest case error power
  stack_*          convex weights over snapshots minimising the case MSE (Bates-Granger)
  sstack_*         the same, solved separately for every output frequency bin

By Parseval the horizon MSE is a sum of per-bin error powers, so bin-wise selection (fuse) and
bin-wise convex weights (sstack) are exact decompositions of the MSE objective.
"known", "oracle" and the "*_known"/"*_oracle" rules use evaluation labels: diagnostics only.
"""

import copy
import time

import numpy as np
import torch
import torch.nn as nn

VARIANTS = (
    "final", "uniform", "soup", "causal_buf",
    "route_knn", "route_knn_soft", "route_dlda", "knn_err",
    "fuse_knn", "fuse_dlda", "fuse_global",
    "stack_global", "stack_knn", "sstack_global", "sstack_knn",
    "known", "oracle", "fuse_known", "fuse_oracle", "causal_known", "stack_known", "sstack_known",
)


def dataset_windows(ds) -> tuple[torch.Tensor, torch.Tensor]:
    """All (x, y) windows of a TimeSeriesDataset as zero-copy views, in dataset order."""
    n = len(ds)
    x = ds.input_data.unfold(0, ds.seq_len, 1)[:n].transpose(1, 2)                  # (n, L, D)
    y = ds.target_data[ds.seq_len:].unfold(0, ds.pred_len, 1)[:n].transpose(1, 2)    # (n, H, C)
    return x, y


def _band_edges(n_bins: int, n_bands: int) -> list[tuple[int, int]]:
    """Log-spaced bands over rfft bins 1..n_bins-1 (DC excluded)."""
    edges = torch.unique(torch.logspace(0, torch.log10(torch.tensor(float(n_bins))), n_bands + 1).long())
    edges = [int(e) for e in edges if e <= n_bins]
    edges[-1] = n_bins
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def window_features(x: torch.Tensor, target_idx: list[int], n_bands: int = 8) -> tuple[torch.Tensor, list[int]]:
    """Level and relative band-power features of windows x: (B, L, D) -> (B, F), group ids (F,).

    Groups: 0 target level, 1 target spectrum, 2 context level, 3 context spectrum.
    """
    mean = x.mean(dim=1)
    std = x.std(dim=1)
    z = (x - mean[:, None]) / (std[:, None] + 1e-3)
    power = torch.fft.rfft(z, dim=1).abs() ** 2                                        # (B, F, D)
    bands = _band_edges(power.shape[1], n_bands)
    bp = torch.stack([power[:, a:b].sum(dim=1) for a, b in bands], dim=1)              # (B, nb, D)
    logbp = torch.log(bp / (bp.sum(dim=1, keepdim=True) + 1e-8) + 1e-4)
    t = target_idx
    feats = [mean[:, t], torch.log(std[:, t] + 1e-3), logbp[..., t].flatten(1)]
    groups = [0] * (2 * len(t)) + [1] * (logbp.shape[1] * len(t))
    if x.shape[-1] > len(t):
        feats += [mean, torch.log(std + 1e-3), logbp.mean(dim=-1)]
        groups += [2] * (2 * x.shape[-1]) + [3] * logbp.shape[1]
    return torch.cat(feats, dim=1), groups


def simplex_weights(G: torch.Tensor) -> torch.Tensor:
    """argmin_a a^T G a over the probability simplex, batched over the leading dims of G (..., S, S).

    Exact for small S: every support set is tried (equality-constrained solution
    a = G_T^-1 1 / 1^T G_T^-1 1, objective 1 / 1^T G_T^-1 1) and the best feasible one kept.
    """
    G = G.double()
    S = G.shape[-1]
    eye = torch.eye(S, dtype=G.dtype)
    G = G + 1e-9 * G.diagonal(dim1=-2, dim2=-1).mean(dim=-1)[..., None, None] * eye
    best_val = torch.full(G.shape[:-2], float("inf"), dtype=G.dtype)
    best = torch.full(G.shape[:-1], 1.0 / S, dtype=G.dtype)
    for mask in range(1, 2 ** S):
        idx = torch.tensor([i for i in range(S) if mask >> i & 1])
        Gt = G[..., idx[:, None], idx[None, :]]
        x = torch.linalg.solve(Gt, torch.ones(Gt.shape[:-1] + (1,), dtype=G.dtype)).squeeze(-1)
        tot = x.sum(dim=-1)
        a = x / tot[..., None]
        ok = (a >= -1e-10).all(dim=-1) & (tot > 0)
        val = torch.where(ok, 1.0 / tot, torch.full_like(tot, float("inf")))
        upd = val < best_val
        full = torch.zeros(G.shape[:-1], dtype=G.dtype)
        full[..., idx] = a.clamp_min(0)
        best = torch.where(upd[..., None], full, best)
        best_val = torch.where(upd, val, best_val)
    return (best / best.sum(dim=-1, keepdim=True)).float()


class SpectralStack(nn.Module):
    """Deployable snapshot knowledge base: per-frequency convex combination of snapshots.

    weights: (F, C, S) simplex weights over snapshots for every output rfft bin and channel.
    Snapshots whose weight is zero at every bin are dropped (no forward pass needed).
    """

    def __init__(self, snapshots: list[nn.Module], weights: torch.Tensor):
        super().__init__()
        keep = [s for s in range(len(snapshots)) if float(weights[..., s].max()) > 1e-6]
        self.snapshots = nn.ModuleList([snapshots[s] for s in keep])
        w = weights[..., keep]
        self.register_buffer("w", w / w.sum(dim=-1, keepdim=True))

    @torch.no_grad()
    def forward(self, x: torch.Tensor, update_memory: bool = False) -> dict:
        preds = torch.stack([m(x, update_memory=False)["y_pred"] for m in self.snapshots], dim=1)
        spec = torch.fft.rfft(preds, dim=2)                                     # (B, S, F, C)
        out = torch.einsum("fcs,bsfc->bfc", self.w.to(spec.dtype), spec)
        return {"y_pred": torch.fft.irfft(out, n=preds.shape[2], dim=1)}


class SnapshotRouter:
    """Snapshots + recognition sets + held-out case segments; evaluates prediction rules."""

    def __init__(self, eval_sets: list, target_indices: list[int] | None, device: str,
                 batch_size: int, k: int = 10, n_recognition: int = 200):
        self.eval_sets = eval_sets
        self.device = device
        self.bs = batch_size
        self.k = k
        self.n_rec = n_recognition
        _, y0 = dataset_windows(eval_sets[0])
        self.target_idx = list(target_indices) if target_indices is not None else list(range(y0.shape[-1]))
        self.snapshots: list[nn.Module] = []
        self.case_sets: list = []
        self._eval_pred: list[list[torch.Tensor]] = []        # [snapshot][regime] -> (n, H, C)
        self._eval_y = [dataset_windows(ds)[1].clone() for ds in eval_sets]
        self._eval_feat = [self._features(dataset_windows(ds)[0]) for ds in eval_sets]
        self._case_pred: dict[tuple[int, int], torch.Tensor] = {}  # (snapshot, regime) -> (m, H, C)
        self._case_y: list[torch.Tensor] = []
        self._case_feat: list[torch.Tensor] = []
        self._rec_feat: list[torch.Tensor] = []
        self.diag: list[dict] = []
        self.update_seconds: list[float] = []   # knowledge-base update cost at each boundary

    # ------------------------------------------------------------------ helpers
    @torch.no_grad()
    def _predict_x(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        out = [model(x[i:i + self.bs].contiguous().to(self.device), update_memory=False)["y_pred"].float().cpu()
               for i in range(0, x.shape[0], self.bs)]
        return torch.cat(out)

    @torch.no_grad()
    def _features(self, x: torch.Tensor) -> torch.Tensor:
        f = [window_features(x[i:i + 1024].contiguous().to(self.device), self.target_idx)[0].cpu()
             for i in range(0, x.shape[0], 1024)]
        if not hasattr(self, "_groups"):
            self._groups = torch.tensor(window_features(x[:2].contiguous().to(self.device), self.target_idx)[1])
        return torch.cat(f)

    def _batched(self, err: torch.Tensor) -> float:
        """Mean of per-batch means, exactly as evaluate_all_tasks averages."""
        return float(torch.stack([c.mean() for c in err.split(self.bs)]).mean())

    def _metrics(self, pred: torch.Tensor, y: torch.Tensor) -> dict:
        return {"mse": self._batched(((pred - y) ** 2).mean(dim=(1, 2))),
                "mae": self._batched((pred - y).abs().mean(dim=(1, 2)))}

    def _knn(self, f: torch.Tensor, ref: torch.Tensor, labels: torch.Tensor, n_cls: int):
        """k nearest reference windows under group-balanced z-scored features: (indices, votes)."""
        mu, sd = ref.mean(dim=0), ref.std(dim=0) + 1e-6
        w = (1.0 / torch.bincount(self._groups).float()[self._groups]).sqrt()
        k = min(self.k, ref.shape[0])
        idx = torch.cdist((f - mu) / sd * w, (ref - mu) / sd * w).topk(k, largest=False).indices
        votes = torch.zeros(f.shape[0], n_cls).scatter_add_(1, labels[idx], torch.ones(f.shape[0], k))
        return idx, votes

    # ------------------------------------------------------------------ knowledge base
    @torch.no_grad()
    def add(self, model: nn.Module, case_set, train_set, seed: int = 0) -> None:
        """Called at the end of a regime: store the snapshot, case segment and recognition set."""
        cuda = str(self.device).startswith("cuda")
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        # knowledge-base update (the method's own cost): snapshot, case segment, case predictions
        frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
        snap = copy.deepcopy(model).eval()
        if self.snapshots:
            # parameters that were frozen and are unchanged (e.g. CLFD's encoder) are stored once
            prev = dict(self.snapshots[-1].named_parameters())
            for name, p in list(snap.named_parameters()):
                if name in frozen and torch.equal(p, prev[name]):
                    mod, attr = snap, name
                    while "." in attr:
                        head, attr = attr.split(".", 1)
                        mod = getattr(mod, head)
                    setattr(mod, attr, prev[name])
        for p in snap.parameters():
            p.requires_grad_(False)
        self.snapshots.append(snap)
        cx, cy = dataset_windows(case_set)
        self.case_sets.append(case_set)
        self._case_y.append(cy.clone())
        for s, sm in enumerate(self.snapshots):
            for r, cs in enumerate(self.case_sets):
                if (s, r) not in self._case_pred:
                    self._case_pred[(s, r)] = self._predict_x(sm, dataset_windows(cs)[0])
        if cuda:
            torch.cuda.synchronize()
        self.update_seconds.append(time.perf_counter() - t0)
        # bookkeeping for evaluating every rule (not part of any rule's deployment cost)
        self._eval_pred.append([self._predict_x(snap, dataset_windows(ds)[0]) for ds in self.eval_sets])
        self._case_feat.append(self._features(cx))
        tx, _ = dataset_windows(train_set)
        g = torch.Generator().manual_seed(seed + len(self._rec_feat))
        pick = torch.randperm(tx.shape[0], generator=g)[: self.n_rec]
        self._rec_feat.append(self._features(tx[pick]))

    def memory_bytes(self) -> dict:
        seen, snap = set(), 0
        for m in self.snapshots:   # count every stored tensor once (shared frozen parameters)
            for t in list(m.parameters()) + list(m.buffers()):
                if t.data_ptr() not in seen:
                    seen.add(t.data_ptr())
                    snap += t.numel() * t.element_size()
        cases = sum(cs.input_data.numel() * 4 + cs.target_data.numel() * 4 for cs in self.case_sets)
        rec = sum(f.numel() * 4 for f in self._rec_feat)
        preds = sum(p.numel() * 4 for p in self._case_pred.values())
        return {"snapshots": snap, "cases": cases, "case_predictions": preds, "recognition": rec}

    @torch.no_grad()
    def deployable(self) -> tuple[SpectralStack, float]:
        """The selected rule (sstack_global) as one module, and the time to fit its weights."""
        t0 = time.perf_counter()
        S = len(self.snapshots)
        Gf = []
        for r in range(len(self.case_sets)):
            e = torch.stack([self._case_pred[(s, r)] for s in range(S)], dim=1) - self._case_y[r][:, None]
            c = torch.fft.rfft(e, dim=2)
            Gf.append(torch.einsum("nsfc,ntfc->fcst", c, c.conj()).real / c.shape[0])
        w = simplex_weights(torch.stack(Gf).mean(dim=0))
        return SpectralStack(self.snapshots, w).to(self.device), time.perf_counter() - t0

    def dump(self, path: str) -> None:
        """All snapshot predictions, targets and features, for offline analysis of new rules."""
        arrs = {"groups": self._groups.numpy()}
        for s, per in enumerate(self._eval_pred):
            for j, p in enumerate(per):
                arrs[f"eval_pred_s{s}_r{j}"] = p.numpy()
        for j, (y, f) in enumerate(zip(self._eval_y, self._eval_feat)):
            arrs[f"eval_y_r{j}"] = y.numpy()
            arrs[f"eval_feat_r{j}"] = f.half().numpy()
        for (s, r), p in self._case_pred.items():
            arrs[f"case_pred_s{s}_r{r}"] = p.numpy()
        for r, (y, f, fr) in enumerate(zip(self._case_y, self._case_feat, self._rec_feat)):
            arrs[f"case_y_r{r}"] = y.numpy()
            arrs[f"case_feat_r{r}"] = f.half().numpy()
            arrs[f"rec_feat_r{r}"] = fr.half().numpy()
        np.savez_compressed(path, **arrs)

    @torch.no_grad()
    def _soup(self) -> nn.Module:
        soup = copy.deepcopy(self.snapshots[-1])
        sd = soup.state_dict()
        for name, t in sd.items():
            if t.is_floating_point():
                sd[name] = torch.stack([m.state_dict()[name] for m in self.snapshots]).mean(dim=0)
        soup.load_state_dict(sd)
        return soup.eval()

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate(self) -> dict[str, list[dict]]:
        S = len(self.snapshots)
        R = len(self.case_sets)
        final = S - 1
        # per regime: case errors of every snapshot, (m_r, S, H, C)
        cerr = [torch.stack([self._case_pred[(s, r)] for s in range(S)], dim=1) - self._case_y[r][:, None]
                for r in range(R)]
        E = torch.stack([(e ** 2).mean(dim=(0, 2, 3)) for e in cerr])            # (R, S) case MSE
        best = E.argmin(dim=1)                                                   # (R,)
        cspec = [torch.fft.rfft(e, dim=2) for e in cerr]                         # (m_r, S, F, C)
        P = torch.stack([(c.abs() ** 2).mean(dim=0) for c in cspec])             # (R, S, F, C)
        bin_sel = P.argmin(dim=1)                                                # (R, F, C)
        glob_sel = P.mean(dim=0).argmin(dim=0)                                   # (F, C)
        # convex combination weights (time domain and per frequency bin)
        G = torch.stack([torch.einsum("nshc,nthc->st", e, e) / e[:, 0].numel() for e in cerr])   # (R, S, S)
        Gf = torch.stack([torch.einsum("nsfc,ntfc->fcst", c, c.conj()).real / c.shape[0] for c in cspec])
        a_reg, a_glob = simplex_weights(G), simplex_weights(G.mean(dim=0))       # (R, S), (S,)
        af_reg, af_glob = simplex_weights(Gf), simplex_weights(Gf.mean(dim=0))   # (R, F, C, S), (F, C, S)

        cf = torch.cat(self._case_feat)
        cl = torch.cat([torch.full((f.shape[0],), r) for r, f in enumerate(self._case_feat)])
        rf = torch.cat(self._rec_feat)
        rl = torch.cat([torch.full((f.shape[0],), r) for r, f in enumerate(self._rec_feat)])
        cerr_w = torch.cat([(e ** 2).mean(dim=(2, 3)) for e in cerr])            # (Ncase, S)
        # diagonal discriminant on case windows
        mu, sd = cf.mean(dim=0), cf.std(dim=0) + 1e-6
        wg = 1.0 / torch.bincount(self._groups).float()[self._groups]
        cz = (cf - mu) / sd
        cent = torch.stack([cz[cl == r].mean(dim=0) for r in range(R)])
        wvar = torch.stack([((cz[cl == r] - cent[r]) ** 2).mean(dim=0) for r in range(R)]).mean(dim=0) + 1e-6
        # causal ensembles: snapshot s has seen regime r iff s >= r
        causal = torch.stack([(torch.arange(S) >= r).float() for r in range(R)])
        causal = causal / causal.sum(dim=1, keepdim=True)                        # (R, S)
        soup = self._soup()

        rows = {v: [] for v in VARIANTS}
        acc = {"knn": 0, "dlda": 0, "buf": 0}
        n_known, to_final = 0, 0
        for j, (y, f) in enumerate(zip(self._eval_y, self._eval_feat)):
            preds = torch.stack([self._eval_pred[s][j] for s in range(S)], dim=1)   # (n, S, H, C)
            n, H = y.shape[0], y.shape[1]
            spec = torch.fft.rfft(preds, dim=2)                                  # (n, S, F, C)

            def pick(s):                                                         # per-window snapshot
                return preds[torch.arange(n), s]

            def mix(a):                                                          # a: (n, S)
                return torch.einsum("ns,nshc->nhc", a, preds)

            def fuse(sel):                                                       # sel: (n, F, C)
                return torch.fft.irfft(torch.gather(spec, 1, sel[:, None]).squeeze(1), n=H, dim=1)

            def smix(a):                                                         # a: (n, F, C, S)
                return torch.fft.irfft(torch.einsum("nfcs,nsfc->nfc", a.to(spec.dtype), spec), n=H, dim=1)

            nn_idx, votes = self._knn(f, cf, cl, R)
            r_knn = votes.argmax(dim=1)
            r_buf = self._knn(f, rf, rl, R)[1].argmax(dim=1)
            z = (f - mu) / sd
            r_dlda = (wg * (z[:, None, :] - cent[None]) ** 2 / wvar).sum(dim=-1).argmin(dim=1)
            soft_w = torch.zeros(n, S).index_add_(1, best, votes / votes.sum(dim=1, keepdim=True))
            s_err = cerr_w[nn_idx].mean(dim=1).argmin(dim=1)
            known = j < R
            if known:
                acc["knn"] += int((r_knn == j).sum())
                acc["dlda"] += int((r_dlda == j).sum())
                acc["buf"] += int((r_buf == j).sum())
                n_known += n
                to_final += int((best[r_knn] == final).sum())
                rj = torch.full((n,), j)
                ev = torch.tensor([self._metrics(preds[:, s], y)["mse"] for s in range(S)])
                pe = (torch.fft.rfft(preds - y[:, None], dim=2).abs() ** 2).mean(dim=0)   # (S, F, C)
            out = {
                "final": preds[:, final],
                "uniform": preds.mean(dim=1),
                "soup": self._predict_x(soup, dataset_windows(self.eval_sets[j])[0]),
                "causal_buf": mix(causal[r_buf]),
                "route_knn": pick(best[r_knn]),
                "route_knn_soft": mix(soft_w),
                "route_dlda": pick(best[r_dlda]),
                "knn_err": pick(s_err),
                "fuse_knn": fuse(bin_sel[r_knn]),
                "fuse_dlda": fuse(bin_sel[r_dlda]),
                "fuse_global": fuse(glob_sel.expand(n, -1, -1)),
                "stack_global": mix(a_glob.expand(n, -1)),
                "stack_knn": mix(a_reg[r_knn]),
                "sstack_global": smix(af_glob.expand(n, -1, -1, -1)),
                "sstack_knn": smix(af_reg[r_knn]),
                # diagnostics (use evaluation labels); fall back to the final snapshot for unseen regimes
                "known": pick(best[rj]) if known else preds[:, final],
                "oracle": preds[:, int(ev.argmin())] if known else preds[:, final],
                "fuse_known": fuse(bin_sel[rj]) if known else preds[:, final],
                "fuse_oracle": fuse(pe.argmin(dim=0).expand(n, -1, -1)) if known else preds[:, final],
                "causal_known": mix(causal[rj]) if known else preds[:, final],
                "stack_known": mix(a_reg[rj]) if known else preds[:, final],
                "sstack_known": smix(af_reg[rj]) if known else preds[:, final],
            }
            for v in VARIANTS:
                rows[v].append(self._metrics(out[v], y))

        self.diag.append({
            "stage": S - 1,
            "case_mse": E.tolist(),
            "selected": best.tolist(),
            "bin_selected": bin_sel[..., 0].tolist(),
            "stack_weights": a_reg.tolist(),
            "stack_global_weights": a_glob.tolist(),
            "regime_acc_knn": acc["knn"] / max(n_known, 1),
            "regime_acc_dlda": acc["dlda"] / max(n_known, 1),
            "regime_acc_buf": acc["buf"] / max(n_known, 1),
            "frac_final_knn": to_final / max(n_known, 1),
        })
        return rows
