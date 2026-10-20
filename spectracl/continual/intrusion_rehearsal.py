from __future__ import annotations

"""
Intrusion-aware Frequency Rehearsal (IFR).

Empirical motivation (experiments/synthetic_overlap.py, real_band_forgetting.py):
after training on a new task, a forecaster's error on an old task grows most
at frequencies where the NEW task has energy the old task lacks -- the new
task's oscillations "intrude" into the old task's forecasts. Frequencies the
two tasks share with the same dynamics are effectively rehearsed for free.

IFR keeps a compact spectral knowledge base (one normalised target spectrum
per task). While training on task t, a replayed sample from old task j is
penalised in the frequency domain with

    w_{j,t}(f) = 1 + kappa * max(0, s_t(f) - s_j(f)) / max_f max(0, s_t - s_j)

normalised to mean 1 over bins (DC kept at average weight), so kappa = 0 is
plain frequency-domain replay with uniform weights.

SpectralWeighter is the reusable part (knowledge base + weighted loss) and can
be plugged into any replay-based method; IntrusionRehearsal adds a
task-balanced buffer for the standalone method.
"""

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def target_spectrum(loader: DataLoader, max_batches: int = 200) -> torch.Tensor:
    """Normalised mean amplitude spectrum of a task's H-step targets (DC zeroed)."""
    acc, n = None, 0
    for b, (_, y) in enumerate(loader):
        if b >= max_batches:
            break
        a = torch.fft.rfft(y.float(), dim=1).abs().sum(dim=(0, 2))
        acc = a if acc is None else acc + a
        n += y.shape[0] * y.shape[2]
    s = acc / n
    s[0] = 0.0
    return s / (s.sum() + 1e-12)


def task_ids_from_sizes(sizes: list[int], idx: torch.Tensor) -> torch.Tensor:
    """Map indices into a concatenation of per-task blocks back to task ids."""
    bounds = torch.cumsum(torch.tensor(sizes, device=idx.device), dim=0)
    return torch.bucketize(idx, bounds, right=True)


WEIGHTINGS = ("intrusion", "old_energy", "overlap", "neglect", "shuffled", "pooled")


def _overlap(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    lo, hi = torch.minimum(a, b), torch.maximum(a, b)
    return (lo / (hi + eps)) * (lo / (0.5 * (a + b) + eps))


class SpectralWeighter:
    """Spectral knowledge base and frequency-weighted replay loss.

    `weighting` selects the per-frequency score that is turned into weights by
    w = 1 + kappa * score / max(score), normalised to mean 1 (DC at the mean):
        intrusion   max(0, s_t - s_j)              -- IFR
        old_energy  s_j                            -- the overlap view's design
        overlap     Delta_S(s_j, s_t)              -- the overlap interference score
        neglect     max(0, s_j - s_t)
        shuffled    intrusion scores randomly permuted across bins (control)
        pooled      intrusion weights averaged over all stored tasks (no per-task knowledge)
    """

    def __init__(self, kappa: float = 4.0, weighting: str = "intrusion", seed: int = 0):
        if weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {WEIGHTINGS}, got {weighting!r}")
        self.kappa = kappa
        self.weighting = weighting
        self._gen = torch.Generator().manual_seed(seed)
        self.spectra: list[torch.Tensor] = []
        self._w: torch.Tensor | None = None   # (n_old_tasks, n_freq) for the current task

    def _score(self, s_old: torch.Tensor, s_new: torch.Tensor) -> torch.Tensor:
        if self.weighting in ("intrusion", "shuffled", "pooled"):
            score = torch.clamp(s_new - s_old, min=0)
        elif self.weighting == "old_energy":
            score = s_old.clone()
        elif self.weighting == "overlap":
            score = _overlap(s_old, s_new)
        else:  # neglect
            score = torch.clamp(s_old - s_new, min=0)
        if self.weighting == "shuffled":
            perm = torch.randperm(score.shape[0] - 1, generator=self._gen) + 1
            score = torch.cat([score[:1], score[perm]])
        return score

    @property
    def is_active(self) -> bool:
        return self._w is not None

    def memory_bytes(self) -> int:
        return sum(s.numel() * s.element_size() for s in self.spectra)

    def begin_task(self, train_loader: DataLoader, device: str) -> None:
        if not self.spectra:
            self._w = None
            return
        s_new = target_spectrum(train_loader)
        rows = []
        for s_old in self.spectra:
            score = self._score(s_old, s_new)
            score[0] = 0.0
            w = 1.0 + self.kappa * score / (score.max() + 1e-12)
            w[0] = w[1:].mean()
            rows.append(w / w.mean())
        w_all = torch.stack(rows)
        if self.weighting == "pooled":
            w_all = w_all.mean(dim=0, keepdim=True).expand_as(w_all).clone()
        self._w = w_all.to(device)

    def end_task(self, train_loader: DataLoader) -> None:
        self.spectra.append(target_spectrum(train_loader))

    def loss(self, pred: torch.Tensor, target: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """Intrusion-weighted squared error in the frequency domain, (B, H, C) inputs."""
        h = pred.shape[1]
        spec = torch.fft.rfft(pred - target, dim=1).abs().pow(2) / h      # (B, F, C)
        w = self._w[task_ids.clamp(max=self._w.shape[0] - 1)].unsqueeze(-1)
        return (spec * w).mean()


def fredf_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    """FreDF objective (Wang et al., ICLR 2025): alpha * L1 on the FFT difference along the
    horizon + (1 - alpha) * time-domain MSE, as in the official implementation."""
    freq = (torch.fft.rfft(pred, dim=1) - torch.fft.rfft(target, dim=1)).abs().mean()
    return alpha * freq + (1.0 - alpha) * (pred - target).pow(2).mean()


class IntrusionRehearsal:
    """Task-balanced replay buffer with intrusion-weighted frequency rehearsal.

    With `fredf_alpha` set, the replay error is measured with the FreDF objective instead
    (used for the ER + FreDF baseline; the intrusion weights are then unused).
    """

    # Device-side copies of the buffer kept only for speed; not additional stored state.
    _cache_attrs = ("_x_all", "_y_all", "_tid_all")

    def __init__(self, capacity: int = 500, kappa: float = 4.0, weight: float = 1.0,
                 weighting: str = "intrusion", seed: int = 0, fredf_alpha: float | None = None):
        self.capacity = capacity
        self.weight = weight
        self.fredf_alpha = fredf_alpha
        self.weighter = SpectralWeighter(kappa, weighting=weighting, seed=seed)
        self.x: list[torch.Tensor] = []
        self.y: list[torch.Tensor] = []
        self._x_all: torch.Tensor | None = None
        self._y_all: torch.Tensor | None = None
        self._tid_all: torch.Tensor | None = None

    @property
    def kappa(self) -> float:
        return self.weighter.kappa

    @property
    def size(self) -> int:
        return sum(t.shape[0] for t in self.x)

    @property
    def n_tasks_stored(self) -> int:
        return len(self.x)

    def memory_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.x + self.y) + self.weighter.memory_bytes()

    def begin_task(self, train_loader: DataLoader, device: str) -> None:
        self.weighter.begin_task(train_loader, device)
        if not self.x:
            return
        self._x_all = torch.cat(self.x).to(device)
        self._y_all = torch.cat(self.y).to(device)
        self._tid_all = torch.cat([torch.full((t.shape[0],), i, dtype=torch.long)
                                   for i, t in enumerate(self.x)]).to(device)

    @torch.no_grad()
    def end_task(self, train_loader: DataLoader) -> None:
        """Store the task's spectrum and a random, task-balanced sample of its windows."""
        self.weighter.end_task(train_loader)
        xs, ys = [], []
        for x, y in train_loader:
            xs.append(x)
            ys.append(y)
        xs, ys = torch.cat(xs), torch.cat(ys)
        per_task = self.capacity // (len(self.x) + 1)
        keep = torch.randperm(xs.shape[0])[:per_task]
        self.x.append(xs[keep].cpu())
        self.y.append(ys[keep].cpu())
        for i in range(len(self.x)):
            if self.x[i].shape[0] > per_task:
                k = torch.randperm(self.x[i].shape[0])[:per_task]
                self.x[i], self.y[i] = self.x[i][k], self.y[i][k]

    def replay_loss(self, model, batch_size: int) -> torch.Tensor | None:
        if not self.weighter.is_active or self._x_all is None:
            return None
        idx = torch.randint(0, self._x_all.shape[0], (batch_size,), device=self._x_all.device)
        pred = model(self._x_all[idx], update_memory=False)["y_pred"]
        if self.fredf_alpha is not None:
            return self.weight * fredf_loss(pred, self._y_all[idx], self.fredf_alpha)
        return self.weight * self.weighter.loss(pred, self._y_all[idx], self._tid_all[idx])
