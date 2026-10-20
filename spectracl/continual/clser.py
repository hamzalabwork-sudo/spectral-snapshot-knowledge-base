from __future__ import annotations

"""
CLS-ER style continual learning adapter for regression forecasting.

This is an adaptation of the ICLR 2022 CLS-ER idea to time-series regression:
- Episodic replay buffer with balanced per-task storage
- Two EMA teachers: plastic (faster) and stable (slower)
- Consistency regularization on replay samples using the better teacher

For regression, the "better" teacher is selected per replay sample by
comparing MSE to the stored ground truth.
"""

from copy import deepcopy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from spectracl.continual.intrusion_rehearsal import SpectralWeighter, task_ids_from_sizes
from spectracl.continual.replay import cat_cached


class CLSERBuffer:
    """Simple balanced replay buffer for CLS-ER adaptation."""

    # Speed-only concatenated copy of the buffer (see cat_cached); not additional stored state.
    _cache_attrs = ("_cat_cache",)

    def __init__(self, capacity: int = 500):
        self.capacity = int(capacity)
        self.x_buffer: list[torch.Tensor] = []
        self.y_buffer: list[torch.Tensor] = []
        self._cat_cache: dict = {}

    @property
    def size(self) -> int:
        return sum(x.shape[0] for x in self.x_buffer)

    def add_from_loader(self, loader: DataLoader, max_samples: int | None = None) -> None:
        """Add one task worth of data with balanced storage across seen tasks."""
        all_x = []
        all_y = []
        for x, y in loader:
            all_x.append(x)
            all_y.append(y)

        if not all_x:
            return

        x_full = torch.cat(all_x, dim=0)
        y_full = torch.cat(all_y, dim=0)

        n_tasks = len(self.x_buffer) + 1
        per_task = max(1, self.capacity // n_tasks)
        n_keep = min(per_task, x_full.shape[0])
        if max_samples is not None:
            n_keep = min(n_keep, int(max_samples))

        idx = torch.randperm(x_full.shape[0])[:n_keep]
        self.x_buffer.append(x_full[idx].cpu())
        self.y_buffer.append(y_full[idx].cpu())
        self._rebalance()

    def _rebalance(self) -> None:
        self._cat_cache = {}
        n_tasks = len(self.x_buffer)
        if n_tasks == 0:
            return

        per_task = max(1, self.capacity // n_tasks)
        for i in range(n_tasks):
            n_curr = self.x_buffer[i].shape[0]
            if n_curr <= per_task:
                continue
            idx = torch.randperm(n_curr)[:per_task]
            self.x_buffer[i] = self.x_buffer[i][idx]
            self.y_buffer[i] = self.y_buffer[i][idx]

    def sample(self, batch_size: int, return_task_ids: bool = False):
        if not self.x_buffer:
            return None

        x = cat_cached(self, "x", self.x_buffer)
        y = cat_cached(self, "y", self.y_buffer)
        k = min(int(batch_size), x.shape[0])
        idx = torch.randperm(x.shape[0])[:k]
        if return_task_ids:
            return x[idx], y[idx], task_ids_from_sizes([t.shape[0] for t in self.x_buffer], idx)
        return x[idx], y[idx]


class CLSERAdapter:
    """Dual-EMA replay consistency adapter for regression continual learning."""

    def __init__(
        self,
        model,
        buffer_capacity: int = 500,
        reg_weight: float = 0.1,
        plastic_model_alpha: float = 0.99,
        stable_model_alpha: float = 0.999,
        plastic_model_update_freq: float = 0.9,
        stable_model_update_freq: float = 0.7,
    ):
        self.buffer = CLSERBuffer(capacity=buffer_capacity)
        self.reg_weight = float(reg_weight)

        self.plastic_model = deepcopy(model).eval()
        self.stable_model = deepcopy(model).eval()

        for p in self.plastic_model.parameters():
            p.requires_grad_(False)
        for p in self.stable_model.parameters():
            p.requires_grad_(False)

        self.plastic_model_alpha = float(plastic_model_alpha)
        self.stable_model_alpha = float(stable_model_alpha)
        self.plastic_model_update_freq = float(plastic_model_update_freq)
        self.stable_model_update_freq = float(stable_model_update_freq)

        self.global_step = 0

    def _ema_update(self, ema_model, source_model, alpha: float) -> None:
        for p_ema, p_src in zip(ema_model.parameters(), source_model.parameters()):
            p_ema.data.mul_(alpha).add_(p_src.data, alpha=(1.0 - alpha))

    def update_ema_models(self, source_model) -> None:
        """Stochastic EMA updates for plastic and stable semantic memories."""
        self.global_step += 1

        alpha_p = min(1.0 - 1.0 / (self.global_step + 1), self.plastic_model_alpha)
        alpha_s = min(1.0 - 1.0 / (self.global_step + 1), self.stable_model_alpha)

        if torch.rand(1).item() < self.plastic_model_update_freq:
            self._ema_update(self.plastic_model, source_model, alpha_p)

        if torch.rand(1).item() < self.stable_model_update_freq:
            self._ema_update(self.stable_model, source_model, alpha_s)

    @torch.no_grad()
    def _select_teacher_prediction(self, x: torch.Tensor, y: torch.Tensor, device: str) -> torch.Tensor:
        """Per sample, select plastic or stable teacher by lower regression error."""
        x = x.to(device)
        y = y.to(device)

        self.plastic_model = self.plastic_model.to(device)
        self.stable_model = self.stable_model.to(device)

        yp = self.plastic_model(x, update_memory=False)["y_pred"]
        ys = self.stable_model(x, update_memory=False)["y_pred"]

        err_p = ((yp - y) ** 2).mean(dim=(1, 2))
        err_s = ((ys - y) ** 2).mean(dim=(1, 2))
        use_stable = (err_s <= err_p).float().view(-1, 1, 1)

        return use_stable * ys + (1.0 - use_stable) * yp

    def consistency_loss(self, model, batch_size: int, device: str,
                         weighter: SpectralWeighter | None = None) -> torch.Tensor | None:
        """CLS-ER consistency term on replay samples.

        With an active `weighter`, the consistency error is measured with the
        intrusion-weighted frequency-domain loss instead of time-domain MSE.
        """
        use_w = weighter is not None and weighter.is_active
        replay_batch = self.buffer.sample(batch_size, return_task_ids=use_w)
        if replay_batch is None:
            return None

        xr, yr = replay_batch[0].to(device), replay_batch[1].to(device)

        with torch.no_grad():
            teacher = self._select_teacher_prediction(xr, yr, device)

        pred = model(xr, update_memory=False)["y_pred"]
        if use_w:
            return self.reg_weight * weighter.loss(pred, teacher, replay_batch[2].to(device))
        return self.reg_weight * F.mse_loss(pred, teacher)
