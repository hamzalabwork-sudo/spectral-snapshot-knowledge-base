from __future__ import annotations

"""
DER++ replay baseline for continual forecasting.

Stores (x, y, y_pred_teacher) from previous regimes.
During future regimes:
- Distill to stored predictions (DER term).
- Supervise with stored labels (replay term).
"""

import torch
from torch.utils.data import DataLoader

from spectracl.continual.intrusion_rehearsal import task_ids_from_sizes
from spectracl.continual.replay import cat_cached


class DERPPBuffer:
    """Balanced regime-wise memory buffer with stored model predictions."""

    # Speed-only concatenated copy of the buffer (see cat_cached); not additional stored state.
    _cache_attrs = ("_cat_cache",)

    def __init__(self, capacity: int = 500, alpha: float = 0.5, beta: float = 0.5):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.x_buffer: list[torch.Tensor] = []
        self.y_buffer: list[torch.Tensor] = []
        self.logits_buffer: list[torch.Tensor] = []
        self._cat_cache: dict = {}

    @property
    def size(self) -> int:
        return sum(x.shape[0] for x in self.x_buffer)

    def _rebalance(self) -> None:
        self._cat_cache = {}
        n_regimes = len(self.x_buffer)
        if n_regimes == 0:
            return

        per_regime = self.capacity // n_regimes
        for i in range(n_regimes):
            n_curr = self.x_buffer[i].shape[0]
            if n_curr <= per_regime:
                continue
            idx = torch.randperm(n_curr)[:per_regime]
            self.x_buffer[i] = self.x_buffer[i][idx]
            self.y_buffer[i] = self.y_buffer[i][idx]
            self.logits_buffer[i] = self.logits_buffer[i][idx]

    @torch.no_grad()
    def add_from_loader(
        self,
        model,
        loader: DataLoader,
        device: str,
        max_samples: int | None = None,
    ) -> None:
        """Add current-regime samples with frozen predictions."""
        model.eval()

        all_x, all_y, all_logits = [], [], []
        for x, y in loader:
            pred = model(x.to(device), update_memory=False)["y_pred"].detach().cpu()
            all_x.append(x.cpu())
            all_y.append(y.cpu())
            all_logits.append(pred)

        if not all_x:
            return

        x_full = torch.cat(all_x, dim=0)
        y_full = torch.cat(all_y, dim=0)
        logits_full = torch.cat(all_logits, dim=0)

        n_regimes = len(self.x_buffer) + 1
        per_regime = max(1, self.capacity // n_regimes)
        n_keep = min(per_regime, x_full.shape[0])
        if max_samples is not None:
            n_keep = min(n_keep, max_samples)

        idx = torch.randperm(x_full.shape[0])[:n_keep]
        self.x_buffer.append(x_full[idx])
        self.y_buffer.append(y_full[idx])
        self.logits_buffer.append(logits_full[idx])
        self._rebalance()

    def sample(self, batch_size: int, return_task_ids: bool = False):
        """Sample replay tuples (x, y_true, y_teacher[, task_ids])."""
        if not self.x_buffer:
            return None

        x = cat_cached(self, "x", self.x_buffer)
        y = cat_cached(self, "y", self.y_buffer)
        logits = cat_cached(self, "logits", self.logits_buffer)

        n = x.shape[0]
        k = min(batch_size, n)
        idx = torch.randperm(n)[:k]
        if return_task_ids:
            tids = task_ids_from_sizes([t.shape[0] for t in self.x_buffer], idx)
            return x[idx], y[idx], logits[idx], tids
        return x[idx], y[idx], logits[idx]
