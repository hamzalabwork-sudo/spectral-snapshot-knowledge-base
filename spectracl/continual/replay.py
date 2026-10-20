from __future__ import annotations

"""
Experience Replay Buffer for SAPHIR-TS continual learning baseline.

Stores a random subset of (x, y) pairs from each regime.
During training on new regimes, old samples are mixed in to
mitigate forgetting.
"""

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


def cat_cached(owner, name: str, tensors: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate per-task buffer blocks once and reuse the result until the buffer changes.

    Sampling a replay batch every step used to re-concatenate the whole buffer, which is
    slow for large inputs (e.g. 500 x 96 x 862 values on Traffic). Owners must reset
    `owner._cat_cache = {}` whenever their blocks change. The sampled data is identical.
    """
    cache = owner.__dict__.setdefault("_cat_cache", {})
    if name not in cache:
        cache[name] = torch.cat(tensors, dim=0)
    return cache[name]


class ReplayBuffer:
    """
    Simple replay buffer that stores samples from previous regimes.

    Usage:
        replay = ReplayBuffer(capacity=500)

        # After training on regime k:
        replay.add_from_loader(train_loader)

        # During training on regime k+1:
        replay_loader = replay.get_loader(batch_size=32)
    """

    # Speed-only concatenated copy of the buffer (see cat_cached); not additional stored state.
    _cache_attrs = ("_cat_cache",)

    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self.x_buffer: list[torch.Tensor] = []
        self.y_buffer: list[torch.Tensor] = []
        self._total_stored = 0
        self._cat_cache: dict = {}

    @property
    def size(self) -> int:
        return sum(x.shape[0] for x in self.x_buffer)

    def add_from_loader(self, loader: DataLoader, max_samples: int | None = None) -> None:
        """
        Randomly sample from a DataLoader and add to buffer.
        Maintains balanced representation across regimes by
        allocating equal capacity to each regime seen so far.
        """
        # Collect all data from loader
        all_x, all_y = [], []
        for x, y in loader:
            all_x.append(x)
            all_y.append(y)
        all_x = torch.cat(all_x, dim=0)
        all_y = torch.cat(all_y, dim=0)

        # Determine how many samples to keep from this regime
        n_regimes = len(self.x_buffer) + 1
        per_regime = self.capacity // n_regimes
        n_keep = min(per_regime, all_x.shape[0])
        if max_samples is not None:
            n_keep = min(n_keep, max_samples)

        # Random sample
        idx = torch.randperm(all_x.shape[0])[:n_keep]
        self.x_buffer.append(all_x[idx].cpu())
        self.y_buffer.append(all_y[idx].cpu())

        # Trim older regimes to maintain capacity balance
        self._rebalance()

    def _rebalance(self) -> None:
        """Ensure total stored samples <= capacity, balanced across regimes."""
        self._cat_cache = {}
        n_regimes = len(self.x_buffer)
        if n_regimes == 0:
            return

        per_regime = self.capacity // n_regimes
        for i in range(n_regimes):
            if self.x_buffer[i].shape[0] > per_regime:
                idx = torch.randperm(self.x_buffer[i].shape[0])[:per_regime]
                self.x_buffer[i] = self.x_buffer[i][idx]
                self.y_buffer[i] = self.y_buffer[i][idx]

    def sample(self, batch_size: int = 32) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Randomly sample a batch from the buffer."""
        if not self.x_buffer:
            return None
        x = cat_cached(self, "x", self.x_buffer)
        y = cat_cached(self, "y", self.y_buffer)
        idx = torch.randperm(x.shape[0])[:batch_size]
        return x[idx], y[idx]

    def get_loader(self, batch_size: int = 32) -> DataLoader | None:
        """Get a DataLoader over all stored replay samples."""
        if not self.x_buffer:
            return None

        x = torch.cat(self.x_buffer, dim=0)
        y = torch.cat(self.y_buffer, dim=0)
        ds = TensorDataset(x, y)
        return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
