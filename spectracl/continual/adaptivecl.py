from __future__ import annotations

"""
Adaptive continual learning baseline (Wu et al., TII 2024) adapted to this codebase.

Implements the paper's core continual objective with a dual-memory design:
1) Main memory buffer M storing (x, y, z_hint) from past tasks
2) Soft memory buffer S_M storing low-error current-task samples
3) Hint regularization against previous model encoder output on current data

The training objective used by the runner is:

    L = L_cur(x_all, y_all)
        + alpha * ||f(x_mem) - y_mem||_2^2
        + beta  * ||h(x_mem) - z_mem||_2^2
        + gamma * ||h_t(x_cur) - h_{t-1}(x_cur)||_2^2

where:
- f(.) is the forecasting model output
- h(.) is the intermediate representation (model output key "h")
- x_all/y_all combines current batch with soft-memory samples

Note:
- This implementation focuses on the continual-learning mechanism and is backbone-agnostic.
- The paper's TimeRelu/STGCN architecture specifics are not required for this runner.
"""

from copy import deepcopy
import torch
import torch.nn.functional as F


class AdaptiveCLBuffer:
    """Dual-memory controller for adaptive continual learning.

    Main memory M uses reservoir-like random retention under a fixed budget.
    Soft memory S_M keeps low-loss samples from recent/current optimization.
    """

    def __init__(
        self,
        capacity: int = 500,
        soft_capacity: int = 64,
        memory_batch_size: int = 32,
        soft_batch_size: int = 16,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 0.5,
    ):
        self.capacity = int(capacity)
        self.soft_capacity = int(soft_capacity)
        self.memory_batch_size = int(memory_batch_size)
        self.soft_batch_size = int(soft_batch_size)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

        # Main memory M: past samples + stored hint representations.
        self._x_mem: torch.Tensor | None = None
        self._y_mem: torch.Tensor | None = None
        self._z_mem: torch.Tensor | None = None

        # Soft memory S_M: low-loss current-task samples.
        self._x_soft: torch.Tensor | None = None
        self._y_soft: torch.Tensor | None = None
        self._soft_scores: torch.Tensor | None = None

        # Snapshot model from previous task for hint loss on current samples.
        self._prev_model = None

    @property
    def memory_size(self) -> int:
        return 0 if self._x_mem is None else int(self._x_mem.shape[0])

    @property
    def soft_size(self) -> int:
        return 0 if self._x_soft is None else int(self._x_soft.shape[0])

    @property
    def has_previous_model(self) -> bool:
        return self._prev_model is not None

    def _trim_random(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        max_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.shape[0] <= max_size:
            return x, y, z
        idx = torch.randperm(x.shape[0])[:max_size]
        return x[idx], y[idx], z[idx]

    def add_main_batch(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> None:
        """Append a batch into main memory and keep fixed capacity."""
        x = x.detach().cpu()
        y = y.detach().cpu()
        z = z.detach().cpu()

        if self._x_mem is None:
            self._x_mem, self._y_mem, self._z_mem = x, y, z
        else:
            self._x_mem = torch.cat([self._x_mem, x], dim=0)
            self._y_mem = torch.cat([self._y_mem, y], dim=0)
            self._z_mem = torch.cat([self._z_mem, z], dim=0)

        self._x_mem, self._y_mem, self._z_mem = self._trim_random(
            self._x_mem, self._y_mem, self._z_mem, self.capacity
        )

    def add_soft_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        per_sample_loss: torch.Tensor,
    ) -> None:
        """Keep low-loss samples in soft memory under fixed capacity."""
        x = x.detach().cpu()
        y = y.detach().cpu()
        score = per_sample_loss.detach().cpu().flatten()

        if x.shape[0] == 0:
            return

        n_select = min(self.soft_batch_size, x.shape[0])
        keep_idx = torch.argsort(score)[:n_select]
        x_sel = x[keep_idx]
        y_sel = y[keep_idx]
        s_sel = score[keep_idx]

        if self._x_soft is None:
            self._x_soft, self._y_soft, self._soft_scores = x_sel, y_sel, s_sel
        else:
            self._x_soft = torch.cat([self._x_soft, x_sel], dim=0)
            self._y_soft = torch.cat([self._y_soft, y_sel], dim=0)
            self._soft_scores = torch.cat([self._soft_scores, s_sel], dim=0)

        if self._x_soft.shape[0] > self.soft_capacity:
            keep = torch.argsort(self._soft_scores)[: self.soft_capacity]
            self._x_soft = self._x_soft[keep]
            self._y_soft = self._y_soft[keep]
            self._soft_scores = self._soft_scores[keep]

    def sample_main(self, batch_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Sample replay triples (x, y, z_hint) from main memory."""
        if self._x_mem is None or self._x_mem.shape[0] == 0:
            return None
        k = self.memory_batch_size if batch_size is None else int(batch_size)
        k = min(k, self._x_mem.shape[0])
        idx = torch.randperm(self._x_mem.shape[0])[:k]
        return self._x_mem[idx], self._y_mem[idx], self._z_mem[idx]

    def sample_soft(self, batch_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Sample low-loss samples from soft memory."""
        if self._x_soft is None or self._x_soft.shape[0] == 0:
            return None
        k = self.soft_batch_size if batch_size is None else int(batch_size)
        k = min(k, self._x_soft.shape[0])
        idx = torch.randperm(self._x_soft.shape[0])[:k]
        return self._x_soft[idx], self._y_soft[idx]

    @torch.no_grad()
    def snapshot_previous_model(self, model: torch.nn.Module) -> None:
        """Freeze a copy of the model from the just-finished task."""
        self._prev_model = deepcopy(model).eval()
        for p in self._prev_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def compute_hint_target(self, x: torch.Tensor, device: str) -> torch.Tensor | None:
        """Return h_{t-1}(x) for current batch if previous model exists."""
        if self._prev_model is None:
            return None
        self._prev_model = self._prev_model.to(device)
        out_prev = self._prev_model(x, update_memory=False)
        return out_prev["h"].detach()

    def main_regularization(
        self,
        model: torch.nn.Module,
        device: str,
        batch_size: int | None = None,
    ) -> torch.Tensor | None:
        """Compute alpha*y-replay + beta*hint-replay terms on sampled M batch."""
        replay_batch = self.sample_main(batch_size=batch_size)
        if replay_batch is None:
            return None

        x_mem, y_mem, z_mem = replay_batch
        x_mem = x_mem.to(device)
        y_mem = y_mem.to(device)
        z_mem = z_mem.to(device)

        out_mem = model(x_mem, update_memory=False)
        loss_y = F.mse_loss(out_mem["y_pred"], y_mem)
        loss_z = F.mse_loss(out_mem["h"], z_mem)
        return self.alpha * loss_y + self.beta * loss_z
