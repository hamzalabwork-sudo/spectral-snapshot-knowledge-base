from __future__ import annotations

"""
Averaged Gradient Episodic Memory (A-GEM) baseline for continual forecasting.

Chaudhry et al., "Efficient Lifelong Learning with A-GEM", ICLR 2019. After the
current-task gradient g is computed, a reference gradient g_ref is computed on a
replay batch; if g conflicts with g_ref (g . g_ref < 0), g is projected so that the
loss on the memory does not increase to first order.
"""

import torch
import torch.nn.functional as F

from spectracl.continual.replay import ReplayBuffer


class AGEM:
    def __init__(self, capacity: int = 500):
        self.buffer = ReplayBuffer(capacity=capacity)

    @property
    def size(self) -> int:
        return self.buffer.size

    def project_gradients(self, model, batch_size: int, device: str) -> None:
        """Call after loss.backward() and before clipping / optimizer.step()."""
        if self.buffer.size == 0:
            return
        params = [p for p in model.parameters() if p.requires_grad]
        g = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]

        xr, yr = self.buffer.sample(batch_size)
        ref_loss = F.mse_loss(model(xr.to(device))["y_pred"], yr.to(device))
        g_ref = torch.autograd.grad(ref_loss, params, allow_unused=True)
        g_ref = [r if r is not None else torch.zeros_like(p) for r, p in zip(g_ref, params)]

        dot = sum((a * b).sum() for a, b in zip(g, g_ref))
        if dot < 0:
            ref_sq = sum((b * b).sum() for b in g_ref).clamp_min(1e-12)
            g = [a - (dot / ref_sq) * b for a, b in zip(g, g_ref)]
        for p, new in zip(params, g):
            p.grad = new
