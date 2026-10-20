from __future__ import annotations

"""
Maximally Interfered Retrieval (MIR) baseline for continual forecasting.

Aljundi et al., "Online Continual Learning with Maximally Interfered Retrieval",
NeurIPS 2019. At every step, a candidate set is drawn from the replay buffer; a
virtual SGD step on the current batch is simulated, and the candidates whose loss
would increase most under that step are replayed.
"""

import torch
import torch.nn.functional as F
from torch.func import functional_call

from spectracl.continual.replay import ReplayBuffer


class MIRReplay:
    def __init__(self, capacity: int = 500, n_candidates: int = 160, virtual_lr: float = 1e-3,
                 weight: float = 1.0):
        self.buffer = ReplayBuffer(capacity=capacity)
        self.n_candidates = n_candidates
        self.virtual_lr = virtual_lr
        self.weight = weight

    @property
    def size(self) -> int:
        return self.buffer.size

    def replay_loss(self, model, current_loss: torch.Tensor, batch_size: int, device: str) -> torch.Tensor | None:
        """Select the most interfered buffer samples and return their replay loss."""
        if self.buffer.size == 0:
            return None
        cand = self.buffer.sample(min(self.n_candidates, self.buffer.size))
        xc, yc = cand[0].to(device), cand[1].to(device)

        names, params = zip(*[(n, p) for n, p in model.named_parameters() if p.requires_grad])
        grads = torch.autograd.grad(current_loss, params, retain_graph=True, allow_unused=True)
        with torch.no_grad():
            virtual = {n: (p - self.virtual_lr * g) if g is not None else p
                       for n, p, g in zip(names, params, grads)}
            before = (model(xc)["y_pred"] - yc).pow(2).mean(dim=(1, 2))
            after = (functional_call(model, virtual, (xc,))["y_pred"] - yc).pow(2).mean(dim=(1, 2))
            top = torch.topk(after - before, k=min(batch_size, xc.shape[0])).indices

        return self.weight * F.mse_loss(model(xc[top])["y_pred"], yc[top])
