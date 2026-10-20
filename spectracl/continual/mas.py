from __future__ import annotations

"""
Memory Aware Synapses (MAS) continual learning regularizer.

Reference:
Aljundi et al., "Memory Aware Synapses", ECCV 2018.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class MAS:
    """Estimates parameter importance from output sensitivity and penalizes drift."""

    def __init__(self, lambda_mas: float = 1.0):
        self.lambda_mas = lambda_mas
        self.omega: dict[str, torch.Tensor] = {}
        self.params_old: dict[str, torch.Tensor] = {}
        self._registered = False

    @property
    def is_registered(self) -> bool:
        return self._registered

    def _compute_importance(
        self, model: nn.Module, dataloader: DataLoader, device: str, n_samples: int = 200
    ) -> dict[str, torch.Tensor]:
        model.eval()
        importance = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        count = 0

        # MAS needs gradients wrt parameters of ||f(x)||^2 even in evaluation mode.
        with torch.enable_grad():
            for x, _ in dataloader:
                if count >= n_samples:
                    break
                x = x.to(device)

                model.zero_grad(set_to_none=True)
                out = model(x, update_memory=False)
                loss = out["y_pred"].pow(2).mean()
                loss.backward()

                for n, p in model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        importance[n] += p.grad.detach().abs()

                count += x.shape[0]

        for n in importance:
            importance[n] /= max(count, 1)

        return importance

    def register_regime(self, model: nn.Module, dataloader: DataLoader, device: str) -> None:
        """Compute MAS importance on a finished regime and consolidate."""
        new_omega = self._compute_importance(model, dataloader, device)

        if not self._registered:
            self.omega = new_omega
        else:
            for n in self.omega:
                self.omega[n] = self.omega[n] + new_omega[n]

        self.params_old = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        self._registered = True

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute MAS penalty term for current parameters."""
        if not self._registered:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.omega:
                omega = self.omega[n].to(p.device)
                old = self.params_old[n].to(p.device)
                loss = loss + (omega * (p - old).pow(2)).sum()

        return self.lambda_mas * loss
