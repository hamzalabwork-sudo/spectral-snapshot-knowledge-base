from __future__ import annotations

"""
Elastic Weight Consolidation (EWC) for SAPHIR-TS.

Prevents catastrophic forgetting by penalizing changes to parameters
that were important for previous regimes, measured by the Fisher
information matrix diagonal.

Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting
in neural networks", PNAS 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from copy import deepcopy


class EWC:
    """
    Elastic Weight Consolidation.

    Usage:
        ewc = EWC(model, lambda_ewc=1000)

        # After training on regime k:
        ewc.register_regime(model, dataloader, device)

        # During training on regime k+1, add penalty to loss:
        loss = task_loss + ewc.penalty(model)
    """

    def __init__(self, lambda_ewc: float = 1000.0):
        self.lambda_ewc = lambda_ewc
        self.params_old: dict[str, torch.Tensor] = {}
        self.fisher: dict[str, torch.Tensor] = {}
        self._registered = False

    @property
    def is_registered(self) -> bool:
        return self._registered

    @torch.no_grad()
    def _compute_fisher(
        self, model: nn.Module, dataloader: DataLoader, device: str, n_samples: int = 200
    ) -> dict[str, torch.Tensor]:
        """
        Estimate diagonal Fisher information matrix using empirical Fisher.

        Args:
            model: trained model
            dataloader: data from the regime just trained on
            device: cuda/cpu
            n_samples: max samples to use for estimation
        """
        model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        count = 0

        # Temporarily enable gradients for Fisher computation
        for x, y in dataloader:
            if count >= n_samples:
                break
            x, y = x.to(device), y.to(device)

            model.zero_grad()
            # Need gradients for Fisher
            with torch.enable_grad():
                out = model(x, update_memory=False)
                loss = F.mse_loss(out["y_pred"], y)
                loss.backward()

            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)

            count += x.shape[0]

        # Average
        for n in fisher:
            fisher[n] /= max(count, 1)

        return fisher

    def register_regime(
        self, model: nn.Module, dataloader: DataLoader, device: str
    ) -> None:
        """
        Call after training on a regime. Stores parameter snapshot
        and computes Fisher information.
        """
        # Compute Fisher for this regime
        new_fisher = self._compute_fisher(model, dataloader, device)

        if not self._registered:
            # First regime -- just store
            self.fisher = new_fisher
        else:
            # Accumulate Fisher across regimes (online EWC)
            for n in self.fisher:
                self.fisher[n] = self.fisher[n] + new_fisher[n]

        # Store parameter snapshot
        self.params_old = {
            n: p.data.clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        self._registered = True

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC penalty: Sum F_i * (theta_i - theta_old_i)^2

        Returns 0 if no regime has been registered yet.
        """
        if not self._registered:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.fisher:
                loss += (self.fisher[n].to(p.device) * (p - self.params_old[n].to(p.device)).pow(2)).sum()

        return self.lambda_ewc * loss
