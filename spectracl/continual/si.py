from __future__ import annotations

"""
Synaptic Intelligence (SI) continual learning regularizer.

Reference:
Zenke et al., "Continual Learning Through Synaptic Intelligence", ICML 2017.
"""

import torch
import torch.nn as nn


class SI:
    """Tracks parameter importance online within a regime and penalizes drift."""

    def __init__(self, lambda_si: float = 1.0, epsilon: float = 0.1):
        self.lambda_si = lambda_si
        self.epsilon = epsilon

        self.omega: dict[str, torch.Tensor] = {}
        self.theta_star: dict[str, torch.Tensor] = {}

        self._small_omega: dict[str, torch.Tensor] = {}
        self._theta_prev: dict[str, torch.Tensor] = {}
        self._theta_start: dict[str, torch.Tensor] = {}
        self._registered = False

    @property
    def is_registered(self) -> bool:
        return self._registered

    @torch.no_grad()
    def begin_regime(self, model: nn.Module) -> None:
        """Initialize SI path tracking at the start of each regime."""
        self._small_omega = {}
        self._theta_prev = {}
        self._theta_start = {}

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            p_detached = p.detach().clone()
            self._small_omega[name] = torch.zeros_like(p_detached)
            self._theta_prev[name] = p_detached.clone()
            self._theta_start[name] = p_detached.clone()

    @torch.no_grad()
    def update_after_step(self, model: nn.Module) -> None:
        """Accumulate contribution to parameter importance after every optimizer step."""
        if not self._theta_prev:
            return

        for name, p in model.named_parameters():
            if not p.requires_grad or p.grad is None or name not in self._theta_prev:
                continue

            theta_new = p.detach()
            delta = theta_new - self._theta_prev[name]
            self._small_omega[name] += -p.grad.detach() * delta
            self._theta_prev[name] = theta_new.clone()

    @torch.no_grad()
    def end_regime(self, model: nn.Module) -> None:
        """Consolidate SI importance and store anchor parameters."""
        for name, p in model.named_parameters():
            if not p.requires_grad or name not in self._small_omega:
                continue

            theta_curr = p.detach()
            total_delta = theta_curr - self._theta_start[name]
            omega_add = self._small_omega[name] / (total_delta.pow(2) + self.epsilon)
            omega_add = torch.relu(omega_add)

            if name not in self.omega:
                self.omega[name] = omega_add.clone()
            else:
                self.omega[name] = self.omega[name] + omega_add

            self.theta_star[name] = theta_curr.clone()

        self._registered = True

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute SI penalty term for current parameters."""
        if not self._registered:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, p in model.named_parameters():
            if not p.requires_grad or name not in self.omega:
                continue
            omega = self.omega[name].to(p.device)
            theta_ref = self.theta_star[name].to(p.device)
            loss = loss + (omega * (p - theta_ref).pow(2)).sum()

        return self.lambda_si * loss
