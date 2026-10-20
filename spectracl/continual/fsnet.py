from __future__ import annotations

"""
FSNet (Fast and Slow Learning Network) for SAPHIR-TS framework.

Adapted from: Pham et al., "Learning Fast and Slow for Online Time Series
Forecasting", ICLR 2023.

Core components:
    1. Per-layer Adapter -- maintains gradient EMA per layer to model temporal
       smoothness; computes adaptation coefficients that blend fast (current)
       and slow (EMA) knowledge.
    2. Associative Memory -- stores adaptation coefficients for recurring
       patterns; triggered on substantial representation change (measured
       by cosine similarity between two gradient EMAs).

This implementation adapts FSNet's online (batch=1) mechanism to work in
the regime-based continual learning setup used by SAPHIR-TS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from copy import deepcopy


class AssociativeMemory:
    """
    Sparse associative memory that stores and retrieves adaptation
    coefficients keyed by gradient signatures.

    Stores (key, value) pairs where:
        key   = gradient EMA signature (flattened, normalized)
        value = per-layer adaptation coefficients
    """

    def __init__(self, capacity: int = 32, key_dim: int = 256):
        self.capacity = capacity
        self.key_dim = key_dim
        self.keys: list[torch.Tensor] = []
        self.values: list[dict[str, torch.Tensor]] = []

    @property
    def size(self) -> int:
        return len(self.keys)

    @torch.no_grad()
    def write(self, key: torch.Tensor, value: dict[str, torch.Tensor]) -> None:
        """Store a (key, value) pair. Evicts oldest if at capacity."""
        key_norm = F.normalize(key.flatten().unsqueeze(0), dim=-1).squeeze(0)

        # Check for near-duplicate
        if self.size > 0:
            stored = torch.stack(self.keys)
            sim = F.cosine_similarity(key_norm.unsqueeze(0), stored, dim=-1)
            if sim.max().item() > 0.95:
                # Update existing entry
                idx = sim.argmax().item()
                for n in value:
                    self.values[idx][n] = value[n].clone()
                return

        if self.size >= self.capacity:
            self.keys.pop(0)
            self.values.pop(0)

        self.keys.append(key_norm.clone())
        self.values.append({n: v.clone() for n, v in value.items()})

    @torch.no_grad()
    def read(self, query: torch.Tensor) -> dict[str, torch.Tensor] | None:
        """Retrieve the most similar stored adaptation coefficients."""
        if self.size == 0:
            return None

        q_norm = F.normalize(query.flatten().unsqueeze(0), dim=-1).squeeze(0)
        stored = torch.stack(self.keys)
        sim = F.cosine_similarity(q_norm.unsqueeze(0), stored, dim=-1)

        best_sim, best_idx = sim.max(dim=0)
        if best_sim.item() > 0.5:
            return self.values[best_idx.item()]
        return None


class FSNetAdapter:
    """
    FSNet adapter that wraps a model and optimizer.

    Maintains per-layer gradient EMA (slow learner) and uses it to
    compute adaptation coefficients that accelerate learning on
    both new and recurring patterns.

    Usage:
        adapter = FSNetAdapter(model, gamma=0.9, ...)

        # In training loop:
        loss.backward()
        adapter.step(model, optimizer)  # replaces optimizer.step()
    """

    def __init__(
        self,
        model: nn.Module,
        gamma: float = 0.9,
        gamma_prime: float = 0.99,
        memory_capacity: int = 32,
        trigger_threshold: float = 0.1,
        adapt_lr: float = 0.01,
    ):
        self.gamma = gamma           # EMA coefficient for fast gradient
        self.gamma_prime = gamma_prime  # EMA coefficient for slow gradient (drift detection)
        self.trigger_threshold = trigger_threshold
        self.adapt_lr = adapt_lr

        # Per-layer gradient EMAs
        self.grad_ema: dict[str, torch.Tensor] = {}
        self.grad_ema_slow: dict[str, torch.Tensor] = {}

        # Per-layer adaptation coefficients
        self.adapt_coeff: dict[str, torch.Tensor] = {}

        # Associative memory
        self.memory = AssociativeMemory(capacity=memory_capacity)

        self._initialized = False
        self._step_count = 0

    def _initialize(self, model: nn.Module) -> None:
        """Initialize EMAs and adaptation coefficients from model parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_ema[name] = torch.zeros_like(param.data)
                self.grad_ema_slow[name] = torch.zeros_like(param.data)
                self.adapt_coeff[name] = torch.zeros_like(param.data)
        self._initialized = True

    def _compute_gradient_key(self) -> torch.Tensor:
        """Compute a compact gradient signature for memory lookup."""
        grads = []
        for name in sorted(self.grad_ema.keys()):
            g = self.grad_ema[name]
            # Use per-layer norm as compact signature
            grads.append(g.flatten()[:64] if g.numel() > 64 else g.flatten())
        return torch.cat(grads)[:256]

    def _check_drift(self) -> bool:
        """Check if a substantial representation change occurred."""
        if not self.grad_ema or not self.grad_ema_slow:
            return False

        # Compute cosine similarity between fast and slow EMAs
        fast_flat = torch.cat([self.grad_ema[n].flatten() for n in sorted(self.grad_ema.keys())])
        slow_flat = torch.cat([self.grad_ema_slow[n].flatten() for n in sorted(self.grad_ema_slow.keys())])

        if fast_flat.norm() < 1e-8 or slow_flat.norm() < 1e-8:
            return False

        cos_sim = F.cosine_similarity(fast_flat.unsqueeze(0), slow_flat.unsqueeze(0)).item()
        return cos_sim < (1.0 - self.trigger_threshold)

    @torch.no_grad()
    def step(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        """
        Perform adapted gradient step.

        1. Update gradient EMAs
        2. Compute adaptation coefficients
        3. Check for drift -> interact with memory
        4. Apply adapted gradients
        """
        if not self._initialized:
            self._initialize(model)

        self._step_count += 1

        # Update gradient EMAs and adaptation coefficients
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                g = param.grad.data

                if name not in self.grad_ema:
                    self.grad_ema[name] = torch.zeros_like(g)
                    self.grad_ema_slow[name] = torch.zeros_like(g)
                    self.adapt_coeff[name] = torch.zeros_like(g)

                # Fast EMA: captures recent gradient trends
                self.grad_ema[name] = self.gamma * self.grad_ema[name] + (1 - self.gamma) * g

                # Slow EMA: for drift detection
                self.grad_ema_slow[name] = self.gamma_prime * self.grad_ema_slow[name] + (1 - self.gamma_prime) * g

                # Adaptation coefficient: difference between current and EMA
                self.adapt_coeff[name] = self.adapt_lr * self.grad_ema[name]

        # Check for drift and interact with memory
        if self._step_count % 10 == 0:  # Don't check every step (expensive)
            drift = self._check_drift()
            if drift:
                # Store current adaptation in memory
                key = self._compute_gradient_key()
                self.memory.write(key, {n: v.clone() for n, v in self.adapt_coeff.items()})
            else:
                # Try to retrieve from memory
                key = self._compute_gradient_key()
                retrieved = self.memory.read(key)
                if retrieved is not None:
                    # Blend retrieved coefficients with current
                    for name in self.adapt_coeff:
                        if name in retrieved:
                            self.adapt_coeff[name] = (
                                0.5 * self.adapt_coeff[name] + 0.5 * retrieved[name].to(self.adapt_coeff[name].device)
                            )

        # Apply adapted gradients: add adaptation coefficient to gradients
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None and name in self.adapt_coeff:
                param.grad.data = param.grad.data + self.adapt_coeff[name]

        # Standard optimizer step with modified gradients
        optimizer.step()

    def end_regime(self, model: nn.Module) -> None:
        """
        Called at the end of each regime.
        Store final adaptation state in memory for future retrieval.
        """
        if self._initialized and self.grad_ema:
            key = self._compute_gradient_key()
            self.memory.write(key, {n: v.clone() for n, v in self.adapt_coeff.items()})
