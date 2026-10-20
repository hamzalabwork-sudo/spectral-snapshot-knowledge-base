from __future__ import annotations

"""
Null-Space Gradient Projection for Mamba in Time Series Forecasting.

Adapts the null-space projection approach (OWM, NSCL, Mamba-CL) to the
time series forecasting setting with two key modifications:

1. Exponential covariance decay: Older task feature statistics are
   exponentially down-weighted, handling the natural non-stationarity
   of time series data.

2. Selective constraint application: Only applied to parameters specified
   by the StatePartitioner (Protected params), not all parameters.

Reference:
    - Wang et al., "Training Networks in Null Space of Feature Covariance
      for Continual Learning", CVPR 2021
    - He et al., "Mamba-CL: Optimizing Selective State Space Model in
      Null Space for Continual Learning", arXiv 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional


class NullSpaceProjector:
    """
    Maintains per-layer feature covariance matrices and projects gradients
    into the null space of previous tasks' feature spaces.

    For a linear layer y = Wx, updating W in the null space of previous
    inputs X_old ensures that W_new @ x_old = W_old @ x_old for all
    old inputs x_old (i.e., zero forgetting on old task features).

    Usage:
        projector = NullSpaceProjector(covariance_ema=0.95)

        # After each task:
        projector.update_subspace(model, train_loader, device)

        # During training on next task, after loss.backward():
        projector.project_gradients(model)
        optimizer.step()
    """

    def __init__(
        self,
        covariance_ema: float = 0.95,
        threshold: float = 0.99,
        min_freedom_ratio: float = 0.2,
        protected_param_names: list[str] | None = None,
        covariance_task_decay: float = 1.0,
    ):
        """
        Args:
            covariance_ema: exponential moving average coefficient for
                feature covariance across tasks (higher = more weight to old tasks)
            threshold: singular value threshold for null space computation.
                Captures this fraction of total variance as "used" space.
            min_freedom_ratio: minimum fraction of dimensions kept free for
                new-task learning, regardless of threshold. Prevents the
                projector from collapsing to near-zero rank null space.
            protected_param_names: list of parameter name substrings to protect.
                If None, protects all linear layer weights.
            covariance_task_decay: multiplicative decay applied to all stored
                covariances before incorporating a new task's covariance.
                Values < 1.0 (e.g. 0.7-0.8) cause older task directions to
                fade, preventing freedom from collapsing over many tasks.
        """
        self.covariance_ema = covariance_ema
        self.threshold = threshold
        self.min_freedom_ratio = min_freedom_ratio
        self.protected_param_names = protected_param_names
        self.covariance_task_decay = covariance_task_decay

        # Per-layer: accumulated feature covariance matrix
        self._covariance: dict[str, torch.Tensor] = {}
        # Per-layer: null-space projection matrix
        self._projectors: dict[str, torch.Tensor] = {}
        self._n_tasks = 0

    @property
    def is_active(self) -> bool:
        return self._n_tasks > 0

    def _should_protect(self, name: str) -> bool:
        """Check if a parameter should be protected."""
        if self.protected_param_names is None:
            # Default: protect all linear weights
            return "weight" in name and "norm" not in name
        return any(p in name for p in self.protected_param_names)

    def _get_hook_layers(self, model: nn.Module) -> dict[str, nn.Module]:
        """Find linear layers whose inputs we need to track."""
        layers = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Check if any parameter of this layer should be protected
                for pname, _ in module.named_parameters():
                    full_name = f"{name}.{pname}" if name else pname
                    if self._should_protect(full_name):
                        layers[name] = module
                        break
        return layers

    @torch.no_grad()
    def update_subspace(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: str,
        max_batches: int = 30,
    ) -> None:
        """
        Update feature covariance matrices after training on a task.

        Computes the input feature covariance for each protected linear layer,
        then updates the running covariance with EMA and recomputes the
        null-space projection matrices.

        Args:
            model: trained model
            loader: training data from the just-completed task
            device: compute device
            max_batches: limit batches for efficiency
        """
        model.eval()

        # Collect input features for each linear layer
        hook_layers = self._get_hook_layers(model)
        feature_accum: dict[str, list[torch.Tensor]] = {
            name: [] for name in hook_layers
        }
        handles = []

        def make_hook(layer_name):
            def hook_fn(module, input, output):
                if len(input) > 0 and input[0] is not None:
                    # input[0]: (*, in_features)
                    inp = input[0].detach()
                    # Flatten all dims except last
                    inp = inp.reshape(-1, inp.shape[-1])
                    # Random subsample for memory efficiency
                    if inp.shape[0] > 256:
                        idx = torch.randperm(inp.shape[0], device=inp.device)[:256]
                        inp = inp[idx]
                    feature_accum[layer_name].append(inp.cpu())
            return hook_fn

        for name, module in hook_layers.items():
            h = module.register_forward_hook(make_hook(name))
            handles.append(h)

        # Forward pass to collect features
        for batch_idx, (x, _) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            x = x.to(device)
            model(x, update_memory=False)

        # Remove hooks
        for h in handles:
            h.remove()

        # Compute covariance for each layer
        for name in hook_layers:
            features_list = feature_accum[name]
            if not features_list:
                continue

            features = torch.cat(features_list, dim=0)  # (N, in_features)
            N, D = features.shape

            # Compute uncentered covariance: X^T X / N
            cov = (features.T @ features) / N  # (D, D)

            # Decay old covariances before merging new task
            if name in self._covariance and self.covariance_task_decay < 1.0:
                self._covariance[name] = self._covariance[name] * self.covariance_task_decay

            # Update running covariance with EMA
            if name in self._covariance:
                self._covariance[name] = (
                    self.covariance_ema * self._covariance[name]
                    + (1 - self.covariance_ema) * cov
                )
            else:
                self._covariance[name] = cov

            # Compute null-space projection matrix
            self._projectors[name] = self._compute_null_projector(
                self._covariance[name]
            )

        self._n_tasks += 1

    def _compute_null_projector(self, covariance: torch.Tensor) -> torch.Tensor:
        """
        Compute the null-space projection matrix from feature covariance.

        P = I - U @ U^T  where U spans the feature subspace used by old tasks.

        Args:
            covariance: (D, D) uncentered feature covariance matrix

        Returns:
            projector: (D, D) projection matrix into null space
        """
        D = covariance.shape[0]

        # SVD of covariance
        try:
            U, S, _ = torch.linalg.svd(covariance, full_matrices=True)
        except RuntimeError:
            # Fallback if SVD fails
            return torch.eye(D)

        # Find rank based on threshold (fraction of total variance)
        total_var = S.sum()
        if total_var < 1e-10:
            return torch.eye(D)

        cumvar = S.cumsum(0) / total_var
        rank = (cumvar < self.threshold).sum().item() + 1
        # Freedom floor: ensure at least min_freedom_ratio of dims remain free
        max_rank = D - max(int(D * self.min_freedom_ratio), 1)
        rank = min(rank, max_rank)

        # Null-space projection: P = I - U_r @ U_r^T
        U_r = U[:, :rank]  # (D, rank) -- basis of used subspace
        projector = torch.eye(D) - U_r @ U_r.T

        return projector

    def project_gradients(self, model: nn.Module) -> None:
        """
        Project gradients of protected parameters into the null space
        of previous tasks' feature subspaces.

        Call this AFTER loss.backward() and BEFORE optimizer.step().
        """
        if not self.is_active:
            return

        hook_layers = self._get_hook_layers(model)

        for layer_name, module in hook_layers.items():
            if layer_name not in self._projectors:
                continue

            projector = self._projectors[layer_name]
            device = next(module.parameters()).device
            P = projector.to(device)

            # Project the weight gradient
            # For y = Wx, grad_W has shape (out, in)
            # Project columns: grad_W_new = grad_W @ P
            if module.weight.grad is not None:
                original_norm = module.weight.grad.data.norm()
                module.weight.grad.data = module.weight.grad.data @ P
                projected_norm = module.weight.grad.data.norm()
                if projected_norm > 1e-8:
                    scale = min(original_norm / projected_norm, 1.5)
                    module.weight.grad.data *= scale

    def recompute_projectors(self) -> None:
        """
        Recompute null-space projectors from stored covariances.

        Call after changing min_freedom_ratio or threshold to update
        projectors without re-running forward hooks.
        """
        for name, cov in self._covariance.items():
            self._projectors[name] = self._compute_null_projector(cov)

    def get_projection_stats(self) -> dict[str, dict]:
        """Return statistics about the null-space projections."""
        stats = {}
        for name, proj in self._projectors.items():
            # Effective dimensionality of null space
            eigenvals = torch.linalg.eigvalsh(proj)
            null_dim = (eigenvals > 0.5).sum().item()
            total_dim = proj.shape[0]
            stats[name] = {
                "total_dim": total_dim,
                "null_dim": null_dim,
                "used_dim": total_dim - null_dim,
                "freedom_ratio": null_dim / total_dim,
            }
        return stats
