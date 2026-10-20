from __future__ import annotations

"""
Frequency-Partitioned State Protection (FPSP) for SpectraCL.

Exploits the heterogeneous memory timescales inherent in Mamba's SSM
state space. The A matrix in Mamba controls how fast each hidden state
dimension decays:
    A_bar = exp(Delta * A)
    |A| close to 0  -> slow decay -> long memory  -> Protected
    |A| large       -> fast decay -> short memory -> Plastic

By protecting only the slow-decaying state dimensions (which encode
long-range temporal patterns most vulnerable to forgetting) and allowing
fast-decaying dimensions to adapt freely, we create a principled
stability-plasticity partition that respects the SSM structure.

This is distinct from Mamba-CL (He et al., 2024) which applies uniform
null-space projection to ALL SSM parameters. FPSP is the first to exploit
the spectral structure of the A matrix for continual learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class StatePartitioner:
    """
    Partitions SSM state dimensions into Protected and Plastic groups
    based on their effective memory timescale (A matrix decay rates).

    Protected dimensions: slow-decaying, encode long-range patterns
        -> Constrained during training (via null-space or EWC)
    Plastic dimensions: fast-decaying, encode local/transient features
        -> Free to adapt for current task

    Usage:
        partitioner = StatePartitioner(protection_ratio=0.5)

        # After each task:
        partitioner.update_partition(model, train_loader, device)

        # Get parameter names to protect:
        protected_names = partitioner.get_protected_param_names(model)
    """

    def __init__(
        self,
        protection_ratio: float = 0.5,
        criterion: str = "decay_rate",
    ):
        """
        Args:
            protection_ratio: fraction of state dims to protect (0-1)
            criterion: how to rank state dimensions
                "decay_rate" -- protect slowest-decaying dims (by A magnitude)
                "fisher_weighted" -- protect dims with high Fisher x slow decay
        """
        self.protection_ratio = protection_ratio
        self.criterion = criterion

        # Per-layer masks: True = Protected, False = Plastic
        self._masks: dict[str, torch.Tensor] = {}
        # Per-layer importance scores
        self._importance: dict[str, torch.Tensor] = {}
        self._n_tasks = 0

    @property
    def is_active(self) -> bool:
        return self._n_tasks > 0

    @torch.no_grad()
    def update_partition(
        self,
        model: nn.Module,
        loader: DataLoader | None = None,
        device: str = "cpu",
    ) -> None:
        """
        Recompute state dimension partitions after a task.

        Analyzes the A_log parameters in each Mamba layer to determine
        which state dimensions have slow vs fast decay rates.

        Args:
            model: trained model
            loader: training data (needed for fisher_weighted criterion)
            device: compute device
        """
        for name, param in model.named_parameters():
            if "A_log" not in name:
                continue

            # A_log has shape (d_inner, d_state) for PureTorchMambaBlock
            # or (d_inner,) for official Mamba
            A_log = param.data.clone()

            # Compute effective decay rate per state dimension
            # A = -exp(A_log), so |A| = exp(A_log)
            # Smaller |A| -> slower decay -> longer memory -> should protect
            decay_rates = A_log.exp()  # higher = faster decay

            if decay_rates.dim() == 2:
                # (d_inner, d_state) -> average across d_inner for per-state-dim score
                decay_per_state = decay_rates.mean(dim=0)  # (d_state,)
            else:
                decay_per_state = decay_rates  # (d_inner,)

            if self.criterion == "fisher_weighted" and loader is not None:
                # Weight by Fisher information (output importance)
                fisher = self._compute_param_fisher(model, name, loader, device)
                if fisher is not None and fisher.shape == decay_per_state.shape:
                    # importance = slow decay (small rate) x high Fisher
                    importance = (1.0 / (decay_per_state + 1e-8)) * fisher
                else:
                    importance = 1.0 / (decay_per_state + 1e-8)
            else:
                # Pure decay rate: protect slowest-decaying (smallest rate)
                importance = 1.0 / (decay_per_state + 1e-8)

            self._importance[name] = importance

            # Create protection mask: top protection_ratio dimensions by importance
            n_dims = importance.numel()
            n_protect = max(1, int(n_dims * self.protection_ratio))
            _, top_indices = importance.topk(n_protect)
            mask = torch.zeros(n_dims, dtype=torch.bool)
            mask[top_indices] = True
            self._masks[name] = mask

        self._n_tasks += 1

    @torch.no_grad()
    def _compute_param_fisher(
        self,
        model: nn.Module,
        a_log_name: str,
        loader: DataLoader,
        device: str,
        max_batches: int = 10,
    ) -> torch.Tensor | None:
        """Compute Fisher information for the A_log parameter."""
        model.eval()

        param = dict(model.named_parameters()).get(a_log_name)
        if param is None:
            return None

        fisher = torch.zeros_like(param.data)
        count = 0

        for batch_idx, (x, y) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            x, y = x.to(device), y.to(device)

            model.zero_grad()
            with torch.enable_grad():
                out = model(x, update_memory=False)
                loss = F.mse_loss(out["y_pred"], y)
                loss.backward()

            if param.grad is not None:
                fisher += param.grad.data.pow(2)
                count += x.shape[0]

        if count > 0:
            fisher /= count

        # Aggregate to per-state-dim if needed
        if fisher.dim() == 2:
            fisher = fisher.mean(dim=0)

        return fisher

    def get_protected_param_names(self, model: nn.Module) -> list[str]:
        """
        Get names of parameters that should be protected.

        For each Mamba layer with a partition, returns the names of
        parameters within that Mamba block: in_proj, out_proj, x_proj.

        Does NOT protect prediction_head or backbone.input_proj:
        - prediction_head needs full plasticity (protected by replay instead)
        - backbone.input_proj has a 1x1 covariance for univariate input,
          making its null space empty (NSP would zero the entire gradient)

        These names are passed to NullSpaceProjector.protected_param_names.
        """
        protected_names = []

        for a_log_name in self._masks:
            # Extract the layer prefix
            # e.g., "backbone.layers.0.mamba.A_log" -> "backbone.layers.0.mamba"
            parts = a_log_name.rsplit(".", 1)
            if len(parts) < 2:
                continue
            layer_prefix = parts[0]

            # Protect all projection weights within the Mamba block:
            # in_proj (maps d_model -> 2*d_inner) -- upstream, changes cascade
            # out_proj (maps d_inner -> d_model)
            # x_proj (contains B, C projections -- maps d_inner -> d_state)
            for pname, _ in model.named_parameters():
                if not pname.startswith(layer_prefix):
                    continue
                if any(k in pname for k in ["in_proj", "out_proj", "x_proj"]):
                    protected_names.append(pname)

        return list(set(protected_names))

    def get_partition_stats(self) -> dict[str, dict]:
        """Return statistics about the current partition."""
        stats = {}
        for name, mask in self._masks.items():
            n_total = mask.numel()
            n_protected = mask.sum().item()
            stats[name] = {
                "total_dims": n_total,
                "protected_dims": n_protected,
                "plastic_dims": n_total - n_protected,
                "protection_ratio": n_protected / n_total,
            }
            if name in self._importance:
                imp = self._importance[name]
                stats[name]["importance_range"] = (
                    imp.min().item(), imp.max().item()
                )
        return stats

    def apply_gradient_mask(self, model: nn.Module) -> None:
        """
        Alternative to null-space: directly zero out gradients of
        Protected A_log dimensions while allowing Plastic dims to update.

        Call AFTER loss.backward() and BEFORE optimizer.step().
        This is a simpler alternative to null-space projection that
        directly prevents A matrix changes for Protected state dimensions.
        """
        if not self.is_active:
            return

        for name, param in model.named_parameters():
            if name in self._masks and param.grad is not None:
                mask = self._masks[name].to(param.device)
                if param.grad.dim() == 2 and mask.dim() == 1:
                    # A_log is (d_inner, d_state), mask is (d_state,)
                    mask = mask.unsqueeze(0).expand_as(param.grad)
                # Zero gradients for Protected dimensions
                param.grad.data[mask] = 0.0
