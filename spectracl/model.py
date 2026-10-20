from __future__ import annotations

"""
MambaForecaster: Clean Mamba backbone for time series forecasting.

Used by all continual learning baselines (finetune, EWC, Replay, etc.)
and as the backbone for SpectraCL. Stripped of atlas/memory/restoration
components that were shown to hurt performance in SAPHIR-TS.
"""

import torch
import torch.nn as nn

from spectracl.backbone.mamba import MambaBackbone
from spectracl.utils.revin import RevIN


class MambaForecaster(nn.Module):
    def __init__(
        self,
        num_features: int = 7,
        output_features: int | None = None,
        seq_len: int = 96,
        pred_len: int = 24,
        # Backbone
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 2,
        dropout: float = 0.1,
        # Normalization
        use_revin: bool = True,
        # Target channel indices (for RevIN reverse with mixed dims)
        target_indices: list[int] | None = None,
    ):
        super().__init__()

        self.pred_len = pred_len
        self.seq_len = seq_len
        self.input_features = num_features
        self.output_features = output_features if output_features is not None else num_features
        self.num_features = self.output_features
        self.d_model = d_model
        self.d_state = d_state
        self.use_revin = use_revin
        self.target_indices = target_indices

        # RevIN
        self.revin = RevIN(num_features=num_features)

        # Mamba Backbone
        self.backbone = MambaBackbone(
            num_features=num_features,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Linear prediction head
        self.prediction_head = nn.Linear(d_model, pred_len * self.output_features)

        # Optional input-conditioned spectral output gate (see spectracl/spectral_gate.py).
        self.spectral_gate = None

    def forward(
        self, x: torch.Tensor, update_memory: bool = False
    ) -> dict[str, torch.Tensor | None]:
        """
        Args:
            x: (B, L, D) input time series
            update_memory: ignored, kept for API compatibility
        Returns:
            dict with 'y_pred': (B, H, D_out) predictions
        """
        B = x.shape[0]

        # RevIN normalize
        if self.use_revin:
            x = self.revin(x)

        # Backbone
        h = self.backbone(x)  # (B, d_model)

        # Predict
        out = self.prediction_head(h)
        y_pred = self.apply_gate(x, out.view(B, self.pred_len, self.output_features))

        # RevIN denormalize
        if self.use_revin:
            if self.output_features == self.input_features:
                y_pred = self.revin.reverse(y_pred)
            elif self.target_indices is not None:
                y_pred = self.revin.reverse(y_pred, target_indices=self.target_indices)

        return {
            "y_pred": y_pred,
            "h": h,
        }

    def apply_gate(self, x_norm: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Apply the spectral output gate (if any) to the normalised forecast."""
        if self.spectral_gate is None:
            return y_pred
        if self.output_features == self.input_features:
            x_target = x_norm
        elif self.target_indices is not None:
            x_target = x_norm[..., self.target_indices]
        else:
            raise ValueError("spectral gate needs target_indices when inputs and outputs differ")
        return self.spectral_gate(x_target, y_pred)

    def get_ssm_params(self) -> dict[str, list[nn.Parameter]]:
        """Return SSM-specific parameters grouped by role.

        Used by SpectraCL for frequency-partitioned protection.
        """
        params = {
            "A_log": [],      # decay rates (controls memory timescale)
            "dt_proj": [],    # discretization step projection
            "B_proj": [],     # state input mapping
            "C_proj": [],     # state output mapping
            "input_proj": [], # input projection
            "out_proj": [],   # output projection
            "other": [],      # everything else
        }

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "A_log" in name:
                params["A_log"].append((name, param))
            elif "dt_proj" in name:
                params["dt_proj"].append((name, param))
            elif "x_proj" in name:
                # x_proj contains B and C projections
                params["B_proj"].append((name, param))
            elif "in_proj" in name:
                params["input_proj"].append((name, param))
            elif "out_proj" in name:
                params["out_proj"].append((name, param))
            else:
                params["other"].append((name, param))

        return params
