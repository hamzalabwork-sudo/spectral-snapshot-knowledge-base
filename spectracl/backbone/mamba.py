from __future__ import annotations

"""
Mamba Backbone Encoder for SAPHIR-TS.

Implements a Mamba-based (Selective State Space Model) encoder for
temporal feature extraction. Falls back to a pure-PyTorch SSM
implementation if mamba_ssm is not installed.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Try importing the optimized CUDA Mamba kernel; fall back to pure PyTorch.
# ---------------------------------------------------------------------------
try:
    from mamba_ssm import Mamba as MambaBlock

    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


# ===================================================================
# Pure-PyTorch Mamba block (used when mamba_ssm is not installed)
# ===================================================================
class PureTorchMambaBlock(nn.Module):
    """Minimal selective SSM block in pure PyTorch."""

    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        d_inner = d_model * expand

        # Input projection  ->  (x, z) split
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # Depth-wise causal convolution
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True,
        )

        # SSM parameters (selectivity via input-dependent Delta, B, C)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)  # Delta, B, C
        self.dt_proj = nn.Linear(1, d_inner, bias=True)

        # Log-space initialisation for A
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).expand(d_inner, -1).clone())

        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            (B, L, D)
        """
        B, L, _ = x.shape

        xz = self.in_proj(x)                          # (B, L, 2*d_inner)
        x_branch, z = xz.chunk(2, dim=-1)             # each (B, L, d_inner)

        # Causal conv  (B, d_inner, L)
        x_conv = self.conv1d(x_branch.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Input-dependent SSM parameters
        x_ssm = self.x_proj(x_conv)                   # (B, L, 2*d_state+1)
        dt = F.softplus(self.dt_proj(x_ssm[..., :1]))  # (B, L, d_inner)
        B_param = x_ssm[..., 1:1 + self.d_state]      # (B, L, d_state)
        C_param = x_ssm[..., 1 + self.d_state:]       # (B, L, d_state)

        A = -torch.exp(self.A_log)                     # (d_inner, d_state)

        # Discretised scan (simplified sequential for compatibility)
        y = self._selective_scan(x_conv, dt, A, B_param, C_param)

        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv
        y = y * F.silu(z)
        return self.out_proj(y)

    def _selective_scan(self, x, dt, A, B, C):
        """Sequential selective scan (pure PyTorch, no custom CUDA)."""
        Bat, L, d_inner = x.shape
        d_state = self.d_state

        h = torch.zeros(Bat, d_inner, d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(L):
            dt_t = dt[:, t]                            # (B, d_inner)
            B_t = B[:, t]                              # (B, d_state)
            C_t = C[:, t]                              # (B, d_state)
            x_t = x[:, t]                              # (B, d_inner)

            dA = torch.exp(dt_t.unsqueeze(-1) * A)     # (B, d_inner, d_state)
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1) # (B, d_inner, d_state)

            h = h * dA + dB * x_t.unsqueeze(-1)
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1)   # (B, d_inner)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)             # (B, L, d_inner)


# ===================================================================
# Single Mamba layer (norm -> mamba -> residual)
# ===================================================================
class MambaLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if MAMBA_AVAILABLE:
            self.mamba = MambaBlock(
                d_model=d_model, d_state=d_state,
                d_conv=d_conv, expand=expand,
            )
        else:
            self.mamba = PureTorchMambaBlock(
                d_model=d_model, d_state=d_state,
                d_conv=d_conv, expand=expand,
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mamba(self.norm(x)))


# ===================================================================
# Full Backbone Encoder
# ===================================================================
class MambaBackbone(nn.Module):
    """
    Mamba backbone encoder.

    Input:  X_t in R^(B, L, D)
    Output: h_t in R^(B, d_model)
    """

    def __init__(self, num_features: int, d_model: int = 128,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        # Project raw features to model dimension
        self.input_proj = nn.Linear(num_features, d_model)

        # Stacked Mamba layers
        self.layers = nn.ModuleList([
            MambaLayer(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)  raw time-series window
        Returns:
            h: (B, d_model)  pooled temporal representation
        """
        h = self.input_proj(x)          # (B, L, d_model)

        for layer in self.layers:
            h = layer(h)                # (B, L, d_model)

        h = self.norm(h)
        h = h.mean(dim=1)              # global average pool -> (B, d_model)
        return h
