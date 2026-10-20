from __future__ import annotations

"""
Input-conditioned spectral output gate.

Spectral intrusion means that, after learning a new regime, the forecaster emits that
regime's oscillations for windows of an earlier regime even though those windows contain
little energy at the corresponding frequencies. The gate acts directly on this failure
mode at prediction time: each frequency component of the (normalised) forecast is scaled
by a gate that depends on how strongly the same frequency is present in the input window,

    Y(k) <- Y(k) * sigmoid(a_k * z_k(x) + b_k),   k = 1 .. H/2,

where z_k(x) is the centred log-amplitude of the input window at the frequency of output
bin k. The DC (level) bin is never gated. With two parameters per bin, the gate learns a
regime-agnostic rule ("pass what the input contains") rather than regime-specific content.
No task identity is needed.
"""

import torch
import torch.nn as nn


class SpectralGate(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, init_bias: float = 3.0):
        super().__init__()
        self.seq_len, self.pred_len = seq_len, pred_len
        n_out = pred_len // 2 + 1
        n_in = seq_len // 2 + 1
        # Input rfft bin with the same frequency (cycles per step) as output bin k.
        idx = torch.round(torch.arange(n_out) * seq_len / pred_len).long().clamp(max=n_in - 1)
        self.register_buffer("in_idx", idx)
        self.slope = nn.Parameter(torch.zeros(n_out))
        self.bias = nn.Parameter(torch.full((n_out,), init_bias))  # starts close to identity

    def gates(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, C) normalised input of the forecast channels -> gates (B, F_out, C)."""
        amp = torch.fft.rfft(x, dim=1).abs()[:, self.in_idx]            # (B, F_out, C)
        logamp = torch.log(amp + 1e-6)
        z = logamp - logamp[:, 1:].mean(dim=1, keepdim=True)
        g = torch.sigmoid(self.slope[None, :, None] * z + self.bias[None, :, None])
        g = torch.cat([torch.ones_like(g[:, :1]), g[:, 1:]], dim=1)     # DC passes unchanged
        return g

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Gate forecast y (B, H, C) given normalised input x (B, L, C)."""
        spec = torch.fft.rfft(y, dim=1) * self.gates(x)
        return torch.fft.irfft(spec, n=self.pred_len, dim=1)
