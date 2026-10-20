from __future__ import annotations

"""
Time-series decomposition utilities for the Seasonal Atlas Encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAvgDecomposition(nn.Module):
    """
    Decompose a time-series into trend and seasonal components
    using a causal moving average filter.

    x = trend + seasonal
    trend = MovingAvg(x)
    seasonal = x - trend
    """

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        if kernel_size % 2 == 0:
            self.kernel_size += 1  # ensure odd for symmetric padding

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L, D)
        Returns:
            trend:    (B, L, D)
            seasonal: (B, L, D)
        """
        # Pad both sides for same-length output
        pad = self.kernel_size // 2
        # Replicate-pad along the time dimension
        x_padded = F.pad(x.transpose(1, 2), (pad, pad), mode="replicate")
        # Average pool across time
        trend = F.avg_pool1d(x_padded, self.kernel_size, stride=1).transpose(1, 2)
        seasonal = x - trend
        return trend, seasonal


class FFTFeatureExtractor(nn.Module):
    """
    Extract frequency-domain features from a time-series segment
    using the Fast Fourier Transform.

    Returns the top-k magnitude components as a feature vector.
    """

    def __init__(self, n_features: int = 32):
        super().__init__()
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            freq_features: (B, n_features)
        """
        # Average across feature dimension, then FFT along time
        x_mean = x.mean(dim=-1)                          # (B, L)
        fft_vals = torch.fft.rfft(x_mean, dim=-1)        # (B, L//2+1) complex
        magnitudes = fft_vals.abs()                       # (B, L//2+1)

        # Take top-k frequency magnitudes
        n = min(self.n_features, magnitudes.shape[-1])
        topk_vals, _ = magnitudes.topk(n, dim=-1)        # (B, n)

        # Pad if needed
        if n < self.n_features:
            pad_size = self.n_features - n
            topk_vals = F.pad(topk_vals, (0, pad_size))

        return topk_vals
