from __future__ import annotations

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization.
    
    Supports mixed input/output dimensions: when predicting a subset of
    input features (e.g., 7 inputs -> 1 target), reverse() can denormalize
    using only the target channels' stored statistics.
    """

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

        self.mean = None
        self.stdev = None

    def forward(self, x):
        # x: (B, L, D_in)
        self.mean = x.mean(dim=1, keepdim=True).detach()
        self.stdev = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()

        x_norm = (x - self.mean) / self.stdev

        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias

        return x_norm

    def reverse(self, y, target_indices=None):
        """
        Reverse the normalization.
        
        Args:
            y: (B, H, D_out)  predictions
            target_indices: list of int or None
                If D_out < D_in, these are the column indices in the original
                input that correspond to the output channels.
                E.g., if input is [HUFL, HULL, ..., OT] and output is [OT],
                then target_indices = [6] (OT is the last column).
                If None, assumes D_out == D_in and uses all channels.
        """
        if target_indices is not None:
            # Select only the target channels' statistics
            mean = self.mean[:, :, target_indices]       # (B, 1, D_out)
            stdev = self.stdev[:, :, target_indices]     # (B, 1, D_out)
            if self.affine:
                weight = self.affine_weight[:, :, target_indices]
                bias = self.affine_bias[:, :, target_indices]
                y = (y - bias) / (weight + self.eps)
        else:
            mean = self.mean
            stdev = self.stdev
            if self.affine:
                y = (y - self.affine_bias) / (self.affine_weight + self.eps)

        y_denorm = y * stdev + mean
        return y_denorm
