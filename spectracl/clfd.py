from __future__ import annotations

"""
CLFD adapted to forecasting (Liu et al., "Continual Learning in the Frequency Domain", NeurIPS 2024).

CLFD's frequency-domain feature encoder (FFE) applies a Haar wavelet transform to the input,
keeps the low- and high-frequency sub-bands at half resolution, and fuses them with 1x1
convolutions into a low-frequency, a high-frequency and a global feature that are fed to the
feature extractor; the encoder is frozen after the first task. For time series we use the 1-D
Haar transform along the lookback window and per-time-step linear maps (the 1-D analogue of
1x1 convolutions). CLFD's class-aware feature selection (CFFS) relies on class labels and has
no counterpart in forecasting, so it is not included. As in the paper, CLFD is combined with a
rehearsal method (ER).
"""

import math

import torch
import torch.nn as nn

from spectracl.backbone.mamba import MambaBackbone
from spectracl.model import MambaForecaster


class CLFDForecaster(MambaForecaster):
    def __init__(self, num_features: int, **kwargs):
        super().__init__(num_features=num_features, **kwargs)
        d = num_features
        self.ffe_low = nn.Linear(d, d)
        self.ffe_high = nn.Linear(d, d)
        self.ffe_global = nn.Linear(2 * d, d)
        bb = self.backbone
        self.backbone = MambaBackbone(
            num_features=3 * d, d_model=self.d_model, d_state=self.d_state,
            d_conv=kwargs.get("d_conv", 4), expand=kwargs.get("expand", 2),
            n_layers=len(bb.layers), dropout=kwargs.get("dropout", 0.1),
        )

    def freeze_encoder(self) -> None:
        for layer in (self.ffe_low, self.ffe_high, self.ffe_global):
            for p in layer.parameters():
                p.requires_grad_(False)

    @staticmethod
    def _haar(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """1-D Haar DWT along time: (B, L, D) -> approximation and detail, each (B, L/2, D)."""
        even, odd = x[:, 0::2], x[:, 1::2]
        return (even + odd) / math.sqrt(2.0), (even - odd) / math.sqrt(2.0)

    def forward(self, x: torch.Tensor, update_memory: bool = False) -> dict:
        b = x.shape[0]
        if self.use_revin:
            x = self.revin(x)
        low, high = self._haar(x)
        feats = torch.cat([self.ffe_low(low), self.ffe_high(high),
                           self.ffe_global(torch.cat([low, high], dim=-1))], dim=-1)
        h = self.backbone(feats)
        y_pred = self.apply_gate(x, self.prediction_head(h).view(b, self.pred_len, self.output_features))
        if self.use_revin:
            if self.output_features == self.input_features:
                y_pred = self.revin.reverse(y_pred)
            elif self.target_indices is not None:
                y_pred = self.revin.reverse(y_pred, target_indices=self.target_indices)
        return {"y_pred": y_pred, "h": h}
