from __future__ import annotations

"""
Linear forecaster: one linear map from the (instance-normalised) lookback window of all
input channels to the forecast horizon of the target channels. It is the model class
covered by the frequency decomposition of forgetting in the paper (Proposition 1) and
serves as a second backbone for the frequency-resolved forgetting analysis.
"""

import torch
import torch.nn as nn

from spectracl.utils.revin import RevIN


class LinearForecaster(nn.Module):
    def __init__(self, num_features: int, output_features: int, seq_len: int, pred_len: int,
                 use_revin: bool = True, target_indices: list[int] | None = None):
        super().__init__()
        self.input_features = num_features
        self.output_features = output_features
        self.seq_len, self.pred_len = seq_len, pred_len
        self.use_revin = use_revin
        self.target_indices = target_indices
        self.revin = RevIN(num_features=num_features)
        self.linear = nn.Linear(seq_len * num_features, pred_len * output_features)

    def forward(self, x: torch.Tensor, update_memory: bool = False) -> dict:
        b = x.shape[0]
        if self.use_revin:
            x = self.revin(x)
        y = self.linear(x.reshape(b, -1)).view(b, self.pred_len, self.output_features)
        if self.use_revin:
            if self.output_features == self.input_features:
                y = self.revin.reverse(y)
            elif self.target_indices is not None:
                y = self.revin.reverse(y, target_indices=self.target_indices)
        return {"y_pred": y, "h": None}
