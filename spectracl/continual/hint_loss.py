from __future__ import annotations

"""
Frequency-Selective Hint Loss (FSHL) for SpectraCL.

Replaces hard null-space gradient projection (NSP) with a soft,
frequency-aware knowledge distillation mechanism that operates on
the model's learned temporal representations.

Key insight: catastrophic forgetting in multivariate time series is
FREQUENCY-SELECTIVE - the model overwrites specific frequency
components where new-task spectral content overlaps with past tasks.
FSHL selectively preserves the model's learned frequency responses
using per-frequency importance weights derived from past-task spectral
profiles.

After each task, a frozen copy of the model is stored. During training
on subsequent tasks, FSHL computes a frequency-domain distillation loss
between the current and frozen models' pre-pooling temporal
representations:

    L_FSHL = (1/K) sum_k  sum_f  w_f^(k) * ||H_new(f) - H_old^(k)(f)||^2

where:
    H_new(f)    - FFT of current model's temporal representation at freq f
    H_old^(k)(f)- FFT of frozen model k's temporal representation at freq f
    w_f^(k)     - per-frequency importance weight from task k's spectral
                  profile (high energy -> strong protection)

Advantages over NSP:
    1. Soft constraint: allows controlled drift, never locks out directions
    2. Frequency-selective: protects only the frequencies that matter
    3. No freedom collapse: doesn't shrink monotonically across tasks
    4. Operates in temporal-frequency space, not raw feature covariance

Reference: Novel contribution of SpectraCL (this work).
"""

import torch
import torch.nn as nn
from copy import deepcopy


class FrequencyHintLoss:
    """
    Frequency-selective knowledge distillation for continual learning.

    Stores frozen copies of the model after each task and computes a
    distillation loss in the frequency domain of the backbone's temporal
    representations (before mean pooling).

    Usage:
        hint = FrequencyHintLoss(weight=1.0)

        # After each task:
        hint.snapshot(model)
        hint.update_frequency_weights(task_profiles)

        # During training on next task:
        loss = loss + hint.compute_loss(model, x, device)
    """

    def __init__(
        self,
        weight: float = 1.0,
        decay: float = 0.7,
    ):
        """
        Args:
            weight: overall scaling for the hint loss
            decay: exponential recency weighting for frozen models
                   (most recent = 1.0, previous = decay, etc.)
        """
        self.weight = weight
        self.decay = decay

        self._frozen_models: list[nn.Module] = []
        self._freq_weights: list[torch.Tensor] = []
        self._n_tasks = 0

    @property
    def is_active(self) -> bool:
        return self._n_tasks > 0

    def snapshot(self, model: nn.Module) -> None:
        """Freeze a copy of the current model after task training."""
        frozen = deepcopy(model)
        frozen.eval()
        for p in frozen.parameters():
            p.requires_grad_(False)
        self._frozen_models.append(frozen)
        self._n_tasks += 1

    def update_frequency_weights(
        self,
        task_profiles: list[torch.Tensor],
    ) -> None:
        """
        Derive per-frequency importance from stored task spectral profiles.

        Frequencies where past tasks had high spectral energy receive
        stronger distillation protection.
        """
        self._freq_weights = []
        for profile in task_profiles:
            w = profile / (profile.mean() + 1e-8)
            w = 0.5 + 0.5 * w
            w = w.clamp(min=0.1, max=5.0)
            self._freq_weights.append(w)

    @staticmethod
    def _get_temporal_repr(
        model: nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract the backbone's temporal representation BEFORE mean pooling.

        Returns: (B, L, d_model)  - the full sequential representation
                 that retains temporal frequency information.
        """
        if model.use_revin:
            x = model.revin(x)
        h = model.backbone.input_proj(x)
        for layer in model.backbone.layers:
            h = layer(h)
        h = model.backbone.norm(h)
        return h

    def compute_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        """
        Frequency-selective hint loss between current and frozen models.

        Computes FFT of temporal representations, then per-frequency
        weighted MSE to selectively protect important frequencies.
        """
        if not self._frozen_models:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Current model's temporal repr - WITH gradients for backprop
        curr_repr = self._get_temporal_repr(model, x)
        curr_fft = torch.fft.rfft(curr_repr, dim=1)

        total_loss = torch.tensor(0.0, device=device)
        total_weight = 0.0
        n_frozen = len(self._frozen_models)

        for i, frozen_model in enumerate(self._frozen_models):
            frozen_model = frozen_model.to(device)

            with torch.no_grad():
                froz_repr = self._get_temporal_repr(frozen_model, x)
                froz_fft = torch.fft.rfft(froz_repr, dim=1)

            # Per-frequency squared error
            diff = (curr_fft - froz_fft).abs().pow(2)  # (B, n_freq, d_model)

            # Apply per-frequency importance weights
            if i < len(self._freq_weights):
                n_freq = diff.shape[1]
                w = self._freq_weights[i]
                w_len = min(w.shape[0], n_freq)
                freq_w = torch.ones(n_freq, device=device)
                freq_w[:w_len] = w[:w_len].to(device)
                diff = diff * freq_w.unsqueeze(0).unsqueeze(-1)

            # Recency weighting: most recent frozen model matters most
            task_w = self.decay ** (n_frozen - 1 - i)
            total_loss = total_loss + task_w * diff.mean()
            total_weight += task_w

        return self.weight * total_loss / max(total_weight, 1e-8)
