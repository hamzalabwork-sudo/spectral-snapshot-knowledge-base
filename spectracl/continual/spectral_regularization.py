from __future__ import annotations

"""
Spectral Frequency-Response Regularisation (SFR) for SpectraCL.

Preserves the model's learned relationship with individual frequency
components across tasks by probing with frequency-isolated inputs and
penalising per-frequency drift.

Theoretical motivation
----------------------
A time series forecasting model M implicitly learns a nonlinear mapping
from input frequency components to output predictions.  After training on
task k the model responds to input frequency omega with some characteristic
output; overwriting that response during task k+1 is one mechanism of
catastrophic forgetting.

SFR makes this explicit.  After each task it constructs a set of N pure-
sinusoidal "probe" inputs psi_i  (i = 1 .. N_probes,  each isolating one
frequency bin) and records the model's output:

    r_i^{(k)} = M_k(psi_i)

During training on task k+1 a regularisation term penalises drift:

    L_SFR  =  (1/K) sum_{k=1}^{K} sum_{i=1}^{N}
              w_i^{(k)} || M_{k+1}(psi_i) - r_i^{(k)} ||^2

where w_i^{(k)} is a per-frequency importance weight derived from task k's
spectral energy density.  Frequencies vital to old tasks are locked;
frequencies only needed by the new task are free.

This constraint space -- the model's *function frequency response* -- is
fundamentally different from:
    - parameter space  (EWC / SI)
    - gradient space   (NSP / OWM)
    - data space       (replay / distillation)

No time-domain equivalent exists: standard distillation uses arbitrary
inputs and cannot selectively protect individual frequency responses.
"""

import math
import torch
import torch.nn as nn


class SpectralRegularizer:
    """
    Probes the model with frequency-isolated inputs and penalises
    per-frequency output drift weighted by past-task spectral energy.

    Usage:
        sfr = SpectralRegularizer(n_probes=32, reg_weight=1.0)

        # After each task:
        sfr.snapshot(model, seq_len, n_features, device,
                     spectral_profile=buffer.task_profiles[-1])

        # During training on next task (every sfr_interval batches):
        loss = loss + sfr.compute_loss(model, seq_len, n_features, device)
    """

    def __init__(
        self,
        n_probes: int = 32,
        reg_weight: float = 1.0,
    ):
        self.n_probes = n_probes
        self.reg_weight = reg_weight

        self._probes: torch.Tensor | None = None
        self._frozen: list[torch.Tensor] = []     # per-task frozen outputs
        self._weights: list[torch.Tensor] = []    # per-task freq importance

    @property
    def is_active(self) -> bool:
        return len(self._frozen) > 0

    # ------------------------------------------------------------------
    # Probe generation
    # ------------------------------------------------------------------

    def _build_probes(
        self,
        seq_len: int,
        n_features: int,
        device: str,
    ) -> torch.Tensor:
        """
        Build frequency-isolated probe inputs.

        Each probe is a pure sinusoid at a single frequency bin, spanning
        [1 cycle / window .. Nyquist].  Probes are normalised to zero mean
        and unit variance so that RevIN does not distort them.

        Returns: (n_probes, seq_len, n_features)
        """
        if (
            self._probes is not None
            and self._probes.shape[1] == seq_len
            and self._probes.shape[2] == n_features
        ):
            return self._probes.to(device)

        t = torch.arange(seq_len, dtype=torch.float32) / seq_len
        freqs = torch.linspace(1.0, seq_len / 2, self.n_probes)

        probes = []
        for f in freqs:
            s = torch.sin(2 * math.pi * f * t)
            s = (s - s.mean()) / (s.std() + 1e-8)
            probes.append(s.unsqueeze(-1).expand(-1, n_features).clone())

        self._probes = torch.stack(probes)
        return self._probes.to(device)

    # ------------------------------------------------------------------
    # Snapshot: record model's frequency response after a task
    # ------------------------------------------------------------------

    @torch.no_grad()
    def snapshot(
        self,
        model: nn.Module,
        seq_len: int,
        n_features: int,
        device: str,
        spectral_profile: torch.Tensor | None = None,
    ) -> None:
        """
        Probe the model and freeze its per-frequency responses.

        Args:
            model: just-trained model (still in memory)
            seq_len: input window length
            n_features: number of input channels
            device: compute device
            spectral_profile: (n_freq,) mean FFT magnitude of the task's
                data.  Used to weight regularisation -- frequencies where
                the task had more energy receive stronger protection.
        """
        probes = self._build_probes(seq_len, n_features, device)
        model.eval()
        out = model(probes, update_memory=False)
        self._frozen.append(out["y_pred"].detach().cpu())

        # --- per-frequency importance weights ---
        w = torch.ones(self.n_probes)
        if spectral_profile is not None:
            n_bins = spectral_profile.shape[0]
            if n_bins >= self.n_probes:
                idx = torch.linspace(0, n_bins - 1, self.n_probes).long()
                w = spectral_profile[idx].clone()
            else:
                w[:n_bins] = spectral_profile
            w = w / (w.mean() + 1e-8)
            w = (0.5 + 0.5 * w).clamp(0.1, 5.0)

        self._weights.append(w)

    # ------------------------------------------------------------------
    # Loss: penalise per-frequency drift
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model: nn.Module,
        seq_len: int,
        n_features: int,
        device: str,
    ) -> torch.Tensor | None:
        """
        Compute SFR loss across all stored task snapshots.

        One forward pass of (n_probes) samples through the model,
        compared against each past task's frozen responses.

        Returns weighted scalar loss, or None if no snapshots stored.
        """
        if not self._frozen:
            return None

        probes = self._build_probes(seq_len, n_features, device)
        out = model(probes, update_memory=False)
        current = out["y_pred"]  # (n_probes, pred_len, n_features)

        total = torch.tensor(0.0, device=device)
        for frozen, weights in zip(self._frozen, self._weights):
            frozen_d = frozen.to(device)
            weights_d = weights.to(device)
            per_probe = ((current - frozen_d) ** 2).mean(dim=(1, 2))
            total = total + (per_probe * weights_d).mean()

        return self.reg_weight * total / len(self._frozen)
