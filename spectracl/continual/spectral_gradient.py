from __future__ import annotations

"""
Spectral Gradient Band Decomposition (SGBD) for SpectraCL.

Replaces binary FPSP (Frequency-Partitioned State Protection) with a
continuous, frequency-aware gradient scaling mechanism for the SSM's
A_log parameters.

Key insight: each dimension of Mamba's A_log parameter controls the
decay rate of one hidden state dimension.  The decay rate determines
which temporal frequencies that state dimension is sensitive to:

    slow decay  (small exp(A_log))  ->  long memory  ->  low frequency
    fast decay  (large exp(A_log))  ->  short memory ->  high frequency

FPSP applies a binary mask (protect/plastic) based on a fixed ratio.
SGBD instead applies CONTINUOUS gradient scaling: state dimensions
whose effective frequency was important to past tasks receive damped
gradients (near-zero but not zero), while state dimensions tuned to
frequencies unused by past tasks receive full gradients.

    scale(d) = 1 - importance(f_d) * protection_strength

where f_d is the effective frequency of dimension d (derived from its
decay rate), and importance(f) is derived from stored task spectral
profiles.

Advantages over FPSP:
    1. Continuous scaling instead of binary mask - smoother optimisation
    2. Spectral-profile-driven: protection adapts to actual task content
    3. No arbitrary protection_ratio hyperparameter
    4. Naturally adapts as new tasks introduce new frequency demands

Reference: Novel contribution of SpectraCL (this work).
"""

import torch
import torch.nn as nn


class SpectralGradientDecomposer:
    """
    Continuous frequency-aware gradient scaling for SSM parameters.

    Maps each A_log dimension to its effective temporal frequency,
    then scales gradients based on how important that frequency was
    to past tasks.

    Usage:
        sgbd = SpectralGradientDecomposer(protection_strength=0.7)

        # After each task:
        sgbd.update_importance(task_profiles)

        # During training, after loss.backward():
        sgbd.scale_gradients(model)
        optimizer.step()
    """

    def __init__(
        self,
        protection_strength: float = 0.7,
        min_scale: float = 0.1,
    ):
        """
        Args:
            protection_strength: how strongly to dampen gradients at
                important frequencies (0 = no protection, 1 = full lock)
            min_scale: minimum gradient scale to prevent total lockout
                (guarantees some plasticity at all frequencies)
        """
        self.protection_strength = protection_strength
        self.min_scale = min_scale

        self._importance_spectrum: torch.Tensor | None = None
        self._n_tasks = 0

    @property
    def is_active(self) -> bool:
        return self._n_tasks > 0

    def update_importance(
        self,
        task_profiles: list[torch.Tensor],
    ) -> None:
        """
        Compute per-frequency importance from past task spectral profiles.

        Aggregates all tasks' frequency energy distributions into a single
        importance vector normalised to [0, 1].  Frequencies where past
        tasks concentrated energy receive high importance.
        """
        if not task_profiles:
            self._importance_spectrum = None
            return

        n_freq = max(p.shape[0] for p in task_profiles)
        agg = torch.zeros(n_freq)
        for p in task_profiles:
            n = min(p.shape[0], n_freq)
            agg[:n] += p[:n]
        agg /= len(task_profiles)

        # Normalise to [0, 1]
        agg_max = agg.max()
        if agg_max > 1e-8:
            self._importance_spectrum = agg / agg_max
        else:
            self._importance_spectrum = torch.zeros(n_freq)

        self._n_tasks += 1

    def _lookup_importance(self, normalised_freq: torch.Tensor) -> torch.Tensor:
        """
        Look up spectral importance for each A_log element based on
        its normalised frequency position.

        Args:
            normalised_freq: tensor with values in [0, 1], mapping each
                state dimension to a frequency bin
        Returns:
            importance values in [0, 1] for each element
        """
        if self._importance_spectrum is None:
            return torch.zeros_like(normalised_freq)

        n_bins = self._importance_spectrum.shape[0]
        indices = (normalised_freq * (n_bins - 1)).long().clamp(0, n_bins - 1).cpu()
        return self._importance_spectrum[indices].to(normalised_freq.device)

    @torch.no_grad()
    def scale_gradients(self, model: nn.Module) -> None:
        """
        Scale A_log gradients based on per-frequency importance.

        Call AFTER loss.backward() and BEFORE optimizer.step().

        Slow-decaying dimensions (low frequency, important to past tasks)
        receive damped gradients; fast-decaying dimensions (high frequency,
        less important) receive full gradients.
        """
        if not self.is_active or self._importance_spectrum is None:
            return

        for name, param in model.named_parameters():
            if "A_log" not in name or param.grad is None:
                continue

            # Map decay rates to normalised frequency positions
            decay_rates = param.data.exp()  # higher = faster decay = higher freq
            flat = decay_rates.flatten()

            dr_min = flat.min()
            dr_max = flat.max()
            if dr_max - dr_min < 1e-8:
                continue

            normalised = (flat - dr_min) / (dr_max - dr_min)

            # Look up importance and compute gradient scale
            importance = self._lookup_importance(normalised)
            scale = 1.0 - importance * self.protection_strength
            scale = scale.clamp(min=self.min_scale)

            param.grad.data *= scale.view_as(param.grad.data)
