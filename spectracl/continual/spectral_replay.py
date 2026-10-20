from __future__ import annotations

"""
Spectral Replay Buffer for SpectraCL.

Implements three frequency-aware innovations for continual time series
forecasting, unified by the thesis that catastrophic forgetting in time
series is FREQUENCY-SELECTIVE: the model overwrites specific frequency
components where new task spectral content overlaps with past tasks.

1. Spectral Memory Curation (SMC)
   At task boundaries, selects samples via farthest-point sampling in
   frequency space, maximising temporal pattern diversity within a fixed
   memory budget. 500 spectrally diverse samples cover far more pattern
   variety than 500 random samples from consecutive sliding windows.

2. Spectral Interference-Aware Replay (SIAR)
   During training, samples replay data with probability proportional to
   spectral overlap with the current batch. Samples whose frequency
   content CLASHES with the current batch are at highest risk of being
   overwritten -- SIAR targets rehearsal where it matters most.

3. Frequency-Domain Rehearsal Loss (FDRL)
   Computes replay loss in the frequency domain with per-bin importance
   weights derived from past task spectral profiles. Directly protects
   learned temporal patterns (trend, seasonality) and is robust to minor
   phase shifts that inflate standard time-domain MSE.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class SpectralReplayBuffer:
    """
    Replay buffer with spectral intelligence for SpectraCL.

    Usage:
        buffer = SpectralReplayBuffer(capacity=500)

        # After each task:
        buffer.add_from_loader_smc(train_loader, device)
        buffer.store_task_profile(train_loader, device)

        # During training on next task:
        xr, yr = buffer.sample_siar(batch_size, x_current, device)
        out_r = model(xr)
        loss = buffer.compute_fdrl(out_r["y_pred"], yr, device)
    """

    def __init__(
        self,
        capacity: int = 500,
        n_fft_fingerprint: int = 48,
        fdrl_weight: float = 1.0,
        siar_temperature: float = 0.5,
        siar_mse_weight: float = 0.0,
        use_siar: bool = True,
        use_smc: bool = True,
        use_csd: bool = False,
        csd_blend: float = 0.3,
    ):
        self.capacity = capacity
        self.n_fft_fingerprint = n_fft_fingerprint
        self.fdrl_weight = fdrl_weight
        self.siar_temperature = siar_temperature
        self.siar_mse_weight = siar_mse_weight
        self.use_siar = use_siar
        self.use_smc = use_smc
        self.use_csd = use_csd
        self.csd_blend = csd_blend

        self.x_buffer: list[torch.Tensor] = []
        self.y_buffer: list[torch.Tensor] = []
        self.fp_buffer: list[torch.Tensor] = []

        self.task_profiles: list[torch.Tensor] = []
        # CSD: per-task cross-spectral energy profile (n_freq,)
        self.task_csd_profiles: list[torch.Tensor] = []

    @property
    def size(self) -> int:
        return sum(x.shape[0] for x in self.x_buffer)

    @property
    def n_tasks_stored(self) -> int:
        return len(self.x_buffer)

    # ------------------------------------------------------------------
    # Spectral fingerprinting
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_fingerprints(self, x: torch.Tensor) -> torch.Tensor:
        """Compact spectral fingerprint: (B, L, D) -> (B, n_fft).

        When CSD is enabled, enriches the magnitude fingerprint with
        cross-spectral density energy -- capturing how much channels
        co-vary at each frequency.  This makes SMC and SIAR sensitive
        to cross-channel structure in multivariate settings.
        """
        fft = torch.fft.rfft(x, dim=1)          # (B, n_freq, D)
        mag = fft.abs().mean(dim=-1)             # (B, n_freq)

        if self.use_csd and fft.shape[-1] > 1:
            # Cross-spectral energy per frequency -- O(D) not O(D^2)
            # ||sum_i X_i(f)||^2 - sum_i ||X_i(f)||^2
            sum_fft = fft.sum(dim=-1)                          # (B, n_freq)
            total_power = sum_fft.abs().pow(2)                 # (B, n_freq)
            auto_power = fft.abs().pow(2).sum(dim=-1)          # (B, n_freq)
            cross_power = (total_power - auto_power).clamp(min=0)
            cross_norm = cross_power / (auto_power + 1e-8)
            mag = mag + self.csd_blend * cross_norm

        n_freq = mag.shape[1]
        if n_freq >= self.n_fft_fingerprint:
            fp = mag[:, :self.n_fft_fingerprint]
        else:
            fp = F.pad(mag, (0, self.n_fft_fingerprint - n_freq))
        return F.normalize(fp, p=2, dim=-1)

    # ------------------------------------------------------------------
    # SMC: Spectral Memory Curation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def add_from_loader_smc(
        self,
        loader: DataLoader,
        device: str,
        max_batches: int = 100,
    ) -> None:
        """
        Add samples from a completed task using farthest-point sampling
        in spectral space to maximise frequency diversity.
        """
        all_x, all_y, all_fp = [], [], []
        for batch_idx, (x, y) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            x = x.to(device)
            fp = self._compute_fingerprints(x)
            all_x.append(x.cpu())
            all_y.append(y)
            all_fp.append(fp.cpu())

        if not all_x:
            return

        all_x = torch.cat(all_x, dim=0)
        all_y = torch.cat(all_y, dim=0)
        all_fp = torch.cat(all_fp, dim=0)

        n_tasks = len(self.x_buffer) + 1
        per_task = self.capacity // n_tasks
        n_select = min(per_task, all_x.shape[0])

        if n_select <= 0:
            return

        if self.use_smc:
            selected = self._farthest_point_sampling(all_fp, n_select)
        else:
            # Uniform random curation for ablation: disables spectral diversity selection.
            selected = torch.randperm(all_x.shape[0])[:n_select]

        self.x_buffer.append(all_x[selected])
        self.y_buffer.append(all_y[selected])
        self.fp_buffer.append(all_fp[selected])

        self._rebalance()

    def _farthest_point_sampling(
        self,
        fingerprints: torch.Tensor,
        n_select: int,
    ) -> torch.Tensor:
        """Greedy farthest-point sampling in spectral space."""
        N = fingerprints.shape[0]
        if n_select >= N:
            return torch.arange(N)

        selected = [torch.randint(N, (1,)).item()]
        min_dist = torch.full((N,), float("inf"))

        for _ in range(n_select - 1):
            last_fp = fingerprints[selected[-1]].unsqueeze(0)
            cos_sim = (fingerprints @ last_fp.T).squeeze(-1)
            dist = 1.0 - cos_sim
            min_dist = torch.min(min_dist, dist)
            for s in selected:
                min_dist[s] = -1.0
            selected.append(min_dist.argmax().item())

        return torch.tensor(selected, dtype=torch.long)

    def _rebalance(self) -> None:
        """Trim older tasks to maintain balanced capacity via SMC."""
        n_tasks = len(self.x_buffer)
        if n_tasks == 0:
            return
        per_task = self.capacity // n_tasks
        for i in range(n_tasks):
            if self.x_buffer[i].shape[0] > per_task:
                keep = self._farthest_point_sampling(self.fp_buffer[i], per_task)
                self.x_buffer[i] = self.x_buffer[i][keep]
                self.y_buffer[i] = self.y_buffer[i][keep]
                self.fp_buffer[i] = self.fp_buffer[i][keep]

    # ------------------------------------------------------------------
    # Task spectral profiles (for FDRL weights)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def store_task_profile(
        self,
        loader: DataLoader,
        device: str,
        max_batches: int = 50,
    ) -> None:
        """Store mean spectral profile and CSD energy of a task."""
        all_mag = []
        all_csd = []
        for batch_idx, (x, y) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            fft_x = torch.fft.rfft(x, dim=1)  # (B, n_freq_x, D)
            mag_x = fft_x.abs().mean(dim=-1)   # (B, n_freq_x)
            if y.dim() == 2:
                y = y.unsqueeze(-1)
            mag_y = torch.fft.rfft(y, dim=1).abs().mean(dim=-1)  # (B, n_freq_y)
            all_mag.append(torch.cat([mag_x, mag_y], dim=1).cpu())

            # CSD: cross-spectral energy on input
            if self.use_csd and fft_x.shape[-1] > 1:
                sum_fft = fft_x.sum(dim=-1)                       # (B, n_freq)
                total_p = sum_fft.abs().pow(2)                    # (B, n_freq)
                auto_p = fft_x.abs().pow(2).sum(dim=-1)           # (B, n_freq)
                cross_p = (total_p - auto_p).clamp(min=0)         # (B, n_freq)
                all_csd.append(cross_p.mean(dim=0).cpu())         # (n_freq,)

        if all_mag:
            self.task_profiles.append(torch.cat(all_mag, dim=0).mean(dim=0))
        if all_csd:
            self.task_csd_profiles.append(torch.stack(all_csd).mean(dim=0))

    # ------------------------------------------------------------------
    # SIAR: Spectral Interference-Aware Replay
    # ------------------------------------------------------------------

    def sample_siar(
        self,
        batch_size: int,
        current_x: torch.Tensor,
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        Sample replay data prioritising spectral overlap with the
        current batch -- targets the actual mechanism of forgetting.
        """
        if not self.x_buffer:
            return None

        all_x = torch.cat(self.x_buffer, dim=0)
        all_y = torch.cat(self.y_buffer, dim=0)
        n_total = all_x.shape[0]
        if n_total == 0:
            return None
        n_sample = min(batch_size, n_total)

        if self.use_siar:
            current_fp = self._compute_fingerprints(current_x)
            batch_fp = current_fp.mean(dim=0, keepdim=True).cpu()

            all_fp = torch.cat(self.fp_buffer, dim=0)
            overlap = (all_fp @ batch_fp.T).squeeze(-1).clamp(min=0)

            logits = overlap / (self.siar_temperature + 1e-8)
            probs = F.softmax(logits, dim=0)
            idx = torch.multinomial(probs, n_sample, replacement=(n_total < batch_size))
        else:
            # Uniform replay sampling for ablation: disables interference-aware prioritization.
            idx = torch.randint(0, n_total, (n_sample,))

        return all_x[idx].to(device), all_y[idx].to(device)

    # ------------------------------------------------------------------
    # FDRL: Frequency-Domain Rehearsal Loss
    # ------------------------------------------------------------------

    def _compute_fdrl_weights(self, pred_len: int, device: str) -> torch.Tensor:
        """Per-frequency importance weights from stored task profiles.

        When CSD profiles are available, frequencies with high cross-channel
        coupling receive boosted weights -- the model's learned cross-channel
        relationships at those frequencies are prioritised for preservation.
        """
        n_freq = pred_len // 2 + 1
        if not self.task_profiles:
            return torch.ones(n_freq, device=device)

        # Recency-weighted aggregation to prevent early-task bias accumulation
        agg = torch.zeros(n_freq)
        decay = 0.7
        n_profiles = len(self.task_profiles)
        total_w = 0.0
        for i, p in enumerate(self.task_profiles):
            w = decay ** (n_profiles - 1 - i)
            n = min(p.shape[0], n_freq)
            agg[:n] += w * p[:n]
            total_w += w
        agg /= total_w

        w = agg / (agg.mean() + 1e-8)
        w = 0.5 + 0.5 * w

        # CSD boost: upweight frequencies with high cross-channel energy
        if self.use_csd and self.task_csd_profiles:
            csd_agg = torch.zeros(n_freq)
            for csd in self.task_csd_profiles:
                n = min(csd.shape[0], n_freq)
                csd_agg[:n] += csd[:n]
            csd_agg /= len(self.task_csd_profiles)
            csd_norm = csd_agg / (csd_agg.mean() + 1e-8)
            csd_boost = (1.0 + 0.5 * csd_norm).clamp(max=3.0)
            w = w * csd_boost[:n_freq]

        return w.clamp(min=0.1, max=5.0).to(device)

    def compute_fdrl(
        self,
        y_pred: torch.Tensor,
        y_target: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        """Frequency-domain rehearsal loss with spectral importance weighting."""
        pred_fft = torch.fft.rfft(y_pred, dim=1)
        target_fft = torch.fft.rfft(y_target, dim=1)
        diff = (pred_fft - target_fft).abs().pow(2)
        w = self._compute_fdrl_weights(y_pred.shape[1], device)
        weighted = diff * w.unsqueeze(0).unsqueeze(-1)
        return self.fdrl_weight * weighted.mean()

    # ------------------------------------------------------------------
    # Spectral divergence (for adaptive protection)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_data_divergence(
        self,
        loader: DataLoader,
        device: str,
        max_batches: int = 20,
    ) -> float:
        """Cosine distance between new task spectrum and stored profiles."""
        if not self.task_profiles:
            return 0.0

        all_mag = []
        for batch_idx, (x, y) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            # Per-channel FFT then average magnitudes (consistent with store_task_profile)
            mag_x = torch.fft.rfft(x, dim=1).abs().mean(dim=-1)
            if y.dim() == 2:
                y = y.unsqueeze(-1)
            mag_y = torch.fft.rfft(y, dim=1).abs().mean(dim=-1)
            all_mag.append(torch.cat([mag_x, mag_y], dim=1).cpu())

        if not all_mag:
            return 0.0

        new_mag = torch.cat(all_mag, dim=0).mean(dim=0)
        divs = []
        for p in self.task_profiles:
            n = min(new_mag.shape[0], p.shape[0])
            sim = F.cosine_similarity(
                new_mag[:n].unsqueeze(0), p[:n].unsqueeze(0), dim=1
            ).item()
            divs.append(1.0 - max(sim, 0.0))
        return sum(divs) / len(divs)
