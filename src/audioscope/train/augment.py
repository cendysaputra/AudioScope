"""Training-only augmentation: time shift, noise, SpecAugment, mixup.

Every function here is called from the training branch of the loop only.
Validation and test batches never pass through it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np
import torch


@dataclass
class AugmentConfig:
    time_shift: float = 0.0  # max |roll| as a fraction of the clip; 0 disables
    noise_prob: float = 0.0  # probability a clip gets additive Gaussian noise
    noise_snr_db: list[float] = field(default_factory=lambda: [10.0, 40.0])
    specaugment: bool = False
    freq_mask_param: int = 8  # max width of a frequency mask, in bands
    time_mask_param: int = 24  # max width of a time mask, in frames
    num_freq_masks: int = 2
    num_time_masks: int = 2
    mixup_alpha: float = 0.0  # Beta(alpha, alpha); 0 disables
    mixup_prob: float = 1.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "AugmentConfig":
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown augmentation settings: {sorted(unknown)}")
        return cls(**d)

    @property
    def any_enabled(self) -> bool:
        return self.time_shift > 0 or self.noise_prob > 0 or self.specaugment or self.mixup_alpha > 0


def random_time_shift(wave: torch.Tensor, max_fraction: float, gen: torch.Generator) -> torch.Tensor:
    """Circularly roll each waveform in ``(B, samples)`` by a random offset."""
    if max_fraction <= 0:
        return wave
    n = wave.shape[-1]
    max_shift = int(max_fraction * n)
    shifts = torch.randint(-max_shift, max_shift + 1, (wave.shape[0],), generator=gen)
    idx = (torch.arange(n).unsqueeze(0) - shifts.unsqueeze(1)) % n
    return wave.gather(1, idx.to(wave.device))


def add_noise(wave: torch.Tensor, prob: float, snr_db: list[float], gen: torch.Generator) -> torch.Tensor:
    """Add white Gaussian noise at a random SNR (dB) to a random subset of clips."""
    if prob <= 0:
        return wave
    b = wave.shape[0]
    apply = torch.rand(b, generator=gen) < prob
    snr = torch.empty(b).uniform_(snr_db[0], snr_db[1], generator=gen)
    power = wave.pow(2).mean(dim=1).clamp_min(1e-10)
    noise_power = power / (10.0 ** (snr / 10.0))
    noise = torch.randn(wave.shape, generator=gen) * noise_power.sqrt().unsqueeze(1)
    noise = noise * apply.unsqueeze(1).float()
    return wave + noise.to(wave.device)


def spec_augment(spec: torch.Tensor, cfg: AugmentConfig, gen: torch.Generator) -> torch.Tensor:
    """Zero (= the normalized mean) random frequency bands and time spans of ``(B, 1, F, T)``."""
    if not cfg.specaugment:
        return spec
    spec = spec.clone()
    b, _, n_f, n_t = spec.shape
    for i in range(b):
        for _ in range(cfg.num_freq_masks):
            w = int(torch.randint(0, cfg.freq_mask_param + 1, (1,), generator=gen))
            if w and w < n_f:
                f0 = int(torch.randint(0, n_f - w + 1, (1,), generator=gen))
                spec[i, :, f0 : f0 + w, :] = 0.0
        for _ in range(cfg.num_time_masks):
            w = int(torch.randint(0, cfg.time_mask_param + 1, (1,), generator=gen))
            if w and w < n_t:
                t0 = int(torch.randint(0, n_t - w + 1, (1,), generator=gen))
                spec[i, :, :, t0 : t0 + w] = 0.0
    return spec


def mixup(
    x: torch.Tensor, y_onehot: torch.Tensor, alpha: float, prob: float, rng: np.random.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix a batch with a shuffled copy of itself (Zhang et al., 2018)."""
    if alpha <= 0 or rng.random() >= prob:
        return x, y_onehot
    lam = float(rng.beta(alpha, alpha))
    perm = torch.from_numpy(rng.permutation(x.shape[0])).to(x.device)
    return lam * x + (1 - lam) * x[perm], lam * y_onehot + (1 - lam) * y_onehot[perm]
