"""Log-mel spectrogram and MFCC front end.

All settings live in :class:`FeatureConfig`, which is saved inside every
checkpoint. Training, evaluation, and the Streamlit app build the front end
from that saved config, so they compute the same representation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np
import torch
import torch.nn as nn
import torchaudio.functional as AF
import torchaudio.transforms as T


@dataclass
class FeatureConfig:
    sample_rate: int = 22050
    clip_seconds: float = 5.0
    n_fft: int = 1024
    win_length: int | None = None  # defaults to n_fft
    hop_length: int = 512
    window: str = "hann"
    n_mels: int = 64
    f_min: float = 0.0
    f_max: float | None = None  # defaults to sample_rate / 2
    mel_scale: str = "slaney"  # "slaney" matches librosa's default; "htk" is the alternative
    mel_norm: str | None = "slaney"
    log: str = "natural"  # "natural": log(mel + log_offset); "db": 10*log10(max(mel, amin))
    log_offset: float = 1e-6
    amin: float = 1e-10
    kind: str = "logmel"  # "logmel" or "mfcc"
    n_mfcc: int = 20
    normalize: str = "per_band"  # "per_band", "global", or "none"; stats come from training folds

    @classmethod
    def from_dict(cls, d: dict | None) -> "FeatureConfig":
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown feature settings: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def num_samples(self) -> int:
        return int(round(self.sample_rate * self.clip_seconds))

    @property
    def num_frames(self) -> int:
        # torch.stft with center=True gives 1 + floor(n / hop) frames.
        return 1 + self.num_samples // self.hop_length

    @property
    def n_features(self) -> int:
        return self.n_mfcc if self.kind == "mfcc" else self.n_mels

    @property
    def fmax(self) -> float:
        return self.f_max if self.f_max is not None else self.sample_rate / 2


class LogMel(nn.Module):
    """Waveform ``(..., samples)`` -> log-mel ``(..., n_mels, frames)``."""

    def __init__(self, cfg: FeatureConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.window != "hann":
            raise ValueError("only the Hann window is supported")
        self.mel = T.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length or cfg.n_fft,
            hop_length=cfg.hop_length,
            f_min=cfg.f_min,
            f_max=cfg.fmax,
            n_mels=cfg.n_mels,
            power=2.0,
            center=True,
            pad_mode="reflect",
            norm=cfg.mel_norm,
            mel_scale=cfg.mel_scale,
        )

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        mel = self.mel(wave)
        if self.cfg.log == "db":
            return 10.0 * torch.log10(torch.clamp(mel, min=self.cfg.amin))
        return torch.log(mel + self.cfg.log_offset)


class MFCCFrontend(nn.Module):
    """MFCCs computed from the same log-mel front end (DCT-II, ortho norm)."""

    def __init__(self, cfg: FeatureConfig):
        super().__init__()
        self.logmel = LogMel(cfg)
        dct = AF.create_dct(cfg.n_mfcc, cfg.n_mels, norm="ortho")  # (n_mels, n_mfcc)
        self.register_buffer("dct", dct)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        lm = self.logmel(wave)  # (..., n_mels, frames)
        return torch.matmul(lm.transpose(-1, -2), self.dct).transpose(-1, -2)


def build_frontend(cfg: FeatureConfig) -> nn.Module:
    if cfg.kind == "logmel":
        return LogMel(cfg)
    if cfg.kind == "mfcc":
        return MFCCFrontend(cfg)
    raise ValueError(f"unknown feature kind {cfg.kind!r}")


class Normalizer(nn.Module):
    """Standardize features with statistics fitted on training data only."""

    def __init__(self, n_features: int, mode: str = "per_band"):
        super().__init__()
        self.mode = mode
        self.register_buffer("mean", torch.zeros(n_features, 1))
        self.register_buffer("std", torch.ones(n_features, 1))

    @torch.no_grad()
    def fit(self, feats: torch.Tensor) -> "Normalizer":
        """``feats``: ``(N, n_features, frames)``."""
        if self.mode == "per_band":
            self.mean.copy_(feats.mean(dim=(0, 2)).unsqueeze(-1))
            self.std.copy_(feats.std(dim=(0, 2)).clamp_min(1e-5).unsqueeze(-1))
        elif self.mode == "global":
            self.mean.fill_(feats.mean().item())
            self.std.fill_(max(feats.std().item(), 1e-5))
        return self

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return feats
        return (feats - self.mean) / self.std


def mel_band_centers(cfg: FeatureConfig) -> np.ndarray:
    """Center frequency in Hz of each mel band (for labeling the spectrogram axis)."""
    import librosa

    htk = cfg.mel_scale == "htk"
    # n_mels + 2 edges; band i is centered on edge i + 1.
    edges = librosa.mel_frequencies(n_mels=cfg.n_mels + 2, fmin=cfg.f_min, fmax=cfg.fmax, htk=htk)
    return edges[1:-1]
