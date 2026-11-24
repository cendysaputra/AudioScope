"""Read audio files, mix to mono, resample, and fix the clip length."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF


def to_mono(wave: np.ndarray) -> np.ndarray:
    """``(samples,)`` or ``(samples, channels)`` -> ``(samples,)`` float32."""
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    return wave


def _read(source) -> tuple[np.ndarray, int]:
    try:
        data, sr = sf.read(source, dtype="float32", always_2d=False)
        return data, sr
    except (sf.LibsndfileError, RuntimeError, TypeError):
        # Formats libsndfile cannot open (some MP3/M4A builds): fall back to librosa/audioread.
        import librosa

        if isinstance(source, (bytes, bytearray)):
            source = io.BytesIO(source)
        if hasattr(source, "seek"):
            source.seek(0)
        data, sr = librosa.load(source, sr=None, mono=False)
        if data.ndim == 2:
            data = data.T
        return data.astype(np.float32), int(sr)


def load_audio(source, sample_rate: int | None = None) -> tuple[torch.Tensor, int]:
    """Load a file path, bytes, or file-like object as a mono float32 tensor.

    If ``sample_rate`` is given the waveform is resampled to it.
    Returns ``(waveform[samples], sample_rate)``.
    """
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    elif isinstance(source, Path):
        source = str(source)
    data, sr = _read(source)
    wave = torch.from_numpy(to_mono(data))
    if sample_rate is not None and sr != sample_rate:
        wave = AF.resample(wave, sr, sample_rate)
        sr = sample_rate
    return wave.contiguous(), sr


def fix_length(wave: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Zero-pad at the end or crop from the start to ``num_samples`` (last dim)."""
    n = wave.shape[-1]
    if n == num_samples:
        return wave
    if n > num_samples:
        return wave[..., :num_samples]
    pad = torch.zeros(*wave.shape[:-1], num_samples - n, dtype=wave.dtype)
    return torch.cat([wave, pad], dim=-1)
