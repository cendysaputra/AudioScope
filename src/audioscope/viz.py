"""Matplotlib figures with physical axes: seconds and hertz."""

from __future__ import annotations

import numpy as np

from .features.frontend import FeatureConfig, mel_band_centers


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_waveform(wave, sample_rate: int, ax=None, title: str | None = None):
    plt = _plt()
    wave = np.asarray(wave)
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 2.2))
    t = np.arange(len(wave)) / sample_rate
    ax.plot(t, wave, lw=0.5, color="#1f5f99")
    ax.set_xlim(0, t[-1] if len(t) else 1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    if title:
        ax.set_title(title)
    if fig is not None:
        fig.tight_layout()
    return fig or ax.figure


def _mel_axis(ax, cfg: FeatureConfig, n_ticks: int = 7):
    centers = mel_band_centers(cfg)
    ticks = np.linspace(0, cfg.n_mels - 1, n_ticks).round().astype(int)
    ax.set_yticks(ticks, [f"{centers[i]:.0f}" if centers[i] < 1000 else f"{centers[i] / 1000:.1f}k" for i in ticks])
    ax.set_ylabel("frequency (Hz, mel-band center)")


def plot_logmel(spec, cfg: FeatureConfig, ax=None, title: str | None = None, cam=None, cam_alpha: float = 0.7):
    """``spec``: ``(n_mels, frames)``. ``cam``: optional map of the same shape in [0, 1]."""
    plt = _plt()
    spec = np.asarray(spec)
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 3.2))
    dur = spec.shape[1] * cfg.hop_length / cfg.sample_rate
    extent = (0, dur, -0.5, spec.shape[0] - 0.5)
    im = ax.imshow(spec, origin="lower", aspect="auto", cmap="magma", extent=extent, interpolation="nearest")
    if cam is not None:
        cam = np.clip(np.asarray(cam, dtype=float), 0, 1)
        # Opacity follows the map, so low-importance regions leave the spectrogram visible.
        ax.imshow(cam, origin="lower", aspect="auto", cmap="viridis", alpha=cam_alpha * cam, extent=extent, vmin=0, vmax=1)
    ax.set_xlabel("time (s)")
    _mel_axis(ax, cfg)
    if title:
        ax.set_title(title)
    if fig is not None:
        fig.colorbar(im, ax=ax, pad=0.01, label="log mel power" if cam is None else "log mel power (Grad-CAM overlaid)")
        fig.tight_layout()
    return fig or ax.figure
