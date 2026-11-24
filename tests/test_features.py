"""Front-end shapes, a fixed-seed snapshot, and agreement with librosa."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from audioscope.features import FeatureConfig, LogMel, MFCCFrontend, Normalizer, build_frontend, fix_length
from audioscope.train.loop import FeaturePipeline

SNAPSHOT = Path(__file__).parent / "snapshots" / "logmel_snapshot.json"


def snapshot_signal(cfg: FeatureConfig) -> torch.Tensor:
    g = torch.Generator().manual_seed(1234)
    t = torch.arange(cfg.num_samples) / cfg.sample_rate
    chirp = torch.sin(2 * torch.pi * (200 + 800 * t) * t)
    return 0.3 * chirp + 0.05 * torch.randn(cfg.num_samples, generator=g)


def test_logmel_shape_default():
    cfg = FeatureConfig()
    out = LogMel(cfg)(torch.zeros(3, cfg.num_samples))
    assert out.shape == (3, 64, cfg.num_frames) == (3, 64, 216)
    assert torch.isfinite(out).all()  # silence stays finite thanks to the log offset


def test_mfcc_shape():
    cfg = FeatureConfig(kind="mfcc", n_mfcc=20)
    out = build_frontend(cfg)(torch.randn(2, cfg.num_samples))
    assert isinstance(build_frontend(cfg), MFCCFrontend)
    assert out.shape == (2, 20, cfg.num_frames)


def test_db_log_mode():
    cfg = FeatureConfig(log="db")
    x = snapshot_signal(cfg)
    nat = LogMel(FeatureConfig())(x)
    db = LogMel(cfg)(x)
    # 10*log10(m) = (10/ln 10) * ln(m) away from the clamps
    mask = nat > -6  # where the 1e-6 offset is negligible
    assert torch.allclose(db[mask], nat[mask] * 10 / np.log(10), atol=1e-2)


def test_fix_length():
    assert fix_length(torch.ones(10), 4).tolist() == [1, 1, 1, 1]
    assert fix_length(torch.ones(2), 4).tolist() == [1, 1, 0, 0]


def test_logmel_snapshot():
    cfg = FeatureConfig()
    out = LogMel(cfg)(snapshot_signal(cfg))
    stats = {
        "mean": out.mean().item(),
        "std": out.std().item(),
        "band_means_0_8_16_32_63": [out[i].mean().item() for i in (0, 8, 16, 32, 63)],
        "frame_100_bands_0_10_20_40": [out[i, 100].item() for i in (0, 10, 20, 40)],
    }
    if not SNAPSHOT.exists():  # first run writes the snapshot; it is committed with the tests
        SNAPSHOT.parent.mkdir(exist_ok=True)
        SNAPSHOT.write_text(json.dumps(stats, indent=2) + "\n")
    ref = json.loads(SNAPSHOT.read_text())
    assert stats["mean"] == pytest.approx(ref["mean"], abs=1e-3)
    assert stats["std"] == pytest.approx(ref["std"], abs=1e-3)
    assert np.allclose(stats["band_means_0_8_16_32_63"], ref["band_means_0_8_16_32_63"], atol=1e-3)
    assert np.allclose(stats["frame_100_bands_0_10_20_40"], ref["frame_100_bands_0_10_20_40"], atol=1e-3)


def test_matches_librosa():
    librosa = pytest.importorskip("librosa")
    cfg = FeatureConfig()
    x = snapshot_signal(cfg)
    ta = LogMel(cfg)(x).numpy()
    mel = librosa.feature.melspectrogram(
        y=x.numpy(), sr=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        n_mels=cfg.n_mels, fmin=0.0, fmax=cfg.fmax, center=True, pad_mode="reflect", power=2.0,
    )
    assert np.abs(ta - np.log(mel + cfg.log_offset)).max() < 1e-3


def test_normalizer_uses_fitted_stats():
    feats = torch.randn(10, 4, 20) * 3 + 5
    norm = Normalizer(4, "per_band").fit(feats)
    out = norm(feats)
    assert out.mean(dim=(0, 2)).abs().max() < 1e-4
    assert (out.std(dim=(0, 2)) - 1).abs().max() < 1e-3
    assert torch.equal(Normalizer(4, "none")(feats), feats)


def test_feature_pipeline_is_deterministic():
    cfg = FeatureConfig(sample_rate=8000, clip_seconds=1.0)
    pipe = FeaturePipeline(cfg)
    w = torch.randn(4, cfg.num_samples)
    pipe.fit(w)
    assert torch.equal(pipe(w), pipe(w))
    assert pipe(w).shape == (4, 1, cfg.n_mels, cfg.num_frames)


def test_unknown_feature_key_rejected():
    with pytest.raises(ValueError):
        FeatureConfig.from_dict({"n_mel": 64})
