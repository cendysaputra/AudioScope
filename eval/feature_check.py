#!/usr/bin/env python
"""Compare the torchaudio log-mel front end with librosa on real ESC-50 clips.

    python eval/feature_check.py --n 32 --out eval/feature_check.json

librosa is the reference implementation. Both sides use the same resampled
waveform, the same STFT settings (Hann, centered, reflect padding), and the
same Slaney mel filterbank. The check passes if every clip's maximum absolute
log-mel difference is below ``--atol`` (natural log units), ignoring bins
whose power is below ``--floor`` relative to the clip maximum, where float32
rounding dominates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import torch

from audioscope.config import load_config
from audioscope.data import load_esc50
from audioscope.features import FeatureConfig, LogMel, fix_length, load_audio


def librosa_logmel(wave: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=wave,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        win_length=cfg.win_length or cfg.n_fft,
        hop_length=cfg.hop_length,
        window="hann",
        center=True,
        pad_mode="reflect",
        power=2.0,
        n_mels=cfg.n_mels,
        fmin=cfg.f_min,
        fmax=cfg.fmax,
        htk=cfg.mel_scale == "htk",
        norm="slaney" if cfg.mel_norm == "slaney" else None,
    )
    return np.log(mel + cfg.log_offset)


def run(config: str, n: int, atol: float, floor_db: float, seed: int) -> dict:
    cfg = FeatureConfig.from_dict(load_config(config)["features"])
    cat = load_esc50(load_config(config)["dataset"]["root"])
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(cat), size=n, replace=False))
    front = LogMel(cfg)
    rows = []
    for i in idx:
        wave, _ = load_audio(cat.paths[i], cfg.sample_rate)
        wave = fix_length(wave, cfg.num_samples)
        ta = front(wave).numpy().astype(np.float64)
        lr = librosa_logmel(wave.numpy(), cfg)
        assert ta.shape == lr.shape, (ta.shape, lr.shape)
        mask = lr > lr.max() + floor_db / 10 * np.log(10)  # dB floor -> natural log
        diff = np.abs(ta - lr)
        rows.append(
            {
                "filename": cat.filenames[i],
                "shape": list(ta.shape),
                "max_abs_diff": float(diff[mask].max()) if mask.any() else 0.0,
                "mean_abs_diff": float(diff[mask].mean()) if mask.any() else 0.0,
                "max_abs_diff_all_bins": float(diff.max()),
                "bins_checked_fraction": float(mask.mean()),
            }
        )
    worst = max(r["max_abs_diff"] for r in rows)
    return {
        "config": config,
        "features": cfg.to_dict(),
        "librosa": librosa.__version__,
        "torch": torch.__version__,
        "n_clips": n,
        "seed": seed,
        "atol": atol,
        "floor_db": floor_db,
        "worst_max_abs_diff": worst,
        "mean_of_mean_abs_diff": float(np.mean([r["mean_abs_diff"] for r in rows])),
        "passed": worst < atol,
        "clips": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--floor-db", type=float, default=-80.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = run(args.config, args.n, args.atol, args.floor_db, args.seed)
    print(
        f"{res['n_clips']} clips: worst max |diff| = {res['worst_max_abs_diff']:.2e} "
        f"(atol {args.atol:g}, bins within {args.floor_db:g} dB of clip max) -> "
        f"{'PASS' if res['passed'] else 'FAIL'}"
    )
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
