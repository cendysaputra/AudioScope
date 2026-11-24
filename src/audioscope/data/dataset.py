"""Cached, resampled waveforms and the train / validation / test split per fold."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..features.audio import fix_length, load_audio
from ..features.frontend import FeatureConfig
from .catalog import AudioCatalog


def _cache_key(catalog: AudioCatalog, fcfg: FeatureConfig) -> str:
    h = hashlib.sha1()
    for f in catalog.filenames:
        h.update(f.encode())
    return f"{catalog.name}_sr{fcfg.sample_rate}_n{fcfg.num_samples}_{h.hexdigest()[:10]}"


def load_waveform_cache(
    catalog: AudioCatalog,
    fcfg: FeatureConfig,
    cache_dir: str | Path | None = "data/cache",
    logger=None,
) -> torch.Tensor:
    """Return ``(N, num_samples)`` float32 waveforms at ``fcfg.sample_rate``.

    Resampling every clip once and caching the result keeps the training loop
    cheap. Features are computed from these waveforms on the fly, so waveform
    augmentation and feature extraction stay in one code path.
    """
    path = None
    if cache_dir is not None:
        path = Path(cache_dir) / f"{_cache_key(catalog, fcfg)}.pt"
        if path.exists():
            return torch.load(path, weights_only=True)
    waves = torch.empty(len(catalog), fcfg.num_samples, dtype=torch.float32)
    for i, p in enumerate(catalog.paths):
        wave, _ = load_audio(p, fcfg.sample_rate)
        waves[i] = fix_length(wave, fcfg.num_samples)
        if logger is not None and (i + 1) % 500 == 0:
            logger.info("cached %d/%d clips", i + 1, len(catalog))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(waves, path)
    return waves


def split_indices(
    catalog: AudioCatalog,
    test_fold: int,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Official-fold split: test = ``test_fold``; train/val drawn from the other folds.

    The validation set is taken from the training folds only, stratified by
    class and grouped by source recording, so clips cut from one recording do
    not sit on both sides of the train/val boundary.
    """
    if test_fold not in set(catalog.folds.tolist()):
        raise ValueError(f"fold {test_fold} not in dataset")
    test = catalog.fold_indices(test_fold)
    pool = np.flatnonzero(catalog.folds != test_fold)
    if val_fraction <= 0:
        return pool, np.array([], dtype=int), test

    rng = np.random.default_rng(seed)
    val: list[int] = []
    for cls in np.unique(catalog.targets[pool]):
        cls_idx = pool[catalog.targets[pool] == cls]
        groups = np.unique(catalog.groups[cls_idx])
        rng.shuffle(groups)
        want = max(1, int(round(val_fraction * len(cls_idx))))
        taken: list[int] = []
        for g in groups:
            if len(taken) >= want:
                break
            taken.extend(cls_idx[catalog.groups[cls_idx] == g].tolist())
        val.extend(taken)
    val_set = set(val)
    # Any other clip sharing a group with a validation clip also goes to validation.
    val_groups = set(catalog.groups[list(val_set)].tolist())
    val_idx = np.array(sorted(i for i in pool if catalog.groups[i] in val_groups), dtype=int)
    train_idx = np.array(sorted(set(pool.tolist()) - set(val_idx.tolist())), dtype=int)
    return train_idx, val_idx, test


class WaveformDataset(Dataset):
    """Index into a cached waveform tensor. Returns ``(waveform, target)``."""

    def __init__(self, waves: torch.Tensor, targets: np.ndarray, indices: np.ndarray):
        self.waves = waves
        self.targets = torch.as_tensor(targets, dtype=torch.long)
        self.indices = torch.as_tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        j = self.indices[i]
        return self.waves[j], self.targets[j]
