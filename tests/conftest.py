from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from audioscope.data.catalog import AudioCatalog

REPO = Path(__file__).resolve().parents[1]
ESC50_ROOT = REPO / "data" / "esc50"

torch.set_num_threads(1)


@pytest.fixture(scope="session")
def esc50():
    if not (ESC50_ROOT / "meta" / "esc50.csv").exists():
        pytest.skip("ESC-50 not downloaded (python -m audioscope.data.download)")
    from audioscope.data import load_esc50

    return load_esc50(ESC50_ROOT)


def make_synthetic(num_classes=4, per_class_per_fold=3, folds=5, sr=8000, seconds=1.0, seed=0):
    """A tiny catalog plus waveforms: each class is a sine at its own frequency."""
    rng = np.random.default_rng(seed)
    n = num_classes * per_class_per_fold * folds
    t = np.arange(int(sr * seconds)) / sr
    waves, targets, fold_ids, groups = [], [], [], []
    for f in range(1, folds + 1):
        for c in range(num_classes):
            for j in range(per_class_per_fold):
                freq = 300 * (c + 1)
                w = 0.5 * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi)) + 0.01 * rng.standard_normal(len(t))
                waves.append(w.astype(np.float32))
                targets.append(c)
                fold_ids.append(f)
                groups.append(f"f{f}c{c}g{j // 2}")
    cat = AudioCatalog(
        name="synthetic",
        root=Path("."),
        paths=[Path(f"{i}.wav") for i in range(n)],
        filenames=[f"{i}.wav" for i in range(n)],
        folds=np.array(fold_ids),
        targets=np.array(targets),
        groups=np.array(groups),
        class_names=[f"tone{c}" for c in range(num_classes)],
    )
    return cat, torch.from_numpy(np.stack(waves))


@pytest.fixture
def synthetic():
    return make_synthetic()
