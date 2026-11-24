"""Metadata for ESC-50 and UrbanSound8K, keyed by the official fold column.

Folds are read from the published metadata. They are never reshuffled.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AudioCatalog:
    name: str
    root: Path
    paths: list[Path]  # absolute audio paths
    filenames: list[str]
    folds: np.ndarray  # official fold id per clip (1-based)
    targets: np.ndarray  # class index per clip
    groups: np.ndarray  # source-recording id per clip (keeps related clips together)
    class_names: list[str]

    @property
    def num_folds(self) -> int:
        return int(self.folds.max())

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def __len__(self) -> int:
        return len(self.paths)

    def fold_indices(self, fold: int) -> np.ndarray:
        return np.flatnonzero(self.folds == fold)


def _class_names(pairs: list[tuple[int, str]]) -> list[str]:
    names: dict[int, str] = {}
    for idx, name in pairs:
        if names.setdefault(idx, name) != name:
            raise ValueError(f"class {idx} has two names: {names[idx]!r} and {name!r}")
    if sorted(names) != list(range(len(names))):
        raise ValueError("class indices are not contiguous")
    return [names[i] for i in range(len(names))]


def load_esc50(root: str | Path = "data/esc50", esc10_only: bool = False) -> AudioCatalog:
    """Read ``meta/esc50.csv``. ``root`` is the unpacked ESC-50 repository."""
    root = Path(root)
    with open(root / "meta" / "esc50.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if esc10_only:
        rows = [r for r in rows if r["esc10"] == "True"]
    if esc10_only:
        # Re-index the ten ESC-10 classes to 0..9, ordered by original target.
        orig = sorted({int(r["target"]) for r in rows})
        remap = {t: i for i, t in enumerate(orig)}
        names = {int(r["target"]): r["category"] for r in rows}
        class_names = [names[t] for t in orig]
        targets = np.array([remap[int(r["target"])] for r in rows])
    else:
        class_names = _class_names([(int(r["target"]), r["category"]) for r in rows])
        targets = np.array([int(r["target"]) for r in rows])
    return AudioCatalog(
        name="esc10" if esc10_only else "esc50",
        root=root,
        paths=[root / "audio" / r["filename"] for r in rows],
        filenames=[r["filename"] for r in rows],
        folds=np.array([int(r["fold"]) for r in rows]),
        targets=targets,
        groups=np.array([r["src_file"] for r in rows]),
        class_names=class_names,
    )


def load_urbansound8k(root: str | Path = "data/urbansound8k") -> AudioCatalog:
    """Read ``metadata/UrbanSound8K.csv``; audio is under ``audio/fold{k}/``."""
    root = Path(root)
    with open(root / "metadata" / "UrbanSound8K.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return AudioCatalog(
        name="urbansound8k",
        root=root,
        paths=[root / "audio" / f"fold{r['fold']}" / r["slice_file_name"] for r in rows],
        filenames=[r["slice_file_name"] for r in rows],
        folds=np.array([int(r["fold"]) for r in rows]),
        targets=np.array([int(r["classID"]) for r in rows]),
        groups=np.array([r["fsID"] for r in rows]),
        class_names=_class_names([(int(r["classID"]), r["class"]) for r in rows]),
    )


def load_catalog(name: str, root: str | Path) -> AudioCatalog:
    if name == "esc50":
        return load_esc50(root)
    if name == "esc10":
        return load_esc50(root, esc10_only=True)
    if name == "urbansound8k":
        return load_urbansound8k(root)
    raise ValueError(f"unknown dataset {name!r}")
