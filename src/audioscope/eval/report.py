"""Write the files that make a run reportable: metrics.json, confusion matrices, figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def write_confusion_csv(cm: np.ndarray, class_names: list[str], path: Path) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["true\\pred", *class_names])
        for name, row in zip(class_names, cm):
            w.writerow([name, *row.tolist()])


def plot_confusion(cm: np.ndarray, class_names: list[str], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(class_names)
    size = max(6.0, 0.22 * n + 2)
    fig, ax = plt.subplots(figsize=(size, size))
    row = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row, out=np.zeros_like(cm, dtype=float), where=row > 0)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), class_names, rotation=90, fontsize=6 if n > 20 else 9)
    ax.set_yticks(range(n), class_names, fontsize=6 if n > 20 else 9)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of true class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_json(obj: dict, path: Path) -> None:
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
        fh.write("\n")
