"""YAML experiment configs with single-parent inheritance.

A config may name a parent with ``base: other.yaml`` (path relative to the
child file). The child is deep-merged over the parent, so an augmentation
config only lists what it changes relative to ``baseline.yaml``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(parent: dict, child: dict) -> dict:
    out = copy.deepcopy(parent)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a config file and resolve its ``base`` chain."""
    path = Path(path)
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    base = cfg.pop("base", None)
    if base is not None:
        parent = load_config(path.parent / base)
        cfg = deep_merge(parent, cfg)
    cfg.setdefault("name", path.stem)
    return cfg


def dump_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
