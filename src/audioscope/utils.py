"""Seeding, logging, and environment capture shared by training and eval."""

from __future__ import annotations

import logging
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_revision(short: bool = False) -> str:
    try:
        args = ["git", "rev-parse", "--short" if short else "HEAD"]
        if short:
            args.append("HEAD")
        rev = subprocess.check_output(args, cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "HEAD", "--", "src", "eval", "configs"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return rev + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment_report() -> str:
    import torchaudio

    lines = [
        f"python: {sys.version.split()[0]}",
        f"platform: {platform.platform()}",
        f"torch: {torch.__version__}",
        f"torchaudio: {torchaudio.__version__}",
        f"numpy: {np.__version__}",
    ]
    for mod in ("torchvision", "librosa", "soundfile", "matplotlib"):
        try:
            lines.append(f"{mod}: {__import__(mod).__version__}")
        except ImportError:
            lines.append(f"{mod}: not installed")
    lines.append(f"torch threads: {torch.get_num_threads()}")
    lines.append(f"cuda available: {torch.cuda.is_available()}")
    lines.append(f"git revision: {git_revision()}")
    return "\n".join(lines) + "\n"


def get_logger(name: str = "audioscope", log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file is not None:
        for h in list(logger.handlers):
            if isinstance(h, logging.FileHandler):
                logger.removeHandler(h)
                h.close()
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def resolve_device(name: str | None = "auto") -> torch.device:
    if name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
