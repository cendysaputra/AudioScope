"""Train one model without scoring its held-out fold.

    python -m audioscope.train --config configs/baseline.yaml --holdout-fold 1 \
        --out checkpoints/baseline-holdout1.pt

This is for producing a checkpoint (e.g. for the Streamlit demo). Reportable
numbers come only from ``eval/crossval.py``, which trains and scores all folds.
"""

from __future__ import annotations

import argparse

import torch

from ..config import load_config
from ..data import load_catalog, load_waveform_cache
from ..features import FeatureConfig
from ..utils import get_logger, resolve_device
from .loop import save_checkpoint, train_fold


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--holdout-fold", type=int, default=1, help="official fold left out of training")
    ap.add_argument("--out", required=True, help="checkpoint path (.pt)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--cache-dir", default="data/cache")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = load_config(args.config)
    logger = get_logger()
    catalog = load_catalog(cfg["dataset"]["name"], cfg["dataset"]["root"])
    waves = load_waveform_cache(catalog, FeatureConfig.from_dict(cfg.get("features")), args.cache_dir, logger)
    res = train_fold(cfg, catalog, waves, args.holdout_fold, resolve_device(args.device), logger, evaluate_test=False)
    save_checkpoint(res.checkpoint, args.out)
    logger.info("saved %s", args.out)


if __name__ == "__main__":
    main()
