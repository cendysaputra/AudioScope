"""Training loop and augmentation."""

from .augment import AugmentConfig
from .loop import FoldResult, predict, train_fold

__all__ = ["AugmentConfig", "FoldResult", "predict", "train_fold"]
