"""Models that take a normalized spectrogram ``(B, 1, n_features, frames)``."""

from __future__ import annotations

import torch.nn as nn

from .baseline import BaselineCNN
from .gradcam import GradCAM
from .panns import Cnn14Classifier
from .resnet import ResNetSpectrogram


def build_model(model_cfg: dict, num_classes: int) -> nn.Module:
    cfg = dict(model_cfg)
    name = cfg.pop("name")
    cfg.pop("gradcam_layer", None)
    if name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes, **cfg)
    if name == "resnet":
        return ResNetSpectrogram(num_classes=num_classes, **cfg)
    if name == "panns_cnn14":
        return Cnn14Classifier(num_classes=num_classes, **cfg)
    raise ValueError(f"unknown model {name!r}")


def gradcam_layer(model_cfg: dict, model: nn.Module) -> str:
    """Layer Grad-CAM reads: the config value if set, else the model's default."""
    return model_cfg.get("gradcam_layer") or model.default_gradcam_layer


__all__ = ["BaselineCNN", "Cnn14Classifier", "GradCAM", "ResNetSpectrogram", "build_model", "gradcam_layer"]
