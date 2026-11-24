"""ImageNet ResNet fine-tuned on log-mel spectrograms (Stage 3).

The single-channel spectrogram is repeated to three channels and resized to
``input_size``. Grad-CAM maps are interpolated straight back to the original
``(n_mels, frames)`` grid, so overlays stay in seconds and hertz.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResNetSpectrogram(nn.Module):
    def __init__(
        self,
        num_classes: int,
        arch: str = "resnet18",
        pretrained: str | None = "imagenet",
        weights_path: str | None = None,
        input_size: tuple[int, int] | list[int] = (224, 224),
        dropout: float = 0.0,
        imagenet_norm: bool = False,
    ):
        """``pretrained``: ``"imagenet"`` (torchvision default weights, downloaded
        on first use or read from ``weights_path``) or ``None`` (random init)."""
        super().__init__()
        self._backbone_frozen = False
        ctor = getattr(torchvision.models, arch)
        self.backbone = ctor(weights=None)
        self.source_checkpoint = None
        if pretrained == "imagenet":
            if weights_path:
                state = torch.load(Path(weights_path), map_location="cpu", weights_only=True)
                self.source_checkpoint = str(weights_path)
            else:
                enum = torchvision.models.get_model_weights(arch).DEFAULT
                state = enum.get_state_dict(progress=False)
                self.source_checkpoint = f"torchvision:{arch}:{enum.name}"
            self.backbone.load_state_dict(state)
        elif pretrained is not None:
            raise ValueError(f"unknown pretrained source {pretrained!r}")
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feats, num_classes))
        self.input_size = tuple(input_size)
        self.imagenet_norm = imagenet_norm
        self.register_buffer("img_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.default_gradcam_layer = "backbone.layer4"

    def head_parameters(self):
        return self.backbone.fc.parameters()

    def set_backbone_trainable(self, trainable: bool) -> None:
        self._backbone_frozen = not trainable
        for name, p in self.backbone.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = trainable
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if self._backbone_frozen:
            # A frozen backbone keeps its pretrained BatchNorm statistics.
            for name, m in self.backbone.named_modules():
                if isinstance(m, nn.BatchNorm2d) and not name.startswith("fc"):
                    m.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        x = x.repeat(1, 3, 1, 1)
        if self.imagenet_norm:
            x = (x - self.img_mean) / self.img_std
        return self.backbone(x)
