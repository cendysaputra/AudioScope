"""PANNs Cnn14 (Kong et al., 2020) with a new classification head (Stage 3).

The convolutional trunk mirrors the published Cnn14 so its AudioSet
checkpoint (``Cnn14_mAP=0.431.pth``, Zenodo record 3987831) loads by key
name. The checkpoint expects 32 kHz audio, n_fft 1024, hop 320, 64 mel bands
from 50 Hz to 14 kHz, and a dB log-mel without external normalization (the
model's own ``bn0`` normalizes per band); ``configs/transfer-panns-cnn14.yaml``
sets exactly that front end.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor, pool: tuple[int, int] = (2, 2)) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=pool)


class Cnn14Classifier(nn.Module):
    TRUNK_PREFIXES = ("bn0.", "conv_block", "fc1.")

    def __init__(
        self,
        num_classes: int,
        n_mels: int = 64,
        checkpoint_path: str | None = None,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(n_mels)
        widths = [1, 64, 128, 256, 512, 1024, 2048]
        for i in range(6):
            setattr(self, f"conv_block{i + 1}", _ConvBlock(widths[i], widths[i + 1]))
        self.fc1 = nn.Linear(2048, 2048)
        self.head = nn.Linear(2048, num_classes)
        self.dropout = dropout
        self.source_checkpoint = None
        if checkpoint_path:
            self.load_audioset(checkpoint_path)
        self.default_gradcam_layer = "conv_block5"

    def load_audioset(self, path: str | Path) -> None:
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        trunk = {k: v for k, v in state.items() if k.startswith(self.TRUNK_PREFIXES)}
        missing, unexpected = self.load_state_dict(trunk, strict=False)
        missing = [k for k in missing if not k.startswith("head.")]
        if missing or unexpected:
            raise RuntimeError(f"Cnn14 checkpoint mismatch: missing={missing} unexpected={unexpected}")
        self.source_checkpoint = str(path)

    def head_parameters(self):
        return self.head.parameters()

    def set_backbone_trainable(self, trainable: bool) -> None:
        for name, p in self.named_parameters():
            if not name.startswith("head."):
                p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, frames) -> Cnn14 layout (B, 1, frames, n_mels)
        x = x.transpose(2, 3)
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        p = 0.2 if self.training else 0.0
        for i in range(1, 7):
            pool = (2, 2) if i < 6 else (1, 1)
            x = getattr(self, f"conv_block{i}")(x, pool=pool)
            x = F.dropout(x, p=p, training=self.training)
        x = x.mean(dim=3)  # over mel
        x = x.max(dim=2).values + x.mean(dim=2)  # over time
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)
