"""Small CNN trained from scratch on log-mel spectrograms (Stage 1)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, convs: int = 1, pool: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(convs):
            layers += [
                nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
        self.body = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(pool) if pool > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.body(x))


class BaselineCNN(nn.Module):
    """conv blocks -> global average pool -> dropout -> linear.

    ``blocks.<i>`` are the convolutional stages; Grad-CAM reads the last one
    before pooling (``blocks.<last>.body``) by default.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        channels: tuple[int, ...] | list[int] = (32, 64, 128, 256),
        convs_per_block: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        chans = [in_channels, *channels]
        self.blocks = nn.Sequential(
            *[ConvBlock(chans[i], chans[i + 1], convs_per_block) for i in range(len(channels))]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(chans[-1], num_classes)
        self.default_gradcam_layer = f"blocks.{len(channels) - 1}.body"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = x.mean(dim=(2, 3))
        return self.fc(self.dropout(x))
