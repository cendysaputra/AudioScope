"""PANNs Cnn14 and ResNet38 (Kong et al., 2020) with a new classification head (Stage 3).

The convolutional trunk mirrors the published Cnn14 so its AudioSet
checkpoint (``Cnn14_mAP=0.431.pth``, Zenodo record 3987831) loads by key
name. The checkpoint expects 32 kHz audio, n_fft 1024, hop 320, 64 mel bands
from 50 Hz to 14 kHz, and a dB log-mel without external normalization (the
model's own ``bn0`` normalizes per band); ``configs/transfer-panns-cnn14.yaml``
sets exactly that front end. ``Cnn14_16k_mAP=0.438.pth`` is the same network
trained on a 16 kHz front end (n_fft 512, hop 160, 50 Hz to 8 kHz), and
``ResNet38_mAP=0.434.pth`` shares Cnn14's 32 kHz front end and its pooling and
``fc1`` embedding, with a residual trunk in between.
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


class _PannsClassifier(nn.Module):
    """Shared PANNs head logic: ``bn0`` -> trunk -> mean over mel, max + mean over time -> ``fc1`` -> head.

    Subclasses build the trunk and implement :meth:`_trunk`. The same interface
    serves fine-tuning and the frozen-trunk probe (``pooled`` features, ``fc1``, ``head``).
    """

    TRUNK_PREFIXES: tuple[str, ...] = ()
    default_gradcam_layer = ""

    def __init__(self, num_classes: int, n_mels: int, dropout: float):
        super().__init__()
        self._backbone_frozen = False
        self.cam_time_major = True  # conv maps are (time, mel); Grad-CAM transposes them
        self.bn0 = nn.BatchNorm2d(n_mels)
        self.fc1 = nn.Linear(2048, 2048)
        self.head = nn.Linear(2048, num_classes)
        self.dropout = dropout
        self.source_checkpoint = None

    def load_audioset(self, path: str | Path) -> None:
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        trunk = {k: v for k, v in state.items() if k.startswith(self.TRUNK_PREFIXES)}
        missing, unexpected = self.load_state_dict(trunk, strict=False)
        missing = [k for k in missing if not k.startswith("head.")]
        if missing or unexpected:
            raise RuntimeError(f"{type(self).__name__} checkpoint mismatch: missing={missing} unexpected={unexpected}")
        self.source_checkpoint = str(path)

    def head_parameters(self):
        return self.head.parameters()

    def set_backbone_trainable(self, trainable: bool) -> None:
        self._backbone_frozen = not trainable
        for name, p in self.named_parameters():
            if not name.startswith("head."):
                p.requires_grad = trainable
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if self._backbone_frozen:
            # Frozen trunk: pretrained BatchNorm statistics, no trunk dropout.
            for name, m in self.named_modules():
                if name and not name.startswith("head"):
                    m.eval()
        return self

    @property
    def _trunk_training(self) -> bool:
        return self.training and not self._backbone_frozen

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        """2048-d pooled conv features, the input of ``fc1``."""
        # x: (B, 1, n_mels, frames) -> PANNs layout (B, 1, frames, n_mels)
        x = x.transpose(2, 3)
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        x = self._trunk(x)
        x = x.mean(dim=3)  # over mel
        return x.max(dim=2).values + x.mean(dim=2)  # over time

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """2048-d clip embedding (ReLU(fc1)), the PANNs embedding layer."""
        x = F.dropout(self.pooled(x), p=self.dropout, training=self._trunk_training)
        return F.relu_(self.fc1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)


class Cnn14Classifier(_PannsClassifier):
    TRUNK_PREFIXES = ("bn0.", "conv_block", "fc1.")
    default_gradcam_layer = "conv_block6"

    def __init__(self, num_classes: int, n_mels: int = 64, checkpoint_path: str | None = None, dropout: float = 0.5):
        super().__init__(num_classes, n_mels, dropout)
        widths = [1, 64, 128, 256, 512, 1024, 2048]
        for i in range(6):
            setattr(self, f"conv_block{i + 1}", _ConvBlock(widths[i], widths[i + 1]))
        if checkpoint_path:
            self.load_audioset(checkpoint_path)

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        p = 0.2 if self._trunk_training else 0.0
        for i in range(1, 7):
            pool = (2, 2) if i < 6 else (1, 1)
            x = getattr(self, f"conv_block{i}")(x, pool=pool)
            x = F.dropout(x, p=p, training=self._trunk_training)
        return x


class _BasicBlock(nn.Module):
    """PANNs residual block: average-pool (not strided conv) when downsampling."""

    def __init__(self, inplanes: int, planes: int, stride: int = 1):
        super().__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(inplanes, planes, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride == 2:
            self.downsample = nn.Sequential(nn.AvgPool2d(2), nn.Conv2d(inplanes, planes, 1, bias=False),
                                            nn.BatchNorm2d(planes))
        elif inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, 1, bias=False), nn.BatchNorm2d(planes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.avg_pool2d(x, 2) if self.stride == 2 else x
        out = F.relu(self.bn1(self.conv1(out)))
        out = F.dropout(out, p=0.1, training=self.training)
        out = self.bn2(self.conv2(out))
        identity = self.downsample(x) if self.downsample is not None else x
        return F.relu(out + identity)


class _ResNetTrunk(nn.Module):
    def __init__(self, layers: tuple[int, ...] = (3, 4, 6, 3)):
        super().__init__()
        inplanes = 64
        for i, (planes, n) in enumerate(zip((64, 128, 256, 512), layers)):
            blocks = []
            for j in range(n):
                blocks.append(_BasicBlock(inplanes, planes, stride=2 if (i > 0 and j == 0) else 1))
                inplanes = planes
            setattr(self, f"layer{i + 1}", nn.Sequential(*blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


class ResNet38Classifier(_PannsClassifier):
    """PANNs ResNet38 (AudioSet mAP 0.434): Cnn14's first block, a ResNet-34 trunk, a 512 -> 2048 block."""

    TRUNK_PREFIXES = ("bn0.", "conv_block1.", "resnet.", "conv_block_after1.", "fc1.")
    default_gradcam_layer = "conv_block_after1"

    def __init__(self, num_classes: int, n_mels: int = 64, checkpoint_path: str | None = None, dropout: float = 0.5):
        super().__init__(num_classes, n_mels, dropout)
        self.conv_block1 = _ConvBlock(1, 64)
        self.resnet = _ResNetTrunk((3, 4, 6, 3))
        self.conv_block_after1 = _ConvBlock(512, 2048)
        if checkpoint_path:
            self.load_audioset(checkpoint_path)

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        t = self._trunk_training
        x = F.dropout(self.conv_block1(x, pool=(2, 2)), p=0.2, training=t)
        x = F.dropout(F.avg_pool2d(self.resnet(x), 2), p=0.2, training=t)
        return F.dropout(self.conv_block_after1(x, pool=(1, 1)), p=0.2, training=t)
