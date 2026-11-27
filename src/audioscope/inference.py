"""Load a saved checkpoint and score audio with the exact front end it was trained with."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .features.audio import fix_length
from .features.frontend import FeatureConfig, mel_band_centers
from .models import GradCAM, build_model
from .train.loop import FeaturePipeline


@dataclass
class Prediction:
    labels: list[str]
    probs: list[float]
    indices: list[int]


class Predictor:
    def __init__(self, checkpoint: dict, device: str = "cpu"):
        self.ckpt = checkpoint
        self.device = torch.device(device)
        self.fcfg = FeatureConfig.from_dict(checkpoint["features"])
        self.class_names: list[str] = checkpoint["class_names"]
        model_cfg = dict(checkpoint["config"]["model"])
        # Weights come from the checkpoint; never re-download a pretrained source here.
        if model_cfg["name"] == "resnet":
            model_cfg["pretrained"] = None
        if model_cfg["name"] in ("panns_cnn14", "panns_resnet38"):
            model_cfg["checkpoint_path"] = None
        self.model = build_model(model_cfg, len(self.class_names)).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.pipe = FeaturePipeline(self.fcfg).to(self.device)
        self.pipe.norm.load_state_dict(checkpoint["normalizer_state"])
        self.gradcam_layer = checkpoint["gradcam_layer"]

    @classmethod
    def from_path(cls, path: str | Path, device: str = "cpu") -> "Predictor":
        return cls(torch.load(Path(path), map_location="cpu", weights_only=False), device)

    def prepare(self, wave: torch.Tensor) -> torch.Tensor:
        """Mono waveform at ``fcfg.sample_rate`` -> fixed-length ``(1, samples)``."""
        return fix_length(wave, self.fcfg.num_samples).unsqueeze(0).to(self.device)

    def features(self, wave: torch.Tensor) -> torch.Tensor:
        return self.pipe(self.prepare(wave))

    @torch.no_grad()
    def predict(self, wave: torch.Tensor, k: int = 5) -> Prediction:
        logits = self.model(self.features(wave))[0]
        probs = torch.softmax(logits, dim=0)
        top = probs.topk(min(k, len(probs)))
        idx = top.indices.tolist()
        return Prediction([self.class_names[i] for i in idx], top.values.tolist(), idx)

    def gradcam(self, wave: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        cam, _ = GradCAM(self.model, self.gradcam_layer)(self.features(wave), class_idx)
        return cam[0].cpu().numpy()

    def band_centers_hz(self) -> np.ndarray:
        return mel_band_centers(self.fcfg)
