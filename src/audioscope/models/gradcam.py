"""Grad-CAM (Selvaraju et al., 2017) projected onto the input spectrogram grid."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Compute a class-activation map from one named convolutional layer.

    The map is upsampled to the *model input* size ``(n_features, frames)``.
    Models that resize internally (the ImageNet ResNet) therefore still get a
    map on the original mel-by-time grid.
    """

    def __init__(self, model: nn.Module, layer_name: str):
        self.model = model
        self.layer_name = layer_name
        self.layer = dict(model.named_modules())[layer_name]
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None

    def _hook(self, _module, _inputs, output):
        self._acts = output
        output.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self._grads = grad

    def __call__(self, x: torch.Tensor, class_idx: int | torch.Tensor | None = None):
        """``x``: ``(B, 1, H, W)``. Returns ``(cam[B, H, W] in [0, 1], logits)``."""
        was_training = self.model.training
        self.model.eval()
        handle = self.layer.register_forward_hook(self._hook)
        try:
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                logits = self.model(x)
                if class_idx is None:
                    class_idx = logits.argmax(dim=1)
                class_idx = torch.as_tensor(class_idx, device=logits.device).reshape(-1)
                if class_idx.numel() == 1 and logits.shape[0] > 1:
                    class_idx = class_idx.expand(logits.shape[0])
                score = logits.gather(1, class_idx.view(-1, 1)).sum()
                self.model.zero_grad(set_to_none=True)
                score.backward()
        finally:
            handle.remove()
            self.model.train(was_training)
        acts, grads = self._acts.detach(), self._grads.detach()
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        if getattr(self.model, "cam_time_major", False):
            # The model's conv maps are (time, mel), e.g. Cnn14; bring them to the input's (mel, time).
            cam = cam.transpose(-1, -2)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        flat = cam.flatten(1)
        lo = flat.min(dim=1).values.view(-1, 1, 1)
        hi = flat.max(dim=1).values.view(-1, 1, 1)
        cam = (cam - lo) / (hi - lo).clamp_min(1e-8)
        return cam, logits.detach()
