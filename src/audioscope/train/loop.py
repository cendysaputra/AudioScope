"""Train one model for one official fold.

The test fold is touched exactly once, by :func:`predict`, after training
has finished and the checkpoint has been selected on the validation split
(which comes from the training folds only).
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.catalog import AudioCatalog
from ..data.dataset import WaveformDataset, split_indices
from ..features.frontend import FeatureConfig, Normalizer, build_frontend
from ..models import build_model, gradcam_layer
from ..utils import git_revision, seed_everything
from .augment import AugmentConfig, add_noise, mixup, random_time_shift, spec_augment

TRAIN_DEFAULTS = {
    "seed": 0,
    "epochs": 60,
    "batch_size": 32,
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "warmup_epochs": 2,
    "val_fraction": 0.1,
    "early_stopping_patience": None,
    "freeze_backbone_epochs": 0,
    "label_smoothing": 0.0,
}


@dataclass
class FoldResult:
    test_fold: int
    test_indices: np.ndarray
    test_targets: np.ndarray
    test_logits: np.ndarray
    best_epoch: int
    best_val_accuracy: float | None
    epochs_run: int
    train_seconds: float
    history: list[dict] = field(default_factory=list)
    checkpoint: dict | None = None


class FeaturePipeline(nn.Module):
    """waveform -> features -> normalized ``(B, 1, F, T)``."""

    def __init__(self, fcfg: FeatureConfig):
        super().__init__()
        self.fcfg = fcfg
        self.frontend = build_frontend(fcfg)
        self.norm = Normalizer(fcfg.n_features, fcfg.normalize)

    @torch.no_grad()
    def raw(self, wave: torch.Tensor) -> torch.Tensor:
        return self.frontend(wave)

    @torch.no_grad()
    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        return self.norm(self.frontend(wave)).unsqueeze(1)

    @torch.no_grad()
    def fit(self, waves: torch.Tensor, batch_size: int = 128) -> None:
        feats = torch.cat([self.frontend(waves[i : i + batch_size]) for i in range(0, len(waves), batch_size)])
        self.norm.fit(feats)


def _lr_lambda(warmup_steps: int, total_steps: int):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return fn


@torch.no_grad()
def predict(model: nn.Module, pipe: FeaturePipeline, waves: torch.Tensor, device, batch_size: int = 64) -> torch.Tensor:
    model.eval()
    out = []
    for i in range(0, len(waves), batch_size):
        x = pipe(waves[i : i + batch_size].to(device))
        out.append(model(x).float().cpu())
    return torch.cat(out)


def train_fold(
    cfg: dict,
    catalog: AudioCatalog,
    waves: torch.Tensor,
    test_fold: int,
    device: torch.device,
    logger,
    run_id: str | None = None,
    evaluate_test: bool = True,
) -> FoldResult:
    tcfg = {**TRAIN_DEFAULTS, **cfg.get("train", {})}
    fcfg = FeatureConfig.from_dict(cfg.get("features"))
    acfg = AugmentConfig.from_dict(cfg.get("augment"))
    seed = int(tcfg["seed"]) * 100 + test_fold
    seed_everything(seed)
    gen = torch.Generator().manual_seed(seed)
    np_rng = np.random.default_rng(seed)

    train_idx, val_idx, test_idx = split_indices(catalog, test_fold, tcfg["val_fraction"], seed=seed)
    logger.info(
        "fold %d: train=%d val=%d test=%d (seed %d)", test_fold, len(train_idx), len(val_idx), len(test_idx), seed
    )

    pipe = FeaturePipeline(fcfg).to(device)
    pipe.fit(waves[train_idx].to(device))  # statistics from training clips only

    num_classes = catalog.num_classes
    model = build_model(cfg["model"], num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("model %s: %.2fM parameters", cfg["model"]["name"], n_params / 1e6)

    freeze_epochs = int(tcfg["freeze_backbone_epochs"] or 0)
    if freeze_epochs and hasattr(model, "set_backbone_trainable"):
        model.set_backbone_trainable(False)
        logger.info("backbone frozen for the first %d epochs", freeze_epochs)

    train_ds = WaveformDataset(waves, catalog.targets, train_idx)
    loader = DataLoader(
        train_ds,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    steps_per_epoch = len(loader)
    total = steps_per_epoch * tcfg["epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(steps_per_epoch * tcfg["warmup_epochs"], total))

    val_waves = waves[val_idx]
    val_targets = torch.as_tensor(catalog.targets[val_idx])
    best_state, best_epoch, best_val, best_val_loss = None, -1, -1.0, float("inf")
    patience = tcfg["early_stopping_patience"]
    history: list[dict] = []
    t0 = time.time()
    epoch = 0
    for epoch in range(1, tcfg["epochs"] + 1):
        if freeze_epochs and epoch == freeze_epochs + 1:
            model.set_backbone_trainable(True)
            logger.info("backbone unfrozen at epoch %d", epoch)
        model.train()
        loss_sum, n_seen, correct = 0.0, 0, 0
        for wave, y in loader:
            wave = random_time_shift(wave, acfg.time_shift, gen)
            wave = add_noise(wave, acfg.noise_prob, acfg.noise_snr_db, gen)
            x = pipe(wave.to(device))
            x = spec_augment(x, acfg, gen)
            y = y.to(device)
            y_soft = F.one_hot(y, num_classes).float()
            if tcfg["label_smoothing"]:
                y_soft = y_soft * (1 - tcfg["label_smoothing"]) + tcfg["label_smoothing"] / num_classes
            x, y_soft = mixup(x, y_soft, acfg.mixup_alpha, acfg.mixup_prob, np_rng)
            logits = model(x)
            loss = torch.sum(-y_soft * F.log_softmax(logits, dim=1), dim=1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(y)
            n_seen += len(y)
            correct += (logits.argmax(1) == y).sum().item()
        rec = {"epoch": epoch, "train_loss": loss_sum / n_seen, "train_acc": correct / n_seen, "lr": sched.get_last_lr()[0]}
        if len(val_idx):
            vlogits = predict(model, pipe, val_waves, device)
            rec["val_loss"] = F.cross_entropy(vlogits, val_targets).item()
            rec["val_acc"] = (vlogits.argmax(1) == val_targets).float().mean().item()
            improved = rec["val_acc"] > best_val or (rec["val_acc"] == best_val and rec["val_loss"] < best_val_loss)
            if improved:
                best_val, best_val_loss, best_epoch = rec["val_acc"], rec["val_loss"], epoch
                best_state = copy.deepcopy(model.state_dict())
        history.append(rec)
        logger.info(
            "fold %d epoch %3d  loss %.4f  acc %.3f%s  (%.0fs)",
            test_fold,
            epoch,
            rec["train_loss"],
            rec["train_acc"],
            f"  val_loss {rec['val_loss']:.4f}  val_acc {rec['val_acc']:.3f}" if "val_acc" in rec else "",
            time.time() - t0,
        )
        if patience and best_epoch > 0 and epoch - best_epoch >= patience:
            logger.info("early stop at epoch %d (best %d)", epoch, best_epoch)
            break
    train_seconds = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epoch
    logger.info("fold %d: selected epoch %d (val_acc %s)", test_fold, best_epoch, f"{best_val:.4f}" if best_val >= 0 else "n/a")

    test_logits = predict(model, pipe, waves[test_idx], device).numpy() if evaluate_test else np.zeros((0, num_classes))

    checkpoint = {
        "format": "audioscope-checkpoint-v1",
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "normalizer_state": {k: v.cpu() for k, v in pipe.norm.state_dict().items()},
        "config": cfg,
        "features": fcfg.to_dict(),
        "model_name": cfg["model"]["name"],
        "gradcam_layer": gradcam_layer(cfg["model"], model),
        "source_checkpoint": getattr(model, "source_checkpoint", None),
        "class_names": catalog.class_names,
        "dataset": catalog.name,
        "held_out_fold": test_fold,
        "best_epoch": best_epoch,
        "run_id": run_id,
        "git_revision": git_revision(),
    }
    return FoldResult(
        test_fold=test_fold,
        test_indices=test_idx,
        test_targets=catalog.targets[test_idx],
        test_logits=test_logits,
        best_epoch=best_epoch,
        best_val_accuracy=best_val if best_val >= 0 else None,
        epochs_run=epoch,
        train_seconds=train_seconds,
        history=history,
        checkpoint=checkpoint,
    )


def save_checkpoint(checkpoint: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
