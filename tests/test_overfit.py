"""A tiny model must be able to memorize a handful of clips, and the whole
train -> checkpoint -> Predictor path must round-trip on synthetic audio."""

import torch

from audioscope.inference import Predictor
from audioscope.train.loop import FeaturePipeline, train_fold
from audioscope.features import FeatureConfig
from audioscope.models import BaselineCNN
from audioscope.utils import get_logger

from conftest import make_synthetic


def test_tiny_overfit():
    torch.manual_seed(0)
    cat, waves = make_synthetic(num_classes=4, per_class_per_fold=1, folds=2)
    cfg = FeatureConfig(sample_rate=8000, clip_seconds=1.0, n_mels=32)
    pipe = FeaturePipeline(cfg)
    pipe.fit(waves)
    x = pipe(waves)
    y = torch.as_tensor(cat.targets)
    model = BaselineCNN(num_classes=4, channels=[8, 16], dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(60):
        loss = torch.nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    assert (model(x).argmax(1) == y).all()
    assert loss.item() < 0.1


def test_train_fold_and_predictor_roundtrip(tmp_path):
    cat, waves = make_synthetic(num_classes=3, per_class_per_fold=4, folds=5)
    cfg = {
        "dataset": {"name": "synthetic", "root": "."},
        "features": {"sample_rate": 8000, "clip_seconds": 1.0, "n_mels": 32, "n_fft": 256, "hop_length": 128},
        "model": {"name": "baseline_cnn", "channels": [8, 16], "dropout": 0.0},
        "augment": {"specaugment": True, "mixup_alpha": 0.2, "time_shift": 0.2, "noise_prob": 0.5},
        "train": {"epochs": 25, "batch_size": 8, "lr": 3e-3, "warmup_epochs": 1, "val_fraction": 0.25},
    }
    res = train_fold(cfg, cat, waves, 3, torch.device("cpu"), get_logger("test"))
    assert res.test_logits.shape == (12, 3)
    assert (res.test_logits.argmax(1) == res.test_targets).mean() > 0.6
    path = tmp_path / "m.pt"
    torch.save(res.checkpoint, path)
    pred = Predictor.from_path(path)
    test_wave = waves[res.test_indices[0]]
    out = pred.predict(test_wave, k=2)
    assert len(out.labels) == 2 and abs(sum(out.probs) - 1) <= 1 and out.probs[0] >= out.probs[1]
    # Same checkpoint, same audio -> same logits as the training-side pipeline.
    expected = torch.softmax(torch.from_numpy(res.test_logits[0]), 0).max().item()
    assert abs(out.probs[0] - expected) < 1e-4
    cam = pred.gradcam(test_wave)
    assert cam.shape == (32, pred.fcfg.num_frames)


def test_frozen_embedding_head(tmp_path):
    from audioscope.models import Cnn14Classifier

    cat, waves = make_synthetic(num_classes=3, per_class_per_fold=4, folds=5)
    feats = {"sample_rate": 8000, "clip_seconds": 1.0, "n_mels": 32, "n_fft": 256, "hop_length": 128,
             "log": "db", "normalize": "none"}
    src = Cnn14Classifier(num_classes=527, n_mels=32)
    torch.save({"model": src.state_dict()}, tmp_path / "cnn14.pth")
    cfg = {
        "dataset": {"name": "synthetic", "root": "."},
        "features": feats,
        "model": {"name": "panns_cnn14", "n_mels": 32, "checkpoint_path": str(tmp_path / "cnn14.pth")},
        "augment": {},
        "train": {"mode": "frozen_embeddings", "epochs": 40, "batch_size": 16, "lr": 1e-2, "val_fraction": 0.25},
    }
    res = train_fold(cfg, cat, waves, 2, torch.device("cpu"), get_logger("test"), cache_dir=tmp_path)
    assert res.test_logits.shape == (12, 3)
    assert list(tmp_path.glob("emb_*.pt"))  # embeddings cached once for all folds
    # Trunk untouched: identical to the loaded checkpoint.
    state = res.checkpoint["model_state"]
    assert torch.equal(state["conv_block1.conv1.weight"], src.state_dict()["conv_block1.conv1.weight"])
    assert torch.equal(state["bn0.running_mean"], src.state_dict()["bn0.running_mean"])
    pred = Predictor(res.checkpoint)
    out = pred.predict(waves[res.test_indices[0]], k=1)
    expected = torch.softmax(torch.from_numpy(res.test_logits[0]), 0).max().item()
    assert abs(out.probs[0] - expected) < 1e-4
