import torch

from audioscope.models import (
    BaselineCNN,
    Cnn14Classifier,
    GradCAM,
    ResNet38Classifier,
    ResNetSpectrogram,
    build_model,
)


def test_baseline_forward_and_gradcam():
    m = BaselineCNN(num_classes=50)
    x = torch.randn(2, 1, 64, 216)
    assert m(x).shape == (2, 50)
    cam, logits = GradCAM(m, m.default_gradcam_layer)(x)
    assert cam.shape == (2, 64, 216) and logits.shape == (2, 50)
    assert cam.min() >= 0 and cam.max() <= 1 + 1e-6


def test_resnet_gradcam_projects_to_spectrogram_grid():
    m = ResNetSpectrogram(num_classes=5, pretrained=None, input_size=(64, 64))
    x = torch.randn(1, 1, 40, 100)
    assert m(x).shape == (1, 5)
    cam, _ = GradCAM(m, "backbone.layer4")(x, class_idx=2)
    assert cam.shape == (1, 40, 100)


def test_resnet_freeze():
    m = ResNetSpectrogram(num_classes=5, pretrained=None)
    m.set_backbone_trainable(False)
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable == {"backbone.fc.1.weight", "backbone.fc.1.bias"}


def test_cnn14_loads_trunk_from_checkpoint(tmp_path):
    src = Cnn14Classifier(num_classes=527)
    # Simulate the published checkpoint layout: trunk keys + an AudioSet head + front-end buffers.
    state = {k: v for k, v in src.state_dict().items() if not k.startswith("head.")}
    state["fc_audioset.weight"] = torch.zeros(527, 2048)
    state["logmel_extractor.melW"] = torch.zeros(1025, 64)
    torch.save({"model": state}, tmp_path / "cnn14.pth")
    m = Cnn14Classifier(num_classes=50, checkpoint_path=str(tmp_path / "cnn14.pth"))
    assert torch.equal(m.conv_block1.conv1.weight, src.conv_block1.conv1.weight)
    m.eval()
    assert m(torch.randn(2, 1, 64, 101)).shape == (2, 50)


def test_resnet38_loads_trunk_and_gradcam(tmp_path):
    src = ResNet38Classifier(num_classes=527)
    state = {k: v for k, v in src.state_dict().items() if not k.startswith("head.")}
    state["fc_audioset.weight"] = torch.zeros(527, 2048)
    torch.save({"model": state}, tmp_path / "resnet38.pth")
    m = ResNet38Classifier(num_classes=50, checkpoint_path=str(tmp_path / "resnet38.pth"))
    assert torch.equal(m.resnet.layer2[0].downsample[1].weight, src.resnet.layer2[0].downsample[1].weight)
    m.eval()
    x = torch.randn(2, 1, 64, 101)
    assert m(x).shape == (2, 50) and m.pooled(x).shape == (2, 2048)
    cam, _ = GradCAM(m, m.default_gradcam_layer)(x)
    assert cam.shape == (2, 64, 101)


def test_resnet38_checkpoint_mismatch_is_an_error(tmp_path):
    import pytest

    torch.save({"model": Cnn14Classifier(num_classes=527).state_dict()}, tmp_path / "cnn14.pth")
    with pytest.raises(RuntimeError, match="mismatch"):
        ResNet38Classifier(num_classes=50, checkpoint_path=str(tmp_path / "cnn14.pth"))


def test_build_model_names():
    assert isinstance(build_model({"name": "baseline_cnn", "channels": [4, 8]}, 3), BaselineCNN)
    assert isinstance(build_model({"name": "panns_resnet38"}, 3), ResNet38Classifier)


def test_frozen_backbone_keeps_batchnorm_stats():
    m = ResNetSpectrogram(num_classes=5, pretrained=None, input_size=(32, 32))
    m.set_backbone_trainable(False)
    m.train()
    before = m.backbone.bn1.running_mean.clone()
    m(torch.randn(4, 1, 20, 30))
    assert torch.equal(before, m.backbone.bn1.running_mean)
    m.set_backbone_trainable(True)
    m.train()
    m(torch.randn(4, 1, 20, 30))
    assert not torch.equal(before, m.backbone.bn1.running_mean)


def test_every_config_builds_its_model():
    """Each config in configs/ resolves to constructor arguments its model accepts."""
    from pathlib import Path

    from audioscope.config import load_config

    for path in sorted(Path(__file__).parents[1].joinpath("configs").glob("*.yaml")):
        cfg = load_config(path)
        model_cfg = dict(cfg["model"])
        # no weight files in the test: only the argument set is checked
        for key in ("weights_path", "checkpoint_path"):
            if key in model_cfg:
                model_cfg[key] = None
        if model_cfg.get("pretrained"):
            model_cfg["pretrained"] = None
        build_model(model_cfg, num_classes=50)


def test_gradcam_time_major_model_maps_back_to_mel_time():
    """A model whose conv maps are (time, mel) must get its CAM transposed to the input's (mel, time)."""
    import torch.nn as nn

    class TimeMajor(nn.Module):
        cam_time_major = True

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 1, bias=False)
            nn.init.ones_(self.conv.weight)

        def forward(self, x):  # x: (B, 1, mel, time)
            return self.conv(x.transpose(2, 3)).mean(dim=(2, 3)).repeat(1, 2)

    x = torch.zeros(1, 1, 8, 32)
    x[0, 0, 6, 20] = 1.0  # one bright cell at mel band 6, frame 20
    cam, _ = GradCAM(TimeMajor(), "conv")(x, 0)
    assert cam.shape == (1, 8, 32)
    assert tuple((cam[0] == cam[0].max()).nonzero()[0].tolist()) == (6, 20)
