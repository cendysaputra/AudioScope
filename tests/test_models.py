import torch

from audioscope.models import BaselineCNN, Cnn14Classifier, GradCAM, ResNetSpectrogram, build_model


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


def test_build_model_names():
    assert isinstance(build_model({"name": "baseline_cnn", "channels": [4, 8]}, 3), BaselineCNN)
