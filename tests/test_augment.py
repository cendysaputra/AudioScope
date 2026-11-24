import numpy as np
import torch

from audioscope.train.augment import AugmentConfig, add_noise, mixup, random_time_shift, spec_augment


def test_time_shift_is_a_roll():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(3, 100)
    out = random_time_shift(w, 0.3, g)
    for i in range(3):
        assert any(torch.equal(out[i], torch.roll(w[i], s)) for s in range(-30, 31))
    assert torch.equal(random_time_shift(w, 0.0, g), w)


def test_noise_snr():
    g = torch.Generator().manual_seed(0)
    w = torch.sin(torch.linspace(0, 200, 20000)).repeat(8, 1)
    out = add_noise(w, 1.0, [20.0, 20.0], g)
    snr = 10 * torch.log10(w.pow(2).mean(1) / (out - w).pow(2).mean(1))
    assert torch.allclose(snr, torch.full_like(snr, 20.0), atol=0.5)
    assert torch.equal(add_noise(w, 0.0, [20.0, 20.0], g), w)


def test_specaugment_masks_and_disabled_noop():
    g = torch.Generator().manual_seed(0)
    x = torch.ones(4, 1, 64, 100)
    cfg = AugmentConfig(specaugment=True, freq_mask_param=10, time_mask_param=20)
    out = spec_augment(x, cfg, g)
    assert (out == 0).any() and (out == 1).any()
    assert torch.equal(x, torch.ones_like(x))  # input untouched
    assert spec_augment(x, AugmentConfig(), g) is x


def test_mixup_labels_sum_to_one():
    rng = np.random.default_rng(0)
    x = torch.randn(8, 1, 4, 4)
    y = torch.nn.functional.one_hot(torch.arange(8) % 3, 3).float()
    xm, ym = mixup(x, y, 0.4, 1.0, rng)
    assert torch.allclose(ym.sum(1), torch.ones(8))
    x2, y2 = mixup(x, y, 0.0, 1.0, rng)
    assert x2 is x and y2 is y


def test_augment_config_rejects_unknown():
    import pytest

    with pytest.raises(ValueError):
        AugmentConfig.from_dict({"specaugmnt": True})
