
import numpy as np
import pytest
import torch

from transformer.finetune import (
    FinetuneDataset,
    build_regime_labels,
    freeze_bottom_layers,
)
from transformer.encoder import MultiScaleEncoder


def _make_ohlcv(n=500):
    rng = np.random.default_rng(7)
    c = 50_000 * np.cumprod(1.0 + rng.normal(0, 0.005, n))
    h = c * (1 + rng.uniform(0.001, 0.005, n))
    l = c * (1 - rng.uniform(0.001, 0.005, n))
    o = np.roll(c, 1); o[0] = c[0]
    v = rng.uniform(100, 1000, n)
    return np.column_stack([o, h, l, c, v]).astype(np.float32)


def test_finetune_dataset_shape():
    ohlcv = _make_ohlcv(500)
    bar_indices = np.array([89, 100, 200, 300])
    labels = np.array([1, 2, 0, 1], dtype=np.int64)
    ds = FinetuneDataset(ohlcv, bar_indices, labels)
    window, label = ds[0]
    assert window.shape == (720, 5)
    assert label.shape == ()
    assert label.dtype == torch.int64


def test_finetune_dataset_pads_short_windows():
    ohlcv = _make_ohlcv(200)   # only 200 bars
    bar_indices = np.array([89])  # only 90 bars of history
    labels = np.array([1], dtype=np.int64)
    ds = FinetuneDataset(ohlcv, bar_indices, labels)
    window, _ = ds[0]
    assert window.shape == (720, 5)
    # First 630 positions should be zero-padded
    assert (window[:630] == 0.0).all()


def test_build_regime_labels_routing():
    # For trending bars: trend label; for ranging bars: range label
    n = 200
    trend_labels = np.full(n, 1, dtype=np.int8)    # all HOLD
    range_labels = np.full(n, 2, dtype=np.int8)    # all BUY
    regime_mask = np.zeros(n, dtype=bool)
    regime_mask[10:20] = True  # bars 10..19 are ranging
    combined = build_regime_labels(trend_labels, range_labels, regime_mask)
    assert combined[5] == 1    # trending → trend label (HOLD)
    assert combined[15] == 2   # ranging → range label (BUY)
    assert combined.dtype == np.int64


def test_freeze_bottom_layers():
    model = MultiScaleEncoder()
    freeze_bottom_layers(model, n_freeze=2)
    # Bottom 2 layers of enc_1h should have no grad
    for layer in model.enc_1h.transformer.layers[:2]:
        for p in layer.parameters():
            assert not p.requires_grad
    # Top 2 layers of enc_1h should still have grad
    for layer in model.enc_1h.transformer.layers[2:]:
        for p in layer.parameters():
            assert p.requires_grad
    # Fusion should have grad
    for p in model.fusion.parameters():
        assert p.requires_grad


def test_freeze_also_freezes_input_proj():
    model = MultiScaleEncoder()
    freeze_bottom_layers(model, n_freeze=2)
    for p in model.enc_1h.input_proj.parameters():
        assert not p.requires_grad
    assert not model.enc_1h.pos_embed.requires_grad
