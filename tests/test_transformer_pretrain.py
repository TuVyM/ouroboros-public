import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch

from transformer.pretrain import OHLCVDataset, apply_mask, make_train_val_split


def _make_ohlcv(n=1000):
    rng = np.random.default_rng(42)
    c = 50_000 * np.cumprod(1.0 + rng.normal(0, 0.005, n))
    h = c * (1 + rng.uniform(0.001, 0.005, n))
    l = c * (1 - rng.uniform(0.001, 0.005, n))
    o = np.roll(c, 1); o[0] = c[0]
    v = rng.uniform(100, 1000, n)
    return np.column_stack([o, h, l, c, v]).astype(np.float32)


def test_dataset_length():
    ohlcv = _make_ohlcv(1000)
    ds = OHLCVDataset(ohlcv)
    # Samples where we have at least 720 bars of history: indices 719..999 → 281 samples
    assert len(ds) == 1000 - 720 + 1


def test_dataset_item_shape():
    ohlcv = _make_ohlcv(1000)
    ds = OHLCVDataset(ohlcv)
    item = ds[0]
    assert item.shape == (720, 5)
    assert item.dtype == torch.float32


def test_dataset_last_item_is_last_720_bars():
    ohlcv = _make_ohlcv(1000)
    ds = OHLCVDataset(ohlcv)
    last = ds[-1].numpy()
    expected = ohlcv[-720:]
    np.testing.assert_allclose(last, expected)


def test_apply_mask_ratio():
    x = torch.ones(4, 168, 5)
    x_masked, mask = apply_mask(x, mask_ratio=0.20)
    # mask should be roughly 20% True (allow ±10% slack for small batches)
    frac = mask.float().mean().item()
    assert 0.10 < frac < 0.30


def test_apply_mask_zeroes_positions():
    x = torch.ones(2, 10, 5)
    x_masked, mask = apply_mask(x, mask_ratio=0.5)
    # Masked positions should be zero
    assert (x_masked[mask] == 0.0).all()
    # Unmasked positions should be 1.0
    assert (x_masked[~mask] == 1.0).all()


def test_train_val_split_gap():
    ohlcv = _make_ohlcv(2000)
    train_ds, val_ds = make_train_val_split(ohlcv, val_frac=0.10, gap=168)
    # No overlap: last train index + gap < first val index
    last_train_idx = train_ds.indices[-1]
    first_val_idx = val_ds.indices[0]
    assert first_val_idx - last_train_idx > 168


def test_train_val_split_sizes():
    ohlcv = _make_ohlcv(2000)
    train_ds, val_ds = make_train_val_split(ohlcv, val_frac=0.10, gap=168)
    assert len(train_ds) > len(val_ds)
    assert len(val_ds) > 0
