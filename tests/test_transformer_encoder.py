import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch

from transformer.encoder import MultiScaleEncoder, TransformerEncoder


def test_transformer_encoder_output_shape():
    enc = TransformerEncoder(seq_len=168)
    x = torch.randn(2, 168, 5)
    out = enc(x)
    assert out.shape == (2, 64)


def test_transformer_encoder_return_sequence():
    enc = TransformerEncoder(seq_len=42)
    x = torch.randn(2, 42, 5)
    out = enc(x, return_sequence=True)
    assert out.shape == (2, 42, 64)


def test_multiscale_encoder_output_shape():
    enc = MultiScaleEncoder()
    x = torch.randn(2, 720, 5)
    out = enc(x)
    assert out.shape == (2, 64)


def test_znorm_zero_mean_unit_std():
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]] * 10])  # (1, 10, 5)
    normed = MultiScaleEncoder._znorm(x)
    assert abs(float(normed.mean())) < 1e-5


def test_resample_volume_sums():
    x = torch.ones(1, 168, 5)
    x[0, :, 4] = 2.0  # volume = 2 per 1h bar → 8 per 4h bar
    out = MultiScaleEncoder._resample(x, 4)
    assert out.shape == (1, 42, 5)
    assert abs(float(out[0, 0, 4]) - 8.0) < 1e-5


def test_resample_open_close():
    x = torch.zeros(1, 24, 5)
    x[0, 0, 0] = 3.0   # open of first 1h bar = 3
    x[0, 23, 3] = 7.0  # close of last 1h bar = 7
    out = MultiScaleEncoder._resample(x, 24)
    assert out.shape == (1, 1, 5)
    assert abs(float(out[0, 0, 0]) - 3.0) < 1e-5  # open = first
    assert abs(float(out[0, 0, 3]) - 7.0) < 1e-5  # close = last


def test_embed_returns_float32_numpy():
    enc = MultiScaleEncoder()
    x = np.random.randn(720, 5).astype(np.float32)
    emb = enc.embed(x)
    assert emb.shape == (64,)
    assert emb.dtype == np.float32


def test_embed_pads_short_input():
    enc = MultiScaleEncoder()
    x = np.random.randn(200, 5).astype(np.float32)
    emb = enc.embed(x)  # should not raise
    assert emb.shape == (64,)


def test_batch_embed_shape():
    enc = MultiScaleEncoder()
    ohlcv = np.random.randn(300, 5).astype(np.float32)
    # out_indices: the ohlcv bar indices for 5 output rows
    out_indices = np.array([89, 100, 150, 200, 299])
    embs = enc.batch_embed(ohlcv, out_indices)
    assert embs.shape == (5, 64)
    assert embs.dtype == np.float32


def test_batch_embed_alignment():
    # Verify that changing bar 150 does not affect embedding for out_index 89
    enc = MultiScaleEncoder()
    ohlcv = np.random.randn(300, 5).astype(np.float32)
    indices = np.array([89, 150])
    embs1 = enc.batch_embed(ohlcv, indices)
    ohlcv_modified = ohlcv.copy()
    ohlcv_modified[200:] *= 999  # change bars after index 150
    embs2 = enc.batch_embed(ohlcv_modified, indices)
    np.testing.assert_allclose(embs1, embs2, atol=1e-5)
