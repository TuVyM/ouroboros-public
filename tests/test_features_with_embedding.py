
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _make_ohlcv(n=800):
    rng = np.random.default_rng(99)
    c = 50_000 * np.cumprod(1.0 + rng.normal(0, 0.005, n))
    h = c * 1.002; l = c * 0.998; o = np.roll(c, 1); o[0] = c[0]
    return np.column_stack([o, h, l, c, rng.uniform(100, 1000, n)]).astype(np.float32)


def _make_timestamps(ohlcv):
    return np.arange(len(ohlcv), dtype=np.int64) * 3600 + 1_700_000_000


def _make_encoder(n_out=64):
    enc = MagicMock()
    def _batch_embed(ohlcv, out_indices, **kw):
        return np.zeros((len(out_indices), n_out), dtype=np.float32)
    enc.batch_embed.side_effect = _batch_embed
    return enc


def test_build_features_with_embedding_shape():
    from orderflow.features import build_features_with_embedding, WINDOW
    ohlcv = _make_ohlcv(800)
    ts = _make_timestamps(ohlcv)
    enc = _make_encoder()
    with patch("orderflow.features.build_features") as mock_bf:
        M = 711
        mock_bf.return_value = (np.zeros((M, 24), dtype=np.float32), ts[WINDOW:WINDOW+M])
        X, out_ts = build_features_with_embedding("BTCUSDT", ohlcv, ts, enc)
    assert X.shape == (M, 88)
    assert X.dtype == np.float32


def test_build_features_with_embedding_correct_bar_indices():
    from orderflow.features import build_features_with_embedding, WINDOW
    ohlcv = _make_ohlcv(800)
    ts = _make_timestamps(ohlcv)
    enc = _make_encoder()
    captured = {}
    def _batch_embed(ohlcv_arg, out_indices, **kw):
        captured["indices"] = out_indices.copy()
        return np.zeros((len(out_indices), 64), dtype=np.float32)
    enc.batch_embed.side_effect = _batch_embed
    with patch("orderflow.features.build_features") as mock_bf:
        M = 711
        mock_bf.return_value = (np.zeros((M, 24), dtype=np.float32), ts[WINDOW:WINDOW+M])
        build_features_with_embedding("BTCUSDT", ohlcv, ts, enc)
    expected = np.arange(WINDOW, WINDOW + M)
    np.testing.assert_array_equal(captured["indices"], expected)


def test_build_range_features_with_embedding_shape():
    from orderflow.range_features import build_range_features_with_embedding, N_RANGE_FEATURES
    from orderflow.features import WINDOW
    ohlcv = _make_ohlcv(800)
    ts = _make_timestamps(ohlcv)
    enc = _make_encoder()
    M = 711
    va = [{"val": 1.0, "poc": 2.0, "vah": 3.0}] * M
    with patch("orderflow.range_features.build_range_features") as mock_brf:
        mock_brf.return_value = (np.zeros((M, N_RANGE_FEATURES), dtype=np.float32), ts[WINDOW:WINDOW+M], va)
        X, out_ts, va_out = build_range_features_with_embedding("BTCUSDT", ohlcv, ts, enc)
    assert X.shape == (M, 93)
    assert X.dtype == np.float32
    assert len(va_out) == M


def test_build_features_with_embedding_passes_encoder():
    from orderflow.features import build_features_with_embedding, WINDOW
    ohlcv = _make_ohlcv(800)
    ts = _make_timestamps(ohlcv)
    enc = _make_encoder()
    with patch("orderflow.features.build_features") as mock_bf:
        M = 50
        mock_bf.return_value = (np.zeros((M, 24), dtype=np.float32), ts[WINDOW:WINDOW+M])
        build_features_with_embedding("BTCUSDT", ohlcv, ts, enc)
    assert enc.batch_embed.called
