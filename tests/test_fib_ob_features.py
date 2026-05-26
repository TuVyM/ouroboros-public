# tests/test_fib_ob_features.py
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orderflow.fib_ob import (
    _fib_time_ob_features_batch,
    _fib_channel_features_batch,
    _ict_features_batch,
    _tpo_features_batch,
    build_structural_features,
)

_RNG = np.random.default_rng(42)


def _make_windows(B=8, T=89):
    """Synthetic OHLCV windows. close is a random walk; high/low bracket it."""
    c = 50_000 + _RNG.standard_normal((B, T)).cumsum(axis=1) * 200
    o = c + _RNG.standard_normal((B, T)) * 50
    hl = np.abs(_RNG.standard_normal((B, T))) * 100 + 50
    h = np.maximum(c, o) + hl
    l = np.minimum(c, o) - hl
    v = _RNG.uniform(1_000, 10_000, (B, T))
    return np.stack([o, h, l, c, v], axis=2).astype(np.float32)


def test_ob_fib_output_shape():
    w = _make_windows()
    out = _fib_time_ob_features_batch(w)
    assert out.shape == (8, 5), f"Expected (8,5), got {out.shape}"


def test_ob_fib_values_finite():
    out = _fib_time_ob_features_batch(_make_windows(B=16))
    assert np.isfinite(out).all(), "OB-Fib features contain inf/nan"


def test_ob_fib_values_clipped():
    out = _fib_time_ob_features_batch(_make_windows(B=16))
    assert out.min() >= -2.01 and out.max() <= 2.01, "OB-Fib output outside [-2, 2]"


def test_fib_channel_shape():
    out = _fib_channel_features_batch(_make_windows())
    assert out.shape == (8, 3)


def test_fib_channel_finite():
    assert np.isfinite(_fib_channel_features_batch(_make_windows(B=16))).all()


def test_ict_shape():
    out = _ict_features_batch(_make_windows())
    assert out.shape == (8, 3)


def test_ict_finite():
    assert np.isfinite(_ict_features_batch(_make_windows(B=16))).all()


def test_ict_sweep_bounded():
    out = _ict_features_batch(_make_windows(B=32))
    assert set(out[:, 1].tolist()).issubset({-1.0, 0.0, 1.0}), \
        "Sweep feature should be -1, 0, or +1"


def test_ict_killzone_with_timestamps():
    w = _make_windows()
    # London open at 08:00 UTC = 8 * 3600 * 1000 ms
    ts_ms = np.full(8, 8 * 3600 * 1000, dtype=np.int64)
    out = _ict_features_batch(w, timestamps_ms=ts_ms)
    np.testing.assert_allclose(out[:, 2], 0.5, err_msg="London open should give killzone=0.5")


def test_tpo_shape():
    out = _tpo_features_batch(_make_windows())
    assert out.shape == (8, 3)


def test_tpo_finite():
    assert np.isfinite(_tpo_features_batch(_make_windows(B=16))).all()


def test_tpo_poc_within_range():
    """POC should be within the window range → (close-POC)/range roughly in [-1,1]."""
    out = _tpo_features_batch(_make_windows(B=32))
    assert out[:, 0].min() > -2.0 and out[:, 0].max() < 2.0


def test_tpo_vah_above_val():
    """VAH should be above VAL for every window → tpo_vah < tpo_val (both relative to close)."""
    w = _make_windows(B=32)
    out = _tpo_features_batch(w)
    # tpo_vah = (close - VAH)/range, tpo_val = (close - VAL)/range
    # VAH > VAL → close - VAH < close - VAL → tpo_vah < tpo_val
    assert (out[:, 1] <= out[:, 2]).all(), "VAH should be >= VAL (tpo_vah <= tpo_val)"


def test_build_structural_features_shape():
    ohlcv = np.random.default_rng(0).random((200, 5)).astype(np.float32)
    ohlcv[:, 3] = 50_000 + np.cumsum(ohlcv[:, 3] - 0.5) * 100  # close = random walk
    ohlcv[:, 1] = ohlcv[:, 3] + np.abs(ohlcv[:, 0]) * 50       # high > close
    ohlcv[:, 2] = ohlcv[:, 3] - np.abs(ohlcv[:, 0]) * 50       # low < close
    ts_s = np.arange(200, dtype=np.int64) * 3600
    out = build_structural_features(ohlcv, ts_s, window=89)
    assert out.shape == (200 - 89, 14), f"Expected ({200-89}, 14), got {out.shape}"


def test_build_structural_features_finite():
    ohlcv = np.random.default_rng(1).random((200, 5)).astype(np.float32)
    ohlcv[:, 3] = 50_000 + np.cumsum(ohlcv[:, 3] - 0.5) * 100
    ohlcv[:, 1] = ohlcv[:, 3] + 50
    ohlcv[:, 2] = ohlcv[:, 3] - 50
    ts_s = np.arange(200, dtype=np.int64) * 3600
    out = build_structural_features(ohlcv, ts_s)
    assert np.isfinite(out).all(), "build_structural_features produced inf/nan"


def test_build_features_returns_24_cols():
    """Full integration: N_FEATURES constant should be 24."""
    from orderflow.features import FEATURE_NAMES, N_FEATURES
    assert N_FEATURES == 24, f"Expected 24 features, got {N_FEATURES}"
    assert len(FEATURE_NAMES) == 24
