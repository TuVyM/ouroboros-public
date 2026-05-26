# tests/test_range_features.py
import numpy as np
import pytest
from unittest.mock import patch

from orderflow.range_features import (
    _rsi, _compute_value_area, build_range_features,
    N_RANGE_FEATURES, RANGE_FEATURE_NAMES,
)


def _ohlcv(n=120, price=50000.0):
    rng = np.random.default_rng(42)
    c = np.full(n, price, dtype=np.float32) + rng.normal(0, 100, n).astype(np.float32)
    h = c + 500
    l = c - 500
    o = np.roll(c, 1); o[0] = c[0]
    v = np.ones(n, dtype=np.float32) * 1000
    return np.column_stack([o, h, l, c, v])


def test_n_range_features_is_29():
    assert N_RANGE_FEATURES == 29


def test_range_feature_names_length():
    assert len(RANGE_FEATURE_NAMES) == 29


def test_rsi_length():
    closes = np.linspace(50000, 55000, 50).astype(np.float64)
    rsi = _rsi(closes, period=14)
    assert len(rsi) == len(closes)


def test_rsi_bounded():
    closes = np.linspace(50000, 55000, 60).astype(np.float64)
    rsi = _rsi(closes, period=14)
    assert np.all(rsi[14:] >= 0) and np.all(rsi[14:] <= 100)


def test_rsi_high_on_uptrend():
    closes = np.linspace(50000, 55000, 30).astype(np.float64)
    rsi = _rsi(closes, period=14)
    assert rsi[-1] > 70


def test_rsi_low_on_downtrend():
    closes = np.linspace(55000, 50000, 30).astype(np.float64)
    rsi = _rsi(closes, period=14)
    assert rsi[-1] < 30


def test_compute_value_area_structure():
    ohlcv = np.zeros((20, 5), dtype=np.float32)
    ohlcv[:, 1] = 50500; ohlcv[:, 2] = 49500
    ohlcv[:, 3] = 50000; ohlcv[:, 4] = 1000
    va = _compute_value_area(ohlcv)
    assert {"val", "poc", "vah"} == set(va)
    assert va["val"] <= va["poc"] <= va["vah"]
    assert 49500 <= va["poc"] <= 50500


def test_compute_value_area_val_below_vah():
    ohlcv = np.zeros((20, 5), dtype=np.float32)
    ohlcv[:, 1] = 51000; ohlcv[:, 2] = 49000
    ohlcv[:, 3] = 50000; ohlcv[:, 4] = 100
    va = _compute_value_area(ohlcv)
    assert va["val"] < va["vah"]


def test_build_range_features_shape():
    n = 120
    ohlcv = _ohlcv(n)
    ts    = np.arange(n, dtype=np.int64) * 3600
    mock_base = np.zeros((n - 89, 24), dtype=np.float32)
    mock_ts   = ts[89:]
    with patch("orderflow.range_features.build_features", return_value=(mock_base, mock_ts)):
        X, out_ts, va_levels = build_range_features("BTCUSDT", ohlcv, ts)
    assert X.shape == (n - 89, N_RANGE_FEATURES)
    assert X.dtype == np.float32
    assert len(va_levels) == n - 89


def test_build_range_features_va_levels_keys():
    n = 100
    ohlcv = _ohlcv(n)
    ts    = np.arange(n, dtype=np.int64) * 3600
    mock_base = np.zeros((n - 89, 24), dtype=np.float32)
    mock_ts   = ts[89:]
    with patch("orderflow.range_features.build_features", return_value=(mock_base, mock_ts)):
        _, _, va_levels = build_range_features("BTCUSDT", ohlcv, ts)
    for va in va_levels:
        assert isinstance(va["val"], float)
        assert isinstance(va["poc"], float)
        assert isinstance(va["vah"], float)
