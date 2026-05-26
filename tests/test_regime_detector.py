# tests/test_regime_detector.py
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lgbm.regime_detector import RegimeDetector


def _trending_ohlcv(n=60):
    rng = np.random.default_rng(0)
    c = 50_000.0 * np.cumprod(1.0 + 0.008 + rng.normal(0, 0.001, n))
    h = c * (1.0 + rng.uniform(0.001, 0.003, n))
    l = c * (1.0 - rng.uniform(0.0001, 0.001, n))
    o = np.roll(c, 1); o[0] = c[0]
    v = np.ones(n) * 1000.0
    return np.column_stack([o, h, l, c, v]).astype(np.float32)


def _flat_ohlcv(n=60):
    rng = np.random.default_rng(1)
    c = 50_000.0 + rng.normal(0, 200, n)
    h = c + rng.uniform(100, 300, n)
    l = c - rng.uniform(100, 300, n)
    o = np.roll(c, 1); o[0] = c[0]
    v = np.ones(n) * 1000.0
    return np.column_stack([o, h, l, c, v]).astype(np.float32)


def test_classify_batch_trending():
    adx = np.array([28.0, 30.0, 35.0])
    assert list(RegimeDetector.classify_batch(adx)) == ["trending", "trending", "trending"]


def test_classify_batch_ranging():
    adx = np.array([10.0, 15.0, 19.9])
    assert list(RegimeDetector.classify_batch(adx)) == ["ranging", "ranging", "ranging"]


def test_classify_batch_volatile():
    adx = np.array([21.0, 23.5, 24.9])
    assert list(RegimeDetector.classify_batch(adx)) == ["volatile", "volatile", "volatile"]


def test_classify_batch_mixed():
    adx = np.array([10.0, 26.0, 22.0])
    result = RegimeDetector.classify_batch(adx)
    assert result[0] == "ranging"
    assert result[1] == "trending"
    assert result[2] == "volatile"


def test_compute_adx_output_length():
    ohlcv = _flat_ohlcv(50)
    adx = RegimeDetector.compute_adx(ohlcv)
    assert len(adx) == len(ohlcv) - 1


def test_compute_adx_values_bounded():
    ohlcv = _flat_ohlcv(60)
    adx = RegimeDetector.compute_adx(ohlcv)
    assert np.all(adx[14:] >= 0) and np.all(adx[14:] <= 100)


def test_classify_requires_three_bars_to_switch():
    det = RegimeDetector()
    det._current_regime = "ranging"
    ohlcv = _flat_ohlcv(60)
    det._threshold = lambda adx, atr, med: "trending"
    det.classify(ohlcv); assert det._current_regime == "ranging" and det._confirmation_count == 1
    det.classify(ohlcv); assert det._current_regime == "ranging" and det._confirmation_count == 2
    det.classify(ohlcv); assert det._current_regime == "trending" and det._confirmation_count == 0
    del det._threshold


def test_classify_hold_zone_does_not_advance_count():
    det = RegimeDetector()
    det._current_regime = "ranging"
    det._threshold = lambda adx, atr, med: None
    ohlcv = _flat_ohlcv(50)
    det.classify(ohlcv)
    assert det._current_regime == "ranging"
    assert det._confirmation_count == 0
    del det._threshold


def test_classify_regime_returns_string():
    det = RegimeDetector()
    result = det.classify(_flat_ohlcv(60))
    assert result in ("trending", "ranging", "volatile")


def test_classify_batch_returns_object_array():
    adx = np.array([10.0, 26.0])
    result = RegimeDetector.classify_batch(adx)
    assert result.dtype == object
