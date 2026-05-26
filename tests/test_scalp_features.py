import sys
from collections import deque
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from orderflow.scalp_features import (
    SCALP_FEATURE_NAMES,
    N_SCALP_FEATURES,
    build_scalp_features,
)

CLOSE_BASE = 80_000.0


def _make_bar(close=CLOSE_BASE, vol=1000.0, ob_imb=0.1, liq_vol=0.0):
    open_ = close * 0.9995
    high  = close * 1.001
    low   = close * 0.999
    ohlcv = np.array([open_, high, low, close, vol], dtype=np.float32)
    return (ohlcv, ob_imb, liq_vol)


def _make_buf(n=30, close_start=CLOSE_BASE):
    buf = deque(maxlen=30)
    for i in range(n):
        close = close_start + i * 10.0
        buf.append(_make_bar(close=close))
    return buf


def test_names_and_count():
    assert len(SCALP_FEATURE_NAMES) == N_SCALP_FEATURES == 15


def test_returns_none_when_buf_lt_30():
    buf = _make_buf(n=29)
    result = build_scalp_features(buf, 1, 0.5, 0)
    assert result is None


def test_returns_array_when_buf_eq_30():
    buf = _make_buf(n=30)
    result = build_scalp_features(buf, 1, 0.5, 0)
    assert result is not None
    assert result.shape == (15,)
    assert result.dtype == np.float32


def test_htf_features_encoded_correctly():
    buf = _make_buf(n=30)
    result = build_scalp_features(buf, htf_signal=2, htf_conf=0.75, htf_regime=1)
    assert result[12] == pytest.approx(2.0)   # htf_signal
    assert result[13] == pytest.approx(0.75)  # htf_conf
    assert result[14] == pytest.approx(1.0)   # htf_regime


def test_ob_imbalance_and_liq_score_zeroed_in_training_sometimes():
    buf = _make_buf(n=30)
    # patch all bars with non-zero ob_imb and liq_vol
    for i in range(len(buf)):
        ohlcv, _, _ = buf[i]
        buf[i] = (ohlcv, 0.8, 500.0)
    # With 1000 trials about 50% should be zeroed
    zeroed_ob = 0
    zeroed_liq = 0
    trials = 400
    for _ in range(trials):
        r = build_scalp_features(buf, 1, 0.5, 0, training=True)
        assert r is not None
        if r[6] == 0.0:
            zeroed_ob += 1
        if r[11] == 0.0:
            zeroed_liq += 1
    # Expect 40–60% zeroed
    assert 0.30 * trials < zeroed_ob < 0.70 * trials
    assert 0.30 * trials < zeroed_liq < 0.70 * trials


def test_ob_imbalance_not_zeroed_when_not_training():
    # Ensure without training=True, ob_imbalance is preserved
    buf = _make_buf(n=30)
    for i in range(len(buf)):
        ohlcv, _, _ = buf[i]
        buf[i] = (ohlcv, 0.8, 500.0)
    results = [build_scalp_features(buf, 1, 0.5, 0, training=False) for _ in range(20)]
    ob_vals = [r[6] for r in results]
    # None should be zero since ob_imb=0.8 and training=False
    assert all(v != 0.0 for v in ob_vals)


def test_liq_score_not_zeroed_when_not_training():
    buf = _make_buf(n=30)
    # Set liq_vol on last 10 bars to a large value so liq_score is clearly nonzero
    for i in range(len(buf)):
        ohlcv, ob_imb, _ = buf[i]
        buf[i] = (ohlcv, ob_imb, 10_000.0)
    results = [build_scalp_features(buf, 1, 0.5, 0, training=False) for _ in range(20)]
    liq_vals = [r[11] for r in results]
    assert all(v != 0.0 for v in liq_vals)


def test_range_pos_between_0_and_1():
    buf = _make_buf(n=30)
    result = build_scalp_features(buf, 1, 0.5, 0)
    assert 0.0 <= result[9] <= 1.0


def test_rsi_between_0_and_100():
    buf = _make_buf(n=30)
    result = build_scalp_features(buf, 1, 0.5, 0)
    assert 0.0 <= result[8] <= 100.0


def test_vol_ratio_positive():
    buf = _make_buf(n=30)
    result = build_scalp_features(buf, 1, 0.5, 0)
    assert result[10] > 0.0
