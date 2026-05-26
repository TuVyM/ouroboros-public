import numpy as np
import pytest

from lgbm.labels import triple_barrier_labels, BUY, SELL, HOLD


def _flat(n=20, price=100.0):
    return np.full(n, price, dtype=np.float64)


def test_output_length():
    closes = _flat(20)
    highs  = _flat(20, 101.0)
    lows   = _flat(20, 99.0)
    y = triple_barrier_labels(closes, highs, lows, tp=0.01, sl=0.005, horizon=8)
    assert len(y) == 20 - 8


def test_buy_label_when_tp_hit():
    # Flat entry at 100, then spike to 102 on bar 3 — hits TP (1%)
    n = 20
    closes = _flat(n, 100.0)
    highs  = _flat(n, 100.5)
    lows   = _flat(n, 99.8)  # Just above SL (99.5)
    highs[3] = 102.0  # TP hit at bar 3
    y = triple_barrier_labels(closes, highs, lows, tp=0.01, sl=0.005, horizon=8)
    assert y[0] == BUY


def test_sell_label_when_sl_hit():
    n = 20
    closes = _flat(n, 100.0)
    highs  = _flat(n, 100.5)
    lows   = _flat(n, 99.5)
    lows[2] = 94.0  # SL hit at bar 2 (drops below 100 * 0.995 = 99.5)
    y = triple_barrier_labels(closes, highs, lows, tp=0.01, sl=0.005, horizon=8)
    assert y[0] == SELL


def test_hold_label_when_neither_hit():
    # Flat series — no TP or SL ever hit in 8 bars
    n = 20
    closes = _flat(n, 100.0)
    highs  = closes * 1.001  # only 0.1% above — never hits TP of 1%
    lows   = closes * 0.999
    y = triple_barrier_labels(closes, highs, lows, tp=0.01, sl=0.005, horizon=8)
    assert y[0] == HOLD


def test_tp_before_sl_wins():
    # Both TP and SL hit on same bar — close is near TP
    n = 20
    closes = np.full(n, 100.0)
    closes[1] = 101.5  # close near TP (101.0)
    highs  = np.full(n, 102.0)   # hits TP
    lows   = np.full(n, 94.0)    # hits SL
    y = triple_barrier_labels(closes, highs, lows, tp=0.01, sl=0.005, horizon=8)
    assert y[0] == BUY  # close[1]=101.5 is closer to TP (101.0) than SL (99.5)


def test_labels_are_int8():
    closes = _flat(20)
    highs  = _flat(20, 101.0)
    lows   = _flat(20, 99.0)
    y = triple_barrier_labels(closes, highs, lows)
    assert y.dtype == np.int8
