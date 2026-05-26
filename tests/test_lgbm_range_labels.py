# tests/test_lgbm_range_labels.py
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lgbm.labels_range import value_area_labels, BUY, SELL, HOLD


def _base_setup(n=30, val=49800.0, poc=50000.0, vah=50200.0, price=None):
    if price is None:
        price = val * 1.001  # just above VAL
    closes      = np.full(n, price, dtype=np.float64)
    highs       = closes * 1.001
    lows        = closes * 0.999
    atr14       = np.full(n, 200.0)
    va_levels   = [{"val": val, "poc": poc, "vah": vah}] * n
    regime_mask = np.ones(n, dtype=bool)
    X_base      = np.zeros((n, 24), dtype=np.float32)
    X_base[:, 10:15] = 0.003   # OB-Fib within 0.5% → confluence item (b) satisfied
    return closes, highs, lows, atr14, va_levels, regime_mask, X_base


def test_buy_label_when_price_reaches_poc():
    closes, highs, lows, atr14, va_levels, regime_mask, X_base = _base_setup()
    highs[3] = 50001.0   # hits POC (50000) on bar 3
    y = value_area_labels(closes, highs, lows, atr14, va_levels, regime_mask, X_base)
    assert y[0] == BUY


def test_sell_label_when_sl_hit_on_buy_candidate():
    closes, highs, lows, atr14, va_levels, regime_mask, X_base = _base_setup()
    # SL = VAL - 0.5 * ATR = 49800 - 100 = 49700
    lows[2] = 49699.0
    y = value_area_labels(closes, highs, lows, atr14, va_levels, regime_mask, X_base)
    assert y[0] == SELL


def test_sell_label_at_vah_price_reaches_poc():
    n, val, poc, vah = 30, 49800.0, 50000.0, 50200.0
    price       = vah * 0.999
    closes      = np.full(n, price, dtype=np.float64)
    highs       = closes * 1.001
    lows        = closes.copy()
    lows[3]     = 49999.0   # hits POC going down
    atr14       = np.full(n, 200.0)
    va_levels   = [{"val": val, "poc": poc, "vah": vah}] * n
    regime_mask = np.ones(n, dtype=bool)
    X_base      = np.zeros((n, 24), dtype=np.float32)
    X_base[:, 10:15] = 0.003
    y = value_area_labels(closes, highs, lows, atr14, va_levels, regime_mask, X_base)
    assert y[0] == SELL


def test_buy_label_when_sl_hit_on_sell_candidate():
    n, val, poc, vah = 30, 49800.0, 50000.0, 50200.0
    price       = vah * 0.999
    closes      = np.full(n, price, dtype=np.float64)
    highs       = closes.copy()
    # SL = VAH + 0.5 * ATR = 50200 + 100 = 50300
    highs[2]    = 50301.0
    lows        = closes * 0.999
    atr14       = np.full(n, 200.0)
    va_levels   = [{"val": val, "poc": poc, "vah": vah}] * n
    regime_mask = np.ones(n, dtype=bool)
    X_base      = np.zeros((n, 24), dtype=np.float32)
    X_base[:, 10:15] = 0.003
    y = value_area_labels(closes, highs, lows, atr14, va_levels, regime_mask, X_base)
    assert y[0] == BUY


def test_hold_when_not_near_val_or_vah():
    _, _, _, atr14, va_levels, regime_mask, X_base = _base_setup()
    closes = np.full(30, 50000.0, dtype=np.float64)  # at POC, not VAL/VAH
    highs  = closes * 1.01; lows = closes * 0.99
    y = value_area_labels(closes, highs, lows, atr14, va_levels, regime_mask, X_base)
    assert y[0] == HOLD


def test_hold_when_non_ranging_bar():
    closes, highs, lows, atr14, va_levels, _, X_base = _base_setup()
    regime_mask = np.zeros(30, dtype=bool)  # all trending
    y = value_area_labels(closes, highs, lows, atr14, va_levels, regime_mask, X_base)
    assert all(v == HOLD for v in y)


def test_hold_when_insufficient_confluence():
    closes, highs, lows, atr14, va_levels, regime_mask, _ = _base_setup()
    X_base = np.zeros((30, 24), dtype=np.float32)
    X_base[:, 10:15] = 0.01  # OB-Fib outside 0.5% → (b) not satisfied
    highs[3] = 50001.0
    y = value_area_labels(closes, highs, lows, atr14, va_levels, regime_mask, X_base)
    assert y[0] == HOLD  # confluence = 1 (only (a)) < 2 → no label


def test_hold_when_horizon_not_reached():
    closes, highs, lows, atr14, va_levels, regime_mask, X_base = _base_setup()
    # highs never reach POC within 6 bars
    y = value_area_labels(closes, highs, lows, atr14, va_levels, regime_mask, X_base)
    assert y[0] == HOLD


def test_labels_dtype():
    closes, highs, lows, atr14, va_levels, regime_mask, X_base = _base_setup()
    y = value_area_labels(closes, highs, lows, atr14, va_levels, regime_mask, X_base)
    assert y.dtype == np.int8
