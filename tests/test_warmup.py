# tests/test_warmup.py
import sys, os

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _make_trader():
    """Return a LiveTrader with LGBM path, warmup=True, balance=10_000."""
    with patch("live_trader.LGBMPredictor"), \
         patch("live_trader.RegimeDetector"), \
         patch("live_trader.torch"):
        from live_trader import LiveTrader
        t = LiveTrader("BTCUSDT", device="cpu", dry_run=True, bph=60, use_lgbm=True)
        t.balance = 10_000.0
        t._warmup = True
        return t


def test_warmup_flag_exists():
    """LiveTrader must have a _warmup attribute defaulting to False."""
    with patch("live_trader.LGBMPredictor"), \
         patch("live_trader.RegimeDetector"), \
         patch("live_trader.torch"):
        from live_trader import LiveTrader
        t = LiveTrader("BTCUSDT", device="cpu", dry_run=True, bph=60, use_lgbm=True)
    assert hasattr(t, "_warmup")
    assert t._warmup is False


def test_manage_position_no_op_during_warmup():
    """_manage_position_lgbm must not open a position while _warmup=True."""
    t = _make_trader()
    assert t.position is None
    t._manage_position_lgbm(
        sig="buy", conf=0.99, price=50_000.0,
        atr_sl=0.005, atr_tp=0.025,
        predict_result={"regime": "trending", "signal": "buy", "confidence": 0.99},
    )
    assert t.position is None, "No position should open during warmup"


def test_manage_position_works_after_warmup():
    """_manage_position_lgbm must open positions once _warmup=False."""
    t = _make_trader()
    t._warmup = False
    t._manage_position_lgbm(
        sig="buy", conf=0.99, price=50_000.0,
        atr_sl=0.005, atr_tp=0.025,
        predict_result={"regime": "trending", "signal": "buy", "confidence": 0.99},
    )
    assert t.position is not None, "Position should open once warmup is over"


# ── Task 2: _run_warmup tests ────────────────────────────────────────────────

import time as _time


def _make_ohlcv(n: int, base_price: float = 50_000.0) -> np.ndarray:
    arr = np.zeros((n, 5), dtype=np.float64)
    arr[:, 0] = base_price
    arr[:, 1] = base_price * 1.001
    arr[:, 2] = base_price * 0.999
    arr[:, 3] = base_price
    arr[:, 4] = 1.0
    return arr


def test_run_warmup_primes_buf():
    """_run_warmup replays bars and fills ScalpLayer._buf without opening positions."""
    from live_trader import _run_warmup
    from lgbm.scalp_layer import ScalpLayer

    t = _make_trader()
    t._warmup = False

    scalp = ScalpLayer(
        balance_slice=3_000.0,
        total_balance=10_000.0,
        dry_run=True,
        model_path=None,
    )
    scalp._model = MagicMock()
    scalp._model.predict.return_value = np.array([[0.1, 0.8, 0.1]])
    t._scalp = scalp

    n_bars = 40
    ohlcv_1m = _make_ohlcv(n_bars + 5)
    ts_1m = np.arange(len(ohlcv_1m), dtype=np.int64)
    ohlcv_1h = _make_ohlcv(10)
    ts_1h = np.arange(0, 10, dtype=np.int64)

    with patch("live_trader._find_cache_file") as mock_find, \
         patch("live_trader.np.load") as mock_load:

        def _find_side(symbol, cache_dir, interval=None):
            return "fake_1m.npy" if interval == "1m" else "fake_1h.npy"

        def _load_side(path):
            if "ts" in path:
                return ts_1m if "1m" in path else ts_1h
            return ohlcv_1m if "1m" in path else ohlcv_1h

        mock_find.side_effect = _find_side
        mock_load.side_effect = _load_side

        t.on_candle = MagicMock(return_value={
            "signal": "hold", "confidence": 0.5, "regime": 0,
            "atr_sl": 0.005, "atr_tp": 0.025,
        })

        _run_warmup(t, "BTCUSDT", n_bars=n_bars)

    assert len(scalp._buf) == 30
    assert t.position is None
    assert scalp._position is None
    assert t._warmup is False


def test_run_warmup_no_scalp():
    """_run_warmup works when no ScalpLayer is attached."""
    from live_trader import _run_warmup

    t = _make_trader()
    t._scalp = None

    n_bars = 20
    ohlcv_1m = _make_ohlcv(n_bars + 5)
    ts_1m = np.arange(len(ohlcv_1m), dtype=np.int64)
    ohlcv_1h = _make_ohlcv(5)
    ts_1h = np.arange(5, dtype=np.int64)

    with patch("live_trader._find_cache_file") as mock_find, \
         patch("live_trader.np.load") as mock_load:

        mock_find.side_effect = lambda s, d, interval=None: (
            "fake_1m.npy" if interval == "1m" else "fake_1h.npy"
        )
        mock_load.side_effect = lambda path: (
            ts_1m if "ts" in path and "1m" in path else
            ts_1h if "ts" in path else
            ohlcv_1m if "1m" in path else ohlcv_1h
        )

        t.on_candle = MagicMock(return_value={
            "signal": "hold", "confidence": 0.5, "regime": 0,
            "atr_sl": 0.005, "atr_tp": 0.025,
        })

        _run_warmup(t, "BTCUSDT", n_bars=n_bars)

    assert t._warmup is False


def test_run_warmup_skips_when_no_cache():
    """_run_warmup returns without error when 1m cache is absent."""
    from live_trader import _run_warmup

    t = _make_trader()
    t._scalp = None

    with patch("live_trader._find_cache_file", return_value=None):
        _run_warmup(t, "BTCUSDT", n_bars=500)

    assert t._warmup is False
