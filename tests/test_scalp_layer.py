import logging
from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


CLOSE = 80_000.0


def _make_ohlcv(close=CLOSE, vol=1000.0):
    return np.array([close * 0.9995, close * 1.001, close * 0.999, close, vol],
                    dtype=np.float32)


def _make_layer(balance_slice=300.0, total_balance=900.0, dry_run=True):
    """Build a ScalpLayer with a mocked LightGBM model (always returns hold)."""
    from lgbm.scalp_layer import ScalpLayer

    fake_model = MagicMock()
    # probs: [P(sell)=0.1, P(hold)=0.8, P(buy)=0.1] → hold
    fake_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])

    layer = ScalpLayer(balance_slice=balance_slice,
                       total_balance=total_balance,
                       dry_run=dry_run,
                       model_path=None)
    layer._model = fake_model
    return layer


def _fill_buf(layer, n=30, close=CLOSE):
    """Feed n 1m bars so the feature buffer is full. Returns last on_1m_bar result."""
    result = None
    for i in range(n):
        result = layer.on_1m_bar(_make_ohlcv(close), 0.0, 0.0, 1_700_000_000_000 + i * 60_000)
    return result


class TestColdStartGuard:
    def test_returns_hold_before_htf_init(self):
        layer = _make_layer()
        result = layer.on_1m_bar(_make_ohlcv(), 0.0, 0.0, 1_700_000_000_000)
        assert result["signal"] == "hold"
        assert result["conf"] == 0.0

    def test_does_not_call_model_before_htf_init(self):
        layer = _make_layer()
        for _ in range(35):
            layer.on_1m_bar(_make_ohlcv(), 0.0, 0.0, 1_700_000_000_000)
        layer._model.predict.assert_not_called()

    def test_trades_after_htf_init(self):
        layer = _make_layer()
        layer.update_htf_context("hold", 0.5, "ranging")
        # Fill buffer to 30 bars so features are non-None
        result = _fill_buf(layer, 30)
        # Model was called at some point
        layer._model.predict.assert_called()


class TestHTFEncoding:
    def test_buy_signal_encoded_2(self):
        layer = _make_layer()
        layer.update_htf_context("buy", 0.8, "trending")
        assert layer._htf_signal == 2
        assert layer._htf_conf == pytest.approx(0.8)
        assert layer._htf_regime == 1

    def test_sell_signal_encoded_0(self):
        layer = _make_layer()
        layer.update_htf_context("sell", 0.6, "volatile")
        assert layer._htf_signal == 0
        assert layer._htf_regime == 2

    def test_unknown_signal_defaults_hold(self):
        layer = _make_layer()
        layer.update_htf_context("unknown", 0.5, "unknown_regime")
        assert layer._htf_signal == 1   # HOLD default
        assert layer._htf_regime == 0  # ranging default


class TestPositionManagement:
    def _make_buy_layer(self):
        layer = _make_layer()
        layer.update_htf_context("hold", 0.5, "ranging")
        # Override model to return strong BUY
        layer._model.predict.return_value = np.array([[0.05, 0.05, 0.90]])
        _fill_buf(layer, 30)
        return layer

    def test_opens_long_on_buy_signal(self):
        layer = self._make_buy_layer()
        assert layer._position is not None
        assert layer._position["direction"] == "long"

    def test_no_position_below_min_confidence(self):
        layer = _make_layer()
        layer.update_htf_context("hold", 0.5, "ranging")
        # conf=0.50 < MIN_SCALP_CONFIDENCE=0.60
        layer._model.predict.return_value = np.array([[0.0, 0.0, 0.50]])
        _fill_buf(layer, 30)
        assert layer._position is None

    def test_sl_closes_position_and_updates_balance(self):
        layer = self._make_buy_layer()
        assert layer._position is not None
        entry = layer._position["entry_price"]
        sl    = layer._position["stop_loss"]
        size  = layer._position["size_usdc"]
        init_bal = layer.balance

        # Feed a bar with close below SL (long SL hit)
        ohlcv_below = _make_ohlcv(close=sl - 1.0)
        layer._model.predict.return_value = np.array([[0.1, 0.8, 0.1]])  # hold
        result = layer.on_1m_bar(ohlcv_below, 0.0, 0.0, 1_700_000_000_000 + 31 * 60_000)

        assert layer._position is None
        assert result["trade_pnl"] is not None
        assert result["trade_pnl"] < 0.0          # SL = loss
        assert layer.balance < init_bal

    def test_tp_closes_position_with_profit(self):
        layer = self._make_buy_layer()
        tp    = layer._position["take_profit"]
        init_bal = layer.balance

        ohlcv_above = _make_ohlcv(close=tp + 1.0)
        layer._model.predict.return_value = np.array([[0.1, 0.8, 0.1]])
        result = layer.on_1m_bar(ohlcv_above, 0.0, 0.0, 1_700_000_000_000 + 31 * 60_000)

        assert layer._position is None
        assert result["trade_pnl"] > 0.0
        assert layer.balance > init_bal

    def test_position_capped_at_1pct_total_balance(self):
        # balance_slice=300 but total_balance=900 → cap=9.0 USDC
        layer = self._make_buy_layer()
        if layer._position is not None:
            assert layer._position["size_usdc"] <= 0.01 * 900.0 + 1e-6


class TestBalanceWarning:
    def test_warning_below_60pct(self, caplog):
        layer = _make_layer(balance_slice=100.0)
        layer.update_htf_context("hold", 0.5, "ranging")
        layer._balance = 55.0   # 55% of 100 → below 60%
        with caplog.at_level(logging.WARNING, logger="lgbm.scalp_layer"):
            _fill_buf(layer, 30)
        assert any("60%" in r.message for r in caplog.records)


class TestResultDict:
    def test_result_keys_always_present(self):
        layer = _make_layer()
        result = layer.on_1m_bar(_make_ohlcv(), 0.0, 0.0, 1_700_000_000_000)
        for key in ("signal", "conf", "trade_pnl", "balance", "position"):
            assert key in result

    def test_balance_property_matches_internal(self):
        layer = _make_layer(balance_slice=300.0)
        assert layer.balance == 300.0
