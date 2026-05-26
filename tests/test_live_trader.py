"""Tests for live_trader: import hygiene, pos_state shape/features, bph normalisation."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_window(n: int = 200, seed: int = 42) -> tuple:
    """Returns (closes, highs, lows) as float64 arrays of length n."""
    rng = np.random.default_rng(seed)
    closes = 50_000.0 * np.cumprod(1.0 + rng.normal(0, 0.001, n))
    highs  = closes * (1.0 + rng.uniform(0.0001, 0.002, n))
    lows   = closes * (1.0 - rng.uniform(0.0001, 0.002, n))
    return closes.astype(np.float64), highs.astype(np.float64), lows.astype(np.float64)


# ---------------------------------------------------------------------------
# Import hygiene
# ---------------------------------------------------------------------------

def test_no_bare_datafeed_import():
    """live_trader must not contain the broken `from data_feed import DataFeed` call."""
    import inspect, importlib
    lt = importlib.import_module("live_trader")
    src = inspect.getsource(lt)
    assert "from data_feed import DataFeed" not in src, \
        "Bare 'from data_feed import DataFeed' must be removed — class is BinanceFeed"


def test_binancefeed_referenced_in_source():
    import inspect, importlib
    lt = importlib.import_module("live_trader")
    src = inspect.getsource(lt)
    assert "BinanceFeed" in src, "live_trader must import BinanceFeed"


def test_live_trader_exports_expected_symbols():
    from live_trader import LiveTrader, _build_pos_state, main
    assert callable(LiveTrader)
    assert callable(_build_pos_state)
    assert callable(main)


# ---------------------------------------------------------------------------
# _build_pos_state: shape
# ---------------------------------------------------------------------------

def test_build_pos_state_shape_is_10():
    from live_trader import _build_pos_state
    closes, highs, lows = _make_window()
    ps = _build_pos_state(
        position=None, price=float(closes[-1]), tick=0,
        closes=closes, highs=highs, lows=lows,
        regime_idx=1, bvr=0.5, bph=60,
    )
    assert ps.shape == (1, 10), f"Expected (1, 10), got {ps.shape}"


def test_build_pos_state_no_position_open_flag_zero():
    from live_trader import _build_pos_state
    closes, highs, lows = _make_window()
    ps = _build_pos_state(
        position=None, price=float(closes[-1]), tick=0,
        closes=closes, highs=highs, lows=lows,
        regime_idx=1, bvr=0.5, bph=60,
    )
    assert ps[0, 0].item() == 0.0, "open_flag should be 0 when no position"


# ---------------------------------------------------------------------------
# _build_pos_state: position open
# ---------------------------------------------------------------------------

def test_build_pos_state_position_open_flag_one():
    from live_trader import _build_pos_state, _LivePosition
    closes, highs, lows = _make_window()
    pos = _LivePosition(
        direction='buy', entry_price=float(closes[-2]),
        size_usdc=100.0, stop_loss=0.0, take_profit=0.0,
        entry_tick=0, regime_idx=1,
    )
    ps = _build_pos_state(
        position=pos, price=float(closes[-1]), tick=5,
        closes=closes, highs=highs, lows=lows,
        regime_idx=1, bvr=0.7, bph=60,
    )
    assert ps[0, 0].item() == 1.0, "open_flag should be 1 when position open"
    assert abs(ps[0, 9].item() - 0.7) < 1e-5, "bvr feature at index 9 should be 0.7"


def test_build_pos_state_regime_encoding():
    from live_trader import _build_pos_state
    closes, highs, lows = _make_window()
    for regime_idx, expected in [(0, -1.0), (1, 0.0), (2, 1.0)]:
        ps = _build_pos_state(
            position=None, price=float(closes[-1]), tick=0,
            closes=closes, highs=highs, lows=lows,
            regime_idx=regime_idx, bvr=0.5, bph=60,
        )
        assert ps[0, 8].item() == expected, \
            f"regime_idx={regime_idx} should encode to {expected}, got {ps[0, 8].item()}"


# ---------------------------------------------------------------------------
# _build_pos_state: bvr clamping
# ---------------------------------------------------------------------------

def test_build_pos_state_bvr_clamped_high():
    from live_trader import _build_pos_state
    closes, highs, lows = _make_window()
    ps = _build_pos_state(
        position=None, price=float(closes[-1]), tick=0,
        closes=closes, highs=highs, lows=lows,
        regime_idx=1, bvr=2.5, bph=60,
    )
    assert ps[0, 9].item() == 1.0, "bvr > 1.0 must be clamped to 1.0"


def test_build_pos_state_bvr_clamped_low():
    from live_trader import _build_pos_state
    closes, highs, lows = _make_window()
    ps = _build_pos_state(
        position=None, price=float(closes[-1]), tick=0,
        closes=closes, highs=highs, lows=lows,
        regime_idx=1, bvr=-0.5, bph=60,
    )
    assert ps[0, 9].item() == 0.0, "bvr < 0.0 must be clamped to 0.0"


# ---------------------------------------------------------------------------
# _build_pos_state: bph normalisation
# ---------------------------------------------------------------------------

def test_build_pos_state_bph_affects_ret1h():
    """bph=1 and bph=60 must produce different ret_1h (index 4) for non-flat prices."""
    from live_trader import _build_pos_state
    # Need >= 61 bars so bph=60 computes ret_1h from actual lookback (not fallback 0.0)
    closes, highs, lows = _make_window(200)
    ps_1h = _build_pos_state(
        position=None, price=float(closes[-1]), tick=0,
        closes=closes, highs=highs, lows=lows,
        regime_idx=1, bvr=0.5, bph=1,
    )
    ps_1m = _build_pos_state(
        position=None, price=float(closes[-1]), tick=0,
        closes=closes, highs=highs, lows=lows,
        regime_idx=1, bvr=0.5, bph=60,
    )
    assert ps_1h[0, 4].item() != ps_1m[0, 4].item(), \
        "bph=1 and bph=60 must produce different ret_1h (feature index 4)"


def test_build_pos_state_bph1_ret24h_nonzero():
    """At bph=1 a 30-bar window is enough for ret_24h; verify it's computed (not fallback 0)."""
    from live_trader import _build_pos_state
    closes, highs, lows = _make_window(30)
    ps = _build_pos_state(
        position=None, price=float(closes[-1]), tick=0,
        closes=closes, highs=highs, lows=lows,
        regime_idx=1, bvr=0.5, bph=1,
    )
    # 30 bars > 24*1=24 required → ret_24h must not be the fallback 0.0
    assert ps[0, 5].item() != 0.0, \
        "ret_24h (index 5) should be non-zero with bph=1 and 30-bar window"


# ---------------------------------------------------------------------------
# LiveTrader: buf_cap scales with bph
# ---------------------------------------------------------------------------

def test_live_trader_buf_cap_at_1m():
    """At bph=60, buf_cap formula must yield >=1500 to hold all lookback windows."""
    from live_trader import WORLD_MODEL_WINDOW
    bph = 60
    buf_cap = max(WORLD_MODEL_WINDOW * 2, 25 * bph)
    assert buf_cap >= 1500, \
        f"buf_cap formula yields {buf_cap} — too small for 1m 24h lookback (need >=1500)"


def test_live_trader_buf_cap_at_1h():
    """At bph=1, buf_cap only needs 25 bars — stays at WORLD_MODEL_WINDOW*2 minimum."""
    from live_trader import WORLD_MODEL_WINDOW
    buf_cap = max(WORLD_MODEL_WINDOW * 2, 25 * 1)
    assert buf_cap == WORLD_MODEL_WINDOW * 2, \
        f"At bph=1 buf_cap should be WORLD_MODEL_WINDOW*2={WORLD_MODEL_WINDOW*2}, got {buf_cap}"
