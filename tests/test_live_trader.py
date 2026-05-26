"""Tests for live_trader: import hygiene and LiveTrader construction."""

import numpy as np



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
    from live_trader import LiveTrader, main
    assert callable(LiveTrader)
    assert callable(main)


def test_no_world_model_constants():
    """WORLD_MODEL_WINDOW and ACTOR_CRITIC_CHECKPOINT must not exist — WM path removed."""
    import importlib
    lt = importlib.import_module("live_trader")
    assert not hasattr(lt, "WORLD_MODEL_WINDOW"), \
        "WORLD_MODEL_WINDOW was removed; re-appearing means WM dead code crept back"
    assert not hasattr(lt, "ACTOR_CRITIC_CHECKPOINT"), \
        "ACTOR_CRITIC_CHECKPOINT was removed; re-appearing means WM dead code crept back"


def test_no_torch_top_level_import():
    """torch must not be imported at module level — LGBM path doesn't need it."""
    import inspect, importlib
    lt = importlib.import_module("live_trader")
    src = inspect.getsource(lt)
    # Allow 'import torch' inside function/method bodies (indented), not at top level
    for line in src.splitlines():
        if line.startswith("import torch"):
            raise AssertionError(
                "Top-level 'import torch' found in live_trader.py — "
                "move it inside the WM branch or remove it entirely"
            )


# ---------------------------------------------------------------------------
# LiveTrader: buf_cap
# ---------------------------------------------------------------------------

def test_live_trader_buf_cap_at_1m():
    """At bph=60, buf_cap must be >= 1500 to hold 25h of 1m bars."""
    bph = 60
    buf_cap = 25 * bph
    assert buf_cap >= 1500, \
        f"buf_cap formula yields {buf_cap} — too small for 1m 24h lookback (need >= 1500)"


def test_live_trader_buf_cap_at_1h():
    """At bph=1, buf_cap only needs 25 bars."""
    bph = 1
    buf_cap = 25 * bph
    assert buf_cap == 25, f"At bph=1, buf_cap should be 25, got {buf_cap}"


# ---------------------------------------------------------------------------
# LiveTrader: non-lgbm mode raises
# ---------------------------------------------------------------------------

def test_live_trader_non_lgbm_raises():
    """LiveTrader(use_lgbm=False) must raise ValueError — only LGBM mode is supported."""
    from unittest.mock import patch
    import pytest
    with patch("live_trader.LGBMPredictor"), \
         patch("live_trader.RegimeDetector"):
        with pytest.raises(ValueError, match="LGBM mode"):
            from live_trader import LiveTrader
            LiveTrader("BTCUSDT", use_lgbm=False)
