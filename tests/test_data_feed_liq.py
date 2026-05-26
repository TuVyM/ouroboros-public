import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_feed import SymbolBuffer


def test_get_liq_since_empty_returns_zero():
    buf = SymbolBuffer("BTCUSDT")
    assert buf.get_liq_since(0) == 0.0


def test_get_liq_since_short_liq_positive():
    buf = SymbolBuffer("BTCUSDT")
    now = int(time.time() * 1000)
    buf.update_liquidation(now, 1000.0, is_short_liq=True)   # short liq = bullish
    result = buf.get_liq_since(now - 1)
    assert result == 1000.0


def test_get_liq_since_long_liq_negative():
    buf = SymbolBuffer("BTCUSDT")
    now = int(time.time() * 1000)
    buf.update_liquidation(now, 500.0, is_short_liq=False)   # long liq = bearish
    result = buf.get_liq_since(now - 1)
    assert result == -500.0


def test_get_liq_since_excludes_older_events():
    buf = SymbolBuffer("BTCUSDT")
    now = int(time.time() * 1000)
    buf.update_liquidation(now - 120_000, 999.0, is_short_liq=True)   # 2 min ago
    buf.update_liquidation(now,           200.0, is_short_liq=True)   # now
    result = buf.get_liq_since(now - 60_000)   # only last 1 min
    assert result == 200.0


def test_get_liq_since_mixed_signs():
    buf = SymbolBuffer("BTCUSDT")
    now = int(time.time() * 1000)
    buf.update_liquidation(now,           300.0, is_short_liq=True)
    buf.update_liquidation(now + 1,       100.0, is_short_liq=False)
    result = buf.get_liq_since(now - 1)
    assert result == 300.0 - 100.0
