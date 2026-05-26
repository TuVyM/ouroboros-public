"""Verify all Fibonacci constants are correct Fibonacci numbers or φ-derived values."""
import math
import re


_PHI = (1 + math.sqrt(5)) / 2


def _fibs_up_to(limit: int) -> set:
    s, a, b = set(), 0, 1
    while b <= limit:
        s.add(b)
        a, b = b, a + b
    return s


_FIBS = _fibs_up_to(10_000)


# ---------------------------------------------------------------------------
# lgbm.shadow_trainer constants
# ---------------------------------------------------------------------------

def test_shadow_min_trade_buffer_is_fib():
    from lgbm.shadow_trainer import MIN_TRADE_BUFFER
    assert MIN_TRADE_BUFFER in _FIBS, f"{MIN_TRADE_BUFFER} not Fibonacci"


def test_shadow_swap_pf_margin_is_above_one():
    from lgbm.shadow_trainer import SWAP_PF_MARGIN
    assert SWAP_PF_MARGIN > 1.0, "SWAP_PF_MARGIN must be > 1.0 (shadow must beat live)"


def test_trade_buffer_default_maxlen_is_fib():
    from lgbm.shadow_trainer import LGBMTradeBuffer
    buf = LGBMTradeBuffer()
    assert buf._buf.maxlen in _FIBS, f"LGBMTradeBuffer default maxlen {buf._buf.maxlen} not Fibonacci"


def test_trade_buffer_enforces_maxlen():
    """Observable: LGBMTradeBuffer must not grow beyond its declared Fibonacci maxlen."""
    from lgbm.shadow_trainer import LGBMTradeBuffer
    import numpy as np
    buf = LGBMTradeBuffer()
    cap = buf._buf.maxlen
    dummy_features = np.zeros(24, dtype=np.float32)
    for i in range(cap + 50):
        buf.push(dummy_features, action=1, pnl=float(i))
    assert len(buf._buf) == cap, f"Expected {cap}, got {len(buf._buf)}"
