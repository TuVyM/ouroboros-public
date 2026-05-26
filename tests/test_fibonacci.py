"""Verify all Fibonacci constants are correct Fibonacci numbers or φ-derived values."""
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_PHI = (1 + math.sqrt(5)) / 2


def _fibs_up_to(limit: int) -> set:
    s, a, b = set(), 0, 1
    while b <= limit:
        s.add(b)
        a, b = b, a + b
    return s


_FIBS = _fibs_up_to(10_000)


# ---------------------------------------------------------------------------
# shadow_trainer constants
# ---------------------------------------------------------------------------

def test_shadow_min_trade_buffer_is_fib():
    from shadow_trainer import MIN_TRADE_BUFFER
    assert MIN_TRADE_BUFFER in _FIBS, f"{MIN_TRADE_BUFFER} not Fibonacci"


def test_shadow_min_bar_buffer_is_fib():
    from shadow_trainer import MIN_BAR_BUFFER
    assert MIN_BAR_BUFFER in _FIBS, f"{MIN_BAR_BUFFER} not Fibonacci"


def test_shadow_swap_pf_margin_is_above_one():
    from shadow_trainer import SWAP_PF_MARGIN
    assert SWAP_PF_MARGIN > 1.0, "SWAP_PF_MARGIN must be > 1.0 (shadow must beat live)"


def test_bar_buffer_default_maxlen_is_fib():
    from shadow_trainer import BarBuffer
    buf = BarBuffer()
    # Push more items than any plausible non-Fibonacci cap to find the actual limit
    import collections
    assert isinstance(buf._buf, collections.deque)
    assert buf._buf.maxlen in _FIBS, f"BarBuffer default maxlen {buf._buf.maxlen} not Fibonacci"


def test_bar_buffer_enforces_maxlen():
    """Observable: buffer must not grow beyond its declared Fibonacci maxlen."""
    from shadow_trainer import BarBuffer
    import numpy as np, torch
    buf = BarBuffer()
    cap = buf._buf.maxlen
    dummy_w = np.zeros((64, 5), dtype=np.float32)
    dummy_h = torch.zeros(1, 1, 256)
    dummy_z = torch.zeros(1, 1, 128)
    for _ in range(cap + 50):
        buf.push(dummy_w, dummy_h, dummy_z)
    assert len(buf._buf) == cap, f"Expected {cap}, got {len(buf._buf)}"


def test_trade_buffer_default_maxlen_is_fib():
    from shadow_trainer import TradeBuffer
    buf = TradeBuffer()
    assert buf._buf.maxlen in _FIBS, f"TradeBuffer default maxlen {buf._buf.maxlen} not Fibonacci"


def test_trade_buffer_enforces_maxlen():
    """Observable: TradeBuffer must not grow beyond its declared Fibonacci maxlen."""
    from shadow_trainer import TradeBuffer
    import torch
    buf = TradeBuffer()
    cap = buf._buf.maxlen
    dummy_h  = torch.zeros(1, 1, 256)
    dummy_z  = torch.zeros(1, 1, 128)
    dummy_ps = torch.zeros(1, 10)
    for i in range(cap + 50):
        buf.push(dummy_h, dummy_z, dummy_ps, action=0, pnl=float(i))
    assert len(buf._buf) == cap, f"Expected {cap}, got {len(buf._buf)}"


# ---------------------------------------------------------------------------
# train_actor_critic constants
# ---------------------------------------------------------------------------

def test_q_pretrain_epochs_is_fib():
    import train_actor_critic as tac
    assert tac.Q_PRETRAIN_EPOCHS in _FIBS, f"Q_PRETRAIN_EPOCHS={tac.Q_PRETRAIN_EPOCHS} not Fibonacci"


def test_horizon_is_fib():
    src = Path("train_actor_critic.py").read_text()
    m = re.search(r'_HORIZON\s*=\s*(\d+)', src)
    assert m, "_HORIZON not found in train_actor_critic.py"
    val = int(m.group(1))
    assert val in _FIBS, f"_HORIZON={val} not Fibonacci"
