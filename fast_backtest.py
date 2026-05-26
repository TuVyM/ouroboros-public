"""
fast_backtest.py — Batch backtest bypassing queue/threading overhead.

Speedup over main.py --backtest --no-teacher:
  Phase 1: preprocess sequentially (RunningNorm correctness preserved)
  Phase 2: Chronos encoding in batches of B=64 (10-20x GPU utilisation)
  Phase 3: RSSM + executor inline — no queue, no threads, no JSONL writes

Expected time: 30-day BTC ~3-5 min vs ~2hr with main.py

Usage:
  .venv/bin/python3 fast_backtest.py --symbol BTCUSDT --days 30
  .venv/bin/python3 fast_backtest.py --symbol BTCUSDT --start-offset 2938685
  .venv/bin/python3 fast_backtest.py --symbol BTCUSDT --days 30 --batch-size 128
"""
import argparse
import logging
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional

import numpy as np
from pathlib import Path

_HERE = Path(__file__).resolve().parent

import torch

# Load .env so CONSENSUS_THRESHOLD / MIN_CONFIDENCE overrides are honoured
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

try:
    from .config import (
        AGENT_CHECKPOINT_DIR,
        BANKROLL_FRACTION,
        CONSENSUS_THRESHOLD,
        MAX_POSITION_PCT,
        MIN_CONFIDENCE,
        PAPER_PORTFOLIO_USDC,
        WORLD_MODEL_CHECKPOINT,
        WORLD_MODEL_WINDOW,
    )
    from .backtest_feed import DATA_CACHE_DIR, _find_cache_file
    from .executor import STOP_LOSS_PCT, TRAIL_PCT, PROBE_SIZE_FRACTION, PROBE_HORIZON_MS, PROBE_COOLDOWN_MS
except ImportError:
    from config import (
        AGENT_CHECKPOINT_DIR,
        BANKROLL_FRACTION,
        CONSENSUS_THRESHOLD,
        MAX_POSITION_PCT,
        MIN_CONFIDENCE,
        PAPER_PORTFOLIO_USDC,
        WORLD_MODEL_CHECKPOINT,
        WORLD_MODEL_WINDOW,
    )
    from backtest_feed import DATA_CACHE_DIR, _find_cache_file
    from executor import STOP_LOSS_PCT, TRAIL_PCT, PROBE_SIZE_FRACTION, PROBE_HORIZON_MS, PROBE_COOLDOWN_MS

# Fibonacci + ICT regime-adaptive take-profit multipliers (R:R = TP_dist / SL_dist)
#   Trending  → φ² = 2.618  (ICT: ride the trend, target next liquidity)
#   Ranging   → φ  = 1.618  (golden ratio, mean-reversion to range boundary)
#   Volatile  → φ  = 1.618  (same — don't over-extend in volatile conditions)
_PHI  = 1.6180339887498949
_PHI2 = 2.6180339887498949
_TP_MULT = {0: _PHI2, 1: _PHI, 2: _PHI}  # regime_idx → R:R multiplier

# LGBM-specific TP/SL — must match train_lgbm.py triple_barrier_labels parameters
_LGBM_SL_PCT  = 0.005  # 0.5% SL — matches train_lgbm.py triple_barrier_labels sl=0.005
_LGBM_TP_MULT = 2.0    # 1.0% TP — matches train_lgbm.py tp=0.010 (SL × 2:1 R:R)
# Minimum confidence required on the OPPOSING signal before closing a position.
# Prevents single-tick noise from flushing positions (root cause of 87% sig= exits).
SIGNAL_FLIP_MIN_CONF = 0.40  # Fibonacci 0.382 rounded up
try:
    from .bocpd import BOCPD
    from .grid import GridResult, GridTrader
    from .swarm import AgentVote, ProbeSignal, SwarmOrchestrator, TradeDecision
    from .world_model import WorldModelWrapper
    from .world_model_v3 import V3WorldModelWrapper
    from .world_model import (
        _vp_features_batch, _fib_time_ob_features_batch, _ict_features_batch,
        _fib_channel_features_batch, _microstructure_proxy_batch,
        _catch22_features_batch, _adx_features_batch,
    )
except ImportError:
    from bocpd import BOCPD
    from grid import GridResult, GridTrader
    from swarm import AgentVote, ProbeSignal, SwarmOrchestrator, TradeDecision
    from world_model import WorldModelWrapper
    from world_model_v3 import V3WorldModelWrapper
    from world_model import (
        _vp_features_batch, _fib_time_ob_features_batch, _ict_features_batch,
        _fib_channel_features_batch, _microstructure_proxy_batch,
        _catch22_features_batch, _adx_features_batch,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fake order book — preprocess() requires bids/asks but they're zero in backtest
# ---------------------------------------------------------------------------
class _FakeBook:
    bids = [(0.0, 0.0)] * 5
    asks = [(0.0, 0.0)] * 5


# ---------------------------------------------------------------------------
# Trade tracking (inline — no JSONL, no PaperExecutor overhead)
# ---------------------------------------------------------------------------
@dataclass
class _Position:
    direction: str
    entry_price: float
    size_usdc: float
    stop_loss: float
    entry_tick: int = 0
    high_water: float = 0.0   # best price seen; drives trailing SL
    take_profit: float = 0.0  # fixed Fibonacci target; 0.0 = disabled
    initial_sl: float = 0.0   # SL at entry — distinguishes trail_sl from initial_sl exits
    entry_regime_idx: int = -1    # 0=trending 1=ranging 2=volatile; -1=unknown
    entry_regime_conf: float = 0.0


@dataclass
class _ClosedTrade:
    pnl_usdc: float
    pnl_pct: float
    reason: str  # "initial_sl" | "trail_sl" | "take_profit" | "signal" | "bocpd" | "eod"
    entry_regime_idx: int = -1
    entry_regime_conf: float = 0.0


@dataclass
class _ProbePosition:
    direction: str
    entry_price: float
    size_usdc: float
    entry_tick: int


_KELLY_RR = 3.5 / 1.5  # run47: TP=3.5×ATR, SL=1.5×ATR → R:R=2.33

def _kelly_size(confidence: float, balance: float) -> float:
    p, q = confidence, 1.0 - confidence
    kelly_f = max(0.0, (p * _KELLY_RR - q) / _KELLY_RR)
    size_pct = min(kelly_f * BANKROLL_FRACTION, MAX_POSITION_PCT)
    return balance * size_pct


def _ict_risk_size(balance: float, risk_pct: float = 0.00382) -> float:
    """ICT risk-first sizing: define the loss before the position.

    size = (bankroll × risk_pct) / TRAIL_PCT
    At 0.382% risk on £300: full-balance position, £1.15 risked.
    """
    return (balance * risk_pct) / TRAIL_PCT


# ---------------------------------------------------------------------------
# Kill zone helpers (ICT — London 07:00-10:00 UTC, NY 12:00-15:00 UTC)
#
# Raw data has no embedded timestamps. We recover the UTC minute-of-day for
# candle 0 from the cache file's mtime (= time the data was fetched ≈ last
# candle). For any N*24h slice, start_minute == end_minute, so:
#   minute_of_day(candle i) = (data_start_minute + i) % 1440
# ---------------------------------------------------------------------------
_LONDON_START = 7 * 60    # 420
_LONDON_END   = 10 * 60   # 600
_NY_START     = 12 * 60   # 720
_NY_END       = 15 * 60   # 900


def _data_start_minute(cache_path: str) -> int:
    """Estimate the UTC minute-of-day for candle 0 of the loaded slice."""
    mtime = os.path.getmtime(cache_path)
    dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
    return dt.hour * 60 + dt.minute


def _in_kill_zone(candle_idx: int, start_minute: int = 0) -> bool:
    """Return True when candle falls in a kill zone window."""
    minute = (start_minute + candle_idx) % 1440
    return _LONDON_START <= minute < _LONDON_END or _NY_START <= minute < _NY_END


# ---------------------------------------------------------------------------
# Choppiness Index — 100*log10(sum_ATR1 / (HH-LL)) / log10(n)
# >61.8 = choppy/ranging, <38.2 = clean trend. Fibonacci gates.
# ---------------------------------------------------------------------------
def _choppiness_index(raw: np.ndarray, candle_idx: int, n: int = 14) -> float:
    start = max(0, candle_idx - n + 1)
    window = raw[start: candle_idx + 1]
    if len(window) < 2:
        return 50.0  # neutral when insufficient data
    highs  = window[:, 1].astype(np.float64)
    lows   = window[:, 2].astype(np.float64)
    closes = window[:, 3].astype(np.float64)
    # True range per candle (use prev close for first candle's gap)
    prev_closes = np.concatenate([[closes[0]], closes[:-1]])
    tr = np.maximum(highs - lows,
         np.maximum(np.abs(highs - prev_closes), np.abs(lows - prev_closes)))
    sum_atr = tr.sum()
    hh_ll = highs.max() - lows.min()
    if hh_ll < 1e-10:
        return 100.0  # flat price = perfectly choppy
    return 100.0 * np.log10(sum_atr / hh_ll) / np.log10(len(window))


# ---------------------------------------------------------------------------
# V3 backtest path — single compact model, no specialist voting
# ---------------------------------------------------------------------------
def _run_v3(args, raw: np.ndarray, data_start_min: int, t_start: float):
    _min_conf = args.min_confidence if args.min_confidence is not None else MIN_CONFIDENCE
    n_windows = len(raw) - WORLD_MODEL_WINDOW
    log.info("V3 mode: %d windows", n_windows)

    # Load V3 model
    v3_wm = V3WorldModelWrapper(device=args.device, checkpoint_path=str(_HERE / "checkpoints" / "v3"))
    v3_wm.load()
    v3_wm.model.eval()
    log.info("V3 model loaded.")

    # Phase 1: precompute book features for all windows in batches
    log.info("Phase 1: precomputing book + macro features...")
    t1 = time.time()
    CHUNK = 1024
    closes = raw[:, 3].astype(np.float64)
    M = len(closes)
    cs = np.zeros(M + 1, dtype=np.float64)
    np.cumsum(closes, out=cs[1:])

    all_book: List[np.ndarray] = []
    for chunk_start in range(0, n_windows, CHUNK):
        chunk_end = min(chunk_start + CHUNK, n_windows)
        wins = np.stack([raw[i: i + WORLD_MODEL_WINDOW] for i in range(chunk_start, chunk_end)])
        ends = np.arange(chunk_start, chunk_end) + WORLD_MODEL_WINDOW - 1

        fib = np.concatenate([
            _vp_features_batch(wins),
            _microstructure_proxy_batch(wins),
            _ict_features_batch(wins),
            _fib_channel_features_batch(wins),
            _fib_time_ob_features_batch(wins),
        ], axis=1)  # (B, 24)
        c22 = _catch22_features_batch(wins)   # (B, 7)
        adx = _adx_features_batch(wins)       # (B, 3)

        cur_p = closes[ends]
        macro = np.zeros((len(ends), 4), dtype=np.float32)
        for col, (lb, clip_v) in enumerate([(240, 0.10), (1440, 0.20), (10080, 0.40)]):
            mask = ends >= lb
            ref = np.where(mask, closes[np.clip(ends - lb, 0, M - 1)], cur_p)
            macro[:, col] = np.where(mask, np.clip((cur_p - ref) / (ref + 1e-8), -clip_v, clip_v), 0.0)
        sma_s = np.maximum(0, ends - 28800)
        wlen  = ends - sma_s + 1
        valid = wlen >= 100
        sma20 = np.where(valid, (cs[ends + 1] - cs[sma_s]) / wlen, cur_p)
        macro[:, 3] = np.where(valid, np.clip((cur_p - sma20) / (sma20 + 1e-8), -0.30, 0.30), 0.0)

        book = np.concatenate([c22, adx, fib, macro], axis=1).astype(np.float32)  # (B, 38)
        all_book.append(book)
    book_all = np.concatenate(all_book, axis=0)  # (n_windows, 38)
    log.info("Phase 1 done in %.1fs", time.time() - t1)

    # Phase 2+3: sequential V3 forward + trade execution
    log.info("Phase 2+3: V3 inference + executor...")
    t2 = time.time()
    balance = PAPER_PORTFOLIO_USDC
    position: Optional[_Position] = None
    trades: List[_ClosedTrade] = []
    consensus_count = 0
    positions_opened = 0
    _persist_buf: Deque[str] = deque(maxlen=args.persistence)
    _bocpd = BOCPD(hazard_rate=args.bocpd_hazard) if args.bocpd_exit else None

    # Pre-stack all windows for indexed access
    wins_all = np.stack([raw[i: i + WORLD_MODEL_WINDOW] for i in range(n_windows)]).astype(np.float32)

    v3_wm.model._h = None
    v3_wm.model._z = None

    for i in range(n_windows):
        price = float(raw[i + WORLD_MODEL_WINDOW - 1, 3])

        # Kill zone gate
        in_kz = False
        if args.kill_zones:
            tick_min = (data_start_min + i) % 1440
            in_kz = (_LONDON_START <= tick_min < _LONDON_END or _NY_START <= tick_min < _NY_END)

        # Exit checks on open position
        _MIN_TRAIL_HOLD = 3
        if position is not None:
            _bars_held = i - position.entry_tick
            _win_e = i + WORLD_MODEL_WINDOW
            _hl_arr = raw[max(0, _win_e - 25):_win_e, 1] - raw[max(0, _win_e - 25):_win_e, 2]
            _atr_trail = float(np.clip(0.75 * float(_hl_arr[-24:].mean()) / (price + 1e-8), 0.004, 0.012))
            _atr_sl_sw = float(np.clip(1.5 * float(_hl_arr[-24:].mean()) / (price + 1e-8), 0.003, 0.015))
            if position.direction == "buy":
                pnl_pct = (price - position.entry_price) / position.entry_price
                position.high_water = max(position.high_water, price)
                if _bars_held >= _MIN_TRAIL_HOLD and pnl_pct >= _atr_sl_sw:
                    trail_sl = position.high_water * (1 - _atr_trail)
                    stop = max(position.stop_loss, trail_sl)
                else:
                    stop = position.stop_loss
            else:
                pnl_pct = (position.entry_price - price) / position.entry_price
                position.high_water = min(position.high_water, price)
                if _bars_held >= _MIN_TRAIL_HOLD and (-pnl_pct) >= _atr_sl_sw:
                    trail_sl = position.high_water * (1 + _atr_trail)
                    stop = min(position.stop_loss, trail_sl)
                else:
                    stop = position.stop_loss
            hit_sl = (position.direction == "buy" and price <= stop) or \
                     (position.direction == "sell" and price >= stop)
            hit_tp = (position.take_profit > 0 and (
                (position.direction == "buy"  and price >= position.take_profit) or
                (position.direction == "sell" and price <= position.take_profit)))
            bocpd_exit = False
            if _bocpd is not None and _bocpd.p_changepoint > args.bocpd_threshold:
                bocpd_exit = True
            if hit_tp:
                pnl = position.size_usdc * pnl_pct
                balance += pnl
                trades.append(_ClosedTrade(pnl, pnl_pct, "take_profit"))
                position = None
            elif bocpd_exit:
                pnl = position.size_usdc * pnl_pct
                balance += pnl
                trades.append(_ClosedTrade(pnl, pnl_pct, "bocpd"))
                position = None
            elif hit_sl:
                pnl = position.size_usdc * pnl_pct
                balance += pnl
                _trail_active = (position.direction == "buy" and stop > position.initial_sl) or \
                                (position.direction == "sell" and stop < position.initial_sl)
                trades.append(_ClosedTrade(pnl, pnl_pct, "trail_sl" if _trail_active else "initial_sl"))
                position = None

        # V3 inference (B=1, maintains RSSM state)
        with torch.inference_mode():
            ohlcv_t = torch.from_numpy(wins_all[i:i+1]).to(args.device)
            book_t  = torch.from_numpy(book_all[i:i+1]).to(args.device)
            out = v3_wm.model(ohlcv_t, book_t)

        sig_probs = torch.softmax(out.signal_logits[0], dim=-1).cpu().numpy()
        reg_probs = torch.softmax(out.regime_logits[0], dim=-1).cpu().numpy()
        sig_idx = int(sig_probs.argmax())
        conf = float(sig_probs[sig_idx])
        sig = ("buy", "sell", "hold")[sig_idx]  # matches V2 world_model signal_names ordering
        regime_idx = int(reg_probs.argmax())
        regime_confidence = float(reg_probs.max())

        if conf < _min_conf:
            sig = "hold"

        decision = None
        if sig in ("buy", "sell") and regime_confidence >= args.regime_min_conf:
            decision = TradeDecision(sig, conf, [], 1)
            consensus_count += 1
            if consensus_count <= 5:
                log.info("DECISION #%d at tick %d: %s conf=%.3f regime_idx=%d regime_conf=%.3f",
                         consensus_count, i, sig, conf, regime_idx, regime_confidence)

        if _bocpd is not None:
            _net = float(sig_probs[0]) - float(sig_probs[1])  # buy_prob - sell_prob
            _bocpd.update(_net)

        if decision and decision.direction != "hold":
            _persist_buf.append(decision.direction)
        else:
            _persist_buf.clear()

        persistent_decision = None
        if (decision and decision.direction != "hold"
                and len(_persist_buf) >= args.persistence
                and len(set(_persist_buf)) == 1):
            persistent_decision = decision

        if (persistent_decision is not None
                and position is None
                and (not args.kill_zones or in_kz)):
            d = persistent_decision.direction
            if args.ict_sizing:
                size = min(balance * 0.00382 / max(TRAIL_PCT, 1e-6), balance * MAX_POSITION_PCT)
            else:
                f = min(BANKROLL_FRACTION * persistent_decision.consensus_confidence, MAX_POSITION_PCT)
                size = balance * f
            size = max(1.0, size)
            if balance >= size:
                balance -= size
                sign = 1 if d == "buy" else -1
                _sl_pct  = _LGBM_SL_PCT  if _lgbm_predictor is not None else STOP_LOSS_PCT
                _tp_mult = _LGBM_TP_MULT if _lgbm_predictor is not None else _TP_MULT.get(regime_idx, _PHI)
                sl = price * (1 - sign * _sl_pct)
                sl_dist = abs(price - sl)
                tp = price + sign * sl_dist * _tp_mult
                position = _Position(d, price, size, sl, entry_tick=i, high_water=price, take_profit=tp, initial_sl=sl,
                                     entry_regime_idx=regime_idx, entry_regime_conf=regime_confidence)
                positions_opened += 1
                if _bocpd is not None:
                    _bocpd.reset()

        if i % 50_000 == 0 and i > 0:
            n = len(trades)
            wins = sum(1 for t in trades if t.pnl_usdc > 0)
            log.info("  %d/%d ticks | trades=%d win_rate=%.1f%% balance=$%.2f",
                     i, n_windows, n, (wins / n * 100) if n > 0 else 0, balance)

    log.info("Phase 2+3 done in %.1fs", time.time() - t2)

    if position is not None:
        price_final = float(raw[-1, 3])
        pnl_pct = (price_final - position.entry_price) / position.entry_price \
                  if position.direction == "buy" \
                  else (position.entry_price - price_final) / position.entry_price
        pnl = position.size_usdc * pnl_pct
        balance += pnl
        trades.append(_ClosedTrade(pnl, pnl_pct, "eod"))

    n = len(trades)
    wins = sum(1 for t in trades if t.pnl_usdc > 0)
    total_pnl = sum(t.pnl_usdc for t in trades)
    win_rate = wins / n if n > 0 else 0
    consensus_rate = positions_opened / n_windows if n_windows > 0 else 0
    exit_reasons = {k: sum(1 for t in trades if t.reason == k)
                    for k in ("initial_sl", "trail_sl", "take_profit", "bocpd", "eod")}
    returns = [t.pnl_pct for t in trades]
    sharpe = 0.0
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(525600)
    bal, peak, max_dd = PAPER_PORTFOLIO_USDC, PAPER_PORTFOLIO_USDC, 0.0
    for t in trades:
        bal += t.pnl_usdc
        peak = max(peak, bal)
        max_dd = max(max_dd, (peak - bal) / peak if peak > 0 else 0)

    gate_wr = win_rate > 0.52
    gate_cr = 0.10 <= consensus_rate <= 0.25
    total_time = time.time() - t_start

    print()
    print("=" * 56)
    print(f"  FAST BACKTEST V3 — {args.symbol} ({len(raw)/1440:.1f} days, {n_windows} ticks)")
    flags = []
    if args.ict_sizing:      flags.append("ict-sizing")
    if args.bocpd_exit:      flags.append(f"bocpd(h={args.bocpd_hazard:.0f},t={args.bocpd_threshold})")
    if args.persistence > 1: flags.append(f"persist={args.persistence}")
    if flags: print(f"  Flags:           {' '.join(flags)}")
    print("=" * 56)
    print(f"  Trades:          {n}")
    print(f"  Win rate:        {win_rate:.1%}  {'✅' if gate_wr else '❌'} (gate: >52%)")
    print(f"  Consensus rate:  {consensus_rate:.1%}  {'✅' if gate_cr else '❌'} (gate: 10-25%)")
    print(f"  Total P&L:       ${total_pnl:.2f}")
    print(f"  Final balance:   ${PAPER_PORTFOLIO_USDC + total_pnl:.2f}")
    print(f"  Max drawdown:    {max_dd:.1%}")
    print(f"  Sharpe:          {sharpe:.2f}")
    print(f"  Exit reasons:    SL={exit_reasons['initial_sl']}  trail={exit_reasons['trail_sl']}  TP={exit_reasons['take_profit']}  bocpd={exit_reasons['bocpd']}  eod={exit_reasons['eod']}")
    print(f"  Elapsed:         {total_time:.1f}s")
    print()
    if gate_wr and gate_cr:
        print("  ✅ V3 PASSES — within gate on win_rate + consensus_rate")
    else:
        print("  ❌ V3 FAILS — check distillation quality / thresholds")
    print("=" * 56)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fast batch backtest")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--days", type=int, default=30,
                        help="Number of days from end of cache (ignored if --start-offset set)")
    parser.add_argument("--start-offset", type=int, default=None,
                        help="Exact candle index to start from (overrides --days)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Chronos encoding batch size (larger = faster on GPU)")
    parser.add_argument("--min-confidence", type=float, default=None,
                        help="Override MIN_CONFIDENCE from config (e.g. 0.618)")
    parser.add_argument("--consensus-threshold", type=int, default=None,
                        help="Override CONSENSUS_THRESHOLD from config (e.g. 4)")
    parser.add_argument("--hold-threshold", type=float, default=None,
                        help="Treat vote as hold if argmax prob < threshold (e.g. 0.40). "
                             "Injects hold votes when model is uncertain, reducing lockstep consensus.")
    parser.add_argument("--calibration", type=str, default=None,
                        help="Path to calibration.json (per-specialist temperatures from calibrate_temperature.py). "
                             "Overrides global T_inv with per-agent temperature scaling.")
    parser.add_argument("--regime-exclusive", action="store_true",
                        help="Only the regime-appropriate specialist votes (momentum=trending, "
                             "mean_reversion=ranging, breakout=volatile). Trade when regime_conf "
                             ">= regime-min-conf AND specialist_conf >= min_confidence.")
    parser.add_argument("--regime-min-conf", type=float, default=0.5,
                        help="Minimum regime oracle confidence required for regime-exclusive trades (default 0.5)")
    parser.add_argument("--epistemic-soft", action="store_true",
                        help="Epistemic soft gate: weight all 3 votes by regime-alignment × confidence")
    parser.add_argument("--kill-zones", action="store_true",
                        help="ICT kill zones: only open NEW positions during London (07-10 UTC) and NY (12-15 UTC). "
                             "Existing positions remain open outside windows.")
    parser.add_argument("--persistence", type=int, default=1,
                        help="Signal persistence filter: require N consecutive ticks agreeing before entry (default 1 = off)")
    parser.add_argument("--grid", action="store_true",
                        help="Grid trading: when ranging (regime_idx=1) + outside kill zone, switch to "
                             "Fibonacci grid mode instead of directional mode.")
    parser.add_argument("--grid-risk-pct", type=float, default=0.001,
                        help="Risk per grid level as fraction of balance (default 0.001 = 0.1%%)")
    parser.add_argument("--ict-sizing", action="store_true",
                        help="Use ICT risk-first sizing (size = balance × 0.382%% / TRAIL_PCT) instead of Kelly.")
    parser.add_argument("--bocpd-exit", action="store_true",
                        help="Replace signal-flip exits with Bayesian Online Changepoint Detection. "
                             "Exit only when P(regime changed) > bocpd-threshold (default 0.7). "
                             "Requires 3+ consecutive opposing ticks to exit, ignoring single-tick churn.")
    parser.add_argument("--bocpd-threshold", type=float, default=0.7,
                        help="P(bearish/bullish) threshold to trigger BOCPD exit (default 0.7)")
    parser.add_argument("--bocpd-hazard", type=float, default=30.0,
                        help="BOCPD hazard rate: expected ticks between genuine regime changes (default 30)")
    parser.add_argument("--regime-stability", type=int, default=0,
                        help="Block new entries for N ticks after a regime change (default 0 = disabled)")
    parser.add_argument("--conf-weighted-size", action="store_true",
                        help="Scale ICT position size by regime_confidence (higher oracle confidence = larger bet)")
    parser.add_argument("--specialist-tp", action="store_true",
                        help="Use per-specialist TP multipliers: breakout=phi2 (2.618), "
                             "mean_reversion=1.0, momentum=phi2 (default: all use regime_idx mapping)")
    parser.add_argument("--mr-tp", type=float, default=None,
                        help="Override mean_reversion TP multiplier (e.g. 0.382, 0.5, 0.618). "
                             "Only applies when --specialist-tp is set.")
    parser.add_argument("--no-bocpd-ranging", action="store_true",
                        help="Disable BOCPD exits for positions entered in ranging regime "
                             "(mean_reversion specialist). Let SL/TP decide instead.")
    parser.add_argument("--actor-critic", action="store_true",
                        help="Replace specialist stack with DreamerV3 actor-critic. "
                             "Requires checkpoints/v4_actor/{actor,critic,reward}.pt")
    parser.add_argument("--actor-critic-dir", type=str, default=str(_HERE / "checkpoints" / "v4_actor"),
                        help="Path to actor-critic checkpoint directory.")
    parser.add_argument("--ci-filter", action="store_true",
                        help="Suppress breakout entries when Choppiness Index > --ci-threshold.")
    parser.add_argument("--ci-threshold", type=float, default=61.8,
                        help="CI gate for breakout entries (default: 61.8 = φ⁻¹ × 100). "
                             "Entries blocked when CI > threshold.")
    parser.add_argument("--ci-lookback", type=int, default=14,
                        help="Lookback period for Choppiness Index (default: 14 candles).")
    parser.add_argument("--model", choices=["v2", "v3"], default="v2",
                        help="Which model to backtest: v2=specialist swarm (default), v3=compact distilled")
    parser.add_argument("--specialist-dir", type=str, default=None,
                        help="Override specialist checkpoint directory (e.g. checkpoints/v3_agents)")
    parser.add_argument("--interval", type=str, default=None,
                        help="Bar interval to select matching cache file (e.g. '1h', '5m')")
    parser.add_argument("--output-log", type=str, default=None,
                        help="Path to write trade log as JSONL (one trade per line + summary)")
    parser.add_argument("--shadow-train", action="store_true",
                        help="Run ShadowTrainer daemon: background AC fine-tuning from live trade outcomes. "
                             "Requires --actor-critic. Saves checkpoint on each swap.")
    parser.add_argument("--shadow-sync", action="store_true",
                        help="(no-op) Sync shadow-trained weights back to checkpoint")
    parser.add_argument("--mode", choices=["wm", "llm", "lgbm", "tcn"], default="wm",
                        help="'wm' uses the world model (default); 'llm' uses the LLM swarm (qwen3:14b via Ollama); 'lgbm' uses LightGBM orderflow predictor; 'tcn' uses FibTCN multi-timeframe predictor")
    args = parser.parse_args()

    t_start = time.time()

    # Apply overrides
    _min_conf = args.min_confidence if args.min_confidence is not None else MIN_CONFIDENCE
    _consensus_thresh = args.consensus_threshold if args.consensus_threshold is not None else CONSENSUS_THRESHOLD
    _hold_thresh = args.hold_threshold  # None = disabled

    # Per-specialist temperature scaling (Guo et al. 2017)
    _cal_temps: dict = {}
    if args.calibration and os.path.exists(args.calibration):
        import json
        with open(args.calibration) as f:
            _cal_temps = json.load(f)
        log.info("Loaded calibration temperatures: %s", _cal_temps)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    path = _find_cache_file(args.symbol, DATA_CACHE_DIR, interval=args.interval)
    if path is None:
        log.error("No cache file for %s in %s (interval=%s)", args.symbol, DATA_CACHE_DIR, args.interval)
        sys.exit(1)

    raw = np.load(path)
    _bvr_path = path.replace(".npy", "_bvr.npy")
    _bvr_full = np.load(_bvr_path) if os.path.exists(_bvr_path) else None
    _data_start_min = _data_start_minute(path)  # UTC minute-of-day for candle 0 of full file

    # Candles-per-day depends on the bar interval
    _interval = args.interval or "1m"
    _interval_minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}.get(_interval, 1)
    _candles_per_day = 1440 // _interval_minutes
    # bars_per_hour: number of bars that equal 1 wall-clock hour — used to normalise pos_state lookbacks
    _bph = max(1, 60 // _interval_minutes)

    if args.start_offset is not None:
        raw = raw[args.start_offset:]
        _bvr_full = _bvr_full[args.start_offset:] if _bvr_full is not None else None
        _data_start_min = (_data_start_min + args.start_offset * _interval_minutes) % 1440
    else:
        n_candles = args.days * _candles_per_day
        # For whole-day slices, start_minute == end_minute == mtime minute (N*1440 % 1440 = 0)
        raw = raw[-n_candles:]
        _bvr_full = _bvr_full[-n_candles:] if _bvr_full is not None else None
        # start_minute unchanged for whole-day slices

    if args.kill_zones:
        log.info("Kill zone start_minute=%d UTC (London=%d-%d, NY=%d-%d)",
                 _data_start_min, _LONDON_START, _LONDON_END, _NY_START, _NY_END)

    log.info("Loaded %s: %d candles (%.1f days, interval=%s)", args.symbol, len(raw), len(raw) / _candles_per_day, _interval)

    if len(raw) < WORLD_MODEL_WINDOW + 1:
        log.error("Not enough candles (%d) for window size %d", len(raw), WORLD_MODEL_WINDOW)
        sys.exit(1)

    if args.model == "v3":
        _run_v3(args, raw, _data_start_min, t_start)
        return

    # ------------------------------------------------------------------
    # Load world models
    # ------------------------------------------------------------------
    _llm_swarm      = None
    _lgbm_predictor   = None
    _lgbm_shadow      = None
    _last_lgbm_feat   = None
    _last_lgbm_action = 1
    _tcn_predictor  = None
    _llm_signal_log: list = []
    if args.mode == "wm":
        log.info("Loading world models on %s...", args.device)
        _specialist_base = args.specialist_dir if args.specialist_dir else AGENT_CHECKPOINT_DIR
        log.info("Specialist dir: %s", _specialist_base)
        agent_wms: Dict[str, WorldModelWrapper] = {}
        for name in ("momentum", "mean_reversion", "breakout", "sentiment"):
            ckpt_path = os.path.join(_specialist_base, name)
            wm = WorldModelWrapper(device=args.device, checkpoint_path=ckpt_path)
            if os.path.exists(os.path.join(ckpt_path, "model.pt")):
                wm.load()
            else:
                wm.load(path=WORLD_MODEL_CHECKPOINT)
            agent_wms[name] = wm

        sentinel_wm = agent_wms["sentiment"]
        swarm = SwarmOrchestrator(agent_wms)

        # Actor-critic wrapper (optional — replaces specialist stack when --actor-critic active)
        _ac_wrapper = None
        h_s = z_s = None
        if args.actor_critic:
            from world_model import ActorCriticWrapper
            _ac_wrapper = ActorCriticWrapper(
                sentinel_wm.model, device=args.device,
                checkpoint_path=args.actor_critic_dir,
            )
            _ac_wrapper.load()
            log.info("ActorCritic loaded from %s", args.actor_critic_dir)

        _shadow = None
        if args.shadow_train:
            if _ac_wrapper is None:
                log.warning("--shadow-train requires --actor-critic; ignoring")
            else:
                from shadow_trainer import ShadowTrainer
                _shadow = ShadowTrainer(sentinel_wm, _ac_wrapper, device=args.device)
                _shadow.start()
                log.info("ShadowTrainer daemon started")
    elif args.mode == "llm":
        # LLM mode: no WM, no specialist swarm, no shadow trainer
        from llm_swarm import LLMSwarm
        _llm_model = os.environ.get("LLM_SWARM_MODEL", "qwen3:14b")
        _llm_swarm = LLMSwarm(model=_llm_model)
        log.info("LLM swarm loaded (%s via Ollama)", _llm_model)
        agent_wms  = {}
        _ac_wrapper = None
        _shadow     = None
        # Per-tick decision log for prompt analysis
        _llm_signal_log: list = []
    elif args.mode == "lgbm":
        # LightGBM mode: no WM, no Ollama
        from lgbm.predictor import LGBMPredictor
        from orderflow.features import build_features, FEATURE_NAMES
        from transformer.embed import load_encoder as _load_enc
        _lgbm_predictor = LGBMPredictor()
        _encoder = _load_enc()
        log.info("LGBMPredictor loaded from lgbm/btc_model.lgbm  encoder=%s",
                 "88/93-feat" if _encoder is not None else "24/29-feat (no finetuned.pt)")
        if args.shadow_train:
            from lgbm.shadow_trainer import LGBMShadowTrainer
            _lgbm_shadow = LGBMShadowTrainer(live_predictor=_lgbm_predictor)
            log.info("LGBMShadowTrainer initialised")
        agent_wms   = {}
        _ac_wrapper = None
        _shadow     = None
        _llm_signal_log: list = []
    else:
        # TCN mode: FibTCN multi-timeframe predictor
        if args.interval != "1h":
            log.error("--mode tcn requires --interval 1h (TCN features are 1h-aligned)")
            sys.exit(1)
        from tcn.predictor import TCNPredictor
        from tcn.dataset import compute_features as _tcn_compute_features, SEQ_LEN as _TCN_SEQ_LEN
        _tcn_predictor = TCNPredictor(device="cpu")
        log.info("TCNPredictor loaded from tcn/btc_model.pt")

        def _load_tf_feat(interval: str, bars_per_day: int) -> np.ndarray:
            tf_path = _find_cache_file(args.symbol, DATA_CACHE_DIR, interval=interval)
            if tf_path is None:
                raise ValueError(
                    f"No {interval} cache for {args.symbol}. Run:\n"
                    f"  python3 pretrain.py --symbols {args.symbol} --days {args.days}"
                    f" --interval {interval} --cache-only"
                )
            arr = np.load(tf_path)
            n = int(args.days * bars_per_day)
            return _tcn_compute_features(arr[-n:] if len(arr) > n else arr)

        _feat_15m = _load_tf_feat("15m", 96)
        _feat_4h  = _load_tf_feat("4h",  6)
        _feat_1d  = _load_tf_feat("1d",  1)
        _feat_1h  = _tcn_compute_features(raw)   # raw is the 1h OHLCV loaded above

        def _tcn_window(feat: np.ndarray, start: int, end: int) -> np.ndarray:
            w = feat[max(0, start):end]
            if len(w) < _TCN_SEQ_LEN:
                w = np.pad(w, ((_TCN_SEQ_LEN - len(w), 0), (0, 0)))
            return w.astype(np.float32)

        log.info("TCN feature arrays: 15m=%d  1h=%d  4h=%d  1d=%d",
                 len(_feat_15m), len(_feat_1h), len(_feat_4h), len(_feat_1d))
        agent_wms   = {}
        _ac_wrapper = None
        _shadow     = None
        _llm_signal_log: list = []

    dummy_book = _FakeBook()

    if args.mode == "wm":
        # Cast Chronos encoder to fp16 — Chronos is the bottleneck (320 T5 sequences per batch).
        # fp16 gives ~2x throughput on RTX 3080 with negligible accuracy loss (feature extractor, frozen).
        # Only the shared sentinel encoder matters; all 4 agents share it.
        if args.device == "cuda":
            enc = sentinel_wm.model.encoder
            if hasattr(enc, "chronos"):
                log.info("Casting Chronos encoder to fp16...")
                enc.chronos.half()
                enc.proj.half()
                log.info("fp16 cast done.")
            else:
                log.info("HRM encoder — skipping fp16 cast.")

        log.info("World models loaded.")

        # ------------------------------------------------------------------
        # Phase 1: Sequential preprocessing
        # Normalize each window in order so RunningNorm stats evolve correctly.
        # Store the resulting tensors for batched encoding in Phase 2.
        # ------------------------------------------------------------------
        n_windows = len(raw) - WORLD_MODEL_WINDOW
        log.info("Phase 1: preprocessing %d windows...", n_windows)
        t1 = time.time()

        normalized: List[torch.Tensor] = []    # list of (1, W, 5) CPU tensors
        book_tensors: List[torch.Tensor] = []  # list of (1, 15) CPU tensors
        for i in range(n_windows):
            window = raw[i: i + WORLD_MODEL_WINDOW]

            # Update all specialist norms (mirrors swarm.vote() sync logic)
            for name, wm in agent_wms.items():
                if name != "sentiment":
                    wm.norm.update(window)

            # sentinel norm update + normalize (sentinel is sentiment)
            ohlcv_t, book_t = sentinel_wm.preprocess(window, dummy_book)  # (1, W, 5), (1, 15)
            normalized.append(ohlcv_t.cpu())
            book_tensors.append(book_t.cpu())

            if (i + 1) % 50_000 == 0:
                log.info("  %d/%d preprocessed", i + 1, n_windows)

        log.info("Phase 1 done in %.1fs", time.time() - t1)
    else:
        n_windows    = len(raw) - WORLD_MODEL_WINDOW
        normalized   = []
        book_tensors = []
        log.info("Phase 1 skipped (LLM mode — raw OHLCV used directly)")

    # ------------------------------------------------------------------
    # Phase 2 + 3: Batched Chronos encoding → sequential RSSM + executor
    # ------------------------------------------------------------------
    log.info("Phase 2+3: encoding batches (B=%d) + RSSM + executor...", args.batch_size)
    t2 = time.time()

    balance = PAPER_PORTFOLIO_USDC
    _dd_peak_balance = PAPER_PORTFOLIO_USDC
    position: Optional[_Position] = None
    _mean_hl: float = 1e-8   # 24-bar mean HL range; updated each tick by AC path, fallback keeps fixed SL
    _sl_cooldown: dict[str, int] = {}   # direction -> tick idx of last SL exit
    _sig_cooldown: dict[str, int] = {}  # direction -> tick idx of last signal-flip exit (5-tick cooldown)
    trades: List[_ClosedTrade] = []
    consensus_count = 0
    positions_opened = 0
    _last_ac_pos_state = None   # pos_state tensor from last AC bar — used by shadow push_trade at close
    _probes: list[_ProbePosition] = []
    _closed_probes: list[_ClosedTrade] = []
    _last_probe_tick: int = -999   # cooldown sentinel

    # Kill zone + persistence + grid + BOCPD state
    _persist_buf: Deque[str] = deque(maxlen=args.persistence)  # last N decision directions
    _swing_prices: Deque[float] = deque(maxlen=100)            # rolling window for swing H/L
    _grid = GridTrader(balance, risk_pct_per_level=args.grid_risk_pct) if args.grid else None
    _grid_trades: List[_ClosedTrade] = []
    _bocpd = BOCPD(hazard_rate=args.bocpd_hazard) if args.bocpd_exit else None
    _last_regime_idx: int = -1       # previous tick's regime
    _regime_changed_tick: int = -999 # tick when regime last changed

    # Pre-compute LightGBM feature matrices (aligned to OHLCV timestamps)
    _lgbm_X        = None
    _lgbm_X_range  = None
    _lgbm_va_levels = []
    _regime_det    = None
    if _lgbm_predictor is not None:
        from orderflow.features import build_features
        from orderflow.range_features import build_range_features
        from lgbm.regime_detector import RegimeDetector as _RegimeDetector
        _ts_path = path.replace(".npy", "_ts.npy")
        if not os.path.exists(_ts_path):
            log.error("Timestamp file not found: %s — needed for --mode lgbm", _ts_path)
            sys.exit(1)
        _raw_ts = np.load(_ts_path)[-len(raw):]
        _trend_n_feat = _lgbm_predictor._trend_model.num_feature()
        _range_n_feat = _lgbm_predictor._range_model.num_feature()
        _trend_uses_emb = _encoder is not None and _trend_n_feat > 24
        _range_uses_emb = _encoder is not None and _range_n_feat > 29
        if _trend_uses_emb:
            from orderflow.features import build_features_with_embedding
            _lgbm_X, _ = build_features_with_embedding(args.symbol, raw, _raw_ts, _encoder)
        else:
            _lgbm_X, _ = build_features(args.symbol, raw, _raw_ts)
        log.info("LightGBM trend feature matrix (%d feat): %s rows", _trend_n_feat, _lgbm_X.shape[0])
        if _range_uses_emb:
            from orderflow.range_features import build_range_features_with_embedding
            _lgbm_X_range, _, _lgbm_va_levels = build_range_features_with_embedding(
                args.symbol, raw, _raw_ts, _encoder
            )
        else:
            _lgbm_X_range, _, _lgbm_va_levels = build_range_features(args.symbol, raw, _raw_ts)
        log.info("LightGBM range feature matrix (%d feat): %s rows", _range_n_feat, _lgbm_X_range.shape[0])
        _regime_det = _RegimeDetector()

    _cur_lgbm_regime  = "trending"
    _cur_lgbm_va_last = {}

    for batch_start in range(0, n_windows, args.batch_size):
        batch_end = min(batch_start + args.batch_size, n_windows)
        B = batch_end - batch_start

        if args.mode == "wm":
            # Batch encode with Chronos + sequential RSSM (all under inference_mode)
            batch_ohlcv = torch.cat(normalized[batch_start:batch_end], dim=0).to(args.device)  # (B, W, 5)
            batch_book = torch.cat(book_tensors[batch_start:batch_end], dim=0).to(args.device)  # (B, 15)
            with torch.inference_mode():
                batch_embs = sentinel_wm.model.encoder(batch_ohlcv, batch_book)  # (B, T', hidden)

        # Sequential RSSM + executor over items in this batch
        for j in range(B):
            i = batch_start + j
            price = float(raw[i + WORLD_MODEL_WINDOW - 1, 3])  # close of window's last candle
            embs = batch_embs[j: j + 1, -10:, :] if args.mode == "wm" else None  # last 10 Chronos steps

            # Track rolling swing H/L for grid mode
            _swing_prices.append(price)
            swing_h = max(_swing_prices)
            swing_l = min(_swing_prices)

            # Compute kill zone from candle index (minute-of-day proxy)
            in_kz = _in_kill_zone(i, _data_start_min) if args.kill_zones else True

            # Grid mode tick — runs regardless of kill zone (grid manages its own exits)
            if _grid is not None and len(_swing_prices) >= 10:
                grid_results = _grid.update(price, swing_h, swing_l)
                for gr in grid_results:
                    balance += gr.pnl_usdc
                    _grid_trades.append(_ClosedTrade(gr.pnl_usdc, gr.pnl_pct, f"grid_{gr.reason}"))

            # Check existing position — trailing SL ratchet then SL check
            _MIN_TRAIL_HOLD = 3
            if position:
                is_buy = position.direction == "buy"
                _bars_held = i - position.entry_tick

                # Ratchet trailing SL toward price, never backward — only after min hold
                # AND a profit gate: only trail once trade is ≥1 SL-distance in profit.
                # Without the gate, trail fires on tiny moves and exits winners before TP.
                # ATR-adaptive trail: 0.75 × 24-bar mean H-L (from previous tick), clipped [0.4%, 1.2%]
                _atr_trail_pct = float(np.clip(0.75 * _mean_hl / (price + 1e-8), 0.004, 0.012))
                _atr_sl_now = float(np.clip(1.5 * _mean_hl / (price + 1e-8), 0.003, 0.015))
                _pnl_toward_tp = (
                    (price / position.entry_price) - 1 if is_buy
                    else 1 - (price / position.entry_price)
                )
                _trail_profit_gate = _pnl_toward_tp >= _atr_sl_now
                if _bars_held >= _MIN_TRAIL_HOLD and _trail_profit_gate:
                    if is_buy:
                        position.high_water = max(position.high_water, price)
                        position.stop_loss = max(position.stop_loss,
                                                 position.high_water * (1 - _atr_trail_pct))
                    else:
                        position.high_water = min(position.high_water, price)
                        position.stop_loss = min(position.stop_loss,
                                                 position.high_water * (1 + _atr_trail_pct))
                else:
                    # Still update high_water so trail starts from correct level when it activates
                    if is_buy:
                        position.high_water = max(position.high_water, price)
                    else:
                        position.high_water = min(position.high_water, price)

                tp_hit = (position.take_profit > 0 and (
                    (is_buy  and price >= position.take_profit) or
                    (not is_buy and price <= position.take_profit)))
                sl_hit = (is_buy and price <= position.stop_loss) or \
                         (not is_buy and price >= position.stop_loss)
                if tp_hit:
                    pnl_pct = (price - position.entry_price) / position.entry_price if is_buy else \
                              (position.entry_price - price) / position.entry_price
                    pnl = position.size_usdc * pnl_pct
                    balance += pnl
                    trades.append(_ClosedTrade(pnl, pnl_pct, "take_profit",
                                               position.entry_regime_idx, position.entry_regime_conf))
                    if _ac_wrapper is not None and h_s is not None:
                        _action_idx = {"buy": 0, "sell": 1, "hold": 2}.get(
                            getattr(position, "direction", "hold"), 2)
                        _ac_wrapper.record_trade(h_s, z_s, _action_idx, pnl)
                        if _ac_wrapper._online_step > 0 and _ac_wrapper._online_step % 10 == 0 and len(_ac_wrapper.buffer) >= 32:
                            threading.Thread(target=_ac_wrapper.online_update, daemon=True).start()
                        if _shadow is not None and _last_ac_pos_state is not None:
                            _shadow.push_trade(h_s, z_s, _last_ac_pos_state, _action_idx, pnl)
                            _shadow.set_position_open(False)
                    if _lgbm_shadow is not None and _last_lgbm_feat is not None:
                        _lgbm_cycle = _lgbm_shadow.push_trade(features=_last_lgbm_feat, action=_last_lgbm_action, pnl=float(pnl))
                        _lgbm_cycle = _lgbm_shadow.run_sync_cycle()
                        if not _lgbm_cycle.get("skipped") and _lgbm_cycle.get("swapped"):
                            log.info("Shadow swap completed: %s", _lgbm_cycle)
                    position = None
                elif sl_hit:
                    _sl_dir = position.direction
                    pnl_pct = (price - position.entry_price) / position.entry_price if is_buy else \
                              (position.entry_price - price) / position.entry_price
                    pnl = position.size_usdc * pnl_pct
                    balance += pnl
                    _trail_active = (is_buy and position.stop_loss > position.initial_sl) or \
                                    (not is_buy and position.stop_loss < position.initial_sl)
                    trades.append(_ClosedTrade(pnl, pnl_pct, "trail_sl" if _trail_active else "initial_sl",
                                               position.entry_regime_idx, position.entry_regime_conf))
                    if _ac_wrapper is not None and h_s is not None:
                        _action_idx = {"buy": 0, "sell": 1, "hold": 2}.get(
                            getattr(position, "direction", "hold"), 2)
                        _ac_wrapper.record_trade(h_s, z_s, _action_idx, pnl)
                        if _ac_wrapper._online_step > 0 and _ac_wrapper._online_step % 10 == 0 and len(_ac_wrapper.buffer) >= 32:
                            threading.Thread(target=_ac_wrapper.online_update, daemon=True).start()
                        if _shadow is not None and _last_ac_pos_state is not None:
                            _shadow.push_trade(h_s, z_s, _last_ac_pos_state, _action_idx, pnl)
                            _shadow.set_position_open(False)
                    if _lgbm_shadow is not None and _last_lgbm_feat is not None:
                        _lgbm_shadow.push_trade(features=_last_lgbm_feat, action=_last_lgbm_action, pnl=float(pnl))
                        _lgbm_cycle = _lgbm_shadow.run_sync_cycle()
                        if not _lgbm_cycle.get("skipped") and _lgbm_cycle.get("swapped"):
                            log.info("Shadow swap completed: %s", _lgbm_cycle)
                    position = None
                    _sl_cooldown[_sl_dir] = i

            # Run 4 RSSMs — sentiment = regime oracle only, 3 specialists vote
            T_inv = 1.0
            votes: List[AgentVote] = []
            regime_confidence = 0.0
            regime_idx = 0
            _REGIME_SPECIALIST = {0: "momentum", 1: "mean_reversion", 2: "breakout"}
            _REGIME_NAMES = {0: "trending", 1: "ranging", 2: "volatile"}
            _REGIME_ALIGNMENT = {
                "momentum":       {"trending": 1.0, "ranging": 0.3, "volatile": 0.5},
                "mean_reversion": {"trending": 0.3, "ranging": 1.0, "volatile": 0.5},
                "breakout":       {"trending": 0.5, "ranging": 0.3, "volatile": 1.0},
            }

            if _ac_wrapper is not None:
                # Actor-critic path: run only sentinel to update RSSM state, then decide
                with torch.inference_mode():
                    h_s, z_s = sentinel_wm._rssm_states.get(args.symbol, (None, None))
                    _, new_h_s, new_z_s = sentinel_wm.model.forward_from_embeddings(embs, h=h_s, z=z_s)
                    sentinel_wm._rssm_states[args.symbol] = (new_h_s.detach(), new_z_s.detach())
                    h_s, z_s = new_h_s.detach(), new_z_s.detach()

                # Build pos_state (1, 10) matching training exactly:
                # [open, pnl%, ticks/100, mom_8h, ret_1h, ret_24h, close_vs_sma24, atr_ratio, regime, buy_vol_ratio]
                # All lookbacks are in wall-clock hours (normalised by _bph) so the same actor works at any interval.
                _win_end     = i + WORLD_MODEL_WINDOW
                _fetch_start = max(0, _win_end - 24 * _bph - 1)
                _closes  = raw[_fetch_start:_win_end, 3].astype(np.float64)
                _highs_w = raw[_fetch_start:_win_end, 1].astype(np.float64)
                _lows_w  = raw[_fetch_start:_win_end, 2].astype(np.float64)
                _hl_w    = _highs_w - _lows_w
                # ret_1h: 1-hour log-return
                _ret_1h  = float(np.clip(np.log(_closes[-1] / (_closes[-1 - _bph] + 1e-8)), -0.05, 0.05)) if len(_closes) > _bph else 0.0
                # ret_24h: 24-hour log-return
                _ret_24h = float(np.clip(np.log(_closes[-1] / (_closes[-1 - 24 * _bph] + 1e-8)), -0.20, 0.20)) if len(_closes) > 24 * _bph else 0.0
                # close_vs_sma24: deviation from 24-hour SMA
                _sma24        = float(_closes[-24 * _bph:].mean()) if len(_closes) >= 24 * _bph else price
                _close_vs_sma = float(np.clip((_closes[-1] - _sma24) / (_sma24 + 1e-8), -0.10, 0.10))
                # atr_ratio: current bar H-L / mean 24-hour H-L - 1
                _cur_hl  = float(_hl_w[-1]) if len(_hl_w) >= 1 else 0.0
                _mean_hl = float(_hl_w[-24 * _bph:].mean()) if len(_hl_w) >= 24 * _bph else 1e-8
                _atr_ratio = float(np.clip(_cur_hl / (_mean_hl + 1e-8) - 1.0, -1.0, 2.0))
                # mom_8h: 8-hour log-return
                _mom_8h = float(np.clip(np.log(_closes[-1] / (_closes[-1 - 8 * _bph] + 1e-8)), -0.10, 0.10)) if len(_closes) > 8 * _bph else 0.0
                # regime from WM regime_head (matches training); values {-1, 0, 1}
                with torch.inference_mode():
                    _regime_feat = float(
                        (_ac_wrapper.wm.regime_head(torch.cat([h_s, z_s], dim=-1)).argmax(dim=-1) - 1).item()
                    )
                if position is not None:
                    _open_flag  = 1.0
                    _pnl_pct    = (price - position.entry_price) / position.entry_price
                    if position.direction == "sell":
                        _pnl_pct = -_pnl_pct
                    # clip ticks_held to training range [0, 5] (training sim_ticks ~ Uniform(0,5))
                    _ticks_held = min((i - position.entry_tick) / 100.0, 5.0)
                else:
                    _open_flag  = 0.0
                    _pnl_pct    = 0.0
                    _ticks_held = 0.0
                _buy_vol_ratio = (
                    float(np.clip(_bvr_full[i] / (raw[i, 4] + 1e-8), 0.0, 1.0))
                    if _bvr_full is not None else 0.5
                )
                _ac_pos_state = torch.tensor([[
                    _open_flag, _pnl_pct, _ticks_held, _mom_8h,
                    _ret_1h, _ret_24h, _close_vs_sma, _atr_ratio, _regime_feat,
                    _buy_vol_ratio,
                ]], dtype=torch.float32)

                if _shadow is not None:
                    _shadow.push_bar(raw[i:i + WORLD_MODEL_WINDOW], h_s, z_s)
                _last_ac_pos_state = _ac_pos_state

                ac_result = _ac_wrapper.decide(h_s, z_s, pos_state=_ac_pos_state)
                sig = ac_result["signal"]
                conf = ac_result["confidence"]
                regime_idx = 2   # actor is regime-agnostic; label as volatile for sizing
                regime_confidence = conf

                if _bocpd is not None:
                    _net_signal = conf if sig == "buy" else -conf if sig == "sell" else 0.0
                    _bocpd.update(_net_signal)

                decision = None
                if sig in ("buy", "sell") and conf >= _min_conf:
                    decision = TradeDecision(sig, conf, [], 1)

            elif _llm_swarm is not None:
                # LLM swarm path — no WM, no RSSM, raw OHLCV window passed directly
                _ps = {
                    "open_flag":   1 if position is not None else 0,
                    "pnl_pct":     (price - position.entry_price) / position.entry_price
                                   if position is not None and position.direction == "buy"
                                   else (position.entry_price - price) / position.entry_price
                                   if position is not None else 0.0,
                    "ticks_held":  (i - position.entry_tick) if position is not None else 0,
                    "position":    position.direction if position is not None else None,
                }
                window_raw = raw[i: i + WORLD_MODEL_WINDOW]
                ac_result = _llm_swarm.decide(window_raw, _ps)
                sig  = ac_result["signal"]
                conf = ac_result["confidence"]
                regime_idx  = ac_result["regime"]
                regime_confidence = 1.0  # LLM doesn't give a separate regime confidence

                if _bocpd is not None:
                    _net_signal = conf if sig == "buy" else -conf if sig == "sell" else 0.0
                    _bocpd.update(_net_signal)

                _llm_signal_log.append((i, sig, round(conf, 3), regime_idx, _ps["open_flag"]))
                decision = None
                if sig in ("buy", "sell") and conf >= _min_conf:
                    decision = TradeDecision(sig, conf, [], 1)

            elif _lgbm_predictor is not None:
                # LightGBM regime-aware orderflow path.
                # _lgbm_X[k] and _lgbm_X_range[k] correspond to bar WINDOW+k.
                feat_row_idx = i - 1
                if feat_row_idx < 0 or feat_row_idx >= len(_lgbm_X):
                    sig, conf = "hold", 0.5
                    _cur_lgbm_regime  = "trending"
                    _cur_lgbm_va_last = {}
                else:
                    # Detect regime from rolling htf window
                    _lgbm_bar  = i + WORLD_MODEL_WINDOW - 1
                    htf_window = raw[max(0, _lgbm_bar - 167):_lgbm_bar + 1]
                    _cur_lgbm_regime = _regime_det.classify(htf_window)

                    if _cur_lgbm_regime == "ranging" and feat_row_idx < len(_lgbm_X_range):
                        _features_24    = None
                        _features_range = _lgbm_X_range[feat_row_idx]
                        _cur_lgbm_va_last = _lgbm_va_levels[feat_row_idx] if feat_row_idx < len(_lgbm_va_levels) else {}
                    else:
                        _features_24    = _lgbm_X[feat_row_idx]
                        _features_range = None
                        _cur_lgbm_va_last = {}

                    result = _lgbm_predictor.predict(_features_24, _features_range, _cur_lgbm_regime, _cur_lgbm_va_last)
                    sig    = result["signal"]
                    conf   = result["confidence"]
                    _last_lgbm_feat   = _features_24 if _features_24 is not None else _features_range
                    _last_lgbm_action = {"hold": 1, "buy": 2, "sell": 0}.get(sig, 1)

                _regime_map = {"trending": 0, "ranging": 1, "volatile": 2}
                regime_idx        = _regime_map.get(_cur_lgbm_regime, 0)
                regime_confidence = 1.0

                _llm_signal_log.append((i, sig, round(conf, 3), regime_idx, 1 if position is not None else 0))
                if _bocpd is not None:
                    _net_signal = conf if sig == "buy" else -conf if sig == "sell" else 0.0
                    _bocpd.update(_net_signal)
                decision = None
                if sig in ("buy", "sell") and conf >= _min_conf:
                    decision = TradeDecision(sig, conf, [], 1)
                # Update _mean_hl for ATR-based SL/TP (LGBM path skips the AC block that normally does this)
                _lgbm_bar = i + WORLD_MODEL_WINDOW - 1
                _lgbm_hl  = raw[max(0, _lgbm_bar - 23):_lgbm_bar + 1, 1] - raw[max(0, _lgbm_bar - 23):_lgbm_bar + 1, 2]
                if len(_lgbm_hl) >= 1:
                    _mean_hl = float(_lgbm_hl.mean())

            elif _tcn_predictor is not None:
                # TCN multi-timeframe path — extract 89-bar windows at each resolution.
                # Loop index i is the world-model window offset; actual 1h bar = i + WORLD_MODEL_WINDOW - 1.
                _t = i + WORLD_MODEL_WINDOW - 1
                x_15m = _tcn_window(_feat_15m, (_t + 1) * 4 - _TCN_SEQ_LEN, (_t + 1) * 4)
                x_1h  = _tcn_window(_feat_1h,  _t - _TCN_SEQ_LEN + 1,        _t + 1)
                x_4h  = _tcn_window(_feat_4h,  _t // 4 - _TCN_SEQ_LEN + 1,   _t // 4 + 1)
                x_1d  = _tcn_window(_feat_1d,  _t // 24 - _TCN_SEQ_LEN + 1,  _t // 24 + 1)
                result = _tcn_predictor.predict(x_15m, x_1h, x_4h, x_1d)
                sig, conf = result["signal"], result["confidence"]
                regime_idx        = 1
                regime_confidence = 1.0
                _llm_signal_log.append((i, sig, round(conf, 3), regime_idx, 1 if position is not None else 0))
                if _bocpd is not None:
                    _net_signal = conf if sig == "buy" else -conf if sig == "sell" else 0.0
                    _bocpd.update(_net_signal)
                decision = None
                if sig in ("buy", "sell") and conf >= _min_conf:
                    decision = TradeDecision(sig, conf, [], 1)

            else:
                with torch.inference_mode():
                    # Sentiment (regime oracle) must run first so regime_idx is set before specialists
                    for name, wm in sorted(agent_wms.items(), key=lambda x: 0 if x[0] == "sentiment" else 1):
                        h, z = wm._rssm_states.get(args.symbol, (None, None))
                        out, new_h, new_z = wm.model.forward_from_embeddings(embs, h=h, z=z)
                        wm._rssm_states[args.symbol] = (new_h.detach(), new_z.detach())
                        if name == "sentiment":
                            # Regime oracle — use regime_logits, do not vote
                            regime_probs = torch.softmax(out.regime_logits * T_inv, dim=-1)[0]
                            regime_confidence = float(regime_probs.max())
                            regime_idx = int(regime_probs.argmax())
                            if regime_idx != _last_regime_idx and _last_regime_idx != -1:
                                _regime_changed_tick = i
                            _last_regime_idx = regime_idx
                        else:
                            if name == "breakout" and args.regime_exclusive:
                                # Use calibrated signal_probs for direction (T=0.518 sharpens properly).
                                # Only fire when regime oracle confirms volatile.
                                volatile_conf = float(regime_probs[2]) if regime_idx == 2 else 0.0
                                if volatile_conf > args.regime_min_conf:
                                    agent_T_inv = 1.0 / _cal_temps.get(name, 0.5)
                                    probs = torch.softmax(out.signal_logits * agent_T_inv, dim=-1)[0]
                                    sig_idx = int(probs.argmax())
                                    conf = float(probs[sig_idx])
                                    sig = ("buy", "sell", "hold")[sig_idx]
                                else:
                                    sig, conf = "hold", 0.3
                            else:
                                # Per-agent temperature from calibration.json, fallback to T_inv=2.0
                                if name in _cal_temps:
                                    agent_T_inv = 1.0 / _cal_temps[name]
                                else:
                                    agent_T_inv = T_inv
                                # Specialist voter — use signal_logits
                                probs = torch.softmax(out.signal_logits * agent_T_inv, dim=-1)[0]
                                sig_idx = int(probs.argmax())
                                conf = float(probs[sig_idx])
                                sig = ("buy", "sell", "hold")[sig_idx]
                                # hold-threshold: treat uncertain votes as hold to prevent lockstep
                                if _hold_thresh is not None and conf < _hold_thresh:
                                    sig = "hold"
                            votes.append(AgentVote(name, sig, conf, "fast_backtest"))

                # Tally
                decision = None
                if args.regime_exclusive:
                    # Regime-exclusive: only the regime-appropriate specialist votes.
                    # consensus_confidence scaled by regime_confidence (epistemic certainty).
                    expert_name = _REGIME_SPECIALIST[regime_idx]
                    expert_vote = next((v for v in votes if v.agent_name == expert_name), None)
                    if (expert_vote is not None
                            and expert_vote.direction in ("buy", "sell")
                            and regime_confidence >= args.regime_min_conf
                            and expert_vote.confidence >= _min_conf):
                        epistemic_conf = expert_vote.confidence * regime_confidence
                        decision = TradeDecision(expert_vote.direction, epistemic_conf, votes, 1)
                elif args.epistemic_soft:
                    # Active Inference epistemic soft gate: weight all 3 votes by
                    # regime-alignment × specialist_confidence. regime_confidence is a gate,
                    # not a per-vote multiplier (to avoid triple-penalising each vote).
                    # Threshold 0.5: fires when domain expert + any partial agree at MIN_CONFIDENCE.
                    if regime_confidence >= args.regime_min_conf:
                        regime_name = _REGIME_NAMES[regime_idx]
                        scores: Dict[str, float] = {"buy": 0.0, "sell": 0.0}
                        for v in votes:
                            if v.direction not in scores:
                                continue
                            alignment = _REGIME_ALIGNMENT.get(v.agent_name, {}).get(regime_name, 0.5)
                            scores[v.direction] += alignment * v.confidence
                        best_dir = max(scores, key=scores.__getitem__)
                        best_score = scores[best_dir]
                        if best_score >= 0.48:
                            epistemic_conf = min(1.0, best_score * regime_confidence)
                            decision = TradeDecision(best_dir, epistemic_conf, votes,
                                                     sum(1 for v in votes if v.direction == best_dir))
                else:
                    # Default: 3/3 unanimous always trades; 2/3 agree + regime_confidence >= 0.5 trades
                    direction_votes: Dict[str, list] = {"buy": [], "sell": []}
                    for v in votes:
                        if v.direction in direction_votes:
                            direction_votes[v.direction].append(v)
                    for _dir, _agreeing in direction_votes.items():
                        n_agree = len(_agreeing)
                        if n_agree >= _consensus_thresh:  # 3/3 unanimous
                            _mean_conf = float(np.mean([v.confidence for v in _agreeing]))
                            decision = TradeDecision(_dir, _mean_conf, votes, n_agree)
                            break
                        elif n_agree == 2 and regime_confidence >= 0.5:  # 2/3 + regime unlock
                            _mean_conf = float(np.mean([v.confidence for v in _agreeing]))
                            decision = TradeDecision(_dir, _mean_conf, votes, n_agree)
                            break

                if decision:
                    consensus_count += 1
                    if consensus_count <= 5:
                        log.info("DECISION #%d at tick %d: %s conf=%.3f regime_idx=%d regime_conf=%.3f",
                                 consensus_count, i, decision.direction, decision.consensus_confidence,
                                 regime_idx, regime_confidence)

                # Compute net directional signal for BOCPD (always, not just when decision fires).
                # x_t = buy_score - sell_score from alignment-weighted vote sum.
                # Captures the continuous signal strength, not just the binary decision threshold.
                if _bocpd is not None:
                    if args.epistemic_soft and regime_confidence >= args.regime_min_conf:
                        regime_name = _REGIME_NAMES[regime_idx]
                        _b_scores: Dict[str, float] = {"buy": 0.0, "sell": 0.0}
                        for v in votes:
                            if v.direction in _b_scores:
                                _align = _REGIME_ALIGNMENT.get(v.agent_name, {}).get(regime_name, 0.5)
                                _b_scores[v.direction] += _align * v.confidence
                        _net_signal = _b_scores["buy"] - _b_scores["sell"]
                    else:
                        # Fallback: simple vote tally
                        _net_signal = sum(
                            v.confidence if v.direction == "buy" else
                            -v.confidence if v.direction == "sell" else 0.0
                            for v in votes
                        )
                    _bocpd.update(_net_signal)

            # consensus_count logging for actor-critic path
            if _ac_wrapper is not None and decision:
                consensus_count += 1
                if consensus_count <= 5:
                    log.info("DECISION #%d at tick %d: %s conf=%.3f (actor-critic)",
                             consensus_count, i, decision.direction, decision.consensus_confidence)

            # Persistence filter: accumulate direction, require N consecutive ticks
            if decision and decision.direction != "hold":
                _persist_buf.append(decision.direction)
            else:
                _persist_buf.clear()

            persistent_decision = None
            if (decision and decision.direction != "hold"
                    and len(_persist_buf) >= args.persistence
                    and len(set(_persist_buf)) == 1):
                persistent_decision = decision

            # Determine if we're in grid mode this tick
            in_grid_mode = (
                args.grid
                and regime_idx == 1           # ranging
                and not in_kz                  # outside kill zone
                and len(_swing_prices) >= 10   # enough history for swing H/L
            )

            # If switching into grid mode: close directional position
            if in_grid_mode and position is not None:
                if position.direction == "buy":
                    pnl_pct = (price - position.entry_price) / position.entry_price
                else:
                    pnl_pct = (position.entry_price - price) / position.entry_price
                pnl = position.size_usdc * pnl_pct
                balance += pnl
                trades.append(_ClosedTrade(pnl, pnl_pct, "signal",
                                           position.entry_regime_idx, position.entry_regime_conf))
                if _lgbm_shadow is not None and _last_lgbm_feat is not None:
                    _lgbm_shadow.push_trade(features=_last_lgbm_feat, action=_last_lgbm_action, pnl=float(pnl))
                    _lgbm_cycle = _lgbm_shadow.run_sync_cycle()
                    if not _lgbm_cycle.get("skipped") and _lgbm_cycle.get("swapped"):
                        log.info("Shadow swap completed: %s", _lgbm_cycle)
                position = None

            # If switching OUT of grid mode: collapse grid positions
            if not in_grid_mode and _grid is not None and _grid.is_active:
                for gr in _grid.collapse_all(price, "mode_exit"):
                    balance += gr.pnl_usdc
                    _grid_trades.append(_ClosedTrade(gr.pnl_usdc, gr.pnl_pct, "grid_mode_exit"))

            # BOCPD exit: close position when regime genuinely flips (not single-tick noise)
            # Minimum hold of 13 ticks (Fibonacci horizon) before BOCPD can fire
            _MIN_BOCPD_HOLD = 13
            _bocpd_suppressed = (args.no_bocpd_ranging
                                  and position is not None
                                  and position.entry_regime_idx == 1)
            if (position is not None and _bocpd is not None and not in_grid_mode
                    and not _bocpd_suppressed
                    and i - position.entry_tick >= _MIN_BOCPD_HOLD):
                p_exit = (_bocpd.p_bearish if position.direction == "buy"
                          else _bocpd.p_bullish)
                if p_exit > args.bocpd_threshold:
                    if position.direction == "buy":
                        pnl_pct = (price - position.entry_price) / position.entry_price
                    else:
                        pnl_pct = (position.entry_price - price) / position.entry_price
                    pnl = position.size_usdc * pnl_pct
                    balance += pnl
                    trades.append(_ClosedTrade(pnl, pnl_pct, "bocpd",
                                               position.entry_regime_idx, position.entry_regime_conf))
                    if _ac_wrapper is not None and h_s is not None:
                        _action_idx = {"buy": 0, "sell": 1, "hold": 2}.get(
                            getattr(position, "direction", "hold"), 2)
                        _ac_wrapper.record_trade(h_s, z_s, _action_idx, pnl)
                        if _ac_wrapper._online_step > 0 and _ac_wrapper._online_step % 10 == 0 and len(_ac_wrapper.buffer) >= 32:
                            threading.Thread(target=_ac_wrapper.online_update, daemon=True).start()
                        if _shadow is not None and _last_ac_pos_state is not None:
                            _shadow.push_trade(h_s, z_s, _last_ac_pos_state, _action_idx, pnl)
                            _shadow.set_position_open(False)
                    if _lgbm_shadow is not None and _last_lgbm_feat is not None:
                        _lgbm_shadow.push_trade(features=_last_lgbm_feat, action=_last_lgbm_action, pnl=float(pnl))
                        _lgbm_cycle = _lgbm_shadow.run_sync_cycle()
                        if not _lgbm_cycle.get("skipped") and _lgbm_cycle.get("swapped"):
                            log.info("Shadow swap completed: %s", _lgbm_cycle)
                    position = None

            # Epistemic probe: only when no consensus trade (mirrors live executor behaviour)
            _probe_horizon_ticks = PROBE_HORIZON_MS // 60_000   # ms → 1-min ticks
            _probe_cooldown_ticks = PROBE_COOLDOWN_MS // 60_000
            _no_consensus = decision is None or decision.direction == "hold"
            _ps: Optional[ProbeSignal] = swarm._probe_signal(args.symbol, votes) if (args.mode == "wm" and votes and _no_consensus) else None
            if _ps is not None and i - _last_probe_tick >= _probe_cooldown_ticks:
                _probe_size = balance * PROBE_SIZE_FRACTION
                _probes.append(_ProbePosition(
                    direction=_ps.direction,
                    entry_price=price,
                    size_usdc=_probe_size,
                    entry_tick=i,
                ))
                _last_probe_tick = i

            # Close probes at t+PROBE_HORIZON ticks
            _still_open = []
            for _p in _probes:
                if i - _p.entry_tick >= _probe_horizon_ticks:
                    if _p.direction == "buy":
                        _p_pnl_pct = (price - _p.entry_price) / _p.entry_price
                    else:
                        _p_pnl_pct = (_p.entry_price - price) / _p.entry_price
                    _p_pnl = _p.size_usdc * _p_pnl_pct
                    balance += _p_pnl
                    _closed_probes.append(_ClosedTrade(_p_pnl, _p_pnl_pct, "probe_timeout"))
                else:
                    _still_open.append(_p)
            _probes = _still_open

            # Execute directional decision (only when not in grid mode + in kill zone)
            if not in_grid_mode and persistent_decision and in_kz:
                d = persistent_decision.direction
                conf = persistent_decision.consensus_confidence

                # Signal-flip exit (when bocpd_exit disabled).
                # Require SIGNAL_FLIP_MIN_CONF on the opposing signal to avoid closing
                # on weak noise ticks (root cause of 87% sig= exits).
                if (position and position.direction != d and _bocpd is None
                        and conf >= SIGNAL_FLIP_MIN_CONF):
                    _exited_dir = position.direction
                    if position.direction == "buy":
                        pnl_pct = (price - position.entry_price) / position.entry_price
                    else:
                        pnl_pct = (position.entry_price - price) / position.entry_price
                    pnl = position.size_usdc * pnl_pct
                    balance += pnl
                    trades.append(_ClosedTrade(pnl, pnl_pct, "signal",
                                               position.entry_regime_idx, position.entry_regime_conf))
                    if _ac_wrapper is not None and h_s is not None:
                        _action_idx = {"buy": 0, "sell": 1, "hold": 2}.get(
                            getattr(position, "direction", "hold"), 2)
                        _ac_wrapper.record_trade(h_s, z_s, _action_idx, pnl)
                        if _ac_wrapper._online_step > 0 and _ac_wrapper._online_step % 10 == 0 and len(_ac_wrapper.buffer) >= 32:
                            threading.Thread(target=_ac_wrapper.online_update, daemon=True).start()
                        if _shadow is not None and _last_ac_pos_state is not None:
                            _shadow.push_trade(h_s, z_s, _last_ac_pos_state, _action_idx, pnl)
                            _shadow.set_position_open(False)
                    if _lgbm_shadow is not None and _last_lgbm_feat is not None:
                        _lgbm_shadow.push_trade(features=_last_lgbm_feat, action=_last_lgbm_action, pnl=float(pnl))
                        _lgbm_cycle = _lgbm_shadow.run_sync_cycle()
                        if not _lgbm_cycle.get("skipped") and _lgbm_cycle.get("swapped"):
                            log.info("Shadow swap completed: %s", _lgbm_cycle)
                    position = None
                    _sig_cooldown[_exited_dir] = i  # 5-tick re-entry cooldown on flipped direction

                # Open if above confidence threshold (SL cooldown + signal-flip cooldown + regime stability)
                _regime_stable = (args.regime_stability == 0
                                  or i - _regime_changed_tick >= args.regime_stability)
                _candle_idx = i + WORLD_MODEL_WINDOW - 1
                _ci_ok = (not args.ci_filter or regime_idx != 2
                          or _choppiness_index(raw, _candle_idx, args.ci_lookback) <= args.ci_threshold)
                _dd_peak_balance = max(_dd_peak_balance, balance)
                _current_dd = (_dd_peak_balance - balance) / max(_dd_peak_balance, 1e-8)
                _entry_conf_min = 0.60 if args.symbol == "ETHUSDT" else _min_conf
                if (position is None and conf >= _entry_conf_min
                        and _current_dd < 0.20
                        and i - _sl_cooldown.get(d, -999) >= 5
                        and i - _sig_cooldown.get(d, -999) >= 5
                        and _regime_stable and _ci_ok):
                    if args.ict_sizing:
                        _risk_pct = 0.00382 * (regime_confidence if args.conf_weighted_size else 1.0)
                        size = _ict_risk_size(balance, risk_pct=_risk_pct)
                    else:
                        size = _kelly_size(conf, balance)
                        size = max(1.0, size)  # floor: $1 minimum
                    is_buy = d == "buy"
                    sign = 1 if is_buy else -1
                    # ATR-scaled SL/TP. run47: TP raised 2.5→3.5× to improve R:R from 1.67→2.33.
                    # At 39% win rate, theoretical PF: (0.39×3.5)/(0.61×1.5) = 1.49.
                    _atr_sl_pct = float(np.clip(1.5 * _mean_hl / (price + 1e-8), 0.003, 0.015))
                    _atr_tp_pct = float(np.clip(3.5 * _mean_hl / (price + 1e-8), 0.007, 0.035))
                    sl = price * (1 - sign * _atr_sl_pct)
                    tp = price * (1 + sign * _atr_tp_pct)
                    # For LGBM ranging: use POC as TP (simplified backtest vs. 3-level ladder in live)
                    if _lgbm_predictor is not None and _cur_lgbm_regime == "ranging":
                        _poc = _cur_lgbm_va_last.get("poc") or 0.0
                        _val = _cur_lgbm_va_last.get("val") or price
                        if _poc > 0:
                            tp = _poc
                            sl_abs = abs(price - _val) * 0.5
                            sl = (price - sl_abs) if is_buy else (price + sl_abs)
                    position = _Position(
                        direction=d, entry_price=price, size_usdc=size,
                        stop_loss=sl, entry_tick=i, high_water=price, take_profit=tp,
                        initial_sl=sl,
                        entry_regime_idx=regime_idx, entry_regime_conf=regime_confidence,
                    )
                    positions_opened += 1
                    if _bocpd is not None:
                        _bocpd.reset()  # fresh detection from entry point
                    if _shadow is not None:
                        _shadow.set_position_open(True)

        if (batch_start // args.batch_size) % 100 == 0 and batch_start > 0:
            n = len(trades)
            wins = sum(1 for t in trades if t.pnl_usdc > 0)
            log.info(
                "  %d/%d ticks | trades=%d win_rate=%.1f%% balance=$%.2f",
                batch_start, n_windows, n,
                (wins / n * 100) if n > 0 else 0, balance,
            )

    log.info("Phase 2+3 done in %.1fs", time.time() - t2)

    # LLM signal distribution summary
    if _llm_swarm is not None and _llm_signal_log:
        _sigs = [s for _, s, _, _, _ in _llm_signal_log]
        _confs = [c for _, _, c, _, _ in _llm_signal_log]
        _buys  = _sigs.count("buy");  _sells = _sigs.count("sell"); _holds = _sigs.count("hold")
        print(f"\n  LLM SIGNAL DISTRIBUTION ({len(_sigs)} ticks)")
        print(f"  buy={_buys} ({_buys/len(_sigs):.0%})  sell={_sells} ({_sells/len(_sigs):.0%})  hold={_holds} ({_holds/len(_sigs):.0%})")
        print(f"  conf mean={sum(_confs)/len(_confs):.3f}  min={min(_confs):.3f}  max={max(_confs):.3f}")

    # Close any open directional position at end
    if position is not None:
        price_final = float(raw[-1, 3])
        if position.direction == "buy":
            pnl_pct = (price_final - position.entry_price) / position.entry_price
        else:
            pnl_pct = (position.entry_price - price_final) / position.entry_price
        pnl = position.size_usdc * pnl_pct
        balance += pnl
        trades.append(_ClosedTrade(pnl, pnl_pct, "eod",
                                   position.entry_regime_idx, position.entry_regime_conf))
        if _ac_wrapper is not None and h_s is not None:
            _action_idx = {"buy": 0, "sell": 1, "hold": 2}.get(
                getattr(position, "direction", "hold"), 2)
            _ac_wrapper.record_trade(h_s, z_s, _action_idx, pnl)
            if _ac_wrapper._online_step > 0 and _ac_wrapper._online_step % 10 == 0 and len(_ac_wrapper.buffer) >= 32:
                threading.Thread(target=_ac_wrapper.online_update, daemon=True).start()
            if _shadow is not None and _last_ac_pos_state is not None:
                _shadow.push_trade(h_s, z_s, _last_ac_pos_state, _action_idx, pnl)
                _shadow.set_position_open(False)
        if _lgbm_shadow is not None and _last_lgbm_feat is not None:
            _lgbm_shadow.push_trade(features=_last_lgbm_feat, action=_last_lgbm_action, pnl=float(pnl))
            _lgbm_cycle = _lgbm_shadow.run_sync_cycle()
            if not _lgbm_cycle.get("skipped") and _lgbm_cycle.get("swapped"):
                log.info("Shadow swap completed: %s", _lgbm_cycle)

    if _shadow is not None:
        _shadow.stop()
        log.info("ShadowTrainer stopped (swaps=%d)", _shadow._swap_count)

    # Collapse any open grid positions
    if _grid is not None and _grid.is_active:
        price_final = float(raw[-1, 3])
        for gr in _grid.collapse_all(price_final, "eod"):
            balance += gr.pnl_usdc
            _grid_trades.append(_ClosedTrade(gr.pnl_usdc, gr.pnl_pct, "grid_eod"))

    # Close any remaining probes at EOD
    price_final_for_probes = float(raw[-1, 3])
    for _p in _probes:
        if _p.direction == "buy":
            _p_pnl_pct = (price_final_for_probes - _p.entry_price) / _p.entry_price
        else:
            _p_pnl_pct = (_p.entry_price - price_final_for_probes) / _p.entry_price
        _p_pnl = _p.size_usdc * _p_pnl_pct
        balance += _p_pnl
        _closed_probes.append(_ClosedTrade(_p_pnl, _p_pnl_pct, "probe_eod"))
    _probes = []

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl_usdc > 0)
    total_pnl = sum(t.pnl_usdc for t in trades)
    win_rate = wins / n if n > 0 else 0

    # Consensus rate: positions_opened / eligible_ticks.
    # When kill zones are active, eligible = kill-zone ticks only (~25% of total).
    # Report both so the gate comparison is apples-to-apples.
    if args.kill_zones:
        kz_ticks = sum(1 for idx in range(n_windows) if _in_kill_zone(idx, _data_start_min))
        consensus_rate = positions_opened / kz_ticks if kz_ticks > 0 else 0
        consensus_rate_raw = positions_opened / n_windows if n_windows > 0 else 0
    else:
        kz_ticks = n_windows
        consensus_rate = positions_opened / n_windows if n_windows > 0 else 0
        consensus_rate_raw = consensus_rate

    exit_reasons = {
        "initial_sl":  sum(1 for t in trades if t.reason == "initial_sl"),
        "trail_sl":    sum(1 for t in trades if t.reason == "trail_sl"),
        "take_profit": sum(1 for t in trades if t.reason == "take_profit"),
        "signal":      sum(1 for t in trades if t.reason == "signal"),
        "bocpd":       sum(1 for t in trades if t.reason == "bocpd"),
        "eod":         sum(1 for t in trades if t.reason == "eod"),
    }

    # Grid stats
    n_grid = len(_grid_trades)
    grid_wins = sum(1 for t in _grid_trades if t.pnl_usdc > 0)
    grid_pnl = sum(t.pnl_usdc for t in _grid_trades)

    # Sharpe on combined trades (annualised, interval-aware)
    all_trades = trades + _grid_trades
    returns = [t.pnl_pct for t in all_trades]
    sharpe = 0.0
    if len(returns) > 1 and np.std(returns) > 0:
        _periods_per_year = 525600 // _interval_minutes
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(_periods_per_year)

    # Max drawdown (combined)
    bal = PAPER_PORTFOLIO_USDC
    peak = bal
    max_dd = 0.0
    for t in all_trades:
        bal += t.pnl_usdc
        peak = max(peak, bal)
        max_dd = max(max_dd, (peak - bal) / peak if peak > 0 else 0)

    total_time = time.time() - t_start

    gate_wr = win_rate > 0.52
    gate_cr = 0.10 <= consensus_rate <= 0.25

    # Regime oracle quality (regime-exclusive mode: does the oracle pick the right specialist?)
    _REGIME_NAMES_FULL = {0: "trending", 1: "ranging", 2: "volatile"}
    _SPECIALIST_FOR = {0: "momentum", 1: "mean_reversion", 2: "breakout"}
    regime_stats: dict[int, dict] = {r: {"n": 0, "wins": 0} for r in range(3)}
    conf_hi_trades = [t for t in trades if t.entry_regime_conf >= 0.65]
    conf_lo_trades = [t for t in trades if 0.0 < t.entry_regime_conf < 0.65]
    for t in trades:
        if t.entry_regime_idx in regime_stats:
            regime_stats[t.entry_regime_idx]["n"] += 1
            if t.pnl_usdc > 0:
                regime_stats[t.entry_regime_idx]["wins"] += 1
    # Oracle gate (regime-exclusive only): each regime with ≥20 trades must win >52%
    regime_wr_gate = True
    if args.regime_exclusive:
        for r, s in regime_stats.items():
            if s["n"] >= 20 and (s["wins"] / s["n"]) <= 0.52:
                regime_wr_gate = False

    # Write output log (JSONL) if requested
    if args.output_log:
        import json as _json_out
        os.makedirs(os.path.dirname(args.output_log) or ".", exist_ok=True)
        profit_factor = 0.0
        _gross_win = sum(t.pnl_usdc for t in trades if t.pnl_usdc > 0)
        _gross_loss = abs(sum(t.pnl_usdc for t in trades if t.pnl_usdc < 0))
        if _gross_loss > 0:
            profit_factor = _gross_win / _gross_loss
        with open(args.output_log, "w") as _f:
            for t in trades:
                _f.write(_json_out.dumps({"pnl_usdc": t.pnl_usdc, "pnl_pct": t.pnl_pct, "reason": t.reason}) + "\n")
            _f.write(_json_out.dumps({
                "summary": True, "symbol": args.symbol, "days": args.days,
                "interval": _interval, "n_trades": n, "win_rate": win_rate,
                "profit_factor": profit_factor, "total_pnl": total_pnl,
                "sharpe": sharpe,
            }) + "\n")
        log.info("Trade log written to %s (n=%d, PF=%.3f)", args.output_log, n, profit_factor)

    # Also emit Profit factor to stdout so the notebook grep can pick it up
    _gross_win_pf = sum(t.pnl_usdc for t in trades if t.pnl_usdc > 0)
    _gross_loss_pf = abs(sum(t.pnl_usdc for t in trades if t.pnl_usdc < 0))
    _pf = _gross_win_pf / _gross_loss_pf if _gross_loss_pf > 0 else 0.0
    print(f"Profit factor: {_pf:.4f}")
    # Regime-split PF (LGBM mode only — entry_regime_idx set by regime detector)
    if args.mode == "lgbm":
        _trend_trades = [t for t in trades if t.entry_regime_idx != 1]
        _range_trades = [t for t in trades if t.entry_regime_idx == 1]
        def _split_pf(tt):
            wins   = sum(t.pnl_usdc for t in tt if t.pnl_usdc > 0)
            losses = abs(sum(t.pnl_usdc for t in tt if t.pnl_usdc < 0))
            return wins / losses if losses > 0 else float("inf")
        print(f"PF (trending):  {_split_pf(_trend_trades):.3f}  ({len(_trend_trades)} trades)")
        print(f"PF (ranging):   {_split_pf(_range_trades):.3f}  ({len(_range_trades)} trades)")

    print()
    print("=" * 56)
    print(f"  FAST BACKTEST — {args.symbol} ({len(raw)/_candles_per_day:.1f} days, {n_windows} ticks)")
    flags = []
    if args.kill_zones:
        flags.append("kill-zones")
    if args.persistence > 1:
        flags.append(f"persist={args.persistence}")
    if args.grid:
        flags.append("grid")
    if args.ict_sizing:
        flags.append("ict-sizing")
    if args.bocpd_exit:
        flags.append(f"bocpd(h={args.bocpd_hazard:.0f},t={args.bocpd_threshold})")
    if args.regime_exclusive:
        flags.append("regime-exclusive")
    if flags:
        print(f"  Flags:           {' '.join(flags)}")
    print("=" * 56)
    print(f"  Trades:          {n}")
    print(f"  Win rate:        {win_rate:.1%}  {'✅' if gate_wr else '❌'} (gate: >52%)")
    if args.regime_exclusive:
        oracle_ok = "✅" if regime_wr_gate else "❌"
        print(f"  Oracle quality:  {oracle_ok} (each regime with ≥20 trades wins >52%)")
        for r in range(3):
            s = regime_stats[r]
            if s["n"] > 0:
                rwr = s["wins"] / s["n"]
                ok = "✅" if s["n"] < 20 or rwr > 0.52 else "❌"
                print(f"    {_REGIME_NAMES_FULL[r]:10s} ({_SPECIALIST_FOR[r]:14s}): "
                      f"n={s['n']:4d}  win={rwr:.1%} {ok}")
        if conf_hi_trades:
            hi_wr = sum(1 for t in conf_hi_trades if t.pnl_usdc > 0) / len(conf_hi_trades)
            print(f"  Conf≥0.65 win:   {hi_wr:.1%}  (n={len(conf_hi_trades)})  "
                  f"{'↑ oracle calibrated' if conf_hi_trades and conf_lo_trades and hi_wr > sum(1 for t in conf_lo_trades if t.pnl_usdc>0)/len(conf_lo_trades) else '↓ conf not predictive'}")
    else:
        if args.kill_zones:
            print(f"  Consensus rate:  {consensus_rate:.1%} (in-KZ)  {consensus_rate_raw:.1%} (all)  {'✅' if gate_cr else '❌'} (gate: 10-25% of KZ ticks)")
        else:
            print(f"  Consensus rate:  {consensus_rate:.1%}  {'✅' if gate_cr else '❌'} (gate: 10-25%)")
    _probe_total_pnl = sum(p.pnl_usdc for p in _closed_probes)
    print(f"  Total P&L:       ${total_pnl + _probe_total_pnl:.2f}")
    print(f"  Final balance:   ${PAPER_PORTFOLIO_USDC + total_pnl + _probe_total_pnl:.2f}")
    print(f"  Max drawdown:    {max_dd:.1%}")
    print(f"  Sharpe:          {sharpe:.2f}")
    if args.bocpd_exit:
        print(f"  Exit reasons:    SL={exit_reasons['initial_sl']}  trail={exit_reasons['trail_sl']}  TP={exit_reasons['take_profit']}  bocpd={exit_reasons['bocpd']}  eod={exit_reasons['eod']}")
    else:
        print(f"  Exit reasons:    SL={exit_reasons['initial_sl']}  trail={exit_reasons['trail_sl']}  TP={exit_reasons['take_profit']}  sig={exit_reasons['signal']}  eod={exit_reasons['eod']}")
    if args.grid:
        grid_wr = f"{grid_wins/n_grid:.1%}" if n_grid > 0 else "n/a"
        print(f"  Grid trades:     {n_grid}  win={grid_wr}  P&L=${grid_pnl:.2f}")
    if _closed_probes:
        _p_wins = sum(1 for p in _closed_probes if p.pnl_usdc > 0)
        _p_pnl  = sum(p.pnl_usdc for p in _closed_probes)
        print(f"  Probes:          {len(_closed_probes)} | win={_p_wins/len(_closed_probes):.1%} | P&L=${_p_pnl:.2f}")
    print(f"  Elapsed:         {total_time:.1f}s")
    print()
    _second_gate_pass = regime_wr_gate if args.regime_exclusive else gate_cr
    if gate_wr and _second_gate_pass:
        print("  ✅ V1 PASSES — proceed to V2")
    else:
        print("  ❌ V1 FAILS — check label generation / thresholds")
    print("=" * 56)


if __name__ == "__main__":
    main()
