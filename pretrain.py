"""
Historical pre-training for the world model student.

Fetches OHLCV from Binance, applies hindsight labelling, and trains the world
model directly — no teacher LLM needed. Run this before main.py to give the
model a head start on signal/regime calibration.

Usage:
    .venv/bin/python3 pretrain.py [--symbols BTCUSDT,ETHUSDT,SOLUSDT] [--days 30] [--epochs 3]
"""
import argparse
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from pathlib import Path
import requests
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from .config import (
        SYMBOLS,
        WORLD_MODEL_CHECKPOINT,
        WORLD_MODEL_KL_WEIGHT,
        WORLD_MODEL_LR,
        WORLD_MODEL_REGIME_WEIGHT,
        WORLD_MODEL_RETURN_WEIGHT,
        WORLD_MODEL_STEP,
        WORLD_MODEL_WINDOW,
    )
    from .world_model import symlog
except ImportError:
    from config import (
        SYMBOLS,
        WORLD_MODEL_CHECKPOINT,
        WORLD_MODEL_KL_WEIGHT,
        WORLD_MODEL_LR,
        WORLD_MODEL_REGIME_WEIGHT,
        WORLD_MODEL_RETURN_WEIGHT,
        WORLD_MODEL_STEP,
        WORLD_MODEL_WINDOW,
    )
    from world_model import symlog

# Fibonacci / golden-ratio constants for label calibration
# Boundaries: 61.8% / 38.2% — canonical Fibonacci retracement levels
# (Lo, Mamaysky & Wang, 2000; Fischer, 1993; Frost & Prechter, 2005)
_INV_PHI  = 0.6180339887498949   # 1/φ  — buy/sell quantile boundary
_INV_PHI2 = 0.3819660112501051   # 1/φ² — sell boundary
_INV_PHI3 = 0.2360679774997896   # 1/φ³ — confidence range above boundary
try:
    from .world_model import WorldModelWrapper
except ImportError:
    from world_model import WorldModelWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger(__name__)


def _ddp_setup():
    """Initialise DDP when launched via torchrun (sets RANK/LOCAL_RANK env vars).
    Returns (local_rank, world_size, is_distributed)."""
    if "RANK" not in os.environ:
        return 0, 1, False
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size(), True


def _ddp_cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

BINANCE_REST = "https://api.binance.com/api/v3/klines"
LOOKAHEAD = 5
STEP = WORLD_MODEL_STEP   # configurable stride — default 8 gives ~11% window overlap
BATCH_SIZE = int(os.getenv("PRETRAIN_BATCH_SIZE", "1024"))  # reduce to 512 for chronos-t5-small
SAVE_EVERY = 1000  # steps

# EMA self-distillation — prediction unit for predictive coding transition
# EMA model = stable prediction unit (top-down in PC terms)
# Training model = recognition unit (error-driven, bottom-up in PC terms)
# iPC upgrade path: replace blended target with per-tick E/M steps (V3)
EMA_DECAY  = 0.99    # τ — how slowly the prediction unit tracks the recognition unit
EMA_BLEND  = 0.382   # φ⁻² — EMA weight; Fibonacci label weight = 0.618 = φ⁻¹
CACHE_DIR = str(Path(__file__).resolve().parent / "data_cache")
DIR_AUX_COEF = 0.3   # weight for directional auxiliary loss — forces latents to encode up/down


# ---------------------------------------------------------------------------
# Dummy order book (zeros) for historical data that lacks book snapshots
# ---------------------------------------------------------------------------
@dataclass
class _Level:
    price: float
    qty: float

class _DummyBook:
    bids: list = None
    asks: list = None
    def __post_init__(self):
        self.bids = []
        self.asks = []

_DUMMY_BOOK = _DummyBook()
_DUMMY_BOOK.__post_init__()


# ---------------------------------------------------------------------------
# Binance data fetch (same as generate_training_data.py)
# ---------------------------------------------------------------------------
_INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}

def fetch_klines(symbol: str, days: int, interval: str = "1m", use_cache: bool = True) -> np.ndarray:
    """Fetch from cache if available, otherwise download from Binance and cache."""
    if interval not in _INTERVAL_MS:
        raise ValueError(f"interval must be one of {list(_INTERVAL_MS)}; got {interval!r}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{symbol}_{days}d_{interval}.npy")

    if use_cache and os.path.exists(cache_path):
        arr = np.load(cache_path)
        log.info("%s: loaded %d %s candles from cache (%s)", symbol, len(arr), interval, cache_path)
        return arr
    bvr_path = cache_path.replace(".npy", "_bvr.npy")

    step_ms = _INTERVAL_MS[interval]
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 24 * 3600 * 1000
    candles = []
    fetch_start = start_ms
    log.info("Fetching %s (%dd of %s candles) from Binance…", symbol, days, interval)
    while fetch_start < end_ms:
        for attempt in range(5):
            try:
                resp = requests.get(BINANCE_REST, params={
                    "symbol": symbol, "interval": interval,
                    "startTime": fetch_start, "endTime": end_ms, "limit": 1000,
                }, timeout=30)
                resp.raise_for_status()
                break
            except Exception as e:
                wait = 2 ** attempt
                log.warning("Fetch attempt %d failed (%s), retrying in %ds…", attempt + 1, e, wait)
                time.sleep(wait)
        else:
            raise RuntimeError(f"Failed to fetch {symbol} after 5 attempts")
        batch = resp.json()
        if not batch:
            break
        candles.extend(batch)
        if len(batch) < 1000:
            break
        fetch_start = batch[-1][0] + step_ms
        time.sleep(0.05)
    arr = np.array([[float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])]
                    for c in candles], dtype=np.float32)
    bvr_arr = np.array([float(c[9]) for c in candles], dtype=np.float32)  # taker buy base vol (col 9)
    ts_arr = np.array([int(c[0]) // 1000 for c in candles], dtype=np.int64)  # open time as unix seconds
    np.save(cache_path, arr)
    np.save(cache_path.replace(".npy", "_bvr.npy"), bvr_arr)
    np.save(cache_path.replace(".npy", "_ts.npy"), ts_arr)
    log.info("%s: %d %s candles — cached to %s", symbol, len(arr), interval, cache_path)
    return arr


def fetch_bvr(symbol: str, days: int, interval: str = "1h") -> Optional[np.ndarray]:
    """Load taker buy base volume cache (saved alongside fetch_klines output as _bvr.npy)."""
    bvr_path = os.path.join(CACHE_DIR, f"{symbol}_{days}d_{interval}_bvr.npy")
    if os.path.exists(bvr_path):
        return np.load(bvr_path)
    log.warning("%s: taker buy vol cache missing at %s — run with FORCE_PRETRAIN=True to generate", symbol, bvr_path)
    return None


# ---------------------------------------------------------------------------
# TradingView fetch (broader symbol universe — stocks, forex, commodities)
# Symbol format: "AAPL:NASDAQ", "EURUSD:FX", "XAUUSD:OANDA"
# ---------------------------------------------------------------------------
def fetch_klines_tv(symbol: str, exchange: str, days: int) -> np.ndarray:
    """Fetch 1m OHLCV from TradingView via tvdatafeed (unofficial scraper)."""
    try:
        from tvDatafeed import TvDatafeed, Interval
    except ImportError:
        raise ImportError("pip install tvdatafeed")

    tv = TvDatafeed()
    # TradingView caps at ~5000 bars per request; 390 mins/trading day for equities
    # Crypto trades 24/7: 1440 mins/day
    is_crypto = exchange.upper() in ("BINANCE", "COINBASE", "BYBIT", "KRAKEN")
    mins_per_day = 1440 if is_crypto else 390
    n_bars = min(days * mins_per_day, 5000)

    log.info("Fetching %s:%s (%dd, ~%d bars) via TradingView…", symbol, exchange, days, n_bars)
    df = tv.get_hist(symbol, exchange, interval=Interval.in_1_minute, n_bars=n_bars)

    if df is None or len(df) == 0:
        raise ValueError(f"No data for {symbol}:{exchange}")

    arr = df[["open", "high", "low", "close", "volume"]].values.astype(np.float32)
    log.info("%s:%s — %d bars", symbol, exchange, len(arr))
    return arr


# ---------------------------------------------------------------------------
# Hindsight labelling (mirrors generate_training_data.py)
# ---------------------------------------------------------------------------
def _detect_regime(
    window: np.ndarray,
    vol_thresh: float = 0.003,
    trend_thresh: float = 0.005,
) -> str:
    closes = window[:, 3]
    returns = np.diff(closes) / (closes[:-1] + 1e-8)
    if float(np.std(returns)) > vol_thresh:
        return "volatile"
    half = len(closes) // 2
    slope = (closes[half:].mean() - closes[:half].mean()) / (closes[:half].mean() + 1e-8)
    return "trending" if abs(slope) > trend_thresh else "ranging"


def _make_samples(ohlcv: np.ndarray, symbol: str):
    """
    Yields (ohlcv_window, signal_dist, regime_idx) for every sliding window.
    Uses quantile labelling for balanced buy/sell/hold distribution.

    Regime thresholds are computed adaptively from the symbol's own return distribution
    using Fibonacci percentiles so the regime split is symbol-invariant:
      volatile  : top φ⁻³ = top 23.6%  of windows by return std  (76.4th percentile)
      trending  : top φ⁻¹ = top 38.2%  of non-volatile windows by |slope| (61.8th pct)
      ranging   : remaining ~47%
    This prevents high-vol assets (SOL) from being over-classified as volatile.
    """
    n = len(ohlcv)
    needed = WORLD_MODEL_WINDOW + LOOKAHEAD
    if n < needed:
        return

    indices = list(range(0, n - needed + 1, STEP))

    # --- Adaptive regime thresholds (vectorised, O(N)) -----------------------
    from numpy.lib.stride_tricks import sliding_window_view

    closes_all = ohlcv[:, 3]
    returns_all = np.diff(closes_all) / (closes_all[:-1] + 1e-8)

    # Std of returns for every window start position (length = n - W)
    ret_wins = sliding_window_view(returns_all, WORLD_MODEL_WINDOW - 1)
    all_stds = ret_wins.std(axis=1)                # (n - W,) — aligns with window starts

    # volatile threshold: 76.4th percentile (top φ⁻³ = 23.6% are volatile)
    vol_thresh = float(np.percentile(all_stds, 100 * (1 - _INV_PHI3)))

    # Slope for every window start (length = n - W + 1)
    close_wins = sliding_window_view(closes_all, WORLD_MODEL_WINDOW)
    half = WORLD_MODEL_WINDOW // 2
    all_slopes = np.abs(
        close_wins[:, half:].mean(axis=1) - close_wins[:, :half].mean(axis=1)
    ) / (close_wins[:, :half].mean(axis=1) + 1e-8)

    # trending threshold: 61.8th percentile of slopes among non-volatile windows
    # (top 38.2% of non-volatile → trending; bottom 61.8% → ranging)
    non_vol_mask = all_stds < vol_thresh           # (n - W,) — excludes last close_win
    non_vol_slopes = all_slopes[: len(all_stds)][non_vol_mask]
    trend_thresh = (
        float(np.percentile(non_vol_slopes, 100 * _INV_PHI))
        if len(non_vol_slopes) > 0 else 0.005
    )

    log.info(
        "%s: adaptive regime thresholds — vol_std>%.5f (p76.4), trend_slope>%.5f (p61.8)",
        symbol, vol_thresh, trend_thresh,
    )

    fwd_returns = np.array([
        (ohlcv[i + WORLD_MODEL_WINDOW + LOOKAHEAD - 1, 3] - ohlcv[i + WORLD_MODEL_WINDOW - 1, 3])
        / (ohlcv[i + WORLD_MODEL_WINDOW - 1, 3] + 1e-8)
        for i in indices
    ])
    ranks = np.argsort(np.argsort(fwd_returns)).astype(float) / max(len(fwd_returns) - 1, 1)

    regime_map = {"trending": 0, "ranging": 1, "volatile": 2}

    for j, (i, rank) in enumerate(zip(indices, ranks)):
        window = ohlcv[i: i + WORLD_MODEL_WINDOW]
        regime = _detect_regime(window, vol_thresh, trend_thresh)
        regime_idx = regime_map[regime]

        if rank >= _INV_PHI:
            # Buy zone [0.618, 1.0]: conf = 0.618 at boundary, 0.854 (0.618+0.236) at rank=1
            conf = float(_INV_PHI + (rank - _INV_PHI) / _INV_PHI2 * _INV_PHI3)
            dist = [conf, (1 - conf) * _INV_PHI3, (1 - conf) * (1 - _INV_PHI3)]
        elif rank <= _INV_PHI2:
            # Sell zone [0, 0.382]: mirrors buy zone
            conf = float(_INV_PHI + (_INV_PHI2 - rank) / _INV_PHI2 * _INV_PHI3)
            dist = [(1 - conf) * _INV_PHI3, conf, (1 - conf) * (1 - _INV_PHI3)]
        else:
            # Hold zone (0.382, 0.618)
            conf = float(0.5 + abs(rank - 0.5) * _INV_PHI)
            dist = [(1 - conf) / 2, (1 - conf) / 2, conf]

        yield window, dist, regime_idx, i, float(fwd_returns[j])  # i = window start index


def _apply_target_proj(params: dict, x: torch.Tensor) -> torch.Tensor:
    """Apply EMA target projection without nn.Module overhead.

    Implements spr_proj forward pass using a frozen dict of tensors instead of an
    nn.Module so the EMA state never pollutes TradingWorldModel.state_dict().

    Keys follow nn.Sequential indexing: Linear[0], SiLU[1] (no params), Linear[2].
    """
    x = F.linear(x, params['0.weight'], params['0.bias'])
    x = F.silu(x)
    x = F.linear(x, params['2.weight'], params['2.bias'])
    return x


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def pretrain(
    symbols: List[str],
    days: int,
    epochs: int,
    device_str: str,
    interval: str = "1m",
    use_cache: bool = True,
    seed_v1_path: str = None,
    lr: float = None,
    eta_min: float = 1e-6,
    reset_scheduler: bool = False,
    oversample_months_path: str = None,
) -> None:
    # DDP: detect torchrun environment; single-GPU path unchanged
    local_rank, world_size, is_dist = _ddp_setup()
    if is_dist and device_str == "cuda":
        device_str = f"cuda:{local_rank}"
    is_main = (local_rank == 0)

    # Speed optimisations — safe on RTX 3080 / T4 (Ampere/Turing support TF32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True   # auto-tune kernels for fixed input shapes

    world_model = WorldModelWrapper(device=device_str)
    if seed_v1_path:
        log.info("Seeding V2 model from V1 checkpoint: %s", seed_v1_path)
        world_model.load_v1_into_v2(seed_v1_path)
        log.info("V1→V2 weight surgery complete — RSSM and heads warm-started")
    else:
        world_model.load()

    # torch.compile disabled: Chronos T5 internals mix cpu/cuda tensors in ways that
    # confuse FakeTensor device propagation → RuntimeError on every batch → silent no-op.
    model = world_model.model
    if is_dist:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    _lr = lr if lr is not None else WORLD_MODEL_LR
    # Use model.parameters() — DDP forwards to inner module, optimizer tracks same tensors
    optimizer = torch.optim.AdamW(model.parameters(), lr=_lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=world_model.device.type == "cuda")

    # Load step from checkpoint if available
    step = 0 if reset_scheduler else getattr(world_model, "_step", 0)
    log.info("Starting from step %d (reset_scheduler=%s, lr=%.2e, eta_min=%.2e)",
             step, reset_scheduler, _lr, eta_min)

    # Cosine LR schedule: decay from lr → eta_min over total steps.
    # Estimated after data load; scheduler is re-created per run so warm-starts
    # begin at the correct LR for their position in the schedule.
    _scheduler = None  # initialised after total_steps is known

    # Fetch all data upfront
    # Symbols with ":" are TradingView (e.g. "AAPL:NASDAQ", "EURUSD:FX")
    # Symbols without ":" are Binance REST (e.g. "BTCUSDT")
    raw_samples = []
    ohlcv_by_sym: dict = {}  # symbol → full 1m OHLCV array (for macro lookback)
    for sym in symbols:
        try:
            if ":" in sym:
                tv_sym, tv_exchange = sym.split(":", 1)
                ohlcv = fetch_klines_tv(tv_sym, tv_exchange, days)  # TV has no cache
            else:
                ohlcv = fetch_klines(sym, days, interval=interval, use_cache=use_cache)
        except Exception as e:
            log.error("Failed to fetch %s: %s", sym, e)
            continue
        ohlcv_by_sym[sym] = ohlcv
        samples = list(_make_samples(ohlcv, sym))
        log.info("%s: %d training windows", sym, len(samples))
        # tag each sample with its symbol for macro lookback
        raw_samples.extend((w, d, r, i, sym, fr) for w, d, r, i, fr in samples)

    if not raw_samples:
        log.error("No samples — aborting")
        return

    # Split into columnar arrays — eliminates per-batch Python tuple unpacking
    # windows kept as list of views (avoid 21GB copy); dists/regimes pre-stacked
    all_windows     = [s[0] for s in raw_samples]
    all_dists       = np.array([s[1] for s in raw_samples], dtype=np.float32)   # (N, 3)
    all_regimes     = np.array([s[2] for s in raw_samples], dtype=np.int64)     # (N,)
    all_fwd_returns = np.array([s[5] for s in raw_samples], dtype=np.float32)   # (N,) raw fwd return
    N = len(all_windows)

    # Multi-horizon forward return arrays — filled per-symbol in the loop below
    all_fwd_returns_4h  = np.zeros(N, dtype=np.float32)
    all_fwd_returns_8h  = np.zeros(N, dtype=np.float32)
    all_fwd_returns_24h = np.zeros(N, dtype=np.float32)

    # Precompute macro features (4h/1d/7d return + 20d SMA distance) from 1m closes.
    # Vectorised per-symbol: cumsum SMA avoids the 13.9M-iteration Python loop.
    log.info("Precomputing macro features for %d samples...", N)
    all_macro = np.zeros((N, 4), dtype=np.float32)
    all_next_returns = np.zeros(N, dtype=np.float32)  # TD-MPC2: next-step log return target

    # Build index: symbol → array of (sample_idx, win_start)
    sym_sample_idx: dict = {}
    for k, s in enumerate(raw_samples):
        sym_sample_idx.setdefault(s[4], []).append((k, s[3]))

    for sym, entries in sym_sample_idx.items():
        closes = ohlcv_by_sym[sym][:, 3].astype(np.float64)
        M = len(closes)
        # Cumsum for O(1) window-mean lookups
        cs = np.zeros(M + 1, dtype=np.float64)
        np.cumsum(closes, out=cs[1:])

        k_arr = np.array([e[0] for e in entries], dtype=np.int64)
        ws_arr = np.array([e[1] for e in entries], dtype=np.int64)
        ends = ws_arr + WORLD_MODEL_WINDOW - 1          # last candle index of each window

        cur = closes[ends]

        # ret_4h
        mask = ends >= 240
        ref = np.where(mask, closes[np.clip(ends - 240, 0, M - 1)], cur)
        all_macro[k_arr, 0] = np.clip((cur - ref) / (ref + 1e-8), -0.10, 0.10).astype(np.float32)
        all_macro[k_arr[~mask], 0] = 0.0

        # ret_1d
        mask = ends >= 1440
        ref = np.where(mask, closes[np.clip(ends - 1440, 0, M - 1)], cur)
        all_macro[k_arr, 1] = np.clip((cur - ref) / (ref + 1e-8), -0.20, 0.20).astype(np.float32)
        all_macro[k_arr[~mask], 1] = 0.0

        # ret_7d
        mask = ends >= 10080
        ref = np.where(mask, closes[np.clip(ends - 10080, 0, M - 1)], cur)
        all_macro[k_arr, 2] = np.clip((cur - ref) / (ref + 1e-8), -0.40, 0.40).astype(np.float32)
        all_macro[k_arr[~mask], 2] = 0.0

        # sma20d — O(1) via cumsum: mean = (cs[end+1] - cs[start]) / (end - start + 1)
        sma_starts = np.maximum(0, ends - 28800)
        window_len = ends - sma_starts + 1
        valid = window_len >= 100
        sma20d = np.where(valid, (cs[ends + 1] - cs[sma_starts]) / window_len, cur)
        all_macro[k_arr, 3] = np.where(
            valid,
            np.clip((cur - sma20d) / (sma20d + 1e-8), -0.30, 0.30),
            0.0,
        ).astype(np.float32)

        # TD-MPC2 next-step log return: log(close[t+1] / close[t]) — next-step log-return target
        next_ends = np.minimum(ends + 1, M - 1)
        log_rets = np.log((closes[next_ends] + 1e-8) / (closes[ends] + 1e-8))
        all_next_returns[k_arr] = np.clip(log_rets, -0.05, 0.05).astype(np.float32)
        all_next_returns[k_arr[ends >= M - 1]] = 0.0  # last candle of symbol has no next

        # Multi-horizon targets: predict return from window-end bar into future.
        for horizon_h, target_arr in [(4, all_fwd_returns_4h),
                                      (8, all_fwd_returns_8h),
                                      (24, all_fwd_returns_24h)]:
            valid_mask = ends + horizon_h < M
            target_arr[k_arr[valid_mask]] = np.clip(
                (closes[ends[valid_mask] + horizon_h] - closes[ends[valid_mask]])
                / (closes[ends[valid_mask]] + 1e-8),
                -0.50, 0.50,
            ).astype(np.float32)
            target_arr[k_arr[~valid_mask]] = 0.0  # last <horizon bars have no target

    log.info("Macro features ready — non-zero rows: %d / %d", int((all_macro != 0).any(axis=1).sum()), N)

    # Oversample windows from worst-performing months (identified by analyze.py).
    # Each flagged window is duplicated OVERSAMPLE_FACTOR× so the model sees those
    # market conditions more often during training.
    OVERSAMPLE_FACTOR = 3
    if oversample_months_path:
        import json as _json
        from datetime import datetime as _dt
        with open(oversample_months_path) as _f:
            _bad_months = {(_e["year"], _e["month"]) for _e in _json.load(_f)}
        log.info("Oversampling %d worst months (×%d): %s",
                 len(_bad_months), OVERSAMPLE_FACTOR,
                 sorted(_bad_months))

        # Build per-symbol timestamp arrays from _ts.npy companions
        _sym_ts: dict = {}
        for sym in sym_sample_idx:
            _ts_path = os.path.join(CACHE_DIR, f"{sym}_{days}d_{interval}_ts.npy")
            if os.path.exists(_ts_path):
                _sym_ts[sym] = np.load(_ts_path)  # unix seconds per candle
            else:
                log.warning("%s: no _ts.npy found — skipping oversample for this symbol", sym)

        _dup_idx = []
        for sym, entries in sym_sample_idx.items():
            if sym not in _sym_ts:
                continue
            ts_arr = _sym_ts[sym]
            for k, win_start in entries:
                candle_idx = min(win_start + WORLD_MODEL_WINDOW - 1, len(ts_arr) - 1)
                dt = _dt.utcfromtimestamp(int(ts_arr[candle_idx]))
                if (dt.year, dt.month) in _bad_months:
                    _dup_idx.extend([k] * (OVERSAMPLE_FACTOR - 1))

        if _dup_idx:
            dup = np.array(_dup_idx, dtype=np.int64)
            all_windows          = all_windows + [all_windows[i] for i in dup]
            all_regimes          = np.concatenate([all_regimes,          all_regimes[dup]])
            all_macro            = np.concatenate([all_macro,            all_macro[dup]])
            all_next_returns     = np.concatenate([all_next_returns,     all_next_returns[dup]])
            all_fwd_returns      = np.concatenate([all_fwd_returns,      all_fwd_returns[dup]])
            all_fwd_returns_4h   = np.concatenate([all_fwd_returns_4h,   all_fwd_returns_4h[dup]])
            all_fwd_returns_8h   = np.concatenate([all_fwd_returns_8h,   all_fwd_returns_8h[dup]])
            all_fwd_returns_24h  = np.concatenate([all_fwd_returns_24h,  all_fwd_returns_24h[dup]])
            N = len(all_windows)
            log.info("Oversampling added %d duplicates → %d total windows", len(_dup_idx), N)
        else:
            log.warning("Oversample: no matching windows found — check _ts.npy files and month list")

    del raw_samples, ohlcv_by_sym  # free memory


    total_steps = (N // BATCH_SIZE) * epochs
    # Warm-start: advance scheduler to current step position so LR is correct
    _scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=eta_min
    )
    for _ in range(step):
        _scheduler.step()
    log.info(
        "Total samples: %d  |  epochs: %d  |  batch_size: %d  |  total_steps: %d  |  lr: %.2e→1e-5",
        N, epochs, BATCH_SIZE, total_steps, WORLD_MODEL_LR,
    )

    # EMA for smooth loss display (α=0.02 ≈ 50-step window)
    _ema_loss = None
    _ema_alpha = 0.02

    def _prep_batch(idx):
        """CPU batch prep — run in prefetch thread while GPU works on prior batch."""
        if len(idx) == 0:
            return None
        windows_np = np.stack([all_windows[i] for i in idx])   # (B, T, 5)
        # 18-dim book: [0:3]=ADX [3:11]=VP [11:14]=ICT [14:18]=macro
        macro_feats = all_macro[idx]                     # (B, 4)
        ohlcv_t, book_t = world_model.preprocess_batch(windows_np, macro_feats=macro_feats)
        return (ohlcv_t, book_t, all_regimes[idx], all_next_returns[idx], all_fwd_returns[idx],
                all_fwd_returns_4h[idx], all_fwd_returns_8h[idx], all_fwd_returns_24h[idx])

    # SPR: unwrap DDP if present so we can access model attributes directly.
    # EMA target params live here as a plain dict — not in TradingWorldModel.state_dict().
    raw_model = model.module if hasattr(model, 'module') else model
    spr_target_params = {n: p.data.clone() for n, p in raw_model.spr_proj.named_parameters()}

    for epoch in range(1, epochs + 1):
        # Deterministic permutation per epoch — same across all ranks so stride
        # sharding produces non-overlapping, exhaustive coverage of the dataset.
        perm = np.random.RandomState(epoch * 31337).permutation(N)
        if is_dist:
            perm = perm[local_rank::world_size]
        model.train()

        epoch_loss = 0.0
        num_steps = 0

        n_shard = len(perm)
        batches = range(0, n_shard - BATCH_SIZE + 1, BATCH_SIZE)
        prefetch_q: queue.Queue = queue.Queue(maxsize=2)

        def _prefetch_worker():
            try:
                for bs in batches:
                    batch = _prep_batch(perm[bs: bs + BATCH_SIZE])
                    if batch is not None:
                        prefetch_q.put(batch)
            except Exception as e:
                log.error("Prefetch worker crashed: %s", e, exc_info=True)
            finally:
                prefetch_q.put(None)  # sentinel always sent

        _worker = threading.Thread(target=_prefetch_worker, daemon=True)
        _worker.start()

        while True:
            item = prefetch_q.get()
            if item is None:
                break
            (ohlcv_t, book_t, batch_regimes, batch_returns, batch_fwd_returns,
             batch_fwd_4h, batch_fwd_8h, batch_fwd_24h) = item

            try:

                with torch.amp.autocast(
                    device_type=world_model.device.type,
                    enabled=world_model.device.type == "cuda",
                ):
                    # SPR requires per-timestep (h, z) — use forward_with_h_sequence
                    # instead of model(ohlcv_t, book_t) which discards intermediate states.
                    obs_embeds = raw_model.encoder(ohlcv_t, book_t)   # (B, T', hidden)
                    out, _, _, h_seq, z_seq = raw_model.forward_with_h_sequence(obs_embeds)
                    T_prime = obs_embeds.shape[1]

                    regime_t = torch.from_numpy(batch_regimes).to(
                        world_model.device, non_blocking=True)               # (B,)
                    regime_loss = F.cross_entropy(out.regime_logits, regime_t)
                    return_targets = torch.from_numpy(batch_returns).to(
                        world_model.device, non_blocking=True)               # (B,)
                    return_loss = F.huber_loss(out.return_pred, return_targets.float(), delta=1e-4)

                    # Directional aux loss — train latents to predict N-bar forward direction
                    dir_targets = torch.from_numpy(
                        (batch_fwd_returns > 0).astype(np.float32)
                    ).to(world_model.device, non_blocking=True)              # (B,)
                    dir_loss = F.binary_cross_entropy_with_logits(out.dir_logit, dir_targets)

                    # Multi-horizon return head losses (0.1 per head = 0.3 total)
                    combined_final = torch.cat([h_seq[:, -1, :], z_seq[:, -1, :]], dim=-1)
                    fwd_4h_t  = torch.from_numpy(batch_fwd_4h.astype(np.float32)).to(world_model.device)
                    fwd_8h_t  = torch.from_numpy(batch_fwd_8h.astype(np.float32)).to(world_model.device)
                    fwd_24h_t = torch.from_numpy(batch_fwd_24h.astype(np.float32)).to(world_model.device)
                    rh_4h_loss  = F.mse_loss(raw_model.ret_head_4h(combined_final).squeeze(-1),
                                             symlog(fwd_4h_t))
                    rh_8h_loss  = F.mse_loss(raw_model.ret_head_8h(combined_final).squeeze(-1),
                                             symlog(fwd_8h_t))
                    rh_24h_loss = F.mse_loss(raw_model.ret_head_24h(combined_final).squeeze(-1),
                                             symlog(fwd_24h_t))
                    mh_loss = 0.1 * (rh_4h_loss + rh_8h_loss + rh_24h_loss)

                    # SPR k-step cosine prediction losses (k=1,2,4)
                    spr_losses = []
                    for k in [1, 2, 4]:
                        valid = T_prime - k
                        if valid <= 0:
                            continue  # window too short for this k — skip
                        h_t   = h_seq[:, :valid, :]              # (B, valid, 256)
                        z_t   = z_seq[:, :valid, :]              # (B, valid, 128)
                        h_tk  = h_seq[:, k:k+valid, :].detach()  # target: detached per BYOL design
                        z_tk  = z_seq[:, k:k+valid, :].detach()
                        hz_t  = torch.cat([h_t,  z_t],  dim=-1).reshape(-1, 384)
                        hz_tk = torch.cat([h_tk, z_tk], dim=-1).reshape(-1, 384)
                        online = raw_model.spr_pred(raw_model.spr_proj(hz_t))
                        with torch.no_grad():
                            target_rep = _apply_target_proj(spr_target_params, hz_tk)
                        loss_k = (1 - F.cosine_similarity(online, target_rep)).mean()
                        spr_losses.append(loss_k)
                    spr_loss = (torch.stack(spr_losses).mean() * 0.5
                                if spr_losses else obs_embeds.new_zeros([]))

                    loss = (WORLD_MODEL_REGIME_WEIGHT * symlog(regime_loss)
                            + WORLD_MODEL_KL_WEIGHT * symlog(out.rssm_kl)
                            + WORLD_MODEL_RETURN_WEIGHT * symlog(return_loss + 1e-8)
                            + DIR_AUX_COEF * symlog(dir_loss)
                            + mh_loss
                            + spr_loss)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                _scheduler.step()

                # EMA update for SPR target projection (τ=0.99)
                for _spr_name, _spr_param in raw_model.spr_proj.named_parameters():
                    spr_target_params[_spr_name].mul_(0.99).add_(_spr_param.data, alpha=0.01)

                step += 1
                num_steps += 1
                loss_val = loss.item()
                # Average loss across ranks for accurate display
                if is_dist:
                    _lt = torch.tensor(loss_val, device=world_model.device)
                    dist.all_reduce(_lt, op=dist.ReduceOp.AVG)
                    loss_val = _lt.item()
                epoch_loss += loss_val
                _ema_loss = loss_val if _ema_loss is None else (1 - _ema_alpha) * _ema_loss + _ema_alpha * loss_val

                if is_main and step % 50 == 0:
                    lr_now = _scheduler.get_last_lr()[0]
                    log.info("Epoch %d | step %d | loss %.4f | dir_loss %.4f | lr %.2e",
                             epoch, step, _ema_loss, dir_loss.item(), lr_now)

                if is_main and step % SAVE_EVERY == 0:
                    world_model._step = step
                    world_model.save()
                    log.info("Checkpoint saved at step %d", step)

            except Exception as e:
                log.warning("Batch failed: %s", e)
                optimizer.zero_grad()
                continue

        _worker.join()
        avg_loss = epoch_loss / max(num_steps, 1)
        if is_main:
            log.info("Epoch %d complete | steps: %d | avg loss: %.4f", epoch, num_steps, avg_loss)

    if is_main:
        world_model._step = step
        world_model.save()
        log.info("Pre-training complete. Final step: %d. Checkpoint saved.", step)
    _ddp_cleanup()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train world model on historical OHLCV data")
    parser.add_argument("--symbols", default=",".join(SYMBOLS))
    parser.add_argument("--interval", default="1m", choices=list(_INTERVAL_MS),
                        help="Candle interval for Binance fetch (default: 1m)")
    parser.add_argument("--days", type=int, default=30, help="Days of history to fetch")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-only", action="store_true",
                        help="Download and cache data to disk, then exit (no training)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-download even if cache exists")
    parser.add_argument("--seed-v1", default=None, metavar="PATH",
                        help="Seed V2 model from a V1 checkpoint dir via weight surgery "
                             "(book_proj extended to 26 dims, all other weights transferred). "
                             "Use this when retraining with VP+Fib features to avoid "
                             "re-learning RSSM temporal structure from scratch.")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate (default: WORLD_MODEL_LR from config, 1e-4)")
    parser.add_argument("--eta-min", type=float, default=1e-6,
                        help="Cosine annealing floor LR (default: 1e-6)")
    parser.add_argument("--reset-scheduler", action="store_true",
                        help="Restart cosine LR schedule from step 0 (fresh cycle from --lr)")
    parser.add_argument("--oversample-months", default=None, metavar="PATH",
                        help="JSON file of worst months from analyze.py; windows in those months "
                             "are duplicated 3× to force more gradient signal on hard periods")
    args = parser.parse_args()

    symbols = args.symbols.split(",")

    if args.cache_only:
        os.makedirs(CACHE_DIR, exist_ok=True)
        for sym in symbols:
            fetch_klines(sym, args.days, interval=args.interval, use_cache=False)
        log.info("Cache download complete. Run without --cache-only to train.")
    else:
        pretrain(
            symbols=symbols,
            days=args.days,
            epochs=args.epochs,
            device_str=args.device,
            interval=args.interval,
            use_cache=not args.no_cache,
            seed_v1_path=args.seed_v1,
            lr=args.lr,
            eta_min=args.eta_min,
            reset_scheduler=args.reset_scheduler,
            oversample_months_path=args.oversample_months,
        )
