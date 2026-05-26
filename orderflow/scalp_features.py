# orderflow/scalp_features.py
import random
from collections import deque

import numpy as np

SCALP_FEATURE_NAMES = [
    "ret_1m",           # 0
    "ret_3m",           # 1
    "ret_10m",          # 2
    "ret_30m",          # 3
    "vwap_dev_1m",      # 4
    "delta_ratio_10m",  # 5
    "ob_imbalance",     # 6
    "atr_ratio_1m",     # 7
    "rsi_14_1m",        # 8
    "range_pos_1m",     # 9
    "vol_ratio_10m",    # 10
    "liq_score",        # 11
    "htf_signal",       # 12
    "htf_conf",         # 13
    "htf_regime",       # 14
]
N_SCALP_FEATURES = 15

_EPS = 1e-8


def _rsi14(closes: np.ndarray) -> float:
    """RSI(14) on the last 15 closes. Returns 50.0 if insufficient data."""
    if len(closes) < 15:
        return 50.0
    deltas = np.diff(closes[-15:])
    up = deltas.clip(min=0.0)
    dn = (-deltas).clip(min=0.0)
    avg_up = up.mean()
    avg_dn = dn.mean()
    if avg_dn < _EPS:
        return 100.0 if avg_up > _EPS else 50.0
    return float(100.0 - 100.0 / (1.0 + avg_up / avg_dn))


def _atr14(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
    """Average true range over last 14 bars (simple mean of TR, not Wilder-smoothed)."""
    if len(highs) < 2:
        return float(highs[-1] - lows[-1]) if len(highs) == 1 else 0.0
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1]),
        ),
    )
    return float(tr[-14:].mean())


def build_scalp_features(
    buf: deque,
    htf_signal: int,
    htf_conf: float,
    htf_regime: int,
    training: bool = False,
) -> "np.ndarray | None":
    """
    Returns (15,) float32 feature vector, or None if len(buf) < 30.

    buf: deque of (ohlcv, ob_imbalance, liq_vol) tuples, maxlen=30
      ohlcv: (5,) float32 — open, high, low, close, volume
      ob_imbalance: float — (bid_vol - ask_vol)/(bid_vol + ask_vol); 0.0 if unavailable
      liq_vol: float — signed liquidation notional this bar; positive=short-squeeze bullish
    training: if True, randomly zero ob_imbalance (feat 6) and liq_score (feat 11) at 50% rate
    """
    if len(buf) < 30:
        return None

    items   = list(buf)
    ohlcvs  = np.array([b[0] for b in items], dtype=np.float64)  # (30, 5)
    ob_imbs = np.array([b[1] for b in items], dtype=np.float64)  # (30,)
    liqs    = np.array([b[2] for b in items], dtype=np.float64)  # (30,)

    closes  = ohlcvs[:, 3]
    highs   = ohlcvs[:, 1]
    lows    = ohlcvs[:, 2]
    volumes = ohlcvs[:, 4]

    c = closes[-1]

    # ── Returns (log) ──────────────────────────────────────────────────────────
    ret_1m  = float(np.log((closes[-1]  + _EPS) / (closes[-2]  + _EPS)))
    ret_3m  = float(np.log((closes[-1]  + _EPS) / (closes[-4]  + _EPS)))
    ret_10m = float(np.log((closes[-1]  + _EPS) / (closes[-11] + _EPS)))
    ret_30m = float(np.log((closes[-1]  + _EPS) / (closes[0]   + _EPS)))

    # ── VWAP deviation (volume-weighted close over 30-bar window) ─────────────
    total_vol = volumes.sum()
    vwap_30   = float((closes * volumes).sum() / (total_vol + _EPS))
    vwap_dev  = float((c - vwap_30) / (vwap_30 + _EPS))

    # ── Delta ratio proxy over last 10 bars ────────────────────────────────────
    # buy_vol ≈ volume where close > open; sell_vol ≈ volume where close <= open
    # Consistent between training (OHLCV-only 1m cache) and inference.
    recent = ohlcvs[-10:]
    bar_sign = np.sign(recent[:, 3] - recent[:, 0])  # close - open
    vol_10   = recent[:, 4]
    tot_10   = vol_10.sum() + _EPS
    delta_ratio = float((bar_sign * vol_10).sum() / tot_10)

    # ── Order-book imbalance (most recent bar) ─────────────────────────────────
    ob_imbalance = float(ob_imbs[-1])
    if training and random.random() < 0.5:
        ob_imbalance = 0.0

    # ── ATR(14) / close ────────────────────────────────────────────────────────
    atr      = _atr14(highs, lows, closes)
    atr_ratio = float(atr / (c + _EPS))

    # ── RSI(14) ────────────────────────────────────────────────────────────────
    rsi = _rsi14(closes)

    # ── Range position: (close - low_20m) / (high_20m - low_20m) ─────────────
    low_20   = float(lows[-20:].min())
    high_20  = float(highs[-20:].max())
    range_pos = float((c - low_20) / (high_20 - low_20 + _EPS))
    range_pos = float(np.clip(range_pos, 0.0, 1.0))

    # ── Volume ratio: current / 10-bar mean ───────────────────────────────────
    mean_vol_10 = float(volumes[-10:].mean()) + _EPS
    vol_ratio   = float(volumes[-1] / mean_vol_10)

    # ── Liq score: signed liq sum over last 10 bars / ATR ─────────────────────
    liq_sum = float(liqs[-10:].sum())
    if training and random.random() < 0.5:
        liq_sum = 0.0
    liq_score = float(liq_sum / (atr + _EPS)) if atr > _EPS else 0.0

    return np.array([
        ret_1m, ret_3m, ret_10m, ret_30m,
        vwap_dev, delta_ratio, ob_imbalance,
        atr_ratio, rsi, range_pos, vol_ratio, liq_score,
        float(htf_signal), float(htf_conf), float(htf_regime),
    ], dtype=np.float32)
