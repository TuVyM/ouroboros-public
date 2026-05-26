# orderflow/range_features.py
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

from orderflow.features import build_features, FEATURE_NAMES, WINDOW

RANGE_FEATURE_NAMES = FEATURE_NAMES + [
    "rsi_bull_div",  # 24 — 1 if price lower low AND RSI higher low (4-bar)
    "rsi_bear_div",  # 25 — 1 if price higher high AND RSI lower high (4-bar)
    "val_dist",      # 26 — (price - VAL) / price  [+ = above VAL]
    "vah_dist",      # 27 — (VAH - price) / price  [+ = below VAH]
    "poc_dist",      # 28 — (price - POC) / price  [+ = above POC]
]
N_RANGE_FEATURES = len(RANGE_FEATURE_NAMES)   # 29

_ADX_LOOKBACK  = 60   # max bars to walk back for range period
_VOLUME_BINS   = 50
_VALUE_AREA_PCT = 0.70


def _rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder-smoothed RSI(period). Returns array same length as closes."""
    closes = closes.astype(np.float64)
    delta  = np.diff(closes, prepend=closes[0])
    gain   = np.where(delta > 0, delta, 0.0)
    loss   = np.where(delta < 0, -delta, 0.0)
    avg_g  = np.zeros(len(closes))
    avg_l  = np.zeros(len(closes))
    if len(closes) > period:
        avg_g[period] = gain[1:period + 1].mean()
        avg_l[period] = loss[1:period + 1].mean()
        for i in range(period + 1, len(closes)):
            avg_g[i] = (avg_g[i - 1] * (period - 1) + gain[i]) / period
            avg_l[i] = (avg_l[i - 1] * (period - 1) + loss[i]) / period
    rs  = avg_g / (avg_l + 1e-8)
    return 100.0 - 100.0 / (1.0 + rs)


def _compute_value_area(ohlcv_window: np.ndarray) -> dict:
    """VAL, VAH, POC from a volume profile histogram over ohlcv_window (N,5)."""
    lo  = ohlcv_window[:, 2].astype(np.float64)
    hi  = ohlcv_window[:, 1].astype(np.float64)
    vol = ohlcv_window[:, 4].astype(np.float64)

    price_min = lo.min()
    price_max = hi.max()
    if price_max <= price_min:
        mid = (price_min + price_max) / 2.0
        return {"val": mid, "poc": mid, "vah": mid}

    edges = np.linspace(price_min, price_max, _VOLUME_BINS + 1)
    bins  = np.zeros(_VOLUME_BINS)
    for i in range(len(ohlcv_window)):
        bar_lo, bar_hi, bar_vol = lo[i], hi[i], vol[i]
        overlap_lo = np.maximum(edges[:-1], bar_lo)
        overlap_hi = np.minimum(edges[1:],  bar_hi)
        overlap    = np.maximum(0.0, overlap_hi - overlap_lo)
        bar_range  = max(bar_hi - bar_lo, 1e-8)
        bins      += bar_vol * (overlap / bar_range)

    poc_idx = int(np.argmax(bins))
    poc     = (edges[poc_idx] + edges[poc_idx + 1]) / 2.0

    total_vol    = bins.sum()
    target_vol   = total_vol * _VALUE_AREA_PCT
    val_idx      = poc_idx
    vah_idx      = poc_idx
    included_vol = bins[poc_idx]

    while included_vol < target_vol:
        can_dn = val_idx > 0
        can_up = vah_idx < _VOLUME_BINS - 1
        if not can_dn and not can_up:
            break
        add_dn = bins[val_idx - 1] if can_dn else -1.0
        add_up = bins[vah_idx + 1] if can_up else -1.0
        if add_up >= add_dn:
            vah_idx += 1; included_vol += bins[vah_idx]
        else:
            val_idx -= 1; included_vol += bins[val_idx]

    val = (edges[val_idx] + edges[val_idx + 1]) / 2.0
    vah = (edges[vah_idx] + edges[vah_idx + 1]) / 2.0
    return {"val": float(val), "poc": float(poc), "vah": float(vah)}


def _find_range_start(adx_series: np.ndarray, current_idx: int) -> int:
    """Walk back to find where ADX last crossed below 25, max _ADX_LOOKBACK bars."""
    start = max(0, current_idx - _ADX_LOOKBACK)
    for i in range(current_idx - 1, start - 1, -1):
        if i < len(adx_series) and adx_series[i] > 25.0:
            return i + 1
    return start


def build_range_features(
    symbol:     str,
    ohlcv:      np.ndarray,
    timestamps: np.ndarray,
    data_dir:   Union[str, Path, None] = None,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    Build (M, 29) float32 range feature matrix.

    Calls build_features() internally — callers must NOT also call build_features()
    to avoid double-computing the 24 base features.

    Returns:
        X         — (M, 29) float32
        out_ts    — (M,) int64 timestamps
        va_levels — list of M dicts {"val", "poc", "vah"} in price units
    """
    from lgbm.regime_detector import RegimeDetector

    X_base, out_ts = build_features(symbol, ohlcv, timestamps, data_dir=data_dir)
    M = len(X_base)
    if M == 0:
        return X_base, out_ts, []

    closes = ohlcv[:, 3].astype(np.float64)
    rsi    = _rsi(closes, period=14)
    adx    = RegimeDetector.compute_adx(ohlcv)   # shape (N-1,)

    ts_to_idx    = {int(t): i for i, t in enumerate(timestamps)}
    base_indices = np.array([ts_to_idx[int(t)] for t in out_ts])

    new_feats = np.zeros((M, 5), dtype=np.float32)
    va_levels: List[dict] = []

    for i, bar_idx in enumerate(base_indices):
        # RSI divergence (4-bar lookback)
        n4       = bar_idx - 4
        bull_div = bear_div = 0
        if n4 >= 0:
            bull_div = int(closes[bar_idx] < closes[n4] and rsi[bar_idx] > rsi[n4])
            bear_div = int(closes[bar_idx] > closes[n4] and rsi[bar_idx] < rsi[n4])

        # Dynamic value area
        adx_idx     = min(bar_idx, len(adx) - 1)
        range_start = _find_range_start(adx, adx_idx)
        window      = ohlcv[range_start:bar_idx + 1]
        va          = _compute_value_area(window) if len(window) > 0 else \
                      {"val": float(closes[bar_idx]), "poc": float(closes[bar_idx]),
                       "vah": float(closes[bar_idx])}

        price    = float(closes[bar_idx])
        val_dist = (price - va["val"]) / (price + 1e-8)
        vah_dist = (va["vah"] - price) / (price + 1e-8)
        poc_dist = (price - va["poc"]) / (price + 1e-8)

        new_feats[i] = [bull_div, bear_div, val_dist, vah_dist, poc_dist]
        va_levels.append(va)

    X = np.concatenate([X_base, new_feats], axis=1).astype(np.float32)
    return X, out_ts, va_levels


from orderflow.features import WINDOW, N_EMB_FEATURES  # noqa: E402

N_RANGE_FEATURES_WITH_EMB = N_RANGE_FEATURES + N_EMB_FEATURES   # 93


def build_range_features_with_embedding(
    symbol:     str,
    ohlcv:      np.ndarray,
    timestamps: np.ndarray,
    encoder,
    data_dir=None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Returns (M, 93) feature matrix — existing 29 range features + 64-dim encoder embedding.

    Bar alignment: same as build_features_with_embedding.
    """
    X, out_ts, va_levels = build_range_features(symbol, ohlcv, timestamps, data_dir=data_dir)
    M = len(X)
    out_indices = np.arange(WINDOW, WINDOW + M)
    embeddings = encoder.batch_embed(ohlcv, out_indices)       # (M, 64)
    return np.concatenate([X, embeddings], axis=1).astype(np.float32), out_ts, va_levels
