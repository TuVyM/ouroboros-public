# orderflow/features.py
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd

WINDOW = 89  # matches WORLD_MODEL_WINDOW in config

FEATURE_NAMES = [
    # --- Orderflow (from Binance klines + funding) ---
    "buy_ratio_1h",       # 0  — taker buy fraction this bar [0,1]
    "delta_ratio_8h",     # 1  — 8h cumulative delta / total vol [-1,1]
    "vwap_deviation",     # 2  — (close - vwap) / vwap
    "funding_rate",       # 3  — 8h perp funding rate
    "ret_1h",             # 4  — 1-bar return
    "ret_8h",             # 5  — 8-bar return
    "ret_24h",            # 6  — 24-bar return
    "atr_ratio",          # 7  — ATR(14) / close
    "sma24_dist",         # 8  — (close - SMA24) / SMA24
    "range_position",     # 9  — (close - window_low) / (window_high - window_low)
    # --- OB-anchored Fibonacci distances ---
    "ob_fib_dist_1",      # 10
    "ob_fib_dist_5",      # 11
    "ob_fib_dist_15",     # 12
    "ob_fib_dist_60",     # 13
    "ob_fib_dist_89",     # 14
    # --- Fibonacci channel position ---
    "fib_ch_236",         # 15
    "fib_ch_500",         # 16
    "fib_ch_764",         # 17
    # --- ICT Smart Money ---
    "ict_fvg",            # 18
    "ict_sweep",          # 19
    "ict_killzone",       # 20
    # --- TPO Market Profile ---
    "tpo_poc",            # 21
    "tpo_vah",            # 22
    "tpo_val",            # 23
]
N_FEATURES = len(FEATURE_NAMES)


def build_features(
    symbol:     str,
    ohlcv:      np.ndarray,      # (N, 5) float32: open high low close volume
    timestamps: np.ndarray,      # (N,) int64: unix seconds UTC
    data_dir:   Union[str, Path, None] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (M, 24) float32 feature matrix where M = N - WINDOW.

    data_dir: directory containing {symbol}_aggtrades_1h.parquet, _funding.parquet.
              Defaults to data/orderflow/ relative to this file's parent.

    Returns (features, timestamps) — both arrays have length M.
    Raises FileNotFoundError if parquet files are missing.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "orderflow"
    data_dir = Path(data_dir)

    agg_path  = data_dir / f"{symbol}_aggtrades_1h.parquet"
    fund_path = data_dir / f"{symbol}_funding.parquet"
    for p in (agg_path, fund_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Orderflow data not found: {p}\n"
                "Run: python -m orderflow.fetch --symbol BTCUSDT --days 1095"
            )

    # Load and index orderflow parquets on UTC-floored timestamp
    agg  = pd.read_parquet(agg_path).set_index("timestamp")
    fund = pd.read_parquet(fund_path).set_index("timestamp")

    # Build hourly DatetimeIndex from OHLCV timestamps (unix seconds)
    idx = pd.to_datetime(timestamps, unit="s", utc=True).floor("1h")

    c  = ohlcv[:, 3].astype(np.float64)  # close
    h  = ohlcv[:, 1].astype(np.float64)  # high
    lo = ohlcv[:, 2].astype(np.float64)  # low

    n = len(c)

    # ---- Orderflow features (from parquet, aligned to OHLCV index) ----
    def _align(df: pd.DataFrame, col: str) -> np.ndarray:
        s = df[col].reindex(idx, method="ffill").values.astype(np.float64)
        return np.nan_to_num(s, nan=0.0)

    buy_vol      = _align(agg, "buy_vol")
    sell_vol     = _align(agg, "sell_vol")
    total_vol    = _align(agg, "total_vol")
    vwap         = _align(agg, "vwap")
    funding_rate = _align(fund, "funding_rate")

    buy_ratio_1h   = buy_vol / (total_vol + 1e-12)
    vwap_deviation = (c - vwap) / (vwap + 1e-12)

    delta        = buy_vol - sell_vol
    delta_s      = pd.Series(delta)
    total_vol_s  = pd.Series(total_vol)
    delta_ratio_8h = (
        delta_s.rolling(8, min_periods=1).sum()
        / (total_vol_s.rolling(8, min_periods=1).sum() + 1e-12)
    ).fillna(0.0).values

    # ---- Price features ----
    ret_1h  = np.zeros(n); ret_1h[1:]   = (c[1:]  - c[:-1])  / (c[:-1]  + 1e-12)
    ret_8h  = np.zeros(n); ret_8h[8:]   = (c[8:]  - c[:-8])  / (c[:-8]  + 1e-12)
    ret_24h = np.zeros(n); ret_24h[24:] = (c[24:] - c[:-24]) / (c[:-24] + 1e-12)

    # ATR(14)
    tr = np.concatenate([[0.0], np.maximum(
        h[1:] - lo[1:],
        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(lo[1:] - c[:-1])),
    )])
    atr       = pd.Series(tr).rolling(14, min_periods=1).mean().values
    atr_ratio = atr / (c + 1e-12)

    # SMA-24 distance
    sma24      = pd.Series(c).rolling(24, min_periods=1).mean().values
    sma24_dist = (c - sma24) / (sma24 + 1e-12)

    # Range position
    h_s  = pd.Series(h)
    lo_s = pd.Series(lo)
    win_h = h_s.rolling(WINDOW, min_periods=1).max().values
    win_l = lo_s.rolling(WINDOW, min_periods=1).min().values
    rng   = win_h - win_l
    range_position = np.where(rng > 1e-12, (c - win_l) / rng, 0.5)

    # ---- Stack orderflow + price features, drop first WINDOW rows ----
    X_base = np.column_stack([
        buy_ratio_1h, delta_ratio_8h, vwap_deviation,
        funding_rate,
        ret_1h, ret_8h, ret_24h,
        atr_ratio, sma24_dist, range_position,
    ])[WINDOW:].astype(np.float32)   # (N - WINDOW, 10)

    # ---- Structural features (OB-Fib, Fib-channel, ICT, TPO) ----
    from orderflow.fib_ob import build_structural_features
    X_struct = build_structural_features(ohlcv.astype(np.float32), timestamps)  # (N - WINDOW, 14)

    X = np.concatenate([X_base, X_struct], axis=1)  # (N - WINDOW, 24)

    out_ts = timestamps[WINDOW:]
    return X, out_ts


N_EMB_FEATURES = 64
N_FEATURES_WITH_EMB = N_FEATURES + N_EMB_FEATURES   # 88


def build_features_with_embedding(
    symbol:     str,
    ohlcv:      np.ndarray,
    timestamps: np.ndarray,
    encoder,                   # transformer.encoder.MultiScaleEncoder
    data_dir=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (M, 88) feature matrix — existing 24 features + 64-dim encoder embedding.

    Bar alignment: output row i corresponds to ohlcv[WINDOW + i].
    encoder.batch_embed is called with out_indices = [WINDOW, WINDOW+1, ..., WINDOW+M-1].
    """
    X, out_ts = build_features(symbol, ohlcv, timestamps, data_dir=data_dir)
    M = len(X)
    out_indices = np.arange(WINDOW, WINDOW + M)
    embeddings = encoder.batch_embed(ohlcv, out_indices)       # (M, 64)
    return np.concatenate([X, embeddings], axis=1).astype(np.float32), out_ts
