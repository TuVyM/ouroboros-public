import numpy as np
import pandas as pd
import pytest
import tempfile

from orderflow.features import build_features, FEATURE_NAMES, N_FEATURES

WINDOW = 89


def _make_ohlcv(n=200, base_close=95000.0):
    closes = base_close + np.random.randn(n).cumsum() * 100
    closes = np.maximum(closes, 1.0)
    highs  = closes * 1.005
    lows   = closes * 0.995
    opens  = np.roll(closes, 1); opens[0] = closes[0]
    vols   = np.abs(np.random.randn(n)) * 1000 + 500
    return np.column_stack([opens, highs, lows, closes, vols]).astype(np.float32)


def _make_orderflow_parquet(tmp_dir, symbol, n=200, base_ts_s=1_700_000_000):
    """Write synthetic orderflow parquet files."""
    ts = pd.to_datetime(
        [base_ts_s + i * 3600 for i in range(n)], unit="s", utc=True
    )
    # Generate total_vol first, then split into buy/sell
    total_vol = np.random.rand(n) * 200 + 100
    buy_ratios = np.random.rand(n)
    buy_vol = total_vol * buy_ratios
    sell_vol = total_vol * (1 - buy_ratios)

    agg_df = pd.DataFrame({
        "timestamp":        ts,
        "buy_vol":          buy_vol,
        "sell_vol":         sell_vol,
        "total_vol":        total_vol,
        "vwap":             95000.0 + np.random.randn(n) * 100,
        "large_trade_vol":  np.random.rand(n) * 10,
        "buy_ratio":        buy_ratios,
    })
    fund_df = pd.DataFrame({
        "timestamp":    ts,
        "funding_rate": np.random.randn(n) * 0.0001,
    })
    oi_df = pd.DataFrame({
        "timestamp": ts,
        "oi":        np.random.rand(n) * 1e9 + 1e10,
    })
    agg_df.to_parquet(Path(tmp_dir) / f"{symbol}_aggtrades_1h.parquet",  index=False)
    fund_df.to_parquet(Path(tmp_dir) / f"{symbol}_funding.parquet",       index=False)
    oi_df.to_parquet(  Path(tmp_dir) / f"{symbol}_oi.parquet",            index=False)
    return base_ts_s


def test_output_shape(tmp_path):
    n = 200
    ohlcv = _make_ohlcv(n)
    base_ts_s = _make_orderflow_parquet(tmp_path, "BTCUSDT", n)
    timestamps = np.array([base_ts_s + i * 3600 for i in range(n)], dtype=np.int64)

    X, out_ts = build_features("BTCUSDT", ohlcv, timestamps, data_dir=tmp_path)

    # First WINDOW rows dropped + last 0 rows = n - WINDOW
    assert X.shape[0] == n - WINDOW
    assert X.shape[1] == N_FEATURES
    assert X.dtype == np.float32


def test_feature_names_length():
    assert len(FEATURE_NAMES) == N_FEATURES == 24


def test_no_nan_in_output(tmp_path):
    n = 200
    ohlcv = _make_ohlcv(n)
    _make_orderflow_parquet(tmp_path, "BTCUSDT", n)
    timestamps = np.array([1_700_000_000 + i * 3600 for i in range(n)], dtype=np.int64)
    X, _ = build_features("BTCUSDT", ohlcv, timestamps, data_dir=tmp_path)
    assert not np.isnan(X).any(), "NaN values in feature matrix"


def test_buy_ratio_in_range(tmp_path):
    n = 200
    ohlcv = _make_ohlcv(n)
    _make_orderflow_parquet(tmp_path, "BTCUSDT", n)
    timestamps = np.array([1_700_000_000 + i * 3600 for i in range(n)], dtype=np.int64)
    X, _ = build_features("BTCUSDT", ohlcv, timestamps, data_dir=tmp_path)
    col = FEATURE_NAMES.index("buy_ratio_1h")
    assert X[:, col].min() >= 0.0
    assert X[:, col].max() <= 1.0
