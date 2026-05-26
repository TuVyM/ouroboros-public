# orderflow/fetch.py
"""
Download Binance orderflow data via REST API.
Buy/sell volume sourced from 1h klines taker-buy columns (col 9/5).
Funding rate and OI from Binance Futures REST API.

Usage:
    python -m orderflow.fetch                        # BTC, 3 years
    python -m orderflow.fetch --symbol ETHUSDT --days 365
"""
import argparse
import gzip
import io
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

DATA_DIR     = Path(__file__).parent.parent / "data" / "orderflow"
SPOT_API     = "https://api.binance.com/api/v3/klines"
FUTURES_BASE = "https://fapi.binance.com"

# These parse helpers remain for backwards-compat with tests
def _parse_agg_csv(raw_gz_bytes: bytes) -> pd.DataFrame:
    """Parse a Binance aggTrades .csv.gz byte blob into a DataFrame."""
    with gzip.open(io.BytesIO(raw_gz_bytes)) as f:
        df = pd.read_csv(
            f, header=None,
            names=["agg_id", "price", "qty", "first_id", "last_id",
                   "timestamp_ms", "is_buyer_maker", "best_match"],
        )
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)
    return df[["timestamp_ms", "price", "qty", "is_buyer_maker"]].astype(
        {"timestamp_ms": "int64", "price": "float64", "qty": "float64"}
    )


def _parse_funding_response(data: list) -> pd.DataFrame:
    df = pd.DataFrame(data)
    df["timestamp"]    = pd.to_datetime(df["fundingTime"], unit="ms", utc=True).dt.floor("1h")
    df["funding_rate"] = df["fundingRate"].astype(float)
    return df[["timestamp", "funding_rate"]]


def _parse_oi_response(data: list) -> pd.DataFrame:
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.floor("1h")
    df["oi"]        = df["sumOpenInterest"].astype(float)
    return df[["timestamp", "oi"]]


def _fetch_klines_1h(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Fetch 1h klines from Binance REST API and extract taker buy/sell volumes.

    Kline col 5  = total volume (base asset)
    Kline col 9  = taker buy base asset volume → buy_vol
    sell_vol = total_vol - buy_vol
    large_trade_vol set to 0 (not available from klines)
    """
    bars = []
    params = {
        "symbol": symbol, "interval": "1h",
        "startTime": start_ms, "endTime": end_ms, "limit": 1000,
    }
    while True:
        r = requests.get(SPOT_API, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        bars.extend(data)
        params["startTime"] = data[-1][0] + 1
        if len(data) < 1000:
            break
        time.sleep(0.05)

    if not bars:
        return pd.DataFrame(columns=["timestamp", "buy_vol", "sell_vol",
                                     "total_vol", "vwap", "large_trade_vol", "buy_ratio"])

    df = pd.DataFrame(bars)
    df["timestamp"]       = pd.to_datetime(df[0], unit="ms", utc=True).dt.floor("1h")
    df["total_vol"]       = df[5].astype(float)
    df["buy_vol"]         = df[9].astype(float)
    df["sell_vol"]        = df["total_vol"] - df["buy_vol"]
    df["vwap"]            = df[4].astype(float)   # close ≈ vwap for 1h bars
    df["large_trade_vol"] = 0.0                   # not available from klines
    df["buy_ratio"]       = df["buy_vol"] / (df["total_vol"] + 1e-12)
    return df[["timestamp", "buy_vol", "sell_vol", "total_vol",
               "vwap", "large_trade_vol", "buy_ratio"]]


def _fetch_funding(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows, url = [], f"{FUTURES_BASE}/fapi/v1/fundingRate"
    params = {"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": 1000}
    while True:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        rows.extend(data)
        params["startTime"] = data[-1]["fundingTime"] + 1
        if len(data) < 1000:
            break
        time.sleep(0.1)
    return _parse_funding_response(rows) if rows else pd.DataFrame(columns=["timestamp", "funding_rate"])


def _fetch_oi(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Fetch OI history in 30-day chunks.
    The /futures/data/openInterestHist endpoint only supports ~30 days per request;
    skip chunks that return 400 (data unavailable for that far back).
    """
    rows = []
    url = f"{FUTURES_BASE}/futures/data/openInterestHist"
    chunk_ms = 30 * 24 * 3600 * 1000  # 30 days in ms

    chunk_start = start_ms
    while chunk_start < end_ms:
        chunk_end = min(chunk_start + chunk_ms, end_ms)
        params = {
            "symbol": symbol, "period": "1h",
            "startTime": chunk_start, "endTime": chunk_end, "limit": 500,
        }
        while True:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 400:
                break  # OI data unavailable for this time range — skip chunk
            r.raise_for_status()
            data = r.json()
            if not data:
                break
            rows.extend(data)
            params["startTime"] = data[-1]["timestamp"] + 1
            if len(data) < 500:
                break
            time.sleep(0.1)
        chunk_start = chunk_end + 1

    return _parse_oi_response(rows) if rows else pd.DataFrame(columns=["timestamp", "oi"])


def fetch_all(symbol: str = "BTCUSDT", days: int = 1095) -> None:
    """Download all three orderflow sources via Binance REST API."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    end_dt   = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=days)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp() * 1000)

    agg_path  = DATA_DIR / f"{symbol}_aggtrades_1h.parquet"
    fund_path = DATA_DIR / f"{symbol}_funding.parquet"
    oi_path   = DATA_DIR / f"{symbol}_oi.parquet"

    if not agg_path.exists():
        print(f"Fetching {days} days of 1h klines (taker buy volume) for {symbol}…")
        df = _fetch_klines_1h(symbol, start_ms, end_ms)
        df.to_parquet(agg_path, index=False)
        print(f"  Saved {len(df)} 1h bars → {agg_path}")

    if not fund_path.exists():
        print(f"Fetching funding rate for {symbol}…")
        df = _fetch_funding(symbol, start_ms, end_ms)
        df.to_parquet(fund_path, index=False)
        print(f"  Saved {len(df)} rows → {fund_path}")

    if not oi_path.exists():
        print(f"Fetching open interest for {symbol} (30-day chunks)…")
        df = _fetch_oi(symbol, start_ms, end_ms)
        df.to_parquet(oi_path, index=False)
        print(f"  Saved {len(df)} rows → {oi_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--days",   type=int, default=1095)
    args = p.parse_args()
    fetch_all(args.symbol, args.days)
