# orderflow/aggregate.py
import pandas as pd

LARGE_TRADE_USD = 100_000.0


def aggregate_to_1h(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw aggTrades to 1h bars.

    Input columns: timestamp_ms (int64 ms), price (float64), qty (float64),
                   is_buyer_maker (bool)
    Output columns: timestamp (UTC, floored to hour), buy_vol, sell_vol,
                    total_vol, vwap, large_trade_vol, buy_ratio
    """
    df = trades_df.copy()
    df["timestamp"]    = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.floor("1h")
    df["trade_usd"]    = df["price"] * df["qty"]
    df["is_taker_buy"] = ~df["is_buyer_maker"]

    # Pre-compute per-row contribution columns so we can use named agg (pandas 3 compatible)
    df["buy_vol_raw"]   = df["qty"] * df["is_taker_buy"].astype(float)
    df["sell_vol_raw"]  = df["qty"] * (~df["is_taker_buy"]).astype(float)
    df["large_vol_raw"] = df["qty"] * (df["trade_usd"] >= LARGE_TRADE_USD).astype(float)
    df["pv"]            = df["price"] * df["qty"]   # price × volume for VWAP numerator

    result = df.groupby("timestamp", sort=True).agg(
        buy_vol=("buy_vol_raw",   "sum"),
        sell_vol=("sell_vol_raw",  "sum"),
        total_vol=("qty",          "sum"),
        large_trade_vol=("large_vol_raw", "sum"),
        pv_sum=("pv",             "sum"),
    ).reset_index()

    result["vwap"]      = result["pv_sum"] / (result["total_vol"] + 1e-12)
    result["buy_ratio"] = result["buy_vol"] / (result["total_vol"] + 1e-12)
    result.drop(columns=["pv_sum"], inplace=True)
    return result
