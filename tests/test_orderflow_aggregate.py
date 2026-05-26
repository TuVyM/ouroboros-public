import pandas as pd
import pytest

from orderflow.aggregate import aggregate_to_1h, LARGE_TRADE_USD


def _trades(ts_offsets_s, prices, qtys, is_buyer_maker, base_ms=1_700_000_000_000):
    return pd.DataFrame({
        "timestamp_ms":   [base_ms + t * 1000 for t in ts_offsets_s],
        "price":          prices,
        "qty":            qtys,
        "is_buyer_maker": is_buyer_maker,
    })


def test_buy_sell_split():
    # 3 BUY trades (is_buyer_maker=False), 2 SELL trades — all in same hour
    df = _trades([0, 1, 2, 3, 4], [100.0]*5, [1.0, 2.0, 3.0, 4.0, 5.0],
                 [False, False, False, True, True])
    result = aggregate_to_1h(df)
    assert len(result) == 1
    assert result["buy_vol"].iloc[0]  == pytest.approx(6.0)   # 1+2+3
    assert result["sell_vol"].iloc[0] == pytest.approx(9.0)   # 4+5
    assert result["total_vol"].iloc[0] == pytest.approx(15.0)


def test_vwap_calculation():
    # Two trades at different prices, equal qty — vwap = mean(prices)
    df = _trades([0, 1], [100.0, 200.0], [1.0, 1.0], [False, True])
    result = aggregate_to_1h(df)
    assert result["vwap"].iloc[0] == pytest.approx(150.0)


def test_large_trade_detection():
    # One large trade (price*qty = 200_000 > LARGE_TRADE_USD), one small
    df = _trades([0, 1], [100_000.0, 100.0], [2.0, 1.0], [False, False])
    result = aggregate_to_1h(df)
    assert result["large_trade_vol"].iloc[0] == pytest.approx(2.0)  # only first trade


def test_two_hours_produce_two_rows():
    # Trades 1h apart land in different buckets
    df = _trades([0, 3601], [100.0, 100.0], [1.0, 1.0], [False, False])
    result = aggregate_to_1h(df)
    assert len(result) == 2


def test_buy_ratio_in_zero_to_one():
    df = _trades([0, 1, 2], [100.0]*3, [1.0]*3, [False, True, False])
    result = aggregate_to_1h(df)
    ratio = result["buy_ratio"].iloc[0]
    assert 0.0 <= ratio <= 1.0
