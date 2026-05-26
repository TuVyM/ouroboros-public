import gzip
import io
import json
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from orderflow.fetch import _parse_agg_csv, _parse_funding_response, _parse_oi_response


def _make_csv_gz(rows):
    """Build a gzipped CSV string matching Binance aggTrades format."""
    lines = "\n".join(
        f"{i},{r['price']},{r['qty']},1,1,{r['ts']},{str(r['buyer_maker']).lower()},true"
        for i, r in enumerate(rows)
    )
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as f:
        f.write(lines.encode())
    return buf.getvalue()


def test_parse_agg_csv_returns_dataframe():
    raw = _make_csv_gz([
        {"price": 100.0, "qty": 1.0, "ts": 1_700_000_000_000, "buyer_maker": False},
        {"price": 101.0, "qty": 2.0, "ts": 1_700_001_000_000, "buyer_maker": True},
    ])
    df = _parse_agg_csv(raw)
    assert list(df.columns) == ["timestamp_ms", "price", "qty", "is_buyer_maker"]
    assert len(df) == 2
    assert df["is_buyer_maker"].dtype == bool


def test_parse_funding_response():
    data = [
        {"fundingTime": 1_700_000_000_000, "fundingRate": "0.0001"},
        {"fundingTime": 1_700_028_800_000, "fundingRate": "-0.0002"},
    ]
    df = _parse_funding_response(data)
    assert "timestamp" in df.columns
    assert "funding_rate" in df.columns
    assert df["funding_rate"].iloc[0] == pytest.approx(0.0001)
    assert df["funding_rate"].iloc[1] == pytest.approx(-0.0002)


def test_parse_oi_response():
    data = [
        {"timestamp": 1_700_000_000_000, "sumOpenInterest": "12345.678"},
        {"timestamp": 1_700_003_600_000, "sumOpenInterest": "12400.0"},
    ]
    df = _parse_oi_response(data)
    assert "timestamp" in df.columns
    assert "oi" in df.columns
    assert df["oi"].iloc[0] == pytest.approx(12345.678)
