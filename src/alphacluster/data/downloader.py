"""Download historical data from the Binance Futures REST API.

Provides functions to fetch OHLCV kline data and funding rate history
with automatic pagination and rate limiting.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from alphacluster.config import BINANCE_BASE_URL, DEFAULT_INTERVAL, DEFAULT_SYMBOL

logger = logging.getLogger(__name__)

# Binance returns at most 1500 candles per request.
MAX_CANDLES_PER_REQUEST = 1500

# Binance allows 1200 requests per minute; we stay well under that.
_MIN_REQUEST_INTERVAL_S = 0.06  # ~16 req/s -> 960 req/min

# Column names returned by the /fapi/v1/klines endpoint.
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]

_INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def _ts_ms(dt: datetime) -> int:
    """Convert a datetime to milliseconds since epoch."""
    return int(dt.timestamp() * 1000)


def download_klines(
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    start_date: datetime | str | None = None,
    end_date: datetime | str | None = None,
    *,
    session: requests.Session | None = None,
    progress_callback: callable | None = None,
) -> pd.DataFrame:
    """Fetch 5-min OHLCV candles from the Binance Futures REST API.

    Handles pagination automatically (Binance returns max 1500 candles per
    request) and applies simple rate limiting to stay within the API limits
    (1200 requests/min).

    Parameters
    ----------
    symbol:
        Trading pair, e.g. ``"BTCUSDT"``.
    interval:
        Kline interval, e.g. ``"5m"``.
    start_date, end_date:
        Date range.  Accepts ``datetime`` objects or ISO-format strings.
        Both are inclusive.  ``end_date`` defaults to now.
    session:
        Optional ``requests.Session`` for connection reuse.
    progress_callback:
        Optional callable(n_candles_fetched) invoked after each page.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``open_time, open, high, low, close, volume``
        (and additional Binance fields).  ``open_time`` is a UTC-aware
        datetime index.
    """
    if isinstance(start_date, str):
        start_date = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    if isinstance(end_date, str):
        end_date = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)

    if start_date is None:
        start_date = datetime(2019, 9, 1, tzinfo=timezone.utc)
    if end_date is None:
        end_date = datetime.now(tz=timezone.utc)

    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    sess = session or requests.Session()
    url = f"{BINANCE_BASE_URL}/fapi/v1/klines"
    interval_ms = _INTERVAL_MS.get(interval, 300_000)

    all_rows: list[list] = []
    current_start_ms = _ts_ms(start_date)
    end_ms = _ts_ms(end_date)
    last_request_time = 0.0

    while current_start_ms < end_ms:
        # Rate limiting
        elapsed = time.monotonic() - last_request_time
        if elapsed < _MIN_REQUEST_INTERVAL_S:
            time.sleep(_MIN_REQUEST_INTERVAL_S - elapsed)

        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start_ms,
            "endTime": end_ms,
            "limit": MAX_CANDLES_PER_REQUEST,
        }

        last_request_time = time.monotonic()
        resp = sess.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        all_rows.extend(data)

        if progress_callback is not None:
            progress_callback(len(data))

        # Move start to one interval after the last received candle.
        last_open_time = int(data[-1][0])
        current_start_ms = last_open_time + interval_ms

        if len(data) < MAX_CANDLES_PER_REQUEST:
            break

    if not all_rows:
        return pd.DataFrame(columns=KLINE_COLUMNS)

    df = pd.DataFrame(all_rows, columns=KLINE_COLUMNS)

    # Convert numeric columns.
    float_cols = [
        "open", "high", "low", "close", "volume",
        "quote_volume", "taker_buy_base_volume", "taker_buy_quote_volume",
    ]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("Int64")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    df = df.drop(columns=["ignore"], errors="ignore")
    df = df.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)

    logger.info("Downloaded %d klines for %s (%s)", len(df), symbol, interval)
    return df


# Maximum funding rate records per page.
_MAX_FUNDING_PER_REQUEST = 1000


def download_funding_rates(
    symbol: str = DEFAULT_SYMBOL,
    start_date: datetime | str | None = None,
    end_date: datetime | str | None = None,
    *,
    session: requests.Session | None = None,
    progress_callback: callable | None = None,
) -> pd.DataFrame:
    """Fetch funding rate history from the Binance Futures API.

    Parameters
    ----------
    symbol:
        Trading pair, e.g. ``"BTCUSDT"``.
    start_date, end_date:
        Date range (inclusive).  ``end_date`` defaults to now.
    session:
        Optional ``requests.Session``.
    progress_callback:
        Optional callable(n_records_fetched).

    Returns
    -------
    pd.DataFrame
        Columns: ``funding_time, symbol, funding_rate, mark_price``.
    """
    if isinstance(start_date, str):
        start_date = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    if isinstance(end_date, str):
        end_date = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)

    if start_date is None:
        start_date = datetime(2019, 9, 1, tzinfo=timezone.utc)
    if end_date is None:
        end_date = datetime.now(tz=timezone.utc)

    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    sess = session or requests.Session()
    url = f"{BINANCE_BASE_URL}/fapi/v1/fundingRate"

    all_rows: list[dict] = []
    current_start_ms = _ts_ms(start_date)
    end_ms = _ts_ms(end_date)
    last_request_time = 0.0

    while current_start_ms < end_ms:
        elapsed = time.monotonic() - last_request_time
        if elapsed < _MIN_REQUEST_INTERVAL_S:
            time.sleep(_MIN_REQUEST_INTERVAL_S - elapsed)

        params = {
            "symbol": symbol,
            "startTime": current_start_ms,
            "endTime": end_ms,
            "limit": _MAX_FUNDING_PER_REQUEST,
        }

        last_request_time = time.monotonic()
        resp = sess.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        all_rows.extend(data)

        if progress_callback is not None:
            progress_callback(len(data))

        last_time = int(data[-1]["fundingTime"])
        current_start_ms = last_time + 1

        if len(data) < _MAX_FUNDING_PER_REQUEST:
            break

    if not all_rows:
        return pd.DataFrame(columns=["funding_time", "symbol", "funding_rate", "mark_price"])

    df = pd.DataFrame(all_rows)
    df = df.rename(columns={"fundingTime": "funding_time", "fundingRate": "funding_rate",
                             "markPrice": "mark_price"})
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    df["mark_price"] = pd.to_numeric(df["mark_price"], errors="coerce")
    df["funding_time"] = pd.to_datetime(df["funding_time"], unit="ms", utc=True)

    df = df.drop_duplicates(subset="funding_time").sort_values("funding_time").reset_index(
        drop=True
    )

    logger.info("Downloaded %d funding rate records for %s", len(df), symbol)
    return df
