"""Tests for the data pipeline (alphacluster.data).

Covers the downloader (with mocked API responses), storage round-trips,
validator gap detection / outlier detection, and the live_feed abstractions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from alphacluster.data.downloader import (
    MAX_CANDLES_PER_REQUEST,
    download_funding_rates,
    download_klines,
    download_ls_ratio,
    download_open_interest,
)
from alphacluster.data.live_feed import HistoricalDataSource, LiveDataSource
from alphacluster.data.storage import (
    append_to_parquet,
    get_last_timestamp,
    load_from_parquet,
    save_to_parquet,
)
from alphacluster.data.validator import detect_outliers, fill_gaps, validate_klines

# ---------------------------------------------------------------------------
# Helpers to build fake Binance API responses
# ---------------------------------------------------------------------------

_FIVE_MIN_MS = 300_000


def _make_kline_row(open_time_ms: int, close_price: float = 10000.0) -> list:
    """Build a single kline row matching the Binance /fapi/v1/klines format."""
    return [
        open_time_ms,  # open_time
        str(close_price),  # open
        str(close_price + 10),  # high
        str(close_price - 10),  # low
        str(close_price),  # close
        "100.0",  # volume
        open_time_ms + _FIVE_MIN_MS - 1,  # close_time
        "1000000.0",  # quote_volume
        150,  # trades
        "50.0",  # taker_buy_base_volume
        "500000.0",  # taker_buy_quote_volume
        "0",  # ignore
    ]


def _make_kline_batch(start_ms: int, n: int, close_price: float = 10000.0) -> list[list]:
    return [_make_kline_row(start_ms + i * _FIVE_MIN_MS, close_price) for i in range(n)]


def _make_oi_row(timestamp_s: int) -> dict:
    return {"t": timestamp_s, "o": 49000.0, "h": 51000.0, "l": 48000.0, "c": 50000.0}


def _make_funding_row(funding_time_ms: int) -> dict:
    return {
        "symbol": "BTCUSDT",
        "fundingTime": funding_time_ms,
        "fundingRate": "0.0001",
        "markPrice": "10000.0",
    }


# ---------------------------------------------------------------------------
# Downloader tests
# ---------------------------------------------------------------------------


class TestDownloadKlines:
    """Tests for download_klines with mocked HTTP responses."""

    def test_fetch_klines_returns_dataframe(self):
        """download_klines should return a DataFrame with the expected columns."""
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = datetime(2023, 1, 1, 1, 0, tzinfo=timezone.utc)  # 1 hour = 12 candles
        batch = _make_kline_batch(int(start.timestamp() * 1000), 12)

        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = batch
        resp.raise_for_status = MagicMock()
        session.get.return_value = resp

        df = download_klines(
            symbol="BTCUSDT",
            interval="5m",
            start_date=start,
            end_date=end,
            session=session,
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 12
        for col in ["open_time", "open", "high", "low", "close", "volume"]:
            assert col in df.columns

    def test_klines_columns_types(self):
        """OHLCV columns should be float and open_time should be datetime."""
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = datetime(2023, 1, 1, 0, 30, tzinfo=timezone.utc)
        batch = _make_kline_batch(int(start.timestamp() * 1000), 6)

        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = batch
        resp.raise_for_status = MagicMock()
        session.get.return_value = resp

        df = download_klines(start_date=start, end_date=end, session=session)

        assert pd.api.types.is_datetime64_any_dtype(df["open_time"])
        for col in ["open", "high", "low", "close", "volume"]:
            assert pd.api.types.is_float_dtype(df[col])

    def test_pagination(self):
        """download_klines should paginate when data exceeds MAX_CANDLES_PER_REQUEST."""
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        start_ms = int(start.timestamp() * 1000)

        page1 = _make_kline_batch(start_ms, MAX_CANDLES_PER_REQUEST)
        page2_start = start_ms + MAX_CANDLES_PER_REQUEST * _FIVE_MIN_MS
        page2 = _make_kline_batch(page2_start, 10)

        # End date far enough to require two pages.
        end_ms = page2_start + 10 * _FIVE_MIN_MS
        end = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)

        session = MagicMock()
        resp1 = MagicMock()
        resp1.json.return_value = page1
        resp1.raise_for_status = MagicMock()

        resp2 = MagicMock()
        resp2.json.return_value = page2
        resp2.raise_for_status = MagicMock()

        session.get.side_effect = [resp1, resp2]

        df = download_klines(start_date=start, end_date=end, session=session)

        assert len(df) == MAX_CANDLES_PER_REQUEST + 10
        assert session.get.call_count == 2

    def test_empty_response(self):
        """download_klines returns empty DataFrame when API returns no data."""
        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = []
        resp.raise_for_status = MagicMock()
        session.get.return_value = resp

        df = download_klines(
            start_date=datetime(2099, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2099, 1, 2, tzinfo=timezone.utc),
            session=session,
        )
        assert df.empty

    def test_deduplication(self):
        """Overlapping candles across multiple download calls are deduplicated."""
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        start_ms = int(start.timestamp() * 1000)

        # Simulate two separate downloads with overlapping data.  The first
        # download returns candles 0..4, the second 4..8 (overlap at index 4).
        batch_a = _make_kline_batch(start_ms, 5)
        batch_b = _make_kline_batch(start_ms + 4 * _FIVE_MIN_MS, 5)
        combined = batch_a + batch_b  # 10 rows, 1 duplicate open_time

        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = combined
        resp.raise_for_status = MagicMock()
        session.get.return_value = resp

        end = datetime.fromtimestamp((start_ms + 9 * _FIVE_MIN_MS) / 1000, tz=timezone.utc)
        df = download_klines(start_date=start, end_date=end, session=session)
        # 10 rows with 1 duplicate -> 9 unique.
        assert len(df) == 9

    def test_progress_callback(self):
        """The progress_callback is invoked with the batch size."""
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        batch = _make_kline_batch(int(start.timestamp() * 1000), 5)

        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = batch
        resp.raise_for_status = MagicMock()
        session.get.return_value = resp

        callback = MagicMock()
        download_klines(
            start_date=start,
            end_date=datetime(2023, 1, 1, 0, 30, tzinfo=timezone.utc),
            session=session,
            progress_callback=callback,
        )
        callback.assert_called_once_with(5)


class TestDownloadFundingRates:
    """Tests for download_funding_rates with mocked HTTP responses."""

    def test_fetch_funding_rates(self):
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        start_ms = int(start.timestamp() * 1000)
        eight_hours_ms = 8 * 3600 * 1000
        rows = [_make_funding_row(start_ms + i * eight_hours_ms) for i in range(3)]

        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = rows
        resp.raise_for_status = MagicMock()
        session.get.return_value = resp

        df = download_funding_rates(
            start_date=start,
            end_date=datetime(2023, 1, 2, tzinfo=timezone.utc),
            session=session,
        )
        assert len(df) == 3
        assert "funding_time" in df.columns
        assert "funding_rate" in df.columns
        assert "mark_price" in df.columns

    def test_empty_funding(self):
        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = []
        resp.raise_for_status = MagicMock()
        session.get.return_value = resp

        df = download_funding_rates(
            start_date=datetime(2099, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2099, 1, 2, tzinfo=timezone.utc),
            session=session,
        )
        assert df.empty


class TestDownloadOpenInterest:
    """Tests for download_open_interest with mocked Coinalyze responses."""

    def test_fetch_open_interest(self):
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        start_s = int(start.timestamp())
        day_s = 86400
        rows = [_make_oi_row(start_s + i * day_s) for i in range(5)]

        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = [{"symbol": "BTCUSDT_PERP.A", "history": rows}]
        resp.raise_for_status = MagicMock()
        session.get.return_value = resp

        df = download_open_interest(
            symbol="BTCUSDT",
            start_date=start,
            end_date=datetime(2023, 1, 6, tzinfo=timezone.utc),
            api_key="test-key",
            session=session,
        )
        assert len(df) == 5
        assert "timestamp" in df.columns
        assert "sum_open_interest" in df.columns
        assert "sum_open_interest_value" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_empty_open_interest(self):
        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = [{"symbol": "BTCUSDT_PERP.A", "history": []}]
        resp.raise_for_status = MagicMock()
        session.get.return_value = resp

        df = download_open_interest(
            symbol="BTCUSDT",
            start_date=datetime(2099, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2099, 1, 2, tzinfo=timezone.utc),
            api_key="test-key",
            session=session,
        )
        assert df.empty

    def test_no_api_key_returns_empty(self):
        df = download_open_interest(
            symbol="BTCUSDT",
            start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2023, 1, 2, tzinfo=timezone.utc),
            api_key="",
        )
        assert df.empty


def _make_ls_ratio_row(timestamp_s: int) -> dict:
    return {"t": timestamp_s, "r": 1.5, "l": 60.0, "s": 40.0}


class TestDownloadLsRatio:
    """Tests for download_ls_ratio with mocked Coinalyze responses."""

    def test_fetch_ls_ratio(self):
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        start_s = int(start.timestamp())
        day_s = 86400
        rows = [_make_ls_ratio_row(start_s + i * day_s) for i in range(5)]

        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = [{"symbol": "BTCUSDT_PERP.A", "history": rows}]
        resp.raise_for_status = MagicMock()
        session.get.return_value = resp

        df = download_ls_ratio(
            symbol="BTCUSDT",
            start_date=start,
            end_date=datetime(2023, 1, 6, tzinfo=timezone.utc),
            api_key="test-key",
            session=session,
        )
        assert len(df) == 5
        assert "timestamp" in df.columns
        assert "long_short_ratio" in df.columns
        assert "long_account" in df.columns
        assert "short_account" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_empty_ls_ratio(self):
        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = [{"symbol": "BTCUSDT_PERP.A", "history": []}]
        resp.raise_for_status = MagicMock()
        session.get.return_value = resp

        df = download_ls_ratio(
            symbol="BTCUSDT",
            start_date=datetime(2099, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2099, 1, 2, tzinfo=timezone.utc),
            api_key="test-key",
            session=session,
        )
        assert df.empty

    def test_no_api_key_returns_empty(self):
        df = download_ls_ratio(
            symbol="BTCUSDT",
            start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2023, 1, 2, tzinfo=timezone.utc),
            api_key="",
        )
        assert df.empty


# ---------------------------------------------------------------------------
# Storage tests
# ---------------------------------------------------------------------------


class TestStorage:
    """Test parquet save / load round-trips."""

    def test_parquet_round_trip(self, tmp_path: Path):
        """Data survives a save->load cycle without loss."""
        df = pd.DataFrame(
            {
                "open_time": pd.date_range("2023-01-01", periods=10, freq="5min", tz="UTC"),
                "open": range(10),
                "high": range(10, 20),
                "low": range(10),
                "close": range(10),
                "volume": [100.0] * 10,
            }
        )
        path = tmp_path / "test.parquet"
        save_to_parquet(df, path)

        loaded = load_from_parquet(path)
        assert len(loaded) == 10
        assert list(loaded.columns) == list(df.columns)
        pd.testing.assert_frame_equal(df, loaded)

    def test_load_nonexistent_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_from_parquet(tmp_path / "does_not_exist.parquet")

    def test_get_last_timestamp(self, tmp_path: Path):
        df = pd.DataFrame(
            {
                "open_time": pd.date_range("2023-01-01", periods=5, freq="5min", tz="UTC"),
                "close": range(5),
            }
        )
        path = tmp_path / "ts.parquet"
        save_to_parquet(df, path)

        ts = get_last_timestamp(path, "open_time")
        assert ts == pd.Timestamp("2023-01-01 00:20:00", tz="UTC")

    def test_get_last_timestamp_missing_file(self, tmp_path: Path):
        assert get_last_timestamp(tmp_path / "nope.parquet") is None

    def test_append_to_parquet(self, tmp_path: Path):
        path = tmp_path / "append.parquet"
        df1 = pd.DataFrame(
            {
                "open_time": pd.date_range("2023-01-01", periods=5, freq="5min", tz="UTC"),
                "close": range(5),
            }
        )
        save_to_parquet(df1, path)

        df2 = pd.DataFrame(
            {
                "open_time": pd.date_range("2023-01-01 00:25", periods=3, freq="5min", tz="UTC"),
                "close": range(5, 8),
            }
        )
        n_new = append_to_parquet(df2, path, time_col="open_time")
        assert n_new == 3

        final = load_from_parquet(path)
        assert len(final) == 8

    def test_append_deduplicates(self, tmp_path: Path):
        path = tmp_path / "dedup.parquet"
        df = pd.DataFrame(
            {
                "open_time": pd.date_range("2023-01-01", periods=5, freq="5min", tz="UTC"),
                "close": range(5),
            }
        )
        save_to_parquet(df, path)

        # Append overlapping data.
        n_new = append_to_parquet(df, path, time_col="open_time")
        assert n_new == 0
        assert len(load_from_parquet(path)) == 5


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------


class TestValidator:
    """Tests for kline validation, outlier detection, and gap filling."""

    def _make_clean_df(self, n: int = 20) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open_time": pd.date_range("2023-01-01", periods=n, freq="5min", tz="UTC"),
                "open": [10000.0] * n,
                "high": [10010.0] * n,
                "low": [9990.0] * n,
                "close": [10000.0] * n,
                "volume": [100.0] * n,
            }
        )

    def test_valid_data_passes(self):
        df = self._make_clean_df()
        report = validate_klines(df)
        assert report["is_valid"] is True
        assert report["n_duplicates"] == 0
        assert report["n_nans"] == 0
        assert report["n_gaps"] == 0

    def test_detect_duplicates(self):
        df = self._make_clean_df()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        report = validate_klines(df)
        assert report["n_duplicates"] == 1
        assert report["is_valid"] is False

    def test_detect_nans(self):
        df = self._make_clean_df()
        df.loc[5, "close"] = float("nan")
        report = validate_klines(df)
        assert report["n_nans"] >= 1
        assert report["is_valid"] is False

    def test_detect_gaps(self):
        df = self._make_clean_df(20)
        # Remove row 10 to create a gap.
        df = df.drop(10).reset_index(drop=True)
        report = validate_klines(df)
        assert report["n_gaps"] == 1
        assert report["is_valid"] is False

    def test_detect_outliers_flags_extreme_move(self):
        df = self._make_clean_df(10)
        # Make candle 5 jump +15%.
        df.loc[5, "close"] = 11500.0
        outliers = detect_outliers(df, pct_threshold=0.10)
        assert len(outliers) >= 1
        assert 5 in outliers.index

    def test_detect_outliers_clean_data(self):
        df = self._make_clean_df(10)
        outliers = detect_outliers(df, pct_threshold=0.10)
        assert len(outliers) == 0

    def test_fill_gaps_interpolates_small_gap(self):
        df = self._make_clean_df(20)
        # Remove rows 10 and 11 -> 10-min gap (2 missing candles).
        df = df.drop([10, 11]).reset_index(drop=True)
        assert len(df) == 18

        filled = fill_gaps(df)
        assert len(filled) == 20
        report = validate_klines(filled)
        assert report["n_gaps"] == 0

    def test_fill_gaps_skips_large_gap(self):
        df = self._make_clean_df(20)
        # Remove rows 5..14 -> 50-min gap (10 missing candles) > 30 min threshold.
        df = df.drop(list(range(5, 15))).reset_index(drop=True)
        assert len(df) == 10

        filled = fill_gaps(df)
        # Large gap is NOT interpolated.
        report = validate_klines(filled)
        assert report["n_gaps"] >= 1

    def test_fill_gaps_empty_df(self):
        df = pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])
        result = fill_gaps(df)
        assert result.empty


# ---------------------------------------------------------------------------
# Live-feed abstraction tests
# ---------------------------------------------------------------------------


class TestLiveFeed:
    """Test DataSource implementations."""

    def test_historical_data_source(self, tmp_path: Path):
        df = pd.DataFrame(
            {
                "open_time": pd.date_range("2023-01-01", periods=50, freq="5min", tz="UTC"),
                "open": range(50),
                "high": range(50),
                "low": range(50),
                "close": range(50),
                "volume": [1.0] * 50,
            }
        )
        path = tmp_path / "hist.parquet"
        save_to_parquet(df, path)

        source = HistoricalDataSource(path)
        all_candles = source.get_candles()
        assert len(all_candles) == 50

        # Filter by start/end.
        start = datetime(2023, 1, 1, 1, 0, tzinfo=timezone.utc)
        end = datetime(2023, 1, 1, 2, 0, tzinfo=timezone.utc)
        filtered = source.get_candles(start=start, end=end)
        assert len(filtered) > 0
        assert filtered["open_time"].min() >= pd.Timestamp(start)
        assert filtered["open_time"].max() <= pd.Timestamp(end)

    def test_live_data_source_raises(self):
        source = LiveDataSource()
        with pytest.raises(NotImplementedError):
            source.get_candles()

    def test_historical_source_missing_file(self, tmp_path: Path):
        source = HistoricalDataSource(tmp_path / "missing.parquet")
        with pytest.raises(FileNotFoundError):
            source.get_candles()
