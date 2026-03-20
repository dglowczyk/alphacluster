"""Data source abstraction for historical and live market data.

Defines a ``DataSource`` protocol that both ``HistoricalDataSource`` (reading
from Parquet files) and ``LiveDataSource`` (WebSocket, placeholder) implement.
"""

from __future__ import annotations

import abc
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from alphacluster.data.storage import load_from_parquet


class DataSource(abc.ABC):
    """Abstract interface for retrieving OHLCV candle data."""

    @abc.abstractmethod
    def get_candles(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Return candles for the given date range.

        Parameters
        ----------
        start, end:
            Optional bounds.  If ``None``, return all available data.

        Returns
        -------
        pd.DataFrame
            Must contain at least ``open_time, open, high, low, close, volume``.
        """
        ...


class HistoricalDataSource(DataSource):
    """Read candles from an on-disk Parquet file.

    Parameters
    ----------
    path:
        Path to a Parquet file written by :func:`alphacluster.data.storage.save_to_parquet`.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._df: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if self._df is None:
            self._df = load_from_parquet(self._path)
            # Ensure open_time is datetime for filtering.
            if not pd.api.types.is_datetime64_any_dtype(self._df["open_time"]):
                self._df["open_time"] = pd.to_datetime(self._df["open_time"], utc=True)
        return self._df

    def get_candles(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        df = self._load()
        if start is not None:
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            df = df[df["open_time"] >= pd.Timestamp(start)]
        if end is not None:
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            df = df[df["open_time"] <= pd.Timestamp(end)]
        return df.reset_index(drop=True)


class LiveDataSource(DataSource):
    """Placeholder for a WebSocket-based live data feed.

    This will be implemented in a later phase.  For now it raises
    ``NotImplementedError``.
    """

    def __init__(self, symbol: str = "BTCUSDT", interval: str = "5m") -> None:
        self.symbol = symbol
        self.interval = interval

    def get_candles(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError(
            "LiveDataSource is a stub — WebSocket streaming will be added in a future phase."
        )
