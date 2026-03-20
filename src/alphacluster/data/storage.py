"""Parquet storage for OHLCV and funding rate data.

Provides save/load helpers and incremental update logic so that only
new data is downloaded on subsequent runs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from alphacluster.config import DATA_DIR

logger = logging.getLogger(__name__)

# Default file paths -------------------------------------------------------
DEFAULT_KLINES_PATH = DATA_DIR / "btcusdt_5m.parquet"
DEFAULT_FUNDING_PATH = DATA_DIR / "btcusdt_funding.parquet"


def save_to_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to Parquet, creating parent directories as needed.

    Parameters
    ----------
    df:
        DataFrame to persist.
    path:
        Destination file path.

    Returns
    -------
    Path
        Resolved path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)
    logger.info("Saved %d rows to %s", len(df), path)
    return path


def load_from_parquet(path: str | Path) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame.

    Parameters
    ----------
    path:
        Source file path.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No data file at {path}")
    df = pd.read_parquet(path, engine="pyarrow")
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def get_last_timestamp(path: str | Path, time_col: str = "open_time") -> pd.Timestamp | None:
    """Return the latest timestamp in an existing Parquet file.

    If the file does not exist or is empty, returns ``None``.

    Parameters
    ----------
    path:
        Path to a Parquet file.
    time_col:
        Name of the datetime column to inspect.
    """
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_parquet(path, engine="pyarrow", columns=[time_col])
    if df.empty:
        return None
    return pd.Timestamp(df[time_col].max())


def append_to_parquet(new_df: pd.DataFrame, path: str | Path, time_col: str = "open_time") -> int:
    """Append *new_df* to an existing Parquet file, deduplicating on *time_col*.

    If the file does not exist yet it is created.

    Parameters
    ----------
    new_df:
        New rows to append.
    path:
        Parquet file path.
    time_col:
        Datetime column used for deduplication.

    Returns
    -------
    int
        Number of **new** rows actually added.
    """
    path = Path(path)
    if path.exists():
        existing = load_from_parquet(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = (
            combined.drop_duplicates(subset=time_col).sort_values(time_col).reset_index(drop=True)
        )
        n_new = len(combined) - len(existing)
    else:
        combined = (
            new_df.drop_duplicates(subset=time_col).sort_values(time_col).reset_index(drop=True)
        )
        n_new = len(combined)

    save_to_parquet(combined, path)
    logger.info("Appended %d new rows to %s (total: %d)", n_new, path, len(combined))
    return n_new
