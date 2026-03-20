"""Data quality checks for OHLCV kline DataFrames.

Validates gaps, duplicates, NaN values, and detects outlier candles.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Expected gap between consecutive 5-min candles.
_FIVE_MIN = pd.Timedelta(minutes=5)

# Threshold for "small" gap that can be interpolated (up to 30 min = 6 candles).
_MAX_INTERPOLATE_GAP = pd.Timedelta(minutes=30)

# Price-change threshold that is considered an outlier.
_OUTLIER_PCT = 0.10  # 10 %


def validate_klines(df: pd.DataFrame, expected_interval: pd.Timedelta = _FIVE_MIN) -> dict:
    """Run quality checks on a klines DataFrame.

    Parameters
    ----------
    df:
        DataFrame with at least ``open_time, open, high, low, close, volume``.
    expected_interval:
        Expected time delta between consecutive candles.

    Returns
    -------
    dict
        Keys:

        - ``n_rows`` (int)
        - ``n_duplicates`` (int)
        - ``n_nans`` (int)  -- total NaN cells across OHLCV columns
        - ``n_gaps`` (int)  -- number of time-gap violations
        - ``gaps`` (list[dict]) -- each gap with ``index``, ``prev_time``, ``next_time``, ``delta``
        - ``is_valid`` (bool) -- ``True`` when there are no duplicates, NaNs, or gaps
    """
    result: dict = {"n_rows": len(df)}

    # --- duplicates ---
    n_dup = int(df.duplicated(subset="open_time").sum())
    result["n_duplicates"] = n_dup
    if n_dup:
        logger.warning("Found %d duplicate open_time entries", n_dup)

    # --- NaN values in OHLCV ---
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    present_cols = [c for c in ohlcv_cols if c in df.columns]
    n_nans = int(df[present_cols].isna().sum().sum())
    result["n_nans"] = n_nans
    if n_nans:
        logger.warning("Found %d NaN values in OHLCV columns", n_nans)

    # --- time-gap check ---
    gaps: list[dict] = []
    if len(df) > 1:
        times = df["open_time"].sort_values().reset_index(drop=True)
        deltas = times.diff().iloc[1:]
        for idx_offset, delta in deltas.items():
            if delta != expected_interval:
                gaps.append(
                    {
                        "index": int(idx_offset),
                        "prev_time": str(times.iloc[idx_offset - 1]),
                        "next_time": str(times.iloc[idx_offset]),
                        "delta": str(delta),
                    }
                )
    result["n_gaps"] = len(gaps)
    result["gaps"] = gaps
    if gaps:
        logger.warning("Found %d interval gaps", len(gaps))

    result["is_valid"] = n_dup == 0 and n_nans == 0 and len(gaps) == 0
    return result


def detect_outliers(df: pd.DataFrame, pct_threshold: float = _OUTLIER_PCT) -> pd.DataFrame:
    """Flag candles with extreme price moves.

    Parameters
    ----------
    df:
        DataFrame with ``open_time`` and ``close`` columns.
    pct_threshold:
        Absolute percentage change threshold (default 10 %).

    Returns
    -------
    pd.DataFrame
        Subset of *df* where ``|close_pct_change| > pct_threshold``.
    """
    pct_change = df["close"].pct_change().abs()
    mask = pct_change > pct_threshold
    outliers = df.loc[mask].copy()
    outliers["pct_change"] = pct_change.loc[mask]
    if len(outliers):
        logger.warning(
            "Detected %d outlier candles (>%.0f%% move in one interval)",
            len(outliers),
            pct_threshold * 100,
        )
    return outliers


def fill_gaps(
    df: pd.DataFrame,
    expected_interval: pd.Timedelta = _FIVE_MIN,
    max_interpolate: pd.Timedelta = _MAX_INTERPOLATE_GAP,
) -> pd.DataFrame:
    """Interpolate small time gaps and log warnings for large ones.

    Small gaps (up to *max_interpolate*) are filled by forward-filling OHLCV
    values and setting volume to 0.  Large gaps are left as-is and logged.

    Parameters
    ----------
    df:
        Klines DataFrame with ``open_time`` and OHLCV columns.
    expected_interval:
        Expected interval between candles.
    max_interpolate:
        Maximum gap size that will be interpolated.

    Returns
    -------
    pd.DataFrame
        DataFrame with small gaps filled.
    """
    if df.empty or len(df) < 2:
        return df.copy()

    df = df.sort_values("open_time").reset_index(drop=True)

    new_rows: list[dict] = []
    for i in range(1, len(df)):
        prev_time = df.loc[i - 1, "open_time"]
        curr_time = df.loc[i, "open_time"]
        gap = curr_time - prev_time

        if gap <= expected_interval:
            continue

        if gap <= max_interpolate:
            # Fill with forward-filled candles.
            n_missing = int(gap / expected_interval) - 1
            last_close = df.loc[i - 1, "close"]
            for j in range(1, n_missing + 1):
                new_time = prev_time + j * expected_interval
                new_rows.append(
                    {
                        "open_time": new_time,
                        "open": last_close,
                        "high": last_close,
                        "low": last_close,
                        "close": last_close,
                        "volume": 0.0,
                    }
                )
            logger.info(
                "Interpolated %d candles for gap at %s (gap=%s)",
                n_missing,
                prev_time,
                gap,
            )
        else:
            logger.warning(
                "Large gap at %s -> %s (gap=%s) — not interpolated",
                prev_time,
                curr_time,
                gap,
            )

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        # Ensure the new rows have matching dtypes for open_time.
        if hasattr(df["open_time"].dtype, "tz"):
            new_df["open_time"] = pd.to_datetime(new_df["open_time"], utc=True)
        df = pd.concat([df, new_df], ignore_index=True)
        df = df.sort_values("open_time").reset_index(drop=True)

    return df
