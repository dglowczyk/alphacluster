"""Technical indicators computed from OHLCV data.

All indicators are normalized to roughly [-1, 1] or [0, 1] range so they
can be fed directly into the neural network alongside normalized OHLCV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_indicators(
    df: pd.DataFrame,
    funding_df: pd.DataFrame | None = None,
    oi_df: pd.DataFrame | None = None,
    ls_ratio_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute technical indicators and append them as columns to *df*.

    Adds 14 indicator columns to the DataFrame.  NaN values from warmup
    periods are forward/back-filled so every row has valid data.

    Parameters
    ----------
    df:
        DataFrame with at least ``open, high, low, close, volume`` columns.
        Must also have an ``open_time`` column (datetime64) for sentiment merging.
    funding_df:
        Optional DataFrame with ``funding_time`` and ``funding_rate`` columns.
        When None, the ``funding_rate`` column is filled with zeros.
    oi_df:
        Optional DataFrame with ``timestamp`` and ``sum_open_interest`` columns.
        When None, the ``oi_change`` column is filled with zeros.
    ls_ratio_df:
        Optional DataFrame with ``timestamp`` and ``long_short_ratio`` columns.
        When None, the ``ls_ratio`` column is filled with zeros.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with 14 additional indicator columns.
    """
    df = df.copy()
    if "open_time" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["open_time"]):
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # ── Returns ───────────────────────────────────────────────────────
    df["return_1"] = close.pct_change(1)
    df["return_20"] = close.pct_change(20)

    # ── RSI(14) ───────────────────────────────────────────────────────
    df["rsi_14"] = _rsi(close, 14) / 50.0 - 1.0  # scale to [-1, 1]

    # ── MACD ──────────────────────────────────────────────────────────
    _macd_line, _signal_line, histogram = _macd(close, 12, 26, 9)
    norm = close * 0.01
    norm = norm.replace(0, 1)  # safety
    df["macd_hist"] = histogram / norm

    # ── Bollinger Bands ───────────────────────────────────────────────
    bb_pctb, _bb_width = _bollinger(close, 20, 2)
    df["bb_pctb"] = bb_pctb

    # ── ATR(14) ───────────────────────────────────────────────────────
    df["atr_14"] = _atr(high, low, close, 14) / close

    # ── Volume ratio ──────────────────────────────────────────────────
    vol_mean = volume.rolling(20).mean()
    vol_mean = vol_mean.replace(0, 1)
    df["volume_ratio_20"] = volume / vol_mean - 1.0

    # ── OBV slope ─────────────────────────────────────────────────────
    df["obv_slope"] = _obv_slope(close, volume, 20)

    # ── VWAP distance ─────────────────────────────────────────────────
    df["vwap_dist"] = _vwap_distance(close, high, low, volume)

    # ── EMA trend ────────────────────────────────────────────────────
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema55 = close.ewm(span=55, adjust=False).mean()
    close_safe = close.replace(0, 1)
    df["ema_trend"] = (ema21 - ema55) / close_safe

    # ── CVD slope (Cumulative Volume Delta approximation) ────────
    opn = df["open"]
    high_low_range = (high - low).replace(0, 1)
    cvd_delta = ((close - opn) / high_low_range) * volume
    cvd = cvd_delta.cumsum()
    cvd_slope_raw = cvd.diff(20)
    vol_mean_cvd = volume.rolling(20).mean().replace(0, 1)
    df["cvd_slope"] = cvd_slope_raw / vol_mean_cvd

    # ── Sentiment: Funding rate ──────────────────────────────────────────
    if funding_df is not None and not funding_df.empty:
        fdf = funding_df[["funding_time", "funding_rate"]].copy()
        if not pd.api.types.is_datetime64_any_dtype(fdf["funding_time"]):
            fdf["funding_time"] = pd.to_datetime(fdf["funding_time"], unit="ms", utc=True)
        fdf = fdf.sort_values("funding_time")
        merged = pd.merge_asof(
            df[["open_time"]].copy(),
            fdf.rename(columns={"funding_time": "open_time"}),
            on="open_time",
            direction="backward",
        )
        df["funding_rate"] = (merged["funding_rate"].fillna(0.0) * 1000).values
    else:
        df["funding_rate"] = 0.0

    # ── Sentiment: Open Interest change ─────────────────────────────────
    if oi_df is not None and not oi_df.empty:
        odf = oi_df[["timestamp", "sum_open_interest"]].copy()
        if not pd.api.types.is_datetime64_any_dtype(odf["timestamp"]):
            odf["timestamp"] = pd.to_datetime(odf["timestamp"], unit="ms", utc=True)
        odf = odf.sort_values("timestamp")
        merged_oi = pd.merge_asof(
            df[["open_time"]].copy(),
            odf.rename(columns={"timestamp": "open_time"}),
            on="open_time",
            direction="backward",
        )
        oi_series = merged_oi["sum_open_interest"].ffill().fillna(0.0)
        oi_pct = oi_series.pct_change(20).fillna(0.0)
        df["oi_change"] = oi_pct.clip(-1.0, 1.0).values
    else:
        df["oi_change"] = 0.0

    # ── Sentiment: Long/Short ratio ──────────────────────────────────────
    if ls_ratio_df is not None and not ls_ratio_df.empty:
        ldf = ls_ratio_df[["timestamp", "long_short_ratio"]].copy()
        if not pd.api.types.is_datetime64_any_dtype(ldf["timestamp"]):
            ldf["timestamp"] = pd.to_datetime(ldf["timestamp"], unit="ms", utc=True)
        ldf = ldf.sort_values("timestamp")
        merged_ls = pd.merge_asof(
            df[["open_time"]].copy(),
            ldf.rename(columns={"timestamp": "open_time"}),
            on="open_time",
            direction="backward",
        )
        df["ls_ratio"] = (merged_ls["long_short_ratio"].fillna(1.0) - 1.0).values
    else:
        df["ls_ratio"] = 0.0

    # ── Fill NaNs from warmup periods ─────────────────────────────────
    indicator_cols = [
        "return_1",
        "return_20",
        "rsi_14",
        "macd_hist",
        "bb_pctb",
        "atr_14",
        "volume_ratio_20",
        "obv_slope",
        "vwap_dist",
        "ema_trend",
        "cvd_slope",
        "funding_rate",
        "oi_change",
        "ls_ratio",
    ]
    df[indicator_cols] = df[indicator_cols].ffill().bfill().fillna(0.0)

    return df


INDICATOR_COLUMNS: list[str] = [
    "return_1",
    "return_20",
    "rsi_14",
    "macd_hist",
    "bb_pctb",
    "atr_14",
    "volume_ratio_20",
    "obv_slope",
    "vwap_dist",
    "ema_trend",
    "cvd_slope",
    "funding_rate",
    "oi_change",
    "ls_ratio",
]
"""Names of the 14 indicator columns added by :func:`compute_indicators`."""


# ── Private helpers ───────────────────────────────────────────────────────


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute RSI in [0, 100]."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.fillna(50.0)


def _macd(
    close: pd.Series, fast: int, slow: int, signal: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD line, signal line, and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger(close: pd.Series, period: int, n_std: float) -> tuple[pd.Series, pd.Series]:
    """Compute Bollinger %B and bandwidth."""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + n_std * std
    lower = sma - n_std * std
    band_range = upper - lower
    band_range = band_range.replace(0, np.nan)
    pctb = (close - lower) / band_range
    bandwidth = band_range / sma.replace(0, np.nan)
    return pctb.fillna(0.5), bandwidth.fillna(0.0)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Compute Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def _obv_slope(close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    """Compute OBV slope normalized by mean volume."""
    direction = np.sign(close.diff())
    obv = (volume * direction).cumsum()
    slope = obv.diff(period)
    vol_mean = volume.rolling(period).mean().replace(0, 1)
    return slope / vol_mean


def _detect_swings(
    high: np.ndarray, low: np.ndarray, period: int = 5
) -> tuple[list[int], list[int]]:
    """Detect swing highs and lows using rolling window comparison.

    A swing high at index i requires high[i] to be the maximum of
    high[i-period:i+period+1]. Swing lows are analogous with low.

    Returns lists of indices where swings occur.
    """
    n = len(high)
    swing_highs: list[int] = []
    swing_lows: list[int] = []

    for i in range(period, n - period):
        window_high = high[i - period : i + period + 1]
        if high[i] == window_high.max() and high[i] > high[i - 1] and high[i] > high[i + 1]:
            swing_highs.append(i)

        window_low = low[i - period : i + period + 1]
        if low[i] == window_low.min() and low[i] < low[i - 1] and low[i] < low[i + 1]:
            swing_lows.append(i)

    return swing_highs, swing_lows


def _vwap_distance(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Compute rolling VWAP distance as (close - VWAP) / close.

    Uses a 20-period rolling window as an approximation of session VWAP
    since crypto markets trade 24/7 with no clear session boundaries.
    """
    typical_price = (high + low + close) / 3.0
    cum_tp_vol = (typical_price * volume).rolling(20).sum()
    cum_vol = volume.rolling(20).sum().replace(0, np.nan)
    vwap = cum_tp_vol / cum_vol
    close_safe = close.replace(0, np.nan)
    return (close - vwap) / close_safe
