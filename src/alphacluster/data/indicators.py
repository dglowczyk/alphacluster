"""Technical indicators computed from OHLCV data.

All indicators are normalized to roughly [-1, 1] or [0, 1] range so they
can be fed directly into the neural network alongside normalized OHLCV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators and append them as columns to *df*.

    Adds 11 indicator columns to the DataFrame.  NaN values from warmup
    periods are forward/back-filled so every row has valid data.

    Parameters
    ----------
    df:
        DataFrame with at least ``open, high, low, close, volume`` columns.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with 11 additional indicator columns.
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
]
"""Names of the 11 indicator columns added by :func:`compute_indicators`."""


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
