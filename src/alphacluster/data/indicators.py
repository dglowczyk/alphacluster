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
) -> pd.DataFrame:
    """Compute technical indicators and append them as columns to *df*.

    Adds 20 indicator columns to the DataFrame.  NaN values from warmup
    periods are forward/back-filled so every row has valid data.

    Parameters
    ----------
    df:
        DataFrame with at least ``open, high, low, close, volume`` columns.
    funding_df:
        Optional DataFrame with funding rate data containing ``funding_time``,
        ``funding_rate``, and ``mark_price`` columns (8-hourly from Binance).
        When provided, adds ``funding_rate``, ``funding_cumulative_24h``, and
        ``funding_premium`` columns.  When ``None``, those columns default to 0.0.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with 20 additional indicator columns.
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
    df["return_5"] = close.pct_change(5)
    df["return_20"] = close.pct_change(20)

    # ── Volatility ────────────────────────────────────────────────────
    pct = close.pct_change()
    df["volatility_20"] = pct.rolling(20).std()
    df["volatility_60"] = pct.rolling(60).std()

    # ── Volatility regime ──────────────────────────────────────────────
    vol20 = df["volatility_20"]
    df["vol_percentile"] = vol20.rolling(252, min_periods=1).rank(pct=True)
    vol_std = vol20.rolling(60, min_periods=1).std()
    vol_mean = vol20.rolling(60, min_periods=1).mean().replace(0, 1)
    df["vol_of_vol"] = vol_std / vol_mean
    # vol_regime is derived after NaN fill (see below) so that vol_percentile is consistent

    # ── RSI(14) ───────────────────────────────────────────────────────
    df["rsi_14"] = _rsi(close, 14) / 50.0 - 1.0  # scale to [-1, 1]

    # ── MACD ──────────────────────────────────────────────────────────
    macd_line, signal_line, histogram = _macd(close, 12, 26, 9)
    norm = close * 0.01
    norm = norm.replace(0, 1)  # safety
    df["macd_hist"] = histogram / norm
    df["macd_signal_diff"] = (macd_line - signal_line) / norm

    # ── Bollinger Bands ───────────────────────────────────────────────
    bb_pctb, bb_width = _bollinger(close, 20, 2)
    df["bb_pctb"] = bb_pctb
    df["bb_width"] = bb_width

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

    # ── Funding rate features ──────────────────────────────────────────
    if funding_df is not None and "open_time" in df.columns:
        fdf = funding_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(fdf["funding_time"]):
            fdf["funding_time"] = pd.to_datetime(fdf["funding_time"], utc=True)

        fdf = fdf.sort_values("funding_time")
        fdf["funding_cumulative_24h"] = fdf["funding_rate"].rolling(3, min_periods=1).sum()

        # Funding premium: mark_price / nearest close - 1
        fdf_with_close = pd.merge_asof(
            fdf.sort_values("funding_time"),
            df[["open_time", "close"]]
            .rename(columns={"open_time": "funding_time", "close": "close_at_funding"})
            .sort_values("funding_time"),
            on="funding_time",
            direction="nearest",
        )
        close_at_f = fdf_with_close["close_at_funding"].replace(0, np.nan)
        fdf["funding_premium"] = (fdf_with_close["mark_price"] / close_at_f - 1.0).values

        merge_cols = ["funding_time", "funding_rate", "funding_cumulative_24h", "funding_premium"]
        df = pd.merge_asof(
            df.sort_values("open_time"),
            fdf[merge_cols].sort_values("funding_time"),
            left_on="open_time",
            right_on="funding_time",
            direction="backward",
        )

        df["funding_rate"] = df["funding_rate"].fillna(0.0) * 100
        df["funding_cumulative_24h"] = df["funding_cumulative_24h"].fillna(0.0) * 100
        df["funding_premium"] = df["funding_premium"].fillna(0.0)
    else:
        df["funding_rate"] = 0.0
        df["funding_cumulative_24h"] = 0.0
        df["funding_premium"] = 0.0

    # ── Fill NaNs from warmup periods ─────────────────────────────────
    indicator_cols = [
        "return_1",
        "return_5",
        "return_20",
        "volatility_20",
        "volatility_60",
        "rsi_14",
        "macd_hist",
        "macd_signal_diff",
        "bb_pctb",
        "bb_width",
        "atr_14",
        "volume_ratio_20",
        "obv_slope",
        "vwap_dist",
        "funding_rate",
        "funding_cumulative_24h",
        "funding_premium",
        "vol_percentile",
        "vol_of_vol",
    ]
    df[indicator_cols] = df[indicator_cols].ffill().bfill().fillna(0.0)

    # Derive vol_regime from the already-filled vol_percentile so boundaries are consistent
    df["vol_regime"] = np.where(
        df["vol_percentile"] < 0.25,
        -1.0,
        np.where(df["vol_percentile"] > 0.75, 1.0, 0.0),
    )

    return df


INDICATOR_COLUMNS: list[str] = [
    "return_1",
    "return_5",
    "return_20",
    "volatility_20",
    "volatility_60",
    "rsi_14",
    "macd_hist",
    "macd_signal_diff",
    "bb_pctb",
    "bb_width",
    "atr_14",
    "volume_ratio_20",
    "obv_slope",
    "vwap_dist",
    "funding_rate",
    "funding_cumulative_24h",
    "funding_premium",
    # Volatility regime features
    "vol_percentile",
    "vol_of_vol",
    "vol_regime",
]
"""Names of the 20 indicator columns added by :func:`compute_indicators`."""


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
