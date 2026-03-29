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

    Adds 20 indicator columns to the DataFrame.  NaN values from warmup
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

    # ── SMC Lite indicators ──────────────────────────────────────────
    smc = _smc_indicators(
        high.to_numpy(),
        low.to_numpy(),
        close.to_numpy(),
        swing_period=5,
        decay=10,
    )
    for col_name, values in smc.items():
        df[col_name] = values

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
        "swing_high_dist",
        "swing_low_dist",
        "fvg_bull",
        "fvg_bear",
        "bos_signal",
        "sweep_signal",
    ]
    df[indicator_cols] = df[indicator_cols].ffill().bfill().fillna(0.0)

    return df


INDICATOR_COLUMNS: list[str] = [
    # Price Action (4)
    "rsi_14",
    "bb_pctb",
    "vwap_dist",
    "ema_trend",
    # Momentum (4)
    "return_1",
    "return_20",
    "macd_hist",
    "atr_14",
    # Volume / Microstructure (3)
    "volume_ratio_20",
    "obv_slope",
    "cvd_slope",
    # Sentiment (3)
    "funding_rate",
    "oi_change",
    "ls_ratio",
    # SMC Lite (6)
    "swing_high_dist",
    "swing_low_dist",
    "fvg_bull",
    "fvg_bear",
    "bos_signal",
    "sweep_signal",
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


def _smc_indicators(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    swing_period: int = 5,
    decay: int = 10,
) -> dict[str, np.ndarray]:
    """Compute SMC lite indicators: swing distances, FVG, BOS, sweeps.

    Returns dict with 6 arrays, one per SMC feature.
    """
    n = len(close)
    swing_high_dist = np.zeros(n)
    swing_low_dist = np.zeros(n)
    fvg_bull = np.zeros(n)
    fvg_bear = np.zeros(n)
    bos_signal = np.zeros(n)
    sweep_signal = np.zeros(n)

    # Detect swings
    swing_highs, swing_lows = _detect_swings(high, low, period=swing_period)

    # ── Swing distances ──────────────────────────────────────────
    last_sh_val = None
    sh_idx = 0
    last_sl_val = None
    sl_idx = 0

    for i in range(n):
        while sh_idx < len(swing_highs) and swing_highs[sh_idx] <= i:
            last_sh_val = high[swing_highs[sh_idx]]
            sh_idx += 1
        while sl_idx < len(swing_lows) and swing_lows[sl_idx] <= i:
            last_sl_val = low[swing_lows[sl_idx]]
            sl_idx += 1

        c = close[i] if close[i] != 0 else 1.0
        if last_sh_val is not None:
            swing_high_dist[i] = np.clip((last_sh_val - close[i]) / c, -0.1, 0.1)
        if last_sl_val is not None:
            swing_low_dist[i] = np.clip((close[i] - last_sl_val) / c, -0.1, 0.1)

    # ── FVG detection ────────────────────────────────────────────
    last_bull_fvg_low = None
    last_bear_fvg_high = None

    for i in range(2, n):
        # Bullish FVG: low[i] > high[i-2]
        if low[i] > high[i - 2]:
            last_bull_fvg_low = high[i - 2]

        # Bearish FVG: high[i] < low[i-2]
        if high[i] < low[i - 2]:
            last_bear_fvg_high = low[i - 2]

        # Check if FVGs are filled
        if last_bull_fvg_low is not None and low[i] <= last_bull_fvg_low:
            last_bull_fvg_low = None
        if last_bear_fvg_high is not None and high[i] >= last_bear_fvg_high:
            last_bear_fvg_high = None

        c = close[i] if close[i] != 0 else 1.0
        if last_bull_fvg_low is not None:
            fvg_bull[i] = np.clip((last_bull_fvg_low - close[i]) / c, -0.1, 0.1)
        if last_bear_fvg_high is not None:
            fvg_bear[i] = np.clip((last_bear_fvg_high - close[i]) / c, -0.1, 0.1)

    # ── BOS + Sweep detection ────────────────────────────────────
    last_sh = None
    last_sl = None
    sh_ptr = 0
    sl_ptr = 0
    bos_raw = np.zeros(n)
    sweep_raw = np.zeros(n)

    for i in range(n):
        while sh_ptr < len(swing_highs) and swing_highs[sh_ptr] <= i:
            last_sh = high[swing_highs[sh_ptr]]
            sh_ptr += 1
        while sl_ptr < len(swing_lows) and swing_lows[sl_ptr] <= i:
            last_sl = low[swing_lows[sl_ptr]]
            sl_ptr += 1

        if last_sh is not None:
            if close[i] > last_sh:
                bos_raw[i] = 1.0
                last_sh = close[i]
            if high[i] > last_sh and close[i] <= last_sh:
                sweep_raw[i] = -1.0

        if last_sl is not None:
            if close[i] < last_sl:
                bos_raw[i] = -1.0
                last_sl = close[i]
            if low[i] < last_sl and close[i] >= last_sl:
                sweep_raw[i] = 1.0

    # Apply linear decay to BOS and sweep signals
    for signal_raw, signal_out in [(bos_raw, bos_signal), (sweep_raw, sweep_signal)]:
        remaining_decay = 0.0
        decay_dir = 0.0
        for i in range(n):
            if signal_raw[i] != 0:
                remaining_decay = decay
                decay_dir = signal_raw[i]
            if remaining_decay > 0:
                signal_out[i] = decay_dir * (remaining_decay / decay)
                remaining_decay -= 1
            else:
                signal_out[i] = 0.0

    return {
        "swing_high_dist": swing_high_dist,
        "swing_low_dist": swing_low_dist,
        "fvg_bull": fvg_bull,
        "fvg_bear": fvg_bear,
        "bos_signal": bos_signal,
        "sweep_signal": sweep_signal,
    }


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
