"""Tests for the technical indicators module (alphacluster.data.indicators)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphacluster.data.indicators import INDICATOR_COLUMNS, compute_indicators

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 200, start_price: float = 50_000.0, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    close = start_price + np.cumsum(rng.normal(0, 10, size=n))
    close = np.maximum(close, 100.0)
    high = close + rng.uniform(0, 20, size=n)
    low = close - rng.uniform(0, 20, size=n)
    low = np.maximum(low, 1.0)
    opn = close + rng.normal(0, 5, size=n)
    opn = np.maximum(opn, 1.0)
    volume = rng.uniform(100, 10000, size=n)

    return pd.DataFrame(
        {
            "open_time": pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC"),
            "open": opn,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


# ===========================================================================
# Tests
# ===========================================================================


class TestComputeIndicators:
    """Tests for compute_indicators()."""

    def test_adds_all_columns(self):
        df = _make_ohlcv()
        result = compute_indicators(df)
        for col in INDICATOR_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_nans(self):
        df = _make_ohlcv()
        result = compute_indicators(df)
        for col in INDICATOR_COLUMNS:
            assert not result[col].isna().any(), f"NaN found in {col}"

    def test_all_finite(self):
        df = _make_ohlcv()
        result = compute_indicators(df)
        for col in INDICATOR_COLUMNS:
            assert np.all(np.isfinite(result[col].values)), f"Non-finite value in {col}"

    def test_does_not_modify_original(self):
        df = _make_ohlcv()
        original_cols = set(df.columns)
        compute_indicators(df)
        assert set(df.columns) == original_cols

    def test_preserves_original_columns(self):
        df = _make_ohlcv()
        result = compute_indicators(df)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_rsi_in_range(self):
        """RSI should be scaled to [-1, 1]."""
        df = _make_ohlcv(n=500)
        result = compute_indicators(df)
        rsi = result["rsi_14"].values
        assert rsi.min() >= -1.0 - 0.01
        assert rsi.max() <= 1.0 + 0.01

    def test_bb_pctb_reasonable(self):
        """Bollinger %B should be roughly [0, 1] with some overshoot."""
        df = _make_ohlcv(n=500)
        result = compute_indicators(df)
        pctb = result["bb_pctb"].values
        # Allow some overshoot outside [0, 1] but not extreme
        assert pctb.min() > -2.0
        assert pctb.max() < 3.0

    def test_constant_price(self):
        """Indicators should handle constant price without errors."""
        n = 200
        df = pd.DataFrame(
            {
                "open_time": pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC"),
                "open": np.full(n, 50_000.0),
                "high": np.full(n, 50_000.0),
                "low": np.full(n, 50_000.0),
                "close": np.full(n, 50_000.0),
                "volume": np.full(n, 1000.0),
            }
        )
        result = compute_indicators(df)
        for col in INDICATOR_COLUMNS:
            assert not result[col].isna().any(), f"NaN in {col} with constant price"

    def test_zero_volume(self):
        """Indicators should handle zero volume without errors."""
        df = _make_ohlcv()
        df["volume"] = 0.0
        result = compute_indicators(df)
        for col in INDICATOR_COLUMNS:
            assert not result[col].isna().any(), f"NaN in {col} with zero volume"

    def test_small_dataframe(self):
        """Should work with a small dataframe (< warmup period)."""
        df = _make_ohlcv(n=10)
        result = compute_indicators(df)
        assert len(result) == 10
        for col in INDICATOR_COLUMNS:
            assert not result[col].isna().any()

    def test_indicator_count(self):
        """Should have exactly 20 indicator columns."""
        assert len(INDICATOR_COLUMNS) == 20

    def test_ema_trend_present_and_finite(self):
        df = _make_ohlcv(n=200)
        result = compute_indicators(df)
        assert "ema_trend" in result.columns
        assert not result["ema_trend"].isna().any()
        assert np.all(np.isfinite(result["ema_trend"].values))

    def test_ema_trend_range(self):
        """EMA trend should be small relative values."""
        df = _make_ohlcv(n=500)
        result = compute_indicators(df)
        vals = result["ema_trend"].values[60:]
        assert np.abs(vals).max() < 0.2, "ema_trend values unreasonably large"

    def test_cvd_slope_present_and_finite(self):
        df = _make_ohlcv(n=200)
        result = compute_indicators(df)
        assert "cvd_slope" in result.columns
        assert not result["cvd_slope"].isna().any()
        assert np.all(np.isfinite(result["cvd_slope"].values))

    def test_returns_are_small(self):
        """Return features should be small for normal price data."""
        df = _make_ohlcv(n=500)
        result = compute_indicators(df)
        for col in ["return_1", "return_20"]:
            vals = result[col].values[60:]  # skip warmup
            assert np.abs(vals).max() < 0.5, f"{col} has unreasonably large values"


# ---------------------------------------------------------------------------
# Sentiment helpers
# ---------------------------------------------------------------------------


def _make_funding_df(n: int = 200) -> pd.DataFrame:
    """Create synthetic funding rate data at 8h intervals."""
    n_funding = max(1, n // 96)
    return pd.DataFrame({
        "funding_time": pd.date_range("2025-01-01", periods=n_funding, freq="8h", tz="UTC"),
        "funding_rate": np.random.default_rng(42).normal(0.0001, 0.0005, n_funding),
    })


def _make_oi_df(n: int = 200) -> pd.DataFrame:
    """Create synthetic open interest data at 5m intervals."""
    base_oi = 50000.0 + np.cumsum(np.random.default_rng(42).normal(0, 100, n))
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC"),
        "sum_open_interest": np.maximum(base_oi, 1000.0),
    })


def _make_ls_ratio_df(n: int = 200) -> pd.DataFrame:
    """Create synthetic long/short ratio data at 5m intervals."""
    ratios = 1.0 + np.random.default_rng(42).normal(0, 0.3, n)
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC"),
        "long_short_ratio": np.maximum(ratios, 0.1),
    })


class TestSentimentFeatures:
    """Tests for sentiment features in compute_indicators()."""

    def test_sentiment_columns_present(self):
        df = _make_ohlcv(n=200)
        funding_df = _make_funding_df(n=200)
        oi_df = _make_oi_df(n=200)
        ls_ratio_df = _make_ls_ratio_df(n=200)
        result = compute_indicators(df, funding_df=funding_df, oi_df=oi_df, ls_ratio_df=ls_ratio_df)
        for col in ["funding_rate", "oi_change", "ls_ratio"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_sentiment_no_nans(self):
        df = _make_ohlcv(n=200)
        funding_df = _make_funding_df(n=200)
        oi_df = _make_oi_df(n=200)
        ls_ratio_df = _make_ls_ratio_df(n=200)
        result = compute_indicators(df, funding_df=funding_df, oi_df=oi_df, ls_ratio_df=ls_ratio_df)
        for col in ["funding_rate", "oi_change", "ls_ratio"]:
            assert not result[col].isna().any(), f"NaN in {col}"

    def test_graceful_degradation_no_sentiment(self):
        """Without sentiment data, columns should be zeros."""
        df = _make_ohlcv(n=200)
        result = compute_indicators(df)
        for col in ["funding_rate", "oi_change", "ls_ratio"]:
            assert col in result.columns
            assert (result[col] == 0.0).all(), f"{col} should be all zeros without data"

    def test_oi_change_clipped(self):
        df = _make_ohlcv(n=200)
        oi_df = _make_oi_df(n=200)
        result = compute_indicators(df, oi_df=oi_df)
        vals = result["oi_change"].values
        assert vals.min() >= -1.0 - 0.001
        assert vals.max() <= 1.0 + 0.001


class TestSwingDetection:
    """Tests for swing high/low detection."""

    def test_known_swing_high(self):
        """A clear peak should be detected as swing high."""
        prices = list(range(100, 110)) + list(range(110, 100, -1))
        n = len(prices)
        df = pd.DataFrame({
            "open_time": pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC"),
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": [1000.0] * n,
        })
        from alphacluster.data.indicators import _detect_swings
        swing_highs, swing_lows = _detect_swings(
            df["high"].values, df["low"].values, period=3
        )
        assert len(swing_highs) >= 1
        assert any(8 <= idx <= 11 for idx in swing_highs)

    def test_known_swing_low(self):
        """A clear trough should be detected as swing low."""
        prices = list(range(110, 100, -1)) + list(range(100, 110))
        n = len(prices)
        df = pd.DataFrame({
            "open_time": pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC"),
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": [1000.0] * n,
        })
        from alphacluster.data.indicators import _detect_swings
        swing_highs, swing_lows = _detect_swings(
            df["high"].values, df["low"].values, period=3
        )
        assert len(swing_lows) >= 1
        assert any(8 <= idx <= 11 for idx in swing_lows)

    def test_constant_price_no_swings(self):
        """Constant price should produce no swings."""
        n = 50
        high = np.full(n, 100.0)
        low = np.full(n, 100.0)
        from alphacluster.data.indicators import _detect_swings
        swing_highs, swing_lows = _detect_swings(high, low, period=5)
        assert len(swing_highs) == 0
        assert len(swing_lows) == 0


class TestSMCFeatures:
    """Tests for SMC lite features in compute_indicators()."""

    def test_smc_columns_present(self):
        df = _make_ohlcv(n=500)
        result = compute_indicators(df)
        for col in ["swing_high_dist", "swing_low_dist", "fvg_bull", "fvg_bear",
                     "bos_signal", "sweep_signal"]:
            assert col in result.columns, f"Missing SMC column: {col}"

    def test_smc_no_nans(self):
        df = _make_ohlcv(n=500)
        result = compute_indicators(df)
        for col in ["swing_high_dist", "swing_low_dist", "fvg_bull", "fvg_bear",
                     "bos_signal", "sweep_signal"]:
            assert not result[col].isna().any(), f"NaN in {col}"
            assert np.all(np.isfinite(result[col].values)), f"Inf in {col}"

    def test_swing_dist_clipped(self):
        df = _make_ohlcv(n=500)
        result = compute_indicators(df)
        for col in ["swing_high_dist", "swing_low_dist"]:
            vals = result[col].values
            assert vals.min() >= -0.1 - 0.001
            assert vals.max() <= 0.1 + 0.001

    def test_bos_signal_range(self):
        df = _make_ohlcv(n=500)
        result = compute_indicators(df)
        vals = result["bos_signal"].values
        assert vals.min() >= -1.0 - 0.001
        assert vals.max() <= 1.0 + 0.001

    def test_sweep_signal_range(self):
        df = _make_ohlcv(n=500)
        result = compute_indicators(df)
        vals = result["sweep_signal"].values
        assert vals.min() >= -1.0 - 0.001
        assert vals.max() <= 1.0 + 0.001

    def test_small_dataframe_smc(self):
        """SMC features should work with small data (all zeros)."""
        df = _make_ohlcv(n=15)
        result = compute_indicators(df)
        for col in ["swing_high_dist", "swing_low_dist", "fvg_bull", "fvg_bear",
                     "bos_signal", "sweep_signal"]:
            assert not result[col].isna().any()

    def test_constant_price_smc(self):
        """Constant price should produce zero SMC signals."""
        n = 200
        df = pd.DataFrame({
            "open_time": pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC"),
            "open": np.full(n, 50000.0),
            "high": np.full(n, 50000.0),
            "low": np.full(n, 50000.0),
            "close": np.full(n, 50000.0),
            "volume": np.full(n, 1000.0),
        })
        result = compute_indicators(df)
        assert (result["bos_signal"] == 0.0).all()
        assert (result["sweep_signal"] == 0.0).all()
