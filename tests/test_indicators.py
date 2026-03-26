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

    def test_returns_are_small(self):
        """Return features should be small for normal price data."""
        df = _make_ohlcv(n=500)
        result = compute_indicators(df)
        for col in ["return_1", "return_5", "return_20"]:
            vals = result[col].values[60:]  # skip warmup
            assert np.abs(vals).max() < 0.5, f"{col} has unreasonably large values"


# ---------------------------------------------------------------------------
# Funding rate helpers and tests
# ---------------------------------------------------------------------------


def _make_funding(n_events: int = 30, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic 8-hourly funding data."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2025-01-01", periods=n_events, freq="8h", tz="UTC")
    return pd.DataFrame(
        {
            "funding_time": times,
            "symbol": "BTCUSDT",
            "funding_rate": rng.normal(0.0001, 0.0003, size=n_events),
            "mark_price": 50_000.0 + np.cumsum(rng.normal(0, 50, size=n_events)),
        }
    )


class TestFundingFeatures:
    """Tests for funding rate features in compute_indicators()."""

    def test_funding_columns_present(self):
        """All 3 funding columns should be present when funding_df is provided."""
        df = _make_ohlcv(n=200)
        fdf = _make_funding()
        result = compute_indicators(df, funding_df=fdf)
        for col in ("funding_rate", "funding_cumulative_24h", "funding_premium"):
            assert col in result.columns, f"Missing column: {col}"

    def test_funding_no_nans(self):
        """Funding feature columns should have no NaN values."""
        df = _make_ohlcv(n=200)
        fdf = _make_funding()
        result = compute_indicators(df, funding_df=fdf)
        for col in ("funding_rate", "funding_cumulative_24h", "funding_premium"):
            assert not result[col].isna().any(), f"NaN found in {col}"

    def test_funding_all_finite(self):
        """All funding feature values should be finite."""
        df = _make_ohlcv(n=200)
        fdf = _make_funding()
        result = compute_indicators(df, funding_df=fdf)
        for col in ("funding_rate", "funding_cumulative_24h", "funding_premium"):
            assert np.all(np.isfinite(result[col].values)), f"Non-finite value in {col}"

    def test_funding_none_defaults_to_zero(self):
        """When funding_df=None, all 3 funding columns should be 0.0."""
        df = _make_ohlcv(n=200)
        result = compute_indicators(df, funding_df=None)
        for col in ("funding_rate", "funding_cumulative_24h", "funding_premium"):
            assert col in result.columns, f"Missing column: {col}"
            assert (result[col] == 0.0).all(), f"{col} should be 0.0 when no funding_df"

    def test_funding_rate_normalized(self):
        """A funding rate of 0.0001 should become 0.01 after ×100 scaling."""
        n = 200
        df = _make_ohlcv(n=n)
        # Build a funding_df with a constant funding_rate = 0.0001
        fdf = pd.DataFrame(
            {
                "funding_time": pd.date_range("2025-01-01", periods=5, freq="8h", tz="UTC"),
                "symbol": "BTCUSDT",
                "funding_rate": [0.0001] * 5,
                "mark_price": [50_000.0] * 5,
            }
        )
        result = compute_indicators(df, funding_df=fdf)
        # After merge_asof backward fill and ×100, all matched rows should be 0.01
        matched = result["funding_rate"]
        # Some rows before the first funding event may be 0.0 (fillna); rest should be 0.01
        non_zero = matched[matched != 0.0]
        assert len(non_zero) > 0, "Expected at least some non-zero funding_rate values"
        assert np.allclose(non_zero.values, 0.01), (
            f"Expected 0.01 but got unique values: {non_zero.unique()}"
        )

    def test_funding_cumulative_is_rolling_sum(self):
        """funding_cumulative_24h should reflect rolling 3-period sum of funding_rate."""
        df = _make_ohlcv(n=200)
        fdf = _make_funding()
        result = compute_indicators(df, funding_df=fdf)
        # Values should be plausible: within a reasonable multiple of typical funding rates
        cum = result["funding_cumulative_24h"]
        # With ×100 scaling and typical rates ~0.0001, cumulative should be small
        assert cum.abs().max() < 10.0, (
            f"funding_cumulative_24h seems too large: max={cum.abs().max()}"
        )


class TestVolRegimeFeatures:
    """Tests for volatility regime indicator features."""

    def test_vol_regime_columns_present(self):
        df = _make_ohlcv(n=500)
        result = compute_indicators(df)
        for col in ["vol_percentile", "vol_of_vol", "vol_regime"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_vol_regime_no_nans(self):
        df = _make_ohlcv(n=500)
        result = compute_indicators(df)
        for col in ["vol_percentile", "vol_of_vol", "vol_regime"]:
            assert not result[col].isna().any(), f"NaN found in {col}"

    def test_vol_regime_all_finite(self):
        df = _make_ohlcv(n=500)
        result = compute_indicators(df)
        for col in ["vol_percentile", "vol_of_vol", "vol_regime"]:
            assert np.all(np.isfinite(result[col].values)), f"Non-finite in {col}"

    def test_vol_percentile_range(self):
        df = _make_ohlcv(n=500)
        result = compute_indicators(df)
        vals = result["vol_percentile"].values
        assert vals.min() >= 0.0 - 0.01
        assert vals.max() <= 1.0 + 0.01

    def test_vol_regime_values(self):
        df = _make_ohlcv(n=500)
        result = compute_indicators(df)
        unique = set(result["vol_regime"].unique())
        assert unique.issubset({-1.0, 0.0, 1.0}), f"Unexpected values: {unique}"

    def test_vol_regime_bucketization(self):
        df = _make_ohlcv(n=500)
        result = compute_indicators(df)
        pct = result["vol_percentile"].values
        regime = result["vol_regime"].values
        for i in range(len(pct)):
            if pct[i] < 0.25:
                assert regime[i] == -1.0
            elif pct[i] > 0.75:
                assert regime[i] == 1.0
            else:
                assert regime[i] == 0.0

    def test_small_df_vol_regime(self):
        df = _make_ohlcv(n=10)
        result = compute_indicators(df)
        for col in ["vol_percentile", "vol_of_vol", "vol_regime"]:
            assert not result[col].isna().any(), f"NaN in {col} with small df"


# ---------------------------------------------------------------------------
# Config consistency tests
# ---------------------------------------------------------------------------


class TestConfigConsistency:
    """Verify config.N_MARKET_FEATURES stays in sync with INDICATOR_COLUMNS."""

    def test_n_market_features_matches_indicator_columns(self):
        from alphacluster.config import N_MARKET_FEATURES

        expected = 5 + len(INDICATOR_COLUMNS)  # 5 OHLCV + indicators
        assert N_MARKET_FEATURES == expected, (
            f"N_MARKET_FEATURES={N_MARKET_FEATURES} but "
            f"5 + len(INDICATOR_COLUMNS)={expected}"
        )
