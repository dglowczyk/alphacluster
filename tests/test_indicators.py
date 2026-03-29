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
        """Should have exactly 9 indicator columns."""
        assert len(INDICATOR_COLUMNS) == 9

    def test_returns_are_small(self):
        """Return features should be small for normal price data."""
        df = _make_ohlcv(n=500)
        result = compute_indicators(df)
        for col in ["return_1", "return_20"]:
            vals = result[col].values[60:]  # skip warmup
            assert np.abs(vals).max() < 0.5, f"{col} has unreasonably large values"
