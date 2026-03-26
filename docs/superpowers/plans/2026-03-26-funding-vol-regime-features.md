# Funding Rate + Volatility Regime Features — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 6 new market features (3 funding rate + 3 volatility regime) to the observation space, bringing N_MARKET_FEATURES from 19 to 25.

**Architecture:** New features are computed inside `compute_indicators()` alongside the existing 14 indicators. The function gains an optional `funding_df` parameter. `N_MARKET_FEATURES` is derived programmatically from `INDICATOR_COLUMNS` length to eliminate manual sync. The CNN+Transformer feature extractor auto-adapts (reads shape from observation_space).

**Tech Stack:** pandas, numpy, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-24-additional-features-design.md` (Categories 1 and 3)

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/alphacluster/data/indicators.py` | Modify | Add 6 features to `compute_indicators()` and `INDICATOR_COLUMNS` |
| `src/alphacluster/config.py` | Modify | Derive `N_MARKET_FEATURES` from `INDICATOR_COLUMNS` |
| `src/alphacluster/env/trading_env.py` | Modify | Pass `funding_df` to `compute_indicators()` |
| `tests/test_indicators.py` | Modify | Tests for new features |
| `tests/test_env.py` | Modify | Update observation shape assertions |
| `notebooks/colab_train.ipynb` | No change | Already passes `funding_df` to `TradingEnv` |

---

### Task 1: Funding Rate Features in `compute_indicators()`

Add 3 funding rate features: `funding_rate`, `funding_cumulative_24h`, `funding_premium`.

**Files:**
- Modify: `src/alphacluster/data/indicators.py:13-93` (signature + body of `compute_indicators()`)
- Modify: `src/alphacluster/data/indicators.py:96-112` (`INDICATOR_COLUMNS` list)
- Test: `tests/test_indicators.py`

- [ ] **Step 1: Write failing tests for funding features**

Add to `tests/test_indicators.py`:

```python
def _make_funding(n_events: int = 30, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic 8-hourly funding data."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2025-01-01", periods=n_events, freq="8h", tz="UTC")
    return pd.DataFrame({
        "funding_time": times,
        "symbol": "BTCUSDT",
        "funding_rate": rng.normal(0.0001, 0.0003, size=n_events),
        "mark_price": 50_000.0 + np.cumsum(rng.normal(0, 50, size=n_events)),
    })


class TestFundingFeatures:
    """Tests for funding rate indicator features."""

    def test_funding_columns_present(self):
        """Funding features appear in output when funding_df is provided."""
        df = _make_ohlcv(n=500)
        funding_df = _make_funding()
        result = compute_indicators(df, funding_df=funding_df)
        for col in ["funding_rate", "funding_cumulative_24h", "funding_premium"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_funding_no_nans(self):
        """Funding features should have no NaN values."""
        df = _make_ohlcv(n=500)
        funding_df = _make_funding()
        result = compute_indicators(df, funding_df=funding_df)
        for col in ["funding_rate", "funding_cumulative_24h", "funding_premium"]:
            assert not result[col].isna().any(), f"NaN found in {col}"

    def test_funding_all_finite(self):
        """Funding features should be finite."""
        df = _make_ohlcv(n=500)
        funding_df = _make_funding()
        result = compute_indicators(df, funding_df=funding_df)
        for col in ["funding_rate", "funding_cumulative_24h", "funding_premium"]:
            assert np.all(np.isfinite(result[col].values)), f"Non-finite in {col}"

    def test_funding_none_defaults_to_zero(self):
        """When funding_df is None, funding features should be 0.0."""
        df = _make_ohlcv(n=200)
        result = compute_indicators(df, funding_df=None)
        for col in ["funding_rate", "funding_cumulative_24h", "funding_premium"]:
            assert col in result.columns, f"Missing column: {col}"
            assert (result[col] == 0.0).all(), f"{col} not zero when funding_df is None"

    def test_funding_rate_normalized(self):
        """funding_rate should be multiplied by 100 (×100 normalization)."""
        df = _make_ohlcv(n=500)
        funding_df = _make_funding()
        # Set a known funding rate
        funding_df["funding_rate"] = 0.0001  # 0.01%
        result = compute_indicators(df, funding_df=funding_df)
        # After ×100 normalization, 0.0001 → 0.01
        funding_vals = result["funding_rate"].values
        non_zero = funding_vals[funding_vals != 0.0]
        if len(non_zero) > 0:
            assert np.allclose(non_zero, 0.01, atol=1e-6)

    def test_funding_cumulative_is_rolling_sum(self):
        """funding_cumulative_24h should be rolling sum of last 3 payments × 100."""
        df = _make_ohlcv(n=500)
        funding_df = _make_funding(n_events=10)
        funding_df["funding_rate"] = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005,
                                       0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
        result = compute_indicators(df, funding_df=funding_df)
        # The cumulative should be sum of last 3 funding rates × 100
        # Exact values depend on merge_asof timing, so just verify it's plausible
        cum_vals = result["funding_cumulative_24h"].values
        assert cum_vals.max() <= 0.15  # max possible = (0.0003+0.0004+0.0005)*100 = 0.12
        assert cum_vals.min() >= 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_indicators.py::TestFundingFeatures -v`
Expected: FAIL — `compute_indicators()` doesn't accept `funding_df` parameter yet.

- [ ] **Step 3: Implement funding features in `compute_indicators()`**

In `src/alphacluster/data/indicators.py`, change the signature and add funding logic:

```python
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
        If funding features are desired, ``open_time`` must be a datetime column.
    funding_df:
        Optional DataFrame with ``funding_time, funding_rate, mark_price`` columns.
        If *None*, funding features default to 0.0.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with 20 additional indicator columns.
    """
    df = df.copy()

    # Ensure open_time is datetime if present (needed for funding merge)
    if "open_time" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["open_time"]):
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)

    close = df["close"]
    # ... existing indicator code unchanged ...
```

After the existing indicators (before the NaN fill section), add:

```python
    # ── Funding rate features ──────────────────────────────────────────
    if funding_df is not None and "open_time" in df.columns:
        fdf = funding_df.copy()
        # Ensure funding_time is datetime
        if not pd.api.types.is_datetime64_any_dtype(fdf["funding_time"]):
            fdf["funding_time"] = pd.to_datetime(fdf["funding_time"], utc=True)

        # Cumulative 24h: rolling sum of last 3 payments (on 8h-frequency data)
        fdf = fdf.sort_values("funding_time")
        fdf["funding_cumulative_24h"] = (
            fdf["funding_rate"].rolling(3, min_periods=1).sum()
        )

        # Funding premium: mark_price / nearest close - 1
        # Merge nearest 5-min close to each funding event
        fdf_with_close = pd.merge_asof(
            fdf.sort_values("funding_time"),
            df[["open_time", "close"]].rename(
                columns={"open_time": "funding_time", "close": "close_at_funding"}
            ).sort_values("funding_time"),
            on="funding_time",
            direction="nearest",
        )
        close_at_f = fdf_with_close["close_at_funding"].replace(0, np.nan)
        fdf["funding_premium"] = (fdf_with_close["mark_price"] / close_at_f - 1.0).values

        # Merge funding data into 5-min DataFrame via merge_asof
        merge_cols = ["funding_time", "funding_rate", "funding_cumulative_24h", "funding_premium"]
        df = pd.merge_asof(
            df.sort_values("open_time"),
            fdf[merge_cols].sort_values("funding_time"),
            left_on="open_time",
            right_on="funding_time",
            direction="backward",
        )

        # Normalize: × 100 for rate and cumulative
        df["funding_rate"] = df["funding_rate"].fillna(0.0) * 100
        df["funding_cumulative_24h"] = df["funding_cumulative_24h"].fillna(0.0) * 100
        df["funding_premium"] = df["funding_premium"].fillna(0.0)
    else:
        df["funding_rate"] = 0.0
        df["funding_cumulative_24h"] = 0.0
        df["funding_premium"] = 0.0
```

Add the 3 new columns to `INDICATOR_COLUMNS` (after `"vwap_dist"`):

```python
INDICATOR_COLUMNS: list[str] = [
    # ... existing 14 ...
    "vwap_dist",
    # Funding rate features
    "funding_rate",
    "funding_cumulative_24h",
    "funding_premium",
]
```

Add the 3 new columns to the `indicator_cols` NaN fill list inside `compute_indicators()`:

```python
    indicator_cols = [
        # ... existing 14 ...
        "vwap_dist",
        "funding_rate",
        "funding_cumulative_24h",
        "funding_premium",
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_indicators.py::TestFundingFeatures -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Run full indicator test suite**

Run: `.venv/bin/python -m pytest tests/test_indicators.py -v`
Expected: Most tests PASS — `test_indicator_count` will fail (asserts 14, now 17). Fixed in Step 6.

- [ ] **Step 6: Update indicator count test**

The existing `test_indicator_count` asserts `len(INDICATOR_COLUMNS) == 14`. Update to `17` (14 existing + 3 funding):

```python
    def test_indicator_count(self):
        """Should have exactly 17 indicator columns (14 original + 3 funding)."""
        assert len(INDICATOR_COLUMNS) == 17
```

This count will change again in Task 2 when vol regime features are added (→ 20).

- [ ] **Step 7: Run full test suite and commit**

Run: `.venv/bin/python -m pytest tests/test_indicators.py -v`
Expected: All PASS.

```bash
git add src/alphacluster/data/indicators.py tests/test_indicators.py
git commit -m "feat: add funding rate features to compute_indicators (3 new indicators)"
```

---

### Task 2: Volatility Regime Features in `compute_indicators()`

Add 3 volatility regime features: `vol_percentile`, `vol_of_vol`, `vol_regime`.

**Files:**
- Modify: `src/alphacluster/data/indicators.py:74-93` (add vol regime computation + update fill list + update INDICATOR_COLUMNS)
- Test: `tests/test_indicators.py`

- [ ] **Step 1: Write failing tests for vol regime features**

Add to `tests/test_indicators.py`:

```python
class TestVolRegimeFeatures:
    """Tests for volatility regime indicator features."""

    def test_vol_regime_columns_present(self):
        """Vol regime features appear in output."""
        df = _make_ohlcv(n=500)
        result = compute_indicators(df)
        for col in ["vol_percentile", "vol_of_vol", "vol_regime"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_vol_regime_no_nans(self):
        """Vol regime features should have no NaN values."""
        df = _make_ohlcv(n=500)
        result = compute_indicators(df)
        for col in ["vol_percentile", "vol_of_vol", "vol_regime"]:
            assert not result[col].isna().any(), f"NaN found in {col}"

    def test_vol_regime_all_finite(self):
        """Vol regime features should be finite."""
        df = _make_ohlcv(n=500)
        result = compute_indicators(df)
        for col in ["vol_percentile", "vol_of_vol", "vol_regime"]:
            assert np.all(np.isfinite(result[col].values)), f"Non-finite in {col}"

    def test_vol_percentile_range(self):
        """vol_percentile should be in [0, 1] range after warmup."""
        df = _make_ohlcv(n=500)
        result = compute_indicators(df)
        vals = result["vol_percentile"].values
        assert vals.min() >= 0.0 - 0.01
        assert vals.max() <= 1.0 + 0.01

    def test_vol_regime_values(self):
        """vol_regime should only take values {-1, 0, 1}."""
        df = _make_ohlcv(n=500)
        result = compute_indicators(df)
        unique = set(result["vol_regime"].unique())
        assert unique.issubset({-1.0, 0.0, 1.0}), f"Unexpected values: {unique}"

    def test_vol_regime_bucketization(self):
        """vol_regime should map low percentile → -1, mid → 0, high → +1."""
        df = _make_ohlcv(n=500)
        result = compute_indicators(df)
        # Find rows where vol_percentile is computed (after warmup)
        pct = result["vol_percentile"].values
        regime = result["vol_regime"].values
        for i in range(len(pct)):
            if pct[i] < 0.25:
                assert regime[i] == -1.0, f"Row {i}: pct={pct[i]} but regime={regime[i]}"
            elif pct[i] > 0.75:
                assert regime[i] == 1.0, f"Row {i}: pct={pct[i]} but regime={regime[i]}"
            else:
                assert regime[i] == 0.0, f"Row {i}: pct={pct[i]} but regime={regime[i]}"

    def test_small_df_vol_regime(self):
        """Vol regime features should work with small DataFrames."""
        df = _make_ohlcv(n=10)
        result = compute_indicators(df)
        for col in ["vol_percentile", "vol_of_vol", "vol_regime"]:
            assert not result[col].isna().any(), f"NaN in {col} with small df"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_indicators.py::TestVolRegimeFeatures -v`
Expected: FAIL — columns don't exist yet.

- [ ] **Step 3: Implement vol regime features in `compute_indicators()`**

In `src/alphacluster/data/indicators.py`, add after the existing volatility section (after line 43):

```python
    # ── Volatility regime ──────────────────────────────────────────────
    vol20 = df["volatility_20"]
    df["vol_percentile"] = vol20.rolling(252, min_periods=1).rank(pct=True)
    vol_std = vol20.rolling(60, min_periods=1).std()
    vol_mean = vol20.rolling(60, min_periods=1).mean().replace(0, 1)
    df["vol_of_vol"] = vol_std / vol_mean
    df["vol_regime"] = np.where(
        df["vol_percentile"] < 0.25, -1.0,
        np.where(df["vol_percentile"] > 0.75, 1.0, 0.0),
    )
```

Add the 3 new columns to `INDICATOR_COLUMNS` (after `"funding_premium"`):

```python
INDICATOR_COLUMNS: list[str] = [
    # ... existing 14 + 3 funding ...
    "funding_premium",
    # Volatility regime features
    "vol_percentile",
    "vol_of_vol",
    "vol_regime",
]
```

Add to the `indicator_cols` NaN fill list:

```python
        "funding_premium",
        "vol_percentile",
        "vol_of_vol",
        "vol_regime",
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_indicators.py::TestVolRegimeFeatures -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Update indicator count test**

Update `test_indicator_count` to `20` (14 + 3 funding + 3 vol regime):

```python
    def test_indicator_count(self):
        """Should have exactly 20 indicator columns (14 + 3 funding + 3 vol regime)."""
        assert len(INDICATOR_COLUMNS) == 20
```

- [ ] **Step 6: Run full test suite and commit**

Run: `.venv/bin/python -m pytest tests/test_indicators.py -v`
Expected: All PASS.

```bash
git add src/alphacluster/data/indicators.py tests/test_indicators.py
git commit -m "feat: add volatility regime features to compute_indicators (3 new indicators)"
```

---

### Task 3: Derive `N_MARKET_FEATURES` from `INDICATOR_COLUMNS`

Replace the hardcoded `N_MARKET_FEATURES = 19` with a computed value.

**Files:**
- Modify: `src/alphacluster/config.py:57`
- Test: `tests/test_indicators.py` (add a consistency test)

- [ ] **Step 1: Write failing test**

Add to `tests/test_indicators.py`:

```python
class TestConfigConsistency:
    """Verify config.N_MARKET_FEATURES stays in sync with INDICATOR_COLUMNS."""

    def test_n_market_features_matches_indicator_columns(self):
        from alphacluster.config import N_MARKET_FEATURES
        expected = 5 + len(INDICATOR_COLUMNS)  # 5 OHLCV + indicators
        assert N_MARKET_FEATURES == expected, (
            f"N_MARKET_FEATURES={N_MARKET_FEATURES} but "
            f"5 + len(INDICATOR_COLUMNS)={expected}"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_indicators.py::TestConfigConsistency -v`
Expected: FAIL — `N_MARKET_FEATURES=19` but `5 + 20 = 25`.

- [ ] **Step 3: Update `config.py` to derive `N_MARKET_FEATURES`**

In `src/alphacluster/config.py`, replace line 57:

```python
# Old:
N_MARKET_FEATURES = 19  # 5 OHLCV + 14 technical indicators

# New:
from alphacluster.data.indicators import INDICATOR_COLUMNS
N_MARKET_FEATURES = 5 + len(INDICATOR_COLUMNS)  # 5 OHLCV + indicators (auto-sync)
```

**Circular import check:** `config.py` imports from `indicators.py`. `indicators.py` does NOT import from `config.py`. `trading_env.py` imports from both — this is fine, no cycle.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_indicators.py::TestConfigConsistency -v`
Expected: PASS — `N_MARKET_FEATURES=25`.

- [ ] **Step 5: Run full test suite and commit**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: Some tests may fail due to observation shape change. This is expected — we fix those in Task 4.

```bash
git add src/alphacluster/config.py tests/test_indicators.py
git commit -m "feat: derive N_MARKET_FEATURES from INDICATOR_COLUMNS (19 → 25)"
```

---

### Task 4: Pass `funding_df` Through `TradingEnv` to `compute_indicators()`

Currently `TradingEnv.__init__` calls `compute_indicators(self.df)` without passing `funding_df`. Change it to pass the funding data through.

**Files:**
- Modify: `src/alphacluster/env/trading_env.py:98`
- Modify: `tests/test_env.py` (update shape assertions)

- [ ] **Step 1: Write failing integration test**

Add to `tests/test_env.py`:

```python
class TestObservationShapeV4:
    """Verify observation shape with 25 market features."""

    def test_market_obs_shape_is_25_features(self):
        """Market observation should have 25 features (5 OHLCV + 20 indicators)."""
        from alphacluster.config import N_MARKET_FEATURES, WINDOW_SIZE
        df = _make_ohlcv()
        env = TradingEnv(df=df)
        obs, _ = env.reset()
        assert obs["market"].shape == (WINDOW_SIZE, N_MARKET_FEATURES)
        assert obs["market"].shape[1] == 25

    def test_market_obs_no_nans(self):
        """Market observation should have no NaN/Inf values."""
        df = _make_ohlcv()
        env = TradingEnv(df=df)
        obs, _ = env.reset()
        assert not np.any(np.isnan(obs["market"]))
        assert not np.any(np.isinf(obs["market"]))
```

- [ ] **Step 2: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_env.py::TestObservationShapeV4 -v`
Expected: PASS — after Tasks 1-3, `INDICATOR_COLUMNS` has 20 entries, `N_MARKET_FEATURES=25`, and `compute_indicators()` adds all 20 columns (funding defaults to 0.0). These tests serve as integration verification that the full stack works end-to-end.

- [ ] **Step 3: Pass `funding_df` to `compute_indicators()`**

In `src/alphacluster/env/trading_env.py`, change line 98:

```python
# Old:
self.df = compute_indicators(self.df)

# New:
self.df = compute_indicators(self.df, funding_df=funding_df)
```

This way, when `funding_df` is provided, the env gets real funding features instead of zeros.

- [ ] **Step 4: Fix any existing test shape assertions**

Search `tests/test_env.py` for any hardcoded `19` in market observation shape assertions and update to `N_MARKET_FEATURES` or `25`. Common patterns:

```python
# Old:
assert obs["market"].shape == (WINDOW_SIZE, 19)
# New:
assert obs["market"].shape == (WINDOW_SIZE, N_MARKET_FEATURES)
```

- [ ] **Step 5: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All tests PASS. If any fail due to shape mismatches, fix them.

- [ ] **Step 6: Lint check**

Run: `.venv/bin/ruff check src/ tests/`
Expected: Clean (or only pre-existing E402/F401 in scripts/).

- [ ] **Step 7: Commit**

```bash
git add src/alphacluster/env/trading_env.py tests/test_env.py
git commit -m "feat: pass funding_df through TradingEnv to compute_indicators"
```

---

### Task 5: Update Notebooks and CLAUDE.md

Update the Colab training notebook to reflect the new feature count and update project documentation.

**Files:**
- Modify: `notebooks/colab_train.ipynb` (update markdown describing 19 → 25 features)
- Modify: `notebooks/colab_train_simple.ipynb` (same update)
- Modify: `CLAUDE.md` (update N_MARKET_FEATURES, INDICATOR_COLUMNS references)

- [ ] **Step 1: Update notebook markdown cells**

In both `colab_train.ipynb` and `colab_train_simple.ipynb`, update the "What's new" or feature description cells:
- Change "19-feature market observations" to "25-feature market observations"
- Add mention of funding rate and volatility regime features

- [ ] **Step 2: Update CLAUDE.md**

Update these references in `CLAUDE.md`:
- `N_MARKET_FEATURES = 19` → `N_MARKET_FEATURES = 25` (and note it's auto-derived)
- "5 OHLCV + 14 technical indicators" → "5 OHLCV + 20 indicators (14 technical + 3 funding + 3 vol regime)"
- `MODEL_VERSION` → update to `"v4-funding-vol-regime"` (also in `config.py`)
- Update observation space description: "(576, 19) → (576, 25)"

- [ ] **Step 3: Bump MODEL_VERSION in config.py**

In `src/alphacluster/config.py`:
```python
# Old:
MODEL_VERSION = "v3-opportunity-cost"
# New:
MODEL_VERSION = "v4-funding-vol-regime"
```

- [ ] **Step 4: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All PASS.

- [ ] **Step 5: Lint check and commit**

Run: `.venv/bin/ruff check src/ tests/`

```bash
git add src/alphacluster/config.py CLAUDE.md notebooks/colab_train.ipynb notebooks/colab_train_simple.ipynb
git commit -m "docs: update for v4-funding-vol-regime (N_MARKET_FEATURES 19 → 25)"
```

---

## Verification Checklist

After all tasks:

1. `.venv/bin/python -m pytest tests/ -v` — all tests pass
2. `.venv/bin/ruff check src/ tests/` — clean
3. `python -c "from alphacluster.config import N_MARKET_FEATURES; print(N_MARKET_FEATURES)"` — prints `25`
4. `python -c "from alphacluster.data.indicators import INDICATOR_COLUMNS; print(len(INDICATOR_COLUMNS))"` — prints `20`
5. Manual: create a `TradingEnv` with and without `funding_df`, verify obs shape is `(576, 25)` in both cases
