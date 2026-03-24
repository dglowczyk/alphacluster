# Additional Observation Features — Design Spec

## Motivation

The current observation space has 19 market features (5 OHLCV + 14 indicators).
All indicators are derived from a single 5-minute timeframe of a single asset.
The agent lacks:

- **Carry cost awareness** — funding rates directly impact position profitability
  but are invisible to the model
- **Higher timeframe context** — the agent sees 576 candles (2 days) but has no
  explicit signal about multi-day trends
- **Cross-asset signals** — crypto markets are highly correlated, and ETH often
  leads BTC moves
- **Volatility regime awareness** — the agent treats high-vol and low-vol markets
  identically despite needing different strategies

## Feature Categories

Four categories of new features, ordered by priority:

| # | Category | New Features | Total New | Data Source |
|---|----------|-------------|-----------|-------------|
| 1 | Funding Rate | 3 | 3 | Already downloaded |
| 2 | Multi-Timeframe | 9 | 9 | Resample existing data |
| 3 | Volatility Regime | 3 | 3 | Derive from existing data |
| 4 | Cross-Asset | 4 | 4 | New download required |
| | **Total** | | **19** | |

After all 4 categories: N_MARKET_FEATURES goes from 19 → 38.

## Architecture Impact

The feature extractor in `network.py` dynamically reads `n_channels` from
`observation_space["market"].shape[1]`. No network code changes are needed.

**Important:** `trading_env.py` does NOT auto-adapt — it uses `N_MARKET_FEATURES`
from config for the observation space shape, and `INDICATOR_COLUMNS` for the
indicator array. These must be kept in sync. To eliminate this coupling,
`N_MARKET_FEATURES` should be derived programmatically:

```python
# config.py
from alphacluster.data.indicators import INDICATOR_COLUMNS
N_MARKET_FEATURES = 5 + len(INDICATOR_COLUMNS)  # 5 OHLCV + indicators
```

If circular imports are an issue, define the constant in `indicators.py` and
import it into `config.py`.

Change points for each new feature:

| Component | File | Change |
|-----------|------|--------|
| Indicator list | `indicators.py` | Add to `INDICATOR_COLUMNS` |
| Compute function | `indicators.py` | Add computation in `compute_indicators()` |
| NaN fill list | `indicators.py` | Add to fill list |
| Docstring/comments | `trading_env.py` | Update `_get_market_obs()` docstring |
| Tests | `test_indicators.py`, `test_env.py` | Update shape assertions |

`N_MARKET_FEATURES` updates automatically if derived from `INDICATOR_COLUMNS`.
`network.py` adapts automatically (reads shape from observation_space).

**Model checkpoint invalidation:** Changing `N_MARKET_FEATURES` makes all
existing model checkpoints incompatible (Conv1d `in_channels` changes). After
each phase, ELO ratings should be reset and a fresh training run started.

---

## Category 1: Funding Rate (3 features)

### Data Source

Funding rates are already downloaded by `downloader.py:download_funding_rates()`
and stored in `data/btcusdt_funding.parquet` with columns:
`funding_time, symbol, funding_rate, mark_price`.

Binance funding is paid every 8 hours (3× per day). Typical range: -0.01% to +0.03%.

### Features

| Feature | Formula | Normalization | Rationale |
|---------|---------|---------------|-----------|
| `funding_rate` | Current funding rate | × 100 (to ~[-1, 3] range) | Direct carry cost signal |
| `funding_cumulative_24h` | Sum of last 3 funding payments | × 100 | Trend in funding direction |
| `funding_premium` | `mark_price / close - 1` | Raw (already ~[-0.01, 0.01]) | Futures premium/discount |

### Implementation

**Data Merging:**
Funding rates are 8-hourly but candles are 5-minute. Merge strategy:
1. In `compute_indicators()`, accept optional `funding_df` parameter
2. Resample funding to 5-min by forward-fill (`ffill`) — each 5-min candle
   inherits the most recent funding rate

**`funding_cumulative_24h` concrete algorithm:**
```python
# Step 1: Compute cumulative on the 8h-frequency funding series (BEFORE ffill)
funding_df["funding_cumulative_24h"] = (
    funding_df["funding_rate"]
    .rolling(3, min_periods=1)
    .sum()
)
# Step 2: Merge into 5-min DataFrame via merge_asof (nearest past event)
df = pd.merge_asof(
    df.sort_values("open_time"),
    funding_df[["funding_time", "funding_rate", "funding_cumulative_24h", "mark_price"]]
        .sort_values("funding_time"),
    left_on="open_time",
    right_on="funding_time",
    direction="backward",
)
# Step 3: Forward-fill any remaining NaN, then normalize
df["funding_rate"] = df["funding_rate"].fillna(0.0) * 100
df["funding_cumulative_24h"] = df["funding_cumulative_24h"].fillna(0.0) * 100
```

**`funding_premium` limitation:** `mark_price` is only available at 8h intervals.
After forward-fill, the premium vs live close will be dominated by staleness
during volatile markets. To mitigate: compute premium at funding timestamps,
then forward-fill the result (not the mark_price itself):
```python
# Compute premium at funding event time, then ffill
funding_df["funding_premium"] = funding_df["mark_price"] / funding_df["close_at_funding"] - 1
# merge_asof brings this already-computed value into 5-min df
```
This requires joining the nearest 5-min close to each funding event first.

**Handling missing funding data:**
If `funding_df` is None (no funding data available), all 3 features default
to 0.0. This maintains backward compatibility.

**Signature change:**
```python
def compute_indicators(
    df: pd.DataFrame,
    funding_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
```

**`open_time` dependency:** The resampling and merge_asof operations require
`open_time` to be a datetime column. `trading_env.py` already converts it
before calling `compute_indicators()`. Callers outside the env path (tests,
notebooks) must ensure `open_time` is datetime. Add a guard at the top of
`compute_indicators()`:
```python
if "open_time" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["open_time"]):
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
```

### Edge Cases

- First funding rate appears Sept 2019. Data before that: fill with 0.0.
- Funding rate can spike during extreme events (>0.1%). No clipping needed —
  the network should see extreme values to learn caution.

---

## Category 2: Multi-Timeframe Features (9 features)

### Rationale

The agent sees 576 × 5-min candles = 2 days. But market regime (trending/ranging)
is better assessed on higher timeframes. Rather than requiring the agent to
implicitly learn multi-timeframe patterns from raw 5-min data, we provide
explicit higher-timeframe indicators.

### Features

**1-hour timeframe (3 features):**

| Feature | Formula | Normalization |
|---------|---------|---------------|
| `rsi_14_1h` | RSI(14) on 1h candles | / 50 - 1 (same as 5min RSI) |
| `macd_hist_1h` | MACD histogram on 1h candles | / (1h_close × 0.01), then ffill to 5min |
| `bb_pctb_1h` | Bollinger %B on 1h candles | Raw (0-1 scale) |

**4-hour timeframe (3 features):**

| Feature | Formula | Normalization |
|---------|---------|---------------|
| `rsi_14_4h` | RSI(14) on 4h candles | / 50 - 1 |
| `macd_hist_4h` | MACD histogram on 4h candles | / (4h_close × 0.01), then ffill to 5min |
| `bb_pctb_4h` | Bollinger %B on 4h candles | Raw (0-1 scale) |

**Trend alignment (3 features):**

| Feature | Formula | Normalization |
|---------|---------|---------------|
| `trend_alignment_1h` | Sign(return_12) × Sign(return_1) | {-1, 0, 1} |
| `trend_alignment_4h` | Sign(return_48) × Sign(return_1) | {-1, 0, 1} |
| `higher_tf_momentum` | `(return_12 + return_48) / 2` | / max(volatility_60, 1e-8) |

`trend_alignment` tells the agent whether 5-min momentum agrees with higher
timeframe direction. `higher_tf_momentum` combines both into a single
volatility-normalized signal. Division uses `max(volatility_60, 1e-8)` floor
to avoid extreme values during calm markets.

### Implementation

**Resampling approach (within `compute_indicators()`):**

```python
# Resample 5-min OHLCV to 1h
df_1h = df.resample("1h", on="open_time").agg({
    "open": "first", "high": "max", "low": "min",
    "close": "last", "volume": "sum",
}).dropna()

# Compute indicators on 1h frame
rsi_1h = _rsi(df_1h["close"], 14) / 50.0 - 1.0
# ... etc

# Forward-fill back to 5-min index
df["rsi_14_1h"] = rsi_1h.reindex(df.index, method="ffill")
```

**Key: resampling happens inside `compute_indicators()`**, not at the data
loading level. The raw 5-min parquet remains unchanged.

**Return-based features** (`return_12`, `return_48`) use existing 5-min close:
- `return_12` = `close.pct_change(12)` (= 1 hour of 5-min candles)
- `return_48` = `close.pct_change(48)` (= 4 hours of 5-min candles)

These are computed directly on 5-min data, no resampling needed.

### Warmup

Higher timeframe indicators need more warmup:
- 1h RSI(14): needs 14 × 12 = 168 candles (~14 hours)
- 4h RSI(14): needs 14 × 48 = 672 candles (~2.3 days)

The existing WINDOW_SIZE of 576 is sufficient for 1h warmup but NOT for 4h.
Since the data starts well before the episode window (we slice from the full
DataFrame), the 4h indicators are computed on the full DataFrame before slicing.
The `_indicators` array in `trading_env.py` stores the full-length array and
slices `start:end` at observation time, so warmup is not an issue.

---

## Category 3: Volatility Regime (3 features)

### Rationale

The agent currently sees `volatility_20` and `volatility_60` as raw values.
It has no context for whether these are historically high or low. A vol regime
feature helps the agent distinguish "calm market, small positions are fine"
from "high vol, be cautious."

### Features

| Feature | Formula | Normalization |
|---------|---------|---------------|
| `vol_percentile` | Percentile rank of `volatility_20` over 252 periods | / 100 (0-1) |
| `vol_of_vol` | Std of `volatility_20` over 60 periods | / mean(volatility_20) over 60 |
| `vol_regime` | `vol_percentile` bucketized: <0.25 → -1, 0.25-0.75 → 0, >0.75 → +1 | {-1, 0, 1} |

### Implementation

```python
# Volatility percentile: rolling rank
vol20 = df["volatility_20"]
df["vol_percentile"] = vol20.rolling(252).rank(pct=True)

# Vol of vol: normalized
vol_std = vol20.rolling(60).std()
vol_mean = vol20.rolling(60).mean().replace(0, 1)
df["vol_of_vol"] = vol_std / vol_mean

# Regime bucketization
df["vol_regime"] = pd.cut(
    df["vol_percentile"],
    bins=[-0.01, 0.25, 0.75, 1.01],
    labels=[-1, 0, 1],
).astype(float)
```

### Implementation Note: `vol_regime` bucketization

Avoid `pd.cut` which returns Categorical (type issues across pandas versions).
Use `np.where` instead:
```python
df["vol_regime"] = np.where(
    df["vol_percentile"] < 0.25, -1.0,
    np.where(df["vol_percentile"] > 0.75, 1.0, 0.0),
)
```

### Warmup

`vol_percentile` needs 252 + 20 periods (252 for rolling rank + 20 for the
underlying `volatility_20`) = 272 periods = ~22.7 hours of 5-min data.
The lookback of 252 5-min periods (~21 hours) is intentionally short — it
measures intraday volatility context, not yearly. Well within the data range
before episode start.

---

## Category 4: Cross-Asset Correlations (4 features)

### Rationale

Crypto markets are highly correlated. ETH/BTC correlation is typically >0.8.
But divergences (ETH leading BTC, or correlation breakdown) are strong signals.

### Features

| Feature | Formula | Normalization |
|---------|---------|---------------|
| `eth_btc_corr_20` | Rolling 20-period correlation of returns | Raw (-1 to 1) |
| `eth_btc_corr_60` | Rolling 60-period correlation of returns | Raw (-1 to 1) |
| `eth_relative_strength` | `eth_return_20 - btc_return_20` | Raw |
| `eth_lead_signal` | Rolling correlation: `eth_return_1(t-1)` vs `btc_return_1(t)` over 60 periods | Raw (-1 to 1) |

`eth_lead_signal` measures whether ETH returns from the previous period predict
BTC returns in the current period. This is a lagged correlation — no future data
is used.

### Data Pipeline Changes

**New download required:** ETHUSDT 5-minute candles.

1. **downloader.py** — Already supports arbitrary symbols via `symbol` parameter.
   No code changes needed for download.
2. **storage.py** — Store as `data/ethusdt_5m.parquet` (separate file).
3. **CLI** — Add `--symbol` flag to `download-data` command, or download both
   by default.

**Data merging in `compute_indicators()`:**
```python
def compute_indicators(
    df: pd.DataFrame,
    funding_df: pd.DataFrame | None = None,
    eth_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
```

Merge ETH data by `open_time` using **left join** on the BTC DataFrame. This
preserves all BTC rows even if ETH has gaps. Missing ETH values are NaN-filled
(forward-fill then 0.0).

```python
df = df.merge(
    eth_df[["open_time", "close"]].rename(columns={"close": "eth_close"}),
    on="open_time",
    how="left",
)
df["eth_close"] = df["eth_close"].ffill().fillna(0.0)
```

If `eth_df` is None, cross-asset features default to 0.0.

### `eth_lead_signal` — Lookahead-Safe Implementation

```python
eth_return = eth_close.pct_change(1)
btc_return = btc_close.pct_change(1)

# Shift ETH return back by 1: correlate eth_return(t-1) with btc_return(t)
eth_lead_signal = eth_return.shift(1).rolling(60).corr(btc_return)
```

At time T, this uses `eth_return[T-60..T-1]` and `btc_return[T-59..T]` — all
past data. No future data is accessed.

### Edge Cases

- ETH data may have gaps where BTC doesn't. Left join + ffill handles this.
- During ETH-specific events (merge, forks), correlation breaks down.
  This is actually useful signal — the agent should see it.
- ETH data starts later than BTC: rows before ETH availability get 0.0
  for all cross-asset features.

---

## Implementation Order

Features should be implemented in this order to maximize value with minimum risk:

### Phase 1: Funding Rate + Volatility Regime (6 features)
- **N_MARKET_FEATURES:** 19 → 25
- **Effort:** 4-5 hours
- **Risk:** Low — uses existing data, no external dependencies
- **Dependencies:** Funding parquet must exist (already downloaded)

### Phase 2: Multi-Timeframe (9 features)
- **N_MARKET_FEATURES:** 25 → 34
- **Effort:** 5-7 hours
- **Risk:** Medium — resampling logic adds complexity, warmup considerations
- **Dependencies:** None (resamples existing 5-min data)

### Phase 3: Cross-Asset (4 features)
- **N_MARKET_FEATURES:** 34 → 38
- **Effort:** 6-8 hours
- **Risk:** Medium — requires new data download, merge logic, CLI changes
- **Dependencies:** ETHUSDT data download

### Each phase is independently deployable and testable.

---

## Testing Strategy

### Unit Tests (per feature)

For each new indicator in `test_indicators.py`:

1. **Computation correctness:** Feed known input, verify output matches
   hand-calculated expected values
2. **NaN handling:** Verify warmup NaN values are properly filled
3. **Shape preservation:** Verify DataFrame shape after `compute_indicators()`
4. **Edge cases:** Zero volume, constant price, single row

### Integration Tests (per phase)

In `test_env.py`:

1. **Observation shape:** `obs["market"].shape == (WINDOW_SIZE, N_MARKET_FEATURES)`
2. **No NaN/Inf in observations:** After reset and step
3. **Backward compatibility:** If optional data (funding_df, eth_df) is None,
   features default to 0.0 and existing behavior is unchanged

### Regression Tests

After each phase, re-run full test suite to ensure no breakage:
```bash
make test  # All existing tests must pass
```

---

## Observation Space Summary (after all phases)

| # | Feature | Source | Category |
|---|---------|--------|----------|
| 0 | open (normalized) | OHLCV | Price |
| 1 | high (normalized) | OHLCV | Price |
| 2 | low (normalized) | OHLCV | Price |
| 3 | close (normalized) | OHLCV | Price |
| 4 | volume (normalized) | OHLCV | Volume |
| 5 | return_1 | 5min | Momentum |
| 6 | return_5 | 5min | Momentum |
| 7 | return_20 | 5min | Momentum |
| 8 | volatility_20 | 5min | Volatility |
| 9 | volatility_60 | 5min | Volatility |
| 10 | rsi_14 | 5min | Momentum |
| 11 | macd_hist | 5min | Momentum |
| 12 | macd_signal_diff | 5min | Momentum |
| 13 | bb_pctb | 5min | Volatility |
| 14 | bb_width | 5min | Volatility |
| 15 | atr_14 | 5min | Volatility |
| 16 | volume_ratio_20 | 5min | Volume |
| 17 | obv_slope | 5min | Volume |
| 18 | vwap_dist | 5min | Volume |
| 19 | funding_rate | Funding | Carry |
| 20 | funding_cumulative_24h | Funding | Carry |
| 21 | funding_premium | Funding | Carry |
| 22 | vol_percentile | Derived | Regime |
| 23 | vol_of_vol | Derived | Regime |
| 24 | vol_regime | Derived | Regime |
| 25 | rsi_14_1h | Resample | Multi-TF |
| 26 | macd_hist_1h | Resample | Multi-TF |
| 27 | bb_pctb_1h | Resample | Multi-TF |
| 28 | rsi_14_4h | Resample | Multi-TF |
| 29 | macd_hist_4h | Resample | Multi-TF |
| 30 | bb_pctb_4h | Resample | Multi-TF |
| 31 | trend_alignment_1h | Derived | Multi-TF |
| 32 | trend_alignment_4h | Derived | Multi-TF |
| 33 | higher_tf_momentum | Derived | Multi-TF |
| 34 | eth_btc_corr_20 | ETH data | Cross-asset |
| 35 | eth_btc_corr_60 | ETH data | Cross-asset |
| 36 | eth_relative_strength | ETH data | Cross-asset |
| 37 | eth_lead_signal | ETH data | Cross-asset |

---

## Dependencies

No new pip packages required. All computations use pandas and numpy
(already present).

Cross-asset features require ETHUSDT data download (same Binance API,
same `download_klines()` function with different symbol parameter).

---

## Interaction with Optuna Tuning

The tuning notebook (Spec A) should be run **after** at least Phase 1 features
are added. The optimal reward scales may differ with richer observations —
the agent might trade more effectively when it can see funding rates and
volatility regime, potentially requiring different fee/opportunity cost balance.

If tuning is run before new features, the results are still valid for the
current feature set. Re-tuning after adding features is recommended but not
required for each phase.
