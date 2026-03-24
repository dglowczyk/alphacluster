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

Change points for each new feature:

| Component | File | Change |
|-----------|------|--------|
| Constant | `config.py` | Update `N_MARKET_FEATURES` |
| Indicator list | `indicators.py` | Add to `INDICATOR_COLUMNS` |
| Compute function | `indicators.py` | Add computation in `compute_indicators()` |
| NaN fill list | `indicators.py` | Add to fill list |
| Tests | `test_indicators.py`, `test_env.py` | Update shape assertions |

`trading_env.py` and `network.py` adapt automatically.

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
3. Compute `funding_cumulative_24h` as rolling sum of the last 3 funding events
   (use `shift` at 8h boundaries, not rolling window of 5-min candles)
4. `funding_premium` requires `mark_price` from funding data, forward-filled
   to 5-min and compared to candle close

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

### Edge Cases

- First funding rate appears Sept 2019. Data before that: fill with 0.0.
- Funding rate can spike during extreme events (>0.1%). No clipping needed —
  the network should see extreme values to learn caution.
- Mark price may differ from close by up to 0.5% during volatility.

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
| `macd_hist_1h` | MACD histogram on 1h candles | / (close × 0.01) |
| `bb_pctb_1h` | Bollinger %B on 1h candles | Raw (0-1 scale) |

**4-hour timeframe (3 features):**

| Feature | Formula | Normalization |
|---------|---------|---------------|
| `rsi_14_4h` | RSI(14) on 4h candles | / 50 - 1 |
| `macd_hist_4h` | MACD histogram on 4h candles | / (close × 0.01) |
| `bb_pctb_4h` | Bollinger %B on 4h candles | Raw (0-1 scale) |

**Trend alignment (3 features):**

| Feature | Formula | Normalization |
|---------|---------|---------------|
| `trend_alignment_1h` | Sign(return_12) × Sign(return_1) | {-1, 0, 1} |
| `trend_alignment_4h` | Sign(return_48) × Sign(return_1) | {-1, 0, 1} |
| `higher_tf_momentum` | `(return_12 + return_48) / 2` | / volatility_60 |

`trend_alignment` tells the agent whether 5-min momentum agrees with higher
timeframe direction. `higher_tf_momentum` combines both into a single
volatility-normalized signal.

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

### Warmup

`vol_percentile` needs 252 periods (= 21 hours of 5-min data). Well within
the data range before episode start.

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
| `eth_lead_signal` | Correlation of `eth_return_1(t)` with `btc_return_1(t+1)` over 60 periods | Raw (-1 to 1) |

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

Merge ETH data by `open_time` (inner join — only timestamps present in both).
If `eth_df` is None, cross-asset features default to 0.0.

### Edge Cases

- ETH data may have gaps where BTC doesn't (and vice versa). Inner join
  handles this, but downstream NaN fill catches any remaining gaps.
- During ETH-specific events (merge, forks), correlation breaks down.
  This is actually useful signal — the agent should see it.
- `eth_lead_signal` is the rolling correlation of `eth_return(t)` vs
  `btc_return(t+1)`. This requires shifted data — careful not to introduce
  lookahead bias. Use `btc_return_1.shift(-1)` only during indicator
  computation, and the resulting correlation is stored as a current-time
  feature (no future leakage since it's computed over past 60 windows).

**Wait — lookahead concern:** `eth_lead_signal` correlates eth_return(t) with
btc_return(t+1). At computation time this uses future BTC data. But we compute
it as a *rolling statistic over the past 60 periods*, so each row's value only
uses data up to that row's time minus 1. This is safe — the correlation at time
T is computed using pairs (eth_t-60...eth_t-1, btc_t-59...btc_t), all past data.

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
