# Feature Redesign: Clustered Features + Sentiment + SMC Lite

## Summary

Redesign the market observation features from 19 (5 OHLCV + 14 indicators) to 25
(5 OHLCV + 20 indicators) organized into 5 semantic clusters. Remove redundant
features, add sentiment data from Binance API, and implement SMC (Smart Money
Concepts) lite indicators derived from OHLCV.

## Motivation

- **Redundancy**: Current 14 indicators include 4 volatility measures and overlapping
  momentum features. Removing redundancy reduces noise without losing information.
- **Missing sentiment**: Funding rate exists in the environment but is not in the
  observation space. Open Interest and Long/Short ratio are available from Binance
  but not downloaded.
- **Missing structure**: No support/resistance or market structure features. SMC
  concepts (FVG, BOS, sweeps) provide actionable structure signals derivable from OHLCV.
- **Multi-asset training**: All features must be asset-agnostic (normalized by price).
  Current OHLCV normalization (`/ current_close - 1.0`) already handles this.

## Feature Specification

### Cluster 1: Price Action (4 features)

| Feature | Formula | Range | Notes |
|---------|---------|-------|-------|
| `rsi_14` | RSI(14) / 50 - 1 | [-1, 1] | Existing, unchanged |
| `bb_pctb` | Bollinger %B(20, 2) | [0, 1] | Existing, unchanged |
| `vwap_dist` | (close - VWAP20) / close | ~[-0.05, 0.05] | Existing, unchanged |
| `ema_trend` | (EMA21 - EMA55) / close | ~[-0.05, 0.05] | **New** — short-term trend direction |

### Cluster 2: Momentum (4 features)

| Feature | Formula | Range | Notes |
|---------|---------|-------|-------|
| `return_1` | close.pct_change(1) | ~[-0.05, 0.05] | Existing, unchanged |
| `return_20` | close.pct_change(20) | ~[-0.2, 0.2] | Existing, unchanged |
| `macd_hist` | MACD histogram / (close * 0.01) | ~[-5, 5] | Existing, unchanged |
| `atr_14` | ATR(14) / close | ~[0, 0.05] | Existing, unchanged |

### Cluster 3: Volume / Microstructure (3 features)

| Feature | Formula | Range | Notes |
|---------|---------|-------|-------|
| `volume_ratio_20` | volume / SMA(volume, 20) - 1 | ~[-1, 10] | Existing, unchanged |
| `obv_slope` | OBV.diff(20) / SMA(volume, 20) | ~[-5, 5] | Existing, unchanged |
| `cvd_slope` | CVD.diff(20) / SMA(volume, 20) | ~[-5, 5] | **New** — cumulative volume delta slope; CVD = cumsum((close-open)/(high-low) * volume) |

### Cluster 4: Sentiment (3 features)

| Feature | Source | Formula | Range | Notes |
|---------|--------|---------|-------|-------|
| `funding_rate` | Binance `/fapi/v1/fundingRate` (8h) | rate * 1000, forward-filled to 5m | ~[-0.5, 0.5] | **New** — positive = longs pay shorts |
| `oi_change` | Binance `/futures/data/openInterestHist` (5m) | OI.pct_change(20), clipped [-1, 1] | [-1, 1] | **New** — rising OI = new positions entering |
| `ls_ratio` | Binance `/futures/data/globalLongShortAccountRatio` (5m) | ratio - 1.0 | ~[-0.5, 0.5] | **New** — >0 = more longs, <0 = more shorts |

### Cluster 5: SMC Lite (6 features)

All SMC features use `swing_period=5` (5 candles each side) for swing detection
and `decay=10` candles for signal decay.

| Feature | Formula | Range | Notes |
|---------|---------|-------|-------|
| `swing_high_dist` | (last_swing_high - close) / close | [-0.1, 0.1] clipped | **New** — distance to nearest swing high |
| `swing_low_dist` | (close - last_swing_low) / close | [-0.1, 0.1] clipped | **New** — distance to nearest swing low |
| `fvg_bull` | distance to nearest unfilled bullish FVG / close | [0, 0.1] | **New** — bullish FVG: low[i] > high[i-2] |
| `fvg_bear` | distance to nearest unfilled bearish FVG / close | [-0.1, 0] | **New** — bearish FVG: high[i] < low[i-2] |
| `bos_signal` | +1 bullish / -1 bearish / 0 none | [-1, 1] | **New** — break of structure with linear 10-candle decay |
| `sweep_signal` | +1 sweep low / -1 sweep high / 0 none | [-1, 1] | **New** — liquidity sweep with linear 10-candle decay |

#### SMC Algorithm Details

**Swing detection:** A swing high at index `i` requires `high[i] > max(high[i-N:i])`
AND `high[i] > max(high[i+1:i+N+1])` where N=5. Swing lows are analogous with `low`.
Computed vectorially using rolling max/min.

**FVG detection:** Bullish FVG at candle `i`: `low[i] > high[i-2]` (gap between
candle i and i-2). Bearish: `high[i] < low[i-2]`. An FVG is considered filled when
price returns into the gap. Tracking uses forward-fill with fill mask.

**BOS detection:** Bullish BOS when `close > last_swing_high`. Bearish BOS when
`close < last_swing_low`. Signal decays linearly from 1.0 to 0.0 over 10 candles.

**Sweep detection:** Bullish sweep (sweep low) when `low < last_swing_low` AND
`close > last_swing_low` (wick below, body above). Bearish sweep analogous.
Same 10-candle linear decay.

**Edge cases:** Start of data (no swings yet) = zeros. No FVG in lookback = zeros.
Multiple BOS/sweep in one candle = last signal wins.

## Removed Features

| Feature | Reason |
|---------|--------|
| `return_5` | Redundant — covered by return_1 (short) and return_20 (medium) |
| `volatility_20` | Redundant — atr_14 captures the same information |
| `volatility_60` | Redundant — atr_14 + return_20 cover this |
| `bb_width` | Redundant — atr_14 is a better volatility measure |
| `macd_signal_diff` | Redundant — macd_hist captures the same MACD signal |

## Architecture Changes

### Observation space

- Market: (576, 19) -> (576, 25)
- Account: (12,) — unchanged
- Total: 576 * 25 + 12 = 14,412 floats (was 576 * 19 + 12 = 10,956)

### Neural network (TradingFeatureExtractor)

No code changes needed. First Conv1d layer `in_channels` is parameterized by
`observation_space["market"].shape[-1]`, which will automatically be 25.

### Data pipeline

New files per asset:
- `{symbol}_funding.parquet` — 8h resolution (existing format, need to download for all assets)
- `{symbol}_oi.parquet` — 5m resolution, columns: `timestamp`, `sumOpenInterest`, `sumOpenInterestValue`
- `{symbol}_ls_ratio.parquet` — 5m resolution, columns: `timestamp`, `longShortRatio`, `longAccount`, `shortAccount`

### compute_indicators() signature

```python
def compute_indicators(
    df: pd.DataFrame,
    funding_df: pd.DataFrame | None = None,
    oi_df: pd.DataFrame | None = None,
    ls_ratio_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
```

Graceful degradation: if any sentiment DataFrame is None, corresponding columns
are filled with zeros.

### TradingEnv changes

- New optional params: `oi_df`, `ls_ratio_df` (funding_df already exists)
- Multi-asset mode: new parameter `sentiment_dfs: list[dict] | None` where each
  dict has keys `funding`, `oi`, `ls_ratio` (all optional)
- `_precompute_asset()` passes sentiment data to `compute_indicators()`

### download_multi.py changes

- New flags: `--oi` (default True), `--ls-ratio` (default True)
- New downloader functions: `download_open_interest()`, `download_ls_ratio()`
- Output files: `{symbol}_oi.parquet`, `{symbol}_ls_ratio.parquet`

## Implementation Plan (3 phases)

### Phase 1: Refactor indicators
1. Remove 5 redundant features from `compute_indicators()`
2. Add `ema_trend` and `cvd_slope`
3. Update `INDICATOR_COLUMNS`
4. Update tests in `test_indicators.py`
5. Update `test_env.py` for new observation shape

### Phase 2: Sentiment data + features
1. Add `download_open_interest()` and `download_ls_ratio()` to `data/downloader.py`
2. Add `--oi` and `--ls-ratio` flags to `scripts/download_multi.py`
3. Add `funding_rate`, `oi_change`, `ls_ratio` to `compute_indicators()`
4. Update `TradingEnv` to accept and pass sentiment data
5. Download data for top 10 assets
6. Add tests for new downloader functions and sentiment features

### Phase 3: SMC Lite
1. Implement `_detect_swings()` — vectorized swing high/low detection
2. Implement `_compute_fvg()` — FVG detection and fill tracking
3. Implement `_compute_bos()` — Break of Structure with decay
4. Implement `_compute_sweeps()` — Liquidity sweep with decay
5. Wire into `compute_indicators()` via `_smc_indicators()` helper
6. Add tests with synthetic data (known swings, known FVGs)

## Backward Compatibility

- `compute_indicators()` without sentiment args -> zeros in 3 columns (safe)
- `TradingEnv` without new params -> works as before (zeros in sentiment features)
- Old models (v3) are **NOT compatible** — different observation shape (19 vs 25).
  This is expected since we are training from scratch.

## Performance Impact

- Per-step: ~5-10% slower CNN forward pass (25 vs 19 input channels)
- Per-init: ~2-3s per asset for SMC computation (one-time)
- FPS estimate: ~140-145 (was ~150), training time ~15.5-16h (was ~15h)
- Memory: +32% observation size, negligible in absolute terms (~57KB vs ~44KB per obs)

## Risks

- Binance API may not have historical OI/L-S ratio data going back to 2020 for all
  pairs. Graceful degradation (zeros) handles this.
- SMC features on 5m data may generate noise (false swings). `swing_period=5` is
  tunable if needed.
- 25 features is more for the model to learn from — with 8M steps and 10 assets,
  this should be sufficient data.
