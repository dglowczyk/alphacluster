# CLAUDE.md

This file provides context for Claude Code when working on the AlphaCluster project.

## Project Overview

AlphaCluster is an AlphaZero-inspired deep reinforcement learning system for backtesting crypto perpetual contract (futures) trading strategies. It uses PPO via Stable-Baselines3, a custom Gymnasium environment, and an ELO tournament system for model selection.

## Project Structure

```
src/alphacluster/
  config.py          - Central constants: fees, leverage, window size, hyperparams, paths
  cli.py             - CLI entry point (argparse subcommands: download-data, train, evaluate, tournament)
  data/
    downloader.py    - Binance Futures API client (klines + funding rates) with pagination
    storage.py       - Parquet save/load, incremental append, deduplication
    validator.py     - Gap detection, NaN checks, outlier flagging, interpolation
    indicators.py    - Technical indicators (14 features: returns, volatility, RSI, MACD, BB, ATR, volume, OBV, VWAP)
    live_feed.py     - Historical data source for backtesting (loads from parquet)
  env/
    mechanics.py     - Stateless helpers: fee, funding, PnL, liquidation, slippage
    account.py       - Position tracking, margin, equity, trade history, position modification
    trading_env.py   - Gymnasium TradingEnv (Dict obs, discrete actions, registered as CryptoPerp-v0)
  agent/
    network.py       - TradingFeatureExtractor: CNN+Transformer for market data + MLP for account state
    config.py        - TrainingConfig dataclass (PPO hyperparams, curriculum, tournament settings)
    trainer.py       - create_agent, train loop, CurriculumCallback, TournamentCallback, save/load
  tournament/
    elo.py           - EloRating class with standard formula + JSON persistence
    arena.py         - run_match (head-to-head) + run_tournament (with ELO + promotion)
    versioning.py    - save/load generations, champion.json tracking
  backtest/
    runner.py        - run_backtest: deterministic episode replay, trade log collection
    metrics.py       - Sharpe, Sortino, max drawdown, win rate, profit factor, direction breakdown, leverage distribution, trade quality
    visualizer.py    - Equity curves, trade overlay, action distribution (matplotlib Agg)

notebooks/           - Jupyter notebooks (kaggle_train.ipynb for GPU training on Kaggle)
scripts/             - Standalone entry points (download_data, train, evaluate)
tests/               - pytest suite: test_data, test_env, test_indicators, test_tournament (180 tests)
data/                - Raw and processed OHLCV data (gitignored)
models/              - Saved model checkpoints and ELO ratings (gitignored)
```

## Key Commands

```bash
make setup           # Create venv + install deps
make test            # Run pytest (180 tests)
make lint            # Ruff check + format check
make format          # Auto-fix lint + format
make download-data   # Fetch candles from Binance
make train           # Train RL agent
make evaluate        # Evaluate trained model
make tournament      # Run ELO tournament

# Direct pytest invocation
.venv/bin/python -m pytest tests/ -v

# Lint
.venv/bin/ruff check src/ tests/
.venv/bin/ruff format --check src/ tests/
```

## Design Decisions

1. **OHLCV + Technical Indicators** -- The observation includes 5 raw OHLCV features plus 14 technical indicators: returns (1/5/20), volatility (20/60), RSI, MACD (histogram + signal diff), Bollinger Bands (%B + width), ATR, volume ratio, OBV slope, VWAP distance. All computed in `data/indicators.py`.

2. **Discrete action space (36 actions)**: direction (long/short/flat=3) x position size (2%/5%/10%/15%=4) x leverage (5x/10x/15x=3). No 0% size option — choosing long/short always opens a position; flat is the only way to have no position.

3. **Observation space**: Market observation is (576, 19) — 576 candles x 19 features (5 OHLCV + 14 indicators). Account observation is (12,) — 7 original features + 5 trade-tracking features (steps since trade, last PnL, trade count, unrealized PnL velocity, running win rate).

4. **Episode length**: 2016 candles (= 7 days of 5-min data).

5. **Multi-component reward function** (8 components):
   - Asymmetric PnL reward (winners weighted 1.5x)
   - Fee penalty (trading cost awareness, scaled by `fee_scale`)
   - Opportunity cost (market-driven penalty for flat during trends, 10-step lookback, 0.2% threshold, 0.02 cap)
   - Position management reward (sqrt ramp for winners up to 3x, escalating penalty for losers)
   - Trade completion reward (quadratic duration scaling for winners, inverse for losers)
   - Churn penalty (penalizes trades held fewer than 20 steps, base 0.02, quadratic)
   - Quadratic drawdown penalty
   - Trade quality bonus (per-step profitability for winners, replaces diversity bonus)

6. **Curriculum learning** (3 phases):
   - Phase 1 (0-30%): "Learn to Trade" — moderate entropy (0.05), opportunity_cost_scale=0.3, partial fee (0.5) and churn (0.5) penalties
   - Phase 2 (30-60%): "Learn Quality" — reduced exploration (0.03), opportunity_cost_scale=0.5, full cost penalties
   - Phase 3 (60-100%): "Refine & Exploit" — low entropy (0.005), opportunity_cost_scale=1.0, amplified fee (2.0x) and churn (2.0x) penalties

7. **CNN+Transformer feature extractor**: 2 Conv1d layers compress (576, 19) → (144, 128), then 3-layer Transformer encoder (4 heads, d_model=128) with learnable positional encoding, adaptive pooling to 128-dim. Account MLP outputs 64-dim. Total: 192-dim feature vector.

8. **Parallel training**: SubprocVecEnv with 4 environments + VecNormalize for reward normalization.

9. **Tournament ELO**: Models compete head-to-head on the same market segments. ELO K-factor = 32, initial rating = 1500.

10. **Data source**: Binance Futures public REST API. No API keys required for historical OHLCV.

11. **Isolated margin liquidation**: Each position uses isolated margin (balance × size_pct at open). Liquidation only loses the allocated margin, not the entire account. Episode continues after liquidation, allowing the agent to learn from both the liquidation penalty and subsequent recovery. This makes `size_pct` a meaningful risk-management dimension — smaller positions risk less capital on liquidation.

## Key Constants (config.py)

- TAKER_FEE = 0.0004, MAKER_FEE = 0.0002
- WINDOW_SIZE = 576, EPISODE_LENGTH = 2016
- N_ACTIONS = 36 (3 x 4 x 3), POSITION_SIZE_OPTIONS = [0.02, 0.05, 0.10, 0.15]
- LEVERAGE_OPTIONS = [5, 10, 15]
- N_MARKET_FEATURES = 19 (5 OHLCV + 14 indicators), N_ACCOUNT_FEATURES = 12
- MAX_LEVERAGE = 15
- LEARNING_RATE = 3e-4, BATCH_SIZE = 128, GAMMA = 0.995
- TOTAL_TIMESTEPS = 2_000_000
- INITIAL_ELO = 1500, ELO_K_FACTOR = 32

## Conventions

- Python 3.10+ (use `X | Y` union syntax, not `Union[X, Y]`)
- Formatting / linting: ruff (configured in pyproject.toml)
- Line length: 100 characters
- All paths use `pathlib.Path`
- Environment variables loaded from `.env` via python-dotenv (optional, never required)
- Data stored as Parquet files in `data/` directory
- Models saved as PyTorch `.pt` state dicts in `models/` directory

## Testing Approach

- pytest with test files in `tests/`
- Test files mirror the source structure: test_data, test_env, test_indicators, test_tournament
- 180 tests covering data pipeline, indicators, environment mechanics, trading env, position modification, ELO, arena, and versioning
- Tests use synthetic data and mock HTTP responses (no network calls)
- Run with: `make test` or `.venv/bin/python -m pytest tests/ -v`

## Integration Notes

- The CLI (`cli.py`) delegates to `scripts/` entry points for download, train, and evaluate
- The `TournamentCallback` in `trainer.py` uses the real tournament system (arena + versioning + ELO)
- The `CurriculumCallback` adjusts entropy coefficient and reward_config across training phases
- The backtest runner reuses the same `TradingEnv` that training uses
- The evaluate script supports `--seed` for reproducible evaluation
- All module `__init__.py` files re-export key symbols for convenient imports
