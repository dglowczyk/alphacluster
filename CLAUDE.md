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
    live_feed.py     - Historical data source for backtesting (loads from parquet)
  env/
    mechanics.py     - Stateless helpers: fee, funding, PnL, liquidation, slippage
    account.py       - Position tracking, margin, equity, trade history
    trading_env.py   - Gymnasium TradingEnv (Dict obs, discrete actions, registered as CryptoPerp-v0)
  agent/
    network.py       - TradingFeatureExtractor: CNN for market data + MLP for account state
    config.py        - TrainingConfig dataclass (PPO hyperparams, schedule, tournament settings)
    trainer.py       - create_agent, train loop, TournamentCallback, save/load helpers
  tournament/
    elo.py           - EloRating class with standard formula + JSON persistence
    arena.py         - run_match (head-to-head) + run_tournament (with ELO + promotion)
    versioning.py    - save/load generations, champion.json tracking
  backtest/
    runner.py        - run_backtest: deterministic episode replay, trade log collection
    metrics.py       - Sharpe, Sortino, max drawdown, win rate, profit factor, etc.
    visualizer.py    - Equity curves, trade overlay, action distribution (matplotlib Agg)

scripts/             - Standalone entry points (download_data, train, evaluate)
tests/               - pytest suite: test_data, test_env, test_tournament (116 tests)
data/                - Raw and processed OHLCV data (gitignored)
models/              - Saved model checkpoints and ELO ratings (gitignored)
```

## Key Commands

```bash
make setup           # Create venv + install deps
make test            # Run pytest (116 tests)
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

1. **Raw OHLCV only** -- No pre-computed technical indicators. The neural network should learn its own features from raw price data.

2. **Discrete action space (60 actions)**: direction (long/short/flat=3) x position size (0%/25%/50%/100%=4) x leverage (1x/3x/5x/10x/20x=5).

3. **Observation window**: 576 candles of 5-min data (= 2 days). Stacked as a 2D array the agent sees as a "price image".

4. **Episode length**: 2016 candles (= 7 days of 5-min data).

5. **Reward**: PnL-based with penalties for taker fees (0.04%), funding rates (every 8h), and max drawdown.

6. **Tournament ELO**: Models compete head-to-head on the same market segments. ELO K-factor = 32, initial rating = 1500. The TournamentCallback in trainer.py automatically saves generations and runs matches during training.

7. **Data source**: Binance Futures public REST API. No API keys required for historical OHLCV.

8. **Feature extractor**: CNN (3 conv layers) for the OHLCV window + MLP for account state, producing a 160-dim feature vector fed to PPO.

## Key Constants (config.py)

- TAKER_FEE = 0.0004, MAKER_FEE = 0.0002
- WINDOW_SIZE = 576, EPISODE_LENGTH = 2016
- N_ACTIONS = 60 (3 x 4 x 5)
- MAX_LEVERAGE = 20
- LEARNING_RATE = 3e-4, BATCH_SIZE = 256, GAMMA = 0.99
- INITIAL_ELO = 1500, ELO_K_FACTOR = 32

## Conventions

- Python 3.10+ (use `X | Y` union syntax, not `Union[X, Y]`)
- Formatting / linting: ruff (configured in pyproject.toml)
- Line length: 100 characters
- All paths use `pathlib.Path`
- Environment variables loaded from `.env` via python-dotenv (optional, never required)
- Data stored as Parquet files in `data/` directory
- Models saved as SB3 `.zip` checkpoints in `models/` directory

## Testing Approach

- pytest with test files in `tests/`
- Test files mirror the source structure: test_data, test_env, test_tournament
- 116 tests covering data pipeline, environment mechanics, trading env, ELO, arena, and versioning
- Tests use synthetic data and mock HTTP responses (no network calls)
- Run with: `make test` or `.venv/bin/python -m pytest tests/ -v`

## Integration Notes

- The CLI (`cli.py`) delegates to `scripts/` entry points for download, train, and evaluate
- The `TournamentCallback` in `trainer.py` uses the real tournament system (arena + versioning + ELO)
- The backtest runner reuses the same `TradingEnv` that training uses
- All module `__init__.py` files re-export key symbols for convenient imports
