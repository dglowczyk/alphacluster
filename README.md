# AlphaCluster

AlphaZero-inspired deep reinforcement learning system for backtesting crypto perpetual contract trading strategies.

## Architecture

```
                        +-------------------+
                        |   Binance API     |
                        |  (public OHLCV)   |
                        +--------+----------+
                                 |
                                 v
                    +------------+------------+
                    |    Data Pipeline        |
                    |  download -> parquet    |
                    +------------+------------+
                                 |
                                 v
+----------------+   +-----------+-----------+   +------------------+
|   RL Agent     |<->|  Trading Environment  |   |   Backtester     |
|  (PPO / SB3)  |   |  (Gymnasium)          |<->|  (metrics, PnL)  |
+-------+--------+   +-----------------------+   +------------------+
        |
        v
+-------+--------+
|   Tournament   |
|  (ELO system)  |
+----------------+
```

## Key Design Decisions

- **Raw OHLCV only** -- no pre-computed indicators; the neural network learns its own features
- **Discrete action space**: direction (3) x position size (4) x leverage (5) = 60 actions
- **PnL-based reward** with fee, funding rate, and drawdown penalties
- **576-candle observation window** (2 days of 5-min data)
- **Tournament ELO system** for model selection across generations
- **Binance public API** for data (no API keys needed for backtesting)

## Setup

```bash
# Clone the repository
git clone <repo-url> && cd alphacluster

# Create venv and install dependencies
make setup

# Or manually:
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Quickstart

```bash
# 1. Download historical data (BTCUSDT 5-min candles from Binance)
make download-data
# Or via CLI:
alphacluster download-data --symbol BTCUSDT --days 365

# 2. Train a PPO agent on the downloaded data
make train
# Or via CLI:
alphacluster train --timesteps 1000000

# 3. Train with tournament mode (saves generations, runs ELO matches)
alphacluster train --timesteps 1000000 --tournament

# 4. Evaluate the champion model on held-out data
make evaluate
# Or via CLI:
alphacluster evaluate --model champion --episodes 5 --save-charts

# 5. Run a head-to-head tournament between two saved generations
alphacluster tournament --gen-a 0 --gen-b 1 --episodes 10
```

## Quickstart with scripts

The `scripts/` directory provides standalone entry points:

```bash
# Download data with validation and gap-filling
python scripts/download_data.py --symbol BTCUSDT --start 2023-01-01

# Train with custom hyperparameters
python scripts/train.py --timesteps 500000 --learning-rate 1e-4 --batch-size 128

# Evaluate a specific generation and save charts
python scripts/evaluate.py --model 3 --episodes 5 --save-charts
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `alphacluster download-data` | Download OHLCV candles from Binance Futures API |
| `alphacluster train` | Train a PPO agent with optional tournament callback |
| `alphacluster evaluate` | Evaluate a trained model and print performance metrics |
| `alphacluster tournament` | Run ELO tournament between two saved generations |

## Makefile Targets

| Target          | Description                                  |
|-----------------|----------------------------------------------|
| `make setup`    | Create venv and install all dependencies     |
| `make download-data` | Fetch OHLCV candles from Binance       |
| `make train`    | Train an RL agent                            |
| `make evaluate` | Evaluate a trained model                     |
| `make tournament` | Run ELO tournament between saved models   |
| `make test`     | Run test suite                               |
| `make lint`     | Check code style with ruff                   |
| `make format`   | Auto-format code with ruff                   |
| `make clean`    | Remove venv, data, models, caches            |

## Project Structure

```
alphacluster/
├── src/alphacluster/          # Main package
│   ├── config.py              # Central constants (fees, leverage, paths, etc.)
│   ├── cli.py                 # CLI entry point (argparse subcommands)
│   ├── data/                  # Data pipeline
│   │   ├── downloader.py      # Binance Futures API client with pagination
│   │   ├── storage.py         # Parquet save/load with incremental updates
│   │   ├── validator.py       # Gap detection, outlier flagging, interpolation
│   │   └── live_feed.py       # Historical data source for backtesting
│   ├── env/                   # Gymnasium trading environment
│   │   ├── mechanics.py       # Fee, funding, PnL, slippage calculations
│   │   ├── account.py         # Position tracking, margin, liquidation
│   │   └── trading_env.py     # TradingEnv (registered as CryptoPerp-v0)
│   ├── agent/                 # RL agent
│   │   ├── network.py         # CNN+MLP feature extractor for Dict obs
│   │   ├── config.py          # TrainingConfig dataclass
│   │   └── trainer.py         # Agent factory, training loop, tournament callback
│   ├── tournament/            # ELO tournament system
│   │   ├── elo.py             # EloRating class with persistence
│   │   ├── arena.py           # Head-to-head matches and tournament runner
│   │   └── versioning.py      # Generation save/load, champion tracking
│   └── backtest/              # Backtesting engine
│       ├── runner.py          # Run agent through episodes, collect trade logs
│       ├── metrics.py         # Sharpe, Sortino, drawdown, win rate, etc.
│       └── visualizer.py      # Equity curves, trade charts (matplotlib)
├── scripts/                   # Standalone runnable scripts
│   ├── download_data.py       # Data download with progress bars
│   ├── train.py               # Training with data splitting
│   └── evaluate.py            # Evaluation with backtest metrics and charts
├── tests/                     # Test suite (116 tests)
│   ├── test_data.py           # Data pipeline tests
│   ├── test_env.py            # Environment + account + mechanics tests
│   └── test_tournament.py     # ELO, arena, versioning tests
├── data/                      # Downloaded data (gitignored)
└── models/                    # Trained models and ELO ratings (gitignored)
```

## Stack

- Python 3.10+
- PyTorch >= 2.0
- Stable-Baselines3 >= 2.0
- Gymnasium >= 0.29
- pandas + pyarrow (Parquet storage)
- matplotlib (visualization)
- ruff (linting / formatting)
- pytest (testing)
