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
                    |  + technical indicators |
                    +------------+------------+
                                 |
                                 v
+----------------+   +-----------+-----------+   +------------------+
|   RL Agent     |<->|  Trading Environment  |   |   Backtester     |
| (PPO + CNN +   |   |  (Gymnasium)          |<->|  (metrics, PnL)  |
|  Transformer)  |   |  4x parallel envs     |   +------------------+
+-------+--------+   +-----------------------+
        |
        v
+-------+--------+
|   Tournament   |
|  (ELO system)  |
+----------------+
```

## Key Design Decisions

- **OHLCV + 14 technical indicators** — RSI, MACD, Bollinger Bands, ATR, volume ratio, OBV slope, VWAP distance, returns, volatility
- **CNN+Transformer feature extractor** — Conv1d compression + 3-layer Transformer encoder with learnable positional encoding
- **Discrete action space**: direction (3) x position size (4) x leverage (5) = 60 actions; no 0% size — agent must trade or explicitly go flat
- **Multi-component reward** — asymmetric PnL (1.5x winners), inactivity penalty, position management, trade completion bonus, quadratic drawdown
- **Curriculum learning** — 3 phases: explore (high entropy) → quality (normal) → exploit (strict drawdown)
- **Parallel training** — 4x SubprocVecEnv + VecNormalize reward normalization
- **576-candle observation window** (2 days of 5-min data) with 19 features per candle
- **12-dimensional account state** — balance, position, PnL, drawdown, trade tracking, win rate
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

# 2. Train a PPO agent (2M timesteps, 4 parallel envs, curriculum learning)
make train
# Or via CLI:
alphacluster train --timesteps 2000000

# 3. Train with tournament mode (saves generations, runs ELO matches)
alphacluster train --timesteps 2000000 --tournament

# 4. Evaluate the champion model on held-out data (reproducible with --seed)
make evaluate
# Or via CLI:
alphacluster evaluate --model champion --episodes 5 --seed 42 --save-charts

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

# Train with single environment (no parallelism)
python scripts/train.py --timesteps 500000 --n-envs 1

# Train without curriculum learning
python scripts/train.py --timesteps 500000 --no-curriculum

# Evaluate a specific generation with reproducible seeding
python scripts/evaluate.py --model 3 --episodes 5 --seed 42 --save-charts
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
│   │   ├── indicators.py      # Technical indicators (14 features, pure numpy/pandas)
│   │   └── live_feed.py       # Historical data source for backtesting
│   ├── env/                   # Gymnasium trading environment
│   │   ├── mechanics.py       # Fee, funding, PnL, slippage calculations
│   │   ├── account.py         # Position tracking, margin, liquidation
│   │   └── trading_env.py     # TradingEnv (registered as CryptoPerp-v0)
│   ├── agent/                 # RL agent
│   │   ├── network.py         # CNN+Transformer feature extractor for Dict obs
│   │   ├── config.py          # TrainingConfig (PPO, curriculum, tournament)
│   │   └── trainer.py         # Agent factory, training loop, curriculum + tournament callbacks
│   ├── tournament/            # ELO tournament system
│   │   ├── elo.py             # EloRating class with persistence
│   │   ├── arena.py           # Head-to-head matches and tournament runner
│   │   └── versioning.py      # Generation save/load, champion tracking
│   └── backtest/              # Backtesting engine
│       ├── runner.py          # Run agent through episodes, collect trade logs
│       ├── metrics.py         # Sharpe, Sortino, drawdown, win rate, flat %, streaks
│       └── visualizer.py      # Equity curves, trade charts (matplotlib)
├── scripts/                   # Standalone runnable scripts
│   ├── download_data.py       # Data download with progress bars
│   ├── train.py               # Training with SubprocVecEnv + curriculum
│   └── evaluate.py            # Evaluation with seeding and extended metrics
├── notebooks/                 # Jupyter notebooks
│   └── kaggle_train.ipynb     # GPU training on Kaggle (quiet logging)
├── tests/                     # Test suite (138 tests)
│   ├── test_data.py           # Data pipeline tests
│   ├── test_env.py            # Environment + account + mechanics tests
│   ├── test_indicators.py     # Technical indicators tests
│   └── test_tournament.py     # ELO, arena, versioning tests
├── data/                      # Downloaded data (gitignored)
└── models/                    # Trained models and ELO ratings (gitignored)
```

## Training on Kaggle (free GPU)

Kaggle provides free GPU access (Tesla P100/T4) for faster training.

### Setup

1. **Upload the repo as a Kaggle Dataset:**
   - Zip the project directory and upload it as a new Dataset named `alphacluster`
   - It should contain `src/`, `scripts/`, etc. at the top level

2. **Upload your data as a separate Dataset:**
   - Create a Dataset named `alphacluster-data`
   - Upload `btcusdt_5m.parquet` (and optionally `btcusdt_funding.parquet`) from your local `data/` directory

3. **Create a new Kaggle Notebook:**
   - Import the contents of `notebooks/kaggle_train.ipynb`
   - Attach both Datasets (`alphacluster` and `alphacluster-data`)
   - Enable **GPU accelerator** in notebook settings
   - Enable **Internet** (needed for pip install)

4. **Run all cells.** Training uses quiet mode (one-line progress updates instead of per-rollout logs). The trained model will appear in `/kaggle/working/models/` — download it from the Output tab and place it in your local `models/` directory.

### What the notebook does

- Creates 4 parallel SubprocVecEnv environments with VecNormalize reward normalization
- Trains with curriculum learning (3 phases over 2M timesteps)
- Uses the CNN+Transformer feature extractor
- Evaluates on 5 episodes with deterministic seeding (seed=42)
- Prints extended metrics: trades/episode, flat time %, win/lose streaks

## Stack

- Python 3.10+
- PyTorch >= 2.0
- Stable-Baselines3 >= 2.0
- Gymnasium >= 0.29
- pandas + pyarrow (Parquet storage)
- matplotlib (visualization)
- ruff (linting / formatting)
- pytest (testing)
