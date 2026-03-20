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
# 1. Download historical data
make download-data

# 2. Train an agent
make train

# 3. Evaluate performance
make evaluate

# 4. Run tournament between models
make tournament
```

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
├── src/alphacluster/       # Main package
│   ├── config.py           # Central settings and constants
│   ├── cli.py              # CLI entry point
│   ├── data/               # Data download and processing
│   ├── env/                # Gymnasium trading environment
│   ├── agent/              # RL agent (PPO via Stable-Baselines3)
│   ├── tournament/         # ELO tournament system
│   └── backtest/           # Backtesting engine and metrics
├── scripts/                # Runnable scripts
├── tests/                  # Test suite
├── data/                   # Downloaded data (gitignored)
└── models/                 # Trained models (gitignored)
```

## Stack

- Python 3.10+
- PyTorch >= 2.0
- Stable-Baselines3 >= 2.0
- Gymnasium >= 0.29
- pandas + pyarrow (Parquet storage)
- ruff (linting / formatting)
- pytest (testing)
