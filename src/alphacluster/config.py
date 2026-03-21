"""Central configuration for AlphaCluster.

Loads environment variables from .env if present, defines trading constants,
hyperparameters, and path settings used throughout the project.
"""

from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment variables (optional .env file)
# ---------------------------------------------------------------------------
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# ---------------------------------------------------------------------------
# Trading fee / funding constants (Binance perpetuals)
# ---------------------------------------------------------------------------
TAKER_FEE = 0.0004  # 0.04 %
MAKER_FEE = 0.0002  # 0.02 %
FUNDING_INTERVAL_HOURS = 8  # Funding settlement every 8 h

# ---------------------------------------------------------------------------
# Position / leverage limits
# ---------------------------------------------------------------------------
DEFAULT_LEVERAGE = 1
MAX_LEVERAGE = 20

# ---------------------------------------------------------------------------
# Observation & episode geometry
# ---------------------------------------------------------------------------
WINDOW_SIZE = 576  # 576 x 5-min candles = 2 days look-back
EPISODE_LENGTH = 2016  # 2016 x 5-min candles = 7 days per episode

# ---------------------------------------------------------------------------
# Discrete action space
# ---------------------------------------------------------------------------
N_DIRECTIONS = 3  # long / short / flat
N_POSITION_SIZES = 4  # 25%, 50%, 75%, 100% of max
N_LEVERAGE_LEVELS = 5  # e.g. 1x, 3x, 5x, 10x, 20x
N_ACTIONS = N_DIRECTIONS * N_POSITION_SIZES * N_LEVERAGE_LEVELS  # 60

POSITION_SIZE_OPTIONS = [0.25, 0.50, 0.75, 1.0]

# ---------------------------------------------------------------------------
# Observation dimensions
# ---------------------------------------------------------------------------
N_MARKET_FEATURES = 19  # 5 OHLCV + 14 technical indicators
N_ACCOUNT_FEATURES = 12
LEVERAGE_OPTIONS = [1, 3, 5, 10, 20]

# ---------------------------------------------------------------------------
# Default training hyper-parameters
# ---------------------------------------------------------------------------
LEARNING_RATE = 3e-4
BATCH_SIZE = 256
GAMMA = 0.99  # Discount factor
GAE_LAMBDA = 0.95  # GAE lambda for PPO
TOTAL_TIMESTEPS = 1_000_000
N_EPOCHS = 10  # PPO epochs per update

# ---------------------------------------------------------------------------
# Tournament / ELO
# ---------------------------------------------------------------------------
INITIAL_ELO = 1500
ELO_K_FACTOR = 32
TOURNAMENT_ROUNDS = 10

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
BINANCE_BASE_URL = "https://fapi.binance.com"
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "5m"
