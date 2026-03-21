#!/usr/bin/env python3
"""Train an RL agent on the crypto perpetual trading environment.

Usage:
    python scripts/train.py --timesteps 1000000
    python scripts/train.py --tournament   # train + run ELO tournament
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alphacluster.agent.config import TrainingConfig
from alphacluster.agent.trainer import create_agent, save_agent, train
from alphacluster.config import DATA_DIR, MODELS_DIR
from alphacluster.data.storage import DEFAULT_FUNDING_PATH, DEFAULT_KLINES_PATH
from alphacluster.env.trading_env import TradingEnv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an AlphaCluster RL agent",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps (default: 1_000_000)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (default: from TrainingConfig)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: from TrainingConfig)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Symbol to train on (default: BTCUSDT)",
    )
    parser.add_argument(
        "--tournament",
        action="store_true",
        help="Run ELO tournament after training completes",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save model checkpoints (default: models/checkpoints)",
    )
    return parser.parse_args(argv)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load kline and (optional) funding data from parquet files.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame | None]
        (klines_df, funding_df_or_None)
    """
    if not DEFAULT_KLINES_PATH.exists():
        print(
            f"ERROR: No kline data found at {DEFAULT_KLINES_PATH}\n"
            "Please run the data download script first:\n"
            "  python scripts/download_data.py"
        )
        sys.exit(1)

    print(f"Loading kline data from {DEFAULT_KLINES_PATH} ...")
    klines_df = pd.read_parquet(DEFAULT_KLINES_PATH, engine="pyarrow")
    print(f"  Loaded {len(klines_df):,} candles")

    funding_df = None
    if DEFAULT_FUNDING_PATH.exists():
        print(f"Loading funding data from {DEFAULT_FUNDING_PATH} ...")
        funding_df = pd.read_parquet(DEFAULT_FUNDING_PATH, engine="pyarrow")
        print(f"  Loaded {len(funding_df):,} funding records")
    else:
        print("No funding data found; training without funding rates.")

    return klines_df, funding_df


def split_data(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronologically split data into train / validation / test sets.

    Parameters
    ----------
    df:
        Full kline DataFrame, assumed sorted by time.
    train_frac:
        Fraction for training (default 0.70).
    val_frac:
        Fraction for validation (default 0.15).
        The remainder goes to test.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    print(f"Data split: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")
    return train_df, val_df, test_df


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # ── Load data ────────────────────────────────────────────────────────
    klines_df, funding_df = load_data()

    # ── Split chronologically ────────────────────────────────────────────
    train_df, val_df, _test_df = split_data(klines_df)

    # ── Build config ─────────────────────────────────────────────────────
    config_overrides: dict[str, object] = {}
    if args.learning_rate is not None:
        config_overrides["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        config_overrides["batch_size"] = args.batch_size
    config = TrainingConfig(total_timesteps=args.timesteps, **config_overrides)

    # ── Create environments ──────────────────────────────────────────────
    print("Creating training environment ...")
    train_env = TradingEnv(
        df=train_df,
        funding_df=funding_df,
        window_size=config.window_size,
        episode_length=config.episode_length,
    )

    eval_env = None
    if len(val_df) > config.window_size + config.episode_length:
        print("Creating validation environment ...")
        eval_env = TradingEnv(
            df=val_df,
            funding_df=funding_df,
            window_size=config.window_size,
            episode_length=config.episode_length,
        )
    else:
        print("Validation set too small for evaluation; skipping eval callback.")

    # ── Create agent ─────────────────────────────────────────────────────
    print("Creating PPO agent ...")
    agent = create_agent(train_env, config)

    # ── Train ────────────────────────────────────────────────────────────
    checkpoint_dir = args.checkpoint_dir or str(MODELS_DIR / "checkpoints")
    print(f"Training for {config.total_timesteps:,} timesteps ...")
    print(f"  Checkpoints: {checkpoint_dir}")
    if args.tournament:
        print(f"  Tournament enabled (every {config.tournament_freq:,} steps)")

    agent = train(
        agent=agent,
        config=config,
        eval_env=eval_env,
        checkpoint_dir=checkpoint_dir,
        run_tournament=args.tournament,
    )

    # ── Save final model ─────────────────────────────────────────────────
    final_path = MODELS_DIR / "ppo_trading_final"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_agent(agent, final_path)
    print(f"Final model saved to {final_path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
