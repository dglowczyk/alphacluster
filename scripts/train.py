#!/usr/bin/env python3
"""Train an RL agent on the crypto perpetual trading environment.

Usage:
    python scripts/train.py --timesteps 1000000
    python scripts/train.py --tournament   # train + run ELO tournament
"""

import argparse
import sys


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
        default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size (default: 256)",
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
        help="Directory to save model checkpoints (default: models/)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"train: timesteps={args.timesteps} lr={args.learning_rate} bs={args.batch_size}")
    if args.tournament:
        print("Tournament mode enabled.")
    print("Not yet implemented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
