#!/usr/bin/env python3
"""Evaluate a trained RL model on held-out data and produce performance metrics.

Usage:
    python scripts/evaluate.py --model models/ppo_btcusdt_best.zip
"""

import argparse
import sys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained AlphaCluster model",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model checkpoint (default: latest in models/)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Symbol to evaluate on (default: BTCUSDT)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render equity curve after evaluation",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"evaluate: model={args.model} symbol={args.symbol} episodes={args.episodes}")
    print("Not yet implemented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
