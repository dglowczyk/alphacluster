"""Command-line interface for AlphaCluster."""

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="alphacluster",
        description="AlphaZero-inspired deep RL for crypto perpetual contract trading",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- download-data ---
    sub_download = subparsers.add_parser("download-data", help="Download OHLCV data from Binance")
    sub_download.add_argument("--symbol", default="BTCUSDT", help="Trading pair (default: BTCUSDT)")
    sub_download.add_argument("--interval", default="5m", help="Candle interval (default: 5m)")
    sub_download.add_argument("--days", type=int, default=365, help="Days of history (default: 365)")

    # --- train ---
    sub_train = subparsers.add_parser("train", help="Train an RL agent")
    sub_train.add_argument(
        "--timesteps", type=int, default=1_000_000, help="Total training timesteps"
    )
    sub_train.add_argument("--tournament", action="store_true", help="Run tournament after training")

    # --- evaluate ---
    sub_eval = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    sub_eval.add_argument("--model", type=str, required=False, help="Path to model checkpoint")

    # --- tournament ---
    subparsers.add_parser("tournament", help="Run ELO tournament between saved models")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    print(f"Command '{args.command}' is not yet implemented.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
