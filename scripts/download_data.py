#!/usr/bin/env python3
"""Download OHLCV candle data from the Binance Futures public API.

Usage:
    python scripts/download_data.py --symbol BTCUSDT --interval 5m --days 365
"""

import argparse
import sys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download OHLCV data from Binance Futures API",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading pair symbol (default: BTCUSDT)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="5m",
        choices=["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"],
        help="Candle interval (default: 5m)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days of historical data to fetch (default: 365)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/raw/)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"download_data: symbol={args.symbol} interval={args.interval} days={args.days}")
    print("Not yet implemented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
