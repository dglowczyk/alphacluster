#!/usr/bin/env python3
"""Download OHLCV candle data and funding rates from the Binance Futures API.

Usage:
    python scripts/download_data.py [--symbol BTCUSDT] [--start 2019-09-01]
    python scripts/download_data.py --funding  # also download funding rates
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

from tqdm import tqdm

from alphacluster.config import DATA_DIR, DEFAULT_INTERVAL, DEFAULT_SYMBOL
from alphacluster.data.downloader import download_funding_rates, download_klines
from alphacluster.data.storage import (
    DEFAULT_FUNDING_PATH,
    DEFAULT_KLINES_PATH,
    append_to_parquet,
    get_last_timestamp,
)
from alphacluster.data.validator import detect_outliers, fill_gaps, validate_klines

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download OHLCV data from Binance Futures API",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=DEFAULT_SYMBOL,
        help=f"Trading pair symbol (default: {DEFAULT_SYMBOL})",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default=DEFAULT_INTERVAL,
        choices=["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"],
        help=f"Candle interval (default: {DEFAULT_INTERVAL})",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2019-09-01",
        help="Start date in ISO format (default: 2019-09-01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date in ISO format (default: now)",
    )
    parser.add_argument(
        "--funding",
        action="store_true",
        help="Also download funding rate history",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Output directory (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing data, don't download",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    start_date = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end_date = (
        datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
        if args.end
        else datetime.now(tz=timezone.utc)
    )

    klines_path = DEFAULT_KLINES_PATH
    funding_path = DEFAULT_FUNDING_PATH

    if args.validate_only:
        from alphacluster.data.storage import load_from_parquet

        logger.info("Validating existing klines at %s", klines_path)
        df = load_from_parquet(klines_path)
        report = validate_klines(df)
        logger.info("Validation report: %s", report)
        outliers = detect_outliers(df)
        if len(outliers):
            logger.warning("Outliers:\n%s", outliers[["open_time", "close", "pct_change"]])
        return 0

    # --- Incremental klines download ---
    last_ts = get_last_timestamp(klines_path, time_col="open_time")
    if last_ts is not None:
        logger.info("Resuming klines download from %s", last_ts)
        start_date = last_ts.to_pydatetime().replace(tzinfo=timezone.utc)

    logger.info(
        "Downloading klines: symbol=%s interval=%s start=%s end=%s",
        args.symbol, args.interval, start_date.date(), end_date.date(),
    )

    progress = tqdm(desc="Downloading klines", unit=" candles")

    def _klines_progress(n: int) -> None:
        progress.update(n)

    df_klines = download_klines(
        symbol=args.symbol,
        interval=args.interval,
        start_date=start_date,
        end_date=end_date,
        progress_callback=_klines_progress,
    )
    progress.close()

    if df_klines.empty:
        logger.info("No new klines to download.")
    else:
        n_new = append_to_parquet(df_klines, klines_path, time_col="open_time")
        logger.info("Added %d new klines (total file at %s)", n_new, klines_path)

        # Validate
        from alphacluster.data.storage import load_from_parquet

        all_klines = load_from_parquet(klines_path)
        report = validate_klines(all_klines)
        logger.info("Validation: %d rows, %d gaps, %d NaNs, valid=%s",
                     report["n_rows"], report["n_gaps"], report["n_nans"], report["is_valid"])

        # Fill small gaps
        if report["n_gaps"] > 0:
            all_klines = fill_gaps(all_klines)
            from alphacluster.data.storage import save_to_parquet

            save_to_parquet(all_klines, klines_path)
            logger.info("Gaps filled. Final row count: %d", len(all_klines))

        # Detect outliers
        outliers = detect_outliers(all_klines)
        if len(outliers):
            logger.warning("%d outlier candles detected", len(outliers))

    # --- Funding rates ---
    if args.funding:
        last_funding_ts = get_last_timestamp(funding_path, time_col="funding_time")
        funding_start = start_date
        if last_funding_ts is not None:
            funding_start = last_funding_ts.to_pydatetime().replace(tzinfo=timezone.utc)
            logger.info("Resuming funding download from %s", last_funding_ts)

        logger.info("Downloading funding rates for %s", args.symbol)
        progress_f = tqdm(desc="Downloading funding rates", unit=" records")

        def _funding_progress(n: int) -> None:
            progress_f.update(n)

        df_funding = download_funding_rates(
            symbol=args.symbol,
            start_date=funding_start,
            end_date=end_date,
            progress_callback=_funding_progress,
        )
        progress_f.close()

        if df_funding.empty:
            logger.info("No new funding records.")
        else:
            n_new = append_to_parquet(df_funding, funding_path, time_col="funding_time")
            logger.info("Added %d new funding records", n_new)

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
