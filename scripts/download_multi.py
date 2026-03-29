#!/usr/bin/env python3
"""Download OHLCV data for multiple Binance Futures symbols.

Usage:
    python scripts/download_multi.py
    python scripts/download_multi.py --symbols BTCUSDT ETHUSDT SOLUSDT
    python scripts/download_multi.py --funding
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from alphacluster.config import DATA_DIR
from alphacluster.data.downloader import download_funding_rates, download_klines
from alphacluster.data.storage import append_to_parquet, get_last_timestamp
from alphacluster.data.validator import detect_outliers, fill_gaps, validate_klines

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Top 20 crypto perpetual futures by market cap on Binance Futures
TOP_20_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "TRXUSDT",
    "LINKUSDT",
    "DOTUSDT",
    "MATICUSDT",
    "LTCUSDT",
    "UNIUSDT",
    "ATOMUSDT",
    "NEARUSDT",
    "AAVEUSDT",
    "FILUSDT",
    "ARBUSDT",
    "OPUSDT",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download data for multiple symbols")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=TOP_20_SYMBOLS,
        help="Symbols to download (default: top 20)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2020-01-01",
        help="Start date (default: 2020-01-01)",
    )
    parser.add_argument("--end", type=str, default=None, help="End date (default: now)")
    parser.add_argument("--funding", action="store_true", help="Also download funding rates")
    parser.add_argument("--no-klines", action="store_true", help="Skip OHLCV klines download")
    parser.add_argument(
        "--oi", action="store_true", default=True, help="Download open interest data"
    )
    parser.add_argument(
        "--ls-ratio", action="store_true", default=True, help="Download long/short ratio data"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help=f"Output directory (default: {DATA_DIR})"
    )
    return parser.parse_args(argv)


def download_symbol(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
    funding: bool = False,
    oi: bool = False,
    ls_ratio: bool = False,
    skip_klines: bool = False,
) -> None:
    symbol_lower = symbol.lower()

    if not skip_klines:
        klines_path = output_dir / f"{symbol_lower}_5m.parquet"

        # Incremental: resume from last timestamp
        last_ts = get_last_timestamp(klines_path, time_col="open_time")
        actual_start = start_date
        if last_ts is not None:
            actual_start = last_ts.to_pydatetime().replace(tzinfo=timezone.utc)
            logger.info("[%s] Resuming from %s", symbol, last_ts)

        logger.info("[%s] Downloading klines %s → %s", symbol, actual_start.date(), end_date.date())

        progress = tqdm(desc=f"{symbol} klines", unit=" candles", leave=False)
        df_klines = download_klines(
            symbol=symbol,
            interval="5m",
            start_date=actual_start,
            end_date=end_date,
            progress_callback=lambda n: progress.update(n),
        )
        progress.close()

        if df_klines.empty:
            logger.info("[%s] No new klines", symbol)
        else:
            n_new = append_to_parquet(df_klines, klines_path, time_col="open_time")
            logger.info("[%s] +%d klines (file: %s)", symbol, n_new, klines_path)

            from alphacluster.data.storage import load_from_parquet

            all_klines = load_from_parquet(klines_path)
            report = validate_klines(all_klines)

            if report["n_gaps"] > 0:
                all_klines = fill_gaps(all_klines)
                from alphacluster.data.storage import save_to_parquet

                save_to_parquet(all_klines, klines_path)

            outliers = detect_outliers(all_klines)
            logger.info(
                "[%s] %d rows, %d gaps, %d outliers",
                symbol,
                report["n_rows"],
                report["n_gaps"],
                len(outliers),
            )

    if funding:
        funding_path = output_dir / f"{symbol_lower}_funding.parquet"
        last_funding_ts = get_last_timestamp(funding_path, time_col="funding_time")
        funding_start = start_date
        if last_funding_ts is not None:
            funding_start = last_funding_ts.to_pydatetime().replace(tzinfo=timezone.utc)

        progress_f = tqdm(desc=f"{symbol} funding", unit=" records", leave=False)
        df_funding = download_funding_rates(
            symbol=symbol,
            start_date=funding_start,
            end_date=end_date,
            progress_callback=lambda n: progress_f.update(n),
        )
        progress_f.close()

        if not df_funding.empty:
            n_new = append_to_parquet(df_funding, funding_path, time_col="funding_time")
            logger.info("[%s] +%d funding records", symbol, n_new)

    if oi:
        from alphacluster.data.downloader import download_open_interest

        oi_path = output_dir / f"{symbol_lower}_oi.parquet"
        last_oi_ts = get_last_timestamp(oi_path, time_col="timestamp")
        oi_start = start_date
        if last_oi_ts is not None:
            oi_start = last_oi_ts.to_pydatetime().replace(tzinfo=timezone.utc)

        progress_oi = tqdm(desc=f"{symbol} OI", unit=" records", leave=False)
        df_oi = download_open_interest(
            symbol=symbol,
            start_date=oi_start,
            end_date=end_date,
            progress_callback=lambda n: progress_oi.update(n),
        )
        progress_oi.close()

        if not df_oi.empty:
            n_new = append_to_parquet(df_oi, oi_path, time_col="timestamp")
            logger.info("[%s] +%d OI records", symbol, n_new)

    if ls_ratio:
        from alphacluster.data.downloader import download_ls_ratio

        ls_path = output_dir / f"{symbol_lower}_ls_ratio.parquet"
        last_ls_ts = get_last_timestamp(ls_path, time_col="timestamp")
        ls_start = start_date
        if last_ls_ts is not None:
            ls_start = last_ls_ts.to_pydatetime().replace(tzinfo=timezone.utc)

        progress_ls = tqdm(desc=f"{symbol} L/S", unit=" records", leave=False)
        df_ls = download_ls_ratio(
            symbol=symbol,
            start_date=ls_start,
            end_date=end_date,
            progress_callback=lambda n: progress_ls.update(n),
        )
        progress_ls.close()

        if not df_ls.empty:
            n_new = append_to_parquet(df_ls, ls_path, time_col="timestamp")
            logger.info("[%s] +%d L/S ratio records", symbol, n_new)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    start_date = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end_date = (
        datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
        if args.end
        else datetime.now(tz=timezone.utc)
    )
    output_dir = Path(args.output_dir) if args.output_dir else DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(args.symbols)
    failed = []

    for i, symbol in enumerate(args.symbols, 1):
        print(f"\n{'=' * 60}")
        print(f"  [{i}/{total}] {symbol}")
        print(f"{'=' * 60}")
        try:
            download_symbol(
                symbol,
                start_date,
                end_date,
                output_dir,
                args.funding,
                args.oi,
                args.ls_ratio,
                args.no_klines,
            )
        except Exception as e:
            logger.error("[%s] FAILED: %s", symbol, e)
            failed.append((symbol, str(e)))

    print(f"\n{'=' * 60}")
    print(f"  DONE: {total - len(failed)}/{total} symbols downloaded")
    if failed:
        print(f"  FAILED: {', '.join(s for s, _ in failed)}")
        for sym, err in failed:
            print(f"    {sym}: {err}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 60}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
