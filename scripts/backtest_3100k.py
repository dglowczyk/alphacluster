"""Backtest checkpoint 3100k on all available assets.

Runs 1 episode per asset on the last 5 weeks of data with $1000 initial capital.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alphacluster.agent.trainer import load_agent
from alphacluster.backtest.metrics import calculate_metrics, print_report
from alphacluster.backtest.runner import run_backtest
from alphacluster.config import WINDOW_SIZE
from alphacluster.data.storage import load_from_parquet
from alphacluster.env.trading_env import TradingEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "models 2" / "checkpoints" / "checkpoint_3100k.pt"
INITIAL_BALANCE = 1_000.0
SEED = 42

# 5 weeks = 35 days × 24h × 12 candles/h = 10080 candles
FIVE_WEEKS_CANDLES = 35 * 24 * 12
EPISODE_LENGTH = FIVE_WEEKS_CANDLES - WINDOW_SIZE - 1  # room for window


def get_all_assets() -> list[str]:
    """Find all symbols with 5m parquet files."""
    files = sorted(DATA_DIR.glob("*_5m.parquet"))
    return [f.stem.replace("_5m", "").upper() for f in files]


def load_asset_data(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load OHLCV and optional funding data for a symbol."""
    sym = symbol.lower()
    df = load_from_parquet(DATA_DIR / f"{sym}_5m.parquet")

    funding_df = None
    funding_path = DATA_DIR / f"{sym}_funding.parquet"
    if funding_path.exists():
        funding_df = load_from_parquet(funding_path)

    return df, funding_df


def run_asset_backtest(symbol: str, model) -> dict | None:
    """Run backtest on a single asset, last 5 weeks."""
    try:
        df, funding_df = load_asset_data(symbol)
    except Exception as e:
        logger.warning("Failed to load %s: %s", symbol, e)
        return None

    # Take the last 5 weeks of data
    if len(df) < FIVE_WEEKS_CANDLES:
        logger.warning("%s: only %d candles, need %d. Skipping.", symbol, len(df), FIVE_WEEKS_CANDLES)
        return None

    test_df = df.iloc[-FIVE_WEEKS_CANDLES:].reset_index(drop=True)

    # Trim funding to same period
    if funding_df is not None:
        start_time = test_df["open_time"].iloc[0]
        funding_df = funding_df[funding_df["funding_time"] >= start_time].reset_index(drop=True)

    env = TradingEnv(
        df=test_df,
        funding_df=funding_df,
        episode_length=EPISODE_LENGTH,
        initial_balance=INITIAL_BALANCE,
        simple_actions=True,
    )

    result = run_backtest(model, env, n_episodes=1, seed=SEED)
    metrics = calculate_metrics(result)

    return {
        "symbol": symbol,
        "pnl_pct": metrics.get("avg_episode_return_pct", 0),
        "final_equity": INITIAL_BALANCE * (1 + metrics.get("avg_episode_return_pct", 0) / 100),
        "trades": metrics.get("trade_count", 0),
        "win_rate": metrics.get("win_rate", 0),
        "profit_factor": metrics.get("profit_factor", 0),
        "max_dd_pct": metrics.get("max_drawdown_pct", 0),
        "sharpe": metrics.get("sharpe_ratio", 0),
        "avg_duration": metrics.get("avg_trade_duration", 0),
        "long_count": metrics.get("long_count", 0),
        "short_count": metrics.get("short_count", 0),
        "metrics": metrics,
        "result": result,
    }


def main():
    assets = get_all_assets()
    print(f"\n{'='*70}")
    print(f"  MULTI-ASSET BACKTEST — Checkpoint 3100k")
    print(f"  Capital: ${INITIAL_BALANCE:.0f} | Period: last 5 weeks | Seed: {SEED}")
    print(f"  Assets: {len(assets)} ({', '.join(assets)})")
    print(f"  Episode length: {EPISODE_LENGTH} steps ({EPISODE_LENGTH * 5 / 60 / 24:.1f} days)")
    print(f"{'='*70}\n")

    # Create a dummy env for model loading
    dummy_df, dummy_funding = load_asset_data(assets[0])
    dummy_test = dummy_df.iloc[-FIVE_WEEKS_CANDLES:].reset_index(drop=True)
    dummy_env = TradingEnv(
        df=dummy_test,
        episode_length=EPISODE_LENGTH,
        initial_balance=INITIAL_BALANCE,
        simple_actions=True,
    )
    model = load_agent(str(MODEL_PATH), env=dummy_env)
    print(f"Model loaded from: {MODEL_PATH}\n")

    # Run backtests
    results = []
    for symbol in assets:
        print(f"--- {symbol} ---")
        r = run_asset_backtest(symbol, model)
        if r is not None:
            results.append(r)
            pnl = r["pnl_pct"]
            sign = "+" if pnl >= 0 else ""
            print(
                f"  PnL: {sign}{pnl:.2f}% | Trades: {r['trades']} | "
                f"WR: {r['win_rate']:.1f}% | PF: {r['profit_factor']:.2f} | "
                f"DD: {r['max_dd_pct']:.1f}% | Sharpe: {r['sharpe']:.2f}"
            )
        print()

    if not results:
        print("No results!")
        return 1

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY — {len(results)} assets")
    print(f"{'='*70}")
    print(f"{'Symbol':<12} {'PnL%':>8} {'Equity':>10} {'Trades':>7} {'WR%':>6} "
          f"{'PF':>7} {'DD%':>7} {'Sharpe':>8} {'L/S':>6}")
    print("-" * 85)

    total_equity = 0
    total_pnl = 0
    for r in sorted(results, key=lambda x: x["pnl_pct"], reverse=True):
        pnl = r["pnl_pct"]
        total_equity += r["final_equity"]
        total_pnl += pnl
        print(
            f"{r['symbol']:<12} {pnl:>+8.2f} {r['final_equity']:>10.2f} "
            f"{r['trades']:>7} {r['win_rate']:>6.1f} {r['profit_factor']:>7.2f} "
            f"{r['max_dd_pct']:>7.1f} {r['sharpe']:>8.2f} "
            f"{r['long_count']}/{r['short_count']}"
        )

    avg_pnl = total_pnl / len(results)
    print("-" * 85)
    print(
        f"{'AVERAGE':<12} {avg_pnl:>+8.2f} "
        f"{'':>10} {'':>7} {'':>6} {'':>7} {'':>7} {'':>8}"
    )
    print(
        f"{'PORTFOLIO':<12} {'':>8} {total_equity:>10.2f} "
        f"(${total_equity - INITIAL_BALANCE * len(results):>+.2f})"
    )
    print(f"\nInitial portfolio: ${INITIAL_BALANCE * len(results):.2f}")
    print(f"Final portfolio:   ${total_equity:.2f}")
    portfolio_pnl = (total_equity / (INITIAL_BALANCE * len(results)) - 1) * 100
    print(f"Portfolio return:  {portfolio_pnl:+.2f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
