#!/usr/bin/env python3
"""Evaluate a trained RL model on held-out data and produce performance metrics.

Usage:
    python scripts/evaluate.py [--model champion] [--symbol BTCUSDT] [--episodes 1] [--save-charts]

Examples:
    # Evaluate the current champion model
    python scripts/evaluate.py --model champion

    # Evaluate a specific generation
    python scripts/evaluate.py --model 3 --episodes 5

    # Evaluate with deterministic seeding
    python scripts/evaluate.py --model champion --episodes 10 --seed 42

    # Evaluate and save charts
    python scripts/evaluate.py --model champion --save-charts
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is importable when running as a script
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from alphacluster.agent.trainer import load_agent
from alphacluster.backtest.metrics import calculate_metrics, print_report
from alphacluster.backtest.runner import run_backtest
from alphacluster.backtest.visualizer import save_report
from alphacluster.config import DATA_DIR, MODEL_VERSION, MODELS_DIR, PROJECT_ROOT, WINDOW_SIZE
from alphacluster.data.storage import load_from_parquet
from alphacluster.env.trading_env import TradingEnv
from alphacluster.tournament.versioning import get_champion, load_generation

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained AlphaCluster model",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="champion",
        help=(
            "Model to evaluate: 'champion' for the current champion, "
            "or an integer generation number (default: champion)"
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a saved model .pt file (overrides --model)",
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
        default=1,
        help="Number of evaluation episodes (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible evaluation (default: 42)",
    )
    parser.add_argument(
        "--save-charts",
        action="store_true",
        help="Save charts to reports/ directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for charts (default: reports/)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--simple-actions",
        action="store_true",
        help="Use simplified 3-action space (must match training mode)",
    )
    return parser.parse_args(argv)


def _resolve_model_id(model_arg: str) -> tuple[str, int | None]:
    """Parse the --model argument into a type and generation number.

    Returns
    -------
    tuple[str, int | None]
        ("champion", gen_number) or ("generation", gen_number).
    """
    if model_arg.lower() == "champion":
        gen = get_champion()
        if gen is None:
            return "champion", None
        return "champion", gen

    try:
        gen = int(model_arg)
        return "generation", gen
    except ValueError:
        # Treat as a path
        return "path", None


def _load_data(symbol: str) -> tuple:
    """Load OHLCV and (optional) funding data for the given symbol.

    Returns
    -------
    tuple[DataFrame, DataFrame | None]
    """
    symbol_lower = symbol.lower()
    klines_path = DATA_DIR / f"{symbol_lower}_5m.parquet"
    funding_path = DATA_DIR / f"{symbol_lower}_funding.parquet"

    if not klines_path.exists():
        raise FileNotFoundError(
            f"No OHLCV data found at {klines_path}. "
            f"Run: python scripts/download_data.py --symbol {symbol}"
        )

    df = load_from_parquet(klines_path)

    funding_df = None
    if funding_path.exists():
        funding_df = load_from_parquet(funding_path)

    return df, funding_df


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ── Resolve model ────────────────────────────────────────────────
    use_model_path = args.model_path is not None
    gen_number = None

    if not use_model_path:
        model_type, gen_number = _resolve_model_id(args.model)

        if model_type in ("champion", "generation") and gen_number is None:
            if model_type == "champion":
                print("No champion has been set yet. Train and run a tournament first.")
            else:
                print(f"Could not parse model argument: {args.model}")
            return 1

        print(f"Loading model: {model_type} (generation {gen_number})")
    else:
        model_path = Path(args.model_path)
        if not model_path.exists() and not model_path.with_suffix(".pt").exists():
            print(f"Model file not found: {model_path}")
            return 1
        print(f"Loading model from: {model_path}")

    # ── Load data ────────────────────────────────────────────────────
    try:
        df, funding_df = _load_data(args.symbol)
    except FileNotFoundError as exc:
        print(f"Data error: {exc}")
        return 1

    print(f"Loaded {len(df)} candles for {args.symbol}")

    # Use the last 20% of data as test split
    test_start = int(len(df) * 0.8)
    test_df = df.iloc[test_start:].reset_index(drop=True)

    if len(test_df) < WINDOW_SIZE + 100:
        print(
            f"Insufficient test data: {len(test_df)} candles "
            f"(need at least {WINDOW_SIZE + 100}). Download more data."
        )
        return 1

    print(f"Using {len(test_df)} candles for test evaluation (last 20%)")
    print(f"Seed: {args.seed}")

    # ── Create environment ───────────────────────────────────────────
    env = TradingEnv(
        df=test_df,
        funding_df=funding_df,
        episode_length=min(len(test_df) - WINDOW_SIZE - 1, 2016),
        simple_actions=args.simple_actions,
    )

    # ── Load model ───────────────────────────────────────────────────
    if use_model_path:
        try:
            model = load_agent(args.model_path, env=env)
        except Exception as exc:
            print(f"Failed to load model from {args.model_path}: {exc}")
            return 1
    else:
        try:
            model, metadata = load_generation(gen_number, env=env)
        except Exception as exc:
            print(f"Failed to load model generation {gen_number}: {exc}")
            return 1

        if metadata:
            print(f"Model metadata: {metadata}")

    # ── Run backtest ─────────────────────────────────────────────────
    print(f"Model version: {MODEL_VERSION}")
    print(f"\nRunning backtest ({args.episodes} episode(s))...")
    result = run_backtest(model, env, n_episodes=args.episodes, seed=args.seed)

    # ── Calculate and print metrics ──────────────────────────────────
    metrics = calculate_metrics(result)
    print()
    print_report(metrics)

    # ── Save charts ──────────────────────────────────────────────────
    if args.save_charts:
        output_dir = args.output_dir or str(PROJECT_ROOT / "reports")
        output_dir = Path(output_dir)

        # Provide price data for the trade overlay chart
        prices = test_df["close"].values.tolist()

        save_report(result, metrics, output_dir, prices=prices)
        print(f"\nCharts saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
