"""Command-line interface for AlphaCluster."""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="alphacluster",
        description=("AlphaZero-inspired deep RL for crypto perpetual contract trading"),
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- download-data ---
    sub_download = subparsers.add_parser("download-data", help="Download OHLCV data from Binance")
    sub_download.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Trading pair (default: BTCUSDT)",
    )
    sub_download.add_argument(
        "--interval",
        default="5m",
        help="Candle interval (default: 5m)",
    )
    sub_download.add_argument(
        "--days",
        type=int,
        default=365,
        help="Days of history (default: 365)",
    )

    # --- train ---
    sub_train = subparsers.add_parser("train", help="Train an RL agent")
    sub_train.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps",
    )
    sub_train.add_argument(
        "--tournament",
        action="store_true",
        help="Enable tournament callback during training",
    )
    sub_train.add_argument(
        "--simple-actions",
        action="store_true",
        help="Use simplified 3-action space",
    )

    # --- evaluate ---
    sub_eval = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    sub_eval.add_argument(
        "--model",
        type=str,
        default="champion",
        help="Model to evaluate: 'champion' or generation number",
    )
    sub_eval.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of evaluation episodes (default: 1)",
    )
    sub_eval.add_argument(
        "--save-charts",
        action="store_true",
        help="Save charts to reports/ directory",
    )
    sub_eval.add_argument(
        "--simple-actions",
        action="store_true",
        help="Use simplified 3-action space",
    )

    # --- tournament ---
    sub_tourn = subparsers.add_parser(
        "tournament",
        help="Run ELO tournament between saved model generations",
    )
    sub_tourn.add_argument(
        "--gen-a",
        type=int,
        required=True,
        help="First generation number",
    )
    sub_tourn.add_argument(
        "--gen-b",
        type=int,
        required=True,
        help="Second generation number",
    )
    sub_tourn.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes per match (default: 10)",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "download-data":
        return _cmd_download(args)
    elif args.command == "train":
        return _cmd_train(args)
    elif args.command == "evaluate":
        return _cmd_evaluate(args)
    elif args.command == "tournament":
        return _cmd_tournament(args)

    print(f"Unknown command: {args.command}")
    return 1


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def _cmd_download(args: argparse.Namespace) -> int:
    """Run the data download pipeline."""
    from scripts.download_data import main as download_main

    download_argv = [
        "--symbol",
        args.symbol,
        "--interval",
        args.interval,
        "--start",
        _days_ago_iso(args.days),
    ]
    return download_main(download_argv)


def _cmd_train(args: argparse.Namespace) -> int:
    """Run the training pipeline."""
    from scripts.train import main as train_main

    train_argv = ["--timesteps", str(args.timesteps)]
    if args.tournament:
        train_argv.append("--tournament")
    if args.simple_actions:
        train_argv.append("--simple-actions")
    return train_main(train_argv)


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """Run the evaluation pipeline."""
    from scripts.evaluate import main as evaluate_main

    eval_argv = ["--model", args.model, "--episodes", str(args.episodes)]
    if args.save_charts:
        eval_argv.append("--save-charts")
    if args.simple_actions:
        eval_argv.append("--simple-actions")
    return evaluate_main(eval_argv)


def _cmd_tournament(args: argparse.Namespace) -> int:
    """Run a tournament between two saved model generations."""
    from alphacluster.config import WINDOW_SIZE
    from alphacluster.data.storage import DEFAULT_KLINES_PATH
    from alphacluster.env.trading_env import TradingEnv
    from alphacluster.tournament.arena import run_tournament
    from alphacluster.tournament.elo import EloRating
    from alphacluster.tournament.versioning import (
        load_generation,
        set_champion,
    )

    # Load evaluation data (last 20% of available data)
    if not DEFAULT_KLINES_PATH.exists():
        print(f"No data found at {DEFAULT_KLINES_PATH}. Run 'alphacluster download-data' first.")
        return 1

    import pandas as pd

    df = pd.read_parquet(DEFAULT_KLINES_PATH, engine="pyarrow")
    test_start = int(len(df) * 0.8)
    test_df = df.iloc[test_start:].reset_index(drop=True)

    if len(test_df) < WINDOW_SIZE + 100:
        print("Insufficient data for tournament evaluation.")
        return 1

    env = TradingEnv(
        df=test_df,
        episode_length=min(len(test_df) - WINDOW_SIZE - 1, 2016),
    )

    # Load both generations
    try:
        model_a, meta_a = load_generation(args.gen_a, env=env)
        model_b, meta_b = load_generation(args.gen_b, env=env)
    except Exception as exc:
        print(f"Failed to load models: {exc}")
        return 1

    print(f"Tournament: gen_{args.gen_a} vs gen_{args.gen_b}")
    print(f"  Episodes: {args.episodes}")

    elo = EloRating()
    result = run_tournament(
        candidate=model_a,
        champion=model_b,
        env=env,
        candidate_id=f"gen_{args.gen_a}",
        champion_id=f"gen_{args.gen_b}",
        n_episodes=args.episodes,
        elo=elo,
    )

    print(f"\nResult: winner = {result.winner}")
    print(
        f"  gen_{args.gen_a}: {result.candidate_wins} wins "
        f"(ELO {result.elo_before[0]:.0f} -> {result.elo_after[0]:.0f})"
    )
    print(
        f"  gen_{args.gen_b}: {result.champion_wins} wins "
        f"(ELO {result.elo_before[1]:.0f} -> {result.elo_after[1]:.0f})"
    )
    print(f"  Draws: {result.draws}")

    if result.candidate_promoted:
        set_champion(args.gen_a)
        print(f"\nGeneration {args.gen_a} promoted to champion!")

    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _days_ago_iso(days: int) -> str:
    """Return an ISO date string for *days* ago from today."""
    from datetime import datetime, timedelta, timezone

    dt = datetime.now(tz=timezone.utc) - timedelta(days=days)
    return dt.strftime("%Y-%m-%d")


if __name__ == "__main__":
    sys.exit(main())
