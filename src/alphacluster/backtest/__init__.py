"""Backtesting package: runner, metrics, and visualization."""

from alphacluster.backtest.metrics import calculate_metrics, print_report
from alphacluster.backtest.runner import BacktestResult, run_backtest
from alphacluster.backtest.visualizer import (
    plot_action_distribution,
    plot_elo_history,
    plot_equity_curve,
    plot_trades,
    plot_training_rewards,
    save_report,
)

__all__ = [
    "BacktestResult",
    "run_backtest",
    "calculate_metrics",
    "print_report",
    "plot_equity_curve",
    "plot_trades",
    "plot_action_distribution",
    "plot_elo_history",
    "plot_training_rewards",
    "save_report",
]
