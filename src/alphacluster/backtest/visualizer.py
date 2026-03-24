"""Matplotlib visualizations for backtest results.

All functions use the non-interactive ``Agg`` backend so they work in
headless environments (CI, SSH, containers).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from alphacluster.backtest.runner import BacktestResult  # noqa: E402
from alphacluster.config import MODEL_VERSION  # noqa: E402

logger = logging.getLogger(__name__)


def plot_equity_curve(
    result: BacktestResult,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot the equity curve with drawdown shading.

    Parameters
    ----------
    result:
        Backtest result containing ``equity_curve``.
    ax:
        Optional matplotlib axes to draw on.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    # Plot each episode's equity curve separately with vertical separators
    offset = 0
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    n_episodes = len(result.equity_curve)

    for i, ep_equity in enumerate(result.equity_curve):
        equity = np.array(ep_equity, dtype=np.float64)
        steps = np.arange(offset, offset + len(equity))
        color = colors[i % len(colors)]

        label = "Equity" if i == 0 else None
        ax.plot(steps, equity, color=color, linewidth=1.0, label=label)

        # Drawdown shading per episode
        if len(equity) > 1:
            peak = np.maximum.accumulate(equity)
            ax.fill_between(
                steps,
                equity,
                peak,
                where=(peak > equity),
                color="salmon",
                alpha=0.3,
                label="Drawdown" if i == 0 else None,
            )

        # Vertical separator between episodes
        if i < n_episodes - 1:
            ax.axvline(x=offset + len(equity) - 1, color="gray", linestyle="--", alpha=0.5)

        offset += len(equity)

    ax.set_title("Equity Curve")
    ax.set_xlabel("Step")
    ax.set_ylabel("Equity (USDT)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    return ax


def plot_trades(
    result: BacktestResult,
    prices: Sequence[float] | np.ndarray,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot price chart with trade entry/exit markers.

    Parameters
    ----------
    result:
        Backtest result containing ``trade_log``.
    prices:
        Price series (typically close prices) aligned with backtest steps.
    ax:
        Optional matplotlib axes.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    prices_arr = np.array(prices, dtype=np.float64)
    steps = np.arange(len(prices_arr))
    ax.plot(steps, prices_arr, color="gray", linewidth=0.8, alpha=0.7, label="Price")

    for trade in result.trade_log:
        entry_step = trade.get("step", 0)
        exit_step = trade.get("close_step", entry_step)
        entry_price = trade.get("entry_price", 0)
        exit_price = trade.get("exit_price", 0)
        direction = trade.get("direction", "")
        pnl = trade.get("pnl", 0)

        # Entry marker
        entry_color = "green" if direction == "long" else "red"
        entry_marker = "^" if direction == "long" else "v"
        if 0 <= entry_step < len(prices_arr):
            ax.scatter(
                entry_step,
                entry_price,
                color=entry_color,
                marker=entry_marker,
                s=60,
                zorder=5,
                edgecolors="black",
                linewidths=0.5,
            )

        # Exit marker
        exit_color = "green" if pnl > 0 else "red"
        if 0 <= exit_step < len(prices_arr):
            ax.scatter(
                exit_step,
                exit_price,
                color=exit_color,
                marker="x",
                s=60,
                zorder=5,
                linewidths=1.5,
            )

    ax.set_title("Price Chart with Trades")
    ax.set_xlabel("Step")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)

    return ax


def plot_action_distribution(
    result: BacktestResult,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot a histogram of trade directions.

    Parameters
    ----------
    result:
        Backtest result containing ``trade_log``.
    ax:
        Optional matplotlib axes.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    directions = [t.get("direction", "unknown") for t in result.trade_log]

    if not directions:
        ax.text(0.5, 0.5, "No trades", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Action Distribution")
        return ax

    labels = sorted(set(directions))
    counts = [directions.count(label) for label in labels]

    colors = {"long": "green", "short": "red", "flat": "gray"}
    bar_colors = [colors.get(label, "steelblue") for label in labels]

    ax.bar(labels, counts, color=bar_colors, edgecolor="black", alpha=0.8)
    ax.set_title("Action Distribution")
    ax.set_xlabel("Direction")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3, axis="y")

    return ax


def plot_elo_history(
    elo_ratings: Sequence[float] | dict[int, float],
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot ELO rating progression across generations.

    Parameters
    ----------
    elo_ratings:
        Either a sequence of ELO values (indexed by generation) or a
        dict mapping generation number to ELO rating.
    ax:
        Optional matplotlib axes.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    if isinstance(elo_ratings, dict):
        gens = sorted(elo_ratings.keys())
        elos = [elo_ratings[g] for g in gens]
    else:
        gens = list(range(len(elo_ratings)))
        elos = list(elo_ratings)

    ax.plot(gens, elos, marker="o", color="darkorange", linewidth=1.5, markersize=6)
    ax.set_title("ELO Rating History")
    ax.set_xlabel("Generation")
    ax.set_ylabel("ELO Rating")
    ax.grid(True, alpha=0.3)

    return ax


def plot_training_rewards(
    rewards: Sequence[float],
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot training reward curve.

    Parameters
    ----------
    rewards:
        Sequence of reward values (e.g., per episode or per update).
    ax:
        Optional matplotlib axes.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    episodes = np.arange(len(rewards))
    ax.plot(episodes, rewards, color="steelblue", linewidth=0.8, alpha=0.6, label="Reward")

    # Smoothed curve (rolling mean)
    if len(rewards) > 10:
        window = max(len(rewards) // 20, 5)
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        offset = (len(rewards) - len(smoothed)) // 2
        ax.plot(
            np.arange(offset, offset + len(smoothed)),
            smoothed,
            color="darkblue",
            linewidth=2.0,
            label=f"Smoothed (window={window})",
        )

    ax.set_title("Training Rewards")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    return ax


def save_report(
    result: BacktestResult,
    metrics: dict[str, Any],
    output_dir: str | Path,
    prices: Sequence[float] | np.ndarray | None = None,
) -> Path:
    """Save all charts to a directory.

    Parameters
    ----------
    result:
        Backtest result.
    metrics:
        Metrics dict from :func:`~alphacluster.backtest.metrics.calculate_metrics`.
    output_dir:
        Directory to write chart images to.
    prices:
        Optional price series for the trade overlay chart.

    Returns
    -------
    Path
        The output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Equity curve
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_equity_curve(result, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "equity_curve.png", dpi=150)
    plt.close(fig)
    logger.info("Saved equity_curve.png")

    # 2. Trades on price (if prices provided)
    if prices is not None and len(prices) > 0:
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_trades(result, prices, ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / "trades.png", dpi=150)
        plt.close(fig)
        logger.info("Saved trades.png")

    # 3. Action distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_action_distribution(result, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "action_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved action_distribution.png")

    # 4. Summary text file
    _save_metrics_text(metrics, output_dir / "metrics.txt")
    logger.info("Saved metrics.txt")

    return output_dir


def _save_metrics_text(metrics: dict[str, Any], path: Path) -> None:
    """Write metrics to a plain text file."""
    lines = [
        "BACKTEST METRICS",
        f"Model Version: {MODEL_VERSION}",
        "=" * 50,
        "",
    ]
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key:30s} {value:>14.4f}")
        else:
            lines.append(f"  {key:30s} {str(value):>14s}")
    lines.append("")
    path.write_text("\n".join(lines))
