"""Performance metrics computed from backtest results.

Provides :func:`calculate_metrics` for a full metrics dict and
:func:`print_report` for formatted console output.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from alphacluster.backtest.runner import BacktestResult
from alphacluster.config import MODEL_VERSION


def calculate_metrics(result: BacktestResult) -> dict[str, Any]:
    """Compute comprehensive performance metrics from a backtest result.

    Parameters
    ----------
    result:
        A :class:`BacktestResult` from :func:`run_backtest`.

    Returns
    -------
    dict
        Dictionary containing all computed metrics.
    """
    metrics: dict[str, Any] = {}
    metrics["model_version"] = MODEL_VERSION

    episodes = result.equity_curve  # list[list[float]]
    trades = result.trade_log

    # ── Equity-based metrics ─────────────────────────────────────────
    if episodes:
        initial_equity = episodes[0][0] if episodes[0] else 0.0
        final_equity = episodes[-1][-1] if episodes[-1] else 0.0
    else:
        initial_equity = 0.0
        final_equity = 0.0

    # Total PnL (first episode start to last episode end)
    total_pnl = final_equity - initial_equity
    total_pnl_pct = (total_pnl / initial_equity * 100.0) if initial_equity != 0 else 0.0
    metrics["total_pnl"] = total_pnl
    metrics["total_pnl_pct"] = total_pnl_pct
    metrics["initial_equity"] = initial_equity
    metrics["final_equity"] = final_equity

    # Per-episode returns, Sharpe, Sortino, and max drawdown
    periods_per_year = 288 * 365  # 5-min candles
    ep_sharpes: list[float] = []
    ep_sortinos: list[float] = []
    worst_dd = 0.0
    worst_dd_pct = 0.0

    for ep_equity_list in episodes:
        eq = np.array(ep_equity_list, dtype=np.float64)
        if len(eq) > 1:
            ep_returns = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
            ep_sharpes.append(_sharpe_ratio(ep_returns, periods_per_year))
            ep_sortinos.append(_sortino_ratio(ep_returns, periods_per_year))

            dd, dd_pct = _max_drawdown(eq)
            if dd_pct > worst_dd_pct:
                worst_dd = dd
                worst_dd_pct = dd_pct

    # Average per-episode Sharpe and Sortino
    metrics["sharpe_ratio"] = float(np.mean(ep_sharpes)) if ep_sharpes else 0.0
    metrics["sortino_ratio"] = float(np.mean(ep_sortinos)) if ep_sortinos else 0.0

    # Max drawdown: worst single-episode drawdown
    metrics["max_drawdown"] = worst_dd
    metrics["max_drawdown_pct"] = worst_dd_pct

    # ── Trade-based metrics ──────────────────────────────────────────
    n_trades = len(trades)
    metrics["trade_count"] = n_trades

    if n_trades > 0:
        pnls = np.array([t["pnl"] for t in trades], dtype=np.float64)
        fees = np.array([t["fee"] for t in trades], dtype=np.float64)

        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        metrics["win_rate"] = len(wins) / n_trades * 100.0
        metrics["avg_pnl_per_trade"] = float(np.mean(pnls))
        metrics["total_fees"] = float(np.sum(fees))

        # Profit factor = gross_profit / gross_loss
        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0
        if gross_loss > 0:
            metrics["profit_factor"] = gross_profit / gross_loss
        else:
            metrics["profit_factor"] = float("inf") if gross_profit > 0 else 0.0

        # Average trade duration (in steps)
        durations = []
        for t in trades:
            open_step = t.get("step", 0)
            close_step = t.get("close_step", open_step)
            durations.append(close_step - open_step)
        metrics["avg_trade_duration"] = float(np.mean(durations)) if durations else 0.0

        metrics["avg_win"] = float(np.mean(wins)) if len(wins) > 0 else 0.0
        metrics["avg_loss"] = float(np.mean(losses)) if len(losses) > 0 else 0.0
        metrics["best_trade"] = float(np.max(pnls))
        metrics["worst_trade"] = float(np.min(pnls))
    else:
        metrics["win_rate"] = 0.0
        metrics["avg_pnl_per_trade"] = 0.0
        metrics["total_fees"] = 0.0
        metrics["profit_factor"] = 0.0
        metrics["avg_trade_duration"] = 0.0
        metrics["avg_win"] = 0.0
        metrics["avg_loss"] = 0.0
        metrics["best_trade"] = 0.0
        metrics["worst_trade"] = 0.0

    # ── Direction breakdown (F1) ─────────────────────────────────────
    if n_trades > 0:
        long_trades = [t for t in trades if t.get("direction") == "long"]
        short_trades = [t for t in trades if t.get("direction") == "short"]

        long_count = len(long_trades)
        short_count = len(short_trades)
        metrics["long_count"] = long_count
        metrics["short_count"] = short_count

        long_pnls = np.array([t["pnl"] for t in long_trades], dtype=np.float64)
        short_pnls = np.array([t["pnl"] for t in short_trades], dtype=np.float64)

        metrics["long_win_rate"] = (
            float(np.sum(long_pnls > 0) / long_count * 100.0) if long_count > 0 else 0.0
        )
        metrics["short_win_rate"] = (
            float(np.sum(short_pnls > 0) / short_count * 100.0) if short_count > 0 else 0.0
        )
        metrics["long_avg_pnl"] = float(np.mean(long_pnls)) if long_count > 0 else 0.0
        metrics["short_avg_pnl"] = float(np.mean(short_pnls)) if short_count > 0 else 0.0
        metrics["long_total_pnl"] = float(np.sum(long_pnls)) if long_count > 0 else 0.0
        metrics["short_total_pnl"] = float(np.sum(short_pnls)) if short_count > 0 else 0.0

        # Profit factor per direction
        for prefix, dir_pnls in [("long", long_pnls), ("short", short_pnls)]:
            if len(dir_pnls) > 0:
                gp = float(np.sum(dir_pnls[dir_pnls > 0]))
                gl = float(np.abs(np.sum(dir_pnls[dir_pnls <= 0])))
                if gl > 0:
                    metrics[f"{prefix}_profit_factor"] = gp / gl
                else:
                    metrics[f"{prefix}_profit_factor"] = float("inf") if gp > 0 else 0.0
            else:
                metrics[f"{prefix}_profit_factor"] = 0.0
    else:
        metrics["long_count"] = 0
        metrics["short_count"] = 0
        metrics["long_win_rate"] = 0.0
        metrics["short_win_rate"] = 0.0
        metrics["long_avg_pnl"] = 0.0
        metrics["short_avg_pnl"] = 0.0
        metrics["long_total_pnl"] = 0.0
        metrics["short_total_pnl"] = 0.0
        metrics["long_profit_factor"] = 0.0
        metrics["short_profit_factor"] = 0.0

    # ── Leverage distribution (F2) ────────────────────────────────
    if n_trades > 0:
        lev_groups: dict[int, list[float]] = defaultdict(list)
        for t in trades:
            lev_groups[int(t.get("leverage", 1))].append(t["pnl"])

        leverage_distribution: dict[int, int] = {}
        leverage_win_rates: dict[int, float] = {}
        leverage_avg_pnl: dict[int, float] = {}

        for lev in sorted(lev_groups):
            lev_pnls = np.array(lev_groups[lev], dtype=np.float64)
            leverage_distribution[lev] = len(lev_pnls)
            leverage_win_rates[lev] = float(np.sum(lev_pnls > 0) / len(lev_pnls) * 100.0)
            leverage_avg_pnl[lev] = float(np.mean(lev_pnls))

        metrics["leverage_distribution"] = leverage_distribution
        metrics["leverage_win_rates"] = leverage_win_rates
        metrics["leverage_avg_pnl"] = leverage_avg_pnl
    else:
        metrics["leverage_distribution"] = {}
        metrics["leverage_win_rates"] = {}
        metrics["leverage_avg_pnl"] = {}

    # ── Trade quality metrics (F3) ─────────────────────────────────
    if n_trades > 0:
        win_durations = []
        loss_durations = []
        for t in trades:
            open_step = t.get("step", 0)
            close_step = t.get("close_step", open_step)
            dur = close_step - open_step
            if t["pnl"] > 0:
                win_durations.append(dur)
            else:
                loss_durations.append(dur)

        metrics["avg_win_duration"] = float(np.mean(win_durations)) if win_durations else 0.0
        metrics["avg_loss_duration"] = float(np.mean(loss_durations)) if loss_durations else 0.0

        avg_win = metrics["avg_win"]
        avg_loss = metrics["avg_loss"]
        metrics["risk_reward_ratio"] = (
            abs(avg_win / avg_loss) if avg_loss != 0 else (float("inf") if avg_win > 0 else 0.0)
        )

        total_fees = metrics["total_fees"]
        total_trade_pnl = float(np.sum(pnls))
        metrics["fee_to_pnl_ratio"] = (
            total_fees / abs(total_trade_pnl) * 100.0 if abs(total_trade_pnl) > 1e-12 else 0.0
        )

        win_rate_frac = metrics["win_rate"] / 100.0
        metrics["expectancy"] = (win_rate_frac * avg_win) + ((1 - win_rate_frac) * avg_loss)
    else:
        metrics["avg_win_duration"] = 0.0
        metrics["avg_loss_duration"] = 0.0
        metrics["risk_reward_ratio"] = 0.0
        metrics["fee_to_pnl_ratio"] = 0.0
        metrics["expectancy"] = 0.0

    # ── Episode stats ────────────────────────────────────────────────
    n_eps = len(result.episode_stats)
    metrics["n_episodes"] = n_eps
    if result.episode_stats:
        returns_pct = [s["total_return_pct"] for s in result.episode_stats]
        metrics["avg_episode_return_pct"] = float(np.mean(returns_pct))

        # Extended metrics from episode stats
        trades_per_ep = [s["n_trades"] for s in result.episode_stats]
        metrics["avg_trades_per_episode"] = float(np.mean(trades_per_ep))

        flat_pcts = [s.get("flat_pct", 0.0) for s in result.episode_stats]
        metrics["avg_flat_pct"] = float(np.mean(flat_pcts))

        pos_steps = [s.get("position_steps", 0) for s in result.episode_stats]
        n_trades_per_ep = [s["n_trades"] for s in result.episode_stats]
        avg_time_in_pos = []
        for ps, nt in zip(pos_steps, n_trades_per_ep):
            if nt > 0:
                avg_time_in_pos.append(ps / nt)
        metrics["avg_time_in_position"] = float(np.mean(avg_time_in_pos)) if avg_time_in_pos else 0

        win_streaks = [s.get("max_winning_streak", 0) for s in result.episode_stats]
        lose_streaks = [s.get("max_losing_streak", 0) for s in result.episode_stats]
        metrics["max_winning_streak"] = int(max(win_streaks)) if win_streaks else 0
        metrics["max_losing_streak"] = int(max(lose_streaks)) if lose_streaks else 0
    else:
        metrics["avg_episode_return_pct"] = 0.0
        metrics["avg_trades_per_episode"] = 0.0
        metrics["avg_flat_pct"] = 0.0
        metrics["avg_time_in_position"] = 0.0
        metrics["max_winning_streak"] = 0
        metrics["max_losing_streak"] = 0

    return metrics


def print_report(metrics: dict[str, Any]) -> None:
    """Print a formatted performance report to the console.

    Parameters
    ----------
    metrics:
        Dictionary from :func:`calculate_metrics`.
    """
    sep = "=" * 60
    print(sep)
    print("  BACKTEST PERFORMANCE REPORT")
    print(sep)

    print(f"\n  Initial Equity:      {metrics.get('initial_equity', 0):>14,.2f}")
    print(f"  Final Equity:        {metrics.get('final_equity', 0):>14,.2f}")
    pnl = metrics.get("total_pnl", 0)
    pnl_pct = metrics.get("total_pnl_pct", 0)
    print(f"  Total PnL:           {pnl:>14,.2f}  ({pnl_pct:>+.2f}%)")

    print(f"\n  Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):>14.4f}")
    print(f"  Sortino Ratio:       {metrics.get('sortino_ratio', 0):>14.4f}")
    dd = metrics.get("max_drawdown", 0)
    dd_pct = metrics.get("max_drawdown_pct", 0)
    print(f"  Max Drawdown:        {dd:>14,.2f}  ({dd_pct:>+.2f}%)")

    print(f"\n  Trade Count:         {metrics.get('trade_count', 0):>14d}")
    print(f"  Win Rate:            {metrics.get('win_rate', 0):>13.2f}%")
    print(f"  Profit Factor:       {_fmt_float(metrics.get('profit_factor', 0)):>14s}")
    print(f"  Avg PnL/Trade:       {metrics.get('avg_pnl_per_trade', 0):>14,.2f}")
    print(f"  Avg Trade Duration:  {metrics.get('avg_trade_duration', 0):>14.1f} steps")

    print(f"\n  Avg Win:             {metrics.get('avg_win', 0):>14,.2f}")
    print(f"  Avg Loss:            {metrics.get('avg_loss', 0):>14,.2f}")
    print(f"  Best Trade:          {metrics.get('best_trade', 0):>14,.2f}")
    print(f"  Worst Trade:         {metrics.get('worst_trade', 0):>14,.2f}")
    print(f"  Total Fees:          {metrics.get('total_fees', 0):>14,.2f}")

    n_eps = metrics.get("n_episodes", 0)
    if n_eps > 0:
        print(f"\n  Episodes:            {n_eps:>14d}")
        print(f"  Avg Episode Return:  {metrics.get('avg_episode_return_pct', 0):>13.2f}%")
        print(f"  Avg Trades/Episode:  {metrics.get('avg_trades_per_episode', 0):>14.1f}")
        print(f"  Avg Flat Time:       {metrics.get('avg_flat_pct', 0):>13.1f}%")
        print(f"  Avg Time in Pos:     {metrics.get('avg_time_in_position', 0):>14.1f} steps")
        print(f"  Max Win Streak:      {metrics.get('max_winning_streak', 0):>14d}")
        print(f"  Max Lose Streak:     {metrics.get('max_losing_streak', 0):>14d}")

    # ── Direction breakdown ────────────────────────────────────────
    long_count = metrics.get("long_count", 0)
    short_count = metrics.get("short_count", 0)
    if long_count > 0 or short_count > 0:
        print(f"\n{'── DIRECTION BREAKDOWN ':─<60}")
        print(
            f"  Long Trades:         {long_count:>10d}"
            f"  (win rate: {metrics.get('long_win_rate', 0):.1f}%)"
        )
        print(
            f"  Short Trades:        {short_count:>10d}"
            f"  (win rate: {metrics.get('short_win_rate', 0):.1f}%)"
        )
        print(f"  Long Avg PnL:        {metrics.get('long_avg_pnl', 0):>14,.2f}")
        print(f"  Short Avg PnL:       {metrics.get('short_avg_pnl', 0):>14,.2f}")
        print(f"  Long Profit Factor:  {_fmt_float(metrics.get('long_profit_factor', 0)):>14s}")
        print(f"  Short Profit Factor: {_fmt_float(metrics.get('short_profit_factor', 0)):>14s}")

    # ── Leverage distribution ─────────────────────────────────────
    lev_dist = metrics.get("leverage_distribution", {})
    if lev_dist:
        print(f"\n{'── LEVERAGE DISTRIBUTION ':─<60}")
        total_trades = sum(lev_dist.values())
        lev_win_rates = metrics.get("leverage_win_rates", {})
        lev_avg_pnl = metrics.get("leverage_avg_pnl", {})
        for lev in sorted(lev_dist):
            count = lev_dist[lev]
            pct = count / total_trades * 100.0 if total_trades > 0 else 0.0
            wr = lev_win_rates.get(lev, 0.0)
            avg = lev_avg_pnl.get(lev, 0.0)
            print(
                f"  {lev}x:{count:>7d} trades ({pct:>4.1f}%)"
                f"  win rate: {wr:>5.1f}%"
                f"  avg PnL: {avg:>8.2f}"
            )

    # ── Trade quality ──────────────────────────────────────────────
    trade_count = metrics.get("trade_count", 0)
    if trade_count > 0:
        print(f"\n{'── TRADE QUALITY ':─<60}")
        print(f"  Risk/Reward Ratio:   {_fmt_float(metrics.get('risk_reward_ratio', 0)):>14s}")
        print(f"  Avg Win Duration:    {metrics.get('avg_win_duration', 0):>14.1f} steps")
        print(f"  Avg Loss Duration:   {metrics.get('avg_loss_duration', 0):>14.1f} steps")
        print(f"  Fee/PnL Ratio:       {metrics.get('fee_to_pnl_ratio', 0):>13.1f}%")
        print(f"  Expectancy/Trade:    {metrics.get('expectancy', 0):>14,.2f}")

    print(sep)


# ── Private helpers ──────────────────────────────────────────────────────


def _sharpe_ratio(returns: np.ndarray, periods_per_year: int) -> float:
    """Annualized Sharpe ratio (risk-free rate = 0)."""
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std == 0:
        return 0.0
    mean_ret = float(np.mean(returns))
    return mean_ret / std * math.sqrt(periods_per_year)


def _sortino_ratio(returns: np.ndarray, periods_per_year: int) -> float:
    """Annualized Sortino ratio (target return = 0)."""
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0:
        # No downside — if mean return > 0, return inf; else 0
        mean_ret = float(np.mean(returns))
        return float("inf") if mean_ret > 0 else 0.0
    downside_std = float(np.std(downside, ddof=1))
    if downside_std == 0:
        return 0.0
    mean_ret = float(np.mean(returns))
    return mean_ret / downside_std * math.sqrt(periods_per_year)


def _max_drawdown(equity: np.ndarray) -> tuple[float, float]:
    """Compute maximum drawdown (absolute and percentage).

    Returns
    -------
    tuple[float, float]
        (max_drawdown_absolute, max_drawdown_pct)
    """
    if len(equity) < 2:
        return 0.0, 0.0

    peak = np.maximum.accumulate(equity)
    drawdowns = peak - equity
    max_dd = float(np.max(drawdowns))

    # Percentage relative to peak at time of drawdown
    with np.errstate(divide="ignore", invalid="ignore"):
        dd_pct = np.where(peak > 0, drawdowns / peak * 100.0, 0.0)
    max_dd_pct = float(np.max(dd_pct))

    return max_dd, max_dd_pct


def _fmt_float(value: float) -> str:
    """Format a float, handling inf gracefully."""
    if math.isinf(value):
        return "inf"
    return f"{value:.4f}"
