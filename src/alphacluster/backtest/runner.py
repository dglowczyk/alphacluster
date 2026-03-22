"""Backtesting runner: evaluate a trained agent on environment data.

Runs the agent deterministically through one or more episodes and
collects trade logs, equity curves, and per-episode summary statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest output data.

    Attributes
    ----------
    trade_log:
        List of dicts recording every trade (open + close paired).
        Each entry has: step, action, direction, size, leverage,
        entry_price, exit_price, pnl, fee, balance.
    equity_curve:
        List of per-episode equity curves. Each inner list contains equity
        values (balance + unrealized PnL) at each step within that episode.
    episode_stats:
        List of per-episode summary dicts.
    """

    trade_log: list[dict[str, Any]] = field(default_factory=list)
    equity_curve: list[list[float]] = field(default_factory=list)
    episode_stats: list[dict[str, Any]] = field(default_factory=list)


def run_backtest(
    model: Any,
    env: Any,
    n_episodes: int = 1,
    seed: int | None = None,
) -> BacktestResult:
    """Run the agent deterministically on the environment.

    Parameters
    ----------
    model:
        A trained SB3 model with a ``predict(obs, deterministic)`` method.
    env:
        A TradingEnv instance.
    n_episodes:
        Number of episodes to run.
    seed:
        Base random seed.  Episode *i* uses ``seed + i`` for reproducibility.

    Returns
    -------
    BacktestResult
        Aggregated results across all episodes.
    """
    result = BacktestResult()

    for ep in range(n_episodes):
        ep_seed = seed + ep if seed is not None else None
        obs, info = env.reset(seed=ep_seed)
        episode_equity: list[float] = [env.account.equity]
        episode_trades: list[dict[str, Any]] = []
        episode_reward = 0.0
        step = 0
        done = False

        # Tracking for extended metrics
        flat_steps = 0
        position_steps = 0
        winning_streak = 0
        losing_streak = 0
        max_winning_streak = 0
        max_losing_streak = 0

        # Track open-trade state for pairing open/close entries
        open_trade: dict[str, Any] | None = None
        prev_n_trades = len(env.account.trade_history)

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1

            # Track flat/position time
            if env.account.position_side == "flat":
                flat_steps += 1
            else:
                position_steps += 1

            # Record equity at this step
            episode_equity.append(env.account.equity)

            # Check for new trade history entries
            trade_history = env.account.trade_history
            while prev_n_trades < len(trade_history):
                th_entry = trade_history[prev_n_trades]
                prev_n_trades += 1

                if th_entry["action"] == "open":
                    open_trade = th_entry.copy()
                    open_trade["open_step"] = step
                elif th_entry["action"] == "close" and open_trade is not None:
                    trade_record = {
                        "step": open_trade["open_step"],
                        "close_step": step,
                        "action": "round_trip",
                        "direction": th_entry["side"],
                        "size": th_entry["size"],
                        "leverage": open_trade.get("leverage", 1),
                        "entry_price": th_entry.get("entry_price", open_trade.get("price", 0.0)),
                        "exit_price": th_entry.get("exit_price", 0.0),
                        "pnl": th_entry.get("pnl", 0.0),
                        "fee": open_trade.get("fee", 0.0) + th_entry.get("fee", 0.0),
                        "balance": env.account.balance,
                    }
                    episode_trades.append(trade_record)

                    # Streak tracking
                    if trade_record["pnl"] > 0:
                        winning_streak += 1
                        max_winning_streak = max(max_winning_streak, winning_streak)
                        losing_streak = 0
                    else:
                        losing_streak += 1
                        max_losing_streak = max(max_losing_streak, losing_streak)
                        winning_streak = 0

                    open_trade = None
                elif th_entry["action"] == "close":
                    # Close without a tracked open (e.g., leftover from reset)
                    trade_record = {
                        "step": step,
                        "close_step": step,
                        "action": "close_only",
                        "direction": th_entry["side"],
                        "size": th_entry["size"],
                        "leverage": 1,
                        "entry_price": th_entry.get("entry_price", 0.0),
                        "exit_price": th_entry.get("exit_price", 0.0),
                        "pnl": th_entry.get("pnl", 0.0),
                        "fee": th_entry.get("fee", 0.0),
                        "balance": env.account.balance,
                    }
                    episode_trades.append(trade_record)

        # Force-close any open position at episode end so PnL is realized
        if env.account.position_side != "flat":
            last_price = float(env.unwrapped._close[env.unwrapped._current_idx])
            rpnl, fee = env.account.close_position(last_price)

            # Record the forced close in trade log
            trade_history = env.account.trade_history
            while prev_n_trades < len(trade_history):
                th_entry = trade_history[prev_n_trades]
                prev_n_trades += 1

                if th_entry["action"] == "close" and open_trade is not None:
                    trade_record = {
                        "step": open_trade["open_step"],
                        "close_step": step,
                        "action": "round_trip",
                        "direction": th_entry["side"],
                        "size": th_entry["size"],
                        "leverage": open_trade.get("leverage", 1),
                        "entry_price": th_entry.get("entry_price", open_trade.get("price", 0.0)),
                        "exit_price": th_entry.get("exit_price", 0.0),
                        "pnl": th_entry.get("pnl", 0.0),
                        "fee": open_trade.get("fee", 0.0) + th_entry.get("fee", 0.0),
                        "balance": env.account.balance,
                    }
                    episode_trades.append(trade_record)
                    open_trade = None
                elif th_entry["action"] == "close":
                    trade_record = {
                        "step": step,
                        "close_step": step,
                        "action": "force_close",
                        "direction": th_entry["side"],
                        "size": th_entry["size"],
                        "leverage": 1,
                        "entry_price": th_entry.get("entry_price", 0.0),
                        "exit_price": th_entry.get("exit_price", 0.0),
                        "pnl": th_entry.get("pnl", 0.0),
                        "fee": th_entry.get("fee", 0.0),
                        "balance": env.account.balance,
                    }
                    episode_trades.append(trade_record)

            # Update equity curve with realized close
            episode_equity[-1] = env.account.equity

        # Episode stats
        initial_balance = env.initial_balance
        final_equity = env.account.equity
        total_steps = flat_steps + position_steps
        ep_stat = {
            "episode": ep,
            "initial_balance": initial_balance,
            "final_equity": final_equity,
            "total_return_pct": (final_equity - initial_balance) / initial_balance * 100.0,
            "total_reward": episode_reward,
            "n_trades": len(episode_trades),
            "n_steps": step,
            "max_equity": max(episode_equity) if episode_equity else initial_balance,
            "min_equity": min(episode_equity) if episode_equity else initial_balance,
            "flat_steps": flat_steps,
            "position_steps": position_steps,
            "flat_pct": flat_steps / max(total_steps, 1) * 100.0,
            "max_winning_streak": max_winning_streak,
            "max_losing_streak": max_losing_streak,
        }
        result.episode_stats.append(ep_stat)
        result.trade_log.extend(episode_trades)
        result.equity_curve.append(episode_equity)

        logger.info(
            "Episode %d/%d: equity %.2f -> %.2f (%.2f%%), %d trades, %d steps, %.1f%% flat",
            ep + 1,
            n_episodes,
            initial_balance,
            final_equity,
            ep_stat["total_return_pct"],
            len(episode_trades),
            step,
            ep_stat["flat_pct"],
        )

    return result
