"""Tests for the backtest runner (alphacluster.backtest.runner)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphacluster.backtest.runner import run_backtest
from alphacluster.config import WINDOW_SIZE
from alphacluster.env.trading_env import TradingEnv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(n_candles: int = 3200, start_price: float = 50_000.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start="2025-01-01", periods=n_candles, freq="5min", tz="UTC")
    close = start_price + np.cumsum(rng.normal(0, 10, size=n_candles))
    close = np.maximum(close, 100.0)
    high = close + rng.uniform(0, 20, size=n_candles)
    low = close - rng.uniform(0, 20, size=n_candles)
    low = np.maximum(low, 1.0)
    opn = close + rng.normal(0, 5, size=n_candles)
    opn = np.maximum(opn, 1.0)
    volume = rng.uniform(100, 10000, size=n_candles)
    return pd.DataFrame(
        {
            "open_time": timestamps,
            "open": opn,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _make_env(n_candles: int = 3200, episode_length: int = 200) -> TradingEnv:
    df = _make_df(n_candles=n_candles)
    return TradingEnv(
        df=df,
        funding_df=None,
        window_size=WINDOW_SIZE,
        episode_length=episode_length,
        initial_balance=10_000.0,
    )


class _AlwaysLongModel:
    """Fake model that always goes long 50% at 5x leverage."""

    def predict(self, obs, deterministic=False):
        # direction=1 (long), size=2 (50%), leverage=0 (5x)
        return np.array([1, 2, 0]), None


class _OpenAndHoldModel:
    """Fake model that opens long on step 0 and holds forever (never goes flat)."""

    def predict(self, obs, deterministic=False):
        # direction=1 (long), size=2 (50%), leverage=0 (5x)
        return np.array([1, 2, 0]), None


class _AlwaysFlatModel:
    """Fake model that always stays flat."""

    def predict(self, obs, deterministic=False):
        return np.array([0, 0, 0]), None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBacktestForceClose:
    """Positions open at episode end must be force-closed and counted."""

    def test_open_position_at_episode_end_is_counted(self):
        """If the model holds a position to episode end, it should be counted as a trade."""
        env = _make_env(episode_length=50)
        model = _OpenAndHoldModel()
        result = run_backtest(model, env, n_episodes=1)

        # The model opens long on step 1 and holds. At episode end the position
        # should be force-closed. That's 1 round-trip trade.
        assert result.episode_stats[0]["n_trades"] >= 1, (
            "Open position at episode end should be force-closed and counted as a trade"
        )

    def test_force_close_realizes_pnl(self):
        """Force-close at episode end should realize PnL into balance."""
        env = _make_env(episode_length=50)
        model = _OpenAndHoldModel()
        run_backtest(model, env, n_episodes=1)

        # After force-close, the account should be flat with no unrealized PnL
        assert env.account.position_side == "flat"
        assert abs(env.account.unrealized_pnl) < 1e-6

    def test_flat_model_has_zero_trades(self):
        """A model that stays flat should have exactly 0 trades."""
        env = _make_env(episode_length=50)
        model = _AlwaysFlatModel()
        result = run_backtest(model, env, n_episodes=1)

        assert result.episode_stats[0]["n_trades"] == 0
        assert abs(env.account.equity - 10_000.0) < 1e-6

    def test_force_close_trade_in_trade_log(self):
        """Force-closed position should appear in the trade log."""
        env = _make_env(episode_length=50)
        model = _OpenAndHoldModel()
        result = run_backtest(model, env, n_episodes=1)

        assert len(result.trade_log) >= 1
