"""Gymnasium-compatible trading environment for crypto perpetual contracts.

The environment simulates trading on 5-minute OHLCV candle data with discrete
actions covering direction, position size, and leverage.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from alphacluster.config import (
    EPISODE_LENGTH,
    LEVERAGE_OPTIONS,
    N_DIRECTIONS,
    N_LEVERAGE_LEVELS,
    N_POSITION_SIZES,
    POSITION_SIZE_OPTIONS,
    WINDOW_SIZE,
)
from alphacluster.env.account import Account

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_FUNDING_HOURS = (0, 8, 16)  # UTC hours at which funding is applied
_DRAWDOWN_PENALTY_COEFF = 0.1  # coefficient for drawdown penalty in reward


class TradingEnv(gym.Env):
    """Perpetual-contract trading environment.

    Parameters
    ----------
    df:
        DataFrame with columns ``open_time, open, high, low, close, volume``.
        ``open_time`` must be a timezone-aware or naive UTC datetime / int
        timestamp in **milliseconds**.
    funding_df:
        Optional DataFrame with columns ``time, funding_rate``.  ``time``
        should be a datetime or int timestamp (ms).  If *None*, funding is
        ignored.
    window_size:
        Number of past candles in each observation.
    episode_length:
        Maximum number of steps per episode.
    initial_balance:
        Starting USDT balance.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        funding_df: pd.DataFrame | None = None,
        window_size: int = WINDOW_SIZE,
        episode_length: int = EPISODE_LENGTH,
        initial_balance: float = 10_000.0,
    ) -> None:
        super().__init__()

        # ── Store data ───────────────────────────────────────────────────
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.episode_length = episode_length
        self.initial_balance = initial_balance

        # Ensure open_time is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df["open_time"]):
            self.df["open_time"] = pd.to_datetime(self.df["open_time"], unit="ms", utc=True)

        # Pre-compute OHLCV numpy arrays for speed
        self._open = self.df["open"].to_numpy(dtype=np.float64)
        self._high = self.df["high"].to_numpy(dtype=np.float64)
        self._low = self.df["low"].to_numpy(dtype=np.float64)
        self._close = self.df["close"].to_numpy(dtype=np.float64)
        self._volume = self.df["volume"].to_numpy(dtype=np.float64)
        self._timestamps = self.df["open_time"].to_numpy()

        # Funding lookup
        self._funding_map: dict[int, float] = {}
        if funding_df is not None:
            fdf = funding_df.copy()
            # Support both "time" and "funding_time" column names
            time_col = "funding_time" if "funding_time" in fdf.columns else "time"
            if not pd.api.types.is_datetime64_any_dtype(fdf[time_col]):
                fdf[time_col] = pd.to_datetime(fdf[time_col], unit="ms", utc=True)
            # Map: hour-rounded timestamp -> rate
            for _, row in fdf.iterrows():
                ts = int(pd.Timestamp(row[time_col]).timestamp())
                self._funding_map[ts] = float(row["funding_rate"])

        # ── Spaces ───────────────────────────────────────────────────────
        self.observation_space = spaces.Dict(
            {
                "market": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.window_size, 5),
                    dtype=np.float32,
                ),
                "account": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(7,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.MultiDiscrete(
            [N_DIRECTIONS, N_POSITION_SIZES, N_LEVERAGE_LEVELS]
        )

        # ── Internal state (set properly in reset) ───────────────────────
        self.account = Account(initial_balance=self.initial_balance)
        self._start_idx: int = 0
        self._current_idx: int = 0
        self._step_count: int = 0
        self._prev_equity: float = self.initial_balance

    # ── Gymnasium API ────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)

        # Pick a random start ensuring room for window + episode
        min_start = self.window_size
        max_start = len(self.df) - self.episode_length - 1
        if max_start < min_start:
            max_start = min_start  # dataset is small — will overlap

        self._start_idx = self.np_random.integers(min_start, max_start + 1)
        self._current_idx = self._start_idx
        self._step_count = 0

        self.account.reset()
        self._prev_equity = self.initial_balance

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray | list | tuple
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        direction_idx, size_idx, leverage_idx = int(action[0]), int(action[1]), int(action[2])

        direction = direction_idx  # 0=flat, 1=long, 2=short

        # Collapse flat actions: when direction=flat, size and leverage are irrelevant.
        # This reduces 12 equivalent "flat" actions to a single semantic action.
        if direction == 0:
            size_pct = 0.0
            leverage = 1
        else:
            size_pct = POSITION_SIZE_OPTIONS[size_idx]
            leverage = LEVERAGE_OPTIONS[leverage_idx]

        current_price = float(self._close[self._current_idx])
        total_fees = 0.0
        total_funding = 0.0
        realized_pnl = 0.0

        # ── Map direction to side string ─────────────────────────────────
        desired_side = {0: "flat", 1: "long", 2: "short"}[direction]
        current_side = self.account.position_side

        # ── Execute trade logic ──────────────────────────────────────────
        if desired_side == "flat":
            if current_side != "flat":
                rpnl, fee = self.account.close_position(current_price)
                realized_pnl += rpnl
                total_fees += fee
            # else: no-op — already flat, no fees
        elif desired_side != current_side:
            # Close existing position first (if any)
            if current_side != "flat":
                rpnl, fee = self.account.close_position(current_price)
                realized_pnl += rpnl
                total_fees += fee
            # Open new position (if size > 0)
            if size_pct > 0.0:
                fee = self.account.open_position(
                    side=desired_side,
                    size_pct=size_pct,
                    leverage=leverage,
                    price=current_price,
                )
                total_fees += fee
        # else: same direction — hold, no fees

        # ── Apply funding if 8-hour boundary crossed ─────────────────────
        if self._current_idx > 0:
            total_funding = self._apply_funding_if_due()

        # ── Increment position time ──────────────────────────────────────
        if self.account.position_side != "flat":
            self.account.time_in_position += 1

        # ── Advance to next candle ───────────────────────────────────────
        self._current_idx += 1
        self._step_count += 1

        # Update unrealized PnL with new candle's close
        new_price = float(self._close[self._current_idx])
        self.account.update_unrealized_pnl(new_price)

        # ── Compute reward ───────────────────────────────────────────────
        current_equity = self.account.equity
        equity_change = current_equity - self._prev_equity
        # Normalize by initial balance
        reward = equity_change / self.initial_balance

        # Drawdown penalty: constant per-step cost proportional to drawdown depth
        drawdown = (self.account.peak_equity - current_equity) / self.account.peak_equity
        dd_penalty = _DRAWDOWN_PENALTY_COEFF * max(0.0, drawdown)
        reward -= dd_penalty * 0.001

        self._prev_equity = current_equity

        # ── Terminal conditions ───────────────────────────────────────────
        truncated = False
        terminated = False

        # Liquidation check
        if self.account.position_side != "flat" and self.account.is_liquidated(new_price):
            # Force close — balance goes to near zero
            loss = self.account.equity
            self.account.balance = 0.0
            self.account.unrealized_pnl = 0.0
            self.account.position_side = "flat"
            self.account.position_size = 0.0
            self.account.entry_price = 0.0
            reward = -loss / self.initial_balance  # large negative reward
            truncated = True

        # Episode length
        if self._step_count >= self.episode_length:
            terminated = True

        # Also terminate if out of data
        if self._current_idx >= len(self.df) - 1:
            terminated = True

        obs = self._get_observation()
        info = self._get_info()
        info["fees"] = total_fees
        info["funding"] = total_funding
        info["realized_pnl"] = realized_pnl

        return obs, float(reward), terminated, truncated, info

    # ── Observation helpers ──────────────────────────────────────────────

    def _get_observation(self) -> dict[str, np.ndarray]:
        market = self._get_market_obs()
        account = self._get_account_obs()
        return {"market": market, "account": account}

    def _get_market_obs(self) -> np.ndarray:
        """Return normalized OHLCV window as (window_size, 5) float32."""
        start = self._current_idx - self.window_size
        end = self._current_idx

        current_close = self._close[self._current_idx]
        if current_close == 0:
            current_close = 1.0  # safety

        o = self._open[start:end] / current_close - 1.0
        h = self._high[start:end] / current_close - 1.0
        lo = self._low[start:end] / current_close - 1.0
        c = self._close[start:end] / current_close - 1.0

        # Volume: normalize by rolling mean over the window
        vol_window = self._volume[start:end]
        vol_mean = vol_window.mean()
        if vol_mean == 0:
            vol_mean = 1.0
        v = vol_window / vol_mean - 1.0

        market = np.stack([o, h, lo, c, v], axis=-1).astype(np.float32)
        return market

    def _get_account_obs(self) -> np.ndarray:
        """Return 7-dimensional normalized account feature vector."""
        acct = self.account
        balance_ratio = acct.balance / self.initial_balance - 1.0
        pos_side = {
            "flat": 0.0,
            "long": 1.0,
            "short": -1.0,
        }[acct.position_side]
        size_ratio = 0.0
        if acct.position_side != "flat" and acct.entry_price > 0:
            notional = acct.position_size * acct.entry_price
            size_ratio = notional / self.initial_balance
        pnl_ratio = acct.unrealized_pnl / self.initial_balance
        leverage_ratio = acct.leverage / 20.0  # normalized to [0, 1]
        drawdown_ratio = 0.0
        if acct.peak_equity > 0:
            drawdown_ratio = (acct.peak_equity - acct.equity) / acct.peak_equity
        time_ratio = acct.time_in_position / self.episode_length

        return np.array(
            [
                balance_ratio,
                pos_side,
                size_ratio,
                pnl_ratio,
                leverage_ratio,
                drawdown_ratio,
                time_ratio,
            ],
            dtype=np.float32,
        )

    # ── Funding ──────────────────────────────────────────────────────────

    def _apply_funding_if_due(self) -> float:
        """Check if an 8-hour funding boundary was crossed and apply it.

        Returns the total funding cost applied (positive = paid).
        """
        if self.account.position_side == "flat":
            return 0.0

        prev_ts = pd.Timestamp(self._timestamps[self._current_idx - 1])
        curr_ts = pd.Timestamp(self._timestamps[self._current_idx])

        total_cost = 0.0

        # Check each funding hour
        for hour in _FUNDING_HOURS:
            # Build the funding timestamp for the same day as prev candle
            funding_time = prev_ts.normalize().replace(hour=hour, minute=0, second=0)
            # Also check next day boundary (e.g., prev=23:55, curr=00:05)
            for offset_days in (0, 1):
                ft = funding_time + pd.Timedelta(days=offset_days)
                if prev_ts < ft <= curr_ts:
                    rate = self._lookup_funding_rate(ft)
                    cost = self.account.apply_funding(rate)
                    total_cost += cost

        return total_cost

    def _lookup_funding_rate(self, funding_time: pd.Timestamp) -> float:
        """Look up funding rate from the funding map; return 0 if not found."""
        ts_key = int(funding_time.timestamp())
        return self._funding_map.get(ts_key, 0.0)

    # ── Info ─────────────────────────────────────────────────────────────

    def _get_info(self) -> dict[str, Any]:
        acct = self.account
        return {
            "balance": acct.balance,
            "equity": acct.equity,
            "position_side": acct.position_side,
            "position_size": acct.position_size,
            "entry_price": acct.entry_price,
            "unrealized_pnl": acct.unrealized_pnl,
            "leverage": acct.leverage,
            "step": self._step_count,
            "peak_equity": acct.peak_equity,
            "n_trades": len(acct.trade_history),
        }
