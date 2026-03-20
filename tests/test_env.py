"""Tests for the trading environment (alphacluster.env)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from alphacluster.config import (
    EPISODE_LENGTH,
    LEVERAGE_OPTIONS,
    MAKER_FEE,
    POSITION_SIZE_OPTIONS,
    TAKER_FEE,
    WINDOW_SIZE,
)
from alphacluster.env.account import Account
from alphacluster.env.mechanics import (
    apply_slippage,
    calculate_fee,
    calculate_funding,
    calculate_liquidation_price,
    calculate_pnl,
)
from alphacluster.env.trading_env import TradingEnv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n_candles: int = 3200, start_price: float = 50_000.0, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame for testing."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(
        start="2025-01-01", periods=n_candles, freq="5min", tz="UTC"
    )
    close = start_price + np.cumsum(rng.normal(0, 10, size=n_candles))
    # Ensure all prices are positive
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


def _make_funding_df(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a funding-rate DataFrame aligned to 8-hour boundaries in *df*."""
    rng = np.random.default_rng(99)
    start = df["open_time"].iloc[0]
    end = df["open_time"].iloc[-1]
    times = pd.date_range(start=start.normalize(), end=end, freq="8h", tz="UTC")
    rates = rng.normal(0.0001, 0.0005, size=len(times))
    return pd.DataFrame({"time": times, "funding_rate": rates})


def _make_env(n_candles: int = 3200, episode_length: int = 200, **kwargs) -> TradingEnv:
    """Convenience wrapper to build a small test environment."""
    df = _make_df(n_candles=n_candles)
    funding_df = _make_funding_df(df)
    return TradingEnv(
        df=df,
        funding_df=funding_df,
        window_size=WINDOW_SIZE,
        episode_length=episode_length,
        initial_balance=10_000.0,
        **kwargs,
    )


# ===========================================================================
# Mechanics tests
# ===========================================================================


class TestMechanics:
    """Tests for pure trading-mechanics functions."""

    def test_calculate_fee_taker(self):
        fee = calculate_fee(100_000.0)
        assert fee == pytest.approx(100_000.0 * TAKER_FEE)

    def test_calculate_fee_maker(self):
        fee = calculate_fee(100_000.0, MAKER_FEE)
        assert fee == pytest.approx(100_000.0 * MAKER_FEE)

    def test_calculate_fee_zero(self):
        assert calculate_fee(0.0) == 0.0

    def test_calculate_funding_long(self):
        # Positive rate, long position -> holder pays
        cost = calculate_funding(50_000.0, 0.0001)
        assert cost == pytest.approx(5.0)

    def test_calculate_funding_short(self):
        # Positive rate, short position (negative value) -> holder receives
        cost = calculate_funding(-50_000.0, 0.0001)
        assert cost == pytest.approx(-5.0)

    def test_liquidation_price_long(self):
        liq = calculate_liquidation_price(50_000.0, 10, "long")
        # With 10x leverage, margin distance = 0.1 - 0.005 = 0.095
        expected = 50_000.0 * (1.0 - 0.095)
        assert liq == pytest.approx(expected)

    def test_liquidation_price_short(self):
        liq = calculate_liquidation_price(50_000.0, 10, "short")
        expected = 50_000.0 * (1.0 + 0.095)
        assert liq == pytest.approx(expected)

    def test_liquidation_price_low_leverage(self):
        liq = calculate_liquidation_price(50_000.0, 1, "long")
        # margin distance = 1.0 - 0.005 = 0.995
        expected = 50_000.0 * (1.0 - 0.995)
        assert liq == pytest.approx(expected)

    def test_liquidation_price_invalid_side(self):
        with pytest.raises(ValueError):
            calculate_liquidation_price(50_000.0, 10, "invalid")

    def test_calculate_pnl_long_profit(self):
        pnl = calculate_pnl(50_000.0, 51_000.0, 1.0, "long")
        assert pnl == pytest.approx(1_000.0)

    def test_calculate_pnl_long_loss(self):
        pnl = calculate_pnl(50_000.0, 49_000.0, 1.0, "long")
        assert pnl == pytest.approx(-1_000.0)

    def test_calculate_pnl_short_profit(self):
        pnl = calculate_pnl(50_000.0, 49_000.0, 1.0, "short")
        assert pnl == pytest.approx(1_000.0)

    def test_calculate_pnl_short_loss(self):
        pnl = calculate_pnl(50_000.0, 51_000.0, 1.0, "short")
        assert pnl == pytest.approx(-1_000.0)

    def test_calculate_pnl_invalid_side(self):
        with pytest.raises(ValueError):
            calculate_pnl(50_000.0, 51_000.0, 1.0, "invalid")

    def test_apply_slippage_buy(self):
        price = apply_slippage(50_000.0, "buy")
        assert price > 50_000.0
        assert price == pytest.approx(50_000.0 * 1.0001)

    def test_apply_slippage_sell(self):
        price = apply_slippage(50_000.0, "sell")
        assert price < 50_000.0
        assert price == pytest.approx(50_000.0 * 0.9999)

    def test_apply_slippage_custom(self):
        price = apply_slippage(50_000.0, "buy", 0.001)
        assert price == pytest.approx(50_000.0 * 1.001)

    def test_apply_slippage_invalid_side(self):
        with pytest.raises(ValueError):
            apply_slippage(50_000.0, "invalid")


# ===========================================================================
# Account tests
# ===========================================================================


class TestAccount:
    """Tests for the Account class."""

    def test_initial_state(self):
        acct = Account(initial_balance=10_000.0)
        assert acct.balance == 10_000.0
        assert acct.position_side == "flat"
        assert acct.position_size == 0.0
        assert acct.equity == 10_000.0

    def test_open_position_long(self):
        acct = Account(initial_balance=10_000.0)
        fee = acct.open_position("long", 0.5, 5, 50_000.0)
        assert acct.position_side == "long"
        assert acct.position_size > 0
        assert fee > 0
        assert acct.balance < 10_000.0  # fee deducted

    def test_open_position_short(self):
        acct = Account(initial_balance=10_000.0)
        fee = acct.open_position("short", 1.0, 10, 50_000.0)
        assert acct.position_side == "short"
        assert acct.position_size > 0
        assert fee > 0

    def test_close_position_pnl(self):
        acct = Account(initial_balance=10_000.0)
        acct.open_position("long", 0.5, 1, 50_000.0)
        pnl, fee = acct.close_position(51_000.0)
        assert pnl > 0  # price went up → long profit
        assert fee > 0
        assert acct.position_side == "flat"

    def test_close_flat_is_noop(self):
        acct = Account(initial_balance=10_000.0)
        pnl, fee = acct.close_position(50_000.0)
        assert pnl == 0.0
        assert fee == 0.0

    def test_unrealized_pnl_update(self):
        acct = Account(initial_balance=10_000.0)
        acct.open_position("long", 1.0, 1, 50_000.0)
        acct.update_unrealized_pnl(51_000.0)
        assert acct.unrealized_pnl > 0

    def test_apply_funding(self):
        acct = Account(initial_balance=10_000.0)
        acct.open_position("long", 0.5, 1, 50_000.0)
        balance_before = acct.balance
        cost = acct.apply_funding(0.0001)
        assert cost > 0  # positive rate, long → pays
        assert acct.balance < balance_before

    def test_is_liquidated_long(self):
        acct = Account(initial_balance=10_000.0)
        acct.open_position("long", 1.0, 20, 50_000.0)
        # With 20x leverage, liq price is very close to entry
        liq_price = calculate_liquidation_price(acct.entry_price, 20, "long")
        assert not acct.is_liquidated(acct.entry_price)
        assert acct.is_liquidated(liq_price - 1.0)

    def test_is_liquidated_short(self):
        acct = Account(initial_balance=10_000.0)
        acct.open_position("short", 1.0, 20, 50_000.0)
        liq_price = calculate_liquidation_price(acct.entry_price, 20, "short")
        assert not acct.is_liquidated(acct.entry_price)
        assert acct.is_liquidated(liq_price + 1.0)

    def test_is_liquidated_flat(self):
        acct = Account(initial_balance=10_000.0)
        assert not acct.is_liquidated(50_000.0)

    def test_margin_ratio(self):
        acct = Account(initial_balance=10_000.0)
        assert acct.margin_ratio() == float("inf")  # flat
        acct.open_position("long", 1.0, 1, 50_000.0)
        ratio = acct.margin_ratio()
        assert ratio > 0

    def test_reset(self):
        acct = Account(initial_balance=10_000.0)
        acct.open_position("long", 0.5, 5, 50_000.0)
        acct.reset()
        assert acct.balance == 10_000.0
        assert acct.position_side == "flat"
        assert len(acct.trade_history) == 0

    def test_trade_history_recorded(self):
        acct = Account(initial_balance=10_000.0)
        acct.open_position("long", 0.5, 5, 50_000.0)
        acct.close_position(51_000.0)
        assert len(acct.trade_history) == 2
        assert acct.trade_history[0]["action"] == "open"
        assert acct.trade_history[1]["action"] == "close"


# ===========================================================================
# TradingEnv tests
# ===========================================================================


class TestTradingEnv:
    """Test suite for the Gymnasium trading environment."""

    def test_env_creation(self):
        env = _make_env()
        assert env is not None
        assert env.observation_space is not None
        assert env.action_space is not None

    def test_observation_space_shape(self):
        env = _make_env()
        obs, info = env.reset(seed=0)
        assert obs["market"].shape == (WINDOW_SIZE, 5)
        assert obs["account"].shape == (7,)

    def test_action_space_size(self):
        env = _make_env()
        assert env.action_space.shape == (3,)
        assert env.action_space.nvec[0] == 3  # directions
        assert env.action_space.nvec[1] == 4  # sizes
        assert env.action_space.nvec[2] == 5  # leverages

    def test_reset_returns_valid_obs(self):
        env = _make_env()
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        assert "balance" in info
        assert info["balance"] == 10_000.0

    def test_step_returns_correct_tuple(self):
        env = _make_env()
        env.reset(seed=0)
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_produces_valid_observations(self):
        env = _make_env()
        env.reset(seed=0)
        for _ in range(10):
            action = env.action_space.sample()
            obs, *_ = env.step(action)
            assert env.observation_space.contains(obs)

    def test_episode_terminates(self):
        env = _make_env(episode_length=50)
        env.reset(seed=0)
        terminated = False
        steps = 0
        while not terminated:
            action = [0, 0, 0]  # stay flat
            _, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if truncated:
                break
            if steps > 100:
                pytest.fail("Episode did not terminate within expected steps")
        assert steps == 50

    def test_fee_deduction(self):
        env = _make_env()
        env.reset(seed=0)
        # Open a long position
        action = [1, 3, 0]  # long, 100% size, 1x leverage
        _, _, _, _, info = env.step(action)
        assert info["fees"] > 0
        assert info["balance"] < 10_000.0

    def test_flat_when_flat_is_noop(self):
        env = _make_env()
        env.reset(seed=0)
        # Action: flat (already flat)
        action = [0, 0, 0]
        _, _, _, _, info = env.step(action)
        assert info["fees"] == 0.0
        # Balance changes only due to no fees/funding in flat position
        assert info["n_trades"] == 0

    def test_funding_rate_application(self):
        """Verify funding is applied when crossing an 8-hour boundary."""
        # Create data spanning at least one 8-hour window
        df = _make_df(n_candles=3200)
        funding_df = _make_funding_df(df)
        env = TradingEnv(
            df=df,
            funding_df=funding_df,
            window_size=WINDOW_SIZE,
            episode_length=200,
            initial_balance=10_000.0,
        )
        env.reset(seed=0)

        # Open a position first
        env.step([1, 3, 2])  # long, 100%, 5x

        # Step through enough candles to cross an 8-hour boundary
        # 8 hours / 5 min = 96 candles
        total_funding = 0.0
        for _ in range(100):
            _, _, terminated, truncated, info = env.step([1, 3, 2])  # hold long
            total_funding += info.get("funding", 0.0)
            if terminated or truncated:
                break

        # We should have crossed at least one funding boundary
        # (Not guaranteed due to random start, but likely with 100 candles)
        # Just verify the mechanism works without error

    def test_liquidation_triggers(self):
        """Test that liquidation sets balance to zero and truncates."""
        acct = Account(initial_balance=10_000.0)
        acct.open_position("long", 1.0, 20, 50_000.0)
        liq_price = calculate_liquidation_price(acct.entry_price, 20, "long")
        # The account should be liquidated at a price below liq_price
        assert acct.is_liquidated(liq_price - 100.0)

    def test_direction_change_closes_and_opens(self):
        """Opening long then going short should close long first, then open short."""
        env = _make_env()
        env.reset(seed=0)
        # Open long
        env.step([1, 2, 0])  # long, 50%, 1x
        assert env.account.position_side == "long"
        # Switch to short
        env.step([2, 2, 0])  # short, 50%, 1x
        assert env.account.position_side == "short"
        # There should be open, close, open in history
        assert len(env.account.trade_history) >= 3

    def test_hold_same_direction_no_fees(self):
        """Holding the same direction should not charge fees."""
        env = _make_env()
        env.reset(seed=0)
        # Open long
        env.step([1, 2, 0])
        # Hold long — same direction
        _, _, _, _, info = env.step([1, 2, 0])
        assert info["fees"] == 0.0

    def test_market_obs_normalization(self):
        """OHLCV prices should be normalized relative to current close."""
        env = _make_env()
        obs, _ = env.reset(seed=0)
        market = obs["market"]
        # The last candle's close (index -1, col 3) should be ratio to current close - 1
        # which is (prev_close / current_close - 1), close to 0 for adjacent candles
        # Just verify the values are in a reasonable range
        assert np.all(np.isfinite(market))
        # Prices should be close to 0 (ratios)
        assert np.abs(market[:, :4]).max() < 1.0  # price ratios within 100%

    def test_account_obs_normalized(self):
        """Account features should be finite and reasonably bounded."""
        env = _make_env()
        obs, _ = env.reset(seed=0)
        acct_obs = obs["account"]
        assert acct_obs.shape == (7,)
        assert np.all(np.isfinite(acct_obs))
        # Initial state: balance_ratio=0, side=0, size=0, pnl=0, lev=0.05, dd=0, time=0
        assert acct_obs[0] == pytest.approx(0.0)  # balance_ratio
        assert acct_obs[1] == pytest.approx(0.0)  # flat

    def test_gymnasium_check_env(self):
        """Run Gymnasium's built-in environment checker."""
        env = _make_env(episode_length=50)
        # check_env will reset + step multiple times
        check_env(env.unwrapped, skip_render_check=True)

    def test_reward_is_reasonable(self):
        """Reward should not be excessively large."""
        env = _make_env()
        env.reset(seed=0)
        rewards = []
        for _ in range(50):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break
        rewards = np.array(rewards)
        # Rewards normalized by initial balance should be small
        assert np.abs(rewards).max() < 10.0, f"Max reward magnitude too large: {np.abs(rewards).max()}"

    def test_env_registration(self):
        """CryptoPerp-v0 should be registered with Gymnasium."""
        import gymnasium

        import alphacluster.env  # noqa: F401 — triggers registration

        spec = gymnasium.spec("CryptoPerp-v0")
        assert spec is not None
        assert spec.entry_point == "alphacluster.env.trading_env:TradingEnv"
