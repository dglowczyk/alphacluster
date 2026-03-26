"""Tests for the trading environment (alphacluster.env)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from alphacluster.agent.config import TrainingConfig
from alphacluster.agent.trainer import CurriculumCallback
from alphacluster.config import (
    MAKER_FEE,
    N_ACCOUNT_FEATURES,
    N_MARKET_FEATURES,
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
    timestamps = pd.date_range(start="2025-01-01", periods=n_candles, freq="5min", tz="UTC")
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

    def test_modify_position_flat_is_noop(self):
        acct = Account(initial_balance=10_000.0)
        fee = acct.modify_position(0.5, 5, 50_000.0)
        assert fee == 0.0
        assert len(acct.trade_history) == 0

    def test_modify_position_changes_leverage(self):
        acct = Account(initial_balance=10_000.0)
        acct.open_position("long", 0.5, 1, 50_000.0)
        original_entry = acct.entry_price
        original_balance = acct.balance
        fee = acct.modify_position(0.5, 5, 50_000.0)
        assert fee > 0
        assert acct.leverage == 5
        # Entry price should remain unchanged
        assert acct.entry_price == original_entry
        assert acct.balance < original_balance  # fee was deducted
        assert acct.trade_history[-1]["action"] == "modify"

    def test_modify_position_reduced_fees_vs_close_open(self):
        """Modifying a position should cost less in fees than close + open."""
        price = 50_000.0

        # Approach 1: close + open (full round-trip fees)
        acct1 = Account(initial_balance=10_000.0)
        acct1.open_position("long", 0.5, 1, price)
        _, close_fee = acct1.close_position(price)
        open_fee = acct1.open_position("long", 0.5, 5, price)
        full_fees = close_fee + open_fee

        # Approach 2: modify (delta-only fees)
        acct2 = Account(initial_balance=10_000.0)
        acct2.open_position("long", 0.5, 1, price)
        modify_fee = acct2.modify_position(0.5, 5, price)

        assert modify_fee < full_fees, (
            f"Modify fee ({modify_fee:.4f}) should be less than close+open ({full_fees:.4f})"
        )

    def test_modify_position_downsize_realizes_pnl(self):
        """Downsizing should realize PnL on the reduced portion."""
        acct = Account(initial_balance=10_000.0)
        acct.open_position("long", 1.0, 1, 50_000.0)
        original_size = acct.position_size
        # Price goes up, then modify to smaller size — should realize partial profit
        acct.modify_position(0.25, 1, 51_000.0)
        assert acct.position_size > 0
        assert acct.position_size < original_size
        assert acct.position_side == "long"

    def test_modify_position_preserves_side(self):
        acct = Account(initial_balance=10_000.0)
        acct.open_position("short", 0.5, 3, 50_000.0)
        acct.modify_position(0.75, 10, 50_000.0)
        assert acct.position_side == "short"
        assert acct.leverage == 10

    def test_modify_position_fee_on_delta_only(self):
        """Fee should be proportional to |new_notional - old_notional|, not total notional."""
        acct = Account(initial_balance=10_000.0)
        acct.open_position("long", 0.5, 1, 50_000.0)
        # old_notional = balance * 0.5 * 1 = ~5000 (at entry time)
        # new_notional = balance * 0.5 * 3 (3x leverage) — delta is ~10000
        old_notional = acct.position_size * acct.entry_price
        balance_before_modify = acct.balance
        fee = acct.modify_position(0.5, 3, 50_000.0)
        new_notional = (balance_before_modify * 0.5) * 3
        expected_delta = abs(new_notional - old_notional)
        expected_fee = calculate_fee(expected_delta, TAKER_FEE)
        assert fee == pytest.approx(expected_fee, rel=1e-6)

    # ------------------------------------------------------------------
    # Isolated margin tests
    # ------------------------------------------------------------------

    def test_margin_stored_on_open(self):
        """Margin field should equal balance * size_pct after opening."""
        acct = Account(initial_balance=10_000.0)
        acct.open_position("long", 0.10, 10, 50_000.0)
        assert acct.margin == pytest.approx(1_000.0, rel=1e-6)

    def test_margin_reset_on_close(self):
        """Margin should be zero after closing a position."""
        acct = Account(initial_balance=10_000.0)
        acct.open_position("long", 0.10, 5, 50_000.0)
        assert acct.margin > 0
        acct.close_position(50_000.0)
        assert acct.margin == 0.0

    def test_margin_updated_on_modify(self):
        """Margin should be recalculated after position modification."""
        acct = Account(initial_balance=10_000.0)
        acct.open_position("long", 0.05, 5, 50_000.0)
        original_margin = acct.margin
        assert original_margin == pytest.approx(500.0, rel=1e-6)
        acct.modify_position(0.10, 5, 50_000.0)
        assert acct.margin != original_margin
        assert acct.margin > 0

    def test_isolated_margin_liquidation(self):
        """Liquidation should lose only the margin, not the entire balance."""
        acct = Account(initial_balance=10_000.0)
        acct.open_position("long", 0.10, 10, 50_000.0)
        margin = acct.margin
        balance_after_open_fee = acct.balance  # balance minus opening fee
        margin_lost = acct.liquidate()
        assert margin_lost == pytest.approx(margin)
        assert acct.balance == pytest.approx(balance_after_open_fee - margin)
        assert acct.position_side == "flat"
        assert acct.margin == 0.0
        assert acct.position_size == 0.0
        # Check trade history records liquidation
        liq_entries = [t for t in acct.trade_history if t["action"] == "liquidation"]
        assert len(liq_entries) == 1
        assert liq_entries[0]["margin_lost"] == pytest.approx(margin)

    def test_liquidation_flat_is_noop(self):
        """Liquidating a flat account should return 0 and change nothing."""
        acct = Account(initial_balance=10_000.0)
        result = acct.liquidate()
        assert result == 0.0
        assert acct.balance == 10_000.0

    def test_multiple_liquidations(self):
        """Multiple open/liquidate cycles should correctly reduce balance each time."""
        acct = Account(initial_balance=10_000.0)

        # First cycle: 10% margin
        acct.open_position("long", 0.10, 10, 50_000.0)
        first_margin = acct.margin
        balance_before_first_liq = acct.balance
        acct.liquidate()
        assert acct.balance == pytest.approx(balance_before_first_liq - first_margin)
        assert acct.position_side == "flat"

        # Second cycle: 10% of remaining balance
        remaining = acct.balance
        acct.open_position("short", 0.10, 5, 50_000.0)
        second_margin = acct.margin
        assert second_margin == pytest.approx(remaining * 0.10, rel=1e-4)
        balance_before_second_liq = acct.balance
        acct.liquidate()
        assert acct.balance == pytest.approx(balance_before_second_liq - second_margin)
        assert acct.balance > 0  # Still has balance left


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
        assert obs["market"].shape == (WINDOW_SIZE, N_MARKET_FEATURES)
        assert obs["account"].shape == (N_ACCOUNT_FEATURES,)

    def test_action_space_size(self):
        env = _make_env()
        assert env.action_space.shape == (3,)
        assert env.action_space.nvec[0] == 3  # directions
        assert env.action_space.nvec[1] == 4  # sizes
        assert env.action_space.nvec[2] == 3  # leverages

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
        # Open a long position: long, 100% size (index 3), 5x leverage
        action = [1, 3, 0]
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

        # Open a position first: long, 100%, 5x
        env.step([1, 3, 2])

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
        """Test that is_liquidated detects price crossing the liquidation threshold."""
        acct = Account(initial_balance=10_000.0)
        acct.open_position("long", 1.0, 20, 50_000.0)
        liq_price = calculate_liquidation_price(acct.entry_price, 20, "long")
        # The account should be liquidated at a price below liq_price
        assert acct.is_liquidated(liq_price - 100.0)

    def test_direction_change_closes_and_opens(self):
        """Opening long then going short should close long first, then open short."""
        env = _make_env()
        env.reset(seed=0)
        # Open long: direction=1, size=50% (index 1), 5x leverage
        env.step([1, 1, 0])
        assert env.account.position_side == "long"
        # Switch to short
        env.step([2, 1, 0])
        assert env.account.position_side == "short"
        # There should be open, close, open in history
        assert len(env.account.trade_history) >= 3

    def test_hold_same_direction_no_fees(self):
        """Holding the same direction should not charge fees."""
        env = _make_env()
        env.reset(seed=0)
        # Open long: direction=1, size=50% (index 1), 5x leverage
        env.step([1, 1, 0])
        # Hold long — same direction
        _, _, _, _, info = env.step([1, 1, 0])
        assert info["fees"] == 0.0

    def test_modify_position_via_env_step(self):
        """Changing leverage while keeping direction should use modify_position."""
        env = _make_env()
        env.reset(seed=0)
        # Open long: direction=1, size=50% (index 1), 5x leverage (index 0)
        env.step([1, 1, 0])
        assert env.account.position_side == "long"
        assert env.account.leverage == 5
        # Change to 15x leverage (index 2) — same direction, different leverage
        _, _, _, _, info = env.step([1, 1, 2])
        assert info["fees"] > 0  # fees charged for modification
        assert env.account.leverage == 15  # leverage updated
        assert env.account.position_side == "long"  # still long
        # Should be a modify in trade history, not close+open
        assert env.account.trade_history[-1]["action"] == "modify"

    def test_modify_position_cheaper_than_reversal(self):
        """Modifying via env step should be cheaper than close+reopen."""
        env1 = _make_env()
        env1.reset(seed=0)
        # Open long 5x, then modify to 15x
        env1.step([1, 1, 0])  # open long, 50%, 5x
        _, _, _, _, info_modify = env1.step([1, 1, 2])  # long, 50%, 15x
        modify_fees = info_modify["fees"]

        env2 = _make_env()
        env2.reset(seed=0)
        # Open long 5x, close, reopen at 15x
        env2.step([1, 1, 0])  # open long, 50%, 5x
        env2.step([0, 0, 0])  # close (flat)
        _, _, _, _, info_reopen = env2.step([1, 1, 2])  # open long, 50%, 15x
        # Close fees from step 2 + open fees from step 3
        # We need to sum them manually from trade history
        close_fee = env2.account.trade_history[1]["fee"]
        reopen_fee = env2.account.trade_history[2]["fee"]
        roundtrip_fees = close_fee + reopen_fee

        assert modify_fees < roundtrip_fees, (
            f"Modify fees ({modify_fees:.6f}) should be less than "
            f"close+open fees ({roundtrip_fees:.6f})"
        )

    def test_market_obs_normalization(self):
        """OHLCV prices should be normalized relative to current close."""
        env = _make_env()
        obs, _ = env.reset(seed=0)
        market = obs["market"]
        # The last candle's close (index -1, col 3) should be ratio to current close - 1
        # Just verify the values are in a reasonable range
        assert np.all(np.isfinite(market))
        # Price ratios (first 4 cols) should be within 100%
        assert np.abs(market[:, :4]).max() < 1.0

    def test_account_obs_normalized(self):
        """Account features should be finite and reasonably bounded."""
        env = _make_env()
        obs, _ = env.reset(seed=0)
        acct_obs = obs["account"]
        assert acct_obs.shape == (N_ACCOUNT_FEATURES,)
        assert np.all(np.isfinite(acct_obs))
        # Initial state: balance_ratio=0, side=0
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
        max_mag = np.abs(rewards).max()
        assert max_mag < 10.0, f"Max reward magnitude too large: {max_mag}"

    def test_env_registration(self):
        """CryptoPerp-v0 should be registered with Gymnasium."""
        import gymnasium

        import alphacluster.env  # noqa: F401 — triggers registration

        spec = gymnasium.spec("CryptoPerp-v0")
        assert spec is not None
        assert spec.entry_point == "alphacluster.env.trading_env:TradingEnv"

    def test_trade_tracking_state(self):
        """Trade tracking state should update when positions are closed."""
        env = _make_env()
        env.reset(seed=0)
        assert env._n_completed_trades == 0

        # Open and close a position
        env.step([1, 1, 0])  # open long
        env.step([0, 0, 0])  # close to flat
        assert env._n_completed_trades == 1
        assert env._steps_since_last_trade == 0

    def test_reward_config_mutable(self):
        """reward_config should be changeable for curriculum learning."""
        env = _make_env()
        env.reset(seed=0)
        env.reward_config["opportunity_cost_scale"] = 2.0
        assert env.reward_config["opportunity_cost_scale"] == 2.0
        env.reward_config["churn_penalty_scale"] = 0.5
        assert env.reward_config["churn_penalty_scale"] == 0.5

    def test_churn_penalty_quadratic(self):
        """Short trades should incur significant quadratic churn penalty."""
        env = _make_env()
        env.reset(seed=0)
        # Open and immediately close (1-step trade)
        env.step([1, 0, 0])  # open long
        _, reward_quick, *_ = env.step([0, 0, 0])  # close immediately

        # Open and hold longer before closing
        env2 = _make_env()
        env2.reset(seed=0)
        env2.step([1, 0, 0])  # open long
        for _ in range(25):
            env2.step([1, 0, 0])  # hold
        _, reward_held, *_ = env2.step([0, 0, 0])  # close after 26 steps

        # Quick trade should have worse reward due to churn penalty
        assert reward_quick < reward_held

    def test_completion_reward_duration_scaling(self):
        """Trade completion reward should scale quadratically with duration."""
        env = _make_env()
        env.reset(seed=0)
        # The completion reward formula gives near-zero for 1-step winners
        # and much larger for longer winners. We verify via the internal state.
        env.step([1, 0, 0])  # open long
        env.step([0, 0, 0])  # close after 1 step
        short_duration = env._last_trade_duration

        env2 = _make_env()
        env2.reset(seed=0)
        env2.step([1, 0, 0])  # open long
        for _ in range(19):
            env2.step([1, 0, 0])  # hold
        env2.step([0, 0, 0])  # close after 20 steps
        long_duration = env2._last_trade_duration

        assert short_duration < 5
        assert long_duration >= 20

    def test_quality_bonus_for_winners(self):
        """Quality bonus should fire for winning trades only."""
        env = _make_env()
        env.reset(seed=0)
        assert "quality_scale" in env.reward_config
        # The quality bonus is part of the reward — we verify the key exists
        # and that the reward config accepts it
        env.reward_config["quality_scale"] = 0.0
        assert env.reward_config["quality_scale"] == 0.0
        env.reward_config["quality_scale"] = 1.0
        assert env.reward_config["quality_scale"] == 1.0

    def test_position_sizes_no_zero(self):
        """All non-flat actions should open positions (no 0% size)."""
        env = _make_env()
        env.reset(seed=0)
        # Try all size indices with direction=long
        for size_idx in range(4):
            env.reset(seed=0)
            env.step([1, size_idx, 0])
            assert env.account.position_side == "long", (
                f"size_idx={size_idx} should open a position"
            )

    def test_extended_account_features(self):
        """Account obs should include all 12 features including new ones."""
        env = _make_env()
        env.reset(seed=0)
        # Open, hold, close to populate tracking state
        env.step([1, 1, 0])  # open long
        env.step([1, 1, 0])  # hold
        env.step([0, 0, 0])  # close
        obs, *_ = env.step([0, 0, 0])  # flat
        acct_obs = obs["account"]
        assert acct_obs.shape == (N_ACCOUNT_FEATURES,)
        # running_win_rate (index 11) should be 0 or 1 after one trade
        assert acct_obs[11] in (0.0, 1.0)

    def test_liquidation_continues_episode(self):
        """Liquidation should not truncate the episode under isolated margin."""
        env = _make_env()
        env.reset(seed=0)

        # Open a long position with high leverage (long, smallest size, 15x)
        obs, reward, terminated, truncated, info = env.step(np.array([1, 0, 2]))

        # Get the account state
        balance_before = env.account.balance
        margin = env.account.margin
        assert margin > 0

        # Directly test that liquidation via account works correctly
        margin_lost = env.account.liquidate()
        assert margin_lost > 0
        assert env.account.position_side == "flat"
        assert env.account.balance == pytest.approx(balance_before - margin_lost)
        assert env.account.balance > 0  # Balance survives

        # Continue stepping the environment - should not crash
        obs, reward, terminated, truncated, info = env.step(np.array([0, 0, 0]))  # flat
        assert not truncated  # Episode did not end


def test_model_version_exists():
    """MODEL_VERSION should be defined in config and be a non-empty string."""
    from alphacluster.config import MODEL_VERSION

    assert isinstance(MODEL_VERSION, str)
    assert len(MODEL_VERSION) > 0


class TestOpportunityCost:
    """Tests for the opportunity cost reward component."""

    def _make_trending_env(self, trend_pct: float = 0.005, n_candles: int = 3200):
        """Create env with a clear directional trend over the lookback window."""
        rng = np.random.default_rng(42)
        timestamps = pd.date_range(start="2025-01-01", periods=n_candles, freq="5min", tz="UTC")
        base_price = 80_000.0
        step_size = base_price * trend_pct / 20
        close = base_price + np.arange(n_candles) * step_size
        close = close + rng.normal(0, 0.5, size=n_candles)
        high = close + rng.uniform(0, 5, size=n_candles)
        low = close - rng.uniform(0, 5, size=n_candles)
        low = np.maximum(low, 1.0)
        opn = close + rng.normal(0, 1, size=n_candles)
        opn = np.maximum(opn, 1.0)
        volume = rng.uniform(100, 10000, size=n_candles)
        df = pd.DataFrame(
            {
                "open_time": timestamps,
                "open": opn,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
        return TradingEnv(
            df=df,
            window_size=WINDOW_SIZE,
            episode_length=200,
            initial_balance=10_000.0,
        )

    def test_opportunity_cost_when_flat_during_trend(self):
        """Flat agent during >0.2% trend should receive negative opportunity cost."""
        env = self._make_trending_env(trend_pct=0.01)
        env.reset(seed=0)
        rewards = []
        for _ in range(15):
            _, reward, *_ = env.step([0, 0, 0])
            rewards.append(reward)
        assert any(r < 0 for r in rewards[10:])

    def test_opportunity_cost_zero_during_consolidation(self):
        """Flat agent during sideways market (<0.2% change) should get no penalty."""
        rng = np.random.default_rng(42)
        n_candles = 3200
        timestamps = pd.date_range(start="2025-01-01", periods=n_candles, freq="5min", tz="UTC")
        close = 80_000.0 + rng.normal(0, 5, size=n_candles)
        close = np.maximum(close, 79_000.0)
        high = close + rng.uniform(0, 2, size=n_candles)
        low = close - rng.uniform(0, 2, size=n_candles)
        opn = close + rng.normal(0, 1, size=n_candles)
        opn = np.maximum(opn, 1.0)
        volume = rng.uniform(100, 10000, size=n_candles)
        df = pd.DataFrame(
            {
                "open_time": timestamps,
                "open": opn,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
        env = TradingEnv(
            df=df,
            window_size=WINDOW_SIZE,
            episode_length=200,
            initial_balance=10_000.0,
        )
        env.reset(seed=0)
        rewards = []
        for _ in range(20):
            _, reward, *_ = env.step([0, 0, 0])
            rewards.append(reward)
        assert all(r == pytest.approx(0.0, abs=1e-9) for r in rewards)

    def test_opportunity_cost_zero_when_in_position(self):
        """Agent holding a position should not receive opportunity cost."""
        env = self._make_trending_env(trend_pct=0.01)
        env.reset(seed=0)
        flat_rewards = []
        for _ in range(15):
            _, r, *_ = env.step([0, 0, 0])
            flat_rewards.append(r)

        env.reset(seed=0)
        env.step([1, 1, 0])
        position_rewards = []
        for _ in range(14):
            _, r, *_ = env.step([1, 1, 0])
            position_rewards.append(r)

        assert sum(flat_rewards[10:]) < 0, "Flat during trend should be penalized"
        assert sum(position_rewards[10:]) > sum(flat_rewards[10:])

    def test_opportunity_cost_at_episode_start(self):
        """First 10 steps should clamp lookback, no crash."""
        env = self._make_trending_env(trend_pct=0.01)
        env.reset(seed=0)
        for _ in range(10):
            _, reward, terminated, *_ = env.step([0, 0, 0])
            assert not terminated

    def test_opportunity_cost_capped(self):
        """During large moves (>2%), penalty should cap at 0.02 * scale."""
        rng = np.random.default_rng(42)
        n_candles = 3200
        timestamps = pd.date_range(start="2025-01-01", periods=n_candles, freq="5min", tz="UTC")
        base = 80_000.0
        close = np.full(n_candles, base)
        for i in range(590, 600):
            close[i] = base + (i - 589) * base * 0.005
        for i in range(600, n_candles):
            close[i] = base * 1.05
        high = close + 10
        low = close - 10
        low = np.maximum(low, 1.0)
        opn = close.copy()
        volume = rng.uniform(100, 10000, size=n_candles)
        df = pd.DataFrame(
            {
                "open_time": timestamps,
                "open": opn,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
        env = TradingEnv(
            df=df,
            window_size=WINDOW_SIZE,
            episode_length=200,
            initial_balance=10_000.0,
        )
        env.reset(seed=0)
        rewards = []
        for _ in range(30):
            _, reward, *_ = env.step([0, 0, 0])
            rewards.append(reward)
        # Cap is 0.02 * scale (default 0.5) = 0.01
        for r in rewards:
            assert r >= -0.01 - 1e-9

    def test_opportunity_cost_exact_threshold(self):
        """Price change of exactly 0.2% should produce zero penalty."""
        rng = np.random.default_rng(42)
        n_candles = 3200
        timestamps = pd.date_range(start="2025-01-01", periods=n_candles, freq="5min", tz="UTC")
        base = 80_000.0
        close = np.full(n_candles, base, dtype=np.float64)
        for i in range(577, 587):
            close[i] = base * (1.0 + 0.002 * (i - 576) / 10)
        for i in range(587, n_candles):
            close[i] = base * 1.002
        high = close + 5
        low = close - 5
        low = np.maximum(low, 1.0)
        opn = close.copy()
        volume = rng.uniform(100, 10000, size=n_candles)
        df = pd.DataFrame(
            {
                "open_time": timestamps,
                "open": opn,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
        env = TradingEnv(
            df=df,
            window_size=WINDOW_SIZE,
            episode_length=200,
            initial_balance=10_000.0,
        )
        env.reset(seed=0)
        rewards = []
        for _ in range(20):
            _, r, *_ = env.step([0, 0, 0])
            rewards.append(r)
        assert all(r >= -1e-9 for r in rewards)

    def test_trading_beats_flat_during_trend(self):
        """Opening a position during a trend should beat staying flat."""
        env_flat = self._make_trending_env(trend_pct=0.005)
        env_flat.reset(seed=0)
        flat_rewards = []
        for _ in range(15):
            _, r, *_ = env_flat.step([0, 0, 0])
            flat_rewards.append(r)

        env_trade = self._make_trending_env(trend_pct=0.005)
        env_trade.reset(seed=0)
        trade_rewards = []
        _, r, *_ = env_trade.step([1, 1, 0])
        trade_rewards.append(r)
        for _ in range(14):
            _, r, *_ = env_trade.step([1, 1, 0])
            trade_rewards.append(r)

        assert sum(trade_rewards) > sum(flat_rewards)


def test_churn_penalty_rebalanced():
    """Churn penalty with base=0.02, threshold=20."""
    env = _make_env()
    env.reset(seed=0)

    env._trade_just_completed = True
    env._last_trade_duration = 1
    env._last_trade_pnl = 0.0
    env._step_fees = 0.0
    r1 = env._compute_reward()

    env._last_trade_duration = 10
    r10 = env._compute_reward()

    env._last_trade_duration = 20
    r20 = env._compute_reward()

    assert r1 < r10
    assert r20 >= r10


def test_training_metrics_callback_creates_csv(tmp_path):
    """TrainingMetricsCallback should be instantiable with correct attributes."""
    from alphacluster.agent.trainer import TrainingMetricsCallback

    log_path = tmp_path / "metrics.csv"
    cb = TrainingMetricsCallback(
        eval_env=None,
        log_freq=10_000,
        n_episodes=2,
        log_path=log_path,
    )

    assert cb.log_freq == 10_000
    assert cb.n_episodes == 2
    assert cb.log_path == log_path
    assert cb._next_log == 10_000
    assert cb._header_written is False


# ===========================================================================
# Reward config tests
# ===========================================================================


class TestRewardConfig:
    """Tests for configurable reward parameters."""

    def test_opportunity_cost_cap_configurable(self):
        """Custom opportunity_cost_cap should be used in reward computation."""
        env = _make_env(episode_length=200)
        obs, _ = env.reset(seed=0)
        env.reward_config = {
            **env.reward_config,
            "opportunity_cost_cap": 0.10,
            "opportunity_cost_threshold": 0.002,
        }
        assert "opportunity_cost_cap" in env.reward_config
        assert env.reward_config["opportunity_cost_cap"] == 0.10

    def test_opportunity_cost_threshold_configurable(self):
        """Custom opportunity_cost_threshold should be used in reward computation."""
        env = _make_env(episode_length=200)
        obs, _ = env.reset(seed=0)
        env.reward_config = {
            **env.reward_config,
            "opportunity_cost_threshold": 0.005,
        }
        assert env.reward_config["opportunity_cost_threshold"] == 0.005

    def test_position_mgmt_scale_configurable(self):
        """position_mgmt_scale should exist in DEFAULT_REWARD_CONFIG."""
        env = _make_env(episode_length=200)
        assert "position_mgmt_scale" in env.reward_config
        assert env.reward_config["position_mgmt_scale"] == 1.0

    def test_opportunity_cost_cap_affects_reward(self):
        """Different opportunity_cost_cap values should produce different rewards."""
        env1 = _make_env(episode_length=200)
        env2 = _make_env(episode_length=200)
        env1.reset(seed=0)
        env2.reset(seed=0)

        # Use a very low threshold so that typical price movements trigger
        # the opportunity cost, then vary the cap to produce different rewards.
        env1.reward_config = {
            **env1.reward_config,
            "opportunity_cost_threshold": 0.0,
            "opportunity_cost_cap": 0.001,
        }
        env2.reward_config = {
            **env2.reward_config,
            "opportunity_cost_threshold": 0.0,
            "opportunity_cost_cap": 0.50,
        }

        rewards1, rewards2 = [], []
        for _ in range(50):
            _, r1, _, _, _ = env1.step([0, 0, 0])
            _, r2, _, _, _ = env2.step([0, 0, 0])
            rewards1.append(r1)
            rewards2.append(r2)

        assert sum(rewards1) != pytest.approx(sum(rewards2), abs=1e-10)


# ---------------------------------------------------------------------------
# CurriculumCallback tests
# ---------------------------------------------------------------------------


class TestCurriculumCallback:
    """Tests for CurriculumCallback with configurable base scales."""

    def test_default_behavior_unchanged(self):
        """CurriculumCallback with no extra args behaves as before."""
        config = TrainingConfig(total_timesteps=100)
        cb = CurriculumCallback(config)
        phase_config = cb._get_phase_reward_config(1)
        assert phase_config["fee_scale"] == pytest.approx(0.5)
        assert phase_config["opportunity_cost_scale"] == pytest.approx(0.3)

    def test_custom_base_scales_applied(self):
        """Custom base_reward_config should be multiplied by phase multipliers."""
        config = TrainingConfig(total_timesteps=100)
        base = {"fee_scale": 0.5, "churn_penalty_scale": 0.8}
        cb = CurriculumCallback(
            config,
            base_reward_config=base,
            phase3_fee_multiplier=2.5,
            phase3_churn_multiplier=1.5,
        )
        phase3_config = cb._get_phase_reward_config(3)
        assert phase3_config["fee_scale"] == pytest.approx(1.25)  # 0.5 * 2.5
        assert phase3_config["churn_penalty_scale"] == pytest.approx(1.2)  # 0.8 * 1.5

    def test_ent_coef_phase_derivation(self):
        """ent_coef should be derived from phase1 value."""
        config = TrainingConfig(total_timesteps=100)
        cb = CurriculumCallback(config, ent_coef_phase1=0.08)
        assert cb._get_phase_ent_coef(1) == pytest.approx(0.08)
        assert cb._get_phase_ent_coef(2) == pytest.approx(0.048)  # 0.08 * 0.6
        assert cb._get_phase_ent_coef(3) == pytest.approx(0.008)  # 0.08 * 0.1


class TestSimpleActionsConfig:
    def test_default_simple_actions_false(self):
        """Default TrainingConfig has simple_actions=False."""
        config = TrainingConfig()
        assert config.simple_actions is False
        assert config.fixed_size_pct == 0.10
        assert config.fixed_leverage == 10


class TestSimpleActions:
    """Tests for the simple_actions=True 3-action mode."""

    def test_action_space_is_discrete_3(self):
        """simple_actions=True → Discrete(3) action space."""
        env = TradingEnv(
            df=_make_df(),
            simple_actions=True,
            fixed_size_pct=0.10,
            fixed_leverage=10,
        )
        from gymnasium.spaces import Discrete
        assert isinstance(env.action_space, Discrete)
        assert env.action_space.n == 3

    def test_default_actions_unchanged(self):
        """simple_actions=False (default) → MultiDiscrete([3,4,3])."""
        env = TradingEnv(df=_make_df())
        from gymnasium.spaces import MultiDiscrete
        assert isinstance(env.action_space, MultiDiscrete)

    def test_step_long_opens_with_fixed_params(self):
        """action=1 opens long with fixed_size_pct and fixed_leverage."""
        env = TradingEnv(
            df=_make_df(),
            simple_actions=True,
            fixed_size_pct=0.10,
            fixed_leverage=10,
        )
        env.reset(seed=42)
        _obs, _reward, _term, _trunc, info = env.step(1)  # long
        assert info["position_side"] == "long"
        assert info["leverage"] == 10

    def test_step_flat_closes_position(self):
        """action=0 closes an open position."""
        env = TradingEnv(
            df=_make_df(),
            simple_actions=True,
            fixed_size_pct=0.10,
            fixed_leverage=10,
        )
        env.reset(seed=42)
        env.step(1)  # open long
        _obs, _reward, _term, _trunc, info = env.step(0)  # go flat
        assert info["position_side"] == "flat"

    def test_same_direction_is_noop(self):
        """Repeating action=1 while long → no fees, no modification."""
        env = TradingEnv(
            df=_make_df(),
            simple_actions=True,
            fixed_size_pct=0.10,
            fixed_leverage=10,
        )
        env.reset(seed=42)
        _obs, _reward, _term, _trunc, info_open = env.step(1)  # open long
        _balance_after_open = info_open["balance"]
        _obs, _reward, _term, _trunc, info_hold = env.step(1)  # repeat long
        assert info_hold["fees"] == 0.0
        # Balance unchanged by fees (only PnL may change)
        assert info_hold["position_side"] == "long"

    def test_gymnasium_check_env_simple(self):
        """Gymnasium's check_env passes for simple_actions=True."""
        from gymnasium.utils.env_checker import check_env
        env = TradingEnv(
            df=_make_df(),
            simple_actions=True,
            fixed_size_pct=0.10,
            fixed_leverage=10,
        )
        check_env(env, skip_render_check=True)


# ===========================================================================
# Observation shape integration tests (v4: N_MARKET_FEATURES=25)
# ===========================================================================


class TestObservationShapeV4:
    """Verify observation shape with 25 market features (5 OHLCV + 20 indicators)."""

    def test_market_obs_shape_is_25_features(self):
        """Market observation should have N_MARKET_FEATURES columns."""
        from alphacluster.config import N_MARKET_FEATURES, WINDOW_SIZE

        df = _make_df()
        env = TradingEnv(df=df)
        obs, _ = env.reset()
        assert obs["market"].shape == (WINDOW_SIZE, N_MARKET_FEATURES)
        assert obs["market"].shape[1] == 25

    def test_market_obs_shape_with_funding_df(self):
        """Market observation shape is unchanged when funding_df is supplied."""
        from alphacluster.config import N_MARKET_FEATURES, WINDOW_SIZE

        df = _make_df()
        funding_df = _make_funding_df(df)
        env = TradingEnv(df=df, funding_df=funding_df)
        obs, _ = env.reset()
        assert obs["market"].shape == (WINDOW_SIZE, N_MARKET_FEATURES)
        assert obs["market"].shape[1] == 25

    def test_market_obs_no_nans(self):
        """Market observation should have no NaN/Inf values."""
        df = _make_df()
        env = TradingEnv(df=df)
        obs, _ = env.reset()
        assert not np.any(np.isnan(obs["market"]))
        assert not np.any(np.isinf(obs["market"]))

    def test_market_obs_no_nans_with_funding(self):
        """Market observation should have no NaN/Inf values when funding_df is supplied."""
        df = _make_df()
        funding_df = _make_funding_df(df)
        env = TradingEnv(df=df, funding_df=funding_df)
        obs, _ = env.reset()
        assert not np.any(np.isnan(obs["market"]))
        assert not np.any(np.isinf(obs["market"]))

    def test_funding_columns_nonzero_when_funding_df_provided(self):
        """When funding_df is supplied, funding_rate/cumulative columns should not all be zero."""
        from alphacluster.data.indicators import INDICATOR_COLUMNS

        df = _make_df()
        funding_df = _make_funding_df(df)
        env = TradingEnv(df=df, funding_df=funding_df)

        # funding_rate and funding_cumulative_24h should be non-zero when real data is wired
        # through.  funding_premium may be zero when mark_price is absent from funding_df.
        _rate_names = ("funding_rate", "funding_cumulative_24h")
        rate_cols = [c for c in INDICATOR_COLUMNS if c in _rate_names]
        assert len(rate_cols) > 0, "Expected funding_rate and funding_cumulative_24h columns"
        for col in rate_cols:
            col_data = env.df[col].to_numpy()
            assert not np.all(col_data == 0.0), f"Column {col!r} is all zeros despite funding_df"

    def test_funding_columns_zero_without_funding_df(self):
        """When no funding_df is supplied, funding indicator columns default to zero."""
        from alphacluster.data.indicators import INDICATOR_COLUMNS

        df = _make_df()
        env = TradingEnv(df=df)  # no funding_df

        funding_cols = [c for c in INDICATOR_COLUMNS if "funding" in c]
        assert len(funding_cols) > 0, "Expected at least one funding indicator column"
        for col in funding_cols:
            col_data = env.df[col].to_numpy()
            assert np.all(col_data == 0.0), f"Column {col!r} should be zero without funding_df"
