"""Account state management for simulated perpetual-contract trading.

The :class:`Account` object is the single mutable state container used by
:class:`~alphacluster.env.trading_env.TradingEnv`.  It tracks balances,
positions, PnL, and trade history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alphacluster.config import TAKER_FEE
from alphacluster.env.mechanics import (
    apply_slippage,
    calculate_fee,
    calculate_funding,
    calculate_liquidation_price,
    calculate_pnl,
)


@dataclass
class Account:
    """Simulated trading account with a single open position at a time.

    Parameters
    ----------
    initial_balance:
        Starting USDT balance.
    """

    initial_balance: float = 10_000.0

    # ── State ────────────────────────────────────────────────────────────
    balance: float = field(init=False)
    position_side: str = field(init=False, default="flat")  # "flat" | "long" | "short"
    position_size: float = field(init=False, default=0.0)   # base-asset units
    entry_price: float = field(init=False, default=0.0)
    leverage: int = field(init=False, default=1)
    unrealized_pnl: float = field(init=False, default=0.0)
    peak_equity: float = field(init=False)
    time_in_position: int = field(init=False, default=0)    # candles

    # ── History ──────────────────────────────────────────────────────────
    trade_history: list[dict[str, Any]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.balance = self.initial_balance
        self.peak_equity = self.initial_balance

    # ── Helpers ──────────────────────────────────────────────────────────

    @property
    def equity(self) -> float:
        """Current equity = balance + unrealized PnL."""
        return self.balance + self.unrealized_pnl

    def _update_peak_equity(self) -> None:
        eq = self.equity
        if eq > self.peak_equity:
            self.peak_equity = eq

    # ── Public interface ─────────────────────────────────────────────────

    def open_position(
        self,
        side: str,
        size_pct: float,
        leverage: int,
        price: float,
        fee_rate: float = TAKER_FEE,
    ) -> float:
        """Open a new position.

        Parameters
        ----------
        side:
            ``"long"`` or ``"short"``.
        size_pct:
            Fraction of current balance to allocate (0.0–1.0).
        leverage:
            Leverage multiplier.
        price:
            Raw market price (slippage is applied internally).
        fee_rate:
            Fee rate to use (default: taker fee).

        Returns
        -------
        float
            Total fee paid for this trade.
        """
        if side not in ("long", "short"):
            raise ValueError(f"side must be 'long' or 'short', got {side!r}")
        if size_pct <= 0.0:
            return 0.0

        # Apply slippage
        buy_or_sell = "buy" if side == "long" else "sell"
        exec_price = apply_slippage(price, buy_or_sell)

        # Calculate position
        margin = self.balance * size_pct
        notional = margin * leverage
        size_units = notional / exec_price

        # Fee
        fee = calculate_fee(notional, fee_rate)
        self.balance -= fee

        # Record
        self.position_side = side
        self.position_size = size_units
        self.entry_price = exec_price
        self.leverage = leverage
        self.time_in_position = 0

        self.trade_history.append(
            {
                "action": "open",
                "side": side,
                "price": exec_price,
                "size": size_units,
                "notional": notional,
                "leverage": leverage,
                "fee": fee,
            }
        )
        return fee

    def close_position(self, price: float, fee_rate: float = TAKER_FEE) -> tuple[float, float]:
        """Close the current position.

        Parameters
        ----------
        price:
            Raw market price (slippage is applied internally).
        fee_rate:
            Fee rate to use.

        Returns
        -------
        tuple[float, float]
            ``(realized_pnl, fee)`` — the PnL *after* fee deduction is in
            ``realized_pnl``; ``fee`` is returned separately for bookkeeping.
        """
        if self.position_side == "flat":
            return 0.0, 0.0

        buy_or_sell = "sell" if self.position_side == "long" else "buy"
        exec_price = apply_slippage(price, buy_or_sell)

        pnl = calculate_pnl(self.entry_price, exec_price, self.position_size, self.position_side)
        notional = exec_price * self.position_size
        fee = calculate_fee(notional, fee_rate)

        self.balance += pnl - fee
        realized_pnl = pnl

        self.trade_history.append(
            {
                "action": "close",
                "side": self.position_side,
                "entry_price": self.entry_price,
                "exit_price": exec_price,
                "size": self.position_size,
                "pnl": pnl,
                "fee": fee,
            }
        )

        # Reset position state
        self.position_side = "flat"
        self.position_size = 0.0
        self.entry_price = 0.0
        self.leverage = 1
        self.unrealized_pnl = 0.0
        self.time_in_position = 0

        self._update_peak_equity()
        return realized_pnl, fee

    def update_unrealized_pnl(self, current_price: float) -> None:
        """Recalculate unrealized PnL at *current_price*."""
        if self.position_side == "flat":
            self.unrealized_pnl = 0.0
            return
        self.unrealized_pnl = calculate_pnl(
            self.entry_price, current_price, self.position_size, self.position_side
        )
        self._update_peak_equity()

    def apply_funding(self, funding_rate: float) -> float:
        """Apply a funding-rate payment/receipt.

        Returns the funding *cost* (positive = paid, negative = received).
        """
        if self.position_side == "flat":
            return 0.0
        position_value = self.entry_price * self.position_size
        if self.position_side == "short":
            position_value = -position_value
        cost = calculate_funding(position_value, funding_rate)
        self.balance -= cost
        return cost

    def is_liquidated(self, current_price: float) -> bool:
        """Return True if the position should be liquidated at *current_price*."""
        if self.position_side == "flat":
            return False
        liq_price = calculate_liquidation_price(
            self.entry_price, self.leverage, self.position_side
        )
        if self.position_side == "long":
            return current_price <= liq_price
        else:  # short
            return current_price >= liq_price

    def margin_ratio(self) -> float:
        """Return the current margin ratio (equity / initial margin).

        Returns ``float('inf')`` when there is no open position.
        """
        if self.position_side == "flat":
            return float("inf")
        initial_margin = self.entry_price * self.position_size / self.leverage
        if initial_margin == 0:
            return float("inf")
        return self.equity / initial_margin

    def reset(self) -> None:
        """Reset the account to its initial state."""
        self.balance = self.initial_balance
        self.position_side = "flat"
        self.position_size = 0.0
        self.entry_price = 0.0
        self.leverage = 1
        self.unrealized_pnl = 0.0
        self.peak_equity = self.initial_balance
        self.time_in_position = 0
        self.trade_history.clear()
