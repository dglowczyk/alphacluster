"""Pure trading-mechanics helper functions.

All functions are stateless — they take numeric inputs and return numeric
outputs.  They are used by :class:`Account` and :class:`TradingEnv` to keep
business logic isolated and easily testable.
"""

from __future__ import annotations

from alphacluster.config import MAKER_FEE, TAKER_FEE


# ── Fee calculation ──────────────────────────────────────────────────────────

def calculate_fee(notional_value: float, fee_rate: float | None = None) -> float:
    """Return the trading fee for a given *notional_value*.

    Parameters
    ----------
    notional_value:
        ``price * size`` — always positive.
    fee_rate:
        Override for the fee rate.  If *None*, the taker rate
        (:data:`TAKER_FEE`) is used.

    Returns
    -------
    float
        The absolute fee amount (always >= 0).
    """
    if fee_rate is None:
        fee_rate = TAKER_FEE
    return abs(notional_value) * fee_rate


# ── Funding rate ─────────────────────────────────────────────────────────────

def calculate_funding(position_value: float, funding_rate: float) -> float:
    """Return the funding payment for a position.

    A **positive** return value means the holder *pays*; negative means the
    holder *receives*.

    Parameters
    ----------
    position_value:
        Signed position value (positive for long, negative for short).
    funding_rate:
        The funding rate for the period (e.g. 0.0001 = 0.01 %).
    """
    return position_value * funding_rate


# ── Liquidation price ────────────────────────────────────────────────────────

_MAINTENANCE_MARGIN_RATE = 0.005  # 0.5 %


def calculate_liquidation_price(
    entry_price: float,
    leverage: int | float,
    side: str,
    maintenance_margin_rate: float = _MAINTENANCE_MARGIN_RATE,
) -> float:
    """Return the estimated liquidation price.

    Uses the simplified formula:
        liq_price = entry * (1 -/+ (1/leverage - maintenance_margin_rate))

    Parameters
    ----------
    entry_price:
        The average entry price of the position.
    leverage:
        The leverage multiplier (e.g. 10).
    side:
        ``"long"`` or ``"short"``.
    maintenance_margin_rate:
        Maintenance-margin ratio (default 0.5 %).

    Returns
    -------
    float
        The estimated liquidation price (always > 0).
    """
    margin_distance = 1.0 / leverage - maintenance_margin_rate
    if side == "long":
        return entry_price * (1.0 - margin_distance)
    elif side == "short":
        return entry_price * (1.0 + margin_distance)
    else:
        raise ValueError(f"side must be 'long' or 'short', got {side!r}")


# ── PnL ──────────────────────────────────────────────────────────────────────

def calculate_pnl(
    entry_price: float,
    current_price: float,
    size: float,
    side: str,
) -> float:
    """Return the unrealized PnL for a position.

    Parameters
    ----------
    entry_price:
        Average entry price.
    current_price:
        Latest mark / last price.
    size:
        Position size in base-asset units (always positive).
    side:
        ``"long"`` or ``"short"``.

    Returns
    -------
    float
        Positive means profit, negative means loss.
    """
    if side == "long":
        return (current_price - entry_price) * size
    elif side == "short":
        return (entry_price - current_price) * size
    else:
        raise ValueError(f"side must be 'long' or 'short', got {side!r}")


# ── Slippage ─────────────────────────────────────────────────────────────────

_DEFAULT_SLIPPAGE_PCT = 0.0001  # 0.01 %


def apply_slippage(
    price: float,
    side: str,
    slippage_pct: float = _DEFAULT_SLIPPAGE_PCT,
) -> float:
    """Return *price* adjusted for simulated slippage.

    For a **buy** (long entry / short exit) the price moves *up*; for a
    **sell** (short entry / long exit) it moves *down*.

    Parameters
    ----------
    price:
        The raw price before slippage.
    side:
        ``"buy"`` or ``"sell"``.
    slippage_pct:
        Slippage as a fraction (e.g. 0.0001 = 0.01 %).
    """
    if side == "buy":
        return price * (1.0 + slippage_pct)
    elif side == "sell":
        return price * (1.0 - slippage_pct)
    else:
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
