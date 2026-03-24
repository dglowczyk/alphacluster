# Reward Function Rebalance: Opportunity Cost Design

## Problem

After implementing anti-churn reward shaping, the model overcorrected from churning (347 trades/episode, avg duration 1.3 steps) to complete paralysis (0 trades/episode, 100% flat). The combined effect of doubled fee penalty (x2.0), aggressive churn penalty (base=0.05, threshold=30), and completely disabled inactivity penalty (scale=0.0 in all phases) removed all incentive to trade. Since the inactivity penalty was the only mechanism nudging the agent toward trading, its removal — combined with amplified penalties — made "do nothing" the optimal policy.

## Goal

- Target frequency: ~50-100 trades/episode (~7-14/day on 7-day episodes)
- Target avg duration: ~10-20 steps (50-100 min)
- Agent trades when market moves, stays flat during consolidation
- Fee/PnL ratio below 100%

## Design

### 1. Opportunity Cost Component (new, replaces removed Inactivity Penalty)

The old inactivity penalty (disabled at scale=0.0) is replaced with a market-driven opportunity cost. When the agent is flat, it is penalized proportionally to the directional price movement it is missing.

**Precise formula:**

```python
opportunity_penalty = 0.0
if position_side == "flat":
    lookback = min(10, current_idx)  # handle episode start
    if lookback > 0:
        price_change = abs(close[t] - close[t - lookback]) / close[t - lookback]
        if price_change > 0.002:  # 0.2% noise threshold
            raw_penalty = price_change - 0.002
            opportunity_penalty = min(raw_penalty, 0.02) * rc["opportunity_cost_scale"]
```

**Variable definitions:**
- `price_change`: fractional absolute endpoint-to-endpoint price change over the lookback window. This measures directional trend, not volatility.
- `rc["opportunity_cost_scale"]`: from `reward_config`, controlled by curriculum (0.3 -> 0.5 -> 1.0).
- **Cap at 0.02**: prevents outsized penalties from data gaps, flash crashes, or exchange downtime. Max penalty per step = 0.02 * 1.0 (Phase 3) = 0.02.

**Rolling window (10 steps / 50 min):** Filters single-candle noise. A 50-minute trend is a real move worth capturing.

**Threshold (0.2%):** At BTC ~80k this is ~$160. Smaller moves are noise, not missed opportunities. Agent is not penalized for being flat during sideways markets.

**Episode start handling:** When `current_idx < 10`, `lookback` clamps to available history. First step has lookback=0, producing no penalty.

**No grace period:** Unlike the old inactivity penalty (arbitrary 20-step grace), the rolling window + threshold naturally ignores short flat periods between positions.

### 2. Fee Penalty — Remove x2.0 Multiplier

```python
# Was:  fee_penalty = (step_fees / initial_balance) * fee_scale * 2.0
# New:  fee_penalty = (step_fees / initial_balance) * fee_scale
```

The x2.0 multiplier was the primary reason every trade had negative EV from the agent's perspective. With fee_scale in curriculum (up to 2.0 in Phase 3), fees are still visible without the base doubling.

### 3. Churn Penalty — Reduced Magnitude

```python
# Was:  base=0.05, threshold=30, quadratic
# New:  base=0.02, threshold=20, quadratic
```

- Base 0.05 -> 0.02: Still 2x the original, but not 5x.
- Threshold 30 -> 20: A trade held 20+ steps (100 min) is not churn. Matches target avg duration of 10-20 steps.
- Quadratic form preserved (smooth gradient, no cliff).

Effective penalties:

| Duration | Old (base=0.05, th=30) | New (base=0.02, th=20) |
|----------|------------------------|------------------------|
| 1 step   | 0.045                  | 0.018                  |
| 5 steps  | 0.035                  | 0.011                  |
| 10 steps | 0.022                  | 0.005                  |
| 15 steps | 0.013                  | 0.001                  |
| 20 steps | 0.006                  | 0.000                  |

### 4. Unchanged Components

- **PnL reward:** Asymmetric 1.5x for winners
- **Position management:** Sqrt ramp for winners (coeff 0.4, cap 3.0)
- **Trade completion:** Quadratic duration scaling
- **Drawdown penalty:** Quadratic
- **Quality bonus:** Per-step profitability for winners

### 5. Full Reward Formula After Changes

```python
reward = (
    pnl_reward            # asymmetric PnL (1.5x winners)
    - fee_penalty          # step_fees / balance * fee_scale (no 2x)
    - opportunity_penalty  # replaces inactivity_penalty
    + position_reward      # sqrt ramp winners, escalating losers
    + completion_reward    # quadratic duration scaling
    - churn_penalty        # base=0.02, threshold=20, quadratic
    - dd_penalty           # quadratic drawdown
    + quality_bonus        # per-step profitability for winners
)
```

8 components total (same count as before — opportunity cost replaces inactivity penalty 1:1).

### 6. Curriculum Phases

New key `opportunity_cost_scale` replaces `inactivity_penalty_scale`. Phase boundaries: Phase 1 ends at 30%, Phase 2 ends at 60% (matching code in `config.py`: `phase1_end=0.3`, `phase2_end=0.6`).

**Phase 1 (0-30%): "Learn to Trade"**
```python
ent_coef = 0.05
reward_config = {
    "opportunity_cost_scale": 0.3,
    "fee_scale": 0.5,
    "drawdown_penalty_scale": 0.3,
    "churn_penalty_scale": 0.5,
    "quality_scale": 1.0,
}
```

**Phase 2 (30-60%): "Learn Quality"**
```python
ent_coef = 0.03
reward_config = {
    "opportunity_cost_scale": 0.5,
    "fee_scale": 1.0,
    "drawdown_penalty_scale": 1.0,
    "churn_penalty_scale": 1.0,
    "quality_scale": 1.0,
}
```

**Phase 3 (60-100%): "Refine & Exploit"**
```python
ent_coef = 0.005
reward_config = {
    "opportunity_cost_scale": 1.0,
    "fee_scale": 2.0,
    "drawdown_penalty_scale": 1.5,
    "churn_penalty_scale": 2.0,
    "quality_scale": 0.5,
}
```

**Phase 3 balance check:** fee_scale=2.0 and churn_scale=2.0 are less problematic now because base values are lower. Effective values:
- Fee: x2.0 (was x4.0 = 2.0 base x 2.0 scale)
- Churn at 1 step: 0.036 (was 0.09)

### 7. DEFAULT_REWARD_CONFIG

```python
DEFAULT_REWARD_CONFIG: dict[str, float] = {
    "opportunity_cost_scale": 0.5,
    "fee_scale": 1.0,
    "drawdown_penalty_scale": 1.0,
    "churn_penalty_scale": 1.0,
    "quality_scale": 1.0,
}
```

### 8. Worked Example: Reward Magnitudes

Assume Phase 2 defaults (all scales = 1.0, opportunity_cost_scale = 0.5), initial_balance = $10,000, BTC at $80,000.

**Scenario A: Good trade (15-step long, +0.3% profit)**
- PnL reward: (30 / 10000) * 1.5 = +0.0045
- Fee penalty: ~(0.8 / 10000) * 1.0 = -0.00008
- Opportunity cost: 0 (in position)
- Position reward: 0.4 * 0.003 * (1 + sqrt(15)/5) = ~+0.0017
- Completion: 0.01 * 0.003 * min((15/10)^2, 10) = +0.000068
- Churn: 0.02 * (1 - 15/20)^2 = 0.00013 (minimal)
- Quality bonus: min(20 * 0.003/15, 0.02) * 1.0 = +0.004
- **Net: ~+0.0100** (positive)

**Scenario B: Churn trade (1-step long, +0.01% profit)**
- PnL reward: (1 / 10000) * 1.5 = +0.00015
- Fee penalty: ~(0.8 / 10000) * 1.0 = -0.00008
- Completion: 0.01 * 0.0001 * min(0.01, 10) = ~0
- Churn: 0.02 * (1 - 1/20)^2 = 0.018 (dominant!)
- **Net: ~-0.018** (strongly negative)

**Scenario C: Flat during 0.5% trend (10 steps)**
- All trade-related components: 0
- Opportunity cost: min(0.005 - 0.002, 0.02) * 0.5 = 0.0015 per step
- Over 10 steps: ~0.015 cumulative
- **Net: ~-0.015** (negative, nudges toward trading)

**Takeaway:** Good trades are rewarded (~+0.01), churn is punished (~-0.018), flat during trends is penalized (~-0.0015/step). The agent should prefer good trades > flat > churn.

## Files Changed

1. **`src/alphacluster/env/trading_env.py`** — Reward function: replace inactivity penalty with opportunity cost, remove fee x2.0, reduce churn base/threshold
2. **`src/alphacluster/agent/trainer.py`** — Curriculum phases: `opportunity_cost_scale` replaces `inactivity_penalty_scale`
3. **`tests/test_env.py`** — Updated and new tests
4. **`CLAUDE.md`** — Update phase boundaries description (30/60 not 30/70) and reward component list

## Tests

### New tests
1. `test_opportunity_cost_when_flat_during_trend` — flat 15 steps during >0.2% trend, reward contains negative opportunity cost
2. `test_opportunity_cost_zero_during_consolidation` — flat but rolling change <0.2%, no penalty
3. `test_opportunity_cost_zero_when_in_position` — in position, opportunity cost = 0
4. `test_opportunity_cost_at_episode_start` — first 10 steps, lookback clamps to available history, no crash
5. `test_opportunity_cost_capped` — during 5% flash crash, penalty capped at 0.02 * scale
6. `test_opportunity_cost_exact_threshold` — |change| == 0.002 exactly produces 0 penalty (strictly greater)
7. `test_churn_penalty_rebalanced` — duration=1 -> ~0.018, duration=10 -> ~0.005, duration=20 -> 0.0
8. `test_trading_beats_flat_during_trend` — net reward for opening a position during trend > staying flat (integration test)

### Modified tests
9. `test_reward_config_mutable` — `opportunity_cost_scale` instead of `inactivity_penalty_scale`
10. `test_inactivity_penalty_when_flat` — replaced with opportunity cost tests (old component removed)

## Verification

1. `make test` — all tests pass
2. `make lint && make format` — code compliant with ruff
3. Training on 2M timesteps -> compare metrics
4. Expected: trades/ep ~50-100, avg duration 10-20, fee/PnL <100%
