# Reward Function Rebalance: Opportunity Cost Design

## Problem

After implementing anti-churn reward shaping, the model overcorrected from churning (347 trades/episode, avg duration 1.3 steps) to complete paralysis (0 trades/episode, 100% flat). The combined effect of doubled fee penalty (x2.0), aggressive churn penalty (base=0.05, threshold=30), and disabled inactivity penalty made "do nothing" the optimal policy.

## Goal

- Target frequency: ~50-100 trades/episode (~7-14/day on 7-day episodes)
- Target avg duration: ~10-20 steps (50-100 min)
- Agent trades when market moves, stays flat during consolidation
- Fee/PnL ratio below 100%

## Design

### 1. Opportunity Cost Component (replaces Inactivity Penalty)

When the agent is flat, compute a penalty based on actual missed market movement:

```python
# When flat:
#   1. Compute rolling |price_change| over last 10 steps (50 min)
#   2. If |change| > 0.2% (noise threshold) -> missed opportunity
#   3. opportunity_penalty = scale * (|change| - 0.002)
```

**Rolling window (10 steps):** Filters single-candle noise. A 50-minute trend is a real move worth capturing.

**Threshold (0.2%):** At BTC ~80k this is ~$160. Smaller moves are noise, not missed opportunities. Agent is not penalized for being flat during sideways markets.

**No grace period:** Unlike the old inactivity penalty (arbitrary 20-step grace), the rolling window + threshold naturally ignores short flat periods between positions.

**Default scale:** `opportunity_cost_scale = 0.5` in `DEFAULT_REWARD_CONFIG` (replaces `inactivity_penalty_scale`).

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

- **PnL reward:** Asymmetric 1.5x for winners — OK
- **Position management:** Sqrt ramp for winners (coeff 0.4, cap 3.0) — OK
- **Trade completion:** Quadratic duration scaling — OK
- **Drawdown penalty:** Quadratic — OK
- **Quality bonus:** Per-step profitability for winners — OK

### 5. Curriculum Phases

New key `opportunity_cost_scale` replaces `inactivity_penalty_scale`.

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

### 6. DEFAULT_REWARD_CONFIG

```python
DEFAULT_REWARD_CONFIG: dict[str, float] = {
    "opportunity_cost_scale": 0.5,
    "fee_scale": 1.0,
    "drawdown_penalty_scale": 1.0,
    "churn_penalty_scale": 1.0,
    "quality_scale": 1.0,
}
```

## Files Changed

1. **`src/alphacluster/env/trading_env.py`** — Reward function: replace inactivity penalty with opportunity cost, remove fee x2.0, reduce churn base/threshold
2. **`src/alphacluster/agent/trainer.py`** — Curriculum phases: `opportunity_cost_scale` replaces `inactivity_penalty_scale`
3. **`tests/test_env.py`** — Updated and new tests

## Tests

### New tests
1. `test_opportunity_cost_when_flat_during_trend` — flat 15 steps during >0.2% trend, reward contains negative opportunity cost
2. `test_opportunity_cost_zero_during_consolidation` — flat but rolling change <0.2%, no penalty
3. `test_opportunity_cost_zero_when_in_position` — in position, opportunity cost = 0
4. `test_churn_penalty_rebalanced` — duration=1 -> penalty ~0.018, duration=20 -> penalty 0.0

### Modified tests
5. `test_reward_config_mutable` — `opportunity_cost_scale` instead of `inactivity_penalty_scale`
6. `test_inactivity_penalty_when_flat` — replaced with opportunity cost test (old component removed)

## Verification

1. `make test` — all tests pass
2. `make lint && make format` — code compliant with ruff
3. Training on 2M timesteps -> compare metrics
4. Expected: trades/ep ~50-100, avg duration 10-20, fee/PnL <100%
