# Training Improvement Roadmap

## Context

Optuna v1 (13 params, 40 trials) and v2 (11 params, 20 trials) both failed to find
stable trading configurations. Key findings:

- **v1:** 7/40 viable at 200k screening, 0/7 survived 500k validation
- **v2:** 1/20 viable at 200k screening, 0/1 survived 500k validation
- Win rate stuck at ~45-48% across all configurations
- Phase 3 escalation was killing activity, but removing it didn't help
- Problem is not in reward scales — it's in the RL formulation itself

## Diagnosis

The agent tries to learn **market prediction** and **trade execution** simultaneously
through a 36-action discrete space. This is too hard:

1. 36 actions (3 directions x 4 sizes x 3 leverages) — huge exploration space
2. Reward is delayed and sparse (only meaningful after trade closes)
3. 19 lagging indicators provide no forward-looking signal
4. 200k-500k steps is insufficient for RL on this problem complexity

## Roadmap — 4 Steps

### Step 1: Simplify Action Space (3 actions)

**Goal:** Answer "can the agent learn direction at all?"

Reduce to 3 actions: long / flat / short with fixed size (5%) and leverage (10x).
Same CNN+Transformer feature extractor, same reward function, same curriculum.
Only the action space changes.

**Why first:** If the agent can't learn direction with 3 actions, no amount of
features or architecture changes will help. This is the cheapest diagnostic test.

**Expected outcome:** Either the agent learns to trade (win rate > 50%, positive PnL)
confirming the problem was action space complexity, or it still fails, confirming
the problem is in observation/architecture.

**Effort:** ~4h (env changes + config + notebook update)

---

### Step 2: Additional Observation Features (Plan B Phase 1)

**Goal:** Give the agent fundamentally new information.

Add 6 features from the existing spec (`2026-03-24-additional-features-design.md`):
- Funding rate (3 features) — carry cost signal, uses already-downloaded data
- Volatility regime (3 features) — derived from existing indicators

**Why second:** Funding rate is the only fundamentally new data source (everything
else is derived from the same OHLCV). Vol regime gives context for position sizing.
These 6 features have the highest information-to-complexity ratio.

**Prerequisite:** Step 1 completed (we know whether simplified action space helps).

**Effort:** ~5h (indicators.py + tests + env integration)

---

### Step 3: Supervised Prediction Feature

**Goal:** Give the agent a forward-looking directional signal.

Train a simple supervised model (1D-CNN or LSTM) to predict next-N-step return
direction. Add its prediction as 1 additional feature in the RL observation space.

This separates "understanding market direction" (supervised, much easier to train)
from "deciding when/how to trade" (RL).

**Why third:** Current features are all lagging indicators — they tell the agent
where the market WAS, not where it's going. A supervised prediction provides
forward-looking signal that RL can learn to trust or ignore.

**Approach:**
1. Train a binary classifier: will close be higher/lower in 12 steps (1 hour)?
2. Output: probability (0.0 to 1.0)
3. Add as `direction_forecast` feature in observation
4. Train offline on historical data, freeze weights during RL training

**Prerequisite:** Steps 1-2 completed.

**Effort:** ~8-10h (model training + integration + evaluation)

---

### Step 4: Plan B Phase 2-3 (Multi-TF + Cross-Asset)

**Goal:** Enrich observation with higher-timeframe and cross-asset context.

From existing spec:
- Multi-timeframe (9 features) — 1h and 4h RSI, MACD, Bollinger, trend alignment
- Cross-asset (4 features) — ETH/BTC correlation, relative strength, lead signal

**Why last:** These are refinements, not fundamentals. Multi-TF resamples existing
data (no new information, just different presentation). Cross-asset requires new
data download (ETH). Both add complexity that only pays off if the agent already
knows how to trade.

**Prerequisite:** Steps 1-3 show improvement. If Step 1+2+3 still yields ~47%
win rate, these features won't help either.

**Effort:** ~10-12h (resampling logic + ETH download + merge + tests)

---

## Decision Points

After each step, evaluate before proceeding:

| After Step | Continue if... | Pivot if... |
|------------|----------------|-------------|
| 1 | Win rate > 50% OR clear learning trend | Still ~47%, flat collapse persists |
| 2 | Metrics improve over Step 1 baseline | No change — features aren't helping |
| 3 | Positive PnL on validation set | Supervised model itself has <52% accuracy |
| 4 | Marginal gains on top of Step 3 | Diminishing returns |

If Step 1 fails → consider fundamentally different approach (different timeframe,
different market, different RL algorithm).

If Step 3's supervised model has <52% accuracy → the data itself may not contain
sufficient predictive signal at 5-min frequency. Consider 15-min or 1h candles.
