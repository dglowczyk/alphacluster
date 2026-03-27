# Smart Optuna v3 — Two-Stage Model Improvement

**Date:** 2026-03-27
**Status:** Draft
**Goal:** Improve v3 model from -0.47%/episode to +4% daily (+28%/episode)

## Context

Model v3 (`ppo_trading_simple_final_v3.pt`) trained on 500k steps with default parameters
outperformed all 25 Optuna trials (200-500k steps each). Key insight: **training duration
matters more than hyperparameter tuning at short horizons**. Previous Optuna produced
high-frequency churners (130-788 trades/ep) while v3's patient strategy (9 trades/ep,
223-step avg duration) was far superior.

### Current v3 Performance (5 episodes, test set, seed=42)

- PnL: -0.47% avg/episode
- Win rate: 51.1%, Profit factor: 1.04
- Trades/episode: 9, Avg duration: 223 steps
- Sharpe: -1.02, Max drawdown: 15%
- Leverage: always 10x, Size: always 10%

## Design

### Stage 1 — Extended Training (warm-start)

Resume training from v3 checkpoint with identical parameters:

| Parameter | Value |
|---|---|
| total_timesteps | 2,500,000 (2M new steps from 500k checkpoint) |
| warm_start | `models/ppo_trading_simple_final_v3.pt` |
| simple_actions | True |
| fixed_size_pct | 0.10 |
| fixed_leverage | 10 |
| All reward/curriculum | v3 defaults (see below) |

**v3 default parameters (frozen):**
- fee_scale=1.0, churn_penalty_scale=1.0, drawdown_penalty_scale=1.0
- opportunity_cost_scale=1.0, opportunity_cost_cap=0.02, opportunity_cost_threshold=0.002
- quality_scale=1.0, position_mgmt_scale=1.0
- learning_rate=3e-4, gamma=0.995, ent_coef=0.05
- phase1_end=0.3, phase2_end=0.6
- phase3_fee_multiplier=2.0, phase3_churn_multiplier=2.0

**Curriculum note:** Since we're warm-starting at 500k into a 2.5M run, the model
resumes at 20% progress — still in Phase 1. Phase transitions:
- Phase 1 (0-30%): 0-750k steps
- Phase 2 (30-60%): 750k-1.5M steps
- Phase 3 (60-100%): 1.5M-2.5M steps

**Output:** `models/ppo_trading_simple_v3_2.5M.pt` + evaluation report.

**Estimated GPU time:** ~2.5h on L4.

### Stage 2 — Smart Optuna

Run Optuna with narrow parameter space centered on v3 defaults. Key differences
from previous Optuna run:
1. Longer training per trial (1M vs 200k)
2. Fewer, more important parameters (7 vs 12)
3. v3 defaults as seeded trial
4. Better scoring metric (PnL-focused)
5. Position size and leverage as tunable hyperparameters

#### Search Space (7 parameters)

| Parameter | Type | Range | v3 Default |
|---|---|---|---|
| fee_scale | float | [0.3, 2.0] | 1.0 |
| ent_coef_phase1 | float | [0.03, 0.20] | 0.05 |
| phase1_end | float | [0.2, 0.5] | 0.3 |
| phase3_fee_multiplier | float | [0.5, 3.0] | 2.0 |
| phase3_churn_multiplier | float | [0.5, 3.0] | 2.0 |
| fixed_size_pct | categorical | [0.02, 0.05, 0.10, 0.15] | 0.10 |
| fixed_leverage | categorical | [5, 10, 15] | 10 |

#### Frozen Parameters

These are NOT tuned (low importance or no reason to change):
- learning_rate=3e-4, gamma=0.995 (PPO stable)
- churn_penalty_scale=1.0, drawdown_penalty_scale=1.0 (importance <0.03)
- quality_scale=1.0, position_mgmt_scale=1.0 (importance <0.02)
- opportunity_cost_scale=1.0, opportunity_cost_cap=0.02, opportunity_cost_threshold=0.002

#### Optuna Configuration

- **Sampler:** TPESampler with trial 0 = v3 defaults (seeded)
- **Pruner:** MedianPruner — prune after 500k steps if score below median
- **Training per trial:** 1,000,000 steps (from scratch, no warm-start)
- **Number of trials:** 15 (1 seeded + 14 sampled)
- **Training data:** first 70% of candles (chronological)

#### Scoring Function

Primary metric: **mean PnL% per episode** with hard filters.

```
Hard rejection (score = -1000):
  - trades_per_episode < 5
  - win_rate < 40%
  - flat_pct > 70%
  - avg_trade_duration < 1.5 steps

Score = mean_pnl_pct (no composite weighting)
```

Rationale: Previous composite score (0.4×pnl + 0.3×trades + 0.3×win_rate) rewarded
high trade count, producing churners. Pure PnL with sensible filters aligns the
optimizer with the actual goal.

#### Validation Phase

- **Top 5 screening trials** retrained on **2,000,000 steps** (from scratch)
- **Evaluation:** 5 episodes on held-out test set (last 20%), seed=42
- **Comparison:** vs Stage 1 model (v3_2.5M) and original v3 (500k)
- **Rejection:** trials where validation PnL drops >50% vs screening = overfit

### Estimated GPU Budget

| Phase | Trials | Steps | Time/trial | Total |
|---|---|---|---|---|
| Stage 1 (warm-start) | 1 | 2.5M | ~2.5h | ~2.5h |
| Stage 2 screening | 15 | 1M | ~1h | ~15h |
| Stage 2 validation | 5 | 2M | ~2h | ~10h |
| **Total** | | | | **~27.5h** |

### Deliverable

One Colab notebook: `notebooks/colab_optuna_v3_smart.ipynb`

**Cells:**
1. Setup — mount Drive, install deps, copy source
2. Load data — parquet + chronological split (70/15/15)
3. Stage 1 — warm-start v3 to 2.5M steps
4. Stage 1 — evaluate on test set, print report
5. Stage 2 — define Optuna study (objective, search space, sampler, pruner)
6. Stage 2 — run screening (15 trials x 1M)
7. Stage 2 — validate top 5 (retrain 2M)
8. Results — comparison table, plots, save best model

**Outputs saved to Drive:**
- `models/ppo_trading_simple_v3_2.5M.pt` (Stage 1)
- `reports/optuna_smart_v3/` (trial results, validation, plots, best_params.json)
- `models/ppo_trading_smart_v3_best.pt` (Stage 2 winner)
