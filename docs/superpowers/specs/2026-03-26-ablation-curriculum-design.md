# Ablation Sweep + Slower Curriculum Design

## Problem

Model v4 (25 features) collapses to always-flat policy by 250k steps, while v3 (19 features) with the same reward function learns a viable trading strategy (12 trades/ep, 49% WR, PF 1.58).

Root cause: 25-feature observation space slows direction learning. The agent doesn't learn profitable trading before curriculum Phase 2 (at 30%) escalates penalties, pushing it past the collapse threshold into always-flat.

## Solution

Two changes:

1. **Slower curriculum** — extend Phase 1 from 30% to 50% of training, giving the agent more time to learn directional signal with reduced penalties before escalation.

2. **Ablation sweep** — 4 reward configurations run sequentially in one notebook, each 1M steps. Identifies which reward components cause collapse and finds a stable configuration for 25 features.

## Curriculum Changes

| Phase | Current | New | Purpose |
|-------|---------|-----|---------|
| 1 "Learn to Trade" | 0–30% | **0–50%** | More time to learn direction |
| 2 "Learn Quality" | 30–60% | **50–80%** | Shorter, focused quality phase |
| 3 "Refine & Exploit" | 60–100% | **80–100%** | Short exploitation phase |

`TrainingConfig` already has configurable `phase1_end` and `phase2_end` fields (defaults 0.3/0.6). `CurriculumCallback._get_phase()` already reads these. No code change needed for phase boundaries — just pass different values when constructing `TrainingConfig`:

```python
config = TrainingConfig(phase1_end=0.5, phase2_end=0.8, ...)
```

## Ablation Configurations

All configs use: simple_actions=True (3 actions), 25 features (v4), 1M steps, slow curriculum (phase1_end=0.5, phase2_end=0.8), same random seed (42).

**Note on trade completion reward:** The trade completion reward component (lines 406-415 in trading_env.py) has no configurable scale and is always active. This means even the "pnl_only" config includes a small bonus/penalty on trade close. This is acceptable — it's a weak signal that won't dominate the ablation results, but should be noted when interpreting results.

### Config 1: `pnl_only`

Only asymmetric PnL reward + trade completion (always active). All configurable penalties disabled.

```python
base_reward_config = {
    "fee_scale": 0.0,
    "churn_penalty_scale": 0.0,
    "opportunity_cost_scale": 0.0,
    "drawdown_penalty_scale": 0.0,
    "quality_scale": 0.0,
    "position_mgmt_scale": 0.0,
}
```

**Diagnostic question**: Can the agent learn directional signal at all with 25 features?

### Config 2: `pnl_fees`

PnL + fee penalty only. Fee gradually introduced via curriculum.

```python
base_reward_config = {
    "fee_scale": 0.5,             # Phase 1: 0.5×0.5=0.25, P2: 0.5×1.0=0.5, P3: 0.5×2.0=1.0
    "churn_penalty_scale": 0.0,
    "opportunity_cost_scale": 0.0,
    "drawdown_penalty_scale": 0.0,
    "quality_scale": 0.0,
    "position_mgmt_scale": 0.0,
}
```

**Diagnostic question**: At what fee level does the agent remain stable?

### Config 3: `pnl_fees_opp`

PnL + fees + strengthened opportunity cost. Opportunity cost cap raised from 0.02 to 0.05, threshold lowered from 0.2% to 0.1%.

```python
base_reward_config = {
    "fee_scale": 0.5,
    "churn_penalty_scale": 0.0,
    "opportunity_cost_scale": 1.0,
    "opportunity_cost_cap": 0.05,       # was 0.02
    "opportunity_cost_threshold": 0.001, # was 0.002
    "drawdown_penalty_scale": 0.0,
    "quality_scale": 0.0,
    "position_mgmt_scale": 0.0,
}
```

**Diagnostic question**: Does stronger opportunity cost prevent flat collapse?

### Config 4: `full_slow`

Full 8-component reward with slower curriculum. Uses env defaults for all scales (opportunity_cost_scale=0.5 from DEFAULT_REWARD_CONFIG).

```python
base_reward_config = {
    "opportunity_cost_scale": 0.5,    # match DEFAULT_REWARD_CONFIG
    "opportunity_cost_cap": 0.02,
    "opportunity_cost_threshold": 0.002,
}
# All other keys default to 1.0 in CurriculumCallback._DEFAULT_BASE
```

**Diagnostic question**: Is slower curriculum alone sufficient?

## Code Changes

### 1. `src/alphacluster/agent/trainer.py` — Pass `base_reward_config` to CurriculumCallback from `train()`

The `train()` function currently creates `CurriculumCallback(config, verbose=verbose)` without passing `base_reward_config`. Add an optional `base_reward_config` parameter to `train()`:

```python
def train(
    agent,
    config,
    eval_env=None,
    ...,
    base_reward_config: dict[str, float] | None = None,
) -> PPO:
    ...
    if config.curriculum_enabled:
        callbacks.append(
            CurriculumCallback(config, verbose=verbose, base_reward_config=base_reward_config)
        )
```

This is the only source code change needed. Everything else (phase boundaries, simple_actions) is already configurable.

### 2. `notebooks/colab_ablation.ipynb` — Sweep notebook

Structure:
- Cell 1: Mount Drive, install deps
- Cell 2: Load data, split (70/15/15)
- Cell 3: Define `ABLATION_CONFIGS` list (each with name, total_timesteps, phase boundaries, base_reward_config)
- Cell 4: Sweep loop — for each config: create env, create agent, train 1M, save metrics CSV + model to Drive
- Cell 5: Comparison plots — overlay all 4 configs on same axes (trades/ep, win rate, flat%, reward, PnL)
- Cell 6: Summary table with final metrics per config

**Drive persistence**: Metrics saved directly to `Drive/alphacluster/ablation_results/{config_name}/training_metrics.csv`. Models saved to `Drive/alphacluster/ablation_results/{config_name}/model.pt`.

**TrainingMetricsCallback log_path**: Each run passes a Drive-based path to `TrainingMetricsCallback` via the checkpoint_dir parameter (the callback already saves to `{checkpoint_dir}/training_metrics.csv`).

## Expected Outcomes

| Config | Expected behavior | If fails |
|--------|-------------------|----------|
| `pnl_only` | Agent trades actively, may overtrade | Agent can't learn direction → architecture issue with 25 features |
| `pnl_fees` | Trades less, holds longer | Collapses to flat → fee sensitivity too high, need fee-free warmup |
| `pnl_fees_opp` | Maintains trading even with fees | Collapses → opportunity cost mechanism needs redesign |
| `full_slow` | Like v3 but with 25 features | Collapses → need simpler reward, not just more time |

## Success Criteria

At least one config produces:
- Trades/episode > 5 at end of training
- Win rate > 45%
- No collapse to always-flat
- Stable or improving metrics in last 25% of training

## Not In Scope

- Changes to the reward function structure (only scales/weights)
- Changes to the CNN+Transformer architecture
- Multi-timeframe or cross-asset features
- Full 36-action space (remains simple_actions=True)
- Multiple seeds per config (single seed=42 for all runs)
