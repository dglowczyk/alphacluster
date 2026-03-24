# Optuna Hyperparameter Tuning — Design Spec

## Problem

The RL agent consistently degenerates into 100% flat (no trading) during training.
Root cause: fee and churn penalties dominate opportunity cost and quality rewards,
so the agent learns that "not trading" is the optimal strategy.

Training metrics show clear progression:
- 50k-150k: active trading, chaotic PnL, fee_to_pnl_ratio already 200-300%
- 200k-800k: trade duration collapses to ~2 steps, flat_pct rises to 64%
- 850k-1M: full capitulation — 0 trades, 100% flat

Manual scale tuning is impractical with 11 interacting parameters.

## Solution

A Colab notebook using **Optuna (TPE sampler + MedianPruner)** for automated
Bayesian hyperparameter optimization of reward scales, curriculum multipliers,
and PPO hyperparameters.

## Parameter Space (11 dimensions)

### Reward Scales (6)

| Parameter | Range | Scale | Current Default |
|-----------|-------|-------|-----------------|
| `fee_scale` | 0.1 – 2.0 | uniform | 1.0 |
| `opportunity_cost_scale` | 0.1 – 2.0 | uniform | 0.5 |
| `opportunity_cost_cap` | 0.01 – 0.15 | uniform | 0.02 (hardcoded) |
| `churn_penalty_scale` | 0.1 – 2.0 | uniform | 1.0 |
| `drawdown_penalty_scale` | 0.1 – 2.0 | uniform | 1.0 |
| `quality_scale` | 0.1 – 2.0 | uniform | 1.0 |

Note: `opportunity_cost_cap` is currently hardcoded in `trading_env.py` (line ~362).
It must be made configurable via `reward_config` dict to be tunable.

### Curriculum Phase 3 Multipliers (2)

| Parameter | Range | Scale | Current Default |
|-----------|-------|-------|-----------------|
| `phase3_fee_multiplier` | 1.0 – 3.0 | uniform | 2.0 |
| `phase3_churn_multiplier` | 1.0 – 3.0 | uniform | 2.0 |

These multiply the base `fee_scale` and `churn_penalty_scale` in Phase 3.
Phase 1 and Phase 2 multipliers remain fixed (Phase 1: 0.5x, Phase 2: 1.0x).

### PPO Hyperparameters (3)

| Parameter | Range | Scale | Current Default |
|-----------|-------|-------|-----------------|
| `learning_rate` | 1e-4 – 1e-3 | log-uniform | 3e-4 |
| `ent_coef` | 0.01 – 0.1 | log-uniform | 0.05 |
| `gamma` | 0.99 – 0.999 | uniform | 0.995 |

## Objective Function

### Composite Score

```python
score = total_pnl_pct * 0.4 + trades_norm * 0.3 + win_rate_norm * 0.3
```

Where:
- `total_pnl_pct` — raw total PnL percentage from backtest
- `trades_norm` — `min(trades_per_episode, 200) / 200` (normalized 0-1)
- `win_rate_norm` — `win_rate / 100` (normalized 0-1)

### Hard Constraints (trial rejected → score = -1000)

- `flat_pct > 80%` — agent has collapsed into flat
- `trades_per_episode < 10` — insufficient trading activity

### Evaluation

Score is computed from `TrainingMetricsCallback` output at the final timestep.
Uses `n_episodes=3` with `seed=42` for deterministic comparison.

## Two-Phase Strategy

### Phase 1: Screening (200k steps × 40 trials)

- **Goal:** Rapidly eliminate parameter configurations that lead to flat collapse
- **Timesteps:** 200,000 (~12 min per trial on Colab T4)
- **Trials:** 40
- **Estimated time:** ~8 hours
- **Pruning:** `MedianPruner(n_startup_trials=10, n_warmup_steps=100000)`
  - First 10 trials run to completion (build baseline)
  - After 10 trials, prune at 100k if intermediate score < median
- **Intermediate reporting:** Score reported at 100k and 200k steps via
  Optuna `trial.report()` + `trial.should_prune()`

### Phase 2: Validation (500k steps × top 10)

- **Goal:** Verify that screening winners actually produce viable strategies
- **Input:** Top 10 parameter sets from Phase 1 (by final score)
- **Timesteps:** 500,000 (~30 min per trial)
- **Trials:** 10 (fixed params, no Optuna sampling)
- **Estimated time:** ~5 hours
- **No pruning** — all trials run to completion

### Total estimated time: ~13 hours

## Notebook Structure

### Cell 1 — Setup & Dependencies

```
!pip install optuna optuna-dashboard plotly kaleido
```

Mount Google Drive. Define paths:
- `DRIVE_DIR = "/content/drive/MyDrive/AlphaCluster/optuna_tuning/"`
- `STUDY_DB = DRIVE_DIR + "optuna_study.db"` (SQLite, Optuna native persistence)
- `RESULTS_CSV = DRIVE_DIR + "trial_results.csv"`
- `BEST_PARAMS_JSON = DRIVE_DIR + "best_params.json"`

### Cell 2 — Install AlphaCluster & Load Data

Clone repo, install package, load and split data (same as colab_train.ipynb).

### Cell 3 — Define Objective Function

```python
def objective(trial: optuna.Trial) -> float:
    # Sample 11 parameters
    fee_scale = trial.suggest_float("fee_scale", 0.1, 2.0)
    opportunity_cost_scale = trial.suggest_float("opportunity_cost_scale", 0.1, 2.0)
    opportunity_cost_cap = trial.suggest_float("opportunity_cost_cap", 0.01, 0.15)
    churn_penalty_scale = trial.suggest_float("churn_penalty_scale", 0.1, 2.0)
    drawdown_penalty_scale = trial.suggest_float("drawdown_penalty_scale", 0.1, 2.0)
    quality_scale = trial.suggest_float("quality_scale", 0.1, 2.0)
    phase3_fee_mult = trial.suggest_float("phase3_fee_multiplier", 1.0, 3.0)
    phase3_churn_mult = trial.suggest_float("phase3_churn_multiplier", 1.0, 3.0)
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    ent = trial.suggest_float("ent_coef", 0.01, 0.1, log=True)
    gamma = trial.suggest_float("gamma", 0.99, 0.999)

    # Build reward_config for each curriculum phase
    # Phase 1: base scales × 0.5 (fee, churn), 0.3 (opp, dd)
    # Phase 2: base scales × 1.0
    # Phase 3: base scales × phase3 multipliers

    # Create env with sampled reward_config
    # Create PPO agent with sampled lr, ent_coef, gamma
    # Train for N timesteps with pruning callback

    # Read training_metrics.csv, compute score
    # Apply hard constraints
    # Return score
```

### Cell 4 — Pruning Callback

Custom SB3 callback that:
1. Reads `TrainingMetricsCallback` CSV at checkpoints (100k, 200k)
2. Computes intermediate score
3. Reports to `trial.report(score, step)`
4. Raises `optuna.TrialPruned()` if `trial.should_prune()`

### Cell 5 — Phase 1: Screening

```python
study = optuna.create_study(
    study_name="alphacluster_reward_tuning",
    direction="maximize",
    storage=f"sqlite:///{STUDY_DB}",
    load_if_exists=True,  # Resume if interrupted
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=100_000,
    ),
    sampler=optuna.samplers.TPESampler(seed=42),
)
study.optimize(objective, n_trials=40, timeout=None)
```

Key: `load_if_exists=True` allows resuming if Colab disconnects.

### Cell 6 — Phase 1 Results & Analysis

- Print top 10 trials with scores and parameters
- Save results to `RESULTS_CSV`
- Optuna visualizations: parameter importances, optimization history, slice plots
- Save plots to Drive as PNG

### Cell 7 — Phase 2: Validation

```python
# Extract top 10 param sets from Phase 1
top_params = sorted(study.trials, key=lambda t: t.value or -inf, reverse=True)[:10]

# Run each with 500k timesteps (no pruning)
validation_results = []
for i, trial in enumerate(top_params):
    result = run_validation_trial(trial.params, timesteps=500_000)
    validation_results.append(result)
    # Save incrementally to Drive
```

### Cell 8 — Final Results & Export

- Rank validated configurations by 500k score
- Save `best_params.json` with winning configuration
- Print recommended reward_config for production training
- Save full study DB and all results to Drive
- Summary table: parameter values, Phase 1 score, Phase 2 score, key metrics

### Cell 9 — Apply Best Parameters (Optional)

Template cell to copy best params into a training run:
```python
# Copy-paste these into your training config
best_reward_config = json.load(open(BEST_PARAMS_JSON))
print(f"Recommended config:\n{json.dumps(best_reward_config, indent=2)}")
```

## Required Code Changes

### 1. Make `opportunity_cost_cap` configurable

In `trading_env.py`, change:
```python
# Before (hardcoded):
opportunity_penalty = min(raw_penalty, 0.02) * rc["opportunity_cost_scale"]

# After (configurable):
opp_cap = rc.get("opportunity_cost_cap", 0.02)
opportunity_penalty = min(raw_penalty, opp_cap) * rc["opportunity_cost_scale"]
```

### 2. Make curriculum phases accept dynamic multipliers

The `CurriculumCallback` currently hardcodes phase multipliers. It needs to
accept base scales and phase multipliers as constructor parameters so the
objective function can override them.

### 3. TrainingMetricsCallback — expose via module __init__

Already implemented, just ensure it's importable from the notebook.

## Persistence & Recovery

All artifacts saved to Google Drive for durability:

| Artifact | Path | Purpose |
|----------|------|---------|
| Optuna study DB | `optuna_study.db` | Full trial history, enables resume |
| Trial results CSV | `trial_results.csv` | Human-readable results table |
| Best params JSON | `best_params.json` | Winning config for production |
| Phase 1 plots | `plots/phase1_*.png` | Parameter importance, optimization history |
| Phase 2 results | `validation_results.csv` | 500k validation scores |

`load_if_exists=True` on the Optuna study enables seamless resume after
Colab disconnection — the study picks up from the last completed trial.

## Dependencies

New pip packages for the tuning notebook:
- `optuna>=3.0` — Bayesian optimization framework
- `optuna-dashboard` — optional, for interactive visualization
- `plotly` — Optuna's built-in visualization backend
- `kaleido` — static image export for Plotly charts

All other dependencies (SB3, torch, etc.) already present.

---

## Appendix: Plan B — Additional Features (future work)

Documented here for future implementation. Not part of current scope.

### 1. Funding Rate (high priority — data already collected)

- Binance funding rate is already downloaded by `downloader.py` but not
  included in the observation space
- Add as feature in `indicators.py`: raw funding rate + 24h cumulative funding
- Impact: helps agent understand carry cost and predict liquidation cascades

### 2. Multi-Timeframe Features (medium priority)

- Resample 5min candles to 1h and 4h within `indicators.py`
- Compute RSI_14, MACD_hist, BB_%B for each higher timeframe
- Add 6 features: `rsi_1h`, `macd_hist_1h`, `bb_pctb_1h`, `rsi_4h`,
  `macd_hist_4h`, `bb_pctb_4h`
- Impact: gives agent explicit trend context without relying on long lookback

### 3. Cross-Asset Correlations (lower priority — needs new data pipeline)

- Download ETHUSDT 5min candles (same pipeline, different symbol)
- Compute rolling correlation BTC/ETH (20-period, 60-period)
- ETH relative strength: `eth_return_20 - btc_return_20`
- Optional: DXY daily data via alternative API
- Impact: market regime context, ETH often leads BTC moves
- Requires: extending `downloader.py` to support multiple symbols

### 4. Volatility Regime Features (medium priority)

- Realized volatility percentile: current `volatility_20` rank over trailing
  252 periods (0-1 scale)
- Volatility of volatility (vol-of-vol): std of `volatility_20` over 60 periods
- Regime classifier: low/medium/high vol based on percentile thresholds
- Impact: helps agent adapt position sizing to market regime

### Feature Priority Matrix

| Feature | Data Available | Implementation Effort | Expected Impact |
|---------|---------------|----------------------|-----------------|
| Funding rate | Yes | Low (2-3 hours) | High |
| Multi-timeframe | Yes (resample) | Medium (4-6 hours) | High |
| Volatility regime | Yes (derived) | Low (2-3 hours) | Medium |
| Cross-asset | No (new download) | High (8+ hours) | Medium |

Recommended order: funding rate → multi-timeframe → volatility regime → cross-asset.
