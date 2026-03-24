# Optuna Hyperparameter Tuning — Design Spec

## Problem

The RL agent consistently degenerates into 100% flat (no trading) during training.
Root cause: fee and churn penalties dominate opportunity cost and quality rewards,
so the agent learns that "not trading" is the optimal strategy.

Training metrics show clear progression:
- 50k-150k: active trading, chaotic PnL, fee_to_pnl_ratio already 200-300%
- 200k-800k: trade duration collapses to ~2 steps, flat_pct rises to 64%
- 850k-1M: full capitulation — 0 trades, 100% flat

Manual scale tuning is impractical with 11+ interacting parameters.

## Solution

A Colab notebook using **Optuna (TPE sampler + MedianPruner)** for automated
Bayesian hyperparameter optimization of reward scales, curriculum multipliers,
and PPO hyperparameters.

## Configuration Source

The tuning notebook uses `TrainingConfig` from `agent/config.py` as the single
source of truth (gamma=0.995, batch_size=128). Values in the top-level `config.py`
(gamma=0.99, batch_size=256) are legacy defaults used only by the CLI and are not
relevant to the tuning notebook.

## Parameter Space (13 dimensions)

### Reward Scales (8)

| Parameter | Range | Scale | Current Default |
|-----------|-------|-------|-----------------|
| `fee_scale` | 0.1 – 2.0 | uniform | 1.0 |
| `opportunity_cost_scale` | 0.1 – 2.0 | uniform | 0.5 |
| `opportunity_cost_cap` | 0.01 – 0.15 | uniform | 0.02 (hardcoded) |
| `opportunity_cost_threshold` | 0.001 – 0.005 | uniform | 0.002 (hardcoded) |
| `churn_penalty_scale` | 0.1 – 2.0 | uniform | 1.0 |
| `drawdown_penalty_scale` | 0.1 – 2.0 | uniform | 1.0 |
| `quality_scale` | 0.1 – 2.0 | uniform | 1.0 |
| `position_mgmt_scale` | 0.1 – 2.0 | uniform | 1.0 (new, wraps 0.4 coeff) |

Notes:
- `opportunity_cost_cap` is hardcoded at 0.02 in `trading_env.py` — must be
  extracted to `reward_config`.
- `opportunity_cost_threshold` is hardcoded at 0.002 — also extracted.
- `position_mgmt_scale` is a new scale that wraps the hardcoded 0.4 coefficient
  in the position management reward: `position_reward = 0.4 * scale * ...`

### Curriculum Phase 3 Multipliers (2)

| Parameter | Range | Scale | Current Default |
|-----------|-------|-------|-----------------|
| `phase3_fee_multiplier` | 1.0 – 3.0 | uniform | 2.0 |
| `phase3_churn_multiplier` | 1.0 – 3.0 | uniform | 2.0 |

These multiply the base `fee_scale` and `churn_penalty_scale` in Phase 3.

**Full curriculum phase computation:**

The `CurriculumCallback` will be refactored to accept base scales and compute
phase-specific configs using fixed phase multiplier vectors:

```
Phase 1 multipliers: {fee: 0.5, churn: 0.5, opp: 0.3, dd: 0.3, quality: 1.0, pos_mgmt: 1.0}
Phase 2 multipliers: {fee: 1.0, churn: 1.0, opp: 1.0, dd: 1.0, quality: 1.0, pos_mgmt: 1.0}
Phase 3 multipliers: {fee: phase3_fee_mult, churn: phase3_churn_mult,
                       opp: 1.0, dd: 1.5, quality: 0.5, pos_mgmt: 1.0}
```

Phase config = base_scale × phase_multiplier for each component.

### PPO Hyperparameters (3)

| Parameter | Range | Scale | Current Default |
|-----------|-------|-------|-----------------|
| `learning_rate` | 1e-4 – 1e-3 | log-uniform | 3e-4 |
| `ent_coef_phase1` | 0.01 – 0.1 | log-uniform | 0.05 |
| `gamma` | 0.99 – 0.999 | uniform | 0.995 |

Note on `ent_coef`: The curriculum callback overrides entropy at each phase.
We tune `ent_coef_phase1` and derive Phase 2/3 as fixed fractions:
- Phase 1: `ent_coef_phase1` (sampled)
- Phase 2: `ent_coef_phase1 * 0.6`
- Phase 3: `ent_coef_phase1 * 0.1`

This preserves the decreasing exploration schedule while making it tunable.

## Objective Function

### Composite Score

```python
pnl_norm = np.clip(total_pnl_pct, -50, 50) / 100 + 0.5  # normalized to 0-1
trades_norm = min(trades_per_episode, 200) / 200           # normalized to 0-1
win_rate_norm = win_rate / 100                              # normalized to 0-1

score = pnl_norm * 0.4 + trades_norm * 0.3 + win_rate_norm * 0.3
```

All three terms are on the 0-1 scale, so weights reflect true relative importance.

### Hard Constraints (trial rejected → score = -1000)

- `flat_pct > 70%` — agent has largely collapsed into flat
- `trades_per_episode < 20` — insufficient trading activity
- `avg_trade_duration < 1.5` — pure churn, not real trading

### Evaluation

Score is computed from the **last row** of the `TrainingMetricsCallback` CSV
(regardless of which timestep it corresponds to — handles pruned trials correctly).
Uses `n_episodes=3` with `seed=42` for deterministic comparison.

### Error Handling

The objective function wraps the entire trial in try/except. On crash (OOM, NaN,
environment error):
- Returns score = -1000
- Stores error message in `trial.set_user_attr("error", str(e))`
- Logs the error for debugging

## VecNormalize Interaction

**Reward normalization is DISABLED** during tuning trials (`norm_reward=False`).
Since we are tuning reward scales directly, VecNormalize reward normalization
would partially undo scale changes, making the search ineffective.

Observation normalization (`norm_obs=True`) remains enabled as it helps training
stability without interfering with reward scale tuning.

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
- **Metrics log_freq:** Set to 100,000 (aligned with pruning warmup) to ensure
  CSV has a row at exactly 100k for intermediate reporting

### Phase 2: Validation (500k steps × top 10)

- **Goal:** Verify that screening winners actually produce viable strategies
- **Input:** Top 10 parameter sets from Phase 1 (by final score)
- **Timesteps:** 500,000 (~30 min per trial)
- **Trials:** 10 (fixed params, no Optuna sampling)
- **Estimated time:** ~5 hours
- **No pruning** — all trials run to completion
- **Metrics log_freq:** 100,000 (5 data points per trial for learning curve)

### Total estimated time: ~13 hours

## Notebook Structure

### Cell 1 — Setup & Dependencies

```
!pip install optuna plotly kaleido
```

Mount Google Drive. Define paths:
- `DRIVE_DIR = Path("/content/drive/MyDrive/AlphaCluster/optuna_tuning/")`
- `STUDY_DB = DRIVE_DIR / "optuna_study.db"` (SQLite, Optuna native persistence)
- `RESULTS_CSV = DRIVE_DIR / "trial_results.csv"`
- `BEST_PARAMS_JSON = DRIVE_DIR / "best_params.json"`

### Cell 2 — Install AlphaCluster & Load Data

Clone repo, install package, load and split data (same as colab_train.ipynb).

### Cell 3 — Define Objective Function

```python
def objective(trial: optuna.Trial) -> float:
    try:
        # Sample 13 parameters
        fee_scale = trial.suggest_float("fee_scale", 0.1, 2.0)
        opportunity_cost_scale = trial.suggest_float("opportunity_cost_scale", 0.1, 2.0)
        opportunity_cost_cap = trial.suggest_float("opportunity_cost_cap", 0.01, 0.15)
        opportunity_cost_threshold = trial.suggest_float(
            "opportunity_cost_threshold", 0.001, 0.005,
        )
        churn_penalty_scale = trial.suggest_float("churn_penalty_scale", 0.1, 2.0)
        drawdown_penalty_scale = trial.suggest_float("drawdown_penalty_scale", 0.1, 2.0)
        quality_scale = trial.suggest_float("quality_scale", 0.1, 2.0)
        position_mgmt_scale = trial.suggest_float("position_mgmt_scale", 0.1, 2.0)
        phase3_fee_mult = trial.suggest_float("phase3_fee_multiplier", 1.0, 3.0)
        phase3_churn_mult = trial.suggest_float("phase3_churn_multiplier", 1.0, 3.0)
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
        ent = trial.suggest_float("ent_coef_phase1", 0.01, 0.1, log=True)
        gamma = trial.suggest_float("gamma", 0.99, 0.999)

        # Build base reward_config
        base_reward_config = {
            "fee_scale": fee_scale,
            "opportunity_cost_scale": opportunity_cost_scale,
            "opportunity_cost_cap": opportunity_cost_cap,
            "opportunity_cost_threshold": opportunity_cost_threshold,
            "churn_penalty_scale": churn_penalty_scale,
            "drawdown_penalty_scale": drawdown_penalty_scale,
            "quality_scale": quality_scale,
            "position_mgmt_scale": position_mgmt_scale,
        }

        # Build curriculum config with phase multipliers
        curriculum_config = {
            "base_reward_config": base_reward_config,
            "phase3_fee_multiplier": phase3_fee_mult,
            "phase3_churn_multiplier": phase3_churn_mult,
            "ent_coef_phase1": ent,
        }

        # Create env, agent, train, evaluate
        # ...read last row of metrics CSV, compute score
        # Apply hard constraints (flat_pct > 70%, trades < 20, duration < 1.5)

    except Exception as e:
        trial.set_user_attr("error", str(e))
        logger.exception("Trial %d failed: %s", trial.number, e)
        return -1000.0

    return score
```

### Cell 4 — Pruning Callback

Custom SB3 callback that:
1. Reads last row of `TrainingMetricsCallback` CSV at checkpoints
2. Computes intermediate score
3. Reports to `trial.report(score, step)`
4. Raises `optuna.TrialPruned()` if `trial.should_prune()`

Registered AFTER `TrainingMetricsCallback` to ensure CSV is written first
(SB3 executes callbacks in registration order).

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
top_params = sorted(
    [t for t in study.trials if t.value is not None and t.value > -999],
    key=lambda t: t.value,
    reverse=True,
)[:10]

# Run each with 500k timesteps (no pruning)
validation_results = []
for i, trial_data in enumerate(top_params):
    result = run_validation_trial(trial_data.params, timesteps=500_000)
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

Template cell showing best params and how to apply them to `TrainingConfig`.

## Required Code Changes

### 1. Make `opportunity_cost_cap` and `opportunity_cost_threshold` configurable

In `trading_env.py`, extract hardcoded values to `reward_config`:
```python
# Before (hardcoded):
if price_change > 0.002:
    raw_penalty = price_change - 0.002
    opportunity_penalty = min(raw_penalty, 0.02) * rc["opportunity_cost_scale"]

# After (configurable):
opp_threshold = rc.get("opportunity_cost_threshold", 0.002)
opp_cap = rc.get("opportunity_cost_cap", 0.02)
if price_change > opp_threshold:
    raw_penalty = price_change - opp_threshold
    opportunity_penalty = min(raw_penalty, opp_cap) * rc["opportunity_cost_scale"]
```

### 2. Add `position_mgmt_scale` to reward computation

```python
# Before:
position_reward = 0.4 * upnl_ratio * (1.0 + hold_time_bonus)

# After:
pos_scale = rc.get("position_mgmt_scale", 1.0)
position_reward = 0.4 * pos_scale * upnl_ratio * (1.0 + hold_time_bonus)
```

Apply same scale to the losing-position branch.

### 3. Refactor CurriculumCallback to accept dynamic base scales

Constructor accepts:
- `base_reward_config: dict` — base scales for all reward components
- `phase3_fee_multiplier: float` — Phase 3 fee scale multiplier
- `phase3_churn_multiplier: float` — Phase 3 churn scale multiplier
- `ent_coef_phase1: float` — Phase 1 entropy (Phase 2 = 0.6x, Phase 3 = 0.1x)

`_apply_phase()` computes phase config as base_scale × phase_multiplier.

### 4. Update DEFAULT_REWARD_CONFIG

Add new keys with backward-compatible defaults:
```python
DEFAULT_REWARD_CONFIG: dict[str, float] = {
    "opportunity_cost_scale": 0.5,
    "opportunity_cost_cap": 0.02,
    "opportunity_cost_threshold": 0.002,
    "fee_scale": 1.0,
    "drawdown_penalty_scale": 1.0,
    "churn_penalty_scale": 1.0,
    "quality_scale": 1.0,
    "position_mgmt_scale": 1.0,
}
```

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

## Known Limitations

- **Single seed per trial:** RL training has high variance. A configuration that
  scores well on one seed may fail on another. Phase 2 uses a single seed for
  time reasons. If results look promising but inconsistent, consider re-running
  top 3 configs with 3 seeds each.
- **200k screening may miss slow-converging configs:** Some parameter sets might
  look bad at 200k but converge well at 1M+. The Phase 2 validation at 500k
  partially mitigates this.
- **13 parameters with 40 trials:** TPE will still be partially exploring at
  trial 40. The study can be continued with additional trials if needed.

## Dependencies

New pip packages for the tuning notebook:
- `optuna>=3.0` — Bayesian optimization framework
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
