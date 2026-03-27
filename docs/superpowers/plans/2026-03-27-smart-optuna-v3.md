# Smart Optuna v3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a Colab notebook that extends v3 training to 2.5M steps (warm-start), then runs targeted Optuna optimization with 7 parameters including position size and leverage.

**Architecture:** Single notebook with 2 stages. Stage 1 loads v3 checkpoint and continues training. Stage 2 runs Optuna with narrow search space, PnL-focused scoring, and validation of top candidates. Based on existing `colab_optuna_simple.ipynb` patterns but with redesigned objective and scoring.

**Tech Stack:** Python 3.10+, Stable-Baselines3, Optuna, PyTorch, Gymnasium

---

### Task 1: Create notebook skeleton with setup and data cells

**Files:**
- Create: `notebooks/colab_optuna_v3_smart.ipynb`

- [ ] **Step 1: Create the notebook with markdown + setup cell**

Cell 0 (markdown):
```markdown
# AlphaCluster — Smart Optuna v3

Two-stage optimization:
- **Stage 1:** Extend v3 training to 2.5M steps (warm-start from 500k checkpoint)
- **Stage 2:** Smart Optuna — 7 parameters, 1M steps/trial, PnL-focused scoring

## Setup
```

Cell 1 (code):
```python
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

from google.colab import drive
drive.mount("/content/drive")

DRIVE_ROOT = Path("/content/drive/MyDrive/alphacluster")
DRIVE_SRC = DRIVE_ROOT / "src"

assert DRIVE_SRC.exists(), (
    f"Source not found at {DRIVE_SRC}\n"
    f"Copy your local src/ directory to Google Drive: My Drive/alphacluster/src/"
)

LOCAL_SRC = Path("/content/src")
if LOCAL_SRC.exists():
    shutil.rmtree(LOCAL_SRC)
shutil.copytree(DRIVE_SRC, LOCAL_SRC)
print(f"Copied source to {LOCAL_SRC}")

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "stable-baselines3>=2.4,<3.0", "gymnasium>=1.0,<2.0",
    "pyarrow", "python-dotenv", "tqdm", "rich", "optuna"], check=True)

sys.path.insert(0, str(LOCAL_SRC))

import alphacluster
print(f"AlphaCluster loaded from {Path(alphacluster.__file__).parent}")

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: No GPU detected. Enable GPU runtime: Runtime > Change runtime type > GPU")
```

- [ ] **Step 2: Add data loading cell**

Cell 2 (markdown):
```markdown
## Load & Split Data
```

Cell 3 (code):
```python
import pandas as pd

DATA_DIR = Path("/content/drive/MyDrive/alphacluster/data")
KLINES_PATH = DATA_DIR / "btcusdt_5m.parquet"
FUNDING_PATH = DATA_DIR / "btcusdt_funding.parquet"

assert KLINES_PATH.exists(), f"Kline data not found at {KLINES_PATH}"

klines_df = pd.read_parquet(KLINES_PATH, engine="pyarrow")
print(f"Loaded {len(klines_df):,} candles")
print(f"Date range: {klines_df.iloc[0, 0]} to {klines_df.iloc[-1, 0]}")

funding_df = None
if FUNDING_PATH.exists():
    funding_df = pd.read_parquet(FUNDING_PATH, engine="pyarrow")
    print(f"Loaded {len(funding_df):,} funding records")

# Chronological split: 70% train / 15% val / 15% test
n = len(klines_df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = klines_df.iloc[:train_end].reset_index(drop=True)
val_df = klines_df.iloc[train_end:val_end].reset_index(drop=True)
test_df = klines_df.iloc[val_end:].reset_index(drop=True)

print(f"\nData split: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")
```

- [ ] **Step 3: Verify notebook structure**

Run: open the notebook in a viewer and confirm cells 0-3 are present with correct content.

- [ ] **Step 4: Commit**

```bash
git add notebooks/colab_optuna_v3_smart.ipynb
git commit -m "feat: add smart optuna v3 notebook skeleton with setup and data cells"
```

---

### Task 2: Stage 1 — warm-start v3 to 2.5M steps

**Files:**
- Modify: `notebooks/colab_optuna_v3_smart.ipynb`

- [ ] **Step 1: Add Stage 1 training cell**

Cell 4 (markdown):
```markdown
## Stage 1 — Extend v3 Training (warm-start to 2.5M steps)

Load the v3 checkpoint (500k steps) and continue training with identical parameters
for a total of 2.5M steps. Curriculum phases apply relative to the new total:
- Phase 1 (0-30%): 0-750k
- Phase 2 (30-60%): 750k-1.5M
- Phase 3 (60-100%): 1.5M-2.5M
```

Cell 5 (code):
```python
import time

import torch
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from alphacluster.agent.config import TrainingConfig
from alphacluster.agent.trainer import (
    CurriculumCallback,
    TrainingMetricsCallback,
    create_agent,
    save_agent,
)
from alphacluster.config import MODEL_VERSION
from alphacluster.env.trading_env import TradingEnv

# ── Paths ──
V3_CHECKPOINT = Path("/content/drive/MyDrive/alphacluster/models/ppo_trading_simple_final_v3.pt")
MODELS_DIR = Path("/content/drive/MyDrive/alphacluster/models")
LOCAL_CHECKPOINT_DIR = Path("/content/checkpoints_stage1")

assert V3_CHECKPOINT.exists(), f"v3 checkpoint not found: {V3_CHECKPOINT}"

STAGE1_TIMESTEPS = 2_500_000
N_ENVS = 4


class ProgressCallback(BaseCallback):
    """Print a one-line progress update every log_interval timesteps."""

    def __init__(self, total_timesteps: int, log_interval: int = 50_000):
        super().__init__(verbose=0)
        self.total_timesteps = total_timesteps
        self.log_interval = log_interval
        self._start_time = None
        self._next_log = log_interval

    def _on_training_start(self) -> None:
        self._start_time = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_log:
            elapsed = time.time() - self._start_time
            pct = self.num_timesteps / self.total_timesteps * 100
            fps = self.num_timesteps / max(elapsed, 1)
            eta = (self.total_timesteps - self.num_timesteps) / max(fps, 1)
            print(
                f"  [{pct:5.1f}%] {self.num_timesteps:>10,} / {self.total_timesteps:,} "
                f"| {fps:.0f} fps | ETA {eta/60:.0f}m",
                flush=True,
            )
            self._next_log += self.log_interval
        return True


def make_env(df, config, rank=0):
    """Factory for SubprocVecEnv."""
    def _init():
        import os
        import warnings

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
        warnings.filterwarnings("ignore", message=".*datetime.datetime.utcnow.*")

        env = TradingEnv(
            df=df,
            window_size=config.window_size,
            episode_length=config.episode_length,
            simple_actions=True,
            fixed_size_pct=config.fixed_size_pct,
            fixed_leverage=config.fixed_leverage,
        )
        env.reset(seed=rank)
        return env
    return _init


# ── Config (v3 defaults) ──
config = TrainingConfig(
    total_timesteps=STAGE1_TIMESTEPS,
    simple_actions=True,
    fixed_size_pct=0.10,
    fixed_leverage=10,
    eval_freq=100_000,
)

# ── Create environments ──
print(f"Creating {N_ENVS} parallel training environments ...")
envs = SubprocVecEnv([make_env(train_df, config, rank=i) for i in range(N_ENVS)])
train_env = VecNormalize(envs, norm_obs=False, norm_reward=True, clip_reward=10.0)

eval_env = TradingEnv(
    df=val_df,
    window_size=config.window_size,
    episode_length=config.episode_length,
    simple_actions=True,
    fixed_size_pct=0.10,
    fixed_leverage=10,
)

# ── Create agent and load v3 weights ──
print("Creating PPO agent and loading v3 checkpoint ...")
agent = create_agent(train_env, config, verbose=0)
state_dict = torch.load(str(V3_CHECKPOINT), map_location="cpu", weights_only=True)
agent.policy.load_state_dict(state_dict)
print(f"Loaded v3 weights from {V3_CHECKPOINT}")

# ── Callbacks ──
curriculum_cb = CurriculumCallback(config, verbose=1)
progress_cb = ProgressCallback(STAGE1_TIMESTEPS, log_interval=50_000)
metrics_cb = TrainingMetricsCallback(
    eval_env=eval_env,
    config=config,
    log_dir=str(LOCAL_CHECKPOINT_DIR),
    verbose=0,
)

# ── Train ──
print(f"\nTraining for {STAGE1_TIMESTEPS:,} timesteps (warm-start from v3 500k) ...")
print(f"Model version: {MODEL_VERSION}")
print(f"Curriculum: Phase 1 (0-30%) -> Phase 2 (30-60%) -> Phase 3 (60-100%)")
print(f"Parameters: v3 defaults (fee_scale=1.0, ent_coef=0.05, phase1_end=0.3)\n")

start = time.time()
agent.learn(
    total_timesteps=STAGE1_TIMESTEPS,
    callback=CallbackList([curriculum_cb, progress_cb, metrics_cb]),
    progress_bar=False,
)
elapsed = time.time() - start
print(f"\nTraining complete in {elapsed/60:.1f} minutes")

# ── Save ──
MODELS_DIR.mkdir(parents=True, exist_ok=True)
stage1_path = MODELS_DIR / "ppo_trading_simple_v3_2.5M"
save_agent(agent, stage1_path)
print(f"Stage 1 model saved to {stage1_path}.pt")

# Cleanup
train_env.close()
```

- [ ] **Step 2: Add Stage 1 evaluation cell**

Cell 6 (markdown):
```markdown
## Stage 1 — Evaluation
```

Cell 7 (code):
```python
from alphacluster.backtest.runner import run_backtest
from alphacluster.backtest.metrics import calculate_metrics, print_report

EVAL_SEED = 42
N_EVAL_EPISODES = 5

print("Running Stage 1 evaluation on test set ...")
test_env = TradingEnv(
    df=test_df,
    window_size=config.window_size,
    episode_length=config.episode_length,
    simple_actions=True,
    fixed_size_pct=0.10,
    fixed_leverage=10,
)

result = run_backtest(model=agent, env=test_env, n_episodes=N_EVAL_EPISODES, seed=EVAL_SEED)
metrics = calculate_metrics(result)

print("\n" + "=" * 60)
print("  STAGE 1 RESULTS (v3 warm-start → 2.5M steps)")
print("=" * 60)
print_report(metrics)

stage1_pnl = metrics.get("avg_episode_return_pct", 0)
stage1_trades = metrics.get("avg_trades_per_episode", 0)
stage1_wr = metrics.get("win_rate", 0)
print(f"\nBaseline for Stage 2: PnL={stage1_pnl:.2f}%, trades/ep={stage1_trades:.1f}, WR={stage1_wr:.1f}%")
```

- [ ] **Step 3: Add Stage 1 learning curves cell**

Cell 8 (markdown):
```markdown
## Stage 1 — Learning Curves
```

Cell 9 (code):
```python
import matplotlib.pyplot as plt

metrics_path = LOCAL_CHECKPOINT_DIR / "training_metrics.csv"
if metrics_path.exists():
    mdf = pd.read_csv(metrics_path)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Stage 1 — Training Progress (v3 warm-start → 2.5M)", fontsize=14)

    plots = [
        ("timestep", "mean_reward", "Mean Episode Return (%)"),
        ("timestep", "trades_per_episode", "Trades per Episode"),
        ("timestep", "win_rate", "Win Rate (%)"),
        ("timestep", "avg_trade_duration", "Avg Trade Duration (steps)"),
        ("timestep", "flat_pct", "Flat Time (%)"),
        ("timestep", "total_pnl_pct", "Total PnL (%)"),
    ]

    for ax, (x, y, title) in zip(axes.flat, plots):
        if y in mdf.columns:
            ax.plot(mdf[x], mdf[y], marker="o", markersize=3)
        ax.set_title(title)
        ax.set_xlabel("Timesteps")
        ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        for pct, label in [(0.3, "Phase 2"), (0.6, "Phase 3")]:
            ts = int(STAGE1_TIMESTEPS * pct)
            ax.axvline(x=ts, color="red", linestyle="--", alpha=0.5)
            ax.text(ts, ax.get_ylim()[1], f" {label}", color="red", fontsize=8, va="top")

    plt.tight_layout()
    plt.show()
else:
    print("No training metrics found.")
```

- [ ] **Step 4: Commit**

```bash
git add notebooks/colab_optuna_v3_smart.ipynb
git commit -m "feat: add Stage 1 warm-start training and evaluation cells"
```

---

### Task 3: Stage 2 — Optuna objective and scoring

**Files:**
- Modify: `notebooks/colab_optuna_v3_smart.ipynb`

- [ ] **Step 1: Add evaluation helper and scoring function**

Cell 10 (markdown):
```markdown
## Stage 2 — Smart Optuna (7 params, 1M steps/trial)

**Search space:** fee_scale, ent_coef_phase1, phase1_end, phase3_fee_multiplier,
phase3_churn_multiplier, fixed_size_pct, fixed_leverage

**Scoring:** Mean PnL% per episode (pure, no composite weighting)

**Frozen:** learning_rate=3e-4, gamma=0.995, all other reward scales at v3 defaults
```

Cell 11 (code):
```python
import logging
from collections import Counter

import numpy as np
import optuna

from alphacluster.agent.trainer import CurriculumCallback

logger = logging.getLogger("optuna_smart_v3")

SCREENING_TIMESTEPS = 1_000_000
VALIDATION_TIMESTEPS = 2_000_000
N_ENVS = 4
EVAL_EPISODES = 5

DRIVE_REPORT_DIR = Path("/content/drive/MyDrive/alphacluster/reports/optuna_smart_v3")
DRIVE_REPORT_DIR.mkdir(parents=True, exist_ok=True)
STUDY_DB = DRIVE_REPORT_DIR / "optuna_study.db"


def evaluate_agent(model, eval_env, n_episodes=5):
    """Run deterministic evaluation. Returns (pnl_pct, details_dict).

    Scoring: mean PnL% per episode with hard rejection filters.
    """
    action_counts = Counter()
    episode_pnls = []
    total_trades = 0
    total_wins = 0
    total_steps = 0
    flat_steps = 0
    trade_durations = []
    long_trades = 0
    short_trades = 0

    for ep in range(n_episodes):
        obs, info = eval_env.reset(seed=ep * 42)
        done = False
        prev_n = len(eval_env.account.trade_history)
        open_trade = None
        ep_step = 0
        ep_pnl = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_counts[int(action)] += 1
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            ep_step += 1
            total_steps += 1

            if eval_env.account.position_side == "flat":
                flat_steps += 1

            th = eval_env.account.trade_history
            while prev_n < len(th):
                entry = th[prev_n]
                prev_n += 1
                if entry["action"] == "open":
                    open_trade = {"side": entry.get("side"), "step": ep_step}
                elif entry["action"] == "close" and open_trade is not None:
                    pnl = entry.get("pnl", 0.0)
                    ep_pnl += pnl
                    total_trades += 1
                    if pnl > 0:
                        total_wins += 1
                    duration = ep_step - open_trade["step"]
                    trade_durations.append(duration)
                    if open_trade["side"] == "long":
                        long_trades += 1
                    else:
                        short_trades += 1
                    open_trade = None

        episode_pnls.append(ep_pnl / eval_env.initial_balance * 100)

    trades_per_ep = total_trades / max(n_episodes, 1)
    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    flat_pct = (flat_steps / total_steps * 100) if total_steps > 0 else 100
    avg_duration = np.mean(trade_durations) if trade_durations else 0
    mean_pnl_pct = np.mean(episode_pnls) if episode_pnls else 0

    if long_trades + short_trades > 0:
        minority = min(long_trades, short_trades)
        direction_balance = minority / (long_trades + short_trades) * 2
    else:
        direction_balance = 0.0

    total_actions = sum(action_counts.values())
    if total_actions > 0:
        probs = np.array([action_counts.get(a, 0) / total_actions for a in range(3)])
        probs = probs[probs > 0]
        action_entropy = -np.sum(probs * np.log(probs)) / np.log(3)
    else:
        action_entropy = 0.0

    details = {
        "pnl_pct": round(mean_pnl_pct, 2),
        "trades_per_ep": round(trades_per_ep, 1),
        "win_rate": round(win_rate, 1),
        "flat_pct": round(flat_pct, 1),
        "avg_duration": round(avg_duration, 1),
        "long_trades": long_trades,
        "short_trades": short_trades,
        "direction_balance": round(direction_balance, 3),
        "action_entropy": round(action_entropy, 3),
    }

    # ── Hard rejection filters ──
    if flat_pct > 70:
        return -1000.0, {**details, "reject": "flat_pct > 70%"}
    if trades_per_ep < 5:
        return -1000.0, {**details, "reject": f"trades_per_ep={trades_per_ep:.1f} < 5"}
    if win_rate < 40:
        return -1000.0, {**details, "reject": f"win_rate={win_rate:.1f}% < 40%"}
    if avg_duration < 1.5:
        return -1000.0, {**details, "reject": f"avg_duration={avg_duration:.1f} < 1.5"}

    # ── Score = mean PnL% per episode ──
    return float(mean_pnl_pct), details


print("Evaluation function defined.")
print("Scoring: mean PnL% per episode (no composite weighting)")
print("Hard filters: trades_per_ep >= 5, win_rate >= 40%, flat_pct <= 70%, avg_duration >= 1.5")
```

- [ ] **Step 2: Add Optuna objective function**

Cell 12 (code):
```python
class OptunaPruningCallback(BaseCallback):
    """Report intermediate scores to Optuna and prune bad trials."""

    def __init__(self, trial, eval_env, report_freq):
        super().__init__(verbose=0)
        self.trial = trial
        self.eval_env = eval_env
        self.report_freq = report_freq
        self._next_report = report_freq

    def _on_step(self):
        if self.num_timesteps < self._next_report:
            return True
        self._next_report += self.report_freq
        score, _ = evaluate_agent(self.model, self.eval_env, n_episodes=3)
        self.trial.report(score, self.num_timesteps)
        if self.trial.should_prune():
            raise optuna.TrialPruned()
        return True


def objective(trial):
    """Optuna objective: 7 params, v3 defaults as base, PnL-focused scoring."""
    train_env = None
    try:
        # ── Tunable reward params (3) ──
        fee_scale = trial.suggest_float("fee_scale", 0.3, 2.0)
        ent_coef_phase1 = trial.suggest_float("ent_coef_phase1", 0.03, 0.20, log=True)
        phase1_end = trial.suggest_float("phase1_end", 0.2, 0.5)

        # ── Tunable Phase 3 multipliers (2) ──
        phase3_fee_mult = trial.suggest_float("phase3_fee_multiplier", 0.5, 3.0)
        phase3_churn_mult = trial.suggest_float("phase3_churn_multiplier", 0.5, 3.0)

        # ── Tunable position params (2) ──
        size_pct = trial.suggest_categorical("fixed_size_pct", [0.02, 0.05, 0.10, 0.15])
        leverage = trial.suggest_categorical("fixed_leverage", [5, 10, 15])

        # ── Frozen reward config (v3 defaults) ──
        base_reward_config = {
            "fee_scale": fee_scale,
            "opportunity_cost_scale": 1.0,
            "opportunity_cost_cap": 0.02,
            "opportunity_cost_threshold": 0.002,
            "churn_penalty_scale": 1.0,
            "drawdown_penalty_scale": 1.0,
            "quality_scale": 1.0,
            "position_mgmt_scale": 1.0,
        }

        config = TrainingConfig(
            total_timesteps=SCREENING_TIMESTEPS,
            learning_rate=3e-4,
            gamma=0.995,
            ent_coef=ent_coef_phase1,
            simple_actions=True,
            fixed_size_pct=size_pct,
            fixed_leverage=leverage,
            phase1_end=phase1_end,
            phase2_end=min(phase1_end + 0.3, 0.8),
            eval_freq=SCREENING_TIMESTEPS + 1,  # disable built-in eval
        )

        envs = SubprocVecEnv([
            make_env(train_df, config, rank=i)
            for i in range(N_ENVS)
        ])
        train_env = VecNormalize(envs, norm_obs=False, norm_reward=True, clip_reward=10.0)

        eval_raw = TradingEnv(
            df=val_df,
            window_size=config.window_size,
            episode_length=config.episode_length,
            simple_actions=True,
            fixed_size_pct=size_pct,
            fixed_leverage=leverage,
        )

        agent = create_agent(train_env, config, verbose=0)

        curriculum_cb = CurriculumCallback(
            config,
            base_reward_config=base_reward_config,
            phase3_fee_multiplier=phase3_fee_mult,
            phase3_churn_multiplier=phase3_churn_mult,
            ent_coef_phase1=ent_coef_phase1,
            verbose=0,
        )
        pruning_cb = OptunaPruningCallback(
            trial, eval_raw, report_freq=250_000,
        )

        agent.learn(
            total_timesteps=config.total_timesteps,
            callback=CallbackList([curriculum_cb, pruning_cb]),
            progress_bar=False,
        )

        score, details = evaluate_agent(agent, eval_raw, n_episodes=EVAL_EPISODES)
        for k, v in details.items():
            if isinstance(v, (int, float, str)):
                trial.set_user_attr(k, v)

        return score

    except optuna.TrialPruned:
        raise
    except Exception as e:
        trial.set_user_attr("error", str(e))
        logger.exception("Trial %d failed: %s", trial.number, e)
        return -1000.0
    finally:
        if train_env is not None:
            try:
                train_env.close()
            except Exception:
                pass


print("Objective function defined (7 params).")
print(f"Training: {SCREENING_TIMESTEPS:,} steps/trial, pruning at 500k")
print("Tunable: fee_scale, ent_coef_phase1, phase1_end, phase3_fee/churn_mult, size_pct, leverage")
print("Frozen: lr=3e-4, gamma=0.995, all other reward scales=1.0 (v3 defaults)")
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/colab_optuna_v3_smart.ipynb
git commit -m "feat: add Stage 2 evaluation helper and Optuna objective (7 params)"
```

---

### Task 4: Stage 2 — Screening run

**Files:**
- Modify: `notebooks/colab_optuna_v3_smart.ipynb`

- [ ] **Step 1: Add screening cell**

Cell 13 (markdown):
```markdown
## Stage 2 — Screening (15 trials x 1M steps)
```

Cell 14 (code):
```python
N_SCREENING_TRIALS = 15

print("=" * 60)
print(f"STAGE 2 SCREENING: {SCREENING_TIMESTEPS:,} steps x {N_SCREENING_TRIALS} trials")
print("=" * 60)

study = optuna.create_study(
    study_name="alphacluster_smart_v3",
    direction="maximize",
    storage=f"sqlite:///{STUDY_DB}",
    load_if_exists=True,
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=3,
        n_warmup_steps=500_000,
    ),
    sampler=optuna.samplers.TPESampler(seed=42),
)

# Seed trial 0: v3 defaults
if len(study.trials) == 0:
    study.enqueue_trial({
        "fee_scale": 1.0,
        "ent_coef_phase1": 0.05,
        "phase1_end": 0.3,
        "phase3_fee_multiplier": 2.0,
        "phase3_churn_multiplier": 2.0,
        "fixed_size_pct": 0.10,
        "fixed_leverage": 10,
    })
    print("Enqueued v3 defaults as seed trial")

completed = len([t for t in study.trials
                 if t.state in (optuna.trial.TrialState.COMPLETE,
                                optuna.trial.TrialState.PRUNED)])
remaining = max(0, N_SCREENING_TRIALS - completed)
print(f"Completed: {completed}, remaining: {remaining}")

if remaining > 0:
    study.optimize(objective, n_trials=remaining, timeout=None)
else:
    print(f"All {N_SCREENING_TRIALS} screening trials already completed.")

# ── Results summary ──
print(f"\nBest trial: #{study.best_trial.number}, score={study.best_trial.value:.4f}")
print(f"Best params: {study.best_trial.params}")
```

- [ ] **Step 2: Add screening results export cell**

Cell 15 (code):
```python
# Save trial results to CSV
TRIAL_CSV = DRIVE_REPORT_DIR / "trial_results.csv"

rows = []
for t in study.trials:
    row = {
        "trial": t.number,
        "score": t.value,
        "state": t.state.name,
        **t.params,
        **{k: v for k, v in t.user_attrs.items() if isinstance(v, (int, float, str))},
    }
    rows.append(row)

trial_df = pd.DataFrame(rows).sort_values("score", ascending=False)
trial_df.to_csv(TRIAL_CSV, index=False)
print(f"Trial results saved to {TRIAL_CSV}")
print(trial_df.to_string(index=False))
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/colab_optuna_v3_smart.ipynb
git commit -m "feat: add Stage 2 screening run (15 trials x 1M steps)"
```

---

### Task 5: Stage 2 — Validation of top candidates

**Files:**
- Modify: `notebooks/colab_optuna_v3_smart.ipynb`

- [ ] **Step 1: Add validation cell**

Cell 16 (markdown):
```markdown
## Stage 2 — Validation (top 5 retrained at 2M steps)
```

Cell 17 (code):
```python
print("=" * 60)
print(f"STAGE 2 VALIDATION: {VALIDATION_TIMESTEPS:,} steps x top 5")
print("=" * 60)

top_trials = sorted(
    [t for t in study.trials if t.value is not None and t.value > -999],
    key=lambda t: t.value,
    reverse=True,
)[:5]

print(f"Validating {len(top_trials)} configs at {VALIDATION_TIMESTEPS:,} timesteps\n")

VALIDATION_CSV = DRIVE_REPORT_DIR / "validation_results.csv"
validation_results = []

existing_validated = set()
if VALIDATION_CSV.exists():
    existing_df = pd.read_csv(VALIDATION_CSV)
    existing_validated = set(existing_df["screening_trial"].tolist())
    validation_results = existing_df.to_dict("records")

for i, trial_data in enumerate(top_trials):
    if trial_data.number in existing_validated:
        print(f"  [{i+1}/{len(top_trials)}] Trial #{trial_data.number} — already validated")
        continue

    print(f"  [{i+1}/{len(top_trials)}] Trial #{trial_data.number} "
          f"(screening={trial_data.value:.4f})")
    params = trial_data.params

    train_env = None
    try:
        base_reward_config = {
            "fee_scale": params["fee_scale"],
            "opportunity_cost_scale": 1.0,
            "opportunity_cost_cap": 0.02,
            "opportunity_cost_threshold": 0.002,
            "churn_penalty_scale": 1.0,
            "drawdown_penalty_scale": 1.0,
            "quality_scale": 1.0,
            "position_mgmt_scale": 1.0,
        }

        size_pct = params["fixed_size_pct"]
        leverage = params["fixed_leverage"]

        config = TrainingConfig(
            total_timesteps=VALIDATION_TIMESTEPS,
            learning_rate=3e-4,
            gamma=0.995,
            ent_coef=params["ent_coef_phase1"],
            simple_actions=True,
            fixed_size_pct=size_pct,
            fixed_leverage=leverage,
            phase1_end=params["phase1_end"],
            phase2_end=min(params["phase1_end"] + 0.3, 0.8),
            eval_freq=VALIDATION_TIMESTEPS + 1,
        )

        envs = SubprocVecEnv([
            make_env(train_df, config, rank=r)
            for r in range(N_ENVS)
        ])
        train_env = VecNormalize(envs, norm_obs=False, norm_reward=True, clip_reward=10.0)

        eval_raw = TradingEnv(
            df=val_df,
            window_size=config.window_size,
            episode_length=config.episode_length,
            simple_actions=True,
            fixed_size_pct=size_pct,
            fixed_leverage=leverage,
        )

        agent = create_agent(train_env, config, verbose=0)

        curriculum_cb = CurriculumCallback(
            config,
            base_reward_config=base_reward_config,
            phase3_fee_multiplier=params["phase3_fee_multiplier"],
            phase3_churn_multiplier=params["phase3_churn_multiplier"],
            ent_coef_phase1=params["ent_coef_phase1"],
            verbose=0,
        )

        agent.learn(
            total_timesteps=config.total_timesteps,
            callback=CallbackList([curriculum_cb]),
            progress_bar=False,
        )

        score, details = evaluate_agent(agent, eval_raw, n_episodes=EVAL_EPISODES)

        # Save best validation model
        if score > -999:
            model_path = MODELS_DIR / f"optuna_smart_v3_trial{trial_data.number}"
            save_agent(agent, model_path)

        result = {
            "screening_trial": trial_data.number,
            "screening_score": trial_data.value,
            "validation_score": score,
            **{k: v for k, v in details.items() if isinstance(v, (int, float, str))},
            **params,
        }
        validation_results.append(result)
        pd.DataFrame(validation_results).to_csv(VALIDATION_CSV, index=False)

        print(f"    -> val_score={score:.4f}, pnl={details.get('pnl_pct', '?')}%, "
              f"trades/ep={details.get('trades_per_ep', '?')}, "
              f"WR={details.get('win_rate', '?')}%")

    except Exception as e:
        print(f"    -> FAILED: {e}")
        validation_results.append({
            "screening_trial": trial_data.number,
            "screening_score": trial_data.value,
            "validation_score": -1000.0,
            "error": str(e),
        })
        pd.DataFrame(validation_results).to_csv(VALIDATION_CSV, index=False)
    finally:
        if train_env is not None:
            try:
                train_env.close()
            except Exception:
                pass

print("\nValidation complete.")
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/colab_optuna_v3_smart.ipynb
git commit -m "feat: add Stage 2 validation (top 5 at 2M steps)"
```

---

### Task 6: Results comparison and best model selection

**Files:**
- Modify: `notebooks/colab_optuna_v3_smart.ipynb`

- [ ] **Step 1: Add final comparison cell**

Cell 18 (markdown):
```markdown
## Results — Comparison & Best Model
```

Cell 19 (code):
```python
import json

print("=" * 60)
print("  FINAL COMPARISON")
print("=" * 60)

# ── Collect all results ──
val_df_results = pd.read_csv(VALIDATION_CSV) if VALIDATION_CSV.exists() else pd.DataFrame()
valid_results = val_df_results[val_df_results["validation_score"] > -999].copy()

if len(valid_results) > 0:
    valid_results = valid_results.sort_values("validation_score", ascending=False)

    print("\nTop validated trials:")
    print(valid_results[[
        "screening_trial", "screening_score", "validation_score",
        "pnl_pct", "trades_per_ep", "win_rate", "flat_pct",
        "fixed_size_pct", "fixed_leverage",
    ]].to_string(index=False))

    # ── Best model ──
    best = valid_results.iloc[0]
    best_trial = int(best["screening_trial"])

    print(f"\n{'=' * 60}")
    print(f"  BEST MODEL: Trial #{best_trial}")
    print(f"  Screening PnL: {best['screening_score']:.2f}%")
    print(f"  Validation PnL: {best['validation_score']:.2f}%")
    print(f"  Size: {best['fixed_size_pct']}, Leverage: {best['fixed_leverage']}x")
    print(f"{'=' * 60}")

    # ── Save best params ──
    best_params = {
        "source": "optuna_smart_v3",
        "screening_trial": best_trial,
        "screening_score": float(best["screening_score"]),
        "validation_score": float(best["validation_score"]),
        "params": {
            k: best[k] for k in [
                "fee_scale", "ent_coef_phase1", "phase1_end",
                "phase3_fee_multiplier", "phase3_churn_multiplier",
                "fixed_size_pct", "fixed_leverage",
            ] if k in best.index
        },
    }

    best_params_path = DRIVE_REPORT_DIR / "best_params.json"
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest params saved to {best_params_path}")

    # ── Copy best model as final ──
    best_model_src = MODELS_DIR / f"optuna_smart_v3_trial{best_trial}.pt"
    best_model_dst = MODELS_DIR / "ppo_trading_smart_v3_best.pt"
    if best_model_src.exists():
        shutil.copy2(best_model_src, best_model_dst)
        print(f"Best model copied to {best_model_dst}")

else:
    print("No valid validation results found.")

# ── Stage 1 vs Stage 2 comparison ──
print(f"\n{'=' * 60}")
print("  STAGE 1 vs STAGE 2 COMPARISON")
print(f"{'=' * 60}")
print(f"  Stage 1 (v3 → 2.5M): PnL={stage1_pnl:.2f}%, "
      f"trades/ep={stage1_trades:.1f}, WR={stage1_wr:.1f}%")
if len(valid_results) > 0:
    print(f"  Stage 2 best:        PnL={best['validation_score']:.2f}%, "
          f"trades/ep={best.get('trades_per_ep', '?')}, WR={best.get('win_rate', '?')}%")
    if float(best["validation_score"]) > stage1_pnl:
        print("\n  >>> Stage 2 WINS — Smart Optuna found better parameters")
    else:
        print("\n  >>> Stage 1 WINS — Longer training with v3 defaults is better")
```

- [ ] **Step 2: Add Optuna plots cell**

Cell 20 (code):
```python
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
)

PLOTS_DIR = DRIVE_REPORT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    fig = plot_optimization_history(study)
    fig.write_image(str(PLOTS_DIR / "screening_optimization_history.png"))
    fig.show()
except Exception as e:
    print(f"Could not plot optimization history: {e}")

try:
    fig = plot_param_importances(study)
    fig.write_image(str(PLOTS_DIR / "screening_param_importances.png"))
    fig.show()
except Exception as e:
    print(f"Could not plot param importances: {e}")

try:
    fig = plot_slice(study, params=[
        "fee_scale", "ent_coef_phase1", "phase1_end",
        "fixed_size_pct", "fixed_leverage",
    ])
    fig.write_image(str(PLOTS_DIR / "screening_slice_key_params.png"))
    fig.show()
except Exception as e:
    print(f"Could not plot slice: {e}")

print(f"Plots saved to {PLOTS_DIR}")
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/colab_optuna_v3_smart.ipynb
git commit -m "feat: add results comparison and Optuna visualization cells"
```

---

### Task 7: Final review and cleanup

**Files:**
- Modify: `notebooks/colab_optuna_v3_smart.ipynb`

- [ ] **Step 1: Verify notebook cell order and completeness**

Open the notebook and verify the following cell sequence:
- Cell 0: Title markdown
- Cell 1: Setup (drive mount, deps)
- Cell 2-3: Data loading
- Cell 4-5: Stage 1 training (warm-start)
- Cell 6-7: Stage 1 evaluation
- Cell 8-9: Stage 1 learning curves
- Cell 10-11: Stage 2 eval helper + scoring
- Cell 12: Stage 2 objective
- Cell 13-14: Stage 2 screening run
- Cell 15: Stage 2 screening export
- Cell 16-17: Stage 2 validation
- Cell 18-19: Results comparison
- Cell 20: Optuna plots

- [ ] **Step 2: Verify make_env uses config.fixed_size_pct and config.fixed_leverage**

Confirm that `make_env` reads `config.fixed_size_pct` and `config.fixed_leverage` (not hardcoded 0.10/10), so Optuna trials with different size/leverage create correct environments.

- [ ] **Step 3: Run lint check on extracted code**

```bash
# Extract Python from notebook and check for syntax errors
python3 -c "
import json, ast
nb = json.load(open('notebooks/colab_optuna_v3_smart.ipynb'))
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        try:
            ast.parse(src)
        except SyntaxError as e:
            print(f'Cell {i}: SYNTAX ERROR: {e}')
print('All cells parsed OK')
"
```

- [ ] **Step 4: Final commit**

```bash
git add notebooks/colab_optuna_v3_smart.ipynb
git commit -m "docs: finalize smart optuna v3 notebook structure"
```
