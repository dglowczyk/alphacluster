# Ablation Sweep + Slower Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `base_reward_config` passthrough to `train()`, and create an ablation sweep notebook that runs 4 reward configurations at 1M steps each with a slower curriculum.

**Architecture:** One source change (add parameter to `train()`), one test, one new notebook. The notebook defines 4 ablation configs and loops through them, saving metrics + models to Google Drive.

**Tech Stack:** Python 3.10+, stable-baselines3, gymnasium, matplotlib, pandas

---

## File Structure

- Modify: `src/alphacluster/agent/trainer.py` — add `base_reward_config` param to `train()`
- Modify: `tests/test_env.py` — test that `train()` passes `base_reward_config` through
- Create: `notebooks/colab_ablation.ipynb` — ablation sweep notebook

---

### Task 1: Add `base_reward_config` parameter to `train()`

**Files:**
- Modify: `src/alphacluster/agent/trainer.py:507-593`
- Test: `tests/test_env.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_env.py` in the `TestCurriculumCallback` class:

```python
def test_train_passes_base_reward_config(self):
    """train() should forward base_reward_config to CurriculumCallback."""
    from unittest.mock import patch, MagicMock
    from alphacluster.agent.trainer import train, create_agent
    from stable_baselines3.common.vec_env import DummyVecEnv

    config = TrainingConfig(
        total_timesteps=128,
        n_steps=64,
        batch_size=64,
        curriculum_enabled=True,
        eval_freq=10_000,
    )
    env = DummyVecEnv([lambda: TradingEnv(df=_make_df(), simple_actions=True)])

    base_cfg = {"fee_scale": 0.0, "churn_penalty_scale": 0.0}

    with patch("alphacluster.agent.trainer.CurriculumCallback") as MockCB:
        MockCB.return_value = MagicMock()
        MockCB.return_value._on_step = MagicMock(return_value=True)
        agent = create_agent(env, config)
        train(
            agent=agent,
            config=config,
            base_reward_config=base_cfg,
            progress_bar=False,
            verbose=0,
        )
        MockCB.assert_called_once()
        call_kwargs = MockCB.call_args
        assert call_kwargs.kwargs.get("base_reward_config") == base_cfg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_env.py::TestCurriculumCallback::test_train_passes_base_reward_config -v`
Expected: FAIL — `train()` does not accept `base_reward_config` keyword argument.

- [ ] **Step 3: Implement the change**

In `src/alphacluster/agent/trainer.py`, modify the `train()` function signature (line ~507) to add the parameter:

```python
def train(
    agent: PPO,
    config: TrainingConfig,
    eval_env: gym.Env | None = None,
    checkpoint_dir: str | Path | None = None,
    run_tournament: bool = False,
    progress_bar: bool = True,
    verbose: int = 1,
    extra_callbacks: list[BaseCallback] | None = None,
    base_reward_config: dict[str, float] | None = None,
) -> PPO:
```

Update the docstring to document the new parameter:

```python
    base_reward_config:
        Optional dict overriding default base scales for CurriculumCallback.
        Keys are reward config names (e.g. ``fee_scale``).  Phase-specific
        values are computed as ``base × phase_multiplier``.
```

Modify line ~593 to pass it through:

```python
    if config.curriculum_enabled:
        callbacks.append(
            CurriculumCallback(
                config, verbose=verbose, base_reward_config=base_reward_config
            )
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_env.py::TestCurriculumCallback::test_train_passes_base_reward_config -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All 200 tests PASS (199 existing + 1 new; no regressions — existing callers don't pass the new kwarg).

- [ ] **Step 6: Lint**

Run: `.venv/bin/ruff check src/alphacluster/agent/trainer.py tests/test_env.py`
Expected: Clean

- [ ] **Step 7: Commit**

```bash
git add src/alphacluster/agent/trainer.py tests/test_env.py
git commit -m "feat: add base_reward_config passthrough to train()"
```

---

### Task 2: Create ablation sweep notebook

**Files:**
- Create: `notebooks/colab_ablation.ipynb`

This notebook has 6 cells. It reuses patterns from `notebooks/colab_train_simple.ipynb` (mount Drive, load data, create envs). The key difference is the sweep loop in Cell 4.

- [ ] **Step 1: Create the notebook with Cell 1 — Mount Drive & Install**

Create `notebooks/colab_ablation.ipynb` with a markdown intro cell and Cell 1:

Markdown cell:
```markdown
# AlphaCluster — Ablation Sweep (Reward Configuration Diagnostic)

Run 4 reward configurations sequentially to diagnose which components cause
the always-flat collapse observed in v4 (25 features).

Each config trains for 1M steps with a slower curriculum
(Phase 1: 0-50%, Phase 2: 50-80%, Phase 3: 80-100%).

**Configs:**
1. `pnl_only` — asymmetric PnL only, all penalties disabled
2. `pnl_fees` — PnL + gradual fee penalty
3. `pnl_fees_opp` — PnL + fees + strengthened opportunity cost
4. `full_slow` — full reward, slower curriculum

**Expected runtime:** ~2h on Colab L4 GPU (4 × 1M steps × ~30 min each)

Results saved to Google Drive: `My Drive/alphacluster/ablation_results/`
```

Cell 1 code (same as colab_train_simple.ipynb):
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
    "pyarrow", "python-dotenv", "tqdm", "rich"], check=True)

PROJECT_ROOT = DRIVE_ROOT
sys.path.insert(0, str(LOCAL_SRC))

import alphacluster
print(f"AlphaCluster loaded from {Path(alphacluster.__file__).parent}")

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: No GPU detected. Enable GPU runtime: Runtime > Change runtime type > T4 GPU")
```

- [ ] **Step 2: Add Cell 2 — Load & Split Data**

```python
import pandas as pd

DATA_DIR = Path("/content/drive/MyDrive/alphacluster/data")
KLINES_PATH = DATA_DIR / "btcusdt_5m.parquet"
FUNDING_PATH = DATA_DIR / "btcusdt_funding.parquet"

assert KLINES_PATH.exists(), (
    f"Kline data not found at {KLINES_PATH}\n"
    f"Upload btcusdt_5m.parquet to Google Drive: My Drive/alphacluster/data/"
)

klines_df = pd.read_parquet(KLINES_PATH, engine="pyarrow")
print(f"Loaded {len(klines_df):,} candles")
print(f"Date range: {klines_df.iloc[0, 0]} to {klines_df.iloc[-1, 0]}")

funding_df = None
if FUNDING_PATH.exists():
    funding_df = pd.read_parquet(FUNDING_PATH, engine="pyarrow")
    print(f"Loaded {len(funding_df):,} funding records")
else:
    print("No funding data found; training without funding rates.")

n = len(klines_df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = klines_df.iloc[:train_end].reset_index(drop=True)
val_df = klines_df.iloc[train_end:val_end].reset_index(drop=True)
test_df = klines_df.iloc[val_end:].reset_index(drop=True)

print(f"\nData split: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")
```

- [ ] **Step 3: Add Cell 3 — Define Ablation Configs**

```python
ABLATION_CONFIGS = [
    {
        "name": "pnl_only",
        "description": "Asymmetric PnL only, all penalties disabled",
        "base_reward_config": {
            "fee_scale": 0.0,
            "churn_penalty_scale": 0.0,
            "opportunity_cost_scale": 0.0,
            "drawdown_penalty_scale": 0.0,
            "quality_scale": 0.0,
            "position_mgmt_scale": 0.0,
        },
    },
    {
        "name": "pnl_fees",
        "description": "PnL + gradual fee penalty (base 0.5)",
        "base_reward_config": {
            "fee_scale": 0.5,
            "churn_penalty_scale": 0.0,
            "opportunity_cost_scale": 0.0,
            "drawdown_penalty_scale": 0.0,
            "quality_scale": 0.0,
            "position_mgmt_scale": 0.0,
        },
    },
    {
        "name": "pnl_fees_opp",
        "description": "PnL + fees + stronger opportunity cost (cap 0.05, threshold 0.1%)",
        "base_reward_config": {
            "fee_scale": 0.5,
            "churn_penalty_scale": 0.0,
            "opportunity_cost_scale": 1.0,
            "opportunity_cost_cap": 0.05,
            "opportunity_cost_threshold": 0.001,
            "drawdown_penalty_scale": 0.0,
            "quality_scale": 0.0,
            "position_mgmt_scale": 0.0,
        },
    },
    {
        "name": "full_slow",
        "description": "Full reward with slow curriculum",
        "base_reward_config": {
            "opportunity_cost_scale": 0.5,
            "opportunity_cost_cap": 0.02,
            "opportunity_cost_threshold": 0.002,
        },
    },
]

RESULTS_DIR = Path("/content/drive/MyDrive/alphacluster/ablation_results")
TOTAL_TIMESTEPS = 1_000_000
PHASE1_END = 0.5
PHASE2_END = 0.8

print(f"Ablation sweep: {len(ABLATION_CONFIGS)} configs × {TOTAL_TIMESTEPS:,} steps")
print(f"Curriculum: Phase 1 (0-{PHASE1_END:.0%}) → Phase 2 ({PHASE1_END:.0%}-{PHASE2_END:.0%}) → Phase 3 ({PHASE2_END:.0%}-100%)")
print(f"Results saved to: {RESULTS_DIR}")
for i, cfg in enumerate(ABLATION_CONFIGS, 1):
    print(f"  {i}. {cfg['name']}: {cfg['description']}")
```

- [ ] **Step 4: Add Cell 4 — Sweep Loop**

```python
import time
from alphacluster.agent.config import TrainingConfig
from alphacluster.agent.trainer import create_agent, save_agent, train
from alphacluster.env.trading_env import TradingEnv
from alphacluster.config import MODEL_VERSION
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

N_ENVS = 4


class ProgressCallback(BaseCallback):
    """Print a one-line progress update every ``log_interval`` timesteps."""

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


def make_train_env(rank: int):
    def _init():
        import os, warnings
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
        env = TradingEnv(
            df=train_df, funding_df=funding_df,
            window_size=576, episode_length=2016,
            simple_actions=True, fixed_size_pct=0.10, fixed_leverage=10,
        )
        env.reset(seed=rank)
        return env
    return _init


import numpy as np

all_results = {}

for cfg_idx, ablation_cfg in enumerate(ABLATION_CONFIGS):
    name = ablation_cfg["name"]
    print(f"\n{'='*60}")
    print(f"  CONFIG {cfg_idx+1}/{len(ABLATION_CONFIGS)}: {name}")
    print(f"  {ablation_cfg['description']}")
    print(f"{'='*60}\n")

    # Seed for reproducibility across configs
    torch.manual_seed(42)
    np.random.seed(42)

    # Output directory on Drive
    run_dir = RESULTS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Training config with slow curriculum
        config = TrainingConfig(
            total_timesteps=TOTAL_TIMESTEPS,
            simple_actions=True,
            fixed_size_pct=0.10,
            fixed_leverage=10,
            phase1_end=PHASE1_END,
            phase2_end=PHASE2_END,
            eval_freq=100_000,
        )

        # Create parallel training envs
        envs = SubprocVecEnv([make_train_env(i) for i in range(N_ENVS)])
        train_env = VecNormalize(envs, norm_obs=False, norm_reward=True, clip_reward=10.0)

        # Create eval env
        eval_env = TradingEnv(
            df=val_df, funding_df=funding_df,
            window_size=576, episode_length=2016,
            simple_actions=True, fixed_size_pct=0.10, fixed_leverage=10,
        )

        # Create and train agent
        agent = create_agent(train_env, config, verbose=0)
        progress_cb = ProgressCallback(TOTAL_TIMESTEPS)

        agent = train(
            agent=agent,
            config=config,
            eval_env=eval_env,
            checkpoint_dir=str(run_dir),
            run_tournament=False,
            progress_bar=False,
            verbose=0,
            extra_callbacks=[progress_cb],
            base_reward_config=ablation_cfg["base_reward_config"],
        )

        # Save model to Drive
        model_path = run_dir / "model"
        save_agent(agent, model_path)
        print(f"\nModel saved to {model_path}.pt")

        # Read metrics
        metrics_path = run_dir / "training_metrics.csv"
        if metrics_path.exists():
            all_results[name] = pd.read_csv(metrics_path)
            print(f"Metrics: {len(all_results[name])} checkpoints logged")
        else:
            print("WARNING: No metrics file found!")

    except Exception as e:
        print(f"\nCONFIG {name} FAILED: {e}")
        all_results[name] = None

    finally:
        # Cleanup envs to free memory
        for v in ["train_env", "eval_env", "envs", "agent"]:
            obj = locals().get(v)
            if obj is not None and hasattr(obj, "close"):
                obj.close()

print(f"\n{'='*60}")
succeeded = sum(1 for v in all_results.values() if v is not None)
print(f"  SWEEP COMPLETE — {succeeded}/{len(ABLATION_CONFIGS)} configs succeeded")
print(f"{'='*60}")
```

- [ ] **Step 5: Add Cell 5 — Comparison Plots**

```python
import matplotlib.pyplot as plt

if not all_results:
    # Try to load from Drive (in case re-running just this cell)
    for cfg in ABLATION_CONFIGS:
        p = RESULTS_DIR / cfg["name"] / "training_metrics.csv"
        if p.exists():
            all_results[cfg["name"]] = pd.read_csv(p)

if not all_results:
    print("No results found. Run the sweep first.")
else:
    metrics_to_plot = [
        ("mean_reward", "Mean Episode Return (%)"),
        ("trades_per_episode", "Trades per Episode"),
        ("win_rate", "Win Rate (%)"),
        ("avg_trade_duration", "Avg Trade Duration (steps)"),
        ("flat_pct", "Flat Time (%)"),
        ("total_pnl_pct", "Total PnL (%)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Ablation Sweep — {MODEL_VERSION} (slow curriculum: P1→50%, P2→80%)",
        fontsize=14,
    )

    colors = {"pnl_only": "#2196F3", "pnl_fees": "#FF9800",
              "pnl_fees_opp": "#4CAF50", "full_slow": "#F44336"}

    for ax, (col, title) in zip(axes.flat, metrics_to_plot):
        for name, df in all_results.items():
            if df is not None and col in df.columns:
                ax.plot(df["timestep"], df[col], label=name,
                        color=colors.get(name, None), marker="o", markersize=3)
        ax.set_title(title)
        ax.set_xlabel("Timesteps")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Mark curriculum phase boundaries
        for pct, label in [(PHASE1_END, "P2"), (PHASE2_END, "P3")]:
            ts = int(TOTAL_TIMESTEPS * pct)
            ax.axvline(x=ts, color="gray", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "ablation_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Plot saved to {RESULTS_DIR / 'ablation_comparison.png'}")
    plt.show()
```

- [ ] **Step 6: Add Cell 6 — Summary Table**

```python
if all_results:
    print(f"\n{'Config':<16} {'Trades/ep':>10} {'Win Rate':>10} {'Flat %':>10} {'PnL %':>10} {'Fee/PnL':>10}")
    print("-" * 68)
    for name, df in all_results.items():
        if df is None:
            print(f"{name:<16} {'FAILED':>10}")
            continue
        last = df.iloc[-1]
        print(
            f"{name:<16} "
            f"{last['trades_per_episode']:>10.1f} "
            f"{last['win_rate']:>9.1f}% "
            f"{last['flat_pct']:>9.1f}% "
            f"{last['total_pnl_pct']:>9.2f}% "
            f"{last['fee_to_pnl_ratio']:>9.1f}%"
        )

    print(f"\nResults on Drive: {RESULTS_DIR}")
    print("Next steps:")
    print("  - If pnl_only collapses: architecture issue, consider smaller observation window")
    print("  - If pnl_fees collapses but pnl_only works: fee penalty too aggressive")
    print("  - If pnl_fees_opp works: opportunity cost prevents collapse, use as base config")
    print("  - If full_slow works: only needed slower curriculum, no reward changes")
else:
    print("No results to summarize.")
```

- [ ] **Step 7: Verify notebook structure**

Open the notebook and verify:
- 7 cells total (1 markdown intro + 6 code cells)
- No syntax errors
- All imports present

- [ ] **Step 8: Lint**

Run: `.venv/bin/ruff check src/ tests/`
Expected: Clean

- [ ] **Step 9: Commit**

```bash
git add notebooks/colab_ablation.ipynb
git commit -m "feat: add ablation sweep notebook for reward config diagnostic"
```

---

## Verification

1. `.venv/bin/python -m pytest tests/ -v` — all 200 tests pass
2. `.venv/bin/ruff check src/ tests/` — clean
3. Upload `src/` to Drive, open `colab_ablation.ipynb` on Colab, run all cells
4. After ~2h: check `Drive/alphacluster/ablation_results/` for 4 subdirectories with metrics CSV + model files
5. Cell 5 overlay plots show learning curves for all 4 configs
6. Cell 6 summary table compares final metrics
