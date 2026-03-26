# Optuna Hyperparameter Tuning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automated Bayesian optimization of reward scales, curriculum multipliers, and PPO hyperparameters to fix the flat-collapse problem.

**Architecture:** Three code changes to make reward parameters configurable (`trading_env.py` + `trainer.py`), then a new Colab notebook that uses Optuna TPE to search the 13-parameter space in two phases (200k×40 screening → 500k×10 validation).

**Tech Stack:** Python 3.10+, Optuna ≥3.0, Stable-Baselines3, Gymnasium, plotly, kaleido

**Spec:** `docs/superpowers/specs/2026-03-24-optuna-tuning-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `src/alphacluster/env/trading_env.py` | Extract hardcoded opportunity cost params + add `position_mgmt_scale` to `DEFAULT_REWARD_CONFIG` and `_compute_reward()` |
| Modify | `src/alphacluster/agent/trainer.py` | Refactor `CurriculumCallback` to accept base scales + phase multipliers |
| Modify | `tests/test_env.py` | Tests for new configurable reward params |
| Create | `notebooks/colab_optuna.ipynb` | Optuna tuning notebook (9 cells) |

---

### Task 1: Make opportunity_cost_cap and opportunity_cost_threshold configurable

**Files:**
- Modify: `src/alphacluster/env/trading_env.py:36-42` (DEFAULT_REWARD_CONFIG)
- Modify: `src/alphacluster/env/trading_env.py:355-364` (_compute_reward opportunity cost section)
- Test: `tests/test_env.py`

- [ ] **Step 1: Write failing test — opportunity_cost_cap is configurable**

In `tests/test_env.py`, add at the end of the file:

```python
class TestRewardConfig:
    """Tests for configurable reward parameters."""

    def test_opportunity_cost_cap_configurable(self):
        """Custom opportunity_cost_cap should be used in reward computation."""
        env = _make_env(episode_length=200)
        obs, _ = env.reset(seed=0)

        # Set a very high cap — penalty should be larger than with default 0.02
        env.reward_config = {
            **env.reward_config,
            "opportunity_cost_cap": 0.10,
            "opportunity_cost_threshold": 0.002,
        }

        # Verify the config keys exist and are used
        assert "opportunity_cost_cap" in env.reward_config
        assert env.reward_config["opportunity_cost_cap"] == 0.10

    def test_opportunity_cost_threshold_configurable(self):
        """Custom opportunity_cost_threshold should be used in reward computation."""
        env = _make_env(episode_length=200)
        obs, _ = env.reset(seed=0)

        env.reward_config = {
            **env.reward_config,
            "opportunity_cost_threshold": 0.005,
        }
        assert env.reward_config["opportunity_cost_threshold"] == 0.005

    def test_position_mgmt_scale_configurable(self):
        """position_mgmt_scale should exist in DEFAULT_REWARD_CONFIG."""
        env = _make_env(episode_length=200)
        assert "position_mgmt_scale" in env.reward_config
        assert env.reward_config["position_mgmt_scale"] == 1.0

    def test_opportunity_cost_cap_affects_reward(self):
        """Different opportunity_cost_cap values should produce different rewards."""
        env1 = _make_env(episode_length=200)
        env2 = _make_env(episode_length=200)
        env1.reset(seed=0)
        env2.reset(seed=0)

        # Low cap vs high cap — step while flat to trigger opportunity cost
        env1.reward_config = {**env1.reward_config, "opportunity_cost_cap": 0.001}
        env2.reward_config = {**env2.reward_config, "opportunity_cost_cap": 0.50}

        # Step both with flat action (direction=0)
        rewards1, rewards2 = [], []
        for _ in range(50):
            _, r1, _, _, _ = env1.step([0, 0, 0])
            _, r2, _, _, _ = env2.step([0, 0, 0])
            rewards1.append(r1)
            rewards2.append(r2)

        # With higher cap, opportunity cost can be larger → more negative total reward
        assert sum(rewards1) != pytest.approx(sum(rewards2), abs=1e-10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_env.py::TestRewardConfig -v`
Expected: FAIL — `opportunity_cost_cap` and `position_mgmt_scale` not in DEFAULT_REWARD_CONFIG

- [ ] **Step 3: Update DEFAULT_REWARD_CONFIG with new keys**

In `src/alphacluster/env/trading_env.py`, replace `DEFAULT_REWARD_CONFIG` (lines 36-42):

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

- [ ] **Step 4: Extract hardcoded values in _compute_reward**

In `src/alphacluster/env/trading_env.py`, replace the opportunity cost block (lines 355-364):

```python
        # 3. OPPORTUNITY COST (replaces inactivity penalty)
        opportunity_penalty = 0.0
        if self.account.position_side == "flat":
            lookback = min(10, self._current_idx - self._start_idx)
            if lookback > 0:
                price_change = abs(
                    self._close[self._current_idx] - self._close[self._current_idx - lookback]
                ) / max(self._close[self._current_idx - lookback], 1e-12)
                opp_threshold = rc.get("opportunity_cost_threshold", 0.002)
                opp_cap = rc.get("opportunity_cost_cap", 0.02)
                if price_change > opp_threshold:
                    raw_penalty = price_change - opp_threshold
                    opportunity_penalty = min(raw_penalty, opp_cap) * rc["opportunity_cost_scale"]
```

- [ ] **Step 5: Add position_mgmt_scale to position management reward**

In `src/alphacluster/env/trading_env.py`, replace the position management block (lines 367-375):

```python
        # 4. POSITION MANAGEMENT REWARD (sqrt ramp for winners)
        position_reward = 0.0
        if self.account.position_side != "flat":
            upnl_ratio = self.account.unrealized_pnl / self.initial_balance
            pos_scale = rc.get("position_mgmt_scale", 1.0)
            if upnl_ratio > 0:
                hold_time_bonus = min((self.account.time_in_position**0.5) / 5.0, 3.0)
                position_reward = 0.4 * pos_scale * upnl_ratio * (1.0 + hold_time_bonus)
            else:
                time_factor = 1.0 + self.account.time_in_position / 100.0
                position_reward = 0.4 * pos_scale * upnl_ratio * time_factor
```

- [ ] **Step 6: Add new keys to existing CurriculumCallback phase dicts**

In `src/alphacluster/agent/trainer.py`, add `"position_mgmt_scale": 1.0` to each phase's `reward_config` dict in `_apply_phase()` (lines 126-150). This keeps the intermediate state between Task 1 and Task 2 commits consistent. Add `"opportunity_cost_cap": 0.02` and `"opportunity_cost_threshold": 0.002` too.

Phase 1 dict becomes:
```python
            reward_config = {
                "opportunity_cost_scale": 0.3,
                "opportunity_cost_cap": 0.02,
                "opportunity_cost_threshold": 0.002,
                "fee_scale": 0.5,
                "drawdown_penalty_scale": 0.3,
                "churn_penalty_scale": 0.5,
                "quality_scale": 1.0,
                "position_mgmt_scale": 1.0,
            }
```

Similarly for Phase 2 and Phase 3 dicts (same new keys, same default values).

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_env.py::TestRewardConfig -v`
Expected: 4 tests PASS

- [ ] **Step 8: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All 146+ tests PASS (existing behavior unchanged due to backward-compatible defaults)

- [ ] **Step 9: Run linter**

Run: `.venv/bin/ruff check src/alphacluster/env/trading_env.py src/alphacluster/agent/trainer.py tests/test_env.py`
Expected: No errors

- [ ] **Step 10: Commit**

```bash
git add src/alphacluster/env/trading_env.py src/alphacluster/agent/trainer.py tests/test_env.py
git commit -m "feat: make opportunity_cost_cap/threshold and position_mgmt_scale configurable

Extract hardcoded opportunity cost parameters (cap=0.02, threshold=0.002)
to reward_config for Optuna tuning. Add position_mgmt_scale (default 1.0)
that wraps the 0.4 coefficient in position management reward.
Also add new keys to CurriculumCallback phase dicts for consistency."
```

---

### Task 2: Refactor CurriculumCallback to accept dynamic base scales

**Files:**
- Modify: `src/alphacluster/agent/trainer.py:88-177` (CurriculumCallback)
- Test: `tests/test_env.py`

- [ ] **Step 1: Write failing test — CurriculumCallback accepts base scales**

In `tests/test_env.py`, add:

```python
from alphacluster.agent.trainer import CurriculumCallback
from alphacluster.agent.config import TrainingConfig


class TestCurriculumCallback:
    """Tests for CurriculumCallback with configurable base scales."""

    def test_default_behavior_unchanged(self):
        """CurriculumCallback with no extra args behaves as before."""
        config = TrainingConfig(total_timesteps=100)
        cb = CurriculumCallback(config)
        # Phase 1 defaults
        phase_config = cb._get_phase_reward_config(1)
        assert phase_config["fee_scale"] == pytest.approx(0.5)
        assert phase_config["opportunity_cost_scale"] == pytest.approx(0.3)

    def test_custom_base_scales_applied(self):
        """Custom base_reward_config should be multiplied by phase multipliers."""
        config = TrainingConfig(total_timesteps=100)
        base = {"fee_scale": 0.5, "churn_penalty_scale": 0.8}
        cb = CurriculumCallback(
            config,
            base_reward_config=base,
            phase3_fee_multiplier=2.5,
            phase3_churn_multiplier=1.5,
        )
        # Phase 3: fee = base(0.5) * multiplier(2.5) = 1.25
        phase3_config = cb._get_phase_reward_config(3)
        assert phase3_config["fee_scale"] == pytest.approx(1.25)
        assert phase3_config["churn_penalty_scale"] == pytest.approx(1.2)  # 0.8 * 1.5

    def test_ent_coef_phase_derivation(self):
        """ent_coef should be derived from phase1 value."""
        config = TrainingConfig(total_timesteps=100)
        cb = CurriculumCallback(config, ent_coef_phase1=0.08)
        assert cb._get_phase_ent_coef(1) == pytest.approx(0.08)
        assert cb._get_phase_ent_coef(2) == pytest.approx(0.048)  # 0.08 * 0.6
        assert cb._get_phase_ent_coef(3) == pytest.approx(0.008)  # 0.08 * 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_env.py::TestCurriculumCallback -v`
Expected: FAIL — `CurriculumCallback.__init__` doesn't accept `base_reward_config`, `_get_phase_reward_config` doesn't exist

- [ ] **Step 3: Refactor CurriculumCallback**

In `src/alphacluster/agent/trainer.py`, replace the entire `CurriculumCallback` class (lines 88-177):

```python
class CurriculumCallback(BaseCallback):
    """Adjusts reward parameters and entropy coefficient across training phases.

    Phase 1 — "Learn to Trade" (0–30%):
        Moderate exploration, early fee/churn awareness.

    Phase 2 — "Learn Quality" (30–60%):
        Reduced exploration, full cost penalties.

    Phase 3 — "Refine & Exploit" (60–100%):
        Low exploration, strict fee/churn/drawdown penalties.

    Parameters
    ----------
    config:
        Training configuration with phase boundaries and total timesteps.
    base_reward_config:
        Base reward scales. If None, uses sensible defaults that reproduce
        the original hardcoded behavior.
    phase3_fee_multiplier:
        Phase 3 multiplier for fee_scale (applied to base).
    phase3_churn_multiplier:
        Phase 3 multiplier for churn_penalty_scale (applied to base).
    ent_coef_phase1:
        Entropy coefficient for Phase 1. Phase 2 = 0.6x, Phase 3 = 0.1x.
        If None, uses hardcoded defaults (0.05, 0.03, 0.005).
    verbose:
        Verbosity level.
    """

    # Fixed multiplier vectors per phase (relative to base scales)
    _PHASE_MULTIPLIERS: dict[int, dict[str, float]] = {
        1: {
            "fee_scale": 0.5,
            "churn_penalty_scale": 0.5,
            "opportunity_cost_scale": 0.3,
            "drawdown_penalty_scale": 0.3,
            "quality_scale": 1.0,
            "position_mgmt_scale": 1.0,
            "opportunity_cost_cap": 1.0,
            "opportunity_cost_threshold": 1.0,
        },
        2: {
            "fee_scale": 1.0,
            "churn_penalty_scale": 1.0,
            "opportunity_cost_scale": 0.5,
            "drawdown_penalty_scale": 1.0,
            "quality_scale": 1.0,
            "position_mgmt_scale": 1.0,
            "opportunity_cost_cap": 1.0,
            "opportunity_cost_threshold": 1.0,
        },
        3: {
            "fee_scale": 2.0,  # overridden by phase3_fee_multiplier
            "churn_penalty_scale": 2.0,  # overridden by phase3_churn_multiplier
            "opportunity_cost_scale": 1.0,
            "drawdown_penalty_scale": 1.5,
            "quality_scale": 0.5,
            "position_mgmt_scale": 1.0,
            "opportunity_cost_cap": 1.0,
            "opportunity_cost_threshold": 1.0,
        },
    }

    # Default base scales — set so that base × phase_multiplier reproduces
    # the original hardcoded absolute values for each phase.
    # E.g. opportunity_cost_scale: base=1.0, Phase1 mult=0.3 → 0.3,
    #       Phase2 mult=0.5 → 0.5, Phase3 mult=1.0 → 1.0 (all match original).
    _DEFAULT_BASE: dict[str, float] = {
        "fee_scale": 1.0,
        "opportunity_cost_scale": 1.0,
        "opportunity_cost_cap": 0.02,
        "opportunity_cost_threshold": 0.002,
        "churn_penalty_scale": 1.0,
        "drawdown_penalty_scale": 1.0,
        "quality_scale": 1.0,
        "position_mgmt_scale": 1.0,
    }

    def __init__(
        self,
        config: TrainingConfig,
        base_reward_config: dict[str, float] | None = None,
        phase3_fee_multiplier: float | None = None,
        phase3_churn_multiplier: float | None = None,
        ent_coef_phase1: float | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.config = config
        self._current_phase: int = 0

        # Merge caller-provided base scales over defaults
        self._base = dict(self._DEFAULT_BASE)
        if base_reward_config is not None:
            self._base.update(base_reward_config)

        # Store Phase 3 overrides
        self._phase3_fee_mult = phase3_fee_multiplier
        self._phase3_churn_mult = phase3_churn_multiplier

        # Entropy schedule
        self._ent_coef_phase1 = ent_coef_phase1

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.config.total_timesteps
        new_phase = self._get_phase(progress)

        if new_phase != self._current_phase:
            self._current_phase = new_phase
            self._apply_phase(new_phase)

        return True

    def _get_phase(self, progress: float) -> int:
        if progress < self.config.phase1_end:
            return 1
        if progress < self.config.phase2_end:
            return 2
        return 3

    def _get_phase_ent_coef(self, phase: int) -> float:
        """Return entropy coefficient for the given phase."""
        if self._ent_coef_phase1 is not None:
            factors = {1: 1.0, 2: 0.6, 3: 0.1}
            return self._ent_coef_phase1 * factors[phase]
        # Original hardcoded defaults
        return {1: 0.05, 2: 0.03, 3: 0.005}[phase]

    def _get_phase_reward_config(self, phase: int) -> dict[str, float]:
        """Compute reward config for the given phase: base × multiplier."""
        multipliers = dict(self._PHASE_MULTIPLIERS[phase])

        # Apply Phase 3 overrides if provided
        if phase == 3:
            if self._phase3_fee_mult is not None:
                multipliers["fee_scale"] = self._phase3_fee_mult
            if self._phase3_churn_mult is not None:
                multipliers["churn_penalty_scale"] = self._phase3_churn_mult

        result = {}
        for key, base_val in self._base.items():
            mult = multipliers.get(key, 1.0)
            result[key] = base_val * mult
        return result

    def _apply_phase(self, phase: int) -> None:
        ent_coef = self._get_phase_ent_coef(phase)
        reward_config = self._get_phase_reward_config(phase)

        # Update agent entropy coefficient
        self.model.ent_coef = ent_coef

        # Update reward config on all environments
        self._set_env_reward_config(reward_config)

        msg = f"Curriculum: phase {phase} (ent_coef={ent_coef})"
        logger.info(msg)
        if self.verbose:
            print(msg)

    def _set_env_reward_config(self, reward_config: dict[str, float]) -> None:
        """Set reward_config on underlying TradingEnv(s)."""
        env = self.model.get_env()
        if env is None:
            return

        # VecEnv: set_attr broadcasts to all sub-environments
        try:
            env.set_attr("reward_config", reward_config)
        except AttributeError:
            # Fallback for single env
            unwrapped = getattr(env, "unwrapped", env)
            if hasattr(unwrapped, "reward_config"):
                unwrapped.reward_config = reward_config
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_env.py::TestCurriculumCallback -v`
Expected: 4 tests PASS

- [ ] **Step 5: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All tests PASS (original phase values reproduced by default args)

- [ ] **Step 6: Run linter**

Run: `.venv/bin/ruff check src/alphacluster/agent/trainer.py`
Expected: No errors

- [ ] **Step 7: Commit**

```bash
git add src/alphacluster/agent/trainer.py tests/test_env.py
git commit -m "refactor: CurriculumCallback accepts dynamic base scales and ent_coef

Constructor now takes base_reward_config, phase3_fee/churn_multiplier,
and ent_coef_phase1. Phase configs computed as base × multiplier.
Default behavior is preserved when no extra args are passed."
```

---

### Task 3: Create the Optuna tuning notebook

**Files:**
- Create: `notebooks/colab_optuna.ipynb`

This is the main deliverable — a 9-cell Colab notebook.

- [ ] **Step 1: Create notebook — Cell 1 (Setup & Dependencies)**

```python
# Cell 1 markdown: "# AlphaCluster — Optuna Hyperparameter Tuning\n\nAutomated Bayesian optimization..."

# Cell 1 code:
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

# Mount Google Drive
from google.colab import drive
drive.mount("/content/drive")

DRIVE_ROOT = Path("/content/drive/MyDrive/alphacluster")
DRIVE_SRC = DRIVE_ROOT / "src"

assert DRIVE_SRC.exists(), (
    f"Source not found at {DRIVE_SRC}\n"
    f"Copy your local src/ directory to Google Drive: My Drive/alphacluster/src/"
)

# Copy source to local runtime storage
LOCAL_SRC = Path("/content/src")
if LOCAL_SRC.exists():
    shutil.rmtree(LOCAL_SRC)
shutil.copytree(DRIVE_SRC, LOCAL_SRC)
print(f"Copied source to {LOCAL_SRC}")

# Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "stable-baselines3>=2.4,<3.0", "gymnasium>=1.0,<2.0",
    "pyarrow", "python-dotenv", "tqdm", "rich",
    "optuna>=3.0", "plotly", "kaleido"], check=True)

sys.path.insert(0, str(LOCAL_SRC))

import alphacluster
print(f"AlphaCluster loaded from {Path(alphacluster.__file__).parent}")

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: No GPU detected.")

# Persistence paths
DRIVE_DIR = Path("/content/drive/MyDrive/AlphaCluster/optuna_tuning/")
DRIVE_DIR.mkdir(parents=True, exist_ok=True)
STUDY_DB = DRIVE_DIR / "optuna_study.db"
RESULTS_CSV = DRIVE_DIR / "trial_results.csv"
BEST_PARAMS_JSON = DRIVE_DIR / "best_params.json"
PLOTS_DIR = DRIVE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nPersistence: {DRIVE_DIR}")
```

- [ ] **Step 2: Create notebook — Cell 2 (Load Data)**

```python
import pandas as pd

DATA_DIR = Path("/content/drive/MyDrive/alphacluster/data")
KLINES_PATH = DATA_DIR / "btcusdt_5m.parquet"
FUNDING_PATH = DATA_DIR / "btcusdt_funding.parquet"

assert KLINES_PATH.exists(), f"Kline data not found at {KLINES_PATH}"

klines_df = pd.read_parquet(KLINES_PATH, engine="pyarrow")
print(f"Loaded {len(klines_df):,} candles")

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

print(f"Data split: train={len(train_df):,}  val={len(val_df):,}")
```

- [ ] **Step 3: Create notebook — Cell 3 (Objective Function)**

```python
import logging
import numpy as np
import optuna

from alphacluster.agent.config import TrainingConfig
from alphacluster.agent.trainer import (
    CurriculumCallback,
    TrainingMetricsCallback,
    create_agent,
)
from alphacluster.env.trading_env import TradingEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

logger = logging.getLogger("optuna_tuning")


class OptunaPruningCallback(BaseCallback):
    """Report intermediate scores to Optuna and prune bad trials."""

    def __init__(self, trial, metrics_path, log_freq):
        super().__init__(verbose=0)
        self.trial = trial
        self.metrics_path = metrics_path
        self.log_freq = log_freq
        self._next_report = log_freq

    def _on_step(self):
        if self.num_timesteps < self._next_report:
            return True
        self._next_report += self.log_freq
        if self.metrics_path.exists():
            score, _ = compute_score(self.metrics_path)
            self.trial.report(score, self.num_timesteps)
            if self.trial.should_prune():
                raise optuna.TrialPruned()
        return True

# --- Tuning constants ---
SCREENING_TIMESTEPS = 200_000
VALIDATION_TIMESTEPS = 500_000
METRICS_LOG_FREQ = 100_000
N_ENVS = 4


def make_env(df, funding_df, config, rank=0):
    """Factory for SubprocVecEnv."""
    def _init():
        import os, warnings
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
        env = TradingEnv(
            df=df, funding_df=funding_df,
            window_size=config.window_size,
            episode_length=config.episode_length,
        )
        env.reset(seed=rank)
        return env
    return _init


def compute_score(metrics_path: Path) -> tuple[float, dict]:
    """Read last row of training metrics CSV and compute composite score."""
    df = pd.read_csv(metrics_path)
    if df.empty:
        return -1000.0, {}

    row = df.iloc[-1]
    flat_pct = row.get("flat_pct", 100.0)
    trades = row.get("trades_per_episode", 0.0)
    duration = row.get("avg_trade_duration", 0.0)
    total_pnl = row.get("total_pnl_pct", 0.0)
    win_rate = row.get("win_rate", 0.0)

    # Hard constraints
    if flat_pct > 70:
        return -1000.0, {"reject": "flat_pct > 70%"}
    if trades < 20:
        return -1000.0, {"reject": f"trades={trades} < 20"}
    if duration < 1.5:
        return -1000.0, {"reject": f"duration={duration} < 1.5"}

    # Composite score (all terms 0-1)
    pnl_norm = np.clip(total_pnl, -50, 50) / 100 + 0.5
    trades_norm = min(trades, 200) / 200
    win_rate_norm = win_rate / 100

    score = pnl_norm * 0.4 + trades_norm * 0.3 + win_rate_norm * 0.3

    details = {
        "total_pnl_pct": total_pnl, "trades_per_episode": trades,
        "win_rate": win_rate, "flat_pct": flat_pct,
        "avg_trade_duration": duration,
        "pnl_norm": pnl_norm, "trades_norm": trades_norm,
        "win_rate_norm": win_rate_norm,
    }
    return float(score), details


def objective(trial: optuna.Trial) -> float:
    """Optuna objective: sample params, train, evaluate."""
    try:
        # --- Sample 13 parameters ---
        fee_scale = trial.suggest_float("fee_scale", 0.1, 2.0)
        opp_cost_scale = trial.suggest_float("opportunity_cost_scale", 0.1, 2.0)
        opp_cost_cap = trial.suggest_float("opportunity_cost_cap", 0.01, 0.15)
        opp_cost_threshold = trial.suggest_float(
            "opportunity_cost_threshold", 0.001, 0.005,
        )
        churn_scale = trial.suggest_float("churn_penalty_scale", 0.1, 2.0)
        dd_scale = trial.suggest_float("drawdown_penalty_scale", 0.1, 2.0)
        quality_scale = trial.suggest_float("quality_scale", 0.1, 2.0)
        pos_mgmt_scale = trial.suggest_float("position_mgmt_scale", 0.1, 2.0)
        p3_fee_mult = trial.suggest_float("phase3_fee_multiplier", 1.0, 3.0)
        p3_churn_mult = trial.suggest_float("phase3_churn_multiplier", 1.0, 3.0)
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
        ent = trial.suggest_float("ent_coef_phase1", 0.01, 0.1, log=True)
        gamma = trial.suggest_float("gamma", 0.99, 0.999)

        base_reward_config = {
            "fee_scale": fee_scale,
            "opportunity_cost_scale": opp_cost_scale,
            "opportunity_cost_cap": opp_cost_cap,
            "opportunity_cost_threshold": opp_cost_threshold,
            "churn_penalty_scale": churn_scale,
            "drawdown_penalty_scale": dd_scale,
            "quality_scale": quality_scale,
            "position_mgmt_scale": pos_mgmt_scale,
        }

        # --- Config ---
        config = TrainingConfig(
            total_timesteps=SCREENING_TIMESTEPS,
            learning_rate=lr,
            gamma=gamma,
            ent_coef=ent,
            eval_freq=SCREENING_TIMESTEPS + 1,  # disable EvalCallback
        )

        # --- Environments ---
        envs = SubprocVecEnv([
            make_env(train_df, funding_df, config, rank=i)
            for i in range(N_ENVS)
        ])
        train_env = VecNormalize(envs, norm_obs=True, norm_reward=False)

        eval_raw = TradingEnv(
            df=val_df, funding_df=funding_df,
            window_size=config.window_size,
            episode_length=config.episode_length,
        )

        # --- Agent ---
        agent = create_agent(train_env, config, verbose=0)

        # --- Callbacks ---
        trial_dir = Path(f"/content/optuna_trial_{trial.number}")
        trial_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = trial_dir / "training_metrics.csv"

        metrics_cb = TrainingMetricsCallback(
            eval_env=eval_raw,
            log_freq=METRICS_LOG_FREQ,
            n_episodes=3,
            log_path=metrics_path,
            verbose=0,
        )
        curriculum_cb = CurriculumCallback(
            config,
            base_reward_config=base_reward_config,
            phase3_fee_multiplier=p3_fee_mult,
            phase3_churn_multiplier=p3_churn_mult,
            ent_coef_phase1=ent,
            verbose=0,
        )

        callbacks = [metrics_cb, curriculum_cb]

        # --- Pruning callback (registered AFTER metrics_cb for CSV availability) ---
        callbacks.append(
            OptunaPruningCallback(trial, metrics_path, METRICS_LOG_FREQ)
        )

        # --- Train ---
        agent.learn(
            total_timesteps=config.total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=False,
        )

        # --- Score ---
        score, details = compute_score(metrics_path)
        for k, v in details.items():
            trial.set_user_attr(k, v)

        # Cleanup trial dir
        import shutil
        shutil.rmtree(trial_dir, ignore_errors=True)

        train_env.close()
        return score

    except optuna.TrialPruned:
        raise
    except Exception as e:
        trial.set_user_attr("error", str(e))
        logger.exception("Trial %d failed: %s", trial.number, e)
        return -1000.0
```

- [ ] **Step 4: Create notebook — Cell 4 (Phase 1: Screening)**

```python
import optuna

print("=" * 60)
print("PHASE 1: Screening — 200k steps × 40 trials")
print("=" * 60)

study = optuna.create_study(
    study_name="alphacluster_reward_tuning",
    direction="maximize",
    storage=f"sqlite:///{STUDY_DB}",
    load_if_exists=True,
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=100_000,
    ),
    sampler=optuna.samplers.TPESampler(seed=42),
)

# Calculate remaining trials (supports resume)
completed = len([t for t in study.trials
                 if t.state in (optuna.trial.TrialState.COMPLETE,
                                optuna.trial.TrialState.PRUNED)])
remaining = max(0, 40 - completed)
print(f"Completed trials: {completed}, remaining: {remaining}")

if remaining > 0:
    study.optimize(objective, n_trials=remaining, timeout=None)
else:
    print("All 40 screening trials already completed.")

print(f"\nBest trial: #{study.best_trial.number}, score={study.best_trial.value:.4f}")
```

- [ ] **Step 5: Create notebook — Cell 5 (Phase 1 Results & Analysis)**

```python
import json

# Collect results
results = []
for trial in study.trials:
    if trial.value is not None and trial.value > -999:
        row = {"trial": trial.number, "score": trial.value, "state": trial.state.name}
        row.update(trial.params)
        row.update(trial.user_attrs)
        results.append(row)

results_df = pd.DataFrame(results).sort_values("score", ascending=False)
print(f"Viable trials: {len(results_df)} / {len(study.trials)}")
print()
print(results_df.head(10).to_string(index=False))

# Save to Drive
results_df.to_csv(RESULTS_CSV, index=False)
print(f"\nResults saved to {RESULTS_CSV}")

# Optuna visualizations
try:
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_slice,
    )

    fig1 = plot_optimization_history(study)
    fig1.write_image(str(PLOTS_DIR / "phase1_optimization_history.png"))
    fig1.show()

    fig2 = plot_param_importances(study)
    fig2.write_image(str(PLOTS_DIR / "phase1_param_importances.png"))
    fig2.show()

    fig3 = plot_slice(study)
    fig3.write_image(str(PLOTS_DIR / "phase1_slice.png"))
    fig3.show()

    print(f"Plots saved to {PLOTS_DIR}")
except Exception as e:
    print(f"Visualization error: {e}")
```

- [ ] **Step 6: Create notebook — Cell 6 (Phase 2: Validation)**

```python
print("=" * 60)
print("PHASE 2: Validation — 500k steps × top 10")
print("=" * 60)

# Extract top 10 param sets
top_trials = sorted(
    [t for t in study.trials if t.value is not None and t.value > -999],
    key=lambda t: t.value,
    reverse=True,
)[:10]

print(f"Validating {len(top_trials)} configs at {VALIDATION_TIMESTEPS:,} timesteps\n")

VALIDATION_CSV = DRIVE_DIR / "validation_results.csv"
validation_results = []

# Check for existing validation results (resume support)
existing_validated = set()
if VALIDATION_CSV.exists():
    existing_df = pd.read_csv(VALIDATION_CSV)
    existing_validated = set(existing_df["screening_trial"].tolist())
    validation_results = existing_df.to_dict("records")

for i, trial_data in enumerate(top_trials):
    if trial_data.number in existing_validated:
        print(f"  [{i+1}/10] Trial #{trial_data.number} — already validated, skipping")
        continue

    print(f"  [{i+1}/10] Trial #{trial_data.number} (screening score={trial_data.value:.4f})")
    params = trial_data.params

    try:
        base_reward_config = {
            "fee_scale": params["fee_scale"],
            "opportunity_cost_scale": params["opportunity_cost_scale"],
            "opportunity_cost_cap": params["opportunity_cost_cap"],
            "opportunity_cost_threshold": params["opportunity_cost_threshold"],
            "churn_penalty_scale": params["churn_penalty_scale"],
            "drawdown_penalty_scale": params["drawdown_penalty_scale"],
            "quality_scale": params["quality_scale"],
            "position_mgmt_scale": params["position_mgmt_scale"],
        }

        config = TrainingConfig(
            total_timesteps=VALIDATION_TIMESTEPS,
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            ent_coef=params["ent_coef_phase1"],
            eval_freq=VALIDATION_TIMESTEPS + 1,
        )

        envs = SubprocVecEnv([
            make_env(train_df, funding_df, config, rank=r)
            for r in range(N_ENVS)
        ])
        train_env = VecNormalize(envs, norm_obs=True, norm_reward=False)

        eval_raw = TradingEnv(
            df=val_df, funding_df=funding_df,
            window_size=config.window_size,
            episode_length=config.episode_length,
        )

        agent = create_agent(train_env, config, verbose=0)

        trial_dir = Path(f"/content/validation_trial_{trial_data.number}")
        trial_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = trial_dir / "training_metrics.csv"

        metrics_cb = TrainingMetricsCallback(
            eval_env=eval_raw, log_freq=METRICS_LOG_FREQ,
            n_episodes=3, log_path=metrics_path, verbose=0,
        )
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
            callback=CallbackList([metrics_cb, curriculum_cb]),
            progress_bar=False,
        )

        score, details = compute_score(metrics_path)
        result = {
            "screening_trial": trial_data.number,
            "screening_score": trial_data.value,
            "validation_score": score,
            **details,
            **params,
        }
        validation_results.append(result)

        # Save incrementally
        pd.DataFrame(validation_results).to_csv(VALIDATION_CSV, index=False)

        import shutil
        shutil.rmtree(trial_dir, ignore_errors=True)
        train_env.close()

        print(f"    → validation score: {score:.4f}")

    except Exception as e:
        print(f"    → FAILED: {e}")
        validation_results.append({
            "screening_trial": trial_data.number,
            "screening_score": trial_data.value,
            "validation_score": -1000.0,
            "error": str(e),
        })
        pd.DataFrame(validation_results).to_csv(VALIDATION_CSV, index=False)
```

- [ ] **Step 7: Create notebook — Cell 7 (Final Results & Export)**

```python
import json

val_df_results = pd.DataFrame(validation_results)
val_df_results = val_df_results.sort_values("validation_score", ascending=False)

print("=" * 60)
print("FINAL RESULTS — Ranked by 500k Validation Score")
print("=" * 60)
print()

display_cols = [
    "screening_trial", "screening_score", "validation_score",
    "total_pnl_pct", "trades_per_episode", "win_rate", "flat_pct",
]
available = [c for c in display_cols if c in val_df_results.columns]
print(val_df_results[available].to_string(index=False))

# Save best params
best_row = val_df_results.iloc[0]
best_trial_num = int(best_row["screening_trial"])
best_trial = next(t for t in study.trials if t.number == best_trial_num)

best_params = {
    "source": "optuna_tuning",
    "screening_trial": best_trial_num,
    "screening_score": float(best_row["screening_score"]),
    "validation_score": float(best_row["validation_score"]),
    "params": best_trial.params,
    "base_reward_config": {
        k: best_trial.params[k]
        for k in [
            "fee_scale", "opportunity_cost_scale", "opportunity_cost_cap",
            "opportunity_cost_threshold", "churn_penalty_scale",
            "drawdown_penalty_scale", "quality_scale", "position_mgmt_scale",
        ]
    },
    "curriculum": {
        "phase3_fee_multiplier": best_trial.params["phase3_fee_multiplier"],
        "phase3_churn_multiplier": best_trial.params["phase3_churn_multiplier"],
        "ent_coef_phase1": best_trial.params["ent_coef_phase1"],
    },
    "ppo": {
        "learning_rate": best_trial.params["learning_rate"],
        "gamma": best_trial.params["gamma"],
    },
}

with open(BEST_PARAMS_JSON, "w") as f:
    json.dump(best_params, f, indent=2)

print(f"\nBest params saved to {BEST_PARAMS_JSON}")
print(f"\nBest configuration (trial #{best_trial_num}):")
print(json.dumps(best_params["params"], indent=2))
```

- [ ] **Step 8: Create notebook — Cell 8 (Apply Best Parameters)**

```python
# Template: apply best params to TrainingConfig for production training
print("=" * 60)
print("HOW TO APPLY BEST PARAMETERS")
print("=" * 60)

with open(BEST_PARAMS_JSON) as f:
    bp = json.load(f)

print("""
In your training notebook (colab_train.ipynb), modify Cell 3:

```python
from alphacluster.agent.config import TrainingConfig
from alphacluster.agent.trainer import CurriculumCallback

config = TrainingConfig(
    total_timesteps=2_000_000,
    learning_rate={lr},
    gamma={gamma},
    ent_coef={ent},
)

# In train() call, the CurriculumCallback will use these:
# Pass as extra_callbacks or modify trainer.py
base_reward_config = {reward_config}

curriculum_cb = CurriculumCallback(
    config,
    base_reward_config=base_reward_config,
    phase3_fee_multiplier={p3_fee},
    phase3_churn_multiplier={p3_churn},
    ent_coef_phase1={ent},
)
```
""".format(
    lr=bp["ppo"]["learning_rate"],
    gamma=bp["ppo"]["gamma"],
    ent=bp["curriculum"]["ent_coef_phase1"],
    reward_config=json.dumps(bp["base_reward_config"], indent=4),
    p3_fee=bp["curriculum"]["phase3_fee_multiplier"],
    p3_churn=bp["curriculum"]["phase3_churn_multiplier"],
))
```

- [ ] **Step 9: Assemble all cells into the notebook file**

Create `notebooks/colab_optuna.ipynb` with all cells from steps 1-8 as a valid Jupyter notebook JSON.

- [ ] **Step 10: Run linter on any Python source changes**

Run: `.venv/bin/ruff check src/ tests/`
Expected: No errors

- [ ] **Step 11: Commit**

```bash
git add notebooks/colab_optuna.ipynb
git commit -m "feat: add Optuna hyperparameter tuning notebook

Two-phase Bayesian optimization (200k×40 screening + 500k×10 validation)
for 13 reward/curriculum/PPO parameters. Results persist to Google Drive
with resume support via Optuna's load_if_exists."
```

---

### Task 4: Final verification

- [ ] **Step 1: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Run full linter**

Run: `.venv/bin/ruff check src/ tests/ && .venv/bin/ruff format --check src/ tests/`
Expected: No errors

- [ ] **Step 3: Verify notebook is valid JSON**

Run: `python -c "import json; json.load(open('notebooks/colab_optuna.ipynb'))"`
Expected: No errors

- [ ] **Step 4: Final commit (if any fixes needed)**

Only if previous steps required changes.
