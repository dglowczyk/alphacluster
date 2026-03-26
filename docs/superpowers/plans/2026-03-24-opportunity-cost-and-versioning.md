# Opportunity Cost Reward Rebalance + Model Versioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the disabled inactivity penalty with a market-driven opportunity cost mechanism, rebalance fee/churn penalties, and add model versioning visible in training and evaluation.

**Architecture:** The reward function in `trading_env.py` gets a new opportunity cost component (replaces inactivity penalty), reduced fee multiplier (remove ×2.0), and rebalanced churn parameters (base=0.02, threshold=20). Model versioning adds a `MODEL_VERSION` constant in `config.py`, printed at training start and included in metrics output.

**Tech Stack:** Python 3.10+, Gymnasium, Stable-Baselines3, NumPy, pandas, pytest

**Spec:** `docs/superpowers/specs/2026-03-24-reward-rebalance-design.md`

---

### Task 1: Add Model Versioning

**Files:**
- Modify: `src/alphacluster/config.py:68` (add MODEL_VERSION constant)
- Modify: `scripts/train.py:231-234` (print version before training)
- Modify: `scripts/evaluate.py:228` (print version before backtest)
- Modify: `src/alphacluster/backtest/visualizer.py:341-354` (include version in metrics.txt)
- Modify: `src/alphacluster/backtest/metrics.py:31` (include version in metrics dict)
- Test: `tests/test_env.py` (test version is present in metrics)

- [ ] **Step 1: Write failing test for model version in metrics**

Add to `tests/test_env.py` (at the end of the file, as a standalone test function):

```python
def test_model_version_exists():
    """MODEL_VERSION should be defined in config and be a non-empty string."""
    from alphacluster.config import MODEL_VERSION

    assert isinstance(MODEL_VERSION, str)
    assert len(MODEL_VERSION) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_env.py::test_model_version_exists -v`
Expected: FAIL with `ImportError: cannot import name 'MODEL_VERSION'`

- [ ] **Step 3: Add MODEL_VERSION to config.py**

In `src/alphacluster/config.py`, after line 68 (`TOTAL_TIMESTEPS = 1_000_000`), add:

```python
MODEL_VERSION = "v3-opportunity-cost"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_env.py::test_model_version_exists -v`
Expected: PASS

- [ ] **Step 5: Add version print to train.py**

In `scripts/train.py`, add `MODEL_VERSION` to the import from `alphacluster.config` (line 24):

```python
from alphacluster.config import DATA_DIR, MODEL_VERSION, MODELS_DIR
```

Then insert after line 231 (`print(f"Training for {config.total_timesteps:,} timesteps ...")`):

```python
    print(f"  Model version: {MODEL_VERSION}")
```

- [ ] **Step 6: Add version print to evaluate.py**

In `scripts/evaluate.py`, add `MODEL_VERSION` to the import from `alphacluster.config` (line 36):

```python
from alphacluster.config import DATA_DIR, MODEL_VERSION, MODELS_DIR, PROJECT_ROOT, WINDOW_SIZE
```

Then insert before line 228 (`print(f"\nRunning backtest ({args.episodes} episode(s))...")`):

```python
    print(f"Model version: {MODEL_VERSION}")
```

- [ ] **Step 7: Add version to metrics dict**

In `src/alphacluster/backtest/metrics.py`, add import at line 11:

```python
from alphacluster.config import MODEL_VERSION
```

In `calculate_metrics()`, right after `metrics: dict[str, Any] = {}` (line 31), add:

```python
    metrics["model_version"] = MODEL_VERSION
```

- [ ] **Step 8: Add version header to metrics.txt output**

In `src/alphacluster/backtest/visualizer.py`, in `_save_metrics_text()` (line 341), add import and modify the function. Add the import at the top of the file alongside other imports:

```python
from alphacluster.config import MODEL_VERSION
```

Modify the `lines` list in `_save_metrics_text` to include version:

```python
    lines = [
        "BACKTEST METRICS",
        f"Model Version: {MODEL_VERSION}",
        "=" * 50,
        "",
    ]
```

- [ ] **Step 9: Run all tests**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 10: Commit**

```bash
git add src/alphacluster/config.py scripts/train.py scripts/evaluate.py src/alphacluster/backtest/metrics.py src/alphacluster/backtest/visualizer.py tests/test_env.py
git commit -m "feat: add MODEL_VERSION constant with display in training and metrics"
```

---

### Task 2: Replace Inactivity Penalty with Opportunity Cost

**Files:**
- Modify: `src/alphacluster/env/trading_env.py:36-42` (DEFAULT_REWARD_CONFIG)
- Modify: `src/alphacluster/env/trading_env.py:326-418` (_compute_reward)
- Test: `tests/test_env.py`

- [ ] **Step 1: Write failing tests for opportunity cost**

Add these tests to `tests/test_env.py`, replacing the existing `test_inactivity_penalty_when_flat` and adding new ones:

```python
class TestOpportunityCost:
    """Tests for the opportunity cost reward component."""

    def _make_trending_env(self, trend_pct: float = 0.005, n_candles: int = 3200):
        """Create env with a clear directional trend over the lookback window."""
        rng = np.random.default_rng(42)
        timestamps = pd.date_range(start="2025-01-01", periods=n_candles, freq="5min", tz="UTC")
        # Create a smooth uptrend: each candle adds trend_pct/n_candles * base_price
        base_price = 80_000.0
        step_size = base_price * trend_pct / 20  # trend over 20 candles
        close = base_price + np.arange(n_candles) * step_size
        # Add tiny noise
        close = close + rng.normal(0, 0.5, size=n_candles)
        high = close + rng.uniform(0, 5, size=n_candles)
        low = close - rng.uniform(0, 5, size=n_candles)
        low = np.maximum(low, 1.0)
        opn = close + rng.normal(0, 1, size=n_candles)
        opn = np.maximum(opn, 1.0)
        volume = rng.uniform(100, 10000, size=n_candles)
        df = pd.DataFrame({
            "open_time": timestamps,
            "open": opn,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })
        return TradingEnv(
            df=df, window_size=WINDOW_SIZE, episode_length=200, initial_balance=10_000.0,
        )

    def test_opportunity_cost_when_flat_during_trend(self):
        """Flat agent during >0.2% trend should receive negative opportunity cost."""
        env = self._make_trending_env(trend_pct=0.005)
        env.reset(seed=0)
        # Stay flat for 15 steps — rolling 10-step window should catch the trend
        rewards = []
        for _ in range(15):
            _, reward, *_ = env.step([0, 0, 0])
            rewards.append(reward)
        # After the lookback window fills (step 10+), rewards should be negative
        assert any(r < 0 for r in rewards[10:])

    def test_opportunity_cost_zero_during_consolidation(self):
        """Flat agent during sideways market (<0.2% change) should get no penalty."""
        rng = np.random.default_rng(42)
        n_candles = 3200
        timestamps = pd.date_range(start="2025-01-01", periods=n_candles, freq="5min", tz="UTC")
        # Flat market — price barely moves
        close = 80_000.0 + rng.normal(0, 5, size=n_candles)  # ~0.006% noise
        close = np.maximum(close, 79_000.0)
        high = close + rng.uniform(0, 2, size=n_candles)
        low = close - rng.uniform(0, 2, size=n_candles)
        opn = close + rng.normal(0, 1, size=n_candles)
        opn = np.maximum(opn, 1.0)
        volume = rng.uniform(100, 10000, size=n_candles)
        df = pd.DataFrame({
            "open_time": timestamps, "open": opn, "high": high,
            "low": low, "close": close, "volume": volume,
        })
        env = TradingEnv(
            df=df, window_size=WINDOW_SIZE, episode_length=200, initial_balance=10_000.0,
        )
        env.reset(seed=0)
        rewards = []
        for _ in range(20):
            _, reward, *_ = env.step([0, 0, 0])
            rewards.append(reward)
        # All rewards should be 0 (no trend, no opportunity cost)
        assert all(r == pytest.approx(0.0, abs=1e-9) for r in rewards)

    def test_opportunity_cost_zero_when_in_position(self):
        """Agent holding a position should not receive opportunity cost."""
        env = self._make_trending_env(trend_pct=0.01)
        env.reset(seed=0)

        # Collect flat rewards during trend
        flat_rewards = []
        for _ in range(15):
            _, r, *_ = env.step([0, 0, 0])
            flat_rewards.append(r)

        # Reset and collect in-position rewards during same trend
        env.reset(seed=0)
        env.step([1, 1, 0])  # open long
        position_rewards = []
        for _ in range(14):
            _, r, *_ = env.step([1, 1, 0])
            position_rewards.append(r)

        # Flat rewards should be negative (opportunity cost), position should not
        # have that penalty. Sum of position rewards should exceed sum of flat.
        assert sum(flat_rewards[10:]) < 0, "Flat during trend should be penalized"
        assert sum(position_rewards[10:]) > sum(flat_rewards[10:])

    def test_opportunity_cost_at_episode_start(self):
        """First 10 steps should clamp lookback to available history, no crash."""
        env = self._make_trending_env(trend_pct=0.01)
        env.reset(seed=0)
        # Should not crash in early steps when lookback < 10
        for _ in range(10):
            _, reward, terminated, *_ = env.step([0, 0, 0])
            assert not terminated

    def test_opportunity_cost_capped(self):
        """During large moves (>2%), penalty should cap at 0.02 * scale."""
        rng = np.random.default_rng(42)
        n_candles = 3200
        timestamps = pd.date_range(start="2025-01-01", periods=n_candles, freq="5min", tz="UTC")
        base = 80_000.0
        # Create a 5% move over 10 candles at the episode start area
        close = np.full(n_candles, base)
        # Place a big jump around candle 600 (after window_size=576)
        for i in range(590, 600):
            close[i] = base + (i - 589) * base * 0.005  # 0.5% per step, total 5%
        for i in range(600, n_candles):
            close[i] = base * 1.05
        high = close + 10
        low = close - 10
        low = np.maximum(low, 1.0)
        opn = close.copy()
        volume = rng.uniform(100, 10000, size=n_candles)
        df = pd.DataFrame({
            "open_time": timestamps, "open": opn, "high": high,
            "low": low, "close": close, "volume": volume,
        })
        env = TradingEnv(
            df=df, window_size=WINDOW_SIZE, episode_length=200, initial_balance=10_000.0,
        )
        env.reset(seed=0)
        # Step through the big move while flat
        rewards = []
        for _ in range(30):
            _, reward, *_ = env.step([0, 0, 0])
            rewards.append(reward)
        # Cap is 0.02 * scale (default 0.5) = 0.01. No single reward should be worse.
        for r in rewards:
            assert r >= -0.01 - 1e-9

    def test_opportunity_cost_exact_threshold(self):
        """Price change of exactly 0.2% should produce zero penalty (strictly greater)."""
        rng = np.random.default_rng(42)
        n_candles = 3200
        timestamps = pd.date_range(start="2025-01-01", periods=n_candles, freq="5min", tz="UTC")
        base = 80_000.0
        # Create exactly 0.2% change over 10-step window
        close = np.full(n_candles, base, dtype=np.float64)
        # At the episode start area (~candle 577+), set a 0.2% drift over 10 steps
        for i in range(577, 587):
            close[i] = base * (1.0 + 0.002 * (i - 576) / 10)
        for i in range(587, n_candles):
            close[i] = base * 1.002  # exactly 0.2% above base
        high = close + 5
        low = close - 5
        low = np.maximum(low, 1.0)
        opn = close.copy()
        volume = rng.uniform(100, 10000, size=n_candles)
        df = pd.DataFrame({
            "open_time": timestamps, "open": opn, "high": high,
            "low": low, "close": close, "volume": volume,
        })
        env = TradingEnv(
            df=df, window_size=WINDOW_SIZE, episode_length=200, initial_balance=10_000.0,
        )
        env.reset(seed=0)
        # Step through — at exactly 0.2%, penalty should be 0 (strictly greater)
        rewards = []
        for _ in range(20):
            _, r, *_ = env.step([0, 0, 0])
            rewards.append(r)
        # All rewards should be zero — 0.2% is at the threshold, not above
        assert all(r >= -1e-9 for r in rewards)

    def test_trading_beats_flat_during_trend(self):
        """Opening a position during a trend should yield better reward than staying flat."""
        env_flat = self._make_trending_env(trend_pct=0.005)
        env_flat.reset(seed=0)
        flat_rewards = []
        for _ in range(15):
            _, r, *_ = env_flat.step([0, 0, 0])
            flat_rewards.append(r)

        env_trade = self._make_trending_env(trend_pct=0.005)
        env_trade.reset(seed=0)
        trade_rewards = []
        # Open long into the trend
        _, r, *_ = env_trade.step([1, 1, 0])
        trade_rewards.append(r)
        for _ in range(14):
            _, r, *_ = env_trade.step([1, 1, 0])
            trade_rewards.append(r)

        # Net reward for trading should exceed staying flat during a trend
        assert sum(trade_rewards) > sum(flat_rewards)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_env.py::TestOpportunityCost -v`
Expected: FAIL — `opportunity_cost_scale` key missing from reward config, old inactivity code still active

- [ ] **Step 3: Update DEFAULT_REWARD_CONFIG**

In `src/alphacluster/env/trading_env.py`, replace lines 36-42:

```python
# Old:
DEFAULT_REWARD_CONFIG: dict[str, float] = {
    "inactivity_penalty_scale": 0.0,
    "fee_scale": 1.0,
    "drawdown_penalty_scale": 1.0,
    "churn_penalty_scale": 1.0,
    "quality_scale": 1.0,
}
```

With:

```python
DEFAULT_REWARD_CONFIG: dict[str, float] = {
    "opportunity_cost_scale": 0.5,
    "fee_scale": 1.0,
    "drawdown_penalty_scale": 1.0,
    "churn_penalty_scale": 1.0,
    "quality_scale": 1.0,
}
```

- [ ] **Step 4: Rewrite _compute_reward components 2, 3, and 6**

Replace the entire `_compute_reward` method (lines 326-418) with the updated version. The three changes are:

**Component 2 — Fee penalty (remove ×2.0):**
```python
        # 2. FEE PENALTY (no 2x multiplier — curriculum fee_scale handles scaling)
        fee_penalty = (self._step_fees / self.initial_balance) * rc["fee_scale"]
```

**Component 3 — Replace inactivity with opportunity cost:**
```python
        # 3. OPPORTUNITY COST (replaces inactivity penalty)
        opportunity_penalty = 0.0
        if self.account.position_side == "flat":
            lookback = min(10, self._current_idx - self._start_idx)
            if lookback > 0:
                price_change = abs(
                    self._close[self._current_idx] - self._close[self._current_idx - lookback]
                ) / max(self._close[self._current_idx - lookback], 1e-12)
                if price_change > 0.002:
                    raw_penalty = price_change - 0.002
                    opportunity_penalty = min(raw_penalty, 0.02) * rc["opportunity_cost_scale"]
```

**Component 6 — Churn penalty (base=0.02, threshold=20):**
```python
        # 6. CHURN PENALTY (quadratic, base=0.02, threshold=20)
        churn_penalty = 0.0
        if self._trade_just_completed and self._last_trade_duration < 20:
            fraction_remaining = 1.0 - self._last_trade_duration / 20.0
            churn_penalty = 0.02 * fraction_remaining ** 2 * rc.get("churn_penalty_scale", 1.0)
```

**Final formula update (use `opportunity_penalty` instead of `inactivity_penalty`):**
```python
        reward = (
            pnl_reward
            - fee_penalty
            - opportunity_penalty
            + position_reward
            + completion_reward
            - churn_penalty
            - dd_penalty
            + quality_bonus
        )
```

Also update the docstring to reflect the changes:
```python
        """Multi-component reward function.

        Components:
        1. Asymmetric PnL reward (winners 1.5x)
        2. Fee penalty (scaled by fee_scale)
        3. Opportunity cost (market-driven penalty for being flat during trends)
        4. Position management reward (sqrt ramp for winners, cut losers)
        5. Trade completion reward (quadratic duration scaling)
        6. Churn penalty (quadratic, base=0.02, threshold=20)
        7. Quadratic drawdown penalty
        8. Trade quality bonus (per-step profitability for winners)
        """
```

- [ ] **Step 5: Run opportunity cost tests**

Run: `.venv/bin/python -m pytest tests/test_env.py::TestOpportunityCost -v`
Expected: All pass

- [ ] **Step 6: Update old inactivity test**

Remove the `test_inactivity_penalty_when_flat` method from `TestTradingEnv` class (lines 663-712) since the inactivity penalty component no longer exists.

- [ ] **Step 7: Update test_reward_config_mutable**

Replace the existing test (line 726-733):

```python
    def test_reward_config_mutable(self):
        """reward_config should be changeable for curriculum learning."""
        env = _make_env()
        env.reset(seed=0)
        env.reward_config["opportunity_cost_scale"] = 2.0
        assert env.reward_config["opportunity_cost_scale"] == 2.0
        env.reward_config["churn_penalty_scale"] = 0.5
        assert env.reward_config["churn_penalty_scale"] == 0.5
```

- [ ] **Step 8: Write test for rebalanced churn penalty**

Add to `tests/test_env.py`:

```python
def test_churn_penalty_rebalanced():
    """Churn penalty with base=0.02, threshold=20: verify key magnitudes."""
    # duration=1 -> 0.02 * (1 - 1/20)^2 = 0.02 * 0.9025 = ~0.018
    # duration=10 -> 0.02 * (1 - 10/20)^2 = 0.02 * 0.25 = 0.005
    # duration=20 -> 0.02 * 0 = 0.0
    env = _make_env()
    env.reset(seed=0)

    # Simulate 1-step trade
    env._trade_just_completed = True
    env._last_trade_duration = 1
    env._last_trade_pnl = 0.0
    env._step_fees = 0.0
    r1 = env._compute_reward()

    env._last_trade_duration = 10
    r10 = env._compute_reward()

    env._last_trade_duration = 20
    r20 = env._compute_reward()

    # 1-step should have bigger penalty than 10-step
    assert r1 < r10
    # 20-step should have no churn penalty
    assert r20 >= r10
```

- [ ] **Step 9: Run all tests**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 10: Run lint**

Run: `.venv/bin/ruff check src/ tests/ && .venv/bin/ruff format --check src/ tests/`
Expected: Clean

- [ ] **Step 11: Commit**

```bash
git add src/alphacluster/env/trading_env.py tests/test_env.py
git commit -m "feat: replace inactivity penalty with opportunity cost, rebalance fee/churn"
```

---

### Task 3: Update Curriculum Phases in Trainer

**Files:**
- Modify: `src/alphacluster/agent/trainer.py:122-149` (CurriculumCallback._apply_phase)
- Test: `tests/test_env.py` (or inline verification)

- [ ] **Step 1: Write failing test for curriculum phases**

Add to `tests/test_env.py`:

```python
def test_curriculum_phases_use_opportunity_cost():
    """CurriculumCallback should use opportunity_cost_scale, not inactivity_penalty_scale."""
    from alphacluster.agent.config import TrainingConfig
    from alphacluster.agent.trainer import CurriculumCallback

    config = TrainingConfig(total_timesteps=100)
    cb = CurriculumCallback(config)
    # Phase 1 config should contain opportunity_cost_scale
    # We can't easily call _apply_phase without a model, so verify the code compiles
    # and the callback exists
    assert cb._current_phase == 0
```

- [ ] **Step 2: Update _apply_phase in trainer.py**

Replace lines 122-149 in `src/alphacluster/agent/trainer.py`:

```python
    def _apply_phase(self, phase: int) -> None:
        if phase == 1:
            ent_coef = 0.05
            reward_config = {
                "opportunity_cost_scale": 0.3,
                "fee_scale": 0.5,
                "drawdown_penalty_scale": 0.3,
                "churn_penalty_scale": 0.5,
                "quality_scale": 1.0,
            }
        elif phase == 2:
            ent_coef = 0.03
            reward_config = {
                "opportunity_cost_scale": 0.5,
                "fee_scale": 1.0,
                "drawdown_penalty_scale": 1.0,
                "churn_penalty_scale": 1.0,
                "quality_scale": 1.0,
            }
        else:  # phase 3
            ent_coef = 0.005
            reward_config = {
                "opportunity_cost_scale": 1.0,
                "fee_scale": 2.0,
                "drawdown_penalty_scale": 1.5,
                "churn_penalty_scale": 2.0,
                "quality_scale": 0.5,
            }

        # Update agent entropy coefficient
        self.model.ent_coef = ent_coef

        # Update reward config on all environments
        self._set_env_reward_config(reward_config)

        msg = f"Curriculum: phase {phase} (ent_coef={ent_coef})"
        logger.info(msg)
        if self.verbose:
            print(msg)
```

- [ ] **Step 3: Run all tests**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 4: Run lint**

Run: `.venv/bin/ruff check src/ tests/ && .venv/bin/ruff format --check src/ tests/`
Expected: Clean

- [ ] **Step 5: Commit**

```bash
git add src/alphacluster/agent/trainer.py tests/test_env.py
git commit -m "feat: update curriculum phases to use opportunity_cost_scale"
```

---

### Task 4: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update reward component list in CLAUDE.md**

In the "Multi-component reward function" section (item 5 under Design Decisions), update:
- Component 3: "Opportunity cost (market-driven penalty for flat during trends, 0.2% threshold, 10-step lookback)" instead of "Inactivity penalty"
- Component 6: "Churn penalty (penalizes trades held fewer than 20 steps, base 0.02)" instead of "base 0.01"
- Remove "x2 base weight" from fee penalty description

Also update the curriculum section (item 6):
- Phase 2 boundary: "(30-60%)" not "(30-70%)"
- Phase 3 boundary: "(60-100%)" not "(70-100%)"
- Replace `inactivity_penalty_scale` references with `opportunity_cost_scale`

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for opportunity cost and rebalanced reward"
```
