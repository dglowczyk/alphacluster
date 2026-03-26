# Simple Action Space Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configurable 3-action mode (long/flat/short) with fixed position size and leverage, to diagnose whether the 36-action space is the bottleneck preventing the agent from learning directional trading.

**Architecture:** A `simple_actions` flag on `TrainingConfig` toggles between `Discrete(3)` and `MultiDiscrete([3,4,3])`. When enabled, `TradingEnv` uses fixed size (10%) and leverage (10x). Same direction repeats are no-ops (zero fees). All env instantiation sites (scripts, CLI, notebooks) propagate the flag from config.

**Tech Stack:** Python 3.10+, Gymnasium, Stable-Baselines3, numpy, pytest

**Spec:** `docs/superpowers/specs/2026-03-25-simple-action-space-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/alphacluster/agent/config.py` | Modify | Add `simple_actions`, `fixed_size_pct`, `fixed_leverage` fields |
| `src/alphacluster/env/trading_env.py` | Modify | Conditional action space, step() decoding, same-direction no-op |
| `scripts/train.py` | Modify | Pass simple_actions flags to all TradingEnv instantiations |
| `scripts/evaluate.py` | Modify | Pass simple_actions flags to TradingEnv |
| `src/alphacluster/cli.py` | Modify | Pass simple_actions flags to train/evaluate CLI subcommands |
| `tests/test_env.py` | Modify | Add `TestSimpleActions` class with 6 tests |
| `notebooks/colab_train_simple.ipynb` | Create | Diagnostic notebook for 3-action training |

### Spec Deviations

- **`src/alphacluster/agent/trainer.py`** — Listed in the spec's Files Changed table, but `trainer.py` does not instantiate `TradingEnv`. It receives envs as arguments. SB3's `create_agent()` auto-detects `env.action_space`, so no changes are needed. Omitted intentionally.
- **CLI tournament env** — The spec says "Same propagation needed" for CLI. However, the tournament command (`_cmd_tournament`) creates its own `TradingEnv` for head-to-head matches. Per the spec's own design decision, "models trained with different action space modes cannot compete in tournaments." Adding `simple_actions` to tournament would be meaningless — both models must use the same mode. Tournament env left unchanged intentionally.

---

### Task 1: Add config fields to TrainingConfig

**Files:**
- Modify: `src/alphacluster/agent/config.py:15-17`
- Test: `tests/test_env.py`

- [ ] **Step 1: Write failing test for new config fields**

In `tests/test_env.py`, add at the end of the file:

```python
class TestSimpleActionsConfig:
    def test_default_simple_actions_false(self):
        """Default TrainingConfig has simple_actions=False."""
        config = TrainingConfig()
        assert config.simple_actions is False
        assert config.fixed_size_pct == 0.10
        assert config.fixed_leverage == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_env.py::TestSimpleActionsConfig::test_default_simple_actions_false -v`
Expected: FAIL with `AttributeError: ... has no attribute 'simple_actions'`

- [ ] **Step 3: Add fields to TrainingConfig**

In `src/alphacluster/agent/config.py`, add three fields after the `episode_length` field (line 17):

```python
    # ── Simple action mode ─────────────────────────────────────────────
    simple_actions: bool = False       # True → Discrete(3), False → MultiDiscrete([3,4,3])
    fixed_size_pct: float = 0.10       # position size when simple_actions=True
    fixed_leverage: int = 10           # leverage when simple_actions=True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_env.py::TestSimpleActionsConfig -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/alphacluster/agent/config.py tests/test_env.py
git commit -m "feat: add simple_actions config fields to TrainingConfig"
```

---

### Task 2: Parameterize TradingEnv action space and step()

**Files:**
- Modify: `src/alphacluster/env/trading_env.py:71-78` (constructor), `135-137` (action_space), `202-215` (step decoding), `255-275` (same-direction branch)
- Test: `tests/test_env.py`

This is the core task. It modifies `TradingEnv.__init__()` to accept simple action params, conditionally sets the action space, updates `step()` to decode both scalar and array actions, and makes same-direction a no-op in simple mode.

- [ ] **Step 1: Write failing tests for simple action space**

Add to `tests/test_env.py`:

```python
class TestSimpleActions:
    """Tests for the simple_actions=True 3-action mode."""

    def test_action_space_is_discrete_3(self):
        """simple_actions=True → Discrete(3) action space."""
        env = TradingEnv(
            df=_make_df(),
            simple_actions=True,
            fixed_size_pct=0.10,
            fixed_leverage=10,
        )
        from gymnasium.spaces import Discrete
        assert isinstance(env.action_space, Discrete)
        assert env.action_space.n == 3

    def test_default_actions_unchanged(self):
        """simple_actions=False (default) → MultiDiscrete([3,4,3])."""
        env = TradingEnv(df=_make_df())
        from gymnasium.spaces import MultiDiscrete
        assert isinstance(env.action_space, MultiDiscrete)

    def test_step_long_opens_with_fixed_params(self):
        """action=1 opens long with fixed_size_pct and fixed_leverage."""
        env = TradingEnv(
            df=_make_df(),
            simple_actions=True,
            fixed_size_pct=0.10,
            fixed_leverage=10,
        )
        env.reset(seed=42)
        _obs, _reward, _term, _trunc, info = env.step(1)  # long
        assert info["position_side"] == "long"
        assert info["leverage"] == 10

    def test_step_flat_closes_position(self):
        """action=0 closes an open position."""
        env = TradingEnv(
            df=_make_df(),
            simple_actions=True,
            fixed_size_pct=0.10,
            fixed_leverage=10,
        )
        env.reset(seed=42)
        env.step(1)  # open long
        _obs, _reward, _term, _trunc, info = env.step(0)  # go flat
        assert info["position_side"] == "flat"

    def test_same_direction_is_noop(self):
        """Repeating action=1 while long → no fees, no modification."""
        env = TradingEnv(
            df=_make_df(),
            simple_actions=True,
            fixed_size_pct=0.10,
            fixed_leverage=10,
        )
        env.reset(seed=42)
        _obs, _reward, _term, _trunc, info_open = env.step(1)  # open long
        balance_after_open = info_open["balance"]
        _obs, _reward, _term, _trunc, info_hold = env.step(1)  # repeat long
        assert info_hold["fees"] == 0.0
        # Balance unchanged by fees (only PnL may change)
        assert info_hold["position_side"] == "long"

    def test_gymnasium_check_env_simple(self):
        """Gymnasium's check_env passes for simple_actions=True."""
        from gymnasium.utils.env_checker import check_env
        env = TradingEnv(
            df=_make_df(),
            simple_actions=True,
            fixed_size_pct=0.10,
            fixed_leverage=10,
        )
        check_env(env, skip_render_check=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_env.py::TestSimpleActions -v`
Expected: FAIL with `TypeError: TradingEnv.__init__() got an unexpected keyword argument 'simple_actions'`

- [ ] **Step 3: Implement TradingEnv changes**

**3a. Constructor** — Add parameters to `__init__()` at `trading_env.py:71-78`:

Change the constructor signature from:
```python
    def __init__(
        self,
        df: pd.DataFrame,
        funding_df: pd.DataFrame | None = None,
        window_size: int = WINDOW_SIZE,
        episode_length: int = EPISODE_LENGTH,
        initial_balance: float = 10_000.0,
    ) -> None:
```

To:
```python
    def __init__(
        self,
        df: pd.DataFrame,
        funding_df: pd.DataFrame | None = None,
        window_size: int = WINDOW_SIZE,
        episode_length: int = EPISODE_LENGTH,
        initial_balance: float = 10_000.0,
        simple_actions: bool = False,
        fixed_size_pct: float = 0.10,
        fixed_leverage: int = 10,
    ) -> None:
```

After `self.initial_balance = initial_balance` (line 85), add:
```python
        self._simple_actions = simple_actions
        self._fixed_size_pct = fixed_size_pct
        self._fixed_leverage = fixed_leverage
```

**3b. Action space** — Replace the action_space assignment at lines 135-137:

From:
```python
        self.action_space = spaces.MultiDiscrete(
            [N_DIRECTIONS, N_POSITION_SIZES, N_LEVERAGE_LEVELS]
        )
```

To:
```python
        if self._simple_actions:
            self.action_space = spaces.Discrete(3)  # 0=flat, 1=long, 2=short
        else:
            self.action_space = spaces.MultiDiscrete(
                [N_DIRECTIONS, N_POSITION_SIZES, N_LEVERAGE_LEVELS]
            )
```

**3c. step() signature** — Update type annotation at line 202-203:

From:
```python
    def step(
        self, action: np.ndarray | list | tuple
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
```

To:
```python
    def step(
        self, action: np.ndarray | int | np.integer | list | tuple
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
```

**3d. Action decoding** — Replace lines 205-215 (the action decoding block at top of step()):

From:
```python
        direction_idx, size_idx, leverage_idx = int(action[0]), int(action[1]), int(action[2])

        direction = direction_idx  # 0=flat, 1=long, 2=short

        # Collapse flat actions: when direction=flat, size and leverage are irrelevant.
        if direction == 0:
            size_pct = 0.0
            leverage = 1
        else:
            size_pct = POSITION_SIZE_OPTIONS[size_idx]
            leverage = LEVERAGE_OPTIONS[leverage_idx]
```

To:
```python
        if self._simple_actions:
            direction = int(action)
            if direction == 0:
                size_pct, leverage = 0.0, 1
            else:
                size_pct = self._fixed_size_pct
                leverage = self._fixed_leverage
        else:
            direction_idx, size_idx, leverage_idx = (
                int(action[0]), int(action[1]), int(action[2])
            )
            direction = direction_idx  # 0=flat, 1=long, 2=short
            if direction == 0:
                size_pct = 0.0
                leverage = 1
            else:
                size_pct = POSITION_SIZE_OPTIONS[size_idx]
                leverage = LEVERAGE_OPTIONS[leverage_idx]
```

**3e. Same-direction no-op** — Replace the `else` branch at lines 255-275:

From:
```python
        else:
            # Same direction — check if size/leverage differs
            current_leverage = self.account.leverage
            if self.account.entry_price > 0:
                current_notional = self.account.position_size * self.account.entry_price
                current_size_pct_approx = (
                    current_notional / (self.account.balance * current_leverage)
                    if (self.account.balance * current_leverage) > 0
                    else 0.0
                )
            else:
                current_size_pct_approx = 0.0

            if abs(size_pct - current_size_pct_approx) > 0.01 or leverage != current_leverage:
                fee = self.account.modify_position(
                    size_pct=size_pct,
                    leverage=leverage,
                    price=current_price,
                )
                total_fees += fee
            # else: truly holding the same position, no action
```

To:
```python
        else:
            # Same direction as current position
            if self._simple_actions:
                pass  # no-op: fixed size/leverage, nothing to modify
            else:
                # Check if size/leverage differs
                current_leverage = self.account.leverage
                if self.account.entry_price > 0:
                    current_notional = self.account.position_size * self.account.entry_price
                    current_size_pct_approx = (
                        current_notional / (self.account.balance * current_leverage)
                        if (self.account.balance * current_leverage) > 0
                        else 0.0
                    )
                else:
                    current_size_pct_approx = 0.0

                if (
                    abs(size_pct - current_size_pct_approx) > 0.01
                    or leverage != current_leverage
                ):
                    fee = self.account.modify_position(
                        size_pct=size_pct,
                        leverage=leverage,
                        price=current_price,
                    )
                    total_fees += fee
                # else: truly holding the same position, no action
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_env.py::TestSimpleActions -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Run full test suite to verify no regressions**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All existing tests PASS (default `simple_actions=False` preserves behavior)

- [ ] **Step 6: Run linter**

Run: `.venv/bin/ruff check src/alphacluster/env/trading_env.py`
Expected: Clean

- [ ] **Step 7: Commit**

```bash
git add src/alphacluster/env/trading_env.py tests/test_env.py
git commit -m "feat: add simple_actions mode to TradingEnv (Discrete(3) action space)"
```

---

### Task 3: Propagate simple_actions flags in scripts and CLI

**Files:**
- Modify: `scripts/train.py:147-167` (_make_env, single env, eval env), `206-221`
- Modify: `scripts/evaluate.py:204-208`
- Modify: `src/alphacluster/cli.py:189-192`

All three files instantiate `TradingEnv` directly. Each must pass the three simple_actions config fields. No new tests needed — the existing tests verify `TradingEnv` behavior; these are wiring changes.

- [ ] **Step 1: Update `scripts/train.py`**

**1a. Add `--simple-actions` CLI flag** — In `parse_args()`, after the `--no-curriculum` argument (line 78):

```python
    parser.add_argument(
        "--simple-actions",
        action="store_true",
        help="Use simplified 3-action space (long/flat/short) with fixed size and leverage",
    )
```

**1b. Pass flag to config** — In `main()`, inside `config_overrides` block (around line 188):

```python
    if args.simple_actions:
        config_overrides["simple_actions"] = True
```

**1c. Update `_make_env()` factory** — In the `_make_env` function (line 147), change the inner `TradingEnv` call from:

```python
        env = TradingEnv(
            df=df,
            funding_df=funding_df,
            window_size=config.window_size,
            episode_length=config.episode_length,
        )
```

To:

```python
        env = TradingEnv(
            df=df,
            funding_df=funding_df,
            window_size=config.window_size,
            episode_length=config.episode_length,
            simple_actions=config.simple_actions,
            fixed_size_pct=config.fixed_size_pct,
            fixed_leverage=config.fixed_leverage,
        )
```

**1d. Update single-env path** — At line 206-211:

```python
        train_env = TradingEnv(
            df=train_df,
            funding_df=funding_df,
            window_size=config.window_size,
            episode_length=config.episode_length,
            simple_actions=config.simple_actions,
            fixed_size_pct=config.fixed_size_pct,
            fixed_leverage=config.fixed_leverage,
        )
```

**1e. Update eval env** — At line 216-221:

```python
        eval_env = TradingEnv(
            df=val_df,
            funding_df=funding_df,
            window_size=config.window_size,
            episode_length=config.episode_length,
            simple_actions=config.simple_actions,
            fixed_size_pct=config.fixed_size_pct,
            fixed_leverage=config.fixed_leverage,
        )
```

**1f. Print simple_actions status** — After the curriculum print (line 235):

```python
    if config.simple_actions:
        print(f"  Simple actions: enabled (size={config.fixed_size_pct}, leverage={config.fixed_leverage}x)")
```

- [ ] **Step 2: Update `scripts/evaluate.py`**

**2a. Add `--simple-actions` CLI flag** — In `parse_args()`, after `--verbose`:

```python
    parser.add_argument(
        "--simple-actions",
        action="store_true",
        help="Use simplified 3-action space (must match training mode)",
    )
```

**2b. Pass to env creation** — At line 204-208, change:

```python
    env = TradingEnv(
        df=test_df,
        funding_df=funding_df,
        episode_length=min(len(test_df) - WINDOW_SIZE - 1, 2016),
    )
```

To:

```python
    env = TradingEnv(
        df=test_df,
        funding_df=funding_df,
        episode_length=min(len(test_df) - WINDOW_SIZE - 1, 2016),
        simple_actions=args.simple_actions,
    )
```

- [ ] **Step 3: Update `src/alphacluster/cli.py`**

**3a. Add `--simple-actions` flag to the train subcommand** — After `--tournament` (line 52):

```python
    sub_train.add_argument(
        "--simple-actions",
        action="store_true",
        help="Use simplified 3-action space",
    )
```

**3b. Add `--simple-actions` flag to the evaluate subcommand** — After `--save-charts` (line 72):

```python
    sub_eval.add_argument(
        "--simple-actions",
        action="store_true",
        help="Use simplified 3-action space",
    )
```

**3c. Propagate in `_cmd_train()`** — At line 146-149:

```python
    train_argv = ["--timesteps", str(args.timesteps)]
    if args.tournament:
        train_argv.append("--tournament")
    if args.simple_actions:
        train_argv.append("--simple-actions")
```

**3d. Propagate in `_cmd_evaluate()`** — At line 156-158:

```python
    eval_argv = ["--model", args.model, "--episodes", str(args.episodes)]
    if args.save_charts:
        eval_argv.append("--save-charts")
    if args.simple_actions:
        eval_argv.append("--simple-actions")
```

**3e. Tournament env** — No change needed. See "Spec Deviations" section above for rationale.

- [ ] **Step 4: Run linter**

Run: `.venv/bin/ruff check scripts/train.py scripts/evaluate.py src/alphacluster/cli.py`
Expected: Clean

- [ ] **Step 5: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/train.py scripts/evaluate.py src/alphacluster/cli.py
git commit -m "feat: propagate simple_actions flags to scripts and CLI"
```

---

### Task 4: Create diagnostic training notebook

**Files:**
- Create: `notebooks/colab_train_simple.ipynb`

Copy of `colab_train.ipynb` with simple_actions enabled and 500k steps.

- [ ] **Step 1: Create notebook**

Create `notebooks/colab_train_simple.ipynb` as a copy of `colab_train.ipynb` with these changes:

1. **Title** (Cell 0 markdown): Change to `# AlphaCluster — Simple Actions Diagnostic (3-Action Mode)`
   - Update description to explain this is a diagnostic test with 3 actions (long/flat/short)
   - Replace the "36-action space" bullet with "**3-action space** — long / flat / short with fixed 10% size and 10x leverage"
   - Remove the "Position modification" bullet

2. **Cell 3 (Create Environments)** — Change config creation:

```python
config = TrainingConfig(
    total_timesteps=500_000,
    simple_actions=True,
    fixed_size_pct=0.10,
    fixed_leverage=10,
)
config.eval_freq = 50_000
```

And add `simple_actions`/`fixed_size_pct`/`fixed_leverage` to all `TradingEnv()` calls:

```python
        env = TradingEnv(
            df=train_df,
            funding_df=funding_df,
            window_size=config.window_size,
            episode_length=config.episode_length,
            simple_actions=config.simple_actions,
            fixed_size_pct=config.fixed_size_pct,
            fixed_leverage=config.fixed_leverage,
        )
```

Same for `eval_env` and `test_env` in Cell 5.

3. **Cell 4 (Train)** — Update checkpoint dir:

```python
LOCAL_CHECKPOINT_DIR = Path("/content/checkpoints_simple")
```

Change `ProgressCallback` log_interval to `10_000` (faster feedback for shorter run).

4. **Cell 5 (Evaluate)** — Add `simple_actions` to test env:

```python
test_env = TradingEnv(
    df=test_df,
    funding_df=funding_df,
    window_size=config.window_size,
    episode_length=config.episode_length,
    simple_actions=config.simple_actions,
    fixed_size_pct=config.fixed_size_pct,
    fixed_leverage=config.fixed_leverage,
)
```

- [ ] **Step 2: Verify notebook JSON is valid**

Run: `.venv/bin/python -c "import json; json.load(open('notebooks/colab_train_simple.ipynb')); print('Valid JSON')"`
Expected: `Valid JSON`

- [ ] **Step 3: Commit**

```bash
git add notebooks/colab_train_simple.ipynb
git commit -m "feat: add colab_train_simple.ipynb for 3-action diagnostic training"
```

---

## Verification Checklist

After all tasks are complete:

1. `.venv/bin/python -m pytest tests/ -v` — all tests pass
2. `.venv/bin/ruff check src/ tests/ scripts/` — clean
3. `TrainingConfig()` produces 36-action env (default unchanged)
4. `TrainingConfig(simple_actions=True)` produces 3-action env
5. Notebook exists and has valid JSON
